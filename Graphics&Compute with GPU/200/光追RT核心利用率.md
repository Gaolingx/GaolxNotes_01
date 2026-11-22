# 光追RT核心利用率

## 🧑‍💻 User

话说光追rtcore利用率怎么做，单core单RAY吗

---
## 😀 默认助手 (7)

## 先纠正一个前提：RT core 不是「单核单光线」

这是最容易带偏优化方向的一个误解。RT core 是挂在 SM 上的**固定功能流水线**（Turing / Ampere 大致 1 个 RT core / SM，Ada 也是每 SM 一个但吞吐再翻倍），它**不是**一个「核心」去负责处理一条光线：

- 一次 `TraceRay` 在 SM 里是 **warp 级**发射的 → 32 条 ray 一起进入遍历。
- 每条 ray 的 BVH 遍历长度**变长、不可预测**（取决于命中深度、cache miss），一条 ray 会在管线里卡几百拍。所以「一条 ray 喂一个单元」这种模型根本不存在——**单条 ray 永远填不满**，硬件靠的是大量独立 ray 在流水线里重叠。
- 因此 RT core 的有效吞吐是多个约束里的最小值：

$$\text{实际 ray 吞吐} = \min\bigl(\text{RT core 的 box/triangle test 吞吐},\ \text{BVH 访存带宽},\ \text{ray 供给速率},\ \text{shader 端吞吐}\bigr)$$

NVIDIA 宣传的「10 GRays/s」这类峰值，是在**极简场景（primary ray、低发散、BVH 全在 cache）**下标定出来的，不是 test 吞吐极限——这也说明：**绝大多数真实场景里跑不满，瓶颈不在 RT core 本身。**

---

## 一、「利用率」到底在测什么

坏消息先说：**没有公开的、直接叫 “RT core utilization” 的计数器**。你只能通过代理指标反推。有用的几类：

| 你想知道的 | 看什么 |
|---|---|
| RT 流水线忙不忙 | RT/BVH 相关的 pipe 计数器（Nsight Compute / RGP 里的 RT 段，名字随版本变）+ **warp stall 原因** |
| ray 供给够不够 | Achieved Occupancy、活跃 warp 数、寄存器/共享内存占用 |
| 是不是访存瓶颈 | Memory Throughput、L1/L2 hit rate、DRAM 带宽占峰值比例 |
| RT 到底占了多少时间 | 时间戳 query 切分（raygen / traversal / hit shader），或开/关光追两版对比 |
| 算法本身合不合理 | 实测 rays/s vs 理论峰值、每像素 spp 与噪声收敛曲线 |

**判断顺序建议**：先看是不是 memory-bound（BVH 遍历是极端延迟敏感，十有八九是它）→ 再看 occupancy / ray 供给 → 最后才怀疑 RT core 本身。

<details>
<summary><b>反直觉的一点：命中率高反而可能更慢</b></summary>

如果你在 hit shader 或 **AnyHit / Intersection shader** 里做事（alpha test、程序化几何），遍历就会从「固定功能硬件」退化成「SM 上的可编程循环」——RT core 的利用率会被 shader 抢走。这正是 Ada 引入 **Opacity Micromap** 和 **DMM（Displaced Micro-Mesh）** 的动机：把 alpha test 和细分几何重新塞回硬件。

所以如果你的 profile 显示 RT 阶段很慢但 RT 相关计数器不高，先查 **AnyHit 调用次数**，而不是调 BVH。
</details>

---

## 二、把利用率打上去的几个杠杆（按性价比排序）

1. **保证独立 ray 数量足够**（最常被忽略）
   - 别一次 dispatch 只发几条 ray；batch 起来，让同一个 wave 里有大量独立遍历。
   - 降寄存器压力、控制 shared memory，把 occupancy 拉上去——这是延迟隐藏的前提。

2. **降 divergence**
   - 同一 warp 内 32 条 ray 方向/起点差异大 → 每个 box test 的命中掩码稀疏 → 有效吞吐暴跌。
   - 手段：ray **reordering / sorting**（按方向、按 tile、按 hit/miss 分桶后再发射）、coherent ray packets、primary ray 按屏幕 tile 分批。

3. **BVH 质量比什么都重要**
   - 用 **SAH** build、调 leaf size（太小 → 遍历深；太大 → 叶内三角形测试爆）、动态物体单独 BLAS 用 refit 而非 rebuild、避免 TLAS instance 数量爆炸。
   - 检查 BLAS 的 **build quality flag**（fast build vs. fast trace），静态几何一定要用 fast trace。

4. **减少 ray 数量，而不是硬堆 spp**
   - 重要性采样、denoiser 前置、adaptive sampling、ReSTIR / SER 类算法。多打一条 ray 的成本远高于让一条 ray 更聪明。

5. **别让 shader 成为瓶颈**
   - raygen 里的 ALU 工作、过重的 hit shader、递归调用深度，都会拖累 issue。SER（Shader Execution Reordering）就是专门治这个的。

6. **用对硬件特性**
   - Ada：Opacity Micromap 替代 alpha test、DMM 替代高密度细分网格。
   - 保证驱动/API 路径没退化（DXR 1.1 inline vs. pipeline，Vulkan KHR ray query vs. pipeline）。

---

## 三、一个可落地的诊断流程

```
1. 做微基准：固定 BVH + 只有 primary ray，扫 ray 数量 / occupancy / 发散度
   → 画出 rays/s 曲线，找到"喂饱"的拐点，这就是你场景的天花板
2. 把真实场景叠上去，看 rays/s 掉了多少、掉在哪个计数器上
3. 按 现象 → 病因 定位：
```

| 现象 | 最可能的病因 |
|---|---|
| RT pipe 计数器低、memory 高 | BVH 太大 / cache miss / 叶子太碎 |
| RT pipe 低、warp stall 在 long scoreboard | 访存延迟，occupancy 不够隐藏 |
| RT pipe 低、AnyHit 调用数高 | 可编程遍历（alpha test 等） |
| rays/s 随 spp 线性掉 | 纯供给不足 / 算法层面该减 ray |
| 开关光追时间差不大 | 光追根本不是瓶颈，别优化了 |

**一句话回答你的问题**：不是「单 core 单 RAY」，而是「一个 warp 一批 ray 进流水线，靠足够多的独立 ray + 低发散 + 好 BVH 把 RT core 喂饱」；利用率本身没有直接计数器，要用 rays/s、warp stall、memory throughput 三件套反推。

如果你能说一下具体场景（离线渲染器 / 实时 DXR / Vulkan，静态还是动态几何，有没有 AnyHit），我可以把上面的清单收窄成针对你那套东西的具体动作。
