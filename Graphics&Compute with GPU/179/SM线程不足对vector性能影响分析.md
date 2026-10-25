# SM线程不足对vector性能影响分析

## 🧑‍💻 User

gpukernel一般不会一个sm只能跑一个cta，但是nv的卡一个sm最多只能上2048 threads，如果比2048小（其他家的accelerator不一定是这个）对vector性能有哪些影响？

---
## 😀 默认助手 (7)

## 先说结论

NVIDIA 把 SM 上限定在 2048 threads，本质上是**为"每个 warp 只有 1 个未完成访存就能盖住 DRAM 延迟"而配的容量**。vector 类 kernel（elementwise、SAXPY、scale/copy、norm 前的 load、vector add 之类）几乎都是 *memory-latency / bandwidth bound*，所以"上不满 2048"这件事本身不会直接伤害性能——真正起作用的是一个物理量：

$$\text{bytes in flight} = \text{threads}_{\text{resident}} \times \text{ILP} \times \text{单次访存宽度}$$

只要这个量够大，occupancy 掉一半也无所谓；不够大，就按比例掉带宽。下面拆开讲。

---

## 一、2048 这个数字是怎么算出来的

用 Little's Law：要打满某条通路，必须有

$$B_{\text{in-flight}} \ge \text{BW} \times t_{\text{DRAM}}$$

以 H100 为例（3.35 TB/s、132 个 SM、时钟约 1.8 GHz、HBM 端到端延迟约 350 ns）：

$$\text{BW}_{\text{per SM}} = \frac{3350\ \text{GB/s}}{132} \approx 25.4\ \text{GB/s}$$

$$B_{\text{in-flight}} \approx 25.4\times10^{9} \times 350\times10^{-9} \approx 8.9\ \text{KB per SM}$$

再看一个 warp 的一次访存：32 个线程 × 4 B = **128 B**（正好一条 cache line / 4 个 32 B sector）。

$$64\ \text{warps} \times 128\ \text{B} = 8\ \text{KB} \approx 8.9\ \text{KB}$$

**64 warps = 2048 threads，正好等于"每个 warp 恰好一个访问在飞"所需的量。** 这就是 2048 的来源——它不是随便定的，是被 DRAM 延迟 × 每 SM 带宽反推出来的最小 TLP。

推论：**如果只靠 TLP（满 occupancy、每线程一次 scalar load），你刚刚好够，没有任何余量。** 这就是为什么"occupancy < 100%"对 vector kernel 特别敏感，而对 ALU-bound kernel 不敏感。

<details>
<summary>换个角度：ILP 如何替代 TLP（点开看数字）</summary>

把访存宽度或每线程的独立访存量提上去，需求线程数就往下掉：

| 每线程在飞字节 | 打满带宽所需 resident threads/SM | 对应 occupancy(H100) |
|---|---|---|
| 4 B（1×scalar load） | ~2048 | 100% |
| 16 B（1×`float4`） | ~512 | 25% |
| 16 B（4×独立 scalar load） | ~512 | 25% |
| 64 B（4×`float4`，软件流水） | ~128 | 6% |

所以 **"比 2048 小"不是问题，问题是"小下去之后 ILP/宽度有没有跟上"**。掉 occupancy 掉性能，本质是掉的是 $B_{\text{in-flight}}$，而不是线程数本身。
</details>

---

## 二、occupancy 掉到 2048 以下，具体伤害在哪

### 1. 带宽出现"拐点式"下降（最主要的效应）

带宽 vs occupancy 通常不是线性的：在拐点以上，带宽饱和、曲线是平的；跌破拐点，带宽随 $B_{\text{in-flight}}$ 近似线性下滑并伴随 scheduler 停顿。拐点位置约等于

$$\text{threads}_{\text{knee}} \approx \frac{\text{BW}_{\text{per SM}} \times t_{\text{DRAM}}}{\text{ILP} \times \text{width}}$$

scalar、无 ILP 的 vector kernel 拐点就在 100% 附近——这类 kernel 一掉 occupancy 就掉性能；而 `float4` + 展开的 kernel 拐点在 25% 甚至更低。

### 2. 延迟暴露 → scheduler 空转

每个 SM 有 4 个 sub-partition，每 cycle 每 partition 最多 issue 1 条 warp 指令。warp 少了，所有 warp 同时卡在长延迟 load 上时，partition 就出现 issue 气泡，IPC 直接掉。这跟第 1 点是同一件事的两个观察面。

### 3. ILP ↔ 寄存器 的负反馈

想用 ILP 补偿 occupancy，就要 unroll / 多路独立 load，这会吃寄存器；寄存器一多，occupancy 又被压得更低。存在一个最优点，靠拍脑袋很难找，通常要用 `__launch_bounds__` 或 ncu 扫。

### 4. 一个 CTA/SM 的退化情形（你说的那种）

当 CTA 大到 `threads_per_cta > 1024` 或寄存器/shared memory 用满，一个 SM 只能塞 1 个 CTA，此时：

- **wave quantization 变严重**：CTA 数不是 SM 数的整数倍时，最后一波有 SM 空转，尾部浪费占比约 $1/N_{\text{wave}}$。
- **CTA 内 barrier 变成全 SM barrier**：多个 CTA/SM 时，A 组在 `__syncthreads()` 停顿，B 组还能喂 issue slot；只有 1 个 CTA 时，一次 `__syncthreads()` 就是整个 SM 的空泡。
- **CTA 启动/排空无法重叠**：pipelining 断裂。

### 5. 对纯 ALU 型 vector 运算基本无影响

如果 kernel 是 compute-bound（FMA 密集、数据在寄存器里 reuse），只需要每个 partition 有 ~4–8 个 ready warp 就能盖住 ALU 的 4–6 cycle 延迟，也就是 512–1024 threads/SM 就够。**所以"上不满 2048"是 memory-bound vector 的痛点，不是 ALU-bound 的痛点。** 这个区分很重要。

### 6. CTA 尺寸的"整除效应"

2048 是个漂亮的上限，但 1536 threads 的 CTA 在 2048 的 SM 上只能放 1 个 → 实际 occupancy 75%；而且 SM 上不同 CTA 之间如果线程数不同，会出现 packing 碎片。选 128/256/512/1024 这类能整除的尺寸，能避免"明明还有线程额度却塞不进第二个 CTA"。

---

## 三、不同 accelerator 的 cap 不一样，怎么理解这种差异

cap 的数字本身不决定 vector 性能，**决定性能的是"内存系统能容忍多少在飞字节"，以及硬件提供了几种补齐它的手段**：

| 补齐手段 | 谁在用 | 作用 |
|---|---|---|
| 加 TLP（更多 resident warp） | NVIDIA Volta/Hopper 2048；AMD CDNA 也约 2048/CU | 直白地堆线程 |
| 加 ILP / 更大向量宽度 | 所有 GPU；`float4`/`int4`、128-bit load | 每线程在飞字节 ×4 |
| 异步拷贝解耦 | Ampere `cp.async`、Hopper **TMA** | 在飞字节由 async queue 决定，与线程寄存器解耦 → **occupancy 的重要性大幅下降** |
| 硬件预取 / 更大 MSHR、L1 | 各家不同 | 抬高单 warp 能同时在飞的事务数 |
| 软件流水 / 双缓冲 | 经典 GEMM、DMA 风格 kernel | 用寄存器当 buffer 换延迟 |
| 无 TLP 模型（systolic/vector machine，如 TPU 类） | 张量/向量加速器 | 没有 warp/occupancy 概念，"occupancy" 变成 tile size + double buffering，走 DMA 而非 per-thread load |

所以：
- 如果一家 accelerator 的 **threads/SM < 2048**，它要维持同样带宽就只有两条路：**更大的 ILP/向量宽度**，或**更多 MSHR / 更大的 L1 / 硬件预取**。反过来，如果它的 DRAM 延迟更低、每 SM 带宽更小（$B_{\text{in-flight}}$ 需求小），那 cap 低一点也没关系——**cap 是被 $\text{BW}_{\text{per SM}} \times t_{\text{DRAM}}$ 反推出来的，不是独立的性能指标**。
- 架构演进方向也印证这点：Hopper/TMA、Blackwell 的趋势是**用异步引擎接管延迟隐藏**，让 kernel 不再靠堆满 2048 threads 吃饭。

---

## 四、实践建议（针对 memory-bound vector kernel）

1. **先量 $B_{\text{in-flight}}$，不要盯 occupancy**：`threads × ILP × width × (line/BW × latency)` 算一遍，看差多少。
2. **能用 `float4`/`int4` 就用**：一行改动把所需线程数除以 4，拐点直接下移到 ~25%。
3. **手动 ILP**：单线程多路独立 load（unroll），配 `__launch_bounds__` 控寄存器，防止 occupancy 被 unroll 反噬。
4. **Hopper 上优先 TMA / `cp.async`**：把数据搬进 shared memory 再算，延迟隐藏从"线程级"变成"引擎级"，对 occupancy 不再敏感。
5. **CTA 尺寸选 128/256/512/1024**，避免 packing 碎片和 wave tail。
6. **grid-stride loop / persistent kernel**：减少 CTA 数少导致的 SM 空转和尾部量化。
7. **验证**：`ncu` 看 `Memory Throughput`、`Long Scoreboard` stall、achieved occupancy 三条曲线一起看，定位是"缺线程"还是"缺 ILP"。

一句话收敛：**2048 是 NVIDIA 用 TLP 覆盖 DRAM 延迟的"及格线"；跌破它不是原罪，跌破之后 $B_{\text{in-flight}}$ 没补上才是原罪。** 其他 accelerator 的 cap 可以不同，因为它们补 $B_{\text{in-flight}}$ 的手段（宽度、MSHR、异步引擎）配比不同。

---
## 🧑‍💻 User

不过我之前读某家accelerator模拟器看的是可以同时驻留64个warp，可以同时并行处理两个block，或者同时驻留32个block，有哪些因素会决定这些参数？

---
## 😀 默认助手 (7)

## 先对齐一下你看到的两个数

你描述的现象其实是**同一个 2048-thread 额度被两种方式"切"满**：

| 场景 | 每个 block | block 数 | 占用 warp | 谁在卡上限 |
|---|---|---|---|---|
| 大 block | 1024 threads = 32 warps | **2** | 64 | **warp/thread 槽位**（64 = 上限） |
| 小 block | 64 threads = 2 warps | **32** | 64 | **block 槽位**（32 = 上限） |

两种情况都停在 64 warp / 2048 thread，但**限制它们的物理资源完全不同**。这说明硬件里至少有四五个**互相独立**的资源池，最终驻留量是它们的 min：

$$N_{\text{CTA/SM}} = \min\Big( \underbrace{\left\lfloor \tfrac{T_{\max}}{T_{\text{CTA}}} \right\rfloor}_{\text{线程槽位}},\ \underbrace{\left\lfloor \tfrac{W_{\max}}{W_{\text{CTA}}} \right\rfloor}_{\text{warp 槽位}},\ \underbrace{\left\lfloor \tfrac{R_{\max}}{R_{\text{CTA}}} \right\rfloor}_{\text{寄存器}},\ \underbrace{\left\lfloor \tfrac{S_{\max}}{S_{\text{CTA}}} \right\rfloor}_{\text{shared mem}},\ \underbrace{B_{\max}}_{\text{block 槽位}} \Big)$$

$$W_{\text{resident}} = N_{\text{CTA/SM}} \times W_{\text{CTA}}$$

注意**一个 CTA 的 warp 不能跨 SM 拆分**（CTA 是最小放置单位）——这一条是后面所有"量子化浪费"的根源。

---

## 一、决定「64 warp / 2048 thread」的因素

这一层是**调度器为每个 warp 保存状态的容量**：

| 决定因素 | 具体内容 |
|---|---|
| **寄存器文件大小** | Hopper/Ampere 每 SM 64K×32-bit 寄存器 = 256 KB。$65536 / 2048 = 32$ regs/thread。**"2048 thread" 和 "32 regs/thread" 是同一个约束的两面**——RF 就这么大，想堆更多线程，平均寄存器预算就得再砍。 |
| **每 warp 的状态存储** | PC、active mask、warp ID、scoreboard、warp 所属 CTA/barrier、寄存器基址指针……64 份这样的表项就是 64 个 warp 槽。 |
| **每 SMSP 的 warp 槽** | H100：4 个 SMSP × 16 warp = 64。所以"64"不是一块整表，而是**每分区 16**，warp 按轮转分配到 4 个分区，单个 CTA 也会被均匀摊开。分区槽位会带来隐性上限（20 warps 不能全塞进一个分区）。 |
| **延迟隐藏需求** | 上一轮算过的 Little's Law：$\text{BW}_{per SM} \times t_{DRAM} \approx 8\text{–}9$ KB，而 $64\ \text{warp} \times 128\ \text{B} = 8$ KB。**64 warp 正好是"每 warp 一次未完成访存"覆盖 DRAM 延迟的容量**。 |
| **发射带宽** | 4 个调度器每 cycle 各发射 1 条 warp 指令 → 64 warp 给每个调度器 16 个候选，足以填充 FMA/访存两类停顿。 |
| **面积/功耗** | 越多的槽位 = 越大的调度表和 RF，纯粹的成本权衡。 |

> 关键洞察：**64 不是"性能目标"，而是"RF 面积 × 延迟隐藏需求"解出来的一个点。**

---

## 二、决定「32 block 槽位」的因素

block 槽位是**与 warp 无关**的一套独立硬件，因为每个 CTA 都需要自己的"户口"：

| 决定因素 | 为什么需要 |
|---|---|
| **CTA 元数据表** | 每个 resident CTA 要存 blockIdx(3D)、grid 信息、smem 基址/大小、退出状态、CTA 状态机 → 一块固定深度的表，深度就是 $B_{\max}$。 |
| **barrier / sync 资源** | 每个 CTA 有一套独立的 barrier 状态（`__syncthreads()`、named barrier）。历史上的 16 个硬件 barrier/SM 会把 CTA 数卡在 16——这也是 Turing 世代 block/SM = 16 而 Volta = 32 这一类差异的候选解释之一（究竟是 barrier 还是单纯面积取舍，我不能确定，两者都可能）。 |
| **shared memory 分配槽** | smem 以 per-CTA 为单位保留，分配器需要跟踪每个 CTA 的块。 |
| **GigaThread / 工作分发器的 CTA 描述符池** | SM 之外、GPU 级还要有一份可派发的 CTA 队列。 |
| **最小 block 粒度** | CTA 最小 1 warp = 32 thread。所以 **32 block × 1 warp 只能凑出 32 warp = 1024 thread，即 50% occupancy**——这正是"32"必须配一个"最小 block 别太小"的隐含前提。你模拟器里 32 block 恰好凑满 64 warp，说明它假定 block ≥ 2 warp。 |

**为什么需要单独一套 block 槽位？** 因为 CTA 是**放置与资源保留的单位**：smem、barrier、启动/退休都必须以 CTA 为粒度管理，这些状态没法用 warp 状态替代。

---

## 三、真正天天卡人的：寄存器与 smem 的「分配粒度」

上面三类是"显式上限"，但日常 occupancy 掉下来几乎都是它们 + **粒度舍入**造成的：

- **寄存器分配粒度**：**8 regs/thread，256 regs/warp**。所以一个用 33 个寄存器的线程，实际按 40 个算；一个 2 warp 的 block 至少按 512 寄存器记账。→ 寄存器用的是"向上取整"，不是精确量。
- **shared memory 分配粒度**：通常 **128 B 或 256 B** 向上取整。用 1 KB 的 CTA 实际按 1 KB 或 1.25 KB 记账，累加后就少放一个 CTA。
- **warp 槽位分配粒度**：部分架构按"组"分配 warp 槽（2/4 个一组），小 CTA 会有 packing 碎片。
- **每线程寄存器上限 255**：写满 255 regs 时，$65536/255 \approx 257$ → 一个 SM 连 512 thread 都放不下。

一个具体例子：1024-thread CTA（32 warps）× 64 regs/thread = 65536 regs = **整个 RF**，于是 $N_{\text{CTA/SM}} = 1$，occupancy 只有 50%——不是 thread 或 block 上限卡的，是寄存器。

---

## 四、更细一层：还有两个"上层"约束

<details>
<summary>per-SMSP 约束 与 CTA 分配的量子化</summary>

1. **SMSP 级约束**：CTA 的 warp 会被分散到 4 个 SMSP 上。若 CTA 只有 1 warp，它只能落在一个 SMSP，其余三个空转；32 个 1-warp CTA 也未必能均匀铺开（且被 32-block 上限卡住）。所以 block 上限太低 + block 太小 = 分区失衡。
2. **量子化浪费**：CTA 整体放置 ⇒ 1536-thread CTA 放进 2048-thread SM 会浪费 512 个线程槽（25%）。这就是为什么推荐 128/256/512/1024 这类"能整除 2048"的尺寸。
3. **wave 量子化**：$N_{\text{CTA}}$ 不是 SM 数整数倍时，最后一波有 SM 空转，浪费 $\approx 1/N_{\text{wave}}$。
</details>

<details>
<summary>Hopper 及以后新增的一层：Thread Block Cluster</summary>

Hopper 引入 cluster：最多 8（Blackwell 更多）个 CTA 作为一个**原子调度单元**，必须同时驻留在**同一个 GPC 内的不同 SM** 上，并可访问彼此的 distributed shared memory。这会新增一个约束：

$$N_{\text{cluster}} \le \min\left(\frac{\text{SM}_\text{GPC}}{\text{CTA}_\text{cluster}},\ \dots\right)$$

即 GigaThread 必须找到一个"空闲 SM 数足够整"的 GPC 才能下发 cluster。所以 block 槽位之外，又多了 GPC 级的空间约束。
</details>

---

## 五、把参数映射回设计动机

| 参数 | 主要决定者 | 本质 |
|---|---|---|
| threads/SM = 2048 | RF 大小 ÷ 每线程寄存器预算 | 面积约束 |
| warps/SM = 64 | 延迟隐藏所需 TLP（Little's Law）+ 调度表深度 + 4×16 分区结构 | 性能约束 |
| warps/block ≤ 32 | 单一 CTA 必须能放进一个 SM；barrier/smem 规模 | 编程模型约束 |
| blocks/SM = 32 | CTA 元数据表 + barrier 资源 + 分发器槽位 | 记账面积约束 |
| regs/CTA, smem/CTA | 分配粒度向上取整（8 regs/thd、256 regs/warp、128/256 B） | 常见真实瓶颈 |

不同世代这几个数的差异（Turing 1024/32/16、Ampere 消费者 1536/48/16、Hopper 2048/64/32）正说明它们是**各自独立的成本/收益权衡**，不是一起缩放的。

---

## 六、其他 accelerator 和模拟器里的建模

模拟器通常就把它建成一张资源表：

```text
num_sm, threads_per_sm, warps_per_sm, blocks_per_sm,
regs_per_sm, regs_per_thread_max, reg_alloc_granularity,
smem_per_sm, smem_alloc_granularity,
warps_per_smsp, max_warps_per_block, max_threads_per_block
```

然后 occupancy 求解器对每个 CTA 配置取 min（就是本文开头的公式）。**读懂一个模拟器的 occupancy 行为，只要问三句话**：三个池子（thread / warp / block）各是多少？寄存器和 smem 的粒度是多少？CTA 是否作为不可拆分单位放置？

<details>
<summary>其他厂商的组织方式（数字供参考，非精确规格）</summary>

- **AMD CDNA (MI 系列)**：概念对应 CU 的 resident wavefront，也是 ~2048 threads/CU，wave64，另有 workgroup 数上限；寄存器以 vector+scalar 两套池子分别记账——比 NVIDIA 多一维资源。
- **AMD RDNA**：以 WGP 为调度单位，wave32，threads/CU 更小，资源表结构又是另一套。
- **Intel Xe**：TLP 粒度在 EU / Xe-core 层，模型与"SM + block"不同构。
- **Systolic / vector 加速器（如 TPU 类）**：**根本没有 warp/occupancy 概念**，"occupancy" 退化成 tile size + double buffering 深度，在飞字节由 DMA 引擎而非 per-thread load 提供。这正好呼应上一轮的结论：**cap 数字是内存系统参数的函数，不是独立指标。**
</details>

---

## 一句话总结

**64 warp / 2048 thread 是"寄存器文件大小 × DRAM 延迟隐藏需求"解出来的；32 block 是"CTA 级元数据 + barrier 记账"的独立面积预算。** 硬件里是三个（其实是五六个）互不相干的池子取 min，模拟器把它们分别建模成 `warps_per_sm`、`threads_per_sm`、`blocks_per_sm` 加粒度参数——你看到的"2 个 block"和"32 个 block"只是同一个 64-warp 额度被不同粒度的 CTA 切满而已。
