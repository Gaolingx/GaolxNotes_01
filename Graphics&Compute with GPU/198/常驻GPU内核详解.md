# 常驻GPU内核详解

## 🧑‍💻 User

什么是Persistent GPU kernel

---
## 😀 默认助手 (7)

## 一句话定义

**Persistent GPU kernel（常驻核函数）** 是一种 GPU 编程模式：kernel 启动时申请的 grid 大小 **刚好能被 GPU 一次性全部调度到 SM 上常驻**，kernel 进入后不退出，而是在内部循环反复"领取任务"处理，直到任务耗尽或收到终止信号才结束。它把"一次启动处理一批数据"改成"一次启动，长期驻留，持续领取任务"。

<details>
<summary><b>一、和普通 kernel 的核心区别</b></summary>

| 维度 | 普通 kernel | Persistent kernel |
|---|---|---|
| grid 大小 | 由数据量决定，可远超驻留容量 | 固定为"刚好全部驻留"的上限 |
| 调度方式 | 硬件把 block 分批（wave）调度 | 所有 block 同时驻留，一次调度 |
| 任务获取 | 由 `blockIdx` 静态映射 | 运行时从全局队列动态领取（atomic） |
| 跨 block 同步 | 只能靠多次 launch 分段 | 可在 kernel 内做 grid barrier |
| 启动开销 | 每个 kernel 一次 launch 成本 | 摊销到整个长生命周期 |
| 尾部效应 | 有（最后一个 wave 不满） | 基本消除 |

</details>

<details>
<summary><b>二、工作原理的三个关键机制</b></summary>

### 1. 驻留（residency）

必须满足

$$\text{gridSize} \le \text{SM 数量} \times \text{每个 SM 可驻留的 block 数}$$

每 SM 能驻留多少 block 由寄存器用量、shared memory 用量、thread 数共同决定，可用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 精确算出：

```cpp
int blocksPerSm = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocksPerSm, persistent_kernel, blockSize, /*dynamicSmem=*/0);
int gridSize = blocksPerSm * prop.multiProcessorCount;  // 上限
```

这是**安全性条件**，不只是性能条件：一旦 grid 超过驻留容量，后面的 block 要等前面的 block 退出才能上 SM，此时任何"跨 block 互相等待"的逻辑都会**死锁**。

### 2. 全局任务队列 + 原子领取

标准做法是一个 global memory 里的计数器，每个 block 用 `atomicAdd` 抢一个任务编号，实现动态负载均衡：

```cpp
__global__ void persistent_kernel(const Task* tasks, int* counter, int numTasks) {
    __shared__ int taskId;
    while (true) {
        if (threadIdx.x == 0)
            taskId = atomicAdd(counter, 1);   // 领号
        __syncthreads();
        if (taskId >= numTasks) return;       // 队列空，退出
        process(tasks[taskId]);
        __syncthreads();                      // 复用 shared mem 前必须同步
    }
}
```

这就是经典的 **persistent thread blocks / dynamic work distribution**，比静态 `blockIdx` 映射更能容忍任务粒度不均。

### 3. Grid-wide barrier（可选但常配套）

因为所有 block 保证同时在跑，可以安全实现**全局屏障**，让算法在 kernel 内做多轮迭代而无需反复 launch。CUDA 9+ 通过 cooperative groups 提供：

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::this_grid().sync();   // 需要以 cudaLaunchCooperativeKernel 启动
```

`cudaLaunchCooperativeKernel` 会**在启动时校验** grid 是否真的能全部驻留，不满足直接报错 —— 这是防死锁的官方保险。注意 `grid.sync()` 要求 `CG_GRID_DIM` 级别的同步能力，且必须整体 launch。

</details>

<details>
<summary><b>三、典型使用场景</b></summary>

1. **多轮迭代的算法**：迭代求解器、稀疏线性代数、PageRank、BFS/图遍历、多时间步 stencil —— 用 grid barrier 把"多次 launch"压成一次。
2. **细粒度、短小、成链的 kernel**：单个 kernel 只有几微秒时，launch 开销占比极高，persistent 化收益明显（小 batch 推理、RL 小 kernel 串、decoding 阶段的逐 token 生成）。
3. **通信-计算重叠**：常驻 kernel 在等待 NVLink/RDMA 数据时继续处理队列中其他任务（NVSHMEM、部分 attention/all-reduce 实现）。
4. **Megakernel / 全模型单 kernel**：把整层甚至整个前向压进一个 persistent kernel，配合设备端任务图消除中间 launch。
5. **动态负载不均的问题**：如稀疏矩阵 SpMV、不规则图计算，用原子领号替代静态切分。

</details>

<details>
<summary><b>四、优点</b></summary>

- **消除 launch 开销与 launch latency**，尤其是短 kernel 链。
- **消除尾部效应 / wave quantization**：不会出现"最后一个 wave 只有 30% 的 SM 有活干"。
- **保留片上状态**：寄存器、shared memory、warp 上下文跨轮次存活，省去重复初始化。
- **数据驻留更久**：任务按缓存局部性排布可持续吃到 L2 命中。
- **可实现 grid 级同步**，简化多阶段流水线。

</details>

<details>
<summary><b>五、缺点与坑（这一节最重要）</b></summary>

- **死锁是首要风险**：grid 超过驻留容量 + 任何跨 block 依赖（grid barrier、spin-wait、生产者-消费者）= 挂死。务必用 occupancy API 算上限，或直接用 cooperative launch 让运行时校验。
- **占用率可能下降**：为了"全部驻留"，每 SM 能放的 block 数被限制，可能低于最优 occupancy；且每个 block 常驻意味着 warp 数少、延迟隐藏能力弱，需靠更多 ILP 弥补。
- **吃满整卡资源**：persistent kernel 通常占满全部 SM，**排斥并发**——与其他流、CUDA Graphs、MPS 多租户、其他进程难以共存。
- **编程复杂**：手写队列、原子、memory fence（`__threadfence()`、acquire/release 语义）、负载均衡、终止条件；正确性验证困难。
- **调试与工具不友好**：`cuda-gdb`、profiler 对超长 kernel 支持差；可视化时间线拉得很长。
- **长跑风险**：Windows 显示驱动 TDR 默认约 2 秒会杀掉长 kernel；即便现代 GPU 支持抢占，长驻 kernel 仍影响共享与响应性。
- **可移植性**：驻留上限强依赖具体架构（SM 数、每 SM 资源），代码需运行时查询参数，不能写死常数。

</details>

<details>
<summary><b>六、容易混淆的概念辨析</b></summary>

- **Grid-stride loop ≠ persistent kernel**。grid-stride 只是"少开 block、每个线程循环处理多个元素"，grid 通常仍大于驻留容量，block 仍分 wave 调度。它是**轻量替代品**，能减少 block 数与尾部效应，但**不能**做 grid barrier，也不能保证"全部同时在跑"。很多号称 persistent 的代码其实只是 grid-stride loop。
- **CUDA Graphs**：解决的是同一问题（launch 开销），但走的是完全不同的路线——用图捕获把多次 launch 合并为一次提交，**不改变计算形态**。工程上常优先选 CUDA Graphs，因为它简单、无死锁风险、可与多流共存；只有在需要 grid barrier 或 in-kernel 动态决策时才用 persistent kernel。
- **Megakernel**：persistent kernel 的一种激进应用形态（整个模型一个 kernel），难点在于层间依赖与同步建模。
- **Atomic-based work stealing 是 persistent kernel 的必要配套**，静态 `blockIdx` 划分就退化成了普通 grid。

</details>

<details>
<summary><b>七、实践检查清单</b></summary>

1. 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算 `gridSize` 上限，绝不硬编码。
2. 有跨 block 同步 → 强制用 `cudaLaunchCooperativeKernel`，让运行时替你校验。
3. 队列计数器必须在 kernel 启动前用 `cudaMemset` 或单独 kernel 清零。
4. 每轮循环的 shared memory 复用必须配 `__syncthreads()`，否则出现数据竞争。
5. 终止条件要"全局可见"：用原子操作或 volatile 读，必要时加 `__threadfence()`。
6. 先测有没有必要：如果 kernel 单次运行已 > 50–100 µs，launch 开销占比很小，收益可能为负。
7. 先考虑 CUDA Graphs / grid-stride loop 这类低风险方案，persistent 是"最后的手段"。

</details>

**总结**：persistent GPU kernel 的本质是**用"常驻 + 自取任务"换取"零重复启动开销"和"kernel 内全局同步能力"**，代价是必须精确控制驻留规模、承担死锁与占用率风险，并与 GPU 的共享/抢占机制天然冲突。它在需要 in-kernel 迭代或微秒级 kernel 串联的场景收益最大，在普通大 kernel 上则往往得不偿失。

---
## 🧑‍💻 User

和megakernel的概念有什么区别

---
## 😀 默认助手 (7)

## 一句话区别

- **Persistent kernel 是"调度形态"**：这个 kernel 的 grid **刚好全部驻留**在 SM 上，它因此能长期存活、循环领任务、做全局同步。它不关心这个 kernel 干了几件事。
- **Megakernel 是"程序划分形态"**：把**整个程序（如整个模型的前向、整帧渲染）压进一次 kernel launch**，消灭 kernel 之间的边界。它不关心 grid 是否驻留。

一个是"这个 kernel 怎么被调度、活多久"，一个是"程序被切成几个 kernel"。二者**经常同时出现，但不是同一件事**。

<details>
<summary><b>一、先看 2×2：两个维度是正交的</b></summary>

| | **非 megakernel**（程序仍是多阶段 / 单阶段） | **Megakernel**（整个程序 = 1 次 launch） |
|---|---|---|
| **非 persistent**<br>grid 可 > 驻留，分 wave 调度 | 普通 kernel、grid-stride loop、算子融合 | **单 launch 融合 kernel**：阶段间依赖只是 block 内局部依赖，各 block 自洽跑完，硬件照常分 wave |
| **Persistent**<br>grid = 驻留上限，常驻 | **经典 persistent kernel**：工作队列 + `atomicAdd` 领号，但只做一类任务（如 SpMV、attention、图遍历的一步） | **真·persistent megakernel**：单 launch + 常驻 + `grid.sync()` 把各阶段串起来，中间结果留在片上 |

四个格子都能落地，所以**不能互相定义**：

- 大量只做一件事的 persistent kernel（第四象限那些）根本不是 megakernel；
- 一个巨大 grid、被硬件分上百个 wave 跑完的融合 kernel，是 megakernel 但**完全不 persistent**。

</details>

<details>
<summary><b>二、为什么这两个词总被混着叫</b></summary>

因为**想实现真正的 megakernel，你通常被迫变成 persistent**。

Megakernel 要把阶段 A 的输出交给阶段 B，中间不能靠"kernel 退出 + 下一 kernel 重读 global memory"，就必须在 kernel 内做**跨 block 全局同步**。而全局屏障要求所有 block 同时在跑 —— 否则先到的 block 在自旋等后到的 block，而后到的 block 根本排不上 SM，直接**死锁**。

于是链条是：`megakernel（需跨阶段同步） → 需 grid barrier → 需全部驻留 → 需 persistent`。

反过来不成立。这就是"persistent megakernel"这个合成词流行的原因，也是读到论文标题时必须追问一句的原因：

- 看到 **"persistent kernel"** → 问：**它融合了几个阶段？**（可能只是一个算子）
- 看到 **"megakernel"** → 问：**它的 grid 驻留吗？**（可能是 wave 调度的单 launch）

</details>

<details>
<summary><b>三、设计目标不同，收益来源也不同</b></summary>

| | Persistent kernel | Megakernel |
|---|---|---|
| 主要想省掉 | **launch 开销、尾部效应、block 调度开销** | **kernel 边界本身**：中间结果不再写回 HBM 再读回来 |
| 次要收益 | 片上状态（寄存器/smem/warp 上下文）跨轮复用 | 阶段间流水、无需 host 往返、可做全程序级调度 |
| 核心约束 | **驻留容量**（occupancy 计算） | **全局同步点建模 + 资源预算取"最重阶段"** |
| 典型使能技术 | `atomicAdd` 工作队列、`cudaLaunchCooperativeKernel` | `grid.sync()`、设备端任务图、DSL/编译器（把算子图编译成单 kernel） |

一个直观的差别：persistent kernel 的收益在**时间维度**（少等几次启动），megakernel 的收益在**空间维度**（少走几趟 DRAM）。所以 memory-bound 的低 batch LLM decoding 是 megakernel 的甜点区 —— 每个 token 的 kernel 只有几微秒，launch 开销与中间张量往返都被放大；而在一个已经跑 200 µs 的大 GEMM 上，两个概念都基本没收益。

</details>

<details>
<summary><b>四、失败模式的差别（最实用的部分）</b></summary>

**Persistent kernel 的坑：**
- 死锁（grid 超驻留 + 跨 block 等待）；
- occupancy 被"必须全驻留"这一条压住，延迟隐藏变差；
- 吃满整卡、排斥多流 / CUDA Graphs / MPS 多租户；
- 长跑撞 TDR、难调试。

**Megakernel 独有的坑（更致命）：**
- **不能用库**。不能调 cuBLAS / cuDNN / FlashAttention，所有算子必须自己手写并自己调优 —— 这是 megakernel 在工程上最常翻车的地方，手写 GEMM 很难打赢高度调优的库。
- **资源预算取最坏情况**。一个 kernel 要容纳所有阶段的寄存器 / shared memory 需求，寄存器压力按最重阶段算，occupancy 被整体拉低。
- **并行度被串行化**。原本各阶段可以各自开满最优 grid，融合后共用同一批 block，阶段间还被 barrier 隔断。
- **SIMD 效率可能下降**。经典反例是路径追踪：Laine、Karras、Aila 的 *Megakernels Considered Harmful*（HPG 2013）就指出，megakernel 把不同材质/不同着色逻辑塞进同一 warp 会造成严重 divergence，反而不如 wavefront 的多 kernel 组织。
- **编译期/代码体积爆炸**，调试几乎不可行。

工程含义：**persistent kernel 是可以局部采用的优化手段；megakernel 相当于自己写一个编译器/DSL**（近年把 LLM 推理编译成 megakernel 的工作如 Mirage，本质就是在做这件事）。绝大多数团队的正确顺序是：CUDA Graphs → 算子融合 → grid-stride → persistent kernel 局部化 → 才轮到 megakernel。

</details>

<details>
<summary><b>五、和几个邻近概念划清边界</b></summary>

- **算子融合（kernel fusion）**：megakernel 的"局部版"，只合并相邻两三个算子，通常也不需要全局同步、不要求驻留。
- **CUDA Graphs**：和 megakernel 目标重叠、路径相反 —— 保留多 kernel，只砍掉提交开销。低风险、无死锁、可与多流共存，因此现实中往往是 megakernel 的**替代方案**。
- **Grid-stride loop**：既不是 persistent 也不是 megakernel，只是少开 block 的循环写法。
- **Dynamic parallelism（设备端 launch）**：另一种"一个 kernel 里跑多阶段"的实现路线，靠硬件在 device 上发起子 kernel，不需要常驻假设，但开销与嵌套深度受限。
- **OS 里的类比**很好用：persistent kernel ≈ 常驻线程池 + 任务队列（线程不退出，任务自己领）；megakernel ≈ 单体内核（monolithic kernel）—— 名字本来就是从操作系统设计里借来的。

</details>

**结论**：persistent 回答"**这个 kernel 怎么活**"（驻留、循环、全局同步），megakernel 回答"**程序被切成几块**"（只切一块）。因为跨阶段全局同步必须依赖驻留，megakernel 在实践中通常蕴含 persistent；但 persistent 完全可以在一个只干一件事的小 kernel 上使用。判断一个方案时，**分别问这两个问题**，就不会被术语带偏。
