# Grouped GEMM 性能优势

### 🧑‍💻 User

GroupedGEMM for MoE为什么相较于启动topk专家个独立的batch gemm，性能更高？

---

### 🤖 Assistant

GroupedGEMM（或称为 Grouped GEMM）在 Mixture of Experts (MoE) 模型中表现优于启动多个独立的 GEMM（即针对每个专家启动一个 Kernel，或者尝试使用 padding 后的 Batched GEMM），其核心原因可以归结为以下几点：**消除 Kernel 启动开销**、**解决负载不均衡（Load Imbalance）**以及**避免无效计算**。

以下是详细的技术分析：

### 1. 变长输入问题 (Variable Sequence Length)

在 MoE 中，Token 会经过 Gating Network 被路由到不同的专家（Experts）。这是一个动态过程，导致分配给每个专家的 Token 数量（即矩阵乘法中的 $M$ 维度）是**动态且不均匀**的。

*   **Expert A** 可能分到了 10 个 Token。
*   **Expert B** 可能分到了 1000 个 Token。

传统的 `cublasGemmBatched` 或 `torch.bmm` 通常要求批次中所有矩阵的维度 $(M, N, K)$ 是完全一致的。

#### 方案 A：Padding + Batched GEMM (低效)
为了使用标准的 Batched GEMM，你必须将所有专家的输入 Pad 到最大的那个专家的长度（例如 1000）。
*   **缺点**：大量的无效计算（Compute on zeros）。如果负载极度不均衡，浪费的算力可能高达 50% 以上。

#### 方案 B：For-loop 单独启动 GEMM (低效)
也就是你提到的“启动 topk 专家个独立的 GEMM”。针对每个专家，根据其实际 Token 数量调用一次 cuBLAS。
*   **缺点**：遭遇 Kernel Launch Overhead 和 GPU 尾部效应（详见下文）。

#### 方案 C：GroupedGEMM (高效)
GroupedGEMM 允许在单个 Kernel 中计算多个 $M, N, K$ 各不相同的矩阵乘法。

---

### 2. Kernel Launch Overhead (CPU瓶颈)

如果模型有 $E$ 个专家（例如 $E=64$ 或更多），在方案 B 中，你需要在一个 Forward pass 中启动 $E$ 个 CUDA Kernel。

*   **问题**：CUDA Kernel 的启动是由 CPU 下发的。当每个 Kernel 的计算量（Workload）很小的时候（例如 Inference 阶段，Batch size 较小，分到每个专家的 Token 很少），**CPU 启动 Kernel 的时间可能比 GPU 执行 Kernel 的时间还要长**。
*   **GroupedGEMM 优势**：它将所有专家的计算打包成**一个** CUDA Kernel Launch。CPU 只需要下发一次指令，极大地减少了 CPU 侧的延迟，这对于小 Batch 推理至关重要。

---

### 3. GPU SM 利用率与“波浪量化”效应 (Wave Quantization)

这是性能差异最深层的原因。GPU 由许多流多处理器（SMs）组成。

#### 独立 GEMM 的情况 (串行或多 Stream)
假设你有 4 个专家，GPU 有 80 个 SM。
1.  **Expert 1 (小任务)**：只占用了 10 个 SM。剩下的 70 个 SM 处于空闲状态，或者需要等待内存加载。
2.  **Expert 2 (大任务)**：占满 80 个 SM，并且需要运行很久。
3.  **同步屏障**：通常框架在层结束时需要同步。

即使使用 CUDA Streams 并行启动，由于 GPU 硬件调度器的限制，多个小 Kernel 往往难以完美地“拼图”填满所有 SM。此外，**尾部效应 (Tail Effect)** 非常明显：整个 MoE 层的延迟取决于最慢的那个专家（Straggler）。当大专家的 Kernel 在运行时，处理完小专家的 SM 可能会空闲下来，无法立即转去帮助大专家（因为它们属于不同的 Kernel Grid）。

#### GroupedGEMM 的情况
GroupedGEMM 将所有专家的计算任务视为一个巨大的“任务池”（Task Pool of Thread Blocks）。

*   **统一调度**：GPU 内部的硬件调度器可以将属于 Expert 1 的 Thread Block 和属于 Expert 2 的 Thread Block 同时调度到不同的 SM 上执行。
*   **自动负载均衡**：如果 Expert 1 的计算很快完成了，释放出来的 SM 会立即从任务池中抓取 Expert 2（那个大任务）剩余的 Block 来执行。
*   **结果**：它像玩“俄罗斯方块”一样完美地填满了 GPU 的时空资源图，消除了碎片化的空闲时间。

---

### 4. 显存访问与 L2 Cache

*   **权重加载**：虽然不同专家权重不同，但在 GroupedGEMM 中，Kernel 可以更智能地管理 L2 Cache。
*   **输入复用**：如果使用的是 Shared Experts 或者某些参数共享机制，GroupedGEMM 能更容易地在寄存器或 Shared Memory 层面复用数据，而独立的 Kernel 很难跨 Kernel 共享数据。

### 总结对比

| 特性 | 独立的 Batch GEMM (Loop/Streams) | GroupedGEMM |
| :--- | :--- | :--- |
| **Kernel 启动次数** | $N$ (专家数量)，CPU 开销大 | **1**，CPU 开销极小 |
| **输入维度 ($M$)** | 必须相同 (需 Padding) 或各自独立 | **支持动态变长**，无需 Padding |
| **GPU 并行度** | 难以在 Kernel 间自动填补空闲 SM | **自动流水线填补**，SM 利用率极高 |
| **主要瓶颈** | Kernel Launch Latency, DRAM Bandwidth (碎片化) | Compute Bound / DRAM Bandwidth (饱和) |

因此，**GroupedGEMM** 实际上是针对 MoE 这种“**多路、变长、细粒度**”计算特性专门定制的算子，它通过**合并计算指令**和**统一硬件调度**，解决了传统 GEMM 无法处理动态负载的问题。目前的实现主要来自于 **CUTLASS** 库以及 **MegaBlocks** 等项目。

---

### 🧑‍💻 User

在deepseek-ai的deepgemm（https://github.com/deepseek-ai/DeepGEMM）中，这具体是怎么实现top-k个专家的fused gemm的？

---

### 🤖 Assistant

在 DeepGEMM 的实现中，Top-k 个专家的 Fused GEMM（即 Grouped GEMM）的核心实现机制并非依赖于 CUDA 提供的原生 Batched GEMM API，而是基于 **Persistent Threads（持久线程）** 模式配合 **Soft-Scheduler（软件调度器）** 来实现的。

这种方法允许在一个 CUDA Grid 中完成多个形状不同（M 维度变长）的矩阵乘法，从而避免了多次 Kernel 启动的开销，并最大化了 GPU 利用率。

以下是结合你提供的代码片段（主要是 `scheduler.cuh` 和 kernel 实现文件）的具体实现分析：

### 1. 核心数据结构：`Scheduler` (`scheduler.cuh`)

这是整个 Grouped GEMM 的大脑。在 MoE 场景下，我们通常使用 `GemmType::MGroupedContiguousWithPsumLayout` 或 `MGroupedMasked`。

`Scheduler` 结构体负责将线性的 CUDA Block Index (`blockIdx.x`) 映射到逻辑上的 **(Expert_ID, M_Chunk_ID, N_Chunk_ID)**。

#### 关键逻辑：`get_next_block`
在 `scheduler.cuh` 中，`get_next_block` 函数由每个 Thread Block 调用，以获取下一个要执行的任务。

```cpp
// scheduler.cuh 片段逻辑简化
__device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // 1. 获取下一个线性的全局任务 ID (Persistent Threading 模式)
    const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

    // ... (针对不同 GemmType 的分支)

    // 2. 针对 MoE (MGroupedContiguousWithPsumLayout) 的逻辑
    // grouped_layout 数组存储了每个专家处理的 token 数量的前缀和 (Prefix Sum)
    else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
        while (true) {
            // 如果当前线性 ID 小于当前专家组的任务总数，说明还在当前专家任务内
            if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                break;

            // 否则，切换到下一个专家 (Group)
            if (++ current_group_idx == kNumGroups)
                return false; // 所有专家都计算完了

            // 读取下一个专家的 token 数量 (M 维度)，计算该专家需要多少个 M-Block
            last_psum_m = align(current_psum_m, 128u);
            current_psum_m = __ldg(grouped_layout + current_group_idx);
            current_m_block_cumsum += num_m_blocks; // 累加之前的 block 数
            num_m_blocks = ceil_div(current_psum_m - last_psum_m, BLOCK_M);
        }
      
        // 3. 将线性 ID 转换为当前专家内部的 (m_block, n_block) 坐标
        get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);
      
        // 加上全局的 M 偏移量，使得不同专家的计算对应到 Global Memory 的不同位置
        m_block_idx += last_psum_m / BLOCK_M;
    }
    return true;
}
```

**原理解析：**
*   **输入准备**：在 Kernel 启动前，CPU 端会将 Token 根据路由结果进行排序（Permutation），使得属于 Expert 0 的 Token 连在一起，Expert 1 的连在一起，以此类推。这形成了一个巨大的 `(Total_M, K)` 的输入矩阵。
*   **grouped_layout**：这是一个位于 GPU 显存上的数组，记录了每个专家负责的 Token 范围（通常是 Offset 或 Length）。
*   **动态切换**：当一个 Thread Block 完成了 assigned 的工作后，它不会退出，而是再次调用 `get_next_block`。如果当前专家的工作被分派完了，调度器会自动让该 Thread Block 去计算下一个专家的部分数据。这实现了完美的负载均衡。

### 2. Kernel 执行循环 (`sm90_bf16_gemm.cuh` 等)

Kernel 本身不再假设这是一个单一的矩阵乘法，而是作为一个“Worker”去不断向 Scheduler 索取任务。

```cpp
// sm90_bf16_gemm.cuh 片段
// 初始化 Scheduler
auto scheduler = Scheduler<...>(shape_m, shape_n, shape_k, grouped_layout);

// Persistent Thread Loop (核心循环)
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // 这里 m_block_idx 已经是计算了全局偏移后的索引
  
    // ... TMA (Tensor Memory Accelerator) 加载数据 ...
    // scheduler.current_group_idx 知道当前是哪个专家，可以用于 Batch GEMM 模式下加载不同的权重
  
    // ... WGMMA (Warpgroup Matrix Multiply Accumulate) 计算 ...
  
    // ... Epilogue 结果写回 ...
}
```

### 3. 数据寻址与 TMA (`tma_copy`)

在 MoE 中，不同的专家有不同的权重（Weight $B$），但可能共享输入（Activation $A$ 的不同切片）。DeepGEMM 处理这种寻址的方式如下：

1.  **输入 Activation ($A$)**：通常是一个巨大的连续 Buffer（Contiguous Layout）。
    *   Scheduler 计算出的 `m_block_idx` 是加上了 `last_psum_m` 偏移的。
    *   因此，TMA 直接从大 Buffer 的正确偏移处加载数据，就像在切蛋糕一样。

2.  **专家权重 ($B$)**：
    *   如果 `GemmType` 是 `Batched` (即 MoE 权重不同)，代码中会使用 `scheduler.current_group_idx`。
    *   在 `tma_copy` 调用中：
        ```cpp
        // sm90_bf16_gemm.cuh
        const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
        // ...
        tma_copy<...>(..., batch_idx);
        ```
    *   TMA 描述符（Descriptor）被配置为 3D 模式，其中第 3 维（Z轴）就是 `batch_idx`（专家 ID）。这样，当 Scheduler 切换专家时，TMA 会自动从显存中加载对应专家的权重矩阵。

### 4. 总结：DeepGEMM 实现 Top-k MoE Fused GEMM 的关键点

1.  **Layout 预处理**：要求输入 Token 按照专家顺序连续排列（Contiguous Layout），并提供一个 `grouped_layout` 数组描述每个专家的长度。
2.  **Unified Grid**：只启动**一个** CUDA Kernel，Grid Size 通常设为 GPU SM 数量的倍数（保证占满 GPU），而不是专家数量或 Token 数量。
3.  **Software Pipelining & Scheduling**：
    *   Kernel 内部是一个 `while` 循环。
    *   `Scheduler` 结构体在运行时动态判断当前 Thread Block 应该处理哪个专家的哪一块数据 (`Tile`)。
    *   消除了 Block 之间的同步开销，也消除了尾部效应（Tail Effect），因为先做完 Expert A 的 SM 会立刻无缝切换去帮助计算 Expert B。
4.  **TMA 3D Loading**：利用 Hopper 架构的 TMA 特性，通过 `batch_idx` 快速切换不同专家的权重矩阵，无需 CPU 介入重设指针。

这种实现方式是目前 NVIDIA Hopper 架构上处理变长、多任务 GEMM（如 MoE）最高效的范式之一。