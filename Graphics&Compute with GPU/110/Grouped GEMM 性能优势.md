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