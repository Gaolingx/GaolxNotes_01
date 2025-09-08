# Stream-K技术提升GEMM性能解析

### 🧑‍💻 User

gemm中的stream-k是什么，他是如何提高gemm性能的？附上cutlass samples。

---

### 🤖 Assistant

好的，我们来详细解析一下 GEMM 中的 Stream-K 技术，它是如何提高性能的，并附上相关的 CUTLASS 示例代码。

---

### 1. 什么是 GEMM 中的 Stream-K？

**Stream-K** 是 NVIDIA 的 CUTLASS 库中实现矩阵乘法（GEMM）的一种新颖工作分配策略。它的核心思想是**打破传统的固定、粗粒度的块状划分方式，采用一种更细粒度、动态的“流式”分配方式**，以解决在特定问题规模下 GPU 计算资源（尤其是 Warp）负载不均的问题，从而显著提升性能。

要理解 Stream-K，我们首先需要了解传统的 GEMM 并行化策略（在 CUTLASS 中称为 **“Streaming Multiprocessor (SM)” 策略**）：

#### 传统策略 (SM 策略)
传统的 GEMM 内核（例如 CUTLASS 中的 `gemm_kernel`）会将整个矩阵乘法问题 `C = A * B` 划分为大量固定大小的 **“GEMM 块”**（Tile）。
*   每个 **块（Tile）** 负责计算输出矩阵 `C` 中的一个子矩阵。
*   每个块的计算会由一个 **线程块（Thread Block）** 来执行。
*   这些线程块被静态地、均匀地分配到 GPU 的多个流式多处理器（SM）上。
*   每个 SM 会处理分配给它的一个或多个线程块。

**传统策略的缺点：**
当矩阵的维度（M, N, K）不是 Tile 大小的整数倍时，会出现**尾部效应**。有些 Warp 分配到的 Tile 是不完整的（例如，在矩阵边缘），它们需要处理的有效数据较少，但依然要启动并消耗与完整 Tile 相同的计算和内存访问开销。这会导致这些 Warp 的计算资源（ALU）利用率极低，许多线程在等待或做无效工作，从而拉低了整体性能。在极端情况下（例如非常小的矩阵或某些维度很小），问题会更为突出。

#### Stream-K 策略的创新
Stream-K 采取了完全不同的视角。它不再将问题划分为固定的输出块，而是**沿着内部维度 K 进行划分**。

*   **将计算视为“流”**：它将整个 GEMM 计算视为在 K 维度上的一系列累加操作。每个基本的计算单元是沿着 K 维度的一个“片段”（Segment）。
*   **动态工作分配**：它创建一个全局的、细粒度的工作包（Work Package）池。每个工作包代表了对整个输出矩阵 `C` 的**一部分计算**，但只负责 K 维度上的一个片段。一个完整的输出 Tile 的计算需要累加多个这样的工作包。
*   **协作完成**：GPU 上的多个 Warp 会动态地从全局队列中领取这些工作包。一个 Warp 可能为输出矩阵 `C` 的多个不同部分贡献计算，也可能多个 Warp 协作完成同一个输出块在 K 维度上的累加。

简单比喻：
*   **传统策略 (SM)**：就像把一块大地（输出矩阵 C）划分成许多固定的田块（Tiles），每队工人（Thread Block）负责耕完自己的一块或多块田。如果田块边缘不规则，有些队里的工人（Warp）就会闲着。
*   **Stream-K 策略**：不再分田到队。而是把耕作任务（在 K 上累加）拆成很多份细活（工作包）。所有工人（Warp）从一个公共任务池里抢活干，大家一起流水线作业，最终共同耕完整块地。这样几乎没有一个工人会闲着。

---

### 2. 它是如何提高 GEMM 性能的？

Stream-K 通过以下机制显著提升性能，尤其是在问题规模不理想的情况下：

1.  **极致的负载均衡**：
    这是最核心的优势。由于工作是细粒度的并且是动态分配的，**几乎消除了所有 Warp 的空闲时间**。只要全局还有未完成的工作包，空闲的 Warp 就可以立刻领取并执行，避免了传统策略中某些 Warp 早早完工而另一些 Warp 还在处理低效尾部块的情况。

2.  **减少尾部效应**：
    对于非完美整除的问题尺寸，Stream-K 不再有“不完整的输出块”概念。所有 Warp 处理的工作包大小都是一致的（除了最后一个），极大减少了因边界处理带来的计算效率损失。

3.  **提高硬件利用率**：
    通过最大化 Warp 的忙碌程度，Stream-K 更好地利用了 GPU 的计算单元（CUDA Cores）和张量核心（Tensor Cores），从而在单位时间内完成更多有效计算，提升了吞吐量。

**性能提升场景：**
Stream-K 并非在所有情况下都是最优的。它的优势在以下场景尤为明显：
*   **问题尺寸（M, N, K）** 与 **GPU 和 Kernel 的配置（Tile大小，WarpShape等）** 不匹配时。
*   矩阵维度（尤其是 M 和 N）较小，但 K 维度较大时。
*   矩阵维度不能被 Tile 尺寸完美整除时。
*   在最新的 GPU 架构（如 Hopper）上，其优势更加突出。

**trade-off：**
Stream-K 引入了全局任务队列和动态调度，这会带来少量的额外开销。因此，在问题尺寸非常规整、完美匹配硬件配置的理想情况下，高度优化的传统 SM 策略可能仍然有微弱的优势。但在现实世界的应用中，问题尺寸千变万化，Stream-K 通常能提供更稳健、更高的平均性能。

---

### 3. CUTLASS Samples

CUTLASS 在其示例代码中提供了 Stream-K 的实现。最核心的示例是：

#### **`cutlass/examples/49_hopper_gemm_with_stream_k/`**

这个目录包含了在 NVIDIA Hopper 架构（但概念也适用于其他架构）上使用 Stream-K 实现 GEMM 的完整代码。

**关键文件：**
*   `49_hopper_gemm_with_stream_k.cu`: 主程序文件，包含了使用 Stream-K kernel 的 GEMM 操作实例。

**代码解析（核心部分）：**

1.  **定义 Stream-K GEMM Kernel 类型**：
    代码中使用 `cutlass::gemm::kernel::GemmStreamK` 来定义 Kernel。这与传统的 `Gemm` 或 `GemmUniversal` 不同。

    ```cpp
    using GemmKernel = cutlass::gemm::kernel::GemmStreamK<
        ElementA,                                 // A 矩阵的数据类型 (e.g. cutlass::half_t)
        cutlass::layout::RowMajor,                // A 的布局
        ElementB,                                 // B 矩阵的数据类型
        cutlass::layout::RowMajor,                // B 的布局
        ElementC,                                 // C 和 D 矩阵的数据类型
        cutlass::layout::RowMajor,                // C 和 D 的布局
        ElementAccumulator,                       // 累加器数据类型
        arch::OpClassTensorOp,                    // 使用 Tensor Core 操作
        arch::Sm90,                               // 目标架构 (Hopper)
        TileShape,                                // 线程块处理的 Tile 大小 (e.g. _256x128x64)
        WarpShape,                                // Warp 处理的形状 (e.g. _64x64x64)
        StageCount,                               // 共享内存流水线级数 (e.g. 5)
        KernelSchedule,                           // 调度策略 (e.g. auto)
        ClusterShape                              // GPU 集群形状 (e.g. _1x1x1)
    >;
    ```

2.  **定义 GEMM Operation 并运行**：
    将定义好的 Kernel 类型包装成 `GemmStreamK` Operation，然后调用 `run` 函数执行。

    ```cpp
    using Gemm = cutlass::gemm::device::GemmStreamK<GemmKernel>;
    
    Gemm gemm_op;
    
    // ... 初始化参数 (arguments)
    cutlass::Status status = gemm_op.run(arguments);
    ```

3.  **参数设置**：
    Stream-K 的参数 (`Arguments`) 和传统 GEMM 类似，需要设置矩阵的尺寸、指针、Leading Dimension 等。

    ```cpp
    typename Gemm::Arguments arguments{
        {M, N, K},                                  // 问题尺寸
        {A, lda},                                   // A 矩阵参数
        {B, ldb},                                   // B 矩阵参数
        {C, ldc},                                   // C 矩阵参数
        {D, ldd},                                   // D 矩阵参数
        {alpha, beta},                              // Epilogue 的标量参数
    };
    ```

**如何编译和运行：**
在 CUTLASS 的构建系统中，这个示例通常会被编译。你需要确保你的 CUDA 工具包和编译器支持目标架构（例如 `-arch=sm_90a` for Hopper）。

```bash
# 在 build 目录下，示例通常会被编译成可执行文件
./examples/49_hopper_gemm_with_stream_k/49_hopper_gemm_with_stream_k
```

这个示例程序会使用 Stream-K 和传统的 Universal 内核分别执行相同的 GEMM 计算，并比较结果、输出性能数据（TFLOPS），让你可以直观地看到 Stream-K 带来的性能提升。

**其他相关示例：**
*   `cutlass/examples/47_ampere_gemm_universal_streamk/`: 在 Ampere 架构上使用 Stream-K 的示例。
*   `cutlass/examples/50_hopper_gemm_streamk_with_epilogue_permute/`: 展示了在 Stream-K GEMM 后接一个数据重排（Epilogue Permute）的操作。

### 总结

Stream-K 是 CUTLASS 中一项重要的创新，它通过将 GEMM 计算重新组织为细粒度的、沿 K 维度流式的工作包，并采用动态调度策略，极大地改善了 GPU 上 Warp 级别的负载均衡。这使得它在处理各种非理想问题尺寸时，能提供比传统静态划分方法更优越和更稳健的性能，特别是在最新的 NVIDIA GPU 架构上。通过分析 CUTLASS 提供的示例代码，开发者可以学习如何在自己的应用中集成这一强大的优化技术。

---

### 🧑‍💻 User

为什么说stream-k特别适合用于处理Transformer 模型中的注意力机制（如变长 K/V 序列），在高性能llm inference中大量使用？

---

### 🤖 Assistant

说得非常对！Stream-K 之所以在 LLM Inference 中处理注意力机制时大放异彩，并被高性能推理引擎（如 TensorRT-LLM, vLLM 等）广泛采用，正是因为它完美地解决了此类工作负载的核心痛点。

下面我们从几个关键角度深入分析为什么 Stream-K 是处理变长 K/V 序列注意力计算的“利器”。

---

### 1. 问题背景：注意力机制 GEMM 的特点

在 Transformer 的解码阶段（无论是训练还是推理），一个核心计算是注意力得分矩阵与 V 矩阵的乘法：`Output = Softmax(QK^T) * V`。这个 `(Softmax(QK^T)) * V` 的计算就是一个典型的 GEMM。

在 **自回归推理（Autoregressive Inference）** 中，这个计算呈现出以下**独特且具有挑战性的特征**：

1.  **极度动态变化的尺寸（Dynamic & Ragged Shapes）**：
    *   **批量（Batch Size）** 中的每个序列（Sequence）长度可能完全不同。用户可能同时请求一个很长的对话历史和一個很短的新问题。
    *   **逐步增长（Growing）**：在生成 token 的过程中，每个步骤（Step）的 K/V 序列长度都会比上一个步骤增加 1（因为新增了当前步生成的 token 的 K/V 缓存）。
    *   这意味着，每一步都需要处理大量 **`[Batch_Size, 1, Head_Dim]` x `[Batch_Size, Head_Dim, Seq_Len]`** 的 GEMM 运算，其中 `Seq_Len` 在每个序列、每个生成步上都可能不同。

2.  **大量的小型或不规则 GEMM 问题**：
    *   单个 GEMM 的维度 M (1) 和 N (Head_Dim) 通常很小（例如，M=1, N=128），而 K (Seq_Len) 会从 1 逐步增长到几千甚至上万。
    *   传统的基于固定 Tile 划分的 GEMM 内核在处理这种 `M=1` 或 `N` 很小的问题时，**效率极其低下**。因为一个线程块（Thread Block）可能被分配去计算输出矩阵的一个 Tile（例如 128x128），但实际输出矩阵只有 `1x128`。这意味着一个庞大的线程块中只有极少数线程（第一行）有实际工作，其他绝大多数线程都处于空闲状态，造成巨大的资源浪费。

### 2. Stream-K 如何完美应对这些挑战？

Stream-K 的动态、细粒度负载均衡特性恰好是解决上述问题的“银弹”。

#### a) 极致负载均衡，消除“空闲线程”

*   **传统 GEMM (SM Strategy) 的问题**：对于 `1x128` 的输出矩阵，一个设计用于处理 `128x128` 的线程块几乎完全浪费。GPU 的 SM 上充斥着这种“空转”或“低效运转”的线程块，有效算力密度极低。
*   **Stream-K 的解决方案**：它将整个计算（沿着 `Seq_Len` 维度）分解成大量微小的工作包。多个 Warp 可以共同协作来完成这个 `1x128` 的输出计算。
    *   每个 Warp 领取一个工作包，计算 Seq_Len 上一小段的数据与 Q 的乘法，并部分累加到输出结果中。
    *   由于工作包很小且数量多，**所有 Warp 都始终保持忙碌**，几乎没有空闲。它们共同、高效地“流式”处理完了这个小的 GEMM 运算。

**这就好比：**
*   **传统方式**：派一辆 50 座的巴士（Thread Block）去送一个快递（1x128 GEMM），极度浪费。
*   **Stream-K 方式**：把一个快递拆成几段路，让多个外卖小哥（Warp）骑电动车并行完成，效率极高。

#### b) 天然适应动态和不规则问题尺寸

Stream-K 不关心全局的 M 和 N 有多大。它的工作分配只依赖于总的计算量（正比于 M * N * K）。对于任何尺寸的 GEMM，它都能通过动态调度来最大化硬件利用率。

*   无论是 `1x128x1000` 还是 `1x128x2345` 的 GEMM，Stream-K 都能以近乎 100% 的 Warp 利用率来处理它们。它**自动适应**各种序列长度，无需为每种可能的尺寸专门优化内核。
*   这在处理** ragged batch**（即一个 batch 内序列长度各异）时至关重要。GPU 上同时计算多个不同序列长度的注意力输出时，Stream-K 能确保每个计算都高效利用分配给它的硬件资源，从而大幅提升整个批处理的吞吐量。

#### c) 减少填充（Padding）开销

在没有 Stream-K 之前，处理 ragged batch 的一种常见方法是**填充（Padding）**：将所有序列填充到 batch 内最长的长度，然后执行一个大的、规整的批处理 GEMM。

*   **缺点**：填充引入了大量的无效计算（对填充值的操作），浪费了宝贵的计算资源和内存带宽。
*   **Stream-K 的优势**：由于 Stream-K 能高效处理各种尺寸的小 GEMM，推理引擎可以**减少甚至避免填充**。它可以为每个序列启动独立但并发的 GEMM 操作，而每个操作因为 Stream-K 而保持高效。这直接减少了总计算量，提升了整体性能。

### 3. 在高性能 LLM Inference 中的应用

正是基于以上优势，Stream-K 成为了现代 LLM 推理引擎的标配优化技术：

*   **TensorRT-LLM**：NVIDIA 的官方高性能推理库，在其核心的注意力机制实现中大量使用了基于 Hopper 架构（SM90）的 Stream-K GEMM。
*   **vLLM**：一个非常流行的开源推理引擎，也集成了 Stream-K 来加速其注意力计算。

这些引擎通过 Stream-K 来实现：
1.  **更高的吞吐量（Throughput）**：在相同硬件上，每秒处理更多的用户请求。
2.  **更低的延迟（Latency）**：更快地返回单个请求的响应，因为计算效率更高。
3. 更好的**经济性**：更高的计算效率意味着用更少的 GPU 服务器承载相同的流量，降低了推理成本。

---

### 总结与类比

| 特性 | 传统 GEMM (SM 策略) | Stream-K GEMM | 对 LLM 推理的影响 |
| :--- | :--- | :--- | :--- |
| **工作分配** | 静态，粗粒度（分输出块） | 动态，细粒度（分 K 维度流） | Stream-K 能高效处理大量小尺寸/不规则 GEMM |
| **负载均衡** | 差，尾部效应严重 | 极佳，Warp 利用率近乎 100% | 彻底解决注意力计算中线程空闲问题 |
| **适应性问题** | 需为特定尺寸优化，怕“小M/N” | 通用性强，无惧任何尺寸 | 完美适应动态变化的序列长度，无需填充 |

可以说，Stream-K 的出现，为解决 LLM 自回归推理中**核心的计算瓶颈**——**不规则注意力计算**——提供了一种革命性的解决方案。它不再是“锦上添花”的优化，而是“雪中送炭”的关键技术，直接决定了现代推理引擎的性能底线。这就是为什么它在高性能 LLM Inference 中变得如此重要和普遍。