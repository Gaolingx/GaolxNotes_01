# Stream-K技术提升GEMM性能解析

### 🧑‍💻 User

gemm中的stream-k是什么，他是如何提高gemm性能的？附上cutlass samples。

---

### 🤖 Assistant

好的，我们来详细探讨一下 GEMM 中的 Stream-K 技术，以及它如何提高性能，并附上相关的 CUTLASS 示例。

### 什么是 GEMM 中的 Stream-K？

**GEMM** (General Matrix Multiplication) 是深度学习和高性能计算中最为核心和计算密集型的操作之一，其形式为 $C = \\alpha \\cdot A \\cdot B + \\beta \\cdot C$。

**Stream-K** 是一种用于优化 GEMM 在现代 GPU 上执行效率的技术。它的核心思想是**将 GEMM 计算中对 K 维度的循环（内积维度）进行切分（Split-K），并将这些切分后的计算块（K-splits）并行地调度到 GPU 的多个流处理器（Streaming Multiprocessors, SMs）上执行**。

为了更好地理解，我们先回顾一下传统的 GEMM 执行方式：

  * **传统（非 Stream-K）GEMM**：通常情况下，整个 GEMM 计算被当作一个单一的、巨大的内核（Kernel）来启动。GPU 会将这个大任务分解到各个 SM 上。虽然 SM 之间存在并行，但它们都服务于同一个宏观任务。这种方式在矩阵尺寸很大时效率很高，但在某些情况下（例如，一个 GEMM 任务无法占满整个 GPU），会导致部分 SM 闲置，造成资源浪费。

而 Stream-K 的处理方式则不同：

  * **Stream-K GEMM**：它将 K 维度分割成多个块。例如，计算 $A (M \\times K) \\cdot B (K \\times N)$，可以被分解为多个小的 GEMM 计算：
    $C = \\sum\_{i=1}^{S} A(M \\times K\_i) \\cdot B(K\_i \\times N)$
    其中 $K = \\sum\_{i=1}^{S} K\_i$，$S$ 是切分的块数。

    Stream-K 将每一个切分出的计算块（$A(M \\times K\_i) \\cdot B(K\_i \\times N)$）作为一个独立的计算单元，并将其调度到 GPU 的不同 SM 组上，像流水线（Stream）一样并行处理。最后，通过一个独立的内核或在最后一个计算块中，将所有部分结果进行规约（Reduction）相加，得到最终的矩阵 C。

### Stream-K 如何提高 GEMM 性能？

Stream-K 主要通过以下几种方式来提升性能：

1.  **提升并行度和 GPU 占用率 (Increased Parallelism and Occupancy)**：

      * **问题**：当单个 GEMM 的规模不足以占满整个 GPU 的所有 SM 时，传统方法会导致大量 SM 闲置。这种情况在处理小批量（small batch sizes）或者模型中存在多个并行分支时很常见。
      * **解决方案**：Stream-K 将一个 GEMM 任务分解成多个独立的子任务（K-splits）。这些子任务可以像不同的波次（waves）一样，被分发到 GPU 的不同 SM 上并行执行。这样，即使单个 GEMM 规模不大，也可以通过并行执行其多个 K-splits 来有效利用整个 GPU，从而显著提高 SM 的占用率和计算资源的利用效率。

2.  **改善数据局部性和缓存利用 (Improved Data Locality and Cache Usage)**：

      * **问题**：巨大的 GEMM 操作需要加载大量的 A 和 B 矩阵数据到 SM 的共享内存（Shared Memory）或 L1/L2 缓存中。如果数据量过大，会导致缓存命中率下降和频繁的数据搬运。
      * **解决方案**：每个 K-split 只处理 K 维度的一个片段。这意味着每个 SM 在执行子任务时，只需要加载和处理更小块的矩阵数据。这使得数据更容易被装入和保留在高速的共享内存或 L1 缓存中，从而提高了数据局部性，减少了对慢速全局内存（Global Memory）的访问延迟。

3.  **隐藏数据加载延迟 (Hiding Memory Latency)**：

      * **解决方案**：通过流水线的方式，当一个 SM 正在计算当前的 K-split 时，它可以预取（prefetch）下一个 K-split 所需的数据。计算和数据加载可以重叠进行，有效地隐藏了从全局内存加载数据所需的时间，这在访存密集型场景下尤为重要。

4.  **灵活的负载均衡 (Flexible Load Balancing)**：

      * **解决方案**：Stream-K 允许根据 GPU 的架构和可用资源，动态地调整 K 维度的切分数量和大小。这种灵活性使得算法能够更好地适应不同的硬件和工作负载，实现更优的负载均衡。例如，可以将一个大的 GEMM 分配给大部分 SM，同时将一个小的 GEMM 分配给剩余的 SM，实现任务间的并行。

### CUTLASS Samples 中的 Stream-K 示例

NVIDIA 的 **CUTLASS** 是一个用于在 CUDA C++ 中实现高性能矩阵乘法的模板库，它为 Stream-K 提供了原生支持和丰富的示例。

最典型的例子是 **`34_sm90_splitk_gemm`**，这个示例专门用于展示在 Hopper (SM90) 架构上如何使用 Split-K 和 Stream-K 的概念。虽然命名中提到了 SM90，但其核心思想和实现对其他现代 GPU 架构也具有指导意义。

**示例位置**:
你可以在 CUTLASS 的官方 GitHub 仓库中找到这个示例：
[https://github.com/NVIDIA/cutlass/tree/main/examples/34\_sm90\_splitk\_gemm](https://www.google.com/search?q=https://github.com/NVIDIA/cutlass/tree/main/examples/34_sm90_splitk_gemm)

**关键代码解读**:

在 `splitk_gemm_example.cu` 文件中，你可以看到 Stream-K 的实现逻辑。其核心步骤如下：

1.  **定义 GEMM 操作**:
    首先会定义一个标准的 GEMM 操作，包括其数据类型、布局和数学运算等。

    ```cpp
    // The epilogue functor for thread-level GEMMs
    using Epilogue = cutlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / cutlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementComputeEpilogue
    >;

    // The universal GEMM kernel
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape,
      typename Mma::IteratorA,
      cutlass::gemm::layout::RowMajor,
      typename Mma::IteratorB,
      cutlass::gemm::layout::RowMajor,
      cutlass::gemm::layout::RowMajor,
      ElementC,
      Epilogue
    >;
    ```

2.  **配置 Split-K 参数**:
    Stream-K 的关键在于 `split_k_slices` 参数。当这个值大于 1 时，CUTLASS 就会启用 Split-K 的执行策略。

    ```cpp
    // Initialize the arguments
    typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      split_k_slices, // This is the key parameter to enable Stream-K
      {alpha, beta},
      A.device_data(),
      B.device_data(),
      C.device_data(),
      D.device_data(),
      problem_size.m() * problem_size.k(),
      problem_size.k() * problem_size.n(),
      problem_size.m() * problem_size.n(),
      problem_size.m() * problem_size.n()
    };
    ```

    在此示例中，`split_k_slices` 被设置为一个大于1的整数，比如3。这意味着 K 维度将被切分为3片，并通过并行流来执行。

3.  **执行 GEMM**:
    `Gemm` 对象在构造和初始化时会根据 `split_k_slices` 的值来决定内部的执行策略。如果 `split_k_slices > 1`，它会自动处理 K 维度的切分、部分结果的计算以及最终的规约。

    ```cpp
    // Initialize the GEMM object
    Gemm gemm_op;

    // ...

    // Run the GEMM
    cutlass::Status status = gemm_op(arguments, nullptr, stream);
    ```

4.  **内部机制**:
    虽然用户侧的代码看起来很简单，但在 CUTLASS 内部，`gemm_op` 的执行会变得复杂：

      * 它会计算每个 K-split 的大小。
      * 它可能会启动多个 CUDA Kernel，每个 Kernel 处理一个或多个 K-splits。
      * 第一个 `split_k_slices - 1` 的 Kernel 会计算部分结果并将其写入一个临时的 Workspace 内存中（此时 $\\beta=0$）。
      * 最后一个 Kernel 会计算最后一块 K-split 的结果，并与 Workspace 中的所有部分结果相加（此时 $\\beta=1$），完成规约操作，得到最终的矩阵 D。

这个示例清晰地展示了如何通过简单地设置一个参数，就能利用 CUTLASS 强大的后端来启动一个复杂的 Stream-K GEMM，从而在兼容的硬件上获得显著的性能提升。

### 总结

Stream-K 是一种先进的 GEMM 优化技术，它通过将 K 维度分解为多个并行处理的块，极大地提高了 GPU 的资源利用率和并行度，尤其是在单个 GEMM 不足以占满整个 GPU 的场景下。它通过改善数据局部性、隐藏内存延迟和提供灵活的负载均衡，成为现代 GPU 上实现极致 GEMM 性能的关键技术之一。CUTLASS 库为开发者提供了简单易用的接口，使其能够方便地利用 Stream-K 的强大功能。

---

### 🧑‍💻 User

为什么说stream-k特别适合用于处理Transformer 模型中的注意力机制（如变长 K/V 序列），在高性能llm inference中大量使用？

---

### 🤖 Assistant

好的，这是一个非常深入且关键的问题，触及了现代高性能大语言模型（LLM）推理优化的核心。

简单来说，Stream-K之所以特别适合处理Transformer中的注意力机制（尤其是在LLM推理中），是因为它完美地解决了**因变长K/V序列导致的GEMM计算规模小、形状不规则、难以充分利用GPU硬件**这一核心痛点。

下面我们来详细拆解这个原因。

### 首先，回顾一下LLM推理中注意力计算的特点

在LLM的自回归推理（Autoregressive Inference）过程中，模型一次只生成一个Token。对于每个新生成的Token，其注意力计算的Q, K, V矩阵有如下特点：

1.  **Q (Query) 矩阵非常“瘦”**：因为我们只为当前这一个新Token计算Query，所以Q矩阵的形状是 `[1, num_heads, 1, head_dim]`。在GEMM中，M维度就是1。
2.  **K (Key) 和 V (Value) 矩阵非常“长”**：它们包含了当前新Token以及**所有**历史上下文（Context）中Token的信息。因此，K和V矩阵的形状是 `[1, num_heads, sequence_length, head_dim]`，其中 `sequence_length` 会随着生成的Token不断增长。

注意力计算主要包含两个GEMM操作：
* **GEMM 1 (Scores Calculation)**: $Q \cdot K^T$  -> 形状为 `(1 x d_head) * (d_head x seq_len)` -> 输出 `(1 x seq_len)`
* **GEMM 2 (Values Aggregation)**: $AttentionScores \cdot V$ -> 形状为 `(1 x seq_len) * (seq_len x d_head)` -> 输出 `(1 x d_head)`

这里的 `d_head` 是 `hidden_dim / num_heads`，通常是64, 128或256。

### LLM推理中注意力计算的性能瓶颈

在高性能推理服务中，服务器会把来自不同用户的请求（每个请求有不同的上下文长度）组成一个批次（Batch）进行处理。这就带来了核心的性能挑战：

1.  **GEMM规模小且“形状糟糕” (Small and Awkwardly Shaped GEMMs)**:
    * 对于 **GEMM 1 ($Q \cdot K^T$)**，它的内积维度（K-dimension）是`d_head`，通常只有64或128。这个K维度非常小。
    * 对于单个请求，这个GEMM计算量很小，比如 `(1x128) * (128x2048)`。这样一个小的计算任务，如果只用一个传统的GEMM内核去执行，**完全无法占满现代GPU上成百上千个流处理器（SMs）**。大部分SM会处于空闲状态，导致GPU利用率极低。

2.  **批处理中的序列长度极不均匀 (Highly Variable Sequence Lengths in a Batch)**:
    * 一个批次中可能包含一个序列长度为50的“新对话”请求，和一个序列长度为8000的“长文总结”请求。
    * 传统的Padding方法（将所有序列填充到最长序列的长度）会带来巨大的无效计算和内存浪费，在LLM推理中是不可接受的。
    * 因此，推理引擎（如vLLM, TensorRT-LLM）会使用`PagedAttention`等技术来处理这些变长的K/V序列，但这导致批处理中的GEMM操作大小和形状各异，进一步加剧了GPU调度和利用的难度。

### Stream-K 如何“对症下药”？

Stream-K技术的核心思想是**将GEMM的内积（K）维度切片并行化**，这恰好命中了上述痛点。

1.  **将小GEMM“放大”，提升GPU占用率 (Amplifying Small GEMMs to Improve Occupancy)**:
    * 以 `(1 x 128) * (128 x seq_len)` 这个GEMM为例，它的K维度是128。传统GEMM中，这个128的循环是在一个或少数几个线程块（Thread Block）内部完成的。
    * **Stream-K可以将这个128的维度切成多片**。例如，切成4片，每片大小为32。即 `(1x32)*(32xN) + (1x32)*(32xN) + (1x32)*(32xN) + (1x32)*(32xN)`。
    * **关键在于，这4个切片的计算可以被当作独立的任务，分配到GPU上更多的SM上并行执行**。原来可能只有一个SM在忙，现在可以让4个、8个甚至更多的SM同时参与这一个“小”GEMM的计算。
    * **效果**：对于单个请求而言，原本无法有效利用GPU的小计算任务，现在被“人为地”并行化了，大大提高了GPU的利用率，从而降低了单个请求的处理延迟。

2.  **实现灵活的、细粒度的并行调度 (Enabling Flexible, Fine-grained Parallel Scheduling)**:
    * 在处理一个包含多种序列长度的批次时，Stream-K带来了前所未有的灵活性。
    * 假设一个批次中有两个请求，一个长序列（seq_len=8000），一个短序列（seq_len=50）。
        * **传统方式**：GPU可能会先集中资源处理完长序列的GEMM，再处理短序列的，或者反之。无论如何，资源分配和调度都比较僵化。
        * **Stream-K方式**：推理引擎可以将长序列的GEMM切分成（比如）16个K-splits，短序列的GEMM切分成2个K-splits。然后，这总共18个小的计算任务（K-splits）可以被灵活地、动态地送往GPU的任何空闲SM组去执行。
    * **效果**：GPU就像一个高度灵活的任务处理器，长短任务的计算“波前”（wavefronts）可以混合在一起并行推进。这极大地提高了整个批次的处理效率（吞吐量），并减少了短请求等待长请求完成的时间（公平性）。这正是高性能推理服务追求的目标。

3.  **与FlashAttention等Memory-bound优化相辅相成**:
    * 注意力计算是典型的**内存受限（Memory-bound）**操作，其瓶颈在于数据（K/V Cache）的读取和写入，而非计算本身。
    * Stream-K通过将K维度切片，使得每个并行执行的子任务所需要的数据量更小。这**增强了数据的局部性（Data Locality）**，数据可以更好地被保留在高速的L1/L2缓存中，减少了对慢速HBM（高带宽内存）的访问次数。
    * 这与FlashAttention的思想异曲同工，后者通过Tiling技术在计算过程中避免向全局内存写入巨大的中间注意力矩阵。Stream-K则是在GEMM计算层面，通过并行化来提升访存和计算效率，两者结合使用能达到最佳效果。

### 总结

我们可以用一个比喻来理解：

* **传统GEMM**：像是一个大货车。如果你有一大批货物（大矩阵），它能高效地一次性运完。但如果你只有一个小包裹（小矩阵），开动这辆大货车就非常浪费，大部分空间都空着（GPU SMs空闲）。
* **Stream-K GEMM**：像是把大货车改装成了一个由多辆灵活的摩托车组成的“快递分队”。即使你只有一个小包裹，也可以派出整个分队，每辆摩托车负责运输包裹的一部分，它们同时出发，迅速完成任务。当你同时有多个大小不一的包裹时，这个分队可以被灵活拆分和组合，去最高效地完成所有运输任务。

因此，Stream-K通过在计算粒度上进行智能的“拆分与并行”，完美契合了LLM推理中注意力机制**计算规模小、变长、且对延迟和吞吐量要求极致**的特点，成为了现代高性能LLM推理引擎中不可或缺的一项关键优化技术。