# LLM推理框架性能优化策略

### 🧑‍💻 User

在llm推理框架开发中，运行一次computegraph可能会涉及多个kernel的调用，像一些matmul op会调用tensorcore，一些element-wise op（如activation等）会调用cuda core，这两者交替调用有没有可能出现gap导致整个pipeline无法被充分利用的情况，如何解决？

---

### 🤖 Assistant

这是一个非常专业且切中要害的问题，它直接关系到高性能LLM推理引擎开发的核心挑战。**答案是：是的，这种情况不仅可能出现，而且在未经优化的Naive实现中几乎必然出现。** 这种“Gap”会导致GPU的运算单元（特别是Tensor Cores）和内存带宽无法被充分利用，从而成为限制整个Pipeline性能的关键瓶颈。

### 为什么会出现“Gap”和利用率不足的情况？

根本原因在于**GPU是一种大规模并行处理器，其极高的计算能力需要持续、稳定地“喂饱”数据才能充分发挥**。任何中断、等待或调度开销都会导致性能下降。具体到你的问题，原因包括：

1.  **Kernel启动开销（Launch Overhead）**:
    *   每次调用一个CUDA Kernel（无论是Tensor Core还是CUDA Core），CPU端（Host）都需要通过驱动程序向GPU提交任务。这个过程涉及参数传递、指令设置、资源分配等，虽然单个开销不大（微秒级），但在一个由大量小规模Kernel组成的计算图中，累积起来的总开销会非常可观。

2.  **执行间隙（Execution Gap）**:
    *   当一个Kernel执行完毕，到下一个Kernel开始执行之间，存在一个微小的间隙。GPU的硬件调度器需要为下一个Kernel分配流多处理器（SMs）、寄存器等资源。在频繁交替调用不同Kernel时，这些间隙累积起来就会形成明显的“空泡”（Bubble），使得计算单元在一段时间内无事可做。

3.  **资源竞争与依赖（Resource Contention and Dependencies）**:
    *   **计算资源竞争**：虽然Tensor Core和CUDA Core在物理上可能位于同一个SM内，但GPU的线程调度器、指令发射器等是共享资源。频繁切换不同类型的计算任务可能会带来额外的调度压力。
    *   **内存带宽竞争**：Element-wise操作通常是带宽受限的（Bandwidth-Bound），即它们的速度取决于从显存中读取和写入数据的速度。而MatMul是计算受限的（Compute-Bound）。当一个带宽受限的Kernel和一个计算受限的Kernel交替运行时，它们可能会竞争相同的内存控制器和缓存资源，导致彼此都无法达到峰值性能。

4.  **数据依赖与停顿（Data Dependency and Stalling）**:
    *   标准的同步执行模式下，`Kernel2`必须等待`Kernel1`完全执行完毕并将数据写入显存后，才能开始从显存读取数据进行计算。这个“写-读”的依赖关系造成了天然的流水线停顿，计算单元在等待数据从显存中读取时处于空闲状态。

---

### 如何解决？优化策略与关键技术

解决这些问题的核心思想是：**最大化计算单元的连续工作时间，最小化任何形式的等待和开销**。以下是业界常用的解决方案，从易到难：

#### 1. Kernel融合（Kernel Fusion）—— 最有效的武器

这是解决该问题的**首选且最有效**的方法。其思想是将多个小的、依次执行的Operator（尤其是Element-wise操作）融合到一个大的Kernel中执行。

*   **如何工作**： instead of:
    ```python
    # 未融合：3次Kernel启动，2次显存读写
    c = torch.matmul(a, b)  # 调用TensorCore Kernel，结果写回显存
    d = torch.relu(c)       # 调用CUDA Core Kernel，从显存读c，结果写回显存
    e = torch.sigmoid(d)    # 调用CUDA Core Kernel，从显存读d，结果写回显存
    ```
    Do:
    ```python
    # 已融合：1次Kernel启动，0次中间显存读写
    e = fused_matmul_relu_sigmoid(a, b) # 一个自定义Kernel
    ```
    在这个自定义Kernel中，每个线程块在从显存加载了输入数据`a`和`b`后，会在芯片上的**共享内存（Shared Memory）** 和**寄存器（Registers）** 中完成矩阵乘、ReLU和Sigmoid的所有计算，最后才将最终结果`e`写回显存。

*   **带来的好处**：
    *   **极大减少Kernel启动次数**：将多次调用变为一次调用，消除了启动开销。
    *   **避免中间结果写回显存**：数据在芯片上的高速缓存中流动，极大地降低了了对全局显存带宽的需求和访问延迟。这是性能提升的关键。
    *   **提升数据局部性**：计算过程中的中间数据始终保持在SM内部，访问速度极快。

*   **实际应用**：像NVIDIA的CuDNN、TensorRT等库都大量使用了Kernel融合。例如，将 `MatMul + BiasAdd + ReLU` 融合成一个 “`CUBLASLT_EPILOGUE_RELU`” 操作。

#### 2. 使用CUDA Streams和异步执行（Asynchronous Execution）

*   **如何工作**：将整个计算图的不同部分分配到多个**CUDA流（Streams）** 中。虽然单个Stream内的Kernel是顺序执行的，但不同Stream之间的Kernel可以**并发（Concurrent）** 执行。
*   **带来的好处**：
    *   **隐藏延迟**：当一个Stream中的Kernel在等待数据从显存中读取时（数据依赖或带宽受限），GPU可以切换到另一个Stream中的Kernel进行计算，从而填充“Gap”，保持计算单元忙碌。
    *   ** overlap Compute and Data Transfer**：经典的优化是使用一个Stream进行计算，同时使用另一个Stream将下一批数据从主机内存异步传输到GPU显存，实现计算和通信的重叠。

#### 3. 使用CUDA Graph

这是对大量小Kernel启动开销的“终极解决方案”。

*   **如何工作**：CUDA Graph允许你**预先录制（Capture）** 一个完整的Kernel调用序列（包括它们的依赖关系），然后将其作为一个单一的、原子性的单元（即Graph）来启动。
*   **带来的好处**：
    *   **近乎消除所有Kernel启动开销**：驱动程序不再需要为Graph中的每一个Kernel进行逐一的启动设置，而是直接执行整个预定义的图。这极大地减少了CPU的参与和驱动层的开销。
    *   **为调度器提供全局视野**：GPU调度器在执行前就看到了整个工作流，可能进行更好的优化调度。
    *   特别适用于LLM推理这种计算图结构固定、需要反复执行的场景。

#### 4. 高性能算子库与代码生成

不要重复造轮子。直接使用或借鉴高度优化的算子库：

*   **cuBLAS, cuDNN**: 提供经过极致优化的基础算子（如MatMul、卷积）。
*   **CUTLASS**: NVIDIA提供的用于实现高性能矩阵乘的模板库，可以自定义并生成融合了各种Element-wise操作的MatMul Kernel。
*   **TensorRT**: 内置了强大的图优化器，会自动进行Kernel融合、精度校准（FP16/INT8）、图优化等，专门为推理场景设计。
*   **TVM, OpenAI Triton**: 这些编译器框架允许你用高级语言描述计算，然后自动生成高度优化的GPU代码，它们也集成了强大的Kernel融合和优化能力。

### 总结

| 问题根源 | 解决方案 | 效果 |
| :--- | :--- | :--- |
| **Kernel启动开销** | **Kernel融合**, **CUDA Graph** | 大幅减少甚至消除启动开销 |
| **执行间隙与空泡** | **Kernel融合**, **CUDA Streams** | 创建更大、更连续的计算任务，并发执行填充空泡 |
| **内存带宽瓶颈** | **Kernel融合**（避免中间结果写回） | 极大降低对显存带宽的需求和访问延迟 |
| **数据依赖停顿** | **Kernel融合**（片上缓存通信） | 将显存间的“写-读”依赖变为芯片内的寄存器/共享内存访问 |

因此，在现代LLM推理框架开发中，**积极的Kernel融合策略**是构建高性能引擎的基石，再辅以**CUDA Graph**来固化执行流程以减少开销，并结合**多流异步执行**来进一步挖掘并行性，从而最大限度地压榨GPU硬件（无论是Tensor Cores还是CUDA Cores）的性能潜力。

---

### 🧑‍💻 User

像vllm（with flashinfer）/tensorrt-llm等高性能inference是如何解决这些问题的？

---

### 🤖 Assistant

非常好的问题！vLLM 和 TensorRT-LLM 正是业界解决上述问题的典范。它们采用了我们之前讨论的所有高级策略，并将其发挥到了极致，但两者的侧重点和实现哲学有所不同。

下面我们分别剖析它们是如何解决 Kernel 间隙和利用率问题的。

---

### 1. vLLM (with FlashInfer)

vLLM 的核心创新是 **PagedAttention**（解决 KV Cache 内存碎片和管理问题），但其高性能也离不开底层算子的高度优化。vLLM 本身并不重度自研算子和融合，而是**巧妙地集成并依赖社区最顶尖的高性能算子库**（如 FlashAttention, FlashInfer, CUTLASS）。

**其解决方案可以概括为：**

#### a. 拥抱“Flash”家族算子 (Kernel Fusion 的集大成者)
*   **FlashAttention / FlashInfer**: 这些是解决 Attention 部分性能问题的核武器。它们并非简单的 Kernel 融合，而是**一种革命性的算法重构**。
    *   **传统Attention**: `MatMul (QK^T) -> Softmax -> Mask -> Dropout -> MatMul (Score * V)`。这个过程需要将巨大的中间矩阵 `QK^T` 和 `Softmax` 结果写回显存，然后再读出来，极度浪费带宽。
    *   **FlashAttention**: 采用 **“Tiling”** 技术，将计算分块进行。在 SRAM（共享内存/寄存器）中进行所有的计算（MatMul、Softmax、MatMul），**完全避免将庞大的中间矩阵写回全局显存**。它将整个 Attention 计算融合成了一个极其高效、对内存访问极度友好的超级 Kernel。
    *   **效果**: 这不仅是融合了多个操作，更是彻底改变了数据流，将内存读写量从平方级 `O(N²)` 降低到线性级 `O(N)`。这极大地减轻了内存带宽压力，让计算单元（Tensor Cores）能够持续工作。

#### b. 依赖高性能基础库
*   vLLM 使用 **CUTLASS** 和 **cuBLASLt** 作为其 MatMul 的后端。这些库本身就支持 **Epilogue Fusion**。
    *   这意味着 `Linear = MatMul + Bias` 这个操作本身就是一个融合 Kernel。你甚至可以配置成 `MatMul + Bias + GELU` 一次完成。
    *   vLLM 通过配置这些库，轻松实现了基础算子的高性能融合。

#### c. 灵活的架构和社区生态
*   vLLM 的架构设计允许它**灵活地插入不同的 Kernel 实现**。例如，对于 Llama 的 `SwiGLU` 激活函数，它可能有多个实现：
    *   一个朴素的、多个 Kernel 实现的版本（用于兼容性）。
    *   一个使用 `CUTLASS` 高度融合的 `MatMul + SwiGLU` 的版本。
    *   运行时根据显卡能力自动选择最佳 Kernel。
*   它积极集成像 **FlashInfer** 这样的社区最优方案，从而快速获得最新、最快的算子，而不需要自己重复开发。

**vLLM 的哲学**: **“站在巨人的肩膀上”**。通过集成顶级算子库和算法创新（PagedAttention），优先解决最大的瓶颈（内存），从而实现整体高性能。它的优化重点在**调度层和内存层**，计算层则交给专业库。

---

### 2. TensorRT-LLM

TensorRT-LLM 代表了另一种哲学：**“垂直整合，极致优化”**。它是 NVIDIA 亲儿子，拥有从编译器到 Kernel 每一层的深度控制权，可以进行全栈优化。

**其解决方案可以概括为：**

#### a. 强大的图编译和优化器 (Aggressive Graph Fusion)
TensorRT-LLM 的核心是一个**编译器**。你提供模型（PyTorch / ONNX），它会对整个计算图进行解析和优化：
*   **算子融合**: 它的融合策略极其激进和彻底。它会扫描整个计算图，寻找任何可以融合的模式。
    *   `Linear -> GELU -> Linear`？ 可以融合。
    *   `Attention 的全部路径`？ 必须融合（它有自己的 FlashAttention 实现）。
    *   甚至更复杂的模式，如 `LayerNorm` 或 `RMSNorm` 与前后操作的融合。
*   **常量折叠**: 将图中可以预先计算的节点（如 Shape/Transpose 的调整）在编译期就计算好，减少运行时开销。
*   **层间融合**: 跨层的操作也可能被合并，减少 Kernel 数量。

#### b. 高度特化和手写 Kernel
*   由于是 NVIDIA 官方出品，TensorRT-LLM 包含了大量**为最新硬件（如 Hopper H100）和最新模型架构特调的手写 CUDA Kernel**。
*   这些 Kernel 不仅仅是融合，更是**极致优化**，充分利用 Tensor Cores、Ampere/Hopper 的新指令集、异步拷贝（`async-copy`）、张量内存加速器（TMA）等硬件特性。
*   例如，对于 **FP8 精度** 或者 **4-bit Weight-Only 量化** 的支持，TensorRT-LLM 都有对应的、高度优化的融合 Kernel（如将去量化操作与 MatMul 融合）。

#### c. 内置高效的 Attention 实现
*   TensorRT-LLM 内置了多种 Attention 实现（类似于 FlashAttention），可以根据输入规模（SeqLen）、Batch Size、数据类型自动选择最快的一个。它无需依赖外部库。

#### d. 深度集成 CUDA Graph
*   TensorRT-LLM 在运行时**广泛使用 CUDA Graph** 来捕获整个推理过程（尤其是小Batch Size的场景）。
*   一旦计算图被编译和优化，整个流程就固定了。运行时只需要启动一个或少数几个 CUDA Graph，而不是成千上万个小 Kernel，**几乎完全消除了 Kernel 启动开销**。

#### e. 与 TensorRT 生态无缝集成
*   它可以利用 TensorRT 强大的量化工具（PTQ、QAT）、插件系统，实现精度和性能的最佳平衡。

**TensorRT-LLM 的哲学**: **“编译器驱动，全栈掌控”**。通过编译期进行极其激进的图优化和融合，再为每个融合后的子图提供手写、高度特化的 Kernel，最后用 CUDA Graph 高效执行。它追求的是极致的端到端性能。

---

### 总结对比

| 特性 | vLLM (with FlashInfer) | TensorRT-LLM |
| :--- | :--- | :--- |
| **核心创新** | **PagedAttention** (解决内存瓶颈) | **Aggressive Graph Compiler** (解决计算瓶颈) |
| **优化哲学** | **“集成最佳”**：灵活架构，集成社区最优算子库 | **“垂直整合”**：全栈可控，编译器驱动极致优化 |
| **Kernel 来源** | 依赖外部高性能库 (FlashInfer, CUTLASS) | 大量自研、手写、高度特化的 Kernel |
| **融合策略** | 主要依赖底层库自身的融合能力（如 Epilogue） | 编译器进行全局、跨层的激进融合 |
| **硬件利用** | 通过优秀库间接充分利用新特性 | **直接、深度**利用最新硬件特性 (TMA, async-copy, FP8) |
| **易用性/灵活性** | **高**：易于 hacking 和集成新模型、新算子 | **中**：需要遵循 NVIDIA 的编译流程，定制新算子更复杂 |
| **性能** | 非常优秀，尤其在吞吐量场景 | **极致**，通常是 NVIDIA 硬件上的性能标杆 |

**结论**：
两者都完美解决了你提到的 Kernel Gap 问题，但路径不同：
*   **vLLM** 更像一个“调度大师”和“内存管理大师”，它通过解决内存瓶颈和集成现成的最佳武器来获得高性能，更灵活。
*   **TensorRT-LLM** 更像一个“编译器大师”和“微架构大师”，它通过编译期深度优化和手写每一行汇编级别的代码来榨干硬件的最后一滴性能，更极致。

在实际应用中，如果你的模型是主流架构（如 Llama, GPT），并且追求极致的延迟和吞吐，**TensorRT-LLM 通常是性能冠军**。如果你需要快速支持新模型或进行学术研究，**vLLM 提供了更大的灵活性**。