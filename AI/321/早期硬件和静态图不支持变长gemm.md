# 早期硬件和静态图不支持变长gemm

### 🧑‍💻 User

为什么早期gpu/tpu和以tf为代表的静态图框架对变长gemm非常不友好？像grouped gemm就很难高效跑起来

---

### 🤖 Assistant

早期 GPU/TPU 以及以早期 TensorFlow（TF1.x）为代表的静态图框架，在设计之初的核心哲学是**“通过极致的规则性（Regularity）和确定性（Determinism）来换取极致的性能”**。

变长 GEMM（如 NLP 中的不同句子长度、图神经网络中的不同节点度数）恰恰打破了这种规则性。像 Grouped GEMM（在一个 Kernel 中并行计算多个不同大小的矩阵乘法）在早期很难高效运行，主要是由于**硬件架构的刚性**和**静态编译器的局限性**共同导致的。

以下是具体的深度剖析：

### 一、 硬件层面的极度不友好（早期 GPU / TPU）

早期 AI 硬件的设计是为了最大吞吐量，它们对数据的“形状（Shape）”有严格的物理和微架构偏好。

#### 1. TPU 的脉动阵列（Systolic Array）尺寸固定
TPU 的核心算力来自于脉动阵列（例如 TPU v2/v3 的 MXU 是 $128 \times 128$ 的乘加器阵列）。
*   **对齐要求：** 矩阵的维度必须是 128 的倍数才能完美填满阵列。
*   **变长的代价：** 如果输入是变长的（比如长度为 37, 89, 112），为了在 TPU 上运行，硬件或底层驱动必须在周围**填充（Padding）大量的 0**，使其对齐到 128。这导致大量的计算资源（FLOPs）和显存带宽被浪费在计算 $0 \times 0$ 上。

#### 2. GPU 的分块（Tiling）与线程束（Warp）利用率
早期 GPU 在执行 GEMM 时，依赖于将大矩阵切分成固定大小的 Block 和 Tile，分配给不同的 SM（流多处理器）。
*   **尾部效应（Tail Effect）：** 如果矩阵大小是动态变长的，很难找到一个全局最优的 Tile Size。对于不能被 Tile Size 整除的边缘部分，会导致大量线程闲置（Warp Divergence 或空转），严重拉低计算效率。

#### 3. Kernel 启动开销（Launch Overhead）
在没有 Grouped GEMM 支持的时代，处理变长 Batch 的常规做法是写一个 `for` 循环，针对每一个不同大小的矩阵启动一次 GEMM Kernel。
*   **CPU 瓶颈：** 早期 GPU 的 Kernel 启动开销在微秒级。如果矩阵很小且数量很多，CPU 发送指令的速度会跟不上 GPU 执行的速度，导致 GPU 严重饥饿（Starvation），利用率极低。

---

### 二、 软件层面的极度不友好（静态图框架与 XLA）

以 TensorFlow 1.x 为代表的静态图（Static Graph）框架，采用的是 **Define-and-Run（先定义，后执行）** 的模式。这种模式在遇到变长数据时会遭遇毁灭性的打击。

#### 1. 静态 Shape 推导与频繁的重编译（Recompilation）
TF 的底层编译器（如 XLA）之所以能做极致的优化（如算子融合 Kernel Fusion、内存预分配），前提是它**必须提前知道所有张量的确切 Shape**。
*   如果网络中存在变长 GEMM，每次输入一个新长度的 Batch，计算图的 Shape 就会发生变化。
*   **后果：** XLA 会认为这是一个全新的计算图，从而触发**重新编译（Recompilation）**。XLA 的编译时间通常需要几十秒到几分钟，如果每个 Step 长度都在变，模型根本无法训练。

#### 2. 内存池分配（Memory Allocation）的碎片化
静态图框架在运行前会进行全局的内存生命周期分析，预先分配好一块连续的显存。
*   如果是定长数据，内存可以完美复用。
*   如果是变长数据，系统很难预测下一步需要多大的内存，导致无法高效进行内存池复用，极易引发显存碎片化（Fragmentation）甚至 OOM（Out of Memory）。

#### 3. 动态控制流（Control Flow）性能极差
为了在静态图中处理变长数据而不触发重编译，开发者只能被迫使用 `tf.while_loop` 或 `tf.cond` 配合 `tf.pad`（分桶/Padding）。
*   早期 TF 的动态控制流在底层实现非常复杂且低效，严重破坏了计算图的连续性，导致 XLA 无法跨越控制流边界进行算子融合（Fusion）。

---

### 三、 为什么早期的 Grouped GEMM 很难跑起来？

Grouped GEMM 的概念是：只启动**一次** GPU Kernel，但给 Kernel 传入一组指针，每个指针指向大小不同的矩阵，让 GPU 内部的 Block 自行认领任务。

在早期，这很难高效实现：
1.  **缺乏底层库支持：** 早期的 cuBLAS 没有提供高效的 `cublasGemmGroupedBatched` 接口。即使后来提供了，其内部实现往往也是串行调度或者简单的启发式分配，无法保证各个 SM 的负载均衡。
2.  **负载不均衡（Load Imbalance）：** 假设一个 Kernel 同时处理 $10 \times 10$ 和 $1000 \times 1000$ 的矩阵，处理小矩阵的 SM 瞬间就计算完了，然后只能干等着处理大矩阵的 SM，导致整体性能被最慢的那个矩阵拖垮（木桶效应）。
3.  **框架表达困难：** 静态图框架很难用优雅的 API 来表达这种“不规则张量（Jagged Tensor）”。TF 虽然后来引入了 `RaggedTensor`，但其底层的算子支持极其有限，遇到复杂计算最终还是会退化成 Padding。

---

<details>
<summary><b>💡 延伸：现代软硬件是如何解决这个问题的？（点击展开）</b></summary>

随着 Transformer 的爆发（序列长度天生可变）和 GNN 的流行，业界被迫对软硬件进行了全面升级：

1.  **动态图框架的胜利：** PyTorch（Define-by-Run）成为主流，它天然支持变长操作，没有静态图重编译的烦恼。
2.  **编译器支持动态 Shape：** 现代的编译器（如 PyTorch Inductor, 升级后的 XLA, TVM）引入了符号化 Shape（Symbolic Shape）推导，可以针对一类 Shape 编译出通用的 Kernel。
3.  **硬件与底层库的进化：** 
    *   NVIDIA 推出了 CUTLASS 库，极大地优化了 Grouped GEMM 的调度，通过细粒度的 Work Stealing（工作窃取）机制解决了 SM 负载不均衡的问题。
    *   FlashAttention 和 Triton 的出现，允许开发者直接针对变长序列（通过 block pointer 和 sequence length masking）写出极致优化的融合算子，彻底绕开了传统的 Padding 惩罚。
</details>