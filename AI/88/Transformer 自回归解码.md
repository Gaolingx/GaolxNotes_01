# Transformer 自回归解码

### 🧑‍💻 User

Transformer 自回归解码具有非常特殊的计算特性。为什么由于 KV 缓存的存在，当 BS=1 时， Transformer 中的每一次矩阵乘法实际上都是矩阵向量乘法。

---

### 🤖 Assistant

这是一个非常深刻的系统层面问题，触及了大模型推理（Inference）性能优化的核心。

简单来说，当 Batch Size ($BS$) = 1 且启用了 **KV Cache** 时，Transformer 在解码（Decoding）阶段的每一步输入序列长度实际上变成了 **1**。因此，数学上的矩阵-矩阵乘法（$M \times K$ 与 $K \times N$）中，$M$ 维度退化为 1，从而在计算上退化为向量-矩阵乘法（Vector-Matrix Multiplication，通常称为 GEMV）。

下面我们从原理、数学维度和计算特性三个方面详细剖析原因。

### 1. 核心原因：KV Cache 改变了输入的定义

在 Transformer 的训练（Training）或推理的预填充（Prefill）阶段，我们是一次性处理整个序列的。但在自回归解码（Autoregressive Decoding）阶段，情况发生了变化。

#### 如果没有 KV Cache
为了预测第 $t$ 个 token，我们需要计算 $t$ 之前所有 token 的 Attention。如果不存缓存，我们需要把 $[x_1, x_2, ..., x_{t-1}]$ 整个序列作为输入再次送入模型，以计算第 $t$ 个位置的 Query、Key 和 Value。此时，输入维度是 $[1, t-1, D_{model}]$，与权重矩阵 $[D_{model}, D_{out}]$ 相乘，这依然是 **矩阵-矩阵乘法 (GEMM)**。

#### 启用 KV Cache 后
KV Cache 的作用是存储之前所有 Token ($1$ 到 $t-1$) 在每一层计算出的 Key 和 Value 向量。
*   当我们要预测第 $t$ 个 token 时，之前的 token 的 Key 和 Value 已经存在显存里了，不需要重新计算。
*   我们需要做的，仅仅是处理 **刚刚生成的那一个** 新 token ($x_{t-1}$)。
*   我们需要计算这个新 token 的 $Q, K, V$，然后取出缓存的 $K_{cache}, V_{cache}$ 进行 Attention 计算。

因此，进入模型每一层的 **有效输入 (Input)** 仅仅是当前这一步的一个 token。

### 2. 数学维度分析

假设模型参数为 $D$ (hidden size)，Batch Size 为 $B=1$。

#### 线性层投影 (Linear Projections)
Transformer 中大量的计算量在于 Linear 层（如 $W_Q, W_K, W_V, W_O$ 以及 MLP 中的 $W_{up}, W_{down}$）。

设权重矩阵 $W$ 的形状为 $[D, D]$。
由于启用了 KV Cache，当前步的输入 $x$ 只是这一个 token 的 embedding。

*   **输入形状**: $[BS, SeqLen, D] = [1, 1, D]$。可以看作一个 $1 \times D$ 的行向量 $\mathbf{x}$。
*   **计算过程**:
    $$ \mathbf{y} = \mathbf{x} \times W $$
*   **维度变化**:
    $$ [1, D] \times [D, D] \rightarrow [1, D] $$

虽然我们在代码中调用的可能依然是 `matmul` 函数，但从线性代数的角度看，一个 **行向量** 乘以一个 **矩阵**，这就是典型的 **向量-矩阵乘法 (GEMV)**。

#### 注意力机制 (Attention)
即使在 Attention 内部，计算特性也发生了改变。

1.  **Q, K, V 生成**: 如上所述，都是 Vector $\times$ Matrix。
2.  **Score 计算 ($Q \times K^T$)**:
    *   $Q_{new}$: 当前 token 的 Query，形状 $[1, D]$ (head 维度忽略或视为拆分)。
    *   $K_{cache}$: 缓存的 Key 矩阵，形状 $[SeqLen, D]$。
    *   计算 $Attention\_Scores = Q_{new} \times K_{cache}^T$。
    *   维度: $[1, D] \times [D, SeqLen] \rightarrow [1, SeqLen]$。
    *   这也是 **Vector-Matrix** 乘法。
3.  **Context 计算 ($Score \times V$)**:
    *   $Scores$: 形状 $[1, SeqLen]$。
    *   $V_{cache}$: 形状 $[SeqLen, D]$。
    *   维度: $[1, SeqLen] \times [SeqLen, D] \rightarrow [1, D]$。
    *   同样是 **Vector-Matrix** 乘法。

### 3. 计算特性差异：Compute Bound vs. Memory Bound

理解这一点对于系统优化至关重要，因为 GEMM 和 GEMV 在硬件上的瓶颈完全不同。

*   **GEMM (Training/Prefill 阶段)**:
    *   操作: 矩阵 $\times$ 矩阵。
    *   算术强度 (Arithmetic Intensity): 高。
    *   对于两个 $N \times N$ 的矩阵，计算量是 $O(N^3)$，数据搬运量是 $O(N^2)$。
    *   **瓶颈**: 通常是 **Compute Bound** (受限于 GPU 的 Tensor Core 算力)。

*   **GEMV (Decoding BS=1 阶段)**:
    *   操作: 向量 $\times$ 矩阵。
    *   算术强度: 极低。
    *   为了计算输出向量的一个元素，我们需要加载权重矩阵的一整列。对于 $1 \times N$ 和 $N \times N$ 的乘法，计算量是 $O(N^2)$，但我们需要把整个 $N \times N$ 的权重矩阵从 HBM (显存) 搬运到芯片上，数据搬运量也是 $O(N^2)$。
    *   比率接近 1:1 (或者 1:2，取决于是否累加)。
    *   **瓶颈**: 它是典型的 **Memory Bound** (受限于显存带宽)。

### 总结

当 $BS=1$ 时，Transformer 自回归解码之所以被称为执行矩阵向量乘法，是因为：

1.  **KV Cache 的存在** 使得我们无需重新计算历史 token，每一步仅需输入 **当前的一个 token**。
2.  输入数据的形状从 $[SeqLen, D]$ 塌缩为 $[1, D]$。
3.  数学上，$1 \times D$ 的向量与 $D \times D$ 的权重矩阵相乘，即为 **Matrix-Vector Multiplication**。

这也是为什么在进行大模型推理优化时（如 vLLM, TensorRT-LLM），对于 Decoding 阶段，我们会极度关注 **显存带宽利用率** 和 **KV Cache 的管理**，而不是纯粹的 FLOPS 算力。

---

### 🧑‍💻 User

这意味着计算完全受限于内存带宽，为什么这个阶段应用 CUDAGraphs 是最有效的优化之一？

---

### 🤖 Assistant

这是一个非常敏锐的问题。你指出了“计算受限于内存带宽（Memory Bound）”，这通常意味着 GPU 的显存传输跑满了。

然而，**当 Batch Size = 1 时，除了 Memory Bound，我们面临着一个更致命的隐形杀手：Kernel Launch Overhead（内核启动开销）。**

这就是为什么 CUDA Graphs 如此有效：它解决的不是“显存带宽不够”的问题，而是解决**“GPU 算得太快，CPU 发号施令来不及”**的问题（即 CPU-Bound 或 Overhead-Bound）。

下面我们详细拆解为什么在 BS=1 时，CUDA Graphs 是救命稻草：

### 1. “小内核”问题 (The Small Kernel Problem)

在 Transformer BS=1 的解码阶段，虽然我们在做矩阵向量乘法（GEMV），但每次运算的数据量其实**非常小**。

*   **计算时间极短**：由于只有 1 个 token，GPU 执行一个 LayerNorm 或一个 GEMV 可能只需要几微秒（$\mu s$）。
*   **启动开销固定**：CPU 下发一个 CUDA Kernel（比如调用 `cudaLaunchKernel`）并把它推送到 GPU 的执行队列中，存在一个固定的物理开销，通常在 **10 $\mu s$ 到 20 $\mu s$** 左右（取决于 CPU 单核性能和驱动）。

**矛盾出现了：**
如果 GPU 执行一个算子只需要 $2 \mu s$，但 CPU 告诉 GPU “去执行这个算子”需要 $10 \mu s$。
$$ \text{Total Time} = \text{Launch Time} + \text{Execution Time} \approx 10 + 2 = 12 \mu s $$
在这个过程中，GPU 实际上只有 $2/12 \approx 16\%$ 的时间在工作，剩下时间都在空等 CPU 的指令。

这就是所谓的 **Overhead Bound（开销受限）**。虽然理论上是 Memory Bound，但由于任务太碎，显存带宽甚至还没来得及跑满，算子就结束了，然后 GPU 进入空闲（Idle）状态等待下一个指令。

### 2. CUDA Graphs 的作用机制

CUDA Graphs 的核心思想是 **“录制与回放”**。

*   **没有 CUDA Graphs**：
    CPU 就像一个微观管理的指挥官，每一层、每一个加法、每一个激活函数，都要单独发令：“做 Linear”、“做 ReLU”、“做 Add”。
    $$ \text{Total Time} = \sum_{i=1}^{N} (\text{CPU\_Launch}_i + \text{GPU\_Exec}_i) $$

*   **使用 CUDA Graphs**：
    我们预先定义好整个 Transformer 层的计算图（Graph）。在推理时，CPU 只需要发一次令：“执行这个图”。GPU 拿到整个图后，会在内部自动调度所有的 kernel，完全不需要 CPU 插手。
    $$ \text{Total Time} = \text{CPU\_Launch}_{\text{graph}} + \sum_{i=1}^{N} \text{GPU\_Exec}_i $$

### 3. 图解性能差异

想象一下时间轴：

**未使用 CUDA Graphs (BS=1):**
```text
CPU: [Launch K1]..........[Launch K2]..........[Launch K3]...
GPU: ...........[Exec K1]...........[Exec K2]...........[Exec K3]
                <--Gap-->            <--Gap-->
```
这里有大量的 **Kernel Gaps**，GPU 处于饥饿状态。显存带宽利用率看起来很低，不是因为读写慢，是因为没东西可读写。

**使用 CUDA Graphs (BS=1):**
```text
CPU: [Launch Graph]
GPU: .............[Exec K1][Exec K2][Exec K3][Exec K4]...
```
GPU 上的算子紧密排列，中间没有间隙。

### 4. 为什么这对 Transformer 解码特别有效？

CUDA Graphs 并不是万能的，它要求计算图的拓扑结构是**静态**的（Static）。Transformer 的解码阶段完美符合这一特性：

1.  **固定的网络结构**：每一层 Decoder Layer 的结构完全一样，循环 $L$ 次。
2.  **固定的输入形状 (BS=1)**：在解码阶段，无论生成到第几个 token，进入 Linear 层的输入形状始终是 $[1, D]$（KV Cache 的增长由指针偏移处理，计算核心逻辑不变）。
3.  **算子数量巨大**：一个大模型可能有 80 层，每层几十个微小算子（RMSNorm, RoPE, QKV proj, Attn, MLP...）。不用 CUDA Graph，每一步推理都要经历数千次 CPU 启动开销，这在 BS=1 时是毁灭性的。

### 总结

虽然 Transformer BS=1 的**计算特性**是 Memory Bound（显存带宽瓶颈），但其实际的**执行性能**往往是 **Latency Bound / Overhead Bound**（延迟/开销瓶颈）。

**CUDA Graphs 的核心价值在于：**
它消除了 CPU 的启动开销，消除了 GPU 算子之间的空隙（Gap）。

这使得 GPU 能够连续不断地执行微小的 GEMV 操作，从而真正地把压力转移回显存带宽上（即让我们**真正**达到 Memory Bound 的理论极限，而不是被卡在 CPU 开销上）。在这个阶段，CUDA Graphs 往往能带来 **2x 到 3x** 的推理速度提升。