# FlashAttention Online Softmax实现与优势

### 🧑‍💻 User

在标准的flashattention实现中，说说online softmax是如何实现的，相比完整的softmax优势有哪些？

---

### 🤖 Assistant

在标准的 FlashAttention 实现中，**Online Softmax**（在线 Softmax）是其能够实现 **Tiling（分块计算）** 和 **Kernel Fusion（算子融合）** 的数学基石。

它使得我们可以在**不需要一次性看到所有 Attention Score** 的情况下，分块地计算 Softmax 的归一化因子和最终输出。

以下是 Online Softmax 的实现原理及其相对于完整 Softmax 的优势分析。

---

### 一、 Online Softmax 是如何实现的？

要理解 Online Softmax，我们需要先回顾标准的 Softmax 以及为何它在大矩阵下难以分块。

#### 1. 标准 Safe Softmax (回顾)
给定一个向量 $x$，标准的数值稳定 Softmax（Safe Softmax）计算如下：
1.  找到最大值：$m = \max(x)$
2.  计算指数和：$L = \sum_{i} e^{x_i - m}$
3.  计算结果：$y_i = \frac{e^{x_i - m}}{L}$

**问题：** 为了计算 $L$ 和 $m$，你需要遍历整个向量 $x$。在 Attention 中，$x$ 对应矩阵的一行（长度为序列长度 $N$）。如果 $N$ 很大，我们无法将整行一次性放入 GPU 的快速显存（SRAM）中。如果我们分块计算，每次只能看到 $x$ 的一部分，此时我们不知道全局的 $m$ 和 $L$ 是多少。

#### 2. Online Softmax 算法推导
Online Softmax 允许我们维护局部的统计量，并在读取新数据块时动态更新它们。

假设我们将输入向量 $x$ 分为两个块：$x^{(1)}$ 和 $x^{(2)}$，即 $x = [x^{(1)}, x^{(2)}]$。

**第一步：处理第一个块**
*   局部最大值：$m_1 = \max(x^{(1)})$
*   局部指数和：$l_1 = \sum e^{x^{(1)} - m_1}$
*   暂时输出（未归一化的 Attention $\times$ V）：$O_1 = P_1 V_1$ （这里简化表示）

**第二步：处理第二个块（Online 更新）**
当我们读取 $x^{(2)}$ 时，计算其局部统计量：
*   局部最大值：$m_2 = \max(x^{(2)})$
*   局部指数和：$l_2 = \sum e^{x^{(2)} - m_2}$

此时，我们需要得到全局的 $m_{new}$ 和 $L_{new}$，公式如下：

1.  **更新全局最大值：**
    $$m_{new} = \max(m_1, m_2)$$

2.  **更新全局指数和：**
    我们需要将 $l_1$ 和 $l_2$ 统一到新的最大值 $m_{new}$ 下进行相加。
    $$L_{new} = l_1 \cdot e^{m_1 - m_{new}} + l_2 \cdot e^{m_2 - m_{new}}$$

3.  **更新输出结果（Output）：**
    之前的输出 $O_1$ 是基于 $m_1$ 计算的，现在需要根据 $m_{new}$ 进行“修正”（Rescale），并加上第二块的贡献。
    $$O_{new} = O_1 \cdot e^{m_1 - m_{new}} + P_2 V_2 \cdot e^{m_2 - m_{new}}$$

    *注意：FlashAttention 实际上是在累加非归一化的结果，最后统一除以 $L_{new}$，或者在循环中动态更新归一化结果。*

#### 3. FlashAttention 中的具体实现
在 FlashAttention 的 CUDA Kernel 中，Online Softmax 嵌入在 Tiling 循环中。

设定：
*   **外层循环**：遍历 $K, V$ 的块（Block $j$）。
*   **内层循环**：遍历 $Q$ 的块（Block $i$）。
*   在处理具体的 Query Block $Q_i$ 时，我们不断加载 $K_j, V_j$。

对于当前的 $Q_i$ 和第 $j$ 个块 $K_j, V_j$：
1.  计算 Score：$S_{ij} = Q_i K_j^T$。
2.  计算当前块的 $m_{ij} = \text{rowmax}(S_{ij})$ 和 $l_{ij} = \text{rowsum}(e^{S_{ij} - m_{ij}})$。
3.  **更新运行中的统计量（Running Statistics）：**
    *   $m_{new} = \max(m_{prev}, m_{ij})$
    *   $l_{new} = e^{m_{prev} - m_{new}} \cdot l_{prev} + e^{m_{ij} - m_{new}} \cdot l_{ij}$
4.  **更新输出 $O_i$：**
    $$O_i^{new} = \text{diag}(l_{prev} / l_{new}) \cdot e^{m_{prev} - m_{new}} \cdot O_i^{prev} + \frac{e^{S_{ij} - m_{new}}}{l_{new}} \cdot V_j$$
    *(注：具体的实现中通常会把除法 $1/L$ 留到最后一步，中间只维护分子)*

通过这种方式，FlashAttention 只需要在 SRAM 中存储当前块的 $S_{ij}$ 和累计的 $O, m, l$，而不需要将整个 $N \times N$ 的矩阵写入 HBM。

---

### 二、 相比完整的 Softmax，优势有哪些？

Online Softmax 是 FlashAttention 能够实现 **“IO 感知（IO-Aware）”** 加速的核心，其优势主要体现在内存访问和存储复杂度上。

#### 1. 内存复杂度降低：$O(N^2) \rightarrow O(N)$
*   **标准 Softmax：** 通常需要先算出完整的 $S = QK^T$ 矩阵，大小为 $N \times N$。对于长序列（如 $N=8192$ 或更大），这个矩阵非常巨大，必须存储在显存（HBM）中。
*   **Online Softmax：** 我们不需要由显式地存储 $N \times N$ 的 Attention 矩阵。我们只需要存储统计量 $m$ 和 $l$，其大小为 $N \times 1$（或者 $N \times \text{Head数}$）。这使得空间复杂度从二次方降为线性。

#### 2. 减少 HBM 读写（IO Reduction）与算子融合
*   **标准流程：**
    1.  从 HBM 读 $Q, K$ -> 算 $S$ -> 写 $S$ 到 HBM。
    2.  从 HBM 读 $S$ -> 算 Softmax($P$) -> 写 $P$ 到 HBM。
    3.  从 HBM 读 $P, V$ -> 算 $O$ -> 写 $O$ 到 HBM。
    这里有大量的 HBM 访问，带宽成为瓶颈。
*   **FlashAttention (Online Softmax)：**
    所有的计算都在 SRAM（GPU 的片上高速缓存）中进行。
    1.  加载小块 $Q, K, V$ 到 SRAM。
    2.  计算 Score，应用 Online Softmax 更新 Accumulator。
    3.  只将最终结果 $O$ 写回 HBM。
    **中间巨大的 $S$ 和 $P$ 矩阵从未离开过 SRAM，甚至从未完整生成过。** 这大幅减少了对 HBM 带宽的需求，从而显著提高了速度。

#### 3. 数值稳定性（Numerical Stability）
Online Softmax 在数学上与标准的 Safe Softmax 是等价的。它在每一步更新时都动态地减去了当前的全局最大值 $m_{new}$。这意味着在计算过程中，指数项 $e^{x - m}$ 永远不会溢出（因为指数部分永远 $\le 0$），保证了与标准 Safe Softmax 相同的数值稳定性，甚至在分块累加过程中可能比某些低精度的全局累加更稳健。

### 总结

Online Softmax 的核心价值在于它打破了 Softmax 必须“**先看全貌，再做归一**”的限制。通过**动态缩放（Rescaling）**历史结果，它允许 Attention 机制在流式/分块的数据上计算，从而使得 FlashAttention 能够将 $O(N^2)$ 的内存读写优化掉，实现数倍的性能提升。