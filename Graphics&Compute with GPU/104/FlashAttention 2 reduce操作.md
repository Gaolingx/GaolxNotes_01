# FlashAttention 2 reduce操作

### 🧑‍💻 User

transformer 原版实现的flash attention 2 kernel中有哪些reduce操作？

---

### 🤖 Assistant

FlashAttention-2 是对 FlashAttention 算法的重大改进，其核心优化目标是减少非矩阵乘法（non-matmul）操作的开销，并优化线程块（Thread Block）和 Warp 之间的并行度。

在 FlashAttention-2 的 CUDA Kernel 实现中，**Reduce（规约）操作**主要用于计算 Softmax 的统计量（Max 和 Sum），以维持数值稳定性和归一化。

以下是 FlashAttention-2 原版实现（基于 Tri Dao 的实现）中涉及的主要 Reduce 操作：

### 1. 前向传播 (Forward Pass) 中的 Reduce

在前向传播中，主要目的是计算注意力输出 $O = \text{Softmax}(QK^T)V$。为了实现分块（Tiled）计算且保持数值稳定（Online Softmax），需要维护行方向的统计信息。

#### A. RowMax (行最大值规约)
在计算 $S = QK^T$ 的分块矩阵乘法后，需要对每一行进行 Reduce Max 操作，以找到当前块的最大值。
*   **数学表达：** $m_{ij}^{(block)} = \max_{k} (S_{ik})$
*   **用途：** 用于 Softmax 的数值稳定性技巧（即 $e^{x - m}$），防止指数运算溢出。
*   **Kernel 实现细节：** 
    *   这是一个 **Warp-level reduction**。
    *   每个线程持有该行的一部分数据，首先进行寄存器内的 `max`，然后使用 `__shfl_xor_sync` 指令进行 Warp 内的规约。

#### B. RowSum (行指数和规约)
在利用 Max 值计算出指数项 $P = \exp(S - m_{block})$ 后，需要对每一行进行 Reduce Sum 操作。
*   **数学表达：** $\ell_{ij}^{(block)} = \sum_{k} P_{ik}$
*   **用途：** 作为 Softmax 的分母（归一化因子）。
*   **Kernel 实现细节：**
    *   同样是 **Warp-level reduction**。
    *   利用 `__shfl_xor_sync` 指令在寄存器层面快速求和。

#### C. 全局统计量更新 (Online Softmax Reduce)
这不仅仅是一个简单的操作符，而是一个逻辑上的规约过程。当处理 K 维度的下一个块时，需要将当前块的统计量与之前的全局统计量进行合并（Rescaling）。
*   **更新 Max:** $m_{new} = \max(m_{old}, m_{block})$
*   **更新 Sum:** $\ell_{new} = e^{m_{old} - m_{new}} \ell_{old} + e^{m_{block} - m_{new}} \ell_{block}$
*   **Reduce 属性：** 这本质上是一个跨 Block 的累积规约（Accumulation），最终得到整行的 Max 和 Sum。

---

### 2. 反向传播 (Backward Pass) 中的 Reduce

FlashAttention-2 的反向传播优化尤为重要，因为它需要重计算注意力矩阵。反向传播中的 Reduce 操作主要用于计算梯度的辅助项。

#### A. RowSum (用于计算 $D$ 或 $\Delta$)
在计算 $dQ$（Query 的梯度）之前，需要先计算一个针对每一行的标量项 $D$。
*   **数学表达：** $D_i = \sum_{j} (dO_{ij} \cdot O_{ij})$
    *   其中 $dO$ 是输出的梯度，$O$ 是前向传播的输出。
*   **用途：** 这是 Softmax 梯度推导中的一部分（$\frac{\partial L}{\partial P} = dP = P \cdot (dS - D)$，其中 $dS$ 是分数的梯度）。
*   **Kernel 实现细节：**
    *   这是一个逐元素的乘法后跟随一个 **Row-wise Sum Reduction**。
    *   在 FlashAttention-2 中，通常会在加载 $O$ 和 $dO$ 后立即计算这个值，并将其存储在共享内存或寄存器中供后续使用。

#### B. RowMax (用于重计算 Softmax)
在反向传播中，为了节省显存，通常不保存前向传播的 Attention Matrix $P$，而是根据 $Q$ 和 $K$ 重计算。重计算时同样需要 **RowMax** 来保证数值稳定性，逻辑与前向传播相同。

#### C. RowSum (用于重计算 Softmax 分母 L)
反向传播通常会从全局内存（HBM）读取前向传播计算好的 LogSumExp（即 $L$），但在某些变体或重计算步骤中，可能需要局部重新规约。

---

### 3. FlashAttention-2 相比 V1 在 Reduce 上的优化

FlashAttention-2 在 Reduce 操作上的核心改进在于**并行维度的划分**：

1.  **从 Block 级到 Warp 级：**
    *   **V1:** 将 $Q$ 切分到不同的 Thread Block，在 Block 内部利用 Shared Memory 进行 Reduce。这导致了较多的 Shared Memory 读写同步。
    *   **V2:** 重新设计了算法，使得外层循环在 Warp 级别进行。Reduce 操作主要在 **寄存器（Registers）** 之间通过 **Warp Shuffle (`__shfl_`)** 指令完成。这意味着 Reduce 操作几乎不需要访问 Shared Memory，极大地减少了 bank conflicts 和同步开销。

2.  **减少非矩阵乘法操作：**
    *   FlashAttention-2 特别关注减少 FLOPs 较低但在内存带宽上昂贵的 Reduce 操作。通过优化 pipeline，它将 Reduce 操作（Max/Sum）与 Matmul 操作更紧密地交织在一起，掩盖了 Reduce 的延迟。

### 总结

在 FlashAttention-2 Kernel 中，最关键的 Reduce 操作是：
1.  **`reduce_max`**: 用于计算 Softmax 的数值稳定项 $m$。
2.  **`reduce_sum`**: 用于计算 Softmax 的分母 $\ell$（前向）以及梯度项 $D$（反向）。

这些操作主要通过 CUDA 的 **Warp Shuffle intrinsics** 实现，以避免 Shared Memory 的开销。