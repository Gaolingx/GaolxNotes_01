# LLM推理框架Prefill阶段FFN计算详解

### 🧑‍💻 User

llm推理框架运行一次推理，prefill阶段，feedforward需要进行哪些计算？

---

### 🤖 Assistant

在LLM推理框架的 **Prefill 阶段**，当处理**整个输入序列**时，**FeedForward Network (FFN)** 模块需要进行以下核心计算步骤（针对序列中的**每一个位置/Token**，并且通常是**并行批量计算**的）：

1.  **输入获取：**
    *   接收来自当前 **Transformer Block** 中 **Multi-Head Attention (MHA)** 模块的输出张量（通常已经经过了 Add & Norm 操作）。记这个输入张量为 `X`，其维度通常为 `[batch_size, sequence_length, hidden_dim]` 或 `[sequence_length, hidden_dim]`（单样本推理时）。

2.  **第一个线性变换：**
    *   将输入 `X` 通过一个**稠密层（全连接层）** `W_up`（有时也称为 `intermediate_dense`）。
    *   计算： `H = X @ W_up + b_up` (其中 `@` 表示矩阵乘法)
    *   维度变化：
        *   `X` 形状: `[batch_size, seq_len, hidden_dim]`
        *   `W_up` 形状: `[hidden_dim, intermediate_dim]` (典型值：`intermediate_dim = 4 * hidden_dim`)
        *   `H` 形状: `[batch_size, seq_len, intermediate_dim]`

3.  **激活函数：**
    *   对中间结果 `H` 应用一个**非线性激活函数** `G`。
    *   计算： `A = G(H)`
    *   最常用的激活函数是 **GeLU (Gaussian Error Linear Unit)**，有时也会用 **ReLU (Rectified Linear Unit)** 或 **Swish/SiLU**。
    *   输出 `A` 的形状与 `H` 相同: `[batch_size, seq_len, intermediate_dim]`

4.  **第二个线性变换：**
    *   将激活后的结果 `A` 通过**另一个稠密层** `W_down`（有时也称为 `output_dense`）。
    *   计算： `Y = A @ W_down + b_down`
    *   维度变化：
        *   `A` 形状: `[batch_size, seq_len, intermediate_dim]`
        *   `W_down` 形状: `[intermediate_dim, hidden_dim]`
        *   `Y` 形状: `[batch_size, seq_len, hidden_dim]` (恢复到与输入 `X` 相同的隐藏维度)

5.  **残差连接与层归一化：**
    *   **注意：** 严格来说，残差连接和层归一化 **不属于 FFN 模块本身的计算**，但它们是 FFN 输出后紧接着在 Transformer Block 中发生的操作，与 FFN 的执行紧密相关。
    *   **残差连接：** 将 FFN 的输出 `Y` 与 FFN 的原始输入 `X`（即 MHA 的输出）相加： `Z = Y + X`
    *   **层归一化：** 对残差连接的结果 `Z` 应用层归一化： `Output = LayerNorm(Z)`
    *   这个 `Output` 将作为下一个 Transformer Block 的输入（如果是最后一个 Block，则进入最后的输出层）。

**Prefill 阶段的关键特点（影响 FFN 计算）：**

1.  **处理完整序列：** Prefill 阶段一次性处理用户提供的整个输入提示（Prompt）。这意味着 `sequence_length` 是输入提示的长度（可能很长）。
2.  **批量并行计算：** 推理框架会充分利用硬件（GPU/TPU）的并行能力，对序列中**所有位置**的 FFN 计算进行**批量处理**。框架会调用高度优化的 **GEMM (General Matrix Multiply)** 库（如 cuBLAS, cuDNN, oneDNN）来执行步骤 2 和步骤 4 中的矩阵乘法 `X @ W_up` 和 `A @ W_down`。这是 Prefill 阶段计算开销的主要部分之一。
3.  **参数加载：** FFN 的权重（`W_up`, `b_up`, `W_down`, `b_down`）在整个 Prefill 阶段（以及后续的 Decode 阶段）都是**只读**的，会从模型的参数存储（通常在 GPU 显存或 TPU HBM 中）加载到计算单元。
4.  **计算密集型：** 由于 `intermediate_dim` 通常是 `hidden_dim` 的 4 倍，FFN 中的两个矩阵乘法操作涉及大量的浮点运算（FLOPs），使其成为 Prefill 阶段的主要计算瓶颈之一（另一个是 MHA 中的 QKV 投影和 Attention 计算）。
5.  **内存带宽敏感：** 加载庞大的权重矩阵（尤其是 `W_up` 和 `W_down`）和中间激活张量（`H`, `A`）需要很高的内存带宽。优化内存访问模式对性能至关重要。
6.  **独立于 Token：** FFN 的计算**只依赖于当前 Token 的向量**。在 Prefill 阶段，序列中所有位置的 FFN 计算是**相互独立**的（不像 MHA 需要计算 Token 之间的依赖关系），因此非常适合并行处理。

**总结 Prefill 阶段 FFN 的计算流程：**

> `(MHA Output) -> [FFN: X -> (Linear Up: X @ W_up) -> (Activation: G) -> (Linear Down: A @ W_down) -> Y] -> (Residual Add: Y + X) -> (LayerNorm) -> (Next Block/Output)`

理解 Prefill 阶段 FFN 的这些计算步骤和特点是优化 LLM 推理性能（减少延迟、提高吞吐量）的关键基础。