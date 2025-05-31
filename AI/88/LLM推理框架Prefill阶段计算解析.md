# LLM推理框架Prefill阶段计算解析

### 🧑‍💻 User

llm推理框架的prefill阶段到底在计算什么？

---

### 🤖 Assistant

LLM推理框架中的**Prefill阶段**（也称为**Context Encoding阶段**或**Prompt Processing阶段**）是处理整个输入提示（Prompt）并生成第一个输出token所需全部状态的关键步骤。**其核心计算的是输入序列的完整表示，并为后续的自回归生成构建初始的KV缓存**。

以下是Prefill阶段具体计算内容的详细拆解：

---

### 1. **输入序列的完整嵌入**
   * **Token Embedding:** 将输入提示中的每个token转换为对应的词嵌入向量（Token Embedding）。
   * **Positional Embedding:** 为每个token添加位置编码信息，使模型理解token的顺序。
   * **结果:** 得到一个形状为 `[Batch Size, Sequence Length, Hidden Dimension]` 的张量，表示整个输入序列的上下文化初始表示。

---

### 2. **逐层计算Transformer Block的输出**
   模型会按顺序处理每一层Transformer Block（Decoder层）。在Prefill阶段，**整个输入序列是已知的**，这带来了计算上的关键特点：
   * **a. Self-Attention计算（核心）:**
     * **Q, K, V矩阵计算:** 对每个token的嵌入向量，通过线性变换生成Query、Key、Value矩阵。
     * **全注意力计算:**
       * 计算每个token的Query向量与序列中**所有token**（包括它自身和它之后的token）的Key向量的点积：`Attention Scores = Q * K^T`。
       * **应用因果掩码：** 由于LLM是自回归的，必须屏蔽“未来”信息。将Attention Scores矩阵中对应“未来”token的位置（通常是上三角部分）设置为一个非常大的负数（如 `-inf`），这样在后续的Softmax中这些位置的概率会趋近于0。
       * **Softmax:** 对每个token的Attention Scores行应用Softmax，得到注意力权重矩阵 `Attention Weights`（形状 `[Seq Len, Seq Len]`）。
       * **加权求和:** 用 `Attention Weights` 对 Value 矩阵进行加权求和，得到每个token新的上下文表示 `Attention Output = Attention Weights * V`。
     * **并行性:** **这是Prefill阶段计算量最大且最耗时的部分。** 得益于序列的完全已知性，Prefill阶段可以对整个序列的注意力计算进行高效的**批量矩阵乘法**，充分利用GPU/TPU等硬件的并行计算能力。计算复杂度为 `O(SeqLen^2 * HiddenDim)`。
   * **b. Feed-Forward Network (FFN) 计算:**
     * 对Self-Attention的输出应用前馈神经网络（通常是两层带有激活函数的线性变换）。
   * **c. 残差连接与层归一化:**
     * 将Self-Attention和FFN的输出分别与它们的输入进行残差连接，并应用层归一化。

---

### 3. **构建初始的KV缓存**
   * **目的:** 这是Prefill阶段**极其重要**的输出，直接服务于后续的Decoding（Generate）阶段。
   * **计算内容:** 在计算每一层的Self-Attention时，会生成该层所有token的Key向量和Value向量。
   * **缓存:** 这些Key向量和Value向量（形状通常是 `[Batch Size, Num Heads, Seq Len, Head Dimension]`）会被**存储**在内存中（即KV缓存）。
   * **意义:** 在后续的Decoding阶段，当模型逐个生成新token时，它只需要：
     * 计算**当前新token**的Query向量。
     * 将当前新token的Query向量与KV缓存中**之前所有token**（包括Prefill阶段处理的所有token和已生成的部分token）的Key向量进行点积计算注意力分数。
     * 避免了在生成每个新token时都重新计算整个历史序列的Key和Value向量，从而**极大地提高了生成效率**。

---

### 4. **输出首个Token的Logits**
   * 经过所有Transformer层后，Prefill阶段最终会输出序列**最后一个token位置**对应的隐藏状态。
   * 这个隐藏状态通过一个**语言模型头**（通常是线性层 + Softmax），被转换为整个词汇表上的概率分布（Logits）。
   * **这个Logits就是第一个将要生成的输出token的预测结果。**

---

### 总结Prefill阶段的核心任务

1. **完整理解上下文:** 利用整个输入序列已知的优势，通过并行计算高效地计算出每个输入token在完整上下文下的深度表示。
2. **构建生成起点:** 计算并输出**第一个**输出token的预测（Logits）。
3. **初始化KV缓存:** 计算并存储输入序列中所有token在所有Transformer层中的Key和Value向量，为后续高效的自回归生成奠定基础。

### Prefill vs. Decoding (Generate) 阶段的关键区别

| 特性         | Prefill 阶段                        | Decoding (Generate) 阶段            |
| :----------- | :---------------------------------- | :---------------------------------- |
| **输入**     | **整个**输入提示 (Prompt)           | **单个**新生成的token               |
| **输出**     | 首个输出token的Logits + **完整KV缓存** | 下一个token的Logits + **更新KV缓存** |
| **注意力范围** | 整个输入序列 (应用因果掩码)         | 所有历史token (Prompt + 已生成部分) |
| **并行性**   | **高:** 可对整个序列做并行矩阵计算  | **低:** 主要计算与新token相关的部分 |
| **计算复杂度** | `O(SeqLen² * HiddenDim)` (主要瓶颈) | `O(SeqLen * HiddenDim)` (常数级)    |
| **主要耗时** | 长序列处理可能非常慢                | 通常很快，但累积时间随输出长度增长  |
| **目的**     | 初始化状态，理解整个上下文          | 逐个预测下一个token                 |

**简单来说：Prefill阶段是“备课”，它一次性读完所有教材（输入提示），做好详细笔记（KV缓存），并准备好回答第一个问题（输出第一个token）。而Decoding阶段则是“答题”，它根据备课笔记和已经写出的答案，快速思考并写下下一个词。** 理解Prefill阶段的计算内容对于优化LLM推理延迟（尤其是首Token延迟）至关重要。

---

### 🧑‍💻 User

这个阶段是否会计算attention weight，是怎么计算的？

---

### 🤖 Assistant

是的，**Prefill 阶段会完整地计算输入序列中每个 token 的 Attention Weight（注意力权重）**。这是理解整个输入上下文的核心步骤。

以下是 Prefill 阶段 Attention Weight 计算的详细过程：

### 核心计算步骤

1.  **生成 Q, K, V 矩阵：**
    *   对于当前层的输入（初始为嵌入层输出，后续为前一层的输出），应用三个独立的线性变换（权重矩阵 `W_q`, `W_k`, `W_v`）：
        *   `Q = input * W_q` (Query 向量)
        *   `K = input * W_k` (Key 向量)
        *   `V = input * W_v` (Value 向量)
    *   形状：通常 `[batch_size, seq_len, num_heads, head_dim]`。为了方便后续矩阵运算，通常会重塑（reshape）和转置（transpose）为 `[batch_size * num_heads, seq_len, head_dim]`。

2.  **计算注意力分数 (Attention Scores)：**
    *   计算每个 Query 向量 (`Q[i]`) 与序列中 **所有** Key 向量 (`K[j]`, j = 1...seq_len) 的点积（相似度）：
        *   `Scores = Q @ K.transpose(-2, -1)` (矩阵乘法)
    *   形状：`[batch_size * num_heads, seq_len, seq_len]`。这是一个方阵，其中 `Scores[i, j]` 表示第 `i` 个 token 的 Query 对第 `j` 个 token 的 Key 的注意力分数。

3.  **缩放 (Scaling)：**
    *   为了防止点积结果过大导致 Softmax 梯度消失，通常将分数除以 Key 向量维度的平方根：
        *   `Scores = Scores / sqrt(d_k)` (其中 `d_k` 是 `head_dim`，即每个注意力头的维度)

4.  **应用因果掩码 (Causal Masking)：**
    *   **这是 Prefill 阶段区别于非因果模型（如 BERT）的关键一步！** 因为 LLM 是自回归生成模型，在预测第 `i` 个 token 时，它只能“看到”它之前的 token (`1` 到 `i-1`) 和它自身 (`i`)，不能“看到”它之后的 token (`i+1` 到 `seq_len`)。
    *   创建一个掩码矩阵 `Mask`：
        *   形状与 `Scores` 相同：`[batch_size * num_heads, seq_len, seq_len]`。
        *   对于矩阵的第 `i` 行（对应第 `i` 个 token 的 Query），`Mask[i, j]` 定义为：
            *   `0` (False / 保留)：当 `j <= i` (表示第 `j` 个 token 是第 `i` 个 token 的“过去”或“现在”)
            *   `-inf` (负无穷大 / 屏蔽)：当 `j > i` (表示第 `j` 个 token 是第 `i` 个 token 的“未来”)
    *   将这个掩码矩阵加到 `Scores` 上：
        *   `Masked_Scores = Scores + Mask`
    *   **效果：** 经过 Softmax 后，那些被加上 `-inf` 的位置对应的概率将变为 0。这样，第 `i` 个 token 的注意力权重就只分配给第 1 到第 `i` 个 token。

5.  **应用 Softmax：**
    *   对 `Masked_Scores` 的**每一行**（即每个 Query 对应的那一行）应用 Softmax 函数，将其转换为概率分布（和为 1）：
        *   `Attention_Weights = softmax(Masked_Scores, dim=-1)` (在最后一个维度 `seq_len` 上进行 Softmax)
    *   形状：仍然是 `[batch_size * num_heads, seq_len, seq_len]`。`Attention_Weights[i, j]` 表示第 `i` 个 token 的 Query 分配给第 `j` 个 token 的 Key 的注意力权重。

6.  **计算加权和 (Context Vector)：**
    *   使用 `Attention_Weights` 对 Value 向量 `V` 进行加权求和，得到每个 token 新的上下文表示：
        *   `Output = Attention_Weights @ V` (矩阵乘法)
    *   形状：`[batch_size * num_heads, seq_len, head_dim]`。
    *   这个 `Output` 就是 Self-Attention 的输出，它融合了每个 token 及其允许看到的上下文信息。

7.  **重塑/合并多头输出：**
    *   将 `Output` 的形状重塑/转置回 `[batch_size, seq_len, num_heads * head_dim]` 或 `[batch_size, seq_len, hidden_size]`。
    *   通常还会通过一个线性层 (`W_o`) 来融合多个注意力头的信息。

### Prefill 阶段 Attention Weight 的关键特点

1.  **完整计算：** Prefill 阶段会为输入序列中**每一个** token，计算它对序列中**它自身及其之前所有 token**（因为因果掩码）的 Attention Weight。
2.  **矩阵并行：** 由于整个输入序列 `seq_len` 是已知的，计算 `Q @ K^T` (`Scores`) 和 `Attention_Weights @ V` (`Output`) 都可以高效地通过**批量矩阵乘法 (GEMM)** 完成。这是 Prefill 阶段能利用硬件并行能力的主要原因，但也导致了 `O(seq_len^2)` 的计算复杂度。
3.  **因果性：** 通过严格的下三角掩码（`j <= i`）确保模型在编码每个位置时，只关注该位置及之前的信息，符合自回归生成的要求。
4.  **KV 缓存的基础：** 在计算 Attention Weight 的过程中生成的 `K` 和 `V` 矩阵（形状 `[batch_size, num_heads, seq_len, head_dim]`）会被存储下来，形成 KV 缓存。**KV 缓存本质上就是序列中每个 token 在每个层计算出来的 Key 向量和 Value 向量的集合**。后续 Decoding 阶段生成新 token 时，只需要计算新 token 的 `Q`，然后与缓存中所有历史 `K`（包括 Prefill 阶段计算的和之前生成步骤缓存的）计算 Attention Scores，用缓存的 `V` 进行加权求和。这避免了重复计算历史 token 的 K 和 V。

### 总结

Prefill 阶段的核心计算任务之一就是为输入序列中的**每一个位置（token）**，计算其对**自身及之前所有位置（token）** 的 **Attention Weight**。这个过程通过高效的矩阵乘法并行完成，但复杂度随序列长度平方增长。计算得到的 Attention Weight 用于生成每个 token 的上下文表示（输出），同时该阶段计算出的 **K 和 V 向量被保存为 KV 缓存**，这是后续高效自回归生成的基础。应用**因果掩码**是确保模型符合自回归生成规则的关键步骤。

---

### 🧑‍💻 User

prefill阶段为什么可以并行进行？

---

### 🤖 Assistant

Prefill 阶段能够进行高效并行计算的核心原因在于：**整个输入序列在开始时就是完全已知且固定的**。这与 Decoding 阶段逐个生成 token 的序列特性形成了鲜明对比。这种“已知性”为利用硬件（如 GPU/TPU）强大的并行计算能力（尤其是矩阵/张量运算）创造了理想条件。

以下是 Prefill 阶段可以高度并行化的关键点：

1.  **输入数据的并行性 (Data Parallelism across Tokens):**
    *   **核心优势：** 所有 token 的嵌入向量（经过 Token Embedding + Positional Embedding）在 Prefill 开始时就已经全部存在，构成一个完整的张量 `[batch_size, seq_len, hidden_dim]`。
    *   **并行处理：** 模型可以一次性处理**整个批次 (`batch_size`) 内所有序列的所有 token (`seq_len`)**。后续的线性变换（如生成 Q、K、V 的投影）、激活函数、层归一化等操作，都可以在 `seq_len` 这个维度上进行**完全并行的元素级或向量级操作**。GPU 的数千个核心可以同时处理这些独立的计算单元。

2.  **注意力计算中的矩阵乘法并行性 (Parallel Matrix Multiplications in Attention):**
    *   **Self-Attention 的本质是矩阵乘法：** Prefill 阶段 Attention 计算的关键步骤 `Scores = Q @ K^T` 和 `Output = Attention_Weights @ V` 都是大规模的**稠密矩阵乘法 (GEMM)**。
    *   **利用硬件优势：** GPU/TPU 的架构（尤其是 Tensor Core/Matrix Engine）就是为高效执行大型 GEMM 操作而优化的。它们能在极短时间内完成 `O(seq_len^2 * hidden_dim)` 量级的计算。
    *   **一次性计算整个注意力矩阵：** 因为整个序列已知：
        *   `Q` 矩阵的形状是 `[batch_size * num_heads, seq_len, head_dim]`
        *   `K` 矩阵的形状是 `[batch_size * num_heads, seq_len, head_dim]` (转置后为 `[batch_size * num_heads, head_dim, seq_len]`)
        *   **一次 GEMM 操作**就能计算出整个注意力分数矩阵 `Scores = Q @ K^T`，其形状为 `[batch_size * num_heads, seq_len, seq_len]`。这个操作在硬件上是高度并行的。
        *   同样，**一次 GEMM 操作**就能完成 `Output = Attention_Weights @ V`，得到所有 token 的新表示。
    *   **对比 Decoding：** Decoding 阶段每次只新增一个 token，计算 `Scores` 时是新 token 的 `Q_new` (形状 `[batch_size * num_heads, 1, head_dim]`) 与 KV 缓存中所有历史 Key `K_cache` (形状 `[batch_size * num_heads, current_seq_len, head_dim]`) 的点积，这是一个**矩阵-向量乘法 (GEMV)**，计算量远小于 GEMM，并行度也低很多。

3.  **因果掩码的并行应用 (Parallel Mask Application):**
    *   因果掩码是一个**固定的、预先计算好的**下三角矩阵（主对角线及以下为 0，以上为 `-inf`）。
    *   将掩码加到 `Scores` 矩阵上的操作 (`Masked_Scores = Scores + Mask`) 是一个简单的、在 `[batch_size * num_heads, seq_len, seq_len]` 张量上的**逐元素加法 (Element-wise Addition)**。这种操作在 GPU 上也是高度并行的。

4.  **Softmax 的行级并行 (Row-wise Parallel Softmax):**
    *   Softmax 是在 `Scores`/`Masked_Scores` 矩阵的**每一行**上独立进行的（即对每个 Query token 的注意力分数分布进行归一化）。
    *   由于每一行的计算是相互独立的，GPU 可以**同时并行处理所有行**（所有 `batch_size * num_heads * seq_len` 个行）。

5.  **层间计算的潜在流水线/张量并行 (Potential Pipeline/Tensor Parallelism across Layers):**
    *   虽然 Transformer 层本身是顺序依赖的（第 `L` 层的输入依赖第 `L-1` 层的输出），但在大规模模型推理或分布式系统中，可以利用以下并行策略加速 Prefill：
        *   **流水线并行 (Pipeline Parallelism):** 将模型的不同层划分到不同的设备上。当一个设备完成第 `L` 层对某批数据的计算后，可以立即开始计算下一批数据，而第 `L+1` 层设备同时处理第 `L` 层传来的上一批数据的结果。
        *   **张量并行/模型并行 (Tensor/Model Parallelism):** 将单层内的大型矩阵运算（如庞大的线性层或 Attention 头）拆分到多个设备上协同计算。例如，将 `W_q`/`W_k`/`W_v`/`W_o` 矩阵按列拆分，每个设备计算部分结果，最后通过通信聚合。
    *   这些并行策略在 Prefill 阶段处理完整序列时也能有效利用起来。

**总结 Prefill 阶段高度并行的关键：**

*   **输入已知且固定：** 所有 token 一次性提供。
*   **计算模式统一：** 所有 token 执行相同的操作序列（线性层、Attention、FFN、Norm 等）。
*   **核心操作为大规模 GEMM：** 注意力计算中的 `Q@K^T` 和 `Attn@V` 是大型稠密矩阵乘法，是硬件并行加速的“甜点”。
*   **元素级/向量级操作并行：** 如嵌入、激活函数、残差加、层归一化、掩码加法等，天然适合 SIMD/SIMT 架构。
*   **独立计算单元：** 如 Softmax 按行独立计算。

**与 Decoding 阶段的对比 (突显并行性差异)：**

| 特性             | Prefill 阶段                                | Decoding (Generate) 阶段                    | 对并行性的影响                               |
| :--------------- | :----------------------------------------- | :------------------------------------------ | :------------------------------------------- |
| **输入特性**     | **完整、已知、固定**的长序列 (`seq_len`)    | **单个**新增 token (序列长度 `current_len` 递增) | Prefill 可并行处理所有 token；Decoding 只能串行处理新增 token |
| **核心 Attention 计算** | **GEMM:** `Q_full @ K_full^T` (大矩阵乘)     | **GEMV:** `q_new @ K_cache^T` (矩阵-向量乘) | GEMM 并行度极高；GEMV 并行度较低             |
| **计算复杂度**   | `O(seq_len^2 * hidden_dim)`                | `O(current_len * hidden_dim)` (每次生成)     | Prefill 计算量大但可高度并行；Decoding 单步计算量小但需串行累积 |
| **硬件利用率**   | **高** (能充分利用 GPU/TPU 的算力，尤其是 Tensor Core) | **较低** (计算强度不如 GEMM，易受内存带宽限制) | Prefill 能“喂饱”硬件；Decoding 常“吃不饱”    |
| **主要瓶颈**     | **计算瓶颈** (处理长 `seq_len` 时的大 GEMM 耗时) | **内存带宽瓶颈** (频繁读写 KV 缓存和模型权重) |                                              |

**简单来说：Prefill 阶段像是对一本已知的书进行全文分析，可以同时翻阅所有章节、对比所有段落、做全书索引（KV缓存）。而 Decoding 阶段像是写读后感，只能根据已经写好的部分（和之前做好的索引）一个字一个字地往下写。前者的“全文已知”特性使得大规模并行计算成为可能且高效；后者的“逐步生成”特性则限制了并行能力。** 理解这种并行性差异对于优化 LLM 推理性能（特别是首 Token 延迟和吞吐量）至关重要。