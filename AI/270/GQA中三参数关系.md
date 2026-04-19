# GQA中三参数关系

### 🧑‍💻 User

gqa中，num_key_value_heads、num_attention_heads、head_dim之间有什么关系？

---

### 🤖 Assistant

在 GQA（Grouped-Query Attention，分组查询注意力机制）中，`num_key_value_heads`、`num_attention_heads` 和 `head_dim` 这三个参数共同决定了注意力机制的结构、计算量以及显存占用。

它们之间的核心关系可以总结为：**Query 头被分成若干组，每组共享一个 Key/Value 头，而所有的头（无论是 Q、K 还是 V）都保持相同的内部维度 `head_dim`。**

以下是这三个参数的具体关系和详细解析：

### 1. 核心数学关系：分组比例 (Group Ratio)

在 GQA 中，Query 头的数量通常大于 Key 和 Value 头的数量。它们之间的倍数关系决定了“每组”的大小：

$$ G = \frac{\text{num\_attention\_heads}}{\text{num\_key\_value\_heads}} $$

*   **`num_attention_heads` ($H_q$)**：Query (Q) 头的总数。
*   **`num_key_value_heads` ($H_{kv}$)**：Key (K) 和 Value (V) 头的总数。
*   **$G$ (Group size)**：表示多少个 Query 头共享同一组 Key 和 Value 头。

**约束条件：** `num_attention_heads` 必须能够被 `num_key_value_heads` 整除。

### 2. head_dim (每个头的维度) 的一致性

**`head_dim`** ($d_h$) 是单个注意力头内部的向量特征维度。
*   **关系：** 无论是 Query 头、Key 头还是 Value 头，它们的 `head_dim` **必须完全相同**。
*   **原因：** 注意力分数的计算公式是 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_h}}) V$。由于 Q 和 K 需要进行点积计算 ($Q \cdot K^T$)，它们的特征维度必须匹配。
*   **与模型隐藏层的关系：** 通常情况下，模型的全局隐藏层维度 `hidden_size` ($D$) 与 `head_dim` 的关系为：
    $$ D = \text{num\_attention\_heads} \times \text{head\_dim} $$

---

<details>
<summary><b>👉 展开查看：它们在张量形状 (Tensor Shapes) 中的具体体现</b></summary>

在模型代码的具体实现中（例如经过线性投影层后），这三个参数直接决定了张量的形状（假设输入张量形状为 `[batch_size, seq_len, hidden_size]`）：

1.  **Query (Q) 的形状:**
    `[batch_size, seq_len, num_attention_heads, head_dim]`
2.  **Key (K) 的形状:**
    `[batch_size, seq_len, num_key_value_heads, head_dim]`
3.  **Value (V) 的形状:**
    `[batch_size, seq_len, num_key_value_heads, head_dim]`

在计算注意力时，K 和 V 会通过广播机制（Broadcasting）或者显式地进行重复（Repeat），使其在头的维度上扩展 $G$ 倍，从而匹配 Query 头的数量 `num_attention_heads`，然后才能进行矩阵乘法。
</details>

<details>
<summary><b>👉 展开查看：通过这三个参数区分 MHA, MQA, 和 GQA</b></summary>

这三种注意力机制的 `head_dim` 逻辑是一样的，唯一的区别就在于 `num_attention_heads` 和 `num_key_value_heads` 的比例关系：

| 机制类型 | 参数关系 | 说明 |
| :--- | :--- | :--- |
| **MHA** (Multi-Head) | `num_key_value_heads == num_attention_heads` | 标准 Transformer。每个 Q 头都有独立的 K、V 头。内存占用最大，精度最高。 |
| **MQA** (Multi-Query) | `num_key_value_heads == 1` | 所有的 Q 头共享**唯一的一个** K、V 头。内存占用极小，但可能会损失一些精度。 |
| **GQA** (Grouped-Query) | `1 < num_key_value_heads < num_attention_heads` | 两者的折中方案（例如 LLaMA 2/3）。将 Q 头分组，每组共享一个 K、V 头。在保证接近 MHA 精度的同时，大幅降低 KV Cache 的显存占用。 |

</details>

### 总结

*   **`num_attention_heads`** 决定了模型能从多少个不同的表示子空间关注信息。
*   **`num_key_value_heads`** 决定了在推理时（尤其是生成阶段）**KV Cache 的大小**。它越小，显存占用越低，推理速度越快。
*   **`head_dim`** 是点积计算的基石，确保了 Q、K、V 之间可以在数学维度上对齐，它不受前面两者比例变化的影响。

---

### 🧑‍💻 User

这些参数将会如何影响kv cache和attn计算复杂度？

---

### 🤖 Assistant

在大型语言模型（LLM）的推理过程中，特别是在自回归生成阶段，这三个参数对 **KV Cache 显存占用** 和 **计算复杂度 / 推理速度** 有着极其深远的影响。

GQA 被提出的核心动机，正是为了在“显存/速度”和“模型性能”之间找到最佳平衡点。以下是这三个参数的具体影响机制：

### 一、 对 KV Cache 显存占用的影响

在模型生成新 token 时，为了避免重复计算历史 token 的特征，会将历史的 Key 和 Value 张量缓存起来，这就是 KV Cache。

**KV Cache 完全由 `num_key_value_heads` 和 `head_dim` 决定，与 `num_attention_heads` 无关。**

单层、单条序列（Batch=1）、单个 token 的 KV Cache 容量计算公式为：
$$ \text{Cache Size} = 2 \times \text{num\_key\_value\_heads} \times \text{head\_dim} \times \text{bytes\_per\_param} $$
*(注：公式中的 2 代表 Key 和 Value 两个张量；`bytes_per_param` 在 FP16/BF16 下为 2 字节)*

*   **`num_key_value_heads` 的影响（决定性）：** 在标准的 MHA 中，它等于 `num_attention_heads`。而在 GQA 中，通过大幅减小 `num_key_value_heads`（例如 LLaMA-2 70B 中从 64 减小到 8），可以**成比例地缩小全局 KV Cache 的显存占用**。这使得模型能支持更大的 Batch Size 和更长的上下文长度。
*   **`head_dim` 的影响：** KV Cache 的大小与 `head_dim` 成正比。但在模型架构设计中，为了保持表达能力，`head_dim` 通常保持固定（常见的如 64, 128），不会轻易改动。
*   **`num_attention_heads` 的影响：** 对 KV Cache **无直接影响**。在 GQA 机制下，增加 Query 头的数量不会增加缓存负担。

---

### 二、 对 Attention 计算复杂度与推理速度的影响

为了准确说明，我们需要将 LLM 的推理分为两个阶段：**Prefill（预填充，处理 Prompt）** 和 **Decode（解码，逐字生成）**。这三个参数在不同阶段的影响是不同的。

#### 1. Prefill 阶段 (Compute-Bound 计算瓶颈)
在处理输入的 Prompt 时，计算量主要集中在密集的矩阵乘法（FLOPs）上。

*   **QKV 投影层复杂度：** 将输入隐藏层映射为 Q, K, V。
    复杂度为：$$ \mathcal{O}(N \cdot D \cdot (\text{num\_attention\_heads} + 2 \times \text{num\_key\_value\_heads}) \cdot \text{head\_dim}) $$
    *GQA 通过减少 K、V 头数，略微降低了这部分的计算量。*
*   **注意力分数矩阵计算 ($Q K^T$)：** 
    尽管物理上只有较少的 K 和 V 头，但在计算注意力时，K 和 V 会通过广播机制与所有的 Query 头对齐。因此，核心 Attention 算子的计算量（FLOPs）**依然由 `num_attention_heads` 决定**。
    计算复杂度约为：$$ \mathcal{O}(N^2 \cdot \text{num\_attention\_heads} \cdot \text{head\_dim}) $$
    *(其中 $N$ 为序列长度。可以看出，在这个阶段，GQA 并不能显著减少 Attention 核心机制的数学计算量)*

<details>
<summary><b>👉 展开查看：Decode 阶段 (Memory-Bound 内存带宽瓶颈) 的深度解析</b></summary>

#### 2. Decode 阶段 (Memory-Bound 内存带宽瓶颈)
这是生成回答的阶段，每次只处理 1 个新的 token ($N=1$)，需要频繁读取庞大的历史 KV Cache。此时，**内存带宽（把数据从显存搬运到计算单元的速度）**成为了绝对的瓶颈，而不是算力（FLOPs）。

*   **访存量 (Memory Traffic)：** 每次生成一个 token，模型都需要把完整的 KV Cache 读入芯片计算。
    读取的数据量正比于：$$ \text{Seq\_Len} \times \text{num\_key\_value\_heads} \times \text{head\_dim} $$
*   **GQA 的巨大优势：** 因为 `num_key_value_heads` 被大幅降低，模型每生成一个词所需读取的**显存数据量成比例下降**（例如降低为原来的 1/8）。虽然 CPU/GPU 还需要将这些少量的 KV 数据在内部“广播”给所有的 `num_attention_heads` 去计算，但这种片内计算极快。
*   **结论：** GQA 在 Decode 阶段通过减少显存数据搬运量，极大地缓解了 Memory-Bound 问题，从而**成倍地提升了推理速度 (Tokens per second)**。
</details>

---

### 总结对照表

下表总结了这三个参数增加时，对系统资源和复杂度的影响（保持其他参数不变的前提下）：

| 参数增加 | KV Cache 显存占用 | QKV 投影计算量 (FLOPs) | Attention 计算量 ($QK^T$) | Decode 访存带宽压力 (延迟) |
| :--- | :--- | :--- | :--- | :--- |
| **`num_key_value_heads` $\uparrow$** | **成比例增加** | 增加 | 不变 (广播前已对齐) | **成比例增加 (变慢)** |
| **`num_attention_heads` $\uparrow$** | 不变 | 增加 | **成比例增加** | 不变 (仅增加片内计算) |
| **`head_dim` $\uparrow$** | 成比例增加 | 成比例增加 | 成比例增加 | 成比例增加 |

**一句话核心思想：** GQA 保持了较大的 `num_attention_heads` 以维持模型的高性能（聪明的大脑），同时通过压缩 `num_key_value_heads` 大幅缩减了 KV Cache 的大小（精简的记忆库），从而在破解显存和内存带宽瓶颈的同时，实现了推理的飞速加速。