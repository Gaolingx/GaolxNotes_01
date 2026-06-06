# QKNorm与QKClip区别

### 🧑‍💻 User

llm训练中，我们发现随着训练进行，attention logits会不断增加可能导致训练不稳定，常见的解决方法qk-clip和qk-norm有什么区别？

---

### 🤖 Assistant

在LLM（大语言模型）的预训练过程中，随着训练的进行，模型为了让Attention权重更加“尖锐”（即更加确定地关注某些特定的token），会倾向于不断增大Query（Q）和Key（K）的向量模长。这会导致 Attention logits ($QK^T$) 的数值变得非常大。

当 logits 过大时，经过 Softmax 之后会产生两个严重问题：
1. **梯度消失（Softmax Saturation）：** Softmax的输出接近one-hot向量，导致梯度极小，模型无法继续有效更新。
2. **数值溢出：** 在FP16或BF16精度下，过大的数值会导致NaN，直接引起训练崩溃（Loss Spike）。

为了解决这个问题，业界最常用的两种方案是 **QK-Norm** 和 **QK-Clip (更准确地说是 Logit Soft-Capping)**。它们的本质区别在于**约束发生的位置**和**数学机制**。

---

### 1. QK-Norm (Query-Key Normalization)

QK-Norm 是在 Q 和 K 进行点积（Dot Product）**之前**，对它们分别应用归一化（通常是 LayerNorm 或 RMSNorm）。

*   **计算公式：**
    $$Q' = \text{Norm}(Q), \quad K' = \text{Norm}(K)$$
    $$\text{Logits} = \frac{Q' (K')^T}{\sqrt{d_k}}$$
*   **作用机制：** 它直接限制了 Q 和 K 向量的模长（使其具有固定的均值和方差）。因为归一化后的向量模长受到约束，所以它们的点积（本质上是计算余弦相似度）的上限也被严格限制住了。
*   **优点：** 
    * 从根本上控制了特征空间的爆炸，使得模型在整个深度上保持良好的条件数（Condition Number）。
    * 在缓解训练不稳定的同时，还能在一定程度上提升模型在长上下文（Long-context）下的外推能力。
*   **缺点：** 增加了额外的计算开销（每个Attention层多了两次Norm操作），且稍微改变了标准Attention的表示能力（强迫Q和K分布在超球面上）。

### 2. QK-Clip / Logit Soft-Capping

QK-Clip（现在业界更流行使用基于 $\tanh$ 的 Soft-Capping）是在 Q 和 K 进行点积**之后**、Softmax**之前**，直接对生成的 Logits 进行数值截断或压缩。

*   **计算公式：**
    通常不使用生硬的 `clamp(x, min, max)`，而是使用平滑的 $\tanh$ 函数（例如在 PaLM, Gemma, Grok 等模型中）：
    $$\text{Logits} = C \cdot \tanh\left(\frac{Q K^T}{C \cdot \sqrt{d_k}}\right)$$
    *(其中 $C$ 是一个超参数，通常设为 30 或 50，即 Logits 的最大值被软性限制在 $[-C, C]$ 之间)*
*   **作用机制：** 模型底层生成的 Q 和 K 向量模长依然可以很大，但是计算出的标量 Logits 经过 $\tanh$ 函数后，被强制压缩到了一个安全的范围内。
*   **优点：** 
    * 计算开销极小（仅仅是一个缩放和激活函数）。
    * 非常有效地防止了极值导致的 Softmax 饱和和 FP16/BF16 溢出。
*   **缺点：** 治标不治本。如果模型强烈倾向于生成极大的 logits，它会被推到 $\tanh$ 的平缓区域，依然会导致针对 Q 和 K 的梯度变小（尽管比直接 Softmax 饱和要好）。

---

### 核心区别对比

| 对比维度 | QK-Norm | QK-Clip (Soft-Capping) |
| :--- | :--- | :--- |
| **干预位置** | 点积**前** (作用于 Q/K 向量) | 点积**后** (作用于标量 Logit) |
| **数学本质** | 限制向量的范数 (Norm)，将点积转化为带缩放的余弦相似度计算 | 限制标量输出的天花板 (Value bounding) |
| **计算开销** | 较高 (增加两个 RMSNorm/LayerNorm) | 极低 (一次乘法和一次 tanh) |
| **梯度流动** | 梯度在 Norm 处重新分配，稳定且平缓 | 梯度在接近上限（如 $C$）时会被 $\tanh$ 衰减 |
| **对表征的影响** | 强制 Q, K 空间呈现超球面几何特性 | 保持原始向量空间，仅改变最终注意力分数的分布 |
| **代表模型** | ViT-22B, 某些内部大规模模型 | PaLM 1/2, Gemma 1/2, Grok-1 |

<details>
<summary><b>展开查看：训练实践中的选择建议</b></summary>

在实际训练中，如何选择这两种方案？

1. **当前主流趋势：推荐使用 Soft-Capping ($\tanh$)**。
   目前 Google 的大模型体系（PaLM, Gemma）几乎全面采用了 Logit Soft-Capping。原因在于它几乎不增加显存和计算负担，且修改代码极为简单（仅需在Attention中加入一行代码），就能完美规避由 logits 爆炸引起的 Loss Spike。推荐的 $C$ 值通常在 $30.0$ 到 $50.0$ 之间。

2. **何时使用 QK-Norm？**
   如果你的模型不仅遇到了 Attention logits 爆炸，还遇到了严重的特征范数爆炸（Hidden states norm explosion），或者你在训练极深的 Transformer 模型，QK-Norm 是更好的选择。它有助于约束整个网络的信息流规模。一些旨在极致稳定性的架构会选择此方案。

3. **混合使用：**
   两者并不冲突。在某些极端规模的模型中，可以同时使用 RMSNorm 约束 QK，再用 $\tanh$ 兜底，但通常没有必要，选择其中一种就能解决 99% 的 Attention 相关的数值不稳定问题。
</details>