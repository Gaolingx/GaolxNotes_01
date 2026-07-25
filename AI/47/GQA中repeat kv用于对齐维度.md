# GQA中repeat kv用于对齐维度

### 🧑‍💻 User

gqa模型中为什么会有个repeat kv的算子？

---

### 🤖 Assistant

在GQA（Grouped-Query Attention，分组查询注意力）模型中，存在 `repeat_kv` 算子的核心原因是为了**解决查询头和键值头数量不匹配的问题**，从而使得现有的标准注意力计算算子能够直接复用，而无需为GQA单独重写底层的矩阵乘法内核。

<details>
<summary><b>背景知识：MHA、MQA与GQA的区别</b></summary>

在理解 `repeat_kv` 之前，需要先了解三种注意力机制的演进：

*   **MHA (Multi-Head Attention)**: 标准的多头注意力。查询（Query）、键（Key）、值（Value）具有相同的头数 $h$。
*   **MQA (Multi-Query Attention)**: 极端情况。Query 有 $h$ 个头，而 Key 和 Value 只有 $1$ 个头。所有 Query 头共享同一组 KV。这大大减少了显存占用，但可能导致质量下降。
*   **GQA (Grouped-Query Attention)**: MHA 和 MQA 的折中方案。Query 有 $h$ 个头，而 Key 和 Value 有 $g$ 个头（$1 < g < h$）。每 $h/g$ 个 Query 头组成一个组，共享同一组 KV。

</details>

### 为什么需要 `repeat_kv`？

在 GQA 中，假设 Query 的头数为 $h$，KV 的头数为 $g$。标准的注意力计算公式为：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

在底层硬件实现（如 PyTorch 的 `torch.nn.functional.scaled_dot_product_attention` 或 Flash Attention）中，这些算子通常要求输入的 $Q$、$K$、$V$ 张量在“头”这个维度上的大小是严格一致的。

由于 GQA 的 $h \neq g$，如果直接将形状不同的 $Q$ 和 $K, V$ 送入标准算子，会导致维度不匹配的报错。为了解决这个问题，通常有两种方案：

1.  **重写底层算子**：编写专门支持不同头数对比的 CUDA 算子或 Triton 算子。
2.  **逻辑上扩展 KV 头**：使用 `repeat_kv` 算子，在逻辑上将 $g$ 个 KV 头复制扩展成 $h$ 个头，使其与 $Q$ 的头数一致，然后直接调用标准的注意力算子。

由于方案2不需要修改底层高度优化的算子，实现简单且兼容性极好，因此在很多开源实现（如 HuggingFace Transformers）中被广泛采用。

### 维度变化与数学表达

假设输入的批次大小为 $b$，序列长度为 $s$，隐藏层维度为 $d$。
*   Query 张量 $Q$ 的形状通常为 $(b, h, s_q, d_{head})$
*   Key/Value 张量 $K, V$ 的形状通常为 $(b, g, s_{kv}, d_{head})$

`repeat_kv` 算子的作用就是将 $K$ 和 $V$ 沿着头维度复制 $r = h / g$ 次，使其形状变为：

$$ K_{\text{repeated}}, V_{\text{repeated}} \in \mathbb{R}^{(b, h, s_{kv}, d_{head})} $$

这样，扩展后的 $K_{\text{repeated}}$ 和 $V_{\text{repeated}}$ 就可以直接与 $Q$ 进行标准的注意力计算了。

<details>
<summary><b>实现细节：expand 与 repeat 的内存优化</b></summary>

虽然该算子在逻辑上被称为 `repeat_kv`，但在实际的深度学习框架（如 PyTorch）实现中，为了性能考虑，**通常不会真正地分配新内存去复制数据**，而是利用张量的视图操作。

例如在 PyTorch 中，会使用 `.expand()` 而不是 `.repeat()`：

```python
import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    等同于将 KV 头复制 n_rep 次，但不占用额外显存
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 先增加一个维度，准备用于广播
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # 重新整形合并头维度
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

`.expand()` 返回的是原张量的一个视图，它不会在物理内存中复制数据，而是通过改变步长在计算时“欺骗”底层算子，让它以为这里有 $h$ 个头。只有当后续算子真正需要读取连续内存时（如进入特定的 CUDA Kernel），才会按需读取。这种设计让 GQA 在保持代码简洁的同时，几乎没有引入额外的显存和计算开销。

</details>

总结来说，GQA 模型中的 `repeat_kv` 算子是一个**工程适配层**，它通过广播或复制机制，将数量较少的 KV 头扩展到与 Query 头数量一致，从而无缝对接现有的、经过高度优化的标准多头注意力计算算子。