# RoPE在LLaMA2中的应用

### 🧑‍💻 User

以llama arch为例，rope位置编码是如何嵌入qk的？（结合https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py代码分析）

---

### 🤖 Assistant

结合 Hugging Face 的 `transformers` 库中 `modeling_llama.py` 的源码逻辑以及你提供的代码片段，Llama 架构中 RoPE（Rotary Positional Embeddings，旋转位置编码）嵌入 Q（Query）和 K（Key）的过程可以分为三个核心步骤：**初始化频率**、**计算旋转辅助变量**、**对 Q 和 K 进行旋转变换**。

以下是详细的代码逻辑分析：

### 1. 核心原理简述
RoPE 不像 BERT 那样将位置向量直接**加**在 Embedding 上，而是通过将 Query 和 Key 向量在复数域（或者看作两两一组）进行**旋转**，从而在做点积（Attention Score）时自然地引入相对位置信息。

数学公式核心为：
$$
\text{RoPE}(x, m) = x \cdot \cos(m\theta) + \text{rotate\_half}(x) \cdot \sin(m\theta)
$$
其中 $x$ 是 Q 或 K 向量，$m$ 是位置索引，$\theta$ 是频率。

---

### 2. 代码实现流程分析

基于 Hugging Face 的实现逻辑，整个过程发生在 `LlamaAttention` 模块中。

#### 第一步：初始化频率 (Init)
在模型初始化阶段（如你提供的 Snippet 1 所示），会根据配置选择不同的 RoPE 实现类（如 `LlamaRotaryEmbedding`）。

```python
# 对应 Snippet 1 的逻辑
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 计算逆频率 inv_freq
        # 公式: theta_i = 1 / (base ^ (2i / d))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
      
        # 预计算 cos 和 sin 缓存 (cos_cached, sin_cached)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
```
**关键点**：这里生成了一个 `inv_freq` 向量，决定了旋转的角度粒度。如果使用了 `rope_scaling`（线性或动态NTK），会在这里调整频率的计算方式，但后续应用逻辑不变。

#### 第二步：准备旋转矩阵 (Prepare Cos/Sin)
在 `LlamaAttention.forward` 过程中，Q 和 K 经过线性层投影并 reshape 成 `[batch_size, num_heads, seq_len, head_dim]` 后，需要获取当前序列对应的 `cos` 和 `sin` 值。

```python
# 在 LlamaAttention.forward 中
def forward(self, hidden_states, ...):
    # 1. 线性投影得到 Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
  
    # ... Reshape & Transpose 变成 [bsz, heads, seq_len, head_dim] ...

    # 2. 获取当前序列长度对应的 cos 和 sin
    # rot_emb 实际上调用 LlamaRotaryEmbedding.forward
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
```

#### 第三步：执行旋转变换 (Apply RoPE)
这是问题的核心，RoPE 是如何“嵌入”进去的。HF 定义了一个辅助函数 `rotate_half` 和主函数 `apply_rotary_pos_emb`。

**辅助函数：rotate_half**
将向量的一半取负并交换位置，模拟复数旋转中的 $i(x+iy) = -y+ix$ 操作。
```python
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 假设 x 的最后维度是 head_dim
    x1 = x[..., : x.shape[-1] // 2] # 前半部分
    x2 = x[..., x.shape[-1] // 2 :] # 后半部分
    # 结果拼接：[-x2, x1]
    return torch.cat((-x2, x1), dim=-1)
```

**主函数：apply_rotary_pos_emb**
这里将 Q 和 K 与位置编码融合。
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    q, k: [batch_size, num_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim] (预计算好的)
    position_ids: [batch_size, seq_len] (每个token的位置索引)
    """
  
    # 1. 根据 position_ids 提取对应的 cos 和 sin
    #    结果形状变为 [batch_size, 1, seq_len, head_dim] 以便广播
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
  
    # 2. 执行旋转公式
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
  
    return q_embed, k_embed
```

---

### 3. 总结：嵌入 QK 的具体动作

在 Hugging Face 的 Llama 实现中，RoPE 的嵌入发生在**Attention 核心计算（Dot Product）之前**：

1.  **输入**：投影后的 `query_states` 和 `key_states`。
2.  **变换**：
    *   保留原向量 `q` 乘以 $\cos$。
    *   将原向量 `q` 切分、翻转、部分取负得到 `rotate_half(q)`，然后乘以 $\sin$。
    *   将上述两部分相加。
3.  **公式化表达**：
    $$ Q_{\text{rotated}} = Q \otimes \cos + \text{Rotate}(Q) \otimes \sin $$
    $$ K_{\text{rotated}} = K \otimes \cos + \text{Rotate}(K) \otimes \sin $$
4.  **后续**：使用旋转后的 $Q_{\text{rotated}}$ 和 $K_{\text{rotated}}$ 进行矩阵乘法计算注意力分数：
    $$ \text{AttnScore} = \text{Softmax}\left(\frac{Q_{\text{rotated}} \cdot K_{\text{rotated}}^T}{\sqrt{d}}\right) $$

通过这种方式，位置信息被注入到了 Q 和 K 的向量方向中，使得注意力机制能够通过向量夹角感知 token 之间的相对距离。