# Self-Attention公式及参数量解析

### 🧑‍💻 User

Self-attention的公式及参数量？为什么用多头？为什么要除以根号d？

---

### 🤖 Assistant

### Self-Attention 公式及参数量

**Self-Attention 公式**  
Self-Attention 的核心公式为：  
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$  
其中：  
- $ Q, K, V $：通过可学习参数矩阵 $ W_Q, W_K, W_V $ 对输入 $ X $ 进行线性变换得到：  
  $$
  Q = XW_Q, \quad K = XW_K, \quad V = XW_V
  $$
- $ d_k $：Key 的维度（通常与Query相同），分母 $ \sqrt{d_k} $ 用于缩放点积，防止内积过大。

**参数量计算**  
1. **线性变换参数**：  
   $ W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k} $，假设输入维度为 $ d_{\text{model}} $，每个矩阵的参数量为 $ 3 \times d_{\text{model}} \times d_k $。  
   但实际中，这三个矩阵的维度可能不同，例如在多头注意力中，每个头的 $ d_k $ 可能调整。

2. **多头注意力（Multi-head）参数**：  
   假设有 $ h $ 个头，每个头的维度为 $ d_k = d_{\text{model}} / h $。此时：  
   - 每个头的参数矩阵为 $ W_{Q_i}, W_{K_i}, W_{V_i} \in \mathbb{R}^{d_{\text{model}} \times d_k} $，每个参数量为 $ 3 \times d_{\text{model}} \times d_k $。  
   - **总参数量**为：  
     $$
     \text{总参数量} = 3 \times h \times (d_{\text{model}} \times d_k) + d_{\text{model}} \times d_{\text{model}} 
     $$  
     其中，最后一个全连接层将拼接后的头输出（维度 $ d_{\text{model}} $）映射到最终输出空间。

---

### 多头注意力（Multi-head Attention）的作用

1. **捕捉不同子空间特征**：  
   多头注意力通过多个头并行学习不同子空间的表示，增强模型对位置或语义关系的表达能力。例如：  
   - 不同头可能关注同一序列中的不同位置依赖（如句法结构、语义角色等）。  
   - 多头的并行计算允许模型同时关注局部与全局依赖。

2. **增强模型鲁棒性**：  
   多头设计使得模型在训练时通过不同头的组合，减少对单一表示的过拟合，提升模型的鲁棒性。

3. **参数效率**：  
   每个头的维度 $ d_k $ 通常设计为 $ d_{\text{model}} / h $，保持总参数量与单头模型基本一致，但通过多头间的交互提升表达能力。

---

### 除以 $ \sqrt{d_k} $ 的作用

**问题背景**：  
在点积注意力中，Query 和 Key 的相似度计算公式为 $ QK^T $，其数值会随维度 $ d_k $ 的增大而增大，导致 softmax 梯度消失。

**数学解释**：  
- 假设 Query 和 Key 的元素 $ Q_i, K_j \sim \mathcal{N}(0, 1) $，则它们的点积 $ Q^T K $ 的方差为 $ d_k $。  
- 为保持点积的方差稳定（避免梯度爆炸或消失），需对点积结果进行缩放，即：  
  $$
  \text{Scaled Dot-Product} = \frac{QK^T}{\sqrt{d_k}}
  $$

**直观理解**：  
- **缩放意义**：将点积的均值保持在合理范围内（如方差为 1），避免 softmax 函数进入梯度饱和区域。  
- **类比层归一化**：与批归一化（BatchNorm）或层归一化（LayerNorm）类似，缩放操作是动态调整特征分布的简单方法。

---

### 总结

| 问题                | 回答概要                                                                 |
|---------------------|--------------------------------------------------------------------------|
| **Self-Attention公式** | $ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $ |
| **参数量**           | 单层线性变换参数量为 $ 3 \times d_{\text{model}} \times d_k $，多头时需乘以头数 $ h $。 |
| **多头作用**         | 通过多组参数捕捉不同特征子空间，增强模型表达能力。                                      |
| **除以 $ \sqrt{d_k} $ 原因** | 防止点积过大导致 softmax 梯度消失，保持方差稳定。                               |

**代码实现示例**：  
```python
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        # Q: [batch_size, h, seq_len, d_k]
        # K: [batch_size, h, seq_len, d_k]
        # V: [batch_size, h, seq_len, d_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.d_k**0.5  # [batch, h, seq_len, seq_len]
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # [batch, h, seq_len, d_k]
        return output
```

通过上述分析，可以清晰理解 Self-Attention 的数学本质、多头设计的优势及其缩放技巧的原理。