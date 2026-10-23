# 主流trans模型RMSNorm位置与归属解析

## 🧑‍💻 User

主流trans rmsnorm在ffn之前吧，rmsnorm不是ffn的一部分来着

---
## 😀 默认助手 (7)

你这个说法**对了一半**——它把"norm 放在哪"和"norm 属于哪个模块"这两件事混在一起了。分开看就清楚了。

## 一、位置上：不只 FFN 前有

现在的主流 LLM（Llama 系）是 **pre-norm**，也就是**每个子层前面各挂一个 norm**，一个层里有两个：

```python
# LlamaDecoderLayer.forward (HF transformers)
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)   # ← attention 前的 norm
hidden_states, _ = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)  # ← FFN 前的 norm
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

所以：

- "RMSNorm 在 FFN 之前" ✅ 成立（`post_attention_layernorm`）。
- 但如果理解成"FFN 前面才有 norm" ❌，attention 前面同样有一个，另外 `LlamaModel` 最后一层之后还有一个 `self.norm`（final norm）。**Llama 每层 2 个 RMSNorm，加末尾 1 个。**

## 二、归属上：代码里确实不在 FFN 内部

`LlamaMLP` 里只有三个 `Linear`，没有 norm：

```python
class LlamaMLP(nn.Module):
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

norm 是 `LlamaDecoderLayer` 的**兄弟模块**（`input_layernorm` / `post_attention_layernorm`），不属于 MLP。这一点你说得对。

不过要小心两点：

- **画图 ≠ 代码结构。** 画架构图时通常把 `Norm → FFN` 当成一个 "FFN block"，所以口语里说"FFN 的输入 norm"很常见。
- **有些框架按计算视角把它划进 FFN。** 比如张量并行切分时，某些实现把 pre-FFN norm 和 MLP 一起放进一个 `FeedForward` 模块；`FeedForward(norm=...)` 这种签名在训练 repo 里很常见。

## 三、性能视角：它反而经常"变成" FFN 的一部分

$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot w $$

关键点：RMSNorm **对每个 token 只是逐元素缩放**（分母是整个 hidden 向量的标量 RMS）。所以

- 量化 / 推理优化里，norm 的 $w$ 和缩放常常被**折叠进 `gate_proj`/`up_proj` 的权重**，模块图上就消失了；
- 融合 kernel 里也会把 RMSNorm + FFN 第一个 GEMM 合成一个 op。

这时候从计算图看，RMSNorm 就是"FFN 的一部分"了。

## 四、别把它当铁律：反例不少

- **Post-norm**（原始 Transformer）：norm 在**残差相加之后**，那才真的是"FFN 之后"的 norm。
- **Sandwich norm**（Gemma 2）：一层有 **4 个** norm —— `input_layernorm`、`post_attention_layernorm`、`pre_feedforward_layernorm`、`post_feedforward_layernorm`，FFN **前后各一个**。而且它那个 `post_attention_layernorm` 是在加残差**之前**用的（`hidden_states = residual + norm(attn(x))`），和传统 post-norm 又不一样。

## 结论

- **位置**上：主流 pre-norm 是"每个子层前各一个 norm"，所以"FFN 前有 RMSNorm"对，但别忘了 attention 前也有、末尾还有一个。
- **归属**上：`nn.Module` 结构里它是 decoder layer 的兄弟，不是 `FFN` 的孩子；但在架构图、部分框架的 block 定义、以及量化/融合 kernel 里，它经常被当作 FFN 的一部分。

所以更准确的表述是：**"pre-FFN norm" 是 FFN 子层的输入归一化，模块上属于 layer、不属于 MLP。**
