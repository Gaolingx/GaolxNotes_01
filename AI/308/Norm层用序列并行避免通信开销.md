# Norm层用序列并行避免通信开销

### 🧑‍💻 User

为什么transformer的norm层需要用sequence parallel而不是colwise/rowwise？

---

### 🤖 Assistant

在Transformer大模型分布式训练（尤其是Megatron-LM风格的张量并行 Tensor Parallelism, TP）中，Layer Normalization (Norm层) 采用**序列并行 (Sequence Parallelism, SP)** 而不是列/行级并行 (Colwise/Rowwise Parallelism)，核心原因在于**LayerNorm的数学计算特性**以及**通信与内存的优化需求**。

具体原因可以分为以下几个关键点：

### 1. LayerNorm 的计算维度：沿 $H$ 归一化，沿 $S$ 独立
在Transformer中，输入的张量形状通常为 $(S, B, H)$，其中 $S$ 是序列长度 (Sequence length)，$B$ 是批次大小 (Batch size)，$H$ 是隐藏层维度 (Hidden dimension)。

LayerNorm 的公式为：
$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

关键在于 $\mu$ (均值) 和 $\sigma$ (方差) 是如何计算的：**它们是在隐藏层维度 $H$ 上进行计算的。**
这意味着，对于序列中的**每一个Token，它的LayerNorm计算是完全独立的**，只依赖于它自身的 $H$ 个元素。

### 2. 为什么不能用 Colwise/Rowwise (切分 $H$ 维度)？
列/行级并行的本质是将权重和激活值在**隐藏层维度 $H$** 上进行切分。

如果我们在Norm层对 $H$ 维度进行切分（假设分给2个GPU）：
*   GPU 0 拥有 Token 的前一半特征 $H/2$。
*   GPU 1 拥有 Token 的后一半特征 $H/2$。

**灾难性的后果（引入额外通信）：**
为了计算均值 $\mu$ 和方差 $\sigma$，GPU 0 和 GPU 1 必须将各自局部的和（Sum）与平方和（Sum of Squares）进行全局通信。也就是说，**在LayerNorm计算的内部，必须插入一次 `All-Reduce` 操作**。
LayerNorm 在Transformer中频繁出现（每个Block至少两次），如果在Norm层内部引入 `All-Reduce`，会导致极高的通信延迟，严重拖慢训练速度。

### 3. 为什么选择 Sequence Parallelism (切分 $S$ 维度)？
序列并行（Sequence Parallelism）是指将激活值在**序列长度 $S$** 上进行切分。

如果在Norm层对 $S$ 维度进行切分（假设分给2个GPU）：
*   GPU 0 负责前 $S/2$ 个 Tokens，拥有这部分 Tokens 完整的 $H$ 维度。
*   GPU 1 负责后 $S/2$ 个 Tokens，拥有这部分 Tokens 完整的 $H$ 维度。

**巨大的优势（零通信）：**
由于每个 GPU 都拥有其负责 Token 的完整 $H$ 维度，它们可以**完全独立且精确地计算局部 Tokens 的 LayerNorm，不需要GPU之间进行任何通信**。

### 4. Megatron-LM v3 的极致内存优化 (Reduce-Scatter / All-Gather)
在引入 Sequence Parallelism 之前（即早期的 Megatron-LM 张量并行），Norm 层并没有被并行化，而是**在每个GPU上冗余计算**。
*   **传统TP的做法：** 线性层（Rowwise Linear）计算后，会执行一次 `All-Reduce`，使得所有 GPU 都拥有完整的 $(S, B, H)$ 张量。然后所有 GPU 分别对这份完整的数据进行 Dropout、Residual Add 和 LayerNorm。
*   **缺点：** 冗余计算，且在前向传播中保存了极其庞大的冗余激活值（Activation Memory），导致显存爆炸。

**Sequence Parallelism 的巧妙设计：**
NVIDIA 在论文 *Reducing Activation Recomputation in Large Transformer Models* 中引入了 SP 来解决这个问题：
1.  将 Rowwise Linear 后的 `All-Reduce` 拆分为 `Reduce-Scatter`。这样每个 GPU 只得到部分序列 $(S/p, B, H)$。
2.  在这部分序列上，**无通信地**执行 Dropout、Residual Add 和 LayerNorm（因为它们在序列维度上都是独立的）。
3.  在进入下一个需要完整 $H$ 维度的 Colwise Linear 层之前，执行一次 `All-Gather`，将序列拼接回完整的 $(S, B, H)$。

<details>
<summary><b>点击展开：通信成本对比（All-Reduce vs Reduce-Scatter + All-Gather）</b></summary>

从通信量上来看：
*   **传统TP：** 1次 `All-Reduce`。数学上，`All-Reduce` 的通信量等价于 1次 `Reduce-Scatter` + 1次 `All-Gather`。
*   **引入SP后：** 将 `All-Reduce` 拆解开，中间夹着 Norm 层和 Dropout 层。

**结论：** 引入序列并行后，总的通信带宽消耗并没有增加（甚至在网络拓扑好的情况下更容易打满带宽），**通信成本不变，但激活值的显存占用大幅度降低了 $\frac{1}{p}$（$p$为张量并行度）**。
</details>

### 总结
在 Norm 层使用 Sequence Parallelism 而不是 Colwise/Rowwise 的原因总结如下：

*   **数学一致性：** LayerNorm 沿 $H$ 维度归一化，切分 $H$ (Col/Row-wise) 会导致计算断裂，必须引入昂贵的通信才能算出全局均值和方差。
*   **零通信开销：** 切分 $S$ 维度 (Sequence Parallelism) 使得每个 Token 保留了完整的 $H$，Norm 层可以做到完全本地计算，0通信。
*   **显存节省：** 结合 `Reduce-Scatter` 和 `All-Gather`，SP 将 Norm、Dropout 等非线性层的激活值显存占用成倍降低，是训练超大模型（如 LLaMA, GPT-3 等）不可或缺的技术。