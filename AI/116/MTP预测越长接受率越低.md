# MTP预测越长接受率越低

### 🧑‍💻 User

为什么deepseek的mtp预测的token越多（num_speculative_tokens），接受率越低？
```
(APIServer pid=1) INFO 02-28 04:23:51 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 3.97, Accepted throughput: 0.50 tokens/s, Drafted throughput: 1.35 tokens/s, Accepted: 95 tokens, Drafted: 256 tokens, Per-position acceptance rate: 0.750, 0.594, 0.375, 0.344, 0.312, 0.219, 0.188, 0.188, Avg Draft acceptance rate: 37.1%
```

---

### 🤖 Assistant

在DeepSeek的MTP（Multi-Token Prediction）以及所有的投机解码（Speculative Decoding）机制中，**预测的Token越靠后，接受率（Acceptance Rate）越低，这是一个不可避免的数学与架构规律。**

从你提供的日志中可以完美地观察到这一现象：各个位置的接受率 `Per-position acceptance rate: 0.750, 0.594, 0.375, 0.344, 0.312, 0.219, 0.188, 0.188` 呈现严格的单调递减趋势。

造成这个现象的核心原因可以归结为以下几个方面：

### 1. 严格的前缀匹配原则（最核心原因）
投机解码的验证机制是**串行依赖**的。假设草稿模型（或MTP模块）一次性预测了 $k$ 个Token：$t_1, t_2, \dots, t_k$。
主模型在并行验证时，**只有当第 $i-1$ 个Token被接受时，第 $i$ 个Token才有资格被接受。** 只要中间有任何一个Token被拒绝，后面所有的Token无论好坏，都会立刻被丢弃（因为上文已经被改变了）。

从数学概率的角度来看，假设主模型接受每一个单独Token的独立概率为 $p$（例如 $p = 0.8$），那么第 $k$ 个位置被接受的联合概率是指数级衰减的：
$$ P(\text{Accept}_k) = P(\text{Accept}_1) \times P(\text{Accept}_2 | \text{Accept}_1) \dots \times P(\text{Accept}_k | \text{Accept}_{1 \dots k-1}) $$
即使每个位置的独立命中率都很高，连乘之后也会变得很低。例如：$0.8^5 \approx 0.327$。

---

### 深层原因解析

为了方便阅读，我将更深层的理论原因整理在以下折叠面板中：

<details>
<summary><b>🔍 展开查看：误差累积与自回归偏移 (Autoregressive Drift)</b></summary>
MTP在预测未来的第 $k$ 个Token时，是基于它自己预测的前 $k-1$ 个Token的上下文。
随着 `num_speculative_tokens` 的增加，草稿（Draft）上下文与主模型真实倾向的上下文之间的“语义发散”会越来越严重。一旦前几个Token的预测出现了微小的词义偏差（哪怕这个Token被接受了），基于这个偏差生成的后续Token，就更难满足主模型严格的概率分布要求。这就是所谓的**误差累积**。
</details>

<details>
<summary><b>🔍 展开查看：自然语言的信息熵 (Entropy) 规律</b></summary>
在语言生成中，**下一个词（Next-token）往往受制于强烈的语法或局部语义约束，因此信息熵较低，容易预测。** 
例如：“The quick brown” 后面大概率是 “fox”。
但是，预测第5个或第8个词时，涉及到模型未决定的长程语义选择，信息熵极高。草稿模型很难在不进行完整的深层计算的情况下，准确猜出主模型在几个词之后想要表达的核心观点。
</details>

<details>
<summary><b>🔍 展开查看：DeepSeek MTP 架构的能力瓶颈</b></summary>
DeepSeek-V3/R1 的 MTP（多Token预测）机制并非使用一个完全独立的草稿大模型，而是复用了主模型的隐状态（Hidden States），外挂了几个轻量级的预测头（Prediction Heads）。
这些轻量级的预测头参数量远小于主干网络。它们具备预测短期（1-3个）Token的泛化能力，但如果要预测极远的未来（比如第8个Token），轻量级网络无法进行足够深度的特征变换，表达能力达到瓶颈，导致其输出的概率分布与主模型的概率分布 $P_{\text{target}}(x)$ 产生巨大差异，从而触发拒绝采样（Rejection Sampling）的拒绝条件。
</details>

---

### 结合你的日志分析

```text
Per-position acceptance rate: 0.750, 0.594, 0.375, 0.344, 0.312, 0.219, 0.188, 0.188
```

*   **第 1 个 Token (`0.750`)**：接受率高达 75%。因为它是紧接着真实上下文预测的，MTP模块很容易猜对。
*   **第 2 个 Token (`0.594`)**：这不仅要求第2个Token本身的概率分布合理，还要求前提是第1个Token必须是对的。
*   **第 8 个 Token (`0.188`)**：要到达这里，前7个Token必须**全部命中**。在这个长距离上，接受率已经跌到了 18.8%。

### 调优建议 (Practical Takeaway)

这也是为什么在 vLLM 或 SGLang 等推理框架中，配置 `num_speculative_tokens`（草稿长度）并不是越大越好。

1.  **开销与收益的博弈**：虽然预测 8 个 Token（`Accepted: 95, Drafted: 256`）可以增加理想情况下的吞吐量，但生成那些必定会被拒绝的末尾 Token（如第6、7、8个）实际上浪费了显存带宽和计算资源。
2.  **寻找最优解**：通过你的 `Mean acceptance length: 3.97` 可以看出，平均每次投机能接受大约 4 个 Token。因此，将 `num_speculative_tokens` 设置为 **4 到 5** 之间，通常能达到计算效率（Drafted throughput）和真实收益（Accepted throughput）的最佳平衡。盲目增大该值反而可能导致系统整体的 `tokens/s` 下降。