# BeamSearch面试考点

### 🧑‍💻 User

python实现beam search在算法岗面试中有哪些常见考点？

---

### 🤖 Assistant

在算法岗（尤其是NLP、大模型、语音识别等方向）的面试中，**Beam Search（集束搜索）** 是一个非常高频的考点。它不仅考察候选人对生成式模型解码过程的理解，还能很好地测试数据结构和代码实现能力。

以下是 Python 实现 Beam Search 时的常见考点和注意事项：

### 1. 核心数据结构的选取 (Priority Queue)
面试官会重点关注你如何维护当前的 Top-K 个候选序列。
*   **考点**：不能简单地对所有候选者排序然后切片，这会导致时间复杂度过高。
*   **正解**：应该使用**最小堆（Min-Heap）**。在 Python 中通常使用 `heapq` 模块。为了保留得分最高的 $K$ 个序列，我们维护一个大小为 $K$ 的最小堆。每次加入新元素后，如果堆的大小超过 $K$，就弹出堆顶（得分最小的）元素。

### 2. 概率乘积与下溢出 (Log Probability)
*   **考点**：序列的联合概率是每一步条件概率的乘积：$P(Y|X) = \prod_{i=1}^{T} P(y_i | y_{<i}, X)$。如果直接相乘，随着序列变长，浮点数会迅速逼近 0，导致**数值下溢（Underflow）**。
*   **正解**：必须在代码中将概率转换为对数概率（Log Probability），将连乘转化为连加：
    $$\log P(Y|X) = \sum_{i=1}^{T} \log P(y_i | y_{<i}, X)$$
    在堆中比较时，也是根据累加的 Log 概率进行排序（注意 Log 概率是负数，越接近 0 分数越高）。

### 3. 长度惩罚 (Length Penalty / Normalization)
*   **考点**：由于 Log 概率是负数，序列越长，累加的负数越多，总分越低。如果不做处理，Beam Search 会严重偏好生成短句子。
*   **正解**：需要对最终得分进行长度归一化。经典的公式是除以序列长度的 $\alpha$ 次方（通常 $\alpha \in [0.6, 1.0]$）：
    $$Score = \frac{\sum_{i=1}^{T} \log P(y_i | y_{<i}, X)}{|T|^\alpha}$$
    面试时，能够主动写出或提到 Length Penalty 是一个很大的加分项。

### 4. 结束符 `<EOS>` 的处理机制
*   **考点**：不同候选序列生成 `<EOS>` (End of Sentence) 的时间点不同。当某个 Beam 遇到 `<EOS>` 时，不应该再继续往下扩展。
*   **正解**：需要维护两个集合/列表：一个是当前正在扩展的活跃 Beam（`active_hypotheses`），另一个是已经完成的 Beam（`completed_hypotheses`）。
    *   当某个序列生成 `<EOS>` 时，将其移入 `completed_hypotheses`。
    *   由于 `completed_hypotheses` 中的序列长度固定了，此时必须**立刻应用长度惩罚**，然后再存入结果集。
    *   终止条件通常是：`completed_hypotheses` 收集到了 $K$ 个结果，或者达到了最大生成长度 `max_len`。

---

### Python 面试手撕代码模板

面试中通常不需要你实现完整的模型前向传播，而是要求你写出解码的核心逻辑。假设面试官提供了一个 `get_topk_logprobs(sequence, k)` 函数来模拟模型的预测。

```python
import heapq

def beam_search(start_token, end_token, max_len, beam_width, alpha=0.7):
    # 假设这是一个模型预测函数，返回下一个token的 (log_prob, token_id) 的列表
    def get_topk_logprobs(seq, k):
        # Dummy implementation for illustration
        return [(-0.1, 1), (-0.5, 2), (-1.2, 3)][:k] 

    # 堆中存放的元组: (当前总得分, 当前序列的列表)
    # 注意: heapq是最小堆，为了保留最大得分，我们存入真实得分（负数），堆顶是当前K个中最差的
    active_hypotheses = [(0.0, [start_token])]
    completed_hypotheses = []

    for step in range(max_len):
        next_hypotheses = []
      
        for score, seq in active_hypotheses:
            # 获取当前序列的下一步预测
            topk_preds = get_topk_logprobs(seq, beam_width)
          
            for log_prob, next_token in topk_preds:
                new_score = score + log_prob
                new_seq = seq + [next_token]
              
                if next_token == end_token:
                    # 遇到结束符，进行长度惩罚后放入完成列表
                    length_penalty = (len(new_seq) - 1) ** alpha # 不算start_token
                    final_score = new_score / length_penalty
                    completed_hypotheses.append((final_score, new_seq))
                else:
                    # 未结束，放入下一步的候选池
                    next_hypotheses.append((new_score, new_seq))
      
        # 如果已经收集到足够多的完整句子，可以提前结束 (简化版逻辑)
        if len(completed_hypotheses) >= beam_width:
            break
          
        # 从所有的下一步候选中，保留得分最高的 beam_width 个
        # heapq.nlargest 返回最大的k个元素，时间复杂度 O(N log K)
        active_hypotheses = heapq.nlargest(beam_width, next_hypotheses, key=lambda x: x[0])
      
        # 如果没有活跃序列了（极端情况），直接退出
        if not active_hypotheses:
            break

    # 把达到最大长度但还没遇到EOS的序列也加到完成列表中
    for score, seq in active_hypotheses:
        length_penalty = (len(seq) - 1) ** alpha
        completed_hypotheses.append((score / length_penalty, seq))
      
    # 最终按照得分对所有完成的序列排序，返回最好的
    completed_hypotheses.sort(key=lambda x: x[0], reverse=True)
    return completed_hypotheses[:beam_width]
```

---

<details>
<summary><b>点击展开：进阶追问 (Advanced Follow-up Questions)</b></summary>

在写完基础代码后，面试官通常会进行追问，以考察你的工程经验和前沿视野：

1.  **如何实现 Batch Beam Search？**
    *   *答：* 实际框架（如 HuggingFace `transformers`）中为了利用 GPU 的并行计算能力，需要对多个样本同时进行 Beam Search。这需要维护一个形状为 `(batch_size, beam_width)` 的张量。当某些句子的某些 beam 完成时，需要使用 Mask 或特殊标志位来让它们不再参与后续计算。
2.  **Beam Search 与 Greedy Search、Sampling 的对比？**
    *   *答：* Greedy Search 是 Beam width = 1 的特例，容易陷入局部最优。Beam Search 能找到全局更好的解，但在对话或故事生成任务中，生成的文本容易显得“平淡”和“重复”。Sampling（如 Top-K, Top-p/Nucleus Sampling 加上 Temperature）能增加生成的多样性，更适合开放式文本生成。
3.  **什么是 Diverse Beam Search？**
    *   *答：* 标准的 Beam Search 中，最后的几个候选序列往往只有最后几个词不同，同质化严重。Diverse Beam Search 将 Beam 分成多个组（Groups），在组之间引入排斥惩罚（Penalty），强制不同组探索不同的前缀空间，从而增加生成结果的多样性。

</details>

<details>
<summary><b>点击展开：复杂度分析 (Complexity Analysis)</b></summary>

*   **时间复杂度**：
    假设词表大小为 $V$，Beam width 为 $K$，最大生成长度为 $T$。
    每次循环中，有 $K$ 个序列。模型预测下一个词（如果是朴素的 softmax 选取）需要 $O(V)$ 时间。提取 Top-K 需要 $O(V \log K)$。
    合并所有的后续节点有 $K \times K$ 个，使用 `nlargest` 提取前 $K$ 个需要 $O(K^2 \log K)$。
    因此，不考虑模型推理本身的时间，纯 Beam Search 逻辑的时间复杂度大约为 $O(T \cdot K \cdot V \log K)$。
*   **空间复杂度**：
    需要存储当前的候选序列。空间复杂度为 $O(K \cdot T)$。

</details>