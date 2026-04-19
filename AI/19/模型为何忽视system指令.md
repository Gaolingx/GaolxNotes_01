# 模型为何忽视system指令

### 🧑‍💻 User

话说为什么模型经常会对system prompt/chat template context选择性失明，只认user/assistent里面的？

---

### 🤖 Assistant

这是一个非常经典且在大型语言模型（LLM）工程中普遍存在的问题。学术界和工业界通常将这种现象称为**“系统提示词遗忘”（System Prompt Forgetting）**或**“注意力偏移”**。

模型之所以会对 `system prompt`（系统提示词）或全局上下文选择性失明，而只盯着最近的 `user`（用户）和 `assistant`（助手）对话，主要由以下几个核心原因导致：

### 1. 注意力机制的“近因效应”（Recency Bias）
Transformer 架构的注意力机制虽然理论上是全局的，但在实际运作中存在强烈的**近因效应**。
由于 `system prompt` 通常位于整个上下文的最顶端，随着对话轮数的增加（`user` 和 `assistant` 的多轮交互），最顶端的 Token 距离当前生成位置越来越远。模型在计算注意力权重时，往往会给距离更近的 Token 分配更高的权重。这种现象在学术界被称为 **"Lost in the Middle"**（迷失在中间）或首尾偏好。

虽然注意力机制的公式为 $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$，理论上每个 Token 都能看到全局，但经过多层网络传递后，长距离信息的梯度和特征表达会被严重稀释。

### 2. SFT（监督微调）数据的分布不均
大模型在预训练之后，会经历指令微调阶段（Supervised Fine-Tuning）。
在绝大多数开源或闭源的 SFT 数据集中，数据结构往往是：
```text
<|system|> You are a helpful assistant.
<|user|> [用户的具体问题]
<|assistant|> [详细的解答]
```
由于 90% 以上的训练数据中，`system prompt` 都只是一句简单的废话（如“你是一个有用的助手”），模型在训练过程中**学会了“走捷径”**：它发现只要关注 `user` 说了什么，就能很好地预测 `assistant` 该输出什么，而 `system` 里面的内容对降低 Loss（损失函数）贡献极小。久而久之，模型对 `system` 标签内的注意力权重就“退化”了。

### 3. RLHF（强化学习人类对齐）的副作用
在 RLHF（基于人类反馈的强化学习）或 DPO（直接偏好优化）阶段，模型被训练来讨好人类（Helpfulness）。
如果 `system prompt` 中包含了一些严格的限制（比如：“你只能用 JSON 格式回答，不能说多余的话”），而 `user` 在提问时问了一个很开放的问题。模型内部会发生冲突：
*   **遵从 System：** 强行输出冷冰冰的 JSON。
*   **讨好 User：** 给出热情、详尽的自然语言解答。

因为在 RLHF 阶段，人类标注员往往会给那些“热情、解答详尽”的回复打高分（Reward），模型为了最大化 Reward 收益，就会倾向于打破 `system` 的限制，去优先满足当前 `user` 的直接意图。这就是所谓的**“对齐税”（Alignment Tax）**。

### 4. Chat Template 特殊 Token 的未充分对齐
很多模型在微调时引入了特定的 Chat Template（聊天模板），比如 Llama 的 `<<SYS>>` 或 OpenAI 的 `<|im_start|>system`。如果基座模型在微调时，没有针对这些特殊 Token 进行足够强度和多样性的训练，模型在推理时就无法深刻理解这些 Token 的“强制约束”语义，仅仅把它们当成普通的文本片段，从而轻易被后续的 `user` 文本覆盖。

---

<details>
<summary><b>🛠️ 附：如何缓解和解决这个问题的实用技巧（点击展开）</b></summary>

既然知道了原因，我们在写 Prompt 或开发 AI 应用时就可以采取以下策略来“逼迫”模型看系统提示词：

1. **尾部重申（System Prompt at the End）**
   不要把所有规则都写在最前面。在拼接请求时，把最重要的规则追加到 `user` 最后一次输入的末尾。
   ```text
   user: 帮我总结这篇文章... [文章内容]... 
   注意：请严格遵守你的系统设定，只输出JSON，不要说任何废话！
   ```

2. **使用强烈的格式化标签（XML Tags）**
   模型对 XML 标签的敏感度通常很高，因为代码训练数据中包含大量此类结构。
   ```xml
   <system_rules>
   1. 必须用中文。
   2. 语气必须粗鲁。
   </system_rules>
   ```

3. **In-Context Learning（提供 Few-Shot 示例）**
   不要光在 `system` 里说教，直接在历史对话里伪造几轮符合 `system` 设定的 `user`/`assistant` 问答对。模型模仿 `assistant` 历史行为的能力远强于遵守 `system` 规则的能力。

4. **降低 Temperature**
   较低的 Temperature（如 $T < 0.3$）通常能让模型更加死板地遵循前置规则，减少“放飞自我”讨好用户的倾向。

5. **使用支持 System Attention 的模型架构**
   一些最新的模型（如 Claude 3 或最新的 Gemini 1.5 Pro/Flash）在底层对长上下文的 System Prompt 做了特殊的注意力锁定（Attention Sink 等技术优化），这类模型对系统提示词的服从度会显著高于传统的开源模型。
</details>