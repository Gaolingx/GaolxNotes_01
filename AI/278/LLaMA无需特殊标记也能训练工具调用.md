# LLaMA无需特殊标记也能训练工具调用

### 🧑‍💻 User

对于llama这种没有把tool call注册为special token的模型也可以正常训练sft工具调用吗？

---

### 🤖 Assistant

**答案是：完全可以。**

对于像 LLaMA 这样没有原生将 Tool Call（工具调用）注册为特殊 Token（Special Token）的模型，**不仅可以正常进行 SFT（监督微调）训练，而且这也是目前开源社区中最主流、最成熟的做法之一。**

这种方法的核心思想是：**把工具调用视为一种“特定格式的文本生成任务”**。模型不需要认识特殊的 Token，只需要学会在合适的时机输出特定格式（如 JSON、XML 或自定义特殊字符串）的文本即可。

以下是具体的实现原理、训练方法以及优缺点：

### 1. 实现原理：依赖 Prompt 和结构化输出

既然没有 `<tool_call>` 这样的专门 Token，我们可以通过在 System Prompt 中定义规则，让模型使用标准的文本来表示工具调用。

常见的文本格式有三种：
*   **JSON 格式（最推荐）**：让模型输出一段带有特定标记的 JSON。
*   **XML / HTML 标签格式**：使用 `<tool_action>` 和 `<tool_args>` 等普通文本标签包裹内容。
*   **ReAct 格式**：使用 `Thought:`（思考）、`Action:`（动作）、`Action Input:`（输入）这种纯文本标识符。

### 2. SFT 训练数据如何构造？

在 SFT 阶段，你需要将对话数据构造成包含“工具描述”、“工具调用”和“工具返回结果”的纯文本格式。

<details>
<summary><b>点击展开查看：SFT 训练数据格式示例（基于 JSON）</b></summary>

在你的 SFT 数据集中，一次完整的工具调用对话通常会被格式化为类似下面的多轮文本结构：

**System Prompt (系统提示词)**
```text
你是一个可以调用外部工具的助手。
当前可用的工具如下：
[
  {"name": "get_weather", "description": "获取指定城市的天气", "parameters": {"city": "string"}}
]
如果你需要调用工具，请严格输出以下 JSON 格式：
```json
{"tool_name": "...", "tool_args": {...}}
```
```

**User (用户输入)**
```text
北京今天天气怎么样？
```

**Assistant (模型输出 - SFT的Target)**
```text
好的，我需要查询北京的天气。
```json
{"tool_name": "get_weather", "tool_args": {"city": "北京"}}
```
```

**User (代表工具返回的结果 - 通常可以定义一个特殊的 role 叫 tool 或 function，但在纯文本模型中也可以作为 User 输入)**
```text
Tool Observation: {"temperature": "25℃", "condition": "晴朗"}
```

**Assistant (模型最终输出 - SFT的Target)**
```text
北京今天天气晴朗，温度是25℃。
```

在 SFT 训练时，你只需要让模型学习预测上述结构中的 **Assistant** 部分即可。模型通过学习大量的这种语料，就会知道：“哦，当我看到需要天气的请求时，我应该输出一段特定格式的 JSON 代码块”。
</details>

### 3. 没有 Special Token 的优缺点

**优点：**
1.  **无需修改 Tokenizer 和模型结构：** 训练门槛极低，你可以直接拿原生的 LLaMA 权重使用普通的 SFT 框架（如 LLaMA-Factory、FastChat 等）进行训练。
2.  **通用性极强：** 这种纯文本的 Prompt 协议可以跨模型使用。比如 LangChain 和 LlamaIndex 等框架，默认就是通过解析 JSON 或 XML 文本来实现 Tool Calling 的。
3.  **便于调试：** 生成的过程全是人类可读的纯文本，非常容易排查是模型没理解工具，还是 JSON 格式生成错了。

**缺点：**
1.  **Token 消耗略大：** 使用纯文本（比如 `{"tool_name": ...}` 加上各种括号和引号）比使用单一的 Special Token 要消耗更多的 Token，略微增加推理成本。
2.  **存在格式幻觉（Syntax Error）的风险：** 因为模型是在生成普通文本，偶尔可能会漏掉一个括号、引号，导致 JSON 无法被你的代码解析（不过经过良好 SFT 的模型，这种错误率通常在 1% 以下）。

### 总结

即使没有特殊的 Tool Call Token，你完全可以通过**“System Prompt 定义工具格式” + “SFT 训练模型输出 JSON/XML” + “后处理代码解析文本”** 这一套流程，让 LLaMA 具备极其强大的工具调用能力。著名的伯克利函数调用排行榜（BFCL）中，很多开源模型就是依靠这种纯文本格式化输出的方法名列前茅的。