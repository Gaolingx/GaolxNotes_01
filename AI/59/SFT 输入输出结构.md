# SFT 输入输出结构

### 🧑‍💻 User

在进行大语言模型的监督微调（SFT）时，常用的输入输出结构是什么？

---

### 🤖 Assistant

在进行大语言模型（LLM）的监督微调（Supervised Fine-Tuning, SFT）时，数据的核心目标是将“预训练的续写能力”转化为“指令遵循”或“对话交互”能力。

常用的输入输出结构通常分为 **逻辑结构（数据字段）** 和 **物理结构（Prompt 模版）** 两个层面。以下是详细的分类解析：

### 1. 逻辑数据结构（JSON/JSONL 格式）

在准备训练数据集文件时，最常用的格式是 JSON 或 JSONL。根据任务类型（单轮指令 vs 多轮对话），结构略有不同。

#### A. Alpaca 格式（单轮指令微调）
这是最经典的格式，源自斯坦福的 Alpaca 项目。它将数据分为三个字段：
*   `instruction`:以此描述任务。
*   `input`: 任务的上下文（可选，如需要摘要的文章、需要翻译的句子）。
*   `output`: 模型的预期回答。

**示例：**
```json
{
    "instruction": "请将下面的句子翻译成英文。",
    "input": "今天天气真好。",
    "output": "The weather is really nice today."
}
```

#### B. ShareGPT / 对话格式（多轮对话微调）
随着 ChatGPT 的流行，多轮对话成为了主流。这种结构通常由一个列表组成，列表中包含按时间顺序排列的消息对象。
*   `role`: 角色，通常分为 `system`（系统提示词）、`user`（用户）、`assistant`（模型）。
*   `content`: 具体内容。

**示例：**
```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个乐于助人的AI助手。"},
      {"role": "user", "content": "你好，请介绍一下量子力学。"},
      {"role": "assistant", "content": "量子力学是物理学的一个分支..."},
      {"role": "user", "content": "那薛定谔的猫是什么？"},
      {"role": "assistant", "content": "薛定谔的猫是一个著名的思想实验..."}
    ]
  }
]
```

---

### 2. 物理输入结构（Prompt Template / 模版拼接）

虽然我们在 JSON 中存储了结构化数据，但在真正喂给模型（Model Input）时，这些数据必须被拼接成**一个完整的字符串**，并加上特殊的标记（Special Tokens），以便模型区分哪部分是用户说的，哪部分是它自己该说的。

不同的基座模型（Base Model）通常有特定的模版格式：

#### A. Alpaca 风格模版
通常用于较早期的 LLaMA 微调。
```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```
*注：如果 `input` 为空，则会使用不包含 Input 部分的模版。*

#### B. ChatML 格式 (Qwen, OpenAI 等)
这是一种非常流行且鲁棒的格式，使用显式的特殊 Token 来标记对话轮次。
```text
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{model_response}<|im_end|>
```

#### C. Llama 3 格式
Meta 的 Llama 3 使用了自定义的特殊 Token。
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{model_response}<|eot_id|>
```

#### D. Mistral / Llama 2 格式 ([INST])
使用 `[INST]` 标签包裹用户的输入。
```text
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_input} [/INST] {model_response} </s>
```

---

### 3. 训练时的处理关键：Loss Masking（损失掩码）

这是 SFT 中最重要的技术细节。

虽然输入给模型的是拼接好的完整字符串（User + Assistant），但在计算损失函数（Loss）时，我们**只计算模型输出部分的 Loss**。

假设输入序列 $X = [x_1, x_2, ..., x_n]$，其中 $x_1...x_i$ 是 Prompt（指令+输入），$x_{i+1}...x_n$ 是 Response（回答）。

*   **Prompt 部分**：在 Label 中通常被设置为特殊值（如 PyTorch 中的 `-100`），这样在计算交叉熵损失时，这部分会被忽略。
*   **Response 部分**：保留真实的 Token ID 作为 Label。

**数学表达：**
通常的语言模型损失是最大化似然函数：
$$ \mathcal{L} = - \sum_{t=1}^{n} \log P(x_t | x_{<t}) $$

但在 SFT 中，我们修改为：
$$ \mathcal{L}_{SFT} = - \sum_{t=i+1}^{n} \log P(x_t | x_{<t}) $$
也就是只优化模型生成的回答部分，而不去强迫模型预测用户的提问。

### 总结

| 层面 | 结构名称 | 描述 |
| :--- | :--- | :--- |
| **数据存储** | Instruction/Input/Output | 适用于单项任务指令。 |
| **数据存储** | Messages (Role/Content) | 适用于聊天机器人，含 System/User/Assistant。 |
| **模型输入** | Prompt Template | 将上述数据拼接成带特殊 Token 的长字符串（如 ChatML, Llama3 格式）。 |
| **训练目标** | User Masking | Prompt 部分 Loss 设为 0，只学习 Response 部分。 |

---

### 🧑‍💻 User

为什么sft经常使用三元组格式更精细地控制模型行为？

---

### 🤖 Assistant

使用“指令（Instruction）、输入（Input）、输出（Output）”这种**三元组（Triplet）**格式，而不只是简单的“用户问（User）- 模型答（Assistant）”二元对，核心原因在于**解耦（Decoupling）**。

这种结构将“**要做什么（任务逻辑）**”与“**对什么做（处理对象）**”分离开来。这种分离为 SFT 提供了更精细的控制力，主要体现在以下几个方面：

### 1. 强化模型的“泛化能力”（Generalization）

如果将指令和输入混合在一起（例如：“请把这句话翻译成英文：今天天气很好”），模型可能会倾向于死记硬背整个句子。

通过三元组分离：
*   **Instruction**: 代表函数逻辑 $f(\cdot)$（例如：翻译、摘要、纠错）。
*   **Input**: 代表变量 $x$（例如：新闻文章、代码片段、具体句子）。
*   **Output**: 代表结果 $y$。

**数学类比：**
模型学习的目标不仅仅是拟合数据，而是学习函数映射关系 $y = f(x)$。
通过固定 `Instruction` 而变换不同的 `Input`，或者固定 `Input` 而变换不同的 `Instruction`，模型能更清晰地理解：即便 $x$ 变了，$f$ 的逻辑依然存在。这极大地提升了模型在**Zero-shot（零样本）**场景下的表现。

### 2. 实现“一源多用”的多任务微调

三元组格式非常适合构建多任务数据集。对于同一个客观存在的文本（Input），我们可以通过改变 Instruction 来训练模型不同的能力。

**示例：**
假设 **Input** 是一篇关于《红楼梦》的短文。

*   **样本 A (Instruction: 摘要):** “请概括这段话的主旨。” $\rightarrow$ Output: 简介了红楼梦的历史地位。
*   **样本 B (Instruction: 实体抽取):** “请提取文中出现的人名。” $\rightarrow$ Output: 贾宝玉, 林黛玉。
*   **样本 C (Instruction: 风格改写):** “请用鲁迅的语气重写这段话。” $\rightarrow$ Output: ...

如果不区分 Instruction 和 Input，模型很难在训练数据中通过“对比”来捕捉不同任务之间的细微差异。

### 3. 清晰定义长文本处理的边界

在处理长文本任务（如RAG、长文档摘要）时，Input 可能非常长。如果把 Instruction 和 Input 混在一起，Instruction 很容易被淹没在大量的 Input 文本中，导致“指令遗忘”。

三元组结构在物理拼接 Prompt 模版时，通常会采用显式的分隔符（如 `### Instruction:` 和 `### Input:`）。

*   **精细控制：** 这种结构让模型明确知道，哪部分是**不可违背的命令**，哪部分是**仅供参考的材料**。
*   **抗干扰：** 防止 Input 中的内容（比如 Input 本身包含了一句“忽略前面的指令”）对 Instruction 造成 Prompt Injection（提示词注入）式的干扰。

### 4. 便利于自动化数据合成 (Data Synthesis)

目前高质量的 SFT 数据往往依赖 GPT-4 等强模型进行合成（如 Self-Instruct 方法）。三元组结构极大地简化了数据生成的流水线：

1.  **种子生成**：先生成 100 个不同的 `Instruction`（任务类型）。
2.  **实例化**：对每个 `Instruction`，让大模型生成对应的 `Input`。
3.  **标注**：将 `Instruction + Input` 喂给大模型生成 `Output`。

如果数据结构不分层，很难程序化地控制数据分布（例如：很难统计出“翻译任务”占了多少比例，“摘要任务”占了多少比例）。

### 5. 模拟编程逻辑（Code-like reasoning）

大语言模型（尤其是代码能力强的模型）对结构化数据的理解能力很强。三元组结构本质上是在模拟编程中的函数调用：

$$ \text{Output} = \text{run\_task}(\text{instruction\_code}, \text{data\_input}) $$

这种格式隐含地鼓励模型运用**逻辑推理**能力，而不是简单的文本续写能力。它告诉模型：“这是一个操作，应用于那个数据”，从而减少幻觉（Hallucination），提高遵循指令的准确率。

### 总结

SFT 使用三元组格式，本质上是为了**降低模型的学习难度**并**提高训练数据的信噪比**。

*   **二元组（User/Assistant）**：更像自然的闲聊，适合训练 Chatbot 的对话流畅度。
*   **三元组（Inst/Input/Output）**：更像执行命令，适合训练 Agent 的工具调用能力、逻辑处理能力和特定任务的执行力。