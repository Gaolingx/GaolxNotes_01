# SFT 输入输出结构

### 🧑‍💻 User

在进行大语言模型的监督微调（SFT）时，常用的输入输出结构是什么？

---

### 🤖 Assistant

在大语言模型（LLM）的**监督微调（Supervised Fine-Tuning, SFT）**阶段，最常用、也最成熟的输入输出结构，本质上是**“条件语言建模（conditional LM）”**，即：

> **给定一段“输入上下文（prompt / context）”，预测模型期望输出的“目标回复（response）”。**

从工程和数据组织角度看，通常可以分为以下几种主流结构。

---

## 一、最基础结构：Prompt → Response（单轮指令）

### 形式

```text
[Input / Prompt]
→
[Target / Response]
```

### 典型示例

```json
{
  "instruction": "解释什么是监督学习",
  "output": "监督学习是一种……"
}
```

### 训练方式

* 将 **Prompt + Response** 拼接成一段文本
* 对 **Response 部分计算 loss**
* Prompt 部分仅作为条件，不参与 loss

### 特点

* 最简单
* 数据制作成本低
* 适合 instruction-following 的冷启动

---

## 二、Instruction / Input / Output 三段式（Alpaca 风格）

这是目前**最常见的工业与学术 SFT 数据格式**之一。

### 结构

```text
Instruction: 描述任务要做什么
Input: 具体实例（可选）
Output: 正确答案或期望回复
```

### 示例

```json
{
  "instruction": "将下面的英文翻译成中文",
  "input": "Large language models are powerful.",
  "output": "大语言模型非常强大。"
}
```

### 模型看到的拼接文本（示意）

```text
### Instruction:
将下面的英文翻译成中文

### Input:
Large language models are powerful.

### Response:
大语言模型非常强大。
```

### 特点

* 明确区分 **任务描述 vs 实例**
* 对泛化到“新任务”非常有帮助
* 已成为事实上的 instruction tuning 标准格式

---

## 三、对话式结构（Chat / Multi-turn SFT）

这是**对齐 chat 模型（ChatGPT 类）时的核心结构**。

### 结构

```text
System
User
Assistant
User
Assistant
...
```

### 示例（JSON 形式）

```json
{
  "messages": [
    {"role": "system", "content": "你是一个严谨的AI助手"},
    {"role": "user", "content": "什么是SFT？"},
    {"role": "assistant", "content": "SFT是监督微调……"}
  ]
}
```

### Loss 计算规则

* **只在 assistant 的 token 上计算 loss**
* system / user 作为条件上下文

### 特点

* 能学习：

  * 对话轮次
  * 角色区分
  * 上下文延续
* 是现代 chat LLM 的**必备格式**

---

## 四、统一视角：Token-level Masked Loss

无论外部数据格式如何，**在模型内部几乎都被统一成同一件事**：

> **给定 token 序列，某些 token 计算交叉熵 loss，其余 token mask 掉**

### 形式化表达

给定序列：
[
x = (x_1, x_2, ..., x_T)
]

定义 mask：
[
m_t =
\begin{cases}
1, & x_t \in \text{assistant / output} \
0, & \text{otherwise}
\end{cases}
]

优化目标：
[
\mathcal{L} = -\sum_t m_t \log p_\theta(x_t \mid x_{<t})
]

这也是为什么：

* **SFT 本质上仍然是 next-token prediction**
* 区别只在于 **loss 在哪些 token 上生效**

---

## 五、工程中常见的几种变体

### 1. 单样本多输出（多参考答案）

```json
{
  "instruction": "...",
  "outputs": [
    "...",
    "..."
  ]
}
```

* 用于缓解“一问多答”的模式崩塌
* 实际训练时通常拆成多条样本

---

### 2. CoT / Reasoning SFT

```text
Answer:
Let's think step by step...
```

* 推理链显式写入 response
* 会显著改变模型的推理风格
* 也是“能力偏移”的一个重要来源

---

### 3. Tool / Function Call SFT

```json
{
  "role": "assistant",
  "content": null,
  "function_call": {
    "name": "search",
    "arguments": "{...}"
  }
}
```

* 输出不再是自然语言
* 而是**结构化 token 序列**
* 本质仍是 SFT

---

## 六、总结一句话

**SFT 中最常用、最本质的输入输出结构是：**

> **“以用户/系统提供的上下文作为条件，只对期望的 assistant 输出部分做 next-token 监督。”**

外部可以是：

* instruction–output
* instruction–input–output
* multi-turn chat

但在模型眼里：

* 都是 **带 mask 的语言模型训练**

如果你愿意，我也可以进一步从你之前关心的 **“SFT 为什么会导致能力偏移 / 遗忘”** 或 **“与 RLHF / RLVR 的分工差异”** 角度，把这些结构和能力变化对应起来。


---

### 🧑‍💻 User

为什么sft经常使用三元组格式更精细地控制模型行为？

---

### 🤖 Assistant

SFT 中**经常使用“三元组（instruction / input / output）格式”**，并不是偶然的数据工程选择，而是一个**在“可控性、泛化性与优化稳定性”之间高度折中的设计**。从建模、优化和行为对齐三个层面，可以比较系统地理解这一点。

---

## 一、核心动机：把“要做什么”和“对什么做”解耦

三元组的本质是**显式因子分解**：

[
p(\text{output} \mid \text{instruction}, \text{input})
]

而不是模糊的：

[
p(\text{output} \mid \text{prompt})
]

### 1. Instruction：**任务先验（task prior）**

* 描述“你现在在执行哪一类映射”
* 决定输出的：

  * 语义目标
  * 风格
  * 约束条件（长度、格式、语气）

### 2. Input：**实例条件（instance condition）**

* 提供具体要处理的对象
* 理论上可以为 *empty*（即零样本）

### 3. Output：**监督信号（target behavior）**

* 明确告诉模型在该条件下“应该怎么做”

这种拆分，相当于**把 latent task variable 显式化**，降低了模型在 SFT 阶段的“猜任务”负担。

---

## 二、从优化角度看：减少梯度噪声与行为歧义

### 1. 单 prompt 结构的问题

如果只用：

```text
Prompt → Response
```

那么一个 prompt 往往**同时隐含**：

* 任务类型
* 输入数据
* 输出风格

不同样本之间，这些隐含因素高度耦合，导致：

* 同一表面形式 ≠ 同一任务
* 同一任务 ≠ 同一表面形式

**结果：**

* 梯度方向噪声大
* 更容易出现“能力此消彼长”的偏移

---

### 2. 三元组降低条件熵

三元组相当于告诉模型：

> “在这个 instruction 语义子空间里，input 是变量，output 的分布应该稳定。”

这会带来两个直接好处：

* **更快收敛**
* **更稳定的任务内泛化**

你可以把 instruction 看成一种 **task token / soft routing signal**。

---

## 三、行为可控性：显式对齐“风格、格式和约束”

在实际 SFT 中，instruction 往往不只是“做什么”，而是：

* 用什么语气
* 输出什么结构
* 是否允许解释
* 是否要 step-by-step

### 示例

```text
Instruction:
请用 JSON 格式输出，不要包含任何解释

Input:
{name: "Alice", age: 30}

Output:
{"name":"Alice","age":30}
```

这种约束如果混在 input 里：

* 模型需要**先解析“这是约束”**
* 再判断“约束优先级是否高于内容本身”

三元组让这些约束成为**一等公民**。

---

## 四、对“指令泛化（instruction generalization）”至关重要

三元组格式直接支撑了 instruction tuning 的一个关键能力：

> **对“未见过的 instruction wording”进行泛化**

原因在于：

* Instruction 在训练中是**高频变化维度**
* Input 是低维变化（具体实例）

模型学到的是：
[
f(\text{instruction}, \text{input}) \rightarrow \text{output}
]

而不是：
[
f(\text{prompt surface}) \rightarrow \text{output}
]

这也是为什么：

* Alpaca / FLAN / T0 一类工作都强调该结构
* 没有 instruction 的 SFT 很难“听懂新任务”

---

## 五、对齐与安全：可控拒答与行为边界

在安全与对齐相关的 SFT 数据中，instruction 往往承担“边界声明”的角色：

```text
Instruction:
如果请求涉及违法行为，请拒绝并解释原因
```

然后配合不同 input：

* 合法
* 灰色
* 非法

让模型学到的是：

* **条件化拒答**
* 而不是无条件的“遇到关键词就拒绝”

这在单 prompt 格式下几乎不可控。

---

## 六、从你关心的“能力偏移”角度再看一眼

你之前问过：**为什么 SFT 会强化某些能力、削弱另一些能力**。

三元组的一个隐藏价值是：

> **它在一定程度上“隔离”了能力子空间。**

* 不同 instruction ≈ 不同 task manifold
* 梯度更新在 task 子空间内更集中
* 减少对 base LM 通用分布的无意识拉扯

当然，它不能完全避免遗忘，但**显著比“杂糅 prompt”要好**。

---

## 七、总结一句话

**SFT 使用三元组格式，并不是为了“数据看起来更规整”，而是为了：**

> **显式建模任务条件、降低优化歧义、增强指令泛化，并提高行为与安全的可控性。**

如果你愿意，下一步可以进一步讨论：

* 为什么 **即便使用三元组，SFT 仍然会导致能力偏移**
* 或者三元组在 **MoE / tool calling / reasoning SFT** 中的不同权重分配方式
