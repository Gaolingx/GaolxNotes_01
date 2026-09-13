# AgenticRL概念解析

### 🧑‍💻 User

agenticrl中，Policy，Harness，Environment，rubric有什么区别？

---

### 🤖 Assistant

在 Agentic RL（智能体强化学习）里，这四个概念处于不同层级，可以简单记成：

- **Policy**：决策者，负责“想和做”
- **Environment**：执行现场，负责“接动作、给结果”
- **Harness**：训练/评估管线，负责“把所有东西串起来跑”
- **Rubric**：评分标准，负责“判断做得好不好”

它们的数据流通常类似：

```text
Harness 启动任务
  -> 把当前状态/历史交给 Policy
  -> Policy 生成动作（文本、工具调用等）
  -> Environment 执行动作，返回新状态/观测
  -> Harness 调用 Rubric 对轨迹或最终结果评分
  -> Harness 汇总 reward，交给 RL 算法更新 Policy
```

---

## 1. Policy：决策策略

Policy 是智能体的“大脑”。在数学上就是策略：

$$\pi_\theta(a_t \mid s_t)$$

其中 $s_t$ 是当前状态或上下文历史，$a_t$ 是智能体选择的动作。

在 Agentic RL 中，Policy 通常由大语言模型实现，输出可能是：

- 自然语言回答
- 工具调用，例如 `search("...")`
- 代码片段
- 规划步骤
- 最终答案

它是唯一被训练/优化的对象。RL 的目标就是更新策略参数 $\theta$，让智能体在环境中获得更高 reward。

---

## 2. Environment：环境

Environment 是智能体交互的外部世界。它接收 Policy 产生的动作，执行后返回：

- 新的观测或状态
- 环境反馈，例如命令输出、网页内容、文件内容
- 终止信号，例如任务完成或失败
- 有时也包括环境自带的 reward

在 Agentic RL 中，Environment 可以是：

- 代码解释器或沙盒
- 终端 / Docker 容器
- 浏览器或网页
- 数据库或 API
- 模拟器

Environment 通常是黑盒且不可微的，RL 不直接对它求梯度。

---

## 3. Harness：训练/评估脚手架

Harness 是工程编排层，负责把 Policy、Environment、Rubric 组合起来。它不是环境本身，而是“线束”或“测试台”。

它通常负责：

- 加载任务和数据集
- 管理 rollout 循环
- 并发执行多个 episode
- 调用 Environment 执行动作
- 调用 Rubric 计算 reward
- 收集轨迹、日志和指标
- 把训练数据交给 RL 算法，例如 PPO、GRPO 等
- 管理检查点、早停、评估

可以理解为：Harness 是训练和评估的 driver，定义“一条任务如何从开始跑到结束”。

---

## 4. Rubric：评分标准

Rubric 是评判智能体行为或最终结果的标准。它决定一个 trajectory 或最终输出应该得多少 reward。

Rubric 可以是：

- 规则型：例如最终答案是否等于标准答案
- 程序型：例如单元测试是否通过
- 模型型：例如 LLM-as-a-Judge 按照流畅性、相关性、正确性打分
- 混合型：例如代码题用测试通过率，写作题用 LLM judge

Rubric 通常由 Harness 调用，用于把智能体的行为转化为 reward。它本身一般不被 RL 更新，但它的设计对训练效果影响很大。

---

## 5. 核心区别总结

| 概念 | 核心角色 | 输入 | 输出 | 是否被 RL 训练 |
|------|----------|------|------|----------------|
| **Policy** | 决策者 | 当前状态/历史 | 动作、文本、工具调用 | 是 |
| **Environment** | 执行现场 | 动作 | 新状态、观测、终止信号 | 否 |
| **Harness** | 编排管线 | 任务、Policy、Environment、Rubric | 轨迹、reward、日志、更新信号 | 否 |
| **Rubric** | 评分标准 | 轨迹或最终输出 | 分数 / reward | 否，但设计很关键 |

---

## 6. 一个具体例子

假设任务是“让智能体在 Docker 容器里修复一个代码 bug”：

- **Policy**：一个正在微调的 LLM，它输出“打开文件 X”“修改第 10 行”“运行测试”等动作。
- **Environment**：Docker 容器中的代码仓库和终端。Policy 的动作会在里面真实执行，返回文件内容、命令输出、测试结果。
- **Rubric**：例如“单元测试是否通过”“修复 diff 是否足够简洁”等评分规则。
- **Harness**：整个训练框架。它加载 issue 数据，循环调用 LLM 和 Docker 容器，用 Rubric 计算 reward，记录轨迹，最后把数据交给 RL 算法更新 LLM。

所以一句话区分：

> **Policy 负责决策，Environment 负责执行和反馈，Harness 负责把流程跑起来，Rubric 负责定义什么叫“好”。**