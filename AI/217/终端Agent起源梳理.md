# 终端Agent起源梳理

### 🧑‍💻 User

最早搞terminal agent概念的是什么时候 哪家ailab？

---

### 🤖 Assistant

已经有足够信息了，下面直接给结论。

## 结论先说

如果问的是「**大 AI 实验室**里谁最早把 *terminal agent（终端/CLI 里的 AI agent）* 做成产品」，答案是：

> **Anthropic —— 2025 年 2 月发布的 Claude Code（research preview）**，这是第一家大型 AI lab 推出的 terminal-native agent。

但如果严格追溯「**terminal agent 这个概念本身是谁先搞的**」，那它并不是大厂 lab 发明的，而是**开源社区 + 学术界**更早：

- **2023 年**：`Aider`（Paul Gauthier，2023 年 5 月）和 `Open Interpreter`（Killian Lucas，2023 年 7 月）是最早在终端里跑起来的 AI agent。
- **2024 年 4 月**：普林斯顿 NLP 的 **SWE-agent** 是学术界最早把「agent 在终端/命令行接口里自主操作」系统化的工作（论文 *Agent-Computer Interfaces Enable Automated Software Engineering*）。

所以「哪家 AI lab」这个问题要分两层回答，时间线如下：

## 时间线

| 时间 | 是谁 | 说明 |
|------|------|------|
| 2023-05 | Aider（开源，Paul Gauthier） | 最早的终端 AI 结对编程工具，git 原生、多模型 |
| 2023-07 | Open Interpreter（开源，Killian Lucas） | 在终端里执行代码/操作的 agent，爆火 |
| 2024-03 | Devin（Cognition，创业公司） | 第一个上头条的「自主软件工程师」，云端沙箱里有 shell/浏览器/编辑器 |
| 2024-04 | SWE-agent（普林斯顿） | 学术界把「terminal 操作 agent」正式化，定义 agent-computer interface |
| **2025-02** | **Claude Code（Anthropic）** | **第一家大型 AI lab 的 terminal-native agent** |
| 2025-04/05 | Codex CLI（OpenAI） | Rust 编写、开源的终端 coding agent |
| 2025-06 | Gemini CLI（Google） | Google 的终端 agent |

## 补充两点

1. **「terminal agent」这个品类为什么 2025 年爆发**：Anthropic（Claude Code）、OpenAI（Codex CLI）、Google（Gemini CLI）在 2025 年 2 月到 6 月这几个月内几乎同时推出终端 agent，本质上是把「agent loop」——读文件系统 → 执行 shell 命令 → 观察结果 → 再决策——架在 Unix 终端这个现成的可组合、可 headless、可脚本化的执行环境上。

2. **如果你说的「terminal/终端 agent」指的是「端侧智能体（on-device agent）」**——也就是把 agent 跑在手机/PC 本地而非云端的那个概念——那是另一条线（Apple Intelligence、Gemini Nano、各大厂商的端侧小模型路线），跟上面的 CLI agent 不是一回事。从你问法里的 "terminal agent + ailab" 判断，我上面按「终端命令行 agent」来答了；如果你其实想问端侧，我可以再展开。

**一句话总结**：概念源头在 2023 年的开源社区（Aider / Open Interpreter），学术系统化是 2024 年普林斯顿的 SWE-agent；而**大型 AI lab 里最早的是 Anthropic 的 Claude Code（2025 年 2 月）**。

---

### 🧑‍💻 User

只是那个时候模型性能不行harness不动而已，当时openai好像还在卷stem和chat，computeruse和agentos都是近两年提出的概念了吧，能不能解释下这些新概念

---

### 🤖 Assistant

你这个判断基本是对的，而且点到了关键：**概念和"能落地"是两回事**。Terminal agent 的 idea 在 2023 年就有，但那时候 GPT-4/Claude 3 级别模型的工具调用、长程规划、错误恢复能力都不够，agent 跑几步就崩，所以只能是 demo/开源玩具。真正把它 "harness 起来"，靠的是 2024 年 Claude 3.5 Sonnet、GPT-4o 这一代 + 之后 o1/o3 那批推理模型的规划能力。

OpenAI 在 2023 年确实重心在 chat 和 STEM/coding benchmark（GPT-4、Codex 那套），把 "agent" 当主线产品是 2024 下半年之后的事。你提到的两个新概念我分开讲：

---

## 1. Computer Use（计算机操控 / GUI Agent）

**定义**：让模型像人一样操作电脑的**图形界面**——看屏幕截图 → 判断 → 移动鼠标/点击/输入/滚动，循环往复完成任务。和 terminal agent 的区别在于：terminal 走的是**结构化文本接口**（shell 命令），Computer Use 走的是**像素级 GUI 接口**。

**关键里程碑**：

| 时间 | 事件 |
|------|------|
| 2017 | OpenAI 等发 **World of Bits / MiniWoB**——概念上最早的"agent 操作界面"，但当时是 RL 小模型，纯学术 |
| 2024-10-22 | **Anthropic 发布 Claude "Computer Use"**（Claude 3.5 Sonnet），第一个可用的通用 GUI 操控能力，这个词自此成为品类名 |
| 2025-01-23 | **OpenAI 发布 Operator**（浏览器 agent）+ **CUA（Computer-Using Agent）**模型 |
| 2025-10 | OpenAI 在 API 放出 `computer-use-preview` 模型 |

**为什么现在才成立**：GUI 操控比 terminal 难一个量级——模型要理解像素、做视觉定位、处理弹窗/遮挡/延迟，还要几十上百步不崩。2023 的模型根本扛不住，所以 2017 的概念到 2024 才被 Claude 3.5 的视觉 + agentic 能力"激活"。

---

## 2. Agent OS（Agent 操作系统 / Agent 运行时）

**定义**：不是"给 agent 装的操作系统"，而是**把 agent 当作程序来管理的那一层基础设施**。传统 OS 管的是进程（调度、内存、隔离、进程间通信），Agent OS 管的是 agent（上下文=内存、工具=系统调用、沙箱=隔离、多 agent 通信=IPC）。

核心思想一句话：**"Agents are programs"**，所以它们需要一套 OS 原语：持久状态、调度、权限、恢复、多 agent 编排。

**关键代表**：

| 时间 | 事件 |
|------|------|
| 2023-10 | **MemGPT**（Berkeley，后改名 **Letta**）——最早把"agent 记忆/状态管理"做成核心抽象，后来直接自称 **agent OS** |
| 2024-03 | **AIOS**（Rutgers，arXiv 论文 "AIOS: LLM Agent Operating System"）——学术界最早用"OS 内核"视角给 LLM agent 做调度/内存/权限 |
| 2024-11 | Anthropic 发 **MCP**——相当于 agent 的"驱动/设备协议"，让工具接入标准化 |
| 2025 起 | 各大厂的 agent 运行时/编排层爆发：OpenAI **AgentKit/ChatKit**、Anthropic **Agent SDK**、Google **ADK**、以及一堆开源 AgentOS |

**它的核心隐喻**（这张对应关系很直观）：

| 传统 OS | Agent OS |
|---------|----------|
| 进程 | 一个 agent 会话 |
| 内存 | 上下文窗口 + 外部记忆 |
| 系统调用 | 工具调用（tool call） |
| 驱动/协议 | MCP / tool spec |
| 沙箱/权限 | agent 沙箱、审批、权限边界 |
| 进程调度 | 多 agent 编排、任务队列 |

---

## 三者是什么关系

```
模型能力（推理/规划/视觉）         ← 2024 之前是瓶颈
      ↓
agent 的"感官"：terminal(CLI) / computer use(GUI)   ← 两种接口，同一底层能力
      ↓
Agent OS / 运行时：记忆、调度、权限、多agent、恢复   ← 让 agent 长期跑、规模化
```

- **Terminal agent 和 Computer Use 是并列的两种"接口"**，不是谁替代谁：terminal 可控、可脚本化、便宜（适合 coding/devops），GUI 通用但慢、贵、易出错（适合替人点网页/软件）。
- **Agent OS 在它们下面/外面一层**，是让"单个 agent 能做事"升级到"agent 系统能长期运行"的基础设施。

---

**回到你的判断**：对，terminal agent 的 idea 没变，变的是模型能力让接口终于"带得动"了；Computer Use 和 Agent OS 这两个**词**确实是 2024-2025 年才火起来的（概念种子更早），本质都是模型变强之后，把 agent 从"单次问答"推向"能长期自主操作真实环境"的产物。

如果你对某一块想深入（比如 MCP 为什么成了事实标准、或者 Letta/AIOS 具体怎么实现"agent 内存"），我可以接着展开。