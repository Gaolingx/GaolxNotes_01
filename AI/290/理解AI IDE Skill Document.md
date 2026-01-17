# 理解AI IDE Skill Document

### 🧑‍💻 User

如何理解主流ai ide（codex、claude code、roo code）的skill document？有什么用途？

---

### 🤖 Assistant

在当前主流的 AI 编程工具（如 Cursor、Roo Code/Cline、Claude Code 等）中，所谓的 **"Skill Document"**（技能文档、规则文件或记忆库）可以被理解为 **AI Agent 的“员工手册”或“外挂大脑”**。

它本质上是一种 **Prompt Engineering（提示工程）的持久化形式**，用于弥补大模型（LLM）通用能力的不足，使其适应特定的项目、代码风格和业务逻辑。

以下从**核心概念、具体用途、主流实现方式**三个维度进行深度解析：

---

### 1. 核心概念：如何理解 Skill Document？

大模型（如 GPT-4, Claude 3.5 Sonnet）就像一个刚入职的**天才资深工程师**：
*   **优点**：精通所有编程语言，算法烂熟于心。
*   **缺点**：完全不懂你们公司的“潜规则”（比如：变量命名必须用驼峰？数据库连接必须用连接池？这个模块不能动？）。

**Skill Document 就是为了解决“缺点”而存在的。** 它主要包含两层含义：

1.  **约束与规范（Rules）：** 强制 AI 遵守的编码纪律。
2.  **上下文与记忆（Context/Memory）：** 项目特有的架构知识和历史决策。

从技术实现上讲，这些文档会在 AI 生成代码之前，被注入到 System Prompt（系统提示词）或 Context Window（上下文窗口）中，从而改变模型的输出概率分布。

---

### 2. 具体用途（Why use it?）

Skill Document 的用途主要体现在以下四个方面：

#### A. 统一代码风格（Style Consistency）
LLM 默认的代码风格往往是随机的或通用的。Skill Document 可以强制规定：
*   **语法偏好**：例如，“在 TypeScript 中总是使用 `interface` 而不是 `type`”。
*   **库的选择**：例如，“进行 HTTP 请求时，必须使用 `axios`，严禁使用 `fetch`”。
*   **命名规范**：例如，“数据库字段使用 `snake_case`，API 响应字段使用 `camelCase`”。

#### B. 注入领域知识（Domain Knowledge Injection）
对于复杂的遗留系统或特定领域，AI 需要额外的背景信息：
*   **架构说明**：解释当前项目的目录结构（DDD, Clean Architecture 等）。
*   **业务逻辑**：例如，“计算价格时，必须先扣除折扣，再计算税费”。
*   **Tech Stack 限制**：例如，“这是一个 Next.js 12 的老项目，不要使用 App Router，只能使用 Pages Router”。

#### C. 减少幻觉与重复劳动（Reduction of Hallucination）
如果没有文档，你每次提问都要重复：“请使用 Tailwind CSS”。有了 Skill Document，AI 会默认执行，并且不会编造不存在的 API（如果文档中限制了 API 范围）。

#### D. 自动化工作流（Workflow Automation）
特别是对于 **Roo Code** 这种 Agent 类型的工具，文档可以定义行为模式：
*   “在提交代码前，必须先运行 `npm test`。”
*   “如果遇到报错，必须先读取日志文件，而不是直接猜测。”

---

### 3. 主流 AI IDE 的具体实现案例

不同的工具对 Skill Document 有不同的称呼和实现方式：

#### A. Cursor (`.cursorrules`)
这是目前最流行的实现方式。
*   **形式**：在项目根目录下创建一个 `.cursorrules` 文件。
*   **内容**：自然语言编写的指令。
*   **示例**：
    ```markdown
    You are an expert in Python and Django.
    - Always use Type Hints.
    - When writing views, use Class-Based Views (CBV).
    - Do not use `print()`, use the configured `logger`.
    ```
*   **理解**：Cursor 会在每次对话时，自动读取该文件并将其作为最高优先级的指令发送给模型。

#### B. Roo Code / Cline (Memory Bank & `.clinerules`)
Roo Code（原 Cline）引入了更高级的**Memory Bank（记忆库）**概念，这是一种结构化的 Skill Document 体系。
*   **形式**：通常由一组 Markdown 文件组成，如 `productContext.md`（产品背景）、`activeContext.md`（当前任务）、`techContext.md`（技术栈）。此外还有 `.clinerules` 用于定义行为准则。
*   **机制**：Roo Code 会**自动维护**这些文档。当你完成一个任务，它会更新 `activeContext.md`。这不仅仅是规则，更是**动态的长期记忆**。
*   **用途**：解决了 AI 聊久了就“忘事”的问题，让 AI 知道项目的演进过程。

#### C. Claude Code (Context Awareness)
虽然 Claude Code 是一个 CLI 工具，但它极度依赖项目的**元文档**。
*   **形式**：它会主动扫描 `README.md`、`CONTRIBUTING.md` 以及项目中的类型定义文件。
*   **理解**：它没有单一的“Skill 文件”，而是将项目现有的文档视为 Skill。如果你的 `CONTRIBUTING.md` 写得好，Claude Code 的表现就会由 60 分提升到 90 分。

---

### 4. 总结：Skill Document 的价值公式

如果用数学逻辑来类比，普通的 AI 编程是：

$$ \text{Output} = \text{Model}(\text{Prompt}) $$

加上 Skill Document 后的 AI 编程是：

$$ \text{Output} = \text{Model}(\text{Prompt} + \text{Skill Docs} + \text{Project Context}) $$

**总结来说，Skill Document 是连接“通用大模型”与“私有项目落地”的桥梁。** 它是未来程序员必须掌握的一项新技能——即“**如何编写文档让 AI 看懂**”，这比编写让人类看懂的文档更具生产力价值。

---

### 🧑‍💻 User

也就是说，一个skills document组成包括skill文档库（例如 skills.md），和一些辅助这个文件的其他工具，比如其他的markdown，模板，脚本等，模型可以用某种形式访问。相当于一个按需加载到上下文的文档，平时只露个头，确实有需要再把整张抽出来看。这个理解是正确的吗？

---

### 🤖 Assistant

你的理解非常到位，甚至可以说触及了 **AI Agent（智能体）** 架构的核心。

你所描述的“**平时只露个头，确实有需要再把整张抽出来看**”，在技术上对应的是 **Tool Use (工具调用)** 或 **RAG (检索增强生成)** 的机制。

为了让你更精准地把握这个概念，我们需要把目前的 AI IDE 分为**两个阶段**来理解，因为它们的处理方式略有不同：

### 1. 第一阶段：全局注入（Static Context）
**代表工具：** 早期的 Copilot, 基础版 Cursor (`.cursorrules`)

在这个阶段，你的理解稍微“超前”了一点。
*   **机制**：工具通常会把 Skill Document（比如 `.cursorrules`）**完整地**、**无条件地**塞进 System Prompt 里。
*   **状态**：不管你问什么，这些规则都在上下文里。
*   **缺点**：如果文档太长，会占用 Context Window（上下文窗口），而且过多的规则会让模型“注意力涣散”。

### 2. 第二阶段：按需加载/代理模式（Agentic & Dynamic Context）
**代表工具：** Roo Code (Cline), Claude Code (配合 MCP), 高级 Agent 框架

**这正是你理解的模式，也是目前最先进的形态。** 你的理解完全正确：

#### A. “只露个头” (The Definition/Schema)
模型在初始状态下，并不知道具体的技能细节，它只看到了一个**技能索引（Index）**或**工具描述（Schema）**。

例如，System Prompt 里可能只有这样一段话：
```markdown
I have the following skills available via tools:
1. `deploy_logic`: How to deploy to AWS.
2. `db_migration`: Rules for altering database schema.
3. `test_gen`: Templates for writing unit tests.
```
这就是“露出的头”。

#### B. “把整张抽出来看” (Retrieval/Execution)
当你对 AI 说：“帮我写一个数据库迁移脚本”时：
1.  **决策**：AI 意识到这涉及 `db_migration`。
2.  **调用**：AI 此时才会去读取 `skills/db_migration.md` 文件，或者调用一个名为 `get_migration_rules` 的工具。
3.  **加载**：具体的规则（例如“必须使用 Alembic”、“字段不能为 NULL”）此时才进入上下文。

---

### 关于“辅助工具”的深入理解

你提到的“**辅助这个文件的其他工具（Markdown、模板、脚本）**”，在 **MCP (Model Context Protocol)** 时代变得尤为重要。Skill Document 不再仅仅是死文本，它是**可执行的知识**。

我们可以把一个完整的 **Skill** 结构化为以下三层：

| 层次 | 组成 | 你的比喻 | 作用 |
| :--- | :--- | :--- | :--- |
| **L1: 索引层** | `skills.json` 或 `index.md` | **露个头** | 让 AI 知道有哪些技能可用，并在何时调用。 |
| **L2: 知识层** | `deploy_guide.md`, `style_guide.md` | **抽出整张纸** | 具体的步骤、SOP（标准作业程序）、注意事项。 |
| **L3: 执行层** | `scripts/deploy.sh`, `templates/react_comp.tsx` | **手里的工具** | AI 不仅读取文档，还可以直接运行脚本或填充模板，实现“知行合一”。 |

### 举个实战例子（Roo Code / Cline 场景）

假设你有一个 Skill 叫 **“创建组件”**。

1.  **用户**：创建个新的 Button 组件。
2.  **AI (思考)**：这里需要用到 `CreateComponent` 技能。
3.  **AI (查阅 Skill 文档)**：读取 `skills/frontend-rules.md`。
    *   *文档内容*：所有组件必须是 Functional Component，必须使用 TypeScript，且必须同时生成 Storybook 文件。
4.  **AI (调用辅助模板)**：读取 `templates/component.tsx.ejs`。
5.  **AI (输出)**：根据文档的规则，填充模板，生成符合你公司规范的完美代码。

### 总结

你的理解非常精准：**Skill Document 本质上就是 AI 的“外挂知识库”和“工具箱”。**

*   **静态看**：它是文档。
*   **动态看**：它是**路由系统**。它负责在正确的时间，把正确的知识（Context）加载到模型的“显存”里，既节省了 Token，又提高了准确率。