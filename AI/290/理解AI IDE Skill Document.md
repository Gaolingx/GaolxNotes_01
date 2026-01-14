# 理解AI IDE Skill Document

### 🧑‍💻 User

如何理解主流ai ide（codex、claude code、roo code）的skill document？有什么用途？

---

### 🤖 Assistant

在主流 **AI IDE / Agentic Coding 工具**（如 Codex、Claude Code、Roo Code）中，所谓的 **skill document** 并不是“给人看的使用说明”，而是**给模型看的、可被稳定消费的能力约束与行为规范载体**。它的核心作用，是把“这个 IDE 里有哪些能力、该在什么条件下用、怎么用、边界在哪里”**显式化并外部化**。

下面我从**定义 → 用途 → 为什么一定要是 document → 不同产品的实现差异 → 与 prompt / tool schema 的关系**几个层面系统解释。

---

## 一、什么是 Skill Document（本质定义）

**Skill document = 对一组可调用能力（skills）的结构化、可读、长期稳定描述**

它通常包含以下信息（即便形式不同，语义高度一致）：

1. **Skill 的语义定义**

   * 这个能力“是干什么的”
   * 解决什么类型的问题
   * 适用 / 不适用的场景

2. **调用前提（Preconditions）**

   * 什么时候“值得”调用
   * 什么时候必须先询问用户
   * 是否需要 workspace / repo / build system 已就绪

3. **操作约束（Constraints）**

   * 不允许的行为（例如：不能 silent refactor、不能删文件）
   * 安全 / 合规边界
   * 成本或副作用

4. **执行流程（Operational Contract）**

   * 高层步骤
   * 期望中间状态
   * 成功 / 失败判定

5. **与其他 skills 的关系**

   * 依赖
   * 互斥
   * 推荐组合顺序

从认知上说，它是：

> **“把 agent 的能力空间从隐式 prompt 工程，升级为显式、可治理的技能系统”**

---

## 二、Skill Document 的核心用途（不是给你看的）

### 1️⃣ 让模型“知道自己会什么、不会什么”

如果没有 skill document，模型只能靠：

* 上下文猜测
* 模糊的 system prompt
* 过拟合的 tool 名字

结果是：

* 过度调用（hallucinated tooling）
* 能力抖动（同一问题有时用、有时不用）
* 难以复现和调试

Skill document 提供的是：

> **稳定的能力边界与可供规划的动作空间**

---

### 2️⃣ 支持 Planning / Tool Selection 的“理性化”

在 Codex / Claude Code 这类 agent 中，模型内部通常有：

```
Goal → Plan → Skill Selection → Execution → Verification
```

Skill document 是 **Plan 阶段可被显式引用的知识源**：

* 可以在 chain-of-thought 中枚举 skills
* 可以对比 “这个 skill vs 那个 skill”
* 可以判断 “现在调用是否 premature”

这一步 **无法只靠 tool schema 完成**，因为：

* schema 只有“怎么调”
* document 才有“为什么、何时、是否该调”

---

### 3️⃣ 作为长期 memory / alignment 的稳定锚点

Prompt 是**易碎的**：

* 容易被上下文污染
* 易被 jailbreak
* 不同 session 不稳定

Skill document 的设计目标是：

* **长期存在**
* **低频变更**
* **版本可控**

它起到的是类似：

* API 文档 + SOP + 行为宪章

对 agent 来说，这是“世界观的一部分”。

---

### 4️⃣ 降低 agent 行为的不可预测性（工程视角）

从工程角度，skill document 是：

* 可 diff
* 可 review
* 可 audit
* 可 A/B

这使得：

* 行为变化可以溯源到文档变更
* 不再完全依赖 prompt 魔法
* 更容易做 safety / compliance

---

## 三、为什么一定是「Document」，而不是 Prompt 或 Tool Schema？

这是关键问题。

### ❌ Tool Schema 的局限

Tool schema 只能表达：

* 参数类型
* 必填 / 可选
* 返回值结构

**它完全表达不了：**

* 什么时候不该调用
* 调用的社会语境（ask user first）
* 失败时的 fallback 策略
* 与其他能力的优先级关系

---

### ❌ Prompt 的局限

Prompt 是：

* 非结构化
* 强依赖上下文顺序
* 易被 user prompt override

Skill document 是：

* 可被“引用”的知识对象
* 可以被 planner 单独读取
* 不容易被 prompt 注入污染

---

### ✅ Document 的优势

Skill document 本质是：

> **“可被模型多次、稳定读取的知识型工件（knowledge artifact）”**

这是 agent 系统里极其重要的一层。

---

## 四、不同产品里的 Skill Document 形态差异

### Codex（OpenAI 系）

* 强调 **workspace-aware skills**
* document 通常描述：

  * 文件系统操作
  * build / test / lint 生命周期
* 更偏 **工程自动化 SOP**

典型风格：

> “When modifying code, always run tests unless explicitly told not to.”

---

### Claude Code（Anthropic 系）

* skill document 和 **Constitutional AI** 风格一致
* 强调：

  * 何时应该向用户确认
  * 何时拒绝操作
* 偏 **行为边界 + reasoning guidance**

---

### Roo Code / 社区 Agent

* skill document 更像：

  * DSL 说明
  * agent 能力注册表
* 强调可扩展性、插件化

---

## 五、Skill Document 在更大图景中的位置

如果把现代 AI IDE / Agent 拆层：

```
User Intent
   ↓
Planner / Reasoner
   ↓
Skill Document  ←——— 这是“能力知识层”
   ↓
Tool / API / Code Executor
   ↓
Environment (Repo / FS / OS)
```

**Skill document 是 Reasoner 和 Execution 之间的“语义缓冲层”**。

它的存在，意味着：

* 模型不是在“即兴发挥”
* 而是在“按操作手册执行”

---

## 六、一句话总结（给工程师的理解方式）

> **Skill document 的本质，是把 agent 的“隐性能力假设”升级为“显性、可治理、可推理的能力契约”。**

它不是 UI 文档，不是 marketing 文档，而是：

* 给模型看的
* 给 planner 用的
* 给工程团队治理 agent 行为的

如果你愿意，下一步我可以：

* 拆一份真实的 Claude Code / Roo Code skill doc，逐段解析其设计意图
* 或讨论：**如果你自己要设计一个 AI IDE，skill document 应该如何分层与版本化**


---

### 🧑‍💻 User

也就是说，一个skills document组成包括skill文档库（例如 skills.md），和一些辅助这个文件的其他工具，比如其他的markdown，模板，脚本等，模型可以用某种形式访问。相当于一个按需加载到上下文的文档，平时只露个头，确实有需要再把整张抽出来看。这个理解是正确的吗？

---

### 🤖 Assistant

是的，你这个理解**非常准确**，而且已经触及了主流 AI IDE / agent 设计里一个**关键但很少被明说的实现层事实**。

我会在肯定你判断的基础上，把它**再精确化一层**，避免一些潜在的误解。

---

## 一、结论先行（校准后的准确表述）

你描述的可以概括为：

> **Skill system = 一组可被索引、可按需加载到上下文的“能力文档工件集合”，其中核心是 skills document（如 `skills.md`），外围是用于解释、约束、模板化和执行该 skill 的辅助工件。**

并且：

* **默认不全部注入上下文**
* **只在 planning / decision / execution 需要时被检索或展开**
* **模型知道“它存在”，但不常驻记忆**

这一点在工程上是**刻意设计**的，而不是偶然。

---

## 二、需要精确区分的几个层次（非常重要）

你的理解基本正确，但有 3 个层次值得明确区分，否则后面容易混淆。

### 1️⃣ “Skill 文档库” ≠ “单个 skills.md”

在成熟系统里，几乎一定是：

```
skills/
  ├── index.md          # skills registry / summary
  ├── refactor.md       # 单一 skill 的完整契约
  ├── test.md
  ├── build.md
  ├── ask_user.md
  └── ...
```

`skills.md`（或 index.md）通常只承担：

* skill 名称
* 一句话能力描述
* 适用场景摘要
* 指向完整文档的引用

**它本身就是“露个头”的那一层**。

---

### 2️⃣ “辅助文件”并不是附属品，而是 skill 的一部分

你提到的：

> 其他 markdown、模板、脚本等

这一点理解得非常到位，而且可以再强化一句：

> **一个 skill 的真实定义 = 文档 + 模板 + 执行约束 + 失败处理逻辑**

例如：

* `refactor.md`
* `refactor_checklist.md`
* `refactor_diff_template.md`
* `verify_refactor.sh`

这些不是“工具”，而是：

> **skill 的可执行外延（operational surface）**

模型在调用 skill 时，实际上是在**消费一个“小型操作系统接口”**。

---

### 3️⃣ “按需加载”不是性能优化，而是认知控制

你说：

> 按需加载到上下文，平时只露个头

这是**非常关键的洞察**。

这么做并不只是为了节省 token，而是为了：

* 防止 skill 之间互相干扰
* 防止 agent 行为过拟合
* 防止 context drift

在 planner 看来，典型流程更像是：

```
1. 我知道我有这些 skills（index）
2. 当前目标是否需要某个 skill？
3. 是 → 加载该 skill 的完整契约
4. 在该契约约束下行动
5. 行动结束后，丢弃细节，只保留结果
```

这几乎等价于：

> **“动态链接而非静态链接”**

---

## 三、你这个理解已经接近真实实现，而不是文档说法

很多公开材料会模糊地说：

> “We provide the model with skills”

但你描述的其实更接近**内部真实工程形态**：

* skills 是**文件系统对象**
* 不是 prompt 片段
* 不是常驻 system message
* 而是被检索、被引用、被展开的 artifacts

在 Codex / Claude Code 这类系统里：

* 模型**并不知道全部 skill 的细节**
* 它只知道：

  * 名称
  * 大致能力
  * 如何请求“查看更多”

---

## 四、一个重要的补充修正（避免一个常见误解）

> ❌ Skill document ≠ “随时可自由读取的知识库”

实际上通常有**访问控制**：

* planner 阶段：只能看到 index / summary
* decision 阶段：可以请求展开某个 skill
* execution 阶段：只能引用已展开的 skill
* 有些 skill **只能通过特定 meta-action 触发**

也就是说：

> **模型不是“翻文档”，而是在“走流程”**

---

## 五、用一句更工程化的话总结你的理解

你现在的理解可以升级为这句话：

> **Skill system 是一个可被 agent 通过规划阶段发现、通过显式请求加载、并在执行阶段临时注入上下文的“能力文档 + 操作工件集合”，而不是一个常驻 prompt 或单一 markdown 文件。**

这是**完全正确、而且是偏实现层的理解**。

---

## 六、如果你愿意继续深挖（可选方向）

基于你目前的理解水平，下一步值得讨论的是：

1. **Skill document 如何与 tool calling / function calling 协同而不是重复**
2. **Skill 的 granularity 如何设计（太细 vs 太粗）**
3. **Skill document 的“版本化”如何影响 agent 行为稳定性**
4. **为什么 skill document 往往比模型权重更难对齐**

这些已经是**设计 agent 的核心难题**，而不是使用层问题了。
