# 终端Agent过拟合修复方案

### 🧑‍💻 User

如果你是基座模型后训练团队的leader遇到overfit terminal agent environment的问题用什么办法修？
案例：某个自研terminal agent有两套present： Standard（约25个工具）/Minimal（只有pwsh/read两个基本工具）。某个模型使用自研的Terminal-based AI Agent和sandbox在Minimal present（具备特定的tool/trajectory/system prompt）下进行训练，在Minimal present下可以在某个swe benchmark上拿到70分以上的pass@k，但推理时只要换成其他Agent（例如claude code/codex等）或者Standard present，则相同的bench分数下滑10分以上，但只要复刻第一轮自研agent的Minimal present轨迹后面自由发挥即可恢复训练时的模型性能，复现方法如下：
其他terminal agent/自研agent Standard present下跑分只有 91/92分，且易陷入低效的 “Let me...” 思维链（CoT）轨迹，，自研agent Minimal present是99/96，但它只有两个工具（pwsh/read）。我们做了一个两阶段插件：首轮只给shell+read锚定 Minimal present 推理轨迹，第一次工具调用后立刻恢复Standard present全部25项工具。bench 连跑两次 98/99，均值98.5，基准分回升至Minimal present水平，观察cot发现第二轮cot发生了显著变化且废话变少（“Let me...”出现率变为0，出现大量“We need...” 推理轨迹），证明不是抽卡，也不需要牺牲完整工具能力。
消融实验指出，不是模型能力不行，而是对首次请求的工具结构和trajectory极其敏感——表现高度依赖首轮请求中可见的 API 工具目录（Schema Surface）和system prompt，这个"开局配置"决定了模型整段任务采用哪套推理策略。
请你分析为什么会出现如此巨大的推训差异，如何定位发生的位置以及设计方案如何紧急修复（要求算力花销尽量小，因为模型核心能力并未丢失通常不需要大规模回滚）

---

### 🤖 Assistant

## TL;DR

这不是"能力丢失"，而是 **RL 把"工具目录 + 协议格式"学成了策略的隐式条件变量**。模型实际学到的是 $\pi(a \mid s,\ \mathcal{T},\ \mathcal{P})$，其中 $\mathcal{T}$ 是可见工具 schema surface，$\mathcal{P}$ 是 system prompt / 工具调用协议。训练只覆盖了 $(\mathcal{T}_{\text{min}}, \mathcal{P}_{\text{自研}})$ 这个点，推理时一换成 $(\mathcal{T}_{\text{std}}, \mathcal{P}_{\text{外部}})$ 就进入了训练的低密度区甚至 OOD 区；而 transformer 自回归 + 首轮 KV cache 的"滚雪球"效应，把首轮的推理模式锁定到了任务结束。两阶段插件之所以有效，本质是**把 OOD 的初始状态重新投影回 ID 流形，并利用首轮锚定的吸引域（basin of attraction）锁定正确模式**。

---

## 一、根因拆解：为什么推训差异这么大

### 1. 策略的条件变量被"偷换"了

在 Minimal present 下训练，模型学到的不只是"如何完成任务"，而是"**当上下文里只有 pwsh + read 时，该如何规划、如何调用、如何写 CoT**"。工具目录 $\mathcal{T}$ 和协议 $\mathcal{P}$ 不是无关的输入噪声，而是策略的条件变量：

$$ \pi(a_t \mid s_t) \;\rightarrow\; \pi(a_t \mid s_t,\ \mathcal{T},\ \mathcal{P}) $$

RL 的 reward 只给"任务是否完成"，**从不惩罚策略对 $\mathcal{T}$ 的过拟合**。于是模型走了一条 shortcut：把"只有 2 个工具"这个特征当成了强先验，学出一套 **shell-centric 的推理-行动耦合策略**——因为只有 pwsh/read，所有"读、搜、改、验证"都要在脑内先组合成 shell 管道命令，这迫使 CoT 里出现大量规划（"We need to find X → grep/sed → verify"）。

换成 Standard present（25 工具）后，条件变量 $\mathcal{T}$ 突变，模型训练时**从未在"25 工具 + 自研协议"这个输入分布下被充分强化**（或覆盖极少），于是策略进入低密度区域，表现退化。

### 2. 首轮是"模式开关"，自回归锁死后续

你的消融已经证明：**首轮可见的 schema surface + system prompt 决定了整段任务用哪套推理策略**。机制是两层：

- **模式路由（latent mode selection）**：模型内部在训练中形成了多个策略簇 / 吸引域。首轮的 schema 上下文是一个强条件信号，把 hidden state 推入某个吸引域（Minimal 模式 vs Standard 模式）。两阶段插件就是先用 2 工具把模型"路由"进 shell-centric planning 模式。
- **自回归滚雪球**：一旦首轮 CoT 进入 "We need..." 的 planning 风格，后续 token 的分布就条件在这个已生成的风格上；KV cache 又把首轮的"模式表征"一路传播下去。所以模式一旦选中就不可逆，直到任务结束——这就是为什么"首轮锚定，后面自由发挥"仍然 98/99。

### 3. "Let me..." vs "We need..." 是两种 latent mode 的语言学标记

- **"Let me..."** = 第一人称、动作即时性，典型的**反应式探索**（reactive trial-and-error）。工具一多，模型的选择空间爆炸，它倾向于"先试一个再说"，CoT 变啰嗦、低效，反复试错。
- **"We need..."** = 目标/规划语气，典型的**前瞻式规划**（prospective planning）。Minimal 下模型没有高级工具一步到位，被迫在动手前把命令组合想清楚，反而形成了更高质量的推理。

这两个词不是巧合，它们是两个策略簇的 **surface marker**。Standard 下 "Let me..." 泛滥 = 模型掉进了训练低密度的 reactive 模式；锚定后 "We need..." 回归 = 模型被拉回了训练高密度的 planning 模式。

### 4. 为什么换外部 Agent（claude code / codex）掉得更狠

换 Agent 不只是 25 个工具的问题，而是**三个变量同时变**：

| 变量 | 自研 Minimal | 外部 Agent |
|---|---|---|
| 工具数量 $\mathcal{T}$ | 2 | 多 |
| 工具协议 $\mathcal{P}_{\text{proto}}$ | 自研格式 | function calling / XML 等 |
| system prompt $\mathcal{P}_{\text{sys}}$ | 自研 | 外部模板 |

三个条件变量全变，输入完全 OOD，掉分是必然。你的两阶段插件实际上**只修复了 $\mathcal{T}$ 这一个维度**就恢复了性能，说明 $\mathcal{T}$（schema surface）是主导变量，协议和 prompt 的影响次之——这本身就是一个很有价值的发现。

---

## 二、如何定位发生的位置（诊断 playbook）

按成本从低到高，建议做以下四件事，把"根因"从"现象"变成可复现的定位：

### 1. 变量消融矩阵（几乎零成本，最优先）

把三个变量拆开测，确认主导维度：

```
维度 A：工具数量        (2 vs 25，协议和 prompt 保持自研)
维度 B：工具调用协议    (自研格式 vs 外部格式，数量保持 2)
维度 C：system prompt   (自研 vs 外部，数量/协议保持)
```

做一个 $2\times2\times2$ 或至少两两交叉的矩阵。你已经证明 A 是主因，但还需确认：**是"数量"本身，还是"具体哪些工具"**？可以再加一组"25 个工具但描述改成 Minimal 风格"来区分。

### 2. 首轮分叉点分析（低代码成本）

对比 Minimal（成功）与 Standard（失败）两条轨迹，找**第一次分叉的 token 位置**：

- 统计首轮 CoT 的 token 熵、风格聚类（用 n-gram 或简单 embedding）。
- 定位 "Let me..." 首次出现的 step 和上下文：是否总是伴随"工具选择不确定"（比如同时提到多个候选工具）？
- 看 log-prob：Standard 下首轮工具选择的概率分布是否比 Minimal 更"扁平"（熵更高 = 更困惑）。

这一步能精确告诉你"模式从哪一步开始崩"。

### 3. 隐藏层表征聚类 / probe（中等成本，但能给出硬证据）

- 取模型中间层激活，对"2 工具上下文"和"25 工具上下文"的 hidden state 做聚类/t-SNE。
- 预期结果：两种 schema surface 把首轮表征推到了**不同的流形/簇**。这直接证明"schema 是模式开关"。
- 更进一步，训练一个轻量线性 probe：能否仅凭首轮 hidden state 预测后续是 "Let me..." 还是 "We need..." 轨迹。如果 probe 准确率高，说明模式在首轮就已决定。

### 4. 训练数据覆盖审计（一次性，强烈建议做）

直接查训练集：

- Standard present（或 25 工具）轨迹的占比是多少？如果 <5% 或为 0，那就是典型的 **coverage gap / covariate shift**，而非能力问题。
- 训练轨迹的 CoT 关键词分布：如果 "We need..." 在训练数据里高频、"Let me..." 低频，那 Standard 下的退化就是模型在低密度区的表现。

> **结论验证**：做完 1+4 基本就能定案——"首轮 schema surface 是模式开关，Standard 覆盖率不足导致该模式下策略退化"。

---

## 三、紧急修复方案（按算力从零到大排序）

核心原则：**模型能力没丢，只是模式路由失灵，优先推理侧对齐分布，训练侧只做"路由修复"而非"能力重训"。**

### 方案 A：推理侧修复（零训练，最快上线）

| 手段 | 做法 | 说明 |
|---|---|---|
| **A1. 首轮锚定产品化** | 把你已验证的两阶段插件固化成推理特性：首轮只暴露 pwsh/read + 自研 system prompt，第一次工具调用后释放全部工具 | 已验证 98/99，直接工程化，风险最低 |
| **A2. System prompt 引导词注入** | 不删工具，只在 system prompt 里加一句强引导：*"Operate in a shell-first planning mode: state your plan with 'We need...' before invoking any tool."* | 更轻，但要验证：纯文本引导是否足以复刻 schema 的触发强度（可能不够，schema 是更强的信号） |
| **A3. 渐进式工具披露** | 把 25 工具分层：首轮 2-4 个核心工具 → 第二步扩展到核心集 → 之后全集 | 比"2→25"更平滑，减少模式切换突兀感 |
| **A4. 协议/schema 统一** | 把外部 Agent（claude code/codex）的工具 schema、描述语言改写成与自研一致的格式 | 让 $\mathcal{P}_{\text{proto}}$ 回到训练分布，零训练成本 |
| **A5. 低效 CoT 早期检测** | 推理早期检测到 "Let me..." 试错循环时，触发一次重采样或注入 planning 提示 | 兜底，防止模式滚雪球 |

**优先顺序：A1（已验证）→ A4（兼容外部 Agent）→ A3/A2（体验优化）→ A5（兜底）。**

### 方案 B：轻量训练修复（小算力，LoRA 级别）

如果 A 方案无法覆盖所有部署场景（比如必须首轮就暴露 25 工具），用 LoRA 做**路由修复**而非重训：

| 手段 | 做法 | 算力估计 |
|---|---|---|
| **B1. Standard-present 的 LoRA SFT 混合** | 用几百~几千条 Standard present 高质量轨迹（可用你的两阶段插件批量生成）做 LoRA，专门教"在 25 工具下也用 shell-centric planning" | 单卡几小时~几十 GPU 小时 |
| **B2. DPO/KTO 偏好对齐** | 配对数据：同任务下"Let me...低效轨迹" vs "We need...高效轨迹"，惩罚低效 CoT | 与 B1 同量级 |
| **B3. 课程 RL（从 Minimal checkpoint 出发）** | 在 Standard present 下做一轮轻量 RL（用 bench reward），保留首轮锚定，让模型在 25 工具下也保持 planning 模式 | 略高于 B1，但可控 |

**关键点**：B1 的数据必须包含"首轮 25 工具 + We need 风格 CoT"的轨迹，即明确教模型"schema 变了但推理策略不变"。这是把 $\mathcal{T}$ 的条件依赖"切断"的最直接方式。

### 方案 C：数据/协议侧（工程手段，零训练）

- **C1**：把自研 Minimal 的 system prompt 精髓（角色定义、行为约束、shell 优先原则）蒸馏进外部 Agent 的 system prompt。
- **C2**：如果外部 Agent 允许自定义工具 schema，把它的工具描述改写成自研的命名/描述风格，降低协议 OOD。

---

## 四、长期加固（防止复发）

1. **训练期 schema augmentation**：随机化工具数量、工具名、协议格式、system prompt 模板，让模型对 $\mathcal{T},\mathcal{P}$ 鲁棒——这是根治"首轮敏感"的正解。
2. **RL rollout 混合**：Minimal 与 Standard 的轨迹按比例混合采样，避免单一 present 过拟合。
3. **把"首轮锚定"作为官方推理特性保留**：它本质是一个零成本的模式路由开关，可以作为 feature 而非 workaround 长期存在。
4. **监控 CoT marker 分布**：上线后持续监控 "Let me..." / "We need..." 的出现率，作为模式漂移的早期告警指标。

---

## 机制一句话总结

> **模型在 RL 中把"极小工具目录 + 自研协议"学习成了策略的隐式条件变量；推理时该变量一改，策略的 latent mode 被路由到了训练低密度区；首轮 token 的自回归锁死效应使这一模式不可逆，直到任务结束。两阶段插件的作用是把 OOD 首轮重新投影回 ID 流形并锁定 planning 模式——所以不是抽卡，是确定性的模式恢复。**

**建议的紧急动作顺序**：先做方案 A1（把已验证的插件产品化）+ 诊断第 4 步（数据覆盖审计，确认根因），如果仍有必须首轮全量工具的场景，再上 B1（LoRA 路由修复）。整体算力成本可以控制在"几十 GPU 小时"以内，无需回滚大模型。

---

### 🧑‍💻 User

是的，目前最明显的问题是overfit special sequence触发certain policy，一般的RL overfit 只会overfit policy，harness和policy的相似与差异，在哪里。还是说是等价。如果overfit Sequence，那明显就是数据本身出了问题吧。比如OPD做logits level RL的时候忘记去重了…可能

---

### 🤖 Assistant

这个问题问到根子上了。核心是把"过拟合"拆成不同的层次，否则"overfit policy"和"overfit sequence"会被混为一谈，导致归因错误。

## 一、先分清三层过拟合，它们不是一回事

| 层次 | 过拟合对象 | 数学形式 | 失效条件 |
|---|---|---|---|
| **L1 策略过拟合** | 状态分布 | $\pi(a\mid s)$ 对 $d_{\text{train}}(s)$ 敏感 | 状态 $s$ OOD（经典 covariate shift） |
| **L2 条件/上下文过拟合** | harness 条件依赖 | $\pi(a\mid s,\ \mathcal{H})$ 对 $\mathcal{H}$ 敏感 | harness $\mathcal{H}$ 变化 |
| **L3 序列记忆** | 字面 token 序列 | $\pi \approx \text{Lookup}(x_{\text{prefix}})$ | 前缀序列被扰动 |

你的案例**主因是 L2，不是 L1，也不是纯 L3**。证据是你自己给的：

- 换 harness（2→25 工具、换 Agent）才崩 → 不是 L1（状态 $s$ 没变，任务没变）。
- 锚定首轮后**能生成任务相关的新轨迹**（"We need..." 是生成式规划，不是复读固定串）→ 不是纯 L3（否则锚定也没用，因为 L3 需要逐字匹配）。

所以准确的表述是：**模型没有 overfit policy（能力），而是 overfit 了"policy 对 harness 的依赖"（条件耦合）**。

## 二、Harness 与 Policy：不等价，但被耦合，极限会坍缩

这是你问的关键。在经典 RL 里，environment 是外生的 MDP，policy 是 $\pi(a|s)$，两者天然分离。但在 **LLM agent 里，harness 是输入 token 的一部分**：

$$ \pi(a \mid s,\ \mathcal{H}),\quad \mathcal{H} = \text{system prompt} \oplus \text{tool schema} \oplus \text{协议} $$

所以 harness 不是"环境"，而是**策略的条件变量**。这带来一个关键后果：

> 训练时模型把 harness 从"外部环境"内化成了"内部决策条件"。评估时我们却假设 harness 是"可替换的脚手架"——**这个假设错了**。

而"等价"这件事，只在过拟合的极限发生。看这个坍缩方向：

$$ \pi(a \mid s,\ \mathcal{H}) \;\xrightarrow{\text{过拟合}}\; \pi(a \mid \mathcal{H}) \;\xrightarrow{\text{数据重复}}\; \text{Lookup}(\mathcal{H}_{\text{literal}}) $$

- **正常**：policy 是 $s$ 和 $\mathcal{H}$ 的联合函数，harness 只是条件之一。
- **L2 过拟合**：$\mathcal{H}$ 成为主导信号，$s$（任务本身）的权重被压低 → 换 harness 就崩。
- **L3 极限**：policy 坍缩成 harness 的字面查表 → 这时的确可以说 **policy ≈ harness 的函数**，两者"等价"了。

你的两阶段插件之所以有效，就是因为它**重新提供了训练时的那个 $\mathcal{H}$ 字面值**，把 policy 从 $\mathcal{H}_{\text{std}}$ 的分支切回了 $\mathcal{H}_{\text{min}}$ 的分支。这证明模型内部存在**按 harness 分叉的多套策略**，而不是一套通用策略——这是 L2 的典型特征。

## 三、"Overfit sequence 就是数据出了问题" —— 对，但要拆成两类

你的直觉对了一半。数据问题分两个性质：

### 1. 数据 **bug**：去重失败（你说的情况）

如果 logits-level RL（per-token policy gradient / logits 蒸馏这类）在训练时**没有对 prompt 前缀去重**，后果是真实的：

- 每个样本的 system prompt + tool schema 前缀高度重复，逐 token RL 会在这些重复 token 上**累积巨额梯度**。
- 模型对"这个特定前缀"的响应被反复强化、固化，前缀的任何字面扰动（换协议格式、改工具描述措辞）都会让它脱离训练分布。
- 这会把 **L2 推向 L3**：从"条件依赖"退化为"序列固化"。

这是可以直接查的 bug，查法：**统计训练集 prompt 前缀（去 schema 后）的去重率，以及轨迹开头 n-gram 的重复度**。如果高度重复，L3 风险坐实。

### 2. 数据 **结构**：共享前缀（去重也修不掉）

但这里有个更隐蔽的点，去重解决不了：

> **即使每个任务的 prompt 都不同，它们的 harness 前缀（system prompt + tool schema）是共享的。** 去重后，这个共享前缀仍然在每个样本里出现一次。

所以模型**必然**会对这个共享前缀条件化——这是结构性的，不是去重能修的。去重解决"重复样本过采样"，**augmentation 才能解决"共享前缀条件化"**。两者是不同层次：

| 问题 | 性质 | 修法 |
|---|---|---|
| 重复样本 → 序列固化 | bug | 去重 |
| 共享 harness 前缀 → 条件耦合 | 结构 | schema/prompt augmentation |

## 四、怎么快速定位到底在 L2 还是 L3

三个实验，成本从低到高：

1. **未见任务泛化（判 L3）**：在 Minimal 下跑**训练时没见过的新任务**。若仍 70+ → 排除 L3，是 L2 的通用能力；若崩 → 坐实 L3，数据 bug 优先查。

2. **字面扰动（判 L2 vs L3 的敏感度）**：在 Minimal 下，把 system prompt / 工具描述**同义改写**（不动数量、不动协议）。若性能骤降 → 对字面序列敏感（L3 倾向）；若稳 → 对语义敏感（L2）。

3. **前缀去重率审计（直接查 bug）**：统计训练 prompt 前缀和轨迹开头的重复度。若高 → 你猜的"logits-level RL 忘去重"成立，是 L3 的放大器。

## 结论一句话

> 你的案例是 **L2（harness 条件耦合）为主**：policy 没有过拟合能力，而是把 harness 内化成了强条件变量，换 harness 就切到训练低密度分支。你担心的"logits-level RL 忘去重"是**真实存在但属于 L3 放大器**的 bug——它能把 L2 恶化成序列固化，但**不是根因**；根因是训练时 harness 分布太窄（单一 Minimal present），导致 policy 对 $\mathcal{H}$ 的条件依赖过强。

所以修复优先级应该是：**先跑实验 1/2 确认 L2 还是 L3** → 若是 L3，补去重（便宜）；无论是否 L3，长期都要做 **harness augmentation**（随机化工具数、命名、协议、prompt 模板），这才是切断 $\pi$ 对 $\mathcal{H}$ 依赖的正解。