# NeurIPS JEV

## 🧑‍💻 User

写一篇jev相关研究方向的report要能投nips

---
## 😀 默认助手 (7)

I found what "JEV" refers to in the ML literature — let me gather the primary sources before writing.

# JEV 研究方向报告：从"System One 决策模型"到 NeurIPS 选题

> **快照时间**：2026-10-15 ｜ **文献基线**：arXiv 摘要含 "Jev" 共 **125** 篇（标题含 "Jev" **66** 篇），社区 awesome-list 收录 **174** 条（161 篇预印本，最近核验 2026-10-14）
> **投稿目标**：NeurIPS 2027 Main Track / Evaluations & Datasets Track（NeurIPS 2026 已于 2026-09-24 放榜，主赛道关闭）

---

## 0. 结论先行

**JEV 在本语境下指 TypeSafe AI 的 "Jev" 决策模型（2026-09-15 发布），不是日本脑炎病毒。** 它定义了一个新的模型接口范式：输入 `state` + 一组**带类型的多个选择题**，一次前向返回每题一个**概率分布**，不生成文本。截至今日，围绕它已形成"接口 → 评测 → 安全 → 开源复现 → 应用"五层生态，且**平均每天 3–5 篇新论文**（有二手报道称发布 7 天内出现 13 篇 arXiv 论文）。

核心判断：**"Jev 好用不好用"类实证评测已严重内卷；真正空的是"这个接口在数学上意味着什么"以及"如何为它设计有保证的算法"。**

| 优先级 | 方向 | 一句话卖点 | 目标 track | 可行性 |
|---|---|---|---|---|
| **P0** | **D1 列表条件决策的公理、识别与去偏算法** | 决策 API 返回的概率不是选项的属性，而是"选项列表"的泛函；给出公理刻画 + 列表不变的去偏读出器，在 IIA 违例与精度之间给出可证的 Pareto 前沿 | Main | ★★★★★ |
| **P0** | **D2 类型化决策的复合性：误差传播与联合校准** | 单步 97% → 序列仅 2–33% 的"正确性留存塌陷"是可形式化、可修复的 | Main | ★★★★☆ |
| **P1** | **D3 一次前向决策的可摊销极限** | 证明某些任务族上"任何单次决策头"有 Ω(1) 超额风险，而 k 步序列可达 o(1)（带匹配上界） | Main（理论） | ★★★★☆ |
| **P1** | **D4 列表决策 API 的分布无关风险控制与弃权** | 给"none-of-the-above"和阈值选择以有限样本保证，稳健处理选项集变化导致的非可交换性 | Main / D&B | ★★★★☆ |
| **P2** | **D5 决策接口的变形测试套件与精确真值基准** | 精确条件概率 + 选项集变形 + 模型留置（held-out）+ 版本钉扎的基准 | **Evaluations & Datasets** | ★★★★★ |
| **P2** | **D6 决策委派的代价感知策略与遗憾界** | "局部省钱 ≠ 任务变好"的反直觉发现 + 可证 regret 的委派策略 | Main | ★★★☆☆ |
| **P3** | **D7 决策头的机制可解释性** | 为什么 mode 对而分布错？用因果干预定位选项列表注意力 | Main / Workshop | ★★☆☆☆ |

**推荐组合**：主攻 **D1**（理论+方法+实证齐备，最好写），同时用 **D5** 占一个 D&B 名额（数据/工具可复用 D1 的实验基础设施）；若时间紧，先投 **ICML 2027（截稿约 2027-01）** 占位，NeurIPS 2027 复投升级版。

---

## 1. 研究对象界定：JEV 的接口规范与技术事实

### 1.1 基本事实

| 项 | 内容 |
|---|---|
| 开发方 | TypeSafe AI（旧金山，2024 年成立；Diogo Almeida，前 OpenAI RLHF/InstructGPT/ChatGPT/GPT-4，任 CEO；联合创始人 Erik Gafni、Sasha Sheng） |
| 发布 | 2026-09-15 有限早期访问，同轮宣布 **$40M 种子轮**（DCVC 领投，估值报道约 $200M） |
| 当前版本 | `jev-1.13.0`（**版本漂移是复现性风险点**） |
| 命名 | 致敬 William Stanley Jevons 与 Jevons 悖论（"更便宜的智能 → 更大规模部署"） |
| 训练 | 纯合成数据 + 自研 **RLCD**（Reinforcement Learning for Calibrated Decisions）；疑似监督损失含 Brier/校准项，细节未公开 |
| 架构 | 官方仅称 transformer-based；**未公开架构、权重、技术论文**（外部推测基于开源 LLM 或 BERT-like） |
| 性能宣称 | 70–500 ms 端到端、比前沿 LLM 快 40–200×、便宜 40–400×；厂商自承其 workflow 由自家 team 构造，存在偏置 |

### 1.2 接口（这是所有研究的形式化起点）

输入 `state`（string / JSON object / string 数组）与若干 **question（primitive）**，每题三选一：

| 类型 | 选项空间 |
|---|---|
| `noul` | true / false（二值） |
| `choice` | 无序候选列表 $c_1,\dots,c_k$ |
| `score` | 有序量表 $0,\dots,k-1$ |

形式化记作：

$$
F_\theta:\ (x,\ q,\ L)\ \longmapsto\ p \in \Delta^{k-1},\qquad L=(c_1,\dots,c_k)
$$

其中 $x$ 是 state，$q$ 是指令文本，$L$ 是**被序列化的候选列表**。输出仅此一种：每题一个离散分布。API 额外返回的 `choice` / `score` / `confidence` 均由固定公式从 $p$ 计算，不含新信息：

$$
\text{choice}=\arg\max_i p_i,\qquad
\text{confidence}_{\text{choice}}=\frac{p_{\max}-k^{-1}}{1-k^{-1}},\qquad
\text{score}=\sum_{i=0}^{k-1} i\,p_i
$$

$$
\text{confidence}_{\text{score}}=\max\!\Big(0,\ 1-\frac{\mathrm{MAD}(p_0,\dots,p_{k-1})}{\mathrm{MAD}(1/k,\dots,1/k)}\Big),\quad
\mathrm{MAD}(p)=\tfrac1k\sum_i |i-\operatorname{mode}(p)|
$$

**两个结构性约束（决定了后面所有可做题）**：

1. **同一次调用内所有题目并行作答、类型系统禁止题间依赖**；有依赖必须拆成多次串行请求。
2. **`confidence` 只是分布的集中度度量，与"我该不该弃权"这类决策问题无关**——这是 D4 的切入口。

### 1.3 独立评测给出的基线数字（写论文时的对照锚点）

第三方 37 数据集、346,009 次请求、花费 < $10 的零样本评测（arXiv:2609.37647）给出：

- IMDB / SST-2 / HellaSwag / ARC **95–99%**；Belebele（122 语言）**86.7%**
- 在 27/37 数据集上胜过 Qwen3.8-27B，37/37 胜 Gemma-4-E4B
- `choice` 概率**校准良好**且支持选择性预测；但**二值概率相对固定 0.5 阈值的"落点"很差**（UNFAIR-ToS 上微调阈值后 micro-F1 从 0.50 → 0.75）
- 旋转选项不改变准确率，**抽掉问题文本掉到随机**（排除浅层记忆；但不排除记住问答对）
- 低资源语言、细粒度/噪声标签、rubric 式质量判断上三者都退化

---

## 2. 领域地图（截至 2026-10-15）

### 2.1 增速与体量

| 指标 | 数值 |
|---|---|
| arXiv 摘要含 "Jev" | **125** 篇（其中标题含 "Jev" 66 篇） |
| 摘要含 "Jev" 且含 "decision" | 110 篇 |
| 含 "TypeSafe" | 24 篇 |
| 社区清单条目 | 174 条 / 161 篇预印本 |
| 时间跨度 | 2026-09-15 起算，**约 30 天** |

> ⚠️ 这意味着：**任何"某某场景下 Jev 评测"类选题，投稿周期内（≥6 个月）有极高概率被抢发。** 这直接支持"做类别级（typed decision models）的普适结论"而非"Jev 的某场景表现"。

### 2.2 八条赛道一览

<details>
<summary><b>展开：文献分赛道清单（含 arXiv 号）</b></summary>

| 赛道 | 代表工作 |
|---|---|
| **① 接口/模型本体** | Jev 官方文档（`primitives`、`models`）；Wikipedia 技术规格整理 |
| **② 评测基准** | 37 数据集零样本评测 `2609.37647`；**ProbabilityBench**（精确真值条件概率 + Bayes error 元评测）`2610.17016`；**Jev-ToM**（9 个 ToM 基准，11,117 题，73.2%）`2610.18610`；**Choice2Seq**（选择 vs 串行构造的正确性留存）`2610.16191`；**DecisionRobust**（文本+视觉鲁棒性）`2610.19118`；JevBench v1.7（社区维护）；OpenDecisions（2,352 题）`2610.17205`；EmbodiedSkillBench `2610.16238` |
| **③ 开源复现/替代** | **Releasing the Inner Jev**（logit 字母读出 + LoRA 决策头，超越 Jev 4.3/4.4 Capability）`2610.17205`；**FunDM**（冻结 LM + 标签无关决策头，36 层用 24 层达 99% 精度，吞吐 1.56×）`2610.17175`；AgriJev `2610.19478`；CLM / Laya / Nimble / Von / StartLux / Sol / werr |
| **④ 理论/统计** | **Recalibrating Many Questions with Few Labels**（$m$ 问题 × $n$ 标签的重校准极小极大率与识别极限，69 页）`2610.18488` |
| **⑤ 校准方法** | PRISM（反事实探针 + 岭回归纠正 judge 偏置）`2610.18389`；选项独立打分 vs choice 打分的取舍；阈值调优 |
| **⑥ 安全/鲁棒** | **JevProbe / JPDH**（228 个真实应用、81.5% 出现决策翻转）`2610.18612`；**Poisoned into Action**（执行感知的上下文投毒协议）`2610.15537`；**Same Attack, Different Judge**（首个受控对抗评测：规则型任务最稳、整体判断型最脆）`2610.17820`；JSB（用 Jev 反馈做越狱）`2610.19971` |
| **⑦ 系统与应用** | Jev-Guard（策略树路由+安全分类，58 ms）`2610.18326`；Jev-Mem（System-One 控制平面 + 智能体记忆）`2609.23986`；Jev-Scanner（记忆证据准入）；JevCoder（SWE-bench，token 降 21.6–50.6%）`2610.19596`；BLADE（多智能体失败归因）`2610.18780`；ProAgent/StreamingJev（主动式物理 AI）`2610.16967`；JevSpawn（组合动作空间）`2610.00437`；流式检索/推荐重排 `2610.14631` `2609.40241`；国际象棋搜索+蒸馏 `2610.09188`；顶会/医学/政治学评测 |
| **⑧ 结构性发现（最有价值的一类）** | **IIA 违背**（299,463 次调用；RMS 系统位移 0.20–0.52 nats ≈ 1.2–3× 运行间波动；255 选项时约 1 nat；独立 yes/no 打分可消除依赖但**掉 8.7 个百分点**）`2610.16655`；认知/社会偏置与**合取谬误式表述敏感**（54% 违背；选定答案不变但概率剧变）`2610.19619`；**算术拒答瓶颈**（有答案 99% / 无答案正确拒答仅 7%，阈值可救到 79%）`2609.39496`；同调用内多题**互不干扰**（bias < 0.001）；记忆退休"过早退休"39.2% `2610.15710`；工具调用"推导缺口" `2610.18497` |

</details>

### 2.3 饱和 vs. 空白

| 状态 | 内容 |
|---|---|
| 🔴 **已饱和** | 单模型零样本评测（>15 篇）；领域应用评测（医学/农业/遥感/法律/政治学）；"开源廉价复现"（已 3 家抢跑：Inner Jev、FunDM、AgriJev）；安全探针与鲁棒性基准（4+ 篇）；选项顺序鲁棒性的**实验观测** |
| 🟡 **正在变红** | 对抗/注入类基准；精确真值概率基准（ProbabilityBench 已先行）；单题重校准理论（已被 69 页理论文占据） |
| 🟢 **几乎空白** | **①** 列表条件概率的**公理化与识别**；**②** 同调用内多题"并行无关性"的形式化与其**代价**（类型系统禁止依赖 → 复合决策必须串行或"生成动作空间"）；**③** 一次前向决策的**表达力/可摊销极限**；**④** 决策 API 的**分布无关风险控制**（选项集变化破坏可交换性）；**⑤** 从"决策正确性"到"任务级代价"的**委派理论**；**⑥** 决策头的**机制可解释性** |

---

## 3. Gap 分析：审稿人会问什么

用 NeurIPS 审稿人的典型否决句反推缺口：

| 审稿人异议 | 现有工作的普遍问题 | 必须补上什么 |
|---|---|---|
| "这只是一个闭源模型的黑盒评测" | 绝大多数论文只调用 `jev-1.13.0` API | **版本钉扎 + 开源权重复现**（FunDM/CLM/Nimble/Laya/Von/StartLux 均可跑）+ 随机种子与调用次数披露 |
| "empirical audit，没有可验证主张" | 结构性发现停留在"我们观察到 X 随 Y 变化" | 形式化 + **定理**（下界 / 刻画 / 保证）或**可证算法** |
| "这会被下一次模型更新淘汰" | 结论绑定具体模型版本 | 把结论建立在**接口类别**上：$F:(x,q,L)\mapsto\Delta^{k-1}$，Jev 只是一个实例 |
| "基准没有留置模型/污染控制" | 用被评测模型参与筛选 | held-out 模型、选项旋转、抽问题、精确真值、Bayes error |
| "增量安全实验" | 攻击-防御的军备竞赛 | 要么给**执行链路级的形式化威胁模型 + 上界**，要么不做 |
| "缺少代价视角" | 只报精度/延迟 | 报**每次成功交付的代价**（多篇智能体论文已发现局部省钱反而抬高该指标） |

---

## 4. 六个候选研究方向（详细规格）

### D1（P0）列表条件决策：公理、识别与列表不变读出器

- **假想标题**：*Probabilities Over Listed Options Are Not Properties of Options: Axioms, Identification, and List-Invariant Decision Heads*
- **一句话主张**：任何被当作"选项属性"使用的概率，都隐含一个可检验的公理系统；违背它可以用一个**位置先验 + 属性项**的模型完全刻画，且该模型同时给出了修复算法。

**形式化**（可直接进论文 §3）：

1. 称 $F$ 满足**列表一致性（IIA）**：若 $L\subset L'$ 且 $c_i,c_j\in L$，则 $\log\frac{p_{i|L}}{p_{j|L}}=\log\frac{p_{i|L'}}{p_{j|L'}}$。
2. 定义**列表敏感度泛函** $\mathcal{S}(F)=\sup_{L\subset L'}\big\|\log p_{\cdot|L}-\log p_{\cdot|L'}\big\|_\infty$，并给出可估的 RMS 版本（现有论文观测到 $0.20\text{–}0.52$ nats，$|L|=255$ 时约 1 nat）。
3. **定理 1（刻画）**：在"题目打包调用互不干扰"（观测：bias $<0.001$）的先验下，违反 IIA 的决策头必然可分解为
$$\operatorname{logit} p_{i|L}=s_\theta(x,q,c_i)+\underbrace{b(i)}_{\text{位置}} +\underbrace{a(c_i;L)}_{\text{近邻属性}},$$
并证明 $s_\theta$ 与 $b(\cdot)$ 在该参数化下仅相差一个可辨别的平移/尺度变换（**识别性**），给出可估性条件。
4. **定理 2（增长律）**：在位置先验模型下证明 $\mathbb{E}\,\mathcal{S}(F)=\Theta(\log |L|)$——把观测到的"255 选项 ≈ 1 nat"从经验现象升级为**定标律**。
5. **定理 3（不可兼得）**：存在常数 $c>0$ 使任意列表不变读出器在某个校准良好的目标分布上有 $\ge c$ 的精度损失——为观测到的"独立打分消除依赖但掉 8.7 点"提供**信息论解释**，并给出达到该前沿的构造。

**方法（论文的算法贡献）**：**De-biased List Scoring (DLS)**——保留选项独立的逐项打分（天然满足 IIA），叠加一个**单调、低容量**的校正头 $g_\phi$，用极小极大最优的重校准（isotonic / 分位数映射）在少量标注上学习，使
$$\tilde p_i \propto \exp\!\big(\phi_\theta(x,q,c_i)+g_\phi(\phi_\theta(x,q,c_i))\big),$$
并证明当 $g_\phi\to 0$ 时 $\mathcal{S}\to 0$；实证给出 (精度, $\mathcal{S}$) 的 **Pareto 前沿**，并给出"以 $\epsilon$-IIA 代价换几分精度"的预算曲线。

**实验矩阵**：
- 模型：Jev（钉扎版本）+ 6 个开源 Jev-style checkpoint（CLM、Laya-typed、Nimble、Von、StartLux、FunDM 头）+ 两个基线（选项独立打分器、prompt logit 读出）
- 数据：BANKING77 / CLINC150 / AG News（与已有预注册研究可比）、ProbabilityBench（有精确真值 → 可测 KL 而非代理指标）、HellaSwag/ARC 规模扫描（$|L|\in\{2,4,8,16,...,256\}$）
- 协议：预注册假设 + 每题多次重复估运行间方差 + 300k+ 调用预算（约 $10–50，成本可忽略）
- 度量：$\mathcal{S}$、IIA 违例率、acc、ECE/KL、选择性预测曲线

**为什么是 NeurIPS**：这是"choice theory（Luce/McFadden IIA）↔ 现代 LLM 决策 API ↔ 可证算法"的三重交汇，既有干净的公理，又有真实且已被独立发现的现象，还有算法与前沿刻画。**差异化**：现有 IIA 论文是纯预注册实证，无定理、无修复方法；重校准论文是单题（`noul`）的统计理论，不涉及选项列表这一结构。

**预判审稿意见**：①"只是 Plackett–Luce 的复述"→ 必须强调 $L$ 是**被序列化的列表**而非集合，故位置先验与序列化顺序（field masking 已被证实可解释全部翻转）是新变量；②"$\Theta(\log k)$ 太弱"→ 补下界证明的匹配构造与实验定标。

**8 周计划**：W1–2 形式化+识别性证明；W3 抓 300k 调用与开源 checkpoint 推理；W4 拟合 $b(i)$ 与 $a(\cdot;L)$ 并做定标律；W5 DLS 实现 + 校准；W6 Pareto 前沿与消融；W7 定理 3 的紧性数值验证；W8 写作。

---

### D2（P0）类型化决策的复合性：误差传播、联合校准与"决策图解码器"

- **假想标题**：*Correct Choices, Failed Sequences: Compositional Guarantees for Typed Decision Models*
- **一句话主张**：并行无关的接口是把双刃剑——它让单步校准良好，却让**多步决策的正确性留存塌陷**；可以形式化误差传播、给出联合校准目标，并把留存率从 2–33% 拉回来。

**现状抓手**：Choice2Seq 显示 Jev 在算术/词典检索上**选择准确率 97.1% / 99.5%，但串行构造后仅留存 33.0% / 2.1%**；同时同调用内多题互不干扰（bias < 0.001）；而 JevSpawn / StreamingJev / JevCoder / Jev-Mem 都在工程上"绕"这个问题而没有理论。

**形式化**：
- 定义**决策图** $G=(V,E)$，节点 $v$ 的选项列表 $L_v$ 依赖于父节点输出：$L_v=L_v(\hat y_{\mathrm{pa}(v)})$。
- 接口约束使一个调用内 $\{d_v\}_{v\in V'}$ 相互独立，故端到端正确率为
$$\Pr[\text{correct path}]=\prod_{\text{calls } t}\prod_{v\in V_t}\Big(1-\epsilon_v\Big)\ \ne\ \prod_v(1-\epsilon_v)\ \text{（跨调用误差相关）}$$
- **定理**：在"每节点条件校准误差 $\le \eta$"下，给出端到端风险的界 $\mathrm{Risk}(G)\le \sum_v \big(\alpha_v\eta+\beta_v\,\mathrm{TV}(L_v,\hat L_v)\big)$，其中 $\beta_v$ 由图的**宽度×深度**与选项列表对父输出的 Lipschitz 敏感度决定；并给出 $\Omega(\text{depth})$ 型下界表明串行开销不可完全消除。
- **推论（可直接检验）**：把列表按父输出做"窄化"（candidate-supported options，Choice2Seq 已观测到有效）应等价于降低 $\mathrm{TV}(L_v,\hat L_v)$，给出可预测的增益曲线。

**方法**：**Decision-Graph Decoder (DGD)**——在决策图上做前向-后向的**联合校准解码**：每个节点输出边际概率，用一小段可学习消息传递在**跨调用的时间维度**上做校准校正（温度/isotonic + 残差），并以"束搜索 + 校准置信度门控"决定何时重试。目标函数是复合风险（加权 Brier/KL）而非逐节点交叉熵。

**实验**：Choice2Seq 的算术/词典任务、合成多层决策图（可控 depth/width/$|L|$）、SWE-bench 子集上的 JevCoder（token↔成功率）、ALFWorld/AndroidWorld 的 Jev 集成（已有 4,500 次执行的对照可复用）。**必须报"每次成功交付代价"**（已有工作发现该指标可能反向恶化）。

**差异化**：现有复合性工作全是"评测+工程 trick"（JevSpawn、JevCoder、Jev-Mem、StreamingJev），无误差传播理论、无联合校准目标。

**风险**：定理容易做成"平凡连乘界"。**对策**：把技术含金量放在"列表窄化 ⇒ $\mathrm{TV}$ 下降 ⇒ 留存提升"的可预测链条上，并给出匹配实验。

---

### D3（P1）一次前向决策的可摊销极限

- **假想标题**：*What Cannot Be Decided in One Pass: Amortization Limits for Typed Decision Models*
- **一句话主张**：System One 模型的"单次前向返回分布"是有表达力上限的；构造任务族证明深度 $d$ 的决策头与 $k$ 步序列之间存在可证的分离，并给出达到上界的构造。

**形式化骨架**：任务是"从 $k$ 个候选里选对"（$k$ 由输入长度决定）。定义深度-$d$ 决策类 $\mathcal{F}_d$（一层/两层决策头 + 冻结骨干）与 $k$ 步序列过程 $\mathcal{S}_k$。目标分离：
$$\inf_{F\in\mathcal{F}_d}\mathrm{risk}(F)\ \ge\ \tfrac12-\delta,\qquad \mathrm{risk}(\mathcal{S}_k)\le \varepsilon_k\to 0 .$$
任务族候选：①**算术拒答**（有答案 99% / 无答案 7%，且原生布尔验证可达 99% → 说明表征里**有信息但无法在一次选择中兑现**，这是极佳的分离证据）；②**计数器/可达性**（在文本中嵌入图，问"从头到目标是否存在长度 $\le m$ 的路径"）；③**多跳检索**（列表窄化依赖中间结果）。

**上界侧**：给出显式的 $O(\log k)$ 深度构造（沿决策图逐步窄化候选），匹配下界。

**为什么好**：这是 CoT 表达力文献在**决策接口**上的对应物，与 D2 天然同源（可合并为一篇的大理论章或拆成两篇）。**审稿人最爱**"简单任务族 + 干净分离 + 匹配上界"。

**风险**：若分离过于人为，会被指为"玩具"。**对策**：至少一个分离族必须是真实基准（算术拒答 / SWE-bench 的可提交判定）。

---

### D4（P1）列表决策 API 的分布无关风险控制与弃权

- **假想标题**：*Risk-Controlled Decision APIs: Conformal Guarantees Under Option-Set Dependence*
- **一句话主张**：开发者需要的不是"置信度"，而是"在目标风险下必须弃权/追询"的**有限样本保证**；选项列表变化破坏可交换性，因此需要条件化的保形框架。

**问题**：官方 `confidence` 是分布的集中度函数（$\frac{p_{\max}-k^{-1}}{1-k^{-1}}$ 等），与"我的效用函数下的最优阈值"无关；实证上二值概率相对固定 0.5 阈值**落点很差**（F1 0.50→0.75 靠调阈值）；医学评测显示几乎所有模型"既认不出缺失答案、也不会用可选弃权"；算术拒答从 7% 靠 dev 集阈值救到 79%（**无保证**）。

**方法**：
1. 定义**列表条件选择风险** $R(F;\lambda)=\mathbb{E}\big[\ell(y,\hat y)\cdot\mathbb{1}\{s(p,L)>\lambda\}\big]$（$s$ 为可选性得分，可用 `confidence`/margin/熵）。
2. 由于 $L$ 随请求变化、同一实例的 $k$ 可变，得分分布**非可交换**：引入列表相似度协变量 $z=\psi(L)$，用**加权保形 / 条件保形（Mondrian by $z$ 分箱）**恢复边际与分组条件覆盖：$\Pr\big(R\le \alpha\big)\ge 1-\delta$。
3. 扩展到"拒绝集"（conformal set-valued choice）：输出一个**选项子集**，保证 $\Pr[y\in \mathcal{C}]\ge 1-\alpha$；这是对"none-of-the-above"的严格化版本。
4. 给出**列表变化下的分布漂移界**：$\mathrm{cov.\ gap}\le \mathrm{TV}(\mu_L,\mu_{L'})$，并用 D1 的敏感度泛函上界之。

**实验**：BANKING77/CLINC150/AG News（与 IIA 论文可比）、UNFAIR-ToS（有 0.50→0.75 的阈值基线）、医学 15 数据集（8,742 实例）、`noul` 二值任务。对照：固定阈值、dev 集调参、`confidence` 直接门控。

**为什么 NeurIPS**：保形预测 + 选择模型 + 新接口的交叉，且**有真实部署需求**。**差异化**：现有校准工作全是"事后修分数"，没有覆盖/风险保证，也没处理选项集导致的非可交换性。

---

### D5（P2）决策接口的变形测试套件与精确真值基准（Evaluations & Datasets）

- **假想标题**：*Metamorphic Decision Bench: Exact-Truth, Held-Out-Model, Version-Pinned Evaluation of Decision APIs*
- **必须承认的先行者**：ProbabilityBench 已经做了"精确真值条件概率 + Bayes error + 校准度量的元评测"。**因此不要重复**，改为三个它没有的维度：
  1. **变形关系（metamorphic relations）**作为一等公民：选项旋转 / 选项近似重复（paraphrase，已有观测显示只继承 1/3 概率质量）/ 选项数扫描 / 离域填充 / 问题打包 / 字段遮蔽。每条关系配套**可判定的人为真值不变式**（例如"近似重复选项之间应近似均分"）。
  2. **复合决策的真值**：多层决策图上的**联合正确性分布**（衔接 D2），而不是单节点边际。
  3. **版本钉扎 + 模型留置 + 结构化卡**：记录 provider/version/日期/调用次数/温度；held-out 模型不参与任何数据挑选；提供"随版本漂移的结论稳定性"报告。
- **交付物**：代码 + 原始响应 + 排行榜 + 元评测（用已知真值检验常用校准与 Bayes-error 估计器本身）。
- **审稿要点**：D&B track 明确要求数据集**永久可得、有文档、有伦理审查、有基线**；务必配套 `Croissant`/数据卡与长时归档。

---

### D6（P2）决策委派的代价感知策略与遗憾界

- **假想标题**：*When Local Savings Do Not Pay: Cost-Aware Delegation with Calibrated Decision Judges*
- **现状抓手**：5 个智能体、4,500 次执行显示 Jev 显著降低决策成本与延迟，但**任务成功率在移动端智能体上明显下降**，且**有一个智能体"每次成功交付的平均代价反而上升"**；JevCoder 降 token 21.6–50.6%；FunDM 显示 24/36 层可保留 99% 精度。
- **形式化**：把"哪些决策节点交给廉价 judge"建模为**带预算的序列委派（learning-to-defer + regret）**：每节点 $v$ 有廉价判断 $\hat y_v$（校准但列表偏差已知，可由 D1 建模）与昂贵生成器 $G$，代价 $c_J\ll c_G$；回报是任务完成。给出在"judge 校准误差 $\eta$、列表敏感度 $\mathcal{S}$"下的次优界，并证明忽略 $\mathcal{S}$ 的贪心策略 regret 可随 $\log k$ 增长。
- **方法**：带"进度保持 + 恢复 + 完成验证"的委派策略（对照观察到的失败模式：失败恢复差、过早完成）。
- **度量**：以**每次成功交付代价**为主指标（这是该文献群最新提出的正确指标）。

---

### D7（P3）决策头的机制可解释性

- **一句话主张**：同样的骨干 + 不同读出（字母 logit vs 决策头）为何在 OOD 下行为迥异？用因果干预定位"选项列表注意力"。
- **抓手**：RSAJEV 观测到 out-of-vocabulary 时置信度**反向**（AUROC 0.38，"silent hallucination"）；Inner Jev 证明字母 logit 读出已具备大部分能力；FunDM 证明一个 2/3 深度的标签无关头足够。这些提供了**可解剖的开源对象**。
- **风险**：NeurIPS 对"只做解释不做干预/预测"的容忍度低。**建议先投 Workshop**（NeurIPS 2027 有 workshop，或 ICML/ICLR workshop），成熟后再升级。

---

## 5. 投稿策略与时间线

### 5.1 NeurIPS 真实日历（2026 年实测，用于推演 2027）

| 环节 | NeurIPS 2026（已发生） | 推演 NeurIPS 2027 |
|---|---|---|
| 摘要截止 | 2026-05-04 AoE | ≈ 2027-05-03/04 |
| 全文截止 | 2026-05-06 AoE | ≈ 2027-05-05/07 |
| 评审放出 | 2026-07-22 | ≈ 2027-07 下旬 |
| Author-reviewer 讨论 | 2026-07-27 → 08-03 | ≈ 同期 |
| 通知 | 2026-09-24 | ≈ 2027-09 下旬 |
| 会议 | 2026-12-06/12，悉尼（+亚特兰大、巴黎卫星场） | ≈ 2027-12 |

**赛道选择**（2026 年实际开设）：Main Track、**Evaluations & Datasets**、**Position Papers**、Reproducibility、Tutorials、Competitions、Workshops、Creative AI、Journal-to-Conference。

> 注意：NeurIPS 2026 引入了 **AI Reviewing Experiment**，且 2026 年审稿流程出现过"评审重新发布"的技术事故——**写作要机器可读、主张要前置**，把核心定理与关键图表放在正文前 4 页。

**中间档期（强烈建议）**：
- **ICML 2027（约 2027-01 截稿）**：D1/D2 若在 12 月底前完成 70%，先投这里占位；
- **ICLR 2027（约 2026-09 截稿）**：本届已过，作为 2028 循环的备选；
- arXiv 预印本**立即发布**以确立优先权（该领域以"周"为单位竞速）。

### 5.2 与 NeurIPS 2026 对齐的工程检查表

- [ ] 正文 ≈ 9 页 + 参考文献/附录（以当年 CFP 为准）；双盲、模板化
- [ ] **Reproducibility Checklist** 必填；代码与原始响应公开（尽量给 DOI/长时归档）
- [ ] **Ethics / Broader Impact** 声明；如涉及智能体安全实验，说明负责任披露与负责人沟通
- [ ] **LLM 使用披露**（按其当年 policy 执行）
- [ ] 关键主张配**预注册**（该文献群已形成预注册惯例，是显著加分项）
- [ ] 所有 API 实验：**版本钉扎 + 调用次数 + 日期 + 随机性重复次数**，并提供"模型更新后重跑"的脚本

### 5.3 三种投稿组合（按你的资源选一种）

| 组合 | 内容 | 适合 |
|---|---|---|
| **保守** | D5（D&B，可复用 D1 基础设施）+ D1（Main） | 有数据工程能力、想拿"两篇" |
| **均衡（推荐）** | **D1（Main）** + 一个 Position Paper（"决策 API 的概率是列表泛函"） | 想冲主赛道、时间紧 |
| **激进** | D2 + D3 合成一篇"复合性与可摊销极限"的大理论+方法论文 | 有理论功底、能承受被拒一次 |

---

## 6. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| **赛道冷却**（Jev 专属术语在 6 个月后失焦） | 高 | 高 | 论文的**主语写成"类型化/列表条件决策模型"**，Jev 只作旗舰实例；结论对任意满足接口的模型成立 |
| **被抢发** | 高 | 中 | 立即发预印本；选"需要定理/保证"的方向（D1/D3/D4）而非"再跑一遍评测" |
| **API 版本漂移**（`1.13.0` → 后续） | 中 | 高 | 版本钉扎 + 开源替代复现 + 提供重跑脚本；结论给"稳定性区间" |
| **闭源不可复现导致拒稿** | 中 | 高 | 至少一半实验在开源 checkpoint 上完成（FunDM/CLM/Laya/Nimble/Von/StartLux 均可获取） |
| **理论过于玩具化** | 中 | 中 | 每个分离任务族必须有一个来自真实基准（算术拒答、SWE-bench 提交判定） |
| **伦理/安全敏感**（攻击类实验） | 中 | 中 | 走负责任披露、不做可直接复用的攻击配方、NeurIPS ethics 审查提前自检 |

---

## 7. 资源清单

<details>
<summary><b>展开：数据、代码、接口与预算</b></summary>

**数据/基准（可直接用）**
- `BANKING77`、`CLINC150`、`AG News`（选项集依赖的标准三件套）
- `ProbabilityBench`（精确真值条件概率，可测 KL 而非代理标）
- `Choice2Seq`、`Jev-ToM`（HKUST-KnowComp 的代码与评测材料公开）、`DecisionRobust`（yifan-lan.github.io/DecisionRobust）、`OpenDecisions`（2,352 题）
- `JevBench` v1.7（社区维护，注意版本）、`EmbodiedSkillBench`（LIBERO-Pro）
- `UNFAIR-ToS`（阈值/校准基线）、15 个医学数据集（8,742 实例，来自医学评测论文）

**开源模型/代码（复现对象）**
- FunDM：`github.com/Ujjawal-K-Panchal/FunDM`（冻结 LM + 标签无关决策头）
- Inner Jev / OpenDecisions 配套模型与服务代码
- JevProbe / JevProbeBench：`github.com/xiaoxiaohg8d/JevProbe-`
- Jev 中毒实验：`github.com/WuXixiong/jev-poisoning-bench`
- 官方基线与原始响应：`github.com/AppliedMachineLearning-Lab/jev-benchmarking`，DOI `10.5281/zenodo.23039006`
- 社区文献图谱：`github.com/Oscar-dzy/Awesome-jev-papers`（174 条，含交互式 atlas）、`github.com/OmniJev/awesome-jev-papers`

**接口文档**：`docs.typesafe.ai/primitives`、`docs.typesafe.ai/models`、`docs.typesafe.ai/concepts/system-one`

**预算量级**：官方定价约 **$0.042 / 1M input tokens**；37 数据集 × 346k 请求的实验花费 **< $10**。→ **D1 的 300k 调用实验成本在数十美元量级**，真正的成本是开源 checkpoint 的推理算力（FunDM 只需 24 层的骨干，单卡 A100/H100 即可）。

</details>

<details>
<summary><b>展开：核心文献速查表（按方向对齐到 D1–D7）</b></summary>

| arXiv | 对应方向 | 关键数字 |
|---|---|---|
| `2610.16655` 选项集依赖（预注册） | D1 | RMS 位移 0.20–0.52 nats；255 选项 ≈ 1 nat；独立打分消除依赖但 −8.7 点 |
| `2610.18488` 多问题重校准（69 页） | D1/D4 | 极小极大复合风险 $(mn)^{-2/3}+\min\{\tau^2,1/n\}+\min\{\tau^2,V^2,(V/n)^{2/3}\}$ |
| `2610.17016` ProbabilityBench | D1/D5 | 精确真值 + Bayes error；16 模型中前沿多为决策模型 |
| `2610.16191` Choice2Seq | D2/D3 | 选择 97.1%/99.5% → 串行留存 33.0%/2.1% |
| `2609.39496` 算术拒答瓶颈 | D3/D4 | 有答案 99% / 正确拒答 7%；阈值救到 79% |
| `2610.17205` Inner Jev + OpenDecisions | D5/D7 | 未微调即 79.6 vs Jev 80.1；微调后 84.4 / JevBench 81.5 |
| `2610.17175` FunDM | D3/D6 | 24/36 层保留 99% 精度；吞吐 1.56×，显存 −1/3 |
| `2610.19619` 认知/社会偏置 | D1 | 41,500 响应；合取类任务 54% 违背；选项顺序敏感 |
| `2610.19118` DecisionRobust | D5 | 30/17 组配置；24/12 留置模型 |
| `2610.18612` JevProbe/JPDH | D6（威胁模型） | 228 应用 / 2,273 用例，81.5% 决策翻转，98.6% 有下游后果 |
| `2610.15537` 上下文投毒 | D6 | 执行感知协议；三分预设对比经 Holm 校正后不显著 |
| `2610.17820` Same Attack, Different Judge | D5 | 规则型任务最强、整体判断型最弱（钓鱼 62% vs 77–81%） |
| `2610.18389` PRISM | D1（校准） | 反事实探针 + 残差校正提升与人类排序一致性 |
| `2610.19596` JevCoder | D6 | SWE-bench 上 token −21.6–50.6% |
| `2610.13092` 智能体工作流 | D2/D6 | 4,500 次执行；移动端成功率下降；**单次成功代价可能上升** |
| `2609.37647` 37 数据集零样本 | 基线 | 346k 请求 < $10；Belebele 86.7%；阈值使 F1 0.50→0.75 |
| `2609.23986` Jev-Mem / `2610.18326` Jev-Guard / `2610.18780` BLADE / `2610.16967` ProAgent / `2610.00437` JevSpawn | D2（系统） | 记忆平面、58 ms 路由、失败归因、主动式干预、组合动作空间 |

</details>

---

## 8. 下一步（未来 14 天）

1. **W1**：把 D1 的公理系统与识别性写成一页 LaTeX，确认与 Plackett–Luce / McFadden / Luce 公理的差异（这是审稿人第一问）。
2. **W1–2**：钉扎 `jev-1.13.0` + 拉取 3 个开源 checkpoint，跑通 $|L|\in\{2,\dots,256\}$ 的敏感度估计，复现 0.20–0.52 nats 与 log 增长。
3. **W2**：同时起草 D5 的数据卡与变形关系清单（这部分与 D1 共享 80% 基础设施，边际成本最低）。
4. **决策门**：若 W2 末能给出"敏感度随 $\log k$ 线性且截距可跨模型迁移"的证据 → 全力 D1；否则转 D4（风险控制，理论路径更短）。

> **一句话总结**：这个生态目前 90% 的论文在回答"Jev 表现如何"，**没有人回答"这个接口的概率在数学上是什么、能保证什么"**。把论文的主语从 Jev 换成"列表条件的类型化决策模型"，你就同时获得了新颖性、可证性与抗过时性——这正是 NeurIPS 主赛道要的东西。
