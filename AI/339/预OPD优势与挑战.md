# 预OPD优势与挑战

## 🧑‍💻 User

Pre-RL OPD为什么现在正在被开始尝试，并且似乎获得了较正面的社区评价？

---
## 😀 默认助手 (7)

I'll look for tools that can help me research this topic.

I found web browsing/fetch tools. Let me research this topic. I'll search the web.

OPD = **On-Policy Distillation**. Let me dig into the key papers and community discussions.

Good — I'm finding the key papers. Let me get the full details and community discussion.

Found the key paper: **"RL Starts before RL: On Policy Distillation for Better..."** (2609.28145) — exactly the "Pre-RL OPD" concept. Let me get details and community takes.

Let me get the full paper details and more community reaction.

Let me gather the remaining community/review perspectives.

Let me pull a couple more sources for the "why it works" mechanism and industry framing.

下面是我通过检索 arXiv、技术博客和中文社区整理出的结论。先给一个一句话判断，再展开。

> **一句话结论：** "Pre-RL OPD" 指的是**把 On-Policy Distillation（在线策略蒸馏）当作 RL 之前的"预备/冷启动"阶段**（即 OPD-then-RL）。它现在火起来，是因为 OPD 本身已从"RL 之后的收尾手段"升级为后训练标配，而 2026 年一批论文系统性地证明"先用 OPD 把初始策略调好、再跑 RL"比"直接 RL""SFT→RL""OPD+RL 混合"都更稳、更强、更便宜；它社区口碑好，则是因为这是一个**又简单、又便宜、又有机制解释、又踩中行业痛点**的方案。

<details>
<summary><b>先厘清术语：OPD 与 "Pre-RL OPD" 到底是什么</b></summary>

**OPD（On-Policy Distillation，在线策略蒸馏）** 的核心是：学生模型**自己采样轨迹**，教师模型在**学生走过的每个 token 上**给出密集信号。它把 RL 的"on-policy 状态覆盖"和蒸馏的"逐 token 密集监督"结合了起来。

逐 token 的 advantage（伪代码里就是一行改动）为：

$$ \hat{A}_t = \mathrm{sg}\!\left[\log \frac{\pi_{\text{teacher}}(y_t \mid x, y_{<t})}{\pi_{\text{student}}(y_t \mid x, y_{<t})}\right] $$

然后代入策略梯度损失（$\mathrm{sg}[\cdot]$ 为 stop-gradient）：

$$ \mathcal{L}(\theta) = -\,\mathbb{E}_{x,\,y\sim\pi_\theta}\!\left[\frac{1}{|y|}\sum_{t}\hat{A}_t \log \pi_\theta(y_t \mid x, y_{<t})\right] $$

这就是 Thinking Machines Lab 那篇博客说的 "a one-line change"（在 GRPO 里把组内归一化 reward 换成 teacher-student log ratio）。

**"Pre-RL OPD" 的定位**：传统上 OPD 多用在 **RL 之后**（GLM-5 用它修复多阶段 RL 造成的灾难性遗忘，DeepSeek-V4 用它替代混合 RL）。而"Pre-RL OPD"是把它**前移**——在学生进入 RL 之前，先做一段 OPD 来塑造更好的初始策略，然后再跑 RLVR。对应论文里的名字有 `OPD-then-RL`、`OPD as preparation for RL`、`OPD as cold start for RL`。
</details>

---

## 一、为什么"现在"才开始被尝试

### 1. 时代背景：OPD 已从"技巧"变成"标配"
过去一年，**Qwen3、GLM-5、MiMo-V2、DeepSeek-V4** 相继在技术报告里把 OPD 写进后训练主流程，Thinking Machines Lab 的博客（2025-10）又用"50–100× 算力效率"把它推到了社区聚光灯下。同时 **verl、ROLL、tinker-cookbook** 等主流框架都内置了 OPD / on-policy distill pipeline。也就是说：**做 Pre-RL OPD 所需的一切基础设施和心智模型都已经就位**，这是它能"被开始尝试"的前提。

### 2. 纯 RL 有个绕不开的"初始策略依赖"
RLVR 的本质是**在策略已有支撑集上做"锐化"**：它只能放大模型本来就能采样到的高质量路径，很难凭空拓宽覆盖面。所以：

- 如果 base 策略对正确推理路径的覆盖（pass@k）太低，RL 会大量浪费算力、容易 plateau；
- 最终性能强烈依赖"从哪个策略开始训"。

这就把问题从"怎么把 RL 调得更好"转移到了"**怎么在 RL 之前把初始策略调好**"——而这正是 Pre-RL OPD 要回答的。

### 3. 旧的 RL 冷启动方式（SFT→RL）出了名的问题
行业标准流程一直是 **SFT 冷启动 → RL**。但 SFT/off-policy 蒸馏有两个根本缺陷：

- **分布错位（exposure bias）**：训练时看的是教师路径，推理时走的是自己的路径；
- **Confident Conflict**：教师输出里学生认为概率极低的 token 被强行拟合，梯度很暴力，是**灾难性遗忘**的主要来源之一。

OPD 天然避开这两点（reverse-KL 是 mode-seeking、且只在学生自己采样的 token 上算信号）。于是"**用 OPD 替代 SFT 来做 RL 的冷启动**"就成了一个极其自然的想法。

### 4. 2026 年的关键实证结果把这条路"坐实"了
几篇论文几乎同时给出了正面结论（详见下节），把"OPD 放在 RL 前还是后/还是混合"这个问题第一次系统性地讲清楚了。

---

## 二、为什么社区评价偏正面

<details>
<summary><b>1）核心实证：Sequential Beats Joint（arXiv:2609.04108）——"先 OPD 后 RL"稳赢</b></summary>

这篇论文直接对比了三类做法：

| 方案 | 结构 |
|---|---|
| 纯 OPD | 只用蒸馏 |
| 纯 RLVR | 只用可验证奖励 |
| **联合优化**（weighted-additive / teacher-modulated rescaling） | 把 OPD 的密集信号与 RL 稀疏 reward 融进同一步 |
| **OPD-then-RL（顺序）** | 先 OPD，再 RL |

结论：**两阶段的 OPD-then-RL 在逻辑与数学推理基准上一致优于纯 OPD、纯 RLVR 以及所有"联合"基线。**

机制解释（这是它说服力强的原因）：

- **OPD 拓展覆盖**（expands coverage of teacher-supported solutions）；
- **RL 在支撑集内锐化**（sharpens within that support）；
- 而**联合优化会让两个信号相互干扰**——参数更新层面出现 **sign conflict**（梯度方向冲突）。

还给出两条实用配方：
1. **用 OPD 的 validation score 作为"何时切到 RL"的开关信号**；
2. **OPD 是比 SFT 更好的 RL 冷启动**。
</details>

<details>
<summary><b>2）机制深挖：RL Starts before RL（arXiv:2609.28145）——收益不只是"初始准确率更高"</b></summary>

这篇（29 位作者，直接把 OPD 定义为 RL 的 preparation stage）发现：

- 在**相同 RL 设置**下，用 OPD 初始化的学生，最终性能高于 **直接 RL** 和 **SFT→RL**；
- **关键在于：即使 OPD 没带来多少即时的准确率提升，这种优势依然出现**；
- **Pre-RL 的 pass@k 并不能完全解释收益**——相似甚至更高的 pass@k 不一定带来更好的 RL 结果；
- 真正起作用的是**"超出 top-1 一致性的、与教师分布的对齐"**：保留了对 RL 后续有用的**替代解与不确定性**，而非过早坍缩到单一答案。

还给出一个很实用的结论：**reverse-KL OPD 在 RL 之前更好，forward-KL OPD 在 RL 之后反超**；且最优目标取决于轨迹来源与后续训练。
</details>

<details>
<summary><b>3）为什么"OPD 做冷启动"胜过 SFT：分布几何角度</b></summary>

- **SFT ≈ forward-KL（mean-covering）**：被迫覆盖教师的所有输出，容易熵崩塌、遗忘；
- **OPD/RL ≈ reverse-KL（mode-seeking）**：只在自己高置信的路径上向教师靠拢，**不产生 Confident Conflict**，**保留不确定性**——这让后续 RL 仍有探索与细化的空间。

配套的机理研究也支持这一点：
- **Rethinking OPD（arXiv:2604.13016，清华）**：给出 OPD 成败的两大前提（**思维模式一致** + **引入新知识**）与 token 级机制（高概率 token 的**渐进式对齐**、**Overlap Sufficiency**），并指出失败的补救方式是"off-policy 冷启动 / teacher-aligned prompts"；
- **On the Geometry of OPD（arXiv:2606.07082）**：OPD 在参数空间走出一条**独立于 SFT 与 RLVR 的更新几何**（relaxed off-principal + subspace locking），说明它不是两者之间的简单插值。
</details>

### 4. 还有几个"加分项"
- **便宜**：OPD 比 RL 便宜得多（Thinking Machines 声称 50–100× 算力效率；相同数据下训练速度快 8–16×），把它当前置阶段**几乎不增加总成本**；
- **简单**："boring two-stage recipe"，工程上就是一行改动级别的接入；
- **踩中行业趋势**：四大厂已用 OPD，"把它放到 RL 前"是这条主线上最自然的下一步；
- **主动传播**：中文社区（知乎、plmblog、yaoyuanzhou 的《OPD 完全指南》、zhichai 深度长文、papers.cool 等）密集解读，形成了"事实新范式"的叙事，进一步推高口碑。

---

## 三、值得注意的保留意见（并非没有反方）

<details>
<summary><b>OPD 本身很脆弱，前置也不是万能的</b></summary>

- **OPD 极易失败**：教师"能力强"不等于"可蒸馏"，思维模式差异过大时，更高分的教师也可能毫无增益甚至倒退；
- **长序列退化**：密集奖励在长轨迹后段（约 10K–15K token）质量崩塌，会反向污染优化；
- **方向尚未定论**：reverse vs forward KL、教师选取（同架构 / 多专家 / 跨尺度）、轨迹来源，都还依赖具体设置；
- **与"OPD after RL"存在张力**：GLM-5/DeepSeek-V4 更多把 OPD 当收尾整合，而 Pre-RL 的用法是把它当起手式，两者孰优仍需更多工作；
- 部分"Pre-RL"结论目前主要集中在**数学/逻辑推理基准**，能否推广到代码、Agent、开放式对话仍是开放问题。
</details>

---

## 四、小结

| 问题 | 简短回答 |
|---|---|
| **它是什么** | OPD 作为 RL 之前的冷启动/预备阶段（OPD-then-RL） |
| **为什么现在** | OPD 生态成熟（四大厂 + 框架）；RL 对初始策略敏感；SFT 冷启动有缺陷；2026 年多篇论文把"顺序优于联合"证明清楚 |
| **为什么口碑好** | 简单、便宜、有机制解释（覆盖拓展 + 锐化 / 无 confident conflict / 保留不确定性）、且踩中"RL 效率与冷启动"这一核心痛点 |
| **主要风险** | OPD 脆弱、长序列退化、KL 方向与教师选择未定、结论偏数学推理域 |

**关联论文速查：**
- `arXiv:2609.04108` Sequential Beats Joint（OPD-then-RL 优于联合）
- `arXiv:2609.28145` RL Starts before RL（OPD 作为 RL 的准备阶段）
- `arXiv:2604.13016` Rethinking OPD（成败条件与机制，THUNLP）
- `arXiv:2606.07082` On the Geometry of OPD（参数空间几何）
- `arXiv:2607.05394` Direct-OPD（弱到强泛化，另一种前置迁移思路）

如果你想，我可以进一步把这几篇的**方法细节 / 实验设置 / 公式推导**逐篇展开成对照表。

---
## 🧑‍💻 User

Pre-RL OPD相比于Post-RL OPD有什么优缺点？为什么？
请详细的阅读数篇论文，尝试抽出数学结构，再看看是否真的能解释现象，作为回答。

---
## 😀 默认助手 (7)

This is a deep research task. Let me actually pull the full texts of the key papers so I can extract their mathematical structure rather than rely on summaries.

Let me extract just the relevant mathematical/mechanistic passages from each paper.

以下基于我实际读取的四篇论文原文（`arXiv:2609.04108`、`2609.28145`、`2604.13016`、`2606.07082`）来回答。我先抽出它们共享的数学骨架，再用这套骨架去比较 **Pre-RL OPD（OPD→RL，当冷启动）** 与 **Post-RL OPD（RL→OPD，当收尾/巩固）**，最后诚实评估这套数学"到底解释了多少"。

---

## 0. 先把两种顺序写成同一套"算子"语言

把 RL 和 OPD 都看成参数空间上的梯度流算子：

- $\mathcal{R}$：沿 $\mathcal{J}_{\text{GRPO}}$ 上升（锐化到 outcome reward）；
- $\mathcal{D}$：沿 $\mathcal{J}_{\text{OPD}}$ 上升（向 teacher 分布对齐）。

于是两种安排是**算子顺序不同**：

$$ \theta_0 \xrightarrow{\ \mathcal{D}\ } \theta_1 \xrightarrow{\ \mathcal{R}\ } \theta_2 \quad(\text{Pre-RL OPD}) \qquad\text{vs.}\qquad \theta_0 \xrightarrow{\ \mathcal{R}\ } \theta_1' \xrightarrow{\ \mathcal{D}\ } \theta_2' \quad(\text{Post-RL OPD}) $$

**两者都只在 student 自己采样的轨迹上算梯度**，因此真正的差异只能来自三件事：RL 的梯度在哪里非零、OPD 的信号在哪里有效、两个算子在参数空间是否打架。下面逐一抽取。

<details>
<summary><b>数学结构一：RL 是"支撑集受限的锐化算子"</b>（顺序的因果方向来自这里）</summary>

论文把两者统一为**逐 token 策略梯度**：

$$ \mathcal{J}(\theta)=\mathbb{E}_{x,\,y\sim\pi_\theta}\Big[\tfrac{1}{|y|}\sum_t A_t\,\log\pi_\theta(y_t\mid h_t)\Big] $$

GRPO 的 advantage 在**整条 rollout 上恒定**，且是组内归一化的：

$$ \hat A^{(i)}=\frac{R^{(i)}-\mathrm{mean}(\{R^{(j)}\})}{\mathrm{std}(\{R^{(j)}\})} $$

关键推论：当一组 rollout **全对或全错**时（$\mathcal{C}(x)=\mathbb{1}[\sum_j R^{(j)}=0]$），$\hat A\equiv 0$，梯度整体消失。

$\Rightarrow$ **RL 只能在"初始策略已经能采样到正负混合"的 prompt 上工作**，它是"在已有支撑集内重新分配概率质量"的算子，**不能凭空创造正确行为**。这直接决定了顺序的因果方向：**coverage 必须先建立，否则 RL 在难题上没有梯度。**

反向地，OPD 的 advantage 是逐 token 的 log-ratio：

$$ d_t\triangleq \log\pi_T(y_t\mid h_t)-\log\pi_\theta(y_t\mid h_t),\qquad \mathcal{J}_{\text{OPD}}=\mathbb{E}_{y\sim\pi_\theta}\Big[\sum_t d_t\Big] $$

它把 $\mathcal{L}_{\text{OPD}}=\mathrm{KL}(\pi_\theta\Vert\pi_T)$ 通过自回归链式法则拆成逐 token 求和——这就是 OPD 的"密集信号"。
</details>

<details>
<summary><b>数学结构二：OPD 的信号是 mode-seeking，且只在"重叠高概率 token"上有效</b></summary>

- **reverse-KL 是 mode-seeking**：$d_t$ 只在 student 真正发出的 token 上拉概率，因此它倾向**覆盖 teacher 的高概率核心**、而**欠覆盖 teacher 的低概率长尾**。论文 1 的实测正是如此：训练中 student 覆盖的 teacher top-K token 变少（overlap ratio 下降），但保留的那些 token 概率更大——"集中在更小但仍是 teacher-supported 的核内"。
- 论文 3 的 token 级机制：成功 OPD 的标志是**高概率 token 上的渐进对齐**——overlap ratio 从 72%→91%，student–teacher 的 entropy gap 收窄，且共享 top-$k$ token 承载 **97%–99%** 的概率质量；**只监督 overlap token 就能复现完整 top-$k$ 的效果**。这说明 OPD 的有效梯度其实落在一个很稀疏的子集上。
- OPD 成立的两个条件（论文 3）：(i) student/teacher **思维模式相容**；(ii) teacher 必须带来 student **没见过的新知识**——"更强的 teacher 也可能完全失败"。

</details>

<details>
<summary><b>数学结构三：两个算子占据部分冲突的参数子空间</b></summary>

- **sign conflict（论文 1）**：定义 $\Delta\theta^m=\theta^m-\theta^{\text{base}}$，取按 $|\Delta\theta^{\text{OPD}}|$ 排序的 top-$K\%$ 参数，统计符号冲突率 SCR。结果（K&K / Countdown，10% 档）：RL 20.36 / 9.57，KDRL 7.73 / 6.79，TRRD 0.04 / 1.31，**OPD-then-RL 0.00 / 0.02**。
- **干预实验**：把 KDRL 中冲突参数的一部分（比例 $\alpha$）替换成 OPD 的更新方向，会**单调提升 pass@$k$、同时降低 pass@1**；随机替换同样数量则不出现该 trade-off。$\Rightarrow$ 冲突子空间正是"锐化 vs 扩展"的战场。
- **subspace locking（论文 4）**：OPD 累积更新很快进入一个**低维窄通道**（relaxed off-principal regime，介于 SFT 的 dense/principal-aligned 与 RLVR 的 sparse/off-principal 之间）。把梯度投影到早期 $V_{16}$ 子空间：**OPD 几乎无损，SFT 明显退化**。该 lock 对 token 稀疏化、off-policy rollout 鲁棒，**但对"把 OPD 与 RLVR 混合"敏感**——从参数几何上再次佐证"顺序优于联合"。

</details>

---

## 1. Pre-RL OPD 的优缺点

| 维度 | 机制（来自哪条数学） |
|---|---|
| **优点** | |
| 扩大 RL 的可达集 | 结构一：OPD 抬高 pass@$k$（大数据集上尤为明显，hard OOD 上仍继承 OPD 的高 pass@128），等于先把 RL 的"$A_t\neq0$ 区域"撑大；RL 再在其中把 pass@1 拉起来 |
| 保留可塑性 | 结构二 + 论文 2：OPD 后的策略**在大 $k$ 仍保留替代解与不确定性**，给后续 RL 留出 reweight 空间 |
| 保持更新方向、避免冲突 | 结构三：sequential 使 OPD 的更新结构基本完好（SCR≈0），不牺牲 capability expansion |
| 比 SFT 更好的冷启动 | 论文 1：OPD 冷启动在多数据集上优于 SFT，且**过完 RL 后差距拉大** |
| **缺点** | |
| 条件脆弱 | 结构二：若 base 与 teacher 思维模式不相容，前置 OPD 直接失败，还会**污染 RL 的起点** |
| 长视界下信号崩塌 | 论文 3：逐 token 奖励质量随轨迹深度退化，不稳定从后段 token 起向后传播；对长链推理/agent 不友好 |
| reverse-KL 欠覆盖长尾 | 结构二：mode-seeking 只覆盖 teacher 高概率核；若正确解恰在 teacher 低概率区，前置反而收窄 |
| 只对齐行为代理 | $\mathcal{J}_{\text{OPD}}$ 模仿 teacher $\neq$ 最大化任务目标，teacher 的偏差会被 RL 继承 |
| 额外成本 | 多一个阶段 + 需要选择切换点（论文 1：切换处的 OPD validation score 基本决定 post-RL 精度） |

---

## 2. Post-RL OPD 的优缺点

（这里要坦白：这四篇主要是关于顺序/联合的，**没有**给出 RL-first vs OPD-first 的头对头实验——论文 1 明确写"teacher-first 与 student-first 在同等算力下的受控比较留待未来工作"。所以下面对 Post-RL 的判断部分是**由上面的数学外推**。）

| 维度 | 机制 |
|---|---|
| **优点** | |
| 修复 RL 副作用 | 论文 1 的分解 $\mathrm{KL}(p_S\Vert p_T)=H(p_S,p_T)-H(p_S)$：RL 阶段 $H(p_S)$ 下降快于 cross-entropy，说明 RL 在**teacher 支撑集内**锐化、熵塌缩。后置 OPD 正好重建熵 / 重新对齐 teacher support，缓解遗忘（GLM-5、DeepSeek-V4 的用法） |
| 巩固 RL 收益 | 后置时正确路径更密集，OPD 相当于把 RL 已找到的高奖励行为"压回"一个更平滑、更接近 teacher 的分布 |
| 不引入新能力、风险低 | 因为 RL 已把边界定死，后置 OPD 只在支撑集内做重分配，不易造成剧烈漂移 |
| **缺点** | |
| 无法回溯扩展覆盖 | 结构一：RL 不能创造新能力 $\Rightarrow$ 后置 OPD 也**只在 RL 已锐化的支撑集内**做对齐；Post-RL 根本不会发生"capability boundary 扩张" |
| 改进空间小 | 结构二：RL 后分布已经很尖，$d_t$ 只在尾部 token 显著，密集信号"无处施展" |
| 与已锁定的 RL 子空间可能冲突 | 结构三：论文 4 显示 lock 对"目标混合"敏感；RL 先建好 off-principal 结构，随后的 OPD 目标与之未必同向 |
| 救不了 RL 的冷启动失败 | 结构一：若 RL 一开始因 pass@$k$ 太低而梯度消失/plateau，后置 OPD 无力回天 |

---

## 3. 这套数学"真的解释"了现象吗？——分层评估

**解释得相当好的部分（可判可证）：**

1. **顺序的因果方向**：结构一（支撑集约束）+ 结构三（sign conflict / subspace lock）合起来给出了"为什么 sequential > joint"的**机制性、带干预证据**的解释。SCR 对照实验（$\alpha$ 扫描 + 随机替换对照）从相关升级到近似因果；rank-16 投影对 OPD/SFT 的差异也是可复现的结构证据。这属于**真解释**。
2. **"RL 没有离开 teacher 支撑集"**：$\mathrm{KL}=H(\text{cross-entropy})-H(\text{entropy})$ 的分解干净地解释了"KL 上升但仍在支持集内"这一反直觉现象。这是**真解释**。

**只是"现象学/假设"，还没被解释的部分：**

1. ✗ **论文 2 最核心的发现**——"OPD 即使几乎不提分，也能让 post-RL 更好；pre-RL pass@$k$ 不解释收益"——**结构一/二都解释不了**。他们把原因归给"超越 top-1 的分布对齐 + 保留不确定性"，但 token uncertainty 只是**代理变量**，没有形式化的量（某个势函数）把"保留替代解"与"post-RL 增益"连起来。这是全篇最需要理论化的地方。
2. ✗ **pass@$k$ 不是充分预测子**：论文 1/2 都观察到更高 pre-RL pass@$k$ 不一定带来更好 post-RL，数学上目前**没有一个正确的"起点好坏"泛函**。
3. ✗ **reverse-KL vs forward-KL 的翻转**（前置用 reverse-KL 更好、后置 forward-KL 反超；换成 teacher 轨迹又变回 reverse-KL 领先）：mode-seeking / mode-covering 的直觉只给了方向，**给不出 crossover 条件**。
4. ✗ **长视界崩塌**：论文 3 说"失败不在信号质量而在局部优化几何（landscape 变平）"，但这是**断言**而非推导，没有随深度变化的奖励质量模型。
5. ✗ **Pre-RL vs Post-RL 本身**：没有同等算力的头对头实验，所谓"Post-RL 改进空间小"更多是从 entropy 塌缩外推的。

---

## 4. 小结

- **一句话**：从数学上看，顺序之所以重要，是因为 $\mathcal{R}$ 是**支撑集受限的锐化算子**、$\mathcal{D}$ 是**只在重叠高概率 token 上有效的 mode-seeking 密集信号**，而两者又占据**部分 sign-conflict 的参数子空间**。这三点共同推出"**先扩展覆盖、再锐化**"（Pre-RL OPD）在结构上优于"先锐化、后对齐"（Post-RL OPD）——后者**无法回溯扩展能力边界**，信号作用空间也更小，但它的价值在于**修复 RL 的熵塌缩与遗忘**，属于"巩固"而非"赋能"。
- **但**："OPD 不涨分也能帮 RL"这一最关键的实证，目前只有假设级解释；KL 目标翻转、长视界退化、以及 Pre/Post 的真实差距都还没有被数学**闭环**解释。所以更准确的表述是：**这套数学解释了"为什么顺序优于混合"，部分解释了"为什么覆盖要先于锐化"，但尚未解释"为什么起点分布的这种细节对 RL 如此关键"。**

如果你想，我可以把论文 1 里两种"联合法"（weighted-additive 与 teacher-modulated）的**逐 token advantage 公式**也完整抄出来，做成一张"Pre / Joint / Post 三者的目标函数与参数几何对照表"。
