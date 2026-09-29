# SFT遮羞布现象

## 🧑‍💻 User

我是大厂被裁mllm校招后训练工程师
越来越觉得SFT是RL的遮羞布，甚至可能是以后opd的遮羞布
最近在做agentic rl的项目，好几次RL没有提上去，重新筛了很多轮RL数据，调了rollout和temperature以及reward，效果没有增长
就尝试rollback洗了一下sft数据，然后再基于SFT的数据再次rl，结果rl的提升显现出来了

导火索就是之前leader强行要grpo相较于sft取得明显提升，我换用ppo被他骂了，我尝试搞grpo的数据以及换reward没有成功
最后整理sft数据后，再次进行grpo，终于取得明显提升了，

这是怎么回事？

感觉以后sft做得好的话，甚至不需要rl以及opd了，就是可能就是纯数据工程...
现在很多大厂也在蒸馏opus的cot用于sft，我们之前rl的reward有一次还直接拿gemini去做

---
## 🤖 Assistant

# 先给结论

你的观察不是错觉，而且它正在成为 post-training 的主流共识方向。准确的说法是：

**RL 基本不创造能力，它只是在 SFT 给定的输出分布上做重加权/锐化。SFT 决定能力上限、以及 reward landscape 的几何形状；RL 决定你最终落到哪个 mode 上。**

所以"洗了 SFT 数据后 RL 才涨"的本质是：你把 policy 从 **能力盆地** 里挪到了 RL 真正能起作用的位置。你没换对算法，你换对了初始化。

---

## 1. 机制：为什么脏 SFT 会让 RL「提不上去」

### (a) RL 啃不动 support 外的行为

policy gradient 的期望是:

$$\nabla_\theta J = \mathbb{E}_{o \sim \pi_\theta}\left[\hat{A}_i \,\nabla_\theta \log \pi_\theta(o_i)\right]$$

**只有模型已经采样得到、概率非零的轨迹才能被加权。** 脏 SFT = 往高概率区塞了一堆垃圾模态，好轨迹被稀释、坏轨迹被抬高，梯度方向被污染。你在 support 外想要的行为，采样都采不到，更别说加权。

### (b) GRPO 的组内方差：RL 只在「踮脚够得着」的区间有效

GRPO 的优势是组内相对值：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}$$

- 一组 rollout **全失败**（脏/弱 SFT）$\Rightarrow \sigma_r \to 0 \Rightarrow$ 梯度 ≈ 0，白跑。
- 一组 rollout **全成功** $\Rightarrow$ 同样无梯度。

RL 真正有效的是 **edge-of-competence** 区间——"有时对、有时错"，这时组内才有承载学习信号的方差。**洗 SFT 的本质，就是把模型抬到这条边缘线上。** 你之前那版 SFT，大概率整体落在"below competence"的死区。

### (c) reward 的有效性 / reward hacking

你的 reward 是 Gemini / 蒸馏 CoT 这类 **proxy 强模型**。RL 会把 reward 的偏差放大 10 倍。

- 脏 SFT 模型本身已经站在"高 reward 的坏解"附近 $\Rightarrow$ 一 RL 就 reward hack，指标虚高、无实质。
- 干净 SFT 离 reward 的真实最优更"正"，hack 空间更小。

这解释了为什么你"**换 reward 练不上去**"——问题不在 reward 选谁，而在 policy 起点已经陷在坏解的吸引域里。

### (d) KL 锚点 = basin

PPO/GRPO 都有 $\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$，把探索锚在 SFT policy 上。

$$\text{坏锚点} \Rightarrow \text{坏 basin} \Rightarrow \text{优化器再努力也只是在坏盆里爬山}$$

---

## 2. 为什么你「换 PPO / 换 reward / 加 RL 数据 / 调 temperature」全都没用

因为它们都在改**下游**（优化器、reward 函数、prompt 分布、采样温度），而瓶颈在**上游**（初始 policy 的 support 与 competence）。

> 在 support 里做再多搜索，也搜不出不在 support 里的东西。

只有动 SFT 才能 break through——这恰好就是你的实验结论。**PPO vs GRPO 是次要变量**（两者都在同一个 basin 里）。leader 逼你用 GRPO 有他工程上的理由（无 value model、便宜、天然适配 verifiable reward），但把算法选择当瓶颈，是一次误诊。

---

## 3. 「SFT 是 RL 的遮羞布」——这个判断对几分

<details>
<summary><b>对的部分（点击展开）</b></summary>

- RLHF/RLVR 在数学上 ≈ **on-policy rejection sampling + reweighting + 对齐**。变分/Bayesian 视角下，`RL with KL` 就是从 reward-tilted 后验里采样。
- 有工作显示 RL 只微调了模型的一小块 subnetwork，是"轻touch 的编辑"，重活是 SFT 干的。
- pass@k 分析显示 RLVR 主要抬 **pass@1**、同时**压低 diversity**——是重加权而非真正教出新推理。
- 你说的"纯数据工程"正是 **data-centric post-training** 的真实趋势：prompt 分布设计、数据 curation、verification、CoT rewrite、rejection sampling 迭代。

</details>

<details>
<summary><b>不完全对 / RL 仍不可替代的部分（点击展开）</b></summary>

- **可验证的稀疏奖励**（代码测试、数学答案、tool-call 成功）：没有现成 teacher 能给你完美轨迹，RL 让你**超越 teacher 的数据覆盖**。
- **探索到 SFT 分布之外**：base 有 latent 能力 + 好的 exploration 时，RL 能到达 teacher 没展示过的解。
- **廉价 scalar reward 的行为塑造**：对齐、格式、拒答。

但这些"不可替代"正在被 **OPD（on-policy distillation）** 侵蚀——见下。

</details>

---

## 4. OPD 和 RL 其实是一家人

| | 采样 | 监督信号 |
|---|---|---|
| RL (GRPO/PPO) | on-policy | scalar reward |
| OPD | on-policy | teacher 的 dense token-level logits |
| Rejection-SFT | on-policy 生成 → 离线 | hard filter |

**三者本质相同：on-policy 生成 + 朝"好"的方向重加权。** 区别只在监督的 **density 和 bias**。OPD = reward 更 dense、bias 更小的 RL。

所以"以后是纯数据工程"更精确的表述是：

> **从 scalar RL 走向 dense / distilled RL，而 OPD 就是这个中间态。**

你平pipeline 已经暴露了这一点：SFT 蒸馏 Opus 的 CoT，reward 又拿 Gemini 当 judge——**整条链其实就是"同一个/更强的 teacher 换着法子蒸馏"**。从这个角度看，,"SFT 是遮羞布"这个说法非常犀利，只是它同样适用于 RL 和 OPD。

---

## 5. 对个人的判断

- **杠杆在数据侧**：prompt 分布设计、可验证性、数据筛选、**初始 SFT 质量**。
- **"调算法（PPO/GRPO/温度/rollout）"往往是低杠杆；"调初始化"是高杠杆。** 你踩到的就是这个。
- 但也别走向"SFT 万能"的极端。真正可稳定放大的是：**可验证奖励 + 好的初始化 + on-policy 数据闭环**。缺了可验证性，SFT 再干净也封顶在 teacher 的水平。

---

## 附：下次 RL 不涨，按这个顺序排查

1. **先量初始化**：SFT 在 RL prompt 分布上的 reward 分布 / pass@k。全 0 或方差 ≈ 0 $\Rightarrow$ 是初始化问题，不是算法问题。
2. **看组内奖励方差**：零方差组的占比有多高。
3. **验 reward**：抽高奖励样本人工看，是否已被 hack。
4. **最后才动** temperature / rollout / KL / 算法（PPO vs GRPO）。

一句话：**先问"模型够不够得着"，再问"怎么推它"。** 你之前一直在回答第二个问题，而瓶颈是第一个。

---
## 🧑‍💻 User

你说的也有一定道理，没有充分SFT，RL很难，因为RL有效的前提是rollout出给正反馈的样本（说白了冷启动也是rl的一部分）

---
## 🤖 Assistant

# 对，而且可以把它说得更狠一点

你说的"RL 有效的前提是 rollout 出给正反馈的样本"，其实就是 **positive-advantage sample 的存在性条件**。不满足，梯度恒为 0，优化器再对也没用。这正是 DeepSeek-R1 里那个显式的 **cold-start SFT** 阶段——官方自己都承认，纯 RL 从 base 起步的探索成本太高，需要先 cold start。

## 把它形式化一下（以 GRPO + 二值 reward 为例）

设某 prompt 上 policy 的成功率是 $p$，rollout 组大小 $G$。GRPO 有信号 $\iff$ 组内**既不是全 0 也不是全 1**：

$$P(\text{有信号}) = 1 - p^{G} - (1-p)^{G}$$

两个直接推论：

1. **提高 $G$ 能救低成功率 prompt。** 固定 $p \in (0,1)$，$G \to \infty$ 时上式 $\to 1$。所以"多采样"有时确实有用——**但只对 $p > 0$ 有效**。
2. **若 $p \approx 0$（直观上 $p \lesssim 1/G$），多采样也救不了。** $(1-p)^G \approx 1$，永远全 0。这就是你之前"调 rollout 没用"的死区——不是 rollout 不够，是 support 里根本没有成功样本。

而且这个式子关于 $p = 1/2$ 对称且在该点最大（$2p(1-p)$ 的推广）。所以：

> **edge-of-competence $\approx$ 每个 prompt 的成功率校准到 $p \approx 1/2$。**

这跟课程学习（curriculum）的经典结论完全一致。**RL 最需要的不是算法，是难度校准。**

## 于是"SFT / cold start 是 RL 的一部分"可以升级成

**SFT = 把探索成本摊销（amortize）到离线的监督数据里。**

| 视角 | 含义 |
|---|---|
| 传统 RL | 初始化 + exploration 由算法自己扛 |
| LLM post-training | exploration 太贵 $\Rightarrow$ 用 SFT/teacher demo 把初始 policy 直接搬到 $p \approx 1/2$ 的区域 |
| 你的 pipeline | SFT(Opus CoT) 提供 $p>0$，RL 只负责把 $p$ 从"有时对"推到"几乎总对" |

所以"SFT 和 RL 的边界"其实是个**算力问题**，不是概念问题：exploration 的预算越大，边界越往 RL 那边挪（这就是 R1-Zero 能 work 的原因——base 本身 latent 能力够 + verifiable reward + 海量 rollout）。绝大多数团队没有这个预算，于是**把探索外化成数据工程**。

## 顺带修正一个常见误解

很多迭代框架（STaR / RFT / rejection-sampling-SFT / 迭代 DPO）本质上都在做同一件事：

$$\text{RL/cold-start 不涨} \Rightarrow \text{rollout} \to \text{筛出高 reward 轨迹} \to \text{SFT} \to \text{再 RL}$$

这就是**用数据循环自动制造 cold start**。所以你说"以后是纯数据工程"有一定道理，但更准确的描述是：

> **数据工程在替 RL 做 exploration；一旦某天 exploration 便宜到能内化，"数据工程"又会退回成 RL 的一个子步骤。**

---

## 一个可操作的判据（接着上次的排查清单）

在做任何 RL 之前，先算每个 prompt 的 **pass rate 分布**（用目标 temperature 采 $G$ 个）：

- 大量 prompt $p < 1/G$ $\Rightarrow$ **别 RL，去补 cold-start 数据 / 降难度 / 加 teacher demo。**
- 大量 prompt $p > 1 - 1/G$ $\Rightarrow$ 已饱和，RL 收益边际极低，别浪费算力。
- 集中在 $p \approx 1/2$ 附近 $\Rightarrow$ 这才是该 RL 的 prompt 子集，把算力压在这里。

**这就是"数据工程"真正在做的事：不是在堆数据，是在给每个 prompt 调 $p$。**
