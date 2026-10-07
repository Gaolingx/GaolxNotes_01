# RFT与RL权衡

## 🧑‍💻 User

reject sampling finetuning在基模后训练中用途有哪些？

---
## 😀 默认助手 (7)

# 拒绝采样微调（Rejection Sampling Fine-tuning, RFT）在基模后训练中的用途

拒绝采样微调（也称 Rejection Sampling Fine-Tuning, **RFT** / **RS**，有时叫 best-of-n distillation）是后训练（post-training）中一种介于纯 SFT 与强化学习之间的"半监督自提升"方法。它不直接做策略梯度更新，而是**用采样+筛选构造高质量数据，再做监督微调**。

## 一、核心机制

给定 prompt $x$，从当前策略 $\pi_\theta$（或更强的教师模型）采样 $n$ 个候选输出 $\{y_1, \dots, y_n\}$，用一个**奖励模型** $r(x,y)$ 或**可验证的验证器** $V(x,y)$ 打分，保留最优或超过阈值的样本：

$$
y^* = \arg\max_{i} r(x, y_i) \quad \text{或} \quad \mathcal{D}^+ = \{(x,y_i) \mid r(x,y_i) \ge \tau\}
$$

再对筛选出的样本做标准 SFT：

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x,y^*)\sim\mathcal{D}} \log \pi_\theta(y^* \mid x)
$$

本质上，它把"模型能生成但还没稳定"的能力，通过筛选**固化**进参数。

## 二、在后训练流水线中的主要用途

<details>
<summary><b>1. 自训练 / 引导推理能力（Bootstrapping Reasoning）</b>（点击展开）</summary>

- 经典代表是 **STaR**（Self-Taught Reasoner）：让模型生成推理链，**只保留最终答案正确的轨迹**，再拿回去微调，迭代若干轮。
- 用途：在**没有大规模人工标注推理过程**的情况下，把模型自身"偶然做对"的推理路径变成训练信号，实现推理能力的自我进化。
- 变体：rationale 补全（Rationale-augmented）、冷启动数据生成。
</details>

<details>
<summary><b>2. 数学 / 代码等可验证领域的提升（RLVR 数据生成）</b>（点击展开）</summary>

- 数学、代码、逻辑题有**客观验证器**（答案匹配、单元测试、编译器），天然适合拒绝采样。
- 用途：
  - 生成大规模的"正确解答"用于 SFT，提升 GSM8K / MATH / HumanEval 表现。
  - 为后续 RL（PPO/GRPO）提供**冷启动（cold-start）SFT 数据**，这是 DeepSeek-R1 等推理模型流程中的关键一步：先生成大量长 CoT，过滤出答案正确的，再做冷启动微调。
- 由于答案可验证，筛选噪声远低于用奖励模型打分。
</details>

<details>
<summary><b>3. RLHF 流水线中的组成部分（替代 / 补充 PPO）</b>（点击展开）</summary>

- 在经典 RLHF（如 Llama 2、InstructGPT 时代的迭代流程）中，"**拒绝采样步**"常被夹在 SFT 与 PPO 之间：
  1. 用当前策略对一批 prompt 采样多个回答；
  2. 用奖励模型选出最佳回答；
  3. 用这些最佳回答做 SFT（即 RFT）。
- 用途：**比 PPO 更稳定、更省算力**的对齐手段；常被用作 RL 的预热（warm-up）或与 RL 交替迭代。
- 也被称为 "best-of-n distillation"：把 best-of-n 采样得到的分布"蒸馏"回策略本身。
</details>

<details>
<summary><b>4. 数据增强与分布对齐（Data Augmentation）</b>（点击展开）</summary>

- 用途：
  - **扩充 SFT 数据**：用强模型/自身采样并筛选，低成本扩充高质量指令数据。
  - **对齐输出分布**：让策略分布靠近奖励模型偏好的分布（类似 offline RL 中的 behavior cloning on filtered data）。
  - **缓解 exposure bias**：用模型自己（而非人类）生成的、但被筛选后高质量的回答来训练，使训练/推理分布更一致。
</details>

<details>
<summary><b>5. 蒸馏与能力迁移（Distillation）</b>（点击展开）</summary>

- 用**更强教师模型**做拒绝采样，得到高质量样例，再微调小模型（Kim et al. 的 "Sequence-Level Knowledge Distillation" 思路）。
- 用途：把大模型（或融合模型、推理模型）的能力迁移到更小、更可部署的模型。
- 也可以**自我蒸馏**：用早期 checkpoint 采样+筛选，训练后续版本。
</details>

<details>
<summary><b>6. 迭代式自我改进（Iterative RFT）</b>（点击展开）</summary>

- 重复"采样 → 筛选 → 微调"多轮，每轮用更新后的策略重新采样，逐步提升筛选通过率与整体能力。
- 代表：ReST（Reinforced Self-Training）、ReST-EM、ETO 等。
- 用途：在无在线 RL 开销的情况下，逼近在线 RL 的收益。
</details>

## 三、典型流程位置

```text
预训练 (Pretrain)
   ↓
SFT（人工数据）
   ↓
【拒绝采样微调 RFT】← 本主题
   ↓  采样 n 个 → 奖励/验证器筛选 → SFT
   ↓
RL（PPO / GRPO / DPO）或 交付
```

它通常位于 **SFT 之后、RL 之前或与 RL 交替**，扮演"数据质量放大器"和"RL 冷启动器"的角色。

## 四、优缺点

| 维度 | 说明 |
|---|---|
| **优点** | 简单稳定、无需策略梯度、可利用可验证奖励、天然抗 reward hacking（有验证器时）、算力可控（$n$ 可调） |
| **缺点** | 依赖筛选信号质量；采样成本随 $n$ 线性增长；只有"做对过"的样本才能被利用（**探索瓶颈**）；对奖励模型打分领域易过拟合奖励 |
| **缓解** | 用 ground-truth 验证器（数学/代码）、KL 约束、与在线 RL 结合、多样性筛选（去重/覆盖度） |

## 五、代表工作对照

- **STaR / V-STaR** —— 自举推理
- **ReST / ReST-EM** —— 迭代式拒绝采样生成 + 微调
- **Llama 2 RLHF** —— 迭代流程中的拒绝采样步
- **DeepSeek-R1** —— 用拒绝采样生成冷启动 SFT 数据 + RLVR
- **Anthropic/OpenAI 早期 RLHF** —— best-of-n + 蒸馏
- **Sequence-Level KD (Kim & Rush)** —— 序列级蒸馏

---

**一句话总结**：拒绝采样微调在基模后训练中的核心用途，是**用"采样 + 筛选"把模型（或教师）偶然生成的高质量输出变成稳定的监督信号**，从而低成本地自举推理能力、生成 RL 冷启动数据、替代或预热 RLHF，并在可验证领域（数学/代码）效果尤为突出。

如果你需要，我可以进一步展开**某个具体场景**（如 R1 式冷启动数据管线、ReST 算法伪代码，或如何在数学任务上设计 $n$ 与阈值 $\tau$）。

---
## 🧑‍💻 User

拒绝采样rft 我记得24年做的人多，好处呢是速度快，训练资源消耗少，缺点是数据利用率比较低，且上限有限
btw，其实现在很多还在rft，上限取决于你data filter和data配比，基本上是纯engineering，花时间就能有提升，而且是monotonous提升，这玩意iteration做多了其实跟rl没啥本质区别（但要做和rl差不多多的轮次 几十轮至少），不过上限确实比不过onpolicy的rl（之前经验是rft＜dpo＜grpo/dpo）
你怎么看

---
## 😀 默认助手 (7)

你的体感基本是对的，我只在两个地方想加一点"结构性"的拆解，因为它们正好解释了**为什么**你观察到的排序和"迭代多了≈RL"会成立。

## 一、先把 RFT 的梯度写出来，很多东西就自解了

把"采样→硬筛→SFT"看成一个整体的估计量：数据来自旧策略 $\pi_{\text{old}}$，只保留 $r(x,y)=1$ 的样本，损失是 $-\log\pi_\theta(y|x)$。它的梯度是

$$
\nabla_\theta \mathcal{L} \;=\; \mathbb{E}_{y\sim\pi_{\text{old}}}\big[\,\mathbb{1}[r(x,y)=1]\,\nabla_\theta \log \pi_\theta(y\mid x)\,\big]
$$

这正好等于**离线 REINFORCE / 策略梯度**，其中 reward 取 0-1、**没有 baseline、没有重要性修正**。于是三个"缺点"其实是一个来源：

- **数据利用率低**：把 $r=0$ 的样本当噪声扔了 → 只做"正向强化"，没有减号去压制错误行为。
- **没有 baseline**：GRPO 式地减掉组内均值后，全对的简单题 advantage≈0、被自动忽略；RFT 是"只要对就要"，**把 90% 通过率的题和 5% 通过率的题一视同仁**，算力大量花在已经会的 prompt 上。这才是"利用率"更隐蔽的那一半浪费。
- **上限有限**：见第三条。

对照一下 DPO：它把**被拒样本当 negative 用**，吃到了 pairwise 的对比信号，所以同样采样量下信息密度天然高于 RFT —— 这和"RFT < DPO"是一致的。

## 二、"单调提升 / 纯 engineering"——基本成立，但有个拐点

同意，它方差小、易 debug，data filter + 配比是主战场。但有两点提醒：

1. **别只看 pass@1**。RFT 是"sharpening without exploration"，迭代多了会出现 **entropy collapse**：pass@1 还在平/微涨，但 pass@k、输出多样性在掉。这时候曲线看着"单调"，其实是分布已经被压窄了。监控 pass@k 比看 loss 有用得多。
2. **difficulty curriculum 是关键工程**。保留通过率在 $[0.1, 0.9]$ 区间的题（太易没信号、太难全丢），等价于手动给了一个"baseline"，能把 RFT 的曲线再抬一截——本质就是人工补上第一条里缺的 advantage。

## 三、"迭代多了≈RL"——对，但它收敛到的是"专家迭代"，不是 on-policy RL

迭代确实让 behavior policy 变成最近的 $\pi_\theta$，逼近 **expert iteration / 类 EM（ReST-EM）**。但结构差异还在：

| | 迭代 RFT | on-policy RL (GRPO) |
|---|---|---|
| 采样来源 | 旧策略（有滞后） | 当前策略 |
| 信号 | 只加正向、无 baseline | 正负 advantage、组内相对 |
| 上限机制 | 受 **base 的 pass@n** 天花板约束 | 可"滚动自举"，突破 pass@n |

天花板那条是本质：RFT 的极限大概是"基模在采样温度下**偶尔能做对**的东西"被你固化下来，它**不会创造新行为**。on-policy RL 因为每次用更新后的策略重新采样，有 compounding 效应，理论上能越过 pass@n——这也是你"上限比不过 on-policy RL"那句的最硬解释。

你给的排序 `RFT < DPO < GRPO/DPO`，我基本认同方向，但会加限定：**它依赖奖励可验证、且基模探索能力够**。在弱验证器或小模型上，RL 极易 reward hack / 熵崩，反而 RFT 更稳，甚至反超。换句话说，这个序不是任务的普适律，是"reward 质量 × 探索能力"的函数。

## 四、我的框架：别当 either/or，它是 pipeline 的两段

- **RFT = consolidation（巩固）**：把"偶然对"变"稳定对"，便宜、可靠、冷启动必需（R1 那套 cold-start SFT 就是这个）。
- **RL = exploration/optimization（探索）**：突破 pass@n、做相对优势、压制坏行为。
- 现实最优解基本是 **RFT 先到一个好 basin，再上 RL 往外推**，而且 RFT 的产物还能顺手变成 DPO 的偏好对（把拒样本利用起来，补掉它最大的短板）。

所以我会说：RFT 不是"过时的 2024 方案"，而是**性价比极高的正则化项和冷启动器**；它的天花板是被自己的目标函数（positive-only、无 baseline、离线）钉死的，工程能逼近那个天花板，但要越过去，还是得把数据重新变成 on-policy、把信号变成带 baseline 的 advantage。

---

一个小问题反问你：你那边 **RFT→DPO 的接力**试过吗（用 RFT 的 accept/reject 直接造偏好对喂 DPO）？如果试过，DPO 那段的收益是主要来自负样本信号，还是来自 on-policy 化的迭代 DPO？这个能判断你们场景到底是被"样本量"还是被"信号类型"卡住。

---
## 🧑‍💻 User

嗯，另外负样本排斥的训练很重要，grpo的问题是他会压负样本，rft不会，他只会推高正样本，如果效果不好可能我们rft的数据和轮次做的不太够（换句话说说要跑到什么程度算收敛）？
btw，既然他不会压负样本，做成产品之后bad case率会不会比dpo这些高呢？
不过压负样本不一定make sense，比如你math response最后一步错了前面都对，你给他整个压下去，这个是错的。
opd的dense reward就能解决这个问题，opd好处就是dense+rev kl产生的mode seeking property，坏处是会entropy塌下来

总之，training experience是只要调的够好，rft可能做到甚至比rl好（视任务或者目标问题而定），但很吃engineering
比如filter的策略啥的，然后把训一半的模型生成的data和一开始的合并起来继续重头训

---
## 😀 默认助手 (7)

几个点我逐个接，尤其"跑到什么程度算收敛"这个我尽量给可测的判据。

## 一、GRPO 压负样本、RFT 不压——但要区分"选择性压制"和"无差别 sharpening"

你说得对，但有个容易混的地方：**RFT 其实也在"压"负样本，只是无差别地压。**

softmax 是归一化的，你把正样本概率推高，剩下的所有 token（包括错误 token、也包括正确答案的另一种合法写法）概率**同比被动下降**。所以 RFT 不是"不压"，而是：

- GRPO：**选择性**压制 —— 只压这一组里采样到、且 advantage<0 的具体 token 序列；
- RFT：**无差别**压制 —— 通过整体 sharpening 把 distribution 收窄，代价是连"没被采样到的正确解"一起压掉了。

这正好解释了为什么 RFT 跑多了会 entropy collapse：它在用"多样化"换"正确率"，换到后面边际收益全是熵换来的。GRPO 的 advantage 里那个负号，是**定向**的，所以它做的是"优化"；RFT 做的是"收敛到已知好解的盆地"。

## 二、"跑到什么程度算收敛"——我的判据

别把收敛当成一个点，应该当成一个 **stopping rule**。我一般盯四个量（$t$ 为迭代轮次）：

| 指标 | 含义 | 停下来/继续的信号 |
|---|---|---|
| $p_{\text{band}}^{(t)}$ | 当前温度下，仍落在**可学习带** $[0.1,0.9]$ 通过率的 prompt 占比 | → 0，说明没有"能学会但还没稳"的题，**信号枯竭，就是收敛** |
| $\Delta_t$ | held-out pass@1 的逐轮增量 | 连续 2 轮 < 1% 绝对 | 
| $H^{(t)}$ | 输出熵 / distinct-n / pass@k | 掉得比 pass@1 涨得快 → 已在过优化 |
| OOD 集 | 通用能力回归 | 一有回归就停 |

**核心那句**：RFT 的收敛点 ≈ **base 策略在学习集上的 pass@n**。所以你其实可以先把 base 的 pass@n 量出来当参考线——当 RFT 的 pass@1 追到 pass@n 的 90%+，后面再练基本就是"熵换正确率"，不是真提升。

**"多少轮"没有普适答案**，它由"你还能不能造出新的可学习数据"决定：
- 冷启动通常 **1–3 轮**；
- 自我迭代一般 **3–6 轮**到甜点，**6–10 轮**之后多半过甜点；
- 能不能继续，取决于你是否**每轮注入新的 hard prompt / 提高难度带**。一旦 prompt 分布固定，$p_{\text{band}}\to 0$，再练就是纯 sharpening。

所以你说"效果不好可能是数据/轮次不够"，我会补一句：**更常见的是"轮次够了、但可选数据被采完了"**——这时候该动的不是 iteration 数，是难度课程和 $n$。

## 三、产品 bad case 率会不会比 DPO 高？

我倾向：**方向上会,但分两种情况**：

- **可验证任务（math/code）**：RFT 的 accept 集是高精度验证过的，in-distribution bad case 不一定比 DPO 高，甚至更低。
- **开放域 / 有害内容规避**：RFT 天生吃亏——它没有"学会不要做什么"的机制，对未见过的失败模式泛化差；而且熵塌之后，**失败会成簇、还更自信**（pass@1 涨但方差变差）。这类上 DPO 通常更稳。

还有个现实缓冲：很多 RFT 产品在**推理时挂了 best-of-n + verifier**，bad case 被推理侧过滤掉了——代价是延迟。所以"产品 bad case 率"其实取决于你有没有把 rejection 留在 inference，而不只是训练目标。

## 四、你的 math 例子恰恰说明序列级负样本是钝器——这也是 OPD 的价值所在

"最后一步错、前面全对 → 整条压低"是典型的 **credit assignment / 稀疏奖励**问题。序列级 advantage 对"前缀是对的"这件事一无所知，压下去等于连正确的推理过程一起 unlearn。解法有三个层次：

1. **数据工程（最便宜）**：构造 DPO 负样本时**保留正确前缀，只在错误步骤处分叉**（minimal-edit negative）。你以为在训练"避开错误"，其实是在训练"这一步别这么走"。
2. **PRM / 过程奖励**：步级信号。
3. **Dense / OPD**：你点到的正解。

**OPD 为什么能修这个**：它给的是**逐 token 的 dense advantage**——本质是"以 teacher 为奖励、advantage ≈ 逐 token 的 $\log(\pi_\theta/\pi_T)$ 的 RL"。梯度只在 teacher 不认同的那几个 token 上有大信号，正确前缀几乎不被波及。这正好是"选择性、细粒度压制"，跟 GRPO 的粗粒度组内压制不是一个量级。

但你对 reverse KL 的判断我完全同意，补个机理和一个缓解工具箱：

- **forward KL** $\mathrm{KL}(\pi_T\|\pi_\theta)$（teacher‖student）= **mass-covering / mean-seeking**，保留多样性但会把多个 mode 平均掉；
- **reverse KL** $\mathrm{KL}(\pi_\theta\|\pi_T)$（student‖teacher）= **mode-seeking / zero-forcing**，抓 teacher 的尖峰、生成更干脆，但**天然降熵**——这就是 collapse 的根。
- 缓解：**truncated reverse KL / top-k teacher**（只对 teacher 高概率区做反向 KL，别对 tail 较劲）、**forward/reverse 插值或 JSD（GKD 那套）**、**熵 bonus / temperature 调度**、以及**teacher 课程**（先 covering 后 seeking）。我个人最常用的是**截断 + 少量 forward KL 打底**。

## 五、你的"半训模型数据 + 原始数据 → 重头训"这个 recipe

这个我很认同，它其实一箭三雕：

1. **replay / 抗遗忘**：原始数据当锚，直接对抗熵塌和分布漂移；
2. **重头训（fresh optimizer）**比在同一 checkpoint 上续训更不容易过尖——相当于每轮把分布"重新摊开"；
3. **数据累积**扩样本量。

代价是要**全量重训**，把 RFT"便宜"的卖点吃掉一半。两个工程细节值得盯：**generated : original 的配比**（生成太多会 drift，太少提升慢），以及**生成数据的偏置会被递归放大**（半训模型自身的坏模式会被当正样本喂回去）。如果嫌重训贵，可以退化成"续训时掺 10–20% 原始数据 + 熵正则"，八成能拿到类似效果。

---

**总 synthesis**：我跟你结论一致——**RFT 调得好能在某些任务上追平甚至超过 RL，但它天花板被"positive-only + 无 baseline + 离线"钉死，本质是巩固而非探索**。真正拉开差距的不是"用不用 RFT"，而是：

- 你能不能**每轮造出落在可学习带里的新数据**（决定轮次上限）；
- 你的**负样本粒度**（序列级 → 最小编辑 → dense/OPD，决定 bad case 和部分正确的问题）；
- 以及**熵管理**（否则 RFT/OPD 都会在同一个地方塌）。

所以我现在的默认 pipeline 是：**RFT 冷启动 → 最小编辑负样本的 DPO → OP(D) 补 dense 信号 → 视算力上 GRPO**，RFT 全程当"巩固器"和"数据工厂"用，而不是当终局方案。

你们现在是卡在"数据造不出新的"还是"熵已经塌了"？这两个的下一步动作完全不一样——前者该上课程/$n$，后者该上 replay/正则。

---
## 🧑‍💻 User

如果业务侧是对bad case率很敏感，比方说因为高bad case会影响用户留存率，用户抽卡n次出不来结果就不付费了
很久之前还做dpo的时候，大部分时间都是研究什么构造负样本
如果对pass@1 要求不高的话，rft这些效果确实很强，而且对entropy影响也相对小
像rl基本上提高pass@1是没啥问题的，但entropy塌的很严重

那有没有什么权衡之计？

---
## 😀 默认助手 (7)

你这个业务画像其实把问题简化了一大半：**你的 KPI 不是 pass@1，是 pass@n × (1 − bad case 率)。** 一旦这么重写，很多"权衡"就变成"解耦"了——因为 pass@n 和 bad case 是两个**正交轴**，而 RL 犯的错正是用一根轴（sharpening）去同时解两个轴。

## 一、先把矛盾画清楚

| | sharpening（RL/DPO 的负压） | coverage（RFT/高熵） |
|---|---|---|
| pass@1 | ↑↑ | ↓（要抽） |
| **pass@n** | **↓↓（塌了就抽不出来了）** | **↑↑** |
| bad case 率 | ↓（选择性压制） | ↑（没压） |
| 熵 | ↓↓↓ | 基本不动 |

RL 的问题是：它拿 pass@1 的收益，**透支了 pass@n**。而你的业务恰恰吃 pass@n，还要靠 n 次里"至少出来一个"来收费。所以 RL 对你的净贡献可能比你想的低，甚至为负。

**结论先给**：对你这种"n 次抽卡 + 留存敏感"的业务，最优解大概率不是"RFT 还是 RL"，而是 **生成端保持高熵（要 pass@n）+ 推理端加验证/拒绝（压 bad case）**。也就是把"探索"和"精确"分到两个组件里去解。

## 二、四个层次的权衡之计

### 1. 把拒绝搬到 inference（最贴合你 n 次抽卡的模式）

你本来就要抽 n 次，那就别让训练把这 n 次的多样性提前压没：

- **训练端**：走 RFT / soft-RFT，**目标就是保熵**，别上会把熵打塌的 RL；
- **推理端**：n 次采样 + **verifier（规则/PRM）+ rejector（安全/质量分类器）** 做 best-of-n 或 early-exit；
- **配套**：验证失败别只重抽，做 **repair 循环**（把拒绝理由回灌成 prompt 再生成），把"bad case"转成"good case"，而不是单纯丢掉。

这一步直接把你业务模型里"抽 n 次"的成本变成**收益杠杆**：n 越大，能容忍训练端熵越高、越不需要压负样本。

### 2. 把"错误"和"危害"分开——只对安全轴硬压

你 math 最后一步错的例子说明：**能力性错误不该硬压**（压了等于 unlearn 正确推理）。但**危害性/掉留存的 bad case**（胡言乱语、冒犯、崩坏输出）必须硬压。所以：

- 能力轴 → 交给 pass@n + retry + RFT；
- 安全/信任轴 → 用 **DPO / 一个专门的 rejector** 去选择性压制。

**只对后者动用"负样本压制"，熵的损失就只发生在一个很小的分布上，几乎不影响 pass@n。** 这是对你"负样本构造很花时间"的最优再分配——把宝贵的人工负样本预算全砸在"伤害留存"的那类上，而不是"答案不对"的那类。

### 3. 负样本构造升级，专治"最后一步错"

既然你 DPO 时代大部分时间在搞负样本构造，那"最小编辑负样本"就是最值钱的升级：

- **保留正确前缀，只在错误步分叉**得到 rejected → 压制只落在分叉处；
- 更激进：**token/segment-level DPO**，loss 只算在错误 span 上（mask 掉正确部分）；
- 负样本用 **base/半训模型自己生成**（模型自己能错的那类才是有效负样本），系统性产出，别全靠人工。

效果 = selective suppression 的收益，熵代价 ≈ 单点抑制的代价，**几乎复刻 OPD 的 dense 好处，但不用付 reverse-KL 塌熵的账**。

### 4. 想要点 RL 的锐度？那就做"软 RL"，别做硬 RL

如果你确实想在能力轴上要一点 pass@1，有个天然的连续旋钮：**advantage 加权回归（reward/advantage-weighted BC）**，权重 $w=\exp(A/\beta)$：

- $\beta\to\infty$ → 退化成 SFT；
- $\beta\to 0$ → 退化成 RFT 硬筛；
- 中间 → **保留所有样本（不丢熵），只是把坏样本软性降权**。

这就是 **RFT 和 RL 之间的插值**，而且它比硬 RL 温和得多。配套再加：

- **entropy bonus** + **加大 KL-to-reference 系数**（DPO 的 $\beta$ 就是这个手刹，它天然带一根对 reference 的 KL 绳）；
- **early stop**：GRPO 只跑 1–2 轮当"nudge"，涨点 pass@1 就收手，别跑到塌。

## 三、推荐给你业务的组合

```text
生成端（保 pass@n）:   RFT / soft-RFT（advantage 加权，别硬筛死）
精确端（压 bad case）:  最小编辑负样本的 DPO  +  独立 rejector
推理端（吃 n 次）:      best-of-n + verifier + repair loop
RL（可选）:            仅当 pass@1 缺口值得时，entropy-bonus 的 GRPO 跑 1-2 轮 early-stop
```

一句话逻辑：**训练不要解决"精确"问题，只解决"能力覆盖"；精确交给推理端的验证+拒绝。** RL 那种在训练里把分布削尖拿 pass@1 的做法，对你这种靠"多次尝试 + 留存"变现的业务，是在用最贵的代价买你最不需要的指标。

---

**所以权衡之计的本质是**：不是找 RFT 和 RL 的中间点，而是**把 pass@n（要多样性）和 bad case（要精确）拆到两个组件**——策略侧保持高熵做纵深，验证/拒绝侧做横向拦截。你 DPO 时代攒的负样本能力，正好可以下沉成"推理端 rejector + 局部 DPO"，既保住了熵，又保住了留存。

想更进一步的话，可以顺手把目标显式写成 $\max\ \text{pass@}n \times (1-\text{badcase})$，甚至用 **pass@n 直接做奖励**去训（多样性感知的 RL）——但那个工程复杂度高，先用上面这套"解耦"基本能拿到八成收益。
