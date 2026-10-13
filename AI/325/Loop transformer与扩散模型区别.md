# Loop transformer与扩散模型区别

## 🧑‍💻 User

讲讲loop transformer，以及和diffusion的区别

---
## 😀 默认助手 (7)

I'll ground the recent works before writing. Let me check a couple of references.

先说一个术语澄清:你说的 "loop transformer" 在文献里通常有四个近义名字——**Looped Transformer / LoopLM / recurrent-depth Transformer / weight-tied Transformer**(有时也叫 "深度递归")。下面都按这个含义讲。它不是什么新算子,而是一种**参数复用策略**。

---

## 一、Loop Transformer 讲的是什么

### 1. 形式化:把"深度"变成"可以反复走的循环"

标准 Transformer 是**每层一套独立参数**:

$$x_{l+1} = F_l(x_l), \qquad l = 0, 1, \dots, L-1$$

Looped Transformer 是**同一套参数走 $K$ 遍**:

$$h_0 = P(x), \qquad h_{k+1} = F_\theta(h_k, e_k), \qquad y = C(h_K)$$

其中 $P$ 是前奏(prelude)、$C$ 是尾声(coda)、$F_\theta$ 是共享的循环块, $e_k$ 是可选的循环步编码(可以没有,也可以像 diffusion 那样给每步不同的条件)。关键恒等式:

$$\text{参数量} \times 1, \qquad \text{FLOPs} \times K \text{ (有效深度} \approx L \times K)$$

也就是说,它把**"加宽加深"**和**"多花算力"**这两件事解耦了:参数量不变,推理算力可以按需加。

### 2. 家族谱系

| 家族 | 代表工作 | 循环什么 | 怎么停机 |
|---|---|---|---|
| Universal Transformer + ACT | Dehghani 2018;Graves 2016 | 每层权重共享 | 动态 halt 概率 |
| 隐式层 / 定点 | DEQ (Bai 2019);ALBERT 式权重共享 | 全层干到收敛 | 不动点判据 |
| Recurrent-depth 潜推理 | Huginn(2025) | 潜空间迭代 + 随机 loop 数训练 | 固定/自适应 |
| LoopLM | Ouro 系列 | 整块 Transformer 重复 | 熵门 / exit gate |
| Looped MoE | LOOM、Foil、Loop Scaling Laws(2026) | MoE 层重复 + 每 loop 独立 router | 固定/自适应 |
| 潜 CoT | LOTUS(2026)、Relit | loop + 对每个潜位置做监督 | 固定 |

### 3. 为什么有人做这个

1. **参数效率**:小模型 + 多循环,能在数学/推理上有超预期表现。LoopCD 的报告里,Ouro-2.6B-Thinking 在 AIME 2024 上通过对比中间循环状态,pass@1 从 61.88% 提到 73.33%,而且能**砍掉一半循环数**还保持水平(FLOPs 降 22.5%~48.2%)。
2. **隐式 CoT**:显式 CoT 要输出 token,又慢又占显存;loop 是在隐空间里"多想几遍再开口"。LoopSLM 报告相对 CoT-SFT 基线,推理准确率涨 20+ 分而**生成 token 少 64.5%**。
3. **算法外推**:循环天然适合"迭代算法"这种归纳偏置(图上的游走、Bellman 迭代、定点求解),对长度泛化友好。
4. **测试时计算扩展的新轴**:可以只增加 loop 数,不增加输出长度。

### 4. 它的真实难点(这部分决定了"为什么不流行")

- **稳定性**:残差更新在循环中累加,隐状态方差随 loop 膨胀,表示会漂移;到某个深度后性能**不涨反崩**(STARS、LOOM 都在处理这个)。LOOM 的结论是:7 亿参数在 iso-FLOP 下**最优点在 5 个 loop**,再往上就回退。
- **监督信号缺失(最要命的一点)**:通常只有最后一个 loop 有 loss,中间态没有任何监督 → 梯度要穿过 $K$ 次 BPTT,既要防梯度爆炸,又学不到"中间该怎么变"。所以 2026 年这批工作几乎都在打这个补丁:深监督(Looped-DiT、LOTUS)、随机 loop 采样训练、Jacobian 谱半径正则(STARS)。
- **KV cache 会随循环线性膨胀**:每轮循环写一份自己的 KV。LPT 的做法是"只让第一轮写 cache,后面只读",反而提质又省 76~79% 上下文显存。
- **收益有界**:Loop Scaling Laws 把"递归"和"稀疏"一起建模,给出的是一个**有上界的**有效参数增益函数——不是循环越多越好。

---

## 二、Diffusion 讲的是什么(只抓和 loop 对照的部分)

前向是**有定义好的**加噪过程:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)$$

有闭式:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

反向学一个去噪器,训练目标在**每一个 $t$ 上都有明确监督**:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]$$

采样就是把这个网络从 $t=T$ 到 $t=0$ 调用 $T$ 次(DDPM / DDIM / flow matching 都是这个骨架)。离散版本(masked diffusion,dLLM)把"加噪"换成"随机 mask 掉 token",代表工作有 LLaDA、Dream、Fast-dLLM、LLaDA MoE v2。

**注意这个结构:同一个网络被反复调用、权重共享、带一个"步"下标做条件。** 就此而言,diffusion 采样器**在结构上就是一个 loop**。这就是为什么这两个东西老被放在一起谈。

---

## 三、核心区别

先给一句话版本:

> **Loop Transformer 的循环下标是"计算深度",Diffusion 的循环下标是"噪声水平"。前者是一个没有中间答案的迭代精化过程,后者是一个自带中间答案的治程。**

| 维度 | Loop Transformer | Diffusion |
|---|---|---|
| 迭代轴语义 | 计算深度 / 精化轮次 $k$ | 噪声水平 / 时间 $t$ |
| 迭代状态是什么 | 干净的隐表示,中间态**不对应任何样本** | 带噪数据 $x_t$,中间态**对应明确定义的分布** $p_t$ |
| 是否有前向过程 | 无。中间路径是学出来的、无约束 | 有。$q(x_t\mid x_0)$ 闭式给定 |
| 中间监督 | 通常**没有**(需人工造:深监督 / corruption 课程) | **每一步都有**($\epsilon$ / $x_0$ / $v$ 都是现成 target) |
| 目标 | 还是 next-token prediction,改的是**内部算多久** | 直接建模 $p(x)$,是**生成模型** |
| 随机性 | 确定性映射,给相同的 loop 数输出相同 | 采样自带随机性 → 内在多样性,还有 score / ELBO 一套理论 |
| 停机准则 | 收敛 / exit gate / 熵门(可以自适应) | 噪声调度走完 $T$ 步(步数不能随便减,除非蒸馏) |
| 生成粒度 | 一般仍是**自回归逐 token**,loop 只影响每个 token 的内部计算 | **并行**去噪整段序列,顺序可任意,天然双向上下文 |
| 算力-质量关系 | 收益有上界,过头会崩 | 步数-质量有 trade-off,少步要蒸馏 |

### 展开说几个最容易被忽略的差别

**1. 中间态有没有"意义" —— 这是最本质的差别。**

Loop 的 $h_k$ 是一个抽象的隐向量,你怎么读它都行,但**没有任何外部定义告诉你 $h_3$"应该"是什么**。Diffusion 的 $x_t$ 则是一个**明确定义的对象**:它就是从干净数据出发走完前向过程第 $t$ 步的那个带噪样本。这个差别带来一连串后果:diffusion 可以在任意 $t$ 上监督,可以做不同的噪声调度,可以用 ODE/SDE 理论分析;loop 全都得自己发明。

`Denoising Recursion Models`(2026)这篇把话说得很直白:diffusion 用不同强度的噪声污染数据、然后要求**一步**还原,这造成了训练/测试行为错位;而 looped 的 Tiny Recursion Model 又只有终点的监督、没有中间路径的课程。**两者的短板正好互为对方的长处**——所以他们的做法是"用 diffusion 式的 corruption 给 loop 造一个中间课程,但要求分多步还原"。

**2. 它们是两个正交的计算轴。**

`Recursive Scaling in Masked Diffusion Models`(R-MDM, 2026)给了目前我看到的最干净的形式化。在 masked diffusion 里,解码是一个**二维决策**:

- **denoising step**:改变状态本身 —— 决定**哪些 token 被 commit**(生成精度 $\approx$ 边的选择)
- **loop / recursion**:在**固定的掩码状态下**反复精化 —— 提升模型在该状态下的边际分布(生成精度 $\approx$ 边际的质量)

他们把它写成"依赖性-保真度"分解:并行解码的误差一部分来自**没搞清楚 token 之间的依赖**,一部分来自**在这些条件上算得不够好**。循环修的是后者,去噪步修的是前者。然后按熵自适应地分配这两个轴,质量-算力前沿就比固定调度更好。

**3. 因此"该加循环还是该加去噪步"是个实证问题,不是理论问题。**

`Looped Diffusion Transformer`(Looped-DiT, 2026)的结论是:**在固定推理预算下,加循环深度比加去噪步数更划算**。他们的 260M 循环化模型在多个文生图基准上超过了 6.5 倍大的模型,推理算力只要 1/4.9,而且"更深的循环能逐步修正前面循环犯的错",表现出来像一种 latent reasoning。但注意:naive 加循环是不work的,他们必须补上跨循环深监督 + 自调制注意力。

**4. 停机逻辑完全不同。**

Diffusion 的 $T$ 是**调度决定的**,你几乎不能"提前停",因为提前停就是停在有噪声的状态上。Loop 的 $K$ 是**可以自适应的**:可以训一个 exit gate、可以用熵决定、可以像 TaH2 那样只给"确实获益的 token"多跑几轮。TaH2 发现很多 token 根本不需要额外循环,而现状是**每个 token 都固定花同样的循环数**,这浪费很大——他们把 AIME 上的准确率-算力斜率提升了 53%。

**5. 一个是"想清楚再说话",一个是"从雾里洗出照片"。**

Loop Transformer 的直觉:同一个编辑反复改同一篇稿子,改 $K$ 遍后交稿。改的过程中你看不到任何"半成品文本"——只有最后一句是算数的。

Diffusion 的直觉:显影 / 从大理石里凿雕像。你每一步手里都有一张**完整的、只是还很糊的图**,而且每一步的糊的程度是事先规定好的。

---

## 四、一个统一视角(我觉得最有价值的说法)

两者都可以写成同一个式子:

$$z_{k+1} = F_\theta(z_k,\ c_k)$$

区别只在条件 $c_k$ 携带什么语义:

- Loop Transformer:$c_k = k$,**无条件语义**,中间路径无约束 → 需要人为制造中间监督
- Diffusion:$c_k = t_k$ **携带一个外部给定的前向过程**,中间路径被过程本身锚定 → 中间监督免费

所以可以这么说(这是直觉不是定理):

> **Diffusion 是一个自带中间监督的 loop;裸的 Loop Transformer 是一个没有监督路径的 diffusion。**

这也解释了为什么现在两条线在**互相抄**:diffusion 那边在加循环(Looped-DiT、R-MDM、Recursive MDM),loop 那边在借 diffusion 的 corruption 造课程(Denoising Recursion Models)。还有个直接的证据是 `A Comparative analysis of Layer-wise Representational Capacity in AR and Diffusion LLMs`(2026)发现:diffusion 目标会让表示**前几层冗余度很高**——也就是说 diffusion 模型内部有很多"可以拿去循环的算力",反过来 loop 正是要把这部分冗余变成有效计算。

---

## 五、几个常见误解

<details>
<summary>展开</summary>

- **"loop transformer 就是 RNN"**:形似而神不同。RNN 递归的是**时间轴**(逐 token 串行,不能并行训练);loop 递归的是**深度轴**(序列维仍然并行 attention、仍然能 teacher forcing)。当然 loop 在推理时的自回归确实是串行的,这是它延迟高的来源。
- **"diffusion 就是一种 looped transformer"**:结构上对(权重共享 + 反复调用 + 步条件),但丢掉了一个关键事实——diffusion 那个"步"是被前向过程定义的外部坐标,而 loop 的下标只是计数器。忽略这个差别会得到错误的工程结论。
- **"循环次数越多越好"**:明确不对。收益有上界,过头会崩;而且专家选择崩溃(looped MoE 里 router 反复选同一批专家)会让额外循环白花算力。
- **"loop 就是省参数版,一定便宜"**:它省参数但**不省 FLOPs**。它是拿参数换算力,不是白送。

</details>

---

## 六、怎么选

| 你的目标 | 更合适 |
|---|---|
| 参数预算卡死,但愿意加推理算力 | Loop Transformer(LoopLM / looped MoE) |
| 想砍掉 CoT 的 token 开销和延迟 | 隐空间循环(LOTUS、Relit、LoopSLM 那一类) |
| 需要并行生成、双向上下文、打破自回归延迟 | masked diffusion LLM(LLaDA、Dream、Fast-dLLM) |
| 需要生成多样性、理论上的概率模型 | diffusion |
| 既要省参数又要扩散式生成 | **两个一起用**:looped diffusion / recursive MDM |

最后一点值得强调:这两者**不是竞争关系,是两个可叠加的算力轴**。R-MDM 的二维解码决策(跑几轮循环 × commit 哪些 token)大概会是接下来一段时间比较主流的框架。

---

<details>
<summary>主要参考(含 arXiv 编号,便于你自己核)</summary>

**Loop / recurrent depth**
- `2609.40305` Looped Diffusion Transformer(260M 超过 6.5× 大模型,4.9× 更低推理算力)
- `2606.18022` Recursive Scaling in Masked Diffusion Models(二维解码:loop × commit)
- `2604.18839` One Step Forward and K Steps Back: Denoising Recursion Models(借 diffusion corruption 给 loop 造课程)
- `2610.02383` The Surprising Effectiveness of Shared Memory in Looped Transformers(kv cache 共享)
- `2610.01153` Looping Beyond Twice: A Scalable Recipe for Looped MoE(9–12 loop 的稳定化配方)
- `2609.40316` Scaling Laws for Looped Mixture of Experts
- `2610.02185` Decoding Looped Transformers Better for (Almost) Free(LoopCD)
- `2609.35748` Improving Test-Time Scaling with Adaptive Looped Transformers(TaH2)
- `2606.31779` Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers(LOTUS)
- `2605.26733` Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning(STARS)
- `2610.02741` On the Chain-of-Thought Monitorability of Looped Language Models(可用性/监控视角)
- 早期:Universal Transformer (Dehghani et al. 2018)、ACT (Graves 2016)、DEQ (Bai et al. 2019)、Huginn / recurrent depth (2025)

**Diffusion / dLLM**
- DDPM (Ho et al. 2020)、DDIM (Song et al. 2021)、Flow Matching (Lipman et al. 2023)、Latent Diffusion (Rombach et al. 2022)、DiT (Peebles & Xie 2023)
- D3PM (Austin et al. 2021)、MaskGIT、LLaDA、Dream、Fast-dLLM v2、LLaDA MoE v2 (`2608.03457`)
- `2603.07475` AR 与 dLLM 的分层表示容量对比

**一个诚实声明**:上面 2026 年的几篇是预印本,文中的具体数字(73.33% pass@1、6.5×、2.5× 等)都是**论文自报结果**,不是经过第三方复现的结论。

</details>

如果你是在具体做某个方向(比如想给一个 LLM 加循环,或者想把 looped 思想用进图像/机器人),告诉我场景,我可以把上面这些配方里的具体做法(深监督怎么加、随机 loop 怎么采样、KV cache 怎么处理、稳定性正则怎么设计)拆开讲细一点。
