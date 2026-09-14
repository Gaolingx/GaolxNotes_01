# GRPO与其变体GSPO DAPO CISPO RLOO差异

### 🧑‍💻 User

比较grpo与其变体gspo dapo cispo rloo差异。

---

### 🤖 Assistant

我找到了 GSPO 和 CISPO 的来源。现在继续获取 GRPO、DAPO、RLOO 的详细信息。

已获取各论文摘要。现在获取更详细的算法公式与技术细节（DAPO 的四项关键技术、RLOO 的 leave-one-out 基线、CISPO 的实现细节）。

信息已比较充分。再补充获取 GSPO 与 CISPO 的精确公式细节。

GSPO 详情已获取。CISPO 细节再补充一下来源（MiniMax-M1 论文的 HTML 版本路径有变，改用其他来源）。

我已经从各原始论文（DeepSeekMath、DAPO、MiniMax-M1、GSPO、Cohere RLOO）及工程实现文档（NeMo RL、verl）核实了细节。下面是完整对比。

# GRPO 及其变体对比：GSPO / DAPO / CISPO / RLOO

这五个算法都属于**在线策略梯度式 LLM 后训练（RLHF / RLVR）**算法，同一血脉：采样一批回答 → 算奖励 → 用策略梯度更新。它们主要在四个问题上分道扬镳：

1. **优化单元**：token 级 vs 序列级
2. **重要性采样比（IS ratio）**：用不用、怎么用、在哪个层级用
3. **裁剪（clipping）**：裁剪目标函数 vs 裁剪权重 vs 不裁剪
4. **优势估计 / 基线**：组内归一化 vs leave-one-out

## 一、总览表

| | **GRPO** | **GSPO** | **DAPO** | **CISPO** | **RLOO** |
|---|---|---|---|---|---|
| 出处 | DeepSeekMath（DeepSeek，2024.02）[2402.03300] | Qwen 团队（阿里，2025.07）[2507.18071] | ByteDance Seed（2025.03）[2503.14476] | MiniMax-M1（2025.06）[2506.13585] | Cohere（2024.02）[2402.14740] |
| 基本思想 | 去掉 PPO 的 critic，用组内相对优势 | 把优化单元从 token 提升到**整条序列** | 给 GRPO 打 4 个工程补丁 | 裁剪 IS 权重本身而非裁剪目标 | 回归最朴素的 REINFORCE + leave-one-out 基线 |
| 优化单元 | token | **序列** | token | token | **序列**（整段生成=一个动作） |
| IS ratio | token 级 $\pi_\theta/\pi_{\theta_{old}}$ | **序列级**（长度归一化） | token 级 | token 级 | **无**（假设近 on-policy） |
| 裁剪方式 | PPO 式 min-clip（对称 $\varepsilon$） | 序列级 min-clip | 不对称 min-clip（$\varepsilon_{low}/\varepsilon_{high}$ 解耦） | 裁剪权重后 **detach**，梯度永不消失 | 不裁剪 |
| 优势/基线 | 组内 $(R_i-mean)/std$ | 组内 $(R_i-mean)/std$ | 组内 $(R_i-mean)/std$ | 组内 $(R_i-mean)/std$ | leave-one-out：$R_i - \text{mean}_{j\ne i}R_j$ |
| 损失聚合 | sample 级（每条回答等权） | 序列级（一条=一份） | **token 级**（$\sum\|o_i\|$ 归一） | token 级（实现多为 token 归一） | 序列级 |
| KL / 参考策略 | 有（$\beta D_{KL}$） | 通常省略 | 去掉 KL | 有（MiniMax-M1 用） | 有（KL-shaped reward） |
| 主要解决 | PPO 内存/价值模型问题 | **GRPO 的 token 级 IS 失效 + MoE 崩溃** | 熵塌缩、零梯度样本、长 CoT 失衡、超长奖励噪声 | PPO/GRPO 裁剪导致**关键 token 梯度被丢弃** | PPO 复杂且贵，RLHF 中其实不需要 |
| 代表性成绩 | DeepSeekMath / R1 系列 | Qwen3 | AIME24 50 分（Qwen2.5-32B） | MiniMax-M1 | 胜过 PPO/DPO/RAFT 的 win-rate |
| 关键代价/限制 | token 级 IS 方差大，长序列易崩溃 | 序列级粒度粗，多轮/细粒度信用分配需 GSPO-token | 需动态采样（多采样）+ 多项超参 | 权重裁剪后仍有偏差；文献较少 | 无 off-policy 校正，样本效率依赖 k |

## 二、各自核心公式

<details>
<summary><b>GRPO：基线算法</b></summary>

对每个问题 $q$ 采样 $G$ 条回答 $\{o_i\}_{i=1}^G$，组内归一化奖励作为优势（所有 token 共享同一优势）：

$$\hat{A}_{i,t}=\hat{A}_i=\frac{R_i-\text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

目标函数（PPO 式 min-clip + KL 惩罚）：

$$\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\big)\hat{A}_{i,t}\Big)-\beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\right]$$

其中 token 级重要性比 $r_{i,t}(\theta)=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}$。

- 优点：去掉价值模型，省内存（4 模型→3 模型）；组内归一化免去 critic。
- 隐藏问题（后面几个变体都是冲着它来的）：**每个 token 的 IS 比只基于单个采样**，按重要性采样原理这不是分布校正而是噪声，随序列变长累积，被裁剪放大后导致不可逆的模型崩溃。
</details>

<details>
<summary><b>GSPO：序列级优化（Qwen3 的算法）</b></summary>

GSPO 的核心论点：**奖励是给整条序列的，优化单元就必须是整条序列**。token 级 IS 比在数学上不成立（每个位置只有一个样本，无法做分布校正），且是 MoE 模型 RL 崩溃的根源。

序列级重要性比（长度归一化，控制数值范围）：

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{old}}(y_i\mid x)}\right)^{1/|y_i|}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}\mid x,y_{i,<t})}\right)$$

目标函数：

$$\mathcal{J}_{\text{GSPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i(\theta)\hat{A}_i,\ \text{clip}\big(s_i(\theta),1-\varepsilon,1+\varepsilon\big)\hat{A}_i\Big)\right]$$

- 梯度上等价于：**同一回答内的所有 token 等权** $\nabla\log\pi_\theta$（GRPO 是每个 token 用各自的权重，权重差异会累积出问题）。
- 因为 $s_i$ 数值量级小，裁剪范围也小（论文用 3e-4/4e-4，GRPO 是 0.2/0.27）。
- 附加收益：① **MoE 训练不再需要 Routing Replay**（token 级路由波动不再干扰 IS 比）；② 只依赖序列似然，对训练/推理引擎精度差异不敏感，**可直接用推理引擎返回的似然做优化**，省掉训练引擎重算，简化 RL 基础设施。
- 附赠 GSPO-token 变体：保留序列级 IS 比、但允许 token 级优势定制（数值上与 GSPO 等价）。
</details>

<details>
<summary><b>DAPO：GRPO 的四个工程补丁</b></summary>

目标函数（注意聚合是 token 级、裁剪是不对称的）：

$$\mathcal{J}_{\text{DAPO}}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta),1-\varepsilon_{low},1+\varepsilon_{high}\big)\hat{A}_{i,t}\Big)\right]$$

四项关键技术（每项在 AIME24 上贡献数分）：

| 技术 | 解决什么 | 做法 |
|---|---|---|
| **1. Clip-Higher** | 熵塌缩（策略过早确定性） | 上下裁剪解耦，抬高 $\varepsilon_{high}$（如 0.2/0.28），给低概率"探索 token"留出提升空间 |
| **2. Dynamic Sampling** | 全对/全错样本优势=0 → 零梯度、梯度缩减 | 过采样并过滤掉 acc=1 和 acc=0 的组，保证 batch 里全是有效梯度样本 |
| **3. Token-Level Loss** | 长 CoT 下 sample 级平均让长回答中高质量 token 被稀释、垃圾 token 逃过惩罚 | 分母从 $\frac{1}{G}\frac{1}{\|o_i\|}$ 改为 $\frac{1}{\sum\|o_i\|}$，每个 token 等权 |
| **4. Overlong Reward Shaping** | 超长截断样本被硬惩罚 → 奖励噪声 | 用长度感知的**软惩罚**（0 → 线性负 → -1），并支持直接 mask 超长样本 |

- 去掉了 KL 惩罚（长 CoT 场景不需要贴近参考策略）。
- 成绩：Qwen2.5-32B 上 AIME24 达 50 分（DeepSeek-R1-Zero 同基座为 47，naive GRPO 仅 30）。
</details>

<details>
<summary><b>CISPO：裁剪权重而非裁剪目标（MiniMax-M1）</b></summary>

CISPO 的诊断：PPO/GRPO 的 `min-clip` 代理目标在 token 越出裁剪区间时**梯度归零**，导致低概率但关键的长 CoT token（如 "Aha"、"Wait" 等反思信号）被丢弃。

做法：保留 token 级 IS 比，但把裁剪后的权重当作 **detach（stop-gradient）系数** 乘在 $\log\pi_\theta$ 上：

$$\hat{r}_{i,t}(\theta)=\text{clip}\left(r_{i,t}(\theta),\ 1-\epsilon_{low}^{IS},\ 1+\epsilon_{high}^{IS}\right),\qquad \epsilon_{low}^{IS}\approx 0\ \text{（下裁剪基本禁用）}$$

$$\mathcal{J}_{\text{CISPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\underbrace{\text{sg}\left[\hat{r}_{i,t}(\theta)\right]\cdot\hat{A}_{i,t}}_{\text{标量系数}}\cdot\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\right]$$

- 梯度始终流过 $\log\pi_\theta$，**每个 token 都参与更新**，同时裁剪约束了权重放大倍数（方差上界）。
- 形式上介于 GRPO 和 REINFORCE 之间：保留组内归一化优势 + token 级 IS 权重，但目标是 REINFORCE 式的。
- 论文报告：比 GRPO 收敛更快、波动更小；AIME 上比 GRPO/DAPO 更早达到同样精度。
</details>

<details>
<summary><b>RLOO：最朴素的 REINFORCE 多采样扩展（Cohere）</b></summary>

RLOO 的立场：在 RLHF 场景（强预训练初始化、概率质量高度集中）下，PPO 的高方差担忧、裁剪、价值模型、partial completion 建模**都是多余的**。把整段生成当做一个"动作"（bandit 形式），直接用 REINFORCE 估计器：

$$\mathcal{L}_{\text{RLOO}}=\frac{1}{k}\sum_{i=1}^{k}\left[R(y^{(i)},x)-\frac{1}{k-1}\sum_{j\neq i}R(y^{(j)},x)\right]\nabla_\theta\log\pi_\theta(y^{(i)}\mid x)$$

- 优势基线是 **leave-one-out**：样本 $i$ 的基线 = 其余 $k-1$ 个样本奖励的均值，参数无关、无偏、逐 prompt 实时构造（比全局滑动平均基线更有效）。
- 奖励采用 KL-shaped 形式 $R(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$。
- 无 IS 比、无裁剪、无价值网络——最省内存（比 PPO 少加载 1-2 个模型）。
- 论文结论：$k=2$ 的 RLOO 就能匹配/超过 $k=4$ 的 RAFT；全面胜过 PPO、DPO、RAFT，且对奖励噪声和 KL 权重更鲁棒。
- 局限：没有 off-policy 校正（mini-batch 多次更新时理论上有偏），主要靠 k 提升样本效率。
</details>

## 三、关键差异：一张"灵魂"对比

### 1. 裁剪哲学（clip 的战场）

| | 裁剪对象 | 越界后梯度 | 类比 |
|---|---|---|---|
| GRPO | token 级 IS 比（min-clip） | 归零 | "踩刹车" |
| DAPO | 同 GRPO 但不对称（抬高上界） | 归零 | "只刹一边" |
| CISPO | token 级 IS 比（detach 后做系数） | **永不归零，只是被限制放大倍数** | "装限流器" |
| GSPO | 整条序列的 IS 比 | 整条回答被排除 | "整单审查" |
| RLOO | 不裁剪 | — | "没刹车" |

### 2. 重要性采样（IS ratio）立场

- **GRPO / DAPO / CISPO**：token 级 IS 比 $\pi_\theta(o_t)/\pi_{\theta_{old}}(o_t)$。GSPO 认为这在数学上是**错用**——每个位置只有一个观测样本，无法完成分布校正，只是引入随序列长度累积的方差噪声。
- **GSPO**：序列级 IS 比 $\big(\pi_\theta(y)/\pi_{\theta_{old}}(y)\big)^{1/|y|}$，符合重要性采样理论（整条序列才是被采样的对象）。
- **RLOO**：完全不用 IS 比，走 on-policy 路线（这也意味着它不能像 GRPO/DAPO 那样在多次 mini-batch 更新里安全复用旧样本）。

### 3. 损失聚合

- GRPO：**sample 级**——每条回答等权 → 长回答的每个 token 贡献被稀释（DAPO 指出的问题）。
- DAPO：**token 级**——每个 token 等权 → 长 CoT 下高质量 token 不被稀释、垃圾模式被有效惩罚。
- GSPO：**序列级**——一条回答一份，配合长度归一化 IS 比。
- RLOO：**序列级**——每条回答就是一份 $\log\pi$（天然 bandit）。

### 4. 各自真正解决的问题（一句话）

| 算法 | 一句话定位 |
|---|---|
| GRPO | 在 PPO 基础上去掉 critic，用组内相对奖励做无价值模型 RL |
| GSPO | 治 GRPO 的"病根"：token 级 IS 失效 → 改为序列级，稳定 MoE、简化基建 |
| DAPO | 治 GRPO 的"症状"：熵塌缩、零梯度组、长序列失衡、超长噪声（4 个补丁） |
| CISPO | 治 GRPO/PPO 裁剪的副作用：关键 token 梯度被丢弃 → 裁剪权重而非目标 |
| RLOO | 论证 RLHF 不需要 PPO 那套复杂装置，最简 REINFORCE + LOO 基线即可 |

### 5. 血缘关系图

```
PPO（clip + critic + GAE）
 ├─ GRPO：去 critic，组内优势 + token 级 min-clip ──┬─► DAPO：+4 个工程补丁（仍 token 级）
 │                                                  └─► CISPO：改裁剪方式（仍 token 级）
 │
 └─► GSPO：把 IS 比/裁剪/奖励全部升到序列级（对 GRPO 的"革命"）

REINFORCE（无裁剪、无 IS）
 └─► RLOO：多采样 leave-one-out 基线（最简路线）
```

## 四、实践选型建议

- **要复现 DeepSeek R1 风格的长 CoT 推理 RL、且在意稳定性和基建简单** → **GSPO**（Qwen3 已量产验证，MoE 友好）。
- **想在现有 GRPO 框架上逐步加料、追求已知的精确增益** → **DAPO**（四项技术可拆开用，verl 有开源实现；注意动态采样会多花采样时间）。
- **长 CoT 训练中发现模型"变笨/反思消失"、怀疑裁剪丢了关键 token** → **CISPO**（改动最小，只换损失函数形式）。
- **做 RLHF（偏好/对齐）而非推理 RLVR，想要最少内存 + 最少超参** → **RLOO**（Cohere 验证在 summarization/dialogue 上全面优于 PPO/DPO/RAFT）。
- **基线对照**：GRPO 仍是最广泛使用的"默认"算法（DeepSeekMath/R1、各路开源框架默认），但长序列、大模型场景已普遍发现需要上述补丁或替换。

## 五、参考资料

| 算法 | 论文 |
|---|---|
| GRPO | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv:2402.03300 |
| GSPO | Group Sequence Policy Optimization, arXiv:2507.18071 |
| DAPO | DAPO: An Open-Source LLM Reinforcement Learning System at Scale, arXiv:2503.14476 |
| CISPO | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention, arXiv:2506.13585 |
| RLOO | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs, arXiv:2402.14740（RLOO 原始思想见 Kool et al., "Buy 4 REINFORCE Samples, Get a Baseline for Free!", ICLR 2019） |

需要我针对其中某一个给出**可直接运行的损失函数 PyTorch 伪代码**、或者在 verl / NeMo 框架里的配置对比吗？

---

### 🧑‍💻 User

reasoningrl非常适合grpo是因为直接利用同一个 prompt 下的 multiple samples 构造相对 advantage吗？因为同一道题比较多个 reasoning trajectories 天然提供了一个 baseline

---

### 🤖 Assistant

你的直觉是对的，而且抓到了最核心的一点。不过我想把它精确化一下——"同一 prompt 下采样多个 trajectories"之所以是关键，不只是因为它"顺便给了个 baseline"，而是因为它给了一个**逐 prompt 条件化的组内基线**。这一点对 reasoning 类任务特别值钱，但理由比"相对比较"要更具体。分几层说。

## 1. 精确化：这个 baseline 好在哪

GRPO 的组内优势是：

$$\hat{A}_i=\frac{R_i-\text{mean}\big(\{R_j\}_{j=1}^G\big)}{\text{std}\big(\{R_j\}_{j=1}^G\big)},\qquad \text{其中 }\{R_j\}\text{ 是同一道题 }q\text{ 的 }G\text{ 条回答}$$

它的本质是**在题目内部做"配对比较"（matched/within-subject design）**。好处的关键不在"有 baseline"，而在 baseline **自动控制了题目难度**：

- 一道题很难 → 所有轨迹的绝对奖励都低；一道题很简单 → 都高。
- 如果用一个全局的价值函数 $V(q)$ 做 baseline，$V(q)$ 必须自己学会"这道题难不难"，学不好就会把难度波动当成动作的优劣（高方差、慢收敛）。
- 组内归一化直接把这些**难度项减掉了**——剩下来的才是"在同样难度下，这条轨迹相对好不好"的信号。

所以更准确的说法是：**组内相对优势 = 一个免费的、逐题难度校准过的 baseline**，它替代了 critic 最重要的功能（减方差），却不需要训练价值网络。这正是 GRPO 能删掉 critic 的前提。

## 2. 但 reasoning 适合 GRPO，原因不止这一个

"多采样给 baseline"是**必要但非充分**条件。能让"多采样 + 无 critic"这套组合跑起来，reasoning 还提供了另外几个天然条件：

| 条件 | 为什么 reasoning 天然满足 | 为什么重要 |
|---|---|---|
| **可验证奖励（RLVR）** | 数学/代码题有判题器，奖励免费、确定、无 RM | GRPO 每题要采 $G=8\sim16$ 条，奖励必须廉价；学出来的 reward model 又贵又噪 |
| **序列级结果奖励** | 只看最终答案对不对（outcome reward），不给中间步骤打分 | 和"组内相对"天然对齐——整条轨迹一个优势值，无需做 token 级 credit assignment |
| **长 horizon** | CoT 动辄几千 token | token 级价值函数在长序列上极难学准，删掉 critic 反而是优势而非妥协 |

这三条里，**可验证奖励是最关键的一条**。如果没有便宜可靠的奖励，多采样成本会爆炸，组内基线再好也用不起。换句话说：reasoning 适合 GRPO，是因为它同时满足了 GRPO 的两个需求——"信号怎么来"（可验证奖励）和"方差怎么降"（组内配对基线）。

## 3. 你的说法里有一个需要小心的点：组内基线的精度

"比较多个 trajectories 天然提供了一个 baseline"——完全正确，但要注意这个基线**并不是无偏的**，而且信号粒度有退化：

**(a) 有偏 vs 无偏。** GRPO 减的是含 $R_i$ 自己的组均值，而 RLOO 减的是 leave-one-out 均值 $\frac{1}{k-1}\sum_{j\ne i}R_j$（不含自己 → 无偏）。GRPO 的做法相当于 $R_i$ 被轻微收缩，引入一点偏差换一点方差下降；`÷ std` 则纯粹是尺度启发式。两者共享同一个"多采样 = 免费基线"的直觉，只是估计器细节不同。

**(b) 二值奖励下的退化。** 数学题的奖励常常只有 0/1。这时组内相对优势**退化成"对 vs 错"的符号**——你无法在"两条都对的轨迹"之间比较谁更好（它们优势值相同）。所以组内基线在 reasoning 里更多是"过滤掉难度、指出哪条错了"，而**不是精细排序**。真正精细的排序（比如 process reward / 逐步打分）恰恰是这个框架给不了的。

**(c) 全对 / 全错时基线失效。** 一道题采 $G$ 条若全对，$\hat A_i=0$ 全组零梯度；全错同理。这就是 DAPO 的 Dynamic Sampling 要过滤的情形。也就是说，组内基线**只在组内有"对错分化"时才产生学习信号**，它假设了采样组总是"部分对、部分错"。

## 小结

你的判断可以修正为一句更精确的话：

> 组内相对优势之所以适合 reasoning RL，是因为**同一道题的多次采样提供了一个逐题难度校准的免费基线**，从而省掉了 critic；而 reasoning 反过来又提供了**可验证的、廉价的序列级奖励**，让这种"多采样换方差"的策略负担得起。

两者是互相成全的：组内配对基线让"无 critic"可行，可验证奖励让"多采样"可行。缺任何一个，GRPO 这套在 reasoning 上的效率优势都会大打折扣——这也是为什么 GRPO 在需要学习型 reward model 的偏好对齐任务里，优势远没有在 RLVR 里那么明显。