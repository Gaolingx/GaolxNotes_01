# MixedRL架构设计解析

### 🧑‍💻 User

mixedrl中，为什么 RLVR 部分用 GRPO 的 on-policy + group normalization（因为 verifier 信号干净，不需要 pool），通用任务部分用 response pool + contrastive RM，KL 项作为全局的安全约束保留，但不再是通用任务的唯一优化信号。

---

### 🤖 Assistant

在 MixedRL（混合强化学习，通常指将基于规则验证的逻辑推理任务与基于人类偏好的通用对话任务结合训练的框架）中，这种非对称的算法设计是非常精妙且符合不同任务本质特征的。

这种设计的核心逻辑在于：**不同类型任务的“奖励信号（Reward Signal）”在客观性、计算成本、噪声分布上存在根本差异，因此必须采用不同的优化策略。**

以下是针对你提到的三个核心部分（RLVR、通用任务、KL项）的深度拆解：

### 1. RLVR 部分：为什么用 GRPO (On-policy) + Group Normalization？

RLVR（Reinforcement Learning with Verifier Rewards）主要针对数学、代码等具有**客观标准答案**的推理任务。

*   **为什么不需要 Pool（经验回放池）且必须 On-policy？**
    *   **信号极度干净：** 基于规则的 Verifier（如代码编译器、数学公式评测器）给出的信号是 $0$ 或 $1$ 的绝对客观标准，不存在“判错”或“偏好摇摆”的噪声。
    *   **推理轨迹的分布偏移极快：** 在推理任务中，模型哪怕改变了一个中间推理步骤（Chain-of-Thought），结果可能截然不同。如果使用过去的 Response Pool（Off-policy），那些旧的错误推理路径对当前策略的指导意义极低（模型可能早就学会不那么错了）。因此，必须使用 On-policy 数据，让模型在**当前能力的边界上**试错。
    *   **验证成本极低：** 跑一次规则校验的成本远低于跑一次千亿参数的 Reward Model（RM），因此可以无所顾忌地实时生成、实时校验，不需要把旧数据存入 Pool 里节省算力。
*   **为什么用 GRPO + Group Normalization？**
    *   **省去 Critic 网络（Value Network）：** 传统的 PPO 需要一个与 Policy 模型一样大的 Critic 网络来预测 Advantage，导致显存翻倍。GRPO 通过对同一个 Prompt 生成 $G$ 个不同的回答，利用这 $G$ 个回答的分数在组内进行标准化（Group Normalization）：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$，直接得到了高质量的 Advantage，大幅节省显存。
    *   **强化探索与相对对比：** 组内标准化意味着即使一道题极难（所有回答得分都很低），只要有一个回答稍微好一点点，它的 Advantage 就是正的；反之亦然。这在信号干净的推理任务中，能极大地促进有效探索。

<details>
<summary>💡 展开查看：GRPO 在逻辑任务中的数学优势</summary>

在传统的 PPO 中，Advantage function $A_t$ 依赖于 Value model $V_\phi(s_t)$：
$$A_t = R_t - V_\phi(s_t)$$
由于 $V_\phi$ 本身存在拟合误差，在需要极长逻辑链（Long CoT）的数学任务中，Value 网络的累积误差会导致训练崩溃。

GRPO 直接舍弃 $V_\phi$，对同一 Prompt $q$ 采样出 $\{y_1, y_2, ..., y_G\}$，计算组内基线。
$$A_i = \frac{r(q, y_i) - \mu_r}{\sigma_r}$$
这种方式天然消除了 Prompt 难度带来的方差（baseline variance），且因为 Verifier 的 $r$ 是绝对正确的，所以组内对比的梯度方向永远是数学上最优的。
</details>

---

### 2. 通用任务部分：为什么用 Response Pool + Contrastive RM？

通用任务（如闲聊、写诗、价值观对齐）没有绝对的对错，只有**人类偏好**。这使得它的奖励信号具有主观性、高噪声和易被利用（Hackable）的特点。

*   **为什么要用 Response Pool（回答池）？**
    *   **防止 Reward Hacking（奖励作弊）：** 神经网络 Reward Model 存在严重的分布外（OOD）漏洞。如果仅用 On-policy 生成数据，模型很快会发现 RM 的漏洞（比如 RM 偏好“废话连篇”或某些特定句式），从而导致能力崩塌。引入历史 Response Pool，可以让当前回答与过去的回答进行对比，提供一个稳定的“锚点”，防止模型在错误的特征上狂飙。
    *   **增加数据多样性：** 通用任务需要多样的表达。Pool 里的历史数据提供了丰富的上下文和不同的生成模式，有助于维持模型的通用生成能力。
*   **为什么用 Contrastive RM（对比式奖励）？**
    *   **RM 的绝对分数不可靠：** 偏好模型（通常基于 Bradley-Terry 模型训练）本质上只擅长做**排序**（A 比 B 好），而不是打绝对分（A 是 80 分，B 是 60 分）。RM 给出的绝对分数往往存在严重的校准问题。
    *   **对比学习还原偏好本质：** 通过将当前的生成结果与 Pool 中的 Baseline 结果组合成对 $(y_{\text{current}}, y_{\text{pool}})$，输入 Contrastive RM 比较哪个更好，可以直接利用偏好模型的相对比较优势，产生更平滑、更鲁棒的梯度。

<details>
<summary>💡 展开查看：Contrastive RM 的机制</summary>

一般的 RM 目标函数是基于 Bradley-Terry 模型：
$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$
由于它是对比训练出来的，因此在强化学习阶段，如果我们直接用 $r_\theta(x, y)$ 作为绝对奖励喂给 PPO，会存在极大的偏差。

MixedRL 中采用的 Response Pool + Contrastive 机制，实际上是在构建实时的对弈：
模型当前生成的 $y_\text{policy}$ 与 Pool 中抽样的 $y_\text{baseline}$ 进行对比，计算 Contrastive Reward：
$$R_{\text{contrastive}} = r_\theta(x, y_\text{policy}) - r_\theta(x, y_\text{baseline})$$
只有当当前策略确实击败了过去的自己，才会获得正向梯度，这极大地稳定了人类偏好对齐的过程。
</details>

---

### 3. KL 散度项：为什么作为全局安全约束，而不是唯一优化信号？

在早期的强化学习微调（如InstructGPT阶段），人们通常把 KL 散度（$\text{KL}(\pi_\theta || \pi_\text{ref})$）作为唯一的惩罚项，目的是让模型“不要忘记通用能力”（即不要偏离预训练或 SFT 模型太远）。

*   **为什么不能只靠 KL 来维持通用能力？（Alignment Tax）**
    *   如果在 MixedRL 中，只用 RLVR 去疯狂优化数学和代码，然后仅仅用 KL 散度去限制它“不要把通用能力忘掉”，这会失败。因为 KL 散度是一个**被动的惩罚项**。随着 RLVR 的梯度不断累积，模型为了拿到极高的数学奖励，宁愿吃一点 KL 惩罚，也会逐渐改变全局的参数分布，最终导致通用对话能力退化（即“对齐税”）。
*   **为什么保留它作为“全局安全约束”？**
    *   在 MixedRL 中，通用任务已经有了自己主动的优化信号（Response Pool + Contrastive RM），数学任务也有了 RLVR。
    *   此时，KL 项退居幕后，变成了一个**兜底的安全护栏（Safety Constraint）**。它的作用不再是“维持通用能力”（这个任务交给了 RM），而是**防止模型产生病态的生成行为**，例如：
        1.  **防止 Mode Collapse（模式崩溃）：** 防止模型只会输出单一的套话。
        2.  **防止 Verbosity（冗长诅咒）：** RL 极度容易让模型产生越来越长的回答（因为长回答往往容易骗过 RM 或覆盖 Verifier 的采分点）。KL 散度可以有效约束输出长度和词汇分布，让它保持像一个正常人类语言。

### 总结

MixedRL 的这种设计是对“奖励工程”深刻理解的体现：
1.  对于**逻辑真理（RLVR）**，采用 **GRPO (On-policy)**：用最极限的探索、最新的试错，去逼近那条绝对正确的逻辑链。
2.  对于**人类主观（通用任务）**，采用 **Pool + Contrastive RM**：用最保守的对比、最稳定的基线，去讨好那个人类模糊的偏好审美，防止模型“钻空子”。
3.  对于**底层基础（KL Constraint）**，采用 **$\text{KL}$ 惩罚**：作为悬在头顶的达摩克利斯之剑，确保模型无论怎么优化，依然说的是“人话”。