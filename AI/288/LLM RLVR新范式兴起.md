# LLM RLVR新范式兴起

### 🧑‍💻 User

以deepseek grpo为代表的rlvr为什么正在成为当今llm rl新范式，相比经典的ppo算法有什么优点？

---

### 🤖 Assistant

DeepSeek-R1 的发布不仅展示了开源模型在推理能力上逼近闭源顶尖模型（如 o1）的潜力，更让其背后的训练算法 **GRPO (Group Relative Policy Optimization)** 以及 **RLVR (Reinforcement Learning with Verifiable Rewards)** 这一概念成为了业界的焦点。

这种新范式正在逐渐取代传统的 RLHF（基于人类反馈的强化学习）流程中的 PPO 算法，主要原因在于它解决了 LLM 尤其是推理类模型在训练时的**计算效率瓶颈**和**奖励信号模糊**问题。

以下从原理、资源效率和效果三个维度，深度解析为什么 GRPO/RLVR 正在成为新范式，并对比其与 PPO 的优劣。

---

### 1. 核心差异概览

在深入细节之前，我们先用一个表格对比 **DeepSeek GRPO** 与经典 **PPO**：

| 特性 | PPO (Proximal Policy Optimization) | GRPO (Group Relative Policy Optimization) |
| :--- | :--- | :--- |
| **核心机制** | 依赖 **Value Function (Critic)** 进行优势估计 (GAE) | 依赖 **Group Normalization** (组内相对优势) |
| **模型数量** | **4个** (Actor, Critic, Ref Model, Reward Model) | **2个** (Actor, Ref Model) *甚至 Ref 可卸载* |
| **显存占用** | 极高 (Critic 需与 Actor 同规模) | **低 (省去 Critic，节省约 50% 显存)** |
| **奖励来源** | 主要是神经网络 (Reward Model) 预测的分数 | 主要是规则/环境验证 (RLVR) + 组内对比 |
| **主要应用** | 风格对齐、闲聊、减少有害性 (RLHF) | 逻辑推理、数学、代码、长思维链 (CoT) |
| **稳定性** | 对超参数极度敏感，训练曲线易崩塌 | 相对稳定，无需训练 Critic |

---

### 2. PPO 的痛点：为什么我们需要新范式？

在 ChatGPT (InstructGPT) 时代，PPO 是绝对的王者。但随着模型参数变大（70B+）以及从“聊天”转向“深度推理”，PPO 暴露出了巨大的局限性：

1.  **显存噩梦 (The VRAM Bottleneck)**：
    *   标准的 PPO 流程需要同时在显存中维护 4 个模型：
        1.  **Actor** (正在训练的模型)
        2.  **Critic** (价值网络，通常与 Actor 同规模，用于评估状态价值 $V(s)$)
        3.  **Reference Model** (冻结的旧模型，用于计算 KL 散度)
        4.  **Reward Model** (奖励模型)
    *   对于一个 70B 的模型，仅加载这 4 个模型就需要巨大的显存集群，导致训练成本极高，且通信开销大。

2.  **Critic 的训练难度**：
    *   PPO 依赖 Critic 来计算 Advantage（优势函数）。如果在推理任务（如长链条数学证明）中，Critic 很难准确判断中间步骤的好坏（Value Function 很难拟合），导致 Critic 提供的信号充满噪声，进而带偏 Actor。

---

### 3. GRPO 的核心创新：去掉了 Critic

DeepSeek-R1 论文中提出的 GRPO 最大的贡献在于：**它通过“组内归一化”完全摒弃了 Critic 模型。**

#### GRPO 的工作原理
GRPO 不再训练一个神经网络来预测“当前这步有多好”，而是采用**以量取胜、相对比较**的策略：

1.  **Group Sampling (组采样)**：对于同一个 Prompt $q$，让模型生成 $G$ 个不同的输出 $\{o_1, o_2, ..., o_G\}$。
2.  **Reward Calculation**：计算这 $G$ 个输出各自的奖励 $\{r_1, r_2, ..., r_G\}$。
3.  **Advantage Estimation (优势估计)**：
    不使用 Critic 预测的基线，而是直接计算这组奖励的**平均值**和**标准差**。第 $i$ 个输出的优势 $A_i$ 为：
    $$
    A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}
    $$
4.  **Policy Update**：鼓励那些优于组内平均水平的输出，抑制低于平均水平的输出。

#### 为什么这是巨大的优势？
*   **计算效率革命**：去掉了 Critic，意味着少了一个巨大的模型需要训练和推理。显存占用大幅下降，通信量减少，使得在有限资源下训练超大模型成为可能。
*   **基线自适应**：组内平均值是一个天然的、动态的 Baseline。对于难的问题，大家分都低，高于平均就是好；对于简单问题，大家分都高，必须更好才行。

---

### 4. RLVR 的崛起：从“讨好人类”到“追求真理”

GRPO 只是算法手段，**RLVR (Reinforcement Learning with Verifiable Rewards)** 才是这一波范式转移的核心场景。

#### 传统 RLHF vs. RLVR
*   **传统 RLHF (Human Feedback)**：
    *   目标：让回答“像人喜欢的样子”。
    *   奖励：来自于一个模仿人类偏好的 Reward Model (RM)。
    *   问题：RM 是个黑盒神经网络，不仅有准确率上限，还容易被 Hack（模型学会输出一些无意义但 RM 给高分的句子）。
*   **RLVR (Verifiable Rewards)**：
    *   目标：让回答“正确”。
    *   奖励：**确定性的规则**。
        *   代码：能否通过编译器？能否通过单元测试？
        *   数学：最终答案是否匹配 Ground Truth？
        *   格式：是否严格遵循了 XML 标签（如 `<think>`）？
    *   优势：**奖励信号纯净，没有噪声**。模型在确定性反馈下，能够学会复杂的 **Chain-of-Thought (CoT)** 策略。

#### DeepSeek-R1 的“Aha Moment” (顿悟时刻)
正是结合了 GRPO（允许大规模采样探索）和 RLVR（提供绝对客观的对错信号），DeepSeek 观察到了模型自发的**自我反思能力**。模型学会了在 `<think>` 标签内进行试错、回溯和验证，因为只有这样才能最大化通过率奖励。PPO 在这种长序列探索中往往因为 Critic 无法收敛而失败。

<details>
<summary><strong>点击查看 GRPO 目标函数数学细节</strong></summary>

GRPO 的目标函数如下：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \text{clip}\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{KL}(\pi_{\theta} || \pi_{\text{ref}}) \right) \right]
$$

其中：
- $G$ 是组的大小（例如 64）。
- $A_i$ 是通过组内归一化计算出的优势。
- $D_{KL}$ 是 KL 散度，防止模型偏离原始模型太远（DeepSeek 使用了特殊的 token-level KL 计算方式）。
- **关键点**：公式中完全没有 Value Function $V(s)$ 的参与。

</details>

---

### 5. 总结：GRPO 相比 PPO 的核心优点

1.  **更低的资源门槛**：去除了 Critic 模型，显著降低显存占用和计算量，使得训练长上下文（Long Context）的推理模型变得更加经济。
2.  **更加鲁棒的优化**：通过 Group Normalization 计算优势，避免了 Critic 训练初期估值不准导致的训练震荡。对于奖励稀疏的逻辑推理任务，这种类似“锦标赛”的对比机制更有效。
3.  **适配推理 scaling**：GRPO 天然适合与 Test-time Compute 结合。训练时采样多个路径进行对比，本质上是在训练模型学习“如何在推理时进行搜索”。
4.  **避免 Reward Hacking（在 RLVR 场景下）**：结合可验证奖励，GRPO 直接优化客观指标，而不是优化一个并不完美的 Reward Model，从而真正提升模型智力。

**结论：**
如果说 PPO 是为了让模型**“说话好听”**（Chat/Style），那么 GRPO + RLVR 就是为了让模型**“脑子好使”**（Reasoning/Logic）。随着 LLM 发展重心从闲聊转向解决复杂问题，这种新范式成为主流是必然趋势。