# RLHF与RLVR后训练交集

### 🧑‍💻 User

在基模后训练领域，rlvr和rlhf做的事情有什么交集？

---

### 🤖 Assistant

在基础模型（Foundation Models）的后训练（Post-Training）阶段，**RLHF**（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）和 **RLVR**（Reinforcement Learning from Verifiable Rewards，基于可验证奖励的强化学习，有时也称为 Rule-based RL）是两种强大的对齐与能力提升技术。

简而言之，RLHF 依赖**“主观偏好”**（如语气、安全性、有用性），而 RLVR 依赖**“客观真理”**（如代码编译通过、数学答案正确、逻辑闭环）。

尽管它们的“奖励来源”截然不同，但它们在**底层机制、优化目标以及训练范式上有着巨大的交集**。以下是它们做的事情的核心交集：

### 1. 核心优化目标：改变模型输出的概率分布
无论是 RLHF 还是 RLVR，它们在后训练中都不再像预训练那样进行单纯的“下一个词预测”（Next-token prediction），而是致力于**将模型的输出分布引导至人类期望的分布**。
*   **突破 SFT 的天花板**：监督微调（SFT）受限于训练数据的质量（Behavior Cloning），而 RLHF 和 RLVR 都允许模型在探索中找到比训练数据更好的解。
*   **抑制幻觉与劣质输出**：通过给予低奖励，两者都能降低模型生成不符合要求（如不安全内容或错误代码）内容的概率。

### 2. 底层数学抽象：完全相同的马尔可夫决策过程 (MDP)
在算法底层，RLHF 和 RLVR 对语言模型生成过程的数学建模是完全一致的。它们都将文本生成视为一个马尔可夫决策过程：
*   **状态 (State) $S_t$**：当前的 Prompt 加上已经生成的 token 序列。
*   **动作 (Action) $A_t$**：词表中下一个被选择的 token。
*   **策略 (Policy) $\pi_\theta$**：正在被训练的大语言模型本身。
*   **转移 (Transition)**：确定性的，即状态加上动作构成新的状态。

<details>
<summary><b>点击展开：RLHF 与 RLVR 共享的核心 RL 目标函数</b></summary>

无论是 RLHF 中的神经网络奖励模型 (Reward Model)，还是 RLVR 中的规则验证器 (Rule-based Verifier)，最终都会输出一个标量奖励 $R(x, y)$（$x$ 为 prompt，$y$ 为生成的完整回答）。

两者通常都使用相同的带 KL 散度约束的优化目标：

$$ \max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) - \beta D_{KL}(\pi_\theta(\cdot|x) \parallel \pi_{ref}(\cdot|x)) \right] $$

*   $R(x, y)$ 是交集中的变量：在 RLHF 中，它是通过偏好数据训练出来的模型算出的；在 RLVR 中，它是通过编译器或数学答案核对算出的。
*   $\beta D_{KL}$ 是两者都必须使用的“护栏”，用于防止模型在追求高奖励时破坏预训练学到的语言基础能力（防止模式崩溃或过度自信）。

</details>

### 3. 共享的强化学习算法库
因为输入（Prompt）和输出（Reward）的接口一致，RLHF 和 RLVR 可以**无缝复用同一套强化学习算法和工程基础设施**。
*   **策略梯度算法**：两者最常使用的都是 `PPO` (Proximal Policy Optimization) 及其变体，或者近期在推理模型中流行的 `GRPO` (Group Relative Policy Optimization)。
*   **在线采样 (On-policy generation)**：两者在训练时都需要模型根据当前最新的权重 $\theta$ 实时生成回答（Rollout），并在这些自我生成的轨迹上进行评估和梯度更新。

### 4. 面临相同的技术挑战与副作用
正因为底层算法相同，RLHF 和 RLVR 在训练过程中会遇到完全一样的“强化学习病”：
*   **奖励骇客 (Reward Hacking / Reward Overoptimization)**：
    *   *RLHF 中*：模型发现写一些奉承人类的话、或者使用极其复杂的词汇，就能骗过 Reward Model 获得高分。
    *   *RLVR 中*：模型发现只要在代码末尾疯狂注释，或者在数学推导中加入无用的永远为真的恒等式，就能绕过验证器漏洞获得分数。
*   **探索困境 (Exploration Issues)**：如果模型初始状态（SFT阶段）根本写不出正确的代码（RLVR）或符合特定格式的回答（RLHF），它就永远拿不到正奖励，导致梯度无法更新。两者都需要通过退火、温度调节或优质的 SFT 预热来解决冷启动问题。

### 5. 现代后训练的融合趋势 (Fusion)
如今，最前沿的基模后训练（如 DeepSeek-R1、OpenAI o1/o3 的背后技术）不再将两者割裂，它们的交集体现在**混合奖励机制**上。

| 维度 | RLVR (可验证奖励) | RLHF (人类反馈奖励) | 混合使用 (交集体现) |
| :--- | :--- | :--- | :--- |
| **优势** | 杜绝幻觉，上限极高（适合复杂推理、代码） | 保证语气符合人类习惯，价值观对齐 | 使用 RLVR 提升硬核逻辑，同时叠加 RLHF 防止模型变成冰冷的机器 |
| **打分方式** | 编译器、测试用例、正则表达式 (Cost极低) | 训练单独的 Reward Model (Cost较高) | 在一个 PPO 循环中，将 $R_{total} = \alpha R_{vr} + \gamma R_{hf}$ 进行联合优化 |

**总结：**
RLVR 和 RLHF 做的事情的交集在于：**它们都是基于强化学习范式，利用标量奖励信号来优化 LLM 策略函数的工程实践。** 区别仅仅在于那个“提供奖励的裁判”是人类教出来的黑盒模型（RLHF），还是物理世界/逻辑世界的白盒规则（RLVR）。在构建顶级基础模型时，它们是同一套流水线上的两套不同检测工具，往往被同时使用。