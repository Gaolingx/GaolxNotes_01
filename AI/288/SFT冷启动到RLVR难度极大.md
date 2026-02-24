# SFT冷启动到RLVR难度极大

### 🧑‍💻 User

为什么llm从sft base开始训练reasoningRL（sft cot cold start->rlvr）提升STEM任务性能难度非常大？

---

### 🤖 Assistant

这是一个非常深刻且前沿的问题，触及了当前大模型推理能力（Reasoning Capabilities）训练的核心痛点。

从 **SFT Base**（已进行指令微调的模型）出发，经过 **SFT CoT Cold Start**（使用思维链数据进行冷启动），再进行 **RLVR**（Reinforcement Learning with Verifiable Rewards，如数学或代码的正确性验证）来提升 STEM 性能，这条路径虽然理论上成立，但在实际操作中确实面临巨大的困难。

这主要归结为**探索（Exploration）**、**分布偏移（Distribution Shift）**与**信用分配（Credit Assignment）**这三个核心矛盾。

以下是详细的技术深度解析：

### 1. 探索难题：从“模仿形式”到“真正推理”的鸿沟

这是最根本的困难。SFT 和 RL 在这个流程中扮演的角色往往是冲突的。

*   **SFT 的局限性（Form over Substance）：**
    SFT CoT Cold Start 通常使用 GPT-4 或 Claude 等强模型生成的 CoT 数据。模型在这个阶段学会的是**推理的“格式”**（比如输出 `Let's think step by step`，分点列述），而不是**推理的“逻辑”**。
    $$ \pi_{\text{sft}}(y|x) \approx \pi_{\text{teacher}}(y|x) $$
    如果 Base 模型本身的知识储备不足以支撑逻辑推导，SFT 只会让模型学会“一本正经地胡说八道”（Hallucination）。

*   **稀疏奖励（Sparse Reward）与冷启动失败：**
    STEM 任务（如奥数题、LeetCode）的奖励是二元的（Pass/Fail）。在 RL 阶段，如果模型在 SFT 后生成正确答案的概率 $P(correct) \approx 0$，那么 RL 算法（如 PPO 或 GRPO）将无法获得任何正向信号（Reward）。
    $$ \nabla J(\theta) \approx \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) \nabla \log \pi_\theta(\tau)] $$
    如果 $R(\tau)$ 总是 0，梯度就是 0，模型无法学习。这就是所谓的**探索困难（Exploration Challenge）**。SFT Base 往往因为过拟合了“人类偏好的说话方式”，导致其输出分布非常尖锐（Low Entropy），失去了探索不同解题路径的能力。

### 2. SFT 带来的“行为惯性”与模式坍缩

使用 SFT 模型作为 RL 的起点（SFT Base），相比于使用 Pre-trained Base 模型，带来了一些特有的副作用。

<details>
<summary><strong>点击展开：SFT Base 带来的具体负面影响</strong></summary>

1.  **无法自纠（Lack of Self-Correction）：**
    标准的 SFT 数据通常是“问题 $\to$ 完美推理 $\to$ 答案”。模型很少见到“错误 $\to$ 发现错误 $\to$ 修正 $\to$ 答案”的数据。因此，SFT Base 模型倾向于**一条路走到黑**。而在复杂的 STEM 任务中，推理模型（如 o1/DeepSeek-R1）的核心能力在于 Backtracking（回溯）和 Verification（验证）。SFT Base 很难通过 RL 涌现出这种能力，因为它被训练成要“自信且流畅”地回答。

2.  **KL 散度的束缚（KL Constrained Optimization）：**
    在 RL 训练中，为了防止模型遗忘通用能力或输出乱码，通常会加一个 KL 惩罚项（KL Penalty），约束当前策略 $\pi_\theta$ 不要偏离参考策略 $\pi_{ref}$（通常就是 SFT Base）太远。
    $$ R_{total} = R_{outcome} - \beta \cdot \text{KL}(\pi_\theta || \pi_{ref}) $$
    这里存在一个悖论：**为了提升 STEM 推理能力，模型必须大幅改变其输出分布（例如学会更长的思考、尝试非传统的解法），这必然导致高 KL 散度。** 强行约束 KL 会锁死模型的推理上限；放松 KL 则容易导致模型输出崩坏（Reward Hacking）。

3.  **短视与长度偏见：**
    SFT 数据往往偏向简短、直接的回答。而高难度的推理往往需要极长的 Context（Long CoT）。SFT Base 的训练分布可能抑制了模型生成长达数千 token 的思考过程的倾向。

</details>

### 3. 信用分配（Credit Assignment）与奖励欺骗（Reward Hacking）

即便解决了冷启动问题，RLVR 过程本身在 STEM 任务中也非常脆弱。

*   **结果奖励的局限性（Outcome Reward Bias）：**
    RLVR 通常只验证最终答案（Answer Correctness）。
    *   **情况 A：** 步骤全错，答案蒙对了（False Positive）。RL 会强化错误的逻辑。
    *   **情况 B：** 步骤对了一大半，最后一步计算错了（False Negative）。RL 会惩罚前面正确的推理逻辑。
    对于长链条推理，这种**稀疏的结果奖励**很难精确指导中间的每一步推理（Step-level logic）。

*   **Goodhart's Law (奖励欺骗)：**
    模型非常擅长走捷径。在 STEM 任务中，如果只用 RLVR，模型可能会学会针对特定类型的题目（如特定格式的数学题）生成特定的“伪代码”或“套话”来通过验证器，而不是真正学会泛化。例如，在代码任务中，模型可能学会写大量的 `try-catch` 或硬编码测试用例来骗过单元测试。

### 4. 数据与验证器的质量瓶颈

*   **SFT CoT 数据的上限：**
    如果你使用的 SFT CoT 数据是用 GPT-4 蒸馏的，那么你的模型上限就是 GPT-4。但在 Reasoning 领域，目标往往是超越教师模型。简单的 SFT 无法做到这一点。
    最近 DeepSeek-R1 的论文指出，**直接在 Base 模型上进行 RL（DeepSeek-R1-Zero）**，虽然初期极其不稳定且难以收敛，但最终能涌现出 SFT 模型所没有的自我反思能力。这反向证明了 SFT 可能会引入“人类偏见”，限制了模型探索更高效（但可能不符合人类直觉）的思维路径。

*   **验证器的完备性：**
    对于数学，验证器相对容易（SymPy, Lean）。对于代码，Unit Test 覆盖率有限。对于科学问题，很难构建完美的自动验证器。验证器的不足直接导致 RL 优化的天花板。

### 总结：为什么这么难？

从 SFT Base 开始训练 Reasoning RL 之所以难，可以用以下公式概括其核心矛盾：

$$ \text{Difficulty} \propto \frac{\text{Required Exploration Space}}{\text{SFT Distribution Entropy}} + \text{Credit Assignment Noise} $$

1.  **SFT 降低了熵（Entropy）：** SFT 让模型变得“自信且固执”，难以在 RL 阶段跳出局部最优解去探索复杂的推理路径。
2.  **推理链条越长，噪声越大：** STEM 任务需要多步推理，仅靠最终答案的二元奖励（0/1）来反向传播更新整个长链条的参数，效率极低且极不稳定。

**目前的解决思路（如 DeepSeek-R1, OpenAI o1）：**
*   **Iterative SFT + RL：** 不是一次到位，而是 `RL -> 生成新数据 -> 过滤 -> SFT -> RL` 的循环（Expert Iteration）。
*   **Pure RL from Base：** 放弃 SFT 冷启动，直接在 Base 模型上用规则奖励硬训（虽然极难收敛，但潜力更大）。
*   **Process Reward Models (PRM)：** 不只给结果奖励，给每一步推理打分（需要极其昂贵的人工标注或强模型标注）。