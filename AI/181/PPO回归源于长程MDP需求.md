# PPO回归源于长程MDP需求

### 🧑‍💻 User

实际上 agentic 强化学习的确有很多理由返回 PPO：1. infra 上，PPO 不需要 decode，只需要 prefill，只增加了 1/4 的计算2. 计算能力上，相当多的 bad pattern（错误 toolcall 等）是应该可以被 critic model 一次前向传播计算的3. agentic RL 越来越长程、越来越近似经典的 MDP，没有理由继续使用 AI4MATH 中 contextual bandit 式的 GRPO 型算法
为什么工业界近几年基模rl经历了从ppo->grpo->ppo范式的转变？

---

### 🤖 Assistant

你提出的这个观察非常敏锐：工业界基座模型 RL 训练的确走过了一条**PPO → GRPO → 再回到 PPO** 的路径，每一次转向背后都有深刻的工程与理论动机。结合你给出的三个理由，我们可以把这条演变脉络拆解成三个阶段来看。

---

### 1. 第一阶段：PPO 的早期统治（InstructGPT 时代）
当 RLHF 刚被大规模应用时，PPO 是自然的选择：
- **经典 MDP 设定**：语言生成被看作序列决策，每步输出一个 token，最后得到奖励。
- **需要价值函数（Critic）**：除了策略模型（Actor），还要训练一个 Critic 来估计状态价值 $V(s)$，用于计算优势函数和裁剪更新。
- **痛点**：需要同时维护两个大模型（或共享 backbone 但增加 Value head），内存和计算开销大，Critic 训练容易不稳定，且当时任务以单轮指令跟随为主，上下文较短，PPO 的 token‑level credit assignment 优势并不明显。

### 2. 第二阶段：GRPO 异军突起（DeepSeek‑R1 为代表）
当数学推理、代码生成这类“单轮长文本生成”任务成为 RL 对齐的主战场时，GRPO 展示出巨大吸引力，迅速取代了 PPO：
- **任务本质更接近 Contextual Bandit**：给一个 prompt，生成一个完整 answer，仅末尾一个奖励，没有多轮工具交互和状态转移。
- **抛弃 Critic，用 Group Sampling 做 baseline**：对同一个 prompt 采样多条响应，用组内平均奖励作为基线计算优势。这样只需要一个 Policy 模型，省去了整个 Critic 的训练和存储。
- **计算量反而更小**：虽然要采样多条响应，但在单轮生成、中等长度场景下，这种“并行采样 + 归一化”的开销远小于额外训练一个大 Critic，而且避免了 Critic 欠拟合、高方差等问题。
- **工程极简、训练稳定**：GRPO 的 clip 仅在组内做相对比较，调参容易，收敛快，尤其适合数学/代码等正确答案唯一、奖励信号明确的任务。

**结论**：在“单轮、长文本、无工具交互”的 Post-Training 场景中，GRPO 用极简的设计换来了巨大的效率提升，因而快速流行。

---

### 3. 第三阶段：Agentic RL 呼唤 PPO 回归
当 RL 的应用从“生成最终答案”拓展到 **Agentic 场景**（多轮工具调用、思考-行动-观察循环、长程规划）时，局面发生了质变：

#### a) 任务从 Contextual Bandit 回归到完整 MDP
Agent 任务由一连串的 **思考 - 调用工具 - 获取结果 - 再思考** 组成，是一个典型的长序列 MDP，具有延迟奖励和部分可观测性。此时 GRPO 的 **组内归一化 baseline** 会出现严重问题：
- GRPO 需要针对同一个 prompt 采样多条**完整的轨迹**，然后计算组平均奖励。在 Agent 场景下，单条轨迹可能长达数千 tokens，采样多条去算一个 advantage 的成本爆炸。
- 而且它把整条轨迹的奖励直接分摊到每个 token 上做归一化，**credit assignment 极其粗糙**：中间的某个错误工具调用可能要到很后面才表现出奖励下降，GRPO 很难准确归因。

#### b) Critic 模型的价值重新凸显（就是你提到的第 2、3 点）
- **Error Pattern 的即时评估**：很多 Agent 错误模式（格式错误、幻觉调用、死循环）在发生的那一步就能被识别。一个训练良好的 Critic 可以**一次前向传播**就给出极低的价值估计，把这些“坏 token”在优势计算时直接惩罚。这比等一条完整轨迹走完再靠奖励信号高效得多。
- **Token‑Level Credit Assignment**：Critic 能提供每个时间步的状态价值 $V(s_t)$，配合 GAE 可以准确分解延迟奖励，这对长程规划任务的稳定训练至关重要。

#### c) 基础设施层面的计算优势（你提到的第 1 点）
在 Agentic RL 的 rollout 中，模型通常会同时扮演**生成者（解码）**和**评估者（prefill）**两个角色：
- **PPO 的 Critic 只需做 prefill**：给定一条（或多条）已经生成的轨迹，Critic 可以**并行 prefill** 整条序列，一次性输出所有 token 的 $V(s_t)$。这通常只增加约 **1/4 的额外计算**（相对于 Actor 的 decode），远小于重采样多条轨迹的开销。
- **GRPO 的隐性代价被放大**：为了计算组 baseline，GRPO 必须**重新采样多条完整轨迹**，每条都需要完整的自回归 decode。在长程 Agent 任务里，这个 decode 成本远超 Critic 的 prefill 成本，彻底翻转了“GRPO 更省计算”的旧认知。

#### d) 业界实践的回摆
目前无论是开源（如 Meta 的 Llama‑Agent 相关研究、Qwen‑Agent 的后训练优化）还是闭源团队，在构建 Agent 训练管线时，都在重新引入 **Actor‑Critic 架构**：
- 用共享 backbone 降低 Critic 训练内存（PPO 的 Value head 只占极少参数）。
- 利用 Critic 的即时反馈减少无效 rollout（例如发现某步工具调用格式错误就提前截断）。
- 回归到经典的 GAE + PPO，利用其成熟的 MDP 优化理论来处理长程信用分配。

---

### 总结：范式转变的底层逻辑
```
PPO → GRPO：在单轮生成（Contextual Bandit）场景下，
           “去掉 Critic + 组采样” > “维护 Critic 的额外成本”
           因此 GRPO 更简单、更高效。

GRPO → PPO：在 Agentic 长程 MDP 场景下，
           “重采样多条完整轨迹”的成本 >> “额外加入一个只做 prefill 的 Critic”的成本，
           且 Critic 的即时错误检测和 token‑level credit assignment 成为刚需。
```

所以，**不是 PPO 本身被“淘汰”过一次，而是 RL 的应用边界发生了变化**。当基座模型的 RL 逐步从简单的单轮后训练，演进到复杂的 Agent 系统时，PPO 天生的 MDP 基因让它重新成为不可替代的选择。你给出的三条 infra / 计算 / 建模理由，恰好是这次回归最核心的推动力。