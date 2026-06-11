# SFT是RL冷启动的关键

### 🧑‍💻 User

```
RL 不是凭空学习,它依赖 policy model 能 sample 出至少偶尔正确的轨迹。如果 policy 在某个任务上从不 sample 出正确解,reward 永远是 0,policy gradient 就是 0 向量,什么都学不到。
SFT 的作用是把 policy 推到"至少能偶尔做对"的初始分布。这就是 cold-start 的字面意思——给 RL 一个非零的起点。
小模型对这个初始分布特别敏感,因为它的探索能力差(softmax 更尖锐、entropy 更低)。大模型可以在 RL 中"自己摸索出"SFT 没教过的解法,小模型不行。
诊断信号:如果 RL on task X 训不动,first hypothesis 不是"调 lr / clip / KL coef",而是"我 SFT mixture 里是不是没有 task X 的样本"。回头补 SFT,再 RL。
```
这对于我们基模后训练reasoningrl/agenticrl有哪些启示？

---

### 🤖 Assistant

这段关于 RL (Reinforcement Learning) 和 SFT (Supervised Fine-Tuning) 关系的论述非常精准，触及了当前大模型后训练（Post-training）的核心痛点。

对于基座模型（Base Model）进行 **Reasoning RL (推理强化学习)** 或 **Agentic RL (智能体强化学习)** 的后训练，这段话带来了以下几个关键的战略性启示：

### 1. SFT 的核心定位：构建“高维搜索空间的初始锚点”

在 Reasoning 和 Agentic 任务中，正确的输出通常是一个长序列（例如 20 步的数学推导，或者“思考-调用工具-观察-再思考”的循环）。
假设模型每一步输出正确的概率是 $p$，那么完成一个 $N$ 步任务的概率就是 $p^N$。如果 Base Model 没有经过 SFT，$p$ 会非常小，导致 $p^N \approx 0$。此时，RL 得到的 Reward 永远是 0，策略梯度 $\nabla_\theta J(\theta) = 0$，模型陷入**“无梯度地狱”**。

*   **对 Reasoning 的启示：** SFT 必须教会模型 **Chain-of-Thought (CoT) 的基本格式和逻辑链条**。你不需要 SFT 覆盖所有极难的奥数题，但你必须在 SFT 中放入足够多的“按步骤解题”的样本，让模型知道“遇到问题先写 `Let's think step by step`”。
*   **对 Agentic 的启示：** SFT 必须教会模型 **ReAct (Reason+Act) 框架或工具调用的 JSON 格式**。如果模型在 SFT 阶段没有学会输出 `{"action": "search", "query": "xxx"}`，RL 阶段的环境（Environment）就永远无法解析它的输出并给予正反馈。

### 2. SFT 数据的“广度覆盖”比“深度挖掘”更重要（Cold-Start 原则）

既然诊断信号是“SFT mixture 里是不是没有 task X 的样本”，这意味着在 SFT 阶段，数据的**多样性 (Diversity) 和任务覆盖率 (Coverage)** 优于单任务的数据量。

*   如果你希望 Agent 能调用 50 种不同的 API，SFT 阶段不一定需要每个 API 准备 10000 条数据，但**必须保证这 50 种 API 都有少量高质量的成功轨迹 (Golden Trajectories)**。
*   只要 SFT 把模型推到了“偶尔能做对”（哪怕成功率只有 5%）的分布，RL 就可以接手，通过 PPO 或 DPO 将成功率放大到 90%。

<details>
<summary><b>💡 展开查看：训练失败时的 Pipeline 调试标准流程 (Standard Debugging Pipeline)</b></summary>
<br>
基于上述逻辑，当我们在 Reasoning/Agentic RL 阶段遇到 Reward 盘旋在 0 附近时，应遵循以下调试顺序：

1.  **检查 Base Model / SFT Model 的 Rollout：** 关闭 RL 更新，设置 Temperature > 0，对 Task X 进行 100 次采样。
2.  **统计命中率 (Hit Rate)：** 
    *   如果 100 次中有 1-2 次是对的（格式正确、逻辑基本通顺）：说明冷启动成功，问题出在 RL 的超参（如 KL 惩罚过大、学习率不当、Reward 过于稀疏）。
    *   如果 100 次全错，或者格式全乱导致解析失败：**立即停止调参**。
3.  **回滚到 Data Pipeline：** 构造 Task X 的高质量人工/合成数据（比如用 GPT-4o 生成正确的思考轨迹），混入 SFT 数据集，重新训练 SFT 模型，再启动 RL。
</details>

### 3. “大小模型”在 RL 策略上的分化 (Scale Differences)

引言中提到“小模型对初始分布极度敏感，大模型可以自己摸索”。这直接决定了我们针对不同规模模型（例如 7B vs 70B）的训练配方：

*   **对于小模型 (如 `Llama-3-8B`, `Qwen-2-7B`)：** 
    *   它们是“能力有限的执行者”。探索空间必须被严格限制。
    *   **强依赖 SFT：** SFT 的质量几乎决定了 RL 的天花板。你需要手把手教（Curriculum Learning）。
    *   **RL 策略：** 尽量使用较小的 KL penalty，或者在 RL 损失函数中加入熵正则化项 $H(\pi_\theta)$，强制防止 softmax 过早变尖锐导致丧失探索能力。
*   **对于大模型 (如 `Llama-3-70B+`)：**
    *   它们具有涌现的“举一反三”和组合泛化能力。
    *   **轻量级 SFT + 深度 RL：** SFT 只需要提供基本的人类偏好对齐和格式规范。在 RL 阶段，大模型有能力在广阔的解空间中碰撞出 SFT 未曾见过的创新解法（例如 OpenAI o1 模型在 RL 阶段自己学会了纠错和回溯）。

### 4. Reward 设计：从稀疏奖励向密集奖励退坡 (Reward Shaping)

如果 SFT 无法完美提供 Task X 的冷启动，或者任务太难（如高阶定理证明），我们必须在 RL 阶段修改 Reward 机制，来“接住”不够聪明的模型：

*   **不要只用 Outcome Reward (结果奖励)：** 如果只在最后答案正确时给 $r=1$，否则给 $r=0$，这就成了稀疏奖励（Sparse Reward）。
*   **引入 Process Reward (过程奖励/PRM)：** 在 Reasoning 和 Agentic 任务中，如果模型写对了第一步公式，或者调用对了第一个 API（哪怕最终任务失败），也要给予部分奖励（Partial Reward，例如 $r=0.2$）。
*   通过 Process Reward Model (PRM)，人为地提升“偶尔做对”的概率，降低 RL 的探索门槛。

### 总结

这段话揭示了后训练的黄金法则：**“SFT 决定模型能不能到达起跑线，RL 决定模型能跑多快。”** 在构建 Reasoning 或 Agentic 模型时，必须将 SFT 和 RL 视为一个耦合的系统，遇到瓶颈时优先做 **Data-centric (以数据为中心)** 的归因，而不是单纯的 **Algorithm-centric (以算法为中心)** 的调参。