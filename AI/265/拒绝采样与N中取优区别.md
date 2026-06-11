# 拒绝采样与N中取优区别

### 🧑‍💻 User

什么是Rejection sampling，什么是best of N？

---

### 🤖 Assistant

在人工智能和大型语言模型（LLM）的对齐（Alignment）与推理领域，**Rejection Sampling（拒绝采样）** 和 **Best-of-N（N中取优）** 是两种经常被提及且紧密相关的技术。它们的核心目的都是**通过生成多个样本并进行筛选，来提升模型最终输出的质量**。

以下是这两者的详细解释以及它们之间的区别。

---

### 1. Best-of-N (N中取优)

**Best-of-N (BoN)** 是一种非常直观且暴力的**推理时（Inference-time）**策略。它主要用于提升模型在实际应用中回答的质量。

**工作原理：**
1. **生成：** 给定一个输入提示（Prompt），使用 LLM（开启一定的温度 `temperature > 0` 以保证多样性）生成 $N$ 个不同的候选回答。
2. **打分：** 使用一个**奖励模型（Reward Model, RM）**或者某种评分函数，对这 $N$ 个候选回答进行打分。
3. **选择：** 直接选出得分最高的那 **1** 个回答作为最终输出，丢弃其余 $N-1$ 个。

**特点：**
* **优势：** 效果立竿见影。只要奖励模型足够好，随着 $N$ 的增加，输出质量会显著提升（这被称为推理时的 Scaling Law）。
* **劣势：** 计算成本极高。生成 $N$ 个回答需要消耗单次生成 $N$ 倍的算力，通常只在打榜、离线评测或对质量要求极高且不在乎延迟的场景下使用。

---

### 2. Rejection Sampling (拒绝采样)

在 LLM 的语境下（例如 LLaMA 2 的论文中），**Rejection Sampling** 通常指的是一种**训练/微调策略（Training/Fine-tuning time）**，用于构建高质量的训练数据。

**在 LLM 微调中的工作原理：**
1. 给定一批 Prompts，让当前的 LLM 为每个 Prompt 生成多个候选回答。
2. 使用奖励模型（Reward Model）对这些回答打分。
3. **筛选（Accept/Reject）：** 设定一个阈值，或者直接保留每个 Prompt 下得分最高的回答（在这个特定操作下，它和 Best-of-N 极其相似）。
4. **微调：** 将这些被“接受（Accepted）”的高分回答作为目标数据，使用监督微调（SFT）的方式重新训练模型。

通过这种方式，模型能够学习到自己生成的最佳输出，从而在下一次迭代中提升基础能力。

<details>
<summary><b>📐 展开查看：经典统计学中的 Rejection Sampling 严格数学定义</b></summary>

在经典统计学和蒙特卡洛方法中，**Rejection Sampling** 是一种从复杂的目标分布 $p(x)$ 中采样的方法。

假设我们要从 $p(x)$ 采样，但直接采样很困难。我们可以找一个容易采样的提议分布（Proposal Distribution）$q(x)$，并找到一个常数 $M$，使得对于所有的 $x$，都有：
$$M \cdot q(x) \geq p(x)$$

**算法步骤：**
1. 从提议分布中采样一个候选值：$x \sim q(x)$。
2. 从均匀分布中采样一个阈值：$u \sim \text{Uniform}(0, 1)$。
3. 接受/拒绝判断：
   * 如果 $u < \frac{p(x)}{M \cdot q(x)}$，则**接受（Accept）** $x$ 作为来自 $p(x)$ 的样本。
   * 否则，**拒绝（Reject）** $x$，并回到步骤 1 重新开始。

在 LLM 的语境中，$q(x)$ 是模型原始的生成概率，$p(x)$ 是我们期望的（符合人类偏好的）高分回答分布。奖励模型（RM）实际上扮演了决定接受概率的角色。

</details>

---

### 3. 两者的关系与区别

在很多最新的 AI 论文中，这两个词经常被混用，因为它们的核心动作都是 **“生成多个 -> 评分 -> 过滤”**。但严格来讲，它们在应用场景上有所不同：

| 特性 | Best-of-N (BoN) | Rejection Sampling (LLM 语境) |
| :--- | :--- | :--- |
| **主要阶段** | **推理阶段 (Inference)** | **训练/数据准备阶段 (Training)** |
| **最终目的** | 在用户提问时，立刻给出一个最好的回答。 | 收集模型自己的高分回答，打包成数据集去微调模型，提升模型的基础能力。 |
| **结果处理** | 选出 Top-1，直接展示给用户，其余丢弃。 | 选出符合标准的数据（可能不止1个，或者选Top-1），用于执行 `model.fit()` 或 SFT。 |
| **对模型权重的改变**| 不改变模型权重。 | **改变模型权重**。 |

#### 总结伪代码对比

**Best-of-N (推理侧):**
```python
def best_of_n_inference(prompt, model, reward_model, N=16):
    candidates = model.generate(prompt, num_return_sequences=N)
    scores = reward_model.score(prompt, candidates)
    best_index = scores.argmax()
    return candidates[best_index] # 直接返回给用户
```

**Rejection Sampling (训练侧):**
```python
def rejection_sampling_finetuning(prompts, model, reward_model):
    training_data = []
    for prompt in prompts:
        candidates = model.generate(prompt, num_return_sequences=16)
        scores = reward_model.score(prompt, candidates)
        best_candidate = candidates[scores.argmax()] 
        # 将最好的回答加入训练集
        training_data.append((prompt, best_candidate))
  
    # 使用收集到的高质量数据微调模型
    model.train(training_data) 
    return model
```

**一句话总结：**
**Best-of-N** 是一种通过暴力计算换取回答质量的**推理技巧**；而 **Rejection Sampling** 是一种利用模型自身的最佳输出来实现自我迭代进化的**训练方法**。