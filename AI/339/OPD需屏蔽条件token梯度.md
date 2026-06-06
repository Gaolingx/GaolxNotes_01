# OPD需屏蔽条件token梯度

### 🧑‍💻 User

我觉得 OPD 和 RL 的主要区别是，reward 的信号是不是分配到每个 token 上，RL 的 reward 是摊大饼。做opd（on policy distillation）的时候，需不需要打qa mask，或者在信用分配的时候把条件token的梯度扣除掉？

---

### 🤖 Assistant

你的直觉非常准确！你对 OPD（On-Policy Distillation）和 RL（Reinforcement Learning，如 PPO）核心区别的理解是一针见血的。

*   **RL（如 PPO）**：确实是“摊大饼”（Sparse Reward / Trajectory-level Reward）。奖励模型通常只在整个句子生成完毕后给出一个全局的标量分数，然后通过 GAE（广义优势估计）等方法进行**信用分配（Credit Assignment）**。这种方式很难精确告诉模型“具体是哪个词生成得好”。
*   **OPD（如基于 KL 散度的蒸馏）**：拥有**密集信号（Dense Signal）**。因为 Teacher 模型会在 Student 生成的每一个 token 上给出概率分布（Logits），Loss 是 Token-level 的，所以信用分配是非常明确的。

针对你的核心问题：**做 OPD 的时候，需不需要打 QA mask，或者在信用分配的时候把条件 token 的梯度扣除掉？**

**答案是：非常需要，必须打 QA mask（扣除条件 token 的梯度）。**

以下是具体原因和机制：

### 1. 明确优化的目标是条件概率 $P(Y|X)$
无论是 SFT、RL 还是 OPD，我们训练 LLM 的核心目标都是让模型学会**根据给定的 Prompt（Question）生成高质量的 Response（Answer）**。
数学上，我们优化的是条件概率分布：
$$ \max_\theta \sum_{t=1}^{|Y|} \log P_\theta(y_t | X, y_{<t}) $$
其中 $X$ 是条件 token（Prompt），$Y$ 是生成的 token。
如果你不打 QA mask，把 Prompt 的 token 也算进 Loss 里，你的优化目标就变成了联合概率 $P(X, Y)$。这意味着你在**训练模型去预测用户会问什么问题**，这在指令微调和蒸馏阶段是毫无意义的，会白白浪费模型的拟合能力，甚至导致模型行为退化。

### 2. On-Policy 的本质是“状态”与“动作”分离
用强化学习的视角来看，条件 token（Prompt）属于环境给定的**初始状态（State）**，而模型生成的 token（Answer）才是模型做出的**动作（Action）**。
在计算梯度和更新策略时，我们只对模型自己做出的“动作”负责（进行信用分配）。我们不能因为环境给了一个很难的“状态”（比如一段生僻字的 Prompt），就去惩罚模型预测不出这个“状态”。因此，Prompt 部分的梯度必须被屏蔽。

---

<details>
<summary><b>展开查看：OPD 中 QA Mask 的代码实现细节</b></summary>

在具体的深度学习框架（如 PyTorch）中，屏蔽条件 token 的梯度通常是通过设置 `ignore_index` 来实现的。

当我们将输入拼成 `[Prompt, Answer]` 喂给 Student 和 Teacher 模型后，我们会得到两者的 logits。在计算 KL 散度或交叉熵时，我们需要构建一个 `labels` 张量：

1. 将 `labels` 初始化为输入序列。
2. 将 `labels` 中属于 Prompt 部分的 token 替换为 `-100`（PyTorch 默认的 ignore_index）。
3. 计算 Loss 时，损失函数会自动忽略 `-100` 的位置。

```python
import torch
import torch.nn.functional as F

# 假设 student_logits 和 teacher_logits 的 shape 为 [batch_size, seq_len, vocab_size]
# labels 的 shape 为 [batch_size, seq_len]

# 1. 找到 Prompt 的长度 (假设 prompt 长度为 prompt_len)
prompt_len = 10 

# 2. 制作 Mask，将 Prompt 部分的 label 设为 -100
labels[:, :prompt_len] = -100

# 3. 提取有效的部分（排除 ignore_index）
# 这里只计算 Answer 部分的 KL 散度
valid_mask = labels != -100

# 获取有效的 logits 并计算温度缩放后的概率
student_log_probs = F.log_softmax(student_logits[valid_mask] / temperature, dim=-1)
teacher_probs = F.softmax(teacher_logits[valid_mask] / temperature, dim=-1)

# 计算 OPD Loss (Token-level KL Divergence)
loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
```
</details>

<details>
<summary><b>展开查看：对比 RL 中的信用分配与 Mask</b></summary>

在 RL（如 PPO）中，其实也隐式地做了类似 "QA mask" 的操作，只是表现形式不同：

1. **Reward 的赋予**：Reward Model 只对生成的 $Y$ 给出评分，Prompt $X$ 不参与得分。
2. **Advantage 的计算**：在计算每一个 token 的 Advantage $\hat{A}_t$ 时，$t$ 是从生成部分的第一个 token 开始算的。
3. **Loss 的计算**：PPO 的 Policy Loss 公式 $\mathcal{L}^{CLIP}$ 仅仅是在生成的 token 序列上进行求和的，Prompt 的概率 $\pi_\theta(x_t | x_{<t})$ 根本不会进入 PPO 的 Loss 计算图中。

所以，无论你是做 RL 还是 OPD，**不为 Prompt 分配信用/梯度**是一个通用的铁律。
</details>

### 总结
你对 OPD 密集奖励和 RL 稀疏奖励的理解完全正确。正因为 OPD 是在每个 token 上密集地对齐 Teacher，我们才更需要**精准地控制对齐的范围**。打上 QA mask，扣除条件 token 的梯度，才能确保 Student 模型把所有的参数更新额度都花在“如何学好 Teacher 的回复策略”上，而不是去背诵用户的提问。