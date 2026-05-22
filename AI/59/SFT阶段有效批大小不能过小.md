# SFT阶段有效批大小不能过小

### 🧑‍💻 User

为什么llm sft阶段需要的effective batchsize不能比预训练小太多？

---

### 🤖 Assistant

在大型语言模型（LLM）的训练中，尽管监督微调（SFT, Supervised Fine-Tuning）阶段的 Effective Batch Size（有效批大小）通常会比预训练（Pre-training）阶段小，但**确实不能小得太多**（例如，不能从预训练的数百万 Token 级别骤降到几千 Token）。

这主要是由**优化动力学、梯度稳定性、灾难性遗忘以及SFT数据的特性**共同决定的。以下是具体的原因分析：

### 1. 梯度方差与优化稳定性 (核心原因)

在深度学习中，梯度的方差与 Batch Size 成反比。我们可以用以下数学公式表示梯度方差：

$$ \text{Var}(\nabla L) \approx \frac{\sigma^2}{B_{eff}} $$

其中 $B_{eff}$ 是 Effective Batch Size，$\sigma^2$ 是单个样本的梯度方差。

*   **预训练阶段：** 模型习惯了极大的 $B_{eff}$（通常在 $2 \times 10^6$ 到 $4 \times 10^6$ Tokens 甚至更大）。这意味着预训练阶段的梯度更新是非常平滑、噪声极低的。模型最终会收敛到一个相对平坦且稳定的局部最优解（Loss Basin）。
*   **SFT阶段骤降的后果：** 如果 SFT 阶段的 $B_{eff}$ 变得非常小，梯度的随机噪声（Variance）会成倍放大。这种充满巨大噪声的梯度信号会像“随机游走”一样，轻易将模型“踹出”预训练好不容易找到的平缓最优解区域，导致训练震荡甚至发散（Loss 飞坡）。

### 2. 加剧“灾难性遗忘”（Catastrophic Forgetting）

SFT 的目的是让模型学会“遵循指令”和“对齐人类偏好”，而不是重新学习世界知识。世界知识已经储存在预训练权重中。

如果 Batch Size 太小，每个 Step 的梯度将高度偏向于该微小 Batch 内的几个特定样本（例如这几条刚好全是写诗的代码）。这种**高方差、强偏置**的更新会剧烈改变模型的底层权重，导致模型为了迎合当前的几个样本，而迅速遗忘预训练阶段积累的通用知识（即灾难性遗忘）。保持较大的 Batch Size 可以确保每次更新的梯度方向是整个数据集分布的一个良好近似，从而在微调的同时保护原有能力。

<details>
<summary><b>点击展开：3. 优化器状态（Optimizer States）的不匹配问题</b></summary>

LLM 训练通常使用 AdamW 等带有动量（Momentum）的优化器。AdamW 会维护梯度的一阶矩（$m_t$）和二阶矩（$v_t$）。

在预训练结束时，这些动量状态是基于**大 Batch Size 下的低噪声梯度**建立的。如果在 SFT 阶段突然切换到极小的 Batch Size：
1. 新计算出的高噪声梯度会与之前的平滑动量产生严重冲突。
2. 二阶矩估计（基于梯度的平方）会因为极端的梯度峰值而急剧增大，导致自适应学习率变得极其不稳定。
这就要求必须重置优化器状态并极大地降低学习率，但这往往会导致 SFT 阶段收敛极其缓慢或效果不佳。
</details>

<details>
<summary><b>点击展开：4. SFT 数据的高异质性（Heterogeneity）</b></summary>

预训练数据通常是大段的连续文本，而 SFT 数据是由各种各样的指令组成的（例如：写代码、翻译、角色扮演、数学推理等），且长度差异极大。

如果 Effective Batch Size 太小：
*   **极端情况：** 某一个 Batch 可能刚好全是长文本的复杂数学推理，而下一个 Batch 全是极短的日常问候。
*   **结果：** 梯度的方向会在不同的任务空间中剧烈摆动（Oscillation），模型无法学习到一个通用的指令遵循范式。
较大的 Batch Size 能够起到**平滑器（Smoother）**的作用，确保每一个 Batch 内部都混合了足够多类型的任务，使得更新方向始终指向“如何更好地遵循通用指令”。
</details>

### 5. 学习率与 Batch Size 的缩放定律

根据深度学习中的缩放法则（Scaling Laws），学习率（Learning Rate, $LR$）和 Batch Size 之间存在强耦合关系。常见的经验法则是线性或平方根缩放，例如：

$$ LR_{SFT} \propto LR_{PT} \times \sqrt{\frac{B_{SFT}}{B_{PT}}} $$

如果 $B_{SFT}$ 比 $B_{PT}$ 小几个数量级，理论上你必须将学习率降得非常低才能保证不发散。但是，学习率过低会导致模型在 SFT 阶段**欠拟合（Underfitting）**，无法有效学习到新的指令格式；如果强行保持较高的学习率，结合小 Batch 带来的高噪声，模型就会直接崩溃。

### 总结与业界实践法则

由于以上原因，业界在进行 LLM SFT 时，通常会遵循以下规律：

1.  **SFT 的 Effective Batch Size 通常比预训练小，但保持在同一个“数量级邻域”。** 例如，预训练如果是 4M Tokens，SFT 的 Effective Batch Size 通常会设置在 **100K 到 500K Tokens** 左右（或者以序列数计，通常在 128 到 512 个 Sequence 之间）。
2.  **通过梯度累加（Gradient Accumulation）来弥补：** 在显存有限的情况下（单卡 Batch Size 只能开 1 或 2），必须通过增加梯度累加步数，或者使用数据并行（DP/FSDP）来强行撑大 Effective Batch Size，绝不能仅仅为了跑通代码而使用极小的等效批大小。