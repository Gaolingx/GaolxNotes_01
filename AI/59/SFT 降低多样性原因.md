# SFT 降低多样性原因

### 🧑‍💻 User

对llm进行fine tune，我总感觉基于sft的方式几乎一定会降低多样性，导致一个模型总是生成一种风格的答案，因为sft数据是预训练分布的一个子集。。。整个分布的支撑集都变小了，有什么理论依据可以支持吗？

---

### 🤖 Assistant

你的直觉非常敏锐，**你是完全正确的**。

在学术界和工业界，这种现象通常被称为**“对齐税”（Alignment Tax）**或者**“模式坍塌”（Mode Collapse）**。SFT（有监督微调）本质上确实是在以牺牲多样性（Diversity）和熵（Entropy）为代价，换取模型输出的确定性（Determinism）和规范性。

以下从几个理论角度来支持你的观点：

### 1. 熵减原理 (Reduction of Entropy)

从信息论的角度来看，预训练（Pre-training）和SFT的目标虽然都是最小化交叉熵损失，但数据分布的熵值截然不同。

*   **预训练阶段**：模型在大规模语料上训练，目标是建模 $P(x)$。这个分布的熵（Entropy）非常高，因为互联网文本包含各种风格、错误、方言和观点。
    $$ H(P_{\text{pretrain}}) \text{ is high} $$
*   **SFT阶段**：我们使用的是人工标注的高质量指令数据。这些数据通常具有非常统一的范式（例如：“首先...其次...最后...”），语气平和且客观。这本质上是一个**低熵分布**。
*   **理论推导**：SFT 的训练目标是最大化似然函数（MLE）：
    $$ \mathcal{L}_{SFT}(\theta) = - \mathbb{E}_{(x,y) \sim \mathcal{D}_{SFT}} [\log P_\theta(y|x)] $$
    当模型收敛时，模型分布 $P_\theta$ 会极度逼近 $\mathcal{D}_{SFT}$。因为 $H(\mathcal{D}_{SFT}) \ll H(\mathcal{D}_{Pretrain})$，所以模型的输出熵会显著降低。这意味着模型在预测下一个 token 时，概率分布会变得非常尖锐（Peaky），可选的 token 变少，自然就丧失了多样性。

### 2. 支撑集收缩 (Support Set Contraction)

你提到的“分布的支撑集（Support Set）变小”在数学上是非常精准的描述。

*   **高维流形假设**：语言模型可以被视为在高维空间中学习一个数据流形（Manifold）。预训练模型覆盖的流形非常广阔。
*   **SFT 的行为**：SFT 数据通常是预训练分布的一个极小的子集（Subset），而且往往位于流形的一个特定区域（例如“有用且无害”的区域）。
*   **灾难性遗忘 (Catastrophic Forgetting)**：虽然我们不希望发生遗忘，但在微调过程中，为了拟合 SFT 数据，模型参数会更新以最小化 SFT 数据的 loss。根据**神经正切核 (Neural Tangent Kernel, NTK)** 理论或简单的梯度下降动力学，模型会倾向于遗忘那些在 SFT 阶段未出现或与 SFT 目标冲突的分布特征。这导致模型实际上“切断”了通往其他风格或领域的路径。

### 3. MLE 的“寻模”特性 (Mode-Seeking Behavior of MLE)

SFT 通常使用标准的交叉熵损失进行训练，这等价于最小化数据分布 $P_{data}$ 和模型分布 $P_{model}$ 之间的 **前向 KL 散度 (Forward KL Divergence)**：

$$ D_{KL}(P_{data} || P_{model}) $$

虽然前向 KL 通常被认为包含“覆盖”（Mode-Covering）特性，但在 SFT 的特定场景下（数据量少、数据高度同质化），它会导致模型过度拟合数据中的显性模式。

更重要的是，SFT 数据往往对同一个 prompt $x$ 只提供**一个**标准答案 $y$。这意味着我们告诉模型：对于输入 $x$，只有 $y$ 是正确的，其他所有可能在预训练中合理的回答现在的概率都应趋近于 0。这迫使模型分布坍塌到单一的“模式”（Mode）上，即所谓的 **Mode Collapse**。

### 4. 逆向强化学习视角 (Inverse RL Perspective)

如果我们把 SFT 看作是一种简单的行为克隆（Behavior Cloning, BC）：

$$ \pi_{SFT} = \arg\max_{\pi} \mathbb{E}_{(s,a) \sim \pi^*} [\log \pi(a|s)] $$

其中 $\pi^*$ 是人类标注者策略。行为克隆有一个著名的理论缺陷：**分布漂移（Distribution Shift）**和** compounding errors**。为了缓解这个问题，SFT 数据通常非常“安全”和“保守”。

理论上，预训练模型是一个巨大的 Generator，可以生成任何东西。SFT 本质上是在这个 Generator 上施加了一个非常强的**Conditioning**。
$$ P_{SFT}(y|x) \approx P_{Pretrain}(y|x, \text{style=assistant}) $$
因为“assistant style”的定义域非常窄，所以多样性必然降低。

### 5. 秩坍塌 (Rank Collapse)

这是一些关于大模型微调的最新理论研究（如 LoRA 相关的论文中提到的现象）。

研究发现，在对大模型进行微调时，权重矩阵的变化量 $\Delta W$ 通常具有非常低的**本征秩 (Intrinsic Rank)**。这意味着，尽管模型参数量巨大，但在 SFT 过程中，模型实际上只在极少数的方向上进行了更新。这种参数空间的低秩更新，限制了模型表达复杂、多样化分布的能力，使其倾向于输出单一风格。

### 总结

你的直觉完全正确。SFT 在数学上可以被理解为：

1.  **约束优化**：将解空间从“所有可能的文本”约束到“符合指令格式的文本”。
2.  **概率质量转移**：将原本分散在各种风格上的概率质量（Probability Mass），集中到了“助手风格”这一狭窄区域。

**由此带来的后果是：**
SFT 后的模型虽然更听话（Instruction Following 能力强），但往往变得更“无聊”，创造力（Temperature 较高时的表现）通常不如 Base 模型。这也是为什么很多做创意写作的应用，更倾向于使用 Base 模型配合 Few-shot Prompting，而不是使用经过重度 SFT 的模型。