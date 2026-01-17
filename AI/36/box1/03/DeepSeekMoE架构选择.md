# DeepSeekMoE架构选择

### 🧑‍💻 User

为什么 DeepSeekMoE 放弃在第一层使用 MoE layer？后续 DeepSeek-V2 和 DeepSeek-V3 均在 early layer 上使用 dense layer 替换掉了MoE layer。

---

### 🤖 Assistant

DeepSeek 系列模型（包括早期的 DeepSeekMoE、DeepSeek-V2 以及最新的 DeepSeek-V3）确实都在网络的第一层（有时是前几层）放弃了 MoE 结构，转而使用标准的 Dense FFN（前馈神经网络）。

这一设计决策并非偶然，而是基于对 **Transformer 内部表征机制** 以及 **MoE 路由特性** 的深入理解。主要原因可以归纳为以下几点：

### 1. 底层特征的“通用性”与 Shared Expert 理念

DeepSeekMoE 架构的核心创新在于引入了 **Shared Experts（共享专家）**，旨在将“通用知识”与“特定知识”分离。

*   **理论依据**：在 Transformer 模型中，底层的 Layer 主要负责处理词法、句法以及浅层的语义转换。这些处理对于几乎所有的 Token 都是必要的。
*   **Dense Layer 的本质**：一个标准的 Dense FFN 层，可以被视为一种特殊的 MoE，即它拥有 **100% 的 Shared Experts** 且没有 Routed Experts（路由专家）。这意味着所有的参数对所有的 Token 都是激活的。
*   **决策逻辑**：由于第一层需要处理原始的 Embedding 并进行初步的特征提取，这些操作具有极高的通用性。如果强行使用 MoE 将这部分计算分散到不同的专家中，不仅会导致冗余（不同的专家可能学到了相同的底层特征），还会降低模型处理通用特征的效率。因此，使用 Dense Layer 是处理这种高度通用知识的最优解。

### 2. 路由（Routing）的稳定性与冷启动问题

MoE 的核心在于 Router（门控网络）能够根据输入状态选择合适的专家。

*   **输入状态的局限性**：在第一层，输入仅仅是 Word Embedding（加上 Positional Encoding）。这些原始的 Embedding 尚未经过任何注意力机制（Attention）或非线性变换的混合，包含的信息非常离散且局限于单个 Token 本身。
*   **路由困难**：在这种原始状态下，Router 很难做出高质量的决策来区分 Token 的意图或领域。如果 Router 无法有效分类，MoE 的“专家分工”优势就无法体现，甚至可能因为路由的随机性引入噪声。
*   **Dense 作为“预处理”**：使用 Dense Layer 作为第一层，可以确保 Embedding 经过一次统一的、全参数的非线性变换和特征融合，为后续的 MoE 层提供更丰富、更稳定的 Hidden States，从而帮助后续层的 Router 做出更准确的分派。

### 3. 避免语义表示的过早发散

MoE 倾向于将不同的 Token 映射到不同的子空间（专家）中处理，这是一种“分而治之”的策略。

*   如果在网络的最开始（第一层）就立即对 Token 进行分流，可能会破坏 Embedding 空间的连续性。
*   DeepSeek 团队发现，保留第一层为 Dense 可以充当一个“对齐层”或“锚点层”，确保所有 Token 在进入复杂的专家路由路径之前，都基于一个共享的、统一的基准表示。

### 4. 架构设计的延续性（DeepSeek-V2/V3）

在 DeepSeek-V2 和 V3 的技术报告中，这一设计被明确保留。

*   **DeepSeek-V2/V3** 依然沿用了 DeepSeekMoE 的思路，即：**FFN $\approx$ Common Knowledge (Shared) + Specific Knowledge (Routed)**。
*   由于第一层几乎全是 Common Knowledge，因此它退化为了 Dense FFN。
*   此外，除了第一层，DeepSeek 模型通常也会将 **最后一层** 设为 Dense（或是只包含 Shared Experts），目的是汇聚所有专家的信息，为最终的预测（Next Token Prediction）生成一个统一的表征。

### 总结

DeepSeek 在首层使用 Dense Layer 替换 MoE Layer，本质上是因为**首层任务的特殊性（处理通用底层特征）与 MoE 的特长（处理分化的专门任务）不匹配**。

用公式来理解，DeepSeekMoE 的输出 $h'$ 可以表示为：
$$
h' = u^T (h + \sum_{i=1}^{N_s} \text{FFN}_s^{(i)}(h) + \sum_{j=1}^{N_r} g_j(h) \text{FFN}_r^{(j)}(h))
$$
其中 $N_s$ 是共享专家数量，$N_r$ 是路由专家数量。

在第一层，模型认为所有知识都是通用的，因此 $N_r \to 0$，结构自然就退化回了标准的 Dense Layer。

---

### 🧑‍💻 User

为什么浅层的 load balancing loss 收敛更慢？换句话说，为什么 later layers 比 early layers 饱和更快？

---

### 🤖 Assistant

这是一个非常深刻且触及 MoE 训练动力学（Training Dynamics）本质的问题。在训练 MoE 模型（如 Switch Transformer, Mixtral, DeepSeekMoE）时，确实经常观察到：**深层（Later Layers）的 Router 能够更快地学会如何均衡分配 Token，而浅层（Early Layers）的 Load Balancing Loss 往往下降缓慢，甚至难以收敛。**

造成这种现象的原因可以从 **梯度流传播**、**特征语义层级** 以及 **任务分配的本质** 三个维度来解释：

### 1. 梯度传播距离与信号强度（Gradient Propagation）

这是最直接的优化层面的原因。

*   **监督信号的源头**：神经网络的训练是由最终的 Loss（通常是 Cross-Entropy Loss）驱动的。这个监督信号位于网络的**最末端**。
*   **信号衰减与噪声**：在反向传播过程中，梯度需要经过一层层的 Attention 和 FFN 向前传递。
    *   **深层 Router**：距离输出层很近，接收到的来自 Main Loss 的梯度信号非常强且直接。模型能够迅速判断出：“将这个 Token 发送给 Expert A 会显著降低 Loss”。因此，深层 Router 能够快速建立起“Token 特征 $\to$ 专家选择”的强关联，从而迅速稳定下来，进而满足 Load Balancing Loss 的约束。
    *   **浅层 Router**：距离输出层非常远。梯度信号经过了数十层的非线性变换和混合，到达浅层时已经变得相对微弱且充满噪声（Variance 很大）。浅层 Router 很难收到明确的指令去通过“分派专家”来优化最终任务，因此它在权衡“主任务 Loss”和“负载均衡 Loss”时会更加挣扎，导致收敛缓慢。

### 2. 输入特征的语义清晰度（Semantic Representation）

Router 的决策依赖于输入 Hidden State 的特征分布。浅层和深层的特征分布有本质区别。

*   **浅层（Early Layers）：原始且模糊**
    *   浅层的输入非常接近原始的 Word Embedding。此时的 Token 表征主要包含**词法（Lexical）**和**浅层句法**信息，缺乏上下文语境。
    *   **歧义性高**：例如单词 "bank"，在第一层可能只是一个孤立的词向量。Router 很难判断它应该去处理“金融”的专家还是处理“地理”的专家。
    *   **聚类困难**：由于特征缺乏区分度（Indistinct），Router 很难找到清晰的决策边界将 Token 划分给不同的专家。为了满足负载均衡，Router 不得不进行某种程度的“随机分配”或“强行分配”，这种不自然的分配方式是不稳定的，导致 Router 容易在不同的分配策略间震荡，拖慢收敛。

*   **深层（Later Layers）：具体且清晰**
    *   经过多层 Self-Attention 的上下文聚合，深层的 Token 表征已经包含了丰富的**语义（Semantic）**和**语境**信息。
    *   **聚类容易**：此时 "bank" 已经吸收了周围词（如 "money" 或 "river"）的信息，变成了明确的“金融机构”或“河岸”概念。深层特征在空间中呈现出明显的簇（Cluster）结构。
    *   **路由简单**：Router 只需要顺水推舟，将属于“代码”簇的 Token 发给代码专家，将属于“数学”簇的发给数学专家。这种自然的分类与 Load Balancing 的目标更容易达成一致。

### 3. “通用”与“专用”的任务冲突（General vs. Specific）

这与 DeepSeek 放弃首层 MoE 的逻辑是一脉相承的。

*   **浅层任务的同质性**：浅层网络主要负责通用的特征提取和对齐（例如旋转空间、对齐词性）。这些任务对于几乎所有 Token 都是相似的。
    *   当任务本身是**同质（Homogeneous）**的，MoE 强行要求将 Token 均衡分配给不同的专家（即强行异质化），这本身就是一种**优化冲突**。Router 找不到合理的理由把 Token A 给专家 1，Token B 给专家 2，因为它们需要的处理其实是一样的。这种冲突导致 Load Balancing Loss 很难下降，因为模型倾向于让某个“最强”的专家处理所有事（Expert Collapse）。
*   **深层任务的异质性**：深层网络负责逻辑推理、知识检索等复杂任务。这些任务本身就是**高度异质（Heterogeneous）**的。
    *   这与 MoE 的设计初衷完美契合。任务天然就需要分工，因此 Router 能够很自然地实现负载均衡，甚至不需要太强的 Auxiliary Loss 约束就能自发形成专业化分工。

### 总结

Later layers 比 Early layers 饱和更快、Load Balancing 收敛更好，是因为：

1.  **特征更易分**：深层输入是高度抽象、语义明确的，易于聚类；浅层输入是原始、模糊的。
2.  **目标更一致**：深层任务天然需要分工（异质），契合 MoE；浅层任务偏向通用（同质），与 MoE 强制分流相悖。
3.  **反馈更直接**：深层 Router 接收到的梯度信号更强、更准确。

这也进一步佐证了 DeepSeek-V2/V3 等模型的设计哲学：**在浅层使用 Dense Layer（或 Shared Experts），避免强行对尚未分化的通用特征进行 MoE 路由，从而规避了浅层路由收敛慢、效果差的问题。**