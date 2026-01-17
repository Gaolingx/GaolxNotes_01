# Upcycling方法废弃原因

### 🧑‍💻 User

Upcycling 是一个将dense model转化为MoE model的方法，具体做法是我们复制dense model中的FFN layer得到对应MoE layer中的Expert，然后我们再结合router训l练，这样可以提高整体的训l练效率。相关模型有MiniCPM，Qwen1.5和Mixtral MoE（疑似），目前这种方法为什么已经被废弃了？

---

### 🤖 Assistant

**Upcycling（稀疏化/向上循环）** 确实曾在一段时间内非常流行，作为将 Dense 模型低成本转化为 MoE（Mixture of Experts）模型的捷径，它在 Mixtral 8x7B、Qwen1.5-MoE 等模型上都取得了不错的成果。

然而，说它“被废弃”可能略显绝对，更准确的说法是：**在追求极致性能（SOTA）和最新架构的超大模型预训练中，简单的 Upcycling 已经不再是首选方案**。

这主要归结于以下几个核心技术瓶颈和架构演进原因：

### 1. 初始化的同质化与对称性破坏难题 (Symmetry Breaking)

Upcycling 的核心是将 Dense 模型的 FFN（Feed-Forward Network）复制 $N$ 份作为初始的 $N$ 个 Expert。

*   **问题：** 训练刚开始时，所有的 Expert 权重完全相同。
*   **后果：** 尽管引入了 Router（路由器），但在反向传播初期，Router 很难区分这些完全一样的 Expert。虽然加入噪声（Noise）可以打破对称性，但这比从随机初始化（或专门设计的初始化）开始训练要低效得多。
*   **现象：** 这会导致模型在 Upcycling 后期的微调（Fine-tuning）或继续预训练（Continued Pre-training）阶段，花费大量算力去让 Expert 变得“不同”，即实现**专家特化（Expert Specialization）**。如果特化不充分，MoE 就会退化成一个参数量巨大的 Dense 模型，失去了稀疏计算的优势。

### 2. 性能天花板与“伪”最优解

Google 和 DeepMind 的相关研究（如 *Sparse Upcycling* 论文）以及后续的实验表明，Upcycling 存在明显的性能局限性：

*   **损失函数景观（Loss Landscape）：** Dense 模型收敛的局部极小值，并不等同于 MoE 模型的最优解。强行将 Dense 权重作为起点，往往会将 MoE 模型限制在一个次优的区域（Local Minima），很难通过后续训练跳出。
*   **对比结果：** 实验证明，在消耗相同总算力（FLOPs）的情况下，**从头训练（From Scratch）的 MoE 模型通常比 Upcycling 得到的模型性能更好，Loss 更低。** Upcycling 虽然节省了早期的训练步骤，但牺牲了最终模型的上限。

### 3. MoE 架构的演进：DeepSeek 带来的冲击

这是导致传统 Upcycling 方法逐渐式微的最重要原因。早期的 MoE（如 Switch Transformer, Mixtral）架构相对简单，基本就是标准 FFN 的堆叠。但现在的 SOTA MoE 架构（以 **DeepSeek-V2/V3** 为代表）发生了质变：

*   **Fine-Grained Experts (细粒度专家)：** 现在的 Expert 变得非常小且数量众多（例如 DeepSeek-V2 有 160 个专家，每次激活 6 个，甚至更细）。简单的 Dense FFN 结构很大，无法直接拆解或复制成这种细粒度的结构。
*   **Shared Experts (共享专家)：** 现代 MoE 引入了“Shared Expert + Routed Expert”的机制（如 DeepSeek 的 MLA 和 MoE 混合）。Shared Expert 专门负责通用知识，Routed Expert 负责专有知识。
    *   **Upcycling 的困境：** 如果你只有一个 Dense FFN，你应该把它复制给 Shared Expert 还是 Routed Expert？这种架构上的不匹配，使得简单的“复制-粘贴”策略失效。

### 4. 遗留偏差与灾难性遗忘

*   **Dense 模型的特征分布：** Dense 模型的激活模式是为“所有 token 通过同一个 FFN”设计的。
*   **MoE 的特征分布：** MoE 依赖于将输入空间（Input Space）切割成不同的流形，交给不同的 Expert 处理。
*   **冲突：** Upcycling 强迫 MoE 模型继承 Dense 模型的特征处理方式。在继续训练时，Router 试图改变数据的流向，这会导致模型不得不“遗忘”之前 Dense 阶段学到的某些通用激活模式，这种“去学习（Unlearning）”的过程本身就是一种算力浪费。

### 5. 什么时候还在用 Upcycling？

尽管有上述缺点，Upcycling 并没有完全消失，它在以下场景依然活跃：

1.  **算力受限（Budget Constraints）：** 如果你没有几千张 H100 来从头预训练一个 Base 模型，但你手里有一个训练好的 LLaMA-3 或 Mistral，你想低成本搞一个更强的模型，Upcycling 依然是性价比最高的选择（如 Solar-10.7B 的深度扩展）。
2.  **验证性研究：** 学术界验证 Router 算法或 MoE 机制时，为了快速收敛，依然会使用这种方法。

### 总结

Upcycling 之所以被头部玩家（如 DeepSeek, OpenAI, Google）在核心模型训练中“废弃”，是因为：

1.  **架构不兼容：** 现代 MoE (DeepSeek-V2/V3) 结构太复杂，无法简单复制。
2.  **上限不足：** 从头训练虽然贵，但上限更高，对于致力于冲击 AGI 的模型来说，性能优于成本。
3.  **专家特化困难：** 相同的初始权重阻碍了 Expert 的差异化发展。

目前的趋势是：**设计专门的 MoE 架构（如 MLA, Shared+Routed），并从零开始（或使用更复杂的蒸馏初始化）进行预训练。**