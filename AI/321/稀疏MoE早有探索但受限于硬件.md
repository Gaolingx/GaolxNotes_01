# 稀疏MoE早有探索但受限于硬件

### 🧑‍💻 User

在2017年attn is all you need 时代，为什么没有人尝试稀疏mlp实现moe模型？

---

### 🤖 Assistant

这是一个非常专业且敏锐的问题！但首先需要澄清一个历史事实：**在2017年，其实已经有人尝试并提出了稀疏MoE（Mixture of Experts）模型，而且这个人正是《Attention Is All You Need》的核心作者之一 —— Noam Shazeer。**

在2017年初（比Transformer论文早几个月），Noam Shazeer 等人发表了里程碑式的论文 **《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》**。在这篇论文中，他们正是通过在 LSTM 之间插入稀疏门控的 MoE 层（本质上就是并行的多个 MLP），成功训练了一个包含 1370 亿参数的超大模型。

既然技术已经存在，且核心作者重合，**为什么在 Transformer 诞生的 2017 年以及随后的两三年里，稀疏 MoE 没有立刻与 Transformer 的 MLP 层结合并成为主流呢？** 

真正的原因在于当时的**硬件限制、软件生态、训练稳定性以及模型规模的发展阶段**。以下是具体原因的深度剖析：

### 1. 硬件架构与内存带宽的限制 (Hardware & Memory Wall)

在2017年，主流的深度学习 GPU 是 NVIDIA P100 和刚发布的 V100。这些 GPU 极其擅长处理密集矩阵乘法（Dense MatMul），但对稀疏路由非常不友好。

*   **计算访存比（Arithmetic Intensity）低：** 传统的密集 MLP 每一层的所有权重都会参与每一个 Token 的计算。而 MoE 模型在推理或前向传播时，如果采用 $top\text{-}k$ 路由（比如 $k=2$），意味着你需要把庞大的专家（Expert）权重加载到 GPU 的 SRAM 中，但只为了计算极少量的 Token。这导致 MoE 从**计算密集型**变成了**内存带宽受限型（Memory-bandwidth bound）**任务。在2017年，显存带宽根本无法支撑这种低效的读写。
*   **显存容量不足：** MoE 即使只有一部分参数被激活，也必须把所有参数常驻显存。当时单卡显存只有 16GB 到 32GB，根本放不下有意义的 MoE 模型。

### 2. 分布式通信基础设施的缺失 (Communication Bottleneck)

MoE 模型的精髓在于**专家并行（Expert Parallelism）**，即不同的 GPU 上存放不同的专家 MLP 权重。这需要极高的节点间通信能力。

<details>
<summary><b>点击展开：关于 All-to-All 通信的底层痛点</b></summary>
当一个 Batch 的 Token 经过 Router 分配后，它们需要被发送到包含对应专家的不同 GPU 上。计算完成后，结果还需要再发回来。这依赖于一种叫做 `All-to-All` 的集体通信原语。
在 2017 年：
<ul>
<li><b>硬件互联差：</b> NVLink 刚起步，跨节点网络（InfiniBand）带宽极贵且不普及。</li>
<li><b>软件不支持：</b> 当时的 TensorFlow 和 PyTorch 底层的 NCCL 通信库对动态大小的 <code>All-to-All</code> 通信支持极差，这导致通信开销远远超过了稀疏计算省下的时间。这就造成了“参数虽然大了，但训练速度反而慢了”的尴尬局面。</li>
</ul>
</details>

### 3. 训练极度不稳定与“专家坍塌” (Training Instability)

MoE 引入了不可导的路由机制（通常需要注入噪声或使用近似方法），这带来了严重的优化挑战：
*   **专家坍塌（Expert Collapse / Routing Collapse）：** 神经网络在早期倾向于“偷懒”。如果某个专家（MLP）偶然学得好一点，Router 就会把所有 Token 都分配给它。这会导致该专家被过度训练，而其他专家处于“闲置”状态（死神经元现象）。
*   **负载均衡难题：** 为了解决上述问题，Shazeer 在 2017 年的论文中引入了复杂的辅助损失函数（Auxiliary Loss）来强制负载均衡。但在实际操作中，这个 Loss 的权重极难调参，且经常导致主任务的 Loss 震荡甚至崩溃。

### 4. 时代背景：“Dense is All You Need” 的红利期

回到 2017-2018 年的历史语境中：
*   **Dense 模型还没摸到天花板：** 初代 Transformer 只有 65M 和 213M 两个版本。随后的 2018 年，BERT-Large 将规模推到了 340M，GPT-2（2019年）推到了 1.5B。在那个时代，**仅仅通过把 Dense 模型的维度做大、层数加深，就能获得巨大的性能提升**（Scaling Laws 初现端倪）。
*   既然直接把标准的 MLP 变宽就能大幅降 Loss，且在 GPU 上跑得极快、不用写复杂的并行代码、不用调优痛苦的辅助 Loss，学术界和工业界自然没有动力去碰难搞的 MoE。

### 总结与时间线

MoE Transformer 真正开始爆发，是在**模型参数规模突破了单卡乃至单机极限，且 Dense 模型的计算成本变得令人难以忍受之后**。

*   **2017年：** Transformer 诞生；同年 Noam Shazeer 提出 Sparsely-Gated MoE（应用于 LSTM）。
*   **2018-2019年：** Dense 模型的黄金扩容期（BERT, GPT-2）。
*   **2020年：** Google 提出 **GShard**，这是**第一个真正将 MoE 完美集成到 Transformer 架构中的标志性工作**，将模型推向了 6000 亿参数。
*   **2021年：** Google 提出 **Switch Transformer**，将 Router 简化为 Top-1 ($k=1$)，极大地简化了训练和通信，参数量达到 1.6 万亿。
*   **2023年至今：** 随着 GPT-4（被曝为 8x220B MoE）和 Mistral/Mixtral 的出现，配合现代 FlashAttention、高速 NVLink 和成熟的 Megatron-LM 等框架，MoE 终于成为当前大模型的标配。

所以，并不是当时没人想到，而是 Noam Shazeer 等先驱者的思想**超前了当时的硬件和工程生态整整三年**。