# MoE训练用FSDP2加EP因激活参数小TP收益低

### 🧑‍💻 User

为什么大规模moe训练中，通常是大规模fsdp2+ep，而不叠加tp，是因为active params不大 tp收益递减吗（例如许多200b的moe，attention和hidden也就7b dense样子）？

---

### 🤖 Assistant

你的直觉非常准确！在大规模 MoE（混合专家）模型的训练中，通常采用 `FSDP2`（或 ZeRO-3）+ `EP`（专家并行），而不叠加 `TP`（张量并行），**最核心的原因正是你提到的：Active Parameters（激活参数量/等效 Dense 参数量）通常并不大，导致 TP 的收益递减，甚至带来负收益。**

除此之外，还有通信开销、计算效率和集群拓扑等多个维度的考量。以下是深入的解析：

### 1. 核心原因：Active Parameters 决定了单卡计算的“甜点区”

在 MoE 模型中，尽管总参数量可能高达 200B、800B 甚至万亿，但每个 Token 在前向和反向传播时真正激活的参数量（Active Params）其实很小。
*   正如你举的例子，一个 200B 的 MoE，其 shared attention 和被激活的 1~2 个 expert 的参数加起来，可能等效于一个 7B 到 14B 的 Dense 模型。
*   **内存角度：** 现代 GPU（如 80GB 的 A100/H100）完全能够轻松吞下 7B 级别模型的单层权重、梯度和激活值。既然显存不爆炸，就不需要依赖 `TP` 来切分单层权重。
*   **计算利用率 (MFU) 角度：** GPU 喜欢大矩阵乘法（GEMM）。如果对 7B 级别的 active 结构强行应用 `TP`（例如 $TP=4$ 或 $TP=8$），会导致切分后的矩阵维度过小。这会让 GPU 的 Tensor Core 吃不满，计算时间无法掩盖访存时间，导致算力严重浪费。

### 2. 通信复杂度的“灾难”

分布式训练的核心是平衡计算与通信。如果叠加 `FSDP2 + EP + TP`，会引入极度复杂的通信矩阵：

*   **TP 的通信：** 每次前向和反向都需要高频的 `All-Reduce`，且对带宽极其敏感，通常只能被限制在单个 Node（节点，即 8 张卡）内部通过 NVLink 进行。
*   **EP 的通信：** 在 MoE 层需要进行 `All-to-All`，将 Token 路由到不同的 Expert 所在的 GPU 上。
*   **FSDP 的通信：** 需要在层与层之间进行 `All-Gather`（获取权重）和 `Reduce-Scatter`（同步梯度）。

如果引入 `TP`，在 MoE 层，你需要先通过 `All-to-All` 分发 Token，然后在一个 TP Group 内部对切分后的 Expert 进行计算并做 `All-Reduce`，这会导致通信时间严重拖累计算。抛弃 `TP`，让一个 Expert 完整地呆在一张（或多张）卡上，可以使 `EP` 的路由和计算变得异常干净高效。

### 3. FSDP2 的成熟替代了 TP 的部分作用

在早期（如 Megatron-LM 时代），大家喜欢用 `TP + PP + DP`。但 `FSDP2`（Fully Sharded Data Parallel 的进化版）改变了游戏规则。
*   `FSDP2` 采用了 2D 拓扑结构，能够非常高效地切分非 MoE 部分（如 Attention、LayerNorm）的权重、梯度和优化器状态。
*   对于 Active Params 只有 7B 的模型，`FSDP2` 已经在不牺牲单卡矩阵乘法效率的前提下，完美解决了显存墙问题，这使得 `TP` 彻底失去了用武之地。

---

<details>
<summary><b>🔍 扩展阅读：在什么极端情况下，MoE 训练依然会引入 TP？</b></summary>
<br>
虽然通常不叠加 `TP`，但在以下几种特定场景下，我们依然会看到 <code>TP + EP + FSDP</code> 的组合：

1.  **单个 Expert 本身体积过大：** 
    如果未来的 MoE 架构设计中，单个 Expert 的参数量极大（例如单个 Expert 达到了 $50\text{B}$），导致单张显卡无法装下一个 Expert 的计算和激活值，此时必须用 `TP` 将单个 Expert 切分到多张卡上。
2.  **超长上下文（Long Context）训练：**
    当 Sequence Length 极大（例如 $100\text{K}$ 甚至 $1\text{M}$）时，激活值（Activations）的显存占用会呈 $O(N^2)$ 或 $O(N)$ 爆炸式增长。此时为了配合 `Sequence Parallelism` (SP) 或 `Context Parallelism` (CP)，往往需要底层的 `TP` 来分担显存压力。
3.  **Attention 层的特殊设计：**
    如果 MoE 模型的 Dense Attention 部分异常庞大，即便 FSDP 能切分权重，但前向传播时组装出来的完整权重或 KV Cache 依然导致 OOM，这时可能会对 Attention 层施加 `TP`，而对 MoE 层保持纯 `EP`，但这会导致代码实现极其复杂。

</details>

<details>
<summary><b>📐 理论通信量对比简表</b></summary>
<br>

| 并行策略组合 | 非 MoE 层通信 (如 Attention) | MoE 层通信 | 整体评价 |
| :--- | :--- | :--- | :--- |
| **FSDP2 + EP** | `All-Gather` / `Reduce-Scatter` | `All-to-All` | 最佳实践。通信与计算容易 overlap，计算粒度大，MFU 高。 |
| **FSDP2 + EP + TP** | `All-Reduce` + FSDP 通信 | `All-to-All` + `All-Reduce` | 极差。对于小 Active Params 模型，`All-Reduce` 带宽被浪费，矩阵碎片化。 |

</details>

**总结：**
大规模 MoE 训练的本质是**“用存储换算力”**。它的总参数量虽然吓人，但那只是静态的存储。由于它在运行时的数学等效体（Active Model）只是一个几十 B 的中小型模型，因此我们只需要用 `FSDP2` 解决静态存储的切分，用 `EP` 解决专家的路由即可。引入 `TP` 属于“杀鸡用牛刀”，反而会砍伤自己（降低计算效率、增加通信延迟）。