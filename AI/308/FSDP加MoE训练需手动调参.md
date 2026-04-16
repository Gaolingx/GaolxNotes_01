# FSDP加MoE训练需手动调参

### 🧑‍💻 User

为什么fsdp+moe training需要手动控制wrap_policy、limit_all_gathers、use_orig_params等参数？

---

### 🤖 Assistant

在训练大型语言模型时，FSDP（Fully Sharded Data Parallel）与 MoE（Mixture of Experts）的结合是目前业界非常前沿且极具挑战性的方向。

普通稠密（Dense）模型使用 FSDP 时，很多参数可以使用默认设置。但 **MoE 模型的稀疏性、极其庞大的参数量以及特殊的网络结构（Router + Experts）**，导致 FSDP 的默认假设失效。

这就要求我们在训练时必须手动精细控制 `wrap_policy`、`limit_all_gathers` 和 `use_orig_params` 这三个核心参数。以下是详细的原因分析：

---

### 1. 为什么必须自定义 `wrap_policy`？
**核心原因：防止因“全量解片（Un-shard）”导致显存瞬间爆炸 (OOM)。**

*   **普通 FSDP 的逻辑**：默认的 `transformer_auto_wrap_policy` 会将一个完整的 Transformer Block（包含 Attention 和 MLP）作为一个 FSDP Unit。在计算到这一层时，FSDP 会通过 All-Gather 将这个 Block 的参数在当前 GPU 上完整重构出来。
*   **MoE 带来的灾难**：在 MoE 模型中，传统的 MLP 层被替换成了包含几十甚至上百个 Expert 的 MoE 层。如果使用默认的 wrap policy，FSDP 会在计算该层时，**把所有 Expert 的参数同时 All-Gather 到单张显卡上**。
*   **手动控制的必要性**：由于单个 Token 每次只需路由到 1-2 个 Expert，把所有 Expert 的参数全拉过来是极其浪费显存的。因此，必须手动编写 Custom Wrap Policy：
    *   将非 MoE 层（如 Attention、LayerNorm）按常规方式 wrap。
    *   将 Router 层单独 wrap，或者不作 FSDP 切片（保持 Replicate），因为 Router 参数极小且需要频繁访问。
    *   **将每一个 Expert 单独 wrap 成一个 FSDP Unit**。这样 FSDP 就能在计算时，用完一个 Expert 就立刻释放（Free）它的参数显存，然后再拉取下一个，从而将显存峰值控制在单/双 Expert 的大小，而不是整个 MoE 层的大小。

### 2. 为什么需要严格控制 `limit_all_gathers=True`？
**核心原因：限制 CPU 异步调度的激进程度，避免预取过多的 MoE 层耗尽显存。**

*   **CUDA 的异步机制**：PyTorch 在执行时，CPU 往往跑在 GPU 前面，它会不断向 GPU 的指令队列里塞入 All-Gather（收集参数）和 Compute（计算）指令。
*   **MoE 环境下的显存危机**：MoE 模型的参数体量通常是稠密模型的数十倍。如果 `limit_all_gathers=False`（不限制），CPU 可能会在 GPU 还在计算第 $N$ 层 MoE 时，就把第 $N+1$ 甚至 $N+2$ 层所有 Expert 的 All-Gather 指令发给了 GPU。这会导致多个巨大的 MoE 层参数同时驻留在显存中，瞬间引发 OOM。
*   **手动控制的必要性**：将 `limit_all_gathers` 设为 `True`，可以强制 CPU 在前一个 FSDP Unit 的 All-Gather 和计算真正完成、并且参数显存被释放之前，**阻塞并等待**，不发出下一个 All-Gather 请求。这虽然可能会牺牲极少量的通信-计算重叠（Overlap）效率，但对于 MoE 来说是**保命（避免 OOM）**的必要设置。

### 3. 为什么必须开启 `use_orig_params=True`？
**核心原因：兼容 MoE 复杂的路由逻辑、优化器分组以及混合并行策略。**

早期 FSDP（`use_orig_params=False`）会将模型参数展平（Flatten）成一个巨大的一维张量进行切片。但 MoE 模型对此极度不适应：

*   **路由索引与 Masking（前向计算需求）**：MoE 的 Router 需要输出稀疏的索引，并对不同的 Expert 传入不同的 Token 集合。如果参数被展平成 1D Tensor，原始的 `weight.shape` 丢失，很多自定义的 MoE Kernel（如 Flash-Attention 团队的 MegaBlocks 或 Triton 写的稀疏算子）将无法进行正确的矩阵维度广播和切片运算。
*   **复杂的 Optimizer 参数分组（优化器需求）**：在 MoE 训练中，Router 层的参数（决定 Token 去哪个 Expert）和 Expert 本身的参数，往往需要**不同的学习率 (Learning Rate)** 或**权重衰减 (Weight Decay)**，甚至 Router 还需要特定的 Z-loss 辅助损失函数。如果参数被展平，你将无法在 Optimizer 中为 `model.router.weight` 和 `model.experts.weight` 设置不同的超参。
*   **手动控制的必要性**：设置为 `True` 后，FSDP 会在底层保留所有参数原始的 shape、name 和 attribute，仅仅是改变了它们的存储位置（Storage）。这让你可以无缝使用 `torch.compile`，轻松为 Router/Expert 设置不同的优化器组，并兼容复杂的稀疏计算逻辑。

---

<details>
<summary>💡 <b>进阶扩展：当 FSDP 遇到 Expert Parallelism (EP) 时的高阶设置</b> (点击展开)</summary>

在当前最大规模的 MoE 训练（如 Llama 3 MoE, Mixtral 8x22B 等）中，纯 FSDP 往往还是不够的，业界通常采用 **FSDP + EP (Expert Parallelism)** 的混合并行策略：

1. **什么是 EP？**
   非 Expert 参数（Attention 等）用 FSDP 在所有 GPU 上切片；但 Expert 参数不在所有 GPU 上切片，而是将不同的 Expert **完整地放置在不同的 GPU 上**。
2. **Device Mesh 的配合**：
   此时，`wrap_policy` 的编写会变得更加变态。你需要结合 PyTorch 的 `DeviceMesh` 技术：
   * 将 `FSDP_Mesh` 传给 Attention 层。
   * 将 `EP_Mesh` 传给 Experts 层。
3. **为什么 `use_orig_params=True` 在这里是绝对刚需？**
   因为 EP 需要跨 GPU 进 Token 的 All-to-All 通信。发送和接收 Token 时，需要极其精确地知道每个 Expert 当前的 Tensor 形状和设备位置。如果参数被展平为 1D 数组，All-to-All 路由引擎（如 Tutel 或是 PyTorch 原生 DTensor 的 MoE 扩展）将完全瘫痪。

</details>

---

### 🧑‍💻 User

为什么对于dense无所谓，对于moe不要将整个 mlp.experts.* 当成一个 unit 进行分片，最好不要对expert layer使用full shared strategy，并尽量结合ep？

---

### 🤖 Assistant

这是一个非常核心且深入的分布式训练架构问题。

要理解这个问题，我们需要从**计算密度（Compute Density）**和**通信代价（Communication Cost）**两个维度，对比“移动参数（FSDP）”和“移动数据（EP）”的本质区别。

以下是为什么 MoE 模型不能像稠密（Dense）模型那样简单粗暴地使用 FSDP，而必须采用混合并行（FSDP + EP）的深度解析：

---

### 1. 为什么对于 Dense 模型无所谓，而对 MoE 致命？

**核心区别在于：计算利用率（Utilization）和显存峰值（Memory Peak）。**

*   **Dense 模型的逻辑（100% 激活）：**
    对于 Dense 模型，整个 MLP 层虽然很大，但**每一个 Token 都会参与整个 MLP 层的完整计算**。FSDP 把整个 MLP 作为一个 Unit，通过 `All-Gather` 将所有被切片的参数拉到当前 GPU。虽然拉取了海量参数，但这些参数被 100% 充分利用了，计算收益完全覆盖了通信代价。
*   **MoE 模型的灾难（极度稀疏激活）：**
    MoE 层的结构是 `mlp.experts.*`，可能包含 $E = 8$ 甚至 $E = 64$ 个 Expert。如果把它当成一个完整的 FSDP Unit：
    1.  **显存瞬间爆炸 (OOM)：** FSDP 会试图在一瞬间通过 `All-Gather` 重构**所有 $E$ 个 Expert 的参数**。相当于单个设备的显存峰值变成了 Dense 模型的 $E$ 倍。
    2.  **极低的计算收益：** 在 Top-$K$ 路由（通常 $K=1$ 或 $2$）中，当前 GPU 上的 Token 实际上只激活了那 $K$ 个 Expert。这意味着你花费了巨大的网络带宽和显存把剩余 $E-K$ 个 Expert 拉过来，却**一行代码都没跑，直接又丢弃了**。这是一种极度铺张浪费的无效通信。

---

### 2. 为什么最好不要对 Expert Layer 使用 Full Shard (FSDP)？

如果我们退一步，把每一个 Expert 单独作为 FSDP Unit 切片（避免了一次性拉取所有 Expert），为什么依然不推荐对 Expert 使用纯 FSDP 策略？

**核心原理：在 MoE 中，“移动参数（FSDP）” 的代价远大于 “移动 Token（EP）”。**

我们来做一个数学对比。假设隐藏层维度为 $d$，单个 Expert 的参数主要是两个权重矩阵，大小约为 $8d^2$。当前 GPU 分配到需要经过某个 Expert 的 Token 数量（Batch Size）为 $B$。

*   **如果用 FSDP (Full Shard) 处理 Expert：**
    *   **行为：** GPU 为了计算，需要通过 `All-Gather` 从网络中把这个 Expert 被切片的参数全部收集过来。
    *   **通信量：** 与参数大小成正比，即 $\mathcal{O}(d^2)$。
    *   **痛点：** $d$ 通常非常大（如 4096 或 8192），参数量动辄几百 MB 甚至几个 GB。无论当前 GPU 只有 1 个 Token 还是 1000 个 Token 路由到该 Expert，你都必须搬运一整座“参数大山”。
*   **如果用 EP (Expert Parallelism) 处理 Expert：**
    *   **行为：** 参数不切片，Expert 完整地放置在某一张或几张特定的 GPU 上。当前 GPU 不去拉取参数，而是把路由到该 Expert 的 Token 通过 `All-to-All` 通信发送给目标 GPU。目标 GPU 算完后，再把结果 `All-to-All` 发回来。
    *   **通信量：** 与 Token 数量成正比，即 $\mathcal{O}(B \times d)$。
    *   **优势：** 在典型的 LLM 训练中，单 GPU 上的 Token 数 $B$ 通常远小于 $d$。即 $B \times d \ll d^2$。**把少量 Token 打包快递寄走，让持有 Expert 的 GPU 算完再寄回来，比你去网上下载几 GB 的模型参数要快得多。**

---

### 3. 为什么要结合 EP (Expert Parallelism)？

目前训练大模型 MoE 的最优解（State-of-the-Art）是 **FSDP + EP 混合并行**。

<details>
<summary>💡 <b>展开查看 FSDP + EP 的完美配合逻辑</b></summary>

在一个典型的 MoE Transformer Block 中，通常包含 Attention 层、LayerNorm 层、Router 层和 Experts 层。混合并行的策略如下：

1.  **非专家层 (Attention / LayerNorm / Router) -> 使用 FSDP：**
    这些层是稠密的，所有 Token 都要经过它们。因此，使用纯 FSDP 在所有 GPU 上切片，以最大化节省显存。
2.  **专家层 (Experts) -> 使用 EP：**
    *   利用 PyTorch 的 2D `DeviceMesh`，将集群划分为 FSDP 维度和 EP 维度。
    *   不同的 GPU 负责持有不同的 Expert 的完整副本（或者只在一个较小的 EP 通信域内做切片，而不是全局 Full Shard）。
    *   当计算到达 MoE 层时，触发底层的 `All-to-All` 算子，GPU 之间交换 Token。
    *   计算完成后，再次 `All-to-All` 换回 Token，继续进入下一层的 FSDP 计算。

**这种架构带来的三大好处：**
1.  **显存最优：** 非专家层被 FSDP 打散，专家层被分配到不同设备（没有冗余），整体模型可以轻松塞入集群。
2.  **通信最优：** 完美避开了 FSDP 对巨大稀疏参数矩阵的 `All-Gather` 操作，转而使用轻量级的 `All-to-All` Token 交换。
3.  **计算连续性：** Expert 在单卡上保持完整，可以使用最高效的本地稠密矩阵乘法（GEMM）算子进行计算，不需要在计算前做参数拼接，极大降低了 CUDA Kernel 的开销。

</details>

### 总结

对 Dense 模型来说，FSDP 是一种**参数空间上的时间换空间**策略，性价比极高。
但对于 MoE 模型，盲目套用 FSDP 会导致**为了极少量的计算，搬运整个参数海洋**，导致网络拥堵和显存 OOM。因此，通过自定义策略让 FSDP 只处理 Dense 层，并引入 EP 通过 `All-to-All` 处理稀疏的 Expert 层，是目前工业界训练千万亿参数 MoE 模型的唯一正解。