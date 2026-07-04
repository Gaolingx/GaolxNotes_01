# hidden size设为4倍数优化并行效率

### 🧑‍💻 User

为什么transformer hidden size/layer一般用4的倍数，这对于distribute parallel有什么好处？

---

### 🤖 Assistant

在Transformer模型（如GPT, LLaMA等）的设计中，将`hidden_size`（隐藏层维度）和`num_layers`（层数）设置为4的倍数（甚至通常是8、16、64或128的倍数，即2的幂次相关）是业界的一个标准做法。

这不仅仅是一个经验法则，而是为了**完美契合底层硬件架构**以及**最大化分布式并行（Distributed Parallelism）的效率**。

以下是这对于分布式并行（特别是张量并行和流水线并行）的具体好处：

### 1. 张量并行（Tensor Parallelism, TP）与 `hidden_size`

张量并行（如Megatron-LM所采用的技术）会将单个矩阵乘法（GEMM）操作拆分到多个GPU上。

*   **切分注意力头（Attention Heads）：** 在Multi-Head Attention中，模型需要将注意力头分配给不同的GPU。如果总共有 $A$ 个头，张量并行度为 $TP$（通常是一台机器上的GPU数量，如4或8），那么必须满足 $A \bmod TP = 0$。由于每个头的维度通常是固定的（例如128），`hidden_size` $H = A \times 128$。如果 $A$ 是4或8的倍数，那么 $H$ 必然也是。
*   **切分MLP层：** Transformer中的前馈神经网络（FFN/MLP）通常会将 `hidden_size` 放大4倍（即 $4H$），然后再缩小回 $H$。张量并行会将这个 $4H$ 的中间维度按列拆分给多个GPU。为了保证每个GPU分配到的计算量和显存**绝对均匀**，`hidden_size` 必须能够被GPU的数量整除。
*   **避免Padding浪费：** 如果 `hidden_size` 不能被 $TP$ 完美整除，底层框架就必须进行零填充（Zero Padding）。这不仅浪费了显存，还会引入额外的逻辑开销，导致通信（All-Reduce操作）时产生无用的数据传输。

### 2. 流水线并行（Pipeline Parallelism, PP）与 `num_layers`

流水线并行是指将模型的不同层（Layers）切分并放置到不同的GPU或节点上。

*   **完美的负载均衡（Load Balancing）：** 假设模型有 $L$ 层，流水线并行度为 $PP$。为了让每个GPU在流水线阶段执行相同时间的计算（避免某个GPU干完了活而在等待其他GPU，即所谓的Pipeline Bubble），最简单的策略就是让每个GPU分配到 $\frac{L}{PP}$ 层。
*   **节点拓扑匹配：** 在大规模训练中，流水线并行通常在节点（Node）之间进行。由于标准的AI服务器节点通常包含4个或8个GPU，将 $L$ 设置为4或8的倍数，可以极大地简化模型在物理集群上的映射逻辑。例如，一个具有32层的模型在4个节点的流水线并行下，每个节点正好均分8层。

### 3. 硬件拓扑与通信效率

现代AI服务器（如NVIDIA DGX）的物理拓扑结构极度依赖于4和8这两个数字：
*   一台标准的服务器通常有 **4张** 或 **8张** GPU。
*   节点内的 NVLink 通信带宽极高，通常会将张量并行度（TP）设置为4或8。
*   由于分布式通信原语（如 `All-Reduce`, `All-Gather`）在数据块大小一致时效率最高，将模型维度设置为4的倍数，能确保在网卡（NIC）和 PCIe/NVLink 上传输的数据包大小是对齐的，从而打满硬件带宽。

---

<details>
<summary><b>💡 展开查看：底层的微观硬件原因（Tensor Cores 与 内存对齐）</b></summary>

除了宏观的分布式并行，微观的GPU硬件设计也极其偏好4、8、16等倍数：

1.  **Tensor Cores 的矩阵分块（Tiling）：**
    NVIDIA GPU的核心计算单元是 Tensor Cores。它们被硬连线为处理特定大小的矩阵块（例如 $4 \times 4$, $8 \times 8$, $16 \times 16$）。在混合精度训练（FP16/BF16）中，如果矩阵的维度 $M, N, K$ 不是 8 或 16 的倍数，cuBLAS 等底层库会自动对矩阵进行 Padding。这会导致：
    *   增加无用的乘加计算（MACs）。
    *   降低 Tensor Core 的利用率（Utilization）。

2.  **全局内存合并（Memory Coalescing）：**
    GPU从全局内存（Global Memory）读取数据时，是以块（比如 32 bytes 或 128 bytes）为单位的。如果 `hidden_size` 的大小导致每行数据在内存中的起始地址没有对齐到这些块的边界，就会发生未合并的内存访问（Uncoalesced Memory Access），导致内存带宽利用率大幅下降。通常，将维度设置为 64 或 128 的倍数可以实现最佳的内存访问效率。

</details>

### 总结

将 `hidden_size` 和 `num_layers` 设置为4（及8、16、64等2的幂）的倍数，本质上是**软件向硬件妥协与优化**的结果。

在分布式并行中，它保证了**计算任务的完美均分**（无负载不均）和**通信数据块的完美对齐**（无网络气泡）；在单卡底层，它激活了 Tensor Cores 的最高效率并最大化了显存带宽。