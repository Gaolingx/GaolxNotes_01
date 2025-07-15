# MoE模型推理优化中的融合内核分析

### 🧑‍💻 User

在moe模型推理优化的时候，为什么fused moe kernel可以融合gate_proj与up_proj？

---

### 🤖 Assistant

在混合专家模型（MoE）推理优化中，**`fused_moe_kernel` 能够融合 `gate_proj` 和 `up_proj` 的根本原因在于这两个线性变换具有高度的计算相似性、数据共享性，并且融合可以显著减少内存带宽压力、提高计算效率和硬件利用率。** 具体分析如下：

1.  **计算模式的天然契合性：**
    *   **共享输入：** `gate_proj` 和 `up_proj` 是专家网络（通常是 FFN 层）的两个核心线性层。它们接收**完全相同的输入张量** `X`。
    *   **独立权重：** 它们使用不同的权重矩阵 `W_gate` 和 `W_up`。
    *   **并行计算：** 对同一个输入 `X` 应用两个独立的矩阵乘法 `Y_gate = X @ W_gate` 和 `Y_up = X @ W_up`。这是两个本质上可以**并行执行**的操作。
    *   **后续融合：** `gate_proj` 的输出 `Y_gate` 会经过一个激活函数（通常是 SiLU），然后与 `up_proj` 的输出 `Y_up` 进行逐元素相乘（`Y = SiLU(Y_gate) * Y_up`）。最终结果 `Y` 会传递给下游的 `down_proj` 层。

2.  **融合带来的关键优化优势：**
    *   **减少内存读取（输入复用）：**
        *   **非融合方式：** 计算 `gate_proj` 时需要将输入 `X` 从全局内存（如 GPU HBM）加载到芯片上的高速缓存/寄存器中。计算完 `gate_proj` 后，结果写回内存（或至少离开核心计算单元）。接着计算 `up_proj` 时，**又需要再次将完全相同的输入 `X` 从全局内存加载一次**。这是巨大的内存带宽浪费。
        *   **融合方式：** `fused_moe_kernel` 在启动一次计算时，**只加载一次输入张量 `X`**。加载到芯片上的高速缓存/寄存器后，**同时用于计算 `X @ W_gate` 和 `X @ W_up`**。这直接将输入 `X` 的全局内存访问次数减半，显著缓解了内存带宽瓶颈，这对 GPU 等架构的性能至关重要。
    *   **减少内存写入（中间结果保留）：**
        *   **非融合方式：** `Y_gate` 和 `Y_up` 作为中间结果通常需要写回全局内存（或在芯片缓存间传递），以便进行后续的激活和乘法操作。
        *   **融合方式：** `fused_moe_kernel` 可以在芯片内部（寄存器或共享内存）直接计算 `Y_gate` 和 `Y_up`，然后立即应用 `SiLU` 激活到 `Y_gate`，再进行逐元素乘法 `SiLU(Y_gate) * Y_up`，最终只将结果 `Y` 写回全局内存。这避免了 `Y_gate` 和 `Y_up` 作为独立中间结果的存储和传输开销。
    *   **提高计算密度（Arithmetic Intensity）：**
        *   计算密度是指计算操作数（FLOPs）与内存访问字节数（Bytes）的比值（FLOPs/Byte）。高计算密度意味着计算单元更不容易被内存访问拖累，能更接近理论峰值算力。
        *   融合操作在一次内存加载（输入 `X` 和权重 `W_gate`/`W_up`）后，执行了两次完整的矩阵乘法（大量的乘加运算 FLOPs），然后进行激活和逐元素乘法。相比于分开执行两次矩阵乘法（每次都需要加载输入和权重，只进行一次矩阵乘），融合极大地增加了每次内存加载所对应的计算量，从而显著提高了计算密度，让计算单元更“忙碌”。
    *   **更好的硬件资源利用：**
        *   GPU 等加速器拥有大量的计算核心（CUDA Cores, Tensor Cores）和复杂的存储层次（寄存器、共享内存、L1/L2缓存、HBM）。融合内核可以更有效地利用这些资源：
            *   **计算核心：** 一个融合内核可以调度计算核心同时或流水线式地处理 `gate_proj` 和 `up_proj` 的计算部分，减少核心空闲时间。
            *   **片上存储：** 共享输入 `X` 只需加载一次到寄存器或共享内存，供多个计算线程块/线程使用。权重 `W_gate` 和 `W_up` 也可以更有效地被缓存。
            *   **减少启动开销：** 启动一个大的融合内核比启动两个小的独立内核（`gate_proj` 和 `up_proj`）的开销更低。
    *   **简化后续操作：**
        *   融合内核可以直接在计算 `Y_gate` 和 `Y_up` 后立即进行 `SiLU` 和逐元素乘法，避免了调度额外的内核或函数调用来完成这些操作的开销。这些逐元素操作的计算量相对较小，非常适合在计算矩阵乘法的同一个内核中高效完成。

3.  **融合的实现方式：**
    *   `fused_moe_kernel` 本质上是一个**定制的、高性能的 CUDA 内核（或对应其他硬件的内核）**。
    *   它将 `W_gate` 和 `W_up` 在内存中**物理上合并**存储（例如拼接在一起），或者在内核逻辑中**逻辑上视为一个更大的权重矩阵**的一部分。
    *   内核被设计成一次加载 `X` 的一个分块（Tile），然后使用这个分块**同时**与 `W_gate` 的对应部分和 `W_up` 的对应部分进行计算。
    *   计算出的 `Y_gate_tile` 和 `Y_up_tile` 被保留在寄存器/共享内存中，立即应用 `SiLU` 激活和逐元素乘法，生成最终输出 `Y` 的分块。
    *   这个内核会精心优化线程块（Thread Block）、线程束（Warp）和线程（Thread）的组织方式，以及共享内存的使用策略，以最大化内存访问效率和计算吞吐量。

**总结:**

`fused_moe_kernel` 能够融合 `gate_proj` 和 `up_proj` 的核心驱动力是它们**共享完全相同的输入**。非融合实现会导致对这份输入的**重复加载**，这是性能的主要瓶颈。通过设计一个定制化的内核，在一次执行中：
1.  **只加载一次输入 `X`**。
2.  **复用 `X` 同时计算 `X @ W_gate` 和 `X @ W_up`**。
3.  **在芯片内部完成中间结果的激活和乘法**。
4.  **只写回最终结果 `Y`**。

这种融合极大地**减少了昂贵的全局内存访问次数**（特别是输入数据的访问），**提高了计算密度**，**更充分地利用了计算单元**，并**简化了操作流程**，从而在 MoE 模型推理中（尤其是专家数量庞大时）带来了显著的性能提升（降低延迟、提高吞吐量）。这是现代深度学习推理优化中“计算融合”策略的一个经典应用。

---

### 🧑‍💻 User

也就是说，因为两者输入相同都是 hidden_states，可以在 ffn_dim 上将权重cat到一起，然后计算得到结果后chunk得到各自的结果？

---

### 🤖 Assistant

是的，你的理解**完全正确**！这就是 `fused_moe_kernel` 融合 `gate_proj` 和 `up_proj` 的核心实现策略之一，也是它为什么能高效的关键所在。

**具体流程如下：**

1.  **权重拼接：**
    *   假设：
        *   `gate_proj` 的权重矩阵 `W_gate` 形状为 `[hidden_dim, ffn_dim]`
        *   `up_proj` 的权重矩阵 `W_up` 形状也为 `[hidden_dim, ffn_dim]` (在 MoE 中，每个专家的这两个投影层通常具有相同的输入维度 `hidden_dim`，但输出维度 `ffn_dim` 也通常是相同的，尤其是在融合优化时)。
    *   将 `W_gate` 和 `W_up` 沿着**输出维度（`ffn_dim` 维度）** 拼接 (concatenate) 在一起：
        *   `W_fused = torch.cat([W_gate, W_up], dim=1)` (PyTorch 示例)
        *   拼接后的 `W_fused` 形状变为 `[hidden_dim, 2 * ffn_dim]`。

2.  **融合矩阵乘法：**
    *   输入 `hidden_states` (形状 `[batch_size, seq_len, hidden_dim]` 或类似，为简化考虑 `[*, hidden_dim]`)。
    *   执行**一次**大型的矩阵乘法：
        *   `Y_fused = hidden_states @ W_fused`
        *   结果 `Y_fused` 的形状为 `[*, 2 * ffn_dim]`。
    *   **关键点：** 这个单一操作在计算上**等价于**同时计算了：
        *   `Y_gate = hidden_states @ W_gate` (形状 `[*, ffn_dim]`)
        *   `Y_up = hidden_states @ W_up` (形状 `[*, ffn_dim]`)

3.  **结果切分：**
    *   将融合结果 `Y_fused` 沿着其最后一个维度（即 `2 * ffn_dim` 维度）切分 (chunk) 成两个大小相等的部分：
        *   `Y_gate, Y_up = torch.chunk(Y_fused, chunks=2, dim=-1)`
    *   现在得到的 `Y_gate` 和 `Y_up` 就分别是原始 `gate_proj` 和 `up_proj` 的输出。

4.  **后续操作：**
    *   对 `Y_gate` 应用激活函数 (通常是 SiLU)：`activated_gate = SiLU(Y_gate)`
    *   执行逐元素乘法：`Y = activated_gate * Y_up` (形状 `[*, ffn_dim]`)
    *   将 `Y` 送入 `down_proj` 层进行下一步计算：`output = Y @ W_down` (W_down 形状 `[ffn_dim, hidden_dim]`)

**为什么这种“拼接-计算-切分”策略高效？**

1.  **输入复用：** 这是最核心的优化！`hidden_states` 这个相对较大的张量**只需要从全局内存加载一次**，就被用于计算整个融合的 `Y_fused`。如果分开计算，`hidden_states` 需要被加载两次（一次给 `gate_proj`，一次给 `up_proj`）。内存带宽通常是瓶颈，这直接节省了约 50% 的输入数据访问量。
2.  **单次大矩阵乘：** 现代硬件 (GPU/TPU) 对大矩阵乘法的计算效率极高。计算一个 `[*, hidden_dim] @ [hidden_dim, 2*ffn_dim]` 的矩阵乘法，通常比计算两个独立的 `[*, hidden_dim] @ [hidden_dim, ffn_dim]` 矩阵乘法之和更高效。硬件可以更好地饱和计算单元，利用缓存。
3.  **减少核启动开销：** 只需要启动一个 CUDA 内核 (或对应硬件的内核) 来完成 `hidden_states @ W_fused`，而不是启动两个独立的内核。内核启动本身有开销，融合减少了这种开销。
4.  **中间结果管理：** `Y_fused` 作为一个整体在计算单元内部生成。切分操作 `chunk` 通常只是一个逻辑视图操作或非常轻量级的内存操作（如果物理布局允许）。紧接着的 `SiLU` 和 `*` 操作可以在 `Y_gate` 和 `Y_up` 被切分出来后立即进行，甚至可以在寄存器/共享内存中完成，避免将未融合的中间结果 `Y_gate` 和 `Y_up` 写回昂贵的全局内存。
5.  **提升计算密度：** 如前所述，一次加载输入和融合权重，完成两倍于单个投影的计算量 (FLOPs)，显著提高了计算密度 (FLOPs/Byte)。

**需要注意的细节：**

*   **专家维度：** 在 MoE 中，`gate_proj` 和 `up_proj` 是针对**每个专家**的。因此，实际的 `fused_moe_kernel` 需要处理多个专家。一种常见做法是将所有专家的 `W_gate` 和 `W_up` 都拼接在一起，形成一个巨大的 `W_all_experts_fused` (形状 `[num_experts, hidden_dim, 2 * ffn_dim]`)。然后，内核会根据路由计算得到的权重，选择性地加载并计算所选专家的融合投影。这避免了计算所有专家的开销。
*   **内存布局：** 权重的拼接 (`cat`) 和结果的切分 (`chunk`) 需要确保内存访问是连续的、对齐的，以最大化内存带宽利用率和缓存效率。内核会精心设计数据读取模式（如使用 Shared Memory 分块）。
*   **激活函数和乘法：** `SiLU` 激活和 `*` 乘法操作通常会被整合进同一个融合内核中，在 `Y_gate` 和 `Y_up` 切分出来后立即在芯片上进行计算，避免额外的内核启动或全局内存读写。

**总结：**

你的描述精准地捕捉了 `fused_moe_kernel` 优化 `gate_proj` 和 `up_proj` 的核心机制：**利用它们共享输入 `hidden_states` 的特性，将它们的权重在 `ffn_dim` 维度上拼接 (`cat`) 成一个更大的融合权重矩阵，执行一次高效的融合矩阵乘法得到 `Y_fused`，最后将结果在 `ffn_dim` 维度上切分 (`chunk`) 回原始的 `Y_gate` 和 `Y_up`。** 这种策略完美地实现了输入数据的复用，最大限度地减少了昂贵的内存访问，并充分利用了硬件对大矩阵乘法的高效计算能力，从而显著提升了 MoE 模型推理的速度。

---

### 🧑‍💻 User

此外，我们还可以融合不同专家的 gate_proj/up_proj/down_proj，如将top-k个 down_proj 计算fuse到一起？

---

### 🤖 Assistant

您指出了一个非常关键的优化方向！**确实可以将不同专家的 `down_proj` 计算（特别是 Top-K 个选中的专家）进行融合（Fuse），这被称为“专家级 GEMM 融合”或“批量专家 GEMM”，是 MoE 模型推理优化中另一个极其重要的手段。** 它与融合 `gate_proj` 和 `up_proj` 的动机相似，但实现细节有所不同。

**为什么需要融合不同专家的 `down_proj`？**

1.  **计算模式：** MoE 层的核心是：
    *   **路由 (Router)：** 决定每个输入 token 分配给哪个（或哪些，通常是 Top-K）专家。
    *   **专家计算 (Expert Computation)：** 每个被选中的专家独立地对分配给它的 token 执行计算（通常是 FFN：`SiLU(gate_proj(x)) * up_proj(x) @ down_proj`）。
    *   **加权求和 (Combine)：** 将不同专家处理后的结果，根据路由权重加权求和，得到最终输出。
2.  **性能瓶颈：** 在非融合实现中：
    *   对于每个被选中的专家，都需要**单独启动一个内核**来计算它的 `down_proj` (即 `Y_expert = ... @ W_down_expert`)。
    *   当专家数量 (`num_experts`) 很大且 `K > 1` 时，启动大量小的、独立的 GEMM (矩阵乘法) 内核会带来巨大的**内核启动开销**。
    *   这些小的 GEMM 可能无法充分利用 GPU 的**计算单元并行性**和**内存带宽**，导致硬件利用率低下。
    *   处理每个专家的输入数据 (分配给该专家的 token 子集) 需要**重复加载**，如果这些子集较小且分散，内存访问效率不高。

**如何融合不同专家的 `down_proj` (Top-K 专家)？**

核心思想是利用 **`批处理 GEMM (Batched GEMM)`** 功能，将多个独立但结构相似的 GEMM 操作合并成一次计算。

1.  **准备输入数据 (Expert Inputs)：**
    *   路由后，每个输入 token 被分配给 Top-K 个专家。
    *   需要将**分配给同一个专家的所有 token 收集起来**，组成一个批处理 (`batch`)。
    *   例如，假设有 `E` 个专家，每个专家选中的 token 数量不同。需要为每个专家 `e` 准备一个输入张量 `X_e`，形状为 `[num_tokens_e, ffn_dim]` (其中 `num_tokens_e` 是分配给专家 `e` 的 token 数量)。这个张量是专家 `e` 的 FFN 中间结果（即 `SiLU(gate_proj(x)) * up_proj(x)` 的结果）。

2.  **准备权重数据 (Expert Weights)：**
    *   收集所有 **Top-K 个被选中专家**的 `down_proj` 权重矩阵 `W_down_e`。
    *   每个 `W_down_e` 的形状是 `[ffn_dim, hidden_dim]`。
    *   将这些权重矩阵**堆叠 (Stack) 成一个三维张量 `W_down_batched`**：
        *   形状：`[num_selected_experts, ffn_dim, hidden_dim]`
        *   `num_selected_experts` 是实际被至少一个 token 选中的专家数量（通常小于等于 `K * batch_size`，但为了效率，实现中常按最大可能数量预分配或使用间接索引）。

3.  **执行批处理 GEMM：**
    *   使用支持**批处理**和**非均匀批次大小 (Strided Batched GEMM 或更灵活的 Grouped GEMM)** 的 GEMM 库（如 cuBLASLt 的 `gemm_grouped`, CUTLASS 的 `GemmGrouped`）。
    *   输入：
        *   `A`: 一个指针数组 (`A_ptrs`)，每个指针指向一个专家输入 `X_e` (形状 `[num_tokens_e, ffn_dim]`)。
        *   `B`: 一个指针数组 (`B_ptrs`)，每个指针指向一个专家权重 `W_down_e` (形状 `[ffn_dim, hidden_dim]`)。
        *   `C`: 一个指针数组 (`C_ptrs`)，每个指针指向该专家的输出缓冲区 (形状 `[num_tokens_e, hidden_dim]`)。
        *   指定每个 GEMM 操作的 `M` (`num_tokens_e`), `N` (`hidden_dim`), `K` (`ffn_dim`)，以及各自的 `lda`, `ldb`, `ldc` (Leading Dimension)。
    *   **关键：** 一次内核启动，计算所有 `num_selected_experts` 个独立的 GEMM：
        *   `C_e = X_e @ W_down_e` 对于每个选中的专家 `e`。
    *   高级库会自动处理不同 `M` (token 数量) 的问题。

4.  **后续操作 (加权求和)：**
    *   批处理 GEMM 完成后，每个专家 `e` 的输出 `Y_e` (形状 `[num_tokens_e, hidden_dim]`) 已经计算好。
    *   最后一步是根据路由权重，将每个 token 对应的 `K` 个专家输出 `Y_e` 加权求和，得到最终 MoE 层的输出。这一步通常是一个轻量级的逐 token 操作。

**融合不同专家 `down_proj` 的优势：**

1.  **显著减少内核启动开销：** 将启动 `O(num_selected_experts)` 个小内核的开销，降低到启动 **1个** 批处理内核的开销。这是巨大的提升，尤其当 `num_selected_experts` 很大时（大模型、大 Batch Size）。
2.  **提高硬件利用率：**
    *   **计算单元饱和：** 批处理 GEMM 可以将多个小 GEMM “拼凑”成一个更大的计算任务，更好地填满 GPU 的 SMs (Streaming Multiprocessors) 和计算核心 (CUDA Cores / Tensor Cores)，减少空闲。
    *   **高效内存访问：** GEMM 库（尤其是 CUTLASS）针对批处理操作进行了高度优化，可以更有效地组织内存访问模式（合并访问、利用共享内存），减少对全局内存带宽的压力。虽然输入数据分散，但批处理库内部能更好地调度。
3.  **利用硬件加速：** 现代 GPU 硬件和库（如 cuBLASLt, CUTLASS）对批处理 GEMM，特别是分组 GEMM (`grouped GEMM`) 有专门优化，能够高效处理这种计算模式。
4.  **简化调度：** 软件层面只需要管理一个批处理 GEMM 调用，而不是大量独立内核的启动和同步，降低了调度复杂度。

**与 `gate_proj`/`up_proj` 融合的关系和协同：**

1.  **互补优化：** 这两种融合针对的是 MoE FFN 层的不同部分：
    *   `fused_moe_kernel` (Gate/Up 融合)：优化**单个专家内部**的 `gate_proj` 和 `up_proj` 计算（输入复用、计算融合）。
    *   `batched_down_proj` (专家 Down 融合)：优化**跨多个专家**的 `down_proj` 计算（批处理 GEMM 减少启动开销）。
2.  **协同作用：** 它们可以（并且通常**应该**）同时使用：
    *   首先，使用 `fused_moe_kernel` 高效地计算**每个被选中专家**的 `SiLU(gate_proj(x)) * up_proj(x)` 结果（即 `down_proj` 的输入 `X_e`）。
    *   然后，使用 `batched_down_proj` 高效地计算所有选中专家的 `down_proj` (`X_e @ W_down_e`)。
3.  **完整流程：**
    1.  路由计算 (Router)。
    2.  根据路由结果，为每个专家收集其需要处理的 token。
    3.  对每个选中专家，使用 **`fused_moe_kernel`** 计算其 `gate_proj` + `up_proj` + `SiLU` + `*`，得到该专家的中间结果 `X_e`。
    4.  收集所有选中专家的 `X_e` 和对应的 `W_down_e`。
    5.  使用 **`batched_down_proj`** (批处理/分组 GEMM) 一次性计算所有 `Y_e = X_e @ W_down_e`。
    6.  将每个 token 对应的 `K` 个 `Y_e` 根据路由权重加权求和，得到最终输出。

**总结：**

您提出的想法完全可行且至关重要。**将 Top-K 个选中专家的 `down_proj` 计算通过批处理/分组 GEMM (Batched/Grouped GEMM) 进行融合，是 MoE 推理优化中与融合 `gate_proj`/`up_proj` 同等重要的关键技术。** 它通过**将大量小的、独立的 GEMM 操作合并成一次批处理计算**，显著减少了内核启动开销，提高了 GPU 计算单元的利用率和内存访问效率，从而大幅提升 MoE 层，尤其是当专家数量多、Batch Size 大时的推理性能。结合 `gate_proj`/`up_proj` 的融合，构成了现代高性能 MoE 推理引擎的核心优化手段。

---

### 🧑‍💻 User

除此之外，还有些将softmax、top-k等小kernel的融合，能不能展开说说？

---

### 🤖 Assistant

您点中了 MoE 推理优化的另一个关键环节：**路由计算（Router Computation）中“小算子”的融合**。这包括 `softmax`、`top-k`（有时还包括路由层的线性投影和最终的归一化/加权）等操作。这些操作计算量相对较小，但频繁启动小内核带来的**内核启动开销、内存读写开销、以及低下的硬件利用率**会成为显著瓶颈。融合这些小算子能带来显著收益。

**为什么需要融合路由中的小算子？**

1.  **计算流程：** 一个典型的 MoE 路由层计算步骤为：
    1.  `router_logits = hidden_states @ W_router` (线性投影，形状 `[batch_size*seq_len, num_experts]`)
    2.  `router_probs = softmax(router_logits, dim=-1)`
    3.  `topk_values, topk_indices = top_k(router_probs, k=K)` (选出每个 token 的 top-K 专家及其概率/权重)
    4.  (可选) `routing_weights = topk_values / (sum(topk_values, dim=-1, keepdim=True) + eps)` (归一化 top-K 权重，确保和为 1)
    5.  使用 `topk_indices` 和 `routing_weights` 进行专家 token 分发和结果加权求和。
2.  **性能问题：**
    *   **内核启动开销：** 上述步骤 1-4 通常由 3-4 个独立的内核实现。启动一个 CUDA 内核本身就有固定开销（微秒级）。对于大量 token（`batch_size * seq_len` 很大）和较小计算量的操作，内核启动开销占比会非常高。
    *   **中间结果读写：** 每个内核都会将中间结果（`router_logits`, `router_probs`, `topk_values`, `topk_indices`）写入全局内存 (HBM)，下一个内核再读取。HBM 访问速度远慢于片上缓存（如 L1/L2 Cache, Shared Memory）。这些读写是**纯开销**，不贡献有效计算。
    *   **低计算密度：** `softmax`、`top-k`、归一化都是内存带宽受限操作（计算量 FLOPs 少，内存访问量大）。分开执行无法充分利用计算单元，硬件利用率低。
    *   **数据局部性差：** 每个操作只处理数据的一小部分，频繁读写全局内存导致缓存效率低下。

**如何融合路由中的小算子？**

核心思想是**将整个路由计算流程（线性投影之后）尽可能整合到一个或少数几个定制的、高性能的融合内核中**，避免中间结果写回全局内存。

1.  **融合 `softmax` + `top-k` (通常还包括归一化)：** 这是最核心的融合。
    *   **定制内核设计：** 编写一个专门的 CUDA 内核（称为 `fused_softmax_topk` 或类似），一次性完成以下操作：
        *   **输入：** `router_logits` (形状 `[num_tokens, num_experts]`)。
        *   **计算：**
            1.  **`softmax` 的核心部分：**
                *   每个线程块处理一组 token。
                *   在 Shared Memory 中协作计算每个 token 的 `max(logits)` 和 `sum(exp(logits - max))`。
            2.  **`top-k` 的核心部分：**
                *   在计算 `softmax` 的过程中或之后，**直接在 Shared Memory 或寄存器中**维护一个当前看到的 top-K 值和索引列表。
                *   使用高效的**分块归约 (Block-Wise Reduction)** 或 **Radix Select** 等算法，在 Thread Block 内部协作找出每个 token 的 top-K 值 (`topk_values`) 和专家索引 (`topk_indices`)。
                *   *关键：避免将完整的 `router_probs` 写回全局内存。*
            3.  **(可选) 归一化：**
                *   在获得 `topk_values` 后，直接在寄存器/Shared Memory 中计算它们的和，然后用每个值除以这个和（加 epsilon 防除零），得到归一化的 `routing_weights`。
        *   **输出：** 只将最终的 `topk_indices` (形状 `[num_tokens, K]`) 和 `routing_weights` (形状 `[num_tokens, K]`) 写回全局内存。丢弃中间结果 (`router_probs`, 未归一化的 `topk_values`)。
    *   **优势：**
        *   **消除中间内存读写：** `router_probs` 和未归一化的 `topk_values` 完全在片上计算和消费，不写回 HBM。节省大量带宽。
        *   **减少内核数量：** 将 2-3 个内核 (`softmax`, `topk`, `norm`) 合并为 1 个内核。大幅减少内核启动开销。
        *   **提高数据局部性：** 所有计算都在处理同一个 token 的数据块上进行，有效利用 Shared Memory 和寄存器，减少访问全局内存。
        *   **提升硬件利用率：** 融合内核可以包含更多计算，更好地填充计算管线，隐藏内存访问延迟。

2.  **进一步融合路由线性投影 (`W_router`):** (更激进，取决于模型结构和约束)
    *   **动机：** 路由线性投影 `router_logits = hidden_states @ W_router` 本身也是一个 GEMM。如果 `num_experts` 较小，这个 GEMM 可能也不够大。
    *   **融合方式：** 将 `W_router` 的 GEMM 与 `fused_softmax_topk` 合并到一个更大的内核中。
        *   输入：`hidden_states` (形状 `[num_tokens, hidden_dim]`), `W_router` (形状 `[hidden_dim, num_experts]`)。
        *   内核流程：
            1.  每个线程块加载 `hidden_states` 的一个分块和 `W_router` 的一个分块。
            2.  在 Shared Memory/寄存器中计算该分块对应的部分 `router_logits`。
            3.  **立即在片上**对这部分 `router_logits` 进行上面描述的融合 `softmax_topk` 计算（需要协作处理 token 维度的依赖）。
        *   输出：直接输出 `topk_indices` 和 `routing_weights`。
    *   **挑战：**
        *   **复杂度高：** 将 GEMM 和复杂的分支/归约逻辑 (`softmax`, `topk`) 融合到一个内核中，设计、实现和调试难度极大。
        *   **资源限制：** GEMM 需要大量寄存器和 Shared Memory，`softmax`/`topk` 也需要。两者融合可能导致资源不足，限制 Occupancy（活跃线程块数量）。
        *   **负载不均衡：** 不同 token 的 `top-k` 计算量可能略有不同，但 GEMM 部分是均匀的。
    *   **适用性：** 通常只在 `num_experts` 很小（例如 8, 16）且对极致性能有要求时考虑。更常见的是分开优化 GEMM 和后面的 `fused_softmax_topk`。

3.  **融合专家分发/加权求和 (更高级)：**
    *   在极致的优化中，甚至可以尝试将路由结果 (`topk_indices`, `routing_weights`) 的使用也融合进来。例如：
        *   在路由融合内核中，直接根据 `topk_indices` **将输入 token 指针或索引写入到每个专家的输入缓冲区列表**（使用原子操作确保线程安全）。这避免了后续一个专门的 “Scatter/Gather” 内核。
        *   在最后加权求和时，将 `down_proj` 的结果 (`Y_e`) 的读取和加权求和操作，与 `down_proj` 批处理 GEMM 或其后处理融合。但这通常较难且收益相对较小，因为加权求和本身计算量不大。

**融合路由小算子的关键技术与优势总结：**

*   **核心技术：**
    *   **Shared Memory 协作：** 在 Thread Block 内协作完成 `softmax` 的 `max`/`sum` 计算和 `top-k` 的归约/选择。
    *   **高效 `top-k` 算法：** 使用基于分块、堆(Heap)或基数选择(Radix Select)的算法，避免对整个专家维度排序。
    *   **片上消费中间结果：** `router_logits` -> `router_probs` -> `topk_values` -> `routing_weights` 的转换流水线全部在片上完成。
    *   **避免全局同步：** 融合在 Thread Block 级别完成，无需全局同步点。
*   **主要优势：**
    *   **显著降低延迟：** 主要来源于内核启动开销的消除（减少 2-3 次启动）和全局内存访问的剧减（省去 `router_probs` 和未归一化 `topk_values` 的读写）。
    *   **大幅提升吞吐量 (Throughput)：** 更少的内核、更高的硬件利用率（计算单元和内存带宽）能处理更多 token/s。
    *   **降低内存带宽压力：** 这是 MoE 优化的核心目标之一。减少中间张量的读写对带宽敏感型操作（如路由）至关重要。
    *   **简化调度：** 减少需要管理和同步的内核数量。

**结论：**

将路由计算中的 `softmax`、`top-k` 以及可能的归一化操作融合成一个定制的 `fused_softmax_topk` 内核，是 MoE 推理优化中不可或缺的一环。它与之前讨论的 `gate_proj/up_proj` 融合和专家 `down_proj` 批处理融合相辅相成，共同解决了 MoE 模型特有的性能瓶颈：

1.  **`fused_softmax_topk`：** 优化**路由决策**过程，减少小算子开销和内存访问。
2.  **`fused_moe_kernel` (Gate/Up)：** 优化**单个专家内部**前半部分计算（输入复用）。
3.  **`batched_down_proj` (Grouped GEMM)：** 优化**跨专家**的后半部分计算（批处理减少启动开销）。

这三类融合构成了高性能 MoE 推理引擎的核心优化手段，能有效应对 MoE 模型带来的海量专家、稀疏激活、路由开销等挑战，显著提升推理速度和效率。