# 变长注意力与标准SDPA区别

### 🧑‍💻 User

var len attention和标准sdpa有什么区别，为什么varlen导致对infra不友好，为什么需要专门编写算子？

---

### 🤖 Assistant

`Varlen Attention`（变长注意力机制）和标准的 `SDPA`（Scaled Dot-Product Attention，缩放点积注意力）主要解决的是**不同长度序列在同一Batch中如何计算**的问题。

虽然 Varlen 在算法逻辑上减少了冗余计算，但它对底层硬件和基础设施提出了更高的挑战。以下是详细的区别、影响及原因分析。

### 1. `Varlen Attention` 与 标准 `SDPA` 的区别

两者的核心区别在于**数据布局**与**掩码处理**方式。

| 特性 | 标准 `SDPA` (Padding-based) | `Varlen Attention` (Concatenation-based) |
| :--- | :--- | :--- |
| **数据布局** | 矩形矩阵，使用 Pad 填充至等长：`[batch_size, max_seq_len, ...]` | 一维拼接，无 Pad：`[total_seq_len, ...]` |
| **索引方式** | 3D/4D 张量索引：`q[b, i, :]` | 1D 张量索引 + 边界数组：`q[i, :]`，配合 `cu_seqlens` |
| **计算冗余** | 高。需对 Padding 部分计算并 Mask 掉 | 无。只计算有效 Token |
| **掩码处理** | 显式 Mask 矩阵，或利用 `is_causal` 标志 | 隐式边界检查，根据 `cu_seqlens` 动态限制 Softmax 范围 |
| **显存占用** | 较高（需存储大块 Padding 的中间激活值） | 极低（严格按需分配） |

在标准 `SDPA` 中，如果一个 Batch 内序列长度不一，通常需要将短序列 Padding 到最长序列的长度。如果 Batch 内长度差异极大（例如 16 vs 4096），会造成巨大的算力和显存浪费。

`Varlen Attention` 将同一个 Batch 内的所有序列拼接成一维长序列，并使用一个前缀和数组 `cu_seqlens`（如 `[0, 16, 103, 4096, ...]`）来记录每个序列的起止位置。

<details>
<summary>展开查看：计算量对比的数学表达</summary>

假设一个 Batch 有 $B$ 个序列，最大长度为 $S_{max}$，实际序列长度为 $L_i$。

*   **标准 SDPA 计算量**：$O(B \cdot S_{max}^2 \cdot d)$
*   **Varlen 计算量**：$O(\sum_{i=1}^B L_i^2 \cdot d)$

当 $L_i$ 极度不均衡时，标准 SDPA 会将大量计算浪费在 $S_{max} - L_i$ 的 Padding 区域，而 Varlen 彻底消除了这部分开销。
</details>

---

### 2. 为什么 Varlen 对 Infra（基础设施）不友好？

GPU 及其底层加速库（如 cuBLAS, cuDNN）的设计哲学是：**追求规则的、高密度的矩阵乘法**。Varlen 的动态性破坏了这种规则性，主要体现在以下几个方面：

#### 2.1 破坏了 GPU 的内存对齐与 Tile 分块
标准 `SDPA` 的输入是规整的 2D/3D 张量，底层算子（如 FlashAttention）可以轻松地将输入切分为 $64 \times 64$ 或 $128 \times 128$ 的 Block 加载到 SRAM 中。
在 `Varlen` 中，序列在内存中是首尾相连的。如果一个序列的长度不是 Tile 大小的整数倍（例如长度为 100），算子在跨越序列边界读取下一个序列时，会产生**非连续的内存访问**，或者需要复杂的边界判断，导致访存效率急剧下降。

#### 2.2 控制流与 Warp Divergence（线程束分化）
在 GPU 上，一个 Warp（包含 32 个线程）通常同步执行相同的指令。
在标准 `SDPA` 中，由于所有序列等长，Warp 内的执行路径是一致的。而在 `Varlen` 中，如果不做特殊处理，同一个 Warp 可能会同时处理属于不同序列的数据。由于不同序列的长度和 Softmax 归一化范围不同，必须引入 `if-else` 分支来判断当前 Token 属于哪个序列。这会导致 Warp Divergence，部分线程处于闲置状态，严重影响并行效率。

#### 2.3 动态形状导致 Kernel 启动开销与图捕获困难
*   **Kernel 启动**：在 PyTorch 的 eager 模式下，如果不使用专门优化的算子，将长序列拆分成多个变长切片会触发数十甚至数百次微小的 CUDA Kernel 启动，带来巨大的 CPU 开销。
*   **计算图捕获**：现代推理框架（如 vLLM, TensorRT-LLM）严重依赖 CUDA Graph 来消除启动开销。但 `cu_seqlens` 是动态的，每前向传播一次，张量的形状和边界都在变化，这使得标准 CUDA Graph 难以直接应用。

---

### 3. 为什么需要专门编写算子？

为了在享受 `Varlen` 节省算力优势的同时，规避上述 Infra 缺陷，必须编写**底层融合算子**（如 `flash_attn_varlen_func`）。原因如下：

#### 3.1 避免实例化 $N \times N$ Attention 矩阵
如果不写融合算子，按照原生 PyTorch 的计算逻辑：
1. 计算 $Q \times K^T$，得到一个 `(total_tokens, total_tokens)` 的大矩阵。
2. 在这个大矩阵上应用由 `cu_seqlens` 生成的 Block-Diagonal Mask（块对角掩码）。
3. 执行 Softmax。
这会生成一个 $O(N^2)$ 的中间矩阵（其中 $N$ 是 `total_tokens`），显存会瞬间爆炸。专门编写的 FlashAttention 变体算子通过在线 Softmax 算法，将这个 $N^2$ 矩阵的计算融合在 SRAM 中完成，绝不下写到 HBM（显存）。

#### 3.2 动态调整循环边界
在标准 FlashAttention 的 CUDA/Triton 实现中，算子会固定循环遍历 $K, V$ 的列。
在 `Varlen` 专用算子中，算子需要接收 `cu_seqlens` 作为参数，并在 Kernel 内部动态计算：
```python
# 伪代码示意
m_start = cu_seqlens[batch_idx]
m_end = cu_seqlens[batch_idx + 1]
# 仅在 [m_start, m_end] 范围内进行 attention 计算
```
这种动态边界计算必须在 Kernel 内部完成，以确保不会越界读取其他序列的数据，同时保证不需要对 Padding 区域进行无用的 Softmax 归一化。

#### 3.3 消除 Python 层的 For 循环
如果不编写专门的 Varlen 算子，开发者通常只能这样写：
```python
outputs = []
for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
    q_seq = q[start:end]
    k_seq = k[start:end]
    v_seq = v[start:end]
    out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq)
    outputs.append(out)
final_out = torch.cat(outputs, dim=0)
```
这种写法虽然逻辑正确，但会导致 Batch 中有多少个序列就启动多少次 Kernel，且无法利用跨序列的并行性。专用的 Varlen 算子（如 `flash_attn_varlen_func`）通过将 `cu_seqlens` 传入 GPU，在单个 Kernel 启动中并行处理所有序列，极大提升了吞吐量。

### 总结
`Varlen Attention` 是为了打破 Padding 浪费而生，但其动态、非连续的特性与 GPU 追求规则矩阵计算的底层架构相悖。因此，必须通过编写高度定制的融合算子，在 Kernel 内部处理序列边界、管理 SRAM 缓存并动态分配线程块，才能在保持高硬件利用率的同时实现变长计算的高效运行。