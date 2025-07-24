# HGEMM在LLM推理中的应用与优化

### 🧑‍💻 User

HGEMM在llm推理框架中有哪些应用？

---

### 🤖 Assistant

HGEMM（Half-precision GEneral Matrix Multiply）在大型语言模型（LLM）推理框架中扮演着**核心角色**，是推理性能（速度和吞吐量）的关键优化点。它的应用贯穿整个模型的计算过程，主要集中在以下几个方面：

1.  **Transformer 层中的核心运算：**
    *   **注意力机制 (Self-Attention):**
        *   **Q/K/V 投影:** 将输入的嵌入向量序列 (通常形状为 `[batch_size, seq_len, hidden_size]`) 通过三个独立的线性层（矩阵乘）分别投影到 Query(`Q`)、Key(`K`)、Value(`V`) 空间。这些投影都是大型的 `[hidden_size, hidden_size]` 或 `[hidden_size, head_dim * num_heads]` 矩阵乘法，**非常适合且主要使用 HGEMM**。
        *   **Q * K^T:** 计算注意力分数矩阵 (`[batch_size, num_heads, seq_len, seq_len]`)。这通常是一个 `[seq_len, head_dim]` 乘以 `[head_dim, seq_len]` 的矩阵乘（在批量和头维度上广播）。虽然尺寸相对较小，但由于序列长度 `seq_len` 可能很大（尤其在处理长上下文时），**HGEMM 在此也非常高效**。
        *   **注意力分数 * V:** 将计算得到的注意力权重应用于 Value 矩阵，得到上下文向量 (`[batch_size, num_heads, seq_len, head_dim]`)。这也是一个 `[seq_len, seq_len]` 乘以 `[seq_len, head_dim]` 的矩阵乘（在批量和头维度上广播）。**HGEMM 是首选**。
    *   **前馈神经网络 (Feed-Forward Network - FFN):**
        *   每个 FFN 层通常包含两个线性变换和一个非线性激活函数（如 GeLU, SwiGLU, ReLU）。第一个线性层将 `hidden_size` 扩展到更大的维度（如 `4 * hidden_size`），第二个线性层再投影回 `hidden_size`。
        *   这两个线性变换都是大规模的矩阵乘法（输入维度 `[batch_size * seq_len, hidden_size]` 乘以权重矩阵 `[hidden_size, intermediate_size]` 和 `[intermediate_size, hidden_size]`）。**这是 LLM 计算量最大的部分之一，HGEMM 在此处的加速效果最为显著**。

2.  **输入/输出嵌入层:**
    *   **输入嵌入查找 (Embedding Lookup):** 严格来说不是 GEMM，但输入 Token ID 序列通过嵌入表查找后得到的嵌入向量 (`[batch_size, seq_len, hidden_size]`) 是后续所有 HGEMM 的输入源。
    *   **输出投影 (Output Projection/LM Head):** 模型最后一层输出的 `[batch_size, seq_len, hidden_size]` 张量需要乘以一个巨大的词表嵌入矩阵 `[hidden_size, vocab_size]`，将其投影到词汇表空间，以计算每个位置 Token 的概率分布 (`[batch_size, seq_len, vocab_size]`）。**这是另一个计算量巨大的矩阵乘法，HGEMM 是优化关键**。`vocab_size` 通常非常大（数万到数十万）。

3.  **其他线性层:**
    *   Transformer 结构中的层归一化虽然本身不是矩阵乘，但其输入输出是后续 HGEMM 的输入输出。
    *   位置编码通常直接加到嵌入上，不涉及 GEMM。
    *   一些模型架构可能包含额外的投影层或适配器层，这些线性层同样会利用 HGEMM。

**为什么 HGEMM 在 LLM 推理中如此重要？**

1.  **计算速度：** 现代 GPU (NVIDIA Tensor Cores) 和 AI 加速器 (如 TPU, AMD MI300X Matrix Cores, Ascend NPU) 都针对低精度（FP16, BF16）的矩阵乘法进行了**硬件级优化**。与 FP32 GEMM (SGEMM) 相比，HGEMM 在这些硬件上通常能提供 **2倍到8倍甚至更高的理论峰值计算吞吐量**。
2.  **内存带宽和容量节省：**
    *   **数据搬运量减半：** FP16/BF16 数据类型占用 16 位（2 字节），而 FP32 占用 32 位（4 字节）。在加载输入矩阵、权重矩阵和存储输出矩阵时，HGEMM **所需的内存带宽仅为 SGEMM 的一半**。内存带宽通常是深度学习计算的瓶颈，HGEMM 能有效缓解此瓶颈。
    *   **模型权重占用减半：** 使用半精度（FP16/BF16）存储模型权重，相比 FP32，模型文件大小和加载到 GPU 显存所需的空间**减少一半**。这使得在有限的显存中部署更大的模型或在同一张卡上运行更大的 Batch Size 成为可能。
3.  **能源效率：** 更快的计算速度和更低的数据搬运需求通常意味着更低的能耗。

**在推理框架中的具体实现和优化：**

*   **高度依赖硬件库：** 推理框架（如 TensorRT-LLM, vLLM, FasterTransformer, ONNX Runtime, PyTorch (with TorchInductor/CUDA Graphs), TensorFlow Lite, llama.cpp 的 GPU 后端等）自身很少直接实现 HGEMM。它们深度集成并调用底层硬件供应商提供的高度优化的 GEMM 库：
    *   NVIDIA GPU: **cuBLAS** (特别是 `cublasHgemm`, `cublasGemmEx` with FP16/BF16), **CUTLASS** (提供更灵活的模板化 GEMM 实现，框架或内核开发者常用)。
    *   AMD GPU: **rocBLAS**。
    *   Intel GPU: **oneMKL**。
    *   Apple Silicon: **Accelerate Framework** (BLAS)。
*   **Kernel Fusion:** 推理框架会将 HGEMM 与其前后紧邻的操作（如激活函数 GeLU, LayerNorm 的部分计算, Bias Add, 转置等）融合成一个单一的 GPU kernel。这避免了中间结果的显存读写开销，显著提升性能。优化良好的框架会利用 CUTLASS 或自定义 CUDA/ROCm/HIP 内核来实现这些融合模式。
*   **Batch GEMM:** LLM 推理通常以 Batch 为单位处理多个输入序列。框架会利用 GEMM 库的 Batch 接口 (`cublasHgemmStridedBatched`, `cublasGemmStridedBatchedEx` 等) 来高效处理批处理。
*   **支持 BF16:** 除了 FP16，BF16 (Bfloat16) 因其更大的动态范围（与 FP32 指数位相同）越来越受欢迎，尤其在某些模型和硬件上能提供更好的数值稳定性，同时保持 FP16 的速度和内存优势。现代 GEMM 库和硬件都支持 BF16 HGEMM。

**总结：**

HGEMM 是 LLM 推理引擎的**命脉**。它被广泛应用于 Transformer 模型的核心计算模块（注意力机制、FFN）以及输入输出嵌入层中所有的大型矩阵乘法操作。通过利用现代硬件对 FP16/BF16 计算的专用加速单元（Tensor Cores / Matrix Cores）以及节省宝贵的内存带宽和显存空间，HGEMM 是实现高效、低延迟、高吞吐量 LLM 推理服务的关键技术。推理框架的核心任务之一就是最大程度地优化和利用 HGEMM 操作。