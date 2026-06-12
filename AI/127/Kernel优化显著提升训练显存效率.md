# Kernel优化显著提升训练显存效率

### 🧑‍💻 User

越来越觉得无论是推理还是训练infra，kernel的性能对整个系统的throughput至关重要，例如liger fused celoss kernel通过tiling+online softmax，分块计算 vocab 维度，不再存整个[B*S, V]，fused linear+ce既降低激活显存又能减少kernel launch overhead，大大降低了大vocab/bs训练的显存需求，配合peft+activation checkpointing只需不到10g vram就能微调1.5b参数的模型。实测32k vocab/1024 hidden/1.5b params模型在bs8下仅开启liger kernel显存占用瞬间降低5g（13.8g->8.5g），更多的余量意味着我们可以使用更大的bs/更长的ctx/关闭ac加速训练等，不过缺点就是无法看到token level metrics（包括ppl/token acc等），只能看到标量loss，tp不兼容loss parallel...对于训练质量观测会带来一些小困难，但在显存有限的情况下显然是利大于弊的

---

### 🤖 Assistant

你的洞察非常精准！完全同意你的观点。在当前的大模型（LLM）时代，无论是训练还是推理，**Kernel 级别的优化已经从“锦上添花”变成了“生死攸关”的基石**。

你提到的 Liger Kernel 的 Fused Cross-Entropy (CE) Loss 就是一个极其经典的以计算（和少量 SRAM 暂存）换取 HBM（显存）带宽和容量的绝佳案例。

我们可以从系统工程的角度，深入探讨一下你提到的收益与妥协。

### 1. 为什么 Fused Linear + CE 收益如此巨大？

在标准的 PyTorch 实现中，计算 Loss 的最后一步通常是：
1. `logits = hidden_states @ embeddings.T`
2. `loss = CrossEntropyLoss(logits, targets)`

这里最大的痛点在于 `logits` 张量。它的形状是 $ [B \times S, V] $。
假设 $B=8$, $S=2048$, $V=32000$，在 `bfloat16`（2 bytes）下，这个中间张量的大小为：
$$ 8 \times 2048 \times 32000 \times 2 \text{ bytes} \approx 1 \text{ GB} $$

这只是前向传播的激活值，为了反向传播，还需要保存对应的梯度（又是 1 GB）。如果 Vocab Size 更大（例如 Qwen 的 152k 或 Llama-3 的 128k），这个开销会呈线性甚至非线性（导致显存碎片化）暴涨。

**Liger Kernel 的解法本质是对 FlashAttention 思想的延伸：**
*   **Tiling & Online Softmax：** 利用 Online Softmax 的数学性质 $ \max(A \cup B) = \max(\max(A), \max(B)) $，在 GPU 的 SRAM 中对 $V$ 维度进行分块（Tiling）计算。
*   **Fusion：** 将矩阵乘法（Linear）和 CE Loss 融合。每次只计算出一个小块的 logits，立刻参与 local softmax 和 loss 的累加，然后**丢弃该 block 的 logits**，绝不写回 HBM。
*   **极致的显存复杂度：** 显存占用从 $O(B \times S \times V)$ 骤降到 $O(B \times S)$，直接把 $V$ 这个巨大的常数项抹去了。这也是为什么你能看到 5GB 级别的显存断崖式下降。

---

### 2. 关于你提到的“妥协”（Trade-offs）与应对策略

正如你所说，天下没有免费的午餐，极致的融合必然带来观测性和系统兼容性上的挑战。但针对这些痛点，Infra 社区也有一些应对策略：

#### 痛点 A: 丢失 Token-level Metrics (PPL, Token Acc)
因为 `logits` 从未被完整实例化，且 Loss 在 Kernel 内部直接被 Reduce 成了标量（Scalar），所以你无法在外层做 `argmax` 来计算准确率，也无法拿到每个 Token 的 PPL。
*   **妥协策略：** 在训练中，我们可以接受只看标量 Loss 来判断收敛趋势。
*   **工程 Workaround：** 可以在 Validation 阶段（或者训练中每隔一定的 Step），**关闭 Fused Kernel**，走一次标准的 PyTorch Forward 来获取详细的 Token-level Metrics。因为此时不需要保存反向传播的计算图，纯 Forward 的显存占用是可以接受的。

#### 痛点 B: 与 Tensor Parallel (TP) / Loss Parallel 不兼容
Loss Parallel 的核心是将 $V$ 维度切分到不同的 GPU 上计算（每个 GPU 计算部分 logits，然后通过 `All-Gather` 局部 max/sum 来完成全局 Softmax）。一旦 Linear 和 CE 融合在一个 Kernel 里，而且内部还在做 Tiling，跨设备的通信逻辑就被破坏了。
*   **工程 Workaround：** 在单机或显存极度受限的微调场景下（如你提到的 1.5B 模型单卡微调），放弃 TP 完全是合理的。对于超大规模预训练，Megatron-LM 等框架目前更倾向于保留 Loss Parallel，并针对切分后的 logits 开发专门的 Fused 算子，而不是单卡跑完所有的 Linear + CE。

---

### 3. 释放的显存带来了“飞轮效应”

你提到的这一点非常关键：**“更多的余量意味着我们可以使用更大的bs/更长的ctx/关闭ac加速训练”**。
在 LLM Infra 中，显存（HBM Capacity）往往是限制吞吐量（Throughput, tokens/sec）的最大瓶颈，而不是算力（FLOPs）。

*   **关闭 Activation Checkpointing (AC)：** AC 是典型的“以计算换显存”（重算前向）。省下 5GB 显存后关闭部分层的 AC，能直接带来 20%-30% 的训练速度提升。
*   **增大 Batch Size (BS)：** 提高 GPU SM 的利用率（Compute Bound 替代 Memory Bound），进一步摊薄 Kernel Launch Overhead。

<details>
<summary><b>💡 展开阅读：除了 Loss Kernel，还有哪些关键的 Infra Kernel 优化？</b></summary>

除了 Liger CE Loss，现代 LLM Infra 极度依赖以下 Kernel 级别的优化来榨干 GPU 性能：

1.  **FlashAttention 系列 (v1/v2/v3)：** LLM Kernel 优化的鼻祖，通过 Tiling 消除 $O(S^2)$ 的 Attention 矩阵显存读写，是长上下文（Long-Context）模型存在的基石。
2.  **PagedAttention (vLLM 等)：** 针对推理阶段的 KV Cache 碎片化问题，通过类似操作系统虚拟内存的分页机制，大幅提升推理 Batch Size。
3.  **Fused RoPE (Rotary Position Embedding) & SwiGLU：** 像 LLaMA 这类架构，RoPE 和 SwiGLU 包含大量的 element-wise 操作。如果不融合，会在 HBM 和 SRAM 之间来回搬运大量数据，导致极高的 Memory Bandwidth 压力。融合后几乎不占用额外时间。
4.  **Quantization Kernels (AWQ, Marlin, ExLlamaV2)：** 针对 W4A16 或 W8A8 的极致推理优化，直接在寄存器或 Shared Memory 中完成反量化并进行 MMA (Matrix Multiply Accumulate)，打破推理时的 Memory Bound。

</details>

### 总结

你的实践深刻印证了当前 AI Infra 工程师的核心工作逻辑：**把算子从 PyTorch 层下沉到 CUDA/Triton 层，控制数据在 HBM 和 SRAM 之间的流动**。在资源受限的情况下，牺牲一点开发期的“观测便利性”和“部分并行策略兼容性”，换取 5GB 的显存和巨大的吞吐量提升，绝对是一笔划算的买卖！