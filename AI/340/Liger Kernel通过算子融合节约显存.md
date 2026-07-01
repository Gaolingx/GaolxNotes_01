# Liger Kernel通过算子融合节约显存

### 🧑‍💻 User

liger celoss kernel节约显存的原理是什么？

---

### 🤖 Assistant

Liger Kernel 是一个专为大语言模型（LLM）训练设计的、基于 OpenAI Triton 的高性能算子库。其中，`CrossEntropyLoss`（交叉熵损失，简称 celoss）算子在优化大模型训练时，能够显著节约显存（VRAM）。

Liger `celoss` 节约显存的核心原理可以总结为：**算子融合（Kernel Fusion）、消除中间变量的 HBM（高带宽内存）读写、以及原地的梯度计算（In-place Gradient Computation）**。

为了讲清楚这个原理，我们需要对比标准 PyTorch 的实现和 Liger Kernel 的实现。

### 1. 痛点：为什么标准交叉熵损失极其消耗显存？

在大语言模型中，最后一层的输出是 Logits，其形状为 `[batch_size * seq_len, vocab_size]`。现代大模型的词表大小（`vocab_size`，即 $V$）通常很大（如 32k, 128k 甚至更大）。

在标准的 PyTorch `CrossEntropyLoss` 计算过程中：
1. **计算 Softmax**：系统会先对 Logits 计算 Softmax 概率分布 $P$。为了支持反向传播，PyTorch 必须在显存中保留这个巨大的概率矩阵 $P$。
2. **计算 Loss**：根据目标标签 $Y$，从 $P$ 中提取对应的值计算交叉熵。
3. **反向传播**：读取保存的 $P$ 矩阵，利用公式 $\frac{\partial L}{\partial X} = P - Y$ 计算梯度。

如果使用混合精度（如 BF16），一个大小为 `[8192, 128000]` 的 Logits 张量大约需要 2GB 显存。而为了反向传播，保存中间变量又要多消耗 2GB 到 4GB 显存。这成为了扩展上下文长度和 Batch Size 的巨大瓶颈。

---

### 2. Liger Kernel `celoss` 节约显存的核心原理

Liger Kernel 通过定制化的 Triton 算子，彻底改变了这一计算流程：

#### A. 算子融合 (Kernel Fusion) 与 SRAM 内计算
Liger 将 Softmax 操作、Log 运算、Loss 计算以及梯度的生成全部融合到一个单独的 CUDA/Triton Kernel 中。
计算过程中，Logits 的切片（Blocks）被加载到 GPU 的高速 SRAM（共享内存/寄存器）中。在 SRAM 内直接完成最大值寻找、指数运算、求和（Online Softmax 算法）以及最终 Loss 的计算，**绝不将中间的 Softmax 概率矩阵写回全局显存 (HBM)**。

#### B. 避免保存用于反向传播的庞大中间张量
由于交叉熵关于 Logits 的梯度公式非常简单：
$$ \frac{\partial L}{\partial X_i} = P_i - Y_i $$
其中 $P_i$ 是预测概率，$Y_i$ 是 One-hot 标签。
Liger Kernel 在前向传播（Forward Pass）计算 Loss 的同时，或者在反向传播（Backward Pass）时利用 Online Softmax 重新计算概率，直接得出梯度。这就**完全省去了保存概率矩阵 $P$ 所需的庞大显存**。

#### C. 原地更新 (In-place) / 内存复用
在某些实现模式下，Liger 会直接利用原本存放 Logits 的内存空间来存放计算出的梯度 $\frac{\partial L}{\partial X}$（因为一旦梯度计算完成，原始的 Logits 就不再需要了）。这种 In-place 操作再次砍掉了一半的显存需求。

---

<details>
<summary><b>🔍 详细对比：标准 PyTorch vs Liger Kernel 的显存占用账单</b></summary>

假设：
*   $N$ = `batch_size * seq_len` (Token 总数)
*   $V$ = `vocab_size` (词表大小)
*   数据类型为 FP16/BF16 (每个元素 2 Bytes)

**标准 PyTorch 流程：**
1. 输入 Logits 占用： $2 \times N \times V$ 字节。
2. 前向传播中，PyTorch 自动微分机制保存 Softmax 结果 $P$：新增占用 $2 \times N \times V$ 字节。
3. 反向传播分配梯度张量 `dLogits`：新增占用 $2 \times N \times V$ 字节。
*   **总峰值显存占用（仅 Loss 部分）** $\approx 6 \times N \times V$ 字节。

**Liger Kernel 流程：**
1. 输入 Logits 占用： $2 \times N \times V$ 字节。
2. 前向传播：在 SRAM 中分块计算，仅输出标量 Loss。**不保存 $P$**。
3. 反向传播：分块读取 Logits 和 Labels，重新计算局部的 $P$ 并直接减去标签生成梯度。如果采用 In-place 模式，梯度直接覆盖 Logits 内存；如果非 In-place，则新分配梯度内存（$2 \times N \times V$ 字节）。
*   **总峰值显存占用（仅 Loss 部分）** $\approx 2 \times N \times V$ 字节（In-place） 或 $4 \times N \times V$ 字节（非 In-place）。

</details>

### 总结

Liger 的 `celoss` 节约显存的本质是用**计算换显存**（在需要时重新计算局部 Softmax）和**极简的内存调度**（依靠 Triton 算子融合在 SRAM 中完成闭环）。它将 Loss 层的峰值显存需求降低了 **50% 到 70%**，使得用户可以在相同的硬件上训练具有更长 Context 窗口或更大词表的模型。

---

### 🧑‍💻 User

具体是避免存储哪个张量降低显存占用？

---

### 🤖 Assistant

具体来说，Liger celoss 降低显存占用，最核心的是**避免了存储 Softmax 概率分布张量**（我们通常用 $P$ 来表示）。

为了让你更直观地理解，我们把前向和反向的数学过程拆解开来：

### 1. 被干掉的“显存刺客”：Softmax 概率张量 $P$

在标准的交叉熵损失计算中，输入是未归一化的 Logits（记为 $X$）。
标准流程的数学推导如下：

1. **前向传播 (Forward Pass)**：
   系统首先对 Logits $X$ 计算 Softmax，得到每一个 Token 对应词表中每一个词的概率分布 $P$：
   $$ P = \text{Softmax}(X) $$
   这个 $P$ 张量的形状和 Logits 一模一样，都是 `[batch_size * seq_len, vocab_size]`。

2. **反向传播 (Backward Pass)**：
   根据微积分链式法则，交叉熵损失 $L$ 关于输入 Logits $X$ 的梯度公式非常优美和简单：
   $$ \frac{\partial L}{\partial X} = P - Y $$
   （其中 $Y$ 是真实标签的 One-hot 编码，实际上就是把 $P$ 中正确标签对应的概率减去 1）。

**痛点就在这里：**
在标准的 PyTorch 自动微分（Autograd）框架下，为了在反向传播时能算出 $P - Y$，**PyTorch 必须在前向传播时把庞大的 $P$ 矩阵保存在显存（HBM）中。**

如果以 Llama-3 为例：词表大小 $V = 128256$，假设 `batch_size * seq_len = 8192`，使用 BF16 精度。
仅这个 $P$ 张量，就会吃掉：
$8192 \times 128256 \times 2 \text{ Bytes} \approx 2.1 \text{ GB}$ 的显存。

**Liger 的做法：**
Liger Kernel 通过定制 Triton 算子，在前向传播时**只输出最终的标量 Loss，直接丢弃（不保存）概率矩阵 $P$**。
到了反向传播阶段，Liger 会分块（Block-wise）把 Logits $X$ 重新读入 GPU 的片上高速缓存（SRAM），在 SRAM 里**重新计算局部概率 $P$，然后瞬间减去标签 $Y$ 得到梯度，并直接写回显存**。
这样就彻底抹去了这 $2.1\text{ GB}$ 的中间变量存储需求。

---

### 2. （可选）进一步避免存储的第二个张量：新的梯度张量 $\frac{\partial L}{\partial X}$

除了避免存储 $P$，Liger 还支持一种叫做 **In-place（原地）更新** 的机制。

在标准 PyTorch 中，算出的梯度 $\frac{\partial L}{\partial X}$ 会被存放在一个**全新分配**的内存张量中（又需要大约 $2.1\text{ GB}$）。

但实际上，一旦梯度 $\frac{\partial L}{\partial X}$ 计算完成，我们在当前层就不再需要原始的 Logits $X$ 了。因此，Liger 会直接**把算出来的梯度写回到原本存放 Logits $X$ 的显存地址中**，覆盖掉 $X$。

### 总结

Liger celoss 具体避免存储的张量主要就是：
1. **绝对避免了存储：前向传播产生的中间概率矩阵 $P$** （省下一倍 Logits 大小的显存）。
2. **通过 In-place 避免了存储：新分配的梯度矩阵**（再省下一倍 Logits 大小的显存）。