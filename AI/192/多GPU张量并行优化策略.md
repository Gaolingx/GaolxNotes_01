# 多GPU张量并行优化策略

### 🧑‍💻 User

在多gpu实现张量并行（tensor parallel）中，对于transformer模型中的一个transformer block（例如wq、wk、wv、mlp），要如何对tensor做切分和all reduce（尽可能减少gpu之间的通信）？

---

### 🤖 Assistant

好的，这是一个关于在多GPU上实现张量并行（Tensor Parallelism）时，如何对Transformer Block进行切分和通信（All-Reduce）的详细解释，旨在最大程度地减少GPU之间的通信开销。

这个策略主要源于NVIDIA的Megatron-LM论文，它是一种非常高效的张量并行实现方式。

### 核心思想

张量并行的核心思想是将模型中单个巨大的权重矩阵（例如 `nn.Linear` 层）切分到多个GPU上，而不是将不同的层放在不同的GPU上（那是流水线并行）。这样，每个GPU只存储和计算权重矩阵的一部分，从而解决了单个GPU显存不足的问题，并能并行化计算。

为了实现这一点，我们需要两种基本的并行方式：**列并行（Column Parallelism）** 和 **行并行（Row Parallelism）**。

#### 1. 列并行 (Column-Parallel Linear Layer)

当我们将一个权重矩阵 $W$ 按 **列** 切分时，我们称之为列并行。

假设我们有 $N$ 个GPU，权重矩阵 $W$ 被切分为 $[W_1, W_2, ..., W_N]$。
输入是一个在所有GPU上都相同的张量 $X$（replicated）。

前向传播计算为：
$$Y = XW = X [W_1, W_2, ..., W_N] = [XW_1, XW_2, ..., XW_N]$$

每个GPU $i$ 只计算 $Y_i = XW_i$。

* **输入**：$X$ 在所有GPU上是相同的（Replicated）。
* **权重**：$W$ 按列切分 ($W_i$ 在 GPU $i$上)。
* **计算**：每个GPU独立计算 $Y_i = XW_i$。
* **输出**：$Y$ 是按列切分的 ($Y_i$ 在 GPU $i$上)。
* **通信**：在前向传播这一步 **不需要任何通信**。

#### 2. 行并行 (Row-Parallel Linear Layer)

当我们将一个权重矩阵 $W$ 按 **行** 切分时，我们称之为行并行。

权重矩阵 $W$ 被切分为：
$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}$$

输入张量 $X$ 是按 **列** 切分的（通常是前一个列并行层的输出）。

前向传播计算为：
$$Y = XW = [X_1, X_2, ..., X_N] \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix} = X_1W_1 + X_2W_2 + ... + X_NW_N$$

每个GPU $i$ 计算出部分结果 $Y_i = X_iW_i$。为了得到最终的完整结果 $Y$，我们需要将所有GPU上的部分结果相加。

* **输入**：$X$ 是按列切分的 ($X_i$ 在 GPU $i$上)。
* **权重**：$W$ 按行切分 ($W_i$ 在 GPU $i$上)。
* **计算**：每个GPU独立计算部分和 $Y_i = X_iW_i$。
* **输出**：$Y$ 在所有GPU上是相同的（Replicated）。
* **通信**：在前向传播中，需要一个 **All-Reduce** 操作来聚合所有 $Y_i$。

### Transformer Block的切分与通信策略

一个标准的Transformer Block包含一个多头自注意力（Multi-Head Self-Attention, MHSA）模块和一个前馈网络（Feed-Forward Network, FFN/MLP）。Megatron-LM的巧妙之处在于它交替使用列并行和行并行，使得每个Block内部的通信最小化。

假设输入一个Transformer Block的隐状态张量 $H$ 在所有GPU上都是完整的（replicated）。

---

#### A. 多头自注意力模块 (MHSA)

1.  **Q, K, V 投射 (wq, wk, wv)**
    * **操作**: 这本质上是三个并行的线性层: $Q = HW_q$, $K = HW_k$, $V = HW_v$。
    * **切分策略**: 对 $W_q, W_k, W_v$ 这三个权重矩阵使用 **列并行**。
        * 将 $W_q$ 切分为 $[W_{q1}, W_{q2}, ..., W_{qN}]$。
        * 将 $W_k$ 切分为 $[W_{k1}, W_{k2}, ..., W_{kN}]$。
        * 将 $W_v$ 切分为 $[W_{v1}, W_{v2}, ..., W_{vN}]$。
    * **计算**:
        * GPU $i$ 计算 $Q_i = HW_{qi}$, $K_i = HW_{ki}$, $V_i = HW_{vi}$。
    * **张量状态**:
        * 输入 $H$: Replicated
        * 输出 $Q, K, V$: 按列切分（或者说按注意力头切分）
    * **通信**: **无**。这是一个 `[Replicated] -> [Split]` 的过程。

2.  **注意力计算**
    * **操作**: $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    * **计算**: 由于 $Q, K, V$ 都已经被切分到每个GPU上，每个GPU $i$ 都拥有一部分注意力头。因此，注意力分数的计算和与 $V_i$ 的加权求和都可以在每个GPU内部独立完成，无需任何通信。
        * GPU $i$ 计算 $S_i = \text{softmax}(\frac{Q_iK_i^T}{\sqrt{d_k}})V_i$。
    * **张量状态**:
        * 输入 $Q_i, K_i, V_i$: Split
        * 输出 $S_i$: Split
    * **通信**: **无**。

3.  **输出投射 (wo)**
    * **操作**: $Output = SW_o$。这里的 $S$ 是上一步所有 $S_i$ 拼接起来的结果，即 $S = [S_1, S_2, ..., S_N]$。
    * **切分策略**: 对输出投射矩阵 $W_o$ 使用 **行并行**。
        * 将 $W_o$ 切分为 $\begin{bmatrix} W_{o1} \\ W_{o2} \\ \vdots \\ W_{oN} \end{bmatrix}$。
    * **计算**:
        * GPU $i$ 计算部分输出 $O_i = S_iW_{oi}$。
        * 最终的输出是所有部分输出的和：$O = \sum_i O_i$。
    * **张量状态**:
        * 输入 $S$: Split
        * 输出 $O$: Replicated
    * **通信**: 需要一个 **All-Reduce** 来对所有GPU上的 $O_i$ 求和，并将结果分发回所有GPU。这是一个 `[Split] -> [Replicated]` 的过程。

---

#### B. 前馈网络模块 (FFN/MLP)

FFN通常由两个线性层组成，中间有一个激活函数，如GeLU。
$MLP(X) = \text{GeLU}(XW_{fc1})W_{fc2}$

1.  **第一个线性层 (Up-projection)**
    * **操作**: 输入是注意力模块的输出 $O$（经过Add & Norm），它在所有GPU上是Replicated的。
    * **切分策略**: 对第一个全连接层的权重 $W_{fc1}$ 使用 **列并行**。
        * 将 $W_{fc1}$ 切分为 $[W_{fc1,1}, W_{fc1,2}, ..., W_{fc1,N}]$。
    * **计算**:
        * GPU $i$ 计算 $Y_i = \text{GeLU}(OW_{fc1,i})$。
    * **张量状态**:
        * 输入 $O$: Replicated
        * 输出 $Y$: Split
    * **通信**: **无**。这是一个 `[Replicated] -> [Split]` 的过程。

2.  **第二个线性层 (Down-projection)**
    * **操作**: 输入是上一步的输出 $Y$，它是Split状态。
    * **切分策略**: 对第二个全连接层的权重 $W_{fc2}$ 使用 **行并行**。
        * 将 $W_{fc2}$ 切分为 $\begin{bmatrix} W_{fc2,1} \\ W_{fc2,2} \\ \vdots \\ W_{fc2,N} \end{bmatrix}$。
    * **计算**:
        * GPU $i$ 计算部分输出 $Z_i = Y_iW_{fc2,i}$。
        * 最终的输出是所有部分输出的和：$Z = \sum_i Z_i$。
    * **张量状态**:
        * 输入 $Y$: Split
        * 输出 $Z$: Replicated
    * **通信**: 需要一个 **All-Reduce** 来对所有GPU上的 $Z_i$ 求和。这是一个 `[Split] -> [Replicated]` 的过程。

### 总结与通信优化

通过这种精心设计的交替并行策略，我们可以将一个Transformer Block中的通信开销降到最低。

| 模块 | 子层 | 并行策略 | 输入状态 | 输出状态 | 通信操作 (前向) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MHSA** | **Wq, Wk, Wv** | **列并行** | Replicated | Split | **无** |
| | Attention Calc | (Local) | Split | Split | **无** |
| | **Wo** | **行并行** | Split | Replicated | **All-Reduce** |
| **FFN/MLP** | **W_fc1 (Up)** | **列并行** | Replicated | Split | **无** |
| | **W_fc2 (Down)**| **行并行** | Split | Replicated | **All-Reduce** |

**关键优化点**:

1.  **最少的通信**: 在整个Transformer Block的前向传播中，只需要进行 **两次** `All-Reduce` 通信。一次在MHSA的输出，一次在FFN的输出。
2.  **串联的优雅**: FFN的输出经过All-Reduce后变成Replicated状态，这恰好是下一个Transformer Block的MHSA模块所需要的输入状态。这种设计使得多个Block可以无缝地堆叠起来，而不需要额外的通信。
3.  **梯度计算**: 在反向传播中，通信模式是相反的。原本前向是All-Reduce的地方（行并行），反向传播时输入梯度是replicated，计算后输出梯度是split，不需要通信。而原本前向没有通信的地方（列并行），反向传播时需要做一次All-Reduce来聚合权重的梯度。因此，反向传播同样也只需要两次All-Reduce。

这种策略是目前实现大模型张量并行最主流和最高效的方法，它在显存占用、计算效率和通信开销之间取得了绝佳的平衡。