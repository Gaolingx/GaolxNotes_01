# GPU训练切分依据

### 🧑‍💻 User

在transformer llm训练中，为了在多张gpu上训练模型，由于模型太大，必须将weight进行切分。对于mlp layers，我们可以按Column-Parallel or Row-Parallel，为了尽可能降低gpu之间的通信，我们可以将一个列并行线性层（Column-Parallel）的输出，直接作为下一个行并行线性层（Row-parallel）的输入，这样做有什么依据？

---

### 🤖 Assistant

这种做法是目前主流大模型分布式训练框架（如 **Megatron-LM**）中 Tensor Parallelism (TP) 的标准设计范式。

其核心依据在于**矩阵乘法的分块性质**与**通信开销的最小化**。简单来说，这种组合允许我们将中间产生的激活值（Activations）保持在切分状态，而无需在两层之间进行任何通信。

以下从数学推导、通信分析和架构设计三个方面详细解释其依据。

### 1. 数学依据：分块矩阵乘法

假设 MLP 包含两层线性变换和一个激活函数 $\sigma$（如 GeLU）。
输入为 $X$，第一层权重为 $A$（维度 $d_{model} \times d_{ffn}$），第二层权重为 $B$（维度 $d_{ffn} \times d_{model}$）。

整个 MLP 的计算公式为：
$$ Y = \sigma(X A) B $$

为了在两张 GPU 上并行（假设 $N=2$），我们利用分块矩阵乘法的性质：

#### 第一层：列并行 (Column Parallel)
我们将矩阵 $A$ 按**列**切分为 $A_1$ 和 $A_2$。
$$ A = [A_1, A_2] $$
其中 $A_1, A_2$ 的维度均为 $d_{model} \times \frac{d_{ffn}}{2}$。

输入 $X$ 复制在所有 GPU 上（Replicated）。每个 GPU 计算一部分输出：
*   **GPU 1:** $H_1 = \sigma(X A_1)$
*   **GPU 2:** $H_2 = \sigma(X A_2)$

此时，中间隐藏层状态 $H$ 实际上被**按列切分**分布在两张卡上，即 $H = [H_1, H_2]$。
**关键点：** 由于激活函数 $\sigma$ 是逐元素（Element-wise）操作，它不会改变数据的分布特性，因此我们在计算完 $X A_i$ 后可以直接在本地计算激活值，**不需要通信**。

#### 第二层：行并行 (Row Parallel)
为了处理已经被切分的输入 $H = [H_1, H_2]$，我们将第二层权重 $B$ 按**行**切分为 $B_1$ 和 $B_2$。
$$ B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} $$
其中 $B_1, B_2$ 的维度均为 $\frac{d_{ffn}}{2} \times d_{model}$。

根据矩阵乘法规则：
$$ Y = H B = [H_1, H_2] \begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = H_1 B_1 + H_2 B_2 $$

*   **GPU 1:** 持有输入 $H_1$ 和权重 $B_1$，计算局部结果 $Y_1 = H_1 B_1$。
*   **GPU 2:** 持有输入 $H_2$ 和权重 $B_2$，计算局部结果 $Y_2 = H_2 B_2$。

#### 最终结果聚合
最终的输出 $Y$ 是两部分之和：$Y = Y_1 + Y_2$。
此时需要一次跨 GPU 的通信（**All-Reduce Sum**）将各卡的局部结果相加，从而在每张卡上恢复完整的 $Y$。

---

### 2. 通信开销分析

如果不采用 "Column $\to$ Row" 的组合，通信开销会显著增加。我们对比一下方案：

#### 方案 A：最优方案 (Column $\to$ Row)
*   **输入处：** 不需要通信（假设输入 $X$ 已经是复制状态）。
*   **层间（Layer 1 $\to$ Layer 2）：** **0 通信**。GPU 1 直接拿 $X A_1$ 的结果作为下一层的输入，因为下一层需要的正是这部分切片。
*   **输出处：** **1 次 All-Reduce**。用于计算 $Y_1 + Y_2$。
*   **总通信量：** 1 次 All-Reduce。

#### 方案 B：朴素方案 (例如 Column $\to$ Column)
如果两层都使用列并行：
1.  Layer 1 输出 $[H_1, H_2]$ 分布在各卡。
2.  Layer 2 需要完整的输入 $H$ 来与自己的列权重 $B_i$ 进行乘法。
3.  **层间通信：** 必须进行 **All-Gather**，让每张卡都获得完整的 $H = [H_1, H_2]$。
4.  Layer 2 输出又是切分的，如果后续模块（如 Dropout/Residual）需要完整数据，又需要通信。
5.  **总通信量：** 至少 2 次通信操作（All-Gather + All-Gather/Reduce）。

### 3. 代码实现逻辑

在代码实现中（例如 Megatron-LM 的 `layers.py`），这通常被封装在一个模块中。我们可以定义两个算子：
1.  `CopyToModelParallelRegion`: 前向传播是 Identity（不做事），反向传播是 All-Reduce。
2.  `ReduceFromModelParallelRegion`: 前向传播是 All-Reduce，反向传播是 Identity（也就是将梯度广播回去）。

MLP 块的流程如下：

```python
# 伪代码演示

class ParallelMLP(nn.Module):
    def __init__(self):
        # 第一层：列并行
        # 输入: (batch, seq_len, hidden_size)
        # 输出: (batch, seq_len, hidden_size / tp_size)
        self.dense_h_to_4h = ColumnParallelLinear(input_size, hidden_size * 4, gather_output=False)
      
        # 第二层：行并行
        # 输入: (batch, seq_len, hidden_size / tp_size)
        # 输出: (batch, seq_len, hidden_size)
        self.dense_4h_to_h = RowParallelLinear(hidden_size * 4, hidden_size, input_is_parallel=True)

    def forward(self, hidden_states):
        # 1. f(Identity)
        # 输入 hidden_states 在各卡上是相同的
      
        # 2. Column Parallel Linear
        # intermediate_parallel 的部分结果分布在各卡上
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
      
        # 3. Activation (GeLU)
        # 逐元素操作，不需要通信
        intermediate_parallel = gelu(intermediate_parallel)
      
        # 4. Row Parallel Linear
        # 这里内部会执行 All-Reduce
        output = self.dense_4h_to_h(intermediate_parallel)
      
        return output
```

### 总结

将 Column-Parallel 的输出作为 Row-Parallel 的输入，其依据是：

1.  **数学等价性**：$H_1 B_1 + H_2 B_2$ 等价于完整矩阵乘法的一部分，允许我们将求和（Reduce）操作推迟到最后一步。
2.  **通信效率**：这种配对方式消除了两层网络中间的通信同步点（Synchronization Point）。在一个完整的 Transformer Layer 中（Attention + MLP），这种设计使得每个子模块只需要在**最后**进行一次 All-Reduce，极大降低了大规模训练时的通信延迟瓶颈。

---

### 🧑‍💻 User

多头注意力机制（Multi-headed attention）的每个“头”（head）都是独立计算的。这种结构是“可以被轻易地并行化（trivially parallelizable）”。只需将不同的attention heads分配到不同的GPU上即可。那么attention heads的输出的结果又要如何输入到并行的mlp layer中呢？

---

### 🤖 Assistant

在 Tensor Parallelism（张量并行）架构中，Attention 模块的输出进入 MLP 模块的过程，并非直接“硬连线”，而是通过**Attention 的输出线性层（Output Linear Layer）**以及一次**All-Reduce**通信来实现的。

这背后的核心在于：Attention 模块不仅仅包含计算注意力的 Heads，还包含一个最后将所有 Heads 结果融合的线性投影层（通常记为 $W_O$ 或 `c_proj`）。

以下是详细的数据流向和原理分析：

### 1. 关键组件：Attention Output Layer ($W_O$)
在多头注意力机制的标准公式中：
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O $$
其中 $W_O$ 是输出投影矩阵。

当我们将 Heads 分配到不同 GPU 上时（例如 GPU 1 计算 Head 1-6，GPU 2 计算 Head 7-12），此时中间结果（Concat 后的向量）在物理上是被**沿隐藏层维度切分**（Split along hidden dimension）并分布在各卡上的。

为了处理这种分布，**$W_O$ 层采用行并行（Row Parallelism）**。

### 2. 数据流向步骤

让我们看看数据是如何从 Attention Heads 流向 MLP 的（假设 2 张 GPU）：

#### 第一步：独立计算 Attention Heads (Local)
*   **GPU 1** 持有一部分 Head 的结果 $H_1$。
*   **GPU 2** 持有一部分 Head 的结果 $H_2$。
*   此时不需要通信。

#### 第二步：行并行输出投影 (Row Parallel Linear)
我们利用 $W_O$ 来处理这些切分的 Heads。我们将 $W_O$ 按**行**切分为 $W_{O1}$ 和 $W_{O2}$。
根据矩阵分块乘法：
$$ \text{Output} = [H_1, H_2] \begin{bmatrix} W_{O1} \\ W_{O2} \end{bmatrix} = H_1 W_{O1} + H_2 W_{O2} $$
*   **GPU 1** 计算局部结果 $Y_1 = H_1 W_{O1}$。
*   **GPU 2** 计算局部结果 $Y_2 = H_2 W_{O2}$。
*   此时，$Y_1$ 和 $Y_2$ 都是维数为 $(Batch, Seq, Hidden)$ 的矩阵，但它们只是最终结果的一部分（Partial Sum）。

#### 第三步：All-Reduce (Sum) —— 关键的“桥梁”
为了进入后续的层（Residual Add, Layer Norm, MLP），我们需要完整的 Output。
此时执行一次 **All-Reduce (Sum)** 操作：
$$ Y_{final} = Y_1 + Y_2 $$
这个操作完成后，GPU 1 和 GPU 2 上都拥有了完全相同的、完整的 Attention 输出 $Y_{final}$。

#### 第四步：后续操作与 MLP 输入
现在每张卡上都有了完整的 $Y_{final}$（Replicated 状态）：
1.  **Dropout / Residual Add**: $X = X + \text{Dropout}(Y_{final})$
2.  **Layer Norm**: $X_{norm} = \text{LayerNorm}(X)$
3.  **MLP 输入**: MLP 的第一层是**列并行（Column Parallel）**。因为它期望输入是完整的（Replicated），而通过上述的 All-Reduce，输入正好满足要求。

### 3. 总结图解

整个 Transformer Layer 的并行模式是对称的：

| 模块 | 子层 | 并行策略 | 输入状态 | 输出状态 | 通信操作 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attention** | QKV Proj | **Column Parallel** | Replicated | Split (按列) | 无 |
| | Core Attention | (独立计算) | Split | Split | 无 |
| | Output Proj ($W_O$) | **Row Parallel** | Split | Partial Sum | **All-Reduce** (恢复为 Replicated) |
| **中间** | Add & Norm | (复制计算) | Replicated | Replicated | 无 |
| **MLP** | Gate/Up Proj | **Column Parallel** | Replicated | Split (按列) | 无 |
| | Activation | (独立计算) | Split | Split | 无 |
| | Down Proj | **Row Parallel** | Split | Partial Sum | **All-Reduce** (恢复为 Replicated) |

### 4. 为什么要这样设计？

你可能会问：*“为什么不在 Attention Output Proj 之后保持切分状态，直接喂给 MLP，从而省去中间那次 All-Reduce？”*

这是因为中间夹着 **Layer Normalization (LN)**：
*   Layer Norm 需要对整个隐藏层维度（Hidden Dimension）计算均值和方差。
*   如果数据是切分的（Split），计算 LN 需要进行昂贵的通信来同步统计量，或者通过 All-Gather 收集完整数据。
*   因此，标准做法是在 Attention 结束时通过 All-Reduce 同步一次，让每张卡都有完整数据去做 Residual Add 和 Layer Norm，然后再进入 MLP 的列并行层。

**结论：**
Attention Heads 的输出通过 **Row-Parallel 的线性层** 转化为局部部分和（Partial Sum），然后通过 **All-Reduce** 累加成完整向量，恢复成所有 GPU 共享的副本状态，最后再输入到 MLP 的列并行层中。