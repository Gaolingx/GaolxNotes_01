# Muon优化器要点解析

### 🧑‍💻 User

muon优化器设计的依据是什么，为什么在transformer nlp task中可以比adamw更加高效？

---

### 🤖 Assistant

# Muon 优化器的设计依据与高效原因

## 一句话结论

Muon 的核心设计依据是：**对二维权重矩阵，应该用"矩阵级/谱级"的几何结构来归一化更新量，而不是像 Adam(W) 那样做逐元素（对角）缩放**。它把动量化的梯度做**正交化（极分解的近似）**，使每层权重的更新矩阵所有奇异值 ≈ 1，从而在谱意义上保持稳定。这使它在对矩阵结构敏感的 Transformer 隐藏层上比 AdamW 更快、更省、loss 更低。

---

## 一、设计依据

### 1. AdamW 的局限：对角预条件丢失矩阵结构

Adam/AdamW 的更新本质是：

$$
\Delta W = \frac{\hat m}{\sqrt{\hat v} + \epsilon}
$$

其中 $m$ 是一阶矩、$v$ 是二阶矩，除法是**逐元素**进行的。这等价于用一个**对角预条件矩阵** $D = \mathrm{diag}\!\big(1/\sqrt{\hat v}\big)$ 作用于梯度：

- 对角矩阵只能对每个坐标**独立缩放**；
- 无法刻画权重矩阵**行/列之间的相关性**；
- 无法控制矩阵的**奇异值分布（谱性质）**。

而 Transformer 的 QKV、MLP 权重恰恰都是二维矩阵，这些结构信息被 AdamW 白白丢掉了。

### 2. Muon 的思路：控制更新的"谱"与正交性

对权重 $W \in \mathbb{R}^{m \times n}$，其梯度 $G$ 可做极分解：

$$
G = U P, \qquad U^\top U = I,\ \ P \succeq 0
$$

- $U$ 是**正交因子**（方向部分），所有奇异值都是 1；
- $P$ 是**幅值/拉伸因子**（大小部分）。

Muon **丢弃 $P$，只保留 $U$**，即把更新量正交化。这样每层权重的更新是一个（近似）正交矩阵：

- 谱范数被固定为 1，训练在谱意义上"等距、稳定"；
- 避免了奇异值分布漂移（某些方向被放大、另一些被压缩）导致的梯度爆炸/消失和优化景观变差。

这与"谱范数正则化 / 权重正交性有利于深层网络稳定"的长期观察是一致的——Muon 相当于把这种正则直接做进了优化器。

### 3. Newton–Schulz 迭代：低成本近似正交化

对每个 step 都做 SVD 太贵，Muon 用 **Newton–Schulz 迭代** 近似计算 $U = G(G^\top G)^{-1/2}$。

先把 $X$ 归一化使最大奇异值 ≤ 1，然后迭代（通常 5 次）：

$$
X \leftarrow \Big[aI + b\,XX^\top + c\,(XX^\top)^2\Big] X
$$

其中系数 $(a, b, c) = (3.4445,\,-4.7750,\,2.0315)$ 是一个 5 阶多项式近似，保证 $X \to U$（正交极因子）。它的计算量只是**几次小矩阵乘法**，可用 bf16 + 融合算子实现，相对该层本身的 GEMM 开销很小。

<details>
<summary>展开：为什么这个迭代收敛到正交因子</summary>

设 $X = U\Sigma V^\top$（完整 SVD），则

$$
\big[aI + bXX^\top + c(XX^\top)^2\big]X
= U\,\big[a\Sigma + b\Sigma^3 + c\Sigma^5\big]V^\top
= U\,\big[\sigma p(\sigma^2)\big]V^\top
$$

其中 $p(t)=a+bt+ct^2$。只要多项式 $\sigma p(\sigma^2)$ 在 $\sigma\in[0,1]$ 上逼近常数 1，$X$ 就会收敛到 $UV^\top$，即正交极因子。系数 $(3.4445,-4.7750,2.0315)$ 正是为此优化出来的。
</details>

### 4. 非方阵的范数补偿

正交矩阵的 Frobenius 范数是 $\sqrt{\min(m,n)}$，比普通稠密更新偏小。参考实现对 $m>n$ 的矩阵额外乘 $\sqrt{m/n}$（即 $\sqrt{\max(m,n)/\min(m,n)}$），把更新量级补偿到合理水平。

### 5. 只用一阶动量，内存减半

Muon 只维护**一个动量状态**（AdamW 需要 $m$ 和 $v$ 两个状态），优化器状态内存约减半。大模型下这是实打实的收益（可换成更大的 batch 或模型）。

---

## 二、为什么在 Transformer NLP 任务上比 AdamW 更高效

| 维度 | AdamW | Muon |
|---|---|---|
| 预条件类型 | 逐元素（对角） | 矩阵级（谱/全矩阵） |
| 控制对象 | 每个坐标的二阶矩 | 更新矩阵的奇异值/谱范数 |
| 优化器状态 | $m$ + $v$（2 个） | 动量（1 个） |
| 对矩阵结构的利用 | 无 | 充分 |

1. **全矩阵预条件 > 对角预条件**：Muon 的正交化等价于在每个矩阵上用 $G(G^\top G)^{-1/2}$ 做牛顿型/自然梯度式的预处理，收敛方向更"直"，而 AdamW 只能做坐标级缩放。

2. **谱稳定 → 训练稳定**：每层权重增量的谱范数恒定，减少了对初始化、深度、学习率的敏感，允许使用更大 LR 而不发散。

3. **样本效率更高、最终 loss 更低**：
   - modded-nanogpt 的 speedrun 用 Muon 刷新了单卡 NanoGPT 的最快/最低 loss 记录；
   - Moonshot 的 **Moonlight**（3B/16B MoE）用 Muon 训练 5.7T tokens，报告达到同等 loss 大约只需 AdamW **一半的 FLOPs**（≈ 2× 算力效率）。

4. **更省显存**：单状态动量比 AdamW 双状态省一半优化器内存。

5. **对 LR 更不敏感**：正交化把更新幅值归一，最优学习率区间更宽，调参成本更低。

---

## 三、实践经验与后续验证

- **原论文**：Keller Jordan 等，*Muon: An optimizer for hidden layers in neural networks*（2024）。
- **混合使用是标准做法**：2D 权重（Linear/Conv）用 Muon；1D 参数（bias、LayerNorm、embedding、输出 head）继续用 AdamW——因为这些没有矩阵结构，正交化无意义。
- **《Muon is Scalable for LLM Training》**（Liu et al., 2025）把 Muon 系统验证到 2.5B/13B，并提出两点大规模稳定化改进：
  - **Consistent RMS**：让更新与动量的 RMS 保持一致/可调；
  - 把 **weight decay 施加在动量上**而非参数上。

  结论是 Muon 在同等 token 数下训练 loss 更低，且能配合 1.5× 以上 LR。

<details>
<summary>展开：局限与注意点</summary>

- 只对 2D 权重有效，embedding/head/norm 仍需 AdamW；
- Newton–Schulz 引入额外 FLOPs 和局部显存，需要低精度/融合实现才划算；
- 在数十 B 以上规模仍需要更多验证，LR、weight decay 等超参需按 Muon 特性重调；
- 它刻意丢弃了幅值信息 $P$（不像 Adam 保留逐元素自适应幅值），对需要精细逐元素缩放的场景未必优于 AdamW。
</details>

---

**总结**：Muon 的设计依据是"隐藏层权重更新应保持谱稳定/正交"，它用 Newton–Schulz 迭代低成本地实现极分解正交化，把 AdamW 的逐元素对角预条件升级为矩阵级谱预条件。这使得它在 Transformer 的 2D 权重上收敛更快、更稳定、更省显存，因而在 NLP/LLM 训练中比 AdamW 更高效。

---

### 🧑‍💻 User

为什么muon optim仅限2d shape，以及分布式训练需要处理fsdp和通信问题？

---

### 🤖 Assistant

# 为什么 Muon 只适用于 2D？分布式训练为何要处理 FSDP 与通信？

## 一、为什么 Muon 优化器只限于 2D 形状

### 1. 核心操作（正交化）本质上是"矩阵"运算

Muon 的关键是 Newton–Schulz 迭代：

$$
X \leftarrow \Big[aI + b\,XX^\top + c\,(XX^\top)^2\Big]X
$$

以及它要逼近的目标——极分解的正交因子：

$$
U = X\,(X^\top X)^{-1/2}
$$

这些运算成立的前提是 **$X$ 是一个二维矩阵**：

- **奇异值、谱范数、正交性**这些概念只对 2D 数组（线性映射）有定义；
- $XX^\top$ 是"行空间"的 Gram 矩阵，只有矩阵才有行空间/列空间之分；
- 正交化是在**归一化矩阵的奇异值**，而奇异值分解 $X = U\Sigma V^\top$ 只存在于 2D。

所以 Muon 的数学基础天然就是二维的。

### 2. 对 1D 向量：正交化退化成"归一化"，会丢失全部幅值信息

如果 $X$ 是一个向量 $v \in \mathbb{R}^n$，它的极分解正交因子就是单位向量：

$$
U = \frac{v}{\|v\|}
$$

即 Muon 作用在向量上等价于**把梯度归一化成单位长度**。这会带来两个致命问题：

1. **幅值信息全部丢失**：更新量永远只有固定长度 $\|U\|=1$，失去了自适应步长，训练无法正常进行；
2. 没有 AdamW 那种逐元素自适应缩放（$\Delta = \hat m/\sqrt{\hat v}$），对 bias、LayerNorm 这类参数，逐元素缩放恰恰是最合适的。

因此 1D 参数（bias、LayerNorm 的 $\gamma,\beta$）必须继续用 AdamW。

### 3. 对 ≥3D 张量：需要人为 flatten，语义不明确

卷积权重是 $[\text{out}, \text{in}, k, k]$ 这种 4D 张量。虽然可以 reshape 成 2D（如 $[\text{out}, \text{in}\times k\times k]$）再正交化，但：

- 这种 flatten 是**人为强加**的，行/列空间的几何含义不自然；
- 跨空间维度的"正交"没有明确的物理/数学意义；
- 原始论文和全部成功实验都基于 2D 稠密权重（Linear 层）。

所以标准做法是：**2D 稠密权重用 Muon，其余用 AdamW**。

### 4. "2D" 是必要不充分条件：嵌入层虽是 2D 却不用 Muon

Embedding 矩阵也是 2D（`vocab_size × hidden_dim`），但实际训练中**不用 Muon**，原因是：

- 每个 step 只有 batch 中出现的少数 token 行有梯度，**梯度极度稀疏**；
- 正交化是**跨行耦合**的：第 $i$ 行的更新依赖所有其他行（通过 $XX^\top$），会把"刚更新的稀疏行"与"大量未更新的行"混在一起，破坏已学到的嵌入结构、导致不稳定；
- AdamW 的逐元素/逐行缩放天然适合稀疏更新。

同理，输出 head（常与 embedding 共享权重）也用 AdamW。

---

## 二、为什么分布式训练要处理 FSDP 与通信

核心矛盾一句话：**Muon 的 Newton–Schulz 迭代是"全矩阵、跨行跨列耦合"的操作，而 FSDP/ZeRO/Tensor Parallel 恰恰把矩阵切碎了存到不同设备上。**

### 1. Newton–Schulz 需要"完整的"权重矩阵

FSDP（或 ZeRO-1/2/3）会把每个参数的**梯度、动量按行或列切分**到不同 rank。但 Muon 要计算 $XX^\top$、做正交化，这要求：

- 每个 rank 手里必须有**整块矩阵** $X$；
- 而 $XX^\top$ 是行空间 Gram 矩阵，任何一行的更新都依赖**所有其他行**。

这与 AdamW 形成鲜明对比：

| | AdamW | Muon |
|---|---|---|
| 更新粒度 | 逐元素（element-wise） | 全矩阵（spectral） |
| 分片后 | 各 rank 本地算即可，**零通信** | 必须 all-gather 还原完整矩阵 |
| 行间依赖 | 无 | 有（通过 $XX^\top$ 耦合） |

### 2. 必须额外的 all-gather 通信

要跑 Muon，分布式实现通常需要：

```
反向传播 → 梯度分片在本地
        → all-gather 还原完整梯度矩阵（额外通信）
        → 在完整矩阵上做 Newton–Schulz 正交化
        → 更新动量、重新分片
```

这多出来的 **all-gather** 是 AdamW 没有的，会显著增加通信量，尤其在超大模型上可能成为瓶颈。

### 3. 与 FSDP 现有通信模式叠加、需要重调度

FSDP 本身已经有两类通信：

- **forward**：参数 all-gather；
- **backward**：梯度 reduce-scatter。

加上 Muon 后，优化器步骤里又多了梯度/动量的 all-gather。三者叠加需要重新调度、融合（例如与 reduce-scatter 合并、异步 overlap），否则通信开销会吃掉 Muon 省下的算力。

### 4. 动量状态也要按同样的分片方式存储并保持一致

Muon 的动量状态必须与参数**同样切分**存储；而 NS 迭代用的是动量（或 Nesterov 动量），需要保证：

- 各 rank 上的动量分片在 all-gather 后能正确还原；
- 正交化完成后能正确**重新分片**回各 rank；
- 所有 rank 的动量/更新保持一致，不引入分片不一致的 bug。

### 5. Tensor Parallel 下更麻烦

TP 把权重按列（或行）切分，且不同列对应不同的输出神经元。正交化要求完整矩阵，意味着：

- 需要**跨 TP group 做 gather** 才能凑齐完整矩阵；
- 这会改变 TP 原本精心平衡的"计算/通信"比例，可能破坏流水线并行或 TP 的 overlap。

### 6. 实际解决思路（《Muon is Scalable for LLM Training》等）

后续工作让 Muon 在分布式下可用的做法包括：

- 在优化器步骤中**显式 all-gather 梯度/动量**，还原完整矩阵后再做 NS 迭代；
- 与 ZeRO/FSDP 的 sharding 策略对齐，把 NS 迭代放在 gather 之后、re-shard 之前；
- 对通信与 NS 计算做 **overlap/融合**，降低额外开销；
- 同时提出 **Consistent RMS**、把 weight decay 施加在动量上等技巧，保证分布式+大规模下的稳定性。

---

## 总结

- **只限 2D**：因为 Muon 的数学核心是矩阵的极分解/正交化（奇异值归一），只有 2D 稠密矩阵有这种谱结构；1D 会退化为无意义的归一化，≥3D 需人为 flatten，稀疏的 2D（embedding）又会被跨行耦合破坏，所以实际只用在 2D 稠密 Linear 权重上。

- **分布式要处理 FSDP/通信**：因为 FSDP/TP 把矩阵切碎存储，而 Muon 的正交化是**全矩阵耦合**操作，必须先把完整矩阵 all-gather 回来，多出额外通信，需要与现有 reduce-scatter/gather 重调度、融合，并保证动量状态分片一致。