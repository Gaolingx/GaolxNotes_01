# FlashAttention Online Softmax实现与优势

### 🧑‍💻 User

在标准的flashattention实现中，说说online softmax是如何实现的，相比完整的softmax优势有哪些？

---

### 🤖 Assistant

在**标准 FlashAttention（FA v1 / v2）实现中**，所谓的 **online softmax** 指的是：

> **在不显式存储完整 attention score 矩阵的前提下，一边流式（block by block）读取 K/V，一边数值稳定地计算 softmax 归一化结果。**

下面从 **算法形式、数值稳定性处理、与“完整 softmax”的对比优势** 三个层面系统说明。

---

## 一、完整 Softmax 的问题回顾（Attention 的瓶颈来源）

标准 Attention 计算：

[
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
]

经典实现流程：

1. 计算完整 score 矩阵 (S = QK^T)（shape: `[Lq, Lk]`）
2. 对每一行做 softmax（需要整行）
3. 再与 (V) 相乘

### 核心问题

* **显存占用**：
  需要存下 (O(L^2)) 的 score / prob
* **带宽瓶颈**：
  score 写回 HBM，再读出来做 softmax
* **算子分裂**：
  GEMM → softmax → GEMM，难以融合
* **cache locality 极差**

FlashAttention 的目标就是：

> **不物化（materialize）score / prob 矩阵**

---

## 二、Online Softmax 的核心思想

### 1. 数值稳定 softmax 的等价变形

标准 softmax（单行）：

[
\mathrm{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
]

**关键点**：
softmax 只依赖两个量：

* 当前行的 **最大值 m**
* 当前行的 **指数和 l = \sum e^{x_j - m}**

这两个量是**可增量更新的**。

---

### 2. Online / Streaming Softmax 更新公式

假设 attention score 按 **K block** 流式计算：

对第 (t) 个 block，得到一组 score (s^{(t)})

维护两个 running 状态：

* running max：(m^{(t)})
* running exp-sum：(l^{(t)})

初始化：
[
m^{(0)} = -\infty,\quad l^{(0)} = 0
]

**每个 block 更新：**

[
m^{(t)} = \max(m^{(t-1)}, \max(s^{(t)}))
]

[
l^{(t)} =
l^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}}
+
\sum_i e^{s^{(t)}_i - m^{(t)}}
]

这一步是整个 FlashAttention 的数值核心。

---

### 3. 同步更新 Attention 输出（关键）

Attention 输出本质是：

[
O = \sum_j \mathrm{softmax}(s_j) \cdot v_j
]

FlashAttention 在 **softmax 还没完全算完时** 就同步累积输出：

维护 running output (o^{(t)})：

[
o^{(t)} =
o^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}}
+
\sum_i e^{s^{(t)}_i - m^{(t)}} \cdot v_i
]

最终输出：

[
O = \frac{o^{(T)}}{l^{(T)}}
]

**注意：**

* softmax 的分母 (l) 是最后一次性除
* 中间完全不需要存 prob

---

## 三、FlashAttention 中的完整流程（单个 Q block）

对每个 Q block：

```
m = -inf
l = 0
o = 0

for K,V block in sequence:
    s = Q @ K^T          # on-chip
    m_new = max(m, max(s))
    alpha = exp(m - m_new)

    l = l * alpha + sum(exp(s - m_new))
    o = o * alpha + sum(exp(s - m_new) * V)

    m = m_new

O = o / l
```

**全部在 SRAM / registers / shared memory 内完成**

---

## 四、相比“完整 softmax”的优势分析

### 1. **显存复杂度：O(N²) → O(N)**

| 实现             | 是否存 score / prob | 显存复杂度    |
| -------------- | ---------------- | -------- |
| 标准 Attention   | 是                | (O(L^2)) |
| FlashAttention | 否                | (O(L))   |

这是 FA 能支持超长上下文的根本原因。

---

### 2. **带宽大幅降低（这是最关键的）**

* 不写回 score
* 不读回 prob
* 所有中间态在 on-chip 完成

在现代 GPU 上：

> **Attention 是 bandwidth-bound，而不是 compute-bound**

FlashAttention 通常可带来：

* 2×～4× 端到端 speedup
* 在长序列下更明显

---

### 3. **算子融合（Kernel Fusion）**

Online softmax 使得：

* GEMM(QKᵀ)
* softmax
* GEMM(PV)

**融合成一个 kernel**

避免：

* kernel launch 开销
* global memory round-trip

---

### 4. **数值稳定性不下降（等价于标准 softmax）**

* 使用 running max 保证指数稳定
* 与“先算完整 softmax”在数学上严格等价
* 不引入近似误差（区别于一些低精度 tricks）

---

### 5. **非常适合 block / tile 化并行**

* K/V block natural streaming
* 与 GPU 的：

  * shared memory
  * warp-level reduction
  * Tensor Core GEMM
    高度匹配

这也是为什么 FlashAttention 的性能与 GPU 架构强绑定。

---

## 五、与其他“近似 softmax”方法的本质区别

需要区分：

| 方法                              | 是否近似 | 是否改变模型语义 |
| ------------------------------- | ---- | -------- |
| FlashAttention (online softmax) | ❌    | ❌        |
| Linformer / Performer           | ✅    | ✅        |
| Kernelized attention            | ✅    | ✅        |

FlashAttention **只是改变计算顺序，不改变结果**。

---

## 六、一句话总结

> **FlashAttention 的 online softmax 本质是：
> 用“可结合（associative）的数值稳定 softmax 累积公式”，
> 将 attention 的 softmax 从“需要完整矩阵的离线计算”，
> 改写成“block streaming 的在线归一化计算”，
> 从而彻底消灭 O(N²) 的中间存储和带宽瓶颈。**

如果你愿意，下一步我可以进一步展开：

* backward 中 online softmax 的反向传播
* FA v2 相比 v1 在 softmax 组织上的差异
* 与 xFormers / Triton 实现的细节对比
