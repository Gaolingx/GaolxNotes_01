# Transformer 自回归解码

### 🧑‍💻 User

Transformer 自回归解码具有非常特殊的计算特性。为什么由于 KV 缓存的存在，当 BS=1 时， Transformer 中的每一次矩阵乘法实际上都是矩阵向量乘法。

---

### 🤖 Assistant

在**自回归（autoregressive）解码**阶段，KV cache 的引入从根本上改变了 Transformer 的计算形态。结论先行：**当 batch size = 1 时，解码阶段几乎所有“看起来是 GEMM 的算子，在数学上都退化成了 GEMV（matrix–vector multiply）**。原因可以从计算图和张量形状两方面严格推导。

---

## 1. 自回归解码 + KV cache 的前提条件

我们讨论的是 **decode / incremental inference**，而不是 prefill。

核心约束：

* **一次只生成 1 个新 token**
* 历史 token 的 K / V 已经全部缓存，不再重复计算
* 当前 step 的输入序列长度恒为 **1**

因此：

```text
BS = 1
ΔT = 1   （每 step 只增加 1 个 token）
```

---

## 2. Attention 中的所有矩阵乘法如何退化为 GEMV

### 2.1 Q / K / V 的线性投影

以 Q 为例（K、V 同理）：

[
Q = X W_Q
]

* (X)：`[1, d_model]`
* (W_Q)：`[d_model, d_k]`

这在数学上就是：

```text
(1 × d_model) · (d_model × d_k) = (1 × d_k)
```

**这是一个标准的矩阵 × 向量（GEMV）**，只是实现上仍可能调用 GEMM kernel。

---

### 2.2 Q × Kᵀ：注意力打分

KV cache 的关键影响体现在这里。

* 当前 step：

  * (Q)：`[1, d_k]`
* KV cache：

  * (K_{cache})：`[T, d_k]`

计算 attention score：

[
\text{score} = Q K_{cache}^T
]

形状是：

```text
(1 × d_k) · (d_k × T) = (1 × T)
```

**这是一个向量 dot 一个矩阵的操作，本质仍是 GEMV**：

* 输入：一个 query 向量
* 输出：对所有历史 token 的标量相似度

---

### 2.3 Attention × V

softmax 后：

* (A)：`[1, T]`
* (V_{cache})：`[T, d_v]`

[
\text{output} = A V_{cache}
]

```text
(1 × T) · (T × d_v) = (1 × d_v)
```

**依然是 GEMV**。

---

## 3. FFN 层为什么也是 GEMV

FFN 的三次线性层（以 LLaMA 类结构为例）：

```text
x @ W_up
x @ W_gate
(x_up ⊙ silu(x_gate)) @ W_down
```

在 decode 阶段：

* (x)：`[1, d_model]`
* 所有权重矩阵是静态的

所以三次 matmul 全部是：

```text
(1 × N) · (N × M) = (1 × M)
```

**没有任何维度可以形成“真正的矩阵 × 矩阵”**。

---

## 4. 为什么说“每一次 matmul 实际上都是 GEMV”

综合 attention + FFN：

| 模块               | 名义上的操作 | 实际数学本质 |
| ---------------- | ------ | ------ |
| Q/K/V projection | GEMM   | GEMV   |
| Q × Kᵀ           | GEMM   | GEMV   |
| Attn × V         | GEMM   | GEMV   |
| FFN up/gate/down | GEMM   | GEMV   |

唯一“长”的维度 **T（上下文长度）** 只存在于 **cache 维度**，而不是 batch 或 sequence 维度，因此**永远只在一个轴上展开**。

---

## 5. 这对系统与硬件意味着什么（关键后果）

### 5.1 算力利用率极低

* GEMV 的 **算强度（FLOPs / Byte）远低于 GEMM**
* Tensor Core / AMX / SIMD 宽度无法被填满
* 解码阶段往往 **memory bound**

这正是你在调试中看到的典型现象：

> decode token/s 明显低于 prefill
> bf16 / fp16 在 decode 阶段几乎不比 fp32 快

---

### 5.2 KV cache 访问成为主瓶颈

* 每个 token 都要 **线性扫描 K/V cache**
* T 增长 → 带宽线性增长
* 这也是为什么：

  * FlashAttention 的收益主要在 prefill
  * decode 阶段收益有限

---

### 5.3 BS > 1 才能“恢复” GEMM

如果：

```text
BS = B
```

则：

* Q：`[B, d_k]`
* Q × Kᵀ → `[B, T]`

此时：

* 可以把 **B 个 query 堆成一个小矩阵**
* 形成 **GEMM（虽然仍然是 skinny GEMM）**

这也是：

* **continuous batching**
* **speculative decoding**
* **server-side batching**

在推理系统中极其重要的根本原因。

---

## 6. 一句话总结（本质结论）

> **KV cache 使得自回归解码阶段“时间维度被冻结在 cache 中”，而当前 step 只剩下一个 token。当 BS=1 时，Transformer 的所有线性代数操作在数学上全部退化为矩阵–向量乘法（GEMV），这决定了解码阶段天然是低算强度、强带宽受限的计算模式。**

如果你愿意，下一步我们可以进一步讨论：

* 为什么 decode 阶段 **bf16 / int8 / 量化**的收益远低于预期
* 为什么 llama.cpp / vLLM 都要极端优化 **GEMV kernel + cache layout**
* 如何从 **kernel fusion / cache blocking / head-major layout** 角度缓解这一问题


---

### 🧑‍💻 User

这意味着计算完全受限于内存带宽，为什么这个阶段应用 CUDAGraphs 是最有效的优化之一？

---

### 🤖 Assistant

是的，在 **BS = 1 的自回归解码阶段**，计算几乎完全是 **memory-bandwidth bound**。但这并不意味着“算子层面已经无可优化”。恰恰相反：**当算术强度极低时，任何非必要的调度与控制开销都会直接、线性地暴露在端到端延迟中**。这正是 **CUDA Graphs 在 decode 阶段异常有效的根本原因**。

下面从机理层面展开。

---

## 1. Decode 阶段的真实性能瓶颈分解

在 BS=1、KV cache 存在的条件下，单 token 的 wall-time 可以粗略拆成：

[
T_{token} \approx
\underbrace{T_{mem}}_{\text{不可避免}}

* \underbrace{T_{launch}}_{\text{kernel launch / driver}}
* \underbrace{T_{sync}}_{\text{stream / event}}
* \underbrace{T_{framework}}_{\text{Python / runtime}}
  ]

其中：

* (T_{mem})：扫描 KV cache 的带宽消耗（硬下限）
* 后三项：**纯软件调度成本**

在 prefill（GEMM-heavy）阶段：

* (T_{mem} \ll T_{compute})
* launch / sync 成本被吞没

但在 decode 阶段：

* **(T_{mem}) 本身已经很小（~几十微秒）**
* 于是：

> **launch + sync + runtime 成本开始与“真正计算”同量级，甚至更大**

---

## 2. 为什么 decode 阶段 kernel 特别“碎”

一个典型 decoder layer（以 LLaMA 类模型为例）：

* RMSNorm
* QKV projection
* RoPE
* Q·Kᵀ
* softmax
* Attn·V
* output projection
* RMSNorm
* FFN up
* FFN gate
* silu
* FFN down
* residual add

即便你做了大量 fusion，**仍然会有 10～20 个 kernel / layer**。

假设：

* 32 layers
* 每 layer 10 kernels

```text
≈ 300 kernel launches / token
```

而 **每个 kernel 实际只干几 KB～几十 KB 的内存访问**。

---

## 3. CUDA Graphs 的本质：消灭“调度噪声”

CUDA Graphs 并没有改变任何计算或访存模式，它的价值在于：

### 3.1 一次 capture，之后“零调度”

传统 launch 路径：

```text
Host → CUDA runtime → Driver → GPU
```

Graph replay 路径：

```text
GPU 直接执行已编译的 launch DAG
```

消除的包括：

* kernel launch 参数解析
* driver-side validation
* stream dependency resolve
* event record / wait
* Python / C++ runtime 介入

**这些成本在 decode 阶段是“刚性成本”**。

---

### 3.2 对 memory-bound workload 的放大效应

对于 GEMM-heavy kernel：

```text
T_compute >> T_launch
```

消掉 launch，收益有限。

但 decode 阶段：

```text
T_mem ≈ T_launch
```

因此：

> **Graph capture 往往能直接带来 1.3×～2× 的 token latency 改善**

这不是魔法，而是把“纯软件噪声”从 critical path 中移除了。

---

## 4. 为什么 decode 阶段特别适合 CUDA Graphs

### 4.1 计算图高度稳定

自回归 decode：

* batch size 固定（通常为 1 或小 batch）
* sequence length 每 step +1，但 **tensor shape 可 pad 到 max_len**
* control flow 无分支

这意味着：

* **Graph 是可复用的**
* capture 一次即可 replay 上千 token

---

### 4.2 GEMV / memory kernel 对 launch 极其敏感

你已经注意到一个关键事实：

> bf16 / fp16 / int8 在 decode 阶段提速有限

原因之一正是：

* kernel 本身太小
* launch overhead 吃掉了格式带来的 compute 优势

CUDA Graphs 让这些 kernel **至少能接近“纯带宽极限”**。

---

## 5. CUDA Graphs vs 其他优化手段的对比

| 优化手段                | 对 decode 的收益 | 原因                  |
| ------------------- | ------------ | ------------------- |
| Tensor Core         | 很低           | GEMV 无法填满           |
| 更低精度                | 中低           | 受限于带宽 + launch      |
| FlashAttention      | 低            | decode 无 block GEMM |
| kernel fusion       | 中            | 减少 kernel 数量        |
| **CUDA Graphs**     | **高**        | **消灭调度成本**          |
| continuous batching | 极高           | 恢复 GEMM             |

因此在工程实践中，decode 优化的优先级通常是：

```text
batching > CUDA Graphs > kernel fusion > precision
```

---

## 6. 为什么说 CUDA Graphs 是“最有效之一”，而不是“唯一解”

需要强调边界条件：

* CUDA Graphs **不能**

  * 减少 KV cache 访存量
  * 改变算术强度
  * 解决 BS=1 的结构性问题

它做的是：

> **当算子已经退化到“不可再算快”时，把系统开销压到最低**

这在 decode 阶段，恰好是最关键的一环。

---

## 7. 总结（工程视角）

**自回归 decode = GEMV + KV streaming + 极低算强度**

在这种 regime 下：

* 性能上限由 **HBM / L2 带宽** 决定
* 而 **实际性能** 常被：

  * kernel launch
  * stream sync
  * runtime 调度
    拖到远低于带宽上限

**CUDA Graphs 的价值在于：**

> 在不改变任何数学与访存本质的前提下，最大限度地逼近“带宽理论上限”。

如果你愿意，下一步我们可以具体拆解：

* vLLM / TensorRT-LLM 是如何 graph-safe 地处理 KV cache growth 的
* 为什么很多框架在 **prefill 不用 graph、decode 必用 graph**
* 以及在 graph 模式下，哪些 kernel fusion 反而会变成负优化
