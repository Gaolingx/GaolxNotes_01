# MoE MLP使用EP而非TP

### 🧑‍💻 User

为什么参数量巨大的moe模型mlp layers要用大规模ep而不是tp？

---

### 🤖 Assistant

这是一个**体系结构—算子—通信**三者强耦合的问题。结论先行：
**在 MoE 的 MLP（Expert FFN）层上采用大规模 EP（Expert Parallelism），而不是 TP（Tensor Parallelism），是因为 MLP 本身是“天然可切分的专家集合”，用 EP 能把参数、计算与通信模式同时做成最优；而用 TP 会在通信、算子形态和负载均衡上全面劣化。**

下面分层展开。

---

## 一、MoE MLP 的结构决定了“专家级并行”是自然分解

MoE 的 FFN 本质是：

[
\text{FFN}(x) = \sum_{e \in \text{TopK}(x)} g_e(x),\text{MLP}_e(x)
]

其中：

* 每个 expert 的 MLP 是**完全独立的一套权重**
* 每个 token **只走极少数（Top-K）专家**
* 不同 expert 之间**没有任何参数或计算依赖**

这意味着：

> **Expert 维度本身就是一个天然的并行维度**

如果你把并行轴选在 expert 维度上（EP）：

* 每个 rank 持有若干完整 expert
* token 被 router 分发到对应 rank
* 每个 rank 做的是**完整、连续的 GEMM**

这是最“物理正确”的切分方式。

---

## 二、为什么不用 TP：TP 在 MoE-MLP 上是“反结构的”

### 1. TP 强行切 MLP，会制造大量 All-Reduce

对一个普通 dense FFN：

[
\text{MLP}(x) = W_2 ,\sigma(W_1 x)
]

TP 的典型代价是：

* 每一层 linear 后需要 **All-Reduce / Reduce-Scatter**
* 这是可以接受的，因为 dense FFN **所有 token 都参与**

但在 MoE 中：

* 每个 expert 只处理 **极少量 token**
* TP 会导致：

  * **极小 batch GEMM**
  * 但仍然需要 **全 TP 组通信**

结果是：

* 计算量 ∝ 实际 token 数
* 通信量 ∝ TP world size（固定）
* **通信/计算比直接爆炸**

👉 **极度不符合 roofline**

---

### 2. TP 会破坏 token–expert 的稀疏性

MoE 的性能来源之一是：

* token-expert 是稀疏映射
* 非激活专家根本不参与计算

但 TP 的行为是：

* 把一个 expert 的权重拆到多个 rank
* 即使只有 3 个 token 进这个 expert
* **所有 TP rank 都得被拉起来干活 + 通信**

这等价于：

> 把“专家级稀疏”强行降维成“全局密集通信”

这是结构性退化。

---

### 3. TP 会把 GEMM 变成 cache / launch / latency bound

你之前在 CPU、GPU GEMM 的讨论中其实已经触到这一点：

* 大 GEMM → compute-bound
* 小 GEMM → launch / L2 / latency-bound

在 MoE：

* 单个 expert 的 token 数波动极大（几十到几百）
* TP 切完以后：

  * 每个 rank 上的 GEMM 维度更小
  * kernel 启动、pipeline、寄存器利用率全都劣化

**EP 保留了“单 rank 内最大可用 GEMM”**

---

## 三、EP 的通信是“一次性、可隐藏、可聚合”的

### 1. EP 的通信只发生在 Router 边界

EP 的主要通信只有两次：

1. **Dispatch**：token → expert rank（All-to-All / P2P）
2. **Combine**：expert output → 原 token 顺序

特点：

* 发生在 MLP block 的边界
* 可以：

  * 与 attention / 上下游计算 overlap
  * 合并多个 layer
  * 用 fused kernel（如 DeepSpeed / Megatron 的 MoE kernels）

**通信是粗粒度、可调度的**

---

### 2. TP 的通信是“层内、同步、不可隐藏”的

TP 的通信：

* 在每一个 linear 后
* 是算子内部的强同步点
* 很难和别的东西 overlap

这在 MoE 中尤为致命，因为：

* expert MLP 层数多（通常 2–4 层）
* token 数小但通信频繁

---

## 四、参数规模视角：EP 才是真正“能撑起超大模型”的方式

你问的是“**参数量巨大的 MoE**”，这里是关键点：

### 1. EP 是**参数线性扩展**的

假设：

* 每个 expert 10B 参数
* 256 个 expert

用 EP：

* 每个 rank 放 1–2 个 expert
* **无复制**
* 总参数 ∝ #rank

用 TP：

* 每个 expert 的参数要在 TP 组内分片
* 但 router + load balance + sparse token
* 会导致 **隐性复制、激活 buffer、通信 buffer**

在工程上：

> **EP 才是唯一能把 MoE 参数“摊平”的方式**

---

### 2. 工业实现的共识

你看到的主流系统：

* Megatron-LM MoE
* DeepSpeed MoE
* GShard / Pathways
* Qwen / DeepSeek / Mixtral 系列

**全部是：**

* Attention：DP + TP（有时加 CP）
* MoE-MLP：**大规模 EP + 小规模 TP（可选）**

这不是偶然，而是长期踩坑后的稳定解。

---

## 五、什么时候“EP + TP 混合”才有意义？

并不是说 TP 完全没用，而是：

### 合理的混合模式是：

* **EP 在 expert 维度**
* **TP 在 expert 内部（小规模）**

适用条件：

* 单个 expert 本身已经非常大（如 >10B）
* 单 expert GEMM 已经接近单卡算力瓶颈
* token-per-expert 足够多

此时：

* TP 规模通常 ≤2–4
* 绝不会是主并行轴

---

## 六、一句话总结（工程直觉版）

> **MoE 的 MLP 是“很多个互不相干的大 FFN”，
> 最好的并行方式就是“一个人扛一个 FFN”。
> 用 TP 去切它，相当于让 8 个人一起抬一张只有 3 本书的桌子，还要不停对口令。**

如果你愿意，下一步我们可以从：

* **prefill vs decode 下 EP 的负载抖动**
* **expert capacity factor / padding 的真实性能代价**
* **为什么 EP 在 decode 阶段更重要**
* **MoE kernel 中为什么要做 token sort / block-sparse GEMM**

这些角度继续往下拆。
