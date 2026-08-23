# 张量并行FSDP对比

### 🧑‍💻 User

tensor parallel vs fsdp

---

### 🤖 Assistant

Both **Tensor Parallelism (TP)** and **Fully Sharded Data Parallelism (FSDP)** are model parallelism techniques that split a large neural network across multiple GPUs to overcome memory constraints. However, they operate at different granularities and have distinct communication, scaling, and implementation trade-offs.

---

## 1. Tensor Parallelism (TP)

Tensor parallelism splits the **weight matrices of individual layers** (e.g., a linear layer’s $W \in \mathbb{R}^{d_{\text{in}}\times d_{\text{out}}}$) column-wise or row-wise across devices. Each device holds a slice of the weights and computes a partial result; these partials are then combined via efficient **all-reduce** or **all-gather** operations every forward/backward pass.

<details>
<summary>🔍 Example: Column-wise split of a linear layer</summary>

For input $x \in \mathbb{R}^{B \times d_{\text{in}}}$, weight $W$ split into $W_1, W_2$ across two GPUs:

- GPU 1: $y_1 = x W_1$
- GPU 2: $y_2 = x W_2$

The final output $y = [y_1, y_2]$ (concatenation) — an all-gather is often used if subsequent ops need the full tensor. Gradients flow back similarly with an all-reduce on the gradient of $x$.

</details>

- **Communication**: Happens **every layer**, typically small but frequent messages (all-reduce / all-gather).
- **Memory**: Each device stores only a fraction of the layer’s weights, but **activations are typically full** unless combined with activation checkpointing.
- **Scalability**: Best within a single node (high-bandwidth NVLink/InfiniBand). Latency-sensitive because of frequent syncs. Not practical to scale beyond the number of attention heads or hidden dimensions without becoming communication-bound.

---

## 2. Fully Sharded Data Parallelism (FSDP)

FSDP (inspired by **ZeRO Stage 3**) shards **model parameters, gradients, and optimizer states** across all data-parallel workers. Unlike Tensor Parallelism, the sharding is **not layer-local**; instead, each GPU holds only a fraction of the *entire model’s* parameters at any time. When a layer is needed for a forward/backward pass, its parameters are temporarily gathered (or rebuilt) via an **all-gather** communication, and then discarded (resharded) to free memory.

- **Communication**: An all-gather to collect full parameters before a layer’s forward, then a reduce-scatter to aggregate gradients after its backward. These happen **per transformer block** (or per parameter group), not per individual op.
- **Memory**: The model footprint on each GPU is drastically lower because parameters, gradients, and optimizer states are sharded. Activations remain local (unless using additional sharding).
- **Scalability**: Works well across **many nodes** because the communication volume scales with the model size but not directly with the number of GPUs (the total data per GPU for all-gather is independent of world size). It exploits the natural data-parallel hierarchy.

---

## 3. Comparison Table

| Aspect | Tensor Parallelism (TP) | FSDP (ZeRO‑3) |
|--------|--------------------------|---------------|
| **Granularity** | Within a single layer (weight split) | Model‑wide parameter sharding (gather/release per block) |
| **Communication pattern** | All‑reduce / all‑gather **every layer** | All‑gather / reduce‑scatter **per transformer block** |
| **Number of syncs** | Very high (linear in number of layers) | Lower (once per FSDP unit, e.g. per decoder layer) |
| **Per‑GPU memory** | Weihts sharded, but activations still full | Parameters, grads, optimizer states sharded; activations untouched |
| **Bandwidth requirement** | Extremely high (best with NVLink / NVSwitch) | More tolerant; can use lower bandwidth interconnects |
| **Scaling beyond one node** | Difficult due to communication overhead | Excellent (data‑parallel scaling with model sharding) |
| **Ease of implementation** | Requires careful model surgery (column/row splits) | Fairly transparent with frameworks like PyTorch FSDP2 |
| **Mix with other parallelism** | Often combined with PP (pipeline) for huge models | Works alongside PP and DP natively |

---

## 4. When to Use Which

- **Tensor Parallelism** shines when you have **very large single layers** (e.g., huge vocabulary embeddings, massive attention heads) that cannot fit on one device, and you have **high-speed intra‑node links** (NVLink, NVSwitch). It is commonly used inside a single node for models like GPT-3 (175B) in combination with pipeline parallelism.

- **FSDP** is the go‑to choice for **scaling training across many GPUs/nodes** while keeping code modifications minimal. It aggressively reduces memory by sharding optimizer states and parameters, making it ideal for models up to hundreds of billions of parameters when paired with activation checkpointing. Many modern training runs (e.g., Llama 2, Falcon) use FSDP (or ZeRO‑3) as the primary distributed strategy.

- **Hybrid Approaches**: Large model training often **combines** FSDP for data‑parallel sharding with tensor parallelism within a node for the largest layers, and optionally pipeline parallelism to split across nodes. This is how trillion-parameter models are trained.

---

## 5. Summary

- **Tensor Parallelism**: Low‑latency, high‑bandwidth layer splitting for individual large operations. Best inside a node.
- **FSDP**: High‑throughput, memory‑efficient sharding of the entire model across data‑parallel workers. Scales across many nodes.

Choosing between them depends on model architecture, hardware topology, and ease of implementation. In practice, they are often used together for optimal performance at extreme scale.

---

### 🧑‍💻 User

tp的通信频率相比ddp/fsdp高多少以至于强制要求nvlink？

---

### 🤖 Assistant

Tensor 并行（TP）的通信频率**本身并不是比 FSDP 高出一个数量级**（都是每层数次通信），真正导致 NVLink 成为刚需的原因是**通信特征的根本差别**：TP 的通信是**同步、小消息、在计算的关键路径上且极难隐藏**。

下面用定量对比来说明这一点。

---

## 1. 每轮训练的集合通信次数（以 L 层 Transformer 为例）

| 策略 | 一类典型操作的次数（近似） | 说明 |
|------|----------------------------|------|
| **DDP** | **1 次** all‑reduce（梯度桶合并） | 全部梯度打包成一个大消息，在反向传播结束时做一次同步。 |
| **FSDP**（ZeRO‑3） | **≈ 3L 次**集合通信<br>（前向：L 次 all‑gather；反向：L 次 all‑gather + L 次 reduce‑scatter） | 每个 FSDP 单元（通常 1 个 Transformer 块）触发 3 次集体操作。 |
| **TP**（典型 2 卡张量并行） | **≈ 4L ～ 8L 次**集合通信 | 每个注意力/FFN 子层的前向有一个 all‑reduce（或 all‑gather），反向对应一次。例如：QKV 投影一次，attention output 投影一次，MLP 上投影一次，MLP 下投影一次 → 4 对（前向+反向各一次）即 8L 次。 |

> **直观感受**：DDP 整轮只有 1 个同步点，FSDP 有 ~3L 个，TP 有 ~4‑8L 个。对于 80 层的模型，TP 的同步点数量是 DDP 的 **几百倍**。

---

## 2. 为什么 FSDP 能用普通网络，而 TP 必须用 NVLink？

仅仅看“次数”还不够，关键是这些通信的**大小、可重叠程度、以及对延时的敏感度**。

### FSDP 的通信特性
- **消息大小**：每次 all‑gather 拉取一整层的参数（例如几十到几百 MB），reduce‑scatter 尺寸相同。
- **可重叠性**：前向的 all‑gather 可以 **提前预取**，与当前层的计算重叠；反向的 reduce‑scatter 也可以与后续层的计算流水线化。实际暴露在关键路径上的通信时间远小于理论总时间。
- **网络要求**：即使使用 200 Gb/s 的 InfiniBand 也能获得较好的扩展效率，延迟 1‑2 μs 的 RDMA 即可接受，因为单个大消息的传输时间远大于延迟。

### TP 的通信特性（同一层内）
- **消息大小**：每次 all‑reduce 的是 **激活值**（前向）或 **激活梯度**（反向），尺寸为 $B \times S \times d_{model}$。例如 batch=8, seq=4096, hidden=8192 → FP16 大小 8×4096×8192×2 ≈ 512 MB（非常可观）。
- **不可重叠**：这些 all‑reduce 严格位于计算图的 **关键路径** 上。比如经过列切分的线性层后，两个 GPU 的局部输出必须立即通过 all‑reduce 求和，**之后的层必须等待这次 all‑reduce 完全结束**。无法像 FSDP 那样提前拉取下一层的参数来掩盖延迟。
- **频率与带宽的乘积**：假设每张卡上有 N 个 TP 通信 per layer，那么**每秒钟需要完成的 all‑reduce 字节数**为：
  $$
  \text{BW} = \text{层数} \times N \times (2 \times B \times S \times d_{model}) \times 2\text{(前向+反向)} \times \frac{1}{\text{单层计算时间}}
  $$
  对于大模型，这个数值极易冲到 **几百 GB/s 甚至 TB/s**。

### 具体带宽需求示例
以 70B 模型、hidden=8192、seq=4096、batch=8、每层 4 个 all‑reduce 为例：
- 单个 all‑reduce 消息量 ≈ 512 MB（前向）。
- 反向同样大小，即每层 1 GB 的聚合通信量（每个 GPU）。
- 假设一张 GPU 计算一层耗时 ~50 ms，那么需要在这 50 ms 内完成 1 GB 的 all‑reduce → **有效带宽需 ≥ 20 GB/s**。
- 实际上为了不拖慢训练，所需带宽通常是这个值的 2‑3 倍，即 **50‑60 GB/s per GPU**。

常见网络带宽对比：
- **NVLink (A100/SXM)**：900 GB/s **双向**，足以覆盖。
- **InfiniBand HDR (200 Gb/s)**：约 25 GB/s（单向），勉强够但余量很小，一旦 batch 或 hidden 稍大就会成为瓶颈。
- **PCIe 4.0 ×16**：~32 GB/s（双向），受限于延迟和拓扑，实测更差。

更重要的是，**TP 对延迟极其敏感**：每个 all‑reduce 的延迟（通常 5‑20 μs 以上）会累积到关键路径上，而 NVLink 的延迟低至 **＜ 1 μs**，这是普通网络无法比拟的。

---

## 3. 总结——强制 NVLink 的根本原因

TP 的高通信频率**不是单纯次数多**，而是：

1. **所有通信都在关键路径上，无法被计算隐藏**；
2. **单次消息量不小（几百 MB 的激活值）**，总带宽需求轻易超过 50 GB/s/GPU；
3. **微秒级的延迟会直接堆积**成不可忽略的空转时间。

这就使得 TP 只能在具有 **超高带宽、超低延迟** 的域内（同一节点 NVLink/NVSwitch）运行。相比之下，FSDP/DDP 通过大块异步通信降低了带宽和延迟的敏感度，因此用 100‑400 Gb/s 的 RoCE/IB 网络即可高效扩展到多个节点。