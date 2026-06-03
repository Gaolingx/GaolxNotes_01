# FSDP2EPSP实现万卡MoE训练

### 🧑‍💻 User

fsdp2+ep+sp如何实现从单节点到万卡scaling的超大规模moe训练？

---

### 🤖 Assistant

实现从单节点（如 8 张 GPU）到万卡（10,000+ GPUs）级别的超大规模 MoE（Mixture of Experts）模型训练，是当前大模型领域最具挑战性的工程问题之一。

**FSDP2（Fully Sharded Data Parallel v2）+ EP（Expert Parallelism）+ SP（Sequence Parallelism）** 的组合，是目前公认的解决超大参数量（万亿级别）和超长上下文（百万级别 Token）的最优解。

以下是这三者如何协同工作，并实现从单节点到万卡平滑 Scaling 的深度解析。

---

### 一、 核心技术解构：三驾马车如何分工？

在 MoE 架构中，模型被分为**稠密层**（Dense layers，如 Attention、LayerNorm）和**稀疏层**（MoE layers，即 Experts）。这三种并行策略针对不同的维度进行切割：

1.  **FSDP2 (负责稠密层的显存与计算伸缩)**
    *   **机制**：FSDP2（通常指代支持 2D/Hybrid Sharding 的进化版 FSDP）将优化器状态、梯度和参数进行切片。
    *   **作用**：在万卡规模下，纯 1D FSDP 的通信开销（All-Gather 和 Reduce-Scatter）会随着 GPU 数量 $N$ 呈线性增长，导致网络瘫痪。FSDP2 采用**混合分片（HSDP）**：在节点内（如 8 卡）或较小的通信组内进行切片（Sharding），在通信组之间进行数据并行复制（Replication）。
2.  **EP / 专家并行 (负责 MoE 层的参数伸缩)**
    *   **机制**：将不同的专家（Experts）放置在不同的 GPU 上。每个 GPU 只保留部分专家的参数。
    *   **作用**：解决 MoE 模型参数量爆炸的问题。前向传播时，利用 `All-to-All` 通信，将输入 Token 路由到对应的 GPU 上的专家进行计算，计算完成后再通过 `All-to-All` 返回结果。
3.  **SP / 序列并行 (负责超长上下文的显存伸缩)**
    *   **机制**：在 Sequence 维度对输入进行切片。常采用 DeepSpeed-Ulysses 或 Ring-Attention 机制。
    *   **作用**：打破单卡对 Context Length 的显存限制，降低激活值（Activations）的显存占用，使其从 $\mathcal{O}(S^2)$ 或 $\mathcal{O}(S)$ 均摊到各个 GPU 上。

---

### 二、 从单节点到万卡的 Scaling 演进路线

#### 1. 单节点阶段 (Single Node, 1-8 GPUs)
在这个阶段，主要是验证并行策略的正确性和单节点的极致吞吐。
*   **通信环境**：极高带宽的节点内互联（如 NVLink 4.0，双向 900GB/s）。
*   **策略映射**：
    *   **FSDP2**：退化为普通的 FSDP (Sharding Degree = 8)。
    *   **EP**：假设有 8 个专家，每个 GPU 分配 1 个专家。Token 的路由全部通过 NVLink 完成 `All-to-All`。
    *   **SP**：如果是长文本，Sequence 分成 8 份分配给 8 张卡。

#### 2. 中等规模集群 (64 - 1,024 GPUs)
跨节点网络（RoCE v2 或 InfiniBand，通常 400Gbps）成为瓶颈。策略必须具备**拓扑感知（Topology-aware）**能力。
*   **FSDP2 (Hybrid Sharding)**：开始采用 2D 切片。例如 1024 卡，设置 Sharding Degree = 64（一个机柜内的 GPU），Replication Degree = 16。这样将 FSDP 的环大小限制在 64，避免了 1024 卡的全局 All-Gather。
*   **EP (Hierarchical All-to-All)**：不再进行全局的 EP。采用 **EP Group** 概念。比如将 64 张卡设为一个 EP Group，涵盖所有专家。Token 的路由被严格限制在组内，从而避免跨 Spine 交换机的海量 `All-to-All` 通信。
*   **SP**：与 FSDP/EP 正交，SP 的通信尽量安排在同一个节点内或同一个 Tor 交换机下（带宽充裕的地方）。

#### 3. 万卡超大规模 (10,000+ GPUs)
万卡级别的核心痛点是**长尾效应、网络拥塞（Incast）和容错**。

*   **极致的 3D 并行拓扑映射**：
    *   令总卡数 $N = 10,000$。
    *   将网络划分为物理结构（例如：Node -> Rack -> Pod -> Cluster）。
    *   **SP 组**：大小设为 8，完全在 Node 内（NVLink），消除 Sequence 维度的跨节点通信延迟。
    *   **EP 组**：大小设为 64 或 128（通常对应一个 Rack/Leaf Switch 覆盖的范围），利用无阻塞胖树网络进行 `All-to-All`。
    *   **FSDP 组**：Sharding Group 设为 256，Replication Group 设为 40 左右。
*   **计算与通信的极致重叠 (Overlapping)**：
    万卡下，通信延迟极大。必须做到在计算第 $L$ 层的 MoE 时，预取（Pre-fetch/All-gather）第 $L+1$ 层的稠密参数；同时在做稠密层计算时，进行 MoE 的 Token 路由（All-to-All）。

---

### 三、 万卡规模下的关键优化技术 (Secret Sauce)

要让这套组合在万卡上真正跑起来并达到高 MFU（Model Flops Utilization），需要以下深度优化：

<details>
<summary><b>点击展开：万卡 Scaling 的进阶优化技术</b></summary>

#### 1. 专家负载均衡 (Expert Load Balancing)
在万卡规模下，如果 Token 路由倾斜（例如大部分 Token 涌向同一个专家），会导致 9,999 张卡等待 1 张卡。
*   **解决方案**：除了传统的 Auxiliary Loss，现在多采用 **Token Dropping**（丢弃溢出的 Token，但在长文本/高价值数据中不可取）或 **Expert Capacity 动态调整**。
*   **无损均衡 (Dropless MoE)**：使用全局的 Token 重新分配算法，或者引入额外的负载均衡路由层，确保每个 GPU 收到的 Token 数量严格为 $\frac{\text{Batch Size} \times \text{Seq Length}}{\text{EP Group Size}}$。

#### 2. EP 与 SP 的通信融合 (Communication Fusion)
SP 需要进行 Sequence 维度的 All-Gather/Reduce-Scatter，而 EP 需要进行 All-to-All。
*   **优化方案**：在某些架构（如 DeepSpeed Ulysses）中，SP 本身就利用了 All-to-All。可以将 SP 的 All-to-All 和 MoE 路由的 All-to-All 进行算子融合（Kernel Fusion），减少一次网络穿越。

#### 3. FSDP2 的异步执行
万卡级别的梯度同步极慢。FSDP2 的 Reduce-Scatter 必须做到高度异步。
*   使用专用的通信流（CUDA Streams）。
*   利用网络网卡（如 NVIDIA Mellanox NDR）的 **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** 功能，将 Reduce 操作下沉到交换机硬件中，极大减轻万卡级别的 FSDP 通信压力。

#### 4. 显存优化：Activation Checkpointing (AC) 与 SP 的配合
超大模型 + 超长上下文会撑爆显存。
*   **选择性重计算 (Selective AC)**：只丢弃占用显存最大但计算最快的算子（如 Attention Map）的激活值。
*   结合 SP 时，每个 GPU 只保存本地 Sequence 分片的激活值。在反向传播重计算时，通过 FSDP2 的 All-Gather 重新获取参数即可。

</details>

### 四、 数据流转示例：一个 Token 的万卡之旅

假设我们正在训练一个万亿参数、1M 上下文的 MoE 模型：

1.  **输入切片 (SP)**：一段 1M Token 的序列进入集群，被切分成 8 份（每份 128k Token），分配给同一个节点内的 8 张 GPU（SP Group = 8）。
2.  **稠密层前向 (FSDP2)**：GPU 触发异步请求，向所在的 FSDP Sharding Group（可能跨节点）请求 Attention 层的完整参数。拿到参数后，在本地进行 Attention 计算（如果用 Ring-Attention，还要进行本地到本地的 P2P 通信）。
3.  **专家路由 (EP)**：经过 Attention 层后，Token 需要进入 MoE 层。GPU 根据 Router 的结果，将这 128k Token 打包，通过网络交换机（InfiniBand），使用 `All-to-All` 发送给 EP Group（例如 64 张卡）内不同的专家。
4.  **专家计算与返回**：对应的 GPU 接收到 Token，利用本地的专家参数完成 MLP 计算，再次通过 `All-to-All` 将结果发送回原本的 GPU。
5.  **反向与更新**：反向传播时路径相反。计算出的稠密层梯度，通过 `Reduce-Scatter` 规约到 FSDP Group 内更新本地负责的参数分片；专家梯度直接在本地更新。

### 总结

`FSDP2 + EP + SP` 的组合并不是简单的策略堆叠，而是一个精密的 **3D 通信拓扑设计问题**。从单节点到万卡的核心跃迁在于：**从依赖全局通信（Global Communication）转向依赖层级化、局部化的通信（Hierarchical/Domain-restricted Communication）**，并极致榨干计算与通信的重叠空间。

---

### 🧑‍💻 User

torchtitan/deepspeed/Megatron-Core有没有最佳实现供参考？

---

### 🤖 Assistant

这三个框架都是当前大模型训练的顶级基座，但它们对 `FSDP2 + EP + SP` 的支持成熟度、命名习惯和底层哲学有所不同。

如果你要寻找**万卡级别的最佳实践参考**，以下是这三个框架的实现现状及参考指南。

---

### 一、 TorchTitan：PyTorch 原生 FSDP2 的官方“教科书”

**TorchTitan** 是 Meta 和 PyTorch 团队为展示 PyTorch 2.x 最新原生特性而打造的端到端参考实现。如果你明确要求使用 **FSDP2**（即 PyTorch 的 `torch.distributed._composable.fsdp`），这是唯一的官方权威参考。

*   **FSDP2 实现**：TorchTitan 原生支持真正的 2D FSDP（HSDP）。它通过 `DeviceMesh` 定义集群拓扑，并在不同的维度上应用切片和复制。
*   **SP 实现**：支持通过异步 Tensor Parallelism (Async TP) 和原生的 Ulysses SP 来处理长序列。
*   **MoE 支持**：目前正在快速迭代，原生代码中已经包含了基于 `torch.compile` 和 `DTensor` 的 MoE 路由实现。

**代码参考路径：**
```python
# TorchTitan 核心 2D Parallelism (FSDP2) 初始化示例
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard

# 假设 64 卡，构建 8x8 的 2D Mesh (Replicate x Shard)
mesh = init_device_mesh("cuda", (8, 8), mesh_dim_names=("dp_replicate", "dp_shard"))

# 将 FSDP2 应用于模型层
fully_shard(model_layer, mesh=mesh["dp_shard"])
```
*   **最佳适用场景**：想要极致紧跟 PyTorch 原生生态，依赖 `torch.compile` 进行加速，且不想引入复杂第三方 C++ 算子的团队。

---

### 二、 Megatron-Core (NVIDIA)：万卡 Scaling 的“绝对霸主”

在真实的 10,000+ GPU 集群中，**Megatron-Core (MCore)** 是工业界最成熟的选择（如 Llama 3、Grok-1.5 等均基于或参考了其架构）。

需要注意的是，**Megatron 不叫 FSDP2**。Megatron 的等效组合是：**分布式优化器 (Distributed Optimizer/ZeRO-1) + TP (张量并行) + CP (上下文并行，即 SP) + EP (专家并行)**。

*   **EP 实现极强**：MCore 实现了非常高效的基于 `GroupedGEMM` 的无填充（Pad-free）MoE 计算，以及最成熟的跨节点层次化 `All-to-All`。
*   **SP/CP 实现**：MCore 将长文本切分称为 **Context Parallelism (CP)**（支持 Ring-Attention 机制），而将配合 TP 降低显存的技术称为 Sequence Parallelism (SP)。两者结合能完美解决长上下文显存问题。
*   **替代 FSDP 的机制**：万卡下纯 FSDP 通信开销大。MCore 倾向于在节点内做 TP，跨节点做 PP (Pipeline) 和 DP，并配合 Distributed Optimizer 达到类似 FSDP 的显存节省，但通信效率在 NVLink+InfiniBand 架构下更高。

**代码参考路径：**
关注 Megatron-Core 源码中的 `megatron/core/transformer/moe/` 和 `megatron/core/distributed/` 目录。
```bash
# Megatron-Core 的典型万卡运行配置参数示例
--tensor-model-parallel-size 8 \
--context-parallel-size 8 \      # 对应长文本 SP
--expert-model-parallel-size 64 \ # 对应 EP
--sequence-parallel \            # 激活值显存优化
--use-distributed-optimizer      # 对应 FSDP 的参数切片效果
```
*   **最佳适用场景**：真正的万卡集群（特别是全套 NVIDIA A100/H100 硬件），追求极致的 MFU（Model Flops Utilization）和稳定性。

---

### 三、 DeepSpeed：ZeRO 架构与 Ulysses SP 的“发源地”

DeepSpeed 虽然在纯大厂万卡训练中的主导地位近年来受到 MCore 的挑战，但其易用性和对特定并行技术（如 Ulysses）的首创使其依然是极佳的参考。其对应的技术栈是 **ZeRO-3 (等效 FSDP) + ZeRO-EP + DeepSpeed-Ulysses**。

*   **ZeRO-EP**：DeepSpeed 在 MoE 方面有极深的积累（DeepSpeed-MoE），专门针对 MoE 层的参数和梯度通信进行了优化。
*   **DeepSpeed Ulysses (SP)**：当前被业界广泛抄作业的长文本 SP 方案。通过 `All-to-All` 交换序列切片，比 Ring-Attention 更易于实现，且通信效率极高。

**代码参考路径：**
```json
// DeepSpeed 的 JSON config 示例
{
  "zero_optimization": {
    "stage": 3 // 等效于 FSDP 全切片
  },
  "sequence_parallel": {
    "type": "ulysses",
    "degree": 8
  },
  "moe": {
    "ep_world_size": 64, // 专家并行组大小
    "num_experts": 256
  }
}
```
*   **最佳适用场景**：中大规模集群（千卡级别），希望快速通过修改 Config 跑通模型，对极长文本（1M+ Token）的 Ulysses 机制有强烈需求的团队。

---

### 总结与选型建议

<details>
<summary><b>点击展开：三个框架的核心对比矩阵</b></summary>

| 维度 | TorchTitan | Megatron-Core | DeepSpeed |
| :--- | :--- | :--- | :--- |
| **底层实现 (FSDP等效)** | 纯正 PyTorch `FSDP2` | `DistributedOptimizer` + TP + DP | `ZeRO-3` / `ZeRO-1` |
| **长序列/SP 支持** | Async TP / Ulysses 原生化 | `Context Parallelism` (Ring-Attention / Ulysses) | `DeepSpeed-Ulysses` 首创者 |
| **MoE 支持度** | 原生 `DTensor` 实验性 | 极致优化 (`GroupedGEMM`, Dropless) | 成熟的 `DeepSpeed-MoE` / ZeRO-EP |
| **万卡实战证明** | Llama 3 (Meta 内部定制版) | 业界公认标准 (Grok, Nemotron) | 早期 Bloom, 现多用于千卡规模 |
| **代码可读性与上手** | ⭐⭐⭐⭐⭐ (Pythonic) | ⭐⭐ (C++ 算子多，极其硬核) | ⭐⭐⭐⭐ (Config 驱动) |

</details>

**最终建议：**
1.  **如果你想学习最前沿的 PyTorch Native 代码结构**：去看 **TorchTitan**，它是未来 PyTorch 生态发展的风向标。
2.  **如果你明天就要在一个真实的 10,000 张 H100 集群上启动训练，并且背着极高的吞吐 KPI**：无脑选择 **Megatron-Core** 及其衍生版（如 NVIDIA NeMo），这是目前万卡容错、网络拥塞处理和极致性能优化的唯一真理。