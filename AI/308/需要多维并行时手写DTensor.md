# 需要多维并行时手写DTensor

### 🧑‍💻 User

什么情况下才需要使用dtensor devicemesh手写多d parallel，而不仅仅依赖fsdp的auto warp policy？

---

### 🤖 Assistant

在PyTorch中，FSDP（Fully Sharded Data Parallel）的 `auto_wrap_policy` 是一种非常强大且易用的“开箱即用”工具，它能解决大多数百亿量级模型的分布式训练问题。

然而，当你面对**超大规模模型（千亿/万亿参数）**、**超长上下文**或**非标准模型架构**时，FSDP的自动包装就显得捉襟见肘了。这时候，你就必须引入 `DeviceMesh` 和 `DTensor` 来手写多维并行（Multi-D Parallelism）。

以下是必须放弃纯 FSDP auto-wrap，转而使用 `DTensor` / `DeviceMesh` 手写多维并行的几种核心场景：

### 1. 需要引入张量并行（Tensor Parallelism, TP）时
纯 FSDP 本质上属于数据并行（Data Parallelism）的变体（ZeRO-3）。即使是全切片，在进行前向和后向计算时，单个GPU也需要能够容纳下**单个完整的层（Layer）及其计算过程中的激活值（Activations）**。

<details>
<summary><b>展开查看详细原因：单层显存溢出与通信优化</b></summary>

*   **单层参数过大：** 当模型的单层（例如一个巨大的 MLP 层或 Attention 投影矩阵）大到单个 GPU 无法装下时，FSDP 会触发 OOM（Out of Memory）。此时必须使用 `DTensor` 进行张量并行（TP），将单个权重矩阵（如 $W \in \mathbb{R}^{d \times 4d}$）切分到多个 GPU 上，使得计算本身就是分布式的。
*   **降低通信开销：** FSDP 需要在不同层之间频繁执行 `All-Gather` 和 `Reduce-Scatter`。如果 GPU 节点间的网络带宽有限，这种通信会成为瓶颈。引入 TP 后，可以将 TP 限制在单台物理机（如 8 张卡通过高带宽的 NVLink 通信），而将 FSDP 放在跨机之间（通过 RoCE 或 InfiniBand），这就是典型的 `2D DeviceMesh` 应用场景。
</details>

### 2. 处理超长文本（长上下文）的序列并行（Sequence/Context Parallelism）
当你的模型需要处理 100k、1M 甚至更长的 Token 序列时，主要吃显存的不再是模型权重，而是**激活值（Activations）和 KV Cache**。激活值的显存占用通常与序列长度呈 $O(L)$ 或 $O(L^2)$ 的关系。

<details>
<summary><b>展开查看详细原因：打破 FSDP 的 Sequence 限制</b></summary>

*   FSDP 默认每个 GPU 都能处理一个完整的 Sequence。但在超长上下文中，单个 Sequence 就会撑爆单卡显存。
*   此时你需要使用 `DeviceMesh` 构建一个包含 Context Parallel (CP) 的拓扑维度，配合 `DTensor` 将输入的 Sequence 维度切分到不同的 GPU 上（例如 Ring Attention 或 Ulysses Attention）。这种切分逻辑是 FSDP 的 `auto_wrap_policy` 根本无法感知的，必须手动定义张量的 `Shard` 或 `Replicate` 布局（Placements）。
</details>

### 3. 复杂的非标准网络架构，特别是 MoE (Mixture of Experts)
Transformer 架构非常规律，FSDP 的按层 wrap（如 `transformer_auto_wrap_policy`）工作得很好。但到了 MoE 架构，事情就变了。

<details>
<summary><b>展开查看详细原因：专家并行（Expert Parallelism）的需求</b></summary>

*   在 MoE 模型中，模型的大部分参数集中在“专家”层，而每次前向传播每个 Token 只会激活少数专家。
*   如果你对 MoE 层使用普通的 FSDP，会导致巨大的不必要通信（把所有专家的权重都 All-Gather 过来，但实际上只用了其中两个）。
*   此时需要使用 `DeviceMesh` 构建 **Expert Parallelism (EP)** 维度。你需要手动将不同的专家分配到不同的 GPU 上，利用 `DTensor` 控制 Router 发送 Token 的路径。这通常需要结合 TP (张量并行) + EP (专家并行) + DP (数据并行)，这是一个 `3D DeviceMesh` 的复杂场景。
</details>

### 4. 精细的物理拓扑感知（Hardware Topology Awareness）
FSDP 的 auto wrap 是把所有可用的 GPU 看作一个扁平的 1D 列表。但在拥有上千张 GPU 的超算集群中，网络拓扑是层级化的（Node 内 NVLink 高带宽，Rack 内普通交换机，Rack 间跨核心交换机延迟高）。

<details>
<summary><b>展开查看详细原因：最大化硬件利用率</b></summary>

*   为了达到极致的吞吐量（MFU），你需要精确控制什么通信走什么网络。
*   例如，通过 `DeviceMesh`，你可以显式地定义：
    ```python
    mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node))
    # mesh["tp"] 对应 node 内，走 NVLink
    # mesh["dp"] 对应 node 间，走 IB 网络
    ```
*   使用 `DTensor`，你可以手动决定哪些算子在 `mesh["tp"]` 上切分，哪些在 `mesh["dp"]` 上使用 FSDP。这种细粒度的控制力是 `auto_wrap_policy` 无法提供的。
</details>

### 总结：何时切换？

| 场景特征 | 建议使用的方案 | 并行策略组合 |
| :--- | :--- | :--- |
| 常规模型 (1B ~ 70B)，标准上下文，标准 Transformer 集群 | **FSDP + Auto Wrap** | 1D FSDP (ZeRO-3) |
| 超大单层参数，单卡装不下单层模型 | **DeviceMesh + DTensor** | TP + FSDP |
| 百万级超长上下文 (Long Context) | **DeviceMesh + DTensor** | CP + TP + FSDP |
| 混合专家架构 (MoE) | **DeviceMesh + DTensor** | EP + FSDP / EP + TP + DP |
| 千卡以上集群，需要极致优化通信 MFU | **DeviceMesh + DTensor** | 3D/4D Parallelism (TP+PP+DP+CP) |

**注：** 随着 PyTorch 2.x 的发展（尤其是 PyTorch 2.3+ 的 **FSDP2**），FSDP 底层已经完全基于 `DTensor` 重写。但即使在使用 FSDP2 时，如果涉及到 TP 或定制化的通信需求，你依然需要先手动定义 `DeviceMesh` 并通过 `DTensor` 接口来配置切分策略（Placements），而不是简单地扔给一个 auto-wrap 函数。

---

### 🧑‍💻 User

devicemesh+dtensor如何实现moe训练中的3d parallel（fsdp+ep）

---

### 🤖 Assistant

在 MoE（Mixture of Experts）模型的训练中，真正的 3D 并行通常指的是 **FSDP (Data/ZeRO) + EP (Expert) + TP (Tensor)**。如果只考虑 FSDP 和 EP，这在拓扑上是一个 2D 网格（2D DeviceMesh）。

利用 PyTorch 的 `DeviceMesh` 和 `DTensor`，我们可以非常优雅地实现这种多维并行。相比于过去需要手写复杂的 MPI 风格通信代码（如手动 `All-to-All`），现代 PyTorch（特别是 PyTorch 2.2+ 和 FSDP2）允许你通过**声明式的张量切分（Placements）**来完成这一切。

以下是通过 `DeviceMesh` + `DTensor` 实现 FSDP + EP（可平滑扩展至引入 TP 的 3D 并行）的标准步骤和底层逻辑：

### 1. 初始化多维 DeviceMesh

首先，我们需要将物理 GPU 集群划分为逻辑上的多维网格。假设我们有 16 张卡，我们希望 4 张卡做 FSDP（数据并行），4 张卡做 EP（专家并行）。

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

# 假设全局 world_size = 16
# 创建一个 2D 网格：4 x 4，分别命名为 "fsdp" 和 "ep"
mesh = init_device_mesh("cuda", (4, 4), mesh_dim_names=("fsdp", "ep"))

# 如果是真正的 3D 并行 (例如 16张卡 = 2 FSDP x 4 EP x 2 TP)
# mesh_3d = init_device_mesh("cuda", (2, 4, 2), mesh_dim_names=("fsdp", "ep", "tp"))
```

### 2. 定义 MoE 的张量切分策略 (Placements)

在 MoE 中，非专家层（如 Attention 层）和专家层（如 MLP Experts）的分布式策略是完全不同的：

*   **非专家层 (Non-Experts)：** 在 `"ep"` 维度上应该被**复制 (Replicate)**，在 `"fsdp"` 维度上被**切片 (Shard)** 以节省显存。
*   **专家层 (Experts)：** 假设所有专家的权重堆叠在一个张量中，形状为 $W \in \mathbb{R}^{E \times D_{in} \times D_{out}}$（$E$ 为专家总数）。那么它在 `"ep"` 维度上应该沿着第 0 维（专家维度）被**切片 (Shard(0))**，在 `"fsdp"` 维度上同样被**切片 (Shard)**（ZeRO-3 逻辑）。

<details>
<summary><b>展开查看详细的代码实现：使用 distribute_tensor / distribute_module</b></summary>

```python
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, Shard, Replicate

# 假设 MoE 层定义如下
class MoELayer(nn.Module):
    def __init__(self, num_experts, in_dim, out_dim):
        super().__init__()
        self.router = nn.Linear(in_dim, num_experts)
        # 批量化存储所有专家：shape = [num_experts, in_dim, out_dim]
        self.expert_weights = nn.Parameter(torch.randn(num_experts, in_dim, out_dim))

    def forward(self, x):
        # ... 后续讲解路由逻辑 ...
        pass

model = MoELayer(num_experts=8, in_dim=1024, out_dim=4096)

# 1. 针对非专家层/路由网络 (在 EP 维度复制)
router_weight = model.router.weight
dist_router_weight = distribute_tensor(
    router_weight,
    device_mesh=mesh,
    placements=[Shard(0), Replicate()] # [FSDP维度, EP维度]
    # 在 FSDP 上 Shard(0) 节省显存，在 EP 上所有设备都保留一份完整副本
)
model.router.weight = nn.Parameter(dist_router_weight)

# 2. 针对专家层权重 (在 EP 维度切分)
# 如果总共有 8 个专家，ep_size=4，则 EP 维度每个 GPU 负责 2 个专家
expert_weight = model.expert_weights
dist_expert_weight = distribute_tensor(
    expert_weight,
    device_mesh=mesh,
    placements=[Shard(1), Shard(0)] # [FSDP维度切片, EP维度沿专家维度切片]
)
model.expert_weights = nn.Parameter(dist_expert_weight)
```
</details>

### 3. FSDP2 的包装 (Wrapping)

在 PyTorch 2.3+ 中，新的 FSDP API（即 `fully_shard`）原生支持了多维 `DeviceMesh`。你只需要告诉 FSDP 它应该作用于哪个 Mesh 维度即可。

```python
from torch.distributed.fsdp import fully_shard

# 遍历模型的各个子模块
for name, module in model.named_modules():
    if "expert" in name:
        # 专家层：FSDP 只在 "fsdp" 维度上起作用。
        # 由于我们之前已经在 "ep" 维度通过 DTensor 对其进行了 Shard(0)，
        # 所以最终专家权重实现了 2D 并行（FSDP 切分 + 专家分配）。
        fully_shard(module, mesh=mesh["fsdp"])
    elif "attention" in name or "router" in name:
        # 非专家层同样在 "fsdp" 维度上做纯 FSDP
        fully_shard(module, mesh=mesh["fsdp"])

# 最外层再 wrap 一次
fully_shard(model, mesh=mesh["fsdp"])
```

### 4. 数据流与通信逻辑（最核心难点：All-To-All）

定义好了权重分布后，真正麻烦的是前向传播 (Forward) 时的 Token 路由。在纯 FSDP 中，数据分布（Batch 切分）是不变的。但在 EP 中，不同 GPU 上的 Token 可能被分配给其他 GPU 上的专家。

这就是 `DTensor` 的魔力所在：**通过重新分发张量布局 (Redistribute Placements) 自动触发通信**。

<details>
<summary><b>展开查看数据路由逻辑：从 Shard 转换到 EP 对应的设备</b></summary>

在常规的手写 EP 中，你需要显式调用 `torch.distributed.all_to_all` 来交换 Token。但在 DTensor 体系下，这被抽象为了 Placements 的转换：

1.  **输入状态：** 数据 $X$ 初始是在 `"fsdp"` (代表 Data Parallel) 维度上 `Shard(0)`（按 Batch 切分），在 `"ep"` 维度上 `Replicate()` 的 DTensor。
2.  **路由计算：** Router 计算出每个 Token 应该去哪个 Expert，生成一个分发索引（Dispatch Index）。
3.  **触发 All-To-All：** 我们通过操作，将 Token 按照目标专家重新排列为一个新的 DTensor。我们改变这个新 DTensor 的 Placement：
    *   在 `"ep"` 维度上：从 `Replicate()` 变为 `Shard(0)`（使得每个 GPU 收集到分配给自己专家的所有 Token）。
    *   这个布局转换（`redistribute`）在底层会自动触发 **All-To-All 通信**。
4.  **专家计算：** 现在当前 GPU 获取了专门针对其本地专家（经过 EP Shard）的 Token，执行普通的本地矩阵乘法。
5.  **还原回原来 GPU：** 计算完成后，再次执行 `redistribute`，将结果从 `"ep"` 维度的 `Shard(0)` 变回 `Replicate()`（再次触发 All-To-All），拼接到原来的序列位置中。

*注：虽然 DTensor `redistribute` 可以自动触发，但在极度追求性能的大模型代码中，通常会使用诸如 `megablocks` 或 `vllm/TGI` 底层优化过的定制 All-To-All kernel。*
</details>

### 总结：多维 Mesh 的意义

如果你只用传统的 `auto_wrap_policy`，模型只能感知到一个扁平的进程组。如果强行切分专家权重，FSDP 的 All-Gather 也会把所有专家权重拉取到所有卡上，这**违背了专家并行的初衷（每个卡只算属于自己的那几个专家）**。

使用 `DeviceMesh + DTensor` 的组合：
1.  **解耦物理与逻辑：** 你明确划分了哪几张卡用于分担 Batch（FSDP），哪几张卡用于分担专家（EP）。
2.  **避免冗余通信：** FSDP 的 All-Gather 只会在 `"fsdp"` Mesh 的组内发生，不会把网格另一端无关的专家权重拉过来。
3.  **数学表达直观：** 你只需要将专家权重的张量定义为 `[Shard(0) on EP, Shard(x) on FSDP]`，PyTorch 就会在底层自动维护这种复杂的跨维状态，避免了容易写错的全局 Rank 索引数学计算。