# MoE混合并行切片策略

### 🧑‍💻 User

以moe常见的fsdp+ep并行为例，这类模型通常包含self attn、shared exps、router exps等module，如何使用 DTensor 和 DeviceMap 设计 ZeRO-2/3 与 数据并行的混合切片策略？

---

### 🤖 Assistant

在大型混合专家模型（MoE）的训练中，结合 **FSDP（ZeRO-3/2，数据并行/显存优化）** 与 **EP（专家并行）** 是目前业界的主流方案。

在 PyTorch 的最新分布式架构中，我们通常使用 **`DeviceMesh`**（设备网格，你提到的 DeviceMap 的标准称呼）和 **`DTensor`**（分布式张量）来优雅地表达这种复杂的混合并行策略。

以下是如何针对 `self_attn`、`shared_exps` 和 `router_exps` 设计和实现 ZeRO-2/3 与 EP 混合切片策略的详细方案。

---

### 1. 硬件拓扑定义 (`DeviceMesh`)

首先，我们需要将物理 GPU 集群划分为一个二维网格。假设我们有总共 $N$ 张卡，我们定义一个二维 `DeviceMesh`：$DP \times EP$。
*   **DP / FSDP 维度**：用于切片模型参数、梯度和优化器状态（ZeRO-2/3）。
*   **EP 维度**：用于将不同的专家分布到不同的设备上。

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

# 假设 8 卡机器，2 个物理节点，我们采用 DP=4, EP=2 的配置
dp_degree = 4
ep_degree = 2

# 初始化 2D DeviceMesh
# mesh shape: [dp_degree, ep_degree]
mesh_2d = init_device_mesh("cuda", (dp_degree, ep_degree), mesh_dim_names=("dp", "ep"))

# 提取子网格，用于后续针对不同模块的并行策略
dp_mesh = mesh_2d["dp"]
ep_mesh = mesh_2d["ep"]
```

### 2. 各模块的 DTensor 切片策略设计

在 MoE 模型中，不同模块对数据的处理方式不同（稠密 vs 稀疏），因此需要不同的 `DTensor` `Placement`（放置规则）。

| 模块类型 | 包含组件 | 并行策略 | DTensor Placement (网格维度: `[dp, ep]`) |
| :--- | :--- | :--- | :--- |
| **稠密模块** | `self_attn`, `shared_exps`, `LayerNorm` | **纯 FSDP (ZeRO-3)**：在 DP 维度上切片参数，在 EP 维度上复制（因为所有 token 都要经过这些层）。 | `[Shard(0), Replicate()]` |
| **稀疏模块** | `router_exps` (专家层) | **FSDP + EP**：在 EP 维度上切分专家 (Expert Sharding)，在 DP 维度上切片单个专家的参数 (ZeRO-3)。 | `[Shard(-1), Shard(0)]` (假设维度0是专家维) |
| **路由网络** | `router` (Gate 层) | **纯 FSDP / DP**：通常参数量很小，采用数据并行或 FSDP 均可。 | `[Shard(0), Replicate()]` |

### 3. PyTorch FSDP2 + EP 代码实现

在 PyTorch 2.2+ 中，推荐使用基于 `DTensor` 的 `fully_shard` (FSDP2) 接口来实现。以下是核心代码设计：

```python
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import Shard, Replicate

def apply_hybrid_parallelism(model, mesh_2d):
    dp_mesh = mesh_2d["dp"]
  
    for name, module in model.named_modules():
        # 1. 稠密模块: Self Attention & Shared Experts
        # 策略: 仅在 DP_Mesh 上进行 FSDP (ZeRO-3) 切片。
        # 相当于在 mesh_2d 上应用 [Shard(0), Replicate()]
        if isinstance(module, (SelfAttention, SharedExpert, nn.LayerNorm)):
            # fully_shard 会自动将模块参数转换为 DTensor 并沿 dp_mesh 的第 0 维切片
            fully_shard(module, mesh=dp_mesh)
          
        # 2. 路由模块: Gate/Router
        # 策略: 同稠密模块，参数在 DP 维度切片
        elif isinstance(module, RouterGate):
            fully_shard(module, mesh=dp_mesh)
          
        # 3. 稀疏专家模块: Routed Experts
        # 策略: 专家数量在 EP_Mesh 上切分，单专家参数在 DP_Mesh 上 FSDP 切分
        elif isinstance(module, RoutedExperts):
            # 假设 RoutedExperts 内部有一个参数张量形如 [num_experts, in_dim, out_dim]
            # 我们首先需要使用 DTensor 分发专家
            # 注意：这里通常需要修改模型定义，或者使用 distribute_module
            # 为了简化，我们对整个专家组在 dp_mesh 上做 FSDP，
            # 而内部的专家分发（EP）通过自定义的 AllToAll 通信结合 ep_mesh 处理。
          
            # 对专家模块在 DP 维度上做 ZeRO-3
            fully_shard(module, mesh=dp_mesh)
          
            # EP 的实现通常不在 FSDP 层级，而在 forward 逻辑的 Token 路由阶段 (AllToAll)
            module.setup_ep_mesh(ep_mesh) # 伪代码：将 ep_mesh 传给模块以进行 AllToAll

    # 最后对整个模型包装，处理剩余的未切片层
    fully_shard(model, mesh=dp_mesh)
    return model
```

### 4. 核心组件的底层数据流与通信分析

<details>
<summary><b>点击展开：深入理解 Forward / Backward 过程中的通信行为</b></summary>

在使用 `DTensor` 和 `DeviceMesh` 后，底层的通信会被自动隐式触发。

#### 1. 稠密部分（Self-Attn, Shared Exps）
*   **状态**：参数 `DTensor` 的 placement 是 `[Shard(0), Replicate()]`。
*   **Forward**：触发 **All-Gather**。在 DP 维度上收集完整的参数计算，计算完成后立即丢弃（ZeRO-3 特性）。
*   **Backward**：触发 **All-Gather** 收集参数计算梯度；算完后对梯度触发 **Reduce-Scatter**，将累加后的梯度切片分发回各个 DP Rank。

#### 2. 稀疏部分（Routed Exps）
MoE 专家的通信较为复杂，通常包含两个阶段：
*   **Token Routing (EP 维度)**：
    1.  计算 Gate 权重后，决定每个 Token 去哪个专家。
    2.  触发 **All-To-All**（在 `ep_mesh` 上）。将本地的 Token 发送给拥有对应专家的目标 GPU。
*   **Expert Computation (DP/FSDP 维度)**：
    1.  现在目标 GPU 收到了属于自己专家的 Token。
    2.  触发 **All-Gather**（在 `dp_mesh` 上），收集该专家的完整参数（因为专家的参数在 DP 维度上被 ZeRO-3 切片了）。
    3.  进行前向计算。
*   **Token Un-Routing (EP 维度)**：
    1.  计算完成后，再次触发 **All-To-All**（在 `ep_mesh` 上），将计算结果送回 Token 原本所在的 GPU。

数学上，单次 MoE 层的通信量约为：
$$ Comm_{MoE} = Comm_{AllToAll}(Tokens) + Comm_{AllGather}(Expert\_Weights) $$

使用 `DTensor` 的好处是，只要你正确设置了 Tensor 的维度和 `DeviceMesh` 的 placement，PyTorch 分布式引擎会自动插入 `all_gather` 和 `reduce_scatter`，你只需要手动处理 Token 路由的 `all_to_all` 即可。
</details>

### 总结设计原则

1.  **网格正交性**：严格区分 `dp_mesh` 和 `ep_mesh`。稠密计算只认 `dp_mesh`，稀疏计算跨越两个 Mesh。
2.  **ZeRO 降级选择**：如果网络带宽有限，专家层 (`router_exps`) 的参数在前向时的 `All-Gather` 开销太大，可以将专家模块的 FSDP 降级为 **ZeRO-2**（前向不释放参数，仅切片梯度和优化器状态），通过 `fully_shard(module, mesh=dp_mesh, reshard_after_forward=False)` 实现。
3.  **Shared Experts 同步**：`shared_exps` 实际上就是一个巨大的 FFN。它的显存占比通常很大，强制在其上应用 FSDP（ZeRO-3）是必选项。

---

### 🧑‍💻 User

对比只使用fsdp的HYBRID_SHARD模式+包含 SparseMoeBlock 的 ModuleWrapPolicy，这种混合切片策略有什么优势？

---

### 🤖 Assistant

对比 **纯 FSDP (`HYBRID_SHARD` + 包装 `SparseMoeBlock`)** 与 **基于 DeviceMesh/DTensor 的 FSDP + EP 混合切片策略**，后者的核心优势在于**真正利用了 MoE 模型的“稀疏性”本质**。

纯 FSDP 方案将 MoE 层等价于一个极其庞大的稠密层来处理，而 FSDP+EP 方案则在计算和通信上进行了物理级别的解耦。

以下是 FSDP+EP 混合策略相较于纯 FSDP `HYBRID_SHARD` 的四大核心优势：

### 1. 极致的通信降维：从“搬运海量参数”变为“搬运少量 Token”

这是混合策略最大的优势。

*   **纯 FSDP (`HYBRID_SHARD`)**：在执行 `SparseMoeBlock` 的前向和反向传播时，FSDP 必须通过 **All-Gather** 收集该 Layer 中**所有专家**的完整参数。对于动辄包含 8 个、64 个甚至数百个专家的模型，这种全量参数的频繁组装会直接打满机器内部（或节点间）的通信带宽。
*   **FSDP + EP 混合切片**：利用 **All-To-All** 将 Token 路由到特定的 GPU 上（EP 维度）。GPU 只需要 **All-Gather 自己负责的那个（或几个）专家的参数**。由于 Token 的数据量通常远小于庞大的专家权重参数量，这种“让数据找模型”的策略大幅降低了总体通信量。

### 2. 显著降低显存峰值 (Peak VRAM)

*   **纯 FSDP (`HYBRID_SHARD`)**：在 `SparseMoeBlock` 被唤醒（unshard）的瞬间，当前 GPU 必须在显存中存下所有专家的全量权重。如果 MoE 层有 8 个专家，显存峰值就是单专家的 8 倍，极易导致 OOM。
*   **FSDP + EP 混合切片**：在 EP 并行下，一张卡物理上只被分配了 $E / N_{ep}$ 个专家。在任何时刻，该卡只需要 unshard 属于自己的那部分专家参数，显存峰值与专家总数 $E$ 完美解耦。

### 3. 计算密度提升：聚拢零碎矩阵乘法 (GEMM Efficiency)

*   **纯 FSDP (`HYBRID_SHARD`)**：在本地卡上，Token 被门控网络（Router）分配到不同的专家。由于每个卡只处理属于自己的那个 Data Parallel 的 Batch，分到单个专家的 Token 数量变得非常零碎（例如一个 Batch 4096 个 token，分给 64 个专家，每个专家平均只有 64 个 token）。这会导致 GPU 的 Tensor Core 处于饥饿状态，矩阵乘法效率极低。
*   **FSDP + EP 混合切片**：在执行 All-To-All 路由后，所有来自各个 DP 节点的、需要同一个专家处理的 Token 都会**汇聚到同一张卡上**。这意味着该卡上的目标专家可以一次性处理一个大 Batch 的 Token（例如 $64 \times N_{dp}$ 个 token），极大地提升了算力利用率 (MFU)。

### 4. 突破规模上限 (Scalability)

`HYBRID_SHARD` （节点内 FSDP，节点间 DDP）本质上还是受限于单节点显存。如果 MoE 模型的专家数量进一步增加（例如 DeepSeek-V2/V3 的数百个专家），纯 FSDP 将彻底瘫痪，因为不仅单节点显存塞不下，通信耗时也会掩盖所有计算。FSDP+EP 配合 2D DeviceMesh，允许跨节点的横向无缝扩展。

---

<details>
<summary><b>点击展开：深入理论分析：通信代价的数学推导对比</b></summary>

假设：
*   $N$: 总 GPU 数量
*   $E$: 专家总数
*   $W$: 单个专家的参数量大小
*   $T$: 本地单卡上的 Token 数量
*   $D$: Token 的 Hidden Dimension

**1. 纯 FSDP 通信代价 (按参数 All-Gather)**
在纯 FSDP 下，每处理一个 MoE 层，都需要收集所有专家的参数：
$$ Comm_{pure\_fsdp} \approx 2 \times \frac{N-1}{N} \times (E \times W) $$
*(前向一次 All-Gather，反向一次 All-Gather，忽略 Reduce-Scatter)*
**结论**：通信量与专家总数 $E$ 成正比。MoE 越大，通信直接爆炸。

**2. FSDP + EP 混合通信代价 (Token All-To-All + 局部参数 All-Gather)**
Token 首先进行 All-To-All 路由，然后仅针对本地的 $E/N_{ep}$ 个专家进行参数 All-Gather：
$$ Comm_{hybrid} \approx Comm_{Token\_AllToAll} + Comm_{Local\_Weight\_AllGather} $$
$$ Comm_{hybrid} \approx 4 \times \frac{N_{ep}-1}{N_{ep}} \times (T \times D) + 2 \times \frac{N_{dp}-1}{N_{dp}} \times (\frac{E}{N_{ep}} \times W) $$
*(前反向共计 4 次 Token All-To-All，局部专家参数前反向 All-Gather)*
**结论**：参数通信量被缩减到了原来的 $1/N_{ep}$。只要 $(T \times D) \ll (E \times W)$（在 LLM 训练中通常如此，因为专家通常非常巨大），混合策略的优势就极其明显。

</details>

### 总结

纯 FSDP (`HYBRID_SHARD` + `ModuleWrapPolicy`) 是一个**“万金油”**策略，它不需要修改模型代码就能跑，适合小规模的稠密模型或微型 MoE。

但对于生产环境中的大型 MoE 模型（如包含 `self_attn`, `shared_exps`, `router_exps` 的标准架构），必须通过 `DeviceMesh` 引入 EP 维度。这是由于 MoE **参数巨大但单次激活极少**的物理特性决定的。不使用 EP，就是在强行用稠密的逻辑去处理稀疏的问题，必然会导致显存和算力的双重浪费。