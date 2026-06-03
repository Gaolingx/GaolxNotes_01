# tpfsdp2epsp万卡moe训练方案

### 🧑‍💻 User

tp+fsdp2+ep+sp如何实现从单节点到万卡scaling的超大规模moe训练？

---

### 🤖 Assistant

实现从单节点（如 8 张 GPU）到万卡（10,000+ GPUs）级别的超大规模 MoE（Mixture of Experts）模型训练，是目前 AI 系统工程中最具挑战性的任务之一。你提到的 **TP（张量并行） + FSDP2（完全分片数据并行 v2） + EP（专家并行） + SP（序列并行）** 组合，正是目前业界（如 DeepSeek、Meta 等）最前沿的 4D 混合并行架构。

这种架构的核心在于**最大化利用节点内的高带宽（NVLink）和节点间的网络（InfiniBand/RoCE），同时将显存墙和通信墙的压力降到最低**。

以下是实现这一超大规模 Scaling 的系统性设计方案与演进路线。

---

### 核心并行策略与 DeviceMesh 映射

在 PyTorch 2.x（特别是 FSDP2）中，所有这些并行策略都可以通过底层的 `DTensor` 和多维 `DeviceMesh` 统一编排。

假设总卡数为 $N$，我们需要构建一个 3D/4D 的 DeviceMesh：
$$N = N_{FSDP} \times N_{EP} \times N_{TP}$$

<details open>
<summary><b>1. 各种并行的硬件映射原则（点击展开/折叠）</b></summary>

*   **TP + SP (Tensor & Sequence Parallelism):**
    *   **作用：** 切分 Attention 的 QKV、MLP 矩阵。SP 将非线性层（LayerNorm/Dropout）在 Sequence 维度切分，节省激活显存。
    *   **硬件映射：** **严格限制在单节点内**。因为 TP 需要频繁的 All-Reduce 通信，只有节点内的 NVLink（如 H800 的 900GB/s）能撑得住。通常设置 $TP = 2, 4, 8$。
*   **EP (Expert Parallelism):**
    *   **作用：** 将 MoE 层中的不同专家（Experts）分配到不同的 GPU 上。Token 通过 Router 被分发到对应的专家。
    *   **硬件映射：** 优先跨节点（但也尽量在同一个交换机下）。EP 需要 All-to-All 通信，带宽要求高。专家数量通常决定了 EP 的上限（如 64 个专家，则 $EP_{max} = 64$）。
*   **FSDP2 (Fully Sharded Data Parallelism v2):**
    *   **作用：** 处理 Dense 层（如 Attention、Embedding）的数据并行，将参数、梯度、优化器状态切分到各个 DP 组中。FSDP2 的优势在于无需像 FSDP1 那样做复杂的扁平化（Flatten），天生支持与 TP/EP 组合。
    *   **硬件映射：** 全局跨节点。负责兜底消耗剩余的卡数，扩展全局 Batch Size。
</details>

---

### 从单节点到万卡的 Scaling 演进路线

#### Phase 1: 单节点跑通与正确性验证 (1 Node, 8 GPUs)
在这个阶段，重点是利用 FSDP2 的可组合性，跑通模型的前向、反向和通信逻辑，不追求极致性能。

*   **配置示例：** $TP=2, EP=4, FSDP=1$ (总共 8 卡)
*   **实现步骤：**
    1.  **构建 DeviceMesh：**
        ```python
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cuda", (1, 4, 2), mesh_dim_names=("dp", "ep", "tp"))
        ```
    2.  **应用 TP + SP：** 对 Attention 层应用 `ColwiseParallel` 和 `RowwiseParallel`，并开启 Sequence Parallelism。
    3.  **应用 EP：** 对 MoE 层，利用 `ep` mesh 维度将不同专家的权重分配到 4 个组中。
    4.  **应用 FSDP2：** 对整个模型中**非 EP 的 Dense 部分**包裹 FSDP2，沿着 `dp` mesh（这里大小为1，主要用于验证接口）进行切分。

#### Phase 2: 集群级扩展 (10 ~ 100 Nodes, 80 ~ 800 GPUs)
此时显存不再是主要瓶颈，**通信瓶颈（All-to-All 和 All-Gather）开始显现**。

*   **配置示例：** $TP=8, EP=32, FSDP=4$ (总共 1024 卡)
*   **实现步骤与优化：**
    1.  **TP 占满单节点：** 将 TP 设为 8，完全利用单机 NVLink。
    2.  **跨节点 EP 通信优化：** EP 引入了 All-to-All 通信。此时必须实现**层次化 All-to-All (Hierarchical All-to-All)** 或者利用网络拓扑感知路由。
    3.  **FSDP2 策略调整：** FSDP2 开始在跨节点发挥作用。使用 `FULL_SHARD` 策略切分 Dense 层的参数。
    4.  **计算与通信重叠 (Overlap)：** FSDP2 自动处理 All-Gather 与计算的 overlap，但你需要手动优化 MoE 层 All-to-All 通信与 Dispatch/Combine 计算的 Overlap。

#### Phase 3: 万卡超大规模 Scaling (1,000+ Nodes, 10,000+ GPUs)
在万卡规模下，任何微小的负载不均（Load Imbalance）或网络抖动（Straggler）都会导致集群算力利用率（MFU）断崖式下跌。

*   **配置示例：** $TP=8, EP=64, FSDP=20$ (外加 3D 并行中的 Pipeline Parallelism PP=16 等等，单靠这四种并行通常吞不下百万级别 Context Length 或超千亿参数，但在此仅讨论目标组合)

<details>
<summary><b>万卡级别的关键技术挑战与解决方案（点击展开）</b></summary>

1.  **MoE 负载不均与 Token 丢弃 (Token Dropping)**
    *   **问题：** 某些专家（如标点符号专家）会过载，导致其他 GPU 闲置等待。
    *   **方案：** 引入 Auxiliary Loss（辅助损失）强制平衡；在路由阶段设置 Capacity Factor（如 1.25），超过容量的 Token 直接通过残差连接跳过 MoE，避免跨界点排队。采用无丢弃策略（如 DeepSeek-V2 的跨组 Token 交换）需要在底层写定制化 CUDA Kernel。
2.  **网络拓扑与通信风暴**
    *   **问题：** 10,000 张卡同时进行 All-to-All 会引发网络拥塞（Incast）。
    *   **方案：**
        *   **EP 分组限制：** 不让 EP = 总卡数/TP，而是限制 $EP \le 64$ 或 128。剩余的扩展能力交给 FSDP2（Data Parallelism）。也就是说，集群中有多个包含完整专家的副本。
        *   **路由局部性：** 修改 Router 算法，鼓励 Token 优先选择同一个机架（Rack）或同一网段内的 GPU 上的专家。
3.  **FSDP2 与 EP 的兼容解耦**
    *   在 FSDP2 中，Dense 层（被 DP 切分）和 MoE 层（被 EP 切分）的通信域是不同的。
    *   **方案：** 使用 PyTorch 2.4+ 提供的 `fully_shard` API，精细控制不同模块包裹不同维度的 Mesh。Dense 部分绑 `dp_mesh`，Expert 权重绑定 `ep_mesh` 并关闭 FSDP 的参数切分（因为 EP 已经是按专家切分了，不需要再做二次切片，否则会导致极端的通信切碎）。
4.  **容错与弹性（Fault Tolerance）**
    *   **问题：** 万卡级别每天都会有 GPU 掉线。
    *   **方案：** 使用 FSDP2 自带的 `DistributedStateDict` 和异步 Checkpointing。切忌全量保存为单一文件，必须按 DeviceMesh 维度分别并发保存和加载。
</details>

---

### FSDP2 在此架构中的决定性优势

为什么选择 **FSDP2** 而不是 FSDP1 或 Deepspeed ZeRO3？

1.  **基于 `DTensor` 的可组合性：** FSDP1 是把模块打平（Flatten）成一个 1D 数组，这破坏了原始的张量形状，导致它极难与 TP 和 EP 组合。FSDP2 是 Per-parameter sharding，保留了 Tensor 的原始形状信息。
2.  **多维 Mesh 零冲突：** 你可以定义一个张量在 TP 维度上切分，同时在 FSDP(DP) 维度上切分，PyTorch 底层会自动处理这两个维度的通信（如先 All-Gather 恢复完整权重，再做张量计算），使得代码极其清晰。

### 总结的伪代码架构

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.fsdp import fully_shard

# 1. 万卡集群的 Mesh 定义 (假设全局 DP=128, EP=16, TP=8 = 16384 卡)
mesh = init_device_mesh("cuda", (128, 16, 8), mesh_dim_names=("dp", "ep", "tp"))
dp_mesh = mesh["dp"]
ep_mesh = mesh["ep"]
tp_mesh = mesh["tp"]

# 2. 模型初始化 (Meta device 省内存)
with torch.device("meta"):
    model = MegaMoEModel(config)

# 3. 逐层应用并行策略
for layer in model.layers:
    # 3.1 TP + SP 应用于 Attention 和 Dense
    parallelize_module(layer.attention, tp_mesh, {"wq": ColwiseParallel(), "wo": RowwiseParallel()})
  
    # 3.2 EP 应用于 MoE 层 (伪代码：依赖自定义或分布式库的 EP 实现)
    parallelize_moe_experts(layer.moe, ep_mesh)
  
    # 3.3 FSDP2 处理 Dense 部分，跳过 Experts 以避免重复切分
    # FSDP2 只在 dp_mesh 上起作用
    fully_shard(layer.attention, mesh=dp_mesh)
    fully_shard(layer.moe.router, mesh=dp_mesh) 
    # Experts 被 EP 管理，通常不再包裹 FSDP，或者包裹在 dp_mesh 上做单纯的数据并行

# 全局包裹
fully_shard(model, mesh=dp_mesh)
```

**建议：** 在实际落地万卡时，除了 PyTorch 原生接口，通常还需要引入 Nvidia Megatron-Core 的底层 Kernel（尤其是优化过的 fused kernels 和定制的 All-to-All 通信原语）作为补充，才能将 MFU 推向 50% 以上。

---

### 🧑‍💻 User

torchtitan（基于dtensor+devicemesh）/deepspeed/Megatron-Core有没有最佳实现供参考？

---

### 🤖 Assistant

针对你提到的 **TP + FSDP2 (基于 DTensor/DeviceMesh) + EP + SP** 这一非常前沿且特定的架构，业界这三大开源框架的侧重点和成熟度有很大差异。

直接给出结论：
*   **最符合你架构描述（DTensor+FSDP2）的最佳参考：`torchtitan`**
*   **真正经过万卡级 MoE 工业生产验证的黄金标准：`Megatron-Core`**
*   **易用性较好但组件复合能力稍弱的历史参考：`DeepSpeed`**

以下是对这三个框架的深度解析与参考指南。

---

### 1. TorchTitan (PyTorch 官方 Native 架构)
**匹配度：★★★★★ (完全吻合你的技术栈) | 生产成熟度：★★★☆☆**

TorchTitan 是 Meta 和 PyTorch 团队为展示 PyTorch 2.x Native 并行能力（尤其是 3D 并行 + FSDP2）而打造的官方 blueprint。它完全基于 `DeviceMesh` 和 `DTensor` 构建。

<details open>
<summary><b>TorchTitan 最佳实践与代码参考</b></summary>

*   **核心优势：** 代码极其干净，没有繁杂的第三方 C++ 扩展，完全通过 `fully_shard` (FSDP2) 和 `parallelize_module` (TP) 实现组合。
*   **如何实现你提到的架构：**
    *   **DeviceMesh 初始化：** 位于 `torchtitan/parallelisms/parallelize_llama.py`。它展示了如何构建 N 维网格。
    *   **FSDP2 + TP 组合：** TorchTitan 完美展示了如何将 TP (Tensor Parallel) 包裹在 Attention 内部，然后在外部使用 `fully_shard` 进行数据并行切分。
*   **关于 EP (MoE)：**
    *   虽然 TorchTitan 早期主要专注 Dense 模型（如 Llama 3），但 PyTorch 社区正在将基于 DTensor 的 MoE 快速合入主分支。
    *   **参考代码路径：** 建议参考 PyTorch core 中的 `torch.distributed.tensor.parallel` 下对 EP 的初步支持，或者关注 TorchTitan 最新的 MoE PR。它的实现逻辑正如上一问中展示的，使用特定的 mesh 维度（`ep_mesh`）对专家的权重进行 Sharding。
</details>

### 2. Megatron-Core (NVIDIA 工业界绝对霸主)
**匹配度：★★★☆☆ (技术栈不同，但理念一致) | 生产成熟度：★★★★★**

如果你真的有一万张卡要跑 MoE，**Megatron-Core (mcore) 是目前唯一的最优解**。像 DeepSeek-V2/V3、Grok-1、Qwen-MoE 等超大规模模型，其底层通信和计算 Kernel 基本都大量借鉴或直接基于 mcore。

<details>
<summary><b>Megatron-Core 的架构映射与代码参考</b></summary>

*   **架构差异说明：**
    *   mcore **不使用 FSDP2**。它的数据并行采用的是 **DP + Distributed Optimizer (等价于 ZeRO-1)**。在极大规模下，ZeRO-1（只切分优化器状态和梯度）配合 PP（流水线并行）比纯 FSDP 更节省通信开销，也更容易与 TP/EP 组合。
*   **MoE 核心实现路径 (`megatron/core/transformer/moe/`)：**
    *   `moe_layer.py`: MoE 层的入口，处理 Router 和不同并行的组合。
    *   `router.py`: 实现了 Top-K 路由算法，处理 Token 丢弃（Token Dropping）和负载均衡。
    *   `token_dispatcher.py` **(极其关键)**: 实现了 EP 中的 All-to-All 通信。它包含了 `AllGatherAllToAllTokenDispatcher` 等高级调度器，优化了通信和计算的 Overlap。
*   **最佳实践参考：**
    *   NVIDIA 的 `Megatron-LM` 仓库中，关于 Mixtral 或定制 MoE 的训练脚本，是学习如何分配 TP/EP/PP/DP 组的最好教程。
    *   它利用 Transformer Engine (TE) 实现了极高效率的 Grouped GEMM，解决了不同专家接收到不同数量 Token 时的计算低效问题。
</details>

### 3. DeepSpeed (微软的大众化选择)
**匹配度：★★★★☆ (使用 ZeRO 代替 FSDP) | 生产成熟度：★★★★☆**

DeepSpeed 是最早普及 MoE 训练的框架之一（DeepSpeed-MoE 论文）。它的核心是 ZeRO 系列，ZeRO-3 在宏观概念上与 FSDP 非常相似。

<details>
<summary><b>DeepSpeed 的架构映射与代码参考</b></summary>

*   **组合痛点：** DeepSpeed 的 ZeRO-3（参数全切分）与 TP 结合一直是个难点。通常在 DeepSpeed 中，大家倾向于用 **ZeRO-DP + EP**，而较少引入复杂的 TP（或者将 TP 交给 Megatron 管理，即 Megatron-DeepSpeed 架构）。
*   **代码参考路径：**
    *   `deepspeed/moe/layer.py`: DeepSpeed 的 MoE 层实现。
    *   `deepspeed/moe/sharded_moe.py`: 展示了如何在专家组内应用并行。
*   **最佳实践：**
    *   DeepSpeed 对 EP 的分组管理非常成熟（`deepspeed.utils.groups.initialize()`）。它允许你设置 `ep_size`，并自动构建与数据并行（DP）正交的通信组。
    *   如果你不想折腾 PyTorch 底层的 `DeviceMesh`，DeepSpeed 提供的高层 API `deepspeed.moe.layer.MoE` 是最快能跑通 EP + 数据并行的方案。
</details>

---

### 总结与选型建议

1.  **如果你在做前沿系统研究，或者希望完全基于 PyTorch 原生生态 (无缝对接 PyTorch 2.4+ 编译优化)：**
    *   死磕 **`torchtitan`**。阅读它如何用 `init_device_mesh` 构建 3D 拓扑，以及如何用 `fully_shard` API 包裹模型。这是未来的方向，代码也最易读。
2.  **如果你马上要在 1000+ GPUs 上炼丹，追求极致的 MFU (算力利用率)：**
    *   果断选择 **`Megatron-Core`**。深入研究它的 `token_dispatcher.py`，因为在万卡 MoE 训练中，决定生死的不是框架 API 好不好看，而是 All-to-All 通信能不能和 Grouped GEMM 完美掩盖 (Overlap)。
3.  **如果你在中小规模集群 (百卡级)，希望快速复现开源 MoE 模型：**
    *   **`DeepSpeed`** 结合 HuggingFace `transformers` 的 Trainer 是最快落地的路径。