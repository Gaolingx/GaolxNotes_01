# PyTorch24中TP与DP组合并行机制

### 🧑‍💻 User

pytorch2.4+中引入了hsdp策略可以方便的划分devicemesh组合tp/dp/pp/sp等，它是如何将 TP (Tensor Parallel) 包裹在 Attention 内部，然后在外部使用 fully_shard 进行数据并行切分吗，对于无法apply tp的层比如moe experts又是如何解决的？tp与dp通信怎么保证正交互不干扰？

---

### 🤖 Assistant

在 PyTorch 2.4+ 中，基于 `DeviceMesh` 和 `DTensor` 的分布式架构实现了真正的**可组合并行（Composable Parallelism）**。通过引入底层的张量语义，我们可以非常优雅地将 TP（张量并行）、FSDP/HSDP（数据并行）以及 PP/SP 等策略叠加使用。

针对你的三个问题，下面详细解析其工作原理。

### 一、 如何在 Attention 内部包裹 TP，外部使用 `fully_shard`

在 PyTorch 2.4 中，推荐的范式是 **“由内向外” (Inside-Out)** 施加并行策略。即先对底层的 `nn.Module` 应用模型并行（如 TP），然后再在更外层的模块上应用数据并行（如 FSDP2 的 `fully_shard`）。

工作流如下：

1. **构建 DeviceMesh**：首先创建一个二维的设备网格，例如维度大小为 $D \times T$，其中 $D$ 代表 DP 维度，$T$ 代表 TP 维度。
2. **应用张量并行 (TP)**：使用 `parallelize_module` 针对 `DeviceMesh` 的 `"tp"` 维度，将 Attention 层内部的 Linear 权重切分为列并行 (`ColwiseParallel`) 和行并行 (`RowwiseParallel`)。此时，权重变成了分布在 TP 网格上的 `DTensor`。
3. **应用全分片数据并行 (FSDP/HSDP)**：在 Transformer Block 或者整个 Model 级别，调用 `fully_shard(module, mesh=mesh["dp"])`。

**核心原理**：`fully_shard` 原生支持 `DTensor`。当 `fully_shard` 试图在 `"dp"` 维度上切分参数时，它会发现这些参数已经是基于 `"tp"` 维度的 `DTensor`。由于两个网格维度是正交的，`fully_shard` 会直接对局部的 TP 分片（Local Shard）在 DP 维度上进行进一步的展平（Flatten）和切分。

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._composable.fsdp import fully_shard

# 1. 初始化 2D DeviceMesh (例如 8 张卡: 2 DP x 4 TP)
mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))

# 2. 定义 TP 切分计划
tp_plan = {
    "wq": ColwiseParallel(),
    "wk": ColwiseParallel(),
    "wv": ColwiseParallel(),
    "wo": RowwiseParallel(),
}

for block in model.layers:
    # 3. 先在内部应用 TP
    parallelize_module(block.attention, mesh["tp"], tp_plan)
    parallelize_module(block.feed_forward, mesh["tp"], mlp_plan)
  
    # 4. 在外部包裹 FSDP (使用 fully_shard)
    fully_shard(block, mesh=mesh["dp"])

# 整个模型也可以套一层 fully_shard
fully_shard(model, mesh=mesh["dp"])
```

---

### 二、 对于无法应用 TP 的层（如 MoE Experts）如何解决？

混合专家模型 (MoE) 的 Experts 通常使用 **专家并行 (Expert Parallelism, EP)** 而不是纯粹的 TP。在可组合并行的架构下，处理 MoE 的策略非常灵活，主要依赖于**选择性策略应用**。

因为我们是基于具体的子模块（sub-module）来应用策略的，所以可以轻易地对 Dense 层和 MoE 层采取不同的处理方式：

1. **跳过 TP**：在为 Transformer Block 编写 TP Plan 时，直接略过 MoE Experts 部分。此时 Experts 参数在 TP 组内是完整复制的（Replicated）。
2. **应用 EP/FSDP**：我们可以把 Experts 当作独立的一层，利用 `DeviceMesh` 的机制，将其分布在特定的网格维度上（通常复用 DP 维度，或划定一个专属的 EP 维度）。

<details>
<summary><b>点击展开：MoE Experts 的具体并行策略细节</b></summary>

*   **FSDP as EP (利用 FSDP 模拟专家并行)**：
    如果我们对 Experts 的 `nn.ModuleList` 在 `"dp"` 维度上应用 `fully_shard`，并且由于我们在 TP 维度跳过了它，每个计算节点将拥有不同专家的分片。在 AllGather 时，各个节点获取完整的专家用于计算。
*   **原生 EP 支持**：
    在最新的架构中，可以将不同的 Expert 放置在不同的 Rank 上（直接基于 `DTensor` 的分片放置）。当 Token 通过 Router 被分配到不同 Expert 时，使用类似于 `AllToAll` 的通信算子，在 `"ep"`（或 `"dp"`）所在的通信组内交换 Token，计算完成后再通过 `AllToAll` 换回结果。这完全不需要 TP 的介入。
*   **代码示例**：
    ```python
    # 针对 Dense Attention 使用 TP + FSDP
    parallelize_module(block.attention, mesh["tp"], tp_plan)
    fully_shard(block.attention, mesh["dp"])
  
    # 针对 MoE 层，不使用 TP，而是直接按照 DP/EP 维度使用 FSDP 
    for expert in block.moe.experts:
        fully_shard(expert, mesh=mesh["dp"]) # 相当于让每个 DP rank 负责不同的 expert 块
    ```
</details>

---

### 三、 TP 与 DP 的通信如何保证正交互不干扰？

当模型同时运行 TP（需要 AllReduce/AllGather 等通信）和 FSDP（需要 AllGather / ReduceScatter 通信）时，PyTorch 通过底层的 `ProcessGroup` (进程组) 隔离和 CUDA Stream 管理来保证它们绝对正交且不干扰。

#### 1. 进程组 (Process Group) 的正交性
当你调用 `init_device_mesh("cuda", (D, T), mesh_dim_names=("dp", "tp"))` 时，PyTorch 并没有创建一个杂乱无章的通信池，而是严格按照矩阵拓扑创建了独立的子通信组（Sub-ProcessGroups）：
*   针对 `"tp"` 维度，创建了 $D$ 个独立的通信组，每个组包含 $T$ 个设备。
*   针对 `"dp"` 维度，创建了 $T$ 个独立的通信组，每个组包含 $D$ 个设备。

当 Attention 内部发生 TP 通信时，底层会提取出与当前操作对应的 `DTensor` 的 `device_mesh["tp"]`，并在这个特定的通信组上调用 NCCL 算子。同理，`fully_shard` 只会调用 `device_mesh["dp"]` 的通信组。**因为 NCCL communicator 是完全分离的，所以空间上的通信是绝对正交的。**

<details>
<summary><b>点击展开：CUDA 流 (Stream) 的调度与并发避免死锁</b></summary>

仅仅区分通信组还不够，如果通信和计算都在默认流（Default Stream）上串行，会引发严重的性能下降甚至死锁。
*   **FSDP 的流管理**：`fully_shard` 内部接管了参数的预取（Prefetch）。它会在专门的 **Unshard Stream** 上发起 `device_mesh["dp"]` 的 AllGather。
*   **TP 的同步机制**：TP 的通信（如列并行后的前向 AllGather，或者行并行后的 AllReduce）通常是张量计算图中的算子（如 `_c10d.all_gather_tensor`）。这些算子默认在当前的 Compute Stream 上执行。
*   **正交执行**：由于 FSDP 提前在后台流发起 DP 维度的权重收集，而当前计算流在执行前向传播并触发 TP 维度的激活动步通信。PyTorch 的分发器（Dispatcher）利用 CUDA Event 进行同步：确保 FSDP 的权重 AllGather 完成后，计算流才开始使用这些权重进行矩阵乘法（并伴随 TP 通信）。
</details>

#### HSDP 视角的补充
HSDP (Hybrid Sharded Data Parallel) 本质上是 FSDP 的变体。它将 DP 进一步拆分为：**节点内分片 (Sharded)** 和 **节点间复制 (Replicated)**。
在组合 TP 时，它的 `DeviceMesh` 通常是一个 3D 结构：`["replicate", "sharded", "tp"]`。
无论维度扩展到多少，由于张量的每一次切分 (Sharding Spec) 和计算算子的分发 (Placements) 都绑定在 `DTensor` 以及特定的 Mesh 维度上，底层的数学逻辑和通信域组建机制依然保证了完全的正交性。