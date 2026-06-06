# PyTorch24中TP与DP组合并行机制

### 🧑‍💻 User

pytorch2.4+中引入了基于 `DeviceMesh` 和 `DTensor` 的分布式架构实现了真正的可组合并行（Composable Parallelism），过引入底层的张量语义，我们可以非常优雅地将 TP（张量并行）、FSDP/HSDP（数据并行）以及 PP/SP 等策略叠加使用。
试问它是如何将 TP (Tensor Parallel) 包裹在 Attention 内部，然后在外部使用 fully_shard 进行数据并行切分的？对于无法apply tp的层比如moe experts又是如何解决的？tp与dp通信怎么保证正交互不干扰？

---

### 🤖 Assistant

在 PyTorch 2.4+ 中，基于 `DeviceMesh` 和 `DTensor` 的可组合并行（Composable Parallelism）标志着分布式训练 API 的一次范式转变。它从以前“基于 Wrapper 互相嵌套”的复杂逻辑（如 FSDP1 嵌套 TP），转向了**基于张量底层语义（Tensor Semantics）的自然正交组合**。

下面将针对你的三个问题，详细解析其底层运行机制。

---

### 一、 如何将 TP 包裹在 Attention 内部，外部再使用 `fully_shard` (FSDP2)？

在 PyTorch 2.4+ 中（特别是 FSDP2），FSDP 和 TP 的组合不再是通过复杂的代码嵌套实现的，而是通过 **`DTensor` 的多维 Placement（放置策略）** 来实现的。

#### 1. DeviceMesh 定义拓扑
首先，我们定义一个二维的设备网格（Device Mesh），包含数据并行（DP）和张量并行（TP）两个维度：

```python
from torch.distributed.device_mesh import init_device_mesh

# 假设有 8 张卡，构建 4x2 的二维网格
# dim 0 为 "dp" (4 cards), dim 1 为 "tp" (2 cards)
mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
dp_mesh = mesh["dp"]
tp_mesh = mesh["tp"]
```

#### 2. 内部：在 Attention 中 Apply TP
TP 的本质是对权重矩阵的切分。PyTorch 提供了 `parallelize_module`，利用 `ColwiseParallel` 和 `RowwiseParallel` 直接将局部的 `torch.Tensor` 参数转换为 `DTensor`。

对于 Attention 层：
*   **Q, K, V 投影（Column-wise）**：权重在列方向切分。输入 $X$ 复制（Replicate），输出 $Y$ 在特征维度切分。
*   **O 投影（Row-wise）**：权重在行方向切分。输入（上一步的输出）已切分，计算后进行 All-Reduce。

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

tp_plan = {
    "q_proj": ColwiseParallel(),
    "k_proj": ColwiseParallel(),
    "v_proj": ColwiseParallel(),
    "o_proj": RowwiseParallel(),
}
# 此时，Attention 的参数变成了基于 tp_mesh 的一维 DTensor
parallelize_module(attention_layer, tp_mesh, tp_plan)
```

#### 3. 外部：使用 `fully_shard` (FSDP2) 进行数据并行切分
FSDP2 (`fully_shard`) 的核心改进是它原生理解 `DTensor`。当你在外部对整个 Transformer Block 或 Model 调用 `fully_shard` 时，FSDP2 会检查参数：

1.  如果参数已经是基于 `tp_mesh` 的 `DTensor`（例如 Attention 的 QKV 权重），FSDP2 会在其基础上增加一个 `dp_mesh` 的维度，将其转变为**二维 `DTensor`**。
2.  其 Placement 策略会变为：`[Shard(0), Shard(1)]` （假设在 DP 维切分，在 TP 维也是切分）。

```python
from torch.distributed.fsdp import fully_shard

# 将外层 module 按照 dp_mesh 进行 FSDP 切分
fully_shard(transformer_block, mesh=dp_mesh)
```

**运行逻辑：**
*   **前向传播开始前**：FSDP2 触发 All-Gather，但**仅在 `dp_mesh` 通信组内**进行。Gather 完毕后，参数在 DP 维度是完整的（Replicated），但在 TP 维度依然是切分的（Sharded）。
*   **进入 Attention**：执行正常的 TP 计算（包含 TP 组内的 All-Reduce）。
*   **反向传播结束**：FSDP2 触发 Reduce-Scatter，再次将梯度在 `dp_mesh` 内切分释放显存。

---

### 二、 对于无法 Apply TP 的层（如 MoE Experts）如何处理？

对于 MoE 架构，通常采用 **Expert Parallelism (EP)** 而非传统的 TP，因为对小型 Expert MLP 进行张量切分会带来过高的通信开销。`DeviceMesh` 提供了极大的灵活性来处理这些异构层。

#### 1. 局部跳过 TP
如果你定义了 TP plan，你可以简单地**不把** MoE 层纳入 `parallelize_module` 的计划中。当外部的 `fully_shard` 作用于整个模型时，MoE 层的参数会被视为普通的 `torch.Tensor`，FSDP 会直接将其在 `dp_mesh` 上切分为一维 `DTensor`。

#### 2. 将 TP Mesh 复用为 EP Mesh
更优雅的做法是，将 TP 维度的设备组用来做 Expert 切分（EP）。

假设一个 MoE 层有 8 个 Experts，`tp_mesh` 有 2 张卡。我们可以将前 4 个 Experts 放在卡 0，后 4 个放在卡 1。
由于所有操作都是基于 `DTensor` 的，我们可以手动将 Experts 的参数列表构造成在 `tp_mesh` 上按照专家维度（例如第 0 维）切分的 `DTensor`：

$$ Expert\_Weights_{ep} = [Shard(0)] \text{ on tp\_mesh} $$

<details>
<summary><strong>点击展开：MoE 层组合 EP 和 FSDP 的代码实现逻辑</strong></summary>

```python
# 假设 expert 参数形状为 [num_experts, in_features, out_features]
# 在初始化时，将其分布在 tp_mesh (这里复用为 ep_mesh) 上
expert_weight = distribute_tensor(
    raw_expert_weight, 
    device_mesh=tp_mesh, 
    placements=[Shard(0)] # 在专家维度切分
)

# 然后在外层套用 fully_shard
# FSDP 会将这个一维的 DTensor 变成基于 2D mesh (dp, tp) 的二维 DTensor
# 最终 Placements: [Shard(0) on DP, Shard(0) on TP/EP]
fully_shard(moe_layer, mesh=dp_mesh)
```
通过这种方式，Attention 走 TP 逻辑，MoE 走 EP 逻辑，外层统一走 FSDP 逻辑，**完全没有底层冲突，因为它们最终都收敛到了多维 `DTensor` 的表达上。**

</details>

---

### 三、 TP 与 DP 通信怎么保证正交互不干扰？

这是 `DeviceMesh` 架构最核心的优势：**通过拓扑解耦 Process Groups (PG)**。

在传统的 PyTorch 分布式中，你需要手动维护各种 `ProcessGroup`。而在 PyTorch 2.4+ 中，当你调用 `init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))` 时，底层发生了以下事情：

1.  **正交通信组的创建**：
    `DeviceMesh` 会在底层自动实例化正交的 NCCL 组。
    *   **DP Groups**：4 个大小为 4 的进程组（跨节点，负责同一 TP rank 的卡之间的数据同步）。
    *   **TP Groups**：4 个大小为 2 的进程组（节点内，负责同一 DP rank 的卡之间的张量并行通信）。

2.  **`DTensor` 操作的分发机制 (Dispatch)**：
    当一个计算图执行到矩阵乘法（如 Attention 内部）或者触发 FSDP hook 时，PyTorch 底层的 `DTensor` 调度器会检查参与运算的 `DTensor` 的 `placements` 和对应的 `DeviceMesh` 维度。
  
    *   **当 FSDP 需要 All-Gather 时**：它明确指定在 `mesh["dp"]` 上进行操作。底层自动提取对应的 DP Process Group 调用 `ncclAllGather`。这绝对不会波及 TP 维度的卡。
    *   **当 TP 的 RowwiseParallel 需要 All-Reduce 时**：它明确在 `mesh["tp"]` 上进行操作。底层提取 TP Process Group 调用 `ncclAllReduce`。

| 并行策略 | 操作维度 (DeviceMesh Dim) | 调用的底层 Process Group | 典型 NCCL 通信原语 | 通信范围 |
| :--- | :--- | :--- | :--- | :--- |
| **FSDP (DP)** | `mesh["dp"]` | DP 专属 Group | All-Gather / Reduce-Scatter | 跨机器/跨节点 (通常) |
| **TP** | `mesh["tp"]` | TP 专属 Group | All-Reduce (前向/反向) | 机器内/节点内 (NVLink) |

**总结：**
PyTorch 2.4+ 的可组合并行之所以能完美运转，是因为它废弃了“外挂式”的分布式组件，转而将并行的状态机（Sharded, Replicated, Partial）内置到了张量本身（`DTensor`）。由于 `DTensor` 的每个维度天然与一个正交的 NCCL 通信组绑定，因此 TP 的通信和 DP 的通信在物理层和逻辑层都被严格隔离，实现了互不干扰的正交组合。