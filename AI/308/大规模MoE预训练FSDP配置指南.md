# 大规模MoE预训练FSDP配置指南

### 🧑‍💻 User

针对大规模moe预训练如何配置fsdp的auto_wrap_policy、activation_checkpointing_policy、sharding_strategy、forward_prefetch、backward_prefetch、limit_all_gathers、sync_module_states？

---

### 🤖 Assistant

针对大规模 Mixture of Experts (MoE) 模型的预训练，合理配置 PyTorch 的 Fully Sharded Data Parallel (FSDP) 是平衡显存占用与计算/通信吞吐量的关键。

MoE 模型的特殊性在于：**参数量极大，但单次前向传播激活的参数量较小（稀疏计算）**。如果在 MoE 中使用纯粹的 FSDP，每次前向/反向传播都需要 `AllGather` 所有专家的权重，这会带来灾难性的通信开销。因此，**业界标准做法是将 FSDP 与专家并行（Expert Parallelism, EP）结合使用**。

以下是针对大规模 MoE 预训练的 FSDP 核心参数详细配置建议：

### 1. `sharding_strategy` (分片策略)
*   **推荐配置：** `ShardingStrategy.HYBRID_SHARD` (配合多机训练)
*   **原因分析：** 
    *   在大规模多机集群中，节点内（NVLink/NVSwitch）带宽极高，节点间（InfiniBand/RoCE）带宽较低。
    *   `HYBRID_SHARD`（HSDP）会在**节点内**执行 `FULL_SHARD`（Zero-3），在**节点间**执行标准的数据并行（Data Parallel）。这极大地减少了跨节点的极耗时的 `AllGather` 和 `ReduceScatter` 通信操作。
    *   *注意*：如果你的模型实在太大，单节点的总显存（例如 $8 \times 80\text{GB}$）放不下一个模型的副本，才退而求其次使用 `FULL_SHARD`。

### 2. `auto_wrap_policy` (自动包装策略)
*   **推荐配置：** 基于 `transformer_auto_wrap_policy` 自定义，按 **Transformer Block** 级别进行包装。
*   **MoE 核心考量：**
    *   你需要将模型按照 Transformer 层（如 `MoETransformerBlock`）进行包装。这确保了在计算当前层时，只 gather 当前层的参数，计算完立即释放。
    *   **⚠️ MoE 专属处理**：如果你同时使用了专家并行（EP），**绝对不能**用 FSDP 去 wrap 专家层（Expert Modules）。因为 EP 已经将专家分发到了不同的 GPU 上，用 FSDP 去分片会导致逻辑冲突和极大的通信冗余。你应该通过 `ignored_modules` 传入 FSDP，或者针对专家层使用 `NO_SHARD`。

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={MixtralDecoderLayer}, # 替换为你的 MoE Block 类
)
```

### 3. `activation_checkpointing_policy` (激活重计算策略)
*   **推荐配置：** 与 `auto_wrap_policy` 保持一致，在 **Transformer Block** 级别进行 Checkpoint。
*   **原因分析：** MoE 模型的显存瓶颈通常在于庞大的参数量和长上下文带来的激活值。开启 Activation Checkpointing 是必须的。通过在 Block 级别重计算，可以用约 30% 的额外计算时间换取数倍的显存节省。

### 4. `backward_prefetch` (反向预取)
*   **推荐配置：** `BackwardPrefetch.BACKWARD_PRE`
*   **原因分析：** 开启后，在计算当前层 $L$ 的梯度的同时，FSDP 会提前发起对上一层 $L-1$ 参数的 `AllGather` 通信。对于 MoE 这种极度依赖通信掩盖计算的模型，`BACKWARD_PRE` 能提供最好的通信/计算重叠（Overlap）效果，显著提升算力利用率（MFU）。

### 5. `forward_prefetch` (前向预取)
*   **推荐配置：** `True`
*   **原因分析：** 与反向预取类似，它在执行第 $i$ 层的前向传播时，提前发起第 $i+1$ 层的参数 `AllGather`。开启此项对于掩盖 MoE 庞大的参数拉取延迟至关重要。

### 6. `limit_all_gathers` (限制并发 AllGather)
*   **推荐配置：** `True`
*   **原因分析：** FSDP 默认可能会同时发起多个 `AllGather` 请求。在 MoE 模型中，由于还存在由 Expert Parallelism 引入的 `All-to-All` 通信，网络拓扑已经非常拥挤。如果不限制 AllGather，会导致 CUDA 流队列堆积，引发不可预知的显存尖峰（OOM）或 CPU/GPU 同步阻塞。

### 7. `sync_module_states` (同步模块状态)
*   **推荐配置：** `True`（配合 CPU Offload 或单节点初始化时） / `False`（配合分布式 Checkpoint 加载时）
*   **原因分析：**
    *   如果为了节省显存，你在 CPU 上初始化了模型（或者只在 Rank 0 上初始化），必须设置为 `True`，让 FSDP 在开始前把 Rank 0 的参数广播（Broadcast）给所有进程。
    *   **在大规模预训练中**，更高效的做法是每个 Rank 直接从磁盘加载属于自己的 distributed checkpoint (分片模型)，此时应设为 `False` 以避免启动时耗费数十分钟进行无用的广播。

---

<details>
<summary>💡 进阶：MoE + FSDP 伪代码与混合并行架构</summary>

在真正的工业级大规模 MoE 预训练中，FSDP 只是非专家层的基石。完整的配置流通常如下：

```python
import torch
import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 1. 策略配置
fsdp_config = {
    "sharding_strategy": ShardingStrategy.HYBRID_SHARD, # 多机推荐
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "forward_prefetch": True,
    "limit_all_gathers": True,
    "sync_module_states": False, # 假设使用分布式加载
    "use_orig_params": True,     # 编译优化 (torch.compile) 所需
}

# 2. Wrap 策略：只 Wrap 整个 Block
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={YourMoETransformerBlock},
)

# 3. 混合精度配置 (推荐 BF16)
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32, # 梯度累加用 fp32 保证精度
    buffer_dtype=torch.bfloat16,
)

# 4. 忽略专家层的 FSDP (让专家并行 EP 去接管它们)
# 注意：你需要把所有的 Expert 层收集起来传入 ignored_modules
expert_modules = [layer.mlp.experts for layer in model.layers]

# 5. 初始化 FSDP
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    ignored_modules=expert_modules, # 非常重要！
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
    **fsdp_config
)

# 6. 激活重计算 (Activation Checkpointing)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

check_fn = lambda submodule: isinstance(submodule, YourMoETransformerBlock)
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=check_fn,
)
```

**关于通信的补充说明**：
在 MoE 中，非专家层（Attention等）走 FSDP 的 `AllGather` / `ReduceScatter`，专家层走 `All-to-All`。两者的通信可以利用 PyTorch 的自定义通信流（CUDA Streams）进一步做 Overlap，但这通常需要修改底层的 MoE 路由逻辑（如参考 Megatron-LM 的实现）。
</details>

---

### 🧑‍💻 User

如何使用devicemesh进一步重叠fsdp+ep？

---

### 🤖 Assistant

在大规模 MoE 模型预训练中，单纯将 FSDP 和 Expert Parallelism (EP) 放在全局通信组（Global Process Group）中会导致极大的通信冲突：FSDP 的 `AllGather` / `ReduceScatter` 会与 EP 的 `All-to-All` 抢占网络带宽和 CUDA 流，导致计算严重闲置。

使用 PyTorch 2.x 引入的 `DeviceMesh`，可以将显卡集群划分为 **正交的通信拓扑**。通过将 FSDP 和 EP 隔离在不同的维度上，并配合自定义 CUDA 流，可以实现底层的通信与计算重叠（Overlap）。

以下是使用 `DeviceMesh` 进一步重叠 FSDP 与 EP 的核心步骤和机制：

### 1. 核心思想：正交的二维/三维网格通信

假设你有 $N$ 张显卡，你想进行 $DP$ 度的 FSDP 数据并行和 $EP$ 度的专家并行，满足 $N = DP \times EP$。
利用 `DeviceMesh`，我们可以创建一个形状为 `(DP, EP)` 的二维网格：
*   **FSDP 维度 (列通信)**：在同一个 EP 组内，各个节点共享同一个专家的副本，进行 FSDP 的参数拉取和梯度规约。
*   **EP 维度 (行通信)**：在同一个 FSDP 组内，各个节点持有不同的专家，进行 Token 路由的 `All-to-All` 通信。

因为这两个操作发生在正交的网络维度（不同的 Process Group）上，NCCL 可以无冲突地并行执行它们。

### 2. 构建 DeviceMesh 并分离 Process Group

首先，通过 `init_device_mesh` 初始化集群拓扑：

```python
from torch.distributed.device_mesh import init_device_mesh

# 假设总共有 64 张卡，DP=8, EP=8
dp_size = 8
ep_size = 8

# 初始化 2D Mesh，命名维度为 "fsdp" 和 "ep"
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(dp_size, ep_size),
    mesh_dim_names=("fsdp", "ep")
)

# 获取针对这两个维度的专属 Process Group
fsdp_pg = mesh.get_group("fsdp")
ep_pg = mesh.get_group("ep")
```

### 3. FSDP 绑定至专属 Mesh 维度

在初始化 FSDP 时，**不再使用默认的全局 PG**，而是显式传入 FSDP 专属的 mesh 维度。这样，FSDP 所有的后台通信（包括 `forward_prefetch` 和 `backward_prefetch`）都会严格限制在 `fsdp_pg` 对应的网络链路上。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 初始化 FSDP 时传入 device_mesh
model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    ignored_modules=expert_modules, 
    device_mesh=mesh["fsdp"], # <--- 关键：限制 FSDP 的通信范围
    **fsdp_config
)
```

### 4. EP All-to-All 绑定至专属 Mesh 维度

在专家并行的实现中（通常在你自定义的 MoE 路由代码里），执行 `torch.distributed.all_to_all_single` 时，显式传入 `ep_pg`。

```python
import torch.distributed as dist

def route_tokens_to_experts(hidden_states, dispatch_mask):
    # ... 计算 local token ...
  
    # 仅在 EP 维度上进行 All-to-All 通信
    dist.all_to_all_single(
        output_tensors,
        input_tensors,
        group=ep_pg # <--- 关键：使用 EP 专属通信组
    )
    return output_tensors
```

### 5. 实现 FSDP 与 EP 的真正 Overlap (双 CUDA 流)

隔离了 Process Group 后，物理上的网络冲突解决了，但为了让通信和计算真正重叠，我们需要在 PyTorch 中使用多 CUDA 流机制。

**重叠逻辑（以前向传播为例）：**
在第 $i$ 层 Transformer Block 计算时：
1.  **流 A (FSDP 流)**：FSDP 的 `forward_prefetch` 正在异步 `AllGather` 第 $i+1$ 层的 Attention 参数。
2.  **流 B (EP 通信流)**：当前第 $i$ 层的 Token 正在进行 `All-to-All` 发送到对应的专家显卡。
3.  **流 C (计算流)**：GPU 正在计算第 $i$ 层的非路由部分，或上游的计算逻辑。

---

<details>
<summary>💡 进阶：如何用 CUDA Stream 封装 MoE 的异步 All-to-All</summary>

在实际开发中，要让 EP 的 `All-to-All` 和 FSDP 的参数拉取/本地计算完全重叠，需要手动管理 CUDA 异步流。以下是高阶伪代码示例：

```python
import torch
import torch.distributed as dist

class AsyncMoELayer(torch.nn.Module):
    def __init__(self, ep_pg):
        super().__init__()
        self.ep_pg = ep_pg
        # 创建一个专门用于 All-to-All 的非阻塞流
        self.comm_stream = torch.cuda.Stream()
      
    def forward(self, hidden_states):
        # 1. 路由计算 (在默认流中执行)
        router_logits = self.router(hidden_states)
        local_tokens, expert_indices = self.get_dispatch_logic(router_logits)
      
        # 2. 启动异步 All-to-All
        # 使用 wait_stream 确保通信流等待本地 token 计算完毕
        self.comm_stream.wait_stream(torch.cuda.current_stream())
      
        with torch.cuda.stream(self.comm_stream):
            # 这里的 all_to_all 不会阻塞主线程的后续计算
            dist.all_to_all_single(
                recv_buffer, 
                local_tokens, 
                group=self.ep_pg
            )
          
        # 3. 此时，主流（计算流）可以去干别的事！
        # 例如：如果是流水线或者交错计算，这里可以计算无需 All-to-All 的部分
        # 或者是让 FSDP 在后台利用 fsdp_pg 偷偷 prefetch 下一层的参数
        do_some_local_computation()
      
        # 4. 同步流，等待 All-to-All 数据到达，然后进行专家计算
        torch.cuda.current_stream().wait_stream(self.comm_stream)
      
        expert_outputs = self.experts(recv_buffer)
      
        # ... 后续的反向路由 (All-to-All 返回结果) 逻辑类似 ...
        return final_output
```

**为什么这能起作用？**
由于你在 `DeviceMesh` 中分离了 FSDP 和 EP，NCCL 为它们分配了不同的通信环/树（Communicators）。因此，即便你的自定义 `self.comm_stream` 正在狂跑 `All-to-All`，FSDP 底层管控的另一个 CUDA 流仍然可以利用另外的网络带宽并行执行 `AllGather`。如果没有 `DeviceMesh`，这两个操作会被强制排队，导致整个集群的卡顿。
</details>