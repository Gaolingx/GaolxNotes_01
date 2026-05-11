# MoE与HSDP通信隔离方案

### 🧑‍💻 User

在moe training with hsdp中，如何保证 all_to_all 只在 EP 组内发生，且不能干扰 DP 的梯度 AllReduce？

---

### 🤖 Assistant

在 MoE (Mixture of Experts) 与 HSDP (Hybrid Sharded Data Parallel, 通常指结合了 ZeRO-2/3 与 数据并行的混合切片策略，如 PyTorch FSDP 的 `HYBRID_SHARD` 模式) 结合的分布式训练中，通信的解耦和并发控制是核心挑战。

如果 `all_to_all` (用于 MoE Token 路由) 和 `AllReduce` / `ReduceScatter` (用于 HSDP 梯度/参数同步) 发生冲突，轻则导致通信效率低下，重则导致 NCCL 死锁 (Deadlock)。

要保证 `all_to_all` 只在 EP (Expert Parallel) 组内发生，且不干扰 DP/HSDP 的梯度通信，核心在于**通信组正交化**、**网络层 Wrapping 隔离**以及**CUDA Stream 分离**。

以下是具体的实现方案和架构设计：

### 1. 建立正交的进程组 (Orthogonal Process Groups)

最基础的保证是物理和逻辑上的通信组隔离。系统总 GPU 数量 $N$ 可以表示为：$N = DP \times EP \times TP$ (这里假设无 TP，则为 $N = DP_{group} \times EP_{group}$)。

必须使用 `torch.distributed.new_group` 为 EP 和 HSDP 创建**完全独立**的通信组 (Communicator)。

*   **EP Group (Expert Parallel):** 仅包含负责不同专家的 GPU。MoE 的 `all_to_all` 通信必须严格传入此 `ProcessGroup`。
*   **HSDP Sharding Group:** 节点内的切片通信组 (通常对应一个 Node 里的 GPU)。
*   **HSDP Replication Group:** 节点间的数据并行通信组 (跨节点的相同 rank)。

```python
# 示例：假设 8 卡，EP=4, DP=2
# EP 组：[0, 1, 2, 3] 和 [4, 5, 6, 7]
ep_group = dist.new_group(ranks=[0, 1, 2, 3]) 
# 传递给 MoE 路由层
moe_layer = MoELayer(..., ep_group=ep_group) 
```
通过显式指定 `group=ep_group`，NCCL 会使用专门的 Communicator 执行 `all_to_all`，从底层杜绝了向非 EP 节点发送数据的可能。

### 2. 模型 Wrapping 的隔离 (解耦 Dense 与 MoE 权重)

HSDP 会在反向传播时自动触发梯度的 `ReduceScatter` 和 `AllReduce`。为了防止 HSDP 干扰 MoE，**绝对不能将 MoE 的 Expert 层与 Dense 层放在同一个 HSDP Wrap 单元中**。

*   **Dense Layers (如 Attention, LayerNorm):** 使用 HSDP (FSDP with `HYBRID_SHARD`) 进行 Wrap。
*   **Expert Layers (专家网络):** **不要**使用全局的 HSDP Wrap 它们。由于专家权重已经被 EP 分散，它们通常只需要在 DP 维度（如果有对应的 Replica）进行普通的 DDP (Distributed Data Parallel) 同步，或者如果 EP 涵盖了整个集群，则根本不需要梯度 AllReduce。

<details>
<summary><b>点击展开：PyTorch FSDP/HSDP 忽略 MoE 专家的代码示例</b></summary>

在 PyTorch FSDP 中，可以使用 `ignored_modules` 或自定义的 `auto_wrap_policy` 将专家层从 HSDP 中剔除。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 提取所有专家模块
expert_modules = [m for m in model.modules() if isinstance(m, ExpertLayer)]

# Wrap 整个模型，但忽略专家层
model = FSDP(
    model,
    auto_wrap_policy=custom_dense_wrap_policy,
    ignored_modules=expert_modules, # 关键：让 HSDP 不要管专家层
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(sharding_group, replication_group)
)

# 专家层的梯度同步由 EP 组的 DDP 或手动 AllReduce 处理
for expert in expert_modules:
    expert = DDP(expert, process_group=dp_group_for_experts)
```
</details>

### 3. CUDA Stream 分离与同步 (防止 NCCL 死锁)

这是最容易踩坑的地方。即使通信组不同，如果 `all_to_all` 和 HSDP 的 `AllReduce` 在同一个默认 CUDA Stream 上排队，由于不同 Rank 之间计算时间的微小差异，极易产生依赖反转，导致 NCCL 死锁。

**解决方案：**
为 MoE 的 `all_to_all` (前向和反向) 分配独立的 CUDA Stream，使其与 HSDP 后台的通信 Stream 异步。

1.  **HSDP Stream:** FSDP/HSDP 内部已经维护了独立的通信 Stream 用于 overlapping。
2.  **MoE Stream:** 在自定义的 `all_to_all` 算子中，使用 `torch.cuda.Stream()` 包裹通信。

```python
moe_comm_stream = torch.cuda.Stream()

def forward(self, hidden_states):
    # 1. 计算 Router (在 Default Stream)
    gates, indices = self.router(hidden_states)
  
    # 同步：确保 Router 计算完毕
    torch.cuda.current_stream().wait_stream(moe_comm_stream)
  
    # 2. All-To-All 分发 Token (在独立的 Stream)
    with torch.cuda.stream(moe_comm_stream):
        dispatched_tokens = dist.all_to_all_single(..., group=self.ep_group)
  
    # 同步：确保 Token 到达后再进行专家计算
    torch.cuda.current_stream().wait_stream(moe_comm_stream)
  
    # 3. 专家计算 (Default Stream)
    expert_outputs = self.experts(dispatched_tokens)
  
    # 4. All-To-All 收回 Token (在独立的 Stream)
    with torch.cuda.stream(moe_comm_stream):
        combined_output = dist.all_to_all_single(..., group=self.ep_group)
      
    return combined_output
```

### 4. 梯度反向传播 (Backward) 期间的 Hook 隔离

在 Backward 阶段：
*   **HSDP:** 随着 Dense 层的反向传播结束，HSDP 注册的 Autograd Hook 会被触发，开始执行 `ReduceScatter`。
*   **MoE:** 专家计算的反向传播结束后，触发反向的 `all_to_all` 把 Token 梯度还给原来的 Device。

为了不相互干扰：
1. 因为我们在第 2 步中**忽略了专家层的 HSDP Wrap**，HSDP 不会等待专家层的梯度，它只管自己的 Dense 权重。
2. Autograd 引擎会自动处理前向计算图中的 Stream 依赖。只要前向代码中插入了正确的 `wait_stream`，反向传播时 PyTorch 的 Autograd 引擎会以拓扑序自动推导依赖，确保反向的 `all_to_all` 与 Dense 梯度的计算并行且互不阻塞。

### 总结

要实现这两者的完美兼容，你的系统必须满足以下条件：

1.  **空间隔离：** $EP\_Group \cap HSDP\_Group = \emptyset$ (逻辑上的独立实例)。`all_to_all` 严格绑定 `ep_group`。
2.  **权重隔离：** Expert 参数不进入 HSDP 的管理范围，通过 `ignored_modules` 剔除。
3.  **时间/流隔离：** 为 MoE 通信创建专属 `cuda.Stream`，并与默认计算流做好 Event 同步。