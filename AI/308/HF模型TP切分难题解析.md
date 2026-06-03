# HF模型TP切分难题解析

### 🧑‍💻 User

hf模型config的tp plan里面有`replicated_with_grad_allreduce`、`packed_colwise`、`moe_tp_experts`这些无法直接映射到torch的tensor parallel原语怎么办？

---

### 🤖 Assistant

在处理 Hugging Face (HF) 模型的 Tensor Parallelism (TP) 配置时，遇到像 `replicated_with_grad_allreduce`、`packed_colwise`、`moe_tp_experts` 这种特定框架（如 Megatron-LM、vLLM 或 TGI）特有的切分策略，确实无法直接用 PyTorch 原生的 `RowwiseParallel` 或 `ColwiseParallel` 简单映射。

这主要是因为 PyTorch 的 `torch.distributed.tensor.parallel` API 相对通用且基础，而 HF config 中的这些 plan 往往包含了**业务逻辑级的切分策略**（例如多维度的 Packing 或 Expert 维度的切分）。

要解决这个问题，你需要**降维到 PyTorch 的底层分布式原语 (`DTensor` 和自定义 `ParallelStyle`)** 来手动实现这些映射。以下是针对这三种特殊 TP 策略的具体解决方案和映射方法。

---

### 核心解决思路

PyTorch 原生 TP 的底层是基于 `DeviceMesh` 和 `DTensor` (Distributed Tensor) 的。如果上层的 `ParallelStyle`（如 `ColwiseParallel`）不够用，我们可以：
1. 使用底层的 `Placement` 策略（`Replicate()`, `Shard(dim)`）。
2. 自定义 PyTorch 的 `ParallelStyle` 或者手动进行权重替换和前向/反向 Hook。

<details>
<summary><b>1. 解决 `replicated_with_grad_allreduce`</b></summary>

**含义：**
在 TP 组内，前向传播时每个 GPU 上保留完整的权重副本（Replicated）；反向传播时，由于各个 GPU 上的输入数据可能不同（例如在 Sequence Parallel 模式下），需要对该权重的梯度进行 AllReduce 操作，以保证梯度更新后各卡权重一致。这通常用于 LayerNorm 的权重、Bias 或非线性激活层的可学习参数。

**PyTorch 映射方案：**
实际上，PyTorch原生 `DTensor` 的 `Replicate()` placement **默认自带反向传播的 AllReduce 语义**。你不需要写额外的代码，只需将该 tensor 声明为 `Replicate()` 即可。

```python
import torch
from torch.distributed._tensor import distribute_tensor, Replicate
from torch.distributed.tensor.parallel import PrepareModuleInput

# 方案 A: 使用 DTensor 原生特性
# 假设 mesh 是你的 TP DeviceMesh
replicated_weight = distribute_tensor(
    original_weight, 
    device_mesh=mesh, 
    placements=[Replicate()]
)
# 当 replicated_weight 参与计算后，PyTorch 的 Autograd 机制
# 会在 backward 时自动对其梯度触发 AllReduce。

# 方案 B: 自定义 ParallelStyle 传入 parallelize_module
class ReplicatedWithGradAllReduce(ParallelStyle):
    def _apply(self, module, device_mesh):
        # 保持 module 不被切分，但输入输出可能需要特别处理
        # 依赖于其参数已经被放置为 Replicate()
        return module
```

</details>

<details>
<summary><b>2. 解决 `packed_colwise`</b></summary>

**含义：**
这是针对 QKV Attention 投影层或 MLP 的 Gate/Up 投影层设计的。通常，如果 Q, K, V 分开，我们会分别做 `ColwiseParallel`。但在 `packed_colwise` 中，它们被拼接成了一个巨大的矩阵 $W_{QKV} = [W_Q, W_K, W_V]$。
如果直接切分这个大矩阵，可能会破坏各个 Attention Head 的局部性（特别是在 GQA/MQA 中，Q, K, V 的维度大小不一致）。

**PyTorch 映射方案：**
不能直接使用标准的 `ColwiseParallel`，需要**先按逻辑拆包 (Unpack) -> 分别切分 (Shard) -> 再在本地打包 (Pack)**。你可以自定义一个函数来处理这个权重。

```python
from torch.distributed._tensor import Shard

def shard_packed_colwise(weight_tensor, mesh, num_heads, num_kv_heads):
    """
    针对 GQA/MQA 优化的 packed_colwise 切分逻辑
    假设 weight 形状为 [out_features, in_features]
    """
    tp_size = mesh.size(0)
  
    # 1. 逻辑上拆解出 Q, K, V 的权重
    # (这里的切片逻辑取决于具体模型，如 Llama)
    q_size = ... # 计算 Q 的行数
    k_size = ... # 计算 K 的行数
    v_size = ... # 计算 V 的行数
  
    wq, wk, wv = torch.split(weight_tensor, [q_size, k_size, v_size], dim=0)
  
    # 2. 对每个部分分别进行 Shard(0) (Colwise)
    wq_sharded = distribute_tensor(wq, mesh, [Shard(0)])
    wk_sharded = distribute_tensor(wk, mesh, [Shard(0)])
    wv_sharded = distribute_tensor(wv, mesh, [Shard(0)])
  
    # 3. 将切分后的本地 tensor 重新 concat，形成本地的 packed tensor
    # 这样可以保证计算时依然使用单次 GEMM
    local_packed_weight = torch.cat([
        wq_sharded.to_local(), 
        wk_sharded.to_local(), 
        wv_sharded.to_local()
    ], dim=0)
  
    # 将其重新包装为分布式 Tensor，声明其切分状态
    return DTensor.from_local(local_packed_weight, mesh, [Shard(0)])
```
然后在模型初始化时，用这个函数手动替换模型对应的 `weight`。

</details>

<details>
<summary><b>3. 解决 `moe_tp_experts`</b></summary>

**含义：**
这是 Mixture of Experts (MoE) 场景下的 Expert Parallelism (EP)。在 HF config 中复用了 TP 的 plan。意味着不再是对单个 Linear 层的 $W$ 进行切分，而是**将不同的 Expert 分配到不同的 TP Rank 上**。
例如，总共有 8 个 experts，TP=4，那么每个 GPU 分配 2 个完整的 experts。

**PyTorch 映射方案：**
MoE 权重通常被实现为 3D Tensor，形状为 `[num_experts, hidden_size, intermediate_size]` 或者是一个 `ModuleList`。我们需要针对**Expert 维度**（通常是第 0 维）进行切分，而不是 Feature 维度。

```python
# 假设 MoE 层的权重是一个 3D Tensor: [num_experts, out_features, in_features]
def apply_moe_tp_experts(moe_module, mesh):
    # 针对 expert 维度 (dim=0) 进行 Shard
    expert_placement = [Shard(0)]
  
    # 如果权重是 3D Tensor
    if hasattr(moe_module, 'weight'):
        moe_module.weight = torch.nn.Parameter(
            distribute_tensor(moe_module.weight.data, mesh, expert_placement)
        )
  
    # 如果专家是用 nn.ModuleList 存放的
    elif hasattr(moe_module, 'experts'):
        tp_rank = mesh.get_local_rank()
        tp_size = mesh.size(0)
        num_experts = len(moe_module.experts)
      
        # 计算当前 rank 应该保留的 experts
        experts_per_rank = num_experts // tp_size
        start_idx = tp_rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
      
        # 只保留本地需要的 experts，其他的删除以节省内存
        local_experts = moe_module.experts[start_idx:end_idx]
        moe_module.experts = torch.nn.ModuleList(local_experts)
      
        # 注意：前向传播 (Router逻辑) 需要修改，确保 token 
        # 被正确地 all-to-all 发送到对应的 rank 上处理。
```
*注：MoE 的并行不仅仅是权重的切分，还需要在 Router 分配 token 后引入 `AllToAll` 通信机制，这通常需要覆写 MoE 模块的 `forward` 方法或使用专门的 MoE 通信库（如 `megablocks` 或 PyTorch 原生的 `torch.distributed._tensor.ops`）。*

</details>

---

### 总结与推荐实践

当你试图在 PyTorch 中加载带有这些高级特性的 HF 配置文件时，建议的开发流程如下：

1. **放弃纯声明式的 `parallelize_module`**：对于标准的 `nn.Linear` 依然可以使用 `ColwiseParallel` / `RowwiseParallel`。
2. **编写自定义切分工厂函数**：遍历模型的 named parameters/modules。
3. **基于名字应用策略**：
   ```python
   for name, module in model.named_modules():
       if "qkv_proj" in name:
           # 调用上面写的 shard_packed_colwise
           replace_with_packed_colwise(module, device_mesh)
       elif "experts" in name:
           # 调用 apply_moe_tp_experts
           apply_moe_tp_experts(module, device_mesh)
       elif "norm" in name:
           # PyTorch DTensor 默认行为就是 replicated_with_grad_allreduce
           pass 
   ```

如果是为了快速在生产环境部署，建议直接使用集成了这些底层优化的推理/训练框架（如 **vLLM** 应对推理，**Torchtitan** 或 **Megatron-LM** 应对训练），因为它们内部已经为你写好了这些定制化算子和切分逻辑。