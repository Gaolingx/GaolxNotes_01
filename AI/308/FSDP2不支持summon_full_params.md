# FSDP2不支持summon_full_params

### 🧑‍💻 User

pytorch fsdp2支持使用summon_full_params吗？

---

### 🤖 Assistant

在 PyTorch FSDP2 中，**不支持**直接使用 FSDP1 中经典的 `FSDP.summon_full_params` 上下文管理器。

FSDP2（通常通过 `torch.distributed.fsdp.fully_shard` 或 `torch.distributed._composable.fsdp` 调用）采用了全新的**可组合（Composable）API 设计**，不再像 FSDP1 那样将模型包装在一个臃肿的 `FullyShardedDataParallel` 类中。因此，依赖于该包装器类的 `summon_full_params` 方法在 FSDP2 中被弃用并替换。

如果你在 FSDP2 中需要获取完整的模型参数，你需要根据具体的**使用场景**采用不同的替代方案。

### FSDP2 中的替代方案

#### 1. 场景一：保存和加载模型权重（最常见场景）
在 FSDP1 中，开发者经常使用 `summon_full_params` 将所有参数收集到 rank 0 上进行保存（即 `torch.save`）。
在 FSDP2 中，官方强烈推荐使用 **Distributed Checkpoint (DCP)**。DCP 允许各个 GPU 并行地保存其分片的参数，完全不需要将参数集中到单个内存中，从而避免了 OOM 问题。

```python
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import fully_shard

# 1. 使用 FSDP2 分片模型
fully_shard(model)

# 2. 保存模型 (替代 summon_full_params 的做法)
state_dict = model.state_dict()
dcp.save(state_dict, checkpoint_id="/path/to/checkpoint")

# 3. 如果需要单机的完整权重，可以使用 DCP 提供的离线合并工具：
# python -m torch.distributed.checkpoint.format_utils dcp_to_torch /path/to/checkpoint /path/to/full_model.pt
```

#### 2. 场景二：在运行时（In-memory）需要完整参数
如果你的目的不是保存模型，而是在训练或推理过程中临时需要完整的参数（例如计算某种全局范数、特定的自定义验证逻辑等），在 FSDP2 中可以通过调整 `state_dict` 的类型来获取。

你可以配置 FSDP2 以返回完整的、未分片的 `state_dict`：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# 设置 StateDictType 为 FULL_STATE_DICT
# FSDP2 会在调用 model.state_dict() 时自动 gather 所有参数
full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(
    model, 
    StateDictType.FULL_STATE_DICT, 
    full_state_dict_config
):
    full_state_dict = model.state_dict()
    # 此时 full_state_dict 包含了完整的模型权重
```
*(注：虽然 FSDP2 是基于函数式的 `fully_shard`，但它依然与 PyTorch 原生的分布式状态字典 API 和 `StateDictType` 上下文管理器兼容。)*

---

<details>
<summary><b>点击展开：为什么 FSDP2 放弃了 summon_full_params？</b></summary>

1. **架构变更**：FSDP1 的 `summon_full_params` 依赖于 `FullyShardedDataParallel` 这个大 Wrapper 类来管理状态。FSDP2 的核心理念是**非侵入性**（Non-intrusive），直接修改底层 Tensor 的 `__class__`（即 DTensor），不再包装 `nn.Module`，因此无法在 Wrapper 上挂载此类方法。
2. **内存安全性**：`summon_full_params` 极易导致内存溢出（OOM），因为在千亿参数大模型时代，即使是 Rank 0 的 CPU 内存也可能无法装下完整的模型。
3. **生态对齐**：PyTorch 生态正在全面向 `DTensor` (Distributed Tensor) 和 DCP 迁移。未来的分布式操作都基于底层的 Tensor 级别的集合通信（Collective Communications），而不是 Module 级别的上下文管理器。

</details>