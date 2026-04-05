# PyTorch Lightning FSDPStrategy配置指南

### 🧑‍💻 User

pytorch lightning的 FSDPStrategy如何配置？

---

### 🤖 Assistant

在 PyTorch Lightning 中，`FSDPStrategy` (Fully Sharded Data Parallel) 是用于训练超大模型（如 LLMs）的核心策略。它通过跨 GPU 分片参数、梯度和优化器状态，极大地降低了显存占用。

配置 `FSDPStrategy` 可以从极简的默认配置，一直深入到精细的内存和计算优化。以下是详细的配置指南。

### 1. 基础配置 (极简模式)

如果你只是想快速启用 FSDP，可以直接在 `Trainer` 中传入字符串 `"fsdp"`。Lightning 会使用默认配置：

```python
import lightning as L

trainer = L.Trainer(
    devices=4,
    accelerator="gpu",
    strategy="fsdp",
    precision="16-mixed" # 推荐配合混合精度使用
)
```

### 2. 高级配置 (使用 `FSDPStrategy` 类)

为了发挥 FSDP 的最大威力，你需要实例化 `FSDPStrategy` 并自定义参数。最关键的配置是 **Auto Wrapping Policy（自动包装策略）**。如果不配置它，FSDP 会把整个模型当作一个大块，退化成类似普通的 DDP，导致 OOM。

```python
import torch
import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# 假设你的模型由多个 TransformerBlock 组成
from my_model import TransformerBlock 

# 1. 定义自动包装策略 (极度重要)
# 这告诉 FSDP 在哪些层进行切分，通常是 Transformer 的 Block 层
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# 2. 实例化 FSDPStrategy
fsdp_strategy = FSDPStrategy(
    auto_wrap_policy=auto_wrap_policy,
  
    # 分片策略：
    # FULL_SHARD (默认): 分片参数、梯度、优化器状态 (类似 DeepSpeed Zero-3)
    # SHARD_GRAD_OP: 分片梯度和优化器状态 (类似 DeepSpeed Zero-2)
    # NO_SHARD: 不分片 (类似 DDP)
    sharding_strategy=ShardingStrategy.FULL_SHARD,
  
    # CPU 卸载：将不活跃的参数卸载到 CPU 内存，牺牲速度换取显存
    cpu_offload=CPUOffload(offload_params=True), # 内存极度紧张时开启
  
    # 激活检查点 (Activation Checkpointing)：用计算时间换取显存
    # 传入需要做 Checkpoint 的层的类
    activation_checkpointing_policy={TransformerBlock},
  
    # 限制并发通信的流数量 (通常保持默认即可)
    limit_all_gathers=True,
  
    # 模型状态字典类型：决定保存 checkpoint 时的格式
    # "FULL_STATE_DICT" (合并为一个完整模型，适合推理)
    # "SHARDED_STATE_DICT" (保持分片状态，适合继续训练)
    state_dict_type="FULL_STATE_DICT" 
)

# 3. 传入 Trainer
trainer = L.Trainer(
    devices=8,
    accelerator="gpu",
    strategy=fsdp_strategy,
    precision="bf16-mixed", # 推荐使用 bfloat16
)

trainer.fit(model)
```

---

### 3. 核心配置参数详解

<details>
<summary><b>1. auto_wrap_policy (自动包装策略)</b></summary>

FSDP 需要知道如何将你的模型“切块”。
*   **基于类的策略 (推荐 Transformer 使用)**: 使用 `transformer_auto_wrap_policy`，指定 Transformer Block 的类。这样 FSDP 会以每个 Block 为单位进行通信和显存分配。
*   **基于大小的策略**: 如果不是 Transformer，可以使用 `size_based_auto_wrap_policy`，例如设定 `min_num_params=1e6`，超过 100 万参数的子模块就会被单独包装。
</details>

<details>
<summary><b>2. activation_checkpointing_policy (激活重计算)</b></summary>

在反向传播时，不再保存前向传播的所有激活值，而是丢弃它们并在需要时重新计算。这能节省海量显存。在 `FSDPStrategy` 中，你可以直接传入一个包含目标层类名（如 `TransformerBlock`）的集合（`set`），Lightning 会自动为你应用它。
</details>

<details>
<summary><b>3. sharding_strategy (分片策略)</b></summary>

*   `ShardingStrategy.FULL_SHARD`: 默认行为。参数、梯度、优化器状态全部分片。最省显存，通信量最大。
*   `ShardingStrategy.SHARD_GRAD_OP`: 仅分片梯度和优化器状态，参数不分片。类似 DeepSpeed Zero-2。如果你的显存足够装下模型参数，使用这个可以加快训练速度。
*   `ShardingStrategy.HYBRID_SHARD`: 在节点内（如一台 8 卡机器）做 `FULL_SHARD`，在节点间做数据并行。适合多机多卡，能显著降低跨节点网络通信开销。
</details>

### 4. 常见陷阱与注意事项

1.  **模型初始化 OOM**: 如果模型极大（例如 70B），在单卡上 `__init__` 就会 OOM。在 Lightning 中，你应该使用 `configure_model` 钩子，或者结合 `Fabric` 来进行空权重初始化（Empty Init）。
    ```python
    class MyModel(L.LightningModule):
        def configure_model(self):
            # 当使用 FSDP 时，这里的模型初始化会被 Lightning 拦截并做优化
            # 避免在 CPU 或单 GPU 上分配全量内存
            self.model = HugeLLM() 
    ```
2.  **优化器配置**: 必须在 `configure_optimizers` 中直接返回优化器。FSDP 会在底层自动接管并分片优化器状态。
3.  **Checkpoint 保存**: FSDP 保存的模型默认是分片的。如果你想保存一个可以用普通 PyTorch 加载的单体模型（用于推理），确保在 `FSDPStrategy` 中设置了 `state_dict_type="FULL_STATE_DICT"`（注意：合并模型需要有足够的 CPU 内存）。

通过合理搭配 `auto_wrap_policy`、`bf16-mixed` 精度和 `activation_checkpointing_policy`，你可以在消费级多卡机器上训练数十亿甚至上百亿参数的模型。