# FSDP配置FULL_SHARD最大化显存叠加

### 🧑‍💻 User

pytorch lightning中fsdp如何配置sharing strategy等，显存充足情况下一般选哪种（已知ddp已经能跑起来但bs32已经80% vram usage了，希望多卡显存叠加堆更大bs）

---

### 🤖 Assistant

在 PyTorch Lightning 中配置 FSDP（Fully Sharded Data Parallel）的 `sharding_strategy` 非常简单。既然你当前 DDP 跑 BS=32 已经占用了 80% 的显存，并且你的核心诉求是**“多卡显存叠加来堆更大的 Batch Size”**，以下是针对你情况的最佳实践和配置方法。

### 💡 核心结论与策略选择

如果你的目标是尽最大可能**叠加多卡显存以提升 Batch Size**，你应该选择：

1.  **首选：`FULL_SHARD` (相当于 ZeRO-3)**
    *   **原因**：它会将模型参数、梯度和优化器状态全部切片到各个 GPU 上。你的总显存池将变成 $VRAM_{total} = N \times VRAM_{single}$（$N$ 为 GPU 数量）。这是最大化显存叠加的唯一方式，能让你把 Batch Size 推到极限。
2.  **备选：`SHARD_GRAD_OP` (相当于 ZeRO-2)**
    *   **原因**：如果你觉得 `FULL_SHARD` 通信开销太大导致训练变慢，且你只需要把 BS 从 32 提升到比如 64 或 128，这个策略只切分梯度和优化器状态，保留完整的模型参数，速度比 `FULL_SHARD` 快，显存节省介于 DDP 和 `FULL_SHARD` 之间。

---

### 🛠️ 如何在 PyTorch Lightning 中配置

你需要通过引入 `FSDPStrategy` 并将其传给 `Trainer` 来进行配置。

```python
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

# 1. 定义 Auto Wrap Policy (FSDP 必须配置，否则无法真正节省显存)
# 这里以最简单的按参数量切分为例，通常建议使用 transformer_auto_wrap_policy
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=1e5
)

# 2. 配置 FSDP 策略
fsdp_strategy = FSDPStrategy(
    # 选择 FULL_SHARD 以最大化显存池，或者 SHARD_GRAD_OP 追求速度平衡
    sharding_strategy=ShardingStrategy.FULL_SHARD, 
  
    # 必须提供 auto_wrap_policy，否则整个模型作为一个大块，等同于没切片
    auto_wrap_policy=my_auto_wrap_policy, 
  
    # 如果显存绝对充足，设为 False。如果还想更大 BS，可以设为 True（但会大幅拖慢速度）
    cpu_offload=False, 
  
    # 开启激活重计算，对大 Batch Size 极其重要
    activation_checkpointing_policy=None # 可在模型层配置
)

# 3. 传入 Trainer
trainer = Trainer(
    accelerator="gpu",
    devices=4, # 假设你有 4 张卡
    strategy=fsdp_strategy,
    precision="bf16-mixed", # 强烈建议配合 BF16/FP16 使用
)

# trainer.fit(model)
```

---

### 🔍 深度解析与进阶技巧 (点击展开)

<details>
<summary><b>1. 详细的 Sharding Strategy 对比</b></summary>

*   `ShardingStrategy.FULL_SHARD` (默认): 参数、梯度、优化器状态全部切片。**最大化显存，通信开销最大**。适合你的“堆显存”需求。
*   `ShardingStrategy.SHARD_GRAD_OP`: 仅切分梯度和优化器状态，参数在每个 GPU 上保留副本。**显存节省中等，通信开销较小**。
*   `ShardingStrategy.NO_SHARD`: 不切片，基本等同于 DDP。
*   `ShardingStrategy.HYBRID_SHARD`: 节点内 `FULL_SHARD`，节点间 DDP。适用于多机多卡（如 $8 \times 8$ A100）且跨节点网络较慢的场景。
</details>

<details>
<summary><b>2. 为什么 auto_wrap_policy 极其重要？</b></summary>

FSDP 工作原理是将模型“分块”（Wrap）。如果你不配置 `auto_wrap_policy`，FSDP 默认会将整个模型作为一个大块，这意味着在 Forward 和 Backward 时，整个模型都需要被完整 gather 到显存中，这会导致 **OOM**，起不到任何节省显存叠加 BS 的作用。

对于 Transformer 模型，强烈建议使用基于层的 Wrap 策略：
```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer # 以 Llama 为例

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)
```
</details>

<details>
<summary><b>3. 进一步压榨显存扩大 BS 的手段</b></summary>

即使使用了 `FULL_SHARD`，当 Batch Size 变得极大时，**激活值（Activations）** 产生的显存占用会超过模型参数。

如果在使用 `FULL_SHARD` 后想进一步提高 BS，你应该：
1.  **开启 Activation Checkpointing (Gradient Checkpointing)**：用计算时间换显存空间。在 Lightning 中，通常在你的 Model 的 `__init__` 中调用原生 PyTorch / HuggingFace 的 checkpointing 方法，或者在 `FSDPStrategy` 中配置 `activation_checkpointing_policy`。
2.  **使用 Mixed Precision**：`Trainer(precision="bf16-mixed")` 或者 `16-mixed`，直接将参数和激活值的显存占用减半。
</details>

### 总结建议

既然你说“已知 DDP 已经能跑起来且显存充足”，说明你的模型单卡放得下。直接在 `FSDPStrategy` 中使用 **`ShardingStrategy.FULL_SHARD`**，务必配合正确的 **`auto_wrap_policy`**。你的最大可用 Batch Size 将从 $BS_{max}$ 跃升至近似 $BS_{max} \times GPUs$，完美实现“多卡显存叠加堆更大 BS”的目的。