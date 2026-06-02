# PyTorch Lightning中使用FSDP2

### 🧑‍💻 User

pytorch lightning中如何使用新的fsdp2？

---

### 🤖 Assistant

在 PyTorch Lightning 中使用全新的 **FSDP2** (Fully Sharded Data Parallel v2) 可以显著提升大规模模型训练的性能和内存效率。FSDP2 采用了全新的底层架构（基于 `torch.distributed._composable.fsdp`），去除了 FSDP1 中的很多复杂的类包装，变得更加灵活。

要在 PyTorch Lightning 中使用 FSDP2，你需要确保环境满足版本要求，并通过配置 `Trainer` 或自定义 `configure_model` 来实现。

### 1. 环境要求
*   **PyTorch**: $\ge 2.4.0$ (FSDP2 在 PyTorch 2.4 中被正式引入并提供稳定的 Composable API)
*   **PyTorch Lightning**: $\ge 2.4.0$

---

### 2. 最简单的使用方法 (自动包装)

如果你只想快速启用 FSDP2，可以在 `Trainer` 中直接传入字符串 `strategy="fsdp2"`。Lightning 会尝试自动为你进行分片包装。

```python
import lightning as L
import torch
import torch.nn as nn

class MyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024)
        self.layer2 = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.layer2(self.layer1(x))

    def training_step(self, batch, batch_idx):
        # ... 训练逻辑 ...
        pass

# 直接使用 "fsdp2" 字符串
trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy="fsdp2", 
    precision="bf16-mixed" # 推荐搭配混合精度使用
)

model = MyModel()
trainer.fit(model)
```

---

### 3. 高级用法：自定义 FSDP2 策略

你可以通过导入 `FSDP2Strategy` 类来精细控制 FSDP2 的行为，例如修改底层通信策略或启用 CPU 卸载（CPU Offloading）。

```python
from lightning.pytorch.strategies import FSDP2Strategy

# 自定义 FSDP2 策略
strategy = FSDP2Strategy(
    # FSDP2 暂不使用 FSDP1 复杂的 auto_wrap_policy，
    # 更多依赖底层的自动启发式或完全手动包装（见第4节）。
    # 你可以传入底层的配置参数：
    data_parallel_size="auto", # 用于配合张量并行或混合并行
    tensor_parallel_size="auto",
    # 可以添加其他受支持的 FSDP2 kwargs
)

trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy=strategy,
    precision="bf16-mixed"
)
```

---

### 4. 最佳实践：使用 `configure_model` 手动包装 (Manual Wrapping)

FSDP2 最大的改进之一是提供了组合式 API (`fully_shard`)。为了获得最佳性能，强烈建议在 `LightningModule` 的 `configure_model` 钩子中**手动对模型的各个子模块进行分片**。

这种方法允许你精确控制哪些层被放在一起作为一个分片单元，这对 Transformer 架构尤为重要。

```python
import lightning as L
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Linear(4096, 4096)
        # ... attention 等 ...

    def forward(self, x):
        return self.ffn(x)

class MyLLM(L.LightningModule):
    def __init__(self):
        super().__init__()
        # 此时模型尚未在 GPU 上初始化
        self.embedding = nn.Embedding(32000, 4096)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(12)])
        self.lm_head = nn.Linear(4096, 32000)

    def configure_model(self):
        # 1. 遍历并对基础的大型重复模块进行 fully_shard
        for block in self.blocks:
            fully_shard(block)
      
        # 2. 对其他大型层进行分片（可选）
        fully_shard(self.embedding)
        fully_shard(self.lm_head)
      
        # 3. 最后，对整个 LightningModule 进行一次 fully_shard 兜底
        fully_shard(self)

    # ... 其他训练逻辑 ...

# 训练器配置
trainer = L.Trainer(
    accelerator="cuda",
    devices=4,
    strategy="fsdp2", 
    precision="bf16-mixed"
)

model = MyLLM()
trainer.fit(model) # Lightning 会在底层自动调用 configure_model
```

---

<details>
<summary><b>展开查看：FSDP1 与 FSDP2 的核心区别（为什么推荐升级？）</b></summary>

*   **API 设计 (Composable API)**：
    *   **FSDP1** 使用面向对象的包装器 (`FullyShardedDataParallel(model)`)，这会改变模块的类型并破坏一些基于 `isinstance` 的检查。它强迫整个模型变成一个层次结构。
    *   **FSDP2** 使用函数式的组合 API (`fully_shard(model)`)。它不会改变原始模块的类，而是将分片逻辑作为钩子（hooks）注入到 PyTorch 模块中，对用户代码的侵入性极低。
*   **显存管理**：FSDP2 采用了更加扁平化的参数管理机制，不再像 FSDP1 那样严格绑定到模块层级，这使得多维并行（FSDP + 张量并行 / TP）的结合变得容易得多。
*   **Lightning 支持**：在 Lightning 中，`strategy="fsdp"` 指向旧版 FSDP1，而 `strategy="fsdp2"` 指向基于 `torch.distributed._composable.fsdp` 的新实现。

</details>