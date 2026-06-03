# fully_shard与FullyShardedDataParallel区别

### 🧑‍💻 User

torch fsdp的遍历模块fully_shard和直接用FullyShardedDataParallel(model)有什么区别？

---

### 🤖 Assistant

在 PyTorch 中，`torch.distributed.fsdp.fully_shard`（通常被称为 **Composable FSDP** 或可组合 API）和 `FullyShardedDataParallel(model)`（传统的 **Wrapper FSDP** 或包装器 API）是实现完全分片数据并行的两种不同方式。

简单来说，**`FullyShardedDataParallel` 是旧版的、基于类的包装器，而 `fully_shard` 是 PyTorch 2.x 引入的新版、原位（in-place）函数式 API。** 官方目前强烈推荐在新的项目中使用 `fully_shard`。

以下是它们的核心区别：

### 1. 对模型层级结构的影响（侵入性）

*   **`FullyShardedDataParallel(model)` (Wrapper 模式):**
    这是一种“包装”操作。它会改变模型的类型和层级结构。包装后，你的模型类型变成了 `FullyShardedDataParallel`。如果你想访问原来的模型属性，通常需要通过 `model.module` 来访问。这会破坏很多依赖模型特定结构的代码（例如通过名字查找特定层）。
*   **`fully_shard(model)` (Composable 模式):**
    这是一个“原位”修改操作。它不会改变模型的类型（`type(model)` 保持不变），也不会改变层级结构。它通过在模型上注册 hooks 来实现分片。你可以像单卡模型一样，无缝地访问 `model.layer1` 等属性。

### 2. 子模块分片控制（细粒度）

对于大模型，我们不能只对最外层进行 FSDP 包装，因为那样会在前向传播时一次性收集所有参数，导致 OOM。我们需要对模型的各个层（例如 Transformer Block）分别进行分片。

*   **Wrapper 模式:**
    你必须定义一个 `auto_wrap_policy`（例如基于层级或大小的策略），然后将这个策略传给最外层的 `FullyShardedDataParallel`。FSDP 会在内部递归地创建嵌套的 FSDP Wrapper。逻辑相对黑盒，调试较难。
*   **Composable 模式:**
    你可以像写普通 Python 循环一样，遍历模型的子模块，并直接对它们调用 `fully_shard()`。这种方式极其直观，你可以完全控制哪些层被分片，哪些层不被分片（或者使用不同的分片策略）。

### 3. 与 PyTorch 新特性的兼容性（尤其是 `torch.compile`）

*   **Wrapper 模式:**
    由于其复杂的嵌套类结构和深层的 state 改变，它与 PyTorch 2.0 的 `torch.compile`（图编译优化）结合时，经常会遇到图断裂（graph breaks）或兼容性问题。
*   **Composable 模式:**
    `fully_shard` 的设计初衷之一就是为了与 `torch.compile` 完美结合。由于它保留了原始的 `nn.Module` 结构，编译器可以更容易地追踪前向和反向传播的计算图。

---

<details>
<summary><b>展开查看：代码写法对比示例</b></summary>

**传统 Wrapper 模式 (FullyShardedDataParallel):**

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

model = MyTransformerModel()

# 必须定义自动包装策略来切分内部的 Transformer Block
my_auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

# 包装整个模型，返回一个 FSDP 类型的对象
fsdp_model = FSDP(
    model,
    auto_wrap_policy=my_auto_wrap_policy,
    device_id=torch.cuda.current_device()
)

# 访问原模型属性需要加 .module
# print(fsdp_model.layer1) # 报错
print(fsdp_model.module.layer1) # 正确
```

**现代 Composable 模式 (fully_shard):**

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

model = MyTransformerModel()

# 直接遍历并原位分片各个子模块
for layer in model.transformer_blocks:
    fully_shard(layer)

# 最后分片最外层模型
fully_shard(model)

# 模型的类型仍然是 MyTransformerModel
# 可以直接访问属性，就像没用 FSDP 一样
print(model.transformer_blocks[0]) 

# 无缝配合 torch.compile
compiled_model = torch.compile(model)
```
</details>

<details>
<summary><b>展开查看：State Dict (权重保存与加载) 的区别</b></summary>

*   **Wrapper 模式:** 需要使用专门的 Context Manager（如 `FSDP.state_dict_type`）来提取和加载权重，因为参数被展平且名称发生了改变（由于外层包装）。
*   **Composable 模式:** 与 `torch.distributed.checkpoint` (DCP) 集成得更好。因为模型结构没变，其 `state_dict` 的 key 也是原始名称，更容易保存和在不同并行策略间（甚至单卡）进行权重的转换和加载。
</details>

### 总结建议

如果你正在维护基于 PyTorch 1.x 的老代码，你可能会看到大量的 `FullyShardedDataParallel`。但如果你是在 **PyTorch 2.1 及以上版本** 开发新模型，**请务必使用 `fully_shard`**。它更符合 Pythonic 直觉，更容易调试，且与现代 PyTorch 生态（如 `torch.compile`）兼容性极佳。