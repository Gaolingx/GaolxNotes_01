# 使用torchao进行fp8训练

### 🧑‍💻 User

在fp8训练中，torchao的convert_to_float8_training如何使用?

---

### 🤖 Assistant

在 PyTorch 的 `torchao` (Torch Architecture Optimization) 库中，`convert_to_float8_training` 是一个用于将标准模型转换为支持 FP8 混合精度训练的核心函数。它通过将模型中的 `nn.Linear` 替换为 FP8 版本的线性层，并自动处理张量缩放（Scaling）和格式转换（通常前向传播使用 E4M3，反向传播使用 E5M2），从而显著减少显存占用并提升计算速度。

以下是 `convert_to_float8_training` 的详细使用指南。

### 1. 核心工作原理

当你调用该函数时，它会遍历你的模型，并将支持的层（主要是 `nn.Linear`）替换为 FP8 变体。
在底层，它默认使用 **延迟缩放 (Delayed Scaling)** 策略。缩放因子 $S_t$ 的计算通常基于历史的最大绝对值 (amax)：
$$S_{t} = \frac{\text{MAX\_FP8}}{\max(|X_{t-1}|)}$$
其中 $X_{t-1}$ 是前一步的激活值或梯度。

### 2. 基础使用流程

使用该函数通常包含以下四个步骤：
1. 以高精度（如 `bfloat16`）初始化模型。
2. 调用 `convert_to_float8_training` 转换模型。
3. **（强烈建议）** 使用 `torch.compile` 编译模型以获得实际的加速。
4. 正常的训练循环。

### 3. 完整代码示例

```python
import torch
import torch.nn as nn
from torchao.float8 import convert_to_float8_training

# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型并移动到GPU，通常底层权重保持在 bfloat16
device = "cuda"
model = SimpleModel().to(device=device, dtype=torch.bfloat16)

# 2. 将模型转换为 FP8 训练模式
convert_to_float8_training(model)

# 3. 使用 torch.compile 编译模型
# 注意：在 PyTorch 中，必须使用 compile 才能将零碎的 cast 和 amax 计算融合，
# 否则 Eager 模式下的 FP8 训练可能会比 BF16 更慢！
model = torch.compile(model)

# 4. 标准训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 模拟输入数据
x = torch.randn(32, 1024, device=device, dtype=torch.bfloat16)
target = torch.randn(32, 1024, device=device, dtype=torch.bfloat16)

# 训练步骤
optimizer.zero_grad()
output = model(x)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print("FP8 训练步骤完成！")
```

---

### 进阶指南与注意事项

为了帮助您在生产环境中更好地使用该功能，以下提供了一些进阶配置和硬件要求的详细信息：

<details>
<summary><b>🛠️ 硬件与软件环境要求 (展开查看)</b></summary>

*   **硬件要求:** FP8 张量核心 (Tensor Cores) 仅在 **NVIDIA Ada Lovelace** (如 RTX 4090, L40) 和 **Hopper** (如 H100, H800) 及更新架构的 GPU 上受硬件支持。在旧 GPU 上运行会回退到软件模拟，极其缓慢。
*   **软件要求:** 
    *   PyTorch $\ge 2.4$ (建议使用 2.5 或 Nightly 版本以获得最佳编译支持)。
    *   安装最新版的 `torchao`：
        ```bash
        pip install torchao
        ```
</details>

<details>
<summary><b>⚙️ 高级配置 (过滤层与缩放策略) (展开查看)</b></summary>

你可以通过传入 `Float8LinearConfig` 或自定义过滤器来控制哪些层被转换。例如，通常不建议将第一层（Embedding/输入层）或最后一层（LM Head）转换为 FP8，以保证模型精度。

```python
from torchao.float8 import Float8LinearConfig, Float8Recipe

# 只转换特定名称的层，或者排除特定层
def module_filter_fn(module: nn.Module, fqn: str):
    # fqn: Fully Qualified Name (如 'layers.0.mlp.fc1')
    if "lm_head" in fqn or "embed" in fqn:
        return False
    return isinstance(module, nn.Linear)

config = Float8LinearConfig(
    recipe=Float8Recipe(delayed_scaling=True), # 默认使用延迟缩放
)

convert_to_float8_training(
    model,
    config=config,
    module_filter_fn=module_filter_fn
)
```
</details>

<details>
<summary><b>🚀 与 FSDP (完全分片数据并行) 结合使用 (展开查看)</b></summary>

在训练大语言模型 (LLM) 时，FP8 经常需要与分布式训练框架结合。在 PyTorch 中，通常先调用 `convert_to_float8_training`，然后再使用 `FSDP` 包装模型。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 1. 转换模型
convert_to_float8_training(model)

# 2. 使用 FSDP 包装
model = FSDP(
    model,
    use_orig_params=True, # 必须设置为 True，因为 FP8 引入了额外的张量状态
    # ... 其他 FSDP 参数 ...
)

# 3. 编译
model = torch.compile(model)
```
注意：使用 FSDP 时，FP8 的 Amax (最大绝对值) 统计信息可能需要在跨 Rank 之间进行同步，这取决于你使用的 `Float8Recipe` 的具体配置。
</details>