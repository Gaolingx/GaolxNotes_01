# PyTorch Lightning支持TensorBoard模型图显示

### 🧑‍💻 User

pytorch lightning使用tensorboard时候除了log打印，是否还支持模型结构和计算图显示？

---

### 🤖 Assistant

**是的，PyTorch Lightning 完全支持在 TensorBoard 中显示模型结构和计算图。**

要在 TensorBoard 中可视化你的模型结构，你只需要完成两个关键步骤：

1. 在你的 `LightningModule` 中定义一个 **示例输入（`example_input_array`）**。
2. 在初始化 `TensorBoardLogger` 时，将 **`log_graph`** 参数设置为 `True`。

PyTorch 底层需要一个实际的输入张量来进行一次前向传播（forward pass），从而追踪并绘制出计算图。这就是为什么必须提供 `example_input_array` 的原因。

### 完整示例代码

以下是一个简单的实现示例：

```python
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
      
        # 关键点 1：提供一个示例输入张量。
        # 它的形状(shape)和数据类型(dtype)必须与模型 forward 方法接收的输入一致。
        self.example_input_array = torch.randn(1, 28 * 28)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 训练逻辑...
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 关键点 2：配置 TensorBoardLogger，并开启 log_graph=True
logger = TensorBoardLogger(save_dir="tb_logs", name="my_model", log_graph=True)

# 将 logger 传入 Trainer
trainer = pl.Trainer(logger=logger, max_epochs=1)

model = MyModel()

# 即使不调用 trainer.fit()，只要执行了 fit/test/predict，图就会被记录
# trainer.fit(model, train_dataloader)
```

<details>
<summary><b>💡 进阶：手动记录计算图 (如果不使用 example_input_array)</b></summary>

如果你不想在 `__init__` 中定义 `example_input_array`（例如你的输入非常复杂，包含多个字典或列表），你可以直接调用底层 TensorBoard 实例的 `add_graph` 方法。

你可以重写 `LightningModule` 的生命周期钩子（比如 `on_train_start` 或在第一次 `training_step` 时）：

```python
class MyModel(pl.LightningModule):
    # ... 省略 __init__ 和 forward ...

    def on_train_start(self):
        # 创建一个 dummy input
        sample_input = torch.randn(1, 28 * 28).to(self.device)
      
        # 获取底层的 SummaryWriter 实例并记录图
        # 注意：这里需要确保你使用的是 TensorBoardLogger
        self.logger.experiment.add_graph(self, sample_input)
```
</details>

### 如何查看
运行代码后，打开终端并在日志目录下启动 TensorBoard：

```bash
tensorboard --logdir tb_logs/
```

打开浏览器访问 `http://localhost:6006`，在顶部的标签页中找到 **"GRAPHS"** 标签，你就可以看到展开的模型结构和数据流向了。你可以双击各个模块来展开查看内部的具体算子（如 MatMul, Relu 等）。