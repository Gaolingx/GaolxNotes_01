# pytorch冻结参数两种写法区别

### 🧑‍💻 User

pytorch中，`requires_grad_`和
```
for param in model.features.parameters():
    param.requires_grad_(False)
```
有什么区别？

---

### 🤖 Assistant

在 PyTorch 中，直接对模块调用 `requires_grad_()` 和使用 `for` 循环遍历参数调用 `requires_grad_()` **在最终效果上是完全一样的**，但它们在**语法简洁度、适用对象以及 PyTorch 版本历史**上有所区别。

简单来说，以下两段代码的作用等价（都是为了冻结 `model.features` 部分的模型参数）：

**写法 A（现代且简洁）：**
```python
model.features.requires_grad_(False)
```

**写法 B（传统且繁琐）：**
```python
for param in model.features.parameters():
    param.requires_grad_(False)
```

以下是详细的区别和解析：

### 1. 作用对象的区别
*   **`Tensor.requires_grad_()`**：这是作用于**单个张量（Tensor）**的方法。你的 `for` 循环其实就是提取出底层的每一个权重张量，然后逐个修改它们的属性。
*   **`nn.Module.requires_grad_()`**：这是作用于**整个网络模块（Module）**的方法。当你对一个模块（如 `model.features`）调用此方法时，PyTorch 会在底层自动帮你遍历该模块及其所有子模块下的每一个参数（Parameter），并统一修改它们的 `requires_grad` 属性。

### 2. PyTorch 版本历史的原因
你之所以会经常看到 `for` 循环的写法（写法 B），是因为历史遗留原因：
*   **在 PyTorch 1.11.0 之前**：`nn.Module` 并没有 `requires_grad_()` 这个直接的方法。开发者如果想冻结某一层或某个模块，**只能**通过 `for` 循环遍历 `parameters()` 来实现。因此，很多早期的教程、开源代码（比如早期的迁移学习教程）都保留了这种写法。
*   **在 PyTorch 1.11.0 及之后**：官方为了提高代码的简洁性和可读性，为 `nn.Module` 增加了 `requires_grad_()` 方法。官方在底层替你实现了这个 `for` 循环。

### 3. 代码的可读性与推荐用法
*   **推荐使用：`model.features.requires_grad_(False)`**
*   **原因**：它更加符合 Pythonic 的风格，代码更短，意图更明确（即“我要冻结这个特征提取层”），且不容易出错。

### 4. 什么时候必须用 `for` 循环？
虽然直接调用模块的方法更好，但在某些**精细化控制**的场景下，你仍然需要使用 `for` 循环。例如，你只想冻结某一层的偏置（bias），而不冻结权重（weight）：

```python
for name, param in model.features.named_parameters():
    if 'bias' in name:
        param.requires_grad_(False) # 只冻结偏置
    else:
        param.requires_grad_(True)  # 权重继续参与训练
```

<details>
<summary><b>💡 扩展：为什么要使用 requires_grad_(False)？（点击展开）</b></summary>

将参数的 `requires_grad` 设置为 `False` 的操作通常被称为**“冻结（Freezing）”**模型。这在深度学习中有以下常见用途：

1.  **迁移学习 / 微调 (Fine-tuning)**：当你使用一个预训练好的模型（如 ResNet, BERT）时，你可能希望保留其强大的特征提取能力（即 `model.features`），只训练最后新加的分类头（分类器）。冻结特征层可以防止预训练的权重被破坏。
2.  **节省显存和计算时间**：由于被冻结的参数不需要计算梯度（Gradient），反向传播（Backpropagation）时会跳过这些层，从而显著减少显存占用和加快训练速度。
3.  **注意区别于 `torch.no_grad()`**：
    *   `requires_grad_(False)` 是修改**模型参数自身的属性**，主要用于**部分训练**。
    *   `with torch.no_grad():` 是改变**环境上下文**，让接下来的所有计算都不记录计算图，主要用于**纯推理（Inference）/ 测试阶段**。
</details>