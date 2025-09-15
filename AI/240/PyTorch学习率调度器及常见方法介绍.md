# PyTorch学习率调度器及常见方法介绍

### 🧑‍💻 User

什么是学习率调度，介绍下pytorch中常见的学习率调度器以及相关的function。

---

### 🤖 Assistant

当然，这是一个关于学习率调度以及在PyTorch中如何实现的详细解释。

### 1. 什么是学习率调度？

**学习率（Learning Rate）** 是神经网络训练中最重要的超参数之一，它控制着模型根据损失梯度更新权重的步长。

- **学习率过大**：可能导致模型在最优值附近震荡，无法收敛，甚至发散。
- **学习率过小**：可能导致收敛速度极慢，容易陷入局部最优点。

**学习率调度（Learning Rate Scheduling）** 是一种在训练过程中**动态调整学习率**的技术。其核心思想是：并非在整个训练过程中都使用一个固定不变的学习率，而是根据预定义的计划或训练时的反馈（如验证集损失不再下降）来调整学习率。

**为什么需要学习率调度？**
1.  **更快收敛**：初期使用较大的学习率可以快速下降，后期使用较小的学习率可以精细调整，逼近最优解。
2.  **避免局部最优**：有时适当增大学习率可以帮助模型跳出尖锐的局部最优点，找到更平坦、泛化能力更强的区域。
3.  **提高模型性能**：这是最直接的目标，通过更有效的优化过程，最终获得在验证集上表现更好的模型。

---

### 2. PyTorch 中的学习率调度器 (`torch.optim.lr_scheduler`)

PyTorch 在 `torch.optim.lr_scheduler` 模块中提供了多种学习率调度器。这些调度器通常接收一个优化器 (`optimizer`) 实例作为参数，并通过其 `step()` 方法在每个 epoch 或每个 step 后更新学习率。

#### 常见的学习率调度器及其相关函数

以下是几个最常用和经典的学习率调度器：

##### a. `StepLR`
- **工作原理**：每经过固定的训练轮数（epoch），就将学习率乘以一个因子（gamma）。
- **公式**：`lr = initial_lr * gamma ^ (epoch // step_size)`
- **适用场景**：这是最基础的调度策略，适用于经验性调整。
- **代码示例**：
    ```python
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # 每经过 30 个 epoch，学习率变为原来的 0.1 倍
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(100):
        train(...)
        validate(...)
        scheduler.step() # 通常在每个 epoch 结束后调用
    ```

##### b. `MultiStepLR`
- **工作原理**：与 `StepLR` 类似，但不是在固定的间隔步数衰减，而是在指定的 epoch 里程碑（milestones）处衰减。
- **适用场景**：当你知道模型在训练过程的特定阶段需要调整学习率时（例如，在总 epoch 数的 50% 和 75% 时）。
- **代码示例**：
    ```python
    # 当训练到第 50 和第 80 个 epoch 时，各进行一次衰减
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    ```

##### c. `ExponentialLR`
- **工作原理**：每个 epoch 都会将学习率乘以一个衰减系数 gamma。
- **公式**：`lr = initial_lr * gamma ^ epoch`
- **适用场景**：需要非常平滑且持续衰减的学习率计划。
- **代码示例**：
    ```python
    # 每个 epoch 后，学习率都变为原来的 0.95 倍
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    ```

##### d. `CosineAnnealingLR`
- **工作原理**：使用余弦函数来调整学习率。学习率从初始值开始，先缓慢下降，然后在中间阶段快速下降，最后再缓慢接近最小值。
- **公式**：`η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(T_cur / T_max * π))`
    - `η_max` 是初始学习率。
    - `η_min` 是最小学习率。
    - `T_cur` 是当前 epoch 数。
    - `T_max` 是半个余弦周期的 epoch 数（通常设为总 epoch 数）。
- **适用场景**：在许多任务上表现优异，尤其是在计算机视觉领域。它提供了一个非常平滑的退火过程。
- **代码示例**：
    ```python
    # 假设总共训练 100 个 epoch，学习率将从 0.1 按余弦曲线衰减到 0
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    ```

##### e. `ReduceLROnPlateau`
- **工作原理**：**基于验证指标（而非 epoch 数）** 的动态调度器。当某个指标（如验证损失）停止改善超过一个“耐心”（patience）值时，就会降低学习率。
- **适用场景**：非常实用和强大，因为你不需要预先知道何时需要衰减。它让训练过程自动决定何时调整学习率。
- **重要参数**：
    - `mode`：`min`（监控指标是否不再下降）或 `max`（监控指标是否不再上升）。
    - `factor`：衰减因子。
    - `patience`：能容忍指标没有改善的 epoch 数。
    - `threshold`：衡量新的最佳值的阈值。
- **代码示例**：
    ```python
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    for epoch in range(100):
        train(...)
        val_loss = validate(...)

        # 注意：这里需要将监控的指标传入 step() 方法
        scheduler.step(val_loss)
    ```

##### f. `OneCycleLR`
- **工作原理**：一种非常流行的策略，它在一个周期内先让学习率从较低值线性增加到很高的值，然后再线性下降到一个比初始值更低的值。它通常与动量（momentum）的反向周期结合使用。
- **适用场景**：训练速度极快，并且通常能达到很好的性能。被称为“超级收敛”（Super-Convergence）策略。适用于 epoch 数相对较少的训练。
- **代码示例**：
    ```python
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                       max_lr=0.1,        # 峰值学习率
                                       steps_per_epoch=len(train_loader),
                                       epochs=10)
    # 注意：OneCycleLR 通常需要在每个 BATCH 后调用 .step()
    for epoch in range(10):
        for batch in train_loader:
            train_batch(...)
            scheduler.step() # 在每个 batch 后更新
    ```

---

### 3. 核心函数与方法

所有调度器都共享一些核心方法：

- `__init__(optimizer, ...)`：构造函数，第一个参数必须是优化器实例，其他参数因调度器而异。
- `step()`：更新学习率。对于大多数调度器，这是在每个 **epoch** 后调用。对于 `OneCycleLR` 或 `CyclicLR` 等，则需要在每个 **batch** 后调用。
    - **特别注意**：`ReduceLROnPlateau.step()` 需要一个指标值作为参数。
- `get_last_lr()`：返回一个列表，包含所有参数组（parameter groups）在上一次 `step()` 之后使用的学习率。
- `state_dict()` 和 `load_state_dict()`：用于保存和加载调度器的状态，这对于中断后继续训练至关重要。

### 总结与选择

| 调度器 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **StepLR** / **MultiStepLR** | 简单，直观 | 需要预先定义好衰减时机 | 基础任务，经验性调整 |
| **ExponentialLR** | 衰减非常平滑 | 可能衰减过快 | 需要持续平滑衰减的场景 |
| **CosineAnnealingLR** | 理论扎实，效果好，平滑 | 周期需要预先设定 | 图像分类等常见任务，首选之一 |
| **ReduceLROnPlateau** | 自适应，无需预先设定 | 需要验证集反馈 | 实用主义，不知道何时衰减时就用它 |
| **OneCycleLR** | 训练快，性能好 | 参数较多，需调整 | 追求快速收敛和最佳性能 |

**一般建议**：
- 从 `CosineAnnealingLR` 或 `ReduceLROnPlateau` 开始尝试。
- 如果追求快速训练和非常好的结果，可以研究并调试 `OneCycleLR`。
- 始终使用 **TensorBoard** 或 **Weights & Biases** 等工具绘制学习率曲线，直观地了解调度器是如何工作的。

---

### 🧑‍💻 User

假如我需要自己设计一个学习率调度器，要怎么做？

---

### 🤖 Assistant

当然！在PyTorch中自己设计一个学习率调度器是一个非常常见的需求，因为研究或项目常常需要特定的学习率变化策略。PyTorch让这个过程变得非常简单和灵活。

你有两种主要的方法来实现自定义学习率调度器：

1.  **LambdaLR**：使用匿名函数快速实现简单规则。
2.  **继承 `_LRScheduler` 基类**：实现功能完整、复杂的自定义调度器。

---

### 方法一：使用 `LambdaLR` （快速简单）

`LambdaLR` 允许你通过一个简单的Lambda函数（或任何可调用对象）来定义学习率相对于初始学习率的变化倍数。

**步骤：**
1.  定义一个函数，该函数以当前 `epoch` 为输入，返回一个**乘数因子（ multiplier ）**。
2.  将这个函数传递给 `LambdaLR` 调度器。

**示例1：实现一个线性衰减调度器**

假设你想在每个epoch后，将学习率线性衰减到第100个epoch时为0。

```python
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.SGD(model.parameters(), lr=0.1)

# 1. 定义lambda函数
# lr_lambda = lambda epoch: 1. - epoch / 100
def lr_lambda(epoch):
    return 1. - epoch / 100 # 从1.0线性减少到0.0

# 2. 创建调度器
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step() # 更新学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch: {epoch}, Learning Rate: {current_lr}')
```

**示例2：实现一个带热重启的余弦退火（SGDR的简化版）**

```python
import math

def cosine_annealing_with_warm_restart(epoch, T_0=10, T_mult=2, eta_min=0):
    """
    T_0: 第一次重启的周期
    T_mult: 重启后周期变大的倍数（通常>=1）
    eta_min: 最小学习率
    """
    # 计算当前是第几个重启周期
    T_cur = epoch
    T_i = T_0
    while T_cur >= T_i:
        T_cur -= T_i
        T_i *= T_mult
    # 余弦公式，返回值是乘数因子
    return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i))

optimizer = optim.SGD(model.parameters(), lr=0.1)
# 使用lambda包装一下，因为lr_lambda函数只接收一个参数epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: cosine_annealing_with_warm_restart(e, T_0=10))
```

**优点**：快速、代码简洁。
**缺点**：功能受限，难以实现依赖于上一步状态（如上一步的学习率）的复杂逻辑。

---

### 方法二：继承 `_LRScheduler` 基类（功能完整）

这是最强大和推荐的方法。你需要创建一个继承自 `torch.optim.lr_scheduler._LRScheduler` 的类。

**核心步骤：**
1.  **继承 `_LRScheduler`**。
2.  **实现 `__init__` 方法**：调用父类的 `__init__` 方法，并初始化你自己需要的参数。
3.  **重写 `get_lr` 方法**：这是核心！该方法返回一个**列表**，包含所有参数组（parameter groups）**下一步**应该使用的学习率。你在这里实现你的计算逻辑。**注意：** 它计算的是学习率值，而不是乘数因子。

**示例：实现一个指数衰减调度器（`ExponentialLR` 的手动实现）**

```python
from torch.optim.lr_scheduler import _LRScheduler

class MyExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        # 初始化自定义参数
        self.gamma = gamma
        # 调用父类初始化方法，必须做！
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # 如果 last_epoch = -1, 说明是初始阶段，返回初始学习率
        if self.last_epoch == -1:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # 核心计算：新的学习率 = 初始学习率 * (gamma ^ last_epoch)
        # 注意：self.base_lrs 是父类属性，保存了每个参数组的初始学习率
        return [base_lr * (self.gamma ** self.last_epoch)
                for base_lr in self.base_lrs]

# 使用自定义调度器
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = MyExponentialLR(optimizer, gamma=0.95)

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

**示例：实现一个更复杂的自定义调度器（例如，前10个epoch预热，之后余弦衰减）**

```python
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            return [base_lr * (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # 余弦衰减阶段
            current_epoch = self.last_epoch - self.warmup_epochs
            total_cosine_epochs = self.max_epochs - self.warmup_epochs
            # 余弦公式
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * current_epoch / total_cosine_epochs)) / 2
                    for base_lr in self.base_lrs]

# 使用
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = WarmupCosineLR(optimizer, warmup_epochs=5, max_epochs=100, eta_min=1e-5)
```

**优点**：
*   **功能完整**：可以实现任何你能想到的逻辑。
*   **状态管理**：基类帮你处理了 `state_dict` 和 `load_state_dict`，确保断点续训正常工作。
*   **标准接口**：和使用内置调度器一模一样（`.step()`, `.get_last_lr()`）。

**关键点**：
*   始终记得在 `__init__` 中调用 `super().__init__(optimizer, last_epoch, verbose)`。
*   `self.base_lrs` 是初始学习率列表。
*   `self.last_epoch` 记录已经进行了多少个 `step()`（通常是epoch数）。
*   `get_lr()` 返回的是一个列表，长度等于 `optimizer.param_groups` 的数量。

### 总结与建议

| 方法 | 适用场景 | 难度 |
| :--- | :--- | :--- |
| **`LambdaLR`** | 简单的、无状态的、基于epoch数的规则 | 低 |
| **继承 `_LRScheduler`** | 复杂的、有状态的、自适应的规则 | 中到高 |

*   对于绝大多数需求，**方法二（继承 `_LRScheduler`）** 是更专业和可靠的选择。
*   在设计时，**务必使用 TensorBoard 或 matplotlib 绘制出你的学习率变化曲线**，以确保它符合你的预期。
*   参考PyTorch官方源码中其他调度器（如 `CosineAnnealingLR`）的实现，这是最好的学习材料。