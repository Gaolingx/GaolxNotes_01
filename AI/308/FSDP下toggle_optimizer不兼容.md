# FSDP下toggle_optimizer不兼容

### 🧑‍💻 User

pytorchlightning toggle_optimizer / untoggle_optimizer 在 FSDP strategy 下能work吗？

---

### 🤖 Assistant

简短的回答是：**不能直接使用，或者说受到严格限制，官方不推荐在 FSDP 中使用 `toggle_optimizer` 的默认行为。**

在 PyTorch Lightning (PL) 中，`toggle_optimizer` 和 `untoggle_optimizer` 通常用于手动优化（Manual Optimization）场景（例如 GANs），其核心功能之一是动态切换参数的 `requires_grad` 属性，以防止在训练一个网络（如生成器）时计算另一个网络（如判别器）的梯度。

**然而，FSDP (Fully Sharded Data Parallel) 策略与这种动态切换 `requires_grad` 的机制存在根本性的冲突。**

以下是详细的原因分析和解决方案：

### 1. 核心冲突原因

#### A. 参数扁平化与分片 (Parameter Flattening & Sharding)
FSDP 的工作原理是将模型参数扁平化（Flatten）并分片（Shard）到各个 GPU 上。在 FSDP 初始化时，它会基于当前的参数状态构建 `FlatParamHandle`。
*   **冲突点：** FSDP 假设参数的可训练状态（`requires_grad`）是**静态**的。如果你在训练步骤中通过 `toggle_optimizer` 动态修改 `requires_grad`，这会破坏 FSDP 内部的视图和分片逻辑，导致运行时错误或梯度计算异常。

#### B. 梯度同步 (Gradient Synchronization)
`toggle_optimizer` 的设计初衷是为了在 DDP 中避免不必要的梯度同步和计算。
*   **冲突点：** FSDP 的梯度同步机制（Reduce-Scatter）是深度集成在 `backward` pass 中的。FSDP 需要知道确切的参与计算的参数集合。动态开关参数会干扰通信钩子（Communication Hooks）。

### 2. PyTorch Lightning 的行为

在较新的 PyTorch Lightning 版本中，如果你在使用 FSDP 策略时尝试使用 `toggle_optimizer`，通常会发生以下情况之一：

1.  **报错 (Error):** PL 可能会直接抛出 `MisconfigurationException`，告知你 FSDP 不支持修改 `requires_grad`。
2.  **静默失效 (Silent Failure):** 如果你强制使用手动优化，PL 可能会忽略 `requires_grad` 的切换。这意味着即使你调用了 toggle，所有的参数实际上仍然在计算梯度（只是未被该优化器 step 更新），这会导致显存并没有像预期那样节省，且计算量没有减少。

### 3. 正确的替代方案 (Workaround)

在 FSDP 下进行多优化器训练（如 GAN），你需要使用 **手动优化 (Manual Optimization)**，但**不要**去切换 `requires_grad`。

你应该直接获取优化器并执行 `step`，让 FSDP 处理所有的梯度计算。虽然这看起来比 toggle 浪费（因为计算了不需要更新的参数的梯度），但这是 FSDP 目前的限制。

#### 代码示例

**不推荐 (DDP 风格，FSDP 下会报错):**

```python
# 这种写法在 FSDP 下是有问题的
def training_step(self, batch, batch_idx):
    opt_g, opt_d = self.optimizers()
  
    # 训练生成器
    self.toggle_optimizer(opt_g) # <--- FSDP 可能会在这里崩溃或警告
    loss_g = self.compute_g_loss(batch)
    self.manual_backward(loss_g)
    opt_g.step()
    opt_g.zero_grad()
    self.untoggle_optimizer(opt_g)
  
    # ... 训练判别器 ...
```

**推荐 (FSDP 兼容写法):**

不要使用 `toggle_optimizer`。直接进行反向传播。虽然这意味判别器的参数在生成器 loss backward 时也会计算梯度（浪费算力），但你可以通过只调用特定优化器的 `step()` 来保证正确性。

```python
class FSDPGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False # 必须开启手动优化

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
      
        # --- 1. 训练生成器 ---
        # 不要 toggle，直接计算
        loss_g = self.compute_generator_loss(batch)
      
        opt_g.zero_grad()
        # FSDP 会处理所有参数的梯度同步
        self.manual_backward(loss_g) 
        opt_g.step()
      
        # --- 2. 训练判别器 ---
        loss_d = self.compute_discriminator_loss(batch)
      
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
```

### 4. 进阶：如何在使用 FSDP 时通过 `no_sync` 优化？

如果你真的想在 FSDP 下模拟 `toggle` 的某些行为（主要是为了累积梯度或减少通信），你应该使用 PyTorch Lightning 的 `self.trainer.strategy.block_backward_sync` (或者 PyTorch 原生的 `model.no_sync()`) 上下文管理器，而不是去改 `requires_grad`。

但是，对于 GAN 这种交替训练，最稳妥的方式依然是**保持所有参数 `requires_grad=True`**，仅控制哪个 `optimizer.step()` 被调用。

### 总结

*   **能不能用？** 不能按预期工作（主要指 `requires_grad` 切换功能）。
*   **为什么？** FSDP 需要静态的参数图结构来进行分片。
*   **怎么做？** 使用 Manual Optimization，但跳过 toggle/untoggle 步骤，接受全量梯度计算的开销，仅在 step 阶段区分更新对象。

---

### 🧑‍💻 User

为什么fsdp下不能随心所欲的设置requires_grad？

---

### 🤖 Assistant

这是一个非常深入且涉及 FSDP 核心机制的问题。

简单来说，FSDP 为了实现极致的显存优化和并行效率，采用了**“打包（Flattening）”**和**“静态图假设”**。当你试图动态修改 `requires_grad` 时，本质上是在破坏 FSDP 已经构建好的地基。

以下是具体的技术原因分析：

### 1. `FlatParameter` 机制：多个参数变一个 (The Flattening Problem)

这是最根本的原因。FSDP 并不是单独管理你模型中的 `weight` 和 `bias`。

*   **常规模式：** PyTorch 中，每个参数（如 `layer.weight`）都是独立的 Tensor。你可以把 Tensor A 设为 `requires_grad=False`，Tensor B 设为 `True`，互不干扰。
*   **FSDP 模式：** FSDP 会把一个模块（Module）内的所有参数**压扁（Flatten）**成一个巨大的 1D Tensor，称为 `FlatParameter`。

$$ \text{FlatParam} = [W_1, b_1, W_2, b_2, \dots] $$

**冲突点：**
在 PyTorch 中，`requires_grad` 是 **Tensor 级别** 的属性，而不是 **元素（Element）级别** 的属性。
*   你不能说：这个 1D Tensor 的前 100 个元素需要梯度，后 100 个元素不需要。
*   由于 $W_1$ 和 $W_2$ 现在物理上属于同一个 `FlatParameter`，它们必须共享相同的 `requires_grad` 状态。
*   如果你试图单独关闭 $W_1$ 的梯度，FSDP 无法仅在 `FlatParameter` 的一部分上关闭梯度追踪。

### 2. 预分配的通信缓冲区 (Communication Buffers)

FSDP 极其依赖**集体通信（Collective Communication）**，主要是 `AllGather`（在前向/反向传播前收集参数）和 `ReduceScatter`（在反向传播后同步梯度）。

为了速度，这些通信缓冲区通常是在 FSDP 初始化或第一次运行时**静态分配**的。

*   **冲突点：** FSDP 预期每次反向传播时，参与通信的梯度形状（Shape）和分片（Shard）大小是固定的。
*   如果你通过 `untoggle` 关闭了部分参数的梯度，PyTorch 的 Autograd 引擎就不会生成这些参数的梯度张量。
*   当 FSDP 尝试执行 `ReduceScatter` 来同步梯度时，它会发现预期的数据缺失了，或者形状对不上，从而导致通信死锁（Hang）或内存错误。

### 3. 分片逻辑的破坏 (Sharding View Mismatch)

FSDP 将 `FlatParameter` 切分成多个分片（Shards）分配给不同的 GPU。

*   GPU 0 负责维护 `FlatParam[0:N]`
*   GPU 1 负责维护 `FlatParam[N:2N]`

这个切分是纯数学上的索引切分，它不关心这些索引对应的是原来的 $W_1$ 还是 $W_2$。
如果你动态修改了原始参数的 `requires_grad`，你实际上是在逻辑层面上改变了模型的可训练结构。但是 FSDP 底层的物理分片逻辑并不知道这个改变。这会导致**视图（View）不一致**：原始参数认为自己不需要更新，但负责该分片的 FSDP 逻辑认为它需要参与梯度同步。

### 4. 混合精度与梯度缩放 (Mixed Precision & Grad Scaling)

在使用 FSDP 时，通常会配合 AMP（自动混合精度）。FSDP 需要管理 Master Weights（FP32）和 Training Weights（FP16/BF16）。

*   当 `requires_grad` 动态变化时，优化器（Optimizer）的状态管理会变得非常混乱。优化器需要知道哪些参数需要更新状态（Momentum, Variance）。
*   在 FSDP 下，优化器更新通常作用于分片后的参数。如果动态开关梯度，会导致优化器 step 时，部分分片有梯度，部分没有，这在分布式环境下极难协调，容易导致参数不同步。

### 总结与解决方案

**为什么不能随心所欲？**
因为 FSDP 牺牲了**动态性**（Dynamic flexibility）来换取**显存效率**（Memory efficiency）。它把零散的参数“焊死”成了一整块金属板。你没法在运行时只让这块金属板的“左上角”不导电。

**如果你必须冻结部分参数怎么办？**

1.  **静态冻结（推荐）：**
    在初始化 FSDP 包装器**之前**（例如在 `__init__` 或 `configure_sharded_model` 中），就设置好 `requires_grad=False`。
    *   FSDP 在构建 `FlatParam` 时，会检测到这些参数不需要梯度，它会智能地将它们排除在可训练的 `FlatParam` 之外，或者将它们放入单独的 `FlatParam` 组中。

2.  **动态“假”冻结（Workaround）：**
    如果必须在训练中途停止更新某部分（例如 GAN 的判别器/生成器交替）：
    *   保持 `requires_grad=True`。
    *   让 PyTorch 计算梯度（虽然浪费算力）。
    *   **仅控制优化器**：在 `optimizer.step()` 之前，手动将不想更新的参数的梯度置为 `None` 或者只对特定的优化器组调用 step。
    *   或者使用 `model.no_sync()` 上下文（如果显存允许），但这主要是为了梯度累积，而不是为了冻结参数。

对于 GAN 这类场景，最稳妥的做法就是**全量计算梯度，选择性更新参数**。