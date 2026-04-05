# DDP中Rank0操作需注意

### 🧑‍💻 User

为什么pytorch lightning使用ddp需要注意部分操作rank0问题？

---

### 🤖 Assistant

在 PyTorch Lightning (PTL) 中使用 Distributed Data Parallel (DDP) 时，必须特别注意 `rank 0`（即主进程）的操作。这主要是由 **DDP 的底层工作原理**决定的。

简单来说，当你使用 DDP 并在 $N$ 张显卡上训练时，系统实际上启动了 **$N$ 个完全独立的 Python 进程**。这 $N$ 个进程会同时、并行地执行你写的同一份代码。

如果不对特定的操作进行 `rank 0` 隔离，就会导致严重的资源冲突、死锁或逻辑错误。以下是需要特别注意 `rank 0` 问题的核心原因及常见场景：

### 1. 避免文件系统冲突（Race Conditions）
当多个进程同时尝试读写同一个文件时，会发生文件损坏或程序崩溃。
* **数据下载与预处理：** 如果你的代码中包含 `download_dataset()`。在 DDP 模式下，$N$ 个进程会同时尝试下载数据并写入同一个文件夹。这必定会导致下载中断或文件损坏。
* **模型保存（Checkpointing）：** 保存模型权重时，实际上所有 $N$ 个进程中的模型参数在每一步梯度同步后都是完全一致的。只需要让 `rank 0` 保存一次即可。如果 $N$ 个进程同时写同一个 `.ckpt` 文件，会导致 I/O 堵塞甚至文件损坏。

### 2. 避免终端日志爆炸
如果你在代码里写了一句 `print("Epoch finished")`，在 8 卡 DDP 训练中，你的终端会瞬间打印 8 次完全一样的信息。这会极大地降低日志的可读性。对于 TensorBoard 或 Wandb 等日志记录工具，如果所有进程都去初始化并写入同一个项目，也会导致图表混乱或报错。

### 3. 避免进程死锁（Deadlocks）
DDP 的核心是集合通信（Collective Communications，如 `all_reduce`）。在反向传播计算梯度时，所有进程必须到达同一个同步点，互相交换梯度数据。
Total Batch Size 的计算公式为：
$$ \text{Total Batch Size} = N \times \text{Batch Size per GPU} $$
如果你的代码写了类似这样的逻辑：
```python
if self.global_rank == 0:
    # 只有 rank 0 停下来进行某些极其耗时的计算，或者等待用户输入
    do_heavy_computation()

# ... 后续的训练代码 ...
loss.backward() # 同步点
```
此时，`rank 1` 到 `rank N-1` 会迅速到达 `loss.backward()` 并卡在那里等待 `rank 0` 的梯度。而 `rank 0` 还在做耗时计算。这就会导致**死锁（Deadlock）**，整个训练程序永远卡住，最终触发超时错误（Timeout）。

---

<details open>
<summary><b>💡 PyTorch Lightning 是如何优雅解决这些问题的（最佳实践）</b></summary>
<br>

PyTorch Lightning 已经为你封装好了很多工具，让你能轻松处理 `rank 0` 的隔离问题：

#### 1. 使用 `@rank_zero_only` 装饰器
如果你写了自定义的打印函数、日志初始化函数或文件保存逻辑，加上这个装饰器，PTL 就会自动保证它只在 `rank 0` 运行。
```python
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class MyModel(pl.LightningModule):
    @rank_zero_only
    def my_custom_print(self, msg):
        print(f"[{self.global_rank}] {msg}")
      
    def training_step(self, batch, batch_idx):
        # 即使这里被所有进程调用，实际上只有 rank 0 会执行打印
        self.my_custom_print("Training step executed!") 
```

#### 2. 区分 `prepare_data` 和 `setup`
在 `LightningDataModule` 中，PTL 严格区分了这两个生命周期：
* `prepare_data()`: **永远只在 `rank 0` 执行一次**。你应该在这里进行数据集下载、解压、或者全局的 Tokenize 缓存操作。
* `setup()`: **在每个进程上都会执行**。你应该在这里进行数据集的划分（Train/Val/Test split）并构建 `Dataset` 对象，因为每个进程都需要属于自己的 `Dataset` 实例放入内存。

#### 3. 日志记录的控制
在使用 `self.log()` 时，对于某些非标量或者只需要记录一次的信息，可以使用参数控制：
```python
# 只让 rank 0 记录这个指标，避免其他进程的冗余计算/传输
self.log("val_loss", loss, rank_zero_only=True)

# 如果你需要所有进程的指标取平均值，应该使用 sync_dist
self.log("train_loss", loss, sync_dist=True) 
```

#### 4. 内置属性检查
如果你需要写复杂的控制流，可以直接调用内置属性：
```python
if self.trainer.is_global_zero:
    # 仅限 rank 0 的操作
    pass
```
</details>

<details>
<summary><b>⚠️ 进阶注意事项：Rank 0 操作与 Barrier 机制</b></summary>
<br>

如果 `rank 0` 执行了比较耗时的 I/O 操作（比如在 `prepare_data` 中下载了 10GB 的数据集），其他进程不能直接往下走，否则它们去读取数据时会发现文件不存在。

PTL 在底层自动为你处理了这个问题。在执行完 `prepare_data` 后，PTL 内部会调用 `torch.distributed.barrier()`。
* `barrier()` 的作用相当于一堵墙，所有执行得快的进程（`rank 1` 到 `rank N-1`）到达这堵墙时会停下来等待。
* 直到最慢的进程（正在下载数据的 `rank 0`）也到达这堵墙。
* 墙倒塌，所有进程再同时继续往下执行 `setup()` 等后续操作。

**注意：** 如果你在自定义的 `rank 0` 独占代码中做了耗时操作，且后续代码依赖该操作的结果，你也必须手动引入 `barrier`，否则会导致其他进程因找不到数据而崩溃。
</details>

---

### 🧑‍💻 User

如果涉及手动优化+多优化器有没有需要额外注意地方

---

### 🤖 Assistant

在 PyTorch Lightning 中，当你同时使用 **手动优化（`automatic_optimization = False`）**、**多优化器（Multi-Optimizers）** 以及 **DDP** 时，你已经进入了分布式训练中最容易出现 Bug 的“深水区”。

在这种场景下，除了上一条提到的常规 `rank 0` 问题外，**最致命的风险是“进程死锁”（Deadlock）和“计算图状态不一致”**。

以下是你在编写代码时必须额外注意的核心问题及解决方案：

### 1. 绝对禁止“基于单卡状态”的条件分支（死锁的万恶之源）

在使用多优化器时（例如 GAN 中的生成器 $G$ 和判别器 $D$），很多开发者会写条件控制流，比如：“当 $D$ 的 Loss 小于某个阈值时，多训练几次 $G$”。

在 DDP 中，如果这个条件**没有在所有卡上严格同步**，就会导致毁灭性的死锁。

**❌ 错误示例（导致死锁）：**
```python
def training_step(self, batch, batch_idx):
    opt_g, opt_d = self.optimizers()
    loss_d = self.compute_d_loss(batch) # 假设卡0算出来是 0.4，卡1算出来是 0.6
  
    # 灾难发生：卡0 认为 loss_d < 0.5，跳过了 D 的更新，去更新 G
    # 卡1 认为 loss_d >= 0.5，停在这里尝试反向传播 D，并等待卡0的梯度同步（all_reduce）
    # 结果：卡1等卡0的 D 梯度，卡0等卡1的 G 梯度 -> 永远卡死！
    if loss_d >= 0.5: 
        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
    else:
        loss_g = self.compute_g_loss(batch)
        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
```

**✅ 正确做法：全局同步条件判断**
所有的进程必须**共同决定**当前这步该更新哪一个优化器。你需要使用 `all_reduce` 或 `all_gather` 获取全局指标。
```python
def training_step(self, batch, batch_idx):
    opt_g, opt_d = self.optimizers()
    loss_d = self.compute_d_loss(batch)
  
    # 将所有卡上的 loss_d 汇总求平均，确保所有进程拿到同一个数字
    global_loss_d = self.all_gather(loss_d).mean() 
  
    # 现在所有进程的条件判断绝对一致
    if global_loss_d >= 0.5: 
        self.manual_backward(loss_d) # 所有卡一起反向传播 D
        opt_d.step()
        opt_d.zero_grad()
    else:
        loss_g = self.compute_g_loss(batch)
        self.manual_backward(loss_g) # 所有卡一起反向传播 G
        opt_g.step()
        opt_g.zero_grad()
```

---

<details open>
<summary><b>💡 2. 必须使用 `toggle_optimizer` 冻结/解冻网络</b></summary>
<br>

在多优化器交替训练时，为了防止不需要更新的网络部分计算梯度并触发 DDP 同步，我们通常需要设置 `requires_grad = False`。

在 DDP 环境下，如果你手动去写 `for param in net.parameters(): param.requires_grad = False`，很容易破坏 DDP `DistributedDataParallel` 包装器的内部状态，导致梯度同步报错。

**解决方案：** 
PyTorch Lightning 提供了 `toggle_optimizer` 和 `untoggle_optimizer`，它不仅会自动切换 `requires_grad`，还会妥善处理 DDP 下的同步和性能优化。

```python
def training_step(self, batch, batch_idx):
    opt_a, opt_b = self.optimizers()

    # --- 训练网络 A ---
    self.toggle_optimizer(opt_a) # 自动解冻 A 的参数，冻结 B 的参数
    loss_a = self.compute_loss_a(batch)
    self.manual_backward(loss_a)
    opt_a.step()
    opt_a.zero_grad()
    self.untoggle_optimizer(opt_a)

    # --- 训练网络 B ---
    self.toggle_optimizer(opt_b) # 自动解冻 B 的参数，冻结 A 的参数
    loss_b = self.compute_loss_b(batch)
    self.manual_backward(loss_b)
    opt_b.step()
    opt_b.zero_grad()
    self.untoggle_optimizer(opt_b)
```
</details>

<details>
<summary><b>⚠️ 3. 梯度累加（Gradient Accumulation）的特殊处理</b></summary>
<br>

在自动优化模式下，Lightning 会自动处理 DDP 环境下的梯度累加（使用 `no_sync` 避免多余的跨卡通信）。但在**手动优化**中，你必须自己控制这一逻辑，否则会在未进行 `step()` 的步骤中发生大量无用的跨卡通信，严重拖慢训练速度。

对于需要累加 $K$ 步的场景，推荐写法如下：

```python
def training_step(self, batch, batch_idx):
    opt = self.optimizers()
  
    # Lightning 提供的判断函数：当前是否属于累加阶段（不需要跨卡同步）
    is_accumulating = (batch_idx + 1) % self.trainer.accumulate_grad_batches != 0
  
    # 缩放 loss (如果 accumulate_grad_batches > 1)
    loss = self.compute_loss(batch) / self.trainer.accumulate_grad_batches

    # manual_backward 内部能识别 accumulate_grad_batches 并自动应用 DDP no_sync
    self.manual_backward(loss)

    if not is_accumulating:
        opt.step()
        opt.zero_grad()
```
*注：即使是多优化器，也要确保在 `not is_accumulating` 触发时，所有进程都共同执行了 `step`。*
</details>

<details>
<summary><b>📊 4. 日志记录的同步策略</b></summary>
<br>

在多优化器交替训练中，你可能会在一个 `training_step` 里面产生好几个 loss（比如 `loss_g` 和 `loss_d`）。

如果你要在进度条或 TensorBoard 中显示这些指标，**请不要只记录单卡的数值**，也不要让 `rank 0` 特立独行。应该使用 `sync_dist=True` 确保日志反映的是全局真实情况：

```python
# 每次算完一个优化器的 loss 就可以直接 log
self.log("g_loss", loss_g, prog_bar=True, sync_dist=True)
self.log("d_loss", loss_d, prog_bar=True, sync_dist=True)
```
这样 Lightning 会在底层自动做 `all_reduce` 求均值，且只有 `rank 0` 会负责将最终求出的均值写入 TensorBoard，既保证了准确性，又避免了文件冲突。
</details>

### 总结 Checklist：
当你写完手动优化 + 多优化器的 DDP 代码后，请在脑海中问自己三个问题：
1. **控制流同步：** 决定“走哪条 `if/else`”、“跳过哪个优化器”的变量，是否使用了 `all_gather` 或确保基于 `batch_idx`？
2. **状态隔离：** 冻结网络计算图是否使用了官方的 `self.toggle_optimizer()`？
3. **反向传播：** 是否所有参与计算图的卡，都到达了同一个 `self.manual_backward()` 并且没有 `rank 0` 独自在等待或提前跑路？

---

### 🧑‍💻 User

DDP 下 validation_epoch_end 的 evaluation / metrics 聚合有没有坑？

---

### 🤖 Assistant

这是一个非常专业且切中痛点的问题。**答案是：有大坑。**

如果你在 DDP 模式下的 `validation_epoch_end`（新版 PTL 推荐使用 `on_validation_epoch_end`）处理不当，你看到的验证集准确率可能只是 **Rank 0 这一张卡上数据的准确率**，或者是**计算方式错误的全局平均值**。

以下是三大核心“坑”及避坑指南：

### 1. “数据填充”陷阱 (The Padding Trap)

这是 DDP 验证中最隐蔽的坑。

**原理：**
DDP 依赖 `DistributedSampler` 来分配数据。为了保证每张卡分配到的 batch 数量完全一致（方便同步），如果验证集长度不能被 $N \times \text{Batch Size}$ 整除，Sampler 会**重复采样（Duplicate）**部分数据来填补空缺。

**后果：**
如果你简单地将所有卡计算出的 Metric 进行平均（Mean），那些**被重复采样的数据权重会变大**，导致最终指标虚高或虚低。

**✅ 解决方案：使用 TorchMetrics**
不要手动去算 `correct / total`。PyTorch Lightning 官方推荐的 `TorchMetrics` 库已经完美解决了这个问题。它内部维护了状态，能够在 DDP 环境下正确处理重复数据（去重或正确加权），并在 `compute()` 时自动进行进程同步。

```python
# 推荐做法
from torchmetrics import Accuracy

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 指标对象会自动处理 DDP 同步
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        # 只需要在这一步调用 update，不要在这里 log 结果
        self.val_acc.update(preds, y)
      
    def on_validation_epoch_end(self):
        # compute() 会自动收集所有卡的统计数据，算出精确的全局指标
        total_acc = self.val_acc.compute()
        self.log("val_acc", total_acc)
        self.val_acc.reset()
```

---

### 2. `self.log` 的 `sync_dist` 陷阱

如果你没有使用 `TorchMetrics`，而是想手动记录 Loss 或简单的准确率，必须注意 `self.log` 的默认行为。

**默认行为：**
在 `validation_step` 中调用 `self.log` 时，默认 `on_step=False, on_epoch=True`。这意味着 PTL 会在每个 step 累积结果，并在 epoch 结束时自动计算平均值。

**❌ 坑点：**
如果不加 `sync_dist=True`，Rank 0 只会算出 Rank 0 这张卡上所有 batch 的平均 Loss，Rank 1 算 Rank 1 的。最终在 TensorBoard 里，你看到的只是 Rank 0 的局部 Loss，而不是全局 Loss。

**✅ 解决方案：显式开启同步**
```python
def validation_step(self, batch, batch_idx):
    loss = ...
    # sync_dist=True 会让 PTL 在 epoch 结束时调用 dist.all_reduce
    # 将所有卡的平均值再取一次平均（或者加权平均）
    self.log("val_loss", loss, sync_dist=True) 
```

---

### 3. 全局数据收集陷阱 (List Gathering)

有些复杂的评估（如生成任务的 BLEU、目标检测的 mAP），无法简单通过“求平均”得到。你需要拿到验证集**所有样本的预测结果**，拼在一起算一次。

**❌ 常见的 OOM 写法：**
```python
# 这是一个非常危险的写法！
class MyModel(pl.LightningModule):
    def __init__(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        # 危险：将 Tensor 留在了 GPU 上并存入列表
        self.validation_step_outputs.append(pred) 

    def on_validation_epoch_end(self):
        # 此时显存可能已经爆了
        all_preds = torch.cat(self.validation_step_outputs)
        # 且这里只有当前卡的数据，不是全局数据！
        calculate_map(all_preds) 
```

**✅ 解决方案：`all_gather` + `cpu()`**

如果确实需要收集所有预测结果，必须手动使用 `all_gather`。

<details open>
<summary><b>🛠️ 最佳实践代码模板</b></summary>
<br>

```python
class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 用于暂存每个 step 的输出
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        preds = self(batch)
        # 1. 极其重要：转到 CPU 以节省显存
        self.validation_step_outputs.append(preds.cpu())

    def on_validation_epoch_end(self):
        # 1. 拼接当前卡的所有预测结果
        local_preds = torch.cat(self.validation_step_outputs)
      
        # 2. 收集所有卡的结果 (DDP 关键步骤)
        # self.all_gather 会返回一个列表，形状是 [World_Size, N_samples, ...]
        if self.trainer.num_devices > 1:
            global_preds = self.all_gather(local_preds)
            # 展平 tensor: [World_Size, N, ...] -> [Total_N, ...]
            global_preds = global_preds.view(-1, *global_preds.shape[2:])
        else:
            global_preds = local_preds

        # 3. 清空缓存列表，防止下一个 epoch 累加
        self.validation_step_outputs.clear()

        # 4. 只有 Rank 0 进行计算和打印，避免重复计算
        if self.trainer.is_global_zero:
            score = complex_metric_calculation(global_preds)
            self.log("val_complex_metric", score, rank_zero_only=True)
```
</details>

### 总结

1.  **能用 `TorchMetrics` 就一定用**，它帮你屏蔽了 99% 的 DDP 聚合逻辑错误。
2.  简单的 Loss 平均，务必在 `self.log` 中加上 `sync_dist=True`。
3.  复杂的全量数据评估，必须手动 `all_gather`，并注意**先转 CPU** 防止显存爆炸，最后用 `rank_zero_only` 只计算一次。