# FSDP在PTL中自动分片无需手动干预

### 🧑‍💻 User

为什么ptl自动优化中，fsdp基本不需要考虑太多参数分片问题？

---

### 🤖 Assistant

在 PyTorch Lightning (PTL) 中使用 FSDP (Fully Sharded Data Parallel) 进行自动优化时，开发者基本上不需要像使用模型并行（Model Parallelism）或张量并行（Tensor Parallelism）那样去手动处理参数的分片（Sharding）逻辑。

这主要得益于 **FSDP 本身的底层设计哲学** 以及 **PTL 提供的高度抽象封装**。以下是具体原因的详细解析：

### 1. FSDP 的本质：内存切分，而非逻辑切分
FSDP 实际上是 **数据并行（Data Parallel）** 的一种变体，而不是真正的模型并行。
* **在逻辑上**：你的模型代码完全不需要改变。在进行前向传播（Forward）和反向传播（Backward）的那一瞬间，当前计算层（Layer）在 GPU 上是**完整**的。
* **在物理上**：FSDP 将模型参数、梯度和优化器状态打平（Flatten）成一维张量，然后均匀地切分到 $N$ 张显卡上。
由于逻辑上模型没有被切分，所以你不需要像张量并行（如 Megatron-LM）那样去重写 `nn.Linear` 并在代码中手动指定 `gather` 或 `scatter` 操作。

### 2. PTL 的自动包装策略 (Auto-Wrap Policy)
如果要把整个大模型作为一个整体进行 FSDP 切分，通信开销会非常大。为了平衡显存和通信，FSDP 需要将模型按子模块（Submodules，例如 Transformer 的 Block）进行包装（Wrap）。
在原生的 PyTorch 中，这需要写繁琐的代码。但在 PTL 中，它会自动帮你处理：
* **基于大小的自动包装**：PTL 默认支持 `size_based_auto_wrap_policy`。只要一个子模块的参数量超过设定的阈值（例如 100M），PTL 就会自动在它外面套一层 FSDP 包装。
* **基于 Transformer 层的包装**：对于大语言模型，PTL 允许你直接传入指定的 Layer 类（如 `TransformerBlock`），框架会自动找到这些层并进行最优切分。

```python
# 在 PTL 中，你只需要通过 Strategy 传入参数即可，完全不需要改模型代码
from lightning.pytorch.strategies import FSDPStrategy

strategy = FSDPStrategy(
    auto_wrap_policy={
        "module_classes": {MyTransformerBlock}, # 自动按 Block 分片
    },
    sharding_strategy="FULL_SHARD"
)
trainer = Trainer(strategy=strategy, devices=4)
```

### 3. 动态的 All-Gather 和 Reduce-Scatter 机制
FSDP 将参数分片问题转化为**动态的通信问题**。PTL 结合 PyTorch 的底层 Hook 机制，在后台自动完成了所有工作：
* **前向传播前**：触发 `All-Gather`，从其他 GPU 拉取当前 Layer 缺失的参数碎片，重组完整参数。
* **前向传播后**：立刻释放（Free）其他 GPU 的参数碎片，只保留自己负责的那一部分。
* **反向传播时**：再次 `All-Gather` 组装参数计算梯度，计算完成后立刻触发 `Reduce-Scatter`，同步并切分梯度。

### 4. 优化器状态自动接管
在深度学习中，显存占用大头往往是优化器状态（例如 Adam 需要保存一阶动量和二阶动量，占用的显存是参数本身的 2 倍）。
在 PTL 的自动优化流程中，`configure_optimizers` 会在模型被 FSDP 包装**之后**才被调用。这意味着优化器在初始化时，看到的已经是被切分好的 `FlatParameter`。因此，优化器状态天生就是分片好的，完全不需要用户干预。

<details>
<summary><b>💡 深入底层：PyTorch 的 FlatParameter 机制 (点击展开)</b></summary>

FSDP 不需要用户考虑分片，是因为它使用了一个叫做 `FlatParameter` 的黑科技。
1. 当 PTL 用 FSDP 包装你的模型时，它会遍历你模型中的所有 `nn.Parameter`。
2. 它将这些形状各异的张量全部拉平（Flatten）成一个巨大的一维数组。
3. 然后，它像切蛋糕一样，把这个一维数组平均切成 $N$ 份，分给 $N$ 张 GPU。
4. FSDP 会记录每个原始张量在这个一维数组中的**视图（View）和偏移量（Offset）**。
5. 当需要计算时，它把一维数组拼回来，然后利用记录的视图，把 1D 张量重新 Reshape 成原来的多维张量供前向计算使用。

这个过程对用户和原生的 PyTorch 算子是完全透明的。
</details>

---

### 什么时候你**确实需要**考虑分片问题？
虽然 PTL 让 FSDP 变得接近“傻瓜式”，但在以下少数高级场景中，你仍然需要介入：
1. **超大规模模型（如千亿参数）**：单层（一个 Transformer Block）的参数量如果连单张 GPU 的显存都放不下，FSDP 就无能为力了。这时必须结合**张量并行 (TP)** 或使用 PyTorch 的 `FullyShardedDataParallel` 的高级特性。
2. **自定义的奇特网络结构**：如果你的网络不是典型的层级结构（比如有非常复杂的跨层连接），自动包装策略可能会导致频繁的通信，此时你需要手动编写 `custom_auto_wrap_policy` 来优化通信效率。
3. **模型保存与加载 (Checkpointing)**：保存模型时，权重是分片在各个 GPU 上的。虽然 PTL 的 `Trainer.save_checkpoint()` 自动处理了合并（或分布式保存），但如果你要将模型导出给不支持 FSDP 的推理框架（如 vLLM/HuggingFace），需要注意 PTL 提供的 consolidate 权重转换步骤。

**总结**：PTL 中的 FSDP 之所以省心，是因为它在**数学逻辑上保持了数据并行**的特性，把复杂的“参数拆分、通信同步、显存释放”全部下放到了框架的底层 Hook 和自动包装策略中。你只需要专心写单机版的模型结构即可。

---

### 🧑‍💻 User

如果需要访问模型params（例如vae-gan disc adapt weight），会受限吗？

---

### 🤖 Assistant

**是的，会受到非常大的限制。** 

如果在 PTL 中使用了 FSDP，你**不能**像以前那样在 `training_step` 或其他地方随意通过 `model.module.weight` 或 `model.module.weight.grad` 来访问模型参数。

特别是像你提到的 **VAE-GAN 中的判别器自适应权重 (Discriminator Adaptive Weight)**，这种操作在 FSDP 下如果不加处理，**100% 会报错**。

### 为什么会受限？

在 VAE-GAN（例如 Taming Transformers 或 LDM 的底层 VAE）中，计算自适应权重通常需要用到解码器最后一层的梯度，公式类似：
$$ \lambda = \frac{\|\nabla_{w} L_{rec}\|}{\|\nabla_{w} L_{GAN}\|} $$
你需要分别对重构损失和 GAN 损失求导，然后获取最后一层（如 `decoder.conv_out.weight`）的 `.grad` 属性来进行计算。

但是在 FSDP 中：
1. **参数被切分了**：在正常的 `training_step` 代码块中（即不在该层的前向/反向传播瞬间），你访问到的 `weight` 只是一个被拉平且切分的 1D 张量（`FlatParameter`），它的形状和原始形状完全对不上。
2. **梯度被切分并释放了**：FSDP 为了省显存，在反向传播计算完某一层的梯度后，会立刻触发 `Reduce-Scatter` 将梯度切分，并清理完整的梯度张量。你拿不到完整的 `.grad` 来计算范数（Norm）。

---

### 如何在 FSDP 中解决这个问题？

针对这种需要显式访问参数或梯度的需求，有以下几种解决方案：

#### 方案一：将需要访问的层排除在 FSDP 包装之外（最推荐）

对于 VAE-GAN 来说，需要计算梯度的通常只是解码器的**最后一层**（比如一个 `nn.Conv2d`）。这一层的参数量非常小，完全没有必要对其进行 FSDP 切分。

**做法**：在定义 FSDP 的 `auto_wrap_policy` 时，确保这一层**不被** FSDP 包装。这样，这层就会退化为普通的 DDP 行为，它的参数和梯度在任何时候都是完整的。

```python
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

# 自定义包装策略：不包装最后一层
def my_wrap_policy(module, recurse, nonwrapped_numel):
    # 假设你的最后一层名字叫 conv_out
    if module.__class__.__name__ == 'Conv2d' and getattr(module, "is_last_layer", False):
        return False
    # 其他层按照常规逻辑包装（比如参数量大于某个阈值）
    return nonwrapped_numel >= 100000 

strategy = FSDPStrategy(
    auto_wrap_policy=my_wrap_policy
)
```
*优点：代码改动最小，直接兼容原有的自适应权重计算逻辑。*

#### 方案二：使用 `summon_full_params` 唤回完整参数

如果你必须访问被 FSDP 包装的层的参数，PyTorch 提供了 `summon_full_params` 上下文管理器。它会在调用时触发 `All-Gather`，把其他 GPU 上的碎片拉过来拼成完整的参数。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def training_step(self, batch, batch_idx):
    # ... 前向传播计算 loss_rec 和 loss_gan ...
  
    # 获取最后一层的 FSDP 模块
    last_layer_module = self.decoder.conv_out 
  
    # 唤回完整参数和梯度
    with FSDP.summon_full_params(last_layer_module, with_grads=True):
        # 在这个 with 块内部，参数和梯度被还原成了原始的形状！
        weight = last_layer_module.weight
      
        # 计算梯度（注意：这里可能需要手动调用 torch.autograd.grad）
        grads_rec = torch.autograd.grad(loss_rec, weight, retain_graph=True)[0]
        grads_gan = torch.autograd.grad(loss_gan, weight, retain_graph=True)[0]
      
        # 计算自适应权重
        d_weight = torch.norm(grads_rec) / (torch.norm(grads_gan) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

    # 继续正常的反向传播...
```
*缺点：会引入额外的通信开销（因为强行打断了 FSDP 默认的流水线），且代码写起来比较绕。*

#### 方案三：利用 PyTorch 的 Hook 机制（进阶）

如果你只需要在反向传播时获取梯度范数，可以给那一层注册一个 `backward_hook`。因为在 Hook 触发的瞬间，FSDP 刚好把完整的参数和梯度组装好，还没有进行 `Reduce-Scatter` 销毁。

```python
# 在模型初始化时
self.decoder.conv_out.register_full_backward_hook(self._hook_fn)

def _hook_fn(self, module, grad_input, grad_output):
    # 在这里，梯度是完整的
    # 你可以在这里计算 norm 并保存到一个类变量中
    pass
```

### 总结建议
如果你在训练 VAE 或 GAN 模型，**强烈建议使用方案一**。
VAE 的参数量通常远没有 LLM 那么极端（一般在几亿到十亿级别）。将需要频繁交互的极少数层（如判别器输出层、生成器输出层）从 FSDP 中剔除，不仅能完美避开参数访问限制，还能避免复杂的通信死锁问题。