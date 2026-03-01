# FSDP适配难度高于DDP

### 🧑‍💻 User

为什么fsdp适配难度比ddp大？（尤其是手动优化+多优化器情况下）

---

### 🤖 Assistant

Fully Sharded Data Parallel (FSDP) 的适配难度之所以远大于 Distributed Data Parallel (DDP)，核心原因在于两者的**底层设计范式（Paradigm）**完全不同：

*   **DDP 是“复制（Replication）”范式**：每个 GPU 上都有完整的模型副本。
*   **FSDP 是“切分（Sharding）”范式**：模型参数、梯度、优化器状态都被切碎并分散在不同 GPU 上。

这种切分破坏了 PyTorch 用户习惯的许多假设（例如“参数总是存在于内存中”、“可以直接访问 `.data`”），导致在**手动优化（Manual Optimization）**和**多优化器（Multiple Optimizers）**场景下，开发者必须显式地管理参数的生命周期。

以下是详细的技术深度解析：

---

### 1. 根本原因：参数的非持久性 (Transient Parameters)

在 DDP 中，`model.parameters()` 返回的张量在整个训练周期内都是完整的、驻留在显存中的真实数据。

在 FSDP 中，为了节省显存，**参数默认是不存在的**（或者说是被展平并分片的）。
*   **前向传播时**：FSDP 触发 `AllGather` 通信，把参数从其他 GPU 拉取过来，组成完整的层，计算完后立即释放（Free）。
*   **反向传播时**：再次 `AllGather` 参数，计算梯度，然后释放参数，最后对梯度进行 `ReduceScatter`。

**适配难点：**
如果你在标准的 `forward` 或 `backward` 流程之外（例如手动优化循环中）试图访问 `param.data`，你可能会得到：
1.  一个空的 tensor。
2.  一个展平的（Flattened）大一维 tensor（`FlatParameter`），无法对应原本的逻辑形状（如 `[768, 768]`）。

### 2. 手动优化（Manual Optimization）的噩梦

在 GAN（生成对抗网络）或强化学习（RL）中，开发者经常需要手动控制反向传播和梯度更新。

#### DDP 的做法
代码非常直观，完全符合 PyTorch 直觉：
```python
# DDP
d_loss = loss_fn(discriminator(real), discriminator(fake))
d_loss.backward()
d_opt.step() # 正常工作，因为参数和梯度都在本地
```

#### FSDP 的做法
FSDP 必须显式地告诉系统：“我现在需要完整的参数，请不要把它们释放掉”。

**难点 A：`FlatParameter` 问题**
FSDP 默认会将一个 Module 内的所有参数展平成一个巨大的 1D Tensor。如果你手动遍历 `model.parameters()` 并试图根据形状做操作（例如 Weight Decay 排除 bias），你会发现所有参数变成了一个巨大的由 FSDP 管理的 `FlatParameter`，原有的结构信息丢失了。
*   *解决办法*：必须使用 PyTorch 2.0+ 的 `use_orig_params=True`，但这又引入了额外的视图管理开销。

**难点 B：上下文管理器地狱**
在手动计算梯度或操作参数时，必须使用上下文管理器：
```python
# FSDP 手动操作参数示例
with FSDP.summon_full_params(model, writeback=True):
    # 此时会触发昂贵的 AllGather，显存瞬间飙升
    # 必须在这个缩进块内操作参数
    norm = torch.norm(model.layer1.weight)
    model.layer1.weight.data.clamp_(-1, 1)
# 退出块后，参数再次被释放或分片
```
如果你忘记写 `summon_full_params`，或者在块外引用了块内的 tensor，程序会直接报错或产生未定义行为。

### 3. 多优化器（Multiple Optimizers）的复杂性

当模型有多个优化器（如 GAN 中的 G 和 D，或预训练中的不同层率学习）时，FSDP 的分片机制会与优化器的 `step()` 发生冲突。

**难点：参数组与分片的不一致**
*   **DDP**：优化器持有参数的引用。只要参数在本地，优化器就能更新。
*   **FSDP**：优化器状态（Momentum, Variance）也是分片的。
    *   如果一个 Layer 被 FSDP 包装（Wrap）了，它就属于某个 FSDP 单元。
    *   如果你的优化器只优化该 Layer 的一部分参数（例如只优化 Bias），FSDP 需要非常小心地处理这种“部分分片”的更新。
    *   在早期 FSDP 版本中，一个 FSDP 单元内的所有参数必须由同一个优化器管理，否则会报错。虽然 `use_orig_params=True` 缓解了这个问题，但要求用户必须极其精准地构建 `param_groups`，且必须确保优化器初始化发生在 FSDP 包装**之后**。

### 4. 模块包装（Wrapping Policy）的艺术

DDP 通常只需要 `model = DDP(model)` 一行代码。FSDP 则需要递归地包装子模块。

**难点：显存与通信的权衡**
*   **如果不包装（只包最外层）**：这就变成了单纯的 ZeRO-stage 3 类似行为。在前向传播时，整个模型参数会被 Gather 到显存中，导致瞬间 OOM（显存溢出），失去了 FSDP 省显存的意义。
*   **如果包装太细**：通信次数暴增，网络开销巨大，训练速度极慢。
*   **用户心智负担**：用户必须定义 `auto_wrap_policy`（例如基于参数量大小，或基于 TransformerBlock 类）。对于非标准 Transformer 结构（如混合专家模型 MoE 或这种手动优化的复杂网络），自动策略往往失效，需要手动指定哪些层应该共用一个 FSDP 单元。

### 5. Checkpoint 保存与加载

**DDP**：`torch.save(model.module.state_dict())`。简单直接。

**FSDP**：
保存模型变成了分布式系统问题。你有两个选择，都很痛苦：
1.  **Full State Dict**：所有 GPU 把分片发给 Rank 0，拼成完整模型保存。
    *   *代价*：Rank 0 瞬间 OOM，或者 CPU 内存撑爆。速度极慢。
2.  **Sharded State Dict**：每个 GPU 保存自己的分片。
    *   *代价*：生成一堆文件。如果要推理或微调，加载时必须保证拓扑结构（World Size）完全一致，或者运行一个复杂的离线转换脚本将分片合并。

---

<details>
<summary><strong>📐 深度对比：通信原语 (Communication Primitives)</strong></summary>

这里用数学公式对比两者的通信逻辑，展示为什么 FSDP 的状态机更复杂。

#### DDP (Ring-AllReduce)
假设参数为 $\theta$，梯度为 $g$。
1. **Forward**: 本地计算，无通信。
2. **Backward**: 本地计算 $g_{local}$。
3. **Sync**: 全局规约 $$g_{global} = \frac{1}{N} \sum_{i=1}^N g_i$$ (AllReduce)。
4. **Update**: $\theta \leftarrow \theta - \eta \cdot g_{global}$。

#### FSDP (AllGather + ReduceScatter)
假设参数 $\theta$ 被切分为 $N$ 份 $\{\theta_1, ..., \theta_N\}$，Rank $i$ 只有 $\theta_i$。
1. **Forward (Unshard)**:
   对于每个 FSDP Unit：
   $$ \theta_{full} = \text{AllGather}(\theta_i) $$
   计算 $y = f(x, \theta_{full})$。
   **立即释放** $\theta_{full}$，只保留 $\theta_i$。
2. **Backward**:
   再次 $$\theta_{full} = \text{AllGather}(\theta_i)$$。
   计算梯度 $g_{full}$。
   **立即释放** $\theta_{full}$。
3. **Gradient Sync**:
   $g_{full}$ 太大，不能直接保留。
   $$ g_{shard\_i} = \text{ReduceScatter}(g_{full}) $$
   Rank $i$ 现在只持有它负责的那部分梯度 $g_{shard\_i}$。
4. **Update**:
   仅更新本地的分片：$\theta_i \leftarrow \theta_i - \eta \cdot g_{shard\_i}$。

**结论**：FSDP 的状态机包含多次显存分配、释放和通信同步，任何一步被打断（如手动干预）都可能导致死锁或数据错误。

</details>

---

### 总结

fsdp 适配难度大，是因为它从**根本上改变了对象的所有权模型**。

| 特性 | DDP | FSDP |
| :--- | :--- | :--- |
| **参数位置** | 始终在本地显存 | 大部分时间在 CPU 或其他 GPU，用时才取 |
| **Tensor 形状** | 保持原样 `[D_out, D_in]` | 可能是展平的 `[N]` 或分片的 |
| **优化器适配** | 传入参数列表即可 | 需严格对应分片组，顺序敏感 |
| **手动梯度操作** | 直接操作 `p.grad` | 需上下文管理器 `summon_full_params` |
| **开发思维** | 单机代码 + 一行 Wrap | 分布式系统思维，需管理内存生命周期 |

在多优化器和手动优化场景下，FSDP 要求开发者不仅懂模型算法，还必须懂**分布式显存管理**，这就是难度陡增的来源。

---

### 🧑‍💻 User

和deepspeed zero相比，fsdp有哪些细节是需要开发者自己处理的？

---

### 🤖 Assistant

虽然 FSDP (PyTorch Native) 的核心算法与 DeepSpeed ZeRO 是一致的（实际上 PyTorch FSDP 的设计深受 DeepSpeed ZeRO-3 的启发），但在**工程实现**和**开发者交互**层面，DeepSpeed 倾向于“全托管（Magic Box）”，而 FSDP 倾向于“显式控制（Explicit Control）”。

这意味着使用 FSDP 时，你需要手动处理很多 DeepSpeed 通过 `ds_config.json` 默默帮你做掉的事情。

以下是 FSDP 需要开发者**手动介入**的关键细节对比：

---

### 1. 模型包装策略 (Wrapping Policy) —— 最核心的痛点

这是 FSDP 最需要“微操”的地方，也是性能调优的关键。

*   **DeepSpeed**:
    *   通常不需要用户手动指定怎么切分模型。DeepSpeed 会通过劫持（Hijack）或在 `deepspeed.initialize()` 时自动处理参数切分。
    *   它往往能自动识别 Transformer 结构并应用优化。
*   **FSDP**:
    *   **必须手动定义 `auto_wrap_policy`**。如果你不指定，或者指定错了，FSDP 默认只会把整个模型当作一个巨大的单元（Unit）来切分。
    *   **后果**：如果不正确地按层（Layer-wise）包装（例如按 `TransformerBlock` 包装），在通信时就无法实现“计算与通信重叠（Overlap）”。你需要显式告诉 FSDP：“这一层是一个独立的通信单元”。

    ```python
    # FSDP 需要开发者写这种代码
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
  
    fsdp_plugin = TorchFSDPPlugin(
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={MyTransformerBlock}, # 必须显式指定类名
        ),
        ...
    )
    ```

### 2. 优化器的初始化顺序 (Optimizer Initialization Order)

这是一个经典的“先有鸡还是先有蛋”的问题，FSDP 对此非常严格。

*   **DeepSpeed**:
    *   你可以先定义优化器，再传给 `deepspeed.initialize`；或者只传参数组，让 DeepSpeed 帮你构建优化器。DeepSpeed 引擎接管了一切。
*   **FSDP**:
    *   **严格顺序**：必须**先**完成模型的 `FSDP(model)` 包装，**后**初始化优化器。
    *   **原因**：在包装之前，参数是普通的 Tensor；包装之后，参数变成了 `FlatParameter` 或者分片数据。如果在包装前初始化优化器，优化器里存的是“旧参数”的引用，导致梯度更新无效。
    *   **代码负担**：在复杂的 codebase 中（例如这部分代码分散在不同函数里），保证这个顺序需要重构代码逻辑。

### 3. 混合精度配置 (Mixed Precision Granularity)

DeepSpeed 往往只需要一个开关，而 FSDP 提供了极细粒度的控制，但也带来了配置复杂度。

*   **DeepSpeed**:
    *   配置文件里写 `"fp16": { "enabled": true }`，DeepSpeed 自动处理 Master Weights 的备份、Loss Scaling 等。
*   **FSDP**:
    *   你需要手动构建 `MixedPrecision` 对象，并分别指定三个阶段的数据类型：
        1.  **Param dtype**: 参数存储和计算的精度（如 bfloat16）。
        2.  **Reduce dtype**: 梯度通信时的精度（为了数值稳定性，通常用 float32）。
        3.  **Buffer dtype**: BatchNorm 等 Buffer 的精度。
  
    ```python
    # FSDP 显式配置
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, # 这里需要开发者权衡精度与速度
        buffer_dtype=torch.float32
    )
    ```

### 4. 梯度裁剪 (Gradient Clipping)

*   **DeepSpeed**:
    *   配置文件设置 `"gradient_clipping": 1.0`，然后在 `engine.step()` 内部自动完成。用户代码里甚至不需要出现 clip 函数。
*   **FSDP**:
    *   普通的 `torch.nn.utils.clip_grad_norm_` 对 FSDP 模型是**无效或极慢**的（因为它可能触发所有参数的 AllGather）。
    *   开发者必须修改训练循环，调用 FSDP 专用的裁剪方法：
        ```python
        # 必须把 model 显式转换成 FSDP 实例调用
        model.clip_grad_norm_(max_norm=1.0) 
        ```

### 5. Checkpoint (保存与加载)

这是 FSDP 最大的深坑之一，需要开发者根据集群环境做选择。

*   **DeepSpeed**:
    *   调用 `model_engine.save_checkpoint(path)`。DS 会保存成它自己独特的文件夹结构（包含零散的 partition 文件）。虽然通用性差，但存取非常无脑且快。
*   **FSDP**:
    *   原生的 `torch.save(model.state_dict())` 可能会导致 OOM（因为 Rank 0 试图加载所有参数）。
    *   开发者必须使用**上下文管理器**来定义保存策略：
        *   **策略 A (Full)**: 所有卡汇总到一张卡保存（易 OOM，但生成的权重通用）。
        *   **策略 B (Sharded)**: 每个卡存自己的碎片（快，但加载麻烦）。
        *   **策略 C (Local)**: 仅保存本地张量。

    ```python
    # FSDP 保存代码示例
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
  
    # 开发者必须显式告诉 FSDP："我现在要合并参数保存"
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        torch.save(model.state_dict(), "model.pt")
    ```

### 6. 参数初始化 (Parameter Initialization)

*   **DeepSpeed**:
    *   配合 DeepSpeed 的 `OnDevice` 工具，可以在 CPU 初始化甚至 Meta 设备初始化，DeepSpeed 会在切分时负责搬运。
*   **FSDP**:
    *   为了防止在 GPU 显存不足时初始化大模型（例如 70B 模型），开发者通常需要使用 `torch.device("meta")` 或者 `deferred_init` 库。
    *   FSDP 提供了 `param_init_fn` 参数，允许在将模块移动到 GPU 之前对其进行初始化。如果不处理好这里，很容易在包装 FSDP 的瞬间（模型从 CPU 复制到 GPU 时）炸显存。

---

### 总结：为什么这么设计？

*   **DeepSpeed (Config-Driven)**: 这种设计是为了**易用性**。它是一个侵入性很强的框架，接管了你的 Training Loop。优点是快、无脑；缺点是如果你想做一些魔改（比如特殊的梯度操作、非标准的优化器逻辑），你会发现很难绕过 DeepSpeed 的 Engine。
*   **FSDP (Code-Driven)**: 这种设计是为了**Pythonic 和 灵活性**。它只是一个 `nn.Module`。它不接管 Loop，不强制优化器。优点是你可以像写普通 PyTorch 代码一样随意 hack 模型逻辑；代价就是你必须自己处理分布式系统带来的所有状态管理细节。

**一句话概括：**
用 DeepSpeed 像是开**自动挡**车（配置好 JSON，踩油门就行，但没法做特技动作）；
用 FSDP 像是开**手动挡**赛车（你需要自己控制换挡时机、离合器配合，虽然繁琐，但你可以完全掌控车辆的每一个行为）。