# 大规模模型预训练迁移到Fabric

### 🧑‍💻 User

为什么大规模基座模型预训练可以考虑将框架从pytorch lightning迁移到lightning farbic

---

### 🤖 Assistant

在大规模基座模型（如 LLM、VLM）的预训练过程中，将框架从 **PyTorch Lightning (PTL)** 迁移到 **Lightning Fabric** 是一个非常符合工程实践的演进趋势。

PyTorch Lightning 是一个高级封装框架，而 Lightning Fabric 是 Lightning 团队推出的一个底层、轻量级的分布式引擎（提取了 PTL 中最核心的分布式和精度控制能力）。

以下是考虑进行这种迁移的核心原因：

### 1. 极致的灵活性与自定义训练循环 (Custom Training Loop)
大规模预训练通常涉及极其复杂的控制流，传统的 PTL `Trainer` 往往显得过于僵化。
* **PTL 的痛点**：它强制你使用 `LightningModule` 并遵循固定的 `training_step`、`validation_step` 生命周期。如果要在训练中交替进行不同的任务（例如混合专家模型 MoE 的自定义路由逻辑、RLHF 中的生成与训练交替、动态微批次大小调整），在 PTL 中需要编写大量繁琐的 `Callback` 或重写底层 Hook。
* **Fabric 的优势**：Fabric 让你完全掌控训练循环（标准的 Python `for` 循环）。你只需写原生 PyTorch 代码，Fabric 只负责在后台处理多卡同步和精度转换（如 BF16）。你可以随时 `break`、`continue` 或插入任何自定义逻辑。

### 2. 降低框架开销 (Framework Overhead)
在千卡级别的集群上预训练时，计算效率（MFU）是核心指标。
* **PTL 的痛点**：PTL 内部有大量的状态检查、日志聚合、Hook 触发器。虽然单卡下开销不明显，但在分布式环境下，每一次 iteration 的框架开销累积起来，可能会导致 GPU 出现不必要的等待等待期（Idle time）。
* **Fabric 的优势**：Fabric 是对原生 PyTorch 的极薄封装（Thin Wrapper）。它去除了所有不必要的抽象层，最大程度降低了框架本身的代码开销，使系统性能更接近甚至等同于纯净的 `torch.distributed` 代码。

### 3. 更透明的分布式策略控制 (FSDP, DeepSpeed 与 3D 并行)
大模型预训练通常需要张量并行（TP）、流水线并行（PP）和数据并行（DP/FSDP）的组合。
* **PTL 的痛点**：虽然 PTL 支持 FSDP 和 DeepSpeed，但它们被深度封装在 `Strategy` 类中。当你需要细粒度地控制权重的切片方式、通信重叠（Communication Overlap）或者自定义的梯度累积逻辑时，往往会碰到 PTL 的 API 边界。
* **Fabric 的优势**：Fabric 将分布式策略作为独立组件暴露出来（Opt-in）。你可以像写单卡代码一样写模型，然后通过 `fabric.setup(model, optimizer)` 来按需注入 FSDP 或 DeepSpeed。它与 PyTorch 原生的分布式 API（如 `torch.distributed.tensor.parallel`）兼容性极佳，非常适合构建自定义的 3D 并行策略。

### 4. 调试与错误追踪的便利性 (Debugging & Traceability)
* **PTL 的痛点**：当大模型训练在几千步后突然因为 NaN 或 OOM 崩溃时，查看 PTL 的报错堆栈（Stack Trace）通常会看到几十层框架内部的调用逻辑，很难快速定位是模型代码问题还是框架生命周期问题。
* **Fabric 的优势**：由于没有复杂的生命周期管理，Fabric 报错时的堆栈就是你写的原始 PyTorch 训练循环逻辑。这对于复杂的预训练调试（尤其是定位算子溢出或通信死锁）至关重要。

---

<details>
<summary><strong>💡 深入对比：检查点 (Checkpointing) 与内存管理</strong></summary>

在大模型预训练中，保存和加载 Checkpoint 是一个耗时且容易 OOM 的操作。保存几十到几百 GB 的模型参数需要特殊的处理。

*   **PTL 的做法**：主要依赖内置的 Checkpoint Callback。对于 FSDP 这种需要聚合分片权重的策略，PTL 的自动处理有时会导致主节点内存峰值过高（CPU OOM）。
*   **Fabric 的做法**：提供了更底层的 `fabric.save()` 和 `fabric.load()` API。它允许开发者完全自定义如何分片保存（Sharded Checkpointing），甚至可以将模型权重直接流式传输到对象存储（如 AWS S3），或者利用像 `safetensors` 这样的库进行无拷贝加载。这种底层的控制权是大模型工程团队所必需的。

</details>

<details>
<summary><strong>📊 核心差异总结表</strong></summary>

| 特性 | PyTorch Lightning | Lightning Fabric |
| :--- | :--- | :--- |
| **架构设计** | 高度抽象的 `Trainer` 和 `LightningModule` | 轻量级、只提供核心工具，由开发者写主循环 |
| **适用场景** | 快速迭代、标准学术研究、中小型模型微调 | 大规模基座模型预训练、复杂强化学习系统 |
| **学习曲线** | 较陡（需要学习 PTL 独有的生命周期和 Hook） | 极平缓（只要懂原生 PyTorch 即可） |
| **调试难度** | 困难（堆栈极深） | 简单（透明的纯 PyTorch 逻辑） |
| **修改并行策略** | 需要重写内部 Strategy 插件，难度高 | 极其简单，直接修改 `fabric.setup` 或调用原生 API |

</details>

---

### 总结与迁移建议

在算力成本极其高昂的基座模型预训练中，**“可控性”优先于“便捷性”**。

如果你发现你的团队为了实现某个特殊的并行算子、特殊的梯度裁剪方式，或者为了排查某个死锁 Bug，经常需要去翻阅 PyTorch Lightning 的源码并编写各种 Hack 补丁时，这就意味着 PTL 的高度抽象已经成为了负担。

迁移到 **Lightning Fabric** 能够让你在**保留 Lightning 优秀的硬件/精度自动配置能力**的同时，**找回纯 PyTorch 的控制权和极致性能**。迁移成本通常也很低，因为 Fabric 本身就是从 PTL 内部剥离出来的心脏。