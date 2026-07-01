# 非侵入式TP通过钩子和属性替换实现分布式并行

### 🧑‍💻 User

非侵入式tp warpper为什么会大量应用`module.register_forward_pre_hook`和`setattr(module, param_name, new_param)`？

---

### 🤖 Assistant

在分布式深度学习中，“非侵入式张量并行（Non-invasive Tensor Parallelism, TP）Wrapper”的核心目标是：**在不修改原始模型源代码（例如直接使用 Hugging Face 的模型代码）的情况下，动态地将单机模型转换为支持多卡分布式的模型。**

为了实现这种“偷梁换柱”的效果，框架（如 ColossalAI、DeepSpeed-Inference、Megatron 的 auto-wrapper 等）必须依赖 Python 的动态特性和 PyTorch 提供的回调机制。这就解释了为什么会大量使用 `setattr` 和 `register_forward_pre_hook`（以及 `register_forward_hook`）。

以下是具体的原理分析：

### 1. 为什么大量使用 `setattr(module, param_name, new_param)`？
**核心目的：物理替换权重（Weight Sharding），实现显存的真正切分。**

张量并行（TP）的本质是将模型中的巨大权重矩阵切分成小块，分散到不同的 GPU 上。
假设原始模型中有一个标准的线性层 `nn.Linear`，其权重为 $W \in \mathbb{R}^{d_{out} \times d_{in}}$。

*   **侵入式做法**：开发者需要重写模型代码，把 `nn.Linear` 替换成 `mpu.ColumnParallelLinear`。
*   **非侵入式做法**：模型实例化时依然是 `nn.Linear`。Wrapper 会遍历模型的每个模块，拿到全局权重 $W$，对其进行切片（Slice）。例如，按列切片后，当前 GPU 分配到的权重为 $W_i \in \mathbb{R}^{(d_{out}/p) \times d_{in}}$ （其中 $p$ 是 TP 的并行度）。

切片完成后，**必须将这个小的局部权重重新绑定到原始模块上**，以便在执行原生的 `forward` 函数时，使用的是切分后的显存。
```python
# 伪代码：非侵入式替换权重
sharded_weight = get_local_shard(module.weight, tp_rank, tp_world_size, dim=0)
new_param = nn.Parameter(sharded_weight)

# 删除原始的全局权重，释放显存
delattr(module, 'weight') 
# 注入切片后的局部权重
setattr(module, 'weight', new_param) 
```
如果不使用 `setattr` 覆盖原有属性，原生的 `forward` 函数（例如 `F.linear(input, self.weight, self.bias)`）就无法找到正确的切片权重，也无法实现显存的降低。

### 2. 为什么大量使用 `module.register_forward_pre_hook`？
**核心目的：劫持前向传播，动态注入分布式通信原语（Communication Primitives）和张量形状变换。**

在修改了权重之后，仅仅依靠原生的 `forward` 函数是无法完成正确的数学计算的。张量并行需要在计算前后插入通信操作（如 All-Reduce, All-Gather 等）。既然我们不能修改原生 `forward` 函数的代码，就只能通过**钩子（Hooks）**在它执行前后“做手脚”。

`register_forward_pre_hook` 会在模块的 `forward` 函数执行**之前**被调用。它在非侵入式 TP 中主要承担以下任务：

#### A. 输入张量的切分与通信 (Input Manipulation)
有些张量并行策略（如 Row Parallelism 行并行）或者结合序列并行（Sequence Parallelism）时，输入到该层的激活张量（Activations）需要在计算前进行处理。
*   **All-Gather**：如果上一层输出的是切分在序列维度上的张量，而当前层需要完整的输入，`pre_hook` 可以在矩阵乘法发生前，执行一次 All-Gather 操作将输入拼接完整。
*   **切分输入**：对于行并行，输入 $X$ 需要按最后一维切分成 $X_i$。`pre_hook` 可以截获原始的完整输入 $X$，将其切片后，再送入原生的 `forward` 函数。

#### B. 参数的动态拉取 (常用于 ZeRO-3 / FSDP 结合 TP 时)
虽然不是纯粹的 TP，但现代非侵入式框架通常将 TP 与 ZeRO（零冗余优化器）结合。如果权重平时被卸载或切分存储，`pre_hook` 可以在该层开始计算的前一刻，触发 All-Gather 将 TP 所需的当前分片拉取到 GPU 上。

#### C. 改变 RNG 状态 (随机数种子管理)
在 TP 中，Dropout 层是非常棘手的。由于不同 GPU 上的激活张量可能是相同的（例如 Column Parallelism 的输入），也可能是不同的，`pre_hook` 会被用来在执行该层之前，临时切换到对应的分布式随机数生成器（RNG）状态，以保证 Dropout 掩码在多卡上的正确性。

<details>
<summary><b>补充：`register_forward_hook` (Post-hook) 的重要性</b></summary>
除了 `pre_hook`，非侵入式 TP 同样高度依赖后置钩子（Post-hook）。

在**行并行（Row Parallelism）**中：
每个 GPU 使用切分后的输入 $X_i$ 和切分后的权重 $W_i$ 算出局部结果 $Y_i = X_i W_i^T$。
此时计算并没有完成，真正的结果是 $Y = \sum Y_i$。
因为不能改写原生的 `forward`，框架会挂载一个 `post_hook`。当原生的 `F.linear` 算出局部的 $Y_i$ 后，`post_hook` 会拦截这个输出，执行一个全局的 **All-Reduce** 操作，将所有 GPU 的结果相加，然后再返回给下一层。
</details>

### 总结工作流示例：如何非侵入式地转换一个 Linear 层

当一个非侵入式 Wrapper 遇到 `nn.Linear` 时，它的内部逻辑大致如下：

1.  **初始化阶段 (Init)**:
    *   读取 `module.weight`。
    *   计算切片，得到 `local_weight`。
    *   使用 **`setattr(module, 'weight', local_weight)`** 替换参数。
2.  **准备阶段 (Hooks)**:
    *   使用 **`module.register_forward_pre_hook`** 注册前置钩子：拦截输入张量，如果需要（例如 SP 切分输入），执行通信原语，修改输入形状。
    *   使用 `module.register_forward_hook` 注册后置钩子：拦截输出张量，执行 All-Reduce 等通信原语汇总结果。
3.  **运行阶段 (Runtime)**:
    *   用户直接调用 `model(inputs)`。
    *   触发 `pre_hook` -> 准备好输入。
    *   执行 Hugging Face 原生的 `forward` -> 因为 `setattr` 的作用，直接使用切片权重进行了局部计算。
    *   触发 `post_hook` -> 聚合局部结果，完成一整套张量并行逻辑。

通过这种“属性替换 + 钩子拦截”的设计思想，框架做到了与上层模型代码的完全解耦。用户不需要学习一套新的模型编写规范（如 Megatron-LM 的算子），只需一行代码 `wrapper(model)` 即可享受复杂的分布式并行策略。