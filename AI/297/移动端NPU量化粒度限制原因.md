# 移动端NPU量化粒度限制原因

### 🧑‍💻 User

我发现各移动平台npu的量化粒度都很少有per-block quant的，基本都只有per-channel scale，高通/苹果也是这样，是出于什么考量吗？

---

### 🤖 Assistant

你的观察非常敏锐，这确实是目前移动端 NPU（如高通 Hexagon DSP/NPU、苹果 Neural Engine、联发科 APU 等）架构设计上的一个普遍现象。

简单来说，**Per-channel quantization（逐通道量化）是硬件效率（算力/功耗）与模型精度之间的“甜点”**，而 Per-block（逐块/分组）量化虽然能提高精度，但在移动端硬件上实施的代价（Cost）过高。

以下是几个核心的考量因素，按优先级排序：

### 1. MAC 阵列的累加器设计（核心原因）

这是最底层的硬件限制。NPU 的核心是大量的 MAC（Multiply-Accumulate，乘加运算）阵列。

*   **Per-Channel 的数学优势：**
    在 Per-channel 量化中，一个输出通道对应的所有权重（Weights）共享同一个缩放因子（Scale, $S_w$），输入激活（Activations）通常也是共享一个 Scale ($S_x$)。
    卷积或全连接层的计算公式如下：
    $$ y = \sum (w \cdot x) \approx S_w \cdot S_x \cdot \sum (q_w \cdot q_x) $$
    注意这里最为关键的一点：**求和符号 $\sum$ 在缩放因子 $S_w \cdot S_x$ 的内部。**
    这意味着硬件可以先用低精度（如 INT8）疯狂进行乘加运算，将结果累加到一个高精度寄存器（通常是 INT32 Accumulator）中，**直到这一个通道的所有计算全部完成，最后才在 Accumulator 后面挂一个 Quantization Unit 进行一次反量化/重量化操作。** 这样，昂贵的浮点或定点缩放操作被分摊到了成百上千次 MAC 运算之后，开销几乎可以忽略。

*   **Per-Block 的硬件噩梦：**
    如果使用 Per-block（例如 block size = 32），意味着每 32 个权重就有一个新的 $S_{w, block}$。公式变为：
    $$ y = \sum_{k=0}^{N/BlockSize} \left( S_{w, k} \cdot S_x \cdot \sum_{i=0}^{31} (q_{w, i} \cdot q_{x, i}) \right) $$
    这里，**求和符号被 Scale 打断了。**
    硬件不能再一次性累加到底了。NPU 必须每做完 32 次乘法，就停下来，从内存读取新的 Scale，执行一次缩放（通常涉及乘法和移位），将结果存入部分和（Partial Sum），然后再做下一组。
    这会导致：
    1.  MAC 阵列的流水线（Pipeline）被打断，填充率下降。
    2.  需要在 MAC 单元内部或者紧邻处增加大量的缩放逻辑电路，极大增加硅片面积（Area）。
    3.  频繁的逻辑切换带来更高的功耗。

### 2. 内存带宽与数据局部性

移动端对带宽（Bandwidth）极其敏感。

*   **Per-Channel：** Scale 的数量等于输出通道数。对于一个 Conv 层，Scale 向量非常小，可以一次性加载到 NPU 的标量寄存器或片上 SRAM 中常驻，几乎不占用运行时带宽。
*   **Per-Block：** 假设 Block size = 32，权重是 INT4，那么每 32 个 INT4 权重（128 bits）就需要一个 Scale（假设 FP16 或 INT8）。这不仅增加了模型体积（Metadata overhead），更重要的是**Scale 的加载频率大大增加**。NPU 在计算过程中需要不断地从内存或缓存中预取 Scale 数据，这会挤占权重和激活值的加载带宽，容易造成 Memory Wall（内存墙）瓶颈。

### 3. 历史包袱与工作负载的变化（CNN vs Transformer）

目前的移动端 NPU 架构定义（Architectural Definition）大多是在 2-4 年前完成的。

*   **CNN 时代：** 在 ResNet、MobileNet、YOLO 等视觉模型主导的时代，Per-channel INT8 甚至 Per-tensor INT8 已经能提供足够的精度。权重的分布相对均匀，Per-block 带来的精度收益（Accuracy Gain）极低，无法证明其增加的硬件成本是合理的。
*   **Transformer/LLM 时代：** Per-block 的需求主要爆发于 LLM（大语言模型）和 Transformer 架构。因为 LLM 的权重存在严重的离群值（Outliers），且为了极致压缩（如 W4A16，即权重 4bit，激活 16bit），必须使用 Per-block 来保证精度。
*   **现状：** 高通和苹果目前的 NPU 主要是为 CNN 优化的 Systolic Array（脉动阵列）。它们虽然在努力适配 Transformer，但底层的累加逻辑还没完全转向适应“细粒度混合精度计算”。

### 4. 软件栈与编译器的复杂性

*   **SIMD 指令集限制：** 移动端的 CPU/DSP 指令集（如 ARM NEON 或 Hexagon HVX）通常提供了高效的 `DotProduct` 指令。这些指令大多也是基于整条向量共享 Scale 设计的。如果要做 Per-block，编译器往往需要生成效率很低的“解包 -> 乘法 -> 缩放 -> 累加”代码，反而不如直接用高精度计算快。
*   **模型转换工具：** TFLite、CoreML、SNPE (QNN) 等工具链长期以来将 Per-channel 视为标准。虽然 PyTorch/ONNX 支持 Per-block，但在下发到 NPU Compiler 时，如果硬件不支持，往往会回退到 CPU 执行，或者强制转为 Per-channel 导致精度崩塌。

### 总结与展望

**高通和苹果不做 Per-block，是因为在 INT8 CNN 时代，这是“投入产出比”极差的设计。** 它破坏了 MAC 阵列最核心的“累加”效率。

**但是，情况正在发生变化：**

随着 **On-device GenAI（端侧生成式 AI）** 的爆发，INT4 权重 + Per-block quantization 正在成为刚需（因为 INT4 Per-channel 精度太差）。

1.  **高通骁龙 8 Gen 3 / 8 Elite：** 已经在宣传支持 INT4，虽然其底层实现细节很少公开，但为了跑通 7B 参数的 LLM，它们很可能在硬件层面增加了对 Group-wise quantization 的特殊支持，或者通过特殊的微码（Micro-code）在 DSP 中高效模拟。
2.  **苹果：** 虽然 Neural Engine 主要是 Per-channel，但 CoreML 这种软件层通过 Metal (GPU) 能够很好地处理 Per-block 量化。未来几代的 A 系列芯片 NPU 很有可能会专门针对 Transformer 的 W4A16 甚至 MX 格式（微缩放格式）进行硬件级支持。

现在的趋势是向 **MXINT (Microscaling Formats)** 发展，这是 OCP (Open Compute Project) 推出的标准，专门解决 Block 量化的硬件效率问题（通过共享指数位等方式），未来移动端 NPU 很可能会采纳类似标准来实现高效的 Per-block。