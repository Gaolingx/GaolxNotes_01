# MoE训练技术演进与trick淘汰

### 🧑‍💻 User

为什么如今业界的 SOTA MoE training已经不再需要 router jitter，dropout，capability limit这些trick和z_loss？

---

### 🤖 Assistant

如今业界（如 DeepSeek-V2/V3, Mixtral 8x7B, Grok-1 等）的 SOTA MoE (Mixture of Experts) 训练范式确实发生了巨大的转变。早期的 MoE 模型（如 Switch Transformer, GShard）为了解决**训练不稳定**、**负载不均衡**和**硬件通信瓶颈**，引入了大量启发式（Heuristic）的 trick。

然而，随着系统工程（System Engineering）的突破、数据规模的爆炸以及对底层数学原理理解的加深，现代 MoE 训练回归了**“大道至简”**。以下是这些 trick 被逐步废弃的核心原因：

---

### 1. 为什么抛弃了 Capacity Limit (专家容量限制)？
**过去的原因**：为了适应 GPU/TPU 静态图的张量对齐要求，早期 MoE 会设定一个 `capacity_factor`。如果分配给某个专家的 token 数量超过了容量，多余的 token 就会被**丢弃（Dropped）**或直接通过残差连接跳过，导致模型性能受损。
**现在的解法：Dropless MoE 与动态内核**

现代框架（如 `MegaBlocks`、`Triton` 编写的定制 Kernel）通过**系统层面的创新**彻底解决了这个问题，实现了零丢弃（Dropless）：
*   **Token Sorting & Block-Sparse Matrix**：现代 MoE 不再强求专家接收固定维度的输入。系统会将所有的 token 按照分配的专家进行排序（Sorting），然后利用块稀疏矩阵乘法（Block-Sparse GEMM）或变长序列计算（Variable-length operations），让每个专家处理其对应的任意长度 token。
*   **通信与计算重叠（Overlap）**：借助现代的 All-to-All 通信优化，不再需要人为截断 token 来保证通信延迟可控。
*   **结论**：抛弃 Capacity Limit 是因为**软件工程已经能够高效处理动态形状（Dynamic Shapes）**，丢弃 token 对模型性能的损害不再是必须承受的代价。

### 2. 为什么抛弃了 Router Jitter (路由抖动/噪声)？
**过去的原因**：在早期的 Top-1 或 Top-2 路由中，为了防止“马太效应”（少数专家包揽所有 token，其他专家饿死），会在 Router 的 Logits 中加入均匀分布或高斯分布的噪声（Jitter），强制模型进行探索（Exploration）。
**现在的解法：更好的负载均衡与细粒度路由**

*   **优秀的负载均衡损失（Load Balancing Loss）**：如今我们能够通过更平滑的数学公式来约束路由行为，例如主流的辅助损失函数：
    $$L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$
    其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是路由器输出的平均概率。通过最小化点积，可以有效迫使分配变得均匀，而不需要随机噪声的干预。
*   **细粒度专家（Fine-grained Experts）与共享专家（Shared Experts）**：如 DeepSeek-MoE 引入了 Shared Expert（处理通用知识）和大量微型专家。这种架构天然降低了路由崩溃的风险，因为通用特征被剥离，特定专家只需要专注特定方向。
*   **结论**：Jitter 是一种治标不治本的正则化手段，它会破坏路由的确定性（Determinism），影响最终性能。现代方法通过直接优化目标函数来解决均衡问题。

### 3. 为什么抛弃了 Z-Loss？
**过去的原因**：Z-Loss 是 Google 在 ST-MoE 中引入的，公式大致为 $L_z = \log(\sum e^{x_i})^2$。它的目的是惩罚 Router 输出过大的 Logits，防止在 `fp16` 混合精度训练下产生溢出（Overflow）或下溢，从而稳定训练。
**现在的解法：更好的数值精度与归一化设计**

*   **硬件与精度的升级**：现代大模型训练已全面转向 `bf16` (Bfloat16)。`bf16` 拥有和 `fp32` 一样大的指数位范围，极大地减少了因 Logits 过大导致的数值溢出问题，因此不再需要 Z-Loss 来强行压制 Logits。
*   **Router 结构改进**：现代 MoE 的 Router 通常不包含复杂的层，且在输入 Router 之前往往有 `RMSNorm` 进行数值约束，或者采用了类似 `Softmax-1`、无偏置线性层等设计，天然避免了数值爆炸。

<details>
<summary><b>点击展开：关于 Router Z-Loss 的数学历史细节</b></summary>
早期的 Transformer 容易出现 Logits 漂移现象（Logits Drift）。在普通的 Dense 模型中，这种漂移只影响 Softmax 之前的绝对数值，因为 Softmax 是平移不变的：
$$Softmax(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$
但是，MoE 的路由机制通常会对 Logits 进行 Top-K 截断和阈值判断。当 $x_i$ 的绝对值在训练过程中无界增长时，轻微的梯度扰动就会在混合精度下引发灾难性的舍入误差。Z-Loss 强制让配分函数（Partition Function）$Z = \sum e^{x_i}$ 靠近 $1$。如今有了 `bf16` 和 `RMSNorm`，这个数学补丁就显得冗余了。
</details>

### 4. 为什么抛弃了 Dropout？
**过去的原因**：Dropout 是深度学习经典的防过拟合（Overfitting）技巧。早期的 MoE 参数量瞬间膨胀（比如 Switch Transformer 达到万亿参数），研究者担心模型会死记硬背训练数据。
**现在的解法：数据规模的爆炸（Chinchilla Scaling Laws）**

*   **欠拟合（Underfitting）是常态**：现代 SOTA LLM 的预训练数据量动辄达到数万亿甚至 10+ Trillion tokens（如 Llama-3 的 15T）。在这个数据规模下，即使是几百 B 参数的 MoE 模型，通常也只训练 1 到 2 个 Epoch。**模型面临的主要问题不是过拟合，而是欠拟合。**
*   **算力浪费**：Dropout 会随机丢弃一部分神经元的激活，这在极大浪费宝贵算力的同时，还会减慢模型的收敛速度。
*   **结论**：不仅是 MoE，现代 Dense 大模型的预训练阶段也基本完全移除了 Dropout。只在某些数据极其稀缺的 SFT（指令微调）阶段可能会酌情使用极低比例的 Dropout。

### 总结
业界 SOTA MoE 抛弃这些 trick，本质上是从**“算法启发式修补”**走向了**“系统与数学本质的协同优化”**。
*   **系统底层（MegaBlocks/Triton）**消灭了 Capacity Limit。
*   **精度升级（bf16）**消灭了 Z-Loss。
*   **数据规模（Trillions of Tokens）**消灭了 Dropout。
*   **架构演进（Aux Loss/Shared Expert）**消灭了 Router Jitter。

现在的 MoE 训练追求的是**极致的确定性、极简的损失函数**以及**对硬件的极致压榨**。

---

### 🧑‍💻 User

为什么aux_loss仍需保留，但是需要训练router时候需在config.json开启`output_router_logits`？

---

### 🤖 Assistant

这是一个非常精准且深入到工程实践层面的问题。

前面我们提到，现代 MoE 摒弃了大量启发式的 trick，但 `aux_loss`（负载均衡损失）不仅被保留了下来，而且是现代 MoE 训练的**绝对核心**。至于 `output_router_logits=True`，则是主流深度学习框架（如 Hugging Face `transformers`）为了优雅地计算这个 Loss 而做出的**工程设计**。

我们分两部分来解答：

---

### 第一部分：为什么 `aux_loss` 是不可或缺的？

虽然现代系统设计（如 Dropless 的 MegaBlocks）解决了硬件层面的“负载不均衡导致丢 token”的问题，但它**并没有解决算法层面的“专家饿死”问题**。

**1. 抵御“马太效应”与表征坍塌（Representation Collapse）**
在 MoE 的路由机制中，由于使用的是 Top-K 离散选择操作（Non-differentiable routing），模型天然存在**“富者愈富，贫者愈贫”**的倾向。
*   在训练初期，由于随机初始化，某个专家（比如 Expert A）可能碰巧对某类 token 产生了略微好一点的梯度。
*   Router 就会倾向于把更多的 token 发给 Expert A。
*   Expert A 因为吃到了更多的训练数据，变得更加“聪明”，其输出产生的 Loss 更低。
*   Router 进而确信“把 token 发给 A 是最安全的”，最终导致绝大多数 token 都发给了 A，其他专家完全闲置（饿死）。

**2. 保证参数利用率（Parameter Utilization）**
如果你训练一个 8x7B 的模型，但因为没有 `aux_loss`，所有 token 都跑去了其中 1-2 个专家那里，那么你的模型实际上退化成了一个 7B 或 14B 的 Dense 模型。你花了 8x7B 的显存和算力，却只得到了小模型的智商。

**结论：** Dropless 框架只是让你在专家负载不均衡时**“不会报错或掉效”**，但 `aux_loss` 是为了保证模型**“真正把所有专家的脑容量都用上”**。它是维持系统多样性的核心数学约束，而非可以丢弃的 trick。

---

### 第二部分：为什么需要在 Config 中开启 `output_router_logits`？

这完全是一个**软件工程与框架架构**（尤其是 Hugging Face 生态）的设计考量。

**1. Loss 计算的位置问题（变量跨层传递）**
我们来看一下标准的现代 MoE 负载均衡损失公式：
$$L_{aux} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$
其中，$P_i$ 是 Router 经过 Softmax 后输出的概率（依赖于 Router Logits），$f_i$ 是当前批次中实际分配给专家的 token 比例。

*   **痛点**：在 Transformer 架构中，MoE 层通常位于非常深的 Block 内部（例如第 10 层的 MLP 替换为 MoE）。而总损失函数（Cross Entropy Loss + Aux Loss）通常是在模型的**最顶层（Output Layer）或 `Trainer` 外部**统一计算的。
*   **解法**：为了在顶层计算 $L_{aux}$，模型必须把每一层隐藏在深处的 Router Logits 给“捞出来”，并一层层地向上传递，直到最终的输出端。

**2. `output_router_logits=True` 的工程作用**
在 Hugging Face 的代码实现中：
*   如果设为 `False`（通常是**推理阶段**的默认值）：MoE 层的 Router 算完 Top-K 选完专家后，这些 Logits 就作为局部变量被丢弃了。这极大地**节省了显存和显存带宽**，因为在生成文本时我们不需要计算 Loss。
*   如果设为 `True`（**训练阶段**必须开启）：每一层的 MoE 模块会把自己的 Router Logits 收集起来，打包成一个 Tuple 附加在当前层的 Hidden States 后面，随着前向传播（Forward Pass）一路带到模型的最终输出字典中（类似于 `outputs.router_logits`）。

**3. 联合优化的代码逻辑**
当你开启了这个参数后，`Trainer` 或你自定义的训练循环才能拿到这些 Logits。典型的代码逻辑如下：

```python
# 1. 前向传播，要求模型吐出 router_logits
outputs = model(input_ids, output_router_logits=True)

# 2. 提取标准的语言模型 Loss (Next token prediction)
lm_loss = outputs.loss 

# 3. 提取所有层的 router_logits
router_logits = outputs.router_logits 

# 4. 根据 logits 计算辅助损失 (框架通常内置了这个函数)
aux_loss = calculate_aux_loss(router_logits, config.router_aux_loss_coef)

# 5. 总 Loss 联合反向传播
total_loss = lm_loss + aux_loss
total_loss.backward()
```

<details>
<summary><b>点击展开：如果不开启会发生什么？</b></summary>
如果在训练时没有在 `config.json` 中开启 `output_router_logits=True`：
1. `outputs.router_logits` 将会是 `None`。
2. 你的 Loss 计算函数无法获取每一层的路由状态，`aux_loss` 的计算会被跳过或报错。
3. 你的模型会按照仅有 Cross Entropy Loss 的方式进行梯度下降。
4. 训练几千步之后，你会发现模型发生了严重的路由坍塌（Router Collapse），只有少数几个专家在工作，模型收敛停滞，变成了一个“虚假的”巨大参数模型。
</details>

### 总结
保留 `aux_loss` 是由 MoE 离散路由的**数学本质**决定的（为了防止模型退化为小模型）；而 `output_router_logits=True` 是深度学习框架为了实现**前向计算与损失计算解耦**而采取的优雅工程手段（按需传递，节省推理时的内存）。