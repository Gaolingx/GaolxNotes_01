# MoE训练快但依赖底层优化

### 🧑‍💻 User

我们发现 experts 增加后，实际train it/s往往比同尺寸规模的 dense 模型慢非常多，这和moe推理快放在一起看会有点反直觉，为什么pytorch定义的模型推理/训练都存在性能问题，特别依赖moe grouped+fused kernel优化，比如基于 cutile/trton 的 custom tiled kernel、Megatron Core 等

---

### 🤖 Assistant

这是一个非常深刻且直击当前大模型工程痛点的问题。很多算法工程师在初次从 Dense 模型切换到 MoE (Mixture of Experts) 模型时，都会产生这种“反直觉”的困惑：**既然 MoE 每次只有少数专家被激活，计算量（FLOPs）变小了，为什么在 PyTorch 下跑起来，无论是训练还是推理（尤其是吞吐量）却比同等激活参数量的 Dense 模型慢得多？**

这其中的核心矛盾在于：**理论上的计算量减少，转化为了实际硬件上极其糟糕的内存访问模式和极低的计算强度（Arithmetic Intensity）。** 原生的 PyTorch API 无法处理这种动态的、碎片化的计算流。

以下是详细的原因剖析，以及为什么我们极度依赖 Custom Tiled Kernel (如 Grouped GEMM) 和 Megatron Core 优化。

---

### 一、 为什么原生 PyTorch 跑 MoE 会非常慢？

在原生的 PyTorch 中实现 MoE，通常的逻辑是：计算 Router 概率 $\rightarrow$ 获取 Top-K 索引 $\rightarrow$ 使用 `torch.gather` 重新排列 Token $\rightarrow$ 写一个 `for` 循环遍历每一个专家进行线性层计算 $\rightarrow$ 使用 `torch.scatter` 还原顺序。

这个过程在 GPU 硬件层面引发了三场“灾难”：

#### 1. 致命的显存带宽瓶颈（Memory Bound）与 Gather/Scatter 开销
PyTorch 原生的 `gather` 和 `scatter` 操作是纯粹的 Memory Bound 操作。为了把分散在 Batch 中的 Token 喂给对应的专家，GPU 需要进行大量的、非连续的显存读写。
*   在 Dense 模型中，数据是连续进入 Tensor Core 的。
*   在 PyTorch MoE 中，大量时间被浪费在搬运数据上。很多时候，**Token 路由重排的时间甚至超过了专家网络（MLP）本身的计算时间**。

#### 2. Kernel Launch 开销与碎片化的 GEMM
如果你的模型有 $E$ 个专家，在原生 PyTorch 中你通常会写类似这样的代码：
```python
for i in range(num_experts):
    expert_tokens = hidden_states[expert_mask == i]
    output[expert_mask == i] = expert_network[i](expert_tokens)
```
*   **Kernel Launch Overhead:** GPU 必须为每一个专家单独 launch CUDA kernel。当专家数量 $E$ 很大（如 8, 16 甚至 64）时，Kernel Launch 的 CPU 下发时间可能远大于 GPU 执行时间，导致 GPU 处于极度饥饿状态。
*   **低效的矩阵乘法 (Small $M$ GEMM):** 矩阵乘法的效率高度依赖于矩阵的尺寸。假设总 Token 数为 $N$，均匀分布下每个专家只分到 $N/E$ 个 Token。这使得 GEMM 的 $M$ 维度变得非常小，GPU 的 Tensor Core 利用率断崖式下跌。

#### 3. 负载不均衡 (Load Imbalance) 与 Padding 浪费
如果某些专家分到的 Token 特别多，某些特别少，原生 PyTorch 往往需要对齐维度（Padding）或者截断（Dropping），这要么带来了大量无效计算（算 0），要么损害了模型精度。

---

### 二、 为什么训练的下降比推理更严重？

<details>
<summary><b>点击展开：训练期独有的通信灾难 (All-to-All)</b></summary>

推理通常在单卡或少数几张卡上进行（Tensor Parallelism），而训练通常是大规模集群，必须使用 **专家并行 (Expert Parallelism, EP)**。

在 EP 下，不同的专家被放置在不同的 GPU/节点上。
1. **前向传播 (Forward):** 本地 GPU 的 Token 必须通过网络发送到拥有对应专家的目标 GPU。这需要一次全局的 `All-to-All` 通信。
2. **后向传播 (Backward):** 计算完梯度后，还需要一次 `All-to-All` 将梯度/误差传回来。

`All-to-All` 是所有分布式通信原语中最耗时、最考验网络带宽的一种。在 Dense 模型中，通信通常是连续大块的（如 All-Reduce），可以通过计算和通信掩盖 (Overlap) 优化。而在原生 MoE 训练中，由于路由的动态性，通信极其细碎且难以掩盖，导致训练的 it/s 暴跌。

此外，训练期需要保存前向的激活值用于反向传播。MoE 引入的复杂路由逻辑和中间变量（如 mask, gather index）会大幅增加 Activation Memory 的占用，导致不得不缩小 Batch Size，进一步拉低了吞吐量。

</details>

---

### 三、 为什么必须依赖 Grouped + Fused Kernel (如 CUTLASS / Megatron Core)？

为了解决上述问题，业界（如 NVIDIA Megatron、Google、OpenAI）开发了高度定制化的 CUDA/Triton 算子。它们将 MoE 的性能拉回了正轨。

#### 1. Grouped GEMM (分组矩阵乘)
这是解决 `for` 循环和碎片化 GEMM 的核心技术。
*   **原理:** 不是调用 $E$ 次标准的 GEMM，而是调用**一次** Grouped GEMM Kernel。在这个 Kernel 内部，允许传入一个列表，包含不同大小的 $M_i$（每个专家的 Token 数），但共享相同的 $K$ 和 $N$（隐藏层维度）。
*   **优势:** 只需要一次 Kernel Launch，同时硬件底层（如基于 CUTLASS/Triton 的实现）可以在不同的 SM (Streaming Multiprocessor) 上并行调度不同专家的计算，极大地提高了 Tensor Core 的饱和度。

$$ \text{Total FLOPs} = \sum_{i=1}^{E} 2 \cdot M_i \cdot K \cdot N $$

#### 2. Fused Routing (融合路由，消灭 Gather/Scatter)
*   **原理:** Custom Kernel 不再需要在全局显存中做真实的物理数据搬运。算子内部直接接收 Token 的路由索引（Routing Indices），在计算 GEMM 的同时，**直接从输入矩阵的非连续地址中读取数据放入寄存器（SRAM）**参与计算。
*   **优势:** 将 Memory Bound 操作与 Compute Bound 操作融合（Fusion），彻底消灭了中间的显存读写瓶颈。

#### 3. Custom Tiled Kernel (如 Megatron Core 的优化)
Megatron-Core 进一步对通信和计算做了深度绑定：
*   **Token Dropping/Padding 的底层优化:** 在算子层级直接处理 Capacity limit，无需在 Python 层做掩码。
*   **Overlap 优化:** 将局部的 Tensor Parallel 计算与跨节点的 All-to-All 专家通信在底层 Stream 上进行 Overlap（流水线掩盖）。

### 总结

您觉得“反直觉”，是因为我们习惯于用**“理论浮点运算量 (FLOPs)”**来衡量速度，但在实际的硬件工程中，**“访存开销 (Memory Access)”**和**“算子下发开销 (Launch Overhead)”**往往才是真正的瓶颈。

原生 PyTorch 将 MoE 拆解成了大量细碎的、访存密集的独立算子；而通过 CUTLASS/Triton 编写的 Grouped+Fused Kernel，则是将整个 MoE 层打包成了一个“黑盒”，让 GPU 用处理 Dense 模型的方式（高并发、一次性读写）去处理稀疏数据。这就是为什么想要在生产环境中高效训练/部署 MoE，底层算子重写是必经之路。

---

### 🧑‍💻 User

infra不完善是moe很早被提出但没有大规模用于预训练的原因吗？

---

### 🤖 Assistant

这是一个非常敏锐且宏观的问题。**答案是肯定的，但不完全是。**

Infra（基础设施）的不完善，确实是 MoE（混合专家模型）很早被提出，却长达数年未能在大规模预训练中普及的**最核心物理阻碍**。但除此之外，**算法的不稳定性、硬件显存的限制，以及 Dense 模型在当时的“高性价比”**，共同造成了 MoE 的“怀才不遇”。

我们可以将原因归结为以下四个维度：

### 1. Infra 与通信墙：All-to-All 带来的灾难

如前文所述，MoE 在分布式训练中严重依赖**专家并行 (Expert Parallelism, EP)**。EP 需要在不同的 GPU 节点之间交换 Token，这依赖于 `All-to-All` 通信原语。
*   **早期硬件网络带宽不足：** 在 2017 年（Google 提出 Sparsely-Gated MoE）到 2020 年左右，集群的主流网络还是较慢的 InfiniBand，甚至很多还在用以太网，节点内 NVLink 也没有今天这么普及和高速。`All-to-All` 在低带宽下的延迟，轻易就能吞噬掉 MoE 节省下来的计算时间。
*   **缺乏系统级框架支持：** 当时根本没有 Megatron-LM、DeepSpeed-MoE 或 vLLM 这样成熟的框架。要训练 MoE，研究人员需要手动在 TensorFlow 或 PyTorch 底层去写极度复杂的通信和计算流水线重叠（Overlap）逻辑，工程门槛高到只有 Google 等极少数巨头能玩得转。

### 2. 硬件显存墙 (Memory Wall)：MoE 是“空间换时间”的怪兽

MoE 的本质是**保持计算量（FLOPs）不变的情况下，极大地暴增模型参数量**。
*   在 V100 (16GB/32GB VRAM) 甚至更早的时代，GPU 最稀缺的资源其实是显存。
*   一个拥有 8 个专家的 MoE 模型，其前馈网络（FFN）的参数量是同计算量 Dense 模型的 8 倍。这意味着你需要成倍的 GPU 才能把模型权重和 Optimizer States 塞进显存里。
*   在当时，连装下一个百亿参数的 Dense 模型都很费劲，大家都在搞张量并行（TP）和流水线并行（PP），根本没有多余的显存空间去留给 MoE 的闲置专家。直到 A100 (80GB) 和 H100 (80GB/144GB) 的普及，显存容量才勉强跟上了 MoE 的胃口。

### 3. 算法与优化灾难：MoE 极度难以收敛

<details>
<summary><b>点击展开：MoE 训练期的“路由崩塌”与不稳定性问题</b></summary>

早期的 MoE 算法在优化上存在非常致命的问题：
1.  **路由崩塌 (Representation Collapse / Routing Collapse)：** 神经网络是“偷懒”的。在训练初期，如果有几个专家稍微表现好一点，Router 就会倾向于把所有的 Token 都分配给这几个专家。这导致“旱的旱死，涝的涝死”，大部分专家得不到训练（梯度为 0），MoE 最终退化成了低计算量的 Dense 模型。
2.  **复杂的辅助损失 (Auxiliary Loss)：** 为了解决负载不均衡，研究人员被迫引入了各种辅助 Loss（例如 $L_{aux}$ 来惩罚负载不均，或者 Router Z-loss 来稳定 logits）。这些超参数极难调优，调小了不起作用，调大了会干扰主任务（Language Modeling）的收敛，导致模型精度受损。
3.  **梯度方差极大：** 动态路由导致每个 batch 参与计算的网络结构都在变，这使得梯度存在极大的方差，训练过程经常出现 Loss Spike（突然爆炸）。

</details>

### 4. 历史进程：Dense 模型当时处于“甜蜜期”

在 2018 到 2022 年这段时间，NLP 界的主旋律是 **Scaling Laws (缩放定律)**。
*   GPT-2 $\rightarrow$ GPT-3 $\rightarrow$ Chinchilla 的成功告诉所有人：**只要无脑把 Dense 模型做大，投入更多的数据和算力，模型能力就会稳定地线性（对数）增长。**
*   既然 Dense 模型结构简单、工程成熟、且 Scaling 效果确定性极高，为什么要去碰复杂、难调、且需要重写基建的 MoE 呢？

---

### 结论：为什么现在 MoE 又行了？

今天 MoE 成为开源和闭源大模型（如 GPT-4, Mixtral, DeepSeek, Qwen）的标配，是因为上述四个问题在近期被同时解决了：

1.  **Infra 成熟：** Megatron-Core, Triton, CUTLASS 等使得 Grouped GEMM 和 Fused 算子成为开源标配，DeepSpeed/Megatron 把 EP 通信优化做到了极致。
2.  **硬件升级：** A100/H100 以及超高带宽的 NVLink 交换机解决了显存和 `All-to-All` 瓶颈。
3.  **算法突破：** 诸如 DeepSeek-MoE 提出的细粒度专家（Fine-grained experts）、Shared Expert（共享专家）等机制，极大地缓解了路由崩塌，训练变得前所未有地稳定。
4.  **Dense 遇到算力/推理瓶颈：** 当模型规模逼近万亿参数时，Dense 模型的推理成本（延迟和算力）变得无法忍受。MoE 成了唯一能在极低推理延迟下，享受万亿参数容量红利的救命稻草。