# Softmax vs sigmoid MoE

### 🧑‍💻 User

$G(x) = \mathrm{softmax}(W_g x + b) \in \mathbb{R}^N$

$G(x) = \mathrm{sigmoid}(W_g x + b) \in \mathbb{R}^N$

一般来说，在选取 top-K 专家时，我们会对 gating layer 的输出进行归一化，通常我们会使用 softmax function，但是，在 deepseek 的 Loss-Free Balancing 中，作者通过实验发现，使用sigmoid作为激活函数效果更好，实验结果显示，sigmoid function 对于超参数更加 robust，且表现也更好一些。两者作为moe router的activation function有什么区别，有哪些moe模型分别使用softmax or sigmoid？

---

### 🤖 Assistant

这是一个非常深刻且前沿的问题。在 Mixture of Experts (MoE) 的架构演进中，Gating Network (Router) 的激活函数选择确实经历了从 Softmax 到 Sigmoid 的探索。

以下是关于两者在 MoE Router 中的本质区别、为何 DeepSeek 倾向于 Sigmoid，以及代表性模型的详细对比。

### 1. 核心区别：竞争性 vs. 独立性

从数学性质上来看，两者最大的区别在于专家之间是否存在“耦合（Coupling）”或“竞争（Competition）”关系。

#### Softmax: 零和博弈 (Zero-Sum Game)
$$G(x)_i = \frac{e^{h_i}}{\sum_{j} e^{h_j}}$$

*   **性质：** 所有专家的权重之和必须为 1 ($\sum G(x)_i = 1$)。
*   **机制：** 这是一种**竞争机制**。如果 Router 想要提高专家 A 的权重，根据定义，它**必须**降低其他专家的权重。
*   **缺点：**
    *   **梯度耦合：** 当反向传播时，一个专家的梯度会影响所有其他专家的 Logits。这在负载均衡（Load Balancing）时可能导致问题：为了满足负载均衡损失（Auxiliary Loss），模型可能会被迫压低“最强专家”的分数，从而牺牲模型性能。
    *   **Winner-takes-all：** 在训练初期，容易出现少数专家垄断的情况（Collapse），需要较强的 Auxiliary Loss 来纠正。

#### Sigmoid: 独立打分 (Independent Scoring)
$$G(x)_i = \mathrm{sigmoid}(h_i) = \frac{1}{1 + e^{-h_i}}$$

*   **性质：** 每个专家的打分都在 $(0, 1)$ 之间，但**互不影响**，总和不限制为 1。
*   **机制：** 这是一种**独立阈值机制**。Router 独立地询问每个专家：“你对处理这个 token 有多大信心？”
*   **优点：**
    *   **解耦：** 提高专家 A 的分数不需要压低专家 B。这给予了 Router 更大的灵活性。
    *   **负载均衡更友好：** 也就是 DeepSeek 提到的点。在使用 Bias 进行负载均衡时，调整专家 $i$ 的 Bias ($b_i$) 只会改变该专家的激活阈值，不会直接通过分母挤压其他专家的概率分布。

---

### 2. 为什么 DeepSeek (DeepSeek-V2/V3) 选择 Sigmoid？

DeepSeek 在其论文（特别是 DeepSeek-V2 关于 **DeepSeekMoE** 架构和 **Loss-Free Balancing** 的部分）中详细阐述了这一点。

#### A. Loss-Free Balancing 的实现基础
DeepSeek 提出了一种不依赖梯度惩罚（Auxiliary Loss）的负载均衡方法，而是通过动态调整 Bias 来实现。
*   **Softmax 的问题：** 如果使用 Softmax，调整 Bias $b_i$ 会通过分母 $\sum e^{h_j}$ 影响全局分布，难以精确控制单个专家的负载而不产生副作用。
*   **Sigmoid 的优势：** 使用 Sigmoid 时，Router 的输出是解耦的。如果要增加专家 $i$ 的负载，只需减小其 Bias；如果要减少负载，只需增加其 Bias。这种调整是局部的、可控的，不会干扰模型对 Top-K 的核心判断逻辑（Logits 的相对大小）。

#### B. 归一化步骤的后置
值得注意的是，虽然 DeepSeek 使用 Sigmoid 进行**打分（Scoring）**和**选择（Routing）**，但在选中 Top-K 专家后，通常还是会进行一次归一化，以保证后续计算数值的稳定性：

$$
\begin{aligned}
s_i &= \mathrm{sigmoid}(W_g x + b)_i \\
\mathcal{I} &= \mathrm{TopK}(s) \\
p_i &= \frac{s_i}{\sum_{j \in \mathcal{I}} s_j} \quad (\text{对被选中的专家进行归一化})
\end{aligned}
$$

DeepSeek 发现这种 setup 在超参数调整上更加 Robust，且收敛效果更好。

---

### 3. 代表性模型分类

根据 Router 激活函数的不同，我们可以将主流 MoE 模型分类如下：

#### 使用 Softmax 的模型 (传统/主流流派)
大多数基于 Google **Switch Transformer** 范式的模型都使用 Softmax。这是最经典的设计，认为 Router 的输出应当是一个概率分布。

*   **Switch Transformer (Google):** 开创性的 Top-1 模型，使用 Softmax。
*   **GShard (Google):** Top-2 MoE，使用 Softmax。
*   **GLaM (Google):** 使用 Softmax。
*   **Mixtral 8x7B (Mistral AI):** 著名的开源 MoE，使用 Softmax Top-2 路由。
*   **Qwen1.5-MoE (Alibaba):** 沿用了主流的 Softmax 路由机制。
*   **Grok-1 (xAI):** 使用 Softmax。

#### 使用 Sigmoid 的模型 (新锐/专门优化流派)
这一流派认为专家之间应当解耦，或者为了配合特殊的负载均衡策略。

*   **DeepSeek-V2 / DeepSeek-V3:**
    *   明确使用了 Sigmoid 作为 Router 激活函数。
    *   配合其独有的 **Expert Isolation**（Shared Experts + Routed Experts）和 **Grouped-Query Attention** 风格的 Latent Vector 路由机制。
*   **NLLB (No Language Left Behind, Meta):**
    *   虽然主要使用 Top-K Softmax，但在某些由稀疏性研究衍生的变体中，Meta 的研究员探索过 Sigmoid-based routing (例如在 Sparsely Gated Mixture of Experts 的某些改进版实验中)。
*   **Megablocks (Stanford/Databricks):**
    *   Megablocks 是一个高效推理框架，它支持多种路由机制。虽然它不强制模型结构，但在此框架下进行的一些关于 **Dropless MoE** 的研究倾向于非归一化的路由分数或 Sigmoid，以避免强制的概率竞争。

### 总结

| 特性 | Softmax Routing | Sigmoid Routing |
| :--- | :--- | :--- |
| **数学含义** | 概率分布 (和为1) | 独立置信度 (和不固定) |
| **专家关系** | 强竞争 (Coupled) | 独立 (Decoupled) |
| **负载均衡** | 较难，Aux Loss 容易影响性能 | 容易，适合通过 Bias 调整 |
| **代表模型** | **Mixtral 8x7B, Switch Transformer, Grok-1** | **DeepSeek-V2/V3** |
| **DeepSeek 观点** | 容易导致 Collapse，超参敏感 | **Robust，配合 Loss-Free Balancing 效果极佳** |

简单来说，**Softmax 是在做“选择题”（在N个里选最好的K个），而 Sigmoid 是在做“判断题”（对N个专家分别判断是否合格）**。DeepSeek 的成功证明了在 MoE 规模极大时，解耦的“判断题”模式配合 Bias 调整可能是更优的解法。