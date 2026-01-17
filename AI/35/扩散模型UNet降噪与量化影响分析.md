# 扩散模型UNet降噪与量化影响分析

### 🧑‍💻 User

为什么int4 groupwise/channelwise量化对于transformer结构的llm影响相对较小，对于conv based unet diffusion模型sampling生成质量影响较大？

---

### 🤖 Assistant

这是一个非常深刻且前沿的问题。虽然Transformer（LLM）和ConvNet（Diffusion UNet）都是深度神经网络，但由于它们的**任务本质（Task Nature）**、**模型架构（Model Architecture）**以及**误差传播机制（Error Propagation）**存在显著差异，导致Int4量化（特别是Weight-only quantization）对两者的影响截然不同。

以下是导致这种现象的几个核心原因：

### 1. 任务本质：离散分类 vs. 连续回归

这是最根本的区别。

*   **LLM (Transformer): 下一个Token预测 (Classification)**
    *   LLM的本质是分类任务。输出层是一个巨大的Softmax，用于预测下一个token的概率分布。
    *   **容错性高**：假设正确的logit是5.0，第二高的是4.5。量化引入的噪声可能让5.0变成了4.9，4.5变成了4.6。只要**相对顺序（Ranking）**不变，或者Top-K的集合变化不大，生成的文本依然是连贯且正确的。
    *   **离散性**：Token的选择是离散的（argmax或sampling）。一旦选定了Token，量化的微小数值误差在这一步就被“重置”了（虽然Hidden State会传递误差，但Token ID是绝对的）。

*   **Diffusion (UNet): 噪声预测 (Regression)**
    *   Diffusion模型的UNet执行的是回归任务。它需要精确预测添加到Latent上的噪声 $\epsilon$（或者预测速度 $v$）。
    *   **容错性低**：这是一个连续数值的逼近问题。量化引入的量化误差（Quantization Error）直接表现为预测出的噪声数值偏差。
    *   **视觉敏感**：图像生成的质量对像素/Latent值的微小偏移非常敏感。噪声预测的系统性偏差（Bias）会导致图像整体颜色偏移、纹理模糊或结构崩塌。

### 2. 误差传播机制：自回归 vs. 迭代去噪

*   **LLM (Auto-regressive)**
    *   LLM是逐个生成Token。虽然误差会积累，但如前所述，离散的Token选择步骤起到了一个“滤波器”的作用。
    *   此外，LLM通常有过参数化（Over-parameterization）现象，模型内部往往存在冗余的神经元可以补偿精度损失。

*   **Diffusion (Iterative Denoising)**
    *   Diffusion Sampling是一个由$T$步组成的马尔可夫链过程（例如 $T=20$ 到 $50$ 步）。
    *   **轨迹偏离**：第 $t$ 步的输出直接作为第 $t-1$ 步的输入。如果在第 $t$ 步因为Int4量化产生了一个微小的方向偏差，这个偏差会在后续的几十步中被反复放大，导致最终生成的Latent向量偏离了真实的数据流形（Manifold）。
    *   **无自我纠错**：不像LLM每一步都有明确的Token ID作为锚点，Diffusion在采样过程中是在高维空间游走，一旦偏航，很难自动纠正回来。

### 3. 架构与权重分布：GEMM vs. Convolution

Int4 Groupwise量化通常针对权重的通道进行分组。

*   **Transformer (Linear Layers / GEMM)**
    *   **权重特征**：Transformer的主要计算在于巨大的全连接层（Linear）。权重矩阵 $W$ 形状通常很大（如 $4096 \times 11008$）。
    *   **Groupwise有效性**：研究发现LLM的激活值存在明显的“离群通道”（Outlier Channels），但权重分布相对平滑。通过Groupwise（例如 group size=128）量化，可以很好地捕捉局部权重的动态范围。
    *   **参数冗余**：LLM参数量巨大（7B - 175B），单个权重的精度损失可以通过整体的协同作用被平均化。

*   **UNet (Convolutional Layers)**
    *   **权重特征**：UNet的核心是卷积层。卷积核通常很小（如 $3 \times 3$）。
    *   **结构敏感性**：一个 $3 \times 3$ 的卷积核只有9个空间参数。将其强行压缩到Int4（只有16个可能的数值），会极大地破坏其提取边缘、纹理等高频细节的能力。
    *   **感受野与局部性**：卷积操作强依赖于局部特征。Conv层的权重分布往往比Linear层更复杂，且不同Channel之间的方差可能极大。简单的Groupwise量化可能无法同时保留卷积核的**幅度信息（Magnitude）**和**相位/方向信息**。
    *   **参数量**：Stable Diffusion XL (SDXL) 的UNet大约只有2.6B参数，SD 1.5只有860M。相比LLM，它的参数更少，每个参数承载的信息密度更高，因此对精度下降更敏感。

### 4. 激活值的分布差异

虽然问题主要问的是Int4量化（通常指Weight Int4），但激活值的特性也会影响Weight量化的效果（因为 $Y = W \cdot X$）。

*   **Transformer**: 激活值虽然有离群值，但通常集中在特定的Channel上。现在的量化算法（如AWQ、GPTQ）会针对这些由于激活值离群导致的敏感权重进行特殊保护（保留FP16或调整Scale）。
*   **Diffusion**: UNet中的激活值（特别是经过SiLU/ReLU后的Feature Map）在不同时间步（Timestep $t$）下的分布变化极大。
    *   在采样初期（高噪声，$t$ 很大），激活值分布宽广。
    *   在采样末期（低噪声，$t$ 很小），激活值分布完全不同。
    *   静态的Int4权重量化很难适应这种跨越整个采样周期的动态激活分布变化。这就是为什么Diffusion模型往往需要**QAT（Quantization Aware Training）**或者**PTQ Calibration with Temporal Data**（使用不同时间步的数据进行校准）才能在低比特下工作。

### 总结

数学表达上的直观对比：

对于 **LLM**，我们寻找的是：
$$ \text{Token}_{t+1} = \text{argmax}(P(x| \text{Context})) $$
$\text{argmax}$ 操作对 $P$ 的微小扰动 $\delta$ 是鲁棒的。

对于 **Diffusion**，采样更新公式（简化版 DDIM）类似于：
$$ x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \dots $$
这里 $\epsilon_\theta$ 是量化后的模型输出。如果 $\epsilon_\theta$ 有误差 $\Delta \epsilon$，那么 $x_{t-1}$ 就有直接的数值误差。这个误差会作为下一次迭代的输入 $x_{t-1}$，导致误差累积：
$$ \text{Error}_{total} \approx \sum_{t=T}^{1} \text{Scale}_t \cdot \Delta \epsilon_t $$

因此，Int4 Groupwise量化虽然在LLM中不仅能用而且效果很好，但在Diffusion模型中直接使用往往会导致生成的图像出现严重的噪点、色块或结构扭曲。Diffusion模型通常需要更精细的混合精度策略（如关键层FP16，其余Int8）或专门的QAT才能在低比特下保持质量。