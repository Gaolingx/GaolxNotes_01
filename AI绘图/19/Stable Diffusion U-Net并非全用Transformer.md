# Stable Diffusion U-Net并非全用Transformer

### 🧑‍💻 User

Stable Diffusion的U-Net 每一个down/up sample block是resnet+transformer组成的吗？

---

### 🤖 Assistant

简短的回答是：**不是的**。Stable Diffusion (SD) 的 U-Net 并不是每一个 Downsample (下采样) 或 Upsample (上采样) 模块都包含了 Transformer。

在 Stable Diffusion 的架构中，出于对**计算资源（显存和算力）的权衡**，U-Net 的 Block 实际上被分为了两类：**纯 ResNet 模块** 和 **ResNet + Transformer 模块**。

以下是具体的架构解析：

### 1. 两种不同的 Block 类型

在 U-Net 的 Encoder（下采样）和 Decoder（上采样）中，主要交替使用以下两种类型的 Block：

*   **`DownBlock2D` / `UpBlock2D` (仅 ResNet)：**
    这类模块内部**只有**多个 ResNet 模块（通常包含 `GroupNorm`、`SiLU` 激活函数和 `Conv2d`）。它们只负责在空间维度上提取局部特征并进行特征图大小的缩放，**完全不包含**注意力机制（Transformer）。
*   **`CrossAttnDownBlock2D` / `CrossAttnUpBlock2D` (ResNet + Transformer)：**
    这类模块才是你提到的组合。它们在每个 ResNet 模块之后，接入了一个 `SpatialTransformer`。这个 Transformer 包含：
    1.  **Self-Attention（自注意力）：** 用于在图像的全局范围内捕捉像素之间的依赖关系。
    2.  **Cross-Attention（交叉注意力）：** 这是 SD 能够听懂文本指令的核心！它将文本的 Token Embeddings 作为 Key 和 Value，图像特征作为 Query，从而将文本条件注入到图像生成过程中。
    3.  **Feed-Forward Network (FFN)：** 标准的 Transformer 前馈网络。

### 2. 为什么不全用 Transformer？

这完全是由 Transformer 中注意力机制的计算复杂度决定的。Self-Attention 的计算复杂度与序列长度（在这里是图像的像素数量）的平方成正比：$O(N^2)$。

*   在最高分辨率层（Latent 空间大小为 $64 \times 64$），序列长度 $N = 64 \times 64 = 4096$。如果在这里应用 Transformer，计算量和显存占用将极其庞大。
*   在较低分辨率层（如 $32 \times 32$ 或 $16 \times 16$），序列长度骤降为 $N = 1024$ 或 $N = 256$。此时使用 Transformer 既能有效捕捉全局语义，又能控制计算成本。

因此，**在特征图分辨率最大的层级，SD 会省略 Transformer 结构，仅使用 ResNet。**

---

<details>
<summary><b>展开查看：Stable Diffusion v1.5 U-Net 的具体层级结构分布</b></summary>

以经典的 Stable Diffusion v1.5 为例，输入 Latent 图像尺寸为 $64 \times 64$，其 U-Net 结构分为 4 个 Down 层、1 个 Mid 层、4 个 Up 层。你可以清晰地看到 Transformer 在何处被使用：

**Encoder (Downsample Stages):**
1.  **Stage 1 ($64 \times 64$):** `DownBlock2D` (仅 ResNet，不含 Transformer，节约显存)
2.  **Stage 2 ($32 \times 32$):** `CrossAttnDownBlock2D` (ResNet + Transformer)
3.  **Stage 3 ($16 \times 16$):** `CrossAttnDownBlock2D` (ResNet + Transformer)
4.  **Stage 4 ($8 \times 8$):** `DownBlock2D` (仅 ResNet。*注：此分辨率的特征提取交由紧接着的 Mid Block 处理*)

**Middle Block:**
*   **Mid ($8 \times 8$):** `UNetMidBlock2DCrossAttn`，结构为：`ResNet` $\rightarrow$ `SpatialTransformer` $\rightarrow$ `ResNet`。

**Decoder (Upsample Stages):**
*   Decoder 的结构是对称的，接收来自 Encoder 的 Skip-Connection（跳跃连接）：
1.  **Stage 1 ($8 \times 8$):** `UpBlock2D` (仅 ResNet)
2.  **Stage 2 ($16 \times 16$):** `CrossAttnUpBlock2D` (ResNet + Transformer)
3.  **Stage 3 ($32 \times 32$):** `CrossAttnUpBlock2D` (ResNet + Transformer)
4.  **Stage 4 ($64 \times 64$):** `UpBlock2D` (仅 ResNet)

</details>

### 总结

Stable Diffusion 的 U-Net 是一个混合架构：
对于高分辨率特征图，模型依赖 `ResNet` 提取局部纹理；对于经过下采样后的低分辨率特征图，模型才使用 `ResNet` + `Transformer` 的组合来进行全局语义理解和文本条件（Prompt）的注入。

---

### 🧑‍💻 User

和后续的dit diffusion架构（flux/sd3）有什么区别？

---

### 🤖 Assistant

从经典的 U-Net（如 SD 1.5 / SDXL）到后续基于 **DiT (Diffusion Transformer)** 的架构（如 Stable Diffusion 3, FLUX, Sora），是图像生成领域一次**根本性的底层架构范式转换**。

简单来说，核心区别在于：**抛弃了基于卷积网络（CNN）的层级缩放结构（Down/Up Sample），全面拥抱像大语言模型（LLM）一样平铺直叙的纯 Transformer 架构。**

以下是 U-Net 与以 FLUX/SD3 为代表的 DiT 架构的 4 个核心区别：

### 1. 宏观结构：层级分辨率 vs 恒定序列长度
*   **U-Net (SD 1.5 / SDXL)：** 依赖空间特征图的放大与缩小。模型包含 `DownBlock`（下采样提取特征，减小分辨率）和 `UpBlock`（上采样恢复分辨率）。这是一种针对图像设计的**归纳偏置（Inductive Bias）**。
*   **DiT (SD3 / FLUX)：** 完全**没有**下采样和上采样模块。它借鉴了 ViT (Vision Transformer) 的思路：
    1.  把输入的 Latent 图像切分成一个个小图块（Patch，例如 $2 \times 2$）。
    2.  把这些图块展平（Flatten）成一个一维的“Token序列”（就像自然语言处理中的一个个单词）。
    3.  在这个固定长度的序列上，堆叠几十层结构完全相同的 Transformer Block。从第一层到最后一层，序列长度和特征维度**始终保持不变**。

### 2. 文本与图像融合机制：Cross-Attention vs MM-DiT / Joint-Attention
这是 SD3 和 FLUX 文本理解能力（Prompt 跟随能力）远超前代的核心原因。

*   **U-Net：** 图像特征占据主导地位。图像作为 Query (Q)，文本 Token 作为 Key (K) 和 Value (V) 进行单向的 `Cross-Attention`。文本仅仅是作为“外部条件”注入到图像生成过程中，两者是“主从关系”。
*   **SD3 / FLUX：** 采用**多模态扩散 Transformer (MM-DiT)**。模型将文本 Token 序列和图像 Patch 序列拼接或并行处理。在 Transformer 内部，文本和图像进行**双向联合注意力计算（Joint Attention）**。
    *   这意味着不仅图像可以关注文本（画什么），文本的隐向量在这个过程中也会动态更新，去关注图像（画到哪了）。两者变成了地位平等的“对等关系”。

### 3. 条件注入机制：ResNet 附加 vs adaLN (自适应层归一化)
步数（Timestep）对于扩散模型至关重要，模型需要知道当前处于去噪的哪个阶段。
*   **U-Net：** 时间步（Time Embedding）通常通过简单的加法或缩放操作，注入到每一个 ResNet 模块中。
*   **DiT 架构：** 广泛采用 **`adaLN` (Adaptive Layer Normalization) 或 `adaLN-Zero`**。时间步（以及全局文本特征）被送入一个线性层，直接动态生成 Transformer Block 中所需的缩放（Scale）、平移（Shift）和门控（Gate）参数。这是一种非常强力且优雅的条件约束方式，对控制 Transformer 生成极其有效。

### 4. 扩展性法则 (Scaling Laws)
*   **U-Net：** 很难预测扩大模型规模后的效果。强行增加卷积的通道数或深度，往往面临收益递减甚至训练不稳定的问题。这也是为什么 SDXL 止步于 2.6B 参数量就很难再单纯靠加深网络来提升。
*   **DiT 架构：** 完美继承了 LLM 的 **Scaling Laws（扩展定律）**。只要堆更多的算力、给更大的模型参数（FLUX 达到了 12B 参数）、喂更多的数据，它的生成质量（FID 指标）就能呈近乎线性的可预测提升。

---

<details>
<summary><b>🛠️ 展开查看：SD3 / FLUX 的内部 Block 到底长什么样？(MM-DiT 结构解析)</b></summary>

因为不再使用 `DownBlock` / `UpBlock`，SD3 和 FLUX 内部其实是堆叠了几十个同质化的 **Joint Transformer Block (联合 Transformer 模块)**。

以 SD3 的 MM-DiT 模块为例，由于文本和图像的模态差异很大，直接把它们混在一起过同一个线性层效果不好。因此，在每一个 Block 内部，执行了以下极具创新的步骤：

1.  **独立归一化与投影：** 文本 Token 和图像 Token 分别经过**独立**的 `LayerNorm` 和线性投影层（生成各自的 $Q, K, V$）。
2.  **拼接 (Concatenation)：** 将文本的 $Q, K, V$ 和图像的 $Q, K, V$ 拼接在一个超长的序列维度上。
3.  **联合注意力 (Joint Self-Attention)：** 在这个拼接后的超长序列上，统一执行**单次**的 Self-Attention 计算。在这里，文本 Token 和图像 Token 发生了密集的信息交换（图像可以注意文本，文本也可以注意图像）。
4.  **拆分 (Split)：** 计算完成后，再把文本序列和图像序列切开。
5.  **独立前馈网络 (Separate MLPs)：** 切开后的文本特征和图像特征，分别输入到各自专属的 MLP（多层感知机）中进行非线性映射。

**FLUX 进一步的演进：**
FLUX 的前段部分沿用了类似 SD3 的 MM-DiT (Double Stream) 双流架构，但在后半段（为了节省计算量并进一步融合），它引入了 **Single Stream Block (单流模块)**。在后半段，文本和图像 Token 不仅在 Attention 层合并计算，甚至直接通过同一个 MLP 处理，实现了更深度的多模态特征融合，这也是它无需依赖复杂负向提示词就能生成高细节图像的秘密。

</details>

### 总结
你提到的 U-Net 是一种 **“卷积为主，注意力为辅”** 的混合网格，靠缩放空间分辨率来节约计算量；而后续的 SD3 / FLUX 是 **“万物皆序列，纯注意力驱动”** 的现代大模型架构。这种转变虽然对显存提出了更高的要求（比如对显存敏感的 $O(N^2)$ 注意力计算遍布所有层），但换来了断崖式领先的文本理解力、画面细节和极具潜力的可扩展性。