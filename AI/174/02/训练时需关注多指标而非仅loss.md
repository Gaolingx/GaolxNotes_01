# 训练时需关注多指标而非仅loss

### 🧑‍💻 User

为什么autoencoderkl training中当判别器启动后，光看loss是远远不够的？（应该重点关注psnr, ssim, rfid, disc win rate）

---

### 🤖 Assistant

这是一个非常深刻且切中要害的问题。在训练 AutoencoderKL（例如 Stable Diffusion 使用的 VAE）时，一旦判别器（Discriminator）介入（即 GAN Loss 开始生效），单纯观察 Loss 曲线（L1 Loss, GAN Loss, Total Loss）往往具有极大的欺骗性。

简单来说，这是因为**数学上的最优（Loss 最小）并不等于人类感知上的最优（视觉真实感）**。

以下是为什么“光看 Loss 远远不够”的深层原因分析，以及为什么你提到的那些指标（PSNR, SSIM, rFID, Disc Win Rate）才是真正的指南针。

### 1. 像素级 Loss (L1/L2) 与感知质量的矛盾

在判别器启动前（Warmup阶段），我们通常只优化 L1 或 L2 Loss。
$$ \mathcal{L}_{rec} = ||x - \hat{x}||_1 $$

*   **Loss 的倾向：** L1/L2 Loss 倾向于生成所有可能像素值的“平均值”或“中位数”以最小化误差。
*   **结果：** 这种机制会导致图像**模糊（Blurry）**。虽然轮廓是对的，但高频细节（如发丝、纹理）丢失了。
*   **矛盾点：** 当判别器启动后，GAN Loss 强迫模型生成高频细节。此时，L1 Loss 可能会**上升**（因为生成的纹理可能与原图并未逐像素对齐，虽然看起来更真实），但这实际上是模型变好的表现。如果你只盯着 L1 Loss，你会误以为模型变差了。

### 2. 对抗训练的动态博弈 (Moving Goalposts)

GAN 的训练是一个零和博弈（Zero-Sum Game），生成器（Encoder+Decoder）和判别器在互相进化。

$$ \min_G \max_D V(D, G) $$

*   **动态标尺：** Loss 值的大小取决于对手的强弱。
    *   如果 Generator Loss 很低，可能是因为 Generator 生成得好，也可能是因为 Discriminator 太弱（太蠢），分辨不出真假。
    *   如果 Generator Loss 突然升高，可能是 Discriminator 突然学会了识别某种伪影，这反而能倒逼 Generator 进步。
*   **震荡是常态：** 在 GAN 训练中，Loss 曲线通常是剧烈震荡的。寻找 Loss 的全局最小值没有意义，我们追求的是纳什均衡（Nash Equilibrium）。单纯看数值无法判断是否达到了均衡。

### 3. 感知-失真权衡 (The Perception-Distortion Tradeoff)

这是一个著名的理论（由 Blau & Michaeli 提出）：**失真度（Distortion，如 L1/MSE）和感知质量（Perceptual Quality，如人眼看起来的真实度）是不可兼得的。**

*   **Loss 代表失真度：** 越低的 MSE 往往意味着越差的感知质量（越模糊）。
*   **GAN 代表感知度：** 为了提高纹理真实感，必然会在一定程度上牺牲像素级的对齐（增加 MSE）。
*   因此，当 GAN Loss 介入时，Total Loss 的数值并不直接对应图像质量的线性提升。

---

### 4. 关键指标深度解读

既然 Loss 不靠谱，我们需要多维度的指标来监控训练状态。你列出的指标正好涵盖了不同的侧面：

<details>
<summary><strong>1. Disc Win Rate / Logits (判别器胜率/输出值) - 最重要的健康指标</strong></summary>

这是监控训练是否崩塌（Collapse）的最直接手段。

*   **含义：** 判别器判定真实图片为“真”以及重建图片为“假”的准确率。
*   **理想状态：** 我们希望它处于“拉锯战”状态。
    *   对于 PatchGAN，通常监控 Logits 输出是否在 $0$ 附近震荡，或者 Win Rate 在某个非 0 或 1 的区间波动。
*   **危险信号：**
    *   **Win Rate $\approx 1.0$ (或 100%)**：判别器太强了，瞬间秒杀生成器。生成器梯度消失，学不到东西，输出变成纯噪声或伪影。
    *   **Win Rate $\approx 0.0$**：判别器太弱或坏掉了，生成器可以随便生成垃圾都能骗过它。
    *   **Logits 持续发散**：数值越来越大，说明梯度爆炸。

</details>

<details>
<summary><strong>2. rFID (Reconstruction FID) - 整体感知质量</strong></summary>

*   **含义：** 计算原图（Ground Truth）分布与重建图（Reconstruction）分布之间的 Fréchet Inception Distance。
*   **作用：** 它是衡量**“像不像真实图片”**的黄金标准。
*   **为什么重要：** L1 Loss 低但模糊的图，FID 会很高（差）。加入了 GAN Loss 后，虽然 L1 可能变高，但只要纹理逼真，FID 会显著下降（好）。这是判断 GAN Loss 权重是否合适的关键。

</details>

<details>
<summary><strong>3. PSNR (峰值信噪比) & SSIM (结构相似性) - 结构保真度</strong></summary>

*   **PSNR：** 基于 MSE 的对数指标。
*   **SSIM：** 衡量亮度、对比度和结构的相似度。
*   **作用：** 它们主要充当**“锚点”**。
    *   Autoencoder 的任务不仅仅是生成好看的图，还必须**忠实还原**原图。
    *   如果 rFID 很好（图很清晰），但 SSIM 暴跌，说明模型开始“胡编乱造”了（Hallucination），生成的细节虽然清晰但和原图不一致。这对于 VAE 来说是不可接受的。

</details>

<details>
<summary><strong>4. Codebook Usage / KL Divergence - 隐空间健康度</strong></summary>

*   **KL Divergence (对于 VAE)：** 衡量 Latent 分布与标准正态分布的差异。
*   **Codebook Usage (对于 VQ-GAN)：** 衡量有多少 Embedding 被使用了。
*   **为什么重要：**
    *   如果 KL Loss 对应的权重太小，Latent Space 会退化，虽然重建图极好（PSNR高），但你也无法用它来训练下游的 Diffusion Model（因为 Latent 分布不规则）。
    *   这是一个**隐性地雷**，光看重建图看不出来，必须监控 Latent 的统计量。

</details>

### 总结

在 AutoencoderKL 的 GAN 训练阶段，评价体系应如下分层：

1.  **第一优先级（稳定性）：Disc Win Rate / Logits**。确保训练没有崩塌，两者在博弈。
2.  **第二优先级（感知质量）：rFID**。确保 GAN 使得图片变得真实、清晰。
3.  **第三优先级（内容一致性）：PSNR / SSIM**。确保图片没有严重变形，内容与原图一致。
4.  **第四优先级（隐空间质量）：KL Loss 值 / Std / Mean**。确保 Latent Space 符合正态分布，为下游任务做准备。
5.  **参考级：Reconstruction Loss (L1)**。仅供参考，不作为主要判断依据。

**最终的 "Ground Truth"：** 哪怕所有指标都好看，每隔几千个 step **人工肉眼检查（Visual Inspection）** 验证集的一批重建图依然是不可替代的。因为指标是对平均值的统计，而人眼能发现特定的伪影（如棋盘格效应 Checkerboard artifacts）。