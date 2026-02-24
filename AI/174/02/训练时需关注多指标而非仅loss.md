# 训练时需关注多指标而非仅loss

### 🧑‍💻 User

为什么autoencoderkl training中当判别器启动后，光看loss是远远不够的？（应该重点关注psnr, ssim, rfid, disc win rate）

---

### 🤖 Assistant

这是一个非常深刻且切中要害的问题。在训练 AutoencoderKL（VAE + GAN，常见于 Stable Diffusion 的第一阶段训练）时，单纯依赖 Loss 曲线不仅不够，甚至极具**误导性**。

简单来说，这是因为 **Adversarial Loss（对抗损失）的引入改变了优化的博弈性质，且像素级损失（Pixel-level Loss）无法衡量“真实感”。**

以下从数学原理、对抗博弈和评价指标三个维度详细解析原因。

---

### 1. 为什么 Loss 会骗人？

AutoencoderKL 的总 Loss 通常由三部分组成：
$$L_{total} = L_{rec} + \lambda_{kl} L_{kl} + \lambda_{adv} L_{G}$$

#### A. 对抗博弈的“非收敛”特性
当判别器（Discriminator）启动后，训练变成了一个极小极大博弈（Min-Max Game）：
*   **Generator (VAE):** 试图最小化 $L_{G}$（骗过判别器）。
*   **Discriminator:** 试图最大化识别率。

**Loss 的欺骗性在于：**
*   **生成器 Loss 升高不代表变差：** 可能是判别器变强了，变得更挑剔了。
*   **生成器 Loss 降低不代表变好：** 可能是判别器变弱了（Collapse），或者生成器找到了对抗样本的捷径（例如生成高频噪声来愚弄判别器，但人眼看起来很糟糕）。
*   **动态平衡：** 在理想的 GAN 训练中，Loss 往往是震荡的，而不是像普通分类任务那样单调下降。

#### B. 像素损失（L1/L2）偏爱“模糊”
$L_{rec}$（通常是 L1 或 L2）倾向于生成所有可能模式的**平均值**。
*   **现象：** 为了降低 MSE（均方误差），模型倾向于输出模糊的灰色或平滑纹理，因为模糊图像在像素统计上比稍微偏移的锐利图像 Loss 更低。
*   **矛盾：** 判别器的作用正是为了惩罚模糊，强制模型生成高频纹理。此时，$L_{rec}$ 和 $L_{adv}$ 可能会在某种程度上“打架”。光看 Total Loss 无法分辨模型是在变锐利（$L_{adv}$ 占优）还是在变模糊（$L_{rec}$ 占优）。

---

### 2. 关键指标详解：为什么要关注这些？

既然 Loss 不可靠，我们需要解耦的指标来分别监控“还原度”、“真实感”和“训练稳定性”。

#### A. 还原度指标 (PSNR, SSIM)
这两个指标主要监控 $L_{rec}$ 的效果，确保图片内容没有丢失。

<details>
<summary><strong>展开查看：PSNR 与 SSIM 的局限性</strong></summary>

*   **PSNR (Peak Signal-to-Noise Ratio):**
    *   衡量像素级的误差。
    *   **局限：** 对纹理不敏感。一张模糊的图 PSNR 可能很高，但看起来很假。
*   **SSIM (Structural Similarity Index):**
    *   衡量结构、亮度和对比度的相似性。
    *   **作用：** 确保 VAE 重建出来的图，结构上（轮廓、物体位置）和原图是一致的。
    *   **警告：** 如果 SSIM 很高但 rFID 很差，说明图片轮廓对了，但纹理像塑料或者有伪影。
</details>

#### B. 感知与真实感指标 (rFID, LPIPS)
这是对抗训练（GAN）的核心监控指标。

*   **rFID (Reconstruction FID):**
    *   计算真实图片特征分布与重建图片特征分布之间的距离（Fréchet Inception Distance）。
    *   **为什么重要：** 它是目前衡量“看起来像不像真图”的黄金标准。Loss 再低，如果 rFID 不降，说明模型生成的纹理不自然。
*   **LPIPS (Learned Perceptual Image Patch Similarity):**
    *   利用 VGG/AlexNet 提取特征计算距离。
    *   **作用：** 比 L1/L2 更符合人类视觉感知。如果 LPIPS 上升，说明图片在人眼看来变差了。

#### C. 训练稳定性指标 (Disc Win Rate / Logits)
这是监控 GAN 是否崩溃（Collapse）的关键，**比 Loss 重要得多**。

*   **Disc Win Rate (判别器胜率):** 指判别器正确区分真假图片的比例。
    *   **理想状态：** 应该在 **0.5 (50%)** 附近震荡，或者在某些设定下（如 Hinge Loss）保持在一定范围内。
    *   **Win Rate $\to$ 1.0 (100%)：** **危险！** 判别器太强了，生成器完全被碾压，梯度消失，生成器学不到任何东西（Loss 可能甚至不降反升）。
    *   **Win Rate $\to$ 0.0 (0%)：** **危险！** 判别器太弱或崩溃，生成器随便生成什么垃圾都能骗过它。
*   **Logits (判别器输出值):**
    *   观察 Real Logits 和 Fake Logits 的均值分离度。如果两者完全重合，说明判别器没在工作；如果两者距离无穷大，说明梯度可能爆炸。

---

### 3. 总结：如何通过指标判断训练状态？

在 AutoencoderKL 训练中，当判别器启动（通常在几千 step 后 warmup 结束），你应该观察以下组合：

| 现象 | Loss 表现 | Disc Win Rate | rFID / LPIPS | 实际情况诊断 |
| :--- | :--- | :--- | :--- | :--- |
| **理想情况** | 震荡或缓慢下降 | **~0.4 - 0.6** | **持续下降** | 判别器和生成器势均力敌，正在互相进步，纹理越来越清晰。 |
| **判别器过强** | Generator Loss 飙升 | **> 0.95** | 不变或变差 | 判别器太早收敛，生成器梯度消失。需减小判别器 LR 或增加生成器更新频率。 |
| **判别器崩溃** | Generator Loss 极低 | **< 0.1** | 变差 (产生伪影) | 判别器失效，生成器在生成“对抗样本”（High-freq noise）。 |
| **模式崩塌** | L1 Loss 下降 | ~0.5 | **变差** | 模型只学会了输出平均色或单一纹理，虽然像素误差低，但没有细节。 |
| **KL 权重过大** | KL Loss 低 | N/A | rFID 差 | **Posterior Collapse**。Latent Space 变成了标准正态分布，但丢失了图像信息，重建图全是噪声。 |

### 结论
在 AutoencoderKL 的对抗训练阶段：
1.  **Loss 仅作为参考**，确保没有出现 NaN 或数量级爆炸。
2.  **首要关注 Disc Win Rate / Logits**，确保“有的练”（博弈平衡）。
3.  **最终决定权在 rFID 和 LPIPS**，它们代表了真正的图像质量。
4.  **人工检查 (Visual Inspection)**：必须定期把重建图 dump 出来看。有时候指标都很好，但图片里有奇怪的棋盘格伪影（Checkerboard Artifacts），这是数据指标很难反映出来的。