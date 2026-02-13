# TAAU与Dither孔隙问题

### 🧑‍💻 User

taau可以配合dither做假透明，但是超分前的渲染分辨率低的话孔隙会比较明显是为什么？可以taau之前跑个computepass把带dither alpha的部分mask出来然后采样相邻pixel做个bilateral filter吗？感觉有点类似denoise的搞法但不知道有没有意义

---

### 🤖 Assistant

这是一个非常硬核且在渲染管线优化中经常遇到的问题。你的直觉很敏锐，**Dithered Transparency（网点/抖动透明）在低分辨率渲染（Undersampling）下确实是TAAU/Upscaling的一大噩梦**。

针对你的两个问题，我将分开深度解析：

### 一、 为什么渲染分辨率低，Dither的孔隙会特别明显？

核心原因在于**“高频信号在低采样率下的混叠（Aliasing）”**以及**“像素覆盖面积的物理放大”**。

#### 1. 奈奎斯特频率与信息的丢失
Dither transparency 本质上是一种**极高频的噪声信号**。
假设你使用最经典的 $2 \times 2$ Bayer Matrix 或者 Checkerboard pattern，意味着像素的变化频率是像素级别的（on-off-on-off）。

*   **Native Resolution ($1080p$):** 屏幕上的一个像素对应渲染的一个像素。TAA通过Jitter（抖动）在时间轴上积累历史，每一帧填补不同的空缺，人眼看起来是半透明。
*   **Low Resolution (e.g., $540p$ upscaled to $1080p$):**
    如果你在 $540p$ 下做 Dither，渲染缓冲区的 $1$ 个像素最终会覆盖屏幕上的 $4$ 个像素（$2 \times 2$ area）。
    原本应该是“亚像素”级别的噪点，现在变成了硕大的“马赛克块”。TAAU 在尝试重建时，面对的是巨大的黑白空洞，而不是细腻的噪点。

#### 2. TAAU 的 History Rejection（历史剔除）失效
TAA/TAAU 的核心逻辑是：$$Color_{final} = \alpha \cdot Color_{history} + (1-\alpha) \cdot Color_{current}$$
为了防止重影（Ghosting），TAA 会使用 **Neighborhood Clamping/Clipping**（基于当前帧 $3 \times 3$ 邻域颜色范围来约束历史帧）。

*   在低分辨率下，Dither 产生的孔隙（背景色）和实体（前景色）之间的颜色差异巨大。
*   当分辨率降低，单个“孔隙”在屏幕空间占据的区域变大。当摄像机微动时，History 采样点可能正好落在巨大的“孔隙”里。
*   此时，TAA 认为历史颜色（背景色）和当前颜色（前景色）差异过大，触发 Clamping，**丢弃了历史积累**。结果就是你看到了原始的、低分辨率的、闪烁的网点，而不是融合后的透明色。

---

### 二、 在 TAAU 之前跑一个 Bilateral Filter 有意义吗？

你的想法非常有创意，确实借鉴了 **Ray Tracing Denoising（光追降噪）** 的思路。但在光栅化管线配合 TAAU 的场景下，**这个做法弊大于利，甚至可能导致更严重的画质问题**。

#### 1. 为什么这像 Denoise？
你的直觉是对的。在 Ray Tracing 中，我们发射少量光线得到 noisy 的结果，然后用 SVGF (Spatiotemporal Variance-Guided Filtering) 或 A-SVGF 进行降噪。Bilateral Filter 是其中的空间降噪环节（利用 Normal 和 Depth 保持边缘）。
Dither 也可以看作一种“噪声”。

#### 2. 为什么在 TAAU 前做 Bilateral Filter 通常不可行？

如果按照你的设想：`Render LowRes -> Mask Dither -> Bilateral Filter -> TAAU Upscale`

**存在以下致命缺陷：**

1.  **高频细节被抹杀 (Over-blurring):**
    *   Dither 常用于头发（Hair）、树叶（Foliage）或淡入淡出的物体。这些物体通常包含极高频的几何细节。
    *   在低分辨率下跑 Bilateral Filter（即使带 Depth/Normal 权重），由于采样点太少，很容易把原本属于几何边缘的像素糊成一团。
    *   **后果：** 树叶变成绿色的色块，头发变成片状，TAAU 放大后会显得非常“肉”（Blurry），失去了超分辨率的意义。

2.  **污染背景 (Leaking):**
    *   Dither 的孔隙里是**背景的颜色**。Bilateral Filter 是一种加权平均。
    *   如果你把“前景树叶”和“背景天空”在空间上做了一次 Filter，前景边缘就会带上天空的颜色。
    *   **后果：** 当镜头移动，前景树叶移动了，但它身上带着上一帧混合进来的天空颜色，这会导致严重的 **Smearing (拖影)** 或 **Halo (光环)** 现象。

3.  **破坏 TAA 的亚像素重建:**
    *   TAAU 依赖原始像素的精确性来推断亚像素细节。如果你先手动 Blur 了一遍，相当于人为引入了低通滤波。
    *   TAAU 接收到的是“已经模糊且并不准确”的信号，它无法再通过 Jitter 还原出锐利的边缘。

#### 3. 更有意义的替代方案

如果你必须解决低分辨率下的 Dither 问题，工业界通常采用以下几种策略，而不是前置滤波：

<details>
<summary><strong>方案 A: Blue Noise + TAA 协同 (最主流)</strong></summary>

不要使用规则的 Bayer Matrix，改用 **Blue Noise (蓝噪声)** 或 **Interleaved Gradient Noise**。
*   **原理：** 蓝噪声的频谱特性导致其能量集中在高频，且分布均匀，没有低频聚类。
*   **做法：** 每一帧改变 Dither 的 Pattern（例如加上 Time Offset）。
*   TAA 对高频噪声的积分效果极好。即使分辨率低，随机变化的蓝噪声更容易被 TAA 在时间轴上抹平，而不是像 Bayer 那样形成固定的孔洞。

</details>

<details>
<summary><strong>方案 B: 修改 TAAU 的 Heuristic (针对性修改)</strong></summary>

在 TAA pass 中特殊处理 Dither 像素。
*   **Mask:** 不需要单独的 compute pass，利用 Stencil 或者 G-Buffer 的 Shading Model ID 标记出 Dither 材质。
*   **Logic:** 对于这些像素，**放宽 Neighborhood Clamping 的限制**，或者强制混合更多的 History（增加 $\alpha$ 值）。
*   **代价：** 可能会导致这类物体有更多的 Ghosting，但比大颗粒的噪点看起来要顺眼（变成了半透明的拖尾）。

</details>

<details>
<summary><strong>方案 C: Stochastic Transparency (随机透明)</strong></summary>

这是一种类似你提到的 Denoise 思路，但做法不同。
*   不仅仅是 Dither，而是根据 Alpha 概率随机丢弃片元。
*   在 TAA Resolve 之前，不进行 Spatial Filter，而是依赖 TAA 极强的 Temporal Accumulation。
*   **关键点：** 必须保证 Motion Vector 是极其准确的，且 Pattern 是 Screen Space 固定但随时间 Jitter 的。

</details>

<details>
<summary><strong>方案 D: Alpha to Coverage (A2C) - 硬件方案</strong></summary>

如果性能允许（通常在 Forward 管线或者 MSAA 开启时），使用 A2C。它利用 MSAA 的 sample mask 来模拟透明，效果远好于 Shader 内的 `clip()`，但对于 Deferred Shading + TAAU 的管线，这通常不可用。

</details>

### 总结

**不建议在 TAAU 之前做 Bilateral Filter。**

这就好比你有一张低清图，你先把它高斯模糊了一下，再拿去 AI 放大，出来的结果一定是很糊的。

**推荐的解决思路是：**
1.  **优化 Pattern:** 确保使用高质量的 **Animated Blue Noise**。
2.  **调整 MipBias:** Dither 物体在低分辨率渲染时，Texture MipLevel 可能会选得太高（太糊），导致 Alpha 测试更不稳定。尝试对半透明物体使用负的 Texture LOD Bias。
3.  **TAA 参数调优:** 针对 Dither 区域，在 Shader 里 hack TAA 的混合权重，使其更倾向于相信历史帧（Trust History），以此来修补孔洞。

如果你一定要尝试你的思路，建议做成 **"Dilate Reconstruction"**：只针对 Dither 产生的**空洞（背景像素）**，尝试用相邻的前景像素去填补它（Max Filter 而不是 Average Filter），以此来构造一个“全实心”的低分变率图给 TAAU，但这会导致物体边缘外扩，仍然很难处理。