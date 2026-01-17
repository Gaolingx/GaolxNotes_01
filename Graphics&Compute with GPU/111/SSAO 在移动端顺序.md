# SSAO 在移动端顺序

### 🧑‍💻 User

为什么mobile pipeline中，ssao要放在rending gbuffer之前，而不是之后？

---

### 🤖 Assistant

这个问题包含了一个常见的概念混淆，或者是指向了一种特定的移动端优化管线（Mobile Forward/Forward+ Rendering）。

首先，我们需要澄清一个技术事实：**SSAO（屏幕空间环境光遮蔽）必须依赖于深度缓冲区（Depth Buffer）**，通常还需要法线信息。因此，严格来说，SSAO 无法在没有任何“几何渲染”之前进行。

然而，你提到的“放在 Rendering G-Buffer 之前”，通常指的是在移动端主流的 **Forward Rendering（前向渲染）** 管线中，或者是在 Deferred（延迟渲染）管线的 **Lighting Pass（光照计算）** 之前，引入了一个 **Depth Pre-pass（深度预通）** 阶段。

在移动端管线中，将 SSAO 安排在 Main Shading（主着色/主渲染）之前，通常遵循以下流程：
1.  **Depth Pre-pass (Z-Prepass)**：仅渲染几何体深度。
2.  **SSAO Pass**：利用上一步的深度图计算 AO。
3.  **Main Pass / G-Buffer Pass**：进行主渲染，此时采样 SSAO 结果。

为什么要这样安排？主要原因集中在 **GPU 架构特性（TBDR）**、**带宽限制** 以及 **API 限制** 上。

### 1. 打破读写依赖（Texture Feedback Loop）
这是最直接的技术原因。
*   在移动端常用的 Forward Rendering 中，你是在同一个 Pass 里计算光照并写入颜色缓冲区的。
*   如果要在 Main Pass 里直接计算 SSAO，你需要读取当前像素周围邻居的深度信息。
*   **问题**：在当前的渲染 Pass 中，深度缓冲区（Depth Buffer）正在被写入。图形 API（如 OpenGL ES, Vulkan, Metal）通常禁止在同一个 Pass 中既**写入**一个纹理（作为 Render Target），又在 Shader 中**采样**同一个纹理（作为 Texture）。这会导致未定义的行为或反馈循环错误。
*   **解决**：必须先在一个独立的 Pass（Depth Pre-pass）中把整个屏幕的深度渲染好，生成一张完成的 Depth Texture。然后在下一个阶段（SSAO）读取这张图。

### 2. 移动端 GPU 架构：TBDR 与 带宽 (Bandwidth)
移动端 GPU（如 Adreno, Mali, PowerVR）大多采用 **TBDR（Tile-Based Deferred Rendering，基于图块的延迟渲染）** 架构。带宽是移动端最大的性能瓶颈和发热来源。

*   **如果使用标准延迟渲染（G-Buffer -> SSAO -> Lighting）：**
    *   你需要渲染巨大的 G-Buffer（Albedo, Normal, Specular, Depth 等多张纹理）并写出到系统内存（DRAM）。
    *   这会消耗巨大的写带宽。
    *   然后读取这些纹理计算 SSAO，再消耗读带宽。
    *   **这对手机来说太重了。**

*   **如果使用带 Z-Prepass 的流程（Z-Prepass -> SSAO -> Main Forward）：**
    1.  **Z-Prepass**：非常快，通常只写深度，不写颜色。许多移动 GPU 对这种只有深度的 Pass 有极高的优化（双倍速率等）。
    2.  **SSAO**：计算出一个单通道的 AO 纹理。
    3.  **Main Pass**：
        *   利用 Z-Prepass 得到的深度图进行 **Early-Z** 测试（见下一点）。
        *   在 Shader 中直接读取已经算好的 AO 纹理并应用。
    *   **优势**：这种方式避免了读写多张巨大的 G-Buffer 纹理，节省了带宽。

### 3. 利用 Early-Z 减少 Overdraw（过度绘制）
既然为了 SSAO 必须做一个 Depth Pre-pass，那么这个成本能不能被利用起来？答案是可以。

*   **Overdraw 问题**：在 Main Pass 中，如果一个像素被多次绘制（例如远处的墙被近处的椅子挡住，但先画了墙），Fragment Shader 会运行多次，浪费算力。
*   **优化**：因为我们在第一步已经渲染了完整的深度图（为了给 SSAO 用），我们可以在 Main Pass 中将深度比较函数设置为 `GL_EQUAL` 或 `GL_LEQUAL`，并关闭深度写入（或保持开启视硬件而定）。
*   **结果**：GPU 可以在执行昂贵的 Fragment Shader（光照计算）之前，利用已有的深度图剔除掉所有被遮挡的像素（Hi-Z / Early-Z）。这意味着复杂的 Shader 只有在像素真正可见时才会执行。
*   **结论**：把 SSAO 相关的深度渲染放在前面，实际上“免费”赠送了一个减少 Overdraw 的优化，极大提升了 Main Pass 的性能。

### 4. 异步计算（Async Compute）的可能性
虽然在移动端不如主机端常见，但将 SSAO 独立出来放在前面，理论上允许驱动或硬件进行并行调度。
*   SSAO 是计算密集型（Compute heavy）或纹理采样密集型。
*   Geometry Pass 是顶点处理密集型。
*   在某些架构下，计算 SSAO 的 Compute Shader 可能与后续 Pass 的某些顶点处理并行执行（视具体依赖和硬件支持而定）。

### 总结

在移动管线中，流程通常是：
$$ \text{Depth Pre-pass} \rightarrow \text{SSAO Compute} \rightarrow \text{Main Shading (Forward)} $$

SSAO 放在主渲染（或 G-Buffer）之前的原因是：
1.  **依赖性**：必须先生成深度图才能算 SSAO（不能边画边读）。
2.  **带宽优化**：配合 Forward Rendering 避免使用笨重的 G-Buffer。
3.  **性能优化**：强制进行的 Depth Pre-pass 可以顺便用于 Early-Z 剔除，大幅降低主渲染阶段的 Overdraw。

---

### 🧑‍💻 User

为什么desktop deferred的high quality ssao需要从gbuffer rt中sampling depth/normal？

---

### 🤖 Assistant

在 Desktop 平台的 **Deferred Shading（延迟渲染）** 管线中，实现 **High Quality SSAO（如 HBAO+, GTAO 等）** 时，必须同时采样 **Depth（深度）** 和 **Normal（法线）**，这主要取决于两个核心原因：**数学模型的正确性** 和 **细节的表现力**。

简单来说，只用 Depth 算出来的是“凑合”的阴影，加上 Normal 算出来的才是“正确”的遮蔽。

以下是详细的技术分析：

### 1. 定义采样半球（The Sampling Hemisphere）
SSAO 的核心算法是在当前像素周围随机采样，判断这些采样点是否被遮挡。
*   **半球 vs 球**：对于一个位于表面的点 $P$，遮蔽只应该来自于表面“上方”的空间。因此，我们需要构建一个**半球（Hemisphere）**来进行积分。
*   **法线的作用**：法线 $\vec{N}$ 决定了这个半球的**朝向**。
    *   如果没有法线信息，算法只能假设一个指向摄像机的通用半球，或者使用球体采样。这会导致非物理的变黑（比如一个平坦的地板仅仅因为离摄像机角度倾斜就变黑）。
    *   **High Quality 要求**：必须使用 G-Buffer 中精确的 $\vec{N}$ 来旋转采样核心（Kernel），确保只计算表面法线方向的遮蔽。

### 2. 细节表现力：法线贴图（Normal Mapping）的细节
这是区分 High Quality 和 Low Quality 最直观的地方。

*   **从 Depth 重建法线（Low Quality 做法）**：
    *   如果只读取 Depth，我们可以通过计算深度的导数（`ddx`, `ddy`）来重建法线。
    *   **缺陷**：这样得到的法线是**几何法线（Geometric Normal）**。也就是说，它反映的是三角形面片的朝向。如果一个平坦的墙面上贴了一张复杂的砖块法线贴图（Normal Map），深度图上这面墙依然是平的。
    *   **后果**：SSAO 无法感知砖块之间的缝隙，AO 效果会丢失所有高频细节。

*   **从 G-Buffer 读取法线（High Quality 做法）**：
    *   Deferred G-Buffer 中的 Normal RT 存储的是经过 Fragment Shader 计算后的**像素法线（Pixel Normal）**，它已经包含了**法线贴图（Normal Map）**的扰动信息。
    *   **优势**：SSAO 能够感知到法线贴图里的凹凸细节。即使几何体是一个平面的 Quad，只要法线贴图里有凹痕，HQ SSAO 就能在那条凹痕里生成 AO 阴影。

### 3. 现代算法（HBAO / GTAO）的几何依赖
现代高质量 AO 算法（如 Horizon-Based AO 或 Ground Truth AO）不仅是简单的“数点”，而是基于物理的角度计算。

*   **Horizon-Based AO (HBAO)**：
    *   它的原理是寻找当前像素点在各个方向上的“地平线角度”。
    *   公式极其依赖表面法线 $\vec{N}$ 和视线向量 $\vec{V}$ 的夹角。如果法线不准，地平线积分就会出错，导致遮蔽计算完全错误。

*   **Ground Truth AO (GTAO)**：
    *   GTAO 更是严格拟合了渲染方程中的余弦项：$(\vec{N} \cdot \vec{L})$。
    *   如果没有精确的 G-Buffer 法线，GTAO 这种基于物理近似的算法就无法通过 $\text{cos}(\theta)$ term 来正确加权遮蔽贡献。

### 4. 避免自遮挡伪影（Self-Occlusion Artifacts）
在延迟渲染中，深度缓冲区的精度是有限的（通常 24-bit 或 32-bit float）。

*   **Shadow Acne**：如果仅仅依赖深度进行比较，当采样点非常靠近当前像素时，由于深度的离散化误差，表面可能会错误地遮挡自己（出现麻子一样的黑点）。
*   **法线偏置（Normal Bias）**：为了解决这个问题，高质量 SSAO 会沿着法线方向 $\vec{N}$ 将采样原点稍微向外推一点点。
    $$ P_{\text{sample}} = P_{\text{world}} + \vec{N} \cdot \text{bias} $$
*   如果没有 G-Buffer 中准确的法线，这个偏置方向就会出错，导致平坦表面出现波纹或者物体边缘出现漏光。

### 总结

在 Desktop Deferred Pipeline 中，因为 G-Buffer 早就生成好了（这是 Deferred 的本质特征），读取 Normal RT 的带宽开销是可以接受的。

| 特性 | 仅用 Depth (重建法线) | 使用 G-Buffer Depth + Normal |
| :--- | :--- | :--- |
| **法线来源** | 几何体面片导数 | 包含 Normal Map 的像素法线 |
| **细节程度** | 低（丢失纹理凹凸感） | **高（感知纹理细节）** |
| **半球朝向** | 粗糙，有阶梯感 | 精确，平滑 |
| **适用算法** | 传统 SSAO | HBAO+, GTAO, ASSAO |
| **性能** | 更快（少读一张纹理） | 稍慢（带宽压力大），但画质好 |

因此，**High Quality** 的核心在于：**利用 G-Buffer 中的 Normal RT 来获取包含微表面细节（Normal Map）的精确朝向，从而计算出符合物理直觉的接触阴影。**