# 实时GI中的Final Gather

### 🧑‍💻 User

实时全局光照中的finalgather是什么？

---

### 🤖 Assistant

在实时全局光照（Real-Time Global Illumination, RTGI）的语境下，**Final Gather（最终聚集）** 是渲染管线中的一个关键步骤，它的核心任务是**计算着色点处来自所有方向的间接光照强度的总和**，并将其与材质属性结合，最终输出该像素的颜色。

简而言之，就是“**收集**（Gather）”四面八方反弹过来的光线，“**最终**（Final）”决定这个点看起来应该是什么颜色。

以下是关于实时渲染中 Final Gather 的详细技术解析：

### 1. 核心数学原理：渲染方程

从数学角度来看，Final Gather 实际上是在近似求解**渲染方程**（The Rendering Equation）中的积分部分。

$$ L_o(p, \omega_o) = L_e(p, \omega_o) + \underbrace{\int_{\Omega} f_r(p, \omega_i, \omega_o) L_i(p, \omega_i) (\omega_i \cdot n) d\omega_i}_{\text{Final Gather 处理的部分}} $$

其中：
*   $L_o$：出射光 radiance（最终颜色）。
*   $f_r$：BRDF（材质属性，如漫反射、高光）。
*   $L_i$：入射光 radiance（来自环境或其他物体的反弹光）。
*   $(\omega_i \cdot n)$：几何项（光线角度衰减）。

在实时光照中，直接光（Direct Lighting）通常单独计算，而 Final Gather 专注于计算积分中的**间接光（Indirect Lighting）**部分。

---

### 2. 离线渲染 vs. 实时渲染中的 Final Gather

理解两者的区别对于掌握 RTGI 非常重要：

*   **离线渲染（Offline / Mental Ray / V-Ray）：**
    传统的 Final Gather 是一种“缓存”技术。为了不追踪数百万条光线，渲染器会在场景中稀疏地选择一些点发射光线，计算出辐照度（Irradiance），然后在渲染像素时，通过**插值**这些稀疏点来获得平滑的间接光。

*   **实时渲染（Real-Time / UE5 Lumen / RTX）：**
    由于每帧只有几毫秒的预算，我们无法发射成千上万条光线。实时中的 Final Gather 通常面临“**采样不足**（Undersampling）”的问题。
    *   **策略：** 每像素可能只发射 0.5 到 1 条光线（Ray Tracing）。
    *   **结果：** 得到的结果充满噪点（Noisy）。
    *   **关键步骤：** 因此，实时 Final Gather 几乎总是与**降噪（Denoising）**和**时域累积（Temporal Accumulation）**紧密绑定。

---

### 3. 实时 Final Gather 的工作流程

在现代显卡（支持硬件光追）的管线中，Final Gather 通常包含以下几个阶段：

#### A. 采样与光线追踪 (Sampling & Tracing)
为了收集光照，必须知道光从哪里来。
*   **重要性采样 (Importance Sampling)：** 根据 BRDF（粗糙度、法线）决定光线射向哪里。例如，镜面反射会向反射方向发射光线，漫反射则在半球面上随机发射。
*   **光线求交：** 光线打到场景中的几何体，获取该位置的光照信息（可能是发光体，也可能是上一帧缓存的辐照度）。

#### B.  radiance 估算 (Radiance Estimation)
当光线击中一个表面时，我们获取该表面的亮度。
*   在 **Lumen (UE5)** 中，可能会查询 Surface Cache 或 Voxel Lighting。
*   在 **ReSTIR (Reservoir Spatio-Temporal Importance Resampling)** 技术中，会通过复用时空邻域的样本来找到“最亮”的光源路径，从而极大提高 Gather 的质量。

#### C. 重构与降噪 (Reconstruction & Denoising)
这是实时 Final Gather 最不同于离线渲染的地方。因为采样数极低，原始的 Gather 结果是一张充满噪点的图。
*   **时域累积 (Temporal Accumulation)：** 混合上一帧的结果（利用 Motion Vectors 重投影），将有效采样数从 1 提升到几十。
*   **空域滤波 (Spatial Filtering)：** 利用 SVGF (Spatiotemporal Variance-Guided Filtering) 或 A-SVGF 等算法，根据法线、深度和物体ID，借用周围像素的光照信息来平滑当前像素。

---

### 4. 常见的 Final Gather 实现方案

在实时渲染中，"Final Gather" 这个术语可能指代具体的算法或阶段：

#### 1. 基于探针 (Probe-Based) / DDGI
*   **原理：** 场景中布满了光照探针（Light Probes）。光线追踪更新探针的数据。
*   **Final Gather 过程：** 着色像素不直接发射光线，而是查找其周围的探针，对探针记录的 Irradiance 进行三线性插值。这是一种极其高效的 Final Gather，也是 **DDGI (Dynamic Diffuse Global Illumination)** 的核心。

#### 2. 屏幕空间全局光照 (SSGI)
*   **原理：** 仅利用屏幕上的 G-Buffer 信息。
*   **Final Gather 过程：** 在屏幕空间进行光线步进（Ray Marching），“收集”屏幕上可见的颜色作为间接光。

#### 3. 硬件光线追踪 (Hardware Ray Tracing)
*   **原理：** 真实的 Ray Tracing。
*   **Final Gather 过程：** 直接求解渲染方程积分。通常结合 **ReSTIR GI**。ReSTIR 本质上是一种先进的采样策略，它让 Final Gather 能够用极少的光线（每像素 1 条甚至更少）找到高贡献的光路，从而在实时帧率下实现接近离线质量的路径追踪。

### 5. 总结

在实时全局光照中，**Final Gather** 是连接“光线追踪计算”与“最终屏幕像素”的桥梁。

*   **输入：** G-Buffer（法线、粗糙度、颜色）、场景结构（BVH/SDF）、光源。
*   **动作：** 发射光线（或查询探针/缓存），通过 BRDF 加权，对入射光进行积分。
*   **核心挑战：** 在极低采样数下消除噪点。
*   **输出：** 降噪后的间接光照贴图（Indirect Lighting Map），随后它会与直接光照叠加，形成最终画面。