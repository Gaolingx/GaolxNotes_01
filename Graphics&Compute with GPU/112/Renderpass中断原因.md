# Renderpass中断原因

### 🧑‍💻 User

为什么类似ssr这种需要采样前一renderpass渲染结果的post process effect会导致subpass中断，导致必须另起一个新的renderpass？

---

### 🤖 Assistant

这是一个非常经典且触及现代图形API（Vulkan/Metal/DX12）核心设计理念的问题。简单来说，核心原因在于**数据的访问模式（Access Pattern）**与**GPU架构（特别是TBDR架构）的片上内存优化机制**之间的冲突。

为了理解为什么 SSR（Screen Space Reflections）会打断 Subpass，我们需要先理解 **RenderPass/Subpass 的设计初衷** 以及 **SSR 算法对数据的需求**。

以下是详细的深度解析：

### 1. Subpass 的核心机制：Tile-Based 与 片上内存

在现代渲染架构中（特别是移动端的 TBDR 架构，以及桌面端显卡的某些优化模式），RenderPass 和 Subpass 的存在是为了利用 **On-chip Memory（片上内存/Tile Memory）**。

*   **RenderPass 的生命周期**：
    一个 RenderPass 通常代表了一次完整的“加载 -> 绘制 -> 存储”流程。在 TBDR 架构下，GPU 会将屏幕分割成许多小块（Tiles）。
*   **Subpass 的黑魔法**：
    Subpass 允许你在同一个 RenderPass 内进行多个阶段的渲染（例如 Deferred Shading 中的 G-Buffer 写入 $\rightarrow$ Lighting 计算）。
    *   **关键特性**：Subpass 之间的数据传递是通过 **Input Attachment** 进行的。
    *   **Input Attachment 的限制**：它**只能**读取当前像素（`gl_FragCoord.xy`）对应的数据。
    *   **优势**：因为只读当前像素，GPU 不需要把 G-Buffer 写回显存（System Memory/VRAM），而是直接保留在极快的片上内存（On-chip Memory）中供下一个 Subpass 使用。这极大节省了带宽。

### 2. SSR 的算法需求：全局随机访问（Global Random Access）

SSR 的原理是在屏幕空间进行光线步进（Ray Marching）。

*   假设我们正在渲染像素 $P(x, y)$ 的反射。
*   根据法线和视线方向，我们发射一条光线。
*   这条光线可能会击中屏幕上任意位置的像素 $P(x', y')$。

**这就是矛盾所在：**

1.  **坐标不确定性**：在计算像素 $(x, y)$ 时，你需要读取的数据可能位于 $(x+100, y+50)$ 或者屏幕的另一端。
2.  **跨 Tile 访问**：在 TBDR 架构中，当 GPU 正在处理 Tile A 时，Tile B 的数据可能还没渲染出来，或者已经渲染完并写回了显存，并不在当前的片上内存中。

### 3. 为什么必须打断？（技术细节）

由于上述矛盾，导致了以下几个必须终止当前 RenderPass 的理由：

#### A. Input Attachment 的局限性
Subpass 依赖 `Input Attachment` 来保持高效。Vulkan 规范明确规定：**Input Attachment 在 Fragment Shader 中只能读取与当前被着色像素位置完全相同的 texel。**
SSR 需要读取邻域甚至远处的像素，这违反了 Input Attachment 的定义。因此，SSR 无法作为 Subpass 使用 Input Attachment 读取上一阶段的 Color/Depth。

#### B. 纹理采样（Texture Sampling）即读写依赖
既然不能用 Input Attachment，那就把上一阶段的结果当作普通 **Sampled Texture** 来读行不行？
可以，但这导致了**读写依赖（Read-After-Write Hazard）**：

1.  **Pass 1 (G-Buffer/Main Pass)**：正在写入 Color Buffer 和 Depth Buffer。
2.  **Pass 2 (SSR)**：需要采样（读取）Pass 1 输出的整个纹理。

在 GPU 中，为了保证数据一致性，必须确保 **Pass 1 的所有像素都已经完全写入到显存（VRAM）中**，Pass 2 才能开始读取。
*   如果试图在同一个 RenderPass 的 Subpass 中把 Color Attachment 当作 Sampled Texture 读取，会造成反馈循环（Feedback Loop），这在大多数 API 中是未定义行为或非法的。
*   **必须的操作**：结束 Pass 1 $\rightarrow$ **Memory Barrier（内存屏障）** $\rightarrow$ 开始 Pass 2。

#### C. 资源布局转换（Layout Transition）
在图形 API（如 Vulkan）中，纹理是有“状态”的：
*   **渲染时**：图片处于 `VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL`（适合写入）。
*   **SSR 采样时**：图片需要处于 `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`（适合纹理单元读取）。

Subpass 虽然支持 Layout 转换，但前提是基于 Attachment 的行为。由于 SSR 需要全屏随机采样（Sampled Image），这意味着整个 Image 必须在内存中完成从“写状态”到“读状态”的转换。这种全局的同步和转换标志着一个 RenderPass 的结束。

### 4. 总结与对比

我们可以用一个表格来总结为什么 Lighting Pass 可以是 Subpass，而 SSR 必须是 New Pass：

| 特性 | Deferred Lighting (Subpass) | SSR (New RenderPass) |
| :--- | :--- | :--- |
| **数据来源** | G-Buffer (Albedo, Normal, Depth) | Color Buffer, Depth Buffer (上一帧或当前帧结果) |
| **访问坐标** | 仅当前像素 $(x, y)$ | 任意像素 $(x', y')$ (Ray Marching) |
| **内存位置** | **On-chip Tile Memory** (极快，无需带宽) | **VRAM** (必须从显存采样) |
| **同步需求** | 无需全局同步，Tile 内部同步即可 | **全局同步** (整张图必须画完才能采样) |
| **机制** | Input Attachment | Sampled Texture |

### 结论

SSR 导致 Subpass 中断并开启新 RenderPass 的根本原因是：**SSR 算法需要对上一阶段的渲染结果进行全局、随机的纹理采样（Global Random Access），这破坏了 Subpass 依赖的“Tile 局部性”和“片上内存驻留”机制。**

因此，必须结束当前的 RenderPass，将所有 Tile 的数据 Flush 到显存（VRAM），并插入内存屏障，将资源状态从“写入”切换为“读取”，SSR 才能安全地采样全屏数据。