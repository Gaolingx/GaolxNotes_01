# Renderpass中断原因

### 🧑‍💻 User

为什么类似ssr这种需要采样前一renderpass渲染结果的post process effect会导致subpass中断，导致必须另起一个新的renderpass？

---

### 🤖 Assistant

在 **Vulkan/Metal/D3D12 的现代 tiled/deferred GPU** 上，**任何需要读取“前一 pass（或同一 render pass 中之前的 subpass）所输出的 color attachment”的效果** —— 例如 SSR（Screen Space Reflections）、部分 SSAO、Bloom prefilter、TAA、Motion Vector resolve —— 都会导致 **subpass 无法连续、render pass 被迫拆开**。
核心原因不是 API 限制，而是 **tiled GPU 的 render-pass/subpass 模型的本质：attachment 在 render pass 内是“隐式在 tile memory 中”的，不能在写入后立刻被当作普通 texture 读取**。

下面分架构讲清楚为什么必须中断。

---

# ✅ **1. Render Pass/Subpass 的核心限制：Attachment 不是 Texture**

在同一个 render pass 内：

* color / depth attachment **不会落到 VRAM**
* 而会 **待在 tile memory（on-chip memory）**
* GPU 假设：**你不会在 write 后立即把这个 attachment 当作 sampled texture 再读**

因此 GPU 不允许 “attachment write → 立即 attachment sample” 在 render pass 内发生。

这意味着：

> **一旦你要将上一 pass 的 color 作为 texture 读（SSR 必须做），就必须结束 render pass，让 GPU flush tile memory → VRAM → 再作为 sampled image read。**

---

# ✅ **2. 为什么 subpass dependency 也无法解决？**

你可能认识到 subpass 有 input attachments：

* Subpass A 写 color attachment
* Subpass B 可以把它作为 input attachment 读（tile-local read）

但 SSR/TAA/Bloom **完全不能使用 input attachment**，因为：

### ❌ 2.1 Input attachment = 局部 tile access，不是 arbitrary sampling

Input attachment **不支持 arbitrary UV sampling、mipmap、LOD、filtering**。

SSR 需要：

* 任意屏幕 UV 访问
* 多次 sample（ray marching）
* LOD、filter filtering

**input attachment 完全不满足**。

### ❌ 2.2 Input attachment 只能在当前 render pass 中使用

不能跨 render pass；SSR 的前一帧 resolve 通常也需要 VRAM 中的 texture。

所以：

> **SSR 不能用 input attachment ⇒ 必须把 RT 写入 VRAM ⇒ 必须结束 render pass。**

---

# ✅ **3. Tile-based GPUs（移动端）上为什么不能允许“边写边采样”？**

在移动 GPU（Mali、Adreno、Apple GPU）上，render pass 的意义更强：

* 所有 color attachment 都存在 tile memory 中直到整个 tile 完成
* 根本没有 global visibility

如果你在同一个 render pass 内要求读取刚写的 color：

### ❌ GPU 要强制 flush tile → VRAM → reload back to tile

这会：

* 严重破坏 tile-based 延迟渲染的效率
* 增加巨额带宽
* 破坏 render pass 合并优化

于是 API 层面直接禁止。

---

# ✅ **4. Desktop GPU（IMR）的情况**

Desktop GPU（NVIDIA/AMD）不依赖 tile buffer，但 Vulkan/D3D12 **仍然保持 render pass 的概念**：

* attachment write → 未 resolve 的 “layout”
* sampler read 要 image layout = *shader-read-only-optimal*
* attachment write layout 与 texture sampling layout 互不兼容

因此必须：

1. 结束 render pass
2. barrier 转换 layout
3. 重新开始一个新 render pass 或 dispatch compute shader

---

# ✅ **5. 总结：为什么 SSR 一定会导致 subpass 中断？**

因为 SSR 需要：

### ✔ 作为 **texture** 采样

而不是 input attachment

### ✔ 任意坐标的随机访问

input attachment 无法做到

### ✔ 需要 LOD/filter

input attachment 无法支持

### ✔ 需要读前一 pass 甚至前一帧数据

input attachment 不能跨 pass/frame

并且：

> **一旦你需要把 attachment 当 texture 读，就必须结束当前 render pass，让 GPU flush + layout transition。
> 所以 subpass 一定会被中断，必须新起一个 render pass。**

---

# 如果你需要，我可以进一步解释：

* 为什么 HDRP 有大量 mini render pass/subpass，为什么不怕中断对性能影响
* 为什么移动端 forward+ 中会把 SSAO/SSR 放到 prepass 前后不同阶段
* 如何最小化 render pass 中断的开销（tiling size、拼合 pass、async compute）

随时问。
