# Title

### 🧑‍💻 User

为什么mobile pipeline中，ssao要放在rending gbuffer之前，而不是之后？

---

### 🤖 Assistant

在 **PC/主机端的延迟管线（Deferred Rendering）** 中，SSAO 通常放在 **G-Buffer 之后**，因为它需要法线、深度等几何缓冲；
但在 **移动平台（Mobile Pipeline）** 中，情况正好相反：**SSAO 通常被移动到 G-Buffer 之前（或完全不使用真正的 G-Buffer）**。
这是移动 GPU 架构 + Tile/Forward+ 渲染方式共同决定的。

下面分步骤解释原因。

---

# ✔ 核心答案（为什么手游里 SSAO 要放在渲染 G-Buffer 之前？）

因为 **移动 GPU 基本不走真正的延迟渲染管线**，而是：

### **Mobile = Tile-based Deferred + Forward Rendering 混合（TBDR）**

移动 GPU（Mali、Adreno、PowerVR）都有 **Tile-based（基于瓦片）** 渲染架构：

* 不喜欢保存庞大的 G-Buffer（带宽吃不消）
* 不喜欢在有 G-Buffer 的情况下再做一次 full-screen pass
* 任何“放在 G-Buffer 之后”的效果会导致昂贵的 **tile resolve → memory round trip**

➡ 所以移动端把 SSAO 放在 **渲染 G-Buffer 之前**，或完全不产生全尺寸 G-Buffer。
真正原因如下：

---

# 🔥 原因 1：**避免 G-Buffer 读写开销（移动 GPU 最大瓶颈是带宽）**

如果在 G-Buffer 之后再做 SSAO：

1. G-Buffer 需要 flush 到系统内存（tile resolve）
2. SSAO 要全屏读取 G-Buffer → 触发大量 external memory read
3. 做完 SSAO 再写回 RT → 再触发 memory write

移动 GPU 系统内存带宽非常有限（比桌面低 10–20 倍），这样做是 **极度低效**

因此移动端渲染顺序一般变成：

```
Depth Prepass → SSAO（用 depth 推导 normal） → Forward Lighting / Light prepass → Composite
```

完全绕开了 G-Buffer。

---

# 🔥 原因 2：**移动端采用“Fake SSAO / Depth-only SSAO”，根本不依赖 G-Buffer**

手游里的 SSAO（尤其 URP / Unity / Unreal Mobile Pipeline）通常是：

### **基于深度的 SSAO（depth-only AO）**

通过深度图 + reconstructed normal 完成。

而不是延迟管线中那种：

* world normal
* albedo
* roughness
* metallic
* etc.

➡ 放在 G-Buffer 之前，是因为 **不需要 G-Buffer，反而会被 G-Buffer 拖累**。

---

# 🔥 原因 3：**移动端追求早期遮罩（early occlusion），减少后续 shading 负担**

SSAO 作为一种休闲 AO mask：

* 如果在 G-Buffer 之后做，它只是最后的图像合成
* 如果在 **G-Buffer 之前** 做，它可以作为**可见性因子**提前介入 shading

例如：

```
AmbientLight *= SSAO_factor
ImageBasedLighting *= SSAO_factor
DiffuseLight *= SSAO_factor
```

这样 shader 的 lighting 计算会变得更便宜（AO 在光照中提前参与），并且可以避免对深度暗处进行过多计算。

➡ 移动端的优化目标：**让 SSAO 参与 shading，而不是后期叠加**。

---

# 🔥 原因 4：**Mobile forward/clustered 渲染顺序决定了 SSAO 只能在 G-Buffer 之前**

移动 pipeline 大部分是 **Forward+ / Tiled Forward**（非延迟）：

### 执行顺序通常是：

1. Depth Prepass
2. SSAO（输入深度）
3. Light Culling（cluster / tile）
4. Forward Lighting
5. Transparent
6. Post processing

如果 SSAO 放在 G-Buffer 之后，不符合 forward 渲染顺序，没有地方插。

---

# 🔥 原因 5：**保持 Tile Memory-on-chip（GMEM）不被打断**

移动 GPU 的 GMEM（tile buffer）非常宝贵。

如果执行了 G-Buffer pass：

* tile 需要 resolve 到内存
* 就 flush 掉 GMEM 的高效加速路径

而 SSAO 如果提前做：

* 深度信息已经在 GMEM 里
* 直接全屏采样 tile-local depth 不需要内存往返

➡ 移动端先做 SSAO 的根本动力：**避免 tile resolve**。

---

# 📌 总结（一句话）

移动端的 SSAO 是：

* **深度重建法线**
* **tile-based forward pipeline**
* **需要在占用大量带宽的 G-Buffer pass 之前运行**

放在 G-Buffer 之后会触发高带宽、破坏 tile memory 的 resolve → 性能崩溃。

### 因此：

> **移动端 SSAO 必须放在 G-Buffer 之前。**

---

如果你愿意，我可以继续提供：

✔ Unity URP / HDRP Mobile SSAO 的具体顺序对比
✔ Unreal Mobile Pipeline 的实际调用顺序（RenderGraph）
✔ Mali/Adreno 架构中 tile resolve 的代价分析
✔ 真实手游中（阴影/ibl/ssao）完整渲染 pipeline diagram

你想继续了解哪一部分？


---

### 🧑‍💻 User

为什么desktop deferred的high quality ssao需要从gbuffer rt中sampling depth/normal？

---

### 🤖 Assistant

在 **Desktop Deferred Rendering** 中，“High Quality SSAO” 必须从 **G-Buffer 的 depth & normal RT（甚至 roughness/albedo）中采样**，这是其和 mobile SSAO 最大的结构性区别。
根本原因在于：

# 🎯 核心答案

**因为桌面 deferred 的高质量 SSAO 需要真实的世界空间几何属性（world normal / depth / material），这些属性只有在 G-Buffer 中才能以高精度、高一致性地获得。**

Mobile SSAO 基本是 *depth-only reconstruction*，而桌面高质量 SSAO 需要：

* **真实的 per-pixel world normal**（不允许用 depth 重建）
* **线性深度（absolute depth precision）**
* **multi-bounce / bent normal AO**（需要法线一致性）
* **material-aware AO**（粗糙度、金属度、surface type）
* **multi-layer geometry**（不可用 depth 重建正常获取）
* **高采样精度的 surface curvature**

G-Buffer 是唯一能在**屏幕空间一次性保存所有这些属性**的结构。

下面展开解释。

---

# 📌 1. 深度重建 normal 在桌面不够准确（尤其高分辨率 / HDR / PBR）

Mobile SSAO 通常这样 reconstruct normal：

```
normal = reconstruct_from_depth(depth, adjacent depth)
```

但在桌面 deferred 下，这种方法精度不够，原因：

### ① depth-only normal 误差很大

重建 normal 对表面曲率、near-plane 梯度非常敏感：

* 自动产生阶梯噪点
* 锐边处特别不准
* Thin geometry（武器、树枝）完全破坏 normal gradient

高质量 SSAO 需要 **true world-space normal**（gbuffer 存的）。

---

# 📌 2. 高质量 SSAO 需要“surface material”，来自 G-Buffer

顶级 SSAO（HBAO+, GTAO, GTAO+Bent Normal, ASSAO）会用：

| 信息                    | 用途                    |
| --------------------- | --------------------- |
| **roughness**         | 调整 AO 半径（粗糙表面 AO 更柔和） |
| **metallic**          | 金属表面不需要 AO            |
| **albedo / specular** | 用来调节 AO contribution  |
| **normal**            | 必需，用于对比曲率             |

这些信息只在 G-Buffer 里才有。

➡ Mobile depth-only SSAO 无法实现这些高质量特性。

---

# 📌 3. Desktop SSAO 需要 world-space depth，而不是 NDC / reprojected depth

桌面 deferred 的 SSAO 一般做：

```
worldPos = Reconstruct Position from depth
occlusion = sample world-space neighborhood
```

这要求深度是：

* Linear depth
* 高分辨率（MSAA 下多 sample）
* 与 world normal 对齐一致

这是 **G-Buffer depth** 才能提供的。

Mobile depth often：

* 使用低精度 R16 depth
* 甚至用 depth pyramid resolve
* 在 tile 上不保证与 normal 一致（因为 normal 是 reconstruct 的）

---

# 📌 4. 高质量 SSAO 依赖 G-Buffer normal consistency（特别是和光照一致）

桌面 deferred：

* 正式的光照（direct + indirect）使用 **G-Buffer normal**
* SSAO 必须使用 *同一套 normal*，否则 AO 与光照的 shading 定向不一致，会出现 stitching、halo

Mobile SSAO 用 reconstructed normal，反正 mobile lighting 也是 simple-lambertish，会容忍误差。

Desktop PBR 不行。

---

# 📌 5. Desktop deferred SSAO 是 *真正的屏幕空间幂几何算法*

例如：

### NVIDIA HBAO+（高端 SSAO）

* 采样 normal map
* 采样 depth
* 采样 surface orientation
* 使用法线差分来推算遮挡

### AMD GTAO（更高质量）

* 用 normal 来计算 bent normal
* 用 roughness 做 radius modulation
* 用 world-space depth 做 neighborhood curvature

**没有 G-Buffer 信息根本无法执行这些算法。**

---

# 📌 6. Desktop deferred lighting 必须在 G-Buffer 之后，因此 SSAO 必须依赖 G-Buffer

Deferred pipeline：

```
1. Geometry Pass → write G-Buffer (depth, normal, albedo, roughness, metallic…)
2. SSAO (full-screen, read G-Buffer)
3. Lighting Pass (reading G-Buffer & SSAO)
4. Postprocess
```

因为 deferred lighting 在 **SSA**O 后，需要：

* SSAO 遮挡 factor
* 必须用相同 normal/albedo 参与光照

所以 SSAO 必然要从 G-Buffer RT 读取。

---

# 📌 7. Desktop Deferred 使用多个 G-Buffer RTs（G0/G1/G2...）提升 AO 准度

常见：

```
GBuffer0: Albedo + Roughness
GBuffer1: Normal + Metallic
GBuffer2: Depth (Z)  → depth buffer
```

AO 使用 normal、depth、roughness、viewspace position 来计算高质量遮挡。

➡ 这些数据 mobile pipeline 没有，也不想存（太贵）。

---

# 📌 总结（一句话）

**因为 Desktop Deferred 的高质量 SSAO ≠ Mobile depth-only SSAO，它需要真实材质信息（normal / roughness / metallic / albedo / linear depth），这些只有在 G-Buffer 中存在。**

因此：

> **Desktop Deferred 的 High Quality SSAO 必须从 G-Buffer 采样。Mobile 的 SSAO 不必且没办法这么做。**

---

如果你需要，我可以继续写：

✔ Desktop vs Mobile SSAO 的完整 shader 伪代码对比
✔ HBAO+ / GTAO / ASSAO 如何使用 G-Buffer
✔ 为什么 mobile 不能承担 full G-Buffer（tile architecture 详细解释）
✔ Deferred / Forward+ / Clustered 下 SSAO 的精确 pipeline 顺序图

你想看哪部分？
