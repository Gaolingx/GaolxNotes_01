# SSAO与大范围AO计算

### 🧑‍💻 User

为什么ssao非常难以计算大范围的ao，这通常需要一些依赖raytracing的gi才能实现。

---

### 🤖 Assistant

下面从**几何原理、采样分布、噪声与遮挡概率、深度图局限性**四个角度解释，为什么 **SSAO（Screen-Space Ambient Occlusion）几乎无法可靠计算“大范围 AO”**，而通常必须依赖 **RTGI / RTAO（Ray Traced GI/AO）** 才能实现大尺度遮挡。

---

# ✅ 1. **SSAO 的“视锥体/屏幕空间”限制 —— 看不到的东西无法提供遮挡**

SSAO 只依赖 **当前屏幕** 的：

* depth buffer
* normal buffer

它完全不知道 **屏幕外的几何、被前景遮挡的几何、背后的场景拓扑**。

因此，屏幕空间 AO **只能计算到摄像机可见的 surfaces 的局部遮挡**。

> **大尺度 AO（如几米外的柱子投的全局遮挡）在 depth buffer 中根本不可见，所以 SSAO 不可能知道它的存在。**

举例：

一个角落房间结构，远端墙壁对地面的环境光遮挡是巨大的，但在当前画面的 depth 中：

```
墙壁   [被遮住 → depth 中不存在]
地面   [可见]
```

SSAO 不知道墙壁存在 → 无法添加长程 AO。

---

# ✅ 2. **大范围 AO 需要“长程采样”，屏幕空间采不到**

SSAO 的工作方式是：

### 在像素附近的**半径 r 区域内**随机采样深度 → 估计遮挡概率

但半径 r 实际上受多种强约束：

* 采样距离越大，深度 buffer 分辨率不足 → 跳变/不连续
* 采样越远越容易穿透背景（depth discontinuity）
* 投影空间的采样半径非常扭曲
* 采样半径若明显大于物体屏幕投影大小 → AO 全是噪声和错误

实际 SSAO 通常只能稳定处理：

* 半径 ≈ 0.5%–2% 屏幕尺寸（在 world space 中通常 ≈ 0.1–1 m）

> 真正 GI 级别 AO 需要 5–10m 的遮挡半径，而屏幕空间深度根本无法稳定支持这种距离的空间采样。

---

# ✅ 3. **深度图是 2.5D 的 —— AO 是 3D 几何问题**

SSAO 的深度图结构：

```
每个 (x,y) 只有一个 Z 值
没有体积信息
没有背面
没有深度层次
```

也就是说：

* 无法处理多层几何
* 无法响应复杂拓扑（如洞口、门框）
* 只能看到 surfaces，没有 geometry

而 AO 本质是：

> 计算某点周围 360° 立体空间的遮挡率

这种东西 **2.5D 结构是根本描述不了的**。

举例：你在桥下行走，桥上结构对你脚下地面的 AO 贡献极大，但深度图通常是：

```
地面深度 → visible
桥深度 → 被遮挡 → 消失
```

SSAO 中“桥”完全不可见 → 不可能算出 global AO。

---

# ✅ 4. **大范围 AO 会被投影空间畸变彻底破坏**

SSAO 通常在 **view space 或屏幕投影空间**采样。

当采样半径增加时：

* 屏幕空间拉伸变形严重
* 远处采样点聚在同一深度像素
* Z 精度丢失（especially reversed-Z still limited）
* 表面偏斜角导致噪声剧增

SSAO 半径扩大 = AO 错误几何级别增长。

这就是为什么大部分 SSAO 实现（HBAO, GTAO）都会强调：

**不要使用过大采样半径，否则 AO artifact 爆炸。**

---

# ✅ 5. **AO 需要“真实的几何”，SSAO 只有“投影后的片段”**

AO 本质是：某点周围是否被体积阻挡。

RTGI/RTAO 使用 BVH（真实三角形）进行射线查询：

```
射线 → 相交 → 告诉你几米外是否有遮挡物
```

SSAO：

```
屏幕片段 → 邻域采样 → 无法知道真实 3D 拓扑
```

等价于只用“二维照片”下判断三维遮挡 → 本质不可能全局正确。

---

# 🔥 6. **为什么 RTGI / RTAO 可以解决大范围 AO？**

因为 RTGI/RTAO 具有：

### √ 真正的 3D 空间

使用 acceleration structure（BVH）
能查询任意距离的遮挡，甚至几十米也没问题。

### √ 不依赖相机视野

屏幕外几何仍存在于 BVH → 可以贡献 AO。

### √ 不受深度图分辨率限制

世界空间精确相交，而不是屏幕空间模糊投影。

### √ 能处理多层几何、细节、拓扑

桥下、桌子下、洞穴结构等各种情况都能正确计算。

---

# 🔚 总结：为什么 SSAO 难做大范围 AO？

| 问题    | SSAO       | RTGI/RTAO  |
| ----- | ---------- | ---------- |
| 屏幕外几何 | ✗ 无法看到     | ✓ BVH 存在   |
| 长距离遮挡 | ✗ 深度图不支持   | ✓ 射线长度可调   |
| 多层几何  | ✗ 只有 2.5D  | ✓ 完整 3D    |
| 空间畸变  | ✗ 采样半径越大越错 | ✓ 无投影畸变    |
| 光照拓扑  | ✗ 视空间局部    | ✓ 真实 3D 场景 |

所以：

> **SSAO 是局部 AO。要大范围/全局 AO，必须要 ray-traced AO 或 GI。**

---

如果你愿意，我还可以进一步给你：

* SSAO → GTAO → HBAO → GTAO+ 的能力比较
* Ray-traced AO 与 SSRTGI 区别
* 为什么 UE5 使用 DFAO、Lumen、RTAO 混合提供 long-range AO
* 如何在非 RT pipeline 实现近似 long-range AO（GTAO bent-normal 分布）

只要告诉我你想继续深挖哪一块。


---

### 🧑‍💻 User

有没有一种可能 当rtgi的精度足够高的时候，用gi也可以实现ao的很多效果了？

---

### 🤖 Assistant

**不仅有这种可能，而且这正是现代实时渲染（UE5、Frostbite、RE Engine 等）在做的事情：
当 RTGI 的采样密度与噪声控制足够好时，它可以天然地“包含” AO 的绝大多数效果。**

下面我详细解释为什么 **高精度 RTGI 会自然地产生 AO**，以及为什么它可以*取代*传统 AO。

---

# ✅ 1. **AO 本来就是 GI 的一个特殊情形**

从光照方程（Rendering Equation）来看：

AO 的定义其实是：

> 在某点附近的小范围内，半球方向上有多少光线被遮挡。

而 GI 做的事情是：

> 在某点向各个方向射线，查看是否有光照贡献（直接光、间接光、遮挡）。

如果你把 GI 的间接光全部屏蔽掉，只保留遮挡信息，扔掉颜色、反射、bounce 信息，你得到的就是：

> **AO = GI 在 0 bounce、仅算 visibility term 的情况**

因此：

✔ AO ⊂ GI
✔ GI 自然包含 AO
✔ GI 越精确，AO 越自然真实

---

# ✅ 2. **RTGI 由于有 BVH + 真正射线，本质上比 AO 更“正确”**

RTGI 在计算时：

* 对每个像素向半球发射射线
* 查 BVH 得到遮挡情况
* 得到 bounce 光照与能量

AO 只需要其中的一部分：

* 某方向光线是否被遮挡？（visibility 0/1 或 fraction）

如果你把 GI 结果做一些 simplification：

* 抹掉 indirect color
* 抹掉 bounce light
* 把光照强度变成纯遮挡率

就会得到：

### **Ray-traced AO (RTAO) = RTGI 的“灰度遮挡项”**

事实上，许多引擎：

* **UE5 Lumen**
* **Frostbite RTXGI**
* **NVIDIA ReSTIR GI**
  都已经直接用 GI 替代 AO，或者从 GI 中“推导”出 AO。

---

# ✅ 3. 当 RTGI 噪声足够低时：

AO 其实就是**GI 在小范围内的特殊参数化**。

RTGI 默认是 multi-bounce + large radius。

而 AO 只是：

* 0 bounce 或 1 bounce 非颜色项
* bent normal approximation
* 只算 visibility term，不算 radiance term
* 半球范围较小

你甚至可以把 GI 的参数 tune 成 AO：

```
GI_Radius = AO_Radius
GI_Bounce = 0
GI_Color = White
```

这样 RTGI = 高精度 AO
并且永远不受屏幕空间限制。

---

# ✅ 4. **RTGI 精度非常高时，AO = GI 中低频阴影的一部分**

当 RTGI 采样数量充足：

GI 自然会包含：

* 物体遮挡近处的暗角 → AO 效果
* 巨大物体投出的 broad soft shadow → 是 long-range AO 的扩展
* bounce 光照填充 → 让“AO 区域不会死黑”

这比 SSAO/HBAO/GTAO 更正确。

AO 本身是一个**原本应该属于 GI 的补丁**，因为早期无法做完整 GI，只能做局部遮挡项而已。

现在 RTGI 的速度足够了，AO 的原始意义正在被吸收。

---

# 🔥 5. **为什么 UE5 中 Lumen 的 GI 输出中 AO 已经“隐含存在”？**

UE5 Lumen 已经：

* 不再需要传统 SSAO 做全局 AO
* GI 输出中天然包含 broad-scale AO
* 用 screen-space AO 只补一些高频细节（small-scale contact shadows）

因此 Lumen 的 pipeline 是：

* **RTGI 提供大尺度 AO**
* **SSAO 用来提升微细细节（micro contact shadows）**

这正体现了你的观点：

> **当 GI 足够好时，AO 变成 GI 的一个低频成分，仅需少量 screen-space 补高频。**

---

# 🧠 6. **真正的未来：AO 会消失，统一回 GI 中**

业内趋势已经越来越明确：

### **AO 会淡出，最终由 RTGI/RTAO 替代。**

原因：

* GI 能提供更真实遮挡（multi-bounce、color bleeding）
* GI涵盖了 AO 所能提供的所有效果
* GI 精度上来后不需要单独的 AO pass
* AO 仅在极少数情况作为一个“small-scale visibility booster”

例如：

* **Cyberpunk 2077 Path Tracing 模式：不需要 AO**
* **UE5 Lumen High Quality 模式：不需要传统 AO**

---

# 🧩 7. 回到你的问题：

> **有没有一种可能，当 rtgi 精度足够高，用 gi 就可以实现 ao 的很多效果？**

答案：

### ✔ 不仅可能，而且是现代渲染正在发生的事实。

### ✔ RTGI/GI 在高精度下天然包含 AO。

### ✔ AO 正在从独立 pass 被吸收进 GI。

如果你愿意，我可以继续讲：

* RTGI 输出中如何 mathematically recover an AO term
* 为什么现阶段仍保留 SSAO（微影子）
* RTAO 和 RTGI 有什么本质区别
* Lumen、RTXGI、Frostbite 都是如何从 GI 导出 AO 的

告诉我下你想深入哪一块。


---

### 🧑‍💻 User

但现实情况却是，通过rdc查看ssgi pass的indirectdiffusecolor，即便经过Temporal +Spatial denoise之后，结果依旧很模糊（低频gi），精度似乎并不足以作为全局 AO使用，这是为什么？

---

### 🤖 Assistant

你观察到的现象 **完全正确**，并且这是 SSGI/RTGI 在实际实时渲染中面临的最核心技术瓶颈：

> **即使 RTGI/SSGI 理论上能包含 AO，但实际实现中，RTGI 输出的 indirect diffuse 通常非常模糊（低频），根本不足以直接替代 AO。**

下面我从**采样预算 → 几何错误 → 噪声统计特性 → SVGF/ATrous 限制 → GI 不等于 visibility term** 逐层分析为什么真实游戏/引擎里 RTGI 的“精度看上去远不足以当 AO”。

---

# ✦ 1. **SSGI/RTGI 的采样量太少——噪声 → 必须模糊 → 高频 AO 消失**

真实游戏中的 RTGI/SSGI：

* 每像素只有 *1–2 条射线*（甚至 0.5 rays/pixel）
* Temporal accumulation 依赖 motion vector（难以保证稳定）
* Spatial denoise 用 bilateral/A-Trous（会侵蚀高频细节）

**AO 恰好是极高频信号（几乎靠近接触点的瞬间遮挡变化）**
而 GI 的 indirect diffuse 是极低频信号（bounce light 是平滑的）。

结果：

### ➜ 为了去噪，滤波器必须把高频噪声一起模糊掉

### ➜ 而 AO 的信息本质上也是高频

### ➜ 因此 AO 特征被当成噪声去掉了

最终你看到：

> **GI 干干净净，只剩下低频光照。AO 消失了。**

这就是为什么：

* RTGI 看起来雾蒙蒙的
* 而 GTAO/HBAO 看起来清晰锐利

**不是因为 RTGI 不够强，而是它强得需要更多采样，但没有预算。**

---

# ✦ 2. **Temporal 的累积会进一步抹掉高频细节**

SSGI/RTGI 主要靠 Accumulation 去噪：

```
FinalGI(t) = lerp(CurrentSample, PreviousGI, Alpha)
```

但 AO 变化极快（camera 轻微移动就改变）：

* 高频（AO）在 temporal 中不稳定 → 累积时被平均掉
* 低频（间接漫反射）稳定 → 成功叠加

这就是为什么：

### ➜ Temporal denoise 会让 AO 消失

### ➜ 但让 indirect diffuse 越来越干净

所以 indirect diffuse 看起来“像牛奶一样柔和”（blurry），但 AO 却完全不存在。

---

# ✦ 3. **SSGI 不是 visibility query，无法直接计算遮挡概率**

真正 AO 需要的 term 是：

```
Visibility(ω) = 0 or 1
AO = ∫ Visibility(ω) dω
```

SSGI / IndirectDiffuseColor 输出的不是 Visibility term，而是：

```
IndirectDiffuse = Albedo * IncomingRadiance * CosTheta
```

AO 是几何遮挡；
GI 是光能传输。

**二者数学上完全不同！**

如果你试图从 GI 恢复 AO：

* Radiance 会被 albedo、roughness、lighting 强烈影响
* 压根不是纯 visibility term

所以 indirect diffuse 永远不能还原 AO 的锐度，哪怕没有噪声。

---

# ✦ 4. **屏幕空间 SSGI 天然会漏掉背面几何 → AO 缺失无可避免**

SSGI 的局限：

* 只看得到当前帧可见的 surfaces
* 被遮挡的 geometry 完全无法参与 indirect diffuse 贡献
* GI 是错误几何 → AO 也随之错误

例如：

* 桥下的 AO
* 桌子下的 AO
* 远处大体块投射的 broad AO

SSGI 全都会漏掉。

为了掩盖这个问题，引擎只能：

* 用重度 spatial blur（使错误更不明显）
* 减少影响半径（避免穿深度）
* 用 very conservative采样

结果：

> **SSGI 的 GI 越“干净”，AO 越消失。**

---

# ✦ 5. **Screen-space GI 的半球采样范围很小 → 只能产生低频 GI**

SSGI/RTGI (非真正 BVH RTGI) 的半球通常被 clamp 为很小的 radius：

* 超过一定半径会穿背景 → 只能小范围采样
* 小半径 bounce 光必然是极低频 diffuse

这天然导致：

### ➜ 低频 GI = 物理意义上正常

### ➜ 高频 AO = 大部分被 depth discontinuity、denoise、temporal 破坏

---

# ✦ 6. **ATrous/SVGF/BilateralDenoise 的结构天生会杀掉AO**

比如 SVGF、AMD FidelityFX、A-Trous：

* 所有 filter 都假设“GI是低频的、几何边界是分割线”
* 其设计目标就是**滤掉高频噪声**

但是 AO 最难的是：

> AO 是高频的 geometrically significant feature
> 但在数学上和噪声非常相似

导致：

* Edge-aware filter 把 AO 当噪声消除了
* Temporal filter 把 AO 当不稳定信号消除了
* Spatial filter 把 AO 当局部噪声消除了

最终你看到：

**RTGI = 低频大块光能分布**
**AO = 本来属于高频几何遮挡 → 全被灭掉**

---

# ✦ 7. **真正引擎中的解决方案：RTGI + RTAO + SSAO + Micro-shadow**

因为上述所有原因，真实游戏都不会只用 GI 替代 AO：

### UE5 Lumen

* Lumen GI：低频间接光
* Lumen RTAO：高频遮挡（来自 BVH）
* SSAO：补 very near-field micro shadow

### Frostbite RTXGI

* RTXGI：低频 bounce
* RTAO：高频 contact shadow

### Cyberpunk 2077 RTX

* Path tracing 模式：无需 AO（因为真实 visibility）
* RTGI 模式：仍需要 RTAO 补 contact

**理由很简单：GI 和 AO 是两个频率带，不是同一类信号。
AO 是高频 visibility，GI 是低频 radiance。**

所以你看到的 SSGI 输出“模糊、低频”其实就是它应该的样子。

---

# ✦ 结论（回答你的问题）

> **为什么 SSGI 的 indirectDiffuseColor 即便经过去噪仍然很模糊、不足以当 AO？**

因为：

1. **采样太少 → 必须 denoise → 高频 AO 被消掉**
2. **Temporal accumulation 会抹掉 AO 高频**
3. **IndirectDiffuse ≠ VisibilityTerm → 无法表达 AO 的几何锐度**
4. **SSGI 有严重的屏幕空间几何缺失 → AO 本来就错**
5. **ATrous/SVGF 专门滤掉 AO 这种高频信号**
6. **SSGI 的物理半径设定决定它只能生成低频 GI**
7. **真实游戏都必须用 RTAO 或 SSAO 来弥补 AO**

因此：

> **SSGI 的 GI 绝不可能替代 AO（特别是 contact AO）。**
> 只有真正的 Ray-Traced AO（基于 BVH 的 visibility）才能深度替代 AO。

---

如果你愿意，我可以进一步：

* 画频域图解释“为什么 AO 是高频 GI 是低频”
* 解释 SVGF/Atrous 为什么数学上会抹掉 AO
* 分析 UE5 Lumen 如何把 GI/RTAO/SSAO 融合起来形成正确 AO

告诉我你想深挖哪部分。
