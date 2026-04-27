# AlphaTest画树叶易致Overdraw爆炸

### 🧑‍💻 User

alpha test画树叶会导致overdraw爆炸吗？

---

### 🤖 Assistant

**是的，使用 Alpha Test（透明度测试）来画树叶，如果不加干预，确实非常容易导致 Overdraw（过度绘制）“爆炸”，进而引发严重的性能瓶颈。**

不过，严格来说，Alpha Test 导致的性能问题不仅仅是传统意义上的“像素被覆盖重写”，更致命的是它会**破坏 GPU 的 Early-Z（提前深度测试）优化**。

以下是具体的原理分析以及为什么画树叶会成为重灾区：

### 1. 为什么画树叶容易导致问题？
树叶的自然形态决定了它在 3D 渲染中的特点：**高密度、高重叠**。
为了让树冠看起来茂密，美术通常会插片大量的面片（Quads）。当你从任意角度看向一棵树时，视线方向上往往会穿透几十甚至上百层树叶面片。

### 2. Alpha Test 的致命弱点：破坏 Early-Z
在现代 GPU 渲染管线中，为了防止 Overdraw，硬件引入了 **Early-Z** 技术：在执行耗时的 Fragment Shader（片元着色器）之前，先进行一次深度测试。如果当前像素被前面的物体挡住了，就直接丢弃，不执行 Shader。

但是，当你使用 Alpha Test（在代码中使用 `clip()` 或 `discard`）时，**GPU 无法提前知道这个像素到底会不会被剔除**。
* 只有等 Fragment Shader 完整执行完，采样了纹理，计算了 Alpha 值之后，GPU 才能决定这个像素是要写入深度缓冲还是被丢弃。
* 这导致 **Early-Z 失效**。对于重叠了 50 层的树叶，即使只有最前面的一层是可见的，GPU 也不得不把这 50 层的 Fragment Shader 全部跑一遍。这就是所谓的“Fragment Shader 级别的 Overdraw 爆炸”。

### 3. Alpha Test vs Alpha Blend
虽然 Alpha Test 会导致性能问题，但画树叶通常**不能**换成 Alpha Blend（半透明混合）：
* **Alpha Blend 需要严格的从后向前排序（Sorting）：** 树叶相互穿插，几乎不可能进行完美的深度排序，会导致严重的渲染错乱（前面的树叶被后面的遮挡）。
* **Alpha Blend 不写入深度（Z-Write Off）：** 无法产生正确的阴影，也无法与其他不透明物体正确交互。
* **Alpha Blend 的 Overdraw 更严重：** 每一层都会真实地进行颜色混合计算。

因此，业界画树叶的主流方案依然是基于 **Alpha Test**，但必须配合一系列优化手段。

---

<details>
<summary><b>🛠️ 解决与优化方案（点击展开）</b></summary>

为了解决 Alpha Test 画树叶带来的 Overdraw 爆炸问题，游戏工业界通常采用以下几种组合优化方案：

#### 1. 深度预处理（Depth Pre-pass / Z-Prepass）
这是最核心的解决方案。
* **第一遍（Pass 1）：** 只渲染树叶的深度。开启 Alpha Test，关闭 Color Write（不输出颜色，也不做光照计算），只执行极简的 Fragment Shader（只采样 Alpha 通道并 `clip`）。这一步虽然也破坏了 Early-Z，但因为 Shader 极简，代价较小。
* **第二遍（Pass 2）：** 正常渲染树叶的颜色和光照。此时深度缓冲已经填好了正确的树叶深度，设置深度测试为 `Equal` 或 `LEqual`，关闭 Alpha Test。此时可以完美利用 Early-Z 剔除掉所有被遮挡的片元，只有最终可见的像素才会执行昂贵的光照计算。

#### 2. 优化网格形状（Tight Mesh Fitting）
不要使用简单的四边形（Quad）来贴一片复杂的树叶纹理。
* 尽量增加一点顶点，把面片的边缘沿着树叶真实的轮廓裁切掉。
* 虽然这增加了顶点数（Vertex Count），但极大地减少了透明区域（需要被 `discard` 的空白像素），从而大幅降低了 Fragment Shader 的无效执行率。在现代 GPU 上，这点顶点开销远低于大面积的 Overdraw 开销。

#### 3. 视锥体外和距离级别的 LOD (Level of Detail)
* **近处：** 使用较精细的网格和 Alpha Test。
* **远处：** 随着距离拉远，将树冠替换为简单的体块模型，或者直接使用交叉面片 Billboard，大幅减少重叠层数。
* **极远处：** 直接替换为一张 Imposter（贴图公告板），完全消除内部 Overdraw。

#### 4. Alpha to Coverage (A2C)
虽然 A2C 主要用于解决 Alpha Test 边缘锐利（锯齿）的问题，但结合 MSAA（多重采样抗锯齿）使用时，它能在保持 Z-Write 的同时提供更好的边缘过渡，通常作为提升画质的配套方案。

</details>

### 总结
用 Alpha Test 画树叶确实会导致严重的 Fragment Shader Overdraw。如果不做优化，一片茂密的森林足以拖垮高端显卡。在实际开发中，必须配合 **Depth Pre-pass** 和 **更紧凑的网格（Tight Mesh）** 等技术来化解这场“爆炸”。

---

### 🧑‍💻 User

为什么mobile的tbdr对od特别敏感？

---

### 🤖 Assistant

其实，这里隐藏着一个非常经典的“反直觉”现象：TBDR（Tile-Based Deferred Rendering，基于图块的延迟渲染）架构最初的设计目的，恰恰是为了**彻底消除** Overdraw。

对于**不透明（Opaque）**物体，移动端的 TBDR 极其强大，它对 Overdraw 几乎是**免疫**的；但是，对于**半透明（Alpha Blend）**和 **Alpha Test（`clip()` / `discard`）** 产生的 Overdraw，移动端 TBDR 确实表现得极其敏感，甚至比传统 PC 的 IMR（Immediate Mode Rendering）架构更容易崩溃。

之所以会对这类 Overdraw 特别敏感，主要有以下几个核心原因：

### 1. 破坏了 TBDR 的核心护城河：HSR（隐藏面消除）
在 PC 的 IMR 架构中，解决 Overdraw 主要靠 Early-Z。而移动端 TBDR 拥有一项更强大的“黑科技”：**HSR（Hidden Surface Removal，隐藏面消除）** 或类似技术（如 Mali 的 FPK）。

* **正常情况（不透明物体）：** GPU 在光栅化之后、执行 Fragment Shader 之前，会在片上的超高速 Tile Memory 中进行完美的可见性判断。不管你重叠了多少层不透明物体，GPU 都能精确剔除被遮挡的像素，**每个屏幕像素永远只执行 1 次 Fragment Shader**。
* **致命弱点（半透明 / Alpha Test）：** 只要你使用了 Alpha Blend 或 Alpha Test，GPU 就**无法提前判断**这个像素最终到底长什么样，或者到底会不会被丢弃。为了保证渲染结果正确，GPU 被迫**关闭 HSR**，回退到最原始的画家算法：把重叠的几十层树叶或特效，从后往前**每一层都老老实实地执行一遍 Fragment Shader**。这使得 TBDR 的最大优势瞬间荡然无存。

### 2. 移动端算力（ALU）极度羸弱与功耗墙
PC 端（IMR 架构）的显卡体积庞大，插着独立的电源，拥有成千上万个流处理器，算力（ALU）充沛。即使发生了 10 倍的 Overdraw，显卡往往也能靠着暴力的算力硬扛过去。

但移动端 GPU 是集成在手机 SoC 中的：
* 移动端 GPU 的 ALU 数量和频率远远低于 PC。
* **发热与降频（Thermal Throttling）：** 手机没有主动散热风扇。Overdraw 意味着同一个像素的 Fragment Shader 被重复执行了 $N$ 次。大量的数学计算会瞬间产生巨大的热量。一旦触及温度阈值，手机系统就会**强制降频**，导致游戏帧率断崖式下跌。

### 3. Tile 阶段的内存溢出（Parameter Buffer 爆炸）
TBDR 渲染分为两个阶段：Geometry（几何/分块阶段）和 Rasterization（光栅化阶段）。
在第一阶段，GPU 会把所有的顶点数据和渲染指令分配到对应的 Tile（图块）列表中，这个列表存储在系统主存（DRAM）的一块特定区域，称为 **Parameter Buffer（或 Polygon List）**。

* 如果屏幕上某一个局部区域（比如一棵茂密的树、一堆绚丽的爆炸特效）发生了严重的几何 Overdraw（大量面片堆叠在一起）。
* 这个 Tile 对应的渲染指令和顶点数据就会急剧膨胀，导致 **Parameter Buffer 溢出**。
* 一旦溢出，GPU 就不得不中断当前的渲染流程，将数据强制刷入主存，或者打断 Tile 的处理逻辑。这会引发海量的系统内存带宽（Bandwidth）消耗。在移动端，**带宽就是功耗，带宽就是生命**，这种操作会导致性能直接崩溃。

---

<details>
<summary><b>💡 进阶：如何拯救移动端上的 Overdraw？（点击展开）</b></summary>

因为移动端对半透明/Alpha Test 的 Overdraw 如此敏感，移动端游戏开发中必须严格遵守以下优化规范：

1. **绝对不要滥用透明图层拼接：** UI 制作时，不要用多层透明的 `Image` 组件叠加来做效果，尽量在美术软件里合成为一张不透明贴图。
2. **切除透明留白（Tight Mesh）：** 无论是 UI、粒子特效还是树叶面片，尽量用多边形网格包围有效像素，切掉透明的空白区域。多消耗一点顶点（Vertex 阶段在移动端相对廉价），去换取 Fragment Shader 阶段的 Overdraw 减少。
3. **强制不透明化：** 对于远处的树木或粒子，在 LOD 较低时，直接替换为不透明的材质，利用 TBDR 的 HSR 免费剔除。
4. **Z-Prepass（深度预处理）：** 如前文所述，用极简的 Shader 先画一遍深度，再进行真实的颜色渲染，以此来拯救 Alpha Test 导致的 HSR 失效。

</details>

### 总结
移动端 TBDR 并不是对**所有**的 Overdraw 敏感，它只对**破坏了其剔除机制的半透明和 Alpha Test 敏感**。因为一旦护城河（HSR）被打破，移动端羸弱的 ALU 算力、极小的内存带宽和严格的电池/散热限制，就会让性能问题成倍地暴露出来。

---

### 🧑‍💻 User

现代的desktop（nvidia/amd/intel）/mobile gpu（apple/高通/mali）针对alphatest导致od有没有硬件层面优化？（也就是tbdr中这个d）

---

### 🤖 Assistant

这是一个非常深刻且直击现代 GPU 架构痛点的问题。

对于 Alpha Test（`discard` / `clip()`）破坏 Early-Z/HSR 从而导致严重的 Overdraw 爆炸问题，**硬件厂商其实一直没有停止尝试在硅片层面进行拯救**。

然而，由于 Alpha Test 在图形学上的逻辑限制（在 Fragment Shader 执行完之前，确实无法确定它是否可见），硬件厂商无法做到像不透明物体那样完美的“零代价剔除”。但他们确实在硬件底层加入了一些隐式优化策略。

以下是现代 Desktop 和 Mobile GPU 针对 Alpha Test 在硬件层面的优化机制：

### 1. 移动端 GPU（TBDR 阵营：Apple, Qualcomm, Mali）

正如你提到的，“TBDR 中的这个 D (Deferred)” 恰恰是移动端拯救 Alpha Test 的关键切入点。虽然 Alpha Test 破坏了 HSR（隐藏面消除），但现代移动端 GPU 会利用其独特的 Tile 架构尽力挽救。

#### A. PowerVR / Apple Silicon (最积极的 HSR 补救)
PowerVR（Apple 早期使用的架构，其思想深刻影响了 Apple Silicon）拥有业界最强悍的 HSR 硬件。对于 Alpha Test，它采用了一种 **"Delayed/Deferred Fragment Evaluation"（延迟片元评估）** 机制。
* **机制：** 当硬件在 Tile 中遇到一个带有 `discard` 的图元时，它**不会立刻**去执行昂贵的 Fragment Shader 来计算光照和颜色。相反，硬件会**先只执行 Shader 中计算 Alpha 并判定是否 discard 的那极小一部分指令**（通常是采样一张 Mask 贴图）。
* **结果：** 一旦判定这个像素被 `discard`，后续所有复杂的 PBR 光照、阴影计算都会被直接跳过。
* **局限：** 这依然算是一次 Overdraw（因为采样贴图的带宽已经花了），但它把“昂贵光照计算的 Overdraw”降级成了“极简指令的 Overdraw”。

#### B. ARM Mali (Forward Pixel Kill - FPK 的妥协)
Mali 架构使用的是 FPK（Forward Pixel Kill）技术，本质上是一种加强版的 Early-Z 队列。
* **机制：** Mali 碰到 Alpha Test 时，表现相对糟糕。因为 FPK 依赖于确定性的 Z-Write，而 Alpha Test 让 Z-Write 变得不确定。
* **优化：** 较新的 Mali GPU（Valhall / 5th Gen 架构之后）优化了线程调度。如果一个 Quad（2x2 像素块）内的线程触发了 `discard`，硬件会尽快回收这些 ALU 线程，而不是让它们空转等待同一 Warp/Wavefront 中的其他像素计算完毕。但在剔除遮挡关系上，Mali 依然非常依赖开发者手动做 Depth Pre-pass。

#### C. Qualcomm Adreno (LRZ - Low Resolution Z)
Adreno 的杀手锏是 LRZ（低分辨率深度图）。
* **机制：** 在 Binning（分块）阶段，Adreno 会生成一张低分辨率的深度图用来做超粗粒度的快速剔除。
* **对于 Alpha Test：** LRZ 通常会被 Alpha Test **直接打断**（因为深度不可控）。但高通后期的驱动和硬件优化中，允许硬件追踪哪些 Tile 受到 Alpha Test 影响。对于没有被 Alpha Test 污染的区域，继续使用极速的 LRZ 剔除；对于污染区域，则退化到标准的 Early-Z（依然会被 `discard` 破坏）。高通强烈建议开发者不要混合交错渲染（Interleave）不透明物体和 Alpha Test 物体，以尽量保持 LRZ 的存活率。

---

### 2. 桌面端 GPU（IMR 阵营：NVIDIA, AMD, Intel）

桌面端由于没有基于 Tile 的 HSR，它们的优化重点集中在**粗粒度剔除（Coarse Culling）**和**寄存器/线程回收**上。

#### A. NVIDIA (Hierarchical-Z / Z-Cull 的挣扎)
NVIDIA 拥有非常强大的 Hierarchical-Z (Hi-Z)。
* **机制：** Hi-Z 是一种多层级的深度缓存。当遇到 Alpha Test 时，当前像素的 Early-Z 必然失效，**并且它甚至可能导致该位置的后续图元的 Hi-Z 快速剔除失效**（因为硬件不知道这个 Alpha Test 像素最终是否写了 Z）。
* **硬件级补救 (Conservative Z-Cull)：** 较新的架构（Turing/Ada）在遇到 Alpha Test 时，只要 Shader 声明了保守深度（Conservative Depth，即只向更深处写 Z，不向更浅处写 Z），或者干脆不写 Z，硬件依然可以维持部分的 Z-Cull 效率。
* **Quad 级别提前终止：** 如果一个 2x2 Quad 中的 4 个像素全部触发了 `discard`，NVIDIA 的硬件会非常快速地终止整个 Quad，释放寄存器和 Register File 带宽，防止 ALU 空转。

#### B. AMD (可见性测试与 Wavefront 优化)
AMD 的 RDNA 架构同样在 Hi-Z 层面上对 Alpha Test 很敏感。
* **机制优化：** 与 NVIDIA 类似，AMD 优化的核心在于**快速丢弃**。当 Fragment Shader 中执行到 `discard` 指令时，现代 RDNA 硬件会利用细粒度的 Wavefront 控制，迅速释放被丢弃像素占用的 VGPR（矢量通用寄存器）。这不能减少“已经被调用的 Overdraw”，但能提高整体吞吐量，让其他有用的像素更快得到计算资源。

#### C. Intel (Intel Xe / Arc)
Intel 的优化思路与业界主流一致。值得一提的是，Intel 提出了很多基于软件+硬件结合的 Visibility Buffer（可见性缓冲）方案。在某种程度上，Visibility Buffer 就是一种手动的、全平台的“Deferred材质求值”机制，这是目前 PC 端绕过 Alpha Test 性能惩罚的前沿方向（虚幻 5 的 Nanite 就是类似思路）。

---

<details>
<summary><b>💡 总结：硬件到底做了什么？（点击展开）</b></summary>

简而言之，面对 Alpha Test，所有现代 GPU 的硬件优化都**不是（也做不到）魔法般的完美剔除**，而是集中在“**尽早止损**”：

1. **分离着色（Deferred Evaluation - Apple/TBDR首选）：** 尝试把“判定是否可见的计算”和“高成本的光照计算”在硬件层面上分开，一旦判定不可见，立刻砍掉后续昂贵计算。
2. **快速线程终止（ALU/Register Recovery - 桌面端首选）：** 当 Shader 执行到 `discard` 时，硬件以最快速度回收该像素占用的计算资源，防止占着茅坑不拉屎（防止 Warp/Wave 内部散度过大）。
3. **保护 Hi-Z / LRZ 不被完全摧毁：** 尽量只让包含 Alpha Test 的局部区域失效，而不是让整个屏幕的深度剔除机制崩溃。

</details>

**最终结论：**
尽管硬件厂商绞尽脑汁加入了上述隐式优化，但 **Alpha Test 依然是现代渲染管线的性能杀手**。硬件的“止损”往往不足以抵消重叠 50 层树叶带来的灾难。因此，**深度预处理（Z-Prepass）**依然是全平台（无论是 PC 还是 Mobile）公认的、必须由开发者主动实施的最有效解决方案。