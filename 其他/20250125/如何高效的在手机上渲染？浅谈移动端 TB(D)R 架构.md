# 从iPhone到骁龙旗舰：一文看懂移动端GPU如何省电高性能

## 0. 引言：移动端GPU的三座大山

想象一下：你正在玩一款画质精美的开放世界手游，场景细节丰富，光影逼真。但不到十分钟，手机就开始发烫，帧率骤降……

是手机配置不够吗？还是游戏优化有问题？

实际上，这背后隐藏着一个关键矛盾：移动端GPU的强大算力，却被极其有限的内存带宽和功耗预算所束缚。

为了在“画质、流畅、续航”三者之间找到平衡，几乎所有主流手机GPU（包括Apple（苹果）的Apple Silicon系列、Qualcomm（高通）的Adreno系列、ARM（安谋）的Mali系列、Imagination的PowerVR系列、Huawei（华为）的Maleoon系列等）都采用了一种独特的架构设计 —— Tile-Based Deferred Rendering（TBDR）。

本文将带你深入这一被长期忽视却至关重要的技术底层，理解它如何改变渲染流程、节省带宽、降低发热，并告诉你作为开发者该如何利用它写出更高效的渲染代码。

![](imgs/_00.png)

## 1. 名词解释：图形渲染基础概念扫盲

- **图元（Primitive）**：可以绘制的最小单元。通常来说就是三角形，不过也可以是线段、点。
- **片元/片段（Fragment）**：每个图元光栅化以后的产物，每个片元都有可能会覆盖一些屏幕上的像素。
- **常量缓冲区（Constant Buffer，DirectX）/统一缓冲区（Uniform Buffer，Vulkan/OpenGL）**：对于整个渲染流程来说，只有一份的数据（相对于例如顶点缓冲区中有per vertex的数据而言）。通常来说，例如摄像机参数、光源的参数等等，会在常量缓冲区中。
- **着色器（Shader）**：执行在 GPU 上的程序。
  - **顶点着色器（Vertex Shader）**：执行顶点变换的着色器。通常对于三维游戏来说，其将顶点从局部坐标转换到世界坐标，再从世界坐标投影到画面上的坐标。
  - **像素着色器（Pixel Shader, DirectX）/片段着色器（Fragment Shader，Vulkan/OpenGL）**：计算出每个像素上的颜色的着色器。光照计算通常在这里发生（但不限于光照）。
- **IMR（Immediate Mode Rendering）**：立即渲染模式，是一种经典的渲染管线架构。在这种架构中，图形数据（如顶点、颜色、纹理坐标等）在每个绘制调用（draw call）中被直接提交给GPU。每次调用都会立即处理这些数据并渲染出图形。
- **TBR（Tile-Based Rendering）**：基于分块的渲染，是目前主流的移动GPU渲染架构。在这种架构中，屏幕被划分为多个小的Tile（瓦片），然后逐个Tile进行渲染。
- **RenderPass（渲染通道）**：RenderPass定义了整个渲染管线（Render Pipeline）的一次执行过程，它本质上是一份元数据或占位符，描述了完整的渲染流程以及所使用的所有资源。这些资源通常包括颜色缓冲区、深度/模板缓冲区等，被称为Attachments（附着物）。

## 2. 传统架构的困境：IMR在移动端为何行不通？

IMR 全称 Immediately Mode Rendering，顾名思义是即时渲染，IMR会将DrawCall处理为严格的命令流，每个DrawCall进来就会立马渲染并且渲染结果写入FrameBuffer内。

IMR渲染流程中的数据流动如下图所示，顶点着色器处理后的数据会进入一个FIFO中并依次处理，可以看出IMR需要多次和FrameBuffer进行存取操作。

![](imgs/_16.png)

IMR 的整体流程如下所示:

![](imgs/_02.png)

![](imgs/_01.png)

这是个IMR GPU大致的执行流程。从左到右，硬件从显存中读取顶点的数据和全局的世界数据（例如摄像机数据等）输入顶点着色器，顶点着色器将顶点从三维空间中变换到屏幕空间的坐标上。然后经过一些高效的固定功能硬件的处理，这些顶点被光栅化成了一个个片元，并在此过程中剔除了一些对最终画面没有影响的片段（早期深度测试，Early-Z Test）。随后，硬件将采样纹理数据、并读取光源等数据作为输入片段着色器。最终这些颜色经过再一轮的测试（透明度测试 Alpha Test 和后期深度测试 Late-Z Test）确定其对最终画面有影响以后，执行透明混合（Alpha Blend），最终写入到帧缓冲中，显示到屏幕上。

优点：

- 首先每个图元直接提交渲染，无需任何额外处理，因为只需要专注于渲染图元而无需其他的控制逻辑，可以极大地提高GPU的吞吐量，充分利用GPU的性能。
- 顶点数据被填充到FIFO队列里依次处理，即使几何数据再多，也会累积在On-Chip上。直到下一阶段准备好使用这些数据。这意味着GPU使用很少的外部内存带宽来存储和检索中间几何结果。这也让IMR架构下可以处理海量的DrawCall和顶点，这也是IMR的优势之一。

特点：

1. 在 IMR 渲染流程中，由于每个图元都是直接渲染到FB的，假如我们已经渲染了A图元，但如果出现了后渲染的B图元挡住了A图元的情况，就会出现Overdraw，尽管可以通过 Early-Z（硬件支持，但需要软件层面对物体进行排序）提前丢弃不可见片段，但是无法完全消除 Overdraw。（受限于渲染顺序、深度写入冲突、AlphaTest等可能造成Early-Z失效）

2. 因为IMR是全屏FrameBuffer上直接渲染的，FrameBuffer是一个全屏的本身就是很大的内存占用的，一般是放在显存中的。然而IMR架构下会和FrameBuffer的交互（多次存取）。并且在像素着色器之后的Depth Test、Stencil Test和Blend等操作都会有多次读取-修改-写入操作，这会消耗大量的带宽，同时还会产生较高的能耗。

在将这套管线迁移到移动端的过程中，发现了几个问题，先说结论：

1. 显存和GPU之间的数据传输是很慢的（相较于GPU的算力来说）。移动端通常 GPU 和 CPU 共享同一个内存池，而 DRAM 的带宽并不足以支撑图形渲染全流程的巨大带宽要求。同时，也几乎不可能将渲染中会用到的大量数据全部缓存住。移动端寸土寸金的 die area，很难将其花在缓存上。GPU 可能需要在等待显存的数据响应的时候，调度执行其他线程，来隐藏延迟。无论采取上述何种解决方式，都会对 GPU 的调度器、GPU的规模和 GPU 的显存带宽带来更高的要求。

2. 显存的动态功耗是很高的。ARM 官方给出的 rule of thumb 是 80-100mW per GB/s. 太高的功耗在移动端是不可接受的。（5W 的 power budget 下，能接受多少 GB/s 的显存流量？桌面 GPU 通常拥有 200GB/s 以上的显存带宽，甚至顶级消费级显卡可以达到或超过 1TB/s）。

3. 几何形状及其最终投影的关系是难以预测的。来自同一个 drawcall 的顶点数据，实际上极有可能在最终画面上的分布相差很远（绝区零邦布拍照任务？——即使确定的mesh，也可能会因为摄像机参数，或者其他一些原因而在屏幕空间上相差很远）。因此，很难去将屏幕空间的所有缓冲区都cache住，GPU片上的缓存会不停的被污染。

### GPU 算力增长的内存带宽需要和不平衡不充分的发展之间的矛盾

![](imgs/_03.png)

运算芯片的算力在飞速发展的同时，内存（DRAM）的发展速度却像蜗牛爬。如今，CPU 的运算吞吐量已经可以和 DRAM 速度差出数十倍甚至上百倍。CPU 况且如此，GPU 这种生来就是为了高吞吐而生的计算设备，问题就愈发严重了（DDR5 6400 双通道 - 100GB/s，也就是 25GFlops fp32 或者 50GFlops fp16，而 8gen2 GPU 可以有近 2000GFlops fp32 算力，4000GFlops fp16 算力）——不解决内存的问题，就是将许多性能白白的浪费了！

## 3. 破局之道：TBR 架构的核心思想

PowerVR 第一个发现了这个问题，并在 2001 年的时候提出了 TBR 架构。

TBR 简要流程图:

![](imgs/_04.png)

![](imgs/_05.png)

可以看出，TBR 架构直接把完整的渲染管线中各个阶段，拆分成了顶点阶段和着色阶段这两个相对独立的阶段。TBR架构在GPU很近的位置增加了一片高速缓存，通常被称为Tile Memory（图中也叫On-Chip Buffer）。受限于芯片面积与功耗，该缓存容量较小，通常为数十KB量级。

在TBR的渲染流程中，首先，它先将屏幕分成许多小块（叫做tiles），然后根据这些tiles和变换到屏幕空间的顶点的位置关系，将这些顶点排序，并分成很多的bin（这个过程叫binning）。这些排序后的结果存储到System Memory中，这块缓存也被称为Parameter Buffer (PB， 图中Primitive List和Vertex Data)，随后进入下一绘制命令的处理阶段。当所有绘制指令的顶点数据都做好处理存进PB或是PB达到一定容量之后才开始进行管线的下一步，即显卡会以tile为单位从PB中取回相应的顶点数据，进行光栅化、fragment shader以及逐片元处理。原本在逐片元处理中需要频繁的访问System Memory变为代价极低的对Tile Memory的访问。直到这个tile的frament将数据全部更新到Tile Memory上之后，再将该Tile Memory中的数据写回System Memory，然后执行下一个tile的处理。

## 4. 更进一步：TBDR 与隐面剔除技术对比

![](imgs/_06.png)

在TBR对tile进行分组的过程中，这个阶段其实也可以做一些优化：通常 TBR GPU 都会配备叫做隐藏面消除（HSR，Hidden Surface Removal）的功能。其核心思想是：如果一个图元被其他不透明图元完全遮挡的话，这样我们就可以完全不去计算它，减少实际上对最终画面没有影响的计算（overdraw）。隐藏面消除有很多实现，例如 Mali 有两套机制：Early-Z + FPK（Forward Pixel Kill）去分别从正反两个方面去去除无效的图元。
随后，因为我们已经知道了每个bin中的图元对应在屏幕上的tile，我们只要按照顺序去一个一个光栅化和着色这些图元就可以了。

因此，TBDR的优势在于利用PB中缓存的顶点数据，提前对流入到管线剩余部分的片段进行了筛选，来解决传统渲染管线的一个老大难问题——过度绘制（over draw）。

---

**TBR vs TBDR 架构差异对比：**

| 维度                     | TBR（Tile-Based Rendering）  | TBDR（Tile-Based Deferred Rendering） |
| ---------------------- | -------------------------- | ----------------------------------- |
| 架构核心目标                 | **降低外部带宽访问**               | **同时降低带宽 + 减少无效片元计算（Overdraw）**     |
| Pipeline 切分方式          | 顶点阶段 ↔ 片元阶段分离              | 顶点阶段 ↔ **延迟的片元阶段（带筛选）**             |
| Tile 划分                | 屏幕划分为固定大小 Tile             | 相同                                  |
| Binning 阶段             | 将 Primitive 分配到 Tile（bin）  | 相同，但 **附加可见性/遮挡信息收集**               |
| Parameter Buffer（PB）内容 | 顶点数据 + Primitive 列表        | 顶点数据 + Primitive 列表 + **潜在深度/覆盖信息** |
| Tile Memory 使用         | 存放颜色 / 深度 / stencil 等中间结果  | 相同，但 **生命周期更短、写回更少**                |
| Overdraw 处理            | **被动减少**（仅因 Tile Locality） | **主动剔除**（HSR / Early-Z / FPK 等）     |
| Hidden Surface Removal | ❌ 通常没有或非常有限                | ✅ 架构级支持（PowerVR HSR / Mali FPK）     |
| Fragment Shader 调用     | 对 bin 中所有 Primitive 逐片元执行  | **只对最终可见片元执行**                      |
| 深度测试时机                 | Raster / Fragment 阶段       | **Raster 前或极早期**                    |
| 外部 Memory 访问           | Tile 完成后写回                 | 写回次数更少，深度/颜色冗余显著降低                  |
| 对不透明物体的优化              | 中等                         | **极强（最理想场景）**                       |
| 对透明物体的优化               | 有限                         | 有限（通常 fallback 到传统路径）               |
| 对 Shader 复杂度的敏感性       | 较高                         | **更不敏感（因减少 FS invocation）**         |
| 实现复杂度                  | 中等                         | **高（需要精确的可见性判定）**                   |
| 典型代表                   | 早期 PowerVR                 | PowerVR TBDR、ARM Mali、部分 Adreno     |

---

### 题外话：TBDR 中的隐面剔除

**「隐面剔除」** 技术这一术语来自于 PowerVR 的 HSR (Hidden Surface Removal)，通常用来指代 GPU 对最终被遮挡的 Primitive/Fragment 做剔除，避免执行其 PS，以达到减少 Overdraw 的效果。

尽管类似诸如 Depth Prepass （在第一个 pass 中，只渲染物体的深度信息到depth buffer，后续的着色 pass 使用深度测试剔除被遮挡的 fragment）这样的技术已经实现了通过预渲染深度的方式降低fragment overdraw，HSR 算的上是基于硬件实现的discard fragment(由于binning 阶段已经有了深度信息)。

Adreno/Mali/PowerVR 三家在处理隐面剔除除的方式是不一样的。

1. PowerVR 的 HSR 原理是生成一个 visibility buffer，记录了每一个 pixel 的深度，用于对 fragment 做像素级的剔除。因为是逐像素级的剔除，因此需要放到 Rasterization 之后，也就是说每个三角形都需要做 Rasterization。根据 visibility buffer 来决定每一个像素执行哪个 fragment 的 ps。也因此，PowerVR 将自己 TBR (Tile Based Rendering) 称为 TBDR (Tile Based Deferred Rendering)。 而且特别提到了一点，如果当出现一个 fragment 的深度无法在 vs 阶段就确定，那么就会等到 fragment 的 ps 执行完，确定了深度，再来填充对应的 visibility buffer。也就是说这个 fragment 会阻塞了 visibility buffer 的生成。 这个架构来自于 PowerVR 2015年左右的文档，后续 Apple 继承了其架构，但是后面是否有做更进一步的架构优化不得而知。
</br>

2. Adreno 实现隐面剔除技术的流程称为 LRZ (Low Resolution Depth)，其剔除的颗粒度是 Primitive 而不是 Fragment。在 Binning pass 阶段执行 Position-Only VS 时的会生成一张 LRZ buffer （低分辨率的 z buffer），将三角形的最小深度与 z buffer 做对比，以此判断三角形的可见性。Binning pass 之后，将可见的 triangle list 存入 SYSMEM，在 render pass 中再根据 triangle list 来绘制。相比于 PowerVR 的 HSR，LRZ 由于是 binning pass 做的，可以减少 Rasterization 的开销。并且在 render pass 中，也会有 early-z stage 来做 fragment 级别的剔除。 对于那种需要在 ps 阶段才能决定深度的 fragment，就会跳过 LRZ，但是并不会阻塞管线。
</br>

3. Mali 实现隐面剔除的技术称为 FPK (Forward Pixel Killing)。其原理是所有经过 Early-Z 之后的 Quad，都会放入一个 FIFO 队列中，记录其位置与深度，等待执行。如果在执行完之前，队列中新进来一个 Quad A，位置与现队列中的某个 Quad B 相同，但是 A 深度更小，那么队列中的 B 就会被 kill 掉，不再执行。 Early-Z 只可以根据历史数据，剔除掉当前的 Quad。而 FPK 可以使用当前的 Quad，在一定程度上剔除掉老的 Quad。 FPK 与 HSR 类似，但是区别是 HSR 是阻塞性的，只有只有完全生成 visibility buffer 之后，才会执行 PS。但 FPK 不会阻塞，只会kill 掉还没来得及执行或者执行到一半的 PS。

![](imgs/_08.png)
![](imgs/_07.avif)

---

**TBDR 架构中主流隐面剔除技术对比：**

| 维度                    | **PowerVR – HSR**<br>(Hidden Surface Removal) | **Adreno – LRZ**<br>(Low Resolution Z)  | **Mali – FPK**<br>(Forward Pixel Killing) |
| --------------------- | --------------------------------------------- | --------------------------------------- | ----------------------------------------- |
| 架构归类                  | **TBDR（Tile-Based Deferred Rendering）**       | **TBR（Tile-Based Rendering）+ Early-Z**  | **TBR（Tile-Based Rendering）+ 动态像素剔除**     |
| 剔除核心思想                | 构建 **Visibility Buffer**，像素级决定“谁最终可见”         | 构建 **低分辨率深度图**，提前判断三角形是否可能可见            | 利用 **在途 Quad 之间的前后关系**，动态 kill 旧像素        |
| 剔除粒度                  | **Fragment / Pixel 级**                        | **Primitive（三角形）级 + Fragment 级**        | **Quad（像素块）级**                            |
| 主要发生阶段                | **Rasterization 之后，PS 之前**                    | **Binning pass（Position-only VS）阶段**    | **Early-Z 之后、PS 执行之前/期间**                 |
| 是否需要完整 Rasterization  | **是**（每个三角形都要光栅化）                             | **否**（被 LRZ 判定为不可见的三角形不会进入 render pass） | **是**（Quad 已生成）                           |
| 是否阻塞管线                | **是（阻塞型）**                                    | **否（前向剔除）**                             | **否（非阻塞，投机执行）**                           |
| 深度信息来源                | Visibility Buffer（逐像素最小深度）                    | 低分辨率 Z Buffer（如 8×8 / 16×16）            | Early-Z 结果 + 在途 Quad 深度                   |
| 不确定深度（PS 才能写 Z）       | **会阻塞**：必须等 PS 完成才能更新 visibility buffer       | **直接跳过 LRZ**，但不影响流水线                    | **无法提前 kill，但可被后续更近 Quad 反杀**             |
| 对 Overdraw 的抑制        | **极强（理论最优）**                                  | **中等（依赖场景深度一致性）**                       | **中-强（依赖队列长度与时序）**                        |
| 对 Rasterization 成本的影响 | **无法减少**                                      | **显著减少**                                | **无法减少**                                  |
| 对 PS 成本的影响            | **最大化减少**（只执行最终可见 fragment）                   | **减少一部分**（被 early-z 剔除）                 | **减少一部分**（被 kill 的 Quad 不再执行或中断执行）        |
| 典型优势                  | 最小 PS 执行量，理论上零 overdraw                       | 高效、稳定、对流水线友好                            | 不阻塞、对动态场景鲁棒                               |
| 典型劣势                  | 管线阻塞，对 late-Z 极其不友好                           | 粒度粗，可能保留不少“假可见”三角形                      | 依赖 FIFO 深度，效果非确定性                         |
| 代表厂商/产品               | PowerVR<br>（Apple GPU 早期继承该思想）                | Qualcomm Adreno                         | ARM Mali                                  |

---

## 5. TBR的优势：如何实现低功耗高效率？

![](imgs/_17.png)

我们来重点关注一下着色阶段，这才是 TBR 架构真正发挥威力的地方——不到最后一刻，它根本不会去碰显存！注意到 TBR 架构的 GPU 上都有一块专门的 Tile Memory （GMEM），这块空间是和 Cache 一样的 SRAM 打造的，因此有极高的带宽、极低的延迟以及并不需要耗电刷新——也就是功耗低。这块存储空间就像是草稿纸，着色阶段会直接从里面读取之前的数据，也可以往里写入数据去渲染——都是以 Cache 级的极低功耗和延迟、极高带宽来完成的。最后的最后，在所有渲染流程结束的时候，会将GMEM中的内容写入到显存中。请注意，这个箭头是单向的——TBR 架构只会在最终所有工作都完成的时候，才会将这个 tile 写入显存中）这个过程被称之为 Resolve）。随后，GMEM 中的内容可以被完全清空——也不会再需要这些内容了——省去了不少显存带宽的roundtrip、也节省了不少显存容量！

这种分块的特性，使得 MSAA、Alpha Test、Alpha Blend 带来的性能消耗将会变得非常小！例如 MSAA 多重采样、Alpha Test 的片段剔除、Alpha Blend 的混合操作，都只会发生在 Tile Memory 中，因此不会像普通的 IMR GPU 那样带来非常大的显存的带宽消耗。

不仅如此，因为很多 render pass 的并不需要上屏，所以这些 render pass 的帧缓冲区根本不需要写到显存里——直接开始下一个 render pass，并引用已经在 GMEM 中的数据就可以了。

题外话：厂商还是觉得最后这个帧缓冲的写入带来了太多的显存带宽消耗，因此还催生出了例如 AFBC （ARM Frame Buffer Compression）这样的技术，来进一步压缩显存的流量。当然，由于纹理在渲染时也需要被 GPU 读取，因此也产生了 ASTC 这样的纹理压缩技术，其可以分块解压，非常适合 TBR 架构的 GPU （可以分块地将纹理数据存储在 GMEM 中）。

总之，TBR架构的核心思想是将整个屏幕划分为多个Tile并使用tile memory来进行渲染，减少了对于FrameBuffer频繁的读取和写入，从而大大减少带宽损耗。

</br>

> ASTC 纹理压缩：
> **ASTC（Adaptive Scalable Texture Compression）**，即自适应可扩展纹理压缩，是一种高效的纹理压缩格式，由ARM公司开发。
> ASTC支持多种压缩级别和块大小，允许开发者根据具体需求灵活选择。ASTC的压缩分块从4x4到12x12不等，最终可以压缩到每个像素占用1bit以下。
> ASTC采用块压缩的思想，将纹理分为多个小块进行压缩，每个块可以有不同的压缩格式和压缩比，从而根据纹理的特性进行灵活调整。
> 
> ![](imgs/屏幕截图_6-1-2026_33350_developer.arm.com.jpeg)
> [参考](https://community.arm.com/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/how-low-can-you-go-building-low-power-low-bandwidth-arm-mali-gpus)

## 6. TB(D)R的代价：哪些特效难以胜任？

尽管 TBR 有很多好处：降低了很多来自屏幕空间大小的缓冲区的内存带宽消耗，而且可以带来非常低消耗的抗锯齿实现。那么代价是什么呢？

如前所述，在TBR架构中，GPU内部集成有很小的片上缓存（On-Chip Buffer），用于临时存储每个Tile的渲染结果。渲染时，先将一个Tile内的图像渲染到片上缓存，然后再拷贝到主显存中。
回顾下 TBR 的渲染流程，它将屏幕分割成小块（Tile），然后分别对每个小块进行对应的三角形变换，着色，深度测试（binning过程是在Vertex阶段之后），将输出的几何数据存储到显存，然后才被FragmentShader读取。

因此，在这个过程中，几何数据过多的管线，容易在此处有性能瓶颈。如果场景中的mesh如果非常复杂，会直接影响 TBR GPU 的性能。TBR GPU 的这种做法，让大多数的手游都没办法使用非常高精度的模型。减面、降低顶点复杂度，是移动端图形开发至今为止都绕不开的话题。也正因如此，mesh shader 这种在桌面端初露头角的技术，在目前的移动端很难推广开去。（例如 Nanite 的假设就是，模型精度远高于屏幕分辨率精度，三角形可以大量地小于一个像素的大小。这种场景直接用 TBR GPU 渲染，就是最坏的情况：顶点的数量 ~= 或者 > 像素的数量）。

其次，TBR GPU 还有一个问题。其对屏幕空间的效果（比如 SSAO 屏幕空间环境光遮蔽/SSR 屏幕空间反射等），是根本没有招架之力的。因为屏幕空间效果，如SSAO和SSR，通常需要访问整个屏幕或邻近像素的数据。例如，SSAO需要根据当前像素周围的深度信息来计算环境光遮蔽，而SSR则需要根据屏幕空间的反射信息来生成反射效果。这些效果通常需要在像素着色器中采样周围的像素，这意味着需要访问其他Tile的数据。

而如果是 TBR GPU 上，在这种情况下的 TBR 带来的优势都荡然无存了，TBR架构的优势在于处理局部数据，每个Tile独立处理。然而，屏幕空间效果需要跨Tile访问数据，例如，某个Tile中的像素可能需要访问相邻Tile中的深度或颜色信息。这会导致TBR GPU在处理这些效果时，无法有效利用Tile内存，因为所需的数据可能不在当前Tile的内存中，需要从系统内存中读取，这会增加带宽消耗和延迟。举个极端的例子，在 SSR/SSGI 的射线步进（Ray-Marching）可能随机的遍历数十个像素，涉及多个 Tile，需反复从主存加载数据。

同时，这也就意味着移动端要尽可能的减少 renderpass 的切换，因为在同一个renderpass内，所有的shading操作都可以在 GPU 的片上内存（Tile Memory）完成，在一个renderpass结束后，再一次性写回System Memory的framebuffer上，并结合render pass上指定RT（render target）的Load/Store Op来进一步降低带宽开销。

比如，屏幕空间效果通常需要多个渲染Pass。例如，先渲染G-Buffer，再进行SSAO计算，最后进行光照和合成。每个Pass之间需要将中间结果保存到系统内存，并在下一个Pass中重新加载。TBR GPU原本的优势在于减少中间结果的系统内存访问，但多Pass的屏幕空间效果会破坏这种优化，导致频繁的Load和Store操作，增加带宽压力。

值得注意的是，在这些pass里，很多游戏会选择将中间的辅助pass降到半分辨率。这样可以在保证效果的同时，降低显存带宽。（特别是低频的信息，例如动态模糊、bloom这样的效果）（p.s. 米哈游的游戏基本上都会带这个半分辨率的优化，但snl是都没有这些考量的）。

例如，以《崩坏：星穹铁道》为例，在 SSR（屏幕空间反射）的pass中，其中深度和法线的RT都是以当前屏幕分辨率的一半进行，这样可以有效降低显存带宽的开销和ps的压力，减少显存占用，并提高L1/TEX Cache的命中率。

![](imgs/_10.png)
![](imgs/_09.png)

> 补充说明：
> 
> - 即使通过算法优化减少采样或合并 Pass，跨 Tile 访问的硬件代价依然存在。
> - 在移动端等带宽敏感场景，频繁的fallback memory latency被进一步放大。

总结：TBR 的 **分块隔离设计** 与屏幕空间效果的 **全局随机访问需求** 直接冲突，导致数据被迫在主存与 Tile Memory 间反复搬运，带宽和延迟成为无法绕过的瓶颈。

## 7. Mobile GPU 性能优化策略：开发者怎么做？

在聊到渲染管线这块之前，有些概念还是希望大家稍微有所了解：

1. 一次 RenderPass 对于 Tiled-Based GPU 意味者什么？
   我们可以简单的认为，一个 Render Pass 意味着执行一次渲染管线，所有对 Tile 的操作（几何光栅化、着色、混合等）均在 Tile Memory 中完成，仅在该 Pass 结束时将最终结果写回System Memory（Save Action）。（例如，如果后续的RenderPass需要用到当前 FBO Attachment 的内容，或者需要将结果输出到屏幕，就必须保存。）
2. Load/Store Action
   这两个操作直接决定了帧缓冲附件（FBO Attachments）在渲染过程中的内存管理策略，避免不必要的加载和保存，显著减少主存与 Tile Memory 间的数据传输，从而优化渲染性能。
   - **Load Action** 定义了在开始渲染到某个附件之前，如何初始化该附件的内存内容，例如：CLEAR/LOAD/DONT_CARE。
   - **Save Action** 决定了在渲染完成后，如何处理附件的内容，例如：STORE/DONT_CARE。

如果上面的概念很难理解，则记住核心的两点：

1. 一次 RenderPass = 所有 Tile 操作在 Tile Memory 中完成 + 按需回写 System Memory（Save Action）。
2. 是否回写取决于后续操作是否需要数据，而优化目标是 最小化回写次数（通过合并 RenderPass 或使用 Subpass）。

实际上现在一个游戏渲染一帧，很大概率并不是只走一次上面的管线。走一遍上面的管线，现在被称作一个 render pass。而一个render pass渲染出来的帧，很有可能作为其他render pass的输入纹理。例如 shadow map，就需要一个 shadow pass 来计算场景到光源的深度图，随后再在主颜色pass中采样深度图，比较深度，决定是否产生阴影。在这个过程中，不同render pass之间，难免也会存在经过显存的写入和读取操作（每个Attachment的Load Action和Save Action）。

![](imgs/_14.png)

在 Vulkan/DirectX12 这样现代的 Graphics API 以前，是没有 Render Pass 这样的概念的，一般就是指画在同一个RT的 drawcall。

当然这在业界也有解决办法——让应用程序通过 API 层面的 hint，告诉 GPU 硬件这个 Render Pass 要怎么处理。Vulkan 就存在 subpassLoad（input attachment） 的概念，告诉驱动，这个 Render Pass 的输出帧是在别的 pass 中有用到。
其实类似vulkan的这个subpass概念在其他图形 API 中也有所涉及，在metal中叫programmable blending，在更老的opengl es中也是framebuffer fetch和depth stencil fetch（或者较少人用的 PLS (pixel local storage)）。

苹果 Metal 2/A11 中推出的 Tile Shader 和 ImageBlock，实际上也是一个对 Tile Memory 的抽象。并且通过这两个设计，Metal 将 Tile Memory 的控制权完全交给了 Metal 开发者，不愧是有 IMG 血缘的架构啊。值得庆幸的是，DirectX 12 里有 Render Pass 的概念，可以通过 D3D12_FEATURE_D3D12_OPTIONS5 中的 RenderPassesTier 来检查 GPU 驱动对这个特性的支持。但是很可惜的是，现在绝大多数桌面 GPU 都不支持这个特性。这意味者开发者都不会针对这个特性去优化。（X Elite 的 X1-85 支持这个特性，但是在这上边跑的所有 DirectX 12 游戏，应该都没有在代码层面专门调用 Render Pass 相关 API，更不用说专门为这个特性去优化渲染算法了）。

### 应用：Subpass 与 One Pass Single Deferred

通过对 vulkan 的 subpass 介绍可以看出，我们不就可以借助 subpassLoad 特性来完成本来需要多个pass才能做的事情了吗？例如在现代的游戏中，延迟渲染（Deferred Shaing）被广泛的使用，通过将几何与光照两个pass进行分离，完成最终的着色。由于lighting pass对gbuffer的读写是在当前着色像素上完成的，这无疑很适合subpass设计渲染管线。

![](imgs/v2-c8c8b4bffbbed77ff8445841dad9570e_r.png)

> [参考](https://www.saschawillems.de/blog/2018/07/19/vulkan-input-attachments-and-sub-passes/)

以 UE4 Mobile Renderer 为例，让我们来看看 mobile deferred shading 是如何在 tile-based gpu 上高效进行的

![](imgs/_11.png)

Vulkan的做法为：将Render Pass分为多个subpass，每个subpass有自己要执行的任务，将它们放在一个pass可以方便我们表达GBuffer pass和lighting pass之间的依赖关系。这个依赖关系会被GPU driver使用，将多个subpass合并成One single pass。这样我们就不需要把GBuffer store回system memory了。每个subpass都需要声明自己读写的attachment。用于获取input attachment的语法为SubpassLoad，通过它我们就可以在lighting subpass中获取当前像素GBuffer的数据。

![](imgs/_12.png)

Mobile deferred shaderer中最后会得到3个subpass：1.Gbuffer，2.Decal（写入GBuffer），3.Lighting+Transparent（写入SceneColor）。结束后，只将SceneColor写入system memory即可。

![](imgs/_13.png)

OpenGL ES的话则通过extension来实现Mobile Deferred Shading：pixel local storage（Mali和PowervR支持，Adreno不支持，且无法store回system memory），framebufferfetch（Adreno支持，Mali不完全支持，Mali只能读取color0 attachment）。所以UE需要在runtime的时候根据GPU型号改变Shader代码。UE4.27会完全支持PLS和FBFetch。

> 参考：[UOD2020]虚幻引擎4在移动平台上的更新|EpicGames Jack Porter&Dmitriy Dyomin(官方字幕)

我们已知，在 Vulkan 中，一个 RenderPass 中包含多个 SubPass，由于 TBDR 架构，在一个 Tile 上的多个 SubPass 都是按顺序执行（顺序需要手动指定）以便可以让当前 SubPass 使用上一个 SubPass 的结果（通过 Input Attachment 实现），只有当这个 Tile 所有的 SubPass 都执行完了之后，才会把渲染结果写回 System Memory。

这样的好处在于，SubPass 对于 FrameBuffer 的读写带宽开销可以省略很多，有效的利用了 Tile-Based 架构特性节约内存带宽。特别是对于延迟渲染管线而言，将 GBuffer Pass 和 Deferred Lighting Pass 以 SubPass 的形式放在同一个 Render Pass 中执行，可以实现 One Pass Defer，能够让GBuffer 在 On-Chip Memory 保存并直接让 LightingPass 使用，计算完毕后，一次性写入 FrameBuffer，从而大大减少内存损耗，极大地节省了系统带宽。此外，Vulkan中还有 Transient Attachments 机制，针对 GBuffer 在 Geometry Pass 被写入，在 LightingPass 计算完毕后便可丢弃的情况，便可以设置为 Transient Attachments 进一步优化内存分配。

## 8. 对比与总结

传统多Pass延迟渲染管线的问题：

**(1) 传统多 Pass 延迟渲染（带宽敏感）**

```mermaid
graph LR
A[GBuffer Pass] -->|Store G-Buffer| B(System Memory)
B -->|Load G-Buffer| C[Lighting Pass]
```

问题：

- 若 Pass B 依赖 Pass A 的结果（如deferred lighting pass 依赖gbuffer pass的attachment），则 Pass A 的附件必须设为 STORE，Pass B 的附件需设为 LOAD。
- 带宽代价：在这个例子中，G-Buffer 需通过 STORE 写回主存，Lighting Pass 需通过 LOAD 重新加载到 Tile Memory，造成了两次主存带宽消耗（写 + 读），破坏了 Tiled-Based GPU 的带宽优势。

**(2) Subpass 优化后的延迟渲染（零主存带宽）**

```mermaid
graph LR
A[GBuffer Subpass] -->|直接传递| B[Lighting Subpass]
```

关键操作：

- 不执行 STORE 和 LOAD：G-Buffer 数据全程保留在 Tile Memory 中。
- 输入附件（Input Attachment）：Lighting Subpass 通过 subpassLoad 直接读取同一 Tile 内的 G-Buffer 数据。

通过 Subpass 的 subpassLoad 特性，Tiled-Based GPU 实现了：

1. 片内零成本数据传递：避免 G-Buffer 的主存回写与重载。
2. 带宽敏感操作原子化：将延迟光照等操作限制在 Tile Memory 内。
3. 与现代 API 深度结合：Vulkan/Metal 的 Subpass 设计天然适配 Tiled 架构。

## 9. 实测验证：用代码看出GPU的“内心世界”

说完了理论的内容，让我们通过软件观察桌面端与移动端 GPU 渲染图元的差异！
写了一个小程序（在github搜索TriangleBin），这是个渲染顺序的检测器。可以支持在 Windows 和 Android 下面跑。在 Windows 下用 OpenGL 4.x，在 Android 下用 OpenGL ES 3.2。它的原理是在像素着色器的每次调用里都给一个计数器加一，然后超过固定的数量就停止渲染。通过这么一个小程序，我们就可以知道 GPU 是按照什么顺序给画面上的像素着色的了。

设备1：NVIDIA GeForce RTX 4060 Ti

![](imgs/_15.png)

设备2：Qualcomm Adreno (TM) 650（Qualcomm Snapdragon 865 (SM8250)）（Device: Xiaomi Redmi K40）

![](imgs/QQ图片20250127001105.jpg)

设备3：Qualcomm Adreno (TM) 530（Qualcomm Snapdragon 821 (MSM8996 Pro)）（Device: Xiaomi MI 5s）

![](imgs/QQ图片20250127001046.jpg)

测试可以看出，桌面 GPU 其实仍然很大程度上是 IMR 的流程，但是也可以看到，这个顺序并不像传统意义上的 IMR 那样从上往下、从右往左去着色每个像素。其实这个过程中也有一些分块的现象出现。但是由于渲染顺序还是一个个三角形这样处理，因此并不能把它们划入 TBR 的范畴。
而移动端 GPU 中，Mali 是典型的 TBR，我们可以很明显的看到整个屏幕被分成很多小块，GPU 先对每个tile分别渲染，而且顺序是按照块状的一个渲染顺序执行的，它是从右到左，再从左到右的这一个渲染顺序，最终完整的呈现在屏幕上。

而高通的 Adreno GPU 则有一些特殊，它似乎在一些几何非常简单的情况下，会直接放弃分块，转而执行类似 IMR 的行为。而即使在几何复杂到出现 TBR 的时候，它也是在每个 Tile 内执行类似 IMR 的渲染操作（逐个图元渲染），而不是像 Mali 那样先算好 visibility 然后再执行每个像素上的着色操作。

## 10. 总结与展望

通过对移动端 GPU TBR/TBDR 以分块渲染为核心，通过片上缓存与隐面剔除技术，在性能、功耗、带宽间寻求平衡，结合现代图形 API（如 Vulkan Subpass），最大化利用 Tile Memory，可以看出软硬件厂商都在为提高移动端 GPU 渲染性能出谋划策，为移动端提供了低功耗、高带宽效率的渲染方案，但是仍旧需要开发者针对 Tile-Based GPU 特点对渲染管线做针对性的优化，才能最大限度的提高移动端 GPU 的渲染效率。
