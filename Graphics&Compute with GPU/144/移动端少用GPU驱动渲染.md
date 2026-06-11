# 移动端少用GPU驱动渲染

### 🧑‍💻 User

在图形渲染中，有一种叫做drawinstanceindirect/dispatchindirect的技术（又称gpu driven），即让gpu自行调度pass启动和执行，为什么mobile gpu很少使用？

---

### 🤖 Assistant

在图形渲染中，`DrawIndirect` 和 `DispatchIndirect`（即 GPU-Driven Rendering 的核心）在 PC 和主机端是提升性能的利器。通过让 GPU 的 Compute Shader 进行剔除（Culling）、LOD 计算，并直接将绘制参数写入 Buffer 交由 GPU 调度，可以极大地降低 CPU 提交 Draw Call 的开销。

然而，在移动端（Mobile GPU）上，这种技术却较少被大规模使用。这主要归结于**移动端 GPU 独特的硬件架构（TBDR）、内存带宽限制以及生态碎片化**。

以下是具体的深层原因：

### 1. 核心冲突：TBDR 架构与同步开销 (Synchronization Overhead)

桌面 GPU 通常是 **IMR (Immediate Mode Renderer, 立即渲染模式)**，而移动端 GPU 绝大多数是 **TBDR (Tile-Based Deferred Rendering, 基于图块的延迟渲染)**，如高通 Adreno、ARM Mali、苹果 Apple Silicon。

在 TBDR 架构下，一帧的渲染被严格分为两个阶段：
1.  **Binning / Tiling Pass (几何与分块阶段)：** 执行所有的 Vertex Shader，计算图元位置，并将其分配到屏幕对应的 Tile 列表中，写入系统内存（Parameter Buffer）。
2.  **Rasterization / Fragment Pass (光栅化与像素阶段)：** GPU 将一个个 Tile 的数据加载到高速片上内存（On-chip SRAM），执行 Fragment Shader，最后再写回系统内存。

**GPU-Driven 在 TBDR 上的痛点：**
当我们使用 Compute Shader 进行剔除并生成 `DrawIndirect` 的参数时，数据流向是：`Compute Shader -> System Memory -> Command Processor (读取 Indirect Buffer) -> Vertex Shader`。
这导致了一个**极其昂贵的同步点 (Pipeline Barrier/Bubble)**。移动端的 Command Processor 必须等待前面的 Compute Shader **完全执行完毕，并将数据写回较慢的系统内存**，然后才能解析 Indirect 参数以启动后续的 Binning Pass。这种跨阶段的内存屏障会打断 TBDR 的流水线并行，导致 GPU 出现巨大的闲置气泡（Bubble），反而可能比 CPU 直接提交 Draw Call 慢得多。

---

为了方便阅读，以下次要原因已折叠：

<details>
<summary><b>2. 内存带宽与功耗陷阱 (Bandwidth & Power Constraints)</b></summary>
<br>
在移动设备上，发热和电池续航是生命线。功耗与带宽的关系大致成正比，即 $Power \propto Bandwidth$。

*   **桌面端：** GPU 拥有专用的高带宽 VRAM（GDDR6 等），读写 Indirect Buffer 极快。
*   **移动端：** GPU 与 CPU 共享系统内存（LPDDR），带宽极其珍贵。
GPU-Driven 流程要求 GPU 将海量的 Instance 数据和 Indirect Command 写入系统内存，然后再由 Command Processor 读出来。这种频繁的 **System Memory Round-trip (内存往返)** 在移动端会消耗大量的显存带宽，导致设备迅速发热降频。相比之下，如果 CPU 本身性能有富余，让 CPU 进行简单的视锥体剔除（Frustum Culling），反而能节省 GPU 宝贵的带宽。
</details>

<details>
<summary><b>3. 驱动与硬件生态的碎片化 (Driver Fragmentation)</b></summary>
<br>
Android 生态极其复杂，不同厂商、不同年代的 GPU 驱动质量参差不齐。
*   很多中低端 Android 设备的 OpenGL ES / Vulkan 驱动对 `Indirect` 调用的支持存在 Bug，或者底层实现依然是 CPU 回读（Readback）后再提交，完全丧失了 GPU-Driven 的意义。
*   一些老旧的 Mali GPU 在执行 `DispatchIndirect` 时，其 Command Streamer 效率极低。
为了保证游戏在数千款 Android 机型上的兼容性和稳定性，商业引擎（如 Unity, Unreal）通常会选择更保守、但在各平台上表现更一致的 CPU 剔除 + 传统 `DrawInstanced` 方案。
</details>

<details>
<summary><b>4. 移动端场景复杂度不够 (Scene Complexity)</b></summary>
<br>
GPU-Driven 解决的核心痛点是：**百万级物体的超大量 Draw Call 导致的 CPU 瓶颈**。
然而，受限于屏幕大小、分辨率、整体算力以及包体大小，移动端游戏（哪怕是原神、崩坏：星穹铁道这样的重度游戏）同屏的物件数量和多边形数量远远达不到 PC 3A 游戏的量级。
在移动端典型的几百到几千个 Draw Call 的场景下，现代移动端 CPU（如 Cortex-X 系列或 Apple A 系列）结合引擎的多线程渲染（Multithreaded Rendering / SRP Batcher），已经能够很轻松地把渲染命令准备好。引入复杂的 GPU-Driven 架构往往属于“杀鸡用牛刀”，投入产出比不高。
</details>

### 总结与未来趋势

简而言之，移动端很少使用 `DrawIndirect` 是因为：**它打破了移动端 GPU 引以为傲的 TBDR 硬件流水线，且增加了致命的内存带宽开销。**

**未来会改变吗？**
正在改变。随着 API 和硬件的进化，移动端也在逐渐引入更友好的 GPU-Driven 方案：
1.  **Vulkan 1.3 & Metal 3：** 提供了更完善的底层的同步机制和无绑定（Bindless）架构支持。
2.  **Mesh Shader (网格着色器)：** Apple A15+ (Metal 3) 和高通骁龙 8 Gen 2 (Vulkan) 已经支持 Mesh Shader。Mesh Shader 允许在管线内部进行几何剔除和生成（通过 `Task Shader -> Mesh Shader`），**无需通过系统内存回读**，这完美契合了 TBDR 架构，未来它将很大程度上取代传统的 Compute + `DrawIndirect` 方案，成为移动端 GPU-Driven 的终极形态。