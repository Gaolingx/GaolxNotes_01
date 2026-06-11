# 移动端GPU不支持WorkGraphs

### 🧑‍💻 User

为什么mobile gpu没有support work graphs？

---

### 🤖 Assistant

Mobile GPU（移动端显卡，如高通 Adreno、ARM Mali、苹果 GPU 等）目前尚未支持 **Work Graphs（工作图）**，根本原因在于**移动端 GPU 的硬件架构（TBDR）、功耗限制、内存带宽以及现阶段的 API 生态**与 Work Graphs 的设计理念存在严重冲突。

Work Graphs 是 DirectX 12 引入的一项前沿技术，旨在实现完全的“GPU 驱动渲染（GPU-Driven Rendering）”，让 GPU 能够动态地自我调度任务，而无需 CPU 干预。

以下是 Mobile GPU 目前不支持 Work Graphs 的核心原因：

### 1. 硬件架构冲突：TBDR vs IMR
桌面端 GPU（NVIDIA, AMD）通常采用 **IMR（立即渲染模式）**，对算力的突发和无序处理能力极强。而移动端 GPU 为了省电，几乎全部采用 **TBDR（基于图块的延迟渲染）**。

*   **TBDR 的工作方式：** 需要先将整个画面的几何体处理完（Binning 阶段），然后把画面分成一个个小图块（Tile），再逐个图块进行像素渲染。这种架构高度依赖**可预测的、结构化的渲染管线**，并且数据要尽可能留在 GPU 内部极小的 SRAM 中。
*   **Work Graphs 的特点：** 它本质上是一个动态的任务流（DAG，有向无环图）。一个计算着色器可以随时“派生（Spawn）”出成百上千个新的着色器任务。这种**高度动态、不可预测、随时产生新几何体或计算任务**的特性，会直接破坏 TBDR 的 Binning 流程，导致数据频繁溢出到主存，完全抵消了移动端 GPU 的架构优势。

### 2. 硬件调度器（Hardware Scheduler）的缺失
Work Graphs 的核心在于 GPU 芯片内部需要有一个极其强大的**硬件级任务调度器**。
*   在桌面端（如 RDNA 3 或 Ada Lovelace），硬件调度器可以实时管理复杂的节点队列，动态分配寄存器和线程组。
*   在移动端，受限于**芯片面积（Die Size）和功耗**，移动 GPU 的指令分发和调度器设计得相对简单。要在寸土寸金的手机 SoC 上塞进一个能处理 Work Graphs 复杂逻辑的硬件调度器，不仅成本高昂，而且会大幅增加漏电和发热。

### 3. 内存带宽与功耗墙（Power & Bandwidth）
Work Graphs 在节点之间传递数据时，需要频繁地读取和写入显存（VRAM）中的队列缓冲（Queue Buffers）。
*   桌面显卡拥有独立的 GDDR6 显存，带宽动辄 $500 \text{ GB/s}$ 到 $1 \text{ TB/s}$，且功耗可以达到 $300\text{W}+$。
*   移动端 GPU 共享手机的 LPDDR 内存，带宽通常只有 $50 \text{ GB/s}$ 左右，且整个 SoC 的持续功耗被严格限制在 $5\text{W} - 8\text{W}$ 以内。
如果移动 GPU 运行 Work Graphs，频繁的内存读写会瞬间榨干可怜的内存带宽，并导致手机严重发热降频。

---

<details>
<summary><b>点击展开：API 生态与行业优先级的次要原因</b></summary>

### 4. API 生态的错位
*   **Work Graphs 是微软 D3D12 的标准：** 它最初是由微软联合 AMD、NVIDIA 在 DirectX 12 Ultimate 中推出的。
*   **移动端的 API：** 移动端主要使用 Vulkan、Metal 和 OpenGL ES。虽然 Vulkan 也有设备生成命令（Device Generated Commands）的扩展，但要形成像 Work Graphs 这样成熟的跨平台标准还需要很长时间。苹果的 Metal 虽然在 GPU-Driven 方面很先进，但也更倾向于适合自身架构的实现方式，而非照搬桌面端的 Work Graphs。

### 5. 移动端渲染的痛点不同
目前移动端游戏开发者的首要任务并不是消除 CPU 提交（Draw Call）的开销，而是：
1.  **降低功耗和发热**
2.  **空间超分和插帧（如移动端 FSR/DLSS 的替代方案）**
3.  **支持基础的硬件光线追踪（Ray Tracing）**

在 CPU 瓶颈方面，移动端开发者通常使用更为传统的 `Indirect Draw / Dispatch`（间接绘制/调度）技术，这已经能够满足绝大多数移动端手游的需求，暂时没有迫切需要引入 Work Graphs 的动力。
</details>

---

### 未来展望
Mobile GPU 未来**一定会有**类似的技术，但不会直接照搬桌面端的 Work Graphs。
移动端厂商（如 ARM、高通）和 API 维护者（Khronos Group）目前正在研究**适合 TBDR 架构的轻量级 GPU 驱动渲染方案**。未来的演进方向可能是：允许 GPU 在图块渲染（Tile-pass）内部进行有限的自我调度，从而在不破坏移动端功耗优势的前提下，释放 CPU 的压力。