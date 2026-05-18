# 移动GPU架构演进与权衡

### 🧑‍💻 User

为什么很长一段时间，mobile gpu都重点攻克vs和ps的渲染瓶颈，设计了复杂的deferred机制（tbdr），而不是把重点放在gpgpu，例如通过tensorcore加速大规模mm计算，或是增加l2大小，或者优化warp调度能力（scoreboard）以及增加专用copy单元（tma）应对dispatch中的复杂控制流以及async compute latency hide（multi cuda stream）

---

### 🤖 Assistant

这是一个非常深刻且专业的GPU架构问题。长久以来（大约在2007年至2018年间），移动端GPU（如PowerVR、Mali、早期的Adreno）的演进路线与桌面端GPU（NVIDIA, AMD）截然不同。移动端死磕**TBDR（Tile-Based Deferred Rendering）**及相关的VS/PS管线优化，而桌面端则早早迈入了**GPGPU**（通用计算）时代。

导致这种分化的根本原因可以归结为三个词：**功耗（Power）、带宽（Bandwidth）、面积（Area），即严格的PPA限制，以及当时的“市场需求”导向。**

以下是详细的架构解析与历史动因：

### 1. 为什么移动端死磕 TBDR 与 VS/PS 优化？

在移动端，**最大的瓶颈不是算力，而是访存带宽和功耗**。

*   **访存能效比极差：** 在芯片设计中，从外部DRAM读取数据的功耗，是执行一次片上ALU计算（如FMA）功耗的数十倍甚至上百倍。可以用简单的公式表示：$E_{DRAM\_Access} \gg E_{ALU\_Compute}$。
*   **TBDR 的核心目的是“把外部访存转化为片内访存”：** 
    桌面端采用IMR（Immediate Mode Rendering），画完多边形立刻写回显存（VRAM），带宽吃紧。移动端带宽小得多（LPDDR vs GDDR），如果用IMR，电池几分钟就耗尽了，且芯片会严重发热。
    TBDR 通过将屏幕切分为小的Tile，先执行所有的 Vertex Shader (VS)，生成完整的几何信息（Binning），然后利用硬件级**HSR（Hidden Surface Removal，隐面剔除）**，只对绝对可见的像素执行 Pixel Shader (PS)。这样，Depth Test、Blending 等高频读写操作全都在功耗极低的片上 SRAM（Tile Memory）中完成，最后只把颜色结果写回一遍 DRAM。

移动GPU花费大量精力优化VS/PS，是因为当时的**核心KPI是：在2~5W的TDP下，流畅渲染高分辨率的UI和游戏画面。**

---

### 2. 为什么当时不重点做 GPGPU 特性？

你提到的 GPGPU 特性（TensorCore, 大L2, Scoreboard, TMA, Async Compute）在桌面端（如NVIDIA Volta/Ampere架构）是大杀器，但在早期的移动端却是“毒药”或“伪需求”。

<details>
<summary><b>展开查看具体特性的移动端痛点分析 (TensorCore,大L2,调度等)</b></summary>

#### A. 为什么不加 TensorCore (大规模 MM 计算)？
*   **没有业务需求：** 几年前的手机上并没有跑大模型或复杂深度学习的需求。把宝贵的硅片面积拿去做矩阵乘法加速（MAC单元集列），在当时打游戏和刷网页时完全是浪费。
*   **架构分工：** 后来随着移动端AI兴起，苹果、高通、联发科选择的路线是**独立NPU（Neural Processing Unit）**，而不是在GPU里塞TensorCore。NPU专门针对深度学习优化，能效比远高于GPU。手机SOC是异构计算体系，不需要GPU包揽一切。

#### B. 为什么不增加 L2 Cache 大小？
*   **硅片面积（Area）太贵：** SRAM的物理面积非常大。在移动SoC寸土寸金的Die Size中，放巨大的L2 Cache会导致成本飙升。
*   **TBDR 减弱了对大 L2 的依赖：** 因为TBDR已经把最耗带宽的阶段锁死在了局部的Tile Memory中，对全局Cache的依赖相对桌面端IMR要小很多。不过，现代移动SoC（如苹果A系列/M系列，高通8Gen系列）已经开始引入System Level Cache (SLC) 来充当庞大的共享缓存。

#### C. 为什么不优化 Warp 调度（Scoreboard）和复杂控制流？
*   **控制逻辑极其耗电：** NVIDIA为了实现极高效率的线程束调度（Warp Scheduler）和乱序执行，设计了极其复杂的 Scoreboard 和 Dispatch 单元。这些**控制逻辑本身占用的面积和消耗的功耗非常惊人**。
*   **图形渲染对分支预测要求低：** 传统的VS/PS大多是数据并行的SIMD/SIMT操作，控制流很简单（很少有复杂的 `if-else`）。移动GPU的设计哲学是“重计算、轻控制”，宁愿堆ALU，也不愿把面积花在复杂的调度器上。

#### D. 为什么没有 TMA (专用Copy单元) 和 Async Compute？
*   **API 限制：** 早期移动端主流是 OpenGL ES，这是一个严格的单线程同步状态机 API。底层硬件如果做很强的异步计算（Async Compute）和多流（Multi-stream），上层软件也调用不到。
*   **Pipeline 特性：** 桌面端用 Async Compute 是为了在渲染阴影或后处理时，把空闲的ALU塞满计算任务（Latency Hiding）。而移动端的 TBDR 本身就是一个被严格划分为不同阶段（Geometry Pass 和 Raster Pass）的流水线，强行引入复杂的异步计算不仅调度困难，还容易打破 Tile Memory 的局部性优势。

</details>

---

### 3. 时代的转变：现在为何移动 GPU 开始像桌面端靠拢？

历史的演进总是螺旋上升的。如今的移动 GPU（例如高通 Adreno 7xx、Apple A17 Pro / M系列 GPU、Arm Immortalis-G720）**正在疯狂引入你提到的这些 GPGPU 特性**。

导致这种转变的原因是：

1.  **图形 API 的革新：** Vulkan 和 Metal 彻底取代了 OpenGL ES。它们原生支持 Compute Shader 和 Async Compute。
2.  **渲染管线的演进：** 现代移动游戏（如《原神》）大量使用基于 Compute Shader 的后处理、Deferred Shading（区别于TBDR，这里指渲染技术）、GPU Driven Pipeline，这就要求移动 GPU 必须具备强大的 GPGPU 能力和低延迟的 Dispatch 能力。
3.  **光线追踪（Ray Tracing）：** 移动端引入硬件光追后，射线求交导致了极其复杂的 Divergence（线程控制流发散）和巨大的访存随机性。为了应对光追，移动GPU现在**必须**引入类似 Scoreboard 的复杂调度机制、更大的片上缓存（L2/SLC）以及更强的乱序执行能力。
4.  **端侧 AI 大模型：** 虽然有独立的NPU，但很多游戏内的实时计算、图像生成（如端侧Stable Diffusion）开始利用GPU的Compute Shader。因此，类似矩阵加速的指令（如Arm的 `SME` 或 Adreno的点积指令）也被整合进了移动GPU的ALU中。

### 总结

在过去很长一段时间，移动GPU面对的数学题是求解在 $Power < 3W$ 且 $Bandwidth < 30GB/s$ 的极值条件下的最优解，答案就是 **TBDR + 极致优化的图形专用管线**。

而现在，随着制程红利走到个位数纳米，以及移动端游戏画质向3A桌面端看齐（光追、高阶后处理），移动GPU的设计不得不走出“舒适区”，开始在有限的面积里塞入复杂的 GPGPU 调度器、巨大的缓存以及异步计算单元。这也是为什么近两代移动SOC的GPU功耗屡屡突破 $10W$ 的原因之一。

---

### 🧑‍💻 User

目前的mobile gpu对于triton/tilelang这种tilewise kernel支持情况怎么样，对于moe grouped gemm/megakernel这类动态dispatch/稀疏算子相比a100/h100还存在哪些不足？

---

### 🤖 Assistant

这是一个非常前沿且硬核的问题。目前端侧大模型（On-Device LLM）的部署，正在痛苦地经历从“图层级（Graph-level）优化”向“算子层级（Kernel-level）优化”的过渡。

总体而言，**目前的Mobile GPU在面对 `Triton` / `TileLang` 这类基于Tile-wise编程模型的工具时，支持度仍然非常初级且碎片化；而在面对MoE、Grouped GEMM、MegaKernel等复杂的动态/稀疏算子时，与A100/H100相比存在断崖式的底层硬件代差。**

以下是具体的深度对比与不足分析：

### 一、 Mobile GPU 对 Triton / TileLang (Tile-wise Kernel) 的支持现状

`Triton` 和 `TileLang` 的核心思想是**Block-level 编程**。它们极度依赖暴露给程序员的三个硬件抽象：高速共享内存（Shared Memory）、Warp级同步原语、以及硬件级矩阵乘（MMA/Tensor Core）。移动端目前的支持情况可以用“一枝独秀，整体拉垮”来形容：

<details>
<summary><b>1. 苹果生态 (Apple Silicon / Metal)：支持度相对最好，但仍有妥协</b></summary>

苹果由于软硬件高度封闭，且底层硬件设计较早地向统一内存架构靠拢，是目前唯一能跑通 `Triton` 的移动端架构（Triton社区已有实验性的Apple Metal Backend）。
*   **硬件抽象：** Metal 3 暴露了完善的 `threadgroup` 内存（对应Shared Memory）和 `simdgroup_matrix` 函数。
*   **开发体验：** 开发者可以通过Metal直接写出类似CUDA的Tile-wise kernel。但相比NV，Apple的 SIMD group 粒度固定（通常是 $32$），且寄存器堆（Register File）较小，导致在写FlashAttention这类高密度Tile算子时，容易发生 Register Spilling。

</details>

<details>
<summary><b>2. 安卓生态 (Adreno / Mali)：支持极度碎片化，举步维艰</b></summary>

对于高通（Adreno）和联发科（Mali/Immortalis），用 `Triton` 几乎无从下手。
*   **API 阻碍：** 安卓主要依赖 OpenCL 和 Vulkan。虽然 Vulkan 引入了 `VK_KHR_cooperative_matrix` 扩展来支持硬件级Tile MMA，但在不同厂商的驱动实现上极度碎片化。
*   **黑盒驱动：** NV的PTX指令集对编译器是透明的，`Triton` 可以精准控制寄存器分配和指令发射。而移动GPU的指令集是高度保密的，编译器只能生成 SPIR-V，最终交由手机厂商的闭源驱动进行JIT编译。这种“黑盒”导致Tile-wise kernel在安卓上经常出现不可预测的性能暴跌。
*   **现状：** 目前安卓端的大模型部署（如MLC-LLM, MNN）仍然高度依赖自动调优（TVM AutoTVM/MetaSchedule）或者手写特定的 Shader 代码，无法像 `Triton` 那样优雅地泛化。

</details>

---

### 二、 相比 A100/H100，应对 MoE / Grouped GEMM / MegaKernel 的底层不足

当面对端侧 MoE（混合专家模型）或高度融合的 MegaKernel 时，Mobile GPU 与数据中心 GPU 的差距主要体现在以下四个核心维度：

<details>
<summary><b>1. 动态 Dispatch 与并发执行能力 (Grouped GEMM)</b></summary>

*   **A100/H100：** 拥有极强的并发调度能力（Hardware Work Queues, Multi-Stream）。对于 MoE 这种动态路由的网络，H100 可以通过 `Grouped GEMM` 在一个 Kernel 启动中，让不同的 SM（流多处理器）并行计算不同大小的专家矩阵，甚至支持 Device-side enqueue（GPU自己发射Kernel）。
*   **Mobile GPU：** 移动GPU的前端调度器（Command Processor）非常孱弱。它们习惯于处理单一的、规则的大型Draw Call或Compute Dispatch。如果通过多次Dispatch来处理MoE的不同专家，会产生巨大的Host-to-Device提交开销；如果强行在一个Kernel里用动态分支来做，由于移动GPU的 Warp Divergence（线程束分歧）惩罚极高，会导致大批ALU闲置。

</details>

<details>
<summary><b>2. Shared Memory 与寄存器容量瓶颈 (制约 MegaKernel)</b></summary>

*   **MegaKernel 的本质：** 是将多个操作（如 `Matmul + Softmax + Matmul + RoPE`）融合在一个 Kernel 中，数据全部驻留在 Shared Memory (SRAM) 和寄存器中，绝不写回Global Memory（显存）。
*   **A100/H100：** Hopper 架构拥有高达 $228KB$ 的 Shared Memory / SM，以及极其庞大的 Register File，足以容纳极其复杂的 MegaKernel 状态。
*   **Mobile GPU：** 为了省功耗和面积，移动GPU的 `Workgroup` 内存（SRAM）上限通常死死卡在 $32KB$ 或 $64KB$。这意味着在移动端实现 FlashAttention 等 MegaKernel 时，Block Size $M$ 和 $N$ 必须切得很小（例如 $16 \times 64$）。Tile太小会导致外层循环次数激增，无法有效均摊内存读取延迟。一旦算子变复杂，寄存器不够用，就会退化到使用 LPDDR（主存），性能瞬间雪崩。

</details>

<details>
<summary><b>3. 应对稀疏性与不规则访存 (TMA 缺失)</b></summary>

*   **A100/H100：** H100 引入了硬件级 TMA (Tensor Memory Accelerator)。在 MoE 这种需要频繁做 Gather/Scatter（根据 Token 路由随机拉取特定专家权重）的场景下，TMA 可以异步处理这种复杂的、带 Stride（步长）的多维内存搬运。同时支持底层的 $2:4$ 稀疏加速。
*   **Mobile GPU：** 移动端完全没有类似 TMA 的专用异步搬运单元。移动GPU的数据预取（Prefetcher）是为连续渲染像素（线性访存）设计的。遇到 MoE 的随机权重拉取，Cache Miss Rate 会急剧飙升，极度受限于移动端 LPDDR 的低带宽（通常只有几十 $GB/s$，而 H100 是 $3+ TB/s$）。

</details>

<details>
<summary><b>4. 同步开销与 Async Compute 掩盖延迟</b></summary>

*   **A100/H100：** 支持基于硬件的 Barrier 和异步指令（`cp.async`）。计算和访存可以完美流水线化（Software Pipelining / Double Buffering）。
*   **Mobile GPU：** 虽然支持基本的 `barrier()` 原语，但由于线程调度机制偏向静态分配，移动端的线程块同步开销远大于数据中心GPU。在移动端手写 Double Buffering 的 Kernel，往往无法达到预期的延迟隐藏效果，反而可能因为占用额外共享内存而降低了 Occupancy（占有率）。

</details>

### 总结

当前移动GPU的架构本质上是一台**“为了高效光栅化和线性计算而高度妥协的机器”**。它在硬件抽象层面上刻意隐藏了内存层级和调度细节，因此对 `Triton` 这种追求硬件极致压榨的编译器极度不友好。面对 MoE 和 MegaKernel，移动端缺失的不是单一指令，而是**庞大的片上SRAM、强大的乱序前端调度器、以及专用的异步内存搬运硬件**。这也是为什么端侧大模型的真正出路，可能不在于强行用通用GPU去跑复杂的 Kernel，而在于定制化NPU的设计演进。