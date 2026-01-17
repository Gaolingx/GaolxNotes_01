# AsyncCompute 解析

### 🧑‍💻 User

在近2年GDC育碧各工作室分享的ppt渲染管线中都出现了AsyncCompute，基本上就是在Render XXXX渲染各个步骤之外又有一条并行的称为AysncCompute的线（Compute Queue），里边的操作都是Compute Shader实现的。
这个神秘的AsyncCompute线究竟是什么？

---

### 🤖 Assistant

你在GDC（Game Developers Conference）育碧（Ubisoft）或其他3A大厂的技术分享中看到的 **Async Compute（异步计算）**，并不是什么黑科技，而是现代GPU架构中为了**压榨硬件性能极限**而存在的一种并行机制。

简单来说，它就像是在原本拥挤的单行道旁边开辟了一条“潮汐车道”，让不同类型的任务可以同时在GPU上奔跑。

下面为你详细拆解这条“神秘的并行线”究竟是什么，以及育碧在里面都塞了些什么工作。

---

### 1. 核心概念：填补“气泡”（Bubbles）

要理解 Async Compute，首先要理解GPU的工作方式。

#### 传统方式（串行）
在没有 Async Compute 之前，GPU 的 Graphics Queue（图形队列）是串行工作的。
比如，GPU 在渲染 G-Buffer（几何阶段）时，可能大量使用了 **固定功能单元**（如光栅化器、ROP），而此时 GPU 核心的大量 **ALU（算术逻辑单元，也就是跑Shader核心计算能力）** 可能是闲置的。或者在等待显存数据（Memory Stall）时，整个GPU都在“发呆”。

这些闲置的时间段，被称为 **“Bubbles”（气泡）**。

#### Async Compute 方式（并行）
Async Compute 允许开发者开启一个独立的 **Compute Queue（计算队列）**。这个队列里的任务（全是 Compute Shader）可以和 Graphics Queue 里的任务 **并行执行**。

$$ T_{Frame} = \max(T_{Graphics}, T_{Compute}) $$
而不是
$$ T_{Frame} = T_{Graphics} + T_{Compute} $$

**它的核心目的只有一个：** 当图形队列主要在忙着搬运顶点、光栅化或者等待显存时，利用那些闲置的 ALU 来做纯数学计算任务。这就像是“一边烧水（图形任务），一边切菜（计算任务）”。

---

### 2. 那条神秘的线里到底在算什么？

育碧（以及很多3A工作室如 Naughty Dog, id Software）通常会将以下几类任务放入 Async Compute 管道中：

#### A. GPU Driven Culling (GPU 驱动的剔除)
这是育碧极度依赖的技术（如《刺客信条》、《孤岛惊魂》这种大世界游戏）。
*   **主线（Graphics）：** 准备绘制上一帧确定的可见物体。
*   **Async Compute 线：** 计算**下一帧**（或者当前帧后续Pass）有哪些物体是可见的。它会遍历场景中成千上万的物体包围盒，做视锥体剔除（Frustum Culling）和遮挡剔除（Occlusion Culling），生成一个精简的“绘制命令列表”（Indirect Draw buffer）。
*   **为什么放这里？** 这是一个纯数学计算，极度依赖 ALU，跟光栅化无关，非常适合填补 G-Buffer Pass 的空隙。

#### B. 物理与模拟 (Physics & Simulation)
*   **内容：** 头发模拟（如 Lara Croft 的头发）、布料解算（披风飘动）、粒子系统的更新、体积云的流体模拟。
*   **育碧案例：** 在《全境封锁》（Snowdrop 引擎）中，大量的粒子飞舞和积雪覆盖计算往往就在这里。

#### C. 环境光遮蔽与光照 (SSAO / GI)
*   **内容：** SSAO（屏幕空间环境光遮蔽）、HBAO+、或者基于体素的全局光照（VXGI）更新。
*   **原因：** 这些效果通常是计算密集型的（Compute Bound），如果在主图形队列跑，会卡住后面的后处理。把它们挪到 Async Pipe，可以让它们在 Shadow Map 渲染（通常是带宽密集型或固定单元密集型）的同时进行计算。

#### D. 后处理与图像优化 (Post-Process & Denoising)
*   **内容：** 景深（DOF）、模糊（Blur）、Tone Mapping，以及现在最重要的——**光线追踪降噪（Ray Tracing Denoising）**。
*   **育碧案例：** 在《阿凡达：潘多拉边境》中，光追计算后的降噪处理非常繁重，利用 Async Compute 可以大幅减少对帧率的影响。

#### E. 解压缩 (Decompression)
*   现代主机（PS5/XSX）架构允许 GPU 辅助解压纹理或几何数据，这部分工作纯属苦力活，非常适合扔到 Compute Queue 里去，不干扰主渲染流程。

---

### 3. 为什么看PPT里它是一条长长的并行线？

你在PPT里看到的图表通常是 **GPU Profiler（如 PIX, Radeon GPU Profiler, Nsight）** 的可视化结果。

一个典型的育碧游戏帧结构可能长这样：

| 时间轴 $\rightarrow$ | 阶段 1 | 阶段 2 | 阶段 3 |
| :--- | :--- | :--- | :--- |
| **Graphics Queue** | **[Shadow Map Pass]** (吃带宽/光栅化) | **[G-Buffer Pass]** (吃ROP/光栅化) | **[Lighting Pass]** (吃ALU) |
| **Async Compute** | **[SSAO 计算]** (吃ALU) | **[粒子模拟 / 剔除]** (吃ALU) | *(此时ALU满了，这里通常空着)* |

*   **阶段 1 & 2：** 图形队列在忙着画三角形、写深度，ALU 并没有跑满。于是 Async Compute 此时插入 SSAO 和粒子计算，完美利用了原本会浪费的 ALU 资源。
*   **阶段 3：** 到了光照阶段，图形队列自己就要疯狂使用 ALU 算光照了，这时候如果再跑 Async Compute 就会导致两个队列抢资源，反而变慢。所以你会发现在 Lighting 阶段，Async Compute 线通常是断开的。

### 4. 代码层面的简单视点

在 Direct3D 12 或 Vulkan 中，这通过创建不同的 Queue 来实现：

```cpp
// 伪代码：Vulkan / DX12 概念

// 1. 获取两个队列
Queue graphicsQueue = device.getQueue(GRAPHICS_FAMILY);
Queue computeQueue  = device.getQueue(COMPUTE_FAMILY);

// 2. 准备命令
CommandBuffer gfxCmd = ...; // 包含 DrawCall, Rasterization
CommandBuffer compCmd = ...; // 包含 Dispatch (Compute Shader)

// 3. 提交任务（关键在于同步）
// 我们希望 compute 任务和 graphics 任务同时开始，或者在特定的 fence 处同步
Fence frameFence;

// 提交图形任务
graphicsQueue.submit(gfxCmd);

// 提交计算任务 (硬件调度器会尝试并行执行这两个)
computeQueue.submit(compCmd);

// 在帧结束时同步
waitForFence(frameFence);
```

### 总结

那个“神秘的 Async Compute 线”实际上就是 **“GPU 的时间管理大师模式”**。

育碧的工程师们通过精细的调优，把那些**不依赖当前光栅化结果**、**纯数学计算**的任务（如剔除、物理、AO），从主渲染线程中剥离出来，塞进了 GPU 的“空闲时间缝隙”里。

这使得原本需要 $33ms$ (30FPS) 渲染的一帧，可能被压缩到 $25ms$，从而在画质不变的情况下提升帧数，或者在帧数不变的情况下塞入更逼真的物理和光照效果。

---

### 🧑‍💻 User

例如，可以在渲染shadowmap的时候用cs来做SSAO，因为sm和ssao其实对GPU资源的要求是很不同的，因此同时做其实更充分的利用了GPU，那确实，sm重点是vertex处理，ssao主要是对depth texture采样，两者要求非常互补，同时肯定能提升效率，具体细节能不能深入分析一下？这是一个很典型的Async Compute and Overlap问题

---

### 🤖 Assistant

这是一个非常精准且经典的 Async Compute 案例。

你提到的 **Shadow Map (Graphics Queue)** 与 **SSAO (Compute Queue)** 并行，是教科书级别的“资源正交性（Resource Orthogonality）”利用。因为这两个任务在 GPU 微架构层面上，争夺的硬件单元（Hardware Units）几乎是完全错开的。

下面我们从 **GPU 微架构（Micro-architecture）** 和 **资源流水线** 的深度，详细分析为什么这一对组合能如此完美地 Overlap。

---

### 1. 资源争夺分析表：为什么它们是“天作之合”？

我们把 GPU 拆解成几个关键的硬件资源池，对比 Shadow Map（通常指级联阴影 CSM）和 SSAO（基于 Compute Shader）的需求：

| 硬件资源 (Resource) | Shadow Map (Graphics Queue) | SSAO (Compute Queue) | 冲突程度 |
| :--- | :--- | :--- | :--- |
| **Geometry/Tessellation** | **极高** (只有顶点处理，无像素处理) | **无** (不处理几何体) | 🟢 零冲突 |
| **Rasterizer (光栅化器)** | **极高** (将三角形转为深度像素) | **无** (Compute 不走光栅化) | 🟢 零冲突 |
| **ROPs (光栅化输出单元)** | **极高** (写入 Depth Buffer, Z-Test) | **无** (CS 通过 UAV 写入显存) | 🟢 零冲突 |
| **ALU (算术逻辑单元)** | **极低** (仅 VS 有少量变换，PS 通常为空) | **高** (采样、随机化、遮蔽估算) | 🟢 互补 |
| **Texture Units (TMU)** | **低** (一般不需要采样纹理，除非 Alpha Test) | **极高** (疯狂采样 Depth/Normal Buffer) | 🟡 轻微竞争 |
| **VRAM Bandwidth (带宽)** | **高 (Write)** (写入 Shadow Map 深度) | **高 (Read)** (读取 G-Buffer 深度) | 🔴 **主要瓶颈** |
| **L2 Cache** | **中** (深度压缩数据) | **高** (纹理采样缓存) | 🟠 潜在冲突 |

#### 深度解析：

1.  **ALU 利用率（填补气泡的核心）：**
    *   **Shadow Map:** 在渲染阴影贴图时，GPU 主要是在忙着把顶点变换位置（Vertex Shader），然后光栅化器（Rasterizer）判断覆盖范围，最后 ROPs 把深度值写进显存。**在这个过程中，GPU 内部海量的 FP32/INT32 计算单元（CUDA Cores / Stream Processors）大部分时间是在“等活干”的。**
    *   **SSAO:** 全是数学计算和纹理采样。它不需要光栅化，也不需要 ROPs。
    *   **结论：** 当 Async Compute 调度器把 SSAO 的 Wavefronts (AMD) / Warps (Nvidia) 发送给 SM (Streaming Multiprocessor) 时，这些线程正好可以使用那些被 Shadow Map 闲置的 ALU 单元。

2.  **ROPs vs UAVs：**
    *   Shadow Map 极度依赖 **ROPs**（这是图形管线后端的固定单元）。
    *   SSAO 使用 **UAV (Unordered Access View) / ImageStore** 直接写内存，绕过了 ROPs。
    *   **结论：** 输出路径完全不同，互不阻塞。

---

### 2. 潜在的隐患：带宽与缓存（The Bandwidth Bottleneck）

虽然计算资源（ALU/ROP）完美互补，但它们共享同一个 **“显存高速公路”**。这是做 Async Compute 时最容易翻车的地方。

*   **场景：** Shadow Map 需要大量的 **Write Bandwidth**（写深度图）；SSAO 需要大量的 **Read Bandwidth**（读全屏深度/法线）。
*   **风险：** 如果你的显存带宽（Memory Bandwidth）已经打满了，强行并行会导致两者都变慢。
    *   *例如：* 在 4K 分辨率下，SSAO 的采样带宽压力极大，如果此时 Shadow Map 精度设得过高（比如 4096 x 4096 x 4 Cascades），两者同时抢占显存总线，会导致 Memory Stall。
*   **优化方案：**
    *   **SSAO 降采样：** 育碧和很多 3A 引擎通常在半分辨率（Half-Res）下计算 SSAO，既减少了 ALU 压力，大幅降低了 Read Bandwidth，使得它能完美塞进 Shadow Map 的带宽间隙里。
    *   **Shadow Map 压缩：** 确保深度压缩（Delta Color Compression / Z-Compression）正常工作，减少写带宽。

---

### 3. 执行流与依赖管理 (Pipeline & Barriers)

要在代码层面实现这个 Overlap，需要非常清晰的依赖管理。通常的帧结构是这样的：

**假设前提：**
*   SSAO 需要依赖 **Main Camera G-Buffer** 的深度和法线（这些在上一阶段已经画完了）。
*   Shadow Map 生成的是 **Light Depth**，SSAO 不需要它；SSAO 的结果 Shadow Map 也不需要。
*   **因此，它们完全独立。**

**Pipeline 示意图：**

```text
[时间轴 Time] ----------------------------------------------------->

Queue 1 (Graphics):  [ G-Buffer Pass ] | [ Shadow Map Pass (4 Cascades) ] | [ Lighting Pass ]
                                       |                                  |
                                       | <--- Async Start                 |
Queue 2 (Compute):                     | [ SSAO Compute Shader ]          |
                                       | [ GTAO / HBAO+ ]                 |
                                       | [ Ray Tracing Denoise ]          |
                                       |                                  |
Barrier/Fence:       (Signal) -------->| (Wait) ------------------------->| (Sync for Lighting)
```

**关键点：**
1.  **G-Buffer 结束时**，需要发出一个信号（Signal），告诉 Compute Queue：“深度图已经准备好了，可以开始算 SSAO 了”。
2.  **Graphics Queue 开始画 Shadow Map**。此时 Compute Queue 并行启动 SSAO。
3.  **Lighting 阶段开始前**，必须进行同步（Barrier）。因为 Lighting Pass 既需要 Shadow Map 的结果（算阴影），也需要 SSAO 的结果（算遮蔽），所以两者必须都完成才能进入 Lighting。

---

### 4. 硬件调度器的微操（Hardware Scheduler）

为什么我们在 PPT 里看到的是两条线，但实际上 GPU 只有一个核心？

*   **AMD (ACEs - Asynchronous Compute Engines):**
    AMD 的架构（GCN/RDNA）有专门的硬件单元叫 ACEs。它们负责管理 Compute Queue，并且极其擅长将计算任务“填缝”进图形任务。如果在 Shadow Map 阶段，CU（Compute Unit）里的 Vector Units 空闲，ACE 就会瞬间把 SSAO 的波前（Wavefronts）塞进去执行。
*   **NVIDIA (HyperQ / Gigathread):**
    Nvidia 从 Maxwell/Pascal 开始大幅改进了 Async Compute。虽然实现方式不同（更多依赖软件/固件调度与上下文切换），但在 Shadow Map 这种 **ROP-bound (ROP受限)** 的场景下，调度器能够非常有效地识别出 ALU 的空闲，并安排 Compute Warps 执行。

### 5. 总结

**Shadow Map + SSAO** 确实是 Async Compute 的黄金搭档，原因总结如下：

1.  **流水线互补：** 一个卡在 Vertex/Raster/ROP，一个卡在 ALU/Texture。
2.  **数据无依赖：** 一个读 Main Depth，一个写 Light Depth。
3.  **主要风险：** 显存带宽（Bandwidth）。
    *   *Solution:* 只要控制好 SSAO 的分辨率或采样数，不让带宽饱和，你就能白赚性能。

育碧的《刺客信条》系列和《全境封锁》之所以能把画面塞得那么满，很大程度上就是因为他们把这一套“时间管理”玩到了极致，把原本被 Shadow Map 浪费掉的 GPU 算力全部找回来了。

---

### 🧑‍💻 User

About Async Compute and Overlap，a classic mistake is to focus only on SM occupancy (unused warp slots) to identify potential async compute workloads. The GPU is a complex beast and other metrics such as top unit throughputs (SOL) play an equal if not more important role than SM occupancy alone.

Thus, in addition to SM occupancy, you should also consider unit throughputs, register file occupancy, group shared memory, and different datapaths. After the ideal pair is identified, the compute workload is moved to the async queue. It is synchronized with the sync/main queue using fences to ensure the correct execution order.

Question:
1. how to to identify potential async compute workloads?
2. how to synchronize with the sync/main queue using fences with dx12?

---

### 🤖 Assistant

This is a sophisticated question that touches on the core art of low-level GPU optimization. You are absolutely correct: **SM Occupancy is a necessary but insufficient metric.** High occupancy merely means you have enough active warps to hide latency; it does not mean the functional units (ALU, FP32, INT, SFU) are actually busy.

Here is the deep dive into identifying workloads and implementing synchronization in Direct3D 12.

---

### Part 1: How to Identify Potential Async Compute Workloads

To find the perfect "Async Partner," you must analyze the **Hardware Unit Throughput (SOL - Speed of Light)**. The goal is **Resource Orthogonality**: If Task A uses Resource X, Task B should use Resource Y.

You need to look at four specific bottleneck categories using tools like **NVIDIA Nsight Graphics**, **AMD Radeon GPU Profiler (RGP)**, or **Microsoft PIX**.

#### 1. The "Fixed-Function vs. Programmable" Gap
This is the most common opportunity.
*   **The Scenario:** Depth Pre-pass, Shadow Mapping, or G-Buffer generation (depending on complexity).
*   **The Bottleneck:** These passes are often **ROP-bound** (Raster Operations, writing depth/color) or **Rasterizer-bound** (Triangle setup).
*   **The Opportunity:** While the ROPs and Rasterizers are screaming at 100%, the **SMs (Streaming Multiprocessors)** are often waiting or running very lightweight Vertex Shaders.
*   **Target Async Workload:** Heavy Compute Shaders that are pure math (ALU) or compute-based culling.

#### 2. The "ALU Datapath" Split (FP32 vs. INT32)
Modern architectures (like NVIDIA Ampere/Ada or AMD RDNA) often have separate or dual-issue pipelines for Floating Point and Integer operations.
*   **The Scenario:** A graphics shader heavily reliant on standard lighting ($N \cdot L$, PBR math) which is almost entirely FP32.
*   **The Opportunity:** The INT32 pipes might be underutilized.
*   **Target Async Workload:** Integer-heavy tasks, such as Bit-packing/unpacking, address generation for tiling, or certain types of hash-based noise generation.

#### 3. Register File & Occupancy Limiters
Sometimes, a graphics shader has low occupancy not because of lack of work, but because it consumes too many registers (VGPRs).
*   **The Scenario:** A complex Uber-Shader in the graphics queue requires 128 registers per thread, limiting the GPU to 25% theoretical occupancy.
*   **The Opportunity:** The physical registers are allocated, but the **LDS (Shared Memory)** or **L1 Cache** bandwidth might be completely available.
*   **Target Async Workload:** A compute shader with *low* register pressure but high reliance on LDS (e.g., a prefix scan or sorting algorithm) can slip into the execution slots that the heavy graphics shader cannot fill.

#### 4. The "Bandwidth Check" (Crucial Warning)
This is where developers fail.
*   **The Trap:** Overlapping a bandwidth-heavy Graphics pass (e.g., G-Buffer write) with a bandwidth-heavy Compute pass (e.g., Post-Process Blur).
*   **The Result:** **Thrashing.** Both queues stall waiting for VRAM.
*   **The Metric:** Monitor `DRAM Throughput` and `L2 Cache Hit Rate`.
*   **Rule of Thumb:** Combine **Compute-Bound** (ALU heavy) tasks with **Memory-Bound** (Read/Write heavy) tasks.

---

### Part 2: Synchronization with Fences in DX12

In DX12, synchronization between queues (Graphics/Direct Queue and Compute Queue) is handled via `ID3D12Fence`.

**Key Concept:** The CPU submits the `Signal` and `Wait` commands to the queues. These commands are executed by the GPU when it reaches that point in the command buffer. **This is a GPU-side wait; the CPU does not block.**

#### The Scenario
1.  **Graphics Queue:** Renders the G-Buffer.
2.  **Compute Queue:** Wants to read the G-Buffer Depth to compute SSAO.
3.  **Graphics Queue:** Wants to read the SSAO result to do the Lighting Pass.

#### The Implementation

```cpp
// 1. Setup: Create the Fence
ID3D12Fence* pFence;
UINT64 fenceValue = 0;
device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

ID3D12CommandQueue* pGfxQueue = ...;
ID3D12CommandQueue* pComputeQueue = ...;

// ====================================================================================
// FRAME START
// ====================================================================================

// --- 1. Graphics Queue: Render G-Buffer ---
pGfxCommandList->OMSetRenderTargets(...); // Set G-Buffer
pGfxCommandList->DrawIndexedInstanced(...); // Draw Scene
pGfxCommandList->Close();
pGfxQueue->ExecuteCommandLists(1, &pGfxCommandList);

// SIGNAL: Tell the GPU "G-Buffer is done at this point"
// We increment the fence value to represent "Phase 1 Complete"
fenceValue++; 
UINT64 fenceVal_GBufferDone = fenceValue;
pGfxQueue->Signal(pFence, fenceVal_GBufferDone); 

// ====================================================================================
// ASYNC COMPUTE OVERLAP BEGINS
// ====================================================================================

// --- 2. Compute Queue: Calculate SSAO ---

// WAIT: Compute Queue sits idle* until Gfx Queue reaches the Signal above.
// *Note: "Idle" here means this specific command list won't start. 
// Ideally, you submit this EARLY so the scheduler sees it.
pComputeQueue->Wait(pFence, fenceVal_GBufferDone);

// Now record/submit the compute work
pComputeCommandList->SetPipelineState(pSSAOPSO);
pComputeCommandList->SetComputeRootShaderResourceView(..., DepthBufferGPUVirtualAddress);
pComputeCommandList->SetComputeRootUnorderedAccessView(..., SSAOOutputGPUVirtualAddress);
pComputeCommandList->Dispatch(width, height, 1);
pComputeCommandList->Close();
pComputeQueue->ExecuteCommandLists(1, &pComputeCommandList);

// SIGNAL: Tell the GPU "SSAO is done"
fenceValue++;
UINT64 fenceVal_SSAODone = fenceValue;
pComputeQueue->Signal(pFence, fenceVal_SSAODone);

// --- 3. Graphics Queue: Shadow Maps (happening in parallel) ---
// While Compute is waiting or working, Graphics Queue keeps going!
pShadowCommandList->DrawIndexedInstanced(...); // Render Shadow Maps
pShadowCommandList->Close();
pGfxQueue->ExecuteCommandLists(1, &pShadowCommandList);

// ====================================================================================
// SYNCHRONIZATION POINT (Rejoining)
// ====================================================================================

// --- 4. Graphics Queue: Lighting Pass ---

// WAIT: Graphics Queue cannot proceed to Lighting until SSAO is finished.
pGfxQueue->Wait(pFence, fenceVal_SSAODone);

// Now do lighting using the SSAO result
pLightingCommandList->SetGraphicsRootDescriptorTable(..., SSAOTextureSRV);
pLightingCommandList->DrawIndexedInstanced(...);
pLightingCommandList->Close();
pGfxQueue->ExecuteCommandLists(1, &pLightingCommandList);
```

### Critical Details for DX12

1.  **Resource States (Barriers):**
    *   Transitioning resource states (e.g., from `D3D12_RESOURCE_STATE_DEPTH_WRITE` to `D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE`) across queues is tricky.
    *   **Rule:** A resource must be in `D3D12_RESOURCE_STATE_COMMON` when handing it off between different queues.
    *   In the example above: The G-Buffer depth must be transitioned to `COMMON` on the Graphics Queue before the Signal. The Compute Queue then transitions it from `COMMON` to `NON_PIXEL_SHADER_RESOURCE` (read).

2.  **Submit Order:**
    *   You should call `pComputeQueue->Wait` and `ExecuteCommandLists` *as early as possible* in your CPU frame loop. If you wait until the end of the frame to submit the compute work, the driver/scheduler might not have enough time to "fill the bubbles" effectively.

3.  **Command Allocators:**
    *   Remember that Command Allocators are not thread-safe and cannot be reset while the GPU is reading from them. You need separate allocators for your Graphics and Compute command lists.