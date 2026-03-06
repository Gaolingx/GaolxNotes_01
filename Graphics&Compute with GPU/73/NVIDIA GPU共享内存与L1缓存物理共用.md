# NVIDIA GPU共享内存与L1缓存物理共用

### 🧑‍💻 User

nvidia gpu的share memory和l1 cache是共享同一块内存区域吗

---

### 🤖 Assistant

是的，在绝大多数现代 NVIDIA GPU 架构中，**Shared Memory（共享内存）和 L1 Cache（一级缓存）在物理上是共享同一块片上 SRAM（静态随机存取存储器）区域的**。

但是，这并非一成不变，其具体情况取决于 GPU 的**微架构代数（Architecture Generation）**。

为了更清晰地说明，我们将核心结论和不同架构的演进历史分开。你可以展开下方的折叠面板查看具体的架构差异。

### 核心区别：共享物理硬件，但管理方式不同

虽然它们使用同一块物理内存（SRAM），但它们的行为和管理方式完全不同：

*   **Shared Memory（共享内存）：软件管理**。它由程序员在 CUDA 代码中显式分配和管理（使用 `__shared__` 关键字）。它具有极低的延迟，用于同一个 Thread Block 内的线程之间进行高速数据共享和同步。
*   **L1 Cache（一级缓存）：硬件管理**。它由 GPU 硬件自动控制，用于缓存对 Global Memory（全局内存）和 Local Memory（局部内存）的访问，以减少内存延迟。程序员无法直接决定哪些具体数据留在 L1 Cache 中。

---

<details>
<summary><b>点击展开：NVIDIA GPU 各代架构中 Shared Memory 与 L1 Cache 的物理分配史</b> (次要信息)</summary>

#### 1. Fermi 和 Kepler 架构 (统一且可配置)
NVIDIA 首次引入了将 Shared Memory 和 L1 Cache 统一在一块物理 SRAM 上的设计。
*   **总容量**：每个 SM（Streaming Multiprocessor）有 $64 \text{ KB}$ 的片上内存。
*   **配置方式**：开发者可以通过 CUDA API 配置两者的比例，例如：
    *   $48 \text{ KB}$ Shared Memory / $16 \text{ KB}$ L1 Cache
    *   $16 \text{ KB}$ Shared Memory / $48 \text{ KB}$ L1 Cache
    *   $32 \text{ KB}$ Shared Memory / $32 \text{ KB}$ L1 Cache (仅限 Kepler)

#### 2. Maxwell 和 Pascal 架构 (物理分离 - 特例)
在这两代架构中，NVIDIA **改变了策略，将它们在物理上分开了**。
*   **Shared Memory**：拥有自己专用的物理 SRAM（例如 Maxwell 每个 SM $96 \text{ KB}$）。
*   **L1 Cache**：与 Texture Cache（纹理缓存）合并，不再与 Shared Memory 共享空间。

#### 3. Volta 和 Turing 架构 (重新统一)
从 Volta 架构开始，NVIDIA 重新回归了**统一数据缓存（Unified Data Cache）**的设计，并将这种设计延续至今。
*   **Volta**：每个 SM 拥有 $128 \text{ KB}$ 的统一内存块，可由 L1 Cache、Shared Memory 和 Texture Memory 共享。Shared Memory 最多可配置到 $96 \text{ KB}$。
*   **Turing**：每个 SM 拥有 $96 \text{ KB}$ 的统一内存，支持灵活的动态划分（例如 $64 \text{ KB} / 32 \text{ KB}$ 或 $32 \text{ KB} / 64 \text{ KB}$）。

#### 4. Ampere、Ada Lovelace 和 Hopper 架构 (大容量统一内存)
延续了 Volta 的设计思路，统一内存块的容量变得更大，且硬件能够更智能地进行动态划分。
*   **Ampere (例如 A100)**：每个 SM 拥有 $192 \text{ KB}$ 的统一 SRAM。用户可以将 Shared Memory 配置为最大 $164 \text{ KB}$。
*   **Hopper (例如 H100)**：每个 SM 拥有 $228 \text{ KB}$ 的统一 SRAM（Thread Block Cluster 功能进一步增强了共享内存的访问能力）。

</details>

---

### 为什么要共享同一块硬件区域？

1.  **资源利用率最大化**：有些算法极度依赖 Shared Memory（例如矩阵乘法、规约算法），而有些算法则主要依赖硬件的 L1 Cache（例如具有不规则内存访问模式的算法）。共享物理区域可以让 SM 动态适应不同特征的 Workload（工作负载），避免硬件闲置。
2.  **降低成本和功耗**：相比于维护两组独立的、庞大的 SRAM 阵列，合并设计能节省芯片面积并降低功耗。

### 开发者如何控制分配比例？

在支持统一内存架构的 GPU 上，CUDA 提供了一些 API 让开发者向驱动程序提供分配偏好（Preference）：

1.  **设置首选缓存配置 (Device 级别或 Function 级别)**：
    使用 `cudaFuncSetCacheConfig` 可以设置某个内核函数的偏好：
    ```cpp
    // 偏好更大的 Shared Memory
    cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared); 
  
    // 偏好更大的 L1 Cache
    cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferL1);
  
    // 均等分配
    cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferEqual);
    ```

2.  **动态申请超大 Shared Memory (Ampere 及更新架构)**：
    在较新的架构中，如果想使用超过默认上限的 Shared Memory（例如 A100 上的 $164 \text{ KB}$），必须通过 API 显式声明：
    ```cpp
    cudaFuncSetAttribute(
        MyKernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        163840 // 设置为 160 KB
    );
    ```

### 总结
在绝大多数情况下（除了 Maxwell 和 Pascal），你可以认为**它们是同一块物理 SRAM 的不同逻辑划分**。一部分交给你（开发者）精确调度，另一部分交给 GPU 硬件自动缓存数据。

---

### 🧑‍💻 User

在分别运行图形和计算任务的时候，这个区域会怎么划分？

---

### 🤖 Assistant

在分别运行**图形（Graphics）**和**计算（Compute）**任务时，这块统一的物理内存区域（Unified SRAM）的划分逻辑会有显著的差异。

核心原则是：**现代 GPU 的硬件调度器会根据当前真正在 SM（流多处理器）上运行的“着色器（Shader）”或“内核（Kernel）”的实际需求，在底层动态地重新配置这块内存。**

下面为您详细拆解这两种任务下的划分情况：

### 1. 运行图形任务（Graphics Workloads）

在图形渲染任务（如运行 DirectX、Vulkan、OpenGL 时），统一内存区域的划分取决于当前执行的具体是哪种渲染管线阶段。现代架构中，**L1 Cache 和纹理缓存（Texture Cache）也是合并的**。

*   **传统图形管线（顶点着色器、像素着色器等）**：
    *   **划分逻辑**：**极度偏向 L1 / 纹理缓存。**
    *   **原因**：传统的顶点（Vertex）和像素/片段（Pixel/Fragment）着色器在编程模型中，**没有**暴露类似 CUDA 中共享内存的概念供开发者显式控制跨线程通信。图形渲染最核心的访存瓶颈在于**纹理采样（Texture Sampling）**和常量缓冲区读取。
    *   **结果**：此时，GPU 硬件会将绝大部分甚至全部的统一 SRAM 划分为 L1/Texture Cache，以最大化纹理命中率，提升渲染帧率。Shared Memory 区域实质上被压缩到极小（仅保留少量供驱动程序内部数据结构使用）。
*   **计算着色器（Compute Shader）阶段**：
    *   **划分逻辑**：**按需分配（类似 CUDA 计算任务）。**
    *   **原因**：现代图形 API 引入了 Compute Shader 来处理光照剔除、粒子物理等任务。在 Compute Shader 中，开发者可以使用 `groupshared` (HLSL) 或 `shared` (GLSL) 来显式申请共享内存。
    *   **结果**：在这个阶段，硬件会像运行 CUDA 程序一样，根据 Shader 编译出的共享内存需求量进行划分，剩余的容量作为 L1 Cache。

### 2. 运行计算任务（Compute Workloads - CUDA/OpenCL）

在运行纯计算任务时，划分逻辑完全由**开发者编写的内核（Kernel）需求**以及**API的显式偏好**来驱动。

*   **刚性需求优先**：
    编译器会计算出每个 Thread Block（线程块）所需的 Shared Memory 总量（包括静态分配和动态分配的部分）。当这个 Kernel 被调度到 SM 上运行时，硬件**必须首先**保证切出足够的 Shared Memory 满足运行需求，否则该 Thread Block 无法启动。
*   **剩余空间全归 L1**：
    在满足了 Shared Memory 的硬性需求后，这块 SRAM 剩下的所有容量，都会自动变成 L1 Cache（用于缓存 Global Memory 读写）。
*   **软件偏好微调**：
    正如上文提到的，如果一个 Kernel 既没有用满 Shared Memory，也没有明显的 L1 Cache 瓶颈，开发者可以通过 `cudaFuncSetCacheConfig` 向驱动建议偏好，驱动会尽可能按照建议去调整硬件寄存器来划分这块区域。

---

<details>
<summary><b>点击展开：底层硬件是如何实现动态切换的？(代码示例与机制)</b></summary>

**1. 切换的时机**
这种内存划分的切换不是在“运行游戏”或“运行 AI 模型”这种宏观级别进行的，而是在**极细粒度的任务调度级别**进行的。一个 SM 可能在微秒级的时间内：
1. 运行一个像素着色器（全部配置为 L1/Texture Cache）。
2. 紧接着运行一个后处理的计算着色器（重新配置为 $64 \text{ KB}$ Shared Memory + $64 \text{ KB}$ L1 Cache）。

**2. 图形着色器中的共享内存代码体现**
在图形 API 开发中，触发 Shared Memory 分配的代码如下：

*   **HLSL (DirectX)**:
    ```hlsl
    // 声明一个共享内存数组，这将迫使 SM 划分出 Shared Memory
    groupshared float sharedData[256]; 

    [numthreads(256, 1, 1)]
    void CSMain(uint3 DTid : SV_DispatchThreadID, uint GI : SV_GroupIndex) {
        sharedData[GI] = LoadData(DTid);
        GroupMemoryBarrierWithGroupSync(); // 线程同步
        // ... 计算 ...
    }
    ```
*   **GLSL (Vulkan/OpenGL)**:
    ```glsl
    // 使用 shared 关键字
    shared float sharedData[256];
    ```

**3. 硬件寄存器控制**
在编译阶段，CUDA (NVCC) 或图形着色器编译器（DXC/Glslang）会将内核所需的共享内存大小写入编译后的二进制文件（PTX 或 SASS 汇编指令头）中。
当 GPU 命令处理器（Command Processor）将任务派发给具体的 SM 时，它会读取这些头部信息，并通过设置 SM 内部的硬件配置寄存器，在指令真正开始执行前，瞬间完成 SRAM 物理寻址边界的重新划定。
</details>

### 总结

*   **玩游戏（纯渲染）时**：这块区域几乎 $100\%$ 化身为 **L1 / 纹理缓存**，为你提供丝滑的高清纹理加载。
*   **跑 AI / CUDA 运算时**：这块区域变成了一把**可调节的瑞士军刀**，根据你的代码需求，在用于高速线程通信的 Shared Memory 和用于降低全局内存延迟的 L1 Cache 之间动态平衡。
*   **现代游戏（混合管线）时**：在同一个画面帧的生成过程中，这块 SRAM 的划分比例会在图形渲染（L1 为主）和计算着色（混合比例）之间极其频繁地来回切换。