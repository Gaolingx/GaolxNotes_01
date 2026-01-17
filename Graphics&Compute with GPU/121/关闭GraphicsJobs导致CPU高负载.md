# 关闭GraphicsJobs导致CPU高负载

### 🧑‍💻 User

unity urp项目 unity2022.3，il2cpp scripting backend，dx12 graphics api backend下，关闭graphics jobs，开启dev build为什么会造成非常高的cpu overhead导致高drawcall下帧率严重下降？

---

### 🤖 Assistant

在 Unity 2022.3 (URP) 中，使用 **DX12 + IL2CPP**，同时 **关闭 Graphics Jobs** 并 **开启 Development Build**，在由高 DrawCall 造成的 CPU 瓶颈下，确实会是一个“性能杀手”组合。

这并非单一原因造成的，而是多个因素叠加产生的**乘数效应**。以下是技术层面的深度分析：

### 核心原因总结

简单来说：**DX12 是一个需要多线程才能发挥优势的 API，你通过关闭 Graphics Jobs 强行将其变成了单线程运行，而 Development Build 又在这个单线程的渲染循环中插入了大量的 Debug 检测代码和 Profiler 钩子，导致处理每个 DrawCall 的 CPU 开销成倍增加。**

以下是详细的三个关键因素分析：

---

### 1. DX12 的特性 vs. 关闭 Graphics Jobs

**DirectX 12 (DX12)** 与 DX11 的最大区别在于它将驱动层（Driver）的大量管理工作（如资源屏障、状态管理、命令列表构建）移交给了应用层（即 Unity 引擎）。

*   **DX12 的优势：** 它的设计初衷是**并行化**。它允许在多个线程上同时构建 Command List（命令列表），最后合并提交。
*   **关闭 Graphics Jobs 的后果：**
    *   当 `Graphics Jobs` 为 `off` 时，Unity 很大程度上回退到单线程渲染模式（或者非常有限的主线程+渲染线程模式）。
    *   这意味着，Unity 必须在**单个线程**上完成所有的剔除（Culling）、排序、SRP Batcher 的批处理计算，以及最繁重的——**DX12 命令列表的录制**。
    *   **DX12 的单线程开销通常高于 DX11**。DX11 的驱动经过了十多年的优化，非常擅长处理单线程提交，并且驱动层会替你做很多“魔法”优化。而在 DX12 下，Unity 必须显式地处理这些繁琐的底层指令。如果只能用一个核来做这些事，CPU 会被压垮。

### 2. Development Build 的开销 (Overhead)

开启 `Development Build` 会对 CPU 性能产生巨大的负面影响，特别是在高频调用的循环中（即高 DrawCall 场景）：

*   **Profiler 钩子 (Hooks)：** 为了能让你在 Profiler 中看到性能数据，Unity 在几乎每个关键函数（包括渲染底层的 C++ 代码）前后都插入了计时和记录代码。
    *   在高 DrawCall 下，假设原本一次 Draw 耗时 $t$，加上 Profiler 钩子后变成了 $t + \Delta$。当你有 5000 个 DrawCall 时，总增加时间是 $5000 \times \Delta$。这个 $\Delta$ 在 Dev Build 中是不可忽视的。
*   **缺少编译器优化：** 即使是 IL2CPP，Development Build 生成的 C++ 代码通常不会开启最高级别的优化（如 `-O3` 或 LTO），为了保留调试符号和防止指令重排。这意味着紧凑的渲染循环（Render Loop）无法被 CPU 有效地流水线化或向量化（SIMD）。
*   **安全检查与断言 (Asserts)：** Dev Build 会启用大量的内部断言（Asserts）来检查内存越界、空指针或无效状态。
    *   特别是在 DX12 后端，Unity 需要频繁检查资源状态（Resource States）是否正确，Barrier 是否遗漏。这些检查在 Release 版本中会被剥离，但在 Dev 版本中会消耗大量 CPU 周期。

### 3. SRP Batcher 与 IL2CPP 的交互

URP 极其依赖 **SRP Batcher** 来降低 SetPass Call 的开销。

*   **SRP Batcher 的工作原理：** 它通过专门的紧凑循环（Tight Loop）将对象的材质属性绑定到 GPU 常量缓冲区（CBUFFER）。
*   **瓶颈所在：** 这个“紧凑循环”是运行在 CPU 上的。
    *   在 **Release** 模式下，这是高度优化的机器码。
    *   在 **Development** 模式下，由于上述的 Debug 开销和优化缺失，这个循环的处理速度会显著变慢。
    *   当 Graphics Jobs 关闭时，这个变慢的循环还会阻塞主线程的其他逻辑。

### 总结：为什么是“严重下降”？

这是一个连锁反应：

1.  **DX12 需要多线程**，但你**强制单线程**（Graphics Jobs Off），导致基础开销变大。
2.  **高 DrawCall** 意味着需要处理大量的命令。
3.  **Development Build** 让处理每条命令的 CPU 成本增加了数倍（Profiler + Checks）。
4.  这三者相乘：**(基础高开销) × (数量大) × (单次处理变慢) = 帧率崩塌。**

---

### 建议与解决方案

1.  **开启 Graphics Jobs (必须)：**
    *   在使用 DX12 时，**强烈建议开启 Graphics Jobs**。这是 DX12 能够跑赢 DX11 的前提。让 Unity 将渲染命令的录制分发到多个 Worker 线程上。

2.  **不要使用 Development Build 测性能：**
    *   Development Build 只能用于功能测试或查找逻辑 Bug。**永远不要依据 Dev Build 的帧率来评估最终游戏的性能。**
    *   如果需要 Profile 真机性能，请使用 **Release Build** 并开启 `Autoconnect Profiler` (虽然这也会有开销，但比完整 Dev Build 小)，或者使用平台原生的工具（如 PIX on Windows, Xcode Instruments, Snapdragon Profiler 等）分析 Release 包。

3.  **检查 DX12 Debug Layer (如果是在 PC 上)：**
    *   如果在 Editor 或 PC Build 中开启了 Development Build，有时会触发 DirectX 的 Debug Layer 验证。这会带来毁灭性的性能打击。确保在非调试图形 API 问题时关闭它。

4.  **回退 API (如果无法解决)：**
    *   如果你的项目并不使用光线追踪（Raytracing）或其他 DX12 独占特性，且目标硬件对 DX12 支持一般，**回退到 DX11** 通常能在单线程或低端 CPU 上获得更稳定的性能表现。

---

### 🧑‍💻 User

为什么很多图形调试工具都要求dx12/vk（nsight graphics gpu profiler、shader profiler/pix）api下才能work？

---

### 🤖 Assistant

这是一个非常深刻且触及图形编程本质的问题。

简单来说，**DX11/OpenGL 是“保姆式”的高级抽象，而 DX12/Vulkan 是“裸金属”式的直接映射。**

像 Nsight Graphics GPU Trace、PIX Timing Captures 或 Shader Profiler 这种深度调试工具，它们的目标是**把每一行代码和 GPU 上的每一个时钟周期对应起来**。DX12/Vulkan 的架构天然契合这种需求，而老一代 API 的机制则成为了调试工具的噩梦。

以下是具体的技术原因：

### 1. 消除“驱动黑盒” (Driver Black Box)

*   **DX11/OpenGL (黑盒):** 当你调用一个 DrawCall 时，驱动程序（Driver）在后台做了大量不可见的工作：资源重命名、内存搬运、状态检查、指令批处理等。驱动甚至可能会推迟执行你的命令，或者为了优化而改变执行顺序。
    *   **工具的困境：** 工具很难告诉你是你的 Shader 写得慢，还是驱动程序正在忙着做资源打补丁（Patching）。由于驱动是个黑盒，工具无法准确测量“API 调用时刻”与“GPU 执行时刻”之间的对应关系。
*   **DX12/Vulkan (白盒):** 显式 API 要求开发者手动管理内存、同步（Barriers）和命令列表（Command Lists）。
    *   **工具的优势：** **所见即所得。** 当你在 PIX 里看到一个 Barrier 时，那就是真实的硬件同步点。工具可以直接读取 Command Buffer，非常精确地知道 GPU 将要执行什么，而不需要猜测驱动会在中间插入什么魔法代码。

### 2. 精确的流水线状态对象 (PSO)

*   **Shader Profiler 的痛点：** 想要做 Shader 的指令级分析（比如 Nsight Shader Profiler，看每一行 HLSL 代码消耗多少 GPU 周期），工具需要知道 Shader 是如何被编译成机器码（ISA, Instruction Set Architecture）的。
*   **DX12/Vk:** 引入了 **PSO (Pipeline State Object)** 概念。PSO 是一切渲染状态的集合体，它在创建时就必须编译完成。这让调试工具极易追踪 Shader 的二进制代码，并将其映射回源代码。
*   **DX11:** 渲染状态是松散的。你可以在运行时随意切换 Blend State 或 Shader。驱动必须在运行时动态重新编译或修补 Shader（Driver Shader Cache Miss）。这对调试工具来说是灾难性的，因为它很难捕捉到那个“动态生成的、稍纵即逝的”最终 Shader 代码。

### 3. 异步计算与多队列 (Async Compute & Multi-Queue)

*   现代 GPU 能够同时运行图形任务（Graphics）和计算任务（Compute）。
*   **DX12/Vk:** 原生通过 Queue（队列）暴露了这种并行能力。工具（如 Nsight GPU Trace）可以完美地绘制出不同 Queue 上的任务是如何在时间轴上并行的，帮助你分析并行效率。
*   **DX11:** 虽然有 Context 概念，但本质上还是隐式序列化的。工具很难准确展示硬件层面的并行执行情况，因为它经过了驱动层的厚重封装。

### 4. 显式的同步与资源屏障 (Explicit Barriers)

*   **性能分析的关键：** 很多 GPU 性能问题（如 Pipeline Stall）是由于资源未准备好导致的等待。
*   **DX12/Vk:** 开发者必须显式写入 `ResourceBarrier`。调试工具可以清晰地可视化这些 Barrier：
    *   “这个 DrawCall 为什么卡住了？” -> 工具：“看，这里有个 Barrier 正在等待 Texture A 从 Copy 队列传输完成。”
*   **DX11:** 驱动自动处理同步。如果发生卡顿，工具只能告诉你 GPU 闲置了，但很难告诉你具体是哪个隐式的同步操作导致了等待。

### 5. 对新硬件特性的支持

很多高级分析功能依赖于只有 DX12/Vulkan 才支持的硬件特性：
*   **Ray Tracing (DXR / Vulkan RT):** DX11 根本不支持。
*   **Mesh Shaders / Variable Rate Shading:** 只有新 API 支持。
*   **Bindless Rendering (无绑定渲染):** 这种极高效率的资源访问方式，需要新 API 的内存模型支持。如果要分析 Bindless 资源的访问热度，必须在 DX12/Vk 环境下。

### 6. 硬件计数器 (Hardware Counters) 的访问

*   GPU 厂商（NVIDIA/AMD）通常通过扩展接口暴露底层的性能计数器（Cache 命中率、显存带宽占用、SM 占用率等）。
*   在 DX12/Vulkan 中，由于架构更接近硬件底层，工具能够以极低的开销（Low Overhead）采样这些计数器。
*   在 DX11 中，插入采样指令可能会打断驱动的批处理优化，导致测量结果不准确（即“观察者效应”：为了测量性能而严重影响了性能）。

### 总结

图形调试工具要求 DX12/Vulkan，本质上是因为：**为了给你提供手术刀般精准的性能数据，工具必须绕过“中间商”（驱动程序和 API 抽象层），直接对话硬件。** DX12 和 Vulkan 的设计初衷就是为了消除这个中间商，因此它们是高级 Profiling 工具的天然土壤。

---

### 🧑‍💻 User

为什么关闭dev build之后，pix/renderdoc无法正确显示pass的名称、准确的推断pass依赖关系了？为什么保留debug信息会影响性能？

---

### 🤖 Assistant

这涉及到了**“元数据（Metadata）”**与**“运行时性能（Runtime Performance）”**之间的权衡。

简单来说：**PIX 和 RenderDoc 显示的名称不是凭空产生的，而是 Unity 主动“告诉”它们的。而在非 Dev Build（Release）中，为了省电和提高帧率，Unity 这里“闭嘴”了。**

以下是深度解析：

---

### 一、 为什么名称和依赖关系“消失”了？

#### 1. 调试标记（Debug Markers）被剥离
在图形 API（DX12, Vulkan）中，GPU 只认识 `DrawInstanced`、`Dispatch`、`ResourceBarrier` 这些指令，它根本不知道“ShadowMap Pass”是个什么东西。

为了让调试工具能显示漂亮的层级结构（例如 `Camera Loop -> Shadow Pass -> DrawCall`），Unity 在底层使用了**调试标记 API**：
*   **DX12:** `PIXBeginEvent`, `PIXEndEvent`, `PIXSetMarker`
*   **Vulkan:** `vkCmdDebugMarkerBeginEXT` 等

**在 Unity 的源码逻辑中，这些调用通常被包裹在宏定义里：**

```csharp
// 伪代码逻辑
void ExecuteRenderPass(string passName) {
#if DEVELOPMENT_BUILD || UNITY_EDITOR
    cmd.BeginSample(passName); // 这最终会调用 PIXBeginEvent
#endif

    // ... 实际的渲染命令 ...

#if DEVELOPMENT_BUILD || UNITY_EDITOR
    cmd.EndSample(passName);
#endif
}
```

当你**关闭 Development Build** 时，上述宏条件不成立，`BeginSample` 变为空操作（No-op）。编译出的 C++ 代码中根本不包含 `PIXBeginEvent` 的指令。
**结果：** PIX/RenderDoc 收到的是一条扁平的、没有任何注释的命令流。它们只能用默认名称（如 `EID: 154 Draw`）来代替。

#### 2. 为什么依赖关系推断变差了？
RenderDoc 和 PIX 推断依赖关系（Dependency Graph）主要靠两种方式：
1.  **显式的资源屏障（Resource Barriers）：** 这是硬性的依赖，Release 版也有。
2.  **逻辑分组（即上面的 Pass 名称）：** 这是软性的逻辑。

在 Release 模式下，虽然硬件层面的依赖还在（否则画面会出错），但**可视化的逻辑依赖**丢失了。
*   **有名称时：** 工具能告诉你“PostProcess 依赖于 MainCamera 的 Color Texture”。
*   **无名称时：** 工具只能告诉你“DrawCall #500 依赖于 DrawCall #450 输出的资源 ID 0x123456”。
这对人类来说几乎不可读，看起来就像依赖关系乱了一样。

---

### 二、 为什么保留 Debug 信息会影响性能？

保留这些信息并非“免费”的，它在 CPU 和 GPU 两端都有代价。

#### 1. CPU 端开销（主要原因）

*   **字符串操作与内存分配：**
    每一次 `cmd.BeginSample("MyPassName")`，都涉及到字符串的传递、编码转换（C# String -> C++ String -> Driver API）。在 URP 这种大量使用 CommandBuffer 的管线中，每一帧可能有成百上千个 Sample 标记。在 Release 包中省去这些操作，能显著降低主线程开销。
*   **API 调用成本：**
    即便字符串很短，调用 `PIXBeginEvent` 本身也是一次从应用层穿透到驱动层的函数调用。当 DrawCall 数量巨大（如你提到的高 DrawCall 场景）时，积少成多，这些额外的 API 调用会显著增加 CPU 耗时。

#### 2. GPU 端开销（Shader 调试信息）

除了 Pass 名称，"Debug Information" 通常还指 **Shader 内部的调试信息**（Debug Symbols）。如果 Shader 编译时带有调试信息（为了能在 Shader Profiler 中看到源码）：

*   **禁止优化（Disable Optimizations）：**
    编译器（DXC/FXC）为了保证源码行号和指令对应，会**关闭**很多激进的优化，例如：
    *   **指令重排（Instruction Reordering）：** 正常情况下编译器会调整指令顺序以隐藏延迟，Debug 模式下禁止。
    *   **死代码消除（Dead Code Elimination）：** 没用的计算不会被删掉。
    *   **寄存器复用：** 变量会长期占据寄存器以便调试器读取，导致**寄存器压力（Register Pressure）大增**。
*   **占用率（Occupancy）下降：**
    由于寄存器压力变大，GPU 的每个计算单元（SM/CU）能同时运行的线程束（Wave/Warp）数量就会减少。这直接导致 GPU 吞吐量大幅下降。

#### 3. 驱动层开销（Validation）

*   当开启调试标记或调试层时，显卡驱动程序会认为你处于“开发模式”。
*   它可能会开启额外的内部检查（Validation），验证每一个 Barrier 是否正确，每一个 Descriptor 是否越界。
*   在 Release 模式下，驱动假设你是对的，跳过这些检查，“闭眼狂奔”。

### 总结

*   **RenderDoc/PIX 无法显示名称**，是因为 Unity 在 Release 模式下通过**宏**剔除了所有 `PIXBeginEvent` 等标记代码，根本没有发送给 GPU。
*   **性能影响**来自于：CPU 端大量的字符串和 API 调用开销 + GPU Shader 失去了编译器优化（导致寄存器压力大、并行度低）。

**最佳实践：**
如果你需要调试 Release 包的渲染错误，可以在 Unity 中**开启 "Development Build" 但关闭 "Script Debugging"**，或者在代码中手动通过 `UnityEngine.Profiling.Profiler.EmitFrameMetaData` (极其高级的用法) 来注入特定的标记，但这通常很麻烦。最直接的方法还是打一个开启了 `Development Build` 的包专门用于 PIX/RenderDoc 截帧分析，虽然帧率低，但逻辑结构是清晰的。