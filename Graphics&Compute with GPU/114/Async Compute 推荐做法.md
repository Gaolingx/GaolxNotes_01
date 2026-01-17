# Async Compute 推荐做法

### 🧑‍💻 User

About Async Compute and Overlap，为什么以下做法是推荐的？

- Use GPU Trace provided by NVIDIA Nsight Graphics to identify potential overlap pairs:
Look for a combination of low top unit throughput metrics.
  - If the SM occupancy shows a significant portion of unused warp slots, then it’s potentially an effective overlap. SM Idle % without conflicting high throughput units is almost always a guaranteed improvement.
  - Capture another GPU Trace to confirm the result.
- Try overlapping different datapaths. For example, FP, ALU, Memory Requests, RT Core, Tensor Core, Graphics-pipe.
  - FP, ALU, and Tensor share a different register file.
- Overlap compute workloads with other compute workloads. This scenario is very efficient on NVIDIA Ampere Architecture GPUs.
- Consider converting some of the graphics work such as post-processing passes to compute: this can present new overlap opportunities.
- Consider running async work between frames.
- Measure the performance difference over the whole frame or average over multiple frames if intraframe async compute is implemented.
- Verify the behavior across different GPU tiers. High-end GPUs have more SM units, so more potential for overlap.
- Verify the behavior at different resolutions. Low resolution usually means less pixel warps and thus more idle time for the SMs and more overlap potential.

以下不推荐：
- Don’t only focus purely on the SM warp occupancy, start by looking at unit throughputs.
- Don’t use long async compute workloads unless they can finish comfortably before the dependency on the sync queue.
- Don’t overlap workloads that use the same resource for reading and writing, as it causes data hazards.
- Don’t overlap workloads with high L1 and L2 usage and VRAM throughput metrics. Oversubscription or reduction in cache hit-rate will result in performance degradation.
- Be careful with more than two queues if hardware-accelerated GPU scheduling is disabled. Software scheduled workloads from more than two queues (copy queue aside) may result in workload serialization.
- Be careful with overlapping compute-over-compute workloads where both cause WFIs. WFIs during simultaneous compute on both queues can result in synchronization across the workloads. Frequent descriptor heap changes on the async queue may cause additional WFIs.
- Don’t use DX12 command queue priorities to influence async and sync workload priorities. The interface simply dictates the queue from which the commands are consumed first and does not affect warp priorities in any meaningful way.
- Don’t overlap RTCore workloads. Both share the same throughput units and due to interference will degrade performance.

---

### 🤖 Assistant

这些关于 **Async Compute（异步计算）** 和 **Overlap（并行重叠）** 的建议，其核心逻辑在于**最大化 GPU 硬件单元的利用率（Utilization）** 并 **隐藏延迟（Latency Hiding）**。

简单来说，GPU 就像一个巨大的工厂，里面有不同的流水线（FP32、INT32、Tensor Core、Load/Store 等）。如果主渲染任务只用到了部分流水线，或者 SM（流多处理器）有空闲，我们可以通过异步计算插入额外的任务，把这些空闲资源利用起来。

以下是针对你列出的“推荐做法”的详细原理解析：

### 1. 使用 GPU Trace 识别低吞吐量（Low Throughput）和空闲 SM
> *Look for a combination of low top unit throughput metrics... SM Idle %...*

*   **原因：** Async Compute 的本质是“见缝插针”。
    *   **Throughput（吞吐量） vs. Occupancy（占用率）：** 如果一个 Shader 的 Occupancy 很高（很多线程在跑），但 Throughput 很低（例如都在等内存），这说明 GPU 在空转等待。这时插入计算任务可以利用这些等待时间。
    *   **SM Idle：** 如果 SM 本身就是空闲的（例如几何处理阶段，SM 还没满载），这是最安全的重叠时机，几乎是“免费”的性能提升。
    *   **避免冲突：** 如果主队列已经把 ALU（算术逻辑单元）跑满了，再塞进一个重 ALU 的计算任务只会导致资源争抢，反而变慢。

### 2. 重叠不同的数据路径（Datapaths）
> *Try overlapping different datapaths (FP, ALU, Memory Requests, RT Core, Tensor Core...)*

*   **原因：** GPU 内部有独立的硬件单元并行工作。
    *   **硬件隔离：** 现代 NVIDIA GPU 可以在同一周期内同时执行 FP32（浮点运算）、INT32（整数运算）和 Load/Store（内存读写）。RT Core 和 Tensor Core 也是独立的。
    *   **互补效应：** 如果主渲染任务是 **Math-Heavy**（纯数学计算，占用 FP32），而异步任务是 **Memory-Heavy**（如拷贝、后处理，占用 Load/Store），它们互不干扰，完美共存。
    *   **寄存器文件：** 不同的指令类型可能对寄存器堆（Register File）的压力不同，错峰使用可以避免因寄存器不足导致的 warp 阻塞。

### 3. 计算任务与计算任务重叠（Compute over Compute）
> *Overlap compute workloads with other compute workloads... efficient on NVIDIA Ampere...*

*   **原因：** 架构改进带来的调度灵活性。
    *   **Ampere 架构优势：** 相比旧架构，Ampere 及之后的架构（Ada Lovelace, Hopper）改进了调度器，能更细粒度地在 SM 之间或 SM 内部划分资源。
    *   **纯计算的灵活性：** 计算着色器（Compute Shader）没有图形管线（Graphics Pipeline）那么多固定功能的束缚（如 Rasterizer, ROP），更容易被驱动程序打散并填补到空隙中。

### 4. 将图形工作（如后处理）转换为计算
> *Convert some of the graphics work such as post-processing passes to compute...*

*   **原因：** 解锁调度自由度。
    *   **Graphics Queue 的刚性：** 标准的图形 pass（画全屏三角形进行后处理）通常必须在 Graphics Queue 上按顺序执行，且受限于 ROP（光栅化操作单元）吞吐。
    *   **Async Queue 的弹性：** 将后处理改为 Compute Shader 后，可以将其放入 Async Compute Queue。这样它就不必等待上一帧的某个非依赖性的图形工作完全结束，可以提前开始，或者与下一帧的几何处理（Geometry）并行。

### 5. 在帧与帧之间运行异步工作
> *Consider running async work between frames.*

*   **原因：** 利用“死区时间”。
    *   **帧间隔：** 在一帧渲染结束（Present）到下一帧开始之间，或者 CPU 提交命令的间隙，GPU 往往会出现短暂的空闲（Bubble）。
    *   **填充空隙：** 这是运行物理模拟、AI 寻路、资源解压或流式加载等非即时渲染任务的绝佳时机。

### 6. 测量整体帧性能，而非单个 Pass
> *Measure the performance difference over the whole frame...*

*   **原因：** 局部牺牲换取全局胜利。
    *   **执行时间延长：** 当 Async Compute 开启时，主队列的任务（Graphics Queue）可能会因为资源分流而变慢（例如从 2ms 变成 2.5ms），异步任务本身也可能比单独运行慢。
    *   **总时长缩短：** 但是，只要这两个任务并行完成的总时间（例如 2.5ms）小于它们串行完成的时间总和（2ms + 1ms = 3ms），你就赢了。只看单个 Shader 的耗时会产生误导。

### 7. 验证不同 GPU 层级（Tier）的行为
> *Verify the behavior across different GPU tiers...*

*   **原因：** 资源充裕度决定了重叠潜力。
    *   **高端卡（4090）：** SM 极多，单个任务很难喂饱所有 SM，因此有大量空闲资源可供重叠。
    *   **低端卡（3050/Laptop）：** 资源很少，主渲染任务可能已经占满了所有 SM 和带宽。强制重叠反而会导致上下文切换（Context Switching）开销和缓存抖动（Cache Thrashing），导致负优化。

### 8. 验证不同分辨率下的行为
> *Verify the behavior at different resolutions...*

*   **原因：** 瓶颈转移（Bottleneck Shifting）。
    *   **低分辨率（1080p）：** 瓶颈通常在 CPU（DrawCall 提交慢）或几何阶段（顶点多）。此时 Pixel Shader 负载低，SM 有大量空闲时间，适合插入 Async Compute。
    *   **高分辨率（4K）：** 瓶颈在 GPU 光栅化和像素处理。SM 满载运行 Pixel Shader，显存带宽也可能打满。此时插入额外任务很难获得收益，甚至引发争抢。

---

### 总结：为什么“不推荐”那些做法？

这一部分反过来验证了上述逻辑：

*   **不要只看 Occupancy：** 满 Occupancy 不代表满效率，可能是一堆线程都在卡顿等待（Stall）。
*   **不要重叠读写相同资源：** 数据竞争（Data Hazard）需要加锁或屏障（Barrier），这会强制同步，抵消了并行的意义。
*   **不要重叠高 Cache/VRAM 占用的任务：** L1/L2 Cache 是有限的。如果两个任务都在疯狂抢 Cache，会导致 Cache Hit Rate 暴跌，总吞吐量下降（带宽瓶颈）。
*   **不要重叠 RT 任务：** RT Core 是稀缺资源，两个 RT 任务一起跑只会互相排队，没有任何并行的物理基础。

总而言之，**Async Compute 是关于资源管理的艺术**——你必须非常清楚你的主任务“剩下了什么”，然后用恰好需要这些剩余资源的副任务去填补它。