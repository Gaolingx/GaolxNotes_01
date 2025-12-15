# AsyncCompute 解析

### 🧑‍💻 User

在近2年GDC育碧各工作室分享的ppt渲染管线中都出现了AsyncCompute，基本上就是在Render XXXX渲染各个步骤之外又有一条并行的称为AysncCompute的线（Compute Queue），里边的操作都是Compute Shader实现的。
这个神秘的AsyncCompute线究竟是什么？

---

### 🤖 Assistant

在近两年 GDC 上育碧各工作室展示的渲染管线 PPT 中，那条被单独画出来、与传统 **Graphics Queue** 并行的 **Async Compute（Compute Queue）**，本质上并不神秘，它是**现代 GPU 硬件并行执行能力在 API 层的显式暴露与工程化利用**。

可以从 **“它不是什麼”** 和 **“它到底是什么”** 两个层面来理解。

---

## 一、Async Compute 不是“另一张 GPU”，也不是“偷偷更快的 Compute Shader”

这是很多人第一次看到那条线时的直觉误解。

Async Compute **并不是**：

1. **额外的一套算力**

   * 它不提供新的 ALU、TMU 或带宽
   * 所有算力仍然来自同一块 GPU

2. **Compute Shader 的特殊版本**

   * Shader 代码层面完全相同
   * 区别不在 HLSL/GLSL，而在 **提交到哪个硬件队列**

3. **自动加速**

   * 如果资源、同步、带宽没有设计好
   * Async Compute 反而可能 **更慢**

---

## 二、Async Compute 的本质：独立的硬件执行队列（Hardware Queue）

### 1. 现代 GPU 是多队列设备

以 GCN / RDNA / Turing+ 为代表的 GPU，都具备：

* **Graphics Queue**

  * Vertex / Pixel / Geometry
* **Compute Queue**

  * Compute Shader（不走 raster pipeline）
* （有时还有 Copy / DMA Queue）

这些队列在硬件上是**真实存在、可并行调度的**。

> Async Compute = **允许 Compute Queue 与 Graphics Queue 同时活跃**

---

### 2. API 层的体现

在现代 API 中这是显式的：

* **DX12**

  * `D3D12_COMMAND_LIST_TYPE_DIRECT`
  * `D3D12_COMMAND_LIST_TYPE_COMPUTE`
* **Vulkan**

  * Graphics Queue
  * Compute-only Queue

引擎把任务分别提交到不同队列，由 GPU 的 **硬件调度器（Scheduler）** 决定如何交错执行。

---

## 三、为什么 Async Compute 在渲染管线里“看起来像另一条时间线”

### 1. GPU 不是严格串行的

在传统“单队列”心智模型中：

```
Shadow → GBuffer → Lighting → PostProcess
```

但真实 GPU 内部是：

* 不同阶段使用 **完全不同的执行单元**
* 大量 pipeline bubbles 本来就存在

例如：

| 阶段         | 主要瓶颈                |
| ---------- | ------------------- |
| Shadow Map | Vertex / Geometry   |
| SSAO       | Texture fetch + ALU |
| TAA        | ALU + LDS           |
| Lighting   | ALU / Tensor / LDS  |

**这些资源并不完全重叠**。

---

### 2. Async Compute 的核心动机

> **用 Compute Work 填充 Graphics Pipeline 的“空洞”**

典型场景：

* Graphics Queue 在：

  * 等待 memory
  * 等待 raster
  * vertex-heavy、pixel-light
* Compute Queue 可以同时：

  * 跑 SSAO
  * 跑 SSR
  * 跑 TAA
  * 跑 Culling / LOD / Skinning

于是 PPT 上就画成了：

```
Graphics: |---- Shadow ----|---- GBuffer ----|---- Lighting ----|
Compute :       |---- SSAO ----|    |-- TAA --|
```

这并不是“引擎想象出来的”，而是**真实硬件并发**。

---

## 四、为什么育碧这两年特别强调 Async Compute？

因为他们的项目满足了 **Async Compute 成立的三个必要条件**：

### 1. 有大量 Compute-heavy 的屏幕空间 / 后处理算法

例如：

* SSAO / GTAO
* SSR
* TAA / TSR
* Volumetric Fog
* GPU Culling
* Clustered / Tiled Lighting

这些本来就**不需要 raster pipeline**。

---

### 2. 渲染阶段存在明显的资源互补

你前面提到的例子非常典型：

> Shadow Map（Vertex-heavy）
> SSAO（Depth sampling + ALU）

* Shadow Map：

  * VS / GS 压力大
  * PS 很轻
* SSAO：

  * 几乎不碰 VS
  * 吃 ALU + Cache

**资源错峰 = Async Compute 能真正跑满 GPU**

---

### 3. 引擎已经高度 Pipeline 化

Async Compute 不是“加个队列就行”，前提是：

* Pass 之间依赖清晰
* 资源生命周期明确
* Barrier / Fence 管理成熟

这正是 Ubisoft 这类 **多年自研引擎 + 跨世代积累** 的优势。

---

## 五、Async Compute 真正难的地方（也是 PPT 不会细讲的）

### 1. 同步与资源 Hazard

* Compute 写的 UAV
* Graphics 读的 SRV

需要：

* UAV Barrier
* Queue-to-Queue Fence

做错了不是花屏，而是 **隐性性能退化**。

---

### 2. 带宽与 Cache 争用

Async Compute **不保证**：

* 带宽翻倍
* Cache 不抖

在一些场景下：

* Compute 会把 Graphics 的 L2 / L1 冲掉
* 导致总帧时间上升

因此需要：

* Pass 粒度控制
* 合理切分 Dispatch

---

### 3. 不同 GPU 架构差异巨大

* GCN：Async Compute 非常强
* RDNA：需要更谨慎
* NVIDIA：并发能力存在，但调度策略不同

这也是为什么很多引擎：

* 有 Async Compute 路径
* 但 **按 Vendor / GPU Tier 开关**

---

## 六、一句话总结

> **Async Compute 并不是“额外算力”，而是让 GPU 原本被浪费的并行能力真正被用起来。**

在育碧的渲染管线图里，那条“神秘的 AsyncCompute 线”代表的是：

* 一个独立的硬件 Compute 队列
* 被精心挑选、与 Graphics 资源互补的任务
* 通过严格同步与调度，实现真实的 GPU 并行

如果你愿意，下一步我们可以直接用一个 **Shadow Map + SSAO 的真实 GPU 时间线**，逐周期拆解为什么它能 overlap，以及在哪些情况下它一定不能。


---

### 🧑‍💻 User

例如，可以在渲染shadowmap的时候用cs来做SSAO，因为sm和ssao其实对GPU资源的要求是很不同的，因此同时做其实更充分的利用了GPU，那确实，sm重点是vertex处理，ssao主要是对depth texture采样，两者要求非常互补，同时肯定能提升效率，具体细节能不能深入分析一下？这是一个很典型的Async Compute and Overlap问题

---

### 🤖 Assistant

这是一个**非常标准、也非常“教科书级”的 Async Compute overlap 场景**。你这个直觉本身是对的，但要把它从“概念正确”推进到“工程上真的快”，必须把 **GPU 的执行资源、瓶颈位置、以及真实时间线**拆到足够细。

下面我按 **硬件资源 → 两个 pass 的微观特征 → 为什么能 overlap → 什么时候反而不行 → 工程实现细节** 这五个层次来分析。

---

## 一、先明确一个关键前提：GPU 并行不是“两个 kernel 同时跑”

Async Compute 成立的**真正条件**是：

> **两个任务的瓶颈资源不同，且调度器能在同一时间片内把它们映射到不同执行单元**

否则就只是时间片轮转，看起来并行，实际上串行。

---

## 二、Shadow Map Pass 的真实硬件画像（Graphics Queue）

以常见的 Directional Shadow Map 为例：

### 1. 工作负载分解

* **Vertex Shader**

  * 大量 instance
  * World → Light View Projection
* **Primitive Assembly / Raster**

  * 深度-only
* **Pixel Shader**

  * 极简（甚至空）

### 2. 硬件瓶颈在哪里？

| 硬件单元                 | 利用率    |
| -------------------- | ------ |
| VS / Attribute Fetch | **高**  |
| Raster / Setup       | 中      |
| PS ALU               | **极低** |
| Texture Units        | 几乎不用   |
| LDS                  | 不用     |
| L2 / DRAM 带宽         | 低～中    |

> **Shadow Map 是一个典型的 “前端（Vertex）受限” pass**

在这一阶段：

* 大量 SM / CU 中的 **pixel pipe 是空的**
* ALU、Texture pipeline 明显闲置

---

## 三、SSAO（Compute Shader）的真实硬件画像（Compute Queue）

以 GTAO / HBAO 类 SSAO 为例：

### 1. 工作负载分解

* 输入：Depth Texture（只读）
* 核心操作：

  * 多次 depth sampling
  * 随机旋转 kernel
  * ALU-heavy accumulation

### 2. 硬件瓶颈在哪里？

| 硬件单元                | 利用率   |
| ------------------- | ----- |
| ALU                 | **高** |
| Texture Units       | **高** |
| LDS / Shared Memory | 中     |
| VS / Raster         | 不使用   |
| Primitive Setup     | 不使用   |

> **SSAO 是典型的 “后端（ALU + Texture）受限” compute workload**

---

## 四、为什么这两个 pass 能“真·并行”？

### 1. 资源互补（这是最关键的一点）

| 资源            | Shadow Map | SSAO   |
| ------------- | ---------- | ------ |
| VS Front-End  | **吃满**     | 不用     |
| Raster        | 使用         | 不用     |
| PS ALU        | 空闲         | **吃满** |
| Texture Units | 空闲         | **吃满** |
| LDS           | 空闲         | 使用     |
| Register File | 中          | 高      |

> **两者竞争的只有：**
>
> * L2 Cache
> * DRAM 带宽（而且 Shadow Map 还不算重）

因此调度器可以做到：

* 一个 CU 在跑 vertex wave
* 同时另一些 CU 在跑 compute wave
* 甚至 **同一个 CU 内 warp / wave 交错执行**

---

### 2. 硬件调度层面的实际行为

在支持 Async Compute 的 GPU 上（GCN / RDNA / Turing+）：

* Graphics Queue 发射：

  * VS wavefront
* Compute Queue 发射：

  * CS wavefront

硬件调度器做的事情是：

* **按资源依赖而非队列顺序调度**
* 在同一时间 slice 内：

  * 若 pixel pipe idle → 派发 compute wave
  * 若 ALU 有空 → 派发 SSAO wave

这就是 PPT 上那条“并行线”的真实含义。

---

## 五、用“时间线”描述一次真实 overlap

假设：

* Shadow Map Pass：4 ms（VS-bound）
* SSAO Compute：2 ms（ALU-bound）

**没有 Async Compute：**

```
0ms ── Shadow Map ── 4ms ── SSAO ── 6ms
```

**有 Async Compute（理想情况）：**

```
0ms ── Shadow Map ────────────── 4ms
       ├── SSAO (Compute) ──┤
0ms ──────────────── 2ms
```

最终：

* 帧时间：4 ms（不是 6 ms）
* SSAO “几乎免费”

---

## 六、什么时候这个 overlap 会失败？

这是工程中最容易踩坑的地方。

### 1. 带宽开始成为瓶颈

如果：

* SSAO sampling 半径大
* Depth 是 full-res
* Shadow Map 是高分辨率 cascaded SM

那么：

* 两者同时冲击 L2 / DRAM
* 导致：

  * Shadow Map VS stall
  * SSAO texture miss

结果：**两个都慢**

---

### 2. Barrier 放错位置

常见错误：

* SSAO 需要 depth pre-pass
* 但 depth 尚未完成 → 强制全局 sync

结果：

```
Graphics: Shadow Map ── Barrier ── SSAO
Compute :        （空等）
```

Async Compute 直接退化成串行。

---

### 3. Dispatch 粒度过大

* SSAO 一次 dispatch 覆盖全屏
* wave 数量太多
* Compute Queue 抢占过强

结果：

* Graphics Queue 被 starvation
* 反而延长 shadow map 时间

---

## 七、育碧 / 一线引擎的工程实践要点

你在 PPT 里看到的“漂亮 overlap”，背后通常有这些策略：

### 1. 切分 SSAO

* Tile-based SSAO
* 分多次 dispatch
* 中间可被抢占

### 2. 显式限制 Compute 占比

* 限制 wave 数
* 控制 LDS / register 使用
* 避免压死 graphics

### 3. 严格的 Queue Fence 设计

* SSAO **只读 depth**
* 不依赖 shadow map 输出
* barrier 精确到 subresource

---

## 八、一个总结性的判断标准

你可以用这句话快速判断一个 Async Compute overlap 是否“值得做”：

> **如果 Graphics pass 正在吃 VS / Raster，而 Compute pass 正在吃 ALU / Texture，且两者几乎没有写后读依赖，那么这是一个一等公民级的 Async Compute 场景。**

Shadow Map + SSAO，正是这个标准案例。

---

如果你愿意，下一步我可以：

* 用 **AMD GPU 的 wavefront / CU 调度模型** 拆一次更底层的 overlap
* 或对比 **NVIDIA 上为什么这个 overlap 收益更不稳定**
* 或直接给你一个 **DX12/Vulkan 级别的同步与队列提交示意**


---

### 🧑‍💻 User

About Async Compute and Overlap，a classic mistake is to focus only on SM occupancy (unused warp slots) to identify potential async compute workloads. The GPU is a complex beast and other metrics such as top unit throughputs (SOL) play an equal if not more important role than SM occupancy alone.

Thus, in addition to SM occupancy, you should also consider unit throughputs, register file occupancy, group shared memory, and different datapaths. After the ideal pair is identified, the compute workload is moved to the async queue. It is synchronized with the sync/main queue using fences to ensure the correct execution order.

Question:
1. how to to identify potential async compute workloads?
2. how to synchronize with the sync/main queue using fences with dx12?

---

### 🤖 Assistant

Your premise is correct and reflects how mature engines actually reason about Async Compute: **SM occupancy is neither necessary nor sufficient**. What matters is *which parts of the GPU are saturated* and *which are idle* over time.

I will answer the two questions separately, at an engineering-practical level rather than a conceptual one.

---

## 1. How to identify potential Async Compute workloads

Think of this as a **resource orthogonality and dependency analysis problem**, not a shader classification problem.

### 1.1 Start from GPU bottleneck classification, not “compute vs graphics”

For each candidate pass, you want to answer:

> *What hardware units are the limiting factor for this pass at peak load?*

Typical bottleneck classes:

| Bottleneck class           | Dominant limiters                        |
| -------------------------- | ---------------------------------------- |
| Front-end / geometry bound | IA, VS, attribute fetch, primitive setup |
| Raster bound               | Rasterizer, depth/stencil                |
| Pixel ALU bound            | FP/INT ALUs                              |
| Texture bound              | TEX, L1/L2 hit rate                      |
| Memory bound               | DRAM BW, L2                              |
| LDS / shared memory bound  | LDS bandwidth or capacity                |
| Register bound             | Register file pressure → wave throttling |

You obtain this from **vendor profilers**, not from occupancy numbers:

* Nsight Graphics / Nsight Compute
* Radeon GPU Profiler (RGP)
* PIX timing captures

If a workload’s *top stall reason* overlaps heavily with another pass, it is **not** a good Async Compute candidate.

---

### 1.2 Look for *complementary* throughput saturation

A good Async Compute pair looks like this:

| Resource       | Graphics pass | Compute pass |
| -------------- | ------------- | ------------ |
| VS / primitive | High          | None         |
| Raster         | High          | None         |
| Pixel ALU      | Low           | High         |
| Texture        | Low–medium    | High         |
| LDS            | Low           | Medium       |
| Registers      | Moderate      | High         |

Classic examples that survive real profiling:

* Shadow map rendering + SSAO / GTAO
* Depth prepass + light list construction
* G-buffer raster + clustered lighting build
* Skinning / culling CS + early graphics passes

Bad examples (despite “free SM slots”):

* Two memory-bound compute passes
* Compute competing with heavy pixel shading
* Full-res postprocess overlapping bandwidth-heavy passes

---

### 1.3 Explicitly evaluate **shared bottlenecks**

Before moving anything to async, you must verify:

1. **DRAM bandwidth headroom**

   * Async Compute cannot create bandwidth
   * If either pass is BW-bound, overlap hurts

2. **L2 cache contention**

   * Random depth sampling + large vertex streams = cache thrash

3. **Register file pressure**

   * High register usage can reduce active waves and starve graphics

4. **LDS capacity**

   * Large group shared memory allocations reduce concurrent waves

This is why “unused warp slots” alone is misleading:
they often exist *because* another unit is saturated.

---

### 1.4 Dependency analysis: the hidden deal-breaker

A workload is **disqualified** for Async Compute if it has:

* Write-after-read or write-after-write dependencies with the graphics pass
* Frequent UAV writes consumed immediately by graphics
* Frame-critical latency sensitivity (e.g., must finish before visibility)

Ideal async candidates:

* Read-only inputs (SRV)
* Outputs consumed much later in the frame
* Tolerant to partial or tiled execution

---

### 1.5 Practical identification workflow

In production engines, this usually looks like:

1. Capture a GPU frame with everything on the main queue.
2. Identify passes where:

   * Graphics queue shows long idle gaps in certain units.
3. Find compute passes whose stall reasons match those gaps.
4. Temporarily move them to async and re-profile.
5. Keep them async **only if total frame time decreases**.

Async Compute is validated empirically, not theoretically.

---

## 2. How to synchronize Async Compute with the main queue in DX12

DX12 synchronization is explicit and low-level by design. The core tools are:

* **ID3D12Fence**
* **Command queue `Signal` / `Wait`**
* **Resource barriers (for hazards, not execution order)**

### 2.1 The fundamental rule

> **Execution order between queues is undefined unless you enforce it with fences.**

Resource barriers do **not** synchronize queues.

---

### 2.2 Typical Async Compute synchronization pattern

Assume:

* `gfxQueue` → main graphics queue
* `computeQueue` → async compute queue
* `fence` → shared fence object

#### Case A: Compute must start *after* a graphics pass

Example: SSAO needs depth buffer produced by depth prepass.

```cpp
UINT64 fenceValue = ++fenceCounter;

// Graphics queue signals when depth is ready
gfxQueue->Signal(fence, fenceValue);

// Compute queue waits before executing SSAO
computeQueue->Wait(fence, fenceValue);

// Submit SSAO command list
computeQueue->ExecuteCommandLists(1, &ssaoCmdList);
```

Key point:

* The wait is **queue-level**, not command-list-level.
* GPU will not start executing compute work until the signal is reached.

---

#### Case B: Graphics must wait for async compute to finish

Example: Lighting pass consumes SSAO result.

```cpp
UINT64 fenceValue = ++fenceCounter;

// Compute queue signals when SSAO is done
computeQueue->Signal(fence, fenceValue);

// Graphics queue waits before lighting
gfxQueue->Wait(fence, fenceValue);
```

---

### 2.3 Combining with resource barriers

Fence synchronization only guarantees **execution order**, not **memory visibility**.

You still need:

* UAV barriers for UAV writes
* Transition barriers for state changes

Typical pattern:

1. Compute writes UAV
2. Signal fence
3. Graphics waits fence
4. Graphics applies UAV barrier
5. Graphics reads as SRV

---

### 2.4 Avoiding accidental serialization

Common mistakes that kill overlap:

1. **Over-synchronization**

   * Waiting too early instead of just before consumption

2. **Using one fence value for everything**

   * Forces unnecessary global ordering

3. **Large monolithic dispatches**

   * Prevent fine-grained interleaving

High-end engines:

* Use **multiple fence values**
* Synchronize at *pass boundaries*, not frame boundaries

---

### 2.5 Mental model that prevents bugs

Think of DX12 fences as:

> “GPU-side condition variables that gate queue progress.”

They do not flush caches, do not change resource states, and do not imply barriers.

---

## Summary

**1. Identifying Async Compute workloads**

* Ignore SM occupancy in isolation
* Profile bottlenecks at the *unit throughput* level
* Look for complementary saturation, not idle warps
* Ensure minimal data dependencies and bandwidth contention

**2. DX12 synchronization**

* Use `Signal` / `Wait` on command queues
* Fences define execution order, not memory hazards
* Combine fences with explicit resource barriers
* Synchronize as late and as locally as possible

If you want, the natural next step is to walk through a **real SSAO + shadow map timeline in PIX or RGP**, showing exactly where fences are placed to preserve overlap while avoiding hidden serialization.
