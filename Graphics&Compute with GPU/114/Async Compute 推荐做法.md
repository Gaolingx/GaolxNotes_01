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

这些“推荐 / 不推荐”实践的核心逻辑，并不在于某一条经验本身，而在于**Async Compute 是否真的形成了“资源互补的并行”，而不是“同一瓶颈上的时间切片”**。下面我按“硬件调度真实约束 → 指标选择 → 架构层原因”的顺序，把为什么这些做法是正确的系统性地解释清楚。

---

## 一、为什么要从 **Unit Throughput** 而不是 **SM Occupancy** 入手

### 1. SM Occupancy 只是“线程是否能驻留”，不是“是否有可执行的工作”

* SM occupancy 反映的是：

  * Warp 是否能被调度进 SM
  * 是否被寄存器 / shared memory 限制
* 但 **warp 能驻留 ≠ warp 正在执行有用指令**

真实执行中，warp 可能因为：

* memory dependency
* pipeline hazard
* scoreboard stall
* 等待 tensor / RT / LSU pipeline

而处于 **Idle / Waiting** 状态。

### 2. Unit Throughput 直接反映“哪个硬件单元在闲着”

Nsight Graphics 的 Top Unit Throughput（SOL）指标，本质上回答的是：

> “在这段时间内，哪些 execution unit 没被喂饱？”

例如：

* FP32 ALU 20%
* Tensor Core 5%
* LSU 80%
* RT Core 0%

这意味着：

* **算力侧存在大量可被填充的空洞**
* 只要新 workload 不和 LSU 冲突，就几乎是“白捡性能”

这也是为什么文档强调：

> *SM Idle % without conflicting high throughput units is almost always a guaranteed improvement*

这是在**硬件层面近似确定性成立**的判断。

---

## 二、为什么推荐“寻找低吞吐单元的互补对”

### 1. GPU 是多 datapath 并行系统，不是单一算力池

NVIDIA GPU 内部至少包含以下相对独立的 datapath：

* FP / INT ALU
* Tensor Core
* Load/Store (Memory Requests)
* RT Core
* Graphics fixed-function pipe（ROP、Raster、Setup）

Async Compute 的本质是：

> **让不同 datapath 的 pipeline 同时保持 non-empty**

### 2. FP / ALU / Tensor “寄存器文件分离”非常关键

在 Ampere 及之后架构中：

* Tensor Core 使用独立寄存器 bank
* FP / INT 使用通用 RF
* Tensor pipeline 与 FP32 pipeline 结构性分离

这意味着：

* Tensor-heavy CS 可以与 FP-heavy PS 并行
* Tensor-heavy AI denoise 可以和 vertex / raster overlap

这不是“理论可能”，而是 **硬件设计目标**。

---

## 三、为什么“Compute-over-Compute”在 Ampere 上反而高效

这是很多人直觉上会怀疑的一点。

### 1. Ampere 的关键变化

* 更强的 instruction-level interleaving
* 更成熟的 scoreboard 与 dual-issue
* 更高效的 async pipeline draining
* Tensor / FP / LSU 更独立

因此：

* 两个 compute queue 并不等价于“时间切片”
* 而是可能形成 **真正的多 pipeline 并行**

### 2. 但前提仍然是：**unit usage 互补**

如果两个 compute workload：

* 都是 memory-bound
* 或都在 saturate L2 / DRAM

那么 overlap 一定是负收益，这也是后面“不推荐”的来源。

---

## 四、为什么建议把部分 Graphics Pass 转成 Compute

### 1. Graphics pipeline 的固定阶段限制 overlap

Graphics queue：

* 强依赖 pipeline order
* 隐含 barrier（如 render target transitions）
* 受限于 raster / ROP 结构

Compute queue：

* barrier 更显式
* 更容易拆分为小粒度任务
* 更适合异步调度

### 2. Compute 化 = 更可控的 overlap 拓扑

例如：

* SSAO
* Bloom
* TAA resolve
* Post-process blur

这些本质是：

* texture read-heavy
* ALU-heavy
* 对 graphics fixed-function 依赖弱

Compute 化后：

* 可以精准安排在 graphics pipeline 的“空洞”中执行
* 更容易形成 datapath 互补

---

## 五、为什么要“跨帧 async”

### 1. 同一帧内 overlap 的窗口非常短

Intraframe async compute：

* 依赖 tight synchronization
* 容易被一个 barrier 吃掉全部收益

### 2. Interframe async 的优势

* dependency 距离大
* scheduling 自由度高
* 能避开 WFI（Wait For Idle）

这在现实引擎中往往是**最稳妥、最容易获得收益**的 async 模式。

---

## 六、为什么必须“整帧 / 多帧”测量

Async Compute 的收益往往表现为：

* GPU bubble 减少
* frame tail 缩短
* pipeline 更平滑

而不是：

* 单个 pass 时间下降

如果只看单 pass：

* 你甚至可能看到它“变慢了”
  但整帧：
* frame time 更短
* variance 更低

---

## 七、为什么高端 GPU、低分辨率更容易受益

### 1. 高端 GPU

* 更多 SM
* 更多 execution unit
* 更容易出现“未被使用的算力岛”

Async Compute 的空间更大。

### 2. 低分辨率

* Pixel warp 数量减少
* Raster / PS 压力降低
* SM 更容易出现 idle slot

这正是 async compute 最容易“填洞”的场景。

---

## 八、逐条解释“不推荐”的底层原因

### 1. 只看 SM Occupancy 是错误的

因为：

* Occupancy ≠ Execution
* 高 occupancy 也可能 90% 时间在等 memory

### 2. 长 async workload 风险极高

* 会顶住 dependency barrier
* 导致 sync queue 等待
* 直接拉长 frame tail

Async workload 必须是：

> **short, prefetch-like, latency-hiding**

### 3. 读写同一资源 = data hazard

* 隐式 cache flush
* 强制 barrier
* L1/L2 invalidation

直接抹杀 overlap。

### 4. 高 L1/L2 / VRAM 使用不可 overlap

这是最常见的“负收益 async compute”来源：

* cache thrashing
* DRAM oversubscription
* hit-rate 下降 → latency 上升

### 5. 多 queue + 软件调度 = serialization

在未启用 HAGS 时：

* OS 层调度
* driver 串行提交
* 反而比单 queue 更慢

### 6. 双 compute 都触发 WFI 非常危险

WFI 会：

* drain pipeline
* 强制同步
* 抹掉指令级 overlap

Descriptor heap 频繁切换是常见诱因。

### 7. DX12 queue priority 并不等于 warp priority

Queue priority：

* 只影响 command fetch
* 不影响 SM 内 warp scheduling

因此几乎无法用于 fine-grained async tuning。

### 8. 不推荐 overlap RTCore workload

RT Core：

* 吞吐单元高度集中
* cache / memory pressure 极高
* 与其他 RT workload 冲突极大

重叠只会导致：

* traversal 变慢
* BVH cache miss 上升

---

## 总结一句话

**Async Compute 不是“并行多干点活”，而是“用本来闲着的硬件单元干活”。**

因此：

* 从 unit throughput 出发是必然
* 资源互补是唯一成功条件
* memory / cache 冲突是最大敌人
* Ampere 之后 compute-over-compute 才真正可行
* 错误的 async 比没有 async 更糟

如果你愿意，下一步我们可以用一个**具体 Nsight GPU Trace 案例**，逐项对照这些原则，演示“一个 async 优化是如何从看起来合理到最终被否定的”。
