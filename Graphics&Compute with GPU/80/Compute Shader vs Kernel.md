# Compute Shader vs Kernel

### 🧑‍💻 User

为什么关于compute shader bug/性能瓶颈，有ai infra/hpc kernel相关经验的开发人员相比只写过ps/vs shader的ta会处理会更加得心应手？

---

### 🤖 Assistant

这是一个**非常典型、也非常现实的差异**，本质并不在于“谁更聪明”，而在于**问题建模方式、性能直觉来源以及对硬件–编译器–运行时整体的理解深度不同**。可以从以下几个层面来解释，为什么有 **AI infra / HPC kernel** 背景的人，在处理 **compute shader 的 bug 和性能瓶颈**时，往往比只写过 PS/VS 的 TA 更“游刃有余”。

---

## 一、Compute Shader 的本质更接近「通用并行计算」，而不是「图形管线」

PS/VS TA 的经验高度绑定在**图形管线抽象**之上：

* VS → Raster → PS
* 输入/输出语义明确
* 大量行为由 fixed-function stage 隐式保证
* “跑慢了”往往可以归因到：

  * overdraw
  * texture 带宽
  * branch divergence（在 PS 中）

而 **compute shader** 本质上是：

> 一个运行在 GPU 上的 **SPMD 并行程序**，你要为它的 **并行结构、内存访问、同步模型、调度粒度**负责。

这和 AI / HPC kernel 的认知模型高度一致：

| Compute Shader    | AI / HPC Kernel          |
| ----------------- | ------------------------ |
| workgroup         | thread block / CTA       |
| local memory      | shared memory            |
| global buffer     | global memory            |
| barrier           | `__syncthreads()`        |
| occupancy         | SM utilization           |
| LDS bank conflict | shared mem bank conflict |

**AI/HPC 背景的人一眼就会把 CS 当成“CUDA kernel / Triton kernel / OpenCL kernel”来看待**，而不是“没有 PS 的 shader”。

---

## 二、问题定位方式：图形思维 vs 计算思维

### TA 常见的定位路径（图形经验）：

1. “是不是 dispatch 太多？”
2. “是不是纹理采样太多？”
3. “是不是某个 if 分支太重？”
4. “要不要拆 pass？”

这些在 compute 中**经常不命中要害**。

---

### AI / HPC kernel 工程师的第一反应通常是：

1. **并行映射是否合理**

   * 一个 thread 在干多少活？
   * 一个 workgroup 覆盖多少数据？
   * 是否存在串行循环隐藏在 kernel 内？

2. **内存层级分析**

   * global → shared → register
   * 是否重复加载同一数据？
   * 是否发生 bank conflict / uncoalesced access？

3. **调度与 occupancy**

   * register 使用量是否过高？
   * local memory 是否导致 occupancy 崩塌？
   * barrier 是否在热路径？

4. **roofline 模型**

   * 当前 kernel 是 compute-bound 还是 memory-bound？
   * 算术强度是否低得不可能快？

这些分析方式**和你在 AI kernel（GEMM / attention / conv）里做的事情完全一致**。

---

## 三、对“性能异常”的容忍度与直觉完全不同

### PS/VS 世界里

* 性能波动常被管线掩盖
* GPU driver / 硬件替你兜底很多事情
* 即使代码不理想，也可能“还能跑”

### Compute / HPC 世界里

* **性能异常就是 bug**
* 2× 慢通常意味着：

  * 一个隐藏的同步
  * 一个 cache line 被反复抖动
  * 一个 warp 只有 3 个 lane 在干活
* 没有 fixed-function 帮你兜底

所以 AI / HPC 工程师对这些现象极其敏感：

> “这个 kernel 理论上不可能这么慢，一定有结构性问题。”

而不是先去怀疑“驱动/平台”。

---

## 四、Bug 类型完全不同：数据竞争 vs 渲染错误

TA 更熟悉的 bug 类型：

* 颜色不对
* 阴影抖动
* 深度错乱
* UV 对不上

这些是**可视化错误**，反馈非常直接。

---

Compute shader 的常见 bug：

* race condition（偶发、设备相关）
* barrier 放错导致未定义行为
* out-of-bounds 写导致“别的 buffer 坏掉”
* 非法 aliasing 在不同 driver 上表现不同
* 某些 GPU 上必现，某些 GPU 上完全正常

这些 bug 的处理方式是：

* 构造最小 repro
* 推导内存可见性与 happens-before
* 用数学而不是“看画面”来判断正确性

**这是 HPC / AI infra 工程师的日常工作方式**。

---

## 五、编译器与 ISA 意识差距

### TA 通常停留在：

* HLSL / GLSL 语义层面
* “我写的是这样，硬件应该会那样”

### AI / HPC 工程师通常会：

* 看反汇编（SASS / GCN ISA / DXIL）
* 知道：

  * 哪些操作会强制 local memory
  * 哪些 pattern 会被 vectorize / unroll
  * barrier / atomics 的真实代价
* 知道“**你写的 ≠ 编译器生成的**”

所以在 compute shader 性能问题上，他们更容易：

* 判断是 **算法层问题** 还是 **代码生成问题**
* 规避 driver/compiler 的已知坑

---

## 六、工作负载规模感知完全不同

PS/VS TA 的常见规模：

* 每帧 1–10M fragment
* 每个 fragment 干的事情相对固定
* GPU pipeline 高度稳定

Compute / AI 的典型规模：

* 上亿 / 数十亿 element
* 工作负载高度参数化
* 对 launch shape 极其敏感

这让 AI / HPC 工程师天然会问：

* grid / group size 是否合理？
* 是否存在 launch overhead 被放大？
* 是否应该 fuse kernel？

而 TA 更容易停留在“单次 dispatch”的视角。

---

## 七、总结一句话

> **Compute shader 的问题本质是并行程序与硬件协同的问题，而不是渲染问题。**

因此：

* **AI infra / HPC kernel 工程师**

  * 习惯用硬件与并行计算模型思考
  * 对性能异常和并发 bug 极其敏感
  * 自然擅长 compute shader 的 debug 与优化

* **只写过 PS/VS 的 TA**

  * 更熟悉图形管线和视觉反馈
  * 在 compute 场景中需要“重学一套心智模型”

---

如果你愿意，下一步我们可以直接对比：

* **“TA 写出来的 compute shader”**
* vs
* **“HPC 思维下写的 compute shader”**

从 thread mapping、memory layout 到 barrier placement，一行一行拆给你看。
