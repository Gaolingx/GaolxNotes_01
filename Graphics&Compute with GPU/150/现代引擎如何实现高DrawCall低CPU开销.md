# 现代引擎如何实现高DrawCall低CPU开销

### 🧑‍💻 User

为什么现代游戏引擎可以轻轻松松做到10w drawcall而不会出现明显cpu overhead，为什么nsight软件显示上万dc 但dispatch却只有三四百？

---

### 🤖 Assistant

**现代引擎能轻松做到 ~10 万级 draw call 且 CPU 开销可控，核心是「低开销 API + 间接/GPU-driven 绘制」把大量工作从 CPU 挪走或合并；Nsight 里看到上万 DC 但只有三四百 dispatch，通常是因为工具统计的是展开后的有效绘制次数，而实际 API/提交次数很少。**

### 1. 为什么现代引擎能扛 10 万级 DrawCall 且 CPU 不明显爆

传统 DX11/OpenGL 时代，单 draw 的驱动验证、状态切换、内核切换开销很大，几千 draw 就容易 CPU bound。现代做法从根上改了：

- **低开级显式 API（Vulkan / D3D12 / Metal）**
  - 驱动验证大幅前移/减少，draw 时几乎不做重活。
  - 命令列表（Command Buffer / Command List）可多线程并行录制，把录制开销摊到多核。
  - 资源屏障、描述符、同步都由应用显式控制，避免驱动“猜”。
  单 draw 的 CPU 开销可降到非常低（微秒级甚至更低），因此“数量”本身不再是硬瓶颈。

- **Multi-Draw Indirect / ExecuteIndirect（最关键的技术之一）**
  一个（或少数几个）API 调用就能发出成千上万个独立 draw（每个可有不同 vertex/index offset、instance count、PSO 等，取决于实现）。
  CPU 侧几乎只关心“写好间接参数缓冲区 + 发一个 MultiDrawIndirect”，实际 draw 数量对 CPU 开销影响极小。
  这就是“轻松 10 万 draw”的主要来源——它们多数是**间接展开后的逻辑 draw**，不是 10 万次独立的 `DrawIndexed` 调用。

- **GPU-Driven Rendering 管线**
  - 视锥/遮挡剔除、LOD 选择、排序、命令生成全部放到 Compute Shader 做。
  - CPU 只负责踢几个 dispatch + 一两个大的 indirect draw。
  - 再配合 bindless（descriptor indexing）、persistent mapped buffer、GPU culling 后的 compact 等，状态切换和绑定开销也大幅下降。
  - Mesh Shader / meshlet、软件光栅（如 Nanite 一类）进一步把“物体级 draw”变成更细粒度、更高效的提交。

- **其他配套**
  - 实例化、动态合批、材质/PSO 缓存与排序减少切换。
  - 异步计算、多队列、Copy/Compute/Graphics 重叠。
  - 现代引擎（UE、Unity SRP、自研引擎）默认把这些做成默认路径或可选项。

结果就是：逻辑上的“物体/draw 数量”可以很高，但 CPU 真正执行的 API 调用次数、状态验证、锁竞争都很少，所以帧时间里 CPU 开销不明显。

### 2. 为什么 Nsight 显示上万 DC，但 dispatch 只有三四百

这是统计口径差异 + 间接绘制的典型表现：

- **DC（Draw Call）**：工具常把 Multi-Draw Indirect / ExecuteIndirect **展开后的有效 draw 数量**也算进去（或者把 instance/子 draw 累计）。所以你看到上万甚至更多，反映的是 GPU 实际处理的绘制命令规模。
- **Dispatch**（或你看到的“实际提交/顶层调用”）：通常对应：
  - 真正的 API 级 Draw/MultiDraw/Dispatch 调用次数；
  - 或 Command List / Command Buffer 的执行/提交次数；
  - 或 Compute Dispatch + 少量顶层 graphics draw 的总和。

  在 GPU-driven + MDI 管线下，几百个 dispatch（各种 culling、生成、后处理 compute）+ 少数几个大的 MultiDrawIndirect，就能覆盖上万逻辑 draw。因此 Nsight 一边显示“有效 DC 上万”，一边显示“实际 dispatch/API 调用只有三四百”，完全正常。

简单类比：你写了一个循环发出 1 万次 draw 的间接参数，然后只调了一次 `MultiDrawIndirectCount`——工具会报 1 万 DC，但 API stream 里只有 1 次相关调用。

### 3. 实际开发中的注意点

- 即使 API 开销低，**PSO 切换、大资源屏障、过多的 Queue Submit、CPU 端数据准备（上传、排序）** 仍可能成为瓶颈。好的引擎会把这些也压到很低。
- 纯“堆 draw call 数量”不是目标；目标是**有效绘制 + 低 CPU/GPU 浪费**。GPU-driven 的价值在于把剔除做狠 + 提交紧凑。
- 不同工具统计口径略有差异（Nsight Graphics 的事件列表、指标、Range Profiler 等）。看 API Stream / GPU Trace 时重点关注：有多少真正的 `vkCmdDraw*` / `DrawInstanced` / `ExecuteIndirect`，以及 indirect 的 count 是多少。
- 如果 CPU 仍高，优先查：命令录制是否充分多线程、是否有过多同步/锁、描述符更新是否频繁、是否在用旧式绑定路径等。

**总结**：
现代引擎靠 Vulkan/DX12 的低开销 + Multi-Draw Indirect + GPU-driven（剔除与命令生成上 GPU）把“10 万 draw”变成了“CPU 只发几百个实际命令”的问题。Nsight 上万 DC 是展开后的有效绘制统计，dispatch 三四百才是真正的 CPU/API 提交规模，所以两者并不矛盾，反而说明管线已经很高效。

如果你能提供 Nsight 具体是哪个视图/指标（API 事件、GPU Trace、还是某个 counter）以及用的引擎/API，可以更精确地对应到是 MultiDraw 展开还是 Command List 执行次数。