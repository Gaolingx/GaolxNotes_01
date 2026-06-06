# 排查复杂分布式训练组合Bug

### 🧑‍💻 User

说起来 torch.compile+devicemesh+fsdp2(with fwd/bwd prefetch)+activation checkpointing+moe ep的组合万一出bug了到底怎么排查，还是说无法排查？

---

### 🤖 Assistant

这是一个极其硬核的问题。你列出的这个组合（`torch.compile` + `DeviceMesh` + `FSDP2` + `Prefetch` + `Activation Checkpointing` + `MoE Expert Parallelism`）基本上是目前 PyTorch 生态中最复杂、最前沿，也是最容易发生“化学反应”导致崩溃的分布式训练栈。

遇到 Bug 时，**并不是无法排查，而是绝对不能“直接”排查**。如果在这个全量状态下直接看报错堆栈（通常是底层 C++ 报错、CUDA 非同步错误或 NCCL Timeout），你几乎得不到任何有用的信息。

排查这种究极组合的 Bug，核心方法论是：**剥洋葱（控制变量法） + 环境变量显影 + 针对性工具**。

以下是系统性的排查指南：

### 第一步：控制变量，降级定位（最重要的一步）

当你遇到 Bug（无论是显存溢出 OOM、计算图报错、还是死锁 Hang），第一步是**逐个关闭高级特性**，看在哪一步 Bug 消失了。关闭顺序建议如下：

1.  **关闭 `torch.compile`**：把编译关掉，回退到 Eager 模式。如果 Eager 模式报错，说明是逻辑或分布式策略问题，可以直接看 Python 堆栈；如果 Eager 不报错，说明是 Dynamo/Inductor 的图捕获或 AOTAutograd 出了问题。
2.  **关闭 FSDP 的 Prefetch (fwd/bwd)**：异步预取会掩盖通信和计算的依赖关系。关闭后强制同步，看是否是生命周期管理的问题。
3.  **关闭 Activation Checkpointing (AC)**：AC 会修改反向传播的 Autograd 图，且涉及到前向随机数种子（RNG）的保存与恢复。MoE 的路由（Routing）如果有随机性，配合 AC 极易出 Bug。
4.  **降级 MoE EP / 缩小 DeviceMesh**：把 Expert Parallelism 关掉（变成专家副本），或者把多机缩小到单机甚至两张卡，确认是否是跨节点通信或 Mesh 维度切分不匹配导致的问题。

---

### 第二步：利用环境变量“显影”底层错误

如果降级后定位到了具体是哪个模块的冲突，可以通过以下环境变量强行把底层的真实错误暴露出来：

<details>
<summary><b>🛠️ 展开查看关键的 Debug 环境变量</b></summary>

*   **CUDA 异步错误同步化 (捕获真实的报错代码行)**
    ```bash
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_USE_CUDA_DSA=1
    ```
    *由于有 Prefetch 和 FSDP，CUDA 执行是高度异步的。不加这个，报错的 Python 行数永远是错的。*

*   **NCCL 通信死锁/Hang 排查**
    ```bash
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,COLL
    export TORCH_NCCL_BLOCKING_WAIT=1 # 让超时的 NCCL 操作直接抛出异常而不是一直卡住
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    ```

*   **torch.compile 专属日志**
    ```bash
    export TORCH_LOGS="recompiles,graph_breaks,dynamo,aot,inductor"
    ```
    *看看到底是在哪个环节 fallback 到了 Python，或者 AOTAutograd 切分子图时是不是和 FSDP/AC 冲突了。*
</details>

---

### 第三步：针对常见“连环毒药”组合的排查重点

这些高级特性组合在一起时，有一些经典的“坑”。你可以对照排查：

#### 1. `torch.compile` ⚔️ `Activation Checkpointing`
**症状**：编译极其缓慢、不断 Recompile，或者反向传播时显存暴增/报错。
**排查**：
*   PyTorch 的 AC 默认会打断 `torch.compile` 的图（Graph Break）。在较新的 PT2.x 中，必须使用支持编译的 AC API，例如 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`。**千万不要用 `use_reentrant=True`**，它在编译下几乎必挂。
*   检查 MoE 路由的 Dropout 或 Gumbel Softmax，确保它们的 RNG 状态在 AC 重计算时被正确恢复，否则编译出来的反向图会发生梯度不匹配。

#### 2. `FSDP2 (DTensor)` ⚔️ `MoE EP` (基于 DeviceMesh)
**症状**：Loss 不降、张量形状报错，或者 NCCL Hang。
**排查**：
*   FSDP2 和 EP 极度依赖 `DeviceMesh` 的正确配置（通常是一个 2D Mesh，一维做 FSDP，一维做 EP）。
*   **打印 DTensor Placements**：在 MoE 层的输入和输出，显式打印张量的分布状态。
    ```python
    print(f"Tensor placements: {my_tensor.placements}, DeviceMesh: {my_tensor.device_mesh}")
    ```
*   确保从 FSDP 维度进入 EP 维度时，数据分发（All-to-All）前后的张量 Sharding 逻辑（`Shard`, `Replicate`, `Partial`）是数学上正确的。

#### 3. `FSDP Prefetch` ⚔️ `torch.compile`
**症状**：OOM 或者遇到莫名其妙的 C++段错误 (Segfault)。
**排查**：
*   Prefetch 依赖于向 Autograd 引擎注册 Hooks（在反向执行前提前触发 All-Gather）。`torch.compile` (AOTAutograd) 很多时候无法正确追踪这些包含通信的异步 Hooks，导致内存被过早释放或显存双重占用。
*   尝试使用 `torch.compiler.disable()` 装饰器将包含复杂 Prefetch 逻辑的局部代码跳过编译，或者确保使用最新的 PyTorch Nightly 版本（PT 团队一直在修 FSDP+Compile 的兼容性）。

---

### 第四步：终极手段 —— Profiling (性能与内存剖析)

如果代码能跑，但是极其慢、或者跑几十步后悄悄 OOM，你需要使用可视化工具。

<details>
<summary><b>📊 展开查看 Trace 与 Memory Snapshot 技巧</b></summary>

**1. 抓取 Trace timeline 看通信死锁**
使用 `torch.profiler` 抓取 Chrome Trace 格式的文件，用 `chrome://tracing` 或 Perfetto 打开。
重点观察：
*   `nccl:all_gather` 和 `nccl:reduce_scatter` 的时间轴是否对齐？
*   如果有 MoE，`nccl:all_to_all` 是否和计算重叠（Overlap）了，还是说 Prefetch 完全没生效导致了长段的空白等待（Bubble）？

**2. 内存快照 (Memory Snapshot)**
如果是 OOM，不要靠猜，直接上内存快照：
```python
import torch.cuda.memory as mem
mem._record_memory_history(max_entries=100000)
# ... 跑几步训练直到接近 OOM ...
mem._dump_snapshot("memory_snapshot.pickle")
mem._record_memory_history(enabled=False)
```
将生成的 pickle 文件拖入 `https://pytorch.org/memory_viz`。重点看：
*   是不是某几个 Expert 的显存占用异常大（负载不均衡导致 OOM）？
*   FSDP 的 Unsharded Weights 是不是没有在计算后及时释放（通常是 Prefetch + 编译的 Bug）？
</details>

### 总结

这种全套技术栈出 Bug **绝对可以排查**，但它要求你从一个“调包侠”转变为“框架开发者”的心态。

**你的黄金排查路径永远是：**
关闭 `compile` -> 关闭异步/预取 -> 检查单步执行逻辑 -> 检查 DeviceMesh `placements` 切分 -> 逐步开启特性定位冲突源 -> 提交 Issue 给 PyTorch 官方（这种级别的组合遇到框架底层 Bug 的概率极大，附带一个最小可复现脚本(MRE)通常能得到官方的高度关注）。