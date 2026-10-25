# Volta独立线程调度改变了warp执行模型

## 🧑‍💻 User

nvidia volta引入的独立线程调度是什么？

---
## 😀 默认助手 (7)

# NVIDIA Volta 的独立线程调度（Independent Thread Scheduling, ITS）

**一句话总结**：Volta（GV100 / Tesla V100，2017）把 SIMT 执行模型从"整个 warp 共享一个程序计数器和活跃掩码"改为"**每个线程拥有自己的 PC 和调用栈**"，从而让同一 warp 内的线程可以各自独立地前进、让出和重新收敛——这是自 CUDA 诞生以来对 SIMT 模型最根本的一次改动。

---

## 1. 背景：Volta 之前的 SIMT 模型

在 Kepler / Maxwell / Pascal 上：

- 一个 warp 的 32 个线程**共享一个 PC（program counter）**和**一个活跃掩码（active mask）**。
- 代码写成分支时，warp 通过一个"分支同步栈 / 重收敛栈"串行执行各个分支路径：先跑 `if` 分支（掩码只保留部分线程），再跑 `else` 分支，最后在汇合点重收敛。
- 关键后果：**同一时刻整个 warp 只能有一个 PC**。因此一个线程无法在另一个线程之前"越过"某个程序点。

这带来一个经典陷阱：

```cuda
// Volta 之前：这段代码可能死锁
if (threadIdx.x == 0) {
    while (shared_flag == 0) { /* 自旋等待 */ }   // 线程 0 卡在这里
}
__syncthreads();  // 其它线程永远到不了
```

只要自旋的线程和负责设置标志的线程属于**同一个 warp**，由于整 warp 共享 PC，自旋线程无法"让出"，整 warp 就卡死了。

---

## 2. ITS 的机制

Volta 的 SIMT 单元做了如下改动：

| 项目 | Volta 之前 | Volta 及之后 |
|---|---|---|
| 程序计数器 | 每 warp 一个 | **每线程一个** |
| 调用栈 | 每 warp 一个（重收敛栈） | **每线程一个** |
| 分支处理 | 串行化分支路径 + 栈式重收敛 | 线程可独立、交错执行 |
| 让出能力 | 无 | 线程可在等待点**主动 yield** |

硬件在每个周期从 warp 中挑选一组"PC 相同且活跃"的线程组成 SIMT 执行单元，NVIDIA 称之为**收敛组（converged group）**；同时有一个**调度优化器（scheduling optimizer）**负责决定何时让线程让出、何时把分散的线程重新收敛到一起，以尽量维持 SIMT 效率。

因此 ITS 提供三个能力：

1. **细粒度的线程间同步**：warp 内可以做生产者-消费者、锁等模式。
2. **有保证的前向进展（forward progress）**：每个线程都会被调度到，不再有"整 warp 被一个自旋线程拖死"的问题。
3. **子 warp 粒度的重收敛**：分支后不必强制立即全部收敛。

```cuda
// Volta 及之后：可以正常工作
if (threadIdx.x == 0) {
    while (shared_flag == 0) { }   // 线程 0 自旋时，其它线程仍可前进并设置 flag
}
__syncthreads();
```

---

<details>
<summary><b>3. 对编程模型的影响（点开查看，很重要）</b></summary>

### 3.1 不能再假设 "warp 同步执行"

这是最容易被忽视的一点。过去大量代码依赖**隐式 warp 同步**——例如用 volatile 共享内存在同一 warp 内传递数据、或假设 `if (lane < 4) {...}` 之后所有 lane 自动对齐。在 Volta 上**这些假设全部失效**，必须显式同步。

为此 CUDA 9 引入了带掩码的 warp 原语 `_sync` 版本：

```cuda
__syncwarp();                 // 让当前 warp（或指定掩码的线程）重新收敛
__shfl_sync(mask, ...);       // 取代旧的 __shfl
__ballot_sync(mask, pred);    // 取代旧的 __ballot
__any_sync / __all_sync / __activemask()
```

其中 `__syncwarp(mask)` 既是一个同步点，也是一个**重收敛提示**——它告诉硬件"这里需要这些线程对齐"，对性能往往有帮助。旧的 `__shfl`、`__ballot` 等无掩码版本在 Volta 上被弃用/移除（编译会报错或给出错误结果）。

### 3.2 Cooperative Groups（CUDA 9 同步引入）

ITS 让更灵活的同步粒度成为可能，CUDA 9 顺势推出 Cooperative Groups（`cooperative_groups::tiled_partition<32>` 等），用类型安全的方式表达 "一个 tile 内的同步"，而不必手写掩码。

### 3.3 更细粒度的并行算法

- **warp 内锁 / 原子操作**、工作窃取（work stealing）、无锁队列在 warp 级别变得可行。
- **`__activemask()`**：返回当前活跃线程掩码，但它**不是同步原语**，只能用作"当前有多少线程在执行我这一行"的查询，不能保证收敛。

</details>

---

<details>
<summary><b>4. 性能权衡（点开查看）</b></summary>

ITS 不是免费午餐：

- **每个线程独立 PC/栈带来额外状态**，在某些高度发散、频繁切换的代码里，硬件需要更多周期做收敛管理，吞吐可能低于"朴素串行化分支"的旧模型。
- **过度发散会降低 SIMT 效率**：如果 32 个线程各走各的 PC，硬件只能逐个/小批执行，有效吞吐接近标量。
- 因此实践建议仍是：
  - **尽量保持 warp 收敛**（让相邻线程走相同分支）；
  - 在确知需要对齐的地方使用 `__syncwarp()`，它同时充当重收敛提示；
  - 不要因为有了 ITS 就写出大量 warp 内自旋/锁，除非确实需要（例如 `cuda::pipeline`、cooperative groups 的同步场景）。

</details>

---

## 5. 一句话记忆点

> **Volta 之前**：warp 是"一个 32 宽的单指令流"，分支必须串行化。
> **Volta 之后**：warp 是"32 个可以各自跑、可以被调度器临时拼成 SIMT 组的独立线程"——代价是你必须用 `__syncwarp()` 等显式原语来重建你曾经免费获得的 warp 同步。

ITS 此后成为 NVIDIA GPU 的标准特性，Turing、Ampere、Hopper（以及后续架构）都延续了这一模型。
