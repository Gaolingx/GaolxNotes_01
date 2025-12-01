# GPU SM Occupancy 解析

### 🧑‍💻 User

gpu的sm occupancy的含义是什么，是什么制约了occupancy的提高，occupancy一定越高越好吗？

---

### 🤖 Assistant

下面给你一个**工程师视角、贴近 kernel 优化实践的 GPU SM Occupancy 解释**，包括：

1. **Occupancy 的准确定义**
2. **是什么限制了 Occupancy**
3. **Occupancy 是否越高越好（非常关键：不一定）**

---

# 1. SM Occupancy 是什么？

最准确的解释：

> **Occupancy = 当前在一个 SM 上实际 resident 的 warp 数 / 该 SM 理论可同时 resident 的最大 warp 数。**

例如（以典型架构为例）：

* 一个 SM 最多可驻留：

  * 64 warps
  * 32 blocks
  * 2048 threads
  * 100K registers
  * 100KB 或更多 shared memory
* 如果你的 kernel 只允许 32 个 warps 常驻，那么 occupancy = 32/64 = 50%

本质上它衡量：

> **SM 中可用于隐藏延迟的并发程度**。

因为 GPU 依赖“warp 切换”来隐藏以下延迟：

* global memory latency (400~800 cycles)
* L2 latency
* texture latency
* pipeline dependency stalls

Occupancy 高 → warp 调度空间大 → 更容易隐藏延迟。

---

# 2. 什么制约了 Occupancy？

GPU 不会让无限多的线程 block 进入同一个 SM，受 **资源** 限制：

---

## (1) Register per thread 限制（最常见瓶颈）

GPU 上寄存器是物理分配的，NVIDIA 是 register file 分成 32-bit slot。

假设：

* 一个 SM 有 **128K** registers
* kernel 每个线程需要 **96 registers**
* block 是 256 threads

则：

* 一个 block 共需 96×256 = 24576 registers
* 最多能容纳 floor(128K/24576) ≈ **5 blocks**

如果上限是 8 blocks/SM，那么你的 register 需求限制了 occupancy。

> **高寄存器消耗是最常见的 occupancy killer。**

---

## (2) Shared Memory per block 限制（第二大瓶颈）

例如：

* SM 有 100 KB shared mem
* 每个 block 用了 48 KB
* 那么同一个 SM 最大只能容纳 floor(100 / 48) = **2 blocks**

许多高性能 kernel（如 FA、layernorm、attention、grouped GEMM）都会因为 shared memory 耗尽导致 occupancy 降低。

---

## (3) Max blocks / warps per SM 的硬件上限

即使不受 shared/register 限制，也不能超过如下：

* 最多 blocks/SM：例如 32
* 最多 warps/SM：64 或 128
* 最多 threads/SM：2048 或 4096

---

## (4) Thread block size 限制（warp 粒度）

block size 过大（比如 1024 threads/block）：

* block 数变少（因为 SM 只能装有限 blocks）
* occupancy 会掉

---

## (5) register spilling / stack frame 导致的 occupancy 降低

如果编译器寄存器不足，会 spill 到 local memory → 需要更多 memory request → 转而需要更高 occupancy 来隐藏 delay → 实际却更低，很难救。

---

# 3. Occupancy 越高越好吗？ **不！关键点在这里。**

---

## “常见误区”：

> **高 occupancy = 高性能**（错）

真实情况：

### ✔ 如果 kernel 被 memory latency 限制（memory-bound）

→ occupancy 必须足够高（典型需 50%–70%+）才能“压住” globalmem latency。

例如：

* 大量随机访问
* 稀疏算子
* gather/scatter
* attention QK matmul 某些路径

此时**occupancy 太低会导致 SM 频繁 stall → 性能灾难级下降。**

---

### ✖ 如果 kernel 是 compute-bound、高 ILP、使用 Tensor Core

→ **occupancy 可以很低，性能依然几乎不变甚至更高。**

典型例子：

* Triton / CUTLASS GEMM
* WGMMA / MMA kernels
* Many ML kernels (conv, attention kernels with pipelining)
* Fully unrolled matmul with cp.async + double-buffering

为什么？
因为 compute-bound kernel 是靠：

* **ILP（指令级并行）**
* **software pipelining**
* **prefetch / async copy**
  来隐藏延迟。

此时过高 occupancy 反而：

* 需要压低 registers/block
* 降低 ILP
* 降低 tile size
* 减低 shared memory double-buffering
* 结果：**性能变更差**

所以 CUTLASS/Triton GEMM 常常只有 25%–40% occupancy 仍然打满 Tensor Core。

---

# 4. 如何判断你需要的占用率？

简单经验法则：

| Kernel 类型                        | 最佳 occupancy             |
| -------------------------------- | ------------------------ |
| **memory-bound kernels**         | 50–100%（越高越好）            |
| **compute-bound kernels (TC)**   | 20–60% 即可                |
| **warp-specialized / pipelined** | 低于 40% 也常见且完全正常          |
| **大量 shared memory 的 kernel**    | 会自然限制在低 occupancy，无需强行提升 |

---

# 5. 实践中如何利用 occupancy 来调参？

### ❌ 不要盲目降低 register 来提高 occupancy

会导致 spill → 性能崩盘。

### ✔ 优化 Kernel tile size

增大 tile（compute-bound）→ 降低 occupancy 但加快速度
减小 tile（memory-bound）→ 提升 occupancy 来隐藏内存延迟

### ✔ 利用软件流水（async copy、双缓冲）

减少对 occupancy 依赖。

### ✔ 保持 warp-level 合理并行

尤其在 Tensor Core / WGMMA / Warp specialized kernel。

---

# **核心总结：**

> **Occupancy 衡量的是 SM 上并发 warp 的数量，主要用于隐藏 latency（尤其是 memory latency）。**

> **决定因素：register、shared memory、block size、硬件上限。**

> **Occupancy 不一定越高越好。对于 compute-bound/tensor-core kernels，降低 occupancy 反而更快。**

换句话说：

> **Occupancy 是用于隐藏延迟的手段，而不是性能目标。**

---

如果你愿意，我可以继续深入讲：

* 如何具体计算 occupancy
* 给你写一个自适应 tile-size → occupancy → 性能预测模型
* comparisons：Triton vs CUDA vs CUTLASS 在 occupancy 上的差异

你想看哪部分？
