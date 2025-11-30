# Bank conflict影响分析

### 🧑‍💻 User

bank conflict产生的原因到底是什么，又为什么会影响性能呢，量化的分析一下，比如2-way bank conflict，造成了什么影响。

---

### 🤖 Assistant

下面给你一个**真正量化、硬件级解释**的版本，专门回答：

* **bank conflict 本质原因是什么**
* **为什么它会降低性能**
* **2-way conflict 数字上究竟损失了什么**

---

# 🔵 1. Bank Conflict 的本质原因

以 GPU（NVIDIA Shared Memory / AMD LDS）为例讲最典型的 bank。

### ● Shared Memory 被分成 N 个 Bank

例如 NVIDIA Turing/Ampere：

* **32 banks**
* 每个 bank 带宽：**4 bytes/cycle**（一个 32bit word）
* 一个 warp（32 threads）一次访问 ideally 可以做到
  **32 并行访问 → 1 cycle 完成**

### ● Bank index = (address / bank_width) % num_banks

典型 bank 映射：

```
bank_id = (addr / 4B) % 32
```

也就是说，只要两个线程访问的目标地址映射到同一个 bank，就会冲突。

---

# 🔵 2. Bank conflict 为什么会导致性能下降？

### 因为：**一个 bank 每 cycle 只能处理 1 个访问请求（除广播情况）**

当 warp 32 个线程同时发出 32 个地址时：

* 如果全部落入不同 bank → **1 cycle 完成（最大带宽）**
* 如果有 k 个线程访问同一个 bank → **hardware 会序列化成 k 次访问**

**冲突不会让访问失败，只会让访问延长 k 倍。**

⚠️ 关键结论：

> **Conflict level = 实际访问所需 cycles / 理想 cycles**

---

# 🔵 3. 量化：什么是 “2-way bank conflict”？

定义：同一个 bank 上同时有 **2 个线程** 发出访问请求。

### ● 举例

假设 32 threads 要访问的地址模式导致：

* 16 个 bank 完全被 thread 踩满（每 bank 2 threads）
* 也就是说 **每个 bank 有 2 个请求**

### ● 对每个 bank：

单 bank 용处理 1 request/cycle
2 个请求 → 2 cycles

### ● 整个 warp 的 latency：

```
理想：1 cycle 读32个word
2-way conflict：2 cycles 读32个word
```

⚠️ 所以：

> **2-way conflict = 带宽减少为原来的一半**

---

# 🔵 4. 更量化的“性能损失”

假设：1 warp shared memory load 通常≈ 20 cycles (典型)

有 bank conflict 以后，会额外乘上冲突倍率：

| Conflict 类型         | 周期  | 带宽损失  | 性能下降               |
| ------------------- | --- | ----- | ------------------ |
| 无冲突                 | 1×  | 100%  | baseline           |
| 2-way               | 2×  | 50%   | throughput ↓ 50%   |
| 4-way               | 4×  | 25%   | throughput ↓ 75%   |
| 8-way               | 8×  | 12.5% | throughput ↓ 87.5% |
| worst-case (32-way) | 32× | 3%    | 32× 慢              |

### 对 kernel 级别：

如果 kernel 有 30% 的时间 spent on shared memory reads：

```
如果 2-way conflict → 那部分变成 2 倍时间
overall latency = 0.3 * 2 + 0.7 = 1.3
→ 30% slowdown
```

---

# 🔵 5. 为什么广播不算冲突？

因为 GPU 有额外逻辑支持：

> 当多个线程访问 **同一地址** → 实现成一个 bank read + 广播数据

所以冲突只发生在：

* 同 bank
* 不同地址

---

# 🔵 6. 直观理解

可以把每个 bank 看成：

* **一条单车道公路**
* 一个线程访问 = 一辆车要过
* 2-way conflict = 两辆车同时想通过 → 必须排队

结果就是：

```
无冲突：32台车32条路 → 同时通过 = 1 单位时间
有冲突：多辆车挤同一条路 → 需要排队 = k 单位时间
```

所以本质就是**带宽从并行变回串行**。

---

# 🔵 7. 总结（最简版）

**1. conflict 原因：多个线程访问映射到同一个 bank（单 bank 每 cycle 只能处理 1 请求）**
**2. 后果：bank 内访问必须序列化 → latency 乘以冲突倍率**
**3. 2-way conflict：吞吐下降到 50%；warp latency 变成 2 倍**
**4. conflict level = (最大银行内线程数) = 延迟倍率**

---

如果你愿意，我可以进一步提供：

* 一个 CUDA 的例子并给出冲突次数分析
* 为什么 TensorCore kernels 很容易产生 bank conflict（特别是 ldmatrix）
* 如何 padding 消除冲突（and why +1 row works）

你想继续哪部分？
