# 优化matmul kernel分块参数

### 🧑‍💻 User

优化matmul kernel的时候，如何根据架构细节推算出理论上能达到峰值的分块参数。需要根据寄存器文件大小，shared memory size/带宽/延迟，各级cache及global memory的带宽/延迟，有一套方法论，确认矩阵乘分块的tile m/n/k及软流水的级数。能不能展开说说？

---

### 🤖 Assistant

好——下面把那套**从寄存器 / shared / 各级 cache / global memory（带宽 & 延迟）出发推导 tile（BM/BN/BK）和软件流水级数（pipeline stages）**的方法论系统化地展开，给出公式、推导步骤、实践启发以及调参 checklist。尽量用可代入数值的**符号化公式**，你可以把实际硬件参数代进去算出具体结果。

我会按这样的结构来讲：

1. 概念清单（要用到的硬件量）
2. 目标与约束（要同时满足的资源 / 性能目标）
3. 逐步推导：容量约束 → 寄存器约束 → 带宽 / 强度 (operational intensity) → pipeline depth（级数）
4. 公式与示例（把步骤合成成可用公式）
5. 实践建议、常见陷阱与调参 checklist

# 1) 概念清单（硬件量 & 符号）

先把变量列出来（符号化，便于代入）：

* 矩阵与 tile：

  * M,N,K：原始 GEMM 维度（C[M×N] = A[M×K] × B[K×N]）
  * BM, BN, BK：block/tile 大小（我们要求解）
* 数据类型：

  * b = bytes/element（例如 FP16 b=2, FP32 b=4, FP8 b=1）
* 计算资源（per-SM / per-GPU，按实际硬件代入）：

  * FLOPS_peak = GPU 的 peak FLOPS（注意混合精度要看 tensor-core throughput & accumulate type）
  * BW_glob = global memory sustained bandwidth（bytes/s）
  * L_glob = global memory latency（cycles 或 ns）—— 用于估算隐藏延迟需要多少并发
* SM 资源（per-SM）：

  * S_sh = shared memory available per SM（bytes）
  * R_reg = register file per SM（number of 32-bit registers OR bytes），或用 R_reg_total（regs）与 reg_size（4 bytes）区分
  * W_max = max warps per SM（hardware limit）
  * blocks_max = max blocks per SM
* 实现/线程映射：

  * TperBlock = threads per block（通常 128/256/512）
  * warps_per_block = TperBlock / 32
* 时序相关：

  * T_compute_tile = cycles spent computing one tile (BM×BN×BK work mapped to tensor cores) — 可以用理论 FLOPS 转换
  * T_load_tile = cycles to load required global data for that tile (taking bandwidth & coalescing into account)
  * L_glob_cycles = latency from issuing global load to it being usable (in cycles)
* 其它：

  * Overhead_shared_bank_conflicts 等暂用为 constant 否则用经验削减有效 BW / capacity。

# 2) 目标与约束（简述）

优化目标通常是：**让 tensor cores / ALUs 在整体上接近峰值 FLOPS**。为此必须满足：

* 每个 SM 上有足够并发（active warps/blocks）来隐藏 memory latency，并保持 tensor cores 有持续 work。
* 每个 block 的 A_sub (BM×BK) 与 B_sub (BK×BN) 以及必要的 C accumulators 能**fit**到 shared memory + registers 中（或合理分配到 shared/reg）。
* 数据搬运（global → shared → regs）和 compute 能够**重叠**（通过 double buffering / software pipeline / async copy）。
* 操作强度（operational intensity I = FLOPs / bytes_from_global）尽量大，以免被 global bandwidth 限制。

因此我们同时要考虑**容量约束**（shared/reg）与**性能约束**（带宽 vs 计算）与**并发约束**（occupancy）。

# 3) 逐步推导（核心步骤）

### A. 容量约束（shared memory + register）

一个 block 处理的 tile 要占用的 shared/reg 大小必须小于硬件允许值，否则不能跑。首先估算一个 block 所需的 shared memory：

* A_tile bytes = BM × BK × b
* B_tile bytes = BK × BN × b
* C_tile bytes (如果在 shared 中缓存 partial C) = BM × BN × b  （注意很多实现让每线程在寄存器里累加，最终写回 global，不一定把 C 放 shared）
* Shared_total_per_block ≈ A_tile + B_tile + (optional) C_tile + overhead (alignment, double buffering)

**约束 1：**

```
A_tile + B_tile + optional_C <= S_sh_alloc_per_block
```

S_sh_alloc_per_block 取决于你能在一个 SM 上同时激活多少 block（blocks_per_sm），也就是：

```
S_sh_alloc_per_block = floor(S_sh_total / blocks_per_sm_desired)
```

常见做法：先选择 1 或 2 blocks_per_sm（以提高 concurrency），计算允许的 S_sh_alloc_per_block。

**寄存器约束**（更容易成为瓶颈）：
每个 thread 使用的寄存器数目 reg_per_thread 必须满足：

```
reg_per_thread × TperBlock <= R_reg_total_per_SM / blocks_per_sm_desired
```

估算 reg_per_thread 来自：

* accumulators per thread (number of C elements that thread keeps in reg)
* fragments of A/B loaded into registers for compute (dependent on warp-level fragment sizes)
* temporaries, indices, pointers

常见策略：决定每线程累加多少 C 元素（call it c_per_thread），那寄存器需求基本与 c_per_thread 成正比。通常希望每线程至少维持数个 C accumulators，以便用寄存器替代 shared 写读，减少 shared pressure。

---

### B. Operational Intensity（算子强度）→ 判断是否带宽受限

用 roofline/算子强度判断是否能利用算力：

* FLOPs per tile (F_tile) = 2 × BM × BN × BK （GEMM 的乘加计数）
* Bytes read from global per tile (D_tile_global) — 关键点：**如果能复用 A_tile 和 B_tile 多次（在 shared 中）**，那么每 block 从 global 读的次数只与 A_tile/B_tile 的大小有关，而不是每次 compute 都读。

最保守的取法（每 tile 只从 global 读一次 A_tile 和 B_tile）：

```
D_tile_global ≈ (BM×BK + BK×BN + (if read C then BM×BN) ) × b
```

（通常 C 只写回一次，读写合并对带宽影响小）

算子强度 I：

```
I = F_tile / D_tile_global   (FLOPs / byte)
```

判断：

* 如果 `I < FLOPS_peak / BW_glob` → memory-bound（global BW 成瓶颈）
* 如果 `I > FLOPS_peak / BW_glob` → compute-bound（或受更近的层如 L2/L1 bandwidth 限制）

可重写为针对 element b：

```
I = 2*BM*BN*BK / ((BM*BK + BK*BN) * b) = 2*BM*BN / ((BM + BN) * b) * (cancel BK)
```

注意 BK 在分子与分母都出现，最后 BK 反而会部分抵消；但 BK 会影响 A/B tile 大小与 shared/reg 占用，从而影响能否成立。

所以要**增大 I**，常用方法：

* 增大 BM,BN（扩大输出 tile），因为 FLOPs scales ~ BM×BN but bytes ~ BM×BK + BK×BN，扩大 BM/BN 更有利（直到 shared/reg限制）
* 选择合适 BK：BK 太小会降低 reuse（但减少 shared），太大会使 A/B 无法放入 shared

---

### C. 并发 / Occupancy（隐藏 latency）

要把 global latency 隐藏住，需要足够的并发工作（active warps / blocks）。衡量：

* active_warps_per_SM = (warps_per_block × blocks_per_sm_active)
* 硬件能同时活跃的 warps 最大值是 W_max

Latency hiding condition（粗略）：

```
active_warps_per_SM × T_compute_tile_per_warp >= L_glob_cycles
```

更常见的做法：确保有足够的 blocks_per_sm（通常 ≥ 2–4）来 overlap memory ops。若每个 tile 计算时间太短（T_compute_tile small），那就需要更多的 pipeline stages/blocks。

**如何估算 T_compute_tile**：

```
T_compute_tile ≈ F_tile / FLOPS_per_SM_effective (in FLOPs/cycle)
```

FLOPS_per_SM_effective 可以用 peak but 要注意 warp-level mapping and instruction throughput (tensor core efficiency).

然后计算需要的并发度 S_needed：

```
S_needed ~ ceil(L_glob_cycles / T_compute_tile) + 1
```

这与 pipeline stages 的想法一致：要 overlapping loads for future tiles while computing current tile.

---

### D. 软件流水（software pipelining / stages）设计

软件流水的目的是重叠多个步骤：global load → L2/L1 → shared copy → compute → write back。典型的设计是 double-buffer 或 triple-buffer：

* **2-stage (double-buffer)**：一组 A/B 在计算，另一组在 prefetch。从 global 到 shared 用 async copy，compute overlap copy。适合当 T_compute_tile ≥ L_copy_time（或接近）时。
* **S-stage pipeline**：把 long latency chain 拆成 S 段（例如 fetch from global, wait L2, copy to shared, compute, writeback），需要 S buffers。`S` 的选择由下面的公式驱动：

若 L_total_cycle 是从发起 global load 到 load 完成可用的总延迟（以 cycles），且 T_compute_tile 是每个 tile 的 compute time（cycles），则最低需要的并行 tiles 数（即 pipeline depth）：

```
S_min = ceil(L_total_cycle / T_compute_tile) + 1
```

解释：如果一个 tile 的 compute 时间可以“覆盖”L_total_cycle 的一部分，那就需要 S_min tiles 同时处于 pipeline 的不同阶段来完全隐藏那段延迟。常见实现里 L_total_cycle 包含 global memory latency + TMA overhead + any memcopy latency。

实践上，很多高性能实现会选择 S = 2 或 3，如果 global latency 很高或 T_compute_tile 很小（tile 小），就需要更高 S（4–8）。但 S 的增加会增加 shared / register 的 buffering 需求（每个 stage 都需要 buffer），因此 S 不能无限大——回到容量约束。

---

# 4) 把它们合起来：可执行的推算流程（step-by-step）

给出一个工程师可以直接执行的流程（把硬件数值代入）：

**步骤 0：收集硬件参数**（代入实际值）

* S_sh_total，R_reg_total（regs），W_max，blocks_max，BW_glob（bytes/s），L_glob_cycles（cycles），FLOPS_peak (FLOPs/s) 或 per-SM FLOPS
* 选择数据类型 b

**步骤 1：选一个 candidate BM×BN 带来的初始强度**

* 选择初始 BK（常见 32/64/128）。选 BK 受 shared/reg 限制。
* 计算 F_tile = 2*BM*BN*BK
* 计算 D_tile = (BM*BK + BK*BN) * b
* 算子强度 I = F_tile / D_tile

**步骤 2：检查带宽约束（是否 memory-bound）**

* 临界强度 I_crit = FLOPS_peak / BW_glob
* 如果 I < I_crit：说明 global BW 将限制性能，需增大 BM/BN 或降精度（减少 b）或增加 BK 的 reuse（注意 BK 影响 shared）
* 如果 I > I_crit：说明可能 compute-bound 或更近层（L2）带宽成为限制 — 继续下步

**步骤 3：共享内存 / 寄存器预算**

* 计算 A_tile + B_tile + optional C_tile <= S_sh_alloc_per_block

  * S_sh_alloc_per_block = floor(S_sh_total / blocks_per_sm_desired)
* 估算 reg_per_thread：首先决定每线程累加多少 C 元素（c_per_thread），以及为加载 A/B fragments 需要多少寄存器。粗略估：

  ```
  reg_per_thread ≈ c_per_thread + a_frag_regs + b_frag_regs + overhead
  ```

  验证 reg_per_thread × TperBlock ≤ R_reg_total_per_SM / blocks_per_sm_desired

**步骤 4：并发 & latency hiding → pipeline depth**

* 估算 T_compute_tile_cycles = F_tile / FLOPS_per_SM_effective
* S_min = ceil(L_total_cycle / T_compute_tile_cycles) + 1
* 选择 blocks_per_sm (≥ S_min / warps_per_block 等)，保证 occupancy：

  * warps_active = warps_per_block × blocks_per_sm
  * warps_active ≤ W_max
* 确认使用 S_min stages 的 buffering（shared/reg）仍满足容量约束（因为每 stage 需要 buffer）

**步骤 5：迭代调节**

* 如果容量/寄存器超限，减小 BM/BN 或 BK 或减少 c_per_thread（寄存器换 shared），或者降低 blocks_per_sm
* 如果 I < I_crit（memory-bound），优先增加 BM 或 BN（扩大输出 tile），或采用更 aggressive reuse（block 划分策略），或改用 lower-precision
* 在每轮调整后，重新计算上面的所有约束，直到找出满足所有约束的最大 BM/BN/BK 与 pipeline S

---

# 5) 公式汇总（便于直接代入计算）

* F_tile = 2 × BM × BN × BK  （FLOPs）

* D_tile_global ≈ (BM×BK + BK×BN) × b  （bytes from global per tile）

* I = F_tile / D_tile_global = 2×BM×BN×BK / ((BM×BK + BK×BN)×b)

* 条件是否 memory-bound:

  ```
  memory_bound if I < FLOPS_peak / BW_glob
  ```

* shared constraint:

  ```
  A_bytes + B_bytes + C_bytes <= floor(S_sh_total / blocks_per_sm)
  ```

* register constraint:

  ```
  reg_per_thread × TperBlock <= floor(R_reg_total / blocks_per_sm)
  ```

* latency hiding / pipeline depth:

  ```
  S_min = ceil(L_total_cycle / T_compute_tile) + 1
  T_compute_tile (cycles) ≈ F_tile / FLOPS_per_SM_effective
  ```

（注：FLOPS_per_SM_effective = FLOPS_peak_per_SM × expected_tensor_core_utilization_factor）

---

# 6) 数值示例（举一个抽象的「代入示例」帮助理解 — 使用假设数值，非特定卡）

假设（示例，仅演示流程）：

* b = 2 bytes (FP16)
* S_sh_total = 256 KB = 262144 bytes
* R_reg_total = 65536 registers (32-bit each)
* BW_glob = 1 TB/s = 1e12 bytes/s （注意：只是演示）
* FLOPS_peak = 50 TFLOPS = 5e13 FLOPs/s（演示）
* Tclock = 1 GHz → 1 cycle = 1 ns
* L_total_cycle (global) = 500 cycles（示意）
* choose BM=128, BN=128, BK=32

  * F_tile = 2 * 128 * 128 * 32 = 1,048,576 FLOPs (~1.05e6)
  * D_tile = (128*32 + 32*128) * 2 = (4096 + 4096) * 2 = 16384 bytes = 16 KB
  * I = 1.048e6 / 16384 ≈ 64 FLOPs/byte
* I_crit = FLOPS_peak / BW_glob = 5e13 / 1e12 = 50 FLOPs/byte

  * 因此 I (64) > I_crit (50) → **不是 global BW 瓶颈**（可能 compute-bound 或 L2限定）
* T_compute_tile ≈ F_tile / FLOPS_per_SM_effective

  * 假设每 SM effective FLOPS 对应到这个 block 的 share 是 500 GFLOPS（示意），T_compute_tile ≈ 1.048e6 / 5e8 = 0.0021 s? Wait 单位不对 —— 用 cycles 更直观：
  * FLOPS_per_cycle_per_SM = 500 GFLOPS / 1 GHz = 500 FLOPs/cycle → T_compute_tile ≈ 1.048e6 / 500 ≈ 2096 cycles
* S_min = ceil(L_total_cycle / T_compute_tile) + 1 = ceil(500 / 2096) + 1 = 1 + 1 = 2 → double buffering 足够
* Shared usage per block:

  * A = 128*32*2 = 8192 bytes (8 KB)
  * B = same = 8 KB
  * total ≈ 16 KB (plus buffers) → S_sh_total 256 KB 支持很多 blocks
* reg estimate: 假设 reg_per_thread ≈ 32 regs, TperBlock = 256 → 8192 regs per block → R_reg_total 65536 可支撑 8 blocks per SM（受 W_max 限制）
* occupancy OK.

示例说明：在这些假设下 (128,128,32) 是可行且 double-buffer 足够。现实中把具体卡的 BW、FLOPS、L_total_cycle、reg total 精确代入即可得到更准确结论。

---

# 7) 实践建议、heuristics 与 checklist（快速上手）

* 优先做**符号化估算**（上面流程）来排掉明显不可能的 tile（例如会超 shared/reg）。
* 常用起点 tile：BM=BN=128，BK=32 / 64（对 FP16/FP8 很常见）。然后按上面公式检查 I、shared/reg、S_min。
* 寄存器优先级：若寄存器成为瓶颈，把部分 C accum 放 shared（代价：更多 shared），或者减小 c_per_thread（每线程累加更少），然后增加 thread 数/warp mapping。
* 软件流水 stages：

  * 如果 T_compute_tile ≫ L_total_cycle，S=2 就够（double buffer）。
  * 如果 T_compute_tile ≲ L_total_cycle，需要 S>2；S 越高用于遮掩越多 latency，但 buffer 成本增高。
  * 实际上，使用 `cp.async`/TMA + double-buffer + 2–3 stage 常能给很大收益。
* 优化 memory path 的对齐与 coalescing：保证 global loads coalesced（A/B 的访问模式按 warp/seg 配置），减少 L2 thrashing。
* 避免 shared bank conflicts：pad shared rows/cols 以避免冲突。
* Autotune：在多个合理 tile 上跑微基准（测 throughput、bandwidth util、SM utilization）并选择最佳。许多实现（CUTLASS）就是这么做的。
* 若做量化（Q4/K 等）：把解压／转 FP16 的开销纳入 D_tile_global（即从 global 读入后需要转换的代价）；最优是把 dequant + matmul + requant 融合，避免多次写回／读出。

---

# 8) 常见误区与陷阱

* **只看共享内存容量，不看寄存器**：寄存器往往更先成为瓶颈，导致编译器插入 spill 到 local memory，性能暴跌。一定先检查 reg_per_thread × threads_per_block。
* **忽视实际 sustained bandwidth 与理论带宽差**：用 sustained BW（benchmark 测到的）而不是标称值。
* **忽视 L2 / L1 带宽 / 争用**：如果多个 block 同时从同一行读取，会产生 L2/L1 冲突，降低实际 reuse。
* **tile 尺寸不是越大越好**：尽管增大 BM/BN 提高算子强度，但会爆掉 shared/reg/occupancy，从而降低 overall throughput。

---

# 9) 如果你愿意，我可以直接帮你做两件事（选其一或都做）

* 给我你目标 GPU 的几个硬件参数（S_sh_total、R_reg_total、BW_glob、FLOPS_peak、L_glob_cycles），我把上面的流程用一个小表格/脚本跑出一组候选 (BM,BN,BK) + 推荐 pipeline stages 并给出每个候选的 I、shared usage、reg usage、S_min。（我会给出具体数值和推荐）
* 或者我直接写一段 pseudo-code / CUDA kernel skeleton（带 double/triple buffering、cp.async 的伪实现结构），把 buffer 布局、warp-to-fragment mapping、寄存器分配建议都写出来，便于你移植到 CUTLASS 或自己实现。

你想先要哪一种？（我不会等待你去做额外事情 —— 如果你把硬件参数贴上来我立刻把数值代入并给出具体候选；如果你不贴，我就给一个通用的 parametric 表格 + kernel skeleton。）


---

### 🧑‍💻 User

根据这套方法论，是否可以套用到其他compute bound/memory bound kernel上？

---

### 🤖 Assistant

**答案是肯定的。**

这套方法论的核心本质是**资源约束下的性能建模（Resource-Constrained Performance Modeling）**，它结合了 **Roofline Model（算力/带宽权衡）** 和 **Little's Law（延迟隐藏）**。这不仅仅适用于矩阵乘法（GEMM），而是适用于几乎所有**追求极致性能的 GPU Kernel**，包括 Compute-bound（计算密集型）和 Memory-bound（访存密集型）。

只要 Kernel 涉及数据的**搬运（Global $\to$ Shared $\to$ Reg）**和**处理（ALU/TensorCore）**，这套逻辑完全通用。

下面我将这套方法论抽象出来，展示如何“套用”到不同类型的 Kernel 上（如 FlashAttention、卷积、LayerNorm/Softmax 等）。

---

### 一、 通用方法论的抽象（Universal Formula）

不论什么 Kernel，优化步骤都可以抽象为求解以下不等式组：

1.  **容量约束（Capacity）：能塞得进吗？**
    $$ \text{SMem\_Usage}(\text{Tile}) + \text{Reg\_Spill\_Risk} \le \text{Hardware\_Limit} $$
    *   决定了 Tile 的上限。
2.  **强度约束（Intensity）：带宽够吃吗？**
    $$ \frac{\text{Work\_per\_Tile}}{\text{Bytes\_Load\_per\_Tile}} \ge \frac{\text{Peak\_FLOPS}}{\text{Peak\_Bandwidth}} $$
    *   对于 **Compute-bound**：通过增大 Tile 或增加 Reuse 来提升左边，直到大于右边。
    *   对于 **Memory-bound**：左边通常固定（无法复用），目标转变为“如何跑满带宽”。
3.  **延迟约束（Latency）：流水线够深吗？**
    $$ \text{Concurrency} \times \text{Work\_Time} \ge \text{Latency} $$
    *   决定了需要的 Active Warps 数量或软件流水（Prefetch）的级数。

---

### 二、 套用到 Compute-Bound Kernel（计算密集型）

这类 Kernel 的特点是有大量的**数据复用（Data Reuse）**。最典型的例子是 **卷积（Convolution）** 和 **FlashAttention**。

#### 1. 卷积 (Convolution / Implicit GEMM)
卷积本质上是特殊的矩阵乘，或者说滑动窗口计算。

*   **Tiling 映射：**
    *   GEMM 的 $M/N/K$ 对应卷积的 $N \times H \times W$（输出像素数）、$C_{out}$（输出通道）、$C_{in} \times K_h \times K_w$（输入体积）。
    *   我们需要确定：一次计算多少个输出像素（$Tile\_HW$），多少个输出通道（$Tile\_C_{out}$）。
*   **套用方法论：**
    *   **容量：** 输入 Feature Map 的 Tile 需要加载到 Shared Memory。如果卷积核较大（如 $7 \times 7$），Halo（重叠区域）会占用大量 Shared，限制了 Tile 大小。
    *   **Reuse：** 增大 $Tile\_C_{out}$ 和 $Tile\_HW$ 可以复用加载进来的 Input Tile。
    *   **结论：** 参数推导逻辑与 GEMM 完全一致，只是 Shared Memory 的计算公式变了（需要包含 Halo 区域）。

#### 2. FlashAttention (SDPA)
这是目前最火的 Compute-bound 优化案例。

*   **Tiling 映射：**
    *   $Q, K, V$ 矩阵。
    *   外层循环切分 $Q$（块大小 $B_r$），内层循环加载 $K, V$（块大小 $B_c$）。
*   **套用方法论：**
    *   **容量约束（最关键）：** 为了避免 HBM 访问，$K, V$ 的 Tile 必须常驻 SRAM（Shared Memory）。
        $$ B_c \times d \times \text{bytes} \le \text{SMem\_Size} $$
        这是决定 $B_c$（KV block size）的硬性条件。
    *   **寄存器约束：** 每个线程需要维护 $O$（Output）和 $l, m$（统计量）的累加器。如果 $d$（head dimension）很大（如 128 或 256），寄存器会溢出，限制了 $B_r$（Query block size）。
    *   **软件流水：** 同样需要 Prefetch 下一块 $K, V$ 到寄存器/Shared，以隐藏 Softmax 和 Matmul 的计算延迟。

---

### 三、 套用到 Memory-Bound Kernel（访存密集型）

这类 Kernel（如 **Element-wise Add, LayerNorm, Softmax, Reduction**）的算术强度（Arithmetic Intensity）通常很低且固定（无法通过 Tiling 增加复用）。

**此时方法论的目标发生变化：从“最大化计算利用率”变为“最大化内存带宽利用率”。**

#### 1. 向量加法 / Activation (Element-wise)
*   **特点：** 读 1 个数，算 1 次，写 1 个数。$I \approx 1/2$ (FLOPs/Byte)，远低于临界点。
*   **套用方法论：**
    *   **容量约束：** Shared Memory 通常不用（直接 Reg to Global），但寄存器限制了**展开因子（Unroll Factor）**。
    *   **延迟约束（核心）：** 为了跑满带宽，必须有足够的 **In-flight Memory Requests**（在途内存请求）。
    *   **推算参数：** 我们不求 $M/N/K$，而是求 **每个线程处理的元素个数 (ItemsPerThread)** 和 **向量化加载宽度 (VectorWidth)**。
        *   公式：
            $$ \text{Active\_Warps} \times \text{ItemsPerThread} \times \text{VectorWidth} \ge \frac{\text{Latency} \times \text{Bandwidth}}{\text{Frequency}} $$
        *   这告诉我们：如果 Latency 很高，而 Active Warps 受限于寄存器数量上不去，你就必须增加 **ItemsPerThread**（即 ILP，Instruction Level Parallelism），让每个线程一次发 4 个、8 个 load 指令，才能填满内存流水线。

#### 2. Reduction / Softmax / LayerNorm
*   **特点：** 需要在 Block 内部进行数据交换（Shuffle 或 Shared Mem Reduction）。
*   **套用方法论：**
    *   **容量约束：** Tile Size（通常是 1D）不能超过 Shared Memory 大小。
    *   **带宽/延迟：**
        *   关键参数是 **Block Size（线程数）** 和 **Grid Size**。
        *   如果是多级 Reduction（Block Reduce -> Grid Reduce），第一级 Kernel 是 Memory-bound。
        *   推算：每个 Block 处理的数据量应足够大，以分摊启动 Overhead，但又要足够小以 fit 进 Shared Memory 进行快速归约。
    *   **Reuse 分析：** 比如 LayerNorm，需要读 $x$，算 mean/var，再读 $x$ 算 norm。
        *   优化点：能否把 $x$ 存在寄存器里复用，避免读两次 Global Memory？
        *   推算：$ \text{Regs\_per\_Thread} \ge \text{Elements\_per\_Thread} $。如果寄存器够，就在寄存器里做；如果不够，只能读两次 Global（退化带宽）。

---

### 四、 总结：如何迁移这套“参数推算”逻辑

当你面对一个新的 Kernel 时，请按以下表格进行“完形填空”：

| 步骤 | Matmul / Compute-bound | Memory-bound (Element-wise/Reduction) |
| :--- | :--- | :--- |
| **1. 确定瓶颈资源** | TensorCore / FMA | Global Memory Bandwidth |
| **2. 核心变量** | Tile Size ($B_M, B_N, B_K$) | ItemsPerThread, VectorWidth, BlockSize |
| **3. 容量约束** | $A_{tile} + B_{tile} \le \text{SMem}$<br>Accumulators $\le \text{Regs}$ | Input Chunk $\le \text{SMem}$ (For reduction)<br>Temp Variables $\le \text{Regs}$ |
| **4. 强度/带宽检查** | $\frac{FLOPS}{Bytes} > \text{Peak\_Ratio}$ ?<br>若否，增大 Tile | 无法改变强度。<br>确保 Load 指令够多以填满 Bus。 |
| **5. 延迟隐藏** | Pipeline Stages (Prefetch)<br>Double Buffering | Unrolling (ILP)<br>Batch Loading (Vectorized Load) |
| **6. 输出参数** | 最佳 $B_M, B_N, B_K$, Stage 数 | 最佳 Grid Size, Block Size, Unroll 因子 |

### 结论
**这套方法论是高性能计算的“物理学”**。
只要你是在通过显式的内存层级（Global/Shared/Reg）来管理数据流动，这套基于**带宽-延迟积（Bandwidth-Delay Product）**和**资源容量（Capacity）**的推算逻辑就永远适用。

区别仅在于：
*   **计算密集型**是在解一个**“如何切块以最大化重用”**的几何题。
*   **访存密集型**是在解一个**“如何调度指令以塞满管道”**的排队论题。