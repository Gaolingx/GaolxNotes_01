# TMEM张量核心架构

## 🧑‍💻 User

众所周知，nvidia blackwell的gpu hardware层面就是tcgen05 + TMEM + 2SM cluster 范式设计的
我其实现在还没搞明白tcgen05到底是干什么的 和普通的wgmma有什么区别
还有2SM cluster什么情况下用

---
## 😀 默认助手 (7)

我先去核实几个关键细节，尤其是 2SM MMA 的操作数分布和 wgmma 在 sm_100 上的可用性；这些细节很容易记错。

我刚刚从最新的 CUTLASS 文档核对了几个关键数字（1SM 和 2SM 的合法 MMA tile shape、2SM 下每个 CTA 分到的 M、cluster 的 M-mode 必须是偶数等），下面按照“本质、差异、TMEM、2SM 决策”的顺序来讲。

---

# 一、本质差异：MMA 从指令变成异步任务

先给结论：

> **wgmma 是一条 SIMT 风格的“集体指令”，128 个线程一起执行它，结果分散写入 128 个线程的寄存器。**
> **tcgen05.mma 是一条“由单线程发射的异步张量核心任务”：一个线程读一份指令描述符，张量核心自己从 SMEM 或 TMEM 取数据、算完写回 TMEM，完成后通过 mbarrier 通知你。**

这个转变和当年引入 TMA 是同一个设计哲学：**把几十条指令组成的循环，压缩成一次带完成信号的后台作业**。因此理解 tcgen05 时，可以把它当成一个“只做矩阵乘的 DMA 引擎”：

| 维度 | wgmma (Hopper) | tcgen05.mma (Blackwell) |
|---|---|---|
| 发射粒度 | 整个 warpgroup（128 线程）集体执行 | **单个线程**（`elect_one_sync()`） |
| 累加器位置 | **寄存器**（分散在 128 线程） | **TMEM**（专用存储，每 SM 256 KB） |
| 完成通知 | `wgmma.commit_group` / `wait_group`，或 mbarrier | 只能 `tcgen05.commit` → mbarrier |
| 同步语义 | 半同步（指令流内可见） | 完全异步（跨 proxy，需要 fence） |
| A 操作数 | 寄存器 或 SMEM | SMEM 或 TMEM（可选 `collector` 缓存复用） |
| B 操作数 | SMEM | SMEM（必须） |
| 单指令 M | 固定 64 | **64 / 128**（1SM），**128 / 256**（2SM） |
| 单指令 N | ≤ 256 | ≤ 256 |
| 结果读取 | 寄存器里就有 | 必须显式执行 `tcgen05.ld` 搬回寄存器 |
| 吞吐 | 1× | **2×**（legacy）/ **4×**（fp4 窄精度） |
| sm_100 可用性 | ❌（`sm_90a` 专属，架构特性不向前兼容） | ✅ |

最后一行是实际工作中最关键的一点：**从 Hopper 迁移到 Blackwell 会彻底更换 MMA 层，根本谈不上优化**。这一点与当年 Ampere→Hopper 的 `mma.sync`→`wgmma` 不同。

另外，要区分清楚：**RTX 50 系（SM120）根本不是这一套**。它没有 tcgen05、没有 TMEM、cluster 固定为 `1x1x1`、没有 multicast，仍然使用 `mma.sync`（只是增加了 f4、f6 窄精度的 block-scale 支持）。网上讨论“Blackwell 张量核心”时经常把 SM100 和 SM120 混为一谈。

---

# 二、为什么必须有 TMEM：寄存器已经装不下累加器了

这是理解整个设计的核心问题。下面计算一下 Hopper 上的硬性限制：

一个 $64 \times 256$ 的 fp32 累加器需要 $16384$ 个 32 位字。一个 warpgroup 有 128 个线程，因此每个线程需要持有

$$\frac{64 \times 256}{128} = 128 \text{ 个寄存器}$$

寄存器上限是 255，其中 128 个已经被累加器占用。这已经是 `wgmma` 的极限配置（m64n256）。如果你想要一个 $128 \times 256$ 的 tile 来计算：

$$\frac{128 \times 256}{128} = 256 > 255 \quad \text{物理上不可能}$$

**因此，Hopper 上一个 warpgroup 永远无法完成 128×256 的 tile**，这是硬件限制，与代码写法无关。在 Hopper 上，你只能在 64 行这个粒度上工作，依靠 tile 更小来换取更多 SMEM 复用，或者牺牲其他东西。

Blackwell 的解决方案是**把累加器搬出寄存器堆，放入一个专用存储**：

- TMEM 容量：$128 \text{ lanes} \times 512 \text{ columns} \times 4\,\text{B} = 256\ \text{KB/SM}$
- 对比：寄存器堆是 $64\text{K} \times 4\,\text{B} = 256\ \text{KB/SM}$，与 TMEM **容量相同，但完全物理独立**

所以有效累加器容量直接翻倍。$128 \times 256$ 的 fp32 tile 占用 $128 \text{ lane} \times 256 \text{ col}$，正好是 TMEM 的一半，**寄存器占用为零**。

这带来了三个实际收益：

1. **tile 变大**：1SM 就能做 $128 \times 256$（Hopper 上不可能），2SM 能做 $256 \times 256$。MMA 的算术强度更高，SMEM 带宽压力更小。
2. **寄存器被释放给流水线**：Hopper 上著名的“ping-pong 双 warpgroup”结构（一组计算、一组做 epilogue）本质上是为了缓解寄存器压力。Blackwell 上单组 warp 发 MMA、独立 epilogue warps 完全异步地读取 TMEM，warpspecialization 的划分方式发生了变化。
3. **MMA 与 epilogue 可以真正重叠**：张量核心写 TMEM 和 epilogue warps 执行 `tcgen05.ld` 读取 TMEM，由硬件仲裁，不需要软件用两组寄存器轮流切换。

代价是**数据被额外搬运了一次**：以前算完结果就在寄存器里，现在必须执行 `tcgen05.ld`。对于 flash-attention 这种“算完立刻做 softmax 再算下一个”的算子，这意味着 TMEM→寄存器→计算→写回 TMEM 的往返。这也是 FA4 在 Blackwell 上需要重新设计 pipeline 的根本原因。

---

# 三、TMEM 的结构与访问规则

这一部分必须记住，否则代码将无法运行：

- **128 lanes × 512 columns**，每列 4 字节。lane 是行，对应矩阵的 M 维。
- **lane 被切成 4 份，每个 warp 只能访问自己那 32 个 lane**：warp $w$（在 warpgroup 内）访问 lane $[32w,\ 32w+32)$。所以一个 warpgroup（4 warps）刚好覆盖全部 128 lanes，而且要凑满 128 行，**必须由 4 个 warp 协作**。
- **分配粒度 32 列**，必须是 2 的幂，512 列上限。`tcgen05.alloc` 只能由一个 warp 执行，地址会写回 SMEM，然后由所有线程读取。
- 分配、释放和归还许可分别由 `tcgen05.alloc`、`tcgen05.dealloc`、`tcgen05.relinquish_alloc_permit` 完成。**漏掉 dealloc 会导致 CTA 退出时资源未释放**，在 persistent kernel 里就会崩溃。

---

# 四、指令族全景

| 指令 | 作用 | 备注 |
|---|---|---|
| `tcgen05.alloc/dealloc` | TMEM 列分配 | 单 warp 执行，结果写入 SMEM |
| `tcgen05.mma` | 矩阵乘 | 单线程发射，`cta_group::1/2`，`kind::f16/tf32/f8f6f4/mxf8f6f4/mxf4/mxf4nvf4/i8` |
| `tcgen05.mma.sp` | 稀疏版（2:4） | K 维压缩 2× |
| `tcgen05.cp` | **SMEM → TMEM 拷贝** | 把 A 或 block-scale 因子搬进 TMEM；支持 `multicast` 变体 |
| `tcgen05.ld / tcgen05.st` | TMEM ↔ 寄存器 | 形状 `32x32b` / `16x64b` / `16x128b` / `16x256b` 等 |
| `tcgen05.commit` | 把之前所有异步 MMA 的完成状态绑定到 mbarrier | 唯一的完成检测手段 |
| `tcgen05.fence::before/after_thread_sync` | 与 generic proxy 之间的顺序保证 | 在 `__syncthreads` 前后要使用 |
| `tcgen05.wait::ld / ::st` | 等待自己之前的 ld/st 落地 | 不是等待 MMA |
| `tcgen05.shift` | TMEM 内跨 lane 位移 | 少数场景（需要邻居 lane 数据）才有用 |

还有两个容易被忽略但很有价值的**操作数复用机制**：

- **`collector::a` 修饰符**（`.collector::a::fill / use / lastuse / discard`）：在张量核心内部有一个 A 操作数的暂存缓冲。当同一块 A 要乘多个 B 时（例如 attention 中一个 Q block 依次乘多个 KV block），用 `fill` 装一次、后面几十条 MMA 用 `use` 复用，**省掉重复读取 TMEM/SMEM 的带宽**。这是 tcgen05 重要的性能来源，手写 kernel 时务必用上。
- **A 放在 TMEM**：A 一旦进入 TMEM，后续 K-loop 里的多次 MMA 就不消耗 SMEM 带宽。

---

# 五、最小骨架（1SM）

```cpp
__shared__ uint32_t tmem_base_smem;
__shared__ barrier mma_bar;

// ---- 1) 分配 TMEM：只能一个 warp，且全 warp 传同一个 nCols ----
if (warp_id == 0) {
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32
      [tmem_base_smem], /*nCols=*/128;          // 128 列 = 128x128 fp32
}
__syncthreads();
uint32_t tmem = tmem_base_smem;                 // 所有线程都能读

// ---- 2) 单线程发射 MMA（异步，立即返回）----
if (elect_one_sync()) {
  uint32_t idesc = make_idesc(/*dtype=*/F16, /*M=*/128, /*N=*/128, /*transpose...*/);
  tcgen05.mma.cta_group::1.kind::f16
      [tmem],            // D 累加器在 TMEM
      a_desc,            // A：SMEM 描述符（或 TMEM 地址）
      b_desc,            // B：SMEM 描述符
      idesc,
      /*enable-input-d=*/0;   // 0=覆盖，1=累加
}

// ---- 3) 完成只能靠 commit + mbarrier ----
if (elect_one_sync()) {
  tcgen05.commit.cta_group::1.mbarrier::arrive::one
      .shared::cluster.b64 [mma_bar];
}
mbarrier_wait(mma_bar, phase);

// ---- 4) 读回寄存器：4 个 warp 各读 32 lane ----
//     每个 warp 读自己 lane slice 的全部 128 列
float acc[32];
tcgen05.ld.sync.aligned.32x32b.x32.b32
    {acc...}, [tmem + (warp_id_in_wg * 32) * 128];
tcgen05.wait::ld.sync.aligned;                  // 等 ld 完成，注意不是等 MMA

// ---- 5) 释放 ----
__syncthreads();
if (warp_id == 0) {
  tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem, 128;
}
```

需要注意的五个常见问题：

1. **`tcgen05.wait::ld` 等待的是你自己的 `ld`，不是 MMA 完成**。MMA 完成只能通过 `commit` + mbarrier 检测。两者混用会导致读到不完整的结果。
2. **`tcgen05.mma` 和 `tcgen05.ld` 处于不同的内存代理**，跨越 `__syncthreads` 时要使用 `tcgen05.fence::before_thread_sync`，否则编译器或硬件重排会读到不完整的数据。
3. **TMEM 地址布局与 SMEM descriptor 完全不同**，A 放 TMEM 时要用 `tcgen05.cp` 按它要求的 shape（如 `128x256b`）搬入，不能直接手工写入。
4. **`idesc`（instruction descriptor）要手工构造**，32 位里包含 dtype、M、N、转置位、稀疏位、block-scale 位等信息。这里一旦写错，错误不会以非法指令的形式报出来，而是直接得到错误结果。
5. **N 必须是 8 的倍数**，M 只有 64、128 可选。不要把任意 tile 尺寸填进去。

---

# 六、2SM（`cta_group::2`）的准确含义

先澄清一个常见误解：

> **1SM 的 `tcgen05.mma` 已经能打满单个 SM 的张量核心吞吐。2SM 提升的对象是数据搬运和发射开销，而非算力。**

这是关键——Blackwell 文档明确写着 1SM 和 2SM 的指令吞吐都是“Hopper 的 2×”。2SM 是把**两个 SM 的张量核心合成一个逻辑 MMA 单元**，让一条指令完成两倍的活。

## 切分方式（已核对）

根据 CUTLASS 的 tile shape 表，**只有 M 被切开**：

| MMA tile (pair) | 每 SM 实际 tile |
|---|---|
| `2SM 128 x N` | `64 x N` |
| `2SM 256 x N` | `128 x N` |

也就是说：

- **A 按 M 切分**：CTA0 持 $M/2$ 行，CTA1 持另一半。
- **累加器按 M 切分**：每个 SM 的 TMEM 存 $M/2 \times N$。$256 \times 256$ 时每个 SM 占 128 lanes × 256 cols，正好用掉 TMEM 一半。
- **B 不切**：两个 CTA 需要同一份 B。所以**实践中用 TMA multicast（`cp.async.bulk.tensor` + `.multicast::cluster`）从 L2 一次读出，投递到两个 CTA 的 SMEM**。节省的是 L2/DRAM 流量，而不是 SMEM 占用。

所以 2SM 的收益可以精确表述为：**B 的全局内存流量按输出摊薄一半，MMA 指令条数减半（前端和发射开销减半），A 的 SMEM 按 CTA 均摊，因此在同样的 SMEM 预算下可以开设更多 pipeline stage。**

## 强制要求

- 必须是**同一 cluster 内按 M 方向相邻的一对 CTA**（rank $2i$ / $2i+1$）。CUTLASS 原文：*"ClusterShape_MNK 的第一个 mode 必须是 2 的倍数"*。
- 由 **leader（偶数 rank）发射**整条指令，两个 CTA 的 descriptor、TMEM 分配、mbarrier 必须严格对称。
- 必须在 kernel 启动时声明 cluster（`__cluster_dims__(2,1,1)` 或 `cudaLaunchKernelEx` 的 cluster attribute）。
- 两个 CTA **共用一份控制流**（同一份 warp-specialization 结构），想实现分支不对称几乎做不到。

---

# 七、2SM 的决策清单

<details>
<summary><b>✅ 该用 2SM 的场景</b></summary>

**1. 大 GEMM（训练前向/反向、大 batch prefill）**

M、N 都 ≥ 256，用 $256 \times 256$ 的 MMA tile。适用场景是：
- B 的 L2 → SMEM 流量被两个 CTA 的输出摊薄，节省一半；
- 每条 MMA 指令的搬运工作量翻倍，K-loop 里的 TensorCore 占用率提高；
- 在 Blackwell 上，$256 \times 256$ 是唯一能把 N=256 与 M=256 结合起来、同时获得两者收益的配置。1SM 无法实现。

**2. Attention prefill 中，Q 方向能拼到 ≥ 256**

若将 batch × heads 合并后 Q 的 M 维能够拼到 256，两个 CTA 可以共享同一份 K/V tile；而 K/V 在 attention 中就是那个 B 操作数，本来就需要复用。这是 2SM 收益最大的场景。

**3. 你希望增加 SMEM stage 数来提高 pipeline 深度**

2SM 下 A 的 SMEM 按 CTA 均摊，因此在同样的 228 KB 预算下能容纳更多 stage。当 K-loop 的 SMEM 带宽或延迟已经成为瓶颈时，这通常是收益最大的一点。

**4. FP4/FP8 大 GEMM**

窄精度路径的 MMA tile 更大（$256 \times 256 \times 256$），吞吐是 Hopper fp8 的 4×。这里 2SM 几乎是默认配置。

</details>

<details>
<summary><b>❌ 不该用 2SM 的场景</b></summary>

**1. Decode / GEMV / 小 M**

当 per-token 生成时，M = 1~16，$M/2 = 8$ 甚至更小。此时 2SM 的协同（cluster sync、对称 barrier）和 cluster launch 开销完全得不偿失。**这种场景 1SM，甚至不用 tcgen05 可能更好**（它终究是带宽瓶颈）。

**2. 问题尺寸导致 cluster 尾部碎片**

cluster=2 意味着独立的“MMA 单元”数量直接减半。当 SM 数不是 2 的倍数，或者 M/K 的尾部很碎时，wave quantization 带来的损失会放大。先计算一下你的 tile 数量。

**3. MoE grouped GEMM 中 expert 特别小**

当单个 expert 的 M 无法凑到 256，或者各 expert 大小差异悬殊时，强行使用 2SM 只会把一对 CTA 一起拖慢。此时应使用 1SM + 动态 tile 调度。

**4. 需要 CTA 之间负载不对称**

例如 ragged attention（变长序列），不同 CTA 要做不同数量的 KV block。当 cluster 绑定后，两个 CTA 会被 `cluster.sync` 相互限制，无法提前结束工作。

**5. 你正在使用 persistent kernel 配合 work stealing**

两个 CTA 必须处理同一对 tile，调度自由度大幅降低。persistent + 2SM 的组合存在，但 tile 的分配逻辑要改写。

</details>

## 一个实用的判断方法

先看下面这个式子，它决定 B 流量是否成为瓶颈：

$$\text{每个输出的 B 字节数} = \frac{K \cdot \text{sizeof}(B)}{M_{\text{tile}}}$$

如果你的 $M_{\text{tile}} = 128$（1SM 上限），分子是 $K \times 2$ 字节。将 $M_{\text{tile}}$ 提高到 256（2SM）**会把这一项直接减半**。当它和 L2 带宽处于同一量级时，2SM 的收益是确定的；当 DRAM/L2 明显不是瓶颈（小矩阵、计算受限严重、或已经靠大 N 摊销掉）时，2SM 就只能带来开销。

---

# 八、cluster 的其他用途（不要只把它和 2SM MMA 绑定）

cluster 在 Hopper 上就已存在，在 Blackwell 上它仍然是独立的工具：

- **TMA multicast**：一份 L2 读取数据，同时投递到 cluster 内多个 CTA 的 SMEM。即使不用 2SM MMA，做 $2 \times 2$ 或 $4 \times 4$ 的 GEMM cluster 也能减少一半的 A 或 B 流量。
- **DSMEM（distributed shared memory）**：直接读取其他 CTA 的 SMEM，用于 epilogue 的跨 CTA 规约（例如 split-K 归约、layernorm 合并规约），比走 HBM 快一个数量级。
- **`cluster.sync()` / cluster barrier**：用于细粒度的跨 CTA 同步。

所以，“用不用 cluster”和“用不用 2SM MMA”是两个正交的决策。很多场景应该启用 cluster（为了 multicast 和 DSMEM），但 MMA 仍然使用 `cta_group::1`。

---

# 九、实践路径

不建议手写 PTX。建议按以下顺序推进：

1. **先使用 CuTe DSL（`cutlass.cute`）**：它提供 `tcgen05_mma`、`tcgen05_copy`、`tcgen05_ld` 等原语，并自动管理 TMEM 分配和 barrier，适合先验证算法结构。
2. **再使用 CUTLASS 3.x collective builder**：`KernelTmaWarpSpecialized{1,2}SmSm100` 系列策略加上 `MmaTileShape_MNK`，基本上只要选对 dtype/layout，就能跑出接近最优的性能。
3. **只有在你需要修改 pipeline 结构（例如 FA4 那种 custom 调度）时才下沉到 PTX**。

验证矩阵：先用 `cute::SM100_MMA_F16BF16_SS` 之类的 atom 做正确性对照，再逐步替换。

<details>
<summary><b>如果你是从 Hopper 代码移植，注意以下三点必须重写</b></summary>

1. **MMA 层整个更换**：`wgmma` 在 sm_100 上不存在，它属于 `sm_90a` 架构特性，不会向前兼容。
2. **累加器生命周期整个更换**：过去累加器在寄存器里意味着“算完就能用”，现在算完必须经过 `tcgen05.ld` 读取，而且必须由 4 个 warp 按 lane slice 协作读取。原有 epilogue 的线程映射（thread-to-element mapping）需要全部更换。
3. **同步机制整个更换**：`wgmma.wait_group` 变为 `tcgen05.commit` + mbarrier，并且引入了跨 proxy 的 fence 语义。以前能正常工作的 `__syncthreads()` 现在可能不够。

</details>

---

**如果需要我继续深入某一部分**，例如：
- 手写一个完整的 `tcgen05` microkernel，包含 `idesc` 的位域构造过程；
- 或者分析你当前场景（给出 M/N/K 和 dtype、瓶颈所在）应该用 1SM 还是 2SM、tile 如何选择；
- 或者讲解 `collector::a` 与 A-in-TMEM 在具体 kernel（例如 flash attention）中如何编排。

请告诉我，我会进一步展开。

---
## 🧑‍💻 User

感觉对比hopper主要就是独立的tmem吧
可以解决一下寄存器不够用的问题
独立tmem有没有必要搞，这个东西是存疑的，但是至少独立矩阵累加寄存器，现在看起来是必选项了
尤其是CD矩阵很大的实现，当然这种实现在GPU上应该不多

一方面是之前的tensor core他acc buffer 设计在tensor core里面了
并且了里面的操作靠硬件实现
把tmem单独做出来，提供更加细粒度的指令 更好做管理
比如说如果中间结果没有存在acc buffer里面，他可能就得把结果写回寄存器里面，这个好像是不确定的，取决于此时执行的指令顺序，acc buffer里面的中间结果可能被evict掉，就非常考验指令编排？不懂因为专利没说
hopper专利说存在acc buffer一样的东西，减少对RF的写回
但这东西并不受指令控制
所以我觉得应该是硬件控制的，执行顺序决定

---
## 😀 默认助手 (7)

你的直觉大体成型了，但我想把两处调整得更精确：一处比你说的更强，一处比你说的更弱。

---

## 一、必要的不是“独立”，而是“ISA 可见 + 软件可分配”

你说“独立矩阵累加寄存器现在看起来是必选项”。这点我同意，但我认为真正的分界线不在“物理上独立”，而在于：**累加器有没有自己的地址空间，由软件显式分配。**

这个区分很关键，因为它解释了 TCgen05 到底解决了什么：

- Hopper 上累加器是**寄存器操作数**（`wgmma` 的 D 就是寄存器）。它不是“没地方放”，而是**它的存在与否、驻留与否，都不在软件手里**——你只看见一串寄存器名，看不见它此刻在哪个物理结构里。
- Blackwell 上 `tcgen05.alloc` 给你 128–512 列，你拿着这个地址，**它就在那里**，不会被换出去，也不会因为后面插了一条别的指令而消失。

一句话：**Hopper 的累加器是“名字”，Blackwell 的累加器是“资源”。**

软件可分配的累加器带来两个 Hopper 做不到的能力，都跟容量无关：

1. **同时有两个活着的累加器**，才能让 MMA 写 A 的同时，epilogue 读 B。Hopper 只能靠两组 warpgroup 轮流——因为只有一套寄存器名字可分。
2. **自己决定驻留期**。K-loop 多长、什么时候交给 epilogue、要不要和另一个 tile 交错——这些都是静态可算的。

---

## 二、TMEM 的动机不止是容量，主要是带宽

这一处我要把你说的加强。**“寄存器不够用”其实只是边际问题，真正决定性的是端口带宽。**

<details>
<summary><b>算一下（点击展开）</b></summary>

**H100 SXM（FP16 dense 989 TFLOPS，132 SM，~1.98 GHz）**

每个 SM 每 cycle 的 MAC 数：

$$\frac{989\times10^{12}}{132 \times 1.98\times10^{9} \times 2} \approx 1893\ \text{MAC/cycle}$$

`m64n256k16` 一条指令 $= 262144$ MAC，占 $\approx 138$ cycle。
累加器 $= 16384$ 个字，于是累加器流量：

$$\frac{16384}{138} \approx 118\ \text{word/cycle} = 473\ \text{B/cycle (只写)}$$

读改写算一遍就是 $\approx 946\ \text{B/cycle}$。

**B200（FP16 dense ~2.25 PFLOPS，~148 SM，~1.9 GHz）**

$$\frac{2.25\times10^{15}}{148\times1.9\times10^{9}\times2} \approx 4000\ \text{MAC/cycle}$$

1SM 的 $128\times256\times16$：$524288$ MAC → $131$ cycle，累加器 $32768$ 字 →

$$\frac{32768}{131} \approx 250\ \text{word/cycle} = 1000\ \text{B/cycle (只写)}$$

读改写 $\approx 2\ \text{KB/cycle}$。（2SM 的 $256\times256$ 按 SM 摊开是同一个数。）

</details>

一个现代 SM 的寄存器堆读带宽量级是 512 B/cycle 上下，写端口通常还更窄。**Blackwell 光累加器 RMW 就要 2 KB/cycle，比整个 RF 端口高一个量级。**

结论：**无论 Hopper 还是 Blackwell，累加器都不可能每指令都真的走一遍通用寄存器堆。它必然有一条专用的宽通路。** TCgen05 的贡献在于**把这条通路做成可寻址的**，并非“新造了一条通路”。

这正好接上你说的第二点。

---

## 三、关于 Hopper 内部 acc buffer：你的模型我认为是对的

先按可信度分层：

**可以直接确认（ISA 层面）**
- `wgmma` 的 D 是寄存器操作数，PTX 文档如实这么写。
- Hopper 上没有任何指令能查询/分配累加器驻留空间。
- Blackwell 上 `collector::a` 修饰符（`fill` / `use` / `lastuse` / `discard`）**公开暴露了一个硬件暂存缓冲的存在**——这是“内部 buffer 确实存在、且 NVIDIA 选择只暴露一部分”的直接证据。

**上面的定量分析支持的推断**
- 那么宽的累加器流量，不可能走通用 RF。所以 `wgmma` 的“寄存器累加器”在实际硅片里大概率不是一个朴素的 RF 读改写，而是 **“RF 里的名字 + 内部宽缓冲的实际驻留 + 在必须可见时才落回 RF”**。你说的“减少对 RF 的写回”，方向应该就是这个。

**我无法确认（专利细节）**
- 这个缓冲的容量、替换策略、以及“被 evict 的代价”到底怎么算。你没查到是正常的。

所以你那句 **“这东西并不受指令控制，应该是硬件控制的，执行顺序决定”**——我完全同意，而且我认为这正是 NVIDIA 换代的核心动机。

---

## 四、为什么必须从“硬件管”翻成“软件管”

因为硬件管理在这个位置上是**不可推理的**：

- 你看不见容量，所以不知道能不能同时驻留两个累加器。
- 你看不见替换，所以一段 microbenchmark 好、真实 kernel 垮了，你没法归因。
- 你看不见代价，所以 patch 一条指令就可能掉 20% 而你查不出来。

对一个要被编译器和 kernel 工程师批量调优的结构，这是致命的。**可预测性本身就是功能。**

这里有个很漂亮的不对称，值得单独说：

| | A 操作数 | 累加器 |
|---|---|---|
| 生命周期 | 相邻几条 MMA（指令窗口内） | 整个 mainloop + 跨 warp handoff |
| 复用模式 | 局部、规则 | 长驻、跨阶段 |
| 管理方式 | **硬件管 + hint**（`collector::a`） | **软件显式 alloc/dealloc** |

**寿命长度决定管理方式。** A 的复用在紧邻的指令窗口里，硬件用一个小 hint 就能追踪；累加器要活过几百条指令、还要交给另一组 warps，隐式机制必然失效。所以 NVIDIA 保留了 A collector 的隐式管理，只把累加器显式化了——这是个很克制的设计选择，不是无差别地把东西都暴露出来。

---

## 五、TMEM 的代价（公平起见，你该知道它亏在哪）

它并不是纯收益，代价很具体：

1. **形状刚性是为效率付的价。** 128 lanes 对齐单 SM 最大 M，32-lane 切分对齐 warp，4 B 元素。一旦 epilogue 想要转置视图、按行求和、或者跨 lane 取邻居，就得走 `tcgen05.ld` 回 RF 再算——这就是 FA 的 softmax 要在 TMEM↔RF 之间来回的原因，也是 `tcgen05.shift` 这种“补丁指令”存在的原因。
2. **epilogue 多了一次永久性搬运税。** Hopper 上算完就在寄存器里，现在必须显式读回。
3. **TMEM 变成了第三类 occupancy 资源。** 以前算 occupancy 只看寄存器 + SMEM，现在还要看 TMEM 列数。512 列/SM 的预算会实实在在地限制你开几个 CTA。这是 kernel 工程师的新负担。
4. **跨 proxy 的顺序问题。** `tcgen05.fence::before/after_thread_sync` 这类东西，是从“半同步指令”变成“异步任务”必然要付的复杂度代价。

---

## 六、关于“CD 矩阵很大的实现在 GPU 上不多”

这条我要温和地反对一下。

GEMM mainloop 里累加器工作集就是 $M_{tile}\times N_{tile}\times 4$ B。哪怕只是 $128\times256$，也是 **128 KB**——**基本等于一整个 warpgroup 的寄存器预算（128 线程 × 255 × 4 B ≈ 130 KB）**。也就是说，**“C/D 很大”是 GEMM 的常态，而非特例**；并且它恰好卡在寄存器预算的边界上（$128\times256$ 需要 256 reg/线程，物理上直接超了 255）。

Attention 更狠：$S = QK^T$ 的累加器要活过整个 KV loop。

所以这个判断不该是“少数人才需要大 C/D”，而是：**任何正常的 tile-based GEMM/attention，都需要一个在容量和带宽上都超出单组寄存器能力的累加器。** 这也是为什么这条设计从 Ampere 的隐式缓冲一路走到 Blackwell 的显式 TMEM——需求一直在，只是终于被明面化了。

---

## 七、我认为真正开放的架构问题

不是“要不要独立累加器”（这个已经定了），而是：

> **TMEM 会不会重演 SMEM 的历史——从“专用且形状苛刻”演变成“更通用”？**

SMEM 当年也是为特定数据流服务的，后来长成了带 bank、带 TMA、带 DSMEM 的通用可编程 scratchpad。TMEM 目前只有一个形状、一个访问模型、一个用途。如果未来 tile 继续变大、窄精度继续吃掉带宽、或者编译器开始自动做 TMEM 分配，那它往通用化走的压力会很大。

反方向的压力也存在：TMEM 一旦成为 occupancy 资源，它就在和 SMEM 抢同一块面积预算。

两三年内如果看到 `tcgen05` 之外的第二套 TMEM 访问原语（更宽的 ld 形状、可跨 lane 的自由寻址、甚至 TMEM 之间的拷贝），那就说明它已经走上 SMEM 的路了。

---

要不要我接着往下算一个东西：**给定你的实际 kernel（M/N/K、dtype、目标 GPU），列一张 1SM vs 2SM 的 TMEM 占用与 stage 数对照表**？那个能直接把“该不该上 2SM”从定性判断变成查表。
