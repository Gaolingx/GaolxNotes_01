# Marlin内核针对Ampere架构已达极致，但因架构根本差异难以迁移至Hopper/Blackwell

## 🧑‍💻 User

marlin kernel那种应该是已经基本把ampere特性榨干了，但到了hopper/Blackwell可能又不是最优了，阅读cutlass和marlinkernel原版repo试分析原因

---
## 😀 默认助手 (7)

我先确认可用的抓取和浏览工具，然后实际去读 CUTLASS 与 Marlin 的源码。

抓取工具可用。我先摸清仓库结构再读关键文件。

README 里已有一句话直接点题：*"compute capability >= 8.0 (Ampere or Ada, Marlin is not yet optimized for Hopper)"*。现在读内核本体。

Marlin 内核已完整读到，包括 README 里那句“not yet optimized for Hopper”。接下来核对 CUTLASS 侧 Hopper/Blackwell 的硬件约束。

持久化文件是单行巨型文本，无法分页。改用 Range 请求分段读文档。

Range 分段有效。继续读 Blackwell 文档中段，以及 Hopper wgmma 的头文件约束。

核心约束已经拿到（Blackwell 的 `tcgen05.mma` 中 A/B 都是描述符，C 在 TMEM，由单线程 `elect_one_sync()` 发出）。再看同组为 Hopper 另写的 Machete 作为对照证据。

读完两个仓库的关键路径，可以给出一个有源码依据的结论。下面是分析。

---

# 结论先行

Marlin 的性能来自它把 **Ampere 的整条数据通路**（`mma.sync` 寄存器操作数 + `ldmatrix` + `cp.async` + 手工 XOR swizzle + `__syncthreads` 同步）在软件上做到了极限；但它把四个 Ampere 时代的"硬前提"写进了 kernel 结构本身：

$$\text{① 张量核接受寄存器操作数}\quad\text{② } mma.sync \text{ 是满速通路}\quad\text{③ 去量化在寄存器里做}\quad\text{④ 同步靠 }\texttt{\_\_syncthreads()}$$

Hopper 破坏了 ②③，Blackwell 连 ①④ 一起废掉。所以 Marlin 上 Hopper/Blackwell 并非"调参不够好"，而是**它的指令级基座被移除了**。

---

## 一、Marlin 的 Ampere 假设（源码证据）

`marlin/marlin_cuda_kernel.cu` 里这四件事是绑死在一起的：

| 设计点 | 源码 | 为什么 Ampere 上是对的 |
|---|---|---|
| 计算原语 | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | Ampere 上这是**唯一**的满速 TC 通路 |
| B 操作数 | `dequant(int q)` → `FragB`（寄存器里的 `half2`） | 权重从不以 fp16 形式落在 SMEM |
| A 操作数 | `ldsm4()` (`ldmatrix.x4`) → 寄存器，整个 k-loop 常驻 | 复用寄存器里的 A，省 SMEM 带宽 |
| 去量化 | `lop3` ×2 + `__hsub2` + `__hfma2` + `__hmul2` | 向量管线有余量（见第四节） |
| 搬运 | `cp.async.cg.shared.global` 16B/线程 + `createpolicy...L2::evict_first` | 无 TMA，手工 16B 对齐拷贝就是最优 |
| 同步 | `cp_async_wait<N>()` + 每 2 个 k-step 一次 `__syncthreads()` | 单 warp-group 内融合循环，够用 |
| 资源 | `const int STAGES = 4; const int SHARED_MEM = 96*1024;` | 注释明写 "max shared memory on compute capability 8.6" |
| 归约 | `barrier_acquire/release` 全局锁 + `ld.global.acquire.gpu` 自旋 | striped partition 把跨 CTA 归约次数压到最少 |

README 自己就写着：

> "NVIDIA GPU with compute capability >= 8.0 (Ampere or Ada, **Marlin is not yet optimized for Hopper**)"

注意它说"还没针对 Hopper 优化"，实际比这更结构性——**它不是没优化，而是这条路径在 Hopper 上到不了顶。**

---

## 二、Hopper：TC 峰值翻倍了，但只对 `wgmma` 开放

### 2.1 算力翻倍发生在"操作数投递"路径上，不在 `mma.sync` 上

Hopper 每 SM 的 fp16 TC 吞吐约 $2\times$ A100（~2048 vs 1024 MAC/clk）。这个翻倍不是因为 TC 阵列变宽，而是因为 **`wgmma` 让 TC 直接从 SMEM 读操作数、绕过寄存器堆**，从而把操作数投递带宽翻倍。

而 `mma.sync` 每个指令都要从 RF 读 A/B 片段，投递带宽与 Ampere 同级。所以在 Hopper 上：

$$\text{用 } mma.sync \text{ 的理论上限} \approx \tfrac{1}{2}\ \text{TC 峰值}$$

实践中常在峰值的 50–65%。Marlin 正好卡在这条天花板上——**它连"把 SM 喂饱"都不可能，谈何榨干。**

### 2.2 想换 `wgmma`，B 就再也进不了寄存器

这是最致命的一条。看 `cutlass` 的 `include/cute/arch/mma_sm90_gmma.hpp`：

```cpp
// A 可以在寄存器里 —— 但有硬约束
struct MMA_64x8x16_F16F16F16_RS {
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];        // ← B 永远是描述符，永远在 SMEM
  static_assert(tnspA == GMMA::Major::K, // ← A 还必须是 K-major
                "Register source operand A must have K major layout.");
```

- `_SS` 变体：A、B 都是 `desc_a`/`desc_b` → 都在 SMEM。
- `_RS` 变体：A 可进寄存器（且只能 K-major），但 **B 无寄存器形态**。
- 形状固定 $M=64$（一个 warp-group），$N\in[8,256]$ 步长 8，$K=16$。

Marlin 的核心技巧恰恰是"**把 int4 直接去量化成 B 片段塞进寄存器**"。换成 `wgmma` 意味着：

1. 去量化结果必须先**写回 SMEM**（每权重 2 byte 写流量 + SMEM 容量占用，与 K-pipeline 争地盘）；
2. SMEM 布局要改成 `wgmma` 的 core-matrix + swizzle 描述符格式——Marlin 那套 `transform_a` 的 XOR 布局是**为 `ldmatrix` 的 bank-conflict-free 设计**的，与描述符 swizzle 不兼容，必须重写；
3. $M=64$ 的固定粒度对上 Marlin 的 16 行粒度（`thread_m_blocks=1` → 16×128 tile）。

也就是说：**用 `wgmma` 就必须放弃 Marlin 最锋利的那把刀。** 这不是调优问题，是重写问题。（作者团队后来确实另起了 Hopper 专用的 wgmma kernel，而不是把 Marlin 移植过去——这本身就是最强的实证。）

### 2.3 其余 Ampere 痕迹在 Hopper 上变成纯 overhead

<details>
<summary><b>展开：同步、TMA、SMEM 容量、集群五处退化</b></summary>

- **同步范式落后一代**：CUTLASS `media/docs/cpp/pipeline.md` 描述的 Hopper 范式是 *producer/consumer + `mbarrier` + `PipelineTmaAsync`*：TMA 自己更新 barrier、warp specialization 让 producer 与 consumer 完全解耦。Marlin 是**单一融合循环**，全体 256 线程既 fetch 又 compute，每 2 个 k-step 一次 `__syncthreads()` —— 这个屏障会把 TC 和拷贝强制串成同一个节奏，任何 warp specialization 都插不进去。
- **没有 TMA**：`cp.async` 16B/线程需要 8 warp 全部参与发射拷贝指令，占用 issue slot；TMA 是单线程发一条描述符就完成整个 tile（含 swizzle、越界填充、cluster multicast）。Hopper 上这直接省掉一整个 warp 的工作量。
- **SMEM 只用一半**：硬编码 `96*1024` + `STAGES=4`（为 A10 的 100KB 调的）。H100 可用 227KB，本可以 8+ 级流水，Marlin 浪费掉。
- **没有 thread block cluster**：H100 的 DSMEM 可以在一组 CTA 间共享/广播 A tile，Marlin 靠"所有 CTA 各自从 L2 捞 A"来打，这在 A100 上很聪明，在 Hopper 上是次优解。
- **归约方式**：全局自旋锁 + `ld.global.acquire.gpu` + fp16 降精度部分和。Hopper/Blackwell 上更该用 2-SM MMA / cluster 内归约或 fp32 split-K workspace。

</details>

---

## 三、Blackwell：`tcgen05.mma` 把 Marlin 的范式从根上删掉了

### 3.1 没有寄存器操作数了

`include/cute/arch/mma_sm100_umma.hpp`：

```cpp
struct SM100_MMA_F16BF16_SS {
  using ARegisters = uint64_t[1];   // 描述符（SMEM）
  using BRegisters = uint64_t[1];   // 描述符（SMEM）
  using CRegisters = uint32_t[1];   // ← TMEM 地址，不是寄存器！
  fma(desc_a, desc_b, tmem_c, scaleC, idescE) {
    if (cute::elect_one_sync()) {   // ← 单个线程发射整条 MMA
      asm("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, ...");
```

关键点，逐条对撞 Marlin：

| Blackwell 事实 | 对 Marlin 的后果 |
|---|---|
| A/B **只能是 SMEM 描述符或 TMEM**（`SM100_MMA_F16BF16_TS` 是 A-from-TMEM） | "去量化到寄存器直接喂 TC" 的路径**在 ISA 层不存在** |
| 累加器在 **TMEM**（~256KB/SM），不在寄存器 | Marlin 的 `frag_c` 寄存器累加 + 寄存器内 `thread_block_reduce` 全部作废；epilogue 要 `tcgen05.ld` |
| **单线程** `elect_one_sync()` 发射，异步 + mbarrier 完成 | Marlin 的 "8 warp 各自算 partial、再 SMEM 归约" 变成多余的通信 |
| `M ∈ {64,128}`，`N%8≤256`，`cta_group::2` 双 SM 协同 | 16 行粒度、striped partition、单 CTA 视角全部对不上 |
| 比 Hopper WGMMA **快 2–4×** | 论文里"内存受限所以 TC 空转"的余量被彻底吃光，向量路径成为硬瓶颈 |

### 3.2 更硬的墙：TC 再翻倍，而向量管线基本没动

这是我认为最本质的一条量化论证：

$$\frac{\text{TC MAC/clk}}{\text{FP32 lane/clk}}:\quad \text{A100}=\frac{1024}{64}=16 \;\to\; \text{H100}=\frac{2048}{128}=16 \;\to\; \text{B200}=\frac{4096}{128}=32$$

如果按 Marlin 真正依赖的 `half2` 向量吞吐算（$2\times$ FP32 lane），比值是 $8 \to 8 \to 16$。

**含义**：Marlin 每个权重花的 ~1.5 条向量指令（`lop3`×2 + `hsub2` + `hfma2` + `hmul2`，8 个 fp16/12 ops），在 A100 上相对于 TC 消耗速度是有余量的；在 B200 上余量**减半**。也就是说，即使在理想情况下，dequant 也会成为新的 roofline 上限——而 Blackwell 的答案是"别在向量路径上做"，直接用硬件：

### 3.3 Blackwell 给了硬件去量化，但要求你换量化格式

`media/docs/cpp/blackwell_functionality.md`：

- `tcgen05.mma.kind::f8f6f4`：$A=\{f4,f6,f8\}\times B=\{f4,f6,f8\}$，**原生 4-bit 运算**，`float4_t` 对齐要求 128 元素；
- `kind::mxf8f6f4 / mxf4 / mxf4nvf4.block_scale`：$D = C + (A\cdot SFA)(B\cdot SFB)$，**scale 由 MMA 硬件施加**，SF vector size 16/32；
- 代价：`mxf4` 只支持 **TN**，对齐 128，且数据类型必须是 **FP4(e2m1)/MXFP4/NVFP4**，不是 GPTQ 的 INT4 + group-128 对称零点。

于是 Marlin 那条路的终点是：**要么把 checkpoint 重新量化成 FP4/MX 格式、让 TC 原生吃 4-bit（此时 dequant 成本归零，Marlin 的 `dequant()` 成为历史）；要么坚持 INT4，就必须 dequant 进 SMEM，永远拿不到峰值。** Marlin 格式（int4 + per-128 fp16 scale + 自定义 reshuffle 布局）恰好落在两个都不讨好的中间地带。

---

## 四、需要修正的一个常见误判：Hopper 上 Marlin 并非全区间输

这点值得说清楚，否则会得出"Hopper 上 Marlin 全面没用"的错误结论。

<details>
<summary><b>展开：按 batch 分区讨论</b></summary>

在 $m \le 16$ 的 decode 区间，int4 权重流式读取受 HBM 带宽限制（H100 3.35 TB/s）。此时需要喂给 TC 的权重速率是

$$\frac{2048\ \text{MAC/clk}}{16} = 128\ \text{weights/clk per SM},$$

而向量路径能产出约 $256/1.5\approx 170$ weights/clk —— **余量存在，TC 不是瓶颈，`mma.sync` 只有半速也不重要**。这解释了为什么 Marlin 三年来在 H100 上仍然能打、为什么 vLLM 一直在用。

真正失速的是：

- **prefill / 大 batch serving / MoE**（$m \ge 64$）：进入 compute-bound，`mma.sync` 的 $\tfrac12$ 上限直接体现；
- **B200 上更是如此**：8 TB/s HBM + 126MB L2 让"权重搬运"变得极其便宜，**瓶颈前移到 TC 与 dequant**，而 Marlin 在这两处都是最弱的一环。

所以准确表述是：Marlin 是 **Ampere 上、小 batch、INT4 场景的局部最优**；它在 Hopper 上"还能用"是因为那个区间还没触碰它的弱点，而不是因为它适配了 Hopper。

</details>

---

## 五、如果要在 Hopper/Blackwell 上"榨干"，必须改什么

<details>
<summary><b>展开：按 CUTLASS 的范式给出的对照改造清单</b></summary>

| 维度 | Marlin (Ampere) | Hopper 正确做法 | Blackwell 正确做法 |
|---|---|---|---|
| 计算原语 | `mma.sync.m16n8k16` | `wgmma.mma_async.m64nNk16`（$N$ 到 256） | `tcgen05.mma`，1SM/2SM |
| 权重路径 | int4 → 寄存器 → TC | int4 → SMEM(fp16) → 描述符 → TC | 直接 FP4 + `block_scale` 硬件去量化 |
| 搬运 | `cp.async` 16B/线程 | `cp.async.bulk`/TMA + `mbarrier` | TMA + CTA-pair（`cta_group::2`） |
| 并行结构 | 单融合循环，全体线程 | warp specialization（producer/consumer）+ Pingpong 调度 | 加 TMEM 分配器 + `tcgen05.cp` |
| 累加器 | 寄存器 `FragC` | 寄存器（`wgmma` 也在 RF） | **TMEM**，epilogue 用 `tcgen05.ld` |
| 归约 | 全局锁 + fp16 部分和 | cluster/DSMEM 或 fp32 workspace | 2SM MMA 天然跨 SM 归约 |
| SMEM | 96KB / 4 级 | 227KB / 更多级 + 128B swizzle 描述符 | 更大 + TMEM 分担压力 |
| scale | `__hmul2`（组内 16） | 仍在向量路径 | `block_scale` 硬件施加 |

一句话：**Marlin 是"在寄存器里合成 fp16"的极致；Hopper/Blackwell 要求"在 SMEM/TMEM 里准备好、让 TC 自己去读，或者干脆别合成"。范式反了。**

</details>

---

## 六、一句话总结

Marlin 用"寄存器内去量化 + `mma.sync` + 手工 swizzle + 融合循环"这四件 Ampere 的专属武器，把 FLOP:byte 算到了极限，所以它"榨干"的是 **Ampere 的寄存器操作数通路**。Hopper 把峰值转移到了 `wgmma` 的 SMEM 描述符通路上（B 无寄存器形态），Blackwell 更进一步取消了寄存器操作数、把累加器搬进 TMEM、并让 TC 再快 2–4 倍而向量管线原地踏步——**武器被逐条收缴了**。所以它不是"再优化 10%"的问题，而是要不要换成 CUTLASS 那套 TMA + warp-specialized +（Hopper）`wgmma` /（Blackwell）`tcgen05` + 硬件 block-scale 的范式问题；而对 INT4 group-128 这个格式来说，Blackwell 上最优解很可能连"去量化"这个动作都不存在了。

> 说明：以上 Marlin 侧结论均引自 `IST-DASLab/marlin` 的 `marlin/marlin_cuda_kernel.cu` 与 README；CUTLASS 侧引自 `include/cute/arch/mma_sm90_gmma.hpp`、`include/cute/arch/mma_sm100_umma.hpp`、`media/docs/cpp/blackwell_functionality.md`、`media/docs/cpp/pipeline.md`（读的是 `main` 分支）。第四节的分区讨论中具体数值为按规格推算的量级估计，非实测。第五节表格是改造方向的定性对照，未逐项做性能验证。
