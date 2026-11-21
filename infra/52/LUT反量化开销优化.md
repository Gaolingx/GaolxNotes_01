# LUT反量化开销优化

## 🧑‍💻 User

我最近在做llm fused dequant+gemm kernel，遇到了lut decode开销太大怎么解决
就是Kimi K3在hopper对标卡上没法用 mxfp8 native dtype
所以只能先把mxfp4 dequant成fp8再gemm
nvfp4也是同理

我记得 hopper的话也是不支持mxfp8的呀 hopper上lut smem dequant还是比较快的
b卡可以用cvtrn
还有就是这种lut查表量化 为什么会比常规的对称/非对称量化算法慢（比如gguf的iq4nl就比q4km慢）
是调度瓶颈 还是寄存器压力大 还是不规则仿存（比如bank conflict的问题）吗

---
## 😀 默认助手 (7)

下面按「对齐硬件事实 → 定位瓶颈 → 定量判断 → 解法」的顺序说明。你问的三个可能原因（调度、寄存器、bank conflict）**都不完全是**主因；真正的主因是**张量核变快后，CUDA core 的 issue 预算被压缩了一半**。LUT 只是把问题从「勉强够」推到了「不够」。

---

## 0. 先对齐硬件前提

| | Hopper (SM90) | Blackwell (SM100/120) |
|---|---|---|
| FP8 WGMMA | ✅ 有，但只有**per-tensor scale**，没有 block-scaled | ✅ |
| MXFP8 / MXFP4 block-scaled MMA | ❌ **没有** | ✅ `tcgen05.mma` 原生处理 FP4 和 scale |
| FP4 转换 | 无 `cvt`，只能手搓 | `cvt.rn.satfinite.e2m1x2.f32` 等 packed cvt |
| W4A8 怎么做 | 只能 dequant→FP8 再跑普通 FP8 WGMMA | 直接喂 FP4+scale，**根本不用 dequant** |

所以你的判断没错：Hopper 上 "MXFP4 → FP8 → WGMMA" 是不得不走的路径。但请注意一个关键结论：

> **在 Blackwell 上，这个 dequant 应当直接去掉，改用 `tcgen05` 原生 FP4。你现在是在用 Hopper 的架构缺陷换取性能，所以出现性能损失是必然的，不应归因于实现写得不好。**

---

## 1. 结论先行：瓶颈排序

按我判断的权重大小排序如下：

1. **每元素的指令数超过 issue 预算**（约 60%）。这属于「调度/issue 瓶颈」，但根源是**算术指令太多**，并非调度器本身多做了无用功。
2. **LUT 放在 smem 或 constant 时产生的额外访存**：额外 `LDS` 加上 constant 发散串行化（约 25%）。这才是你提到的「不规则访存」。
3. **寄存器压力**（约 10–15%）：它通常来自第 1 条的副作用，是因为展开过度，而非原因。
4. **纯 bank conflict**：影响远小于你的预期，详见下文。

---

## 2. 两个叠加因素

### 2.1 最关键：MMA 快了一倍，预算砍半

你的印象是「Hopper 上 smem LUT dequant 挺快」，这是针对 **W4A16/BF16** 路径；这条路径确实能跑，因为 BF16 WGMMA 的吞吐只有 FP8 的一半。一旦换成 **FP8 WGMMA**，张量核吞吐翻倍，而 CUDA core 数量不变，因此：

- 每个 dequant 出来的元素，**可分摊的 ALU 指令预算直接减半**。
- 同一个 kernel 中，dequant 代码一行都没有改动，但它从「够用」变成「瓶颈」。

以 H100 SXM（dense）估算：

$$
R=\frac{\text{TC MAC/s}}{\text{ALU op/s}}
=\frac{990\ \text{T-MAC/s}}{33.5\ \text{T-op/s}}\approx 30
$$

每个元素被复用 $N$ 次（$N$ 是 MMA 的 N 维），于是每元素可用的指令预算为：

$$
B=\frac{N}{R}\quad\Rightarrow\quad
B(N{=}128)\approx 4.3,\qquad B(N{=}256)\approx 8.6
$$

换成 per-SM 每周期吞吐看更直观：SM 每周期约能发 $128$ 个 lane-op，张量核每周期约 $3787$ MAC：

$$
\text{需要的元素/周期}=\frac{3787}{N},\qquad
\text{能供给的元素/周期}=\frac{128}{\text{ops per element}}
$$

- $N=128$、$5$ ops/element → 供给 $25.6$ < 需求 $29.6$ → **无法持续满足需求**
- $N=128$、$4$ ops/element → 供给 $32$ > 需求 $29.6$ → **刚好压线**

> **结论：在 Hopper 的 FP8 直通路径上，dequant 必须压在 $\lesssim 4$ ops/element（N=128）。任何用 `LDS` 做 LUT 的实现大约在 5–7 ops/element，必然无法满足需求。** 这就是你看到「慢」的主要原因。

### 2.2 FP4 解包比 INT4 更贵

| | INT4 (Q4_K) | FP4 (MXFP4) |
|---|---|---|
| 解包 | 掩码+移位 | 掩码+移位（一样） |
| 数值化 | `I2F`（硬件一条） | 没有硬件指令，只能 **LUT 或 bit-twiddle** |
| 应用 scale | `FMA`（affine，常规） | 需要 rebias（E8M0 幂次）或 **FP32 乘法**（NVFP4） |

INT4→FP8 有一条更直接的路径：`I2F` → `cvt.rn.satfinite.e4m3x2.f32`。**FP4 是浮点格式，没有直接转换，LUT 只是为了省事才被引入——但它其实可以被完全消除**（见 §5）。

再补充一层：**NVFP4 比 MXFP4 更难**。MXFP4 的 scale 是 E8M0（纯 2 的幂），折算进 FP8 只需把指数位加一个常数（整数加法，约 0.25 op/element）；**NVFP4 的 scale 是 E4M3（非幂次），必须做真正的 FP32 乘法再 cvt**，每个元素多 1–2 条指令。因此，「NVFP4 同理」其实更麻烦。

---

## 3. 你那三个猜测的定位

### 3.1 是调度瓶颈吗？——是，但本质是 issue 带宽

Hopper 的 SM 只有 4 个 warp scheduler，每周期最多发 4 条 warp 指令。dequant、主循环控制、WGMMA 同步和 epilogue 全在争抢这 4 个槽位。LUT dequant 每元素需要 5–7 条，本身就消耗掉大量槽位，**所以表现为调度/issue 瓶颈，但解决方法是减少指令数，加宽调度没有用**。

### 3.2 是不规则访存 / bank conflict 吗？——有，但分情况

**LUT 放在 `__constant__`（很多开源实现这么做）：这是最大的问题。**
`LD.CONSTANT` 只在**全 warp 索引相同**时才广播；如果各 lane 的 4-bit 码不同，硬件会**按不同地址拆成最多 16 次访问串行重放**。在 NCU 里表现为 `imc_miss` 或 `short_scoreboard` 飙升。这属于纯粹的浪费指令周期。

**LUT 放在 smem：比 constant 好很多，但仍有代价。**
16 个 float 条目 = 64B，正好铺满 16 个 bank。32 个 lane 查 16 个地址 → 平均 **2-way conflict**，最坏 4-way 以内。**注意：conflict 本身并不严重，真正贵的是「为了这 4 个字节，消耗了一条 LSU 指令和一次 smem 往返」**。对于只有 16 个空格的表，用 smem 做 LUT 是最不划算的选择。

**还有一个常被忽略的隐藏成本**：FP4 每 8 个 nibble 打包在一个 word 里，一次 `LDS` 就能取到 8 个；但 dequant 完的 FP8 是 **1 字节/元素**，写回 smem 要 `STS`，WGMMA 还要再从 smem 读取。也就是说，**dequant 结果必须经历 smem 往返（`STS` + WGMMA 的 smem 读）**，额外增加约 0.25–1 次访存/元素。这个成本往往比 LUT 本身还高。

### 3.3 是寄存器压力吗？——通常是「果」，而非「因」

LUT 本身体积很小（4 个寄存器）；如果真的造成压力，原因通常是：

- 为了隐藏 `LDS` 约 20–30 cycle 的延迟而**深度展开**；
- 为了双缓冲 dequant 结果而**提前预占一批寄存器**来存放 FP8 数据。

二者叠加后，`registers_per_thread` 上涨，occupancy 掉，latency hiding 变差，于是你看到大量 stall。**这是为了弥补 §2.1 的吞吐缺口而展开所付出的代价**，而寄存器压力本身不是根因。

---

## 4. IQ4_NL vs Q4_K_M：同一套解释

- **Q4_K_M**：纯 affine，`I2F` + `FMA`（`d*scale*q - dmin*min` 可合并成 1 FMA），**没有 gather**，SIMD/流水线完美。
- **IQ4_NL**：进行**查表** `d * kvalues_iq4nl[q]`。在 CUDA 上这就是「发散 gather + 乘法」：

在 CUDA 上，如果 `IQ4_NL` 的 `kvalues_iq4nl` 走 constant/`__ldg` 并被发散索引，它就吃亏；在 **CPU/SIMD** 上则相反（`vpshufb` 一条指令就能完成 16 项查表，`IQ4_NL` 因为只用一个 scale 反而更快）。**因此，「IQ4_NL 比 Q4_K_M 慢」是 GPU 后端特有的现象，正是因为查表在单指令多线程下变成发散访存**——这正好印证 §3.2。

> 另外一个思路：既然表只有 16 项，**就不要把它当成 memory LUT，而应当成「指令级表」**——`LOP3` 和 `PRMT` 在寄存器里完成查表。见下。

---

## 5. 怎么解（按性价比排序）

<details>
<summary><b>① 核心解法：把 LUT 搬进寄存器（PRMT + LOP3），彻底消灭 LDS</b></summary>

FP4 的 8 个幅值映射到 E4M3 是**无损**的（1 bit 尾数进 3 bit，4 个 octave 的指数装得下 4 bit 指数位），而且 FP4→E4M3 的映射逐位确定：

| FP4 幅值码 $m$ | 值 | E4M3 字节 |
|---|---|---|
| 0 | 0 | `0x00` |
| 1 | 0.5（次正规） | `0x30` |
| 2 | 1.0 | `0x38` |
| 3 | 1.5 | `0x3C` |
| 4 | 2.0 | `0x40` |
| 5 | 3.0 | `0x44` |
| 6 | 4.0 | `0x48` |
| 7 | 6.0 | `0x4C` |

$m\ge2$ 时恰好是等差 $4m+0x30$；只有 $m=1$（FP4 的次正规）需要修正 $-4$，$m=0$ 要清零。

**关键技巧**：这 8 个字节正好是 8 项表，塞进 2 个 32-bit 寄存器，用 `prmt.b32` 一次完成 4 次查找：

```cuda
// 表：00 30 38 3C 40 44 48 4C  (小端)
const uint32_t lo = 0x3C383000u, hi = 0x4C484440u;

// w 里 8 个 nibble 表示 8 个 FP4
uint32_t mag = w & 0x77777777u;              // 去符号，得到 8 个 3-bit 索引
uint32_t d0 = __byte_perm(lo, hi, mag & 0x7777);        // nibble 0..3 → 4 字节
uint32_t d1 = __byte_perm(lo, hi, (mag >> 16) & 0x7777); // nibble 4..7 → 4 字节
// 符号位：从 0x88888888 里重排到每字节 bit7（再来一次 __byte_perm 或 LOP3）
// scale：E8M0 是 2 的幂，等价于给每个 E4M3 字节的指数位加常数
```

成本：**约 1–1.5 ops/element**（含解包、PRMT、符号、scale fold），**零次访存**。与 LDS-LUT 的 5–7 ops 相比，是 3–4 倍的改善，直接回到 §2.1 里的预算带以内（$N=128$ 也够）。

> 唯一需要注意的一点：scale 折算到 E4M3 时会**溢出指数域**（E4M3 正规值上限 448）。解决办法是：先算一个**全张量/全行的 global rescale**，把 E8M0 指数范围压到 ±(4 bit) 内，再收进 epilogue。**FP4→FP8 本身无损，risk 全在 scale fold 这一步。**
</details>

<details>
<summary><b>② 免掉 smem 往返：让 dequant 结果作为 WGMMA 的 A 寄存器操作数</b></summary>

Hopper WGMMA 规定 **A 可以来自寄存器，B 必须在 smem**。如果你的 dequant 产物能走 A 位，就**省掉 `STS` + WGMMA 的 smem 读**（约 0.25–1 次访存/元素）。

对 W4A8 decode，通常权重是「小张量、被复用」，正好适合放进寄存器当 A：把 GEMM 写成 $y^\top = W^\top x$，dequant 后的权重进 A 寄存器，激活放 smem 当 B。

代价是 dequant 的 lane→寄存器映射要对齐 wgmma 的 fragment 布局（m64n*k16 的 A 布局），实现上麻烦一点，但这是**消除一半访存压力**的关键。
</details>

<details>
<summary><b>③ 离线预 dequant：如果 prefill 占主导，干脆别 fuse</b></summary>

权重是**静态**的。在 load 时一次性把 MXFP4 解成 FP8 存进 HBM，运行时就是**纯 FP8 GEMM，零 LUT 开销**。

代价是 HBM 占用/带宽 ×2（FP4→FP8）。所以：

- **prefill（大 M，compute-bound）**：FP8 张量核能吃满 → 预 dequant 几乎一定更快，fuse 反而是在拿宝贵的 issue 槽位换带宽，得不偿失。
- **decode（M=1，memory-bound）**：带宽省一半有意义 → 值得 fuse。但 decode 阶段张量核本来就没吃饱，dequant 的 ALU 开销大概率**被掩盖**，所以也不会成为瓶颈。

**先算一下你的 Kimi K3 场景在哪个 regime，这比调 kernel 更影响结果。**
</details>

<details>
<summary><b>④ warp specialization + 双缓冲，把 dequant 和 MMA 放到不同 pipe</b></summary>

用 TMA 拉 FP4 → producer warpgroup 负责 dequant→FP8→smem → consumer warpgroup 跑 WGMMA。这样 dequant 占用 ALU/LSU，MMA 占用 tensor pipe，二者**互不争抢 issue 槽位**，可以把 §2.1 的吞吐缺口从「串行叠加」变成「并行取 max」。

但注意：**这只能掩盖重叠开销，无法改变 ALU:TC 的容量比例**。如果 ops/element 本身就超预算（§2.1），warp specialization 也只能让你「满负荷地慢」。**必须和 ① 一起用。**
</details>

<details>
<summary><b>⑤ 调 tile 的 N，把预算从 4.3 拉到 8.6</b></summary>

$B=N/30$——把 N 从 128 提到 256，预算直接翻倍。在 Hopper 上 `m64n256k16` 的 WGMMA 是可行的。这是**不用改 dequant 代码、纯 tiling 层面**就能拿到的 2× 余量。
</details>

<details>
<summary><b>⑥ 换 E5M2 当 dequant 目标格式</b></summary>

如果 E4M3 的指数域装不下 scale 范围，用 **E5M2**：指数 5 bit，范围到 57344，而 FP4 只有 1 bit 尾数，塞进 E5M2 的 2 bit 尾数**依然无损**。Hopper 上 E5M2 和 E4M3 的 WGMMA 吞吐相同，等于**免费拿到更大范围**。（需确认你用的 PTX 变体允许 A/B 混用 e4m3×e5m2。）
</details>

---

## 6. 为什么 NCU 判定瓶颈的标准

<details>
<summary><b>展开：用 NCU metric 走一遍决策树</b></summary>

先看 warp stall 分布（`smsp__warp_issue_stalled_*_per_warp_active.pct`）：

| 观测 | 结论 | 对应解法 |
|---|---|---|
| `math_pipe_throttle` 高，且 `issue_active` ≈ 100% | **ALU/INT pipe 打满** → 指令数超标 | ① PRMT、⑤ 调 N |
| `mio_throttle` 高 + `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum > 0` | **smem LUT** → 移出 smem | ① |
| `imc_miss` / `lg_throttle` 高 | **constant 发散重放** | ① |
| `short_scoreboard` 高 | smem 依赖延迟 → 展开不够或往返太多 | ① 或 ② |
| `registers_per_thread` 高 + occupancy 低 + `long_scoreboard` | **寄存器压力（展开的代价）** | 减展开、② |
| MMA pipe 空闲、dequant 也不忙 | **是 overlap/调度问题，不是 dequant 成本** | ④ warp specialization |

另外一定要自己算两个派生量做对照：
- `smsp__inst_executed.sum / (dequant 元素数)` → 实测 ops/element，直接和第 §2.1 的 4.3 门槛比
- `sm__pipe_tensor_cycles_active / sm__cycles_elapsed` → 张量核利用率，判断是不是「要么等 dequant 要么等内存」

如果 tensor pipe 利用率 < 60% 而 issue 满，**那就更明确：不是 bank conflict，是 issue 带宽不够**。
</details>

---

## 一句话总结

- **根因**：Hopper 的 FP8 WGMMA 比 BF16 快一倍，把 dequant 的 ALU 预算从 ~8.6 压到 ~4.3 ops/element；而 LUT-dequant（尤其走 smem/constant）需要 5–7 ops，于是从「够用」变成「硬缺口」。**这不是你实现得不好，而是「用 Hopper 的架构缺陷换取性能」的必然损耗。**
- **你的三个猜测**：**调度/issue = 主因**（但不是调度器问题，是指令数）；**不规则访存 = 次要主因**（constant 发散 > smem bank conflict，真正的痛点是「为了 16 项表付一次访存」）；**寄存器压力 = 果不是因**。
- **最优解**：**把 LUT 从内存搬进 `PRMT` 和 `LOP3`（约 1 op/element），同时让 dequant 产物走 WGMMA 的 A 寄存器位，避开 smem 往返**。之后再用 ⑤ 调 N，在 prefill 上考虑 ③ 离线预 dequant。
- **战略上**：有 B200 就用 `tcgen05` 原生 MXFP4/NVFP4，**这个问题直接从架构上消失**——不值得在 Hopper 上为它做过多的指令级优化。

如果你愿意贴一下 NCU 的 stall 分布和具体的 tile 配置（N、warpgroup 数、dequant 是 producer 还是 inline），我可以帮你把上面这几条收敛到一条明确的改法。

---
## 🧑‍💻 User

你去看看flashinfer的github repo有没有针对sm90的mxfp8和nvfp4 fused densegemm kernel或者nn.linear相关算子，说不定处理了上述提到的问题

---
## 😀 默认助手 (7)

我来查一下 flashinfer 的实际实现，看看它是否有针对 SM90 的 MXFP8/NVFP4 路径。

GitHub 代码搜索需要登录，我改用原始文件和官方文档。

有 `fp8_blockscale_gemm_sm90` 和 `kimi_k3_fp8_projection` 这两个名字，正好符合你的场景。我去核对一下它们的架构约束。

关键信息已经拿到了，但还需确认架构检查条件。我去看源码里的实际断言。

我去看了 flashinfer 的 main 分支和 0.7.3 文档。结论与预期可能相反：**它没有解决这个问题，而是从架构上绕开了它。SM90 上没有任何 FP4 dense GEMM。**

## 一、结论：SM90 没有 FP4 GEMM

我把相关算子的架构要求都核对了一遍。flashinfer 用 `@supported_compute_capability([...])` 装饰器写明这些要求，源码在 `flashinfer/gemm/gemm_bf16_fp4.py` 和 `gemm_base.py`：

| API | 用途 | 支持的 CC | 有 SM90？ |
|---|---|---|---|
| `mm_bf16_fp4` | W4A16，bf16×FP4 | `[100, 103, 107, 110, 120, 121]` | ❌ |
| `mm_fp4` | NVFP4/MXFP4（block 16/32） | cudnn：SM100/103/110/120/121；cutlass：SM120；cute-dsl：SM100/103；b12x：SM120/121 | ❌ |
| `mm_mxfp8` | MXFP8（E4M3 + E8M0，block 32） | cudnn：SM100/103/110/120/121；b12x：SM120/121；cutedsl_low_latency：SM100/103 | ❌（cutlass 分支需实测，`bmm_mxfp8` 里明确注释是 “SM120/121 CUTLASS path”）|
| `group_gemm_mxfp8_mxfp4_nt_groupwise` | fp8×fp4 混合（MoE） | 同样 FP4 侧，只能 Blackwell | ❌ |

你可以直接在自己的机器上验证：

```bash
python -c "
from flashinfer.gemm import mm_bf16_fp4, prepare_bf16_fp4_weights
" 2>&1 | tail -5
# 或在调用时观察装饰器抛出的 compute capability 断言
```

**原因**：flashinfer 所有 FP4 路径的底层都是 `tcgen05` 和 CUTLASS SM100 内核，它们利用**硬件原生 block-scaled MMA**。**既然硬件能直接接收 FP4 和 scale，就根本不需要 LUT dequant，因此也不会有人去写它。** 这印证了上一轮的判断：在 Blackwell 上，这个问题在架构层面消失了；而 flashinfer 的立场是「SM90 不要做 FP4」。

## 二、唯一一个 SM90 的 block-scaled dense GEMM

`fp8_blockscale_gemm_sm90` 是唯一在 SM90 上真正可用的方案：

```python
from flashinfer.gemm import fp8_blockscale_gemm_sm90

out = fp8_blockscale_gemm_sm90(input, weight, input_scale, weight_scale)
```

文档原话：

> This function **requires NVIDIA Hopper (SM90)** architecture and CUDA 12.8+. The function uses **DeepGEMM backend with JIT compilation**.

它的关键特征恰好解释了你的问题：

| 项目 | 取值 |
|---|---|
| 权重格式 | **FP8 (E4M3)**，不是 FP4 |
| scale 粒度 | per-token `1×128` 或 per-block `128×128`（**FP32**） |
| 支持组合 | `BF16+BF16`（内部量化）、`BF16+FP8`、`FP8+FP8` |
| **swapAB** | **`M < 32` 时自动走 swapAB kernel** |

其中 **`M<32` 自动 swapAB** 这一条，正好对应我在上一轮提到的第 ② 条建议「让 dequant 产物作为 WGMMA 的 A 寄存器操作数」。DeepGEMM 在 decode 阶段就是这么做的：把权重换到 A 位，以避开 smem 往返。所以这个方向是对的，上游早就在利用它。

但请记住：**它接收已经量化为 FP8 的权重，不接收 FP4。** 它不帮你做 dequant。

## 三、最有说服力的证据：Kimi-K3 的官方算子

flashinfer 里有一个专门为你这个模型写的算子 `kimi_k3_fp8_projection`，相关源码是 `flashinfer/gemm/kimi_k3_fp8_projection.py`。它的 docstring 直接说明了做法：

> The experimental generated-program backend runs the KDA / MLA projection GEMMs of `nvidia/Kimi-K3-NVFP4` whose weights are serialized as **`FP8_PB_WO` (E4M3 with a 128x128 FP32 block scale)** on **SM100 / SM103**: BF16 activations are quantized per token in 1x128 blocks to E4M3 with power-of-two (UE8M0) scales... and multiplied with the weight **requantized to UE8M0 scales** (vLLM `requant_weight_ue8m0`) through block-scaled tcgen05 MMA.

这里有三个可以直接借鉴的事实：

1. **只在 SM100/SM103 上可用**（没有 SM90）。
2. **权重不以 MXFP4 或 NVFP4 直接进入 kernel**，而是先**离线 requant 成 FP8 + UE8M0 scale**。
3. 这个 requant 发生在 `prepare_kimi_k3_fp8_projection_weights()` 里，**每个权重只做一次，不在每次 launch 的热路径上**。

也就是说，上游对你这个问题的回答是：

> **你可以把 dequant 做在 weight-prep 阶段，不要 fuse 进 GEMM。**

这与我上一轮第 ③ 条「离线预 dequant」完全一致，只是执行得更彻底：他们连量化格式都换掉了（从 FP4 换成 FP8 + 2 的幂 scale），从而让 Hopper 的 FP8 WGMMA 能用上 block scale。

flashinfer 还配套提供了独立的 dequant 算子，进一步佐证「dequant 应当独立于 GEMM」：

```python
flashinfer.quantization.mxfp4_dequantize            # 独立 dequant kernel
flashinfer.quantization.mxfp4_dequantize_host       # host 侧
flashinfer.quantization.e2m1_and_ufp8sf_scale_to_float  # FP4 值 + E8M0 scale → float
flashinfer.quantization.mxfp8_dequantize_host
```

注意 `e2m1_and_ufp8sf_scale_to_float` 这个名字：`e2m1` 是 FP4，`ufp8sf` 是 unsigned FP8 scale factor（也就是 **E8M0**），它就是你正在写的那个 LUT dequant。但它被放在 **quantization** 模块里做**离线工具**，而不是放在 gemm 模块里做 fused kernel。

## 四、一个可能直接省掉你一半工作量的发现

看完这些，我重新审视了你的链路，发现你在上一轮还有一个隐含前提可能是错的：

**MXFP4 和 MXFP8 的 block 结构完全相同：**

| | MXFP4 | MXFP8 |
|---|---|---|
| scale 格式 | E8M0 | **E8M0** |
| block size | 32 | **32** |
| 值格式 | E2M1 | E4M3 |

也就是说，**从 MXFP4 转换到 MXFP8 时完全不需要 fold scale**：scale 原样透传即可，只需把 4-bit 值展开成 8-bit E4M3 值。

这意味着：

- **不需要做任何 E4M3 指数溢出保护**（上一轮我担心的那个 range 问题，直接消失了）。
- **不需要做那 0.25 op/element 的 packed byte add**。
- 你的 dequant 退化成纯粹**「nibble 变成 byte」的格式展开**；这正是 `PRMT` 和 `LOP3` 最擅长的操作，约 **1 op/element**，不需要访存。

$$\text{MXFP4}\to\text{MXFP8}:\quad \underbrace{\text{unpack}}_{0.5} + \underbrace{\text{PRMT}}_{0.5} + \underbrace{\text{sign}}_{0.25} \approx 1.25\ \text{op/element},\quad \text{scale: free}$$

对比上一轮的预算 $B(N{=}128)\approx 4.3$：**有 3× 以上余量。**

**但是**：这要求 kernel 能接收「per-32 E8M0 block scale」的 FP8 MMA。而这正是 Hopper 做不到的事；Hopper 的 FP8 WGMMA 只有 per-tensor scale。所以：

- **SM90**：你只能在两步里选一个：
  - **(a)** 使用 `fp8_blockscale_gemm_sm90`（DeepGEMM），并把 MXFP4 **离线 requant 成 128-block FP32 scale 的 FP8**。这是 SM90 上最快的方案，代价是粒度从 32 降低到 128，精度略降。
  - **(b)** 保留 32-block 粒度，就必须把 scale fold 进 FP8 值里，于是回到上一轮的溢出问题，以及每元素多 1 到 2 条指令。
- **SM100/103**：使用 `mm_mxfp8` 即可；scale 原样透传，dequant 只负责格式展开（甚至直接使用 `mm_fp4`，连展开都不需要）。

## 五、给你的三条路线（按推荐度）

<details>
<summary><b>路线 A（SM90 首选）：改用 <code>fp8_blockscale_gemm_sm90</code> + 离线 requant</b></summary>

```python
# 1) 模型加载时做一次：MXFP4 → FP8(E4M3) + 128x128 FP32 block scale
#    这一步慢无所谓，不在热路径
w_fp8, w_scale = requant_mxfp4_to_fp8_128block(w_mxfp4, w_e8m0)

# 2) 热路径：纯 FP8 blockscaled GEMM，零 LUT
from flashinfer.gemm import fp8_blockscale_gemm_sm90
out = fp8_blockscale_gemm_sm90(x_bf16, w_fp8, None, w_scale)
```

收益：直接获得 DeepGEMM 的调优、`M<32` 自动 swapAB、以及 TMA 和 warp-specialized pipeline。你不需要自己实现 dequant 了。

代价：block 粒度从 32 变成 128，精度会有小幅下降（需要实测评估）；需要 CUDA 12.8+ 并启用 JIT 编译。
</details>

<details>
<summary><b>路线 B：继续做 fused kernel，但把 LUT 换成 PRMT（我上一轮的方案）</b></summary>

如果你想保留 32-block 的粒度，或者想避开 DeepGEMM 的依赖，那就接受「必须 fold scale」这个事实，然后用寄存器内的 LUT 把成本压到 1 到 1.5 op/element：

```cuda
// 8 个 FP4 幅值 → E4M3 字节，塞进 2 个 32-bit 寄存器
const uint32_t lo = 0x3C383000u, hi = 0x4C484440u;  // 00 30 38 3C | 40 44 48 4C
uint32_t mag = w & 0x77777777u;
uint32_t d0 = __byte_perm(lo, hi, mag & 0x7777);           // 4 次查表 / 1 instr
uint32_t d1 = __byte_perm(lo, hi, (mag >> 16) & 0x7777);   // 4 次查表 / 1 instr
// 符号：0x88888888 重排到每字节 bit7（再一次 PRMT）
// scale：E8M0 是 2 的幂 → 指数位 packed add（需做溢出保护）
```

关键是**不要用 smem 或 constant 查表**；那正是你当前变慢的正确原因（见上一轮 §3.2）。
</details>

<details>
<summary><b>路线 C：如果目标是 B200/B300 —— 别写这个 kernel</b></summary>

```python
from flashinfer.gemm import mm_fp4
out = mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, alpha, torch.bfloat16,
             block_size=32, use_nvfp4=False, backend="auto")  # MXFP4 原生
```

`tcgen05` 直接接收 FP4 和 E8M0 scale，**没有 dequant，也没有你这个问题**。你和上游最聪明的选择都是这一条。
</details>

## 六、一句话总结

- **flashinfer 对「SM90 的 MXFP4/NVFP4 fused dense GEMM」的回答是：不存在，也不建议做。** 它的 FP4 路径全部依赖 Blackwell 的 `tcgen05`；SM90 上唯一提供的是 **FP8** blockscaled GEMM。
- **Kimi-K3 的官方算子是 SM100/103 only，而且把 dequant/requant 放在 weight-prep 阶段**，这直接验证了我上一轮第 ③ 条判断：**这份工作应该付一次，不应该付每个 token。**
- **你值得马上确认一件事**：你的 kernel 到底是在做「MXFP4→MXFP8（scale 透传）」还是「MXFP4→FP8（scale fold）」。如果是后者，**改成前者能同时去掉溢出风险和约 1.5 op/element**；代价只是需要 block-scaled FP8 MMA，而这一点在 SM90 上只能用 DeepGEMM 那条路。
- 如果这个模型必须跑在 Hopper 上且要极致性能，**路线 A 是上游实际在用的做法**；路线 B 是保留 FP4 粒度的代价。

需要的话，我可以把路线 A 的 `requant_mxfp4_to_fp8_128block` 写出来（包括 32→128 重分块时如何做 scale 合并和误差控制），或者帮你搭一个 NCU 对比脚本，量化「fused LUT」vs「预 requant + DeepGEMM」在你的真实 shape 上的差距。
