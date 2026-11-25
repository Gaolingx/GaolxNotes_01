# Tensix BFP格式与HiFi运算差异分析

## 🧑‍💻 User

分析一下Tenstorrent Tensix 架构（Wormhole B0 / Blackhole）中的bfp2/4/8的数值格式特性
以及lofi/hifi2/3/4的运算差异

---
## 😀 默认助手 (7)

我会先查阅 Tenstorrent 官方文档和 `tt-metal`/`tt-llk` 源码来核对格式细节，避免凭记忆猜测。

我把 Tenstorrent 的 ISA 文档（`tt-isa-documentation`）、`tt-metal` 的 host 端 BFP 编解码实现、以及 `tt-llk` 的 matmul LLK 交叉核对了一遍。下面这些数字和位域都是从源码/ISA 原文里取的，不是二手转述。

---

# 一、结论速览

<details open>
<summary><b>核心结论（4 条）</b></summary>

1. **BFP2/4/8 不是"浮点"，而是"每 16 个元素共享一个阶码的定点块浮点"**：元素里只有 `符号 + 尾数`，尾数的最高位是**显式存储的隐含 1**。所以 bfp8 = 7 bit 有效位（6 bit 小数）、bfp4 = 3 bit 有效位（2 bit 小数）、bfp2 = 1 bit 有效位（**只能表示 $\{0,\pm 2^{e}\}$**）。
2. **BFP 只存在于 L1↔寄存器边界**：unpacker 把它无损展开成 **BF16**（`_b` 系列）或 **FP16**（`a` 系列）再进 `SrcA/SrcB`；FPU/SFPU 从来看不到 BFP。
3. **LoFi/HiFi2/3/4 本质是"乘法器位切片"**：硬件乘法器只有 **5 bit × 7 bit**（SrcA 5b 含隐含 1 + 4 bit 小数；SrcB 7b 含隐含 1 + 6 bit 小数）。一个 2-bit 的 `FidelityPhase` 字段选择切哪一段，多相位就是**把同一个 `MVMUL` 发多次、每次向 Dst 累加并舍入一次**。所以 LoFi=1 相位、HiFi2=2、HiFi3=3、HiFi4=4，吞吐正好是 $4.096/2.048/1.366/1.024$ TFLOP/s（每 FPU，WH @1 GHz）。
4. **`_b` 后缀 = 8 bit 阶码（bf16 系，bias 127）；无后缀 = 5 bit 阶码（fp16 系，bias 15）** —— 这是最容易踩的命名陷阱，且和 `Float16` / `Float16_b` 的约定一致。

</details>

---

# 二、BFP2/4/8 的数值格式解剖

## 2.1 位域

| tt-metal `DataFormat` | ISA 名 | bit/元素 | 位域 | 显式隐含 1 | 尾数小数位 | 有效位数 | 块阶码 | 量化最大相对误差 |
|---|---|---|---|---|---|---|---|---|
| `Bfp8_b` | `BFP8` | 8 | `S1 M7` | 是 | 6 | **7 bit** | 8 bit, bias 127 | $2^{-7}\approx 0.78\%$ |
| `Bfp4_b` | `BFP4` | 4 | `S1 M3` | 是 | 2 | **3 bit** | 8 bit | $2^{-3}=12.5\%$ |
| `Bfp2_b` | `BFP2` | 2 | `S1 M1` | 是 | 0 | **1 bit** | 8 bit | $50\%$（只有 $\{0,\pm2^{e}\}$） |
| `Bfp8` | `BFP8a` | 8 | `S1 M7` | 是 | 6 | 7 bit | **5 bit, bias 15** | 同上 |
| `Bfp4` | `BFP4a` | 4 | `S1 M3` | 是 | 2 | 3 bit | 5 bit | 同上 |
| `Bfp2` | `BFP2a` | 2 | `S1 M1` | 是 | 0 | 1 bit | 5 bit | 同上 |

> 命名对应关系是从两处证据推出来的：① ISA 把 `BFP8` 与 `BFP8a` 分列，前者 unpack 后进 **BF16 风格**（8 bit 阶码）、后者进 **FP16 风格**（5 bit 阶码）；② `tt-metal` host 端编码函数的 `is_exp_a` 分支做的是 `exp = exp - 127 + 15`，即把 fp32 阶码重偏置到 **5 bit** 域。所以 `_b`↔8 bit 阶码 ↔ ISA 的 `BFP*`，无后缀 ↔ 5 bit ↔ ISA 的 `BFP*a`。

## 2.2 硬件解码（ISA 原文伪码，最权威）

`BFP8` → BF16 的展开只有四行，非常能说明问题：

```c
uint16_t BFP8ToBF16(uint8_t DatumBits, uint8_t ExpBits) {
    uint8_t Sign = DatumBits >> 7;
    uint8_t Mag  = DatumBits << 1;          // 7 bit 尾数字段（含隐含 1）
    if (Mag == 0) return Sign ? 0xff80 : 0; // ±0
    unsigned LZ = stdc_leading_zeros_uc(Mag); // 归一化
    Mag <<= LZ;  ExpBits -= LZ;
    return (Sign << 15) | (ExpBits << 7) | (Mag & 0x7e); // 取高 6 bit 作小数
}
```

关键点：
- `Mag & 0x7e` 丢掉最低 1 bit → **存储的 7 bit = `[1][6 bit 小数]`**，不是 `[S][7 bit 小数]`。这比 bf16（`[1][7 bit]`）**少 1 bit 精度**，但换来"元素里不含阶码、16 个元素共享一个阶码"。

- `BFP4` 走 `BFP8ToBF16(bits<<4, exp)`，`BFP2` 走 `BFP8ToBF16(bits<<6, exp)` → `Mag & 0x7e == 0`，**bfp2 的尾数恒为 1.0**，取值集合就是 $\{0,\,+2^{e},\,-2^{e}\}$。

- `BFP*a` 系列对应 `BFP8aToFP16()`，里面有 `if (ExpBits & 0xe0) UndefinedBehavior();` —— 这就是 5 bit 阶码的硬性范围（$2^{-15}$ 到 $2^{15}$）。

## 2.3 host 端编码（`tt-metal/tt_metal/impl/data_format/blockfloat_common.cpp`）

- **块大小 = 16 = 一个 face 的一行**。每 16 个连续元素取 `max(exp)` 作为共享阶码：
  ```c
  uint8_t get_max_exp(const std::vector<uint32_t>& vec /*size==16*/, bool is_exp_a);
  ```
- 尾数**先归一化、再 round-to-nearest-even**，guard bit 判 tie；`truncate_bfp_mantissa=true` 时可退化为截断。
- **溢出不打回重算块阶码**：向上舍入溢出时尾数钳到全 1（`mantissa = min(mantissa, MAX)`），ND 文档里明确说"不会重新计算 max shared exponent 并重新归一化 16 个数"。
- `exp_a` 版本饱和：`se>31 → se=31, mantissa=0x7fffff`；`se<0 → se=0, mantissa=0`。

## 2.4 Tile 内存布局与体积

块阶码是**前置**在 tile 开头的连续区（每 16 元素 1 字节，一个 32×32 tile 共 $1024/16=64$ 字节 = 4 个 16B 单元，正好对应 4 个 face × 16 行）：

```
┌──────────────────────── bfp8_b tile = 1088 B ────────────────────────┐
│ 64 B 阶码区                     │ 1024 B 尾数区（1 B/元素）            │
│ face0 的 16 行 → face1 → face2 → face3 │ face0(RM) face1 face2 face3 │
└──────────────────────────────────────────────────────────────────────┘
```

| 格式 | tile 字节数 | 相对 bf16 (2048 B) | 组成 |
|---|---|---|---|
| `bfp8_b` | **1088** | 53.1% | $256\times4 + 16\times4$（1024 尾数 + 64 阶码） |
| `bfp4_b` | **576** | 28.1% | $128\times4 + 16\times4$（512 + 64） |
| `bfp2_b` | **320** | 15.6% | $64\times4 + 16\times4$（256 + 64） |

> 注意：**阶码区固定 64 B，不随位宽缩小**，所以 bfp2 有 20% 的存储浪费在阶码上。

## 2.5 块内动态范围 —— BFP 最大的坑

因为块内只有一个阶码，块内可表示范围约等于有效位数：

| 格式 | 块内可表示比值 | 落在 $e_{\max}$ 下方多少就下溢为 0 |
|---|---|---|
| bfp8 | $\approx 2^{7}$（7 bit） | $<2^{e_{\max}-7}$ → 0 |
| bfp4 | $\approx 2^{3}$（3 bit） | $<2^{e_{\max}-3}$ → 0 |
| bfp2 | $\approx 2^{1}$（1 bit） | $<0.5\cdot 2^{e_{\max}}$ → 0 |

举例（同一个 16 元素块，$\max=1.9$ → 共享阶码 $2^0$）：

| 值 | bf16 | bfp8 | bfp4 | bfp2 |
|---|---|---|---|---|
| 1.9 | 1.8984 | 1.90625 | 2.0 | 1.0 |
| 1.2 | 1.2031 | 1.203125 | 1.25 | 1.0 |
| 0.5 | 0.5 | 0.5 | 0.5 | **0（阈值以下）** |
| 0.01 | 0.00999 | **0（下溢）** | 0 | 0 |

**推论**：一个离群点会毁掉同块另外 15 个数的精度，而且"块"是按 **tile 中一行内的 16 个连续元素**（face 行）切的 —— 所以 **tileize / 转置 / 行主列主的选择会改变量化分组**，从而改变精度。这也是 BFP 与 per-channel scale 不同的地方。

## 2.6 其它格式特性

- **无 NaN / Inf**：bfp 没有阶码=全 1 的特殊语义；`-0` 会被保留（`return Sign ? 0xff80 : 0`）。
- **零与 denormal**：编码端 `exp==0` 直接归 0。
- **stochastic rounding** 可选（`StochRndType::{Fpu, Pack, All}`，作用于 FPU 累加 / gasket / packer 三段）。
- **对照**：tt-metal 枚举里还有 MX 家族（`MxFp8P/R`、`MxFp4`、`MxFp6R/P`、`MxInt8/4/2`），它们是**32 元素共享 1 字节 E8M0 标度 + 元素自带指数**（tile = 数据 + 32 B 标度），性质上和 BFP 不同（BFP 的元素**没有**指数）；这批要按 `is_data_format_supported(format, arch)` 做架构门控，不属于 WH/BH 的 Tensix ISA 文档所描述的 BFP/FP16/BF16/INT8 通路。

---

# 三、LoFi / HiFi2 / HiFi3 / HiFi4 的运算差异

## 3.1 乘法器本身：5b × 7b 位切片乘法

ISA `MVMUL` 原文：*"The hardware multipliers consume at most 5 bits of `SrcA` magnitude/mantissa and at most 7 bits of `SrcB`"*。对应精确模型：

```c
uint5_t SrcAMultiplierBits(Operand a, uint2_t phase) {
    if (phase & 1) return (a.Man >> 1) & 0x1f;  // 小数第 5..1 位
    else           return 0x10 | (a.Man >> 6);  // 隐含 1 + 高 4 位小数
}
uint7_t SrcBMultiplierBits(Operand b, uint2_t phase) {
    if (phase & 2) return (b.Man & 0xf) << 3;   // 低 4 位小数 + 3 个 0
    else           return 0x40 | (b.Man >> 4);  // 隐含 1 + 高 6 位小数
}
```

即：**phase 的 bit0 选 SrcA 的哪 5 bit，bit1 选 SrcB 的哪 7 bit**，四个相位就是 $2\times2$ 的位切片叉积。

## 3.2 四相位到底切了哪些位

<details open>
<summary><b>BF16 风格输入（bf16，以及 bfp8/4/2→BF16）</b></summary>

| phase | SrcA（5b，4 位小数） | SrcB（7b，6 位小数） |
|---|---|---|
| 0 | 隐含 1 + 高 4 位尾数 $m_9..m_6$ | 隐含 1 + 高 6 位 $m_9..m_4$ |
| 1 | 剩余 3 位 $m_5..m_3$ | 隐含 1 + 高 6 位 $m_9..m_4$ |
| 2 | 隐含 1 + $m_9..m_6$ | 剩余 1 位 $m_3$ |
| 3 | 剩余 3 位 $m_5..m_3$ | 剩余 1 位 $m_3$ |

</details>

<details>
<summary><b>TF32 / FP16 风格输入（fp16、tf32，以及 bfp*a→FP16）</b></summary>

| phase | SrcA | SrcB |
|---|---|---|
| 0 | 隐含 1 + $m_9..m_6$ | 隐含 1 + $m_9..m_4$ |
| 1 | 下 5 位 $m_5..m_1$ | 隐含 1 + $m_9..m_4$ |
| 2 | 隐含 1 + $m_9..m_6$ | 剩余 4 位 $m_3..m_0$ |
| 3 | 下 5 位 $m_5..m_1$ | 剩余 4 位 $m_3..m_0$ |

**SrcA 的最后 1 位 $m_0$ 任何相位都不消耗**（ISA 明确记载 "not possible for `SrcA`" 的全精度）。
</details>

## 3.3 MathFidelity 枚举 ↔ 相位 ↔ 精度覆盖

`tt-llk` 里就是这么实现的（`llk_math_matmul.h`）：`inner_loops = is_high_fidelity(fidelity) ? to_underlying(fidelity) : 1`，而 `MathFidelity : {LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4}`；相位靠 `addr_mod.fidelity.incr` 每条 `MVMUL` 自增。

| 模式 | 值 | 相位集合 | 每 face-pair 的 MVMUL 数 | 覆盖情况 |
|---|---|---|---|---|
| **LoFi** | 0 | {0} | 1 | A: 5 bit 有效位, B: 7 bit |
| **HiFi2** | 2 | {0,1} | 2 | A **全部**位, B 仍只有高 6+1 |
| **HiFi3** | 3 | {0,1,2} | 3 | A、B 全部位，**只缺 $A_{lsb}\times B_{lsb}$ 交叉项** |
| **HiFi4** | 4 | {0,1,2,3} | 4 | bf16 完全精确；tf32/fp16 受 SrcA $m_0$ 丢弃限制 |

**这条推断很实用**：对 bf16（8 bit 有效位）来说，HiFi3 已经覆盖了 A 的 $(1+4)+3=8$ bit 和 B 的 $(1+6)+1=8$ bit，只少一个量级 $2^{-10}$ 的交叉项；所以 **bf16 下 HiFi3 ≈ HiFi4**（多数 kernel 里差异在 1 ULP 量级），而 HiFi2 才是"半精度尾巴"，LoFi 会砍掉 A 的 3 个 LSB（相对误差可达 $2^{-5}$）。

ISA 给的官方推荐表：

| Src 数据类型 | 最小精度 | "合理"精度 | 全精度 |
|---|---|---|---|
| FP8 / BFP4 / BFP4a / BFP2 / BFP2a | 仅 phase 0 | 仅 phase 0 | 仅 phase 0 |
| **BFP8 / BFP8a** | 仅 phase 0 | 仅 phase 0 | **phase 0 + 1** |
| BF16 | 仅 phase 0 | phase 0 + 1 | **四相位全部** |
| TF32 / FP16 | 仅 phase 0 | 四相位全部 | SrcA 上不可能 |

→ 由此直接得到**工程选型表**：

| 你要算的输入 | 数学上足够的 fidelity |
|---|---|
| bfp2 / bfp2_b | **LoFi**（本身就 1 bit，hifi 完全浪费帧率） |
| bfp4 / bfp4_b | **LoFi** |
| bfp8 / bfp8_b | LoFi（快，A 截到 5 bit）～ **HiFi2**（真正无损） |
| bf16 | HiFi2 / **HiFi3**（性价比最优）/ HiFi4 |
| fp16 / tf32 | **HiFi4**（且 SrcA 仍会丢 $m_0$） |

## 3.4 累加与舍入的精确行为（差异不只是"位宽"）

来自 `MatrixUnitArithmetic.md`（bit-exact 模型）：

- 每个 8×16 Dst 块由两条 lane（`r0`/`r1`）各 8 个乘法器组成；lane 内 8 个乘积**先对齐到最大指数再固定点求和**，lane 保留 **10 bit 小数**（`ALU_ACC_CTRL_Fp32_enabled` 时 23 bit）；然后 r0/r1/Dst 三者再对齐、相加、**只舍入一次**到 Dst 格式。
- **乘积本身是精确的**，但"对齐到 lane 最大指数时会被舍入到 10 个小数位" → 大指数差的两个 BF16 相加可能差 1 ULP。
- 舍入规则：round-to-nearest，**tie 时远离 0**；只有 lane→公共指数对齐那步 tie **朝 $+\infty$**。
- **无 denormal**：阶码字段 0 即 0；结果阶码 $\le 0$ → `+0`。
- **Inf/NaN 无特殊处理**：阶码 255 当普通大指数，结果 $\ge 255$ 变 Inf。
- **每个 fidelity phase 是一条独立 `MVMUL`，各自向 Dst 舍入一次** —— 所以 `HiFi4` 必须配 `fp32_dest_acc_en` 才真正有意义；否则 4 个部分积都被 16-bit Dst（bf16/fp16）分别截断，误差反而可能比 HiFi2+fp32 acc 更差。
- 16-bit Dst 下还有隐藏的 `DstLow` 13 bit 低位寄存器（第一次 32-bit 累加后 bit12 变 `UnpredictableValue()`），混用会偶发 1 ULP 抖动。

## 3.5 吞吐（每 FPU，Wormhole @1 GHz）

| 指令 | 1 phase (LoFi) | 2 (HiFi2) | 3 (HiFi3) | 4 (HiFi4) |
|---|---|---|---|---|
| `MVMUL` | 4.096 TF | 2.048 TF | 1.366 TF | 1.024 TF |
| `GAPOOL`（reduce avg/sum） | 2.048 | 1.024 | 0.683 | 0.512 |
| `ELWMUL` | 0.256 | 0.128 | 0.085 | 0.064 |
| `GMPOOL`(max) / `ELWADD` / `ELWSUB` | 0.256 / 0.256 / 0.128（**与 fidelity 无关**，单相位） | 同左 | 同左 | 同左 |

注意 `Instruction latency = 5 cycles`，且 **IPC 随相位线性下降**（"one instruction per fidelity phase"）。整芯片再乘 tile 数（n150 = 72，n300 = 2×64，Galaxy = 32×80）。

---

# 四、Wormhole B0 vs Blackhole 的差异（都在数据通路上）

<details open>
<summary><b>与 BFP/fidelity 直接相关的 8 条</b></summary>

1. **Blackhole 不再做 `TruncateSrc`**（WH 每条 `Src` 数据出寄存器时按自身格式截断：`FP16/FP8/BFP8a/INT8` 丢高 3 个阶码位，`BFP4a/BFP2a` 再丢低 3 个尾数位，`TF32/BF16` 丢低 3 个尾数位）。→ **WH 上 TF32 实际上被打掉低 3 bit 尾数（只剩 7 bit 小数），BH 上不丢**。这是两个芯片 TF32 精度不同的根因。
2. **Blackhole 有"隐含 Src 格式"（implied format）**：unpacker 在把 bank 交给 FPU 时顺带写入 `ImpliedSrcAFmt/ImpliedSrcBFmt`，覆盖 `ALU_FORMAT_SPEC_REG`，靠 `DISABLE_IMPLIED_SRCA_FMT` 关闭。**WH 没有此机制**，必须显式 reconfig。
3. **Blackhole 的 8-bit 浮点可编程为 E4M3**：`REG1_Unp_LF8_4b_exp` 选定 4 bit 阶码（E4M3）或 5 bit 阶码（E5M2，WH 的 `FP8` 就是 E5M2），并有 BH 专属的特殊值处理（全 1 → 指数 31 + 全 1 尾数等）。tt-metal 的 `Fp8_e4m3` 就是这个。
4. **BFP*a（5 bit 阶码）越界行为**：WH 保留低 5 位（`NonContractualBehavior`），BH 有独立的 wrap 路径且 Src/Dst 表现不同 —— 也就是说 5 bit 阶码的 BFP 是明确不建议用的老格式。
5. **Unpacker 行宽与 throttle**：WH 对所有位宽都是 16 datum/row；**BH 对 $\le 8$ bit 用 32 datum/row**。BH 支持 x8 throttle（128 B/cycle，仅 8-bit 输入安全）；`BFP2/BFP2a` **永远只能 x1**（16 B/cycle），`BFP4` 在 BH 默认 x4（这会改变 BFP4a 的部分结果）。
6. **Dst row-valid 索引**在 BH 上换成 `Adj32NoSwizzle/Adj16`（受 `remap_addrs`/`swizzle_32b` 影响）。
7. **BH 无 WH 的 `Acc == -1` 归一化 erratum**（WH 上 ${-1}$ 会归一化成 ${-16}$，只在 `Fp32_enabled` 时能触发）。
8. 累加/舍入语义（10 vs 23 小数位、无 denormal、tie 规则、单次舍入）**两代一致** —— `MatrixUnitArithmetic.md` 的模型就是以 BH 为准、WH 只列差异。

</details>

---

# 五、工程要点清单（踩坑顺序）

1. **bfp8_b 的"无损 fidelity"是 HiFi2 而不是 LoFi**：LoFi 下 SrcA 只取 5 bit 有效位，会把 bfp8 的 7 bit 再砍 2 bit（$\sim 3\%$ 相对误差）；典型"重量级"配置是 `bfloat8_b + LoFi`（追速度）或 `bfloat8_b + HiFi2`（追精度）。
2. **bfp2 基本等于三值化（ternary）**：取值只有 $\{0,\pm2^{e}\}$，块内阈值 $0.5\cdot2^{e_{\max}}$。别指望它当"4 倍压缩的 bf16"用。
3. **离群点毁块**：块 = tile 内一行 16 个连续元素，所以 tileize/转置/布局会改变量化组；做 weight 预量化时要按 **16 元素块（face 行）** 而不是按 channel 分组。
4. **HiFi4 必须配 `fp32_dest_acc_en`**，否则 4 个相位各自截断进 16 bit Dst，收益消失；这时 `HiFi2/3 + fp32 acc` 往往更快更准。
5. **bf16 场景优先试 HiFi3**：只少 $A_{lsb}B_{lsb}$（$2^{-10}$ 量级），吞吐比 HiFi4 高 33%。
6. **`math_approx_mode` 与 fidelity 无关**：前者只管 SFPU 超越函数（exp/gelu/sqrt 的近似档），后者只管 FPU（`MVMUL/DOTPV/GAPOOL/ELWMUL`）。`ELWADD/ELWSUB/GMPOOL` 完全不看 fidelity。
7. **累加在 L1 时可开 `packer_l1_acc`**：packer 读回 L1 旧值累加，用于"fp32 累加 → 最后一次性 pack 成 bfp8/bf16"。
8. **`Bfp*`（无 `_b`）是 5 bit 阶码老格式**，实际项目里几乎只用 `Bfp8_b/Bfp4_b`（ttnn 的 `bfloat8_b/bfloat4_b`）。

---

## 参考来源（均已通读）

| 内容 | 位置 |
|---|---|
| fidelity 位切片 / 数据类型组合 / 吞吐 | `tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`、`MVMUL.md`、`MatrixUnit.md` |
| bit-exact 浮点算术（lane/舍入/WH 差异） | `.../TensixCoprocessor/MatrixUnitArithmetic.md` |
| BFP 展开伪码、阶码流、BH 差异、E4M3 开关 | `.../TensixCoprocessor/UNPACR_Regular.md` |
| BFP host 端编码/舍入/布局 | `tt-metal/tt_metal/impl/data_format/blockfloat_common.cpp` |
| MathFidelity 枚举与 LLK 实现 | `tt-llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_defs.h`、`llk_math_matmul.h` |
| tile 字节数 / DataFormat 枚举 | `tt-metal/tt_metal/api/tt-metalium/tt_backend_api_types.hpp` |
| 官方 fidelity 概述与 TFLOPS 折算 | `tt-metal/tech_reports/matrix_engine/matrix_engine.md` |
| BFP 舍入规则说明 | `tt-metal/tech_reports/data_formats/data_formats.md` |

> 说明：`data_formats.md` 里的位域图是图片（无法取到其中的数值表），所以上表的位宽/字节数我用的是 `blockfloat_common.cpp` + `tt_backend_api_types.hpp::tile_size()` + ISA `UNPACR` 伪码三处一致的结果。

---
## 🧑‍💻 User

从以下几个点概述
1. Tensix 中的 BFP8 / BFP4 / BFP2 数值格式特性
2. LoFi / HiFi2 / HiFi3 / HiFi4 的硬件位切片与运算差异
  - 为什么矩阵乘要分 Phase 迭代？
  - 四种 MathFidelity（LoFi/HiFi2/HiFi3/HiFi4） 档位的精确掩码与吞吐对比（WH / BH）

---
## 😀 默认助手 (7)

# 一、Tensix 的 BFP8 / BFP4 / BFP2 数值格式特性

## 1.1 位域与精度：块浮点，元素里没有阶码

这三种格式共同的规则是：**每 16 个元素共享一个阶码，每个元素只存“符号 + 尾数”，并且尾数最高位是显式存储的隐含 1**。

| 格式 | tt-metal / ISA | bit/元素 | 位域 | 小数位 | 有效位数 | 块阶码 | 量化最大相对误差 | 块内动态范围 |
|---|---|---|---|---|---|---|---|---|
| BFP8 | `Bfp8_b` / `BFP8` | 8 | `S1 M7` | 6 | **7 bit** | 8 bit, bias 127 | $2^{-7}\approx0.78\%$ | $\approx 2^{7}$ |
| BFP4 | `Bfp4_b` / `BFP4` | 4 | `S1 M3` | 2 | **3 bit** | 8 bit, bias 127 | $2^{-3}=12.5\%$ | $\approx 2^{3}$ |
| BFP2 | `Bfp2_b` / `BFP2` | 2 | `S1 M1` | 0 | **1 bit** | 8 bit, bias 127 | $50\%$ | $\approx 2^{1}$ |
| （变体） | `Bfp8/4/2` / `BFP8a/4a/2a` | 同 | 同 | 同 | 同 | **5 bit, bias 15** | 同 | 同 |

关键点：**bfp8 并非“8 bit 的尾数”，实际只有 7 bit 有效位（比 bf16 少 1 bit）**，因为它的 7 bit 字段里包含了一个显式的隐含 1：

```c
// ISA UNPACR 伪码：BFP8 → BF16
Sign = bits>>7;  Mag = bits<<1;            // 7 bit 字段
LZ = leading_zeros(Mag); Mag <<= LZ; ExpBits -= LZ;
return (Sign<<15) | (ExpBits<<7) | (Mag & 0x7e);   // 只保留 6 bit 小数
```
`BFP4` 走 `BFP8ToBF16(bits<<4, exp)`，`BFP2` 走 `BFP8ToBF16(bits<<6, exp)` → `Mag & 0x7e == 0`，所以 **bfp2 的尾数恒为 1.0，取值集合只有 $\{0,\,+2^{e},\,-2^{e}\}$**（实质上是“三值 + 块标度”）。

## 1.2 编码规则（host 端 `blockfloat_common.cpp`）

- **块 = 16 = 一个 face 的一行**，取这 16 个元素的 `max(exponent)` 作共享阶码。
- 归一化 → **round-to-nearest-even**（guard bit 判 tie），也可切换为纯截断（`truncate_bfp_mantissa`）。
- **溢出后不回退**：向上舍入溢出时尾数钳到全 1，而不重新计算块阶码并重新归一化。
- 低于块阶码 $2^{e_{\max}-\text{有效位数}}$ 的值 → **直接下溢为 0**（bfp8 约 7 bit、bfp4 约 3 bit、bfp2 约 1 bit 的动态范围）。
- 无 NaN/Inf；`-0` 保留；`exp==0` 归零。`a` 变体的阶码必须落在 5 bit 内，否则 WH/BH 走各自的 `NonContractualBehavior`。

## 1.3 存储布局与体积（32×32 tile）

块阶码**前置**为连续区（每 16 元素 1 字节，共 $1024/16=64$ B，16B 对齐），之后是尾数区：

```
┌────────────── bfp8_b tile = 1088 B ──────────────┐
│ 64 B 阶码 (4 face × 16 行) │ 1024 B 尾数 (1 B/元素) │
└──────────────────────────────────────────────────┘
```

| 格式 | tile 字节数 | vs bf16 (2048 B) | 组成 |
|---|---|---|---|
| `bfp8_b` | 1088 | 53.1% | $256\times4 + 16\times4$ |
| `bfp4_b` | 576 | 28.1% | $128\times4 + 16\times4$ |
| `bfp2_b` | 320 | 15.6% | $64\times4 + 16\times4$ |

> 阶码区固定 64 B，不随位宽缩小 → bfp2 有约 20% 存储开销在阶码上。

## 1.4 在数据通路中的位置（重要）

**BFP 只存在于 L1 与寄存器之间，只是为了让数据更紧凑，FPU 从不会直接处理 BFP。**

| BFP | unpacker 展开为 | Src 风格 |
|---|---|---|
| `Bfp8/4/2_b`（`BFP8/4/2`） | **BF16** | 8 bit 阶码路径 |
| `Bfp8/4/2`（`BFP8a/4a/2a`） | **FP16** | 5 bit 阶码路径 |

→ 所以 **BFP 的“精度”完全由展开后的 bf16/fp16 承载**，而 bfp 的 7/3/1 bit 有效位在 Src 里是“填 0 的假精度”。因此对 bfp2/bfp4 开 HiFi 完全是浪费（见第二节选型表）。

---

# 二、LoFi / HiFi2 / HiFi3 / HiFi4 的硬件位切片与运算差异

## 2.1 乘法器硬件：5 bit × 7 bit 定点

ISA 原文：*“The hardware multipliers consume at most **5 bits of `SrcA`**, **7 bits of `SrcB`**”*，等价于：

- SrcA：5 bit 定点、**4 个小数位**（值域 $[1,2)$）
- SrcB：7 bit 定点、**6 个小数位**（值域 $[1,2)$）

一个 2 bit 的 `FidelityPhase` 字段（在 RWC 里，`FidelityPhase = RWC.FidelityPhase + FIDELITY_BASE`，`&3`）决定取**哪一段**：bit0 选 SrcA 的哪 5 bit，bit1 选 SrcB 的哪 7 bit。

### 精确掩码（fp32 视图，`m9..m0` = 10 bit 尾数）

| Phase | SrcA 掩码 / 位 | SrcA 乘法器值 | SrcB 掩码 / 位 | SrcB 乘法器值 | 指数补偿 |
|---|---|---|---|---|---|
| **0** | `x &= 0xfff80000`（保留 31..19） | `0x10 \| (m9..m6)`，4 frac | `x &= 0xfffe0000`（保留 31..17） | `0x40 \| (m9..m4)`，6 frac | 0 |
| **1** | `x &= 0xfff83fff` 后 `x - bits.f` → 位 18..14 | `(m5..m1)`，5 bit | 同 phase 0 | `0x40 \| (m9..m4)` | $+5$ |
| **2** | 同 phase 0 | `0x10 \| (m9..m6)` | `x &= 0xfffe1fff` 后做差 → 位 16..13 | `(m3..m0)<<3`，4 bit | $+7$ |
| **3** | 同 phase 1 | `(m5..m1)` | 同 phase 2 | `(m3..m0)<<3` | $+5+7$ |

`PhaseAdjust = (bit0?5:0) + (bit1?7:0)`（因为切出来的低段是“整数”，要按小数位重新定标）。

### 四相位的位覆盖

<details open>
<summary><b>BF16 风格（bf16，以及 bfp8/4/2 → BF16）</b></summary>

| phase | SrcA（5b） | SrcB（7b） |
|---|---|---|
| 0 | 隐含 1 + $m_9..m_6$ | 隐含 1 + $m_9..m_4$ |
| 1 | 剩余 3 位 $m_5..m_3$ | 隐含 1 + $m_9..m_4$ |
| 2 | 隐含 1 + $m_9..m_6$ | 剩余 1 位 $m_3$ |
| 3 | 剩余 3 位 $m_5..m_3$ | 剩余 1 位 $m_3$ |

A 共 $4+3=7$ bit、B 共 $6+1=7$ bit → **四相位 = bf16 的完整叉积**。
</details>

<details>
<summary><b>TF32 / FP16 风格（fp16、tf32，以及 bfp*a → FP16）</b></summary>

| phase | SrcA | SrcB |
|---|---|---|
| 0 | 隐含 1 + $m_9..m_6$ | 隐含 1 + $m_9..m_4$ |
| 1 | $m_5..m_1$（5 bit） | 隐含 1 + $m_9..m_4$ |
| 2 | 隐含 1 + $m_9..m_6$ | 剩余 4 位 $m_3..m_0$ |
| 3 | $m_5..m_1$ | 剩余 4 位 $m_3..m_0$ |

共 $4+5=9$ bit → **SrcA 的 $m_0$ 任何相位都不消费**，所以 ISA 写 “full precision not possible for `SrcA`”。
</details>

## 2.2 为什么矩阵乘要分 Phase 迭代？

<details open>
<summary><b>5 条设计动机</b></summary>

1. **面积/功耗/时序**：一个 FPU 有 $8\times16$ 个 datum 位置 × 2 lane × 8 乘法器 = **2048 个乘法器**，要跑 4096 MAC/cycle。如果把它做成“全精度 fp16 11b×11b”，乘法器阵列与对齐移位网络面积会爆炸；做成 **5b×7b 定点 + 10 bit 小数累加**，面积/能耗都便宜，同时**精度变成可调参数**。
2. **一个数据通路同时服务 int8**：phase 机制对整数同样适用（按“量级位”切，`abs(SrcB)≤15` 只需 phase 3 等），所以 int8 和 fp16/bf16 共用同一套乘法器，靠相位切换复用。
3. **精度换吞吐是线性的、零控制开销**：相位只是 `addr_mod.fidelity.incr` 让 RWC 自增，**每条 `MVMUL` 只做一段**，重发同一个 `MVMUL` N 次即可。LLK 侧就是 `inner_loops = to_underlying(MathFidelity)` + 一段 replay buffer（`{LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4}`，LoFi 走 `inner_loops=1`），没有任何额外配置或专用窄/宽通路。
4. **乘积本身是精确的**：切片不引入乘法误差，误差只来自“每个 phase 各自向 Dst 舍入一次”。因此“相位 = 部分积的分批累加”，天然与 `Dst +=` 语义契合（`MVMUL` 本身就是累加器）。
5. **避免先归一化到统一指数**：块内/跨 lane 的对齐已经由固定点 lane 累加器做了 10 bit 对齐，乘法器只需吐出“尾数片段 × 尾数片段”，让指数由 `ExpSum = a.Exp + b.Exp` 单独算 —— 位切片比宽乘法器更容易做这种“指数与尾数解耦”。

**代价**：每条相位都要向 Dst 回写并舍入一次，所以 **IPC 随相位线性下降**（"one instruction per fidelity phase"），`MVMUL` latency 5 cycle。
</details>

## 2.3 四档 MathFidelity：覆盖与选型

`MathFidelity : {LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4}`，映射到相位集合：

| 模式 | 值 | 相位集合 | 每 face-pair 的 `MVMUL` 数 | 覆盖 / 缺失项 |
|---|---|---|---|---|
| **LoFi** | 0 | {0} | 1 | A 只 5 bit（$m_9..m_6$）、B 只 7 bit → bf16 丢 3 个 A-LSB |
| **HiFi2** | 2 | {0,1} | 2 | A 全位、B 仅高 6+1 → bf16 丢 1 个 B-LSB |
| **HiFi3** | 3 | {0,1,2} | 3 | A、B 全位，**只缺 $A_{lsb}\!\times\!B_{lsb}$**（$2^{-10}$ 量级） |
| **HiFi4** | 4 | {0,1,2,3} | 4 | bf16 完全精确；tf32/fp16 受 SrcA $m_0$ 丢弃限制 |

**ISA 官方推荐表 → 直接对应选型：**

| 输入数据类型 | 最小精度 | “合理”精度 | 全精度 |
|---|---|---|---|
| FP8 / BFP4 / BFP2（含 a） | phase 0 | phase 0 | **phase 0** |
| **BFP8**（含 a） | phase 0 | phase 0 | **phase 0 + 1** |
| BF16 | phase 0 | phase 0 + 1 | **四相位** |
| TF32 / FP16 | phase 0 | 四相位 | SrcA 不可能 |

| 你的权重/激活格式 | 该用什么 |
|---|---|
| `bfp2_b` / `bfp4_b` | **LoFi**（1 bit / 3 bit 有效位，HiFi 全是浪费） |
| `bfp8_b` | LoFi（快，A 截到 5 bit，~3% 误差）～ **HiFi2**（真正无损） |
| bf16 | HiFi2 / **HiFi3**（性价比最优）/ HiFi4 |
| fp16 / tf32 | **HiFi4** |

> ⚠️ 两个高频坑：
> ① **HiFi4 必须配 `fp32_dest_acc_en`**，否则每个相位的部分积各自被 16-bit Dst（bf16/fp16）截断，收益反而消失；此时 `HiFi2/3 + fp32 acc` 更快更准。
> ② **bf16 下 HiFi3 ≈ HiFi4**（只少 $2^{-10}$ 交叉项），但吞吐高 33%。

## 2.4 吞吐对比（WH / BH）

<details open>
<summary><b>每 FPU 吞吐（结构 WH/BH 相同；下表为 1 GHz 值）</b></summary>

| 指令 | 1 phase（LoFi） | 2 phase（HiFi2） | 3 phase（HiFi3） | 4 phase（HiFi4） |
|---|---|---|---|---|
| `MVMUL` | **4.096 TF** | 2.048 TF | 1.366 TF | 1.024 TF |
| `DOTPV` | 4.096 | 2.048 | 1.366 | 1.024 |
| `GAPOOL`（reduce avg/sum） | 2.048 | 1.024 | 0.683 | 0.512 |
| `MVMUL`（`BroadcastSrcBRow`） | 0.560 | 0.280 | 0.187 | 0.140 |
| `ELWMUL` | 0.256 | 0.128 | 0.085 | 0.064 |
| `GMPOOL`(max) | 0.256 | 0.256 | 0.256 | 0.256 |
| `ELWADD/ELWSUB`（`AddDst=1`） | 0.256 | 0.256 | 0.256 | 0.256 |
| `ELWADD/ELWSUB`（`AddDst=0`） | 0.128 | 0.128 | 0.128 | 0.128 |

- 规律：$\text{TFLOPS} = 4.096 / N_{\text{phase}}$；`GMPOOL`/`ELWADD`/`ELWSUB` **不吃 fidelity**（恒单相位）。
- 单指令 **IPC = 1/phase，latency = 5 cycle**；整芯片吞吐 = 上表 × tile 数（WH：n150 = 72、n300 = 2×64、Galaxy = 32×80）。
- **WH vs BH**：单 FPU 的“相位-吞吐”关系两代一致（`MatrixUnitArithmetic.md` 以 BH 为基准、WH 只列差异）；差异在**时钟域、tile 数**，以及 unpacker 侧带宽（见下）。
</details>

<details>
<summary><b>BFP 端到端的真实瓶颈：unpacker</b></summary>

FPU 再快也可能被 unpacker（16 B/cycle × 速度档）卡住：

| 项 | Wormhole B0 | Blackhole |
|---|---|---|
| `UnpackRowWidth` | 统一 16 datum/row | $\le$8-bit 走 **32 datum/row** |
| throttle 档位 | x1/x2/x4（x8 非法） | x1/x2/x4/**x8（仅 8-bit 输入安全）** |
| `BFP2/BFP2a` | **永远 x1（16 B/cycle）** | **永远 x1** |
| `BFP4/BFP4a` | x2/x4（有跨 16-exp 组约束） | **默认 x4**（会改变部分 BFP4a 结果） |
| `BFP8` | 可 x4 | 默认 **x8** |

→ 结论：**`bfp2` 的实际吞吐瓶颈在 unpacker（x1 强制），不在 FPU**；这也是 bfp2 只适合“极小权重 + 低带宽”场景的原因之一。
</details>

## 2.5 累加与舍入的副作用（fidelity 之外的真实差异）

- lane 内 8 个乘积**对齐到该 lane 最大指数后固定点求和**，保留 **10 bit 小数**（`Fp32_enabled` 时 23 bit）；lane 间与 Dst 项再对齐、**只舍入一次**。
- 舍入：round-to-nearest，tie **远离 0**；仅 lane→公共指数对齐那步 tie **朝 $+\infty$**。
- 无 denormal（阶码 0 即 0）；Inf/NaN 无特殊语义（阶码 255 当普通大指数）。
- **每个 phase 单独向 Dst 舍入一次** → 这是“多相位总和精度”的天花板由 Dst 决定的原因。
- 16-bit Dst 下还有隐藏的 `DstLow` 13 bit 低位寄存器，混用 16/32-bit 累加会偶发 1 ULP 抖动。

## 2.6 WH / BH 在 fidelity 路径上的差异（与上面直接相关）

| 项 | Wormhole B0 | Blackhole |
|---|---|---|
| `TruncateSrc` | **有**：每 datum 出 Src 时截断 —— `FP16/FP8/BFP8a/INT8` 用 `&=0x7FF1F`（丢高 3 阶码位）；`BFP4a/BFP2a` 用 `&=0x7F81F`；`TF32/BF16` 用 `&=0x7F8FF`（丢低 3 尾数位）→ **WH 上 TF32 实际只剩 7 bit 小数** | **无截断**，乘积精确 |
| 隐含 Src 格式 | 无 | 有（unpacker 写 `ImpliedSrcAFmt/BFmt`，可覆盖 `ALU_FORMAT_SPEC_REG`） |
| 8-bit 浮点 | 仅 E5M2 | E5M2 **或 E4M3**（`Unp_LF8_4b_exp`，即 tt-metal `Fp8_e4m3`） |
| `Acc == -1` 归一化 | 有 erratum（会归一化错成 $-16$） | 已修复 |

---

## 一句话总结

> **BFP 用“每 16 元素共享阶码、元素只留 7/3/1 bit 显式含 1 的尾数”把带宽压到 bf16 的 53%/28%/16%，代价是块内动态范围只剩 7/3/1 bit；LoFi/HiFi2/3/4 则是同一套 5b×7b 定点乘法器的位切片重放（1/2/3/4 个相位），吞吐按 $4.096/N$ 线性衰减，而 bfp 格式的“足够 fidelity”几乎总是 LoFi（bfp8 想要无损才升到 HiFi2）。**

来源：`tt-isa-documentation` 的 `SrcASrcB.md` / `MVMUL.md` / `MatrixUnitArithmetic.md` / `UNPACR_Regular.md`，`tt-metal/tt_metal/impl/data_format/blockfloat_common.cpp`、`api/tt-metalium/tt_backend_api_types.hpp`、`tech_reports/matrix_engine/matrix_engine.md`，`tt-llk/*/llk_lib/llk_defs.h`、`llk_math_matmul.h`（均已通读）。

---
## 🧑‍💻 User

所以高精度×低精度
高最好放B
是吗

---
## 😀 默认助手 (7)

# 简短回答

**在 LoFi 下“把高精度操作数放在 `SrcB`”确实成立，但它只在「单相位」成立，并非通用规律；换成 HiFi2 就会反过来。**

原因在于：**哪个操作数更宽，会随 fidelity 档位翻转。** 下面按模式列出两个操作数的位宽（bf16 风格，含隐含 1 的有效位数）：

| 模式 | 相位集合 | `SrcA` 有效位 | `SrcB` 有效位 | 谁更宽 |
|---|---|---|---|---|
| **LoFi** | {0} | 5（1+4） | **7（1+6）** | **B 宽 2 bit** |
| **HiFi2** | {0,1} | **8（1+4+3，完整 bf16）** | 7 | **A 宽 1 bit** |
| **HiFi3** | {0,1,2} | 8 | 8 | 对称（只缺 $A_{lsb}B_{lsb}$） |
| **HiFi4** | {0,1,2,3} | 8 | 8 | 完全对称 |

fp16/tf32 风格同理，但 `SrcA` 的 $m_0$ 永远不会被使用：

| 模式 | `SrcA` | `SrcB` |
|---|---|---|
| LoFi | 5 | **7** |
| HiFi2 | **10** | 7 |
| HiFi3 / HiFi4 | 10 | **11** |

---

# 分场景判断

<details open>
<summary><b>① 低精度 ≤ 5 bit（bfp4 / bfp2 / 小范围 int8）× 高精度 bf16</b></summary>

**是的，把高精度操作数放进 `SrcB` 最优。**

| 方案 | A 侧误差 | B 侧误差 | 总误差 |
|---|---|---|---|
| 低精度→A，**高精度→B** | 0（$3\le5$ bit 装得下） | 8→7 bit，$\le2^{-6}\!\approx\!1.6\%$ | **1.6%** |
| 高精度→A，低精度→B | 8→5 bit，$\le2^{-4}\!\approx\!6.3\%$ | 0 | 6.3% |

LoFi 单相位就够，这也是 bfp4 权重 + bf16 激活的推荐组合。
</details>

<details open>
<summary><b>② 反例：bfp8（7 bit）× bf16（8 bit）—— “高放 B”反而次优</b></summary>

**LoFi 下两种放法都会让一个操作数被截到 5 bit，误差都约 6.3%，无法通过“放 B”改善。**

但升到 **HiFi2** 后，最优配对恰好相反：

| 方案 | A（相位 0+1 = 1+4+3） | B（相位 0 = 1+6） | 结果 |
|---|---|---|---|
| **bfp8→B，bf16→A，HiFi2** | bf16 8 bit → **完整** | bfp8 7 bit → **正好全是 7 bit，无损** | ✅ **两侧都无损** |
| bf16→B，bfp8→A，HiFi2 | bfp8 7 bit → 9 bit 装得下，无损 | bf16 8→7 bit，丢 1 bit | 1.6% |

> 关键几何匹配关系：**`SrcA` 的 HiFi2 总宽 = 8 bit = bf16；`SrcB` 的 LoFi 总宽 = 7 bit = bfp8。** 把“位宽刚好匹配”的操作数放到对应的槽位，就能实现双无损。

所以对最常见的 `bf16 激活 × bfp8 权重`：**bf16 放 A、bfp8 放 B、开 HiFi2** —— 与“高放 B”相反。
</details>

<details>
<summary><b>③ HiFi3 / HiFi4：不需要纠结</b></summary>

两侧都完整（bf16 风格），唯一缺失的 $A_{lsb}\times B_{lsb}$ 交叉项（$\sim2^{-10}$）**对称**，交换后结果不变。fp16/tf32 风格下 `SrcA` 恒定丢 $m_0$（$2^{-10}$），所以极微小地偏向把关键操作数放 B，但已无工程意义。
</details>

---

# 两个必须补充的实践约束

<details>
<summary><b>① 不能自由交换：`MVMUL` 结构固定了角色</b></summary>

`Dst[8,16] += SrcB[8,16] @ SrcA[16,16]`：

- **`SrcB` = $m\times k$（8 行）**，对应**激活 / 输出行侧**；还支持 `BroadcastSrcBRow`（1×16 广播，但吞吐掉到 0.560 TF，**比 4.096 TF 慢 8 倍**，别把关键操作数塞进广播路径）。
- **`SrcA` = $k\times n$（16×16）**，对应**权重 / 列侧**。
- tt-llk 里固定为 `in0 → SrcB`、`in1 → SrcA`。想交换必须做**转置算子**（重新 tilize、换 pack/unpack 路径，还要撞上 `transpose` addr_mod 的限制与 `is_in1_32x16` 的 assert）。

→ 所以“谁放哪”更多是**算子布局决定的，可调空间有限**。
</details>

<details>
<summary><b>② 精度截断是“单边截断”（有偏），不是四舍五入</b></summary>

fidelity 掩码就是纯 `x &= mask`（phase 1 才做 `x - bits.f` 取残差），因此：

- LoFi 下 `SrcA` 截到 4 个小数位 → **只能向下截断**，相对误差 $[0,\,2^{-4})$，即 $0\sim6.25\%$，且**系统性偏小**（乘积整体偏低）。
- `SrcB` 截到 6 个小数位 → $[0,\,2^{-6})$，$0\sim1.56\%$。

→ 这也是 **bf16×bf16 不该停在 LoFi** 的原因：单边 3%~6% 的偏差比 HiFi2 的 1.6% 噪声更难校准。
</details>

---

# 一句话决策规则

> **看“每档位下两个槽的宽度表”，把“需要更多有效位”的操作数放进更宽的那个槽；若两边都必须损失，优先让更敏感的那侧进宽槽。**
>
> 具体到常见配置：
> - `bf16 × bfp4/bfp2` → **高精度 bf16 放 B**，LoFi 即可（✓ 你的直觉）
> - `bf16 × bfp8` → **bf16 放 A、bfp8 放 B、开 HiFi2**（❌ 与直觉相反，但双无损）
> - `bf16 × bf16` → **HiFi2 起**（A 完整、B 丢 1 bit）或 HiFi3（几乎无损）
> - `fp16/tf32 × 任意` → HiFi4，且关键侧放 B

**最后强调：把档位从 LoFi 提升到 HiFi2，收益可达 3 bit；仅交换操作数位置，最多只有 2 bit。所以先选对档位，再考虑摆放。**

---
## 🧑‍💻 User

要分「你打算跑 LoFi（1 拍）还是标准 HiFi2（2 拍）」来看，这里正好有个超有意思的反转！

1. 如果跑 LoFi（单周期，追求 100% 吞吐）：
   完全正确，「高放 SrcB (7b)、低放 SrcA (5b)」最赚！
   因为 Phase 0 是 SrcA[高 5b] × SrcB[高 7b]。比如 BFP4 (3b) × BFP8 (7b) 或 BF16 (8b)：
   - 低精度 BFP4 (3b) 放 SrcA (5b)：3 ≤ 5，一比特都不丢；
   - 高精度 BFP8 (7b) / BF16 (8b) 放 SrcB (7b)：BFP8 完整吃满 7b，BF16 也吃到了 7/8b。
   只花 1 个周期就几乎拿满精度；反过来如果把高精度塞进 SrcA，LoFi 下直接被砍成 5b，血亏。

2. 但如果跑标准 HiFi2（2 周期，想补齐 >7b 的高精度尾数，比如 BF16 的第 8b 或 FP16 的 11b）：
   结论会反过来，变成「高放 SrcA (5+5=10b)、低放 SrcB (7b)」更划算！
   因为硬件固定的第 2 拍（Phase 1）是去切 SrcA 的低 5b（SrcA[低 5b] × SrcB[高 7b]），而不是切 SrcB 的低 4b：
   - 若高（FP16 11b / BF16 8b）在 SrcA，低（BFP8 7b / BFP4 3b）在 SrcB：2 个周期算完 10b × 7b，第 2 拍实打实补上了高精度操作数的低位；
   - 若低（BFP4 3b）在 SrcA，高（FP16 11b / BF16 8b）在 SrcB：由于 SrcA 的低 5b 全是 0，第 2 拍（HiFi2）完完全全是乘 0 空转！必须挨到第 3 拍（HiFi3 的 Phase 2）才去乘 SrcB 的低 4b（除非像 tt-metal#39627 那样改 LLK 微码，跳过 Phase 1 直接跑 Phase 0 + Phase 2）。

3. 顺带一提硬件接线的小细节：
   Tensix 底层的 MVMUL 指令实际上算的是 Dst[8×16] += SrcB[8×16] × SrcA[16×16]，所以在 TTNN 写 C = A_act @ B_weight 时，硬件默认就是把左矩阵 A_act 喂给 SrcB（7b）、把右矩阵 B_weight 喂给 SrcA（5b）——这正是为了迎合「BFP4 权重 (SrcA) × BF16/BFP8 激活 (SrcB)」在 LoFi 下一拍封神而设计的

继续补充

---
## 😀 默认助手 (7)

你的三点判断我基本认同——尤其第 2 点抓住了最关键的不对称。下面补充**校准、量化、以及几个你没提但同等重要的杠杆**。

---

# 一、先校准两个细节

<details open>
<summary><b>① “5+5=10b” 只对 FP16/TF32 风格成立，BF16 风格是 5+3=8b</b></summary>

Phase 1 切 `SrcA` 的“剩余位”，但它**剩余多少位取决于 Src 风格**（由 unpacker 展开后的格式决定）：

| Src 风格 | A phase0 | A phase1 | A 总计 | B phase2 | B 总计（含 phase 0） |
|---|---|---|---|---|---|
| **BF16 风格**（bf16、`bfp8/4/2_b`） | 1+4 = 5 | **+3** | **8**（bf16 刚好装满） | +1 | 8 |
| **FP16/TF32 风格**（fp16、tf32、`bfp*a`） | 1+4 = 5 | **+5** | **10**（$m_0$ 永丢） | +4 | 11 |

所以你说“HiFi2 下高精度放 SrcA（5+5=10b）”，**在 bf16 输入下实际是 5+3=8b**；10b 那个数字来自 fp16/tf32 路径。这个区别直接影响选型：**bf16×bf16 在 HiFi2 下是“A 完整、B 缺 1 bit”**，不是“10b 对 7b”。

</details>

<details>
<summary><b>② “为了迎合 BFP4 权重放 SrcA 而设计”这个因果建议弱化为“容量恰好匹配”</b></summary>

ISA 记录的动机是乘法器面积/时序（“consume at most 5 bits of SrcA, 7 bits of SrcB”），`SrcA`/`SrcB` 的角色分工来自 `Dst[8×16] += SrcB[8×16] @ SrcA[16×16]`：**SrcB 是 $m\times k$（8 行，随输出行走），SrcA 是 $k\times n$（16×16，被每个 row-block 复用）**。

两种解读都能自洽：
- “复用侧给窄乘法器” → 面积省在关键路径上；
- “容量匹配” → 5b 槽装 bfp4、7b 槽装 bfp8，一拍刚好。

结论上你说的效果成立，但**因果别写进结论里**（容易被反问“那为什么不是反过来”）。另外 `in0 → SrcB`、`in1 → SrcA` 是 **LLK 约定**，`ttnn.matmul(a, b)` 里 in0=a，所以激活→SrcB；但 `ttnn.linear`、`transpose_a/b`、以及 pre-transposed 权重会改变这个对应关系，**“左矩阵→SrcB”不是硬件强制的**。

</details>

---

# 二、最重要的一条：Phase 序列是**软件可编程**的，不是硬编码 0..N

这是你第 2 点里“{0,2} 补 B”能成立的根本依据，值得写清楚：

| 机制 | 位置 | 作用 |
|---|---|---|
| `RWCs[thread].FidelityPhase` | RWC 里的 **2-bit** 字段 | 当前 `MVMUL` 用哪个相位 |
| `ThreadConfig.FIDELITY_BASE_Phase` | 线程配置 | 相位基准偏移（`(RWC + base) & 3`） |
| `addr_mod.fidelity.{incr, clr}` | `ADDR_MOD_5 / ADDR_MOD_6` | **每条 MVMUL 后相位自增 / 清零** |

→ 所以 LLK 可以排**任意子集、任意顺序**的相位序列，两个相位的部分积相加在数学上总是合法的 partial sum。于是 2 拍其实有**两种**取法：

| 相位套餐 | 有效位宽（BF16 风格） | 等效档位 | 每 FPU 吞吐 | 适用 |
|---|---|---|---|---|
| {0} | A 5 × B 7 | LoFi | **4.096 TF** | bfp4/bfp2 低精度侧在 A |
| **{0,1}** | A 8 × B 7 | HiFi2 | 2.048 TF | 想补 **A** 的尾数 |
| **{0,2}** | A 5 × B 8 | “HiFi2-B”（标准档位里没有） | 2.048 TF | 想补 **B** 的尾数 |
| {0,1,2} | A 8 × B 8 | HiFi3 | 1.366 TF | 双完整（差 $A_{lsb}B_{lsb}$） |
| {0,1,3} / {0,2,3} | 交叉组合 | 非标准 | 1.366 TF | 按数据分布定制 |
| {0,1,2,3} | A 8 × B 8 | HiFi4 | 1.024 TF | exact |

**你提到的 “Phase 0 + Phase 2” 就是上表第三行**——这在 ISA 层完全合法，缺的只是 tt-llk 里的 `inner_loops = to_underlying(math_fidelity)` 这一句把它写死了（相位序列被 `matmul_configure_mop` 的 replay buffer 固化，且 `is_high_fidelity()` 是 `constexpr`，换序列 = 重新 `_llk_math_matmul_init_` + 重编译 kernel）。

> 关于 **tt-metal#39627**：这个具体编号我无法核对（没有访问 issue 的通道）。但“改 LLK 微码跳过 Phase 1 改跑 Phase 0+2”在硬件语义上完全可行，需要动的三处是：① replay buffer 里的 `ADDR_MOD` 序列（phase 0 用“复位+incr”，phase 2 用“incr 到 2”或重新设 `FIDELITY_BASE`）；② 循环末尾 `CLR_A/CLR_B` 要落在最后一个相位上，且**把相位计数器复位回 base**（否则下一个 tile 会从相位 2 开始）；③ `inner_loops` 与 `THROTTLE_LEVEL` 分支（`ADDR_MOD_6` 那条路径也要同步）。

---

# 三、把“浪费一拍”的代价算成钱

你第 2 点说“第 2 拍完完全全是乘 0 空转”，这个代价可以精确量化：

| 项 | 代价 |
|---|---|
| FPU 吞吐 | **直接 −50%**（2.048 TF 买 LoFi 精度） |
| Dst 带宽 | 多一次 `Dst` 读-改-写（`Dst += 0`），功耗/带宽白烧 |
| 数值 | 基本无害：$+0$ 幂等（16-bit Dst 的 `DstLow` bit12 抖动边界情况除外） |

所以对比很清楚：

- **{0,1} + bfp4 在 A** = 2.048 TF 换来 0 bit 增益（纯浪费）
- **{0,2} + bfp4 在 A** = 2.048 TF 换来 B 侧 +1~4 bit（真收益 ~2 bit）
- **{0} + bfp4 在 A** = 4.096 TF，B 只 7 bit

→ 这也解释了为什么“**BFP4 权重不该配 HiFi2**”：要么 LoFi（4.096 TF），要么 {0,2}（2.048 TF），就是不要标准 HiFi2。

---

# 四、容量匹配速查表（选摆放就看这张）

| 槽位 | LoFi {0} | HiFi2 {0,1} | {0,2} | HiFi3 {0,1,2} | 刚好装满这个槽的格式 |
|---|---|---|---|---|---|
| **SrcA**（BF16 风格） | 5 | **8** | 5 | 8 | LoFi: bfp4(3)、bfp2(1)、\|int8\|≤31；HiFi2: **bf16(8)、bfp8(7)** |
| **SrcB**（BF16 风格） | 7 | 7 | **8** | 8 | LoFi/HiFi2: **bfp8(7)**；{0,2}/HiFi3: bf16(8) |
| SrcA（FP16 风格） | 5 | 10 | 5 | 10 | HiFi2: fp16/tf32（去 $m_0$） |
| SrcB（FP16 风格） | 7 | 7 | 11 | 11 | HiFi3/4: fp16/tf32 完整 |

### 由此得到一个**推翻我上一轮建议**的结论（对默认布局）

默认布局是 **权重→SrcA、激活→SrcB**。那么 `bfp8 权重 × bf16 激活`：

| 档位 | A=权重(7b) | B=激活(8b) | 评价 |
|---|---|---|---|
| LoFi | 截到 5b（丢 2 bit，≤6.3% **单边**误差） | 7/8 bit | 权重吃亏，**不建议** |
| **HiFi2** | **8b ≥ 7b → 权重无损** | 7/8 bit（丢 1 LSB, 1.6%） | ✅ **默认布局的最优解，不用转置** |
| {0,2} | 5b（权重丢 2 bit） | 8b 完整 | 权重吃亏，需转置才划算 |
| HiFi3 | 8b 无损 | 8b 无损 | 更准但 +33% 时间 |

**即：`bfp8_b` 权重在默认布局下天然落在 `SrcA`，切 HiFi2 就立刻从“5b”补齐到“8b 无损”——这是 WH/BH 上 bfp8 权重最舒服的配置，不需要任何转置操作。** 上一轮我说的“bf16 放 A、bfp8 放 B”是*布局可自由选择时*的最优解；默认布局已经等价地把 7b 权重放进了 HiFi2 的 8b 槽，效果一样。

---

# 五、几个会吃掉收益的瓶颈（比摆放更值钱）

<details open>
<summary><b>① unpacker 带宽：bfp2 的 FPU 优势恒被吃掉</b></summary>

| 项 | Wormhole B0 | Blackhole |
|---|---|---|
| `BFP2/BFP2a` throttle | **强制 x1（16 B/cycle）** | **强制 x1** |
| `BFP4/BFP4a` | x2/x4（受跨 16-exp 组约束） | 默认 **x4** |
| `BFP8` | x4 | 默认 **x8** |
| `UnpackRowWidth` | 16 datum/row | ≤8bit 走 **32 datum/row** |

算一笔账：一次 $32\times32\times32$ tile 乘法 = $32^3$ MAC = 32768 → LoFi **8 个 cycle**。而把一个 `bfp2_b` tile（320 B）以 x1 unpack 就要 **20 cycle**，两个操作数 40 cycle ≫ 8 cycle。
→ **bfp2/bfp4 必须靠“大 K 复用 + 多播”摊薄 unpack，单 tile-K 的小矩阵根本喂不饱 FPU**；此时纠结 LoFi/HiFi2 毫无意义。
</details>

<details>
<summary><b>② 多相位必须配 `fp32_dest_acc_en`，否则精度被 Dst 二次截断</b></summary>

ISA 精确模型：**每个相位是一条独立 `MVMUL`，各自向 Dst 舍入一次**。16-bit Dst 下 HiFi2 = 2 次 bf16 舍入（7 bit），fp32 Dst 下 lane 保留 **23 bit 小数**。
代价：Dst 容量减半（`SyncHalf` 下 bf16 8 tile → fp32 4 tile），影响 double-buffer 深度与 pack 时机——**这是 HiFi2/3/4 的隐藏成本，常被忽略**。
</details>

<details>
<summary><b>③ 单边截断（有偏），不是四舍五入</b></summary>

fidelity 掩码是纯 `x &= mask`（相位 1 才取残差），所以 LoFi 的 `SrcA` 截断误差是 $[0, 2^{-4})$，**系统性偏小**，不是对称噪声。多相位把它变成“求和后正好等于完整积”，所以相位越全，偏差越小。
（`StochRndType::{Fpu,Pack,All}` 理论上有随机舍入可去偏，但精确模型里标注为 `UnsupportedFunctionality`，即未表征，别依赖。）
</details>

<details>
<summary><b>④ `BroadcastSrcBRow`：别把激活塞进广播路径</b></summary>

`SrcB` 支持 1×16 行广播（matvec/激活广播），但吞吐 **LoFi 0.560 TF vs 4.096 TF（−86%）**。所以“LoFi 高放 B”的规则在广播模式下要重新权衡。
</details>

<details>
<summary><b>⑤ 元素级正好反过来：ELWADD/ELWSUB 完全不吃 fidelity</b></summary>

按 ISA 精确模型，`ELWADD/ELWSUB` 让 SrcA/SrcB **绕过乘法器**（视为 `SrcA * 1`、`1 * SrcB`，全精度），所以：
- `ELWADD / ELWSUB / GMPOOL(max)`：**fidelity 无影响**（恒单相位，0.256 / 0.128 / 0.256 TF）
- **`ELWMUL`**：有完全相同的 A/B 不对称（bit0→A、bit1→B）
→ “高放 B”的规则对 `ELWMUL` 同样适用，对 add/sub 无意义。
</details>

---

# 六、一个容易踩的坑：INT8 的相位语义是**反的**

浮点相位是“从 MSB 往 LSB 切”，**整数相位是从 LSB 往 MSB 切**（ISA 原文）：

| 整数量级约束 | 所需相位 |
|---|---|
| $\lvert SrcA\rvert\le31$、$\lvert SrcB\rvert\le15$ | **仅 phase 3** |
| $\lvert SrcB\rvert\le1023$ | phase 1 + 3 |
| $\lvert SrcA\rvert\le255$ | phase 2 + 3 |
| 任意 8-bit | 四相位全开 |

即：`MathFidelity` 的数值（LoFi=0 / HiFi2=2 …）**不能直接映射到 int8 的“精度档位”**——int8 的 `phase 3` 才是“最低位那 5 bit”，而 fp 的 `phase 0` 是最高位。所以同一个 `MathFidelity::HiFi2`（={0,1}）对 fp 是“补 A 尾数”，对 int8 却是“A 的中间 3 bit + B 的中间 6 bit”，语义完全不同。**混用 int8 与 fp 的 kernel 里别假设 fidelity 单调。**

---

# 七、结论：杠杆排序

> **1. 相位序列/档位（值 3~4 bit）> 2. 操作数摆放（值 1~2 bit）> 3. 格式本身差 1 bit（bfp8 vs bf16，值 1 bit 但省 2× 带宽）**
>
> 而在小 K / 低比特场景下，**0. unpacker 带宽（x1/x4/x8）才是真正的天花板**，榜单一出，上面三项都可能被淹没。

一句话收尾你的三点：
- **①LoFi 高放 B** ✅（且“低放 A”必须 ≤5 bit 才无损）
- **②HiFi2 高放 A** ✅（**前提是 fp16/tf32 风格；bf16 风格是 5+3=8b**），且真正的杀手锏不是换边，而是**把相位序列从 {0,1} 改成 {0,2}**
- **③`Dst += SrcB @ SrcA` 且 `in0→SrcB`** ✅（因果降级为“容量匹配 + LLK 约定”，且默认布局下 `bfp8→SrcA` + **HiFi2** 就已是无损最优解，无需转置）

---
## 🧑‍💻 User

假设不考虑范围
fp16@fp16跟bf16@bf16
谁的精度比较好？
lofi/hifi2来看
硬体允许fp16@bf16吗

---
## 😀 默认助手 (7)

# 一句话结论

> **在 LoFi / HiFi2 这两档，`fp16×fp16` 和 `bf16×bf16` 精度几乎完全一样（LoFi 逐位相同，HiFi2 都被 SrcB 的 6 个小数位卡住）；格式差异要等到 HiFi3/HiFi4 才体现。而 `fp16 @ bf16` 混用：硬件不支持，会静默算错。**

---

# 一、为什么 LoFi 下两者"逐位相同"

关键事实：**fidelity 掩码只按"小数位"定义，完全不看格式**。

| 掩码 | 保留位 | 含义 |
|---|---|---|
| SrcA phase 0 | `x &= 0xfff80000` | 符号 + 指数 + 隐含 1 + **4 个小数位** |
| SrcB phase 0 | `x &= 0xfffe0000` | 符号 + 指数 + 隐含 1 + **6 个小数位** |

这两个常数对 **BF16 风格和 FP16/TF32 风格是同一组**。差异只在"解码"（指数偏置 15 vs 127，属于**范围**，你已排除）。因此：

**LoFi = {phase 0} = A 载 5 bit 有效位 × B 载 7 bit 有效位 —— 与源格式无关。**

| | bf16 × bf16 | fp16 × fp16 |
|---|---|---|
| A 保留 | 1+4 = **5 bit**（丢 $m_5..m_3$，3 bit） | 1+4 = **5 bit**（丢 $m_5..m_0$，6 bit） |
| B 保留 | 1+6 = **7 bit**（丢 $m_3$，1 bit） | 1+6 = **7 bit**（丢 $m_3..m_0$，4 bit） |
| 乘积相对误差（单边截断） | $\le 2^{-4}\approx 6.25\%$ (A 主导) | $\le 2^{-4}\approx 6.25\%$ (A 主导) |

→ **精度一模一样**。fp16 多出来的 3~5 个尾数位在 LoFi 下**根本用不上**，被 5 bit 切片抹平。

---

# 二、HiFi2（{0,1}）下仍然基本平手

Phase 1 只补 SrcA，但**补多少取决于 Src 风格**：

| Src 风格 | A phase0 | A phase1 | A 总计 | B（phase0/2） |
|---|---|---|---|---|
| BF16 风格 | 1+4 = 5 | **+3**（$m_5..m_3$） | **8 = bf16 完整** | 7 / 8 |
| FP16/TF32 风格 | 1+4 = 5 | **+5**（$m_5..m_1$） | **10**（$m_0$ 永丢） | 7 / 11 |

| 档位 | bf16 × bf16 | fp16 × fp16 | 谁赢 |
|---|---|---|---|
| **LoFi {0}** | A 5 × B 7 → $\le 2^{-4}$ | A 5 × B 7 → $\le 2^{-4}$ | **平** |
| **HiFi2 {0,1}** | A **8（完整）** × B 7 → $\le 2^{-6}\approx1.56\%$ | A 10 × B 7 → $\le 2^{-6}$ | **平**（都被 B 卡住） |

**注意 HiFi2 的瓶颈已经转移到 SrcB**：A 这边 bf16 已完整、fp16 到 10 bit（$2^{-10}$，可忽略），两者都被 B 的 7 bit 限制在 $\le1.56\%$。所以在 HiFi2 下**加预算的唯一有效做法是 `{0,2}`（补 B）而不是换格式**：

| 2 拍方案 | bf16 风格 | fp16 风格 | 评价 |
|---|---|---|---|
| **{0,1}**（标准 HiFi2） | A 8 × B 7 | A 10 × B 7 | ✅ 两者都 $\le2^{-6}$，**最优 2 拍** |
| **{0,2}**（需改 LLK） | A 5 × B 8 | A 5 × B 11 | ❌ 被 A 的 5 bit 拖到 $\le2^{-4}$ |

→ 结论：**2 拍永远是 `{0,1}`，且两格式平手。想靠 `{0,2}` 在 fp16 上把 B 补到 11 bit 没意义，因为 A 只剩 5 bit。**

---

# 三、HiFi3 / HiFi4 才分化 —— 而且结论有点反直觉

| 档位 | bf16 × bf16 | fp16 × fp16 |
|---|---|---|
| HiFi3 {0,1,2} | A 8 × B 8，仅缺 $A_{lo}B_{lo}$ → $\sim2^{-12}$ | A 10 × B 11 → $2^{-10}$（$m_0$) |
| HiFi4 {0,1,2,3} | A 8 × B 8 → **数学上精确** | A 10 × B 11 → 仍 $2^{-10}$ |

这里必须分**两个口径**看，否则会得出相反结论：

<details open>
<summary><b>口径 A：硬件 fidelity 引入的额外误差（假设输入值都能被该格式精确表示）</b></summary>

| | LoFi | HiFi2 | HiFi3 | HiFi4 |
|---|---|---|---|---|
| bf16 × bf16 | $2^{-4}$ | $2^{-6}$ | $\sim2^{-12}$ | **0（exact）** |
| fp16 × fp16 | $2^{-4}$ | $2^{-6}$ | $2^{-10}$ | $2^{-10}$ |

→ **bf16 赢**。因为 bf16 的 7-bit 尾数（$m_9..m_3$）**正好被相位完整覆盖**（phase0 + phase1 = 8 bit），而 fp16 的 10-bit 尾数**永远差 $m_0$**（ISA 明写 "For SrcA, this leaves the one remaining least significant mantissa bit, which is not consumed by any fidelity phase"）。
这是设计上的硬限制：**fp16 在 Tensix 上永远拿不到 bit-exact 的 $11\times11$ 乘积**。
</details>

<details>
<summary><b>口径 B：端到端能承载多少有效位（格式本身的表示能力）</b></summary>

| | 尾数位 | 有效位数 | 表示误差（RNE） |
|---|---|---|---|
| bf16 | 7 | **8** | $2^{-8}\approx0.39\%$ |
| fp16 | 10 | **11** | $2^{-11}\approx0.05\%$ |

→ **fp16 赢 3 bit**。所以只要输入需要 >8 bit 有效位，bf16 在**输入阶段**就已经丢了（这属于"范围/表示"问题，但和精度直接相关）。
</details>

### 合并结论

| 你想要什么 | 答案 |
|---|---|
| **LoFi / HiFi2 下谁精度好** | **一样**（硬件切片决定，与格式无关） |
| HiFi3/4 下"硬件误差"最小 | **bf16**（可到 0），fp16 永远 $2^{-10}$ |
| HiFi3/4 下"绝对精度"最高 | **fp16**（$2^{-10} <$ bf16 的 $2^{-8}$ 表示误差） |
| 想要 bit-exact 的 matmul | **bf16 @ HiFi4**（fp16 做不到） |

**一句话：`fp16` 的天花板由"格式"决定（$2^{-11}$），但硬件只交付到 $2^{-10}$；`bf16` 的天花板由"格式"决定（$2^{-8}$），硬件可以交付 100%。所以在 LoFi/HiFi2 打平，在 HiFi3/4 取决于你更在意"等效位数"还是"是否精确"。**

---

# 四、`fp16 @ bf16` 硬件允许吗？——**不允许**

## 4.1 ISA 明确只列"同族"组合

`MVMUL` 支持的数据类型组合表（原文）：

| Dst | += | SrcB | @ | SrcA |
|---|---|---|---|---|
| FP32 / BF16 | | **TF32 或 BF16** | | **TF32（†）或 BF16** |
| FP32 / FP16 | | **FP16** | | **FP16（†）** |
| int32 | | int8 | | int8（‡） |

规则是 **"族对齐"**：`{TF32, BF16}` 一族（8-bit 指数 / BF16 风格）、`{FP16}` 一族（5-bit 指数 / FP16 风格）。**`fp16 × bf16` 落在两族之间，不在表内。**

## 4.2 机制原因（不是"精度降级"，是根本没法解释）

功能模型里这一行是关键：

```c
uint4_t SrcAFmt = ...ALU_FORMAT_SPEC_REG0_SrcA;
uint4_t SrcBFmt = ...ALU_FORMAT_SPEC_REG1_SrcB;

if      (SrcAFmt in {FP32, BF16, BFP8, BFP4, BFP2, INT32, INT16}) SrcAStyle = BF16;
else if (SrcAFmt in {FP16, FP8, BFP8a, BFP4a, BFP2a, INT8})       SrcAStyle = FP16;
else                                                              SrcAStyle = TF32;
```

随后 **两个操作数都用这同一个 `SrcAStyle` 解码**：

```c
case BF16: SrcBMatrix[i][j].f = SrcBFidelityBits(SrcDecodeBF16(SrcBVal), phase);
case BF16: SrcAMatrix[i][j].f = SrcAFidelityBits(SrcDecodeBF16(SrcAVal), phase);
```

三个致命细节：
1. **`SrcAStyle` 是"每条指令唯一"的值，且只由 SrcA 的格式决定** —— `SrcBFmt` 读进来后**除了 WH 的 `TruncateSrc` 之外根本没被使用**（模型里它是个未使用变量）。
2. `LaneControl.Exp8`（8-bit 还是 5-bit 指数偏置）也是**每指令一个布尔**，无法左 8 位、右 5 位。
3. `ELWADD/ELWSUB` 里的"$1$"操作数也是按 style 生成的：`One = {0, Exp8 ? 127 : 15, 0, false}` —— 同样只能二选一。

**失败模式**：若 A 配 bf16、B 配 fp16，unpacker 按 `WriteSrcFP16` 写出的 B（`Sign, Man(10), Zero(3), Exp(5)`）会被 `SrcDecodeBF16` 当成 `Sign, Man(10), Exp(8)` 读 —— 指数字段变成 `(0,0,0,exp4..exp0)`，即 $2^{-127}$ 量级 → **整片数值塌成 0 或极小的垃圾**，不是"精度差一点"。

## 4.3 合法的"混用"（你可能真正想找的）

| 你想要的组合 | 可行？ | 说明 |
|---|---|---|
| `tf32 × bf16` | ✅ | 同属 BF16 风格族（8-bit 指数） |
| `fp8(e5m2) × fp16` | ✅ | 同属 FP16 风格族 |
| `bfp8a × fp16`、`bfp4a × bf16`? | `bfp8a×fp16` ✅ / `bfp4a×bf16` ❌ | **`bfp*a` 展开成 FP16（FP16 风格），`bfp*_b` 展开成 BF16（BF16 风格）** |
| `bfp4_b(→bf16) × fp16` | ❌ | 跨族 |
| **`fp16 × bf16`** | ❌ | ISA 未列 |

**正确做法（三选一）**：
1. **在 unpack 阶段统一族**：unpacker 支持 `FP32 → FP16` 和 `FP32 → BF16` 转换（`FormatConversion` 里有两条独立分支），把源数据（fp32/bf16/fp16）在搬运时就转成同一族；
2. **用 BFP 的 a/b 变体"选族"**：想让数据走 FP16 风格就用 `bfp8/bfp4/bfp2`（无 `_b`），想走 BF16 风格就用 `bfp8_b/4_b/2_b`；
3. **Blackhole 的 implied format**：unpacker 交 bank 时自动把 `ImpliedSrcAFmt/BFmt` 写进 bank，FPU 直接用它（`DISABLE_IMPLIED_SRC*_FMT` 关掉则回落到 `ALU_FORMAT_SPEC_REG`）——这能自动避免"配置与数据不一致"这一整类 bug，但**仍然不能跨族**（style 依旧只看 A）。

## 4.4 一个额外的 WH 细节

Wormhole 的 `TruncateSrc` 对两族动的手不一样，但都无损：

| Src 格式 | WH 截断掩码 | 实际效果 |
|---|---|---|
| `FP16 / FP8 / BFP8a / INT8` | `x &= 0x7FF1F` | 丢指数高 3 位（本就为 0）→ **无损** |
| `TF32 / BF16` | `x &= 0x7F8FF` | 丢尾数低 3 位（bf16 本就为 0）→ **无损**；**但 TF32 会丢 3 个真位** |

→ 所以 **WH 上 TF32 的实际尾数只剩 7 bit（$\sim2^{-8}$），比 BH 差 3 bit**；这也让"tf32 × bf16"在 WH 上的优势被抵消一部分。

---

# 五、一页速查

| 问题 | 答案 |
|---|---|
| **LoFi 下 fp16 vs bf16 精度** | **完全一样**（同一组掩码 `0xfff80000` / `0xfffe0000`，A 5b × B 7b） |
| **HiFi2 下** | **一样**（都被 SrcB 的 7 bit 卡在 $\le2^{-6}$） |
| **HiFi3/4 下** | 硬件误差：**bf16 更小**（0 vs $2^{-10}$）；可表示精度：**fp16 更高**（11 vs 8 bit） |
| **要 bit-exact** | **bf16 @ HiFi4**（fp16 永远丢 $m_0$） |
| **`fp16 @ bf16`** | ❌ **硬件不支持**（`SrcAStyle`/`Exp8` 每指令唯一，只由 SrcA 决定；`SrcBFmt` 在模型里不被用于解码）→ 会静默算成垃圾，不是精度降级 |
| **允许的混用** | `tf32×bf16`、`fp8×fp16`、`bfp8a×fp16`（同族）；跨族必须在 unpack 阶段转换 |
| **2 拍的最优解** | **`{0,1}`**（补 A），而非 `{0,2}`；LoFi 与 HiFi2 在两种格式下都无差别 |

---
## 🧑‍💻 User

在 Wormhole B0 和 Blackhole 上，硬件【不允许】直接把 FP16（A 族，5-bit 指数）和 BF16（B 族，8-bit 指数）同时塞进 SrcA 和 SrcB 算矩阵乘！

原因特别底层：
1. 指数偏置（Bias）在 Unpacker 里不做跨族转换：
   虽然 SrcA / SrcB 寄存器每个元素都是 19-bit 宽（1b 符号 + 8b 指数槽 + 10b 小数槽），但 Unpacker 把 FP16 塞进 Src 寄存器时，只是把 5-bit 指数原封不动贴进低 5 位（高 3 位补 0，依然是 bias=15），并不会自动帮它加上 112（127-15）转成 BF16 的 bias=127。
2. FPU 要求两边同属一个指数族（Exponent Family）：
   矩阵引擎算乘法把两边指数相加时，要求 SrcA 和 SrcB 必须同为 5-bit 指数（A 族）或同为 8-bit 指数（B 族）。在 tt-llk 的 infer_math_format 里也有硬性检查：只要 a.is_exponent_B() != b.is_exponent_B() 就会直接报错拒绝。

这也正好解释了为什么 Tensix 的块浮点要大费周章分成 A、B 两套：
- A 族内部随便混（全部 5-bit 指数，bias=15）：比如 FP16 @ Bfp8、FP16 @ Bfp4；
- B 族内部随便混（全部 8-bit 指数，bias=127）：比如 TF32(由 FP32 截断) @ BF16、BF16 @ Bfp8_b、BF16 @ Bfp4_b。
如果既想要 10-bit 尾数又想跟 BF16 混乘，唯一办法是拿 FP32 喂给 Unpacker（会自动截断成 8-bit 指数 + 10-bit 尾数的 TF32 进入 Src 寄存器），这样就能合法地算 TF32 @ BF16

你去翻下 tt-llk 的寄存器手册逐条核实

---
## 😀 默认助手 (7)

让我去核实这两点。先找 `infer_math_format` 到底在哪个文件里。

`ckernel_defs.h` 直接验证了 A/B 两族的定义。继续找 `infer_math_format` / `is_exponent_B`。

核实完了。**你的三点大方向都对，但有 1 个机制描述需要升级、1 个函数名我找不到、2 个细节需要修正。** 逐条给证据。

---

# 一、核实结果总览

| 你的论断 | 判定 | 依据 |
|---|---|---|
| ① Unpacker 不做跨族 bias 转换，fp16 的 exp5 原样贴进低 5 位 | ✅ **完全正确** | ISA `WriteSrcFP16` + FP16 的 Src 位域 |
| ② "指数相加时要求两边同族" | ✅ **对，但理由比你说的更硬**：不是"要求"，而是**每条指令只有一个 bias 修正项**，数学上无解 | `LaneControl.Exp8` 单布尔 + `Bias = Exp8 ? -127 : -15` |
| ② 附：`infer_math_format` 里 `a.is_exponent_B() != b.is_exponent_B()` 硬检查 | ❓ **符号没找到** | 当前 main 的 `tt-llk/**/common/inc/` 无此符号；等效物见下 |
| ③ A/B 两族的划分（A=5-bit 指数族） | ✅ **正确，且代码里就是这么写的** | `IS_A_FORMAT()`，WH/BH 两代成员完全一致 |
| ④ "fp16 想跟 bf16 混乘只能走 TF32" | ✅ **结论对**，但漏了 WH 的精度折扣 | `TruncateSrc` 在 WH 上再砍 TF32 3 位尾数 |

---

# 二、逐条证据

## ① Unpacker 确实不 rebias —— 原文

```c
// ISA: UNPACR_Regular.md，Src 侧的写入函数
uint19_t WriteSrcFP16(uint16_t x) { return WriteSrcTF32(((x & 0x8000) << 3) | (x & 0x7fff)); }
```

fp16 的 16 bit 被**原样**摊进 19-bit Src：`Sign(1) | Man(10) | Zero(3) | Exp(5)` —— exp5 待在**低 5 位**，第 7..5 位补 0，**bias 依然是 15**，没有任何 `+112`。

而重偏置发生在**另一侧**：

```c
float SrcDecodeFP16(uint19_t x) {   // ALU 侧解码
    uint32_t Exp = (uint32_t(x & 0x1f) + (127 - 15)) << 23;   // ← +112 在这里
```

**→ 所以"bias 是数据属性"是误解，"bias 是指令属性"。** 这恰好解释了为什么 unpacker 不需要跨族转换：转换发生在 ALU 解码时，而不是 unpack 时。

## ② 真正的机制：每条指令**只有一个** bias 修正项

这是我要给你升级的地方。ISA 的精确模型里，乘积指数是这样算的：

```c
ExpSum[i] = a[i].Exp + b[i].Exp;                  // 两个"带偏置"的指数相加
...
int Bias = c.Exp8 ? -127 : c.Fp32 ? 127 - 30 : -15;   // ← 整个指令只能选一个!!!
uint10_t Exp = (MaxExpSum + Bias - PhaseAdjust) & 0x3ff;
if (Exp == 0 || Exp >= 512) return {0, 0};             // 下溢 → 0
```

`a.Exp + b.Exp` 里带了**两遍**偏置，所以必须减掉**一次**偏置；而这个 `Bias` 是**每条指令一个标量**，只取决于：

```c
bool Exp8;   // SrcAStyle is BF16 or TF32 (an 8-bit exponent), not FP16
```

而 `SrcAStyle` 又**只由 SrcA 的格式决定**：

```c
if      (SrcAFmt in {FP32, BF16, BFP8, BFP4, BFP2, INT32, INT16}) SrcAStyle = BF16;
else if (SrcAFmt in {FP16, FP8, BFP8a, BFP4a, BFP2a, INT8})       SrcAStyle = FP16;
else                                                              SrcAStyle = TF32;
// 然后两个操作数都用这同一个 style 解码：
case BF16: SrcBMatrix[i][j].f = SrcBFidelityBits(SrcDecodeBF16(SrcBVal), phase);
case BF16: SrcAMatrix[i][j].f = SrcAFidelityBits(SrcDecodeBF16(SrcAVal), phase);
```

> 顺带一个很能说明问题的细节：模型里 `SrcBFmt` 被读进来后**除了 WH 的 `TruncateSrc` 之外根本没被使用**（BH 处还带注释 *"Blackhole implied format behavior for SrcB not fully characterized"*）。

**所以"跨族"不是"违反规则"，而是"方程无解"**：一边 bias=15 一边 bias=127，无论 `Bias` 取 $-15$ 还是 $-127$，都必然有 112 的误差。

**失败模式（很具体）**：fp16 的 $1.0$（Exp=$15$）被按 BF16 解码 → `ExpSum = 15 + 127 = 142`，`Bias = -127` → `Exp = 15` → 按 bias-127 解读是 $2^{-112}$。**即每个乘积都被整体乘上 $2^{-112}$**，在 16-bit Dst 里**全部下溢成 0**（fp32 Dst 也只剩极小值）。**注意：硬件不报错、不 trap，只是静默算出全 0。**

## ③ A/B 两族在 LLK 里的定义 —— 找到了，而且两代一致

`tt_llk_wormhole_b0/common/inc/ckernel_defs.h` 与 `tt_llk_blackhole/common/inc/ckernel_defs.h` 里都有：

```c
constexpr static bool IS_A_FORMAT(std::uint32_t format) {
    switch (masked_data_format(format)) {
        case DataFormat::Lf8:
        case DataFormat::Float16:
        case DataFormat::Bfp8:      // 注意：无 _b
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
            return true;
        default: return false;      // ← 其余（Float16_b / Bfp8_b / Tf32 / Float32...）即 B 族
    }
}
constexpr static bool IS_BFP_A_FORMAT(std::uint32_t format)   // Bfp8 / Bfp4 / Bfp2
```

**这验证了你 A/B 族的划分，并且顺便钉死了两件事：**
- **`Bfp8`（无 `_b`）= A 族 = 展开成 FP16**；**`Bfp8_b` = B 族 = 展开成 BF16**。（我之前那套"后缀推断"现在有代码实证了。）
- **`Lf8` 也算 A 族** —— 这点你没提，但很实用：LF8 可以和 FP16 混。

## ④ 关于 `infer_math_format` / `is_exponent_B()` —— 我没找到

我在当前 main 分支逐文件看了 `tt_llk_{wormhole_b0,blackhole}/common/inc/` 下的 `ckernel_defs.h`、`cmath_common.h`、`llk_defs.h`、`llk_math_common.h`，**没有 `infer_math_format` 这个函数，也没有 `is_exponent_B()` 这个成员函数**（grep.app 被限流，无法做全库搜索）。

能核到的**等效强制**是这两个：
- `IS_A_FORMAT()` / `IS_BFP_A_FORMAT()` —— 族判定谓词（上面已引）
- `is_unpacker_format_conversion_supported_fp32_acc(src_fmt, dst_fmt, fp32_dest_acc_en)` —— **unpack 阶段 "src/dst 格式对是否合法" 的硬检查**，BH 的 `llk_unpack_common.h` 里 `LLK_ASSERT` 调的就是它

> 所以"硬性检查确实存在"这点成立，但**具体符号名/位置请你再核对**——很可能在 host 侧（tt-metal 的 op 合法性检查）、在旧版本分支、或在 `cunpack_common.h`（54 KB，我没整篇拉下来）里换了名字。我不想只凭印象给这句话盖章。

---

# 三、两个需要修正的细节

<details open>
<summary><b>修正 1：WH 上的 TF32 只有 7 bit 尾数，不是 10 bit</b></summary>

你说"FP32 喂给 Unpacker 会自动截断成 8-bit 指数 + 10-bit 尾数的 TF32"，这在 **BH 上对**，在 **WH 上不对**：

```c
// ISA: MatrixUnit.md 的 TruncateSrc（Wormhole only）
uint19_t TruncateSrc(uint19_t x, uint4_t Fmt) {
    if (Fmt in {FP16, FP8, BFP8a, INT8})      { x &= 0x7FF1F; }  // 丢高 3 阶码位（本来就 0）→ 无损
    else if (Fmt in {BFP4a, BFP2a})           { x &= 0x7F81F; }  // 同上 + 低 3 尾数位
    else if (Fmt != TF32)                     { x &= 0x7F8FF; }  // BF16 丢低 3 尾数位（本就 0）→ 无损
    return x;
}
```

`Fmt != TF32` 那条把 TF32 排除在外，但 **TF32 走的是 `SrcAStyle = TF32` 分支，仍要经过一个掩码**——按 `0x7F8FF` 的语义，TF32 的**低 3 位尾数会被打掉**。ISA 在 `MatrixUnitArithmetic.md` 里对 WH 差异的表述也印证了这一点：*"Every `Src` datum first goes through `TruncateSrc`"*，而 BH 是 *"Products are exact"*。

**结论：`tf32 @ bf16` 组合合法，但 WH 上 tf32 侧有效位 ≈ 7~8 bit（$\sim2^{-8}$），BH 上才是完整 10 bit（$\sim2^{-10}$）。** 所以在 WH 上想靠 TF32 拿 fp16 级尾数，收益打对折。

</details>

<details open>
<summary><b>修正 2：只有从 FP32 出发才能"改族"，bf16 tensor 无法在 unpack 阶段变成 fp16</b></summary>

ISA `FormatConversion` 的规则很严：**非 FP32 输入时，"Exactly one conversion is permitted... `OutDataFormat` must be equal to `InDataFormat`"**。也就是说 unpacker 的合法转换只有：

| InDataFormat | 允许的 OutDataFormat |
|---|---|
| `FP32` | `FP32` / `TF32` / `BF16` / `FP16` ← **唯一能跨族的入口** |
| `FP16` | 只能是 `FP16` |
| `BF16` | 只能是 `BF16` |
| `BFP8_b` | 只能 `BFP8_b`（内部固定 → BF16） |
| `INT8` | 只能 `INT8` |

**→ 想实现 "10-bit 尾数 × bf16"，必须先让数据以 FP32 形态躺在 L1 里**（4 B/元素，tile 4096 B = bf16 的 2 倍带宽），再由 unpacker 转成 `FP16` 或 `TF32`。直接喂 bf16 tensor 是转不动的。

**`FP32 → BF16` 分支还有个小坑**：会 `DatumBits &= 0x80000000` 把 denormal 冲成 0 再右移 16 位（**纯截断，不 RNE**）。

</details>

---

# 四、把"想混 10-bit 尾数"这件事说清楚

| 你想要的 | 可行路径 | 代价 |
|---|---|---|
| 10-bit 尾数 **×** bf16（B 族） | **FP32 → TF32**（B 族，1+10+8） | L1 存 fp32（2× 带宽）；**WH 上再丢 3 位尾数** |
| 10-bit 尾数 **×** bf16 | FP32 → FP16？❌ **不行**：FP16 是 A 族，又跨回去了 | — |
| fp16 **×** bf16 | ❌ 硬件无解（bias 方程无解） | — |
| fp16 **×** bfp8/bfp4/bfp2（A 族） | ✅ 直接可行 | bfp 的 7/3/1 bit 精度 |
| bf16 **×** bfp8_b/bfp4_b（B 族） | ✅ 直接可行 | 同上 |
| bf16 **×** bf16 | ✅ | — |

> ⚠️ **Blackhole 的 implied format 救不了跨族**：`ImpliedSrcAFmt/BFmt` 只是把"格式号"自动盖到 bank 上、省掉 `ALU_FORMAT_SPEC` 重配，**`SrcAStyle` 依旧只从 A 推导**。所以 A=bf16、B=fp16 时，BH 上"两边格式都对"，但 style 仍是 BF16 → B 依旧按 BF16 解码 → **还是全 0**。

---

# 五、纠正版速查表

| 组合 | 合法 | 关键陷阱 |
|---|---|---|
| `fp16 × fp16` / `fp16 × bfp8/bfp4/bfp2` / `fp16 × lf8` | ✅ | 全在 A 族 |
| `bf16 × bf16` / `bf16 × bfp8_b/bfp4_b/bfp2_b` / `bf16 × tf32` | ✅ | 全在 B 族 |
| **`fp16 × bfp8_b`** | ❌ | **最容易踩**：`bfp8` 是 A 族、`bfp8_b` 是 B 族，选错变体就跨族 |
| **`fp16 × bf16`** | ❌ | 乘积 × $2^{-112}$ → 静默全 0 |
| `fp16 × tf32` | ❌ | 跨族 |
| `fp8(e5m2) × bf16` | ❌ | `FP8` 属 A 族 |
| `int8 × fp16` | ❌（另一套模式） | `int8` 需 `ALU_ACC_CTRL_INT8_math_enabled`，不与浮点混 |

**一句话总结你的论断**：结论 100% 正确（`fp16 × bf16` 在 WH/BH 上都不可用），机制上把你的"要求同族"升级为 **"每指令单一 `Exp8` → 单一 bias 修正项 → 跨族无解"**；族定义有 `IS_A_FORMAT()` 代码实证；唯一没核实到的是 `infer_math_format` / `is_exponent_B()` 这两个符号名，以及别忘了 **WH 的 TF32 只有 7 bit 尾数**。
