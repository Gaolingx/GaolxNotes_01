# TensorCore128行划分为四乘三十二

## 🧑‍💻 User

为什么感觉tensorcore做128*128的矩阵乘法的时候像是把[128,K]组织成了[4,32,K]啊

---
## 😀 默认助手 (7)

## 一句话结论

你看到的 `[4, 32, K]` 不是显存里的物理排布，而是 **warp 级分区 + fragment 布局** 造成的逻辑视图：

- **4** = 沿 M 方向分给 **4 个 warp**（一个 warpgroup / 一个 SM 的 4 个 subpartition，各有一个 tensor core）；
- **32** = **每个 warp 负责 32 行**（$128 / 4 = 32$，也正好是 warp 的 32 个 lane）；
- **K** = 规约维，**不跨 warp 切**，所有 warp 都要完整走一遍 K。

所以对 A 来说，它就"像"被拆成了 4 个 $[32, K]$ 的子矩阵，而不是一整块 $[128, K]$。

---

## 1. 三层分解：从 CTA 到线程

以最常见的 Ampere/Ada `mma.sync.aligned.m16n8k16`（fp16）为例，A 的视角是层层往下切的：

```
A[128, K]
├── warp0 → rows   0..31   → A0[32, K]
├── warp1 → rows  32..63   → A1[32, K]
├── warp2 → rows  64..95   → A2[32, K]
└── warp3 → rows  96..127  → A3[32, K]        ← 这就是 [4, 32, K]
        │
        └── 每个 warp: 2 条 m16 指令（32 = 2 × 16）
              └── 每条 m16: 8 个 quad × 2 行 = 16 行
                    └── 每个 quad（4 lane）持有整行的 16 个 K 元素
```

关键点在最后一行：在 m16n8k16 的 A fragment 里，**一个 4-lane quad 拥有两条完整的行**，K 方向 16 个元素由 quad 内 4 个 lane 分摊。

<details>
<summary><b>m16n8k16 的 A fragment 索引公式（点开看）</b></summary>

令 `lane = 0..31`，`g = lane >> 2`（quad id，0..7），`t = lane & 3`（quad 内偏移，0..3）：

```
a0 → (row = g,     col = 2t)
a1 → (row = g,     col = 2t+1)
a2 → (row = g+8,   col = 2t)
a3 → (row = g+8,   col = 2t+1)
a4 → (row = g,     col = 2t+8)
a5 → (row = g,     col = 2t+9)
a6 → (row = g+8,   col = 2t+8)
a7 → (row = g+8,   col = 2t+9)
```

可以看到：
- **M（16 行）= 8 个 quad × 2 行**；
- **K（16 列）= 4 个 lane × 4 列**（每个 lane 拿 `2t, 2t+1, 2t+8, 2t+9` 四列）。

所以"4"这个数字在 fragment 层面也反复出现：**4-lane quad 是 tensor core 数据通路的最小粒度**。一个 warp 32 个 lane = 8 个 quad。

对照地，C 累加器同样满足 `row = g` / `g+8`、`col = 2t` / `2t+1`，B 则是 `row = 2t` / `2t+1`、`col = g`。

> 注：不同文档里 a0..a7 的寄存器顺序可能略有差异，但"quad 拥有两整行、K 由 4 个 lane 分摊"这个所有权模式是固定的。

</details>

<details>
<summary><b>为什么偏偏是 4 和 32（而不是 2、8）</b></summary>

- **4 个 warp**：Hopper 的 `wgmma` 就是以一个 **warpgroup = 4 个 warp = 128 线程**为单位；同时每个 SM 有 **4 个 subpartition**，每个 subpartition 一个 tensor core，4 个 warp 刚好铺满。
- **每 warp 32 行**：$128 / 4 = 32$，同时这也等于 warp 的 lane 数（32）。
- **单条指令 M 很小**：`mma.sync` 的 M 上限是 16，`wgmma` 是 64。128 行必须切成多份，不可能一个 warp 全包。
- **寄存器压力**：$[128, K]$ 的 A fragment 放不进一个 warp 的寄存器；B 则可以由 4 个 warp 共享广播。

</details>

---

## 2. 不同架构下，"4"对应不同的东西

这是最容易混淆的地方，取决于你用的指令：

| 架构 | 指令 | 发起单位 | 128 行的切法 | "像"什么 |
|---|---|---|---|---|
| Ampere/Ada | `mma.sync m16n8k16` | 1 warp | 4 warp × 32 行 | `[4, 32, K]` |
| Hopper | `wgmma m64nNk16` | 1 warpgroup（4 warp） | 每 warp 16 行，128 行要发两次 | `[4, 16, K]` ×2 |
| Blackwell | `tcgen05.mma` | 1 个线程 / 1 CTA | 128 行 → 128 条 **TMEM lane** | `[128, N]`（TMEM 视图） |

所以在 Ampere 风格的代码里你会强烈感到 `[4, 32, K]`；到了 Hopper 会变成 `[4, 16, K]`；Blackwell 上 M 直接映射到 TMEM 的 128 条 lane，那个 `[4,32]` 的"错觉"就基本消失了。

<details>
<summary><b>Hopper wgmma 的寄存器 A 布局</b></summary>

`wgmma.mma_async.m64nNk16.f16` 的 A 在寄存器里：每线程 4 个 `.b32` 寄存器（8 个 half），$128 \times 4 \times 2 = 1024 = 64 \times 16$。每个 warp 负责 16 行，4 个 warp 拼成 M=64。要算 128 行就发两次 wgmma（或开两个 warpgroup）。

</details>

---

## 3. shared memory 里也是分块的（core matrix）

`[4, 32]` 只解释了一半。A 从 shared memory 进寄存器时，还有一层 **core matrix** 视图：

- K-major 存储下，A 被切成 **8 行 × 16 字节**（fp16 即 8 个元素）的 core matrix；
- `ldmatrix.x4` 让 **32 个线程分成 4 组 × 8 线程**，每组负责一个 $8\times8$ tile → `[4 groups][8 rows][8 cols]`；
- Swizzle（32B/64B/128B）把同一行附近的 core matrix 交错放置，看起来行又被横着切了 4/8 份。

$128$ 行 = $16$ 个 core-matrix 行。所以如果你是在看 SMEM 的加载代码，"4×32"也可能来自 `ldmatrix` 的 4 组线程 / swizzle 的行分组。

---

## 4. 怎么在你自己的代码里验证

1. 看 CUTLASS 的 `MmaTensorOp` / `warp_shape`：128×128 tile + 4 warp，若 warp 排列是 4×1，则 warp 形状是 `[32, 128]`，A 就是 `[4, 32, K]`；若是 2×2，则每 warp 64 行，A 是 `[2, 64, K]`。
2. 在 Triton 里打印/推理 `warpsPerCTA`：128×128 block、`num_warps=4` 时常见 `[4,1]`（或 `[2,2]`），前者正对应 `[4,32]`。
3. 用 `ncu` 看生成的 `HMMA` 指令的 fragment 布局，或直接按上面的公式反推 `(row, col)`。

<details>
<summary><b>一句话总结</b></summary>

`[4, 32, K]` = **4 个 warp × 每个 warp 32 行 × 完整 K**。它是 warp 级 MMA（M 上限 16/64） + 128 行必须分摊 + 4 warp 铺满 4 个 SM subpartition 的必然结果；再往下一层，warp 内部还会按 4-lane quad 继续切。所以这个"感觉"是对的——tensor core 确实就是这么组织数据的。

</details>

---
## 🧑‍💻 User

我想你有个适配 warp 粒度的最小矩阵就可以了，更多的都是拼起来的，比如 n 可以是 8/16/...；tensor cores 上的 core matrix 是8x16B...总之这么设计性质比较好

---
## 😀 默认助手 (7)

对，你这个概括基本就是本质：**一个和 warp 粒度匹配的"原子"，剩下全是拼装**。只是这里有**两个原子**需要对齐，对齐上了，性质才"好"。

## 两个原子

| | 寄存器侧原子 | 存储侧原子 |
|---|---|---|
| 形状 | `m16n8k16`（fp16） | core matrix `8 × 16B` |
| 谁来定 | warp 的 32 lane + quad 结构 | 16B 访存粒度 + swizzle |
| 分解 | M$=8_{quad}\times2$，N$=4_{lane}\times2$，K 每 lane 2 个 b32 | 8 行 × 16B（fp16 即 8 个元素） |

关键的是它俩**能对上**：
- `ldmatrix` 里 **8 个线程给一个 $8\times8$ tile 的 8 行地址**，每行正好 16B → 8 线程/tile，32 lane 分 4 组就是 `ldmatrix.x4`；
- K 原子 $=16$ 个 fp16 $=2$ 个 core matrix 的 K 宽度。

所以**一条 16B 的行**是内存侧不可再分的单位，**一个 4-lane quad** 是计算侧不可再分的单位，16B 刚好喂饱一个 quad 的一行 chunk。

<details>
<summary><b>K 原子 = 2 个 core matrix，三种 dtype 都对得上</b></summary>

| dtype | 指令 | K 原子 | core matrix K 宽度（16B） | 倍数 |
|---|---|---|---|---|
| fp16 | `m16n8k16` | 16 | 8 | ×2 |
| tf32 | `m16n8k8` | 8 | 4 | ×2 |
| int8/fp8 | `m16n8k32` | 32 | 16 | ×2 |

这不是巧合——K 原子固定对应 **32B 的 K 宽度**（2 个 core matrix），因为一个 lane 每步要拿满若干个 b32 寄存器，同时访存必须是 16B 的倍数。

</details>

## 往上是纯拼装

- **warp tile**：`M ∈ {16, 32, 64}`（多叠几个 m16），`N ∈ {8,16,32,...}`（并排几个 n8），`K` 就是个循环；
- **CTA tile**：4 个 warp 沿 M/N 一摆就是 128×128；
- **wgmma**：指令级 M=64（4×16）、N 可到 256，本质还是 quad/core-matrix 的放大；
- **Blackwell tcgen05**：M=128 直接落到 128 条 TMEM lane，N 可到 256，拼装粒度更粗。

> 一个小澄清：`mma.sync` 指令里的 **N 是固定 8**；你说的"n 可以是 8/16/…"其实是 **tile 层**拼出来的，只有 wgmma / tcgen05 才让 N 在**指令级**变大。M 同理（指令 16，warp tile 拼到 32/64）。

## 为什么这么设计"性质比较好"

1. **16B = 128-bit**：一个 lane 一次 `LDS.128` 拿一整行 chunk，天然对齐，无需拆分；
2. **全是 2 的幂**：$32 \to 4 \to 8 \to 16$，以及 32/64/128B swizzle 都是倍数关系，bank 冲突和边界判断都能消掉；
3. **整除性好**：$32/4=8$、$16/8=2$、K 原子整除各种 dtype，M/N/K 三个方向都不会出现"边角料"，tile 可以无缝拼大；
4. **可广播**：quad 内 A 共享同一行、B 共享同一列，shared memory 可以 broadcast，省带宽；
5. **布局对称**：`ldmatrix.trans` 让 A/B 转置几乎免费，M 和 N 的处理写起来一样；
6. **寄存器用量是可算的常数**：每个 lane 固定几个 b32，occupancy 可以精确预测——这正是 CUTLASS/Triton 能自动 tiling 的基础。

所以你的直觉是对的：**硬件只保证"一个 warp 能直接吃下的一小块"高效，其余全靠规则化拼装**；而之所以拼装得这么顺，就是因为原子尺寸（4-lane quad、16B、2 的幂）之间是整除关系。
