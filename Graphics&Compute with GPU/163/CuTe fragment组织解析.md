# CuTe fragment组织解析

### 🧑‍💻 User

针对nvidia ptx（ldmatrix/mma）或cute中，比如f16matmul f32 acc一个 MMA Atom 可能只做16x8x16，但我真正想做的 CTA tile 可能是128x128x32要如何组织 fragment？

---

### 🤖 Assistant

我先确认一下当前 CuTe/CUTLASS 的准确 API 和 fragment 语义，再给你完整答案。

我已拿到 `mma_atom.hpp` 的关键实现。再确认一下 `ldmatrix` 的 copy atom 和 `make_tiled_copy_A` 的准确 API，以及一个真实 mainloop 例子。

正好有 `sgemm_sm80.cu`，这是最权威的参考。我读一下它的 mainloop 和 fragment 组织。

我已经核对了 CUTLASS `main` 分支里 `mma_atom.hpp`、`mma_traits_sm80.hpp`、`copy_traits_sm75.hpp` 和教程 `examples/cute/tutorial/sgemm_sm80.cu` 的源码，下面给你一个从「源码语义」出发的准确回答。

## 先纠正一个关键直觉

fragment **不是**"一个 MMA Atom = 一个 16×8×16"。fragment 是**每个线程的寄存器 tensor，它的逻辑形状就是整个 CTA tile**（C 是 $128\times128$，A 是 $128\times32$，B 是 $32\times128$），它的 layout 负责把每个元素映射到 `(thread, register)`。16×8×16 只是这个 layout 里最内层的 "MMA mode"。

从 `mma_atom.hpp` 的 `thrfrg_C` 注释可以看得最清楚，一个 TiledMMA 的 C-fragment 结构是：

$$
\underbrace{\Big(\ \texttt{ThrV},\ \underbrace{(\texttt{ThrM},\texttt{ThrN})}_{\text{threads tiled in M,N}}\ \Big)}_{\text{线程维度}}\;\times\;\underbrace{\Big(\ \texttt{FrgV},\ \underbrace{(\texttt{RestM},\texttt{RestN})}_{\text{values tiled in M,N}}\ \Big)}_{\text{每线程的寄存器维度}}
$$

其中（A 把 N 换成 K，B 把 M 换成 K）：

| 符号 | 含义 | SM80 16×8×16 的取值 |
|---|---|---|
| `ThrV` | 一个 atom 内部的线程 | 32（1 个 warp）|
| `ThrM/ThrN/ThrK` | `AtomLayout` 在 M/N/K 方向铺了多少个 atom（**这些是线程**）| 取决于你的 AtomLayout |
| `FrgV` | 一个线程、一个 atom 持有的寄存器数 | A=8 f16，B=4 f16，C=4 f32 |
| `RestM/RestN/RestK` | **超出线程覆盖范围**、在寄存器里循环的部分 | 由 CTA tile ÷ MMA tile 决定 |

## 三层分解：128×128×32 = Atom → MMA tile → CTA tile

```
atom  (16 × 8 × 16)         ← 一条 mma.sync（32 线程）
  │  × AtomLayout (4,2,1)
  ▼
MMA tile (64 × 16 × 16)     ← TiledMMA，256 线程协作
  │  × Rest (2 × 8 × 2)
  ▼
CTA tile (128 × 128 × 32)
```

关键公式（源码里 `ThrLayoutVMNK = tiled_product(AtomThrID, AtomLayoutMNK)`）：

$$\texttt{nthreads} = \underbrace{32}_{\text{ThrV}} \times |\texttt{AtomLayout}|\ ,\qquad \text{MMA tile} = \texttt{atom shape} \times \texttt{AtomLayout shape}$$

**这就是为什么不能把 128×128×32 全铺成 atom layout**：$8\times16\times2=256$ 个 atom，$256\times32=8192$ 个线程，超了 1024 上限。所以超过线程覆盖的部分必须走 `Rest` 模式（在寄存器里循环），而不是线程。

## 具体代码（128×128×32，f32 acc，256 线程）

```cpp
using namespace cute;
using MMA = SM80_16x8x16_F32F16F16F32_TN;   // atom: 16x8x16, f16*f16+f32

// AtomLayout: M×4, N×2, K×1 个 atom
//   -> MMA tile = (4·16, 2·8, 1·16) = 64x16x16
//   -> nthreads  = 32 * 4 * 2 * 1  = 256
// Tile<...> 是第三个参数：每次 MMA 迭代覆盖的数据 tile（决定 Rest 的粒度）
auto mma = make_tiled_mma(
    MMA{},
    Layout<Shape<_4,_2,_1>, Stride<_1,_4,_8>>{},  // atom/thread layout
    Tile<_64,_16,_16>{});                         // data tile per iteration

auto thr_mma = mma.get_slice(threadIdx.x);

// 直接对整个 CTA tile 做 partition，fragment 逻辑形状就是整个 tile
Tensor tCgC = thr_mma.partition_C(gC);            // gmem C: 128x128
Tensor tCrA = thr_mma.partition_fragment_A(sA);   // smem A: 128x32  (f16)
Tensor tCrB = thr_mma.partition_fragment_B(sB);   // smem B: 32x128  (f16)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);      // 累加器, f32

// 得到的分片形状（每线程）：
//   tCrC : (MMA=4, MMA_M=2, MMA_N=8)   = 4*2*8  = 64 f32
//   tCrA : (MMA=8, MMA_M=2, MMA_K=2)   = 8*2*2  = 32 f16
//   tCrB : (MMA=4, MMA_N=8, MMA_K=2)   = 4*8*2  = 64 f16

clear(tCrC);
for (int k = 0; k < size<2>(tCrA); ++k) {        // K 循环 = MMA_K = 32/16 = 2
    cute::gemm(mma, tCrA(_,_,k), tCrB(_,_,k), tCrC);
}
axpby(alpha, tCrC, beta, tCgC);                  // 累加器写回 gmem
```

通用分片形状公式（设 `ThrM/ThrN/ThrK` 为 AtomLayout 的三个 extent）：

$$
\begin{aligned}
\texttt{tCrC} &= \big(\texttt{FrgV}_C,\ \tfrac{\text{BLK}_M}{\texttt{atom}_M\cdot\texttt{ThrM}},\ \tfrac{\text{BLK}_N}{\texttt{atom}_N\cdot\texttt{ThrN}}\big)\\
\texttt{tCrA} &= \big(\texttt{FrgV}_A,\ \tfrac{\text{BLK}_M}{\texttt{atom}_M\cdot\texttt{ThrM}},\ \tfrac{\text{BLK}_K}{\texttt{atom}_K\cdot\texttt{ThrK}}\big)\\
\texttt{tCrB} &= \big(\texttt{FrgV}_B,\ \tfrac{\text{BLK}_N}{\texttt{atom}_N\cdot\texttt{ThrN}},\ \tfrac{\text{BLK}_K}{\texttt{atom}_K\cdot\texttt{ThrK}}\big)
\end{aligned}
$$

## 每线程的寄存器明细（256 线程，f32 acc）

| 操作数 | 总量 | 每线程 | 备注 |
|---|---|---|---|
| C | $128\times128=16384$ f32 | **64 f32**（64 reg）| 累加器 |
| A | $128\times32=4096$ f16 | 32 f16（16 reg）| A 在 N 方向被广播（ThrN=2 组线程共享）|
| B | $32\times128=4096$ f16 | 64 f16（32 reg）| B 在 M 方向被广播（ThrM=4 组线程共享）|

若 K 也做流水（一次只留一个 k-slab 的 A/B），A 降到 16 f16、B 降到 32 f16，C 仍是 64 f32 —— 总约 88 个寄存器，很宽裕。

PTX 视角锚定一下：整块 128×128×32 有 $8\times16\times2=256$ 条 atom；每个 k-slab（K=16）有 128 条 atom。256 线程下，**每个线程每个 k-slab 发 $128\times32/256=16$ 条 `mma.sync`**，两个 k-slab 共 32 条。

## K 维度（32 vs atom 的 16）怎么处理

- **atom 内部 K=16**：由单条 `mma.sync.aligned.m16n8k16...` 在指令内累加，不需要你管。
- **K=32 的剩余部分**：放在 fragment 的 **`MMA_K` 模式**里，就是上面 `for (int k = 0; k < 2; ++k)` 的循环，**同一个 `tCrC` 累加器被复用**。

注意：K 是归约维，所以 **AtomLayout 的 K extent 通常取 1**（`ThrK=1`），不要把 K 铺到线程上去（那需要跨线程归约）。K 的扩展交给 `RestK`/循环即可。教程 `sgemm_sm80.cu` 里也是这么做的——它 BLK_K=64、`MMA_K=4`，在 mainloop 里 `gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC)` 逐 k-block 累加。

## ldmatrix（smem → 寄存器）怎么和 fragment 对齐

`SM75_U32x4_LDSM_N` 就是 `ldmatrix.x4`，它一次搬一个 $16\times16$ f16 tile，给 32 个线程每人 8 个 f16 —— **恰好等于 MMA atom 的 A 操作数布局**（$16\times16$，8 f16/线程）。CuTe 用 `make_tiled_copy_A/B` 自动把 ldmatrix atom 铺成和 MMA fragment 完全一致的寄存器 layout：

```cpp
using LDSM = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;   // ldmatrix.x4

auto s2r_A = make_tiled_copy_A(LDSM{}, mma);   // tile 自动匹配 mma 的 A 分片
auto s2r_B = make_tiled_copy_B(LDSM{}, mma);

auto thr_A = s2r_A.get_slice(threadIdx.x);
Tensor tCsA = thr_A.partition_S(sA);   // smem 源视图 (128x32)
Tensor tCrA = thr_A.retile_D(tCrA0);   // 寄存器视图，layout == mma 的 A fragment

copy(LDSM{}, tCsA(_,_,k), tCrA(_,_,k));  // 这一条展开成 ldmatrix.x4
```

smem 侧通常用 swizzle 消除 bank conflict（教程里是 `Swizzle<3,3,3>` + k-major 布局）；gmem→smem 是另一套 `TiledCopy`（SM80 用 `cp.async`），和 MMA fragment 的组织是正交的。

## 小结：怎么"组织"

1. 选 **AtomLayout**（决定 MMA tile 大小和线程数），让 `32 × |AtomLayout|` 落在 128/256/512/1024 里，比如 `(4,2,1)` → 64×16×16、256 线程。
2. CTA tile 超出 MMA tile 的部分，**不需要你手写循环结构**，它自动成为 fragment 的 `RestM/RestN/RestK` 模式（或第三个参数 `Tile<...>` 指定粒度）。
3. K 方向靠 fragment 的 `MMA_K` 模式循环，同一个 `tCrC` 累加。
4. 数据搬运用 `make_tiled_copy_A/B(LDSM{}, mma)`，保证 ldmatrix 产出的寄存器布局和 MMA 消费的 fragment 布局逐位一致。

一句话：**fragment = "线程铺 atom" 的 `ThrV/ThrM/ThrN` 维 + "每线程寄存器" 的 `FrgV` 维 + "寄存器内循环" 的 `Rest` 维**，16×8×16 只是藏在最里层，128×128×32 是靠 `Rest` 模式把 8×16×2 个 atom 循环补齐的。