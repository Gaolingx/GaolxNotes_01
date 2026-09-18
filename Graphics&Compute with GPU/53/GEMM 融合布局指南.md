# GEMM 融合布局指南

### 🧑‍💻 User

gemm很多时候不是planar layout能跑出正常性能的，比如前一个gemm的output过激活到下一个op的input swizzle layout就够掉点头发的，这个后处理部分要如何design fuse、layout？

---

### 🤖 Assistant

# 后处理(epilogue)的 fuse 与 layout 设计

先把你说的“掉头发”场景定性清楚：**GEMM1 的 accumulator 是固定的 MMA fragment layout，GEMM2 的 A/B operand 又是另一套（受 `ldmatrix` + swizzle 约束的）layout**，两者之间夹了一个逐元素的 activation。你如果中间用 planar(row-major) global 内存串起来，等于在缝线处强制塞了两次 layout 转换,自然掉性能。

---

## 1. 先把被混在一起的三件事拆开

| 空间 | 内容 | 能不能随便改 |
|---|---|---|
| **Value space** | activation、bias、scale 等逐元素数学 | 无所谓,layout-free |
| **Index space** | fragment mapping / swizzle / 转置 | **这里才是成本来源** |
| **Storage space** | register / shared / global | 影响带宽和 bank conflict |

**最关键的一条规则:逐元素算子对 layout 不敏感。** 也就是说 `ReLU/GELU/×scale` 可以在任意 layout 下做,放到 producer 的 register 里做是**几乎免费**的。真正花钱的只有 **layout 转换**(寄存器↔共享↔寄存器、或 global 往返)。

所以设计目标不是“怎么 fuse activation”,而是:

> **把 activation 塞进 producer 的累加器寄存器里算掉,同时让 producer 写出的坐标映射 = consumer 期望的 swizzle,从而让 layout 转换次数归零。**

---

## 2. 为什么 planar 一定跑不出正常性能

以 NVIDIA Tensor Core(Ampere/Hopper)为例,两端都被硬件钉死了:

- **GEMM1 累加器 C**:`m16n8k16` 的 C fragment 是固定的线程分布。对 `m16n8`,线程 $t$ 持有
  $$\text{row} \in \{t/4,\; t/4+8\},\quad \text{col} \in \{2(t\bmod 4),\; 2(t\bmod 4)+1\}$$
  即每线程 4 个 fp32。这个 layout **不可选**。
- **GEMM2 operand**:想喂 `ldmatrix`,共享里的数据必须是 8×8(=16B 一行)粒度的 **XOR swizzle**,否则 `ldmatrix` 直接 bank conflict 到崩。

planar 同时不满足这两者,于是你被迫:
```
C frag ──st.global(还可能 non-coalesced)──▶ global planar
                                            │
                ld.global ──▶ st.shared(swizzle) ──▶ ldmatrix
```
多一次 global 往返 + 一次 smem 写入 + activation 还要再来一遍。**这就是那点头发。**

---

## 3. 核心结构:把 epilogue 做成“layout 缝合线”

现代做法是把 GEMM1 的 epilogue 变成 **producer 侧的 swizzle 写入**,让 GEMM2 的 `ldmatrix`(consumer 侧)直接吃:

```
  GEMM1 MMA ──▶ C fragment (fixed)
                    │
        [epilogue]  ├─ bias / activation   ← 逐元素, 在寄存器里, 免费
                    ├─ 广播/规约处理
                    └─ stmatrix / XOR swizzled st.shared   ← 写出 = GEMM2 期望的坐标
                    │
                 shared tile  (producer 与 consumer 共用同一 swizzle atom)
                    │
             GEMM2 ldmatrix ──▶ operand fragment
```

数学上,你是在共享内存里实现一个从累加器线性下标到消费方物理下标的映射,并让这个映射只是 XOR:

$$\text{offset}_{\text{phys}} = \text{offset}_{\text{lin}} \oplus \big(\phi(\text{row}) \ll c\big)$$

$\phi$ 取行号低几位、$c$ 是列向移位,选到让 `ldmatrix` 的 8 行恰好落在 8 个不同 bank。**要点是 producer 写的时候就用这个 $\oplus$,consumer 读的时候天然匹配,中间 0 转换。**

Hopper 上这个 producer 原语就是 `stmatrix`;Ampere 没有,就用手工 XOR 寻址的 `st.shared`(或用 `movmatrix` 做寄存器内转置)。两者和 `ldmatrix` 是一对齿轮。

---

## 4. 四类 fusion 方案 + 选择

| 方案 | 中间态 | 成本 | 适用 |
|---|---|---|---|
| **A. 直接 swizzle 链(推荐)** | 共享内存里就是 consumer 的 swizzle | 只多一次 smem 写 | tile 能对齐,K-reduction 允许即产即用 |
| **B. 寄存器内 fragment 重映射** | 不落内存 | 依赖 `shfl`/`movmatrix` | 仅当 C layout 与 operand layout 是 warp 内固定置换(少见) |
| **C. 规范化 swizzle(canonical encoding)** | 全局只认一种 swizzle | 边界一次转换 | GEMM 密集图,把转换推到图边界 |
| **D. 双写 epilogue** | 同时写 planar global + swizzle smem | 两次 store | 中间结果也是图输出 / 多消费者 |

> **方案 C 才是根治**:全图约定按 (dtype, tile shape) 索引的 swizzle 家族,每个 GEMM epilogue 都吐它、每个 GEMM prologue 都吃它。于是 GEMM→Act→GEMM 之间**没有任何转换**,planar↔swizzle 的转换被推到图的**输入/输出边界**,整个网络只在最外面转一次。

---

## 5. 具体配方(伪代码)

```cpp
// ===== GEMM1 epilogue: 直接产出 GEMM2 想要的 swizzle =====
// 约定: GEMM1 的输出 tile (BM x BK) == GEMM2 的 A operand tile
frag_C c = mma_accumulator;          // m16n8, 固定 layout

#pragma unroll
for (int i = 0; i < c.size(); ++i) {
    float v = c[i] + bias[col_of(i)]; // 广播按 fragment 下标取
    c[i] = act(v);                    // ReLU/GELU/...  寄存器里做, layout-free
}

// producer 侧: 写出坐标 = consumer 的 ldmatrix swizzle
stmatrix_or_swizzled_store(smem_tile, c, /*atom=*/SW128_XOR);
// Ampere 手工路径: addr = xor_swizzle(row, col, SW128_XOR); st.shared.f16 ...

__syncthreads();                      // 或 mbarrier, 双缓冲时不需要

// ===== GEMM2 prologue: 零转换直接吃 =====
ldmatrix(a_frag, smem_tile);         // 和上面对齐的 swizzle, 无 conflict
```

**对齐条件(必须满足,否则退化成方案 A + 一次转换):**
- GEMM1 的 `BN`(=输出 N/K 维度)= GEMM2 的 `BK`;
- M/N tile 的 CTA 映射一致,避免跨 CTA 重排;
- 若 GEMM1 输出 `[M,K]` 而 GEMM2 要 `[K,M]`(A^T),要在缝线处做转置 → 用 `movmatrix` 或 **带转置的 stmatrix 模式**;
- 16B 对齐 + 双缓冲 smem。

---

## 6. Layout 传播原则(如果画成一个 pass)

1. **Anchor 在 GEMM 的 operand 上** —— 那里的 layout 由 MMA + `ldmatrix` 物理钉死,不可自由选。
2. **逐元素链是“柔性段”** —— activation 边界是允许改 layout 的合法位置(因为它 layout-free),把不可避免的转换调度到这里。
3. **贪心前向(从 GEMM1 输出)+ 后向(从 GEMM2 输入)传播,在 elementwise 链上会合**:若两端的 encoding 能通过逐元素算子传递而不冲突,就**不插入** `convert_layout`;冲突才插入,并用 cost model 估价。
4. **优化目标是转换“次数”而非单次成本** —— 3 次廉价转换 > 1 次昂贵转换通常是错的,要全局算。
5. **优先让 producer 写 consumer 的 layout**,而不是 consumer 读 producer 的 layout(写侧做 swizzle 更自然,因为 `stmatrix` 就是为此而生)。

<details>
<summary><b>7. 各框架落地映射(点开)</b></summary>

- **CUTLASS**:写 epilogue 时用 `TensorOp`/`FragmentIterator`,让 `stmatrix` 的 swizzle 与下游 `ldmatrix` 的 `TensorLayout` 一致;`SmemLayoutAtom` 选同一个 XOR 型号。
- **Triton**:layout 自动推断,`convert_layout` 自动插;你要做的是保证 tile 形状/K 对齐,别让编译器被迫在 seam 插转换。
- **IREE**:把 layout 做成 tensor 的 **encoding**(MVP/MatmulEncoding),前向传播,在图边界才 materialize 成 planar。
- **TVM**:`Layout` + `TransformLayout` + MMA layout inference,在 `schedule` 里决定 swizzle。

</details>

<details>
<summary><b>8. 工程坑清单(点开)</b></summary>

- activation **永远在寄存器里做**,绝不“store→reload→activate”。
- 用 XOR swizzle 而非 padding swizzle(padding 浪费 smem,且破坏 atom 对齐)。
- 注意 dtype:fp16 128B swizzle 的 atom 是 8×16B;fp8/int8 不同,别抄错 $\phi$。
- 别忽略 K-split / split-K:它会改变哪一侧 layout 更划算(有时保持 planar 走 global 反而更快,如果 K 很碎)。
- 非 NVIDIA 平台(AMX / NPU):同样的“producer 吐 consumer 的 swizzle”原则,只是原语名字不同(AMX 用 tile load/store 的 stride 描述)。
- barrier 开销:双缓冲 + `mbarrier` 隐藏,别让 `syncthreads` 变成 seam 的主导成本。
</details>

---

## 一句话总结

**把 activation 当成 layout-free 的东西塞进 GEMM1 的累加器寄存器里做掉,然后让 GEMM1 的 epilogue 用 `stmatrix`/XOR-swizzle 直接写出 GEMM2 `ldmatrix` 期望的坐标——seam 上做到零 layout 转换。** 若不可能,就退一步用**规范化 swizzle encoding**把转换整体推到图的边界,而不是在每个 GEMM 交界处转。真正要 minimize 的是转换次数,不是单次转换的代价。

---

### 🧑‍💻 User

比如说mlp中通常是upproj->activation->downproj，upproj经常会和swiglu融合然后去做downproj的gemm，这里最佳实践是怎么样的，能不能举个cutlass的例子解释？

---

### 🤖 Assistant

# SwiGLU MLP 的 fuse / layout 最佳实践

先钉死计算图($X$ 是 token, $K$ 是 hidden,$N$ 是 intermediate,通常是 $N\approx \frac{8}{3}K$):

$$
G = XW_g^\top,\quad U = XW_u^\top,\quad A = \mathrm{SiLU}(G)\odot U,\quad Y = AW_d^\top
$$

up-proj 的输出宽度是 $2N$(gate+up 拼一起),中间张量 $A\in\mathbb{R}^{M\times N}$ 是**上一个 GEMM 的 output**,同时是**下一个 GEMM 的 A operand**,缝线就在这里。

---

## 1. 核心洞察:利用 `m16n8k16` 累加器的“相邻两列同线程”

这是整个设计里最值钱的一条。对 `mma.m16n8k16` 的 C fragment,线程 $t$ 持有:

$$
\text{row}\in\Big\{\tfrac{t}{4},\ \tfrac{t}{4}+8\Big\},\qquad
\text{col}\in\Big\{2(t\bmod 4),\ 2(t\bmod 4)+1\Big\}
$$

**相邻两列(2i, 2i+1)永远落在同一个线程。** 于是只要把权重交错成

$$
W_{gu}[:,2i] = W_g[:,i],\qquad W_{gu}[:,2i+1]=W_u[:,i]
$$

那么每个线程拿到的 fragment 里,`reg[0]` 和 `reg[1]` **天然就是同一个 intermediate index 的 $(g_i,u_i)$ 对**。$\mathrm{SiLU}(g)\cdot u$ 就变成了**纯线程内、零通信**的一次计算:

```
线程持有:  [g_i, u_i]   ->   out_i = silu(g_i) * u_i      // 无 shfl, 无 smem, 无 cross-thread
```

> 这一步把原本“需要把 $G$ 和 $U$ 两个张量对齐”的 two-source 问题,坍缩成了一个 **单累加器、线程内** 的问题 —— 这是 CUTLASS 里能用一条简单 epilogue functor 搞定的根本原因。

如果**不交错**(直接 `[W_g | W_u]`),$g_i$ 在列 $i$、$u_i$ 在列 $i+N$,二者落到不同线程,你就必须靠 shuffle 或 smem 把配对搬到一起 —— 这就是“掉头发”的来源之一。

而且**交错对 GEMM 本身零成本**:它只是 $B$ 操作数 N 维度的一个全局内存排列,主循环/TMA 完全无感。代价只有 $O(KN)$ 的一次性权重重排(可以烘进 checkpoint)。

---

## 2. 分层最佳实践(推荐栈)

| 层 | 做法 | 收益 | 代价 |
|---|---|---|---|
| L0 | 两个独立 kernel,中间 elementwise kernel | — | 多了读写 $A$ + 一次 launch |
| **L1** | **把 SiLU 融进 up-proj 的 epilogue**(必须做) | 省一次读写 $A$、省一个 kernel | 无 |
| **L2** | **权重交错** $W_{gu}$,gated 激活线程内完成 | epilogue 零通信 | 一次性权重重排 |
| **L3** | 中间张量写成 down-proj **能直接 `ldmatrix`/TMA 的编码**(canonical swizzle) | 缝线零 `convert_layout` | 需两端协调 |
| L4 | **PDL / CUDA Graph / persistent** 隐藏 kernel 边界 | 去掉 launch/尾部空泡 | 调度复杂度 |
| L5 | **fp8 中间态 + per-token scale** | $A$ 的读写减半(fusion 后瓶颈就在这) | 精度/量化 |

**生产上(vLLM / TensorRT-LLM / SGLang 这一档)基本就是 L1+L2+L3,再用 L4 把两个 kernel 的边界藏掉。**

---

## 3. 为什么**不做** full on-chip GEMM→GEMM(和 Attention 的本质区别)

看到“GEMM→elementwise→GEMM”很容易想照抄 FlashAttention 的 on-chip 融合,但这里结构不同:

<details>
<summary><b>展开:为什么 SwiGLU MLP 的 full fusion 不划算(点开)</b></summary>

FlashAttention 能融,是因为内层循环里**驻留的 operand 很小**:$Q$ tile $[B_M, d]$ 在整个 $K$ 循环里待着,$d=128$ 级别。

MLP 若照搬,内层循环 over $n\_tile$,每轮要重算 $G[:,n\_tile]=XW_g[:,n\_tile]^\top$,需要把驻留的 $X[BM, :]$ 放在片上:

$$
X[128, 8192]\times 2\text{B} \approx 2\text{MB} \gg 228\text{KB(sm\_90 smem)}
$$

放不下 → 要么每轮从 L2 重读 $X$(重读 $N/BN\approx 112$ 次,直接爆),要么干脆把 $A$ materialize 出去。**结论:contraction 维驻留的 operand 太大,full on-chip fusion 会退化成“重读 $X$”或“回到两 kernel”,所以不值。**

正确的心智模型:**“GEMM→elementwise→GEMM” 能融的前提是中间态小且即产即用,或驻留 operand 小。** SwiGLU 两者都不满足(中间态 $A$ 很大,驻留 operand $X$ 也大),于是最优解就是**两 kernel + 缝线处零转换**。
</details>

---

## 4. CUTLASS 例子

> ⚠️ CUTLASS API 随版本变化较大,下面用 SM90(CUTLASS 3.x)+ 一个自定义 epilogue 说明**结构**;具体模板参数请对齐你所用版本。核心是那个 gated epilogue 和交错权重布局。

### 4.1 Host:构造交错权重(用 cute 表达布局)

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// W_gate, W_up : [K, N] 行主序 (K=hidden, N=intermediate)
// 目标 W_gu : [K, 2N],  列 = 2*i + h   (h=0->gate, h=1->up)
// offset(k, i, h) = k*(2N) + 2*i + h
auto make_Wgu = [](auto* gu, auto* g, auto* u, int K, int N) {
  auto Wgu = make_tensor(make_gmem_ptr(gu),
      make_layout( make_shape (K, make_shape(Int<2>{}, N)),
                   make_stride(2*N, make_stride(Int<1>{}, Int<2>{})) ));
  auto Wg  = make_tensor(make_gmem_ptr(g),  make_layout(make_shape(K, N), GenRowMajor{}));
  auto Wu  = make_tensor(make_gmem_ptr(u),  make_layout(make_shape(K, N), GenRowMajor{}));
  // 一次性的 host/device 拷贝, 可离线烘进权重
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < N; ++i) {
      Wgu(k, make_coord(0, i)) = Wg(k, i);   // gate -> even
      Wgu(k, make_coord(1, i)) = Wu(k, i);   // up   -> odd
    }
};
```

关键点:这个交错是**纯 global 布局**,kernel 完全无感。

### 4.2 Kernel:自定义 gated epilogue

累加器 tile 是 $[BM, 2N]$,每线程 4 个 float(`m16n8`),`{0,1}` 是一行的 (g,u) 对,`{2,3}` 是 row+8 的对:

```cpp
struct SwiGLUEpilogue {
  using ElementAcc = float;
  using ElementOut = cutlass::bfloat16_t;

  // acc[0..3]: row   {2c, 2c+1} -> (gate, up)
  //            row+8 {2c, 2c+1} -> (gate, up)
  // out[0..1]: 每行一个真实元素 (宽度从 2N 折半到 N)
  CUTLASS_DEVICE
  void operator()(cutlass::Array<ElementAcc,4> const& acc,
                  cutlass::Array<ElementOut,2>&       out) const {
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      float g = acc[2*r + 0];             // gate
      float u = acc[2*r + 1];             // up
      float s = g / (1.f + __expf(-g));   // SiLU(g) = g * sigmoid(g)
      out[r]  = ElementOut(s * u);        // 线程内完成, 无通信
    }
  }
};
```

配套的 GEMM:

```cpp
using UpProjGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // A = X      [M, K]
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // B = W_gu   [2N, K] (NT)
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // D = A      [M, N]
    float,                                            // 累加用 fp32
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    // ... ThreadblockShape / WarpShape / StageCount ...
    SwiGLUEpilogue>;                                  // <-- 就挂在这里
```

### 4.3 缝线:输出宽度折半 + swizzle 编码

<details>
<summary><b>为什么这个 epilogue 和标准 epilogue 不一样,以及怎么落地(点开)</b></summary>

标准 CUTLASS epilogue 假设 `D` 的形状 = 累加器形状。这里 $D$ 的列数是累加器的**一半**($N$ vs $2N$),所以你要:

- **CUTLASS 2.x 风格**:自定义 `EpilogueOutputOp`,把输出 iterator 定义在“只取偶数列”的视图上(`cute::make_layout` 对 N mode 用 stride-2 子选择),让 store 落到 $[M,N]$。
- **CUTLASS 3.x 风格**:用 **Epilogue Visitor Tree(EVT)**。为 gated 激活写一个自定义节点(读同一 accumulator 的相邻两 lane、输出一个 lane),或者用 `fusion::EltwiseApply` 家族 + 自定义 op。EVT 就是为这种 multi-source / 变形 epilogue 设计的机制。
- **alternive**:不折半宽度,而是让 kernel 带**两个累加器**(gate/up 各一个、同 layout、同 tile),epilogue 做普通逐元素 `silu(acc_g)*acc_u`。两个累加器共享 $X$ 的 smem load,是“fused two-GEMM”。代价是主循环要维护两路 B operand。

哪个更优取决于你的 pipeline:交错单 GEMM 只有一路 B 流,对 warp-specialization / TMA 更友好;双累加器复用标准 EVT 节点更省事。
</details>

而且 kernel1 写出的 $A$ **直接按 down-proj 的 A-operand swizzle 落盘/落 smem**(L3),这样 down-proj 的 `ldmatrix`/TMA 一步到位,**缝线零 `convert_layout`**。这就是上一轮说的“producer 写 consumer 的 layout”。

### 4.4 down-proj:极简、无激活

```cpp
using DownProjGemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // A = A      [M, N]
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // B = W_down [K_out, N] (NT)
    cutlass::bfloat16_t, cutlass::layout::RowMajor,   // D = Y      [M, K_out]
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    // ... tile shapes: 让 A-operand 的 K-tile (=N-tile) 与上游对齐 ...
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, /*...*/ cutlass::bfloat16_t, float>>;
```

**tile 对齐的硬约束**:up-proj 的输出 N-tile $\text{BN}_{up}$ 应该等于 down-proj 的 A-operand K-tile $\text{BK}_{dn}$,这样中间 tile 的形状刚好吻合;若走 L4 的 PDL/persistent,这也是让两个 kernel 的 tile 调度相容的前提。

---

## 5. 一张图总结缝线

```
        X [M,K]
           │
   ┌───────▼────────┐   A operand: X  (standard swizzle)
   │  up-proj GEMM  │   B operand: W_gu [2N,K]  <- 交错权重, 一次性
   │  (MMA m16n8k16)│
   └───────┬────────┘
           │  C fragment [BM, 2N]  相邻两列同线程 = (g, u) 对
   ┌───────▼─────────────────────────┐
   │  epilogue:  out = SiLU(g) * u    │  线程内, 零通信, L1+L2
   │  折半宽度 [BM,N], 写 swizzle      │  L3: 直接落成 down-proj 的 A layout
   └───────┬─────────────────────────┘
           │  A [M, N]   (ideal: fp8 + per-token scale, L5)
   ┌───────▼────────┐   A operand: A  (ldmatrix/TMA 直接吃, 零转换)
   │ down-proj GEMM │   B operand: W_down [K_out,N]
   └───────┬────────┘
           ▼
        Y [M,K_out]
```

---

## 6. 落地清单

- ✅ **一定**把 SiLU 融进 up-proj epilogue;**绝不**再单开一个 elementwise kernel。
- ✅ 用**交错权重** $W_{gu}$,靠“相邻两列同线程”实现线程内 gated 激活。
- ✅ 中间态按 down-proj 的 operand 编码写(**canonical swizzle encoding**),缝线零转换。
- ✅ 用 **PDL / CUDA Graph / persistent** 藏掉两个 kernel 的边界空泡。
- ✅ intermediate 上 **fp8 + per-token scale**(此时瓶颈就是 $A$ 的读写)。
- ❌ 别硬上 full on-chip GEMM→GEMM:contraction 维驻留的 $X$ 太大,会退化成重读 $X$。
- ⚠️ epilogue 折半宽度这点,2.x 需自定义 `EpilogueOutputOp`,3.x 走 **EVT** 自定义节点 —— 这是唯一需要“写代码”的地方,其余都是布局/配置。