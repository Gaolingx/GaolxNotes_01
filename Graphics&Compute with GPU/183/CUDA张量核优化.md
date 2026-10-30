# CUDA张量核优化

## 🧑‍💻 User

```
发现根源：矩阵列数为奇数时张量核吞吐直接减半（95 对 190 TFLOPS），而每个原子块的有效列数常为奇数。接下来会补齐空列，让列数凑成 8 的倍数来解决这个问题。
```
这个为什么会影响性能，cuda programming最佳实践是什么？

---
## 😀 默认助手 (7)

## 结论先行

奇数列表述下算力腰斩,几乎不可能是"硬件就慢一半",而是你的 kernel 从**快路径掉到了慢路径**。Tensor Core 的最小粒度决定了:非 8 倍数的维度必然被"向上取整"执行,而取整之外还有**对齐崩塌**(`ldmatrix` / `cp.async` / 128-bit 向量化全部退化)。你说的"补到 8 的倍数"方向是对的,而且在 fp16/bf16 下 **8 元素 = 16 字节**,恰好就是 `ldmatrix` 和 `cp.async` 的最小对齐单位——这不是巧合。

---

## 一、为什么奇数列会把吞吐打掉一半

### 机制 A:MMA 指令粒度量化(B 维最小是 8)

主流 Tensor Core 指令的形状是硬编码的:

| 指令 | 精度 | M | N | K |
|---|---|---|---|---|
| `mma.sync.m16n8k16` | FP16/BF16 | 16 | **8** | 16 |
| `mma.sync.m16n8k8` | TF32 | 16 | **8** | 8 |
| `wgmma.m64nNk16` | FP16/BF16 | 64 | 8…256 (×8) | 16 |
| `tcgen05.mma` | 各精度 | 128 | 8…256 (×8) | 8/16/32 |

关键点:**N 维永远不能小于 8,K 维是 8/16/32 的整数倍**。所以有效列数 $N_{\text{eff}}$ 的张量核利用率上限是

$$U = \frac{N_{\text{eff}}}{\lceil N_{\text{eff}}/8 \rceil \cdot 8}$$

- 若你的"列"落在 **N 维**:$N_{\text{eff}} = 4$ → $U = 4/8 = 50\%$,正好对上你说的 95 vs 190。
- 若你的"列"其实落在 **K 维**(收缩维,粒度 16 for FP16):奇数 K 会多切出一步"半满"的 k-iteration,最坏也是 50%。

> **判别方法**:看你的 `N_eff` 分布。如果平均在 4 左右,那 2× 是纯算术量化,padding 到 8 只会把"浪费"从"硬件自动补"变成"你显式补",算力不会变——**但如果它让你切回了向量化快路径,还是会翻倍**。如果 `N_eff` 已经是 7、13、15 这种,那 padding 只多算 0~1 列就拿回 2×,纯赚。

### 机制 B:对齐崩塌(这才是"恰好 2×"最常见的原因)

矩阵行长度(leading dimension, LDA)为奇数时,第 $r$ 行起始地址是 `base + r*LDA*sizeof(T)`,于是:

1. **`ldmatrix` 直接不可用**。`ldmatrix.sync.aligned.m8n8.x4` 要求 8×8 tile 的每一行起始地址 **16 字节对齐**,奇数 LDA 违反该约束 → 编译器/你只能退回 `lds.b32` 逐元素装载,指令数 ×4,SMEM 流水线直接成为瓶颈。
2. **`cp.async` 宽度塌到 4B**。`cp.async.cg` 要 16B 对齐,否则只能 `cp.async.ca ... , 4`,即每搬 4 字节发一条指令,而不是 16 字节。异步流水线的搬运吞吐和 in-flight 能力同步崩掉。
3. **TMA 的 tile 描述符要求 stride 是 16B 的倍数**,奇数 LDA 无法构造 TMA descriptor → 退回 LDG 循环。
4. **Epilogue 输出同理**:`st.global.v4` 需要 16B 对齐 → 退化成标量 store。

这四条叠加,实际 FLOPs 没变,但**每条 MMA 前的数据供给指令数 ×4**,在 compute-bound 的 GEMM 里观测到的就是"吞吐腰斩"。

### 机制 C:扇区/事务浪费

GPU 访存按 **32B sector / 128B transaction** 计数。一行 $c$ 个 fp16、起始字节偏移 $o$:

$$\text{sectors per row} = \left\lceil \frac{(o \bmod 32) + 2c}{32} \right\rceil$$

LDA 为奇数 ⇒ 每行 $o \bmod 32$ 随机漂移 ⇒ 每行多咬一个 sector(最坏 2× 字节流量)。如果是 memory-bound,这就是现成的 2×。

### 机制 D:调度与尾部

- 奇数 N 让 tile 数不能整除 SM 数 → **wave quantization** 尾部效应。
- 很多库级 kernel(含 cuBLAS 的某些 heuristic)在 `N % 8 != 0` 时**直接选另一套非 tensor-core 或简化 epilogue 的实现**。这不是你代码的问题,是库的分派策略。

<details>
<summary><b>用 Nsight Compute 三条命令定位到底是哪个机制(点开)</b></summary>

| 指标 | 健康值 | 异常含义 |
|---|---|---|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | >60% | 低 ⇒ 机制 B/D(数据供给不上) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | ≈4 | <4 ⇒ 未合并(机制 C) |
| `smsp__inst_executed_pipe_lsu.sum` 相对 `sm__inst_executed_pipe_tensor.sum` 的比值 | 稳定 | 比值暴涨 ⇒ 机制 B(指令数膨胀) |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active` | — | 高 ⇒ 等 HBM(机制 C) |
| `smsp__warp_issue_stalled_mio_throttle` | — | 高 ⇒ SMEM/LSU 争用(机制 B) |
| `sm__ctas_active.avg.pct_of_peak_sustained_active` 末段掉坑 | 平稳 | 掉坑 ⇒ 机制 D(wave quantization) |

快速实验:把矩阵 N 手工 pad 到 8 的倍数但**不改 kernel**,只比 `sm__pipe_tensor_cycles_active` —— 如果 tensor pipe util 不变而 wall time 减半,说明瓶颈在访存侧(机制 B/C);如果 tensor pipe util 本身翻倍,说明是机制 A/D。
</details>

---

## 二、你的修法为什么对,以及三个必须踩住的坑

**为什么对**:FP16/BF16 下 $8 \times 2\text{B} = 16\text{B}$,恰好同时满足 `ldmatrix` 的行对齐、`cp.async` 的 16B、`st.global.v4` 的对齐。补到 8 的倍数,一次性把机制 A/B 里的"功能阻断"全解掉。

**坑 1:8 只是下限,32/128 才影响带宽。** 对齐要求是分层的:

| 目标 | FP16 需 LDA 倍数 | 字节 |
|---|---|---|
| `ldmatrix` / `cp.async.cg` / `float4` 向量化 | **8** | 16B |
| 扇区不浪费(sector) | 16 | 32B |
| 最优 burst / TMA swizzle / HBM 效率 | 64 | 128B |

如果能改全局布局,建议直接按 **64 元素** pad,而不只是 8。

**坑 2:别把 padding 写进 global memory 再读回来——那会把访存压力加大。** 优先级从高到低:

1. **只补 SMEM**:全局布局不动,装载时用 `cp.async` 的 **zfill**(越界自动补 0)或 TMA 的 **OOB-fill** 特性,把残块补零到 SMEM 的对齐 tile 里。零额外 HBM 流量。这是最干净的方案。
2. **一次性重排**:如果矩阵会被复用(多轮迭代/多 kernel),做一次 pre-pass 写入 padded buffer,成本摊薄。
3. **改布局**:`cudaMallocPitch` / 自定义 stride,把 LDA 抬到 64 的倍数。

**坑 3:padded 区域必须"只写不读"或 mask 掉。** 补出来的列是垃圾数据,epilogue 若原样写回会越界或污染;反之若它参与归约(比如 softmax、求和),会算错。epilogue 必须**按真实 N 谓词化**,主循环才可以全速跑满 —— 这就是"**peel the tail**"原则:主循环对整 tile 无分支 + 全展开,残块只在最后单独处理一次。**绝对不要在热循环里每轮都做边界判断**。

---

## 三、CUDA 编程最佳实践清单(按对本问题的相关性排序)

### 1. 布局与对齐(第一优先)

```cuda
// 原则:所有被向量化访问的维度,步长对齐到访问宽度的整数倍
// FP16/BF16:  LDA % 8  == 0   (16B, 功能底线)
//             LDA % 64 == 0   (128B, 性能最优)
// FP32/TF32:  LDA % 4  == 0   (16B)
// FP8:        LDA % 16 == 0   (16B)

// 结构体/指针也别忘了
struct __align__(16) Pack { half x[8]; };
auto* p = reinterpret_cast<float4*>(__builtin_assume_aligned(ptr, 16));
```

- 一次性对外声明对齐:用 `__builtin_assume_aligned` / `__restrict__`,让编译器敢生成 `ld.global.v4`。
- 分配期就保证: `cudaMalloc` 返回 256B 对齐,但**子矩阵/切片**会破坏它 —— 切片是"对齐静默失效"最常见的来源。

### 2. 访存合并(Coalescing)

- 一个 warp 的 32 个线程应访问一个连续的 128B 段;目标是 **4 sectors/request**。
- 行主序 + 线程沿列连续 ⇒ 好;线程沿行跨步 ⇒ 灾难。
- 检查手段:`ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`。

### 3. Shared Memory 与 Tensor Core 专用布局

- 传统 bank conflict 规避:stride 取奇数(`+1` padding)通常**有效**(32 个 bank,奇数步长天然错开)。
- 但接 Tensor Core 时**不要自己手推奇数 stride** —— 用 **XOR swizzle**(CUTLASS/Swizzle<3,3,3> 那种),它是为 `ldmatrix` 的 16B 行对齐 + 无冲突同时设计的。你的"奇数 LDA"在传统视角下恰恰是"无害",这正是它容易被忽略、却仍然致命的原因。
- SMEM tile 的 leading dim 也要 pad:常见做法是 `kTileN + 8`(FP16 时让每行额外错开 16B)。

### 4. 异步拷贝与流水线

| API | 对齐要求 | 备注 |
|---|---|---|
| `cp.async.ca.shared.global [.,4/8/16]` | 4/8/16B | 16B 时最省指令 |
| `cp.async.cg` | 16B | 绕过 L1,GEMM 首选 |
| `cp.async.bulk.tensor`(TMA) | 16B,stride 亦须 16B 倍数 | 支持 OOB-fill 补零 |

- 用 **多级 pipeline**(至少 2~3 级 `cp.async` + `cp.async.commit_group` / `wait_group`)把 HBM 延迟藏住;单缓冲的 ldmatrix 循环永远打不满 Tensor Core。
- `__pipeline_memcpy_async` / `cuda::memcpy_async` 是较易用的封装。

### 5. 指令级与 Tile 选择

- Tile 尺寸必须让 **M、N、K 全是 MMA 粒度的整数倍**(FP16:16/8/16),否则波浪量化和尾部检查必然出现。
- 累加器常驻寄存器,避免 FP32↔TF32 反复转换;`__launch_bounds__` 控制寄存器压力到能容纳双缓冲 accumulator。
- 循环展开:主循环 `#pragma unroll` 全展开,残块独立函数。

### 6. 变宽分块(ragged)的通用策略

你这种"每块有效列数不定"的负载,是有标准解法的:

1. **分桶(Bucketing)**:按 $N_{\text{eff}}$ 排序/分桶,把尺寸相近的块批量调度,同桶内用同一套特化 kernel(每桶编译一份 template 实例)。
2. **聚合拼装(Packing)**:用 gather/scatter 索引把多个小块的列拼进一个 dense tile,补零后统一算,**最后 scatter 回原位**。补零列只占算力、不占额外 HBM 流量。
3. **模板特化 + 对齐分派**:`switch (AlignmentA)` 在 CUTLASS 里就是干这个的。

```cpp
// CUTLASS 里对齐是模板参数,必须显式给对,否则静默走慢路径
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cutlass::gemm::GemmShape<128, 128, 64>,   // 全部是 8/16 的倍数
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,          // stages
    8,          // AlignmentA  <-- 关键:至少要 8
    8,          // AlignmentB
    8           // AlignmentC
>;
```

### 7. 系统化原则(来自 CUDA C++ Best Practices Guide 的核心四问)

按顺序问,不要跳:

1. **能不能更少的活?** —— 算法级:减少 FLOPs / 减少数据搬运(算子融合)。
2. **能不能更少的数据?** —— 精度降低(FP32→TF32→FP16→FP8)、稀疏、复用 SMEM。
3. **能不能更好的访存?** —— 合并、对齐、向量化、`__ldg` / L2 persisting。
4. **能不能更好的并行?** —— occupancy、ILP、async pipeline、避免 divergence。

你现在的问题**已经过了第 1、2 层**,是第 3 层的典型失败:FLOPs 一点没变,但每条 MMA 的数据补充指令数膨胀了。

### 8. 必做的工程化护栏

- **CI 里加对齐断言**:

```cpp
#define ENSURE_ALIGNED(p, bytes) \
  do { assert((reinterpret_cast<uintptr_t>(p) & ((bytes)-1)) == 0); } while (0)

TORCH_CHECK(LDA % 8 == 0, "LDA=", LDA, " violates 16B alignment for <<ldmatrix>>");
```

- **CI 里跑奇偶两种 N** 做性能回归,把"奇数列表述"变成固定测试用例。
- 用 `ncu --set full` + `--kernel-name` 定期比对,把 `sm__pipe_tensor_cycles_active` 当 SLO 守住。

---

<details>
<summary><b>一句话记忆版(点开)</b></summary>

> **Tensor Core 的"最小颗粒"是 N=8 / K=16(TF32 为 K=8),但真正的杀手不是算术取整,而是 16 字节对齐。** 奇数步长 → `ldmatrix`/`cp.async`/`v4 store` 全部退化 → 指令数 ×4 → 吞吐腰斩。补到 8 的倍数(FP16 恰好 = 16B)是必要且几乎无浪费的修法;补到 64(128B)才把带宽也吃满。补零优先在 SMEM 里做(`cp.async` zfill / TMA OOB-fill),不要污染 HBM。主循环整 tile 无分支,边界检查只放在 peeled tail。
</details>

需要我把 `cp.async` zfill 补零 + peeled-tail 的那个 kernel 骨架写出来,或者用一段小 benchmark 帮你验证到底是机制 A 还是机制 B 吗?
