# FP16 GEMM布局处理

### 🧑‍💻 User

在我们编写
```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16..."
);
```
时候通常需要处理
- warp mapping
- register tuple
- A fragment layout
- B fragment layout
- C fragment layout
- shared-memory layout
- ldmatrix layout
- swizzle
- warp tile mapping
- CTA tile mapping
请你用一个fp16 gemm example演示这些要点

---

### 🤖 Assistant

# 手写 `mma.sync.aligned.m16n8k16` FP16 GEMM：十个要点逐一拆解

先用一句话定位：`mma.m16n8k16` 是 **一个 warp（32 个 lane）共同执行** 的指令，计算

$$D_{16\times 8} = A_{16\times 16} \cdot B_{16\times 8} + C_{16\times 8}$$

- A 是 $16 \times 16$（M×K），B 是 $16 \times 8$（K×N），C/D 是 $16 \times 8$（M×N）
- 单个 lane **看不到完整矩阵**，只持有几个寄存器，硬件在 warp 内隐式广播/规约
- 所以你必须精确知道：**每个 lane 的每个寄存器里放的是矩阵的哪个元素** —— 这就是 fragment layout
- 而数据从 global → shared → 寄存器 的每一步，都对应一个 layout 决策（smem 布局、swizzle、ldmatrix、tile 映射）

下面用一个 **CTA 128×128、warp tile 64×64、K-step 64、fp16 输入 + fp32 累加** 的完整例子贯穿全部 10 个要点。

---

## 1. 指令与寄存器 tuple（register tuple）

```cpp
// A: 4 个 .b32 寄存器，每个打包 2 个 fp16（即 half2）
// B: 2 个 .b32 寄存器，每个打包 2 个 fp16
// C/D: 4 个 .f32 寄存器（fp32 累加器）
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
```

| 操作数 | 形状 | 每线程寄存器 | C++ 类型 | 每寄存器内容 |
|---|---|---|---|---|
| A | 16×16 fp16 | 4 × `uint32_t` | `a[4]` | 1 个 half2（2 个连续 k） |
| B | 16×8 fp16 | 2 × `uint32_t` | `b[2]` | 1 个 half2（2 个连续 k） |
| C / D | 16×8 fp32 | 4 × `float` | `c[4]` / `d[4]` | 1 个 fp32 |

> `.row.col` 含义：A 是 row-major fragment（每个寄存器 = 同一行 m 上 2 个连续 k），B 是 col-major fragment（每个寄存器 = 同一列 n 上 2 个连续 k）。这决定了后面 ldmatrix 怎么摆。

---

## 2. Fragment layout（A / B / C）

记 `lane`（0..31），定义 `groupID = lane/4`，`tid_in_group = lane%4`。

### A fragment（16×16，4 个 half2）

$$a_0: A[\tfrac{lane}{4}][2(\tfrac{lane}{4}\bmod 4)+0..1]$$

$$a_1: A[\tfrac{lane}{4}+8][2(\tfrac{lane}{4}\bmod 4)+0..1]$$

$$a_2: A[\tfrac{lane}{4}][2(\tfrac{lane}{4}\bmod 4)+8..9]$$

$$a_3: A[\tfrac{lane}{4}+8][2(\tfrac{lane}{4}\bmod 4)+8..9]$$

```
            k = 0..7          k = 8..15
row lane/4:      a0                a2
row lane/4+8:    a1                a3
```
即：每个 lane 拿 **2 行 × 4 列**，行差 8、列差 8 由寄存器索引表达。

### B fragment（16×8，2 个 half2）

$$b_0: B[2(\tfrac{lane}{4}\bmod 4)+0..1][\tfrac{lane}{4}]$$

$$b_1: B[2(\tfrac{lane}{4}\bmod 4)+8..9][\tfrac{lane}{4}]$$

即：每个 lane 固定一个 n 列（`lane/4`），拿 **4 行 k**（两个 half2，行差 8）。

### C fragment（16×8，4 个 fp32）

$$c_0: C[\tfrac{lane}{4}][2(\tfrac{lane}{4}\bmod 4)],\quad c_1: C[\tfrac{lane}{4}][2(\tfrac{lane}{4}\bmod 4)+1]$$

$$c_2: C[\tfrac{lane}{4}+8][2(\tfrac{lane}{4}\bmod 4)],\quad c_3: C[\tfrac{lane}{4}+8][2(\tfrac{lane}{4}\bmod 4)+1]$$

每个 lane 拿 **2 行 × 2 列** 的 2×2 小块。

> 记忆技巧：`lane/4` 负责行方向（8 组），`lane%4` 负责列/通道方向（4 组）。C 的 `c0,c1` 与 `c2,c3` 行差 8；A 的 `a0↔a1`、`a2↔a3` 行差 8，`a0↔a2`、`a1↔a3` 列差 8；B 的 `b0↔b1` 行(k)差 8。

---

## 3. Tile 映射：CTA → warp → mma

```
CTA 128×128（256 线程 = 8 warps，2×4 排布）
┌─────────────────────────────┐
│ warp(0,0) 64×64 │ warp(0,1) │
│ warp(0,2)       │ warp(0,3) │
│ warp(1,0) ...               │
└─────────────────────────────┘
每个 warp: 4(m) × 8(n) = 32 个 mma.m16n8k16 输出瓦片(16×8)
```

```cpp
const int warp = tid >> 5, lane = tid & 31;
const int wm = warp >> 2;   // 0..1  m 方向
const int wn = warp & 3;    // 0..3  n 方向
// warp(wm,wn) 负责输出行 [64·wm, 64·wm+64)，列 [64·wn, 64·wn+64)
// mma 输出瓦片 (mt, nt) 对应：
//   行 = 64·wm + 16·mt + {0..15}
//   列 = 64·wn +  8·nt + {0..7}
```

**寄存器阻塞（register blocking）**：A 的 16×16 fragment 在 8 个 n-tile 间复用（同一份 `a[4]` 喂 8 次 mma）；B 的 16×8 fragment 在 4 个 m-tile 间复用。于是 64×64 累加器 = `4×8` 个 mma × 4 个 fp32 = **128 个寄存器**，加上 A/B 各 16 个 = ~160 个，正好压在 255 寄存器上限内（这也是 warp tile 通常做 64×64 的原因）。

---

## 4. Shared memory 布局 + swizzle

我们用 **128B XOR swizzle**：16B 为一个基本行块，8 行一组，把"组内行号"与"16B chunk 号"异或，实现 bank 无冲突。

**A smem：128(m) × 64(k) fp16**，每行 128B = 8 个 16B chunk：

```cpp
// c = k/8 ∈ 0..7（16B chunk）
// 128B 组 = 8 行；组基址 = (m/8)*1024B
__device__ size_t sA_off(int m, int k) {
    int c = k >> 3;
    return (size_t)(m >> 3) * 1024 + (((m & 7) ^ c) * 16) + (k & 7) * 2;
}
```

**B smem：64(k) × 128(n) fp16（K-major，行 = k）**，每行 256B = 2 个 128B 组：

```cpp
// c = n/8 ∈ 0..15；在 128B 组内做 ((k&7) ^ (c&7))
__device__ size_t sB_off(int k, int n) {
    int c = n >> 3;
    return (size_t)(k >> 3) * 2048 + ((c >> 3) * 128)
         + (((k & 7) ^ (c & 7)) * 16) + (n & 7) * 2;
}
```

**为什么必须 swizzle？** 以 B 的 `ldmatrix.trans` 为例：不 swizzle 时，8×8 瓦片 8 行都在同一 bank 相位（bank = `4·k`），8 个 lane 同时访问同一 bank → 8 路冲突。swizzle 后 bank = `4·((k&7)^c) + …`，对 8 行是 0,4,…,28 的排列，无冲突。

**bank 计算**：`bank = (byte_addr/4) % 32`。上面两个公式算出的 chunk 起始地址都是 16B 对齐（`sA_off`、`sB_off` 恒为 16 的倍数），满足 ldmatrix/cp.async 的对齐要求。

---

## 5. ldmatrix：smem → 寄存器 fragment

`ldmatrix.sync.aligned.m8n8.x4.shared.b16` 一次把 **4 个 8×8 fp16 瓦片** 载入 4 个寄存器：

- 每个 lane 从每个瓦片拿 2 个 fp16（`lane/4` 行、`2·(lane%4)` 起的 half2）
- 4 个瓦片地址必须**预先写进 smem 里一个 4×32bit 的地址表**（`[addr]` 指向该表）—— 这是最常见的坑
- `.trans` 变体：读入后转置再分发，得到"同一列 n 上 2 个连续 k"——正是 B 的 col fragment

```cpp
__device__ void ldmatrix_x4(uint32_t r[4], const uint32_t* slot) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ void ldmatrix_x4_trans(uint32_t r[4], const uint32_t* slot) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(slot);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
```

**地址顺序必须匹配 mma fragment 顺序**（k-major 的 2×2 瓦片网格）：

- A（无 `.trans`）：`(m0,kk) (m0+8,kk) (m0,kk+8) (m0+8,kk+8)` → `a0 a1 a2 a3`
- B（带 `.trans`）：`(kk,n0) (kk+8,n0) (kk,n0+8) (kk+8,n0+8)` → `b0 b1`（n-tile 2p）与 `b2 b3`（n-tile 2p+1）

---

## 6. 完整 kernel

主循环用 `cp.async` 做双缓冲流水，每阶段 K=64（内部 4 个 k16 子步，每子步 8 次 `ldmatrix.x4` + 32 次 `mma`）。C 的 epilogue 复用 A/B 的 64KB smem 做转存。

<details>
<summary>完整可编译骨架（sm_80+，A/B/C 为 row-major，M/N/K 需为 128/8 的倍数）</summary>

```cpp
#include <cuda_fp16.h>
#include <cstdint>

constexpr int CTA_M = 128, CTA_N = 128;   // CTA 输出瓦片
constexpr int KSTEP = 64;                 // 每流水阶段 K
constexpr int THREADS = 256, WARPS = 8;
constexpr int WARP_M = 64, WARP_N = 64;   // warp 输出瓦片

__shared__ half    sAB[2][2][CTA_M * KSTEP];  // [stage][0=A|1=B]，各 16KB，共 64KB
__shared__ uint32_t sldm[4];                  // ldmatrix 地址表槽

// ---- 128B XOR swizzle（见第 4 节）----
__device__ __forceinline__ size_t sA_off(int m, int k) {
    int c = k >> 3;
    return (size_t)(m >> 3) * 1024 + (((m & 7) ^ c) * 16) + (k & 7) * 2;
}
__device__ __forceinline__ size_t sB_off(int k, int n) {
    int c = n >> 3;
    return (size_t)(k >> 3) * 2048 + ((c >> 3) * 128)
         + (((k & 7) ^ (c & 7)) * 16) + (n & 7) * 2;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// ---- cp.async 16B / commit / wait ----
__device__ __forceinline__ void cp_async16(void* sdst, const void* gsrc) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_u32(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---- ldmatrix ----
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const uint32_t* slot) {
    uint32_t s = smem_u32(slot);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const uint32_t* slot) {
    uint32_t s = smem_u32(slot);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// ---- mma.m16n8k16 ----
__device__ __forceinline__ void mma_m16n8k16(float d[4], const uint32_t a[4],
                                             const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---- 流水线取数：global -> smem（带 swizzle）----
__device__ __forceinline__ void load_stage(const half* A, const half* B,
        int N, int K, int gk, int stage, int bx, int by, int tid) {
    half* sA = sAB[stage][0];
    half* sB = sAB[stage][1];

    #pragma unroll
    for (int i = 0; i < (CTA_M * KSTEP / 16) / THREADS; i++) {   // 4 次/线程
        int cid = tid + i * THREADS;        // 0..1023
        int m   = cid >> 3;                 // 行 0..127
        int kc  = cid & 7;                  // k chunk 0..7 -> k=8*kc
        const half* src = A + (size_t)(by * CTA_M + m) * K + gk + 8 * kc;
        half* dst = (half*)((char*)sA + sA_off(m, 8 * kc));
        cp_async16(dst, src);               // 16B：8 个连续 k
    }
    #pragma unroll
    for (int i = 0; i < (CTA_N * KSTEP / 16) / THREADS; i++) {
        int cid = tid + i * THREADS;        // 0..1023
        int k   = cid >> 4;                 // k 行 0..63
        int nc  = cid & 15;                 // n chunk 0..15 -> n=8*nc
        const half* src = B + (size_t)(gk + k) * N + bx * CTA_N + 8 * nc;
        half* dst = (half*)((char*)sB + sB_off(k, 8 * nc));
        cp_async16(dst, src);               // 16B：同一 k 行上 8 个连续 n
    }
}

// ---- 主计算：每 k16 子步 8 次 ldmatrix.x4 + 32 次 mma ----
__device__ __forceinline__ void mma_mainloop(int wm, int wn, int stage,
                                             float c[4][8][4]) {
    const half* sA = sAB[stage][0];
    const half* sB = sAB[stage][1];

    #pragma unroll
    for (int kk = 0; kk < KSTEP; kk += 16) {
        uint32_t a[4][4], b[4][4];

        // A：4 个 m-tile，各 1 次 ldmatrix.x4（无 trans）
        #pragma unroll
        for (int mt = 0; mt < 4; mt++) {
            int m0 = wm * WARP_M + mt * 16;
            sldm[0] = smem_u32((char*)sA + sA_off(m0,     kk));
            sldm[1] = smem_u32((char*)sA + sA_off(m0 + 8, kk));
            sldm[2] = smem_u32((char*)sA + sA_off(m0,     kk + 8));
            sldm[3] = smem_u32((char*)sA + sA_off(m0 + 8, kk + 8));
            __syncwarp();                       // 写地址表 -> ldmatrix
            ldmatrix_x4(a[mt], sldm);
        }

        // B：4 个 n-tile 对，各 1 次 ldmatrix.x4（带 trans）
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int n0 = wn * WARP_N + p * 16;
            sldm[0] = smem_u32((char*)sB + sB_off(kk,     n0));
            sldm[1] = smem_u32((char*)sB + sB_off(kk + 8, n0));
            sldm[2] = smem_u32((char*)sB + sB_off(kk,     n0 + 8));
            sldm[3] = smem_u32((char*)sB + sB_off(kk + 8, n0 + 8));
            __syncwarp();
            ldmatrix_x4_trans(b[p], sldm);
        }

        // 32 次 mma：a[mt] 复用 8 次，b[p] 复用 4 次
        #pragma unroll
        for (int mt = 0; mt < 4; mt++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const uint32_t* bp = b[nt >> 1];
                uint32_t bv[2] = { bp[(nt & 1) * 2], bp[(nt & 1) * 2 + 1] };
                mma_m16n8k16(c[mt][nt], a[mt], bv, c[mt][nt]);
            }
    }
}

__global__ __launch_bounds__(THREADS)
void gemm_f16(const half* A, const half* B, float* C, int M, int N, int K) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;      // 2×4 warp 网格

    float c[4][8][4];                             // 64×64 累加器 = 128 个寄存器
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            #pragma unroll
            for (int r = 0; r < 4; r++) c[mt][nt][r] = 0.f;

    load_stage(A, B, N, K, 0, 0, bx, by, tid);
    cp_async_commit();

    for (int gk = 0; gk < K; gk += KSTEP) {
        int stage = (gk / KSTEP) & 1;
        if (gk + KSTEP < K) {                     // 预取下一阶段
            load_stage(A, B, N, K, gk + KSTEP, stage ^ 1, bx, by, tid);
            cp_async_commit();
        }
        cp_async_wait<1>();                       // 等当前阶段数据就绪
        __syncthreads();

        mma_mainloop(wm, wn, stage, c);

        __syncthreads();                          // 所有 warp 读完，才能被覆盖
    }

    // ---- epilogue：C 寄存器 -> smem（复用 sAB）-> global ----
    float* sC = reinterpret_cast<float*>(sAB);
    #pragma unroll
    for (int mt = 0; mt < 4; mt++)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++)
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int row = wm * WARP_M + mt * 16 + ((r < 2) ? (lane >> 2) : (lane >> 2) + 8);
                int col = wn * WARP_N + nt * 8 + 2 * (lane & 3) + (r & 1);
                sC[row * CTA_N + col] = c[mt][nt][r];
            }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < CTA_M * CTA_N / THREADS; i++) {  // 64 次/线程
        int idx = tid + i * THREADS;
        int row = idx >> 7, col = idx & (CTA_N - 1);
        C[(size_t)(by * CTA_M + row) * N + bx * CTA_N + col] = sC[idx];
    }
}
// 启动：gemm_f16<<<dim3(N/CTA_N, M/CTA_M), 256>>>(A, B, C, M, N, K);
```
</details>

---

## 7. 十个要点 → 代码对照表

| 概念 | 在示例中的体现 | 出错时的典型症状 |
|---|---|---|
| **warp mapping** | `wm = warp>>2, wn = warp&3`；8 warps 覆盖 128×128 | 输出瓦片错位/越界 |
| **register tuple** | A=`uint32_t[4]`、B=`uint32_t[2]`、C/D=`float[4]`；`"r"`/`"f"` 约束 | 寄存器类型/数量不匹配编译失败 |
| **A fragment layout** | 行 `lane/4`、`+8`，列 `2(lane%4)`、`+8`；ldmatrix 无 trans | 数值错但形状对 |
| **B fragment layout** | 固定 n=`lane/4`，k 对 `2(lane%4)`、`+8`；ldmatrix 带 trans | 同上（且不转置会全错） |
| **C fragment layout** | `c0..c3` = 2×2 块（行差 8，列差 1）；epilogue 回写 | epilogue 错位 |
| **shared-memory layout** | `sA` 128×64（128B 行）、`sB` 64×128（256B 行） | 对齐/越界错误 |
| **ldmatrix layout** | x4 的 4 地址按 k-major 2×2 网格排列，地址表放 smem | 忘记地址表 → 非法地址 |
| **swizzle** | `((m&7)^c)*16` / `((k&7)^(c&7))*16` 的 128B XOR | 性能骤降（bank conflict） |
| **warp tile mapping** | 每 warp 64×64 = 4×8 个 mma 瓦片 | 寄存器复用逻辑错 |
| **CTA tile mapping** | `by*128 + row`、`bx*128 + col`；grid 2D | 结果写到错误位置 |

### 最容易踩的 4 个坑
1. **ldmatrix.x4 的地址表必须存在 smem**（`[addr]` 指向 4 个连续 32bit 地址），不能直接传寄存器地址。
2. **B 必须用 `.trans` 且 smem 按 K-major 存**，否则拿到的是 n-major fragment，与 `.row.col` 不匹配。
3. **swizzle 只在 16B chunk 粒度做**，且 ldmatrix/cp.async 的源、宿地址都要保持 16B 对齐（本例 `k`、`n` 都取 8 的倍数）。
4. **双缓冲的同步顺序**：`wait_group → __syncthreads() → compute → __syncthreads()`，第二个 sync 保证下一阶段 cp.async 不会覆盖正在读的 buffer。

### 性能预期（简评）
- 每 k16 子步：8 次 `ldmatrix.x4` + 32 次 `mma` → 每 warp 每 k16 算 $64\times64\times16\times2 = 131$ KFLOP；K=64 阶段 4 子步。
- 寄存器压力 ~160/255，无溢出；如需更激进可用 `-maxrregcount` 或 `__launch_bounds__` 调整。
- 该结构可直接替换成 Hopper 的 `wgmma.mma_async`（把 smem→寄存器→mma 换成 smem→tensor memory 直连），fragment/warp 映射的思维完全一致。

需要的话，我可以再给出：**归一化（alpha/beta）epilogue、边界裁剪（非 128 倍数）、或 Hopper wgmma 版本**。