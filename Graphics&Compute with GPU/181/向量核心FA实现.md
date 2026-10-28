# 向量核心FA实现

## 🧑‍💻 User

如何在gtx980ti上用vec core实现flashattention？

---
## 😀 默认助手 (7)

# 在 GTX 980 Ti 上用向量核心（Vector / CUDA Core）实现 FlashAttention

## 先把两个问题讲清楚

**1）“vec core”指什么**
- 如果指的是**非 Tensor Core 的普通 FP32/SIMT 通路**（NVIDIA 文档里的 vector core / CUDA core），那么这正是 980 Ti 唯一能走的路线，因为 GM200 **没有任何 Tensor Core**，也没有 `mma.sync`、`wmma`、`cp.async`、`ldmatrix`。
- 如果指的是 **llama.cpp 的 `fattn-vec` 内核家族**（它在没有 Tensor Core 的卡上会自动回退到那批 `fattn-vec-*.cuh`）：见文末“现成方案”，可直接使用，不必自己实现。

**2）可行性结论**：可以，但本质上是「手写一个寄存器分块（register-blocked）的 FA1 前向内核」。上限是 980 Ti 的 FP32 峰值约 $6.0\ \text{TFLOP/s}$（2816 CUDA core × 2 × 1.076 GHz），实际可做到 2–3.5 TFLOP/s。它不会接近 A100 上的 FA2/FA3，但在 $N\ge 2\text{K}$、$D=64$ 时仍比朴素 attention 快 3–4 倍，主要收益来自**不再将 $N\times N$ 的 score 矩阵写入 HBM**。

---

<details>
<summary><b>一、硬件约束：这些数字直接决定内核形状</b></summary>

| 项目 | GTX 980 Ti (GM200, `sm_52`) | 对实现的影响 |
|---|---|---|
| SM 数 / FP32 | 22 / 128 每 SM | 所有计算走 FP32 FMA，别指望半精度加速 |
| Tensor Core | **无** | 无 `wmma`/`mma`，只能 SIMT 矩阵乘 |
| FP16 打包运算 | **无**（`__hfma2` 需 `sm_53`） | FP16 只能当存储格式，转换开销大于收益 → **全程 FP32** |
| 共享内存 | 96 KB/SM，**48 KB/block 上限** | 分块预算的硬性限制，`Br/Bc/D` 三角权衡 |
| 共享内存带宽 | ≈128 B/clk/SM | 每从 smem 读 1 个 float，必须换来 ≥4 次 FMA，否则会被 LDS 带宽限制 |
| `cp.async` / 异步拷贝 | 无（Ampere+） | 无法做 FA3 式流水线，只能双缓冲或接受 `__syncthreads` |
| SFU（`ex2`） | ≈1/4 FP32 速率 | 用 `exp2f` 并把 $\log_2 e$ 折进 scale |
| HBM / L2 | 336 GB/s / 3 MB | $D{=}64$ FP32 时每 2K token 的 K+V 仅 1 MB → **前向基本是 compute-bound，L2 命中友好** |
| 软件栈 | CUDA ≤ 12.6 仍支持 `sm_52`（CUDA 13 移除）；Triton 需 Volta+ | **Triton / flash-attn (Dao-AILab) 都用不了**，必须 CUDA C++ |

</details>

<details>
<summary><b>二、算法核心：online softmax（必须记住的三个递推）</b></summary>

对固定 query 行 $i$，按 KV block $j$ 流式更新：

$$
m_i^{(j)}=\max\!\big(m_i^{(j-1)},\tilde m_i\big),\qquad
\ell_i^{(j)}=e^{m_i^{(j-1)}-m_i^{(j)}}\,\ell_i^{(j-1)}+\tilde\ell_i
$$

$$
O_i^{(j)}=e^{m_i^{(j-1)}-m_i^{(j)}}\,O_i^{(j-1)}+e^{\tilde m_i-m_i^{(j)}}\,\tilde O_i
$$

其中 $\tilde m_i,\tilde\ell_i,\tilde O_i$ 是当前 block 内的局部统计量。全程 FP32 累加，数值上不需要任何技巧；唯一要小心的是**整块被 mask 掉时 $m=-\infty$ 会产生 `NaN`**，需要显式守卫（代码里会标出）。

</details>

## 三、分块与线程映射（$D=64$ 的推荐配置）

| 符号 | 取值 | 说明 |
|---|---|---|
| `Br` | 64 | 每 block 的 query 行数 |
| `Bc` | 32 | 每次迭代的 key 数，等于一个 warp 的 lane 数 |
| `NW`、线程 | 8 warp、256 线程 | 每 warp 独占 8 行 query |
| GEMM1 映射 | `lane = key 列`，float4 沿 $d$ 向量化 | `S[r][lane] = <Q[r], K[lane]>` |
| GEMM2 映射 | `lane` 持有 2 个输出维度 $d=2\cdot\text{lane}$ | 每线程 8 行 × 2 列 = 16 个输出寄存器 |

**共享内存预算**（关键：48 KB/block 限制）

| 数组 | 布局 | 大小 |
|---|---|---|
| `Qs[64][64]` | row-major | 16.0 KB |
| `Ks[32][68]` | row-major, stride 68 | 8.7 KB |
| `Vs[32][68]` | row-major, stride 68 | 8.7 KB |
| `Ps[64][36]` | row-major, stride 36 | 9.2 KB |
| **合计** | | **≈42.6 KB** → 2 blocks/SM，512 线程/SM |

padding 规则如下，都是为消除 bank conflict：`V/K` stride 取 $D+4=68$（$68\bmod 16$ 使 `float4` 对齐且 8-lane 相位内 bank 互斥），`P` stride 取 $Bc+4=36$。

> `D=128` 时不要 stage Q（`64×128` 就要 32 KB），把 Q 行放进寄存器（每线程 $8\times 128/32=32$ 个），smem 只放 `Ks/Vs/Ps` ≈ 43 KB。

---

<details>
<summary><b>四、参考实现（单头、前向、FP32、sm_52；已包含 causal mask）展开看</b></summary>

```cuda
// nvcc -arch=sm_52 -O3 --use_fast_math -lineinfo
// 目标：GTX 980 Ti (GM200)。单头、前向、D=64、FP32。
#include <cuda_runtime.h>

namespace {
constexpr int D   = 64;          // head_dim
constexpr int BR  = 64;          // query rows / block
constexpr int BC  = 32;          // keys per KV step
constexpr int NW  = 8;           // warps per block
constexpr int NT  = NW * 32;
constexpr int RPW = BR / NW;     // 8 rows per warp
constexpr int DPW = D / 32;      // 2 output dims per lane

constexpr int KS = D  + 4;       // smem stride for K/V  (68, 16B 对齐)
constexpr int PS = BC + 4;       // smem stride for P    (36)

__device__ __forceinline__ float shfl_xor(float v, int m) {
  return __shfl_xor_sync(0xffffffffu, v, m);
}

__global__ void __launch_bounds__(NT, 2)
fa_fwd(const float* __restrict__ Q, const float* __restrict__ K,
       const float* __restrict__ V, float* __restrict__ O,
       int N, float softmax_scale, int causal)
{
  __shared__ float Qs[BR][D];
  __shared__ float Ks[BC][KS];
  __shared__ float Vs[BC][KS];
  __shared__ float Ps[BR][PS];

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int q0   = blockIdx.x * BR;
  const int lrow = warp * RPW;          // warp 在 Ps 中的首行
  const int row_w = q0 + lrow;

  // ---------- 1. stage Q (float4) ----------
  for (int i = tid; i < BR * (D / 4); i += NT) {
    int r = i / (D / 4), c = (i % (D / 4)) * 4;
    int gr = q0 + r;
    float4 q4 = (gr < N) ? *reinterpret_cast<const float4*>(Q + (size_t)gr * D + c)
                         : make_float4(0, 0, 0, 0);
    *reinterpret_cast<float4*>(&Qs[r][c]) = q4;
  }
  __syncthreads();

  // ---------- 2. 线程私有状态 ----------
  float m_i[RPW], l_i[RPW], Oacc[RPW][DPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) { m_i[r] = -INFINITY; l_i[r] = 0.f; }
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int d = 0; d < DPW; ++d) Oacc[r][d] = 0.f;

  const float qk = softmax_scale * 1.4426950408889634f;  // log2(e) 折进 scale

  const int nkb = (N + BC - 1) / BC;
  for (int kb = 0; kb < nkb; ++kb) {
    const int k0 = kb * BC;
    if (causal && k0 > q0 + BR - 1) break;      // 整块跳过（block 内一致，break 安全）

    // ---------- 3. stage K / V ----------
    for (int i = tid; i < BC * (D / 4); i += NT) {
      int c = i / (D / 4), d4 = (i % (D / 4)) * 4, gc = k0 + c;
      bool ok = (gc < N);
      float4 k4 = ok ? *reinterpret_cast<const float4*>(K + (size_t)gc * D + d4)
                     : make_float4(0, 0, 0, 0);
      float4 v4 = ok ? *reinterpret_cast<const float4*>(V + (size_t)gc * D + d4)
                     : make_float4(0, 0, 0, 0);
      *reinterpret_cast<float4*>(&Ks[c][d4]) = k4;
      *reinterpret_cast<float4*>(&Vs[c][d4]) = v4;
    }
    __syncthreads();

    // ---------- 4. GEMM1: S[r][lane] = Q[r] · K[lane] ----------
    float s[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) s[r] = 0.f;

#pragma unroll 4
    for (int d = 0; d < D; d += 4) {
      float4 k4 = *reinterpret_cast<const float4*>(&Ks[lane][d]);  // 每 lane 独占
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        float4 q4 = *reinterpret_cast<const float4*>(&Qs[lrow + r][d]); // 广播
        s[r] = fmaf(q4.x, k4.x, s[r]);
        s[r] = fmaf(q4.y, k4.y, s[r]);
        s[r] = fmaf(q4.z, k4.z, s[r]);
        s[r] = fmaf(q4.w, k4.w, s[r]);
      }
    }

    // ---------- 5. online softmax ----------
    float m_new[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      float t = s[r] * qk;
      int gc = k0 + lane;
      if ((causal && gc > row_w + r) || gc >= N) t = -INFINITY;  // mask
      s[r] = t;
      float mx = t;
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) mx = fmaxf(mx, shfl_xor(mx, m));
      m_new[r] = fmaxf(m_i[r], mx);
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      // 整行全被 mask 时 exp2(-inf - -inf) = NaN，必须守卫
      float corr = (m_new[r] == -INFINITY && m_i[r] == -INFINITY)
                     ? 1.f : exp2f(m_i[r] - m_new[r]);
      float p    = exp2f(s[r] - m_new[r]);        // 0 when masked
      s[r] = p;
      l_i[r] *= corr;
#pragma unroll
      for (int d = 0; d < DPW; ++d) Oacc[r][d] *= corr;

      float psum = p;
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) psum += shfl_xor(psum, m);
      l_i[r] += psum;
      m_i[r]  = m_new[r];
      Ps[lrow + r][lane] = p;
    }
    __syncthreads();   // Ps 可见

    // ---------- 6. GEMM2: Oacc += P · V ----------
#pragma unroll 1
    for (int c = 0; c < BC; ++c) {
      float2 v2 = *reinterpret_cast<const float2*>(&Vs[c][2 * lane]);
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        float p = Ps[lrow + r][c];               // 广播
        Oacc[r][0] = fmaf(p, v2.x, Oacc[r][0]);
        Oacc[r][1] = fmaf(p, v2.y, Oacc[r][1]);
      }
    }
    __syncthreads();   // 保护 Ps/Ks/Vs 在下一轮被复用
  }

  // ---------- 7. epilogue: O = Oacc / l ----------
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    int gr = row_w + r;
    if (gr >= N) continue;
    float inv = 1.f / l_i[r];
    float2 o2 = make_float2(Oacc[r][0] * inv, Oacc[r][1] * inv);
    *reinterpret_cast<float2*>(O + (size_t)gr * D + 2 * lane) = o2;
  }
}
} // namespace
```

**要接到真实模型上还差**：3D grid `(ceil(N/BR), nheads, nbatch)` 加指针偏移（`Q += ((size_t)b*nheads + h)*N*D`）；GQA/MQA 的 KV head 映射；dropout（可省略）；反向（重算 $S$，不存 $P$，寄存器压力翻倍，建议先只做前向）。

</details>

<details>
<summary><b>五、Maxwell 专属调优清单（按收益排序）</b></summary>

1. **`float4` 全局 + 共享内存访问全对齐**。Q/K/V 指针 16B 对齐，`D % 4 == 0`，行 stride 16B 对齐（$D=64$ FP32 天然 256B）。这是唯一免费的翻倍。
2. **thread tile 比率 ≥ 4**。每读 1 个 smem float 要有 ≥4 个 FMA，否则被 128 B/clk 的 LDS 带宽限制。上面 GEMM1 靠“K 独占 + Q 广播”达到 ~78% FMA 密度。
3. **`exp2f` + 把 $\log_2 e$ 折进 `softmax_scale`**，配合 `--use_fast_math` 走 `MUFU.EX2`。
4. **不要用双精度**：980 Ti 的 FP64 是 1/32（≈190 GFLOP/s）。
5. **不要用 FP16 存储**：`sm_52` 没有 `__hfma2`，每次 FMA 前都要 `cvt`，净亏。想省 smem 就减小 `Bc`。
6. **没有 `cp.async`**，别抄 FA2/FA3 的 pipeline：预取到寄存器 + 单次 `__syncthreads` 就够，本内核计算量足够掩盖 8.7 KB 的搬运。
7. **causal 时块级跳过**（`break`），对角块做逐元素 mask；这是长序列下 2× 的收益。
8. **reg/occupancy**：`__launch_bounds__(256, 2)`，目标 ≤128 regs/thread（65536/512）。想再提高 occupancy 只能缩 `Br`（会掉 FMA 密度），通常不值得。
9. **L2 是朋友**：3 MB L2 + $D{=}64$ 时 2K token 的 K+V 只有 1 MB，前向多 block 共享 KV 会大量命中 L2。
10. 性能核算：$\text{FLOPs} \approx 4N^2D$。$N{=}4096,D{=}64$ 每头 4.3 GFLOP；@2.5 TFLOP/s ≈ 1.7 ms/头。对比朴素实现仅 $S$ 矩阵往返就 $\sim$400 MB → 1.2 ms/头，**FA 的收益来自访存收益，而非算力**。

</details>

## 六、不想手写？直接用现成的（更现实的选择）

| 方案 | 是否支持 `sm_52` | 说明 |
|---|---|---|
| **llama.cpp 的 `fattn-vec` 内核** | ✅ | `ggml/src/ggml-cuda/fattn-vec*.cuh`，就是“无 Tensor Core 时的向量核心 FA”，`-fa` 开启；在 Maxwell/Pascal 上会自动走 vec 路径。这是你问题的最短路径。 |
| **xformers `memory_efficient_attention`** | ✅（需确认版本） | cutlass-based fMHA 有 `sm_50` 分支，但新版本可能只发 Volta+ 的 wheel；用旧 wheel 或源码编译。 |
| **flash-attn (Dao-AILab)** | ❌ | 需要 `sm_80+`（FP16 Tensor Core） |
| **Triton 版 FA** | ❌ | Triton 已要求 Volta+ |

软件栈务必注意：用 **CUDA 11.8 或 ≤12.6**（CUDA 13 已移除 Maxwell），`nvcc -gencode arch=compute_52,code=sm_52`；PyTorch 官方 wheel 的 `sm_52` 支持逐年收缩，先 `python -c "import torch;print(torch.cuda.get_arch_list())"` 确认，缺了就 `TORCH_CUDA_ARCH_LIST=5.2` 源码编译或退回 2.1.x+cu118。

## 七、一句话决策

- **只想起作用** → 用 llama.cpp 的 `fattn-vec`（或 xformers），别重写。
- **想自己实现/学习** → 按上面的 `Br=64 / Bc=32 / 8 warp / FP32-only` 骨架写，先保证正确性（尤其 mask 和 `m=-inf` 守卫），再用 `float4` + 广播/独占布局把 FMA 密度推过 70%，预期 2–3.5 TFLOP/s，$N\ge2\text{K}$ 时比朴素 attention 快 3–4×。
- **反向传播、长上下文、decode 吞吐** → 980 Ti 的 336 GB/s 才是真正的天花板（decode 时每 token 要扫全部 KV cache），算法优化改变不了这一点，先算清这条内存账再投入。
