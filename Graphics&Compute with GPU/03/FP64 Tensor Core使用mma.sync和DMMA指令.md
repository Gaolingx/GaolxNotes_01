# FP64 Tensor Core使用mma.sync和DMMA指令

### 🧑‍💻 User

fp64 tensorcore mma一般用什么指令？

---

### 🤖 Assistant

**主要用 PTX 的 `mma.sync`（对应 SASS 的 `DMMA`），形状通常是 `m8n8k4`。**

### 1. PTX 层面（最常用、可直接写 inline PTX / 被编译器生成）
FP64 Tensor Core MMA 对应的 PTX 指令是：

```ptx
mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
```

典型写法示例：

```ptx
mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
    {%d0, %d1},          // D (2×f64)
    {%a0, %a1},          // A (2×f64)
    {%b0},               // B (1×f64)
    {%c0, %c1};          // C (2×f64)
```

- **形状固定**：`m8n8k4`（这是 Ampere 及以后 FP64 Tensor Core 的原生形状）。
- 修饰符常见组合：`.aligned`、`.row.col`（或 `.col.row` 等 layout）。
- 从 Ampere（`sm_80`）开始支持，Hopper/Ada/Blackwell 等后续架构继续支持（具体吞吐和是否走专用 Tensor Core 路径以对应白皮书为准）。

在 CUDA C++ 里也可以通过 `nvcuda::wmma`（`fragment` + `mma_sync`）使用 double，底层最终还是落到这类 `mma` 指令；高性能代码（CUTLASS、自研 kernel 等）更常直接写 inline PTX 或使用 CUTLASS 的 MMA atom。

### 2. SASS 层面（最终机器码）
编译后对应的 SASS 指令一般是 **`DMMA`**（Double-precision Matrix Multiply-Accumulate），常见形态类似 `DMMA.884` / 带 layout 与 shape 的变体（具体编码随架构和工具链略有差异）。

对比其他精度：
- FP16 / BF16 / TF32 → 多为 `HMMA`
- INT8/INT4 等 → `IMMA` 等
- FP64 → `DMMA`

### 使用注意点
- **架构要求**：需要 `sm_80+`（A100 及以后）。更早的 Volta/Turing 没有 FP64 Tensor Core。
- **寄存器与 layout**：A/B/C/D 的 fragment 布局、寄存器数量与 `m8n8k4` 严格对应，写错 layout 会直接编译失败或结果错误。
- **性能现实**：FP64 Tensor Core 吞吐远低于 FP16/TF32/BF16 Tensor Core，主要用于需要双精度的科学计算场景；很多时候密集 FP64 GEMM 仍会混用普通 FFMA + Tensor Core 路径，具体以 cuBLAS / CUTLASS 调优结果为准。
- **Hopper 及以后**：大矩阵更常见 `wgmma`（warp-group），但 FP64 的基础 MMA 形态仍以 `mma`/`DMMA` 为主；具体支持以 PTX ISA 与对应架构文档为准。
- 实际开发建议：优先看 CUTLASS 的 FP64 MMA atom / cuBLASLt，或直接反汇编（`cuobjdump -sass`）确认自己的 kernel 是否真正发出了 `DMMA`。

**总结**：日常说的 “FP64 Tensor Core MMA 指令” 就是 **PTX `mma.sync.aligned.m8n8k4....f64.f64.f64.f64`**，落到 SASS 就是 **`DMMA`**。

---

### 🧑‍💻 User

#if __CUDA_ARCH__ >= 800
__device__ void mma_f64f64f64_8_8_4_(double *data)
{
    double d0 = 0.0;
    double d1 = 0.0;
    double a = 0.0;
    double b = 0.0;
    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++)
    {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
            : "=d"(d0), "=d"(d1)
            : "d"(a), "d"(b), "d"(d0), "d"(d1));
        __syncwarp();
    }
    // Each thread holds 2 f64 accumulators; 32 threads cover 8x8 = 64 elements
    double *ptr = &data[threadIdx.y * 8 * 8];
    ptr[threadIdx.x * 2 + 0] = d0;
    ptr[threadIdx.x * 2 + 1] = d1;
}

__global__ void mma_f64f64f64_8_8_4(void *data, int *rc)
{
    mma_f64f64f64_8_8_4_((double *)data);
    *rc = 0;
}
为什么会报非法指令？

---

### 🤖 Assistant

**原因：PTX 里写了不存在的 FP64 MMA 指令形态，导致非法指令（illegal instruction）。**

### 核心错误
你写的指令是：

```ptx
mma.sync.aligned.m16n8k32.row.col.f64.f64.f64.f64
```

**FP64 Tensor Core 根本不支持 `m16n8k32` 这个 shape**。

- Ampere（`sm_80`）及以后，**FP64 MMA 唯一合法的 shape 是 `m8n8k4`**。
- `m16n8k32` 是 INT8 / 某些低精度指令常用的 shape，和 `.f64` 组合在一起是**非法的**。
- 结果就是：要么 `ptxas` 直接报错，要么生成了无效编码，运行时 GPU 抛出 **illegal instruction**（`cudaErrorIllegalInstruction` 或 kernel 直接 trap）。

函数名和注释写的是 `8_8_4`，但 asm 却写成了 `m16n8k32`，明显是从其他精度（比如 INT8/FP16）的代码复制过来改精度时没改 shape。

### 正确的指令应该是
```ptx
mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
```

对应的操作数数量也必须匹配（A 需要 2 个 double，B 需要 1 个，C/D 各 2 个）：

```ptx
mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64
    {%0, %1},          // D (2 × f64)
    {%2, %3},          // A (2 × f64)
    {%4},              // B (1 × f64)
    {%5, %6};          // C (2 × f64)
```

### 修正后的示例代码
```cuda
#if __CUDA_ARCH__ >= 800
__device__ void mma_f64f64f64_8_8_4_(double *data)
{
    double d0 = 0.0, d1 = 0.0;
    double a0 = 0.0, a1 = 0.0;
    double b  = 0.0;

    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++)
    {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
            : "=d"(d0), "=d"(d1)
            : "d"(a0), "d"(a1), "d"(b), "d"(d0), "d"(d1));
        // __syncwarp();   // mma.sync 已经是 warp 同步的，通常可去掉
    }

    // 注意：这里只是简单把每个线程的 2 个 accumulator 写出去
    // 真正的 8×8 结果布局要严格按 m8n8k4 的 fragment mapping 来，
    // 下面这种写法只适合做吞吐 micro-benchmark，不能当正确结果用。
    double *ptr = &data[threadIdx.y * 8 * 8];
    ptr[threadIdx.x * 2 + 0] = d0;
    ptr[threadIdx.x * 2 + 1] = d1;
}
#endif
```

### 其他可能导致 illegal instruction 的原因（排查清单）

| 检查项 | 说明 | 建议 |
|--------|------|------|
| **GPU 架构** | 必须 Ampere 及以上（A100 / 30系 / 40系 / H100 等） | `nvidia-smi` 看 Compute Capability ≥ 8.0 |
| **编译架构** | 必须 `-arch=sm_80` 或更高（或 fatbin 包含 sm_80+） | `nvcc -arch=sm_80 ...` 或 `-gencode=arch=compute_80,code=sm_80` |
| **运行时选择** | 如果 fatbin 里只有 sm_80+ 的代码，在 Volta/Turing 上跑就会 illegal | 用正确的 GPU 或生成多 arch fatbin |
| **操作数数量** | 原代码 A/B 只给了 1 个寄存器，和任何合法 mma 都不匹配 | 按上面改成 A×2 + B×1 |
| **结果正确性** | 即使指令合法，a0/a1/b 全是 0 且没有按 fragment 布局填数，结果也是错的 | 做正确性验证时必须按 PTX 文档的 register mapping 填 A/B |

### 快速验证方法
1. 改成 `m8n8k4` + 正确操作数后重新编译。
2. 用 `cuobjdump -sass your_binary` 看是否真正生成了 `DMMA` 指令。
3. 确认运行的 GPU 是 sm_80+。

**总结**：非法指令的直接原因就是 **`m16n8k32` + `.f64` 这个组合根本不存在**。改成 `m8n8k4` 并配上正确的寄存器数量即可。

---

### 🧑‍💻 User

查询网页，在SM80（A100）/SM90（H100/H200）/SM100–SM110（B200/B300）等架构上，合法的fp64 mma指令有哪些（m8n8k4 m16n8k4 m16n8k8 m16n8k16）？

---

### 🤖 Assistant

已查询 NVIDIA 官方 PTX ISA 文档（v9.3），结论如下。

## 合法 FP64 `mma.sync` 指令与架构支持

| shape | PTX 语法形态 | 最小架构要求 | SM80 A100 | SM90 H100/H200 | SM100–SM110 B200/B300 |
|---|---|---|---|---|---|
| `.m8n8k4` | `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64` | **sm_80+** | ✅ | ✅ | ✅ |
| `.m16n8k4` | `mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64` | **sm_90+** | ❌ | ✅ | ✅ |
| `.m16n8k8` | `mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64` | **sm_90+** | ❌ | ✅ | ✅ |
| `.m16n8k16` | `mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64` | **sm_90+** | ❌ | ✅ | ✅ |

来源（PTX ISA 9.3，§9.7.15.5.14 `mma` 的 Target ISA Notes）：
- `m8n8k4`：**PTX 7.0 引入**，要求 `sm_80` 或更高。
- `m16n8k4` / `m16n8k8` / `m16n8k16`：**PTX 7.8 引入**，要求 `sm_90` 或更高。

> ⚠️ 所以 A100 上只有 `m8n8k4` 合法；`m16n8k4/8/16` 在 A100 上会非法（编译报错或运行时 illegal instruction）。

## 各 shape 的寄存器 fragment 数量

| shape | A（f64） | B（f64） | C/D（f64） |
|---|---|---|---|
| `m8n8k4` | 1 个寄存器 | 1 个寄存器 | 2 个寄存器 |
| `m16n8k4` | 2 个寄存器 | 1 个寄存器 | 4 个寄存器 |
| `m16n8k8` | 4 个寄存器 | 2 个寄存器 | 4 个寄存器 |
| `m16n8k16` | 8 个寄存器 | 4 个寄存器 | 4 个寄存器 |

示例（官方文档写法）：

```ptx
// m16n8k8
mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64.rn
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3};
```

## 关键注意点

1. **layout 固定为 `row.col`**：FP64 的 `mma` 只支持 `row.col`（A row-major、B col-major），没有 `.col.row` 等其它组合（对比 f16 才有多种 layout）。

2. **舍入修饰符**：FP64 `mma` 支持 `.rn`（默认）、`.rz`、`.rm`、`.rp`。

3. **寄存器上限要求**：文档明确说明，使用 `.m16n8k4/8/16` 的 FP64 `mma` 程序**编译时至少需要 64 个寄存器**（否则编译/性能可能有问题）。

4. **`mma.sp`（稀疏）不支持 f64**：稀疏 MMA 只有 f16/bf16/tf32/int8/int4/fp8 等，**没有 FP64 稀疏变体**。

5. **`wgmma`（Hopper warpgroup MMA）不支持 f64**：`wgmma.mma_async` 只支持 f16/bf16/tf32/fp8/int8/int4/b1，FP64 仍需用 warp 级 `mma.sync`。

6. **Blackwell 的 `tcgen05.mma`（第5代 TensorCore）也不暴露 f64**：其 `.kind` 只有 f16/tf32/f8f6f4/i8/mxf8f6f4/mxf4/mxf4nvf4。所以在 SM100/SM110 上做 FP64 Tensor Core GEMM，实际走的还是 `mma.sync`（m8n8k4 / m16n8k4 / m16n8k8 / m16n8k16）。

7. **另有 `wmma` 级别 API**：`wmma.mma.sync.aligned.m8n8k4...f64`（fragment 布局由硬件抽象，无需手动 layout），同样要求 `sm_80+`，但它只有 `m8n8k4` 一个 shape，且是高层封装，底层仍是 `mma.sync`/`DMMA`。

**一句话总结**：SM80 只有 `m8n8k4`；SM90 及以后（含 SM100–SM110）四种 `m8n8k4 / m16n8k4 / m16n8k8 / m16n8k16` 全部合法。

---

### 🧑‍💻 User

SM86/87, SM89 (Ada), SM120+ (consumer Blackwell)全部不支持fp64 mma吗？

---

### 🤖 Assistant

已核实（来源：NVIDIA PTX ISA 9.3 官方文档、NVIDIA 各代架构白皮书、Blackwell 消费/数据中心对照 wiki、TechPowerUp 等）。结论：

**是的，SM86/87、SM89（Ada）、SM120/121（消费级 Blackwell）都不支持原生 FP64 MMA（FP64 Tensor Core）。**

## 核心结论表

| 架构 | CC | 代表芯片 | 原生 FP64 Tensor Core MMA |
|---|---|---|---|
| Ampere 数据中心 | **sm_80** | A100 / A800（GA100） | ✅ 仅 `m8n8k4` |
| Ampere 消费/嵌入式 | **sm_86 / sm_87** | RTX 30 系（GA102）、Jetson Orin（GA10B） | ❌ |
| Ada 消费 | **sm_89** | RTX 40 系（AD102 等） | ❌ |
| Hopper 数据中心 | **sm_90** | H100 / H200 / GH200（GH100） | ✅ `m8n8k4` + `m16n8k4/8/16` |
| Blackwell 数据中心 | **sm_100 / sm_103 / sm_110** | B100 / B200 / B300（GB100/200/300） | ✅ `m8n8k4` + `m16n8k4/8/16` |
| Blackwell 消费/工作站 | **sm_120 / sm_121** | RTX 5090 / 5080 / 5070（GB202/203） | ❌ |

## 为什么容易误判（关键陷阱）

PTX ISA 9.3 的 Target ISA Notes 原文是这样写的：

> `.f64 floating point type mma operation with .m8n8k4 shape requires sm_80 or higher.`
> `.f64 floating point type mma operation with .m16n8k4, .m16n8k8, and .m16n8k16 shapes require sm_90 or higher.`

这里的 **"sm_80 or higher" / "sm_90 or higher" 是「最低编译目标」的表述，不是「所有更高 CC 都有这块硬件」**。

- 数值上 sm_86、sm_89、sm_120 都 ≥ sm_80，所以 **ptxas 编译期不会报错**，你之前代码里的 `#if __CUDA_ARCH__ >= 800` 也会通过。
- 但**运行时硬件里根本没有 FP64 Tensor Core 数据通路**，于是落到消费卡上就变成非法指令（illegal instruction / trap）。

这正是你上一个问题里「为什么报非法指令」的更深层原因之一：即使 shape 写对（`m8n8k4`），只要跑在 sm_86/sm_89/sm_120 上，也一样会炸。

## 为什么消费卡没有 FP64 Tensor Core

这是 NVIDIA 一贯的**产品线切割**，跟 FP32/FP16 不同：

- **FP64 Tensor Core 是数据中心专属硅特性**，只为 HPC/科学计算市场保留：
  - Ampere：只有 GA100（A100）有；GA102（RTX 3090）没有。
  - Hopper/Ada 一代：只有 GH100（H100）有；AD102（RTX 4090）没有。
  - Blackwell 一代：只有 GB100/GB200/GB300（B200/B300）有；GB202（RTX 5090）没有。
- 消费级 Tensor Core 的精度路线是 **FP4/FP6/FP8/INT8/BF16/FP16/TF32**（低精度 AI），从不包含 FP64。
- 消费卡上 FP64 只能走 **CUDA Core 的 `FFMA`**，且被限制在 **1/64 吞吐**（例如 RTX 5090 每 SM 只有 2 个原生 FP64 单元，~1.6 TFLOPS）。

## 一个补充事实（Blackwell 特别）

- **Blackwell 数据中心**（sm_100+）：FP64 GEMM 走的是**老一代的 `mma.sync`/`DMMA` 路径**；新一代 `tcgen05.mma` 的 `.kind` 里**同样没有 f64**。
- **Blackwell 消费级**（sm_120/121）：连 `tcgen05`、TMEM 都没有，只剩 warp 级 `mma.sync`，且只有低精度格式。详情见 [Blackwell GPU Wiki: SM100 vs SM120](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/)。

## 正确写法建议

不要只用 `#if __CUDA_ARCH__ >= 800` 这种粗粒度判断，应**运行时检测真实 CC**：

```cuda
int major = 0, minor = 0;
cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
int cc = major * 10 + minor;

// 真正有 FP64 Tensor Core 的 CC 白名单
bool has_fp64_tc = (cc == 80 || cc == 90 ||
                    cc == 100 || cc == 103 || cc == 110);
if (!has_fp64_tc) {
    // 回退到 FP64 CUDA-core GEMM，或 FP32/TF32 tensor core 模拟（Ozaki 方案）
}
```

**一句话总结**：FP64 MMA 只在 `sm_80 / sm_90 / sm_100(103/110)` 这些数据中心芯片上存在；`sm_86/87、sm_89、sm_120/121` 全都不支持，别被 PTX 文档里 "sm_80 or higher" 的宽松措辞骗了。