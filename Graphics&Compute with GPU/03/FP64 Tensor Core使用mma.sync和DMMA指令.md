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

在ampere/hopper等架构上，合法的fp64 mma指令有哪些？

---

### 🤖 Assistant

**直接回答：在 Ampere（`sm_80`）和 Hopper（`sm_90`）上，FP64 的 MMA 指令只有一种 shape——`m8n8k4`，加上 A/B 两种 layout 的组合，一共 4 条。**

### 1. 合法的 PTX 指令（全部 4 条）

| PTX 指令 | A 布局 | B 布局 |
|---|---|---|
| `mma.sync.aligned.m8n8k4.row.row.f64.f64.f64.f64` | row | row |
| `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64` | row | col |
| `mma.sync.aligned.m8n8k4.col.row.f64.f64.f64.f64` | col | row |
| `mma.sync.aligned.m8n8k4.col.col.f64.f64.f64.f64` | col | col |

- **`m8n8k4` 是唯一合法的 shape**（M=8, N=8, K=4）。没有 `m16n8k8.f64`、`m8n8k4` 之外的任何 FP64 变体。
- 类型全部是 `.f64.f64.f64.f64`（A、B、C、D 都是 double）。
- 只有 `.aligned` 一种形式，没有 FP64 的 `.and` 或其他特殊变体。
- 实践中最常用的是 `.row.col`（对应 GEMM 里 A 行主序 × B 列主序的常见约定）。

### 2. 操作数数量（这里要纠正我上一次的错误）

我上一条回复说 “A 需要 2 个 double、B 需要 1 个” 是**错的**。按 $8\times8\times4$ 的矩阵大小算：

- **A**：$8\times4 = 32$ 个元素 → 每个线程 **1 个 `f64`**
- **B**：$4\times8 = 32$ 个元素 → 每个线程 **1 个 `f64`**
- **C / D**：$8\times8 = 64$ 个元素 → 每个线程 **2 个 `f64`**

也就是说，**你原始代码里的操作数排列（A=1、B=1、C=2、D=2）本来就是对的**，唯一的问题只是把 shape 写成了 `m16n8k32`（那是 INT8 的形状）。把 shape 改成 `m8n8k4` 后，操作数不用改。

### 3. 修正后的完整代码

```cuda
#if __CUDA_ARCH__ >= 800
__device__ void mma_f64f64f64_8_8_4_(double *data)
{
    double d0 = 0.0;
    double d1 = 0.0;
    double a  = 0.0;
    double b  = 0.0;

    for (unsigned k = 0; k < N_LOOP_INTERNAL; k++)
    {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%0, %1};\n"
            : "+d"(d0), "+d"(d1)      // D 和 C 都用 {d0, d1}，原地累加
            : "d"(a), "d"(b));        // A = {a}, B = {b}
        // __syncwarp();  // mma.sync 本身是 warp 同步指令，一般不需要
    }

    double *ptr = &data[threadIdx.y * 8 * 8];
    ptr[threadIdx.x * 2 + 0] = d0;
    ptr[threadIdx.x * 2 + 1] = d1;
}
#endif
```

几个关键点：

- **用 `"+d"(d0), "+d"(d1)` 而不是分别写 `"=d"` 和 `"d"`**：你原来的写法里 `d0/d1` 同时出现在输出和输入约束里，编译器**不保证**它们映射到同一个物理寄存器，可能导致累加结果不对。`+d` 保证读写同一个寄存器（即 D=C，原地累加）。
- **`{%0,%1}` 同时作为 D 和 C**：这样就是标准 accumulate 语义。
- `__syncwarp()` 可以去掉，`mma.sync` 本身是 warp 级同步指令。

### 4. Hopper 的补充说明

- Hopper（`sm_90`）引入的 `wgmma`（warp-group MMA）**不支持 FP64**（只支持 f16/bf16/tf32/fp8/int8/b1）。
- 所以即使在 H100 上，FP64 矩阵乘仍然走 **warp 级 `mma.sync.aligned.m8n8k4.f64...`**，shape 不变。
- SASS 层面对应的指令是 **`DMMA.884`**（Double MMA，8×8×4）。可以用 `cuobjdump -sass` 验证。

### 5. 仍要注意的点

| 事项 | 说明 |
|---|---|
| **架构要求** | `sm_80+`；真正有实用 FP64 Tensor Core 吞吐的是 A100/H100 这类数据中心卡，消费级 Ampere（sm_86）的 FP64 被大幅削减 |
| **fragment 布局** | A/B 的寄存器到矩阵元素的映射**必须**按 PTX ISA 文档的 fragment 表来填，`.row`/`.col` 不同则映射不同，否则结果错乱 |
| **编译选项** | 必须 `-arch=sm_80` 或更高，且运行在对应 GPU 上 |

**总结**：合法的 FP64 MMA 只有 `mma.sync.aligned.m8n8k4.{row.row|row.col|col.row|col.col}.f64.f64.f64.f64` 这 4 条，操作数是 **A=1、B=1、C=2、D=2** 个 `f64`。你原来报非法指令，就是因为 shape 写成了不存在的 `m16n8k32`。