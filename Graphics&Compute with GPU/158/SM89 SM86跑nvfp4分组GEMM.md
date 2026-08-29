# SM89 SM86跑nvfp4分组GEMM

### 🧑‍💻 User

sm89和sm86 arch的gpu要怎么运行nvfp4的cutedsl grouped gemm？

---

### 🤖 Assistant

现在我有足够的信息来给出一个全面的答案了。

---

## NVFP4 CUTEDSL Grouped GEMM 在 SM89 / SM86 上的运行方案

### 一、硬件现实：先搞清楚什么能做什么不能做

| 架构 | SM 版本 | 代表 GPU | 原生 NVFP4 | 原生 FP8 | 可用 Tensor Core 类型 |
|------|---------|-----------|-----------|---------|----------------------|
| Blackwell | SM100/SM120 | B200 / RTX 5090 | ✅ 原生 | ✅ | FP4, FP8, FP16, BF16 |
| **Ada Lovelace** | **SM89** | RTX 4090 | ❌ | ✅ (E4M3/E5M2) | FP8, FP16, BF16, TF32 |
| **Ampere** | **SM86** | RTX 3090 / A100 | ❌ | ❌ | FP16, BF16, TF32 |

**核心结论：SM89 和 SM86 都没有原生 NVFP4 硬件指令。** 需要走 **软件反量化 (dequantization)** 路线——在寄存器/共享内存中把 FP4 展开为 FP16/BF16，再调用各自架构上可用的 Tensor Core。

---

### 二、CUTLASS 中相关的参考示例

在最新的 CUTLASS main 分支中，直接相关的示例：

```
# 原生 NVFP4（仅 Blackwell，不能直接在 SM89/SM86 跑）
examples/72_blackwell_narrow_precision_gemm/   ← 72a: NVFP4×BF16, 72b: NVFP4×NVFP4
examples/75_blackwell_grouped_gemm/             ← Blackwell Grouped GEMM
examples/89_sm103_fp4_ultra_gemm/               ← SM103 FP4 Ultra
examples/90_sm103_fp4_ultra_grouped_gemm/       ← SM103 FP4 Ultra Grouped

# Mixed Dtype / Narrow Precision 模式（可参考改造用于 SM89/SM86）
examples/55_hopper_mixed_dtype_gemm/            ← Hopper int4→bf16/fp8 (关键参考！)
examples/58_ada_fp8_gemm/                       ← Ada FP8 (SM89)
examples/64_ada_fp8_gemm_grouped/               ← Ada FP8 Grouped GEMM (SM89 的直接模板)
examples/69_hopper_mixed_dtype_grouped_gemm/    ← Hopper Mixed Dtype Grouped
examples/24_gemm_grouped/                       ← 通用 Grouped GEMM (SM86 可参考)
examples/86_blackwell_mixed_dtype_gemm/         ← Blackwell mixed dtype
```

---

### 三、SM89 (Ada / RTX 4090) 上的方案

SM89 有 **FP8 Tensor Core**（`cutlass::float_e4m3_t`），但没有 FP4。最佳路径是：

**FP4 权重 → 反量化到 FP16/BF16 → 使用 FP16 Tensor Core 做 MMA**

参考 `55_hopper_mixed_dtype_gemm` 的 README 中的设计思路（虽然在 Hopper 示例里，但 CUTLASS 3.x 的 mixed dtype collective 也适用于 SM89）：

> "always passes the type with fewer bits through the register file and upcasts to the type with the higher bit count"

对于 **Grouped GEMM**，可以直接参考 `64_ada_fp8_gemm_grouped` 的代码结构，核心类型定义为：

```cpp
// 64_ada_fp8_gemm_grouped 的关键模板参数
using ElementA = cutlass::float_e4m3_t;     // ← 替换为你的 FP4 packed 类型
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGroupedPerGroupScale<
    ElementA, LayoutA, ...,
    ElementB, LayoutB, ...,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,                     // ← Ada
    ...
>::GemmKernel;
```

改造为 FP4 的关键步骤：

1. **存储层**：FP4 打包存储（2 个 FP4 值占 1 字节），定义自定义 `cutlass::Array<float_e4m3_t, 2>` 或直接用 `uint8_t` 配合自定义 converter
2. **反量化**：在 mainloop 的 TMA load 或 cp.async load 阶段将 FP4→BF16/FP16，或借助 CUTLASS 3.x 的 `MixedInputMainloop` 自动处理
3. **GroupScheduleMode**：使用 `kDeviceOnly` 或 `kHostPrecompute`

实际编译运行：
```bash
# 参考 Ada FP8 Grouped GEMM 的构建方式
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=89
make 64_ada_fp8_gemm_grouped
./examples/64_ada_fp8_gemm_grouped/64_ada_fp8_gemm_grouped --groups=100
```

---

### 四、SM86 (Ampere / A100, RTX 3090) 上的方案

SM86 **连 FP8 都没有**，只能用 FP16/BF16 Tensor Core。因此：

**FP4 权重 → 反量化到 FP16/BF16 → 使用 FP16/BF16 Tensor Core**

SM86 没有 TMA（Tensor Memory Accelerator），但有 `cp.async`。参考模式：

1. **基础 Grouped GEMM 模板**：`examples/24_gemm_grouped`
2. **混合精度思路**：参考 Hopper mixed dtype（55）中的 converter 设计，但要降级：
   - 取消对 TMA 的依赖
   - 用 `cp.async` 加载 FP4 数据
   - 在寄存器中进行反量化操作（或使用 ldmatrix + 转换）

CUTLASS 2.x API 方式（更适合 SM86）：

```cpp
// 类似于 examples/24_gemm_grouped 的结构
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,     // 自定义 FP4 类型（或 uint8_t）
    LayoutA,
    ...
    ElementB,     // 自定义 FP4 类型（或 uint8_t）
    LayoutB,
    ...
    ElementOutput,     // cutlass::half_t 或 cutlass::bfloat16_t
    LayoutC,
    ElementAccumulator, // float
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,  // ← Ampere
    ...
>::GemmKernel;
```

对于 SM86 的 Grouped GEMM，CUTLASS 3.x 的 **Mixed Dtype Grouped**（`69_hopper_mixed_dtype_grouped_gemm`）也是很好的参考，只需要把 `Sm90` 替换为 `Sm80`，并把 TMA 替换为 `cp.async`。

---

### 五、CuTe / CUTEDSL 层面的操作

如果用 CuTe（CUTLASS 的底层 DSL）直接编写 kernel：

1. **FP4 packing**：用 `cute::uint4_t` 或自定义 `FP4x2` 类型
2. **Layout**：FP4 窄类型必须是 **K-major**：
   - A 矩阵 FP4 → RowMajor
   - B 矩阵 FP4 → ColumnMajor
3. **反量化 atom**：自定义一个 `Copy_Atom`，在数据从 GMEM→SMEM→RF 的过程中将 FP4 展开为对应的高精度类型
4. **MMA atom**：使用该架构支持的 MMA 指令（SM89 用 FP16 MMA，SM86 用 FP16/BF16 MMA）

```cpp
// CuTe 层面的核心思路
auto tiled_copy_fp4_to_bf16 = make_tiled_copy(
    Copy_Atom<FP4ToBF16Converter, Element>,  // 自定义转换 atom
    layout_for_fp4,
    layout_for_threads
);
```

---

### 六、总结对照表

| 方面 | SM89 (Ada) | SM86 (Ampere) |
|------|-----------|---------------|
| 原生 FP4 支持 | ❌ | ❌ |
| 最高效的 MMA 类型 | FP8 / FP16 / BF16 | FP16 / BF16 |
| 推荐反量化目标 | BF16（便于与 activation 对齐） | BF16 或 FP16 |
| 推荐 CUTLASS 示例 | `64_ada_fp8_gemm_grouped` + `55_hopper_mixed_dtype_gemm` | `24_gemm_grouped` + `55` 的 converter 思路 |
| 数据加载方式 | TMA（SM89 支持 TMA） | `cp.async` / `ldmatrix` |
| API 层 | CUTLASS 3.x | CUTLASS 2.x（或 3.x 降级） |
| CMAKE ARCH | `-DCUTLASS_NVCC_ARCHS=89` | `-DCUTLASS_NVCC_ARCHS=86` |

**关键要点**：在 SM89/SM86 上跑 NVFP4 的本质是 **"FP4 存储 + 运行时反量化 + 高精度 MMA"**，性能取决于反量化开销与内存带宽节省之间的权衡。CUTLASS 的 mixed dtype infrastructure（`55_hopper_mixed_dtype_gemm`）已经为此类场景提供了很好的框架，只是需要针对非 Blackwell 架构做适配（替换 MMA 指令和加载路径）。