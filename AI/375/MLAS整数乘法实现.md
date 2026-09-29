# MLAS整数乘法实现

## 🧑‍💻 User

onnx cpu用的那个MLAS有没有整数矩阵乘法相关实现？

---
## 🤖 Assistant

I'll check what tools are available that might help research this question.

The tool listing timed out. Let me retry.

I have browser/fetch tools. Let me research ONNX Runtime's MLAS source directly from GitHub.

I found the MLAS source tree. Let me examine the integer GEMM implementations directly.

## 结论

**有，而且很完整。** ONNX Runtime CPU EP 用的 MLAS 里，整数矩阵乘法就是 **QGEMM（Quantized GEMM）** 这一整套实现——支持 `uint8 × uint8`、`uint8 × int8`、`int8 × uint8`、`int8 × int8` 四种组合，累加器为 `int32`，并带有 zero-point 修正。

---

## 核心实现位置

都在 `onnxruntime/core/mlas/lib/` 下：

| 文件 | 作用 |
|------|------|
| `qgemm.cpp` | QGEMM 的主入口/调度（`MlasGemmBatch`、线程切分、`MlasGemmPackB`） |
| `qgemm.h` | `MLAS_GEMM_QUANT_*` 结构体与 dispatch 声明 |
| `qgemm_kernel_default.cpp` | **纯标量/C++ 通用整数内核（无 SIMD 平台的兜底）** |
| `qgemm_kernel_sse.cpp` / `_sse41.cpp` / `_avx2.cpp` / `_avx512vnni.cpp` | x86 各代 SIMD 整数内核 |
| `qgemm_kernel_amx.cpp` | Intel AMX（int8） |
| `qgemm_kernel_neon.cpp` / `_sdot.cpp` / `_smmla.cpp` / `_udot.cpp` / `_ummla.cpp` | ARM NEON / SDOT / I8MM (SMMLA/UMMLA) |
| `qgemm_kernel_lsx.cpp` / `_wasmsimd.cpp` / `_wasmrelaxedsimd.cpp` | LoongArch / Wasm SIMD |

<details>
<summary><b>默认内核的实际代码逻辑（点开看）</b></summary>

`qgemm_kernel_default.cpp` 里的内核签名和累加逻辑如下：

```cpp
struct MLAS_GEMM_QUANT_KERNEL_DEFAULT {
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;
    static constexpr size_t PackedK = 4;
    ...
};

template<>
size_t MlasGemmQuantKernel(
    const ...PackedAType* A, const ...PackedBType* B, int32_t* C,
    size_t PackedCountK, size_t CountM, size_t CountN, size_t ldc,
    const int32_t* RowSumBuffer, const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB, bool ZeroMode)
{
    while (CountN-- > 0) {
        int32_t Accumulator = *RowSumBuffer;      // A 的行和
        if (ZeroPointB != nullptr)
            Accumulator *= *ZeroPointB++;         // A行和 × B的zero-point
        Accumulator += *ColumnSumBuffer++;        // + B 的列和
        const auto* a = A;
        for (size_t k = 0; k < PackedCountK; k++) {
            Accumulator += a[0] * B[0];           // 整数点积累加
            Accumulator += a[1] * B[1];
            Accumulator += a[2] * B[2];
            Accumulator += a[3] * B[3];
            a += 4; B += 4;
        }
        if (!ZeroMode) Accumulator += C[0];
        C[0] = Accumulator;                       // int32 输出
        C += 1;
    }
    return 1;
}
```

可以看到：
- 输入被 bit-flip（`^ 0x80`）成无符号参与乘法，用 `RowSumBuffer`/`ColumnSumBuffer` 做 zero-point 校正；
- 累加结果写回 `int32_t* C`，**量化中的反量化（scale）是在上层再做的**。
</details>

<details>
<summary><b>调度入口：按 signed/unsigned 组合选内核</b></summary>

`qgemm.cpp` 中：

```cpp
const auto* GemmQuantDispatch = MlasGemmQuantGetDispatch(Shape->AIsSigned, Shape->BIsSigned);
```

即根据 A/B 是否为有符号，选取对应组合的 dispatch，再调用 `Operation` 或 `PackedOperation`。上层对外的入口是 `MlasGemmBatch(...)`（以及 `MlasGemmPackBSize`/`MlasGemmPackB` 用于预打包 B）。
</details>

---

## 相关的其他"整数/低位"矩阵乘法

MLAS 里除了标准 QGEMM，还有几类整数矩阵乘：

- **`sbgemm.h` / `sbgemm_kernel_neon.cpp`** — Signed Byte GEMM（ARM64 上的 s8s8 专用路径）。
- **`qlutgemm.cpp`** — 基于查表（LUT）的量化 GEMM，用于低比特/4bit 权重的快速路径。
- **`qnbitgemm.cpp`** + `sqnbitgemm_kernel_*` — n-bit（2/3/4/8bit）量化权重矩阵乘（AVX2/AVX512/NEON/LASX 等）。
- **`q4gemm.cpp` / `q4gemm_avx512.cpp`** — 4bit 权重的 GEMM。
- **`qdwconv.cpp` / `convsym.cpp`** — 量化卷积里的对称/整数卷积（复用同一套整数内积概念）。

---

## 一个重要区别（避免误解）

⚠️ MLAS 的整数矩阵乘法是**为量化推理路径（QGEMM）服务的**：

- 数据类型限定在 **u8 / s8**，累加用 **int32**；
- 语义上依赖 **zero-point + 列和/行和修正**，**不是**通用的 `int32 × int32 → int32` 稠密矩阵乘（像 NumPy 的整数 matmul / ONNX 的 INT32 `MatMul` 那种）。

所以：如果你要的是**"量化 int8 推理用的整数 GEMM"**——MLAS 有，而且做到各平台的 SIMD/AMX/I8MM 内核（上面已确认，`qgemm.cpp` 中明确写着 *"implements the quantized integer matrix/matrix multiply operation (QGEMM)"*）。
如果你要的是**通用 int32 稠密矩阵乘**——MLAS 主线并不提供，ORT 对这种情况通常走别的实现（如 Eigen/逐点或转换到浮点），而不会命中 MLAS 的整数内核。

---

## 一句话总结

MLAS 的整数矩阵乘法 = **QGEMM**（`qgemm.cpp` + 各平台 `qgemm_kernel_*.cpp`），覆盖 u8/s8 全组合、int32 累加，且从标量兜底到 SSE/AVX2/AVX512-VNNI/AMX、NEON/SDOT/I8MM 都有实现；它是量化推理专用的，不是通用 int32 GEMM。
