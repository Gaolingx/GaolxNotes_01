# 用int8gemm模拟fp64matmul

### 🧑‍💻 User

如何用多个 int8 gemm 再手动进位模拟fp64 matmul解决消费级gpu无fp64tc，给出triton实现，注意kernel launch开销和fused优化

---

### 🤖 Assistant

在消费级 GPU（如 RTX 3090/4090）上，由于 NVIDIA 人为限制了 FP64 Tensor Core 的吞吐量（通常只有 FP32 的 1/64），处理双精度矩阵乘法时性能极差。

为了绕过这一硬件限制，一种极客优化方案是：**将高精度数据（FP64 的尾数）拆分成多个低精度（INT8）的块（Limbs），利用算力极高的 INT8 Tensor Core 进行交叉乘加运算（Cross-Multiplication），最后在 SRAM 中手动进位并组合成高精度结果。**

下面为您详细解析这种方案的数学原理，并给出一个高度融合（Fused）以消除 Kernel Launch 开销的 Triton 实现。

---

### 一、 数学原理：多精度 Limb 拆分与手动进位

FP64 遵循 IEEE 754 标准，包含 1 位符号、11 位指数和 52 位尾数（加上隐式 1，共 53 位）。矩阵乘法 $C = A \times B$ 的核心计算压力在尾数相乘。

为了使用 INT8 Tensor Core，我们需要把 53 位整数（尾数）拆成多个 INT8 块。为了防止 INT8 点积时发生符号溢出，我们通常每次只取 **7 bit** 作为无符号数参与运算。

设 $A$ 和 $B$ 的尾数分别为 $X$ 和 $Y$，按 7-bit 拆分：
$$ X = \sum_{i=0}^{7} x_i \cdot 2^{7i} $$
$$ Y = \sum_{j=0}^{7} y_j \cdot 2^{7j} $$

它们的乘积可以展开为 64 个交叉项：
$$ X \times Y = \sum_{i=0}^{7} \sum_{j=0}^{7} (x_i \cdot y_j) \cdot 2^{7(i+j)} $$

这里的 $x_i \cdot y_j$ 就可以使用 `tl.dot(..., out_dtype=tl.int32)` 利用 INT8 TC 计算。

---

### 二、 核心优化思路（解决开销与 Fusion）

<details>
<summary><b>展开查看：如何解决 Kernel Launch 开销与寄存器压力</b></summary>

1. **彻底消除 Kernel Launch 开销（Full Fusion）：**
   传统的做法可能会在 PyTorch 层面对 Tensor 进行切片，然后调用多次 `torch.matmul`，这会产生数十次 Kernel Launch 和极高的显存读写（Memory Bound）。
   **Triton 优化：** 将“加载 FP64 -> 拆分 8 个 INT8 Limb -> 64 次 INT8 Dot -> 移位与手动进位 -> 组合 FP64” **全部写在一个 Triton Kernel 中**。数据只在首尾进出 HBM 一次，中间全在 SRAM/Registers 中完成。

2. **寄存器压力（Register Pressure）控制：**
   8 个 Limb 产生 64 个累加器，如果每个 block 都在寄存器里维护 64 个 FP32/INT32 累加器，会导致严重的寄存器溢出（Register Spilling）。
   **Triton 优化：** 不去维护 64 个独立的累加器，而是按**权重层级（$i+j$ 的值）**进行合并累加。$i+j$ 的范围是 0 到 14，我们只需要在循环内部维护 15 个累加器即可。

3. **指数对齐（Exponent Alignment）：**
   真正的浮点乘加需要对齐指数。在极度压榨性能的场景下，通常采用 **Block Floating Point (BFP)** 或 **Ozaki Scheme**：在 Block 级别找到最大指数，将所有尾数对齐转换成定点整数，做完 INT8 矩阵乘后再转回浮点数。下面的代码以定点/整数尾数相乘为核心逻辑进行展示。
</details>

---

### 三、 Triton Kernel 实现

为了代码可读性，以下 Triton 代码展示了 **3个Limb (模拟21-bit精度)** 的完整交叉乘加和手动进位逻辑。您可以根据此模板通过代码生成（Metaprogramming）扩展到 8个Limb (53-bit FP64尾数)。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def int8_sim_high_prec_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 计算当前 block 的位置
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 维护按照位移权重 (i+j) 合并的累加器
    # 3x3 交叉相乘，最大权重为 2+2=4，因此需要 5 个累加器
    acc_W0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_W1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_W2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_W3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc_W4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 1. 加载数据 (假设输入已经是缩放并转换为 int64 的尾数)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)

        # 2. Fused Limb 拆分 (使用 7-bit 掩码防止符号位干扰 INT8 TC)
        # 实际 FP64 需要拆分为 8 个 Limb，这里以 3 个 Limb 为例
        mask = 0x7F # 127
      
        a0 = (a >> 0) & mask
        a1 = (a >> 7) & mask
        a2 = (a >> 14) & mask
      
        b0 = (b >> 0) & mask
        b1 = (b >> 7) & mask
        b2 = (b >> 14) & mask

        # 转换为 int8 类型，满足 Tensor Core 的数据类型要求
        a0_i8 = a0.to(tl.int8)
        a1_i8 = a1.to(tl.int8)
        a2_i8 = a2.to(tl.int8)
      
        b0_i8 = b0.to(tl.int8)
        b1_i8 = b1.to(tl.int8)
        b2_i8 = b2.to(tl.int8)

        # 3. 交叉点积 (利用 INT8 Tensor Core，累加到 int32)
        # 权重 W0: i+j = 0
        acc_W0 += tl.dot(a0_i8, b0_i8, out_dtype=tl.int32)
      
        # 权重 W1: i+j = 1
        acc_W1 += tl.dot(a0_i8, b1_i8, out_dtype=tl.int32)
        acc_W1 += tl.dot(a1_i8, b0_i8, out_dtype=tl.int32)
      
        # 权重 W2: i+j = 2
        acc_W2 += tl.dot(a0_i8, b2_i8, out_dtype=tl.int32)
        acc_W2 += tl.dot(a1_i8, b1_i8, out_dtype=tl.int32)
        acc_W2 += tl.dot(a2_i8, b0_i8, out_dtype=tl.int32)
      
        # 权重 W3: i+j = 3
        acc_W3 += tl.dot(a1_i8, b2_i8, out_dtype=tl.int32)
        acc_W3 += tl.dot(a2_i8, b1_i8, out_dtype=tl.int32)
      
        # 权重 W4: i+j = 4
        acc_W4 += tl.dot(a2_i8, b2_i8, out_dtype=tl.int32)

        # 移动指针
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. SRAM 内手动进位与组合 (Fused Carry-over)
    # 将累加器提升为 float64 以防止最终移位时 int32 溢出
    c_fp64 = acc_W0.to(tl.float64)
    c_fp64 += acc_W1.to(tl.float64) * (1 << 7)
    c_fp64 += acc_W2.to(tl.float64) * (1 << 14)
    c_fp64 += acc_W3.to(tl.float64) * (1 << 21)
    c_fp64 += acc_W4.to(tl.float64) * (1 << 28)

    # 5. 写入 HBM
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  
    tl.store(c_ptrs, c_fp64, mask=mask_c)
```

### 四、 代码原理解析与扩展到 53-bit (FP64)

1. **为什么用 `0x7F` (7 bits)？**
   INT8 类型的取值范围是 $[-128, 127]$。如果我们取 8 bits 拆分，最高位会被解析为符号位，导致 `tl.dot` 产生负数计算错误。取 7 bits 能确保所有 Limb 都是正整数。
2. **如何扩展到 FP64 (53 bits)？**
   * 53 bits 尾数需要 $\lceil 53 / 7 \rceil = 8$ 个 Limb。
   * 您需要生成 $a_0$ 到 $a_7$ 和 $b_0$ 到 $b_7$。
   * 交叉相乘会产生 $8 \times 8 = 64$ 次 `tl.dot`。
   * $i+j$ 的权重范围是从 $W0$ 到 $W14$，需要在循环外定义 15 个 `acc_W` 累加器。
   * 手动进位部分改为：`c_fp64 += acc_Wi.to(tl.float64) * (1 << (7 * i))`
3. **性能调优提示 (`num_stages` 和 `num_warps`)：**
   因为 8-Limb 扩展后寄存器占用极高，建议在调用 `int8_sim_high_prec_matmul_kernel[grid](...)` 时，严格限制 `BLOCK_M` 和 `BLOCK_N`（例如 `32x32` 或 `64x64`），并将 `num_warps` 设为 8，`num_stages` 设为 2 或 3，防止因寄存器溢出导致的性能崩塌。