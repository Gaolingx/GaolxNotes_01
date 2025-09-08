# Triton实现FP16矩阵乘法与FP32累加

### 🧑‍💻 User

如何用triton实现一个fp16 matmul,fp32 acc gemm kernel?

---

### 🤖 Assistant

# 使用 Triton 实现 FP16 矩阵乘法与 FP32 累加的 GEMM Kernel

下面我将展示如何使用 Triton 实现一个高效的 FP16 矩阵乘法 kernel，并在累加阶段使用 FP32 精度以防止精度损失。

## 实现思路

1. 使用 FP16 存储输入矩阵，减少内存带宽需求
2. 在计算过程中将 FP16 值转换为 FP32 进行累加，确保数值稳定性
3. 使用分块(tiling)策略优化内存访问模式
4. 利用共享内存(shared memory)减少全局内存访问

## 完整实现代码

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # 矩阵指针
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 步长信息
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 元参数
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 确定程序ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算内存偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 迭代K维度进行计算
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载A和B的块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 将FP16转换为FP32并进行矩阵乘法累加
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        accumulator += tl.dot(a, b)
        
        # 移动指针到下一个K块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 将FP32累加结果转换为FP16存储
    c = accumulator.to(tl.float16)
    
    # 计算C矩阵的写入位置
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 存储结果
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    # 检查输入
    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    assert a.dtype == torch.float16 and b.dtype == torch.float16, "输入必须是FP16"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 启动kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

# 测试函数
def test_matmul():
    # 创建随机FP16矩阵
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # 计算Triton结果
    c_triton = matmul(a, b)
    
    # 计算PyTorch结果作为参考
    c_ref = torch.matmul(a.float(), b.float()).half()
    
    # 比较结果
    print(f"最大误差: {torch.max(torch.abs(c_triton - c_ref))}")
    print(f"平均误差: {torch.mean(torch.abs(c_triton - c_ref))}")
    
    # 验证正确性
    assert torch.allclose(c_triton, c_ref, atol=1e-2, rtol=0), "结果不匹配"

if __name__ == "__main__":
    test_matmul()
```

## 关键实现细节

1. **数据类型处理**：
   - 输入矩阵使用FP16存储
   - 累加器使用FP32防止精度损失
   - 最终结果转换回FP16存储

2. **内存访问优化**：
   - 使用分块策略减少内存访问
   - 利用TL语言内置的`tl.dot`操作进行高效矩阵乘法

3. **自动调优**：
   - 使用`@triton.autotune`装饰器自动选择最佳配置
   - 提供多种块大小配置以适应不同矩阵尺寸

4. **边界处理**：
   - 使用掩码(mask)处理非整除的矩阵维度
   - 确保不会越界访问内存

这个实现提供了高效的FP16矩阵乘法，同时在累加阶段使用FP32精度保持数值稳定性，适合在支持Triton的GPU上运行。