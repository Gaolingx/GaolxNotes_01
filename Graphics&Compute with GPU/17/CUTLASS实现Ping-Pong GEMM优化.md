# CUTLASS实现Ping-Pong GEMM优化

### 🧑‍💻 User

如何用cutlass实现pingpong gemm，掩盖访存延迟？

---

### 🤖 Assistant

在 CUTLASS 中实现 **Ping-Pong GEMM**（双缓冲）的核心思路是**重叠计算与数据加载**，通过预取下一块数据（Prefetching）来掩盖全局内存访问延迟。以下是关键实现步骤和代码示例：

---

### 1. 核心概念：双缓冲流水线
- **Ping 阶段**：加载数据块 `n+1` 到共享内存
- **Pong 阶段**：同时执行数据块 `n` 的 GEMM 计算
- **交替执行**：每个迭代步交换角色，实现计算与访存重叠

---

### 2. 实现步骤（使用 CUTLASS 模板）

#### (1) 选择支持双缓冲的 GEMM 内核
```cpp
#include <cutlass/gemm/device/gemm.h>

// 使用 2 级流水线（双缓冲）
using GemmOperator = cutlass::gemm::device::Gemm<
    float,                                   // A 数据类型
    cutlass::layout::RowMajor,               // A 布局
    float,                                   // B 数据类型
    cutlass::layout::ColumnMajor,            // B 布局（优化访存）
    float,                                   // C 数据类型
    cutlass::layout::RowMajor,
    float,                                   // 累加器类型
    cutlass::arch::OpClassSimt,              // SIMT 核心
    cutlass::arch::Sm80,                     // Ampere GPU
    cutlass::gemm::GemmShape<128, 128, 32>,  // 线程块分块
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp 分块
    cutlass::gemm::GemmShape<1, 1, 1>,       // 指令分块
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>, // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,         // Swizzling
    2                                        // kStages = 2 (双缓冲关键!)
>;
```

#### (2) 内核调用（显式设置双缓冲）
```cpp
GemmOperator gemm_op;
GemmOperator::Arguments args(
    {M, N, K},                      // 问题尺寸
    {d_A, K},                       // A 矩阵 (行主序)
    {d_B, N},                       // B 矩阵 (列主序优化)
    {d_C, N},                       // C 矩阵
    {d_D, N},                       // D 矩阵 (输出)
    {alpha, beta}                   // 标量参数
);

cutlass::Status status = gemm_op(args);  // 启动双缓冲 GEMM
```

---

### 3. 手动优化技巧（如需进一步控制）

#### (1) 共享内存双缓冲声明
```cpp
__shared__ float smem_A[2][TileSize][TileSize];  // Ping-Pong 缓冲区 A
__shared__ float smem_B[2][TileSize][TileSize];  // Ping-Pong 缓冲区 B
```

#### (2) 数据加载与计算重叠
```cpp
for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    int buffer_idx = tile_idx % 2;  // 当前缓冲索引 (0/1)
    int next_buffer_idx = (tile_idx + 1) % 2;

    // 阶段1: 异步加载下一块数据到共享内存
    __syncthreads();
    load_tile_to_shared(
        smem_A[next_buffer_idx], 
        smem_B[next_buffer_idx], 
        tile_idx + 1
    );

    // 阶段2: 计算当前块 (使用 buffer_idx)
    compute_gemm_tile(
        smem_A[buffer_idx], 
        smem_B[buffer_idx]
    );

    // 阶段3: 等待下一块数据加载完成
    __syncthreads();
}
```

---

### 4. 关键优化点
1. **`kStages=2`**：  
   在 CUTLASS 模板中显式设置流水线深度为 2（双缓冲）。

2. **内存布局优化**：  
   - 对 `A` 使用行主序（RowMajor）
   - 对 `B` 使用列主序（ColumnMajor）提升合并访问

3. **共享内存 Bank 冲突**：  
   通过调整 Tile 大小（如 128x128）避免 Bank 冲突。

4. **异步拷贝（Async Copy）**：  
   在 Ampere+ GPU 上使用 `cp.async` 指令：
   ```cpp
   __pipeline_memcpy_async(
       smem_ptr, 
       global_ptr, 
       sizeof(float) * copy_size
   );
   ```

5. **流水线同步**：  
   用 `__pipeline_commit()` 和 `__pipeline_wait_prior(n)` 控制依赖。

---

### 5. 性能验证
- **Nsight Compute 分析**：检查 SM 利用率是否 >90%
- **内存延迟隐藏**：确保 `stall_memory_dependency` 指标显著下降
- **吞吐量**：对比单缓冲版本，预期获得 10-30% 加速

> **注意**：现代 CUTLASS (2.x+) 已自动优化流水线，优先使用模板参数 `kStages=2` 而非手动实现，除非有特殊硬件定制需求。

通过结合 CUTLASS 的高效模板和硬件特性（如异步拷贝），Ping-Pong 机制能有效提升 GEMM 性能至接近理论峰值。