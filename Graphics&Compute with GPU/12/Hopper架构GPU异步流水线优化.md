# Hopper架构GPU异步流水线优化

### 🧑‍💻 User

hopper架构的gpu，cuda中如何用wgmma+tma实现异步流水线，最大化吞吐，隐藏延迟？

---

### 🤖 Assistant

在 Hopper 架构 (H100+) 上使用 **WGMMA (Warp Group Matrix Multiply-Accumulate)** 指令配合 **TMA (Tensor Memory Accelerator)** 实现异步流水线是最大化 GEMM 吞吐量和隐藏内存/计算延迟的关键技术。其核心思想是将**数据搬运 (TMA)**、**计算 (WGMMA)** 和**结果写入/同步**三个阶段进行深度流水化重叠。

以下是实现策略和关键步骤：

## 核心概念与组件

1.  **WGMMA (Warp Group MMA):**
    *   Hopper 引入的革命性特性，允许一个 **Warp Group (4个相邻的 Warp, 128 个线程)** 协作执行一个大型矩阵乘法累加操作（例如 128x128x128）。
    *   **异步执行：** WGMMA 操作是**硬件异步**的。发出 WGMMA 指令后，线程可以立即继续执行后续指令，而计算在后台由张量核心进行。
    *   **依赖跟踪：** 使用 `wgmma.wait_group.sync.aligned` 指令显式等待 WGMMA 操作完成，确保累加结果可用。
    *   **寄存器文件消耗：** WGMMA 需要消耗大量的寄存器文件来保存输入矩阵片段和累加器结果。规划寄存器使用至关重要。
2.  **TMA (Tensor Memory Accelerator):**
    *   专用的硬件单元，用于在 **Global Memory** 和 **Shared Memory** 之间高效地搬运**大块、多维的张量数据**。
    *   **异步与非阻塞：** TMA 拷贝操作 (`cp.async.bulk.tensor`) 也是**硬件异步**和非阻塞的。发出拷贝指令后，线程可以继续执行，数据搬运在后台进行。
    *   **描述符 (Descriptor)：** 操作前需要配置一个 TMA 描述符，定义源/目标地址、张量维度、步长、box 大小等元信息。
    *   **依赖跟踪：** 使用 `cp.async.bulk.group` 或 `cp.async.bulk.wait_group` 指令来同步 TMA 操作的完成。
    *   **Shared Memory 要求：** TMA 要求目标 Shared Memory 地址按特定规则对齐（通常是 128 字节）。
3.  **双缓冲/三缓冲 (Double/Triple Buffering):**
    *   在 Shared Memory 中为输入矩阵 `A` 和 `B` 分配**多个缓冲区**（通常是 2 个或 3 个）。
    *   当一个缓冲区正在被 WGMMA 计算消耗时，TMA 可以同时将下一块数据异步搬运到另一个空闲缓冲区。
    *   计算和搬运在空间上分离的不同缓冲区上同时进行，实现重叠。
4.  **CUDA Pipeline Primitives (cuda::pipeline):**
    *   CUDA 提供的高级抽象（头文件 ``），简化了基于 `__pipeline_commit()` 和 `__pipeline_wait_prior()` 等原语的异步操作依赖管理和同步。
    *   它帮助构建清晰的流水线阶段，自动管理操作之间的依赖关系（如 TMA 完成才能开始计算，计算完成才能开始下一轮搬运）。

## 实现异步流水线的步骤

以下是一个典型的使用 TMA + WGMMA + 双/三缓冲 + CUDA Pipeline 的核函数结构框架：

```cpp
#include <cuda/pipeline>

__global__ void hopperSgemmTmaWgmma(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                    int M, int N, int K) {
    // 0. 常量定义 (Tile sizes, WarpGroup 配置等)
    constexpr int kBlockTileM = 128; // ThreadBlock 在 M 维度计算的 Tile 大小
    constexpr int kBlockTileN = 128; // ThreadBlock 在 N 维度计算的 Tile 大小
    constexpr int kBlockTileK = 64;  // ThreadBlock 在 K 维度计算的 Tile 大小 (TMA 搬运的 K 块大小)
    constexpr int kWarpGroupSize = 4; // 一个 Warp Group 包含 4 个 Warp (128 threads)
    // ... 其他常量 ...

    // 1. 共享内存声明 - 双缓冲 (或三缓冲)
    //   为矩阵 A 和 B 各分配两个 Tile 大小的缓冲区 (+ 可能的填充以满足对齐)
    extern __shared__ float shmem[];
    float* smemA[2] = { shmem, shmem + 2 * kSmemBufferSizeA }; // 假设计算了每个缓冲区的正确大小和偏移
    float* smemB[2] = { ... };
    // 或三缓冲: float* smemA[3]; float* smemB[3];

    // 2. 初始化 TMA 描述符 (通常在 Thread 0 执行一次)
    //   创建描述符需要的信息：全局内存基址、维度、步长、Box大小(kBlockTileM x kBlockTileK 或 kBlockTileK x kBlockTileN)、数据类型等
    if (threadIdx.x == 0) {
        tma::initTensorMapDescA(&gTmaDescA, A, ...);
        tma::initTensorMapDescB(&gTmaDescB, B, ...);
    }
    __syncthreads(); // 确保描述符对所有线程可见

    // 3. CUDA Pipeline 对象创建 (通常每个 ThreadBlock 一个)
    constexpr int kPipelineStages = 2; // 双缓冲 = 2 个流水线阶段
    cuda::pipeline<cuda::thread_scope_block> pipeline = cuda::make_pipeline();

    // 4. 计算当前 ThreadBlock 负责的 C 矩阵 Tile 的全局坐标
    int blockTileM = ...; // 基于 blockIdx.x, blockDim.x 计算
    int blockTileN = ...; // 基于 blockIdx.y, blockDim.y 计算

    // 5. 初始化流水线：预取第0阶段的数据 (K维的第0个Tile)
    //   - 计算当前 K-Tile (k=0) 的 A 和 B 在 Global Mem 中的坐标
    int k_tile = 0;
    //   - 提交第0阶段的 TMA 拷贝 (从 Global 到 smemA[0], smemB[0])
    pipeline.producer_acquire();
    if (current_thread_handles_tma_for_A) {
        tma::load(smemA[0], gTmaDescA, coordA_k0);
    }
    if (current_thread_handles_tma_for_B) {
        tma::load(smemB[0], gTmaDescB, coordB_k0);
    }
    //   - 使用 memory fence 确保 TMA 提交被系统看到
    __pipeline_commit();
    pipeline.producer_commit();
    k_tile++; // 准备下一个 K-Tile

    // 6. 主循环 (遍历 K 维度的 Tiles)
    //    累加器 (通常放在寄存器中) 需要初始化为0 (在循环外或第一次迭代前)
    float accum[kRegsM][kRegsN] = {0.0f}; // 假设每个线程持有累加器的一个片段

    for (; k_tile < (K / kBlockTileK); ++k_tile) {
        // ** Stage 1: 提交下一阶段 (k_tile) 的 TMA 加载 (Double Buffering: 写入下一个缓冲区) **
        pipeline.producer_acquire();
        //   计算下一个 K-Tile (k_tile) 的 A 和 B 的 Global 坐标
        //   提交异步 TMA 加载到下一个缓冲区 (smemA[next_buffer], smemB[next_buffer])
        int next_smem_buffer = k_tile % 2; // 双缓冲切换
        if (current_thread_handles_tma_for_A) {
            tma::load(smemA[next_smem_buffer], gTmaDescA, coordA_kt);
        }
        if (current_thread_handles_tma_for_B) {
            tma::load(smemB[next_smem_buffer], gTmaDescB, coordB_kt);
        }
        __pipeline_commit();
        pipeline.producer_commit();

        // ** Stage 2: 等待当前阶段 (k_tile - 1) 的 TMA 加载完成 (确保当前计算用的数据在 SMEM 中) **
        pipeline.consumer_wait(); // 等待上一轮提交的 TMA (对应当前要计算的 k_tile-1 数据) 完成
        __pipeline_wait_prior(0); // 更细粒度的等待 (可选，pipeline通常管理)
        __syncthreads(); // **重要**：确保整个 ThreadBlock 看到完整的 SMEM 数据 (TMA 是 per-threadblock 操作)

        // ** Stage 3: 执行 WGMMA 计算 (使用当前缓冲区 smemA[current_buffer], smemB[current_buffer] 的数据) **
        int current_smem_buffer = (k_tile - 1) % 2; // 计算当前可用的缓冲区 (上一轮加载的)
        //   a) 将 SMEM 中的数据 (通过 LDMatrix) 加载到寄存器 (WGMMA 操作数的寄存器片段)
        //   b) 发出 WGMMA 指令 (异步!)
        asm volatile (
          "wgmma.mma_async.sync.aligned.m%d.n%d.k%d [%0], [%1], %2, %3;\n"
          : // Outputs (通常是累加器寄存器列表)
          : "r"(smemA[current_smem_buffer] + offsetA), "r"(smemB[current_smem_buffer] + offsetB),
            "r"(descA), "r"(descB) // desc 描述矩阵布局
          : "memory"
        );
        //   c) 发出后线程立即继续，无需等待计算完成

        // ** Stage 4: 为下一次迭代做准备 **
        pipeline.consumer_release(); // 释放当前已计算完数据的 SMEM 缓冲区 (使其可被下一轮 TMA 重用)
        //   (对于双缓冲，这个释放的缓冲区就是下一轮 TMA 将要加载的目标缓冲区)

        // ** 注意：WGMMA 的等待通常在 K 维内积循环结束后进行 **
    }

    // 7. 处理主循环结束后最后一块 K-Tile 的计算 (没有新的 TMA 提交了)
    pipeline.consumer_wait(); // 等待最后提交的 TMA 完成 (对应最后一块K)
    __syncthreads();
    //   执行最后一块 K 的 WGMMA (使用最后加载的缓冲区)
    asm volatile ( ... /* WGMMA for last K tile */ ... );

    // 8. 等待所有未完成的 WGMMA 操作完成 (K 维所有累加完成)
    //    必须在读取累加器结果之前进行
    asm volatile ("wgmma.wait_group.sync.aligned %0;\n" : : "n"(0) /* 通常等待组0 */);
    // 或者使用 `__wgmma_wait()` 内在函数

    // 9. 将最终累加器结果 (在寄存器中) 写回全局内存 C
    //    可能需要经过一次 Epilogue (激活函数、bias 加等)
    //    写回通常也使用向量化存储或 STG/TMA 存储优化
    writeOutput(C, accum, ...);
}
```

## 关键优化点与注意事项

1.  **缓冲策略选择:**
    *   **双缓冲 (Double Buffering):** 最常见，开销最小。需要至少 2 个 Tile 的 Shared Memory (A/B 各 2 个)。在计算 Tile `k` 时，加载 Tile `k+1`。计算和加载完美重叠。
    *   **三缓冲 (Triple Buffering):** 在双缓冲基础上增加一个缓冲区。主要优势在于可以**隐藏 TMA 延迟的波动**。如果某个 K-Tile 的 TMA 加载时间意外变长，三缓冲可以提供额外的缓冲时间，防止计算因等待数据而停顿。代价是更高的 Shared Memory 消耗。
2.  **Shared Memory 布局与 Bank Conflicts:**
    *   **WGMMA 要求：** WGMMA 通过 `ldmatrix` 指令从 Shared Memory 加载数据到寄存器。`ldmatrix` 对 Shared Memory 的访问模式有特定要求（通常是 32 字节或 128 字节访问），并且要**避免 Bank Conflicts**。
    *   **TMA 要求：** TMA 的目标地址需要按 128 字节对齐。
    *   **解决方案：** 仔细设计 `A` 和 `B` 矩阵在 Shared Memory 中的布局（例如使用 **Swizzled Layouts**），或者进行适当的**填充 (Padding)**，以满足对齐要求并最大限度地减少甚至消除 Bank Conflicts。这是性能关键点。
3.  **TMA 与 WGMMA 的协调:**
    *   **Box 大小匹配：** TMA 搬运的 "Box" (例如 `kBlockTileM x kBlockTileK` for A) 必须与 WGMMA 操作期望从 Shared Memory 中读取的数据块大小相匹配。
    *   **坐标计算：** 在循环中正确计算每个 K-Tile 对应的全局内存坐标 (`coordA_kt`, `coordB_kt`) 和 Shared Memory 偏移 (`offsetA`, `offsetB`)。
4.  **依赖与同步管理:**
    *   **TMA->Compute 依赖：** 在计算使用某个 SMEM 缓冲区的数据之前，**必须**确保将该数据搬运到 SMEM 的 TMA 操作已完成 (`pipeline.consumer_wait()` + `__syncthreads()`)。`__syncthreads()` 是必须的，因为 TMA 完成状态是 per-threadblock 的，但 `pipeline.consumer_wait()` 可能只作用于提交线程。
    *   **Compute->TMA (Buffer Reuse) 依赖：** 在 TMA 可以安全地将新数据覆盖写入一个 SMEM 缓冲区之前，**必须**确保所有使用该缓冲区旧数据的 WGMMA 计算都已完成 (`pipeline.consumer_release()`)。在双缓冲中，这通常通过 `pipeline` 自动管理。
    *   **WGMMA Completion 依赖：** 在所有 K-Tile 的 WGMMA 操作都发出后，在读取累加器结果或将其写入全局内存之前，**必须**使用 `wgmma.wait_group.sync.aligned` 等待所有异步 WGMMA 操作完成。
5.  **寄存器压力管理:**
    *   WGMMA 需要大量的寄存器来保存输入操作数片段 (`A_frag`, `B_frag`) 和累加器结果 (`C_frag`)。
    *   优化寄存器使用至关重要，否则会限制 Occupancy（同时活跃的 ThreadBlock 数量），从而影响整体吞吐量。
    *   考虑：
        *   使用最小的符合精度要求的数据类型 (e.g., `fp16`, `bf16` over `fp32`).
        *   调整 Tile 大小 (`kBlockTileM/N/K`)。更大的 Tile 通常能更好地摊销 TMA 和 WGMMA 开销，但会增加寄存器和 Shared Memory 消耗，降低 Occupancy。需要在局部计算效率和全局 Occupancy 之间找到平衡点。
        *   仔细规划循环结构，复用寄存器。
6.  **流水线阶段重叠:**
    *   理想情况下，在整个 K 维循环中，TMA 加载下一个 Tile 的时间 (`T_tma`) 完全被计算当前 Tile 的时间 (`T_wgmma`) 所覆盖。即 `T_tma <= T_wgmma`。
    *   如果 `T_tma > T_wgmma`，计算会因等待数据而停顿。此时可以尝试：
        *   增大 K-Tile 大小 (`kBlockTileK`)：增加计算量 `T_wgmma`。
        *   使用三缓冲：提供额外缓冲应对 `T_tma` 波动或偶尔变长。
        *   优化 Shared Memory 布局减少 Bank Conflicts，加快 `ldmatrix` 速度（间接影响 `T_wgmma`）。
        *   检查 Global Memory 访问是否合并，TMA 描述符配置是否最优。
    *   如果 `T_wgmma > T_tma`，TMA 会提前完成，但计算单元是更宝贵的资源，这通常不是瓶颈。
7.  **使用 CUDA Pipeline Primitives:**
    *   强烈建议使用 `` 提供的 `cuda::pipeline` 机制。它比手动使用 `__pipeline_commit()`/`__pipeline_wait_prior()`/`__syncthreads()` 更简洁、更不易出错，并且能清晰地表达流水线阶段 (`producer_acquire/commit`, `consumer_wait/release`)。

## 总结

在 Hopper GPU 上实现高性能 GEMM 的核心在于：

1.  **利用 TMA 异步搬运:** 高效地将大块数据从 Global 移动到 Shared。
2.  **利用 WGMMA 异步计算:** 让张量核心在后台全力进行矩阵乘累加。
3.  **双缓冲/三缓冲:** 在 Shared Memory 中隔离计算和搬运的空间，使它们能并行发生。
4.  **CUDA Pipeline:** 清晰、安全地管理 TMA 和 WGMMA 操作之间的复杂依赖关系（数据就绪、缓冲区重用）。
5.  **精细的内存布局优化:** 确保 Shared Memory 访问（尤其是 `ldmatrix` 用于 WGMMA）高效无冲突。

通过精心设计缓冲策略、内存布局、同步点和循环结构，可以构建一个高效的异步流水线，将 TMA 的数据搬运延迟、Shared Memory 访问延迟以及 WGMMA 的计算时间最大限度地重叠起来，从而最大化地利用 Hopper 架构的硬件资源（内存带宽、张量核心计算能力），最终实现接近理论峰值的 GEMM 吞吐量。务必使用 Nsight Compute 进行性能分析，重点关注 TMA 利用率、WGMMA 利用率和 Shared Memory Bank Conflicts。