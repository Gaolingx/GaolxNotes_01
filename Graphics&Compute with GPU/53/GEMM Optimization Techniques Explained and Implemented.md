# GEMM Optimization Techniques Explained and Implemented

### 🧑‍💻 User

常见的GEMM 优化技术分为以下这些类型，能不能具体分析下，最后用c++ with simd intrinsics实现一个简单的matmul kernel？
- 分块（Tiling）：将矩阵划分为能放入缓存的小块，从而提升内存访问效率。
- 重排序（Reordering）：通过交换代码中 for 循环的顺序来优化内存访问。
- 向量化（Vectorization）：将矩阵中的多个数据打包在一起，以实现高效的内存访问和计算。
- 布局（Layout）：改变矩阵的排列方式，以匹配特定硬件的内存访问特性。
- 流水线（Pipeline）：通过重叠计算和数据移动的不同阶段来提升性能，实现连续处理并减少操作间的空闲时间。

---

### 🤖 Assistant

好的，我们来详细分析这些常见的GEMM（通用矩阵乘法）优化技术，并最后用C++和SIMD intrinsics实现一个简单的`matmul` kernel。

### GEMM优化技术详解

GEMM是许多科学计算和机器学习应用中的核心操作，其性能至关重要。由于现代处理器中计算速度远超内存访问速度，因此GEMM优化的核心思想是 **最大化数据复用，减少内存访问开销**。

-----

#### 1\. 分块 (Tiling / Blocking)

**核心思想：** 将大矩阵分解成能够装入CPU缓存（L1, L2, L3）的小块（Tiles）。通过对小块进行计算，可以最大程度地复用加载到缓存中的数据，避免重复从主内存中读取。

**分析：**

  * **数据局部性原理：** 计算机存储系统是分层的（寄存器 -\> L1缓存 -\> L2缓存 -\> L3缓存 -\> 主存）。访问速度逐级递减，容量逐级递增。如果处理器需要的数据恰好在缓存中，其访问速度会比在主存中快几个数量级。
  * **问题场景：** 在一个朴素的矩阵乘法 `C = A * B` 中（`ijk` 顺序），计算 `C` 的一个元素需要 `A` 的一整行和 `B` 的一整列。当矩阵很大时，`B` 的访问模式是列式跳跃的（在行主序存储中），这会导致缓存行（Cache Line）的利用率很低，频繁地从主存加载新数据，造成所谓的“缓存抖动”（Cache Thrashing）。
  * **解决方法：** 我们不计算单个 `C` 的元素，而是计算 `C` 的一个子矩阵（`C_sub`）。这个子矩阵的计算只需要 `A` 的一个行块（`A_sub`）和 `B` 的一个列块（`B_sub`）。通过精心选择块的大小，我们可以确保 `A_sub`，`B_sub` 和 `C_sub` 能够同时驻留在某个级别的缓存中。在计算 `C_sub` 的所有元素时，`A_sub` 和 `B_sub` 的数据被反复使用，极大地减少了对主存的访问。

**图解：**
想象一下将矩阵A、B、C都切成小方块。我们加载A的一个行块和B的一个列块到缓存中，然后用它们来更新C的一个目标块。这个过程会持续进行，直到A的行块遍历完所有B的列块。

-----

#### 2\. 重排序 (Reordering)

**核心思想：** 调整`for`循环的嵌套顺序，以改变内存访问模式，从而更好地利用数据局部性。

**分析：**
对于矩阵乘法，有六种可能的循环顺序：`ijk`, `ikj`, `jik`, `jki`, `kij`, `kji`。

  * **`ijk` (朴素实现):**

    ```cpp
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    ```

      - `A` 的访问是行主序连续的（好）。
      - `B` 的访问是列主序跳跃的（差）。
      - `C` 的元素在内层循环中被反复读写（好）。

  * **`ikj`:**

    ```cpp
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    ```

      - `A` 的元素在内层循环中被复用（好）。
      - `B` 的访问是行主序连续的（好）。
      - `C` 的访问是行主序连续的，但每次内层循环都完整遍历一行（还可以）。

  * **`jik`:**

    ```cpp
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    ```

      - 这种顺序通常性能较差，因为外层循环是 `j`，导致 `C` 和 `A` 的访问都可能是不连续的。

**最佳实践：**
通常，结合 **分块** 技术后，循环重排序变得更加强大。在外层循环中对块进行迭代，在内层循环中对块内的元素进行计算。`ikj` 或 `jki` 变体在结合分块后通常能获得很好的性能，因为它们能更好地利用寄存器和缓存。

-----

#### 3\. 向量化 (Vectorization)

**核心思想：** 利用CPU的SIMD（Single Instruction, Multiple Data）指令集（如SSE, AVX, AVX-512），一次性对多个数据执行相同的操作。

**分析：**

  * **硬件支持：** 现代CPU拥有一组特殊的、宽度为128位、256位甚至512位的寄存器。一条SIMD指令就可以将这些寄存器中的多个数据元素（例如，4个32位浮点数或8个16位整数）与另一组数据元素进行并行计算。
  * **实现方式：**
    1.  **编译器自动向量化：** 现代编译器（如GCC, Clang）会尝试自动将循环中的标量操作转换为SIMD指令。但这需要代码结构清晰、无数据依赖。
    2.  **Intrinsics（内联函数）：** 通过在C++代码中直接调用SIMD指令的内联函数（例如 `_mm256_load_ps`, `_mm256_mul_ps`, `_mm256_add_ps`），可以精确地控制向量化过程。这是高性能库（如BLAS）常用的方法。
  * **在GEMM中的应用：** 在计算 `C_sub += A_sub * B_sub` 时，我们可以一次加载 `A_sub` 的一个元素和 `B_sub` 的一行中的多个元素（打包成一个SIMD向量），然后进行乘加运算，结果累加到 `C_sub` 对应行的SIMD向量中。

-----

#### 4\. 布局 (Layout)

**核心思想：** 改变矩阵数据在内存中的物理排列方式，使其更适合硬件的访问模式，特别是SIMD操作和缓存行加载。

**分析：**

  * **行主序 (Row-Major) vs. 列主序 (Column-Major):**
      * C/C++默认使用行主序，即 `A[i][j]` 和 `A[i][j+1]` 在内存中是相邻的。
      * Fortran/MATLAB默认使用列主序，即 `A[i][j]` 和 `A[i+1][j]` 在内存中是相邻的。
  * **问题：** 在行主序下，访问矩阵的一行是连续的（缓存友好），访问一列是跳跃的（缓存不友好）。
  * **优化方法：**
    1.  **转置 (Transpose):** 在计算 `C = A * B` 之前，可以将矩阵 `B` 进行转置得到 `B^T`。这样，乘法就变成了 `A` 的行与 `B^T` 的行进行点积。访问 `B^T` 的行就等同于访问原 `B` 的列，但现在内存访问是连续的。
    2.  **数据重排 (Data Packing/Swizzling):** 更高级的方法是在分块的基础上，将小块内的元素按照特定的顺序（如Z-Morton曲线顺序）重新排列。这种布局可以使得无论按行还是按列访问小块，都能获得较好的缓存局部性。对于向量化，可以将数据预先整理成SIMD向量所需的格式，这个过程称为 "packing"。

-----

#### 5\. 流水线 (Pipelining)

**核心思想：** 重叠计算和数据移动，隐藏内存访问延迟，让计算单元持续工作，避免空闲。

**分析：**

  * **指令级并行 (Instruction-Level Parallelism, ILP):** 现代CPU可以同时执行多条没有相互依赖的指令。通过精心安排指令，可以使得加载数据、执行计算、存储结果等操作并行进行。
  * **软件流水线 (Software Pipelining):**
    1.  **预取 (Prefetching):** 在当前数据块计算的同时，使用专门的预取指令（如 `_mm_prefetch`）提前将下一个要用到的数据块加载到缓存中。当计算单元完成当前任务时，下一个任务所需的数据已经准备好了，从而隐藏了内存访问的延迟。
    2.  **双缓冲 (Double Buffering):** 使用两块缓冲区。当CPU在处理缓冲区A中的数据时，DMA（直接内存访问）控制器可以异步地将下一批数据加载到缓冲区B中。当CPU处理完A后，立即切换到B，同时DMA开始向A中填充再下一批数据。

**综合应用：**
一个高度优化的GEMM实现（如OpenBLAS, MKL）会综合运用以上所有技术。其典型的执行流程如下：

1.  **宏内核 (Macro-Kernel):** 将大矩阵分块（Tiling）。循环遍历这些大块。
2.  **数据打包 (Packing):** 在处理每个大块之前，先将 `A` 和 `B` 的数据块按照特定的内存布局（Layout）复制到一个连续的缓冲区中。这个过程可以优化缓存和SIMD的效率。
3.  **微内核 (Micro-Kernel):** 这是最优化的核心部分。它使用汇编或Intrinsics编写，负责计算打包后的小块矩阵。微内核内部会充分利用向量化（Vectorization）、循环重排序（Reordering）和指令级并行（Pipelining）技术，将数据加载到寄存器并高效计算。通过预取指令实现软件流水线，隐藏数据加载延迟。

-----

### C++ with SIMD Intrinsics 简单实现

下面是一个简化的示例，它主要演示 **向量化** 和 **分块** 的基本思想。我们将使用Intel的AVX intrinsics（需要支持AVX的CPU和相应的编译器标志，如 `-mavx`）。

这个例子将实现一个`4x4`的微内核，用它来构建一个分块矩阵乘法。

**前提:**

  * 编译器：GCC 或 Clang
  * 编译指令：`g++ -o matmul matmul.cpp -O3 -mavx`
  * CPU：支持AVX指令集

<!-- end list -->

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h> // Intel Intrinsics头文件

// 假设矩阵是行主序存储
void matmul_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 核心：一个4x8的微内核，使用AVX进行向量化计算
// 计算 C[i:i+4][j:j+8] += A[i:i+4][k] * B[k][j:j+8]
// 注意：为了简化，这里假设K是1的倍数
static void kernel_4x8(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
    // 8个256位的向量寄存器来存储C的4x8子块的结果
    __m256 c_row0, c_row1, c_row2, c_row3;
    
    // 加载C的当前值
    c_row0 = _mm256_loadu_ps(&C[0 * ldc]);
    c_row1 = _mm256_loadu_ps(&C[1 * ldc]);
    c_row2 = _mm256_loadu_ps(&C[2 * ldc]);
    c_row3 = _mm256_loadu_ps(&C[3 * ldc]);

    for (int k = 0; k < K; ++k) {
        // 加载B的一行 (8个浮点数)
        __m256 b_vec = _mm256_loadu_ps(&B[k * ldb]);

        // 加载A的一列中的4个元素，并广播到整个向量
        __m256 a_val0 = _mm256_broadcast_ss(&A[0 * lda + k]);
        __m256 a_val1 = _mm256_broadcast_ss(&A[1 * lda + k]);
        __m256 a_val2 = _mm256_broadcast_ss(&A[2 * lda + k]);
        __m256 a_val3 = _mm256_broadcast_ss(&A[3 * lda + k]);
        
        // FMA: Fused Multiply-Add (a * b + c)
        // c_row0 = a_val0 * b_vec + c_row0;
        c_row0 = _mm256_fmadd_ps(a_val0, b_vec, c_row0);
        c_row1 = _mm256_fmadd_ps(a_val1, b_vec, c_row1);
        c_row2 = _mm256_fmadd_ps(a_val2, b_vec, c_row2);
        c_row3 = _mm256_fmadd_ps(a_val3, b_vec, c_row3);
    }
    
    // 将结果写回C
    _mm256_storeu_ps(&C[0 * ldc], c_row0);
    _mm256_storeu_ps(&C[1 * ldc], c_row1);
    _mm256_storeu_ps(&C[2 * ldc], c_row2);
    _mm256_storeu_ps(&C[3 * ldc], c_row3);
}


// 使用分块和SIMD优化的Matmul
void matmul_optimized(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    // 初始化C矩阵为0
    std::fill(C.begin(), C.end(), 0.0f);

    // 定义块大小
    const int BLOCK_M = 4;
    const int BLOCK_N = 8; // AVX一次处理8个float

    // 分块循环
    for (int i = 0; i < M; i += BLOCK_M) {
        for (int j = 0; j < N; j += BLOCK_N) {
            // 对每个C的小块调用微内核
            // 这里我们假设M和N是块大小的倍数，简化边界处理
            kernel_4x8(BLOCK_M, BLOCK_N, K, &A[i * K], &B[j], &C[i * N + j], K, N, N);
        }
    }
}


int main() {
    // 确保维度是块大小的倍数以简化示例
    const int M = 256;
    const int N = 512;
    const int K = 128;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_naive(M * N);
    std::vector<float> C_optimized(M * N);

    // 初始化随机数据
    for (int i = 0; i < M * K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = (float)rand() / RAND_MAX;

    // --- 朴素版本 ---
    auto start_naive = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B, C_naive, M, N, K);
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naive_ms = end_naive - start_naive;
    std::cout << "Naive Matmul took: " << naive_ms.count() << " ms\n";

    // --- 优化版本 ---
    // 为了更好的性能，B矩阵应该按列打包(Layout)，这里为了简化没有做
    auto start_optimized = std::chrono::high_resolution_clock::now();
    matmul_optimized(A, B, C_optimized, M, N, K);
    auto end_optimized = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> optimized_ms = end_optimized - start_optimized;
    std::cout << "Optimized Matmul took: " << optimized_ms.count() << " ms\n";

    // 验证结果 (可选)
    float max_err = 0.0f;
    for(int i=0; i<M*N; ++i) {
        max_err = std::max(max_err, std::abs(C_naive[i] - C_optimized[i]));
    }
    std::cout << "Max error between naive and optimized: " << max_err << std::endl;


    return 0;
}
```

**代码分析：**

1.  **`kernel_4x8` (微内核):**
      * **目标:** 计算`4x8`大小的 `C` 子矩阵。`C_sub += A_sub * B_sub`。
      * **向量化:**
          * `__m256` 是一个可以容纳8个32位浮点数的AVX向量类型。
          * 我们用4个`__m256`向量 (`c_row0`到`c_row3`) 来作为累加器，分别对应 `C` 子矩阵的4行。
          * 在`k`循环中，`_mm256_loadu_ps` 从 `B` 中加载连续的8个浮点数（`B`的一行的一部分）。`loadu`表示非对齐加载，更灵活。
          * `_mm256_broadcast_ss` 从 `A` 中加载一个浮点数，并将其复制到向量的所有8个位置。这使得我们可以将一个标量与一个向量相乘。
          * `_mm256_fmadd_ps(a, b, c)` 是核心操作，它执行 `a * b + c`，即融合乘加。这比单独的乘法和加法更快，且精度更高。
2.  **`matmul_optimized` (宏内核):**
      * **分块:** 代码通过 `i` 和 `j` 循环以`BLOCK_M` (4) 和 `BLOCK_N` (8) 的步长遍历 `C` 矩阵，实现了分块。
      * **调用微内核:** 在循环体内，直接调用`kernel_4x8`来处理每个`4x8`的块。
3.  **局限与改进空间：**
      * **边界处理:** 这个简单的实现假设 `M` 和 `N` 都是块大小的整数倍。一个完整的库需要处理边缘情况。
      * **Layout (Packing):** `kernel_4x8` 中 `B` 的加载是连续的，这是好的。但 `A` 的加载是 `_mm256_broadcast_ss`，效率不是最高。一个更高级的实现会预先对 `A` 进行打包，使其数据也变得连续，从而可以用 `_mm256_load_ps` 直接加载，进一步提升效率。
      * **缓存分块:** 这里的`K`循环没有分块。对于非常大的`K`，`B`矩阵的行无法完全放入缓存。完整的实现会增加一个`k`方向的分块循环。
      * **流水线:** 没有显式的预取指令。在一个更复杂的内核中，会插入`_mm_prefetch`来提前加载后续循环需要的数据。

即便如此，这个结合了分块和向量化的`matmul_optimized`版本，相比`matmul_naive`，通常也能展现出显著的性能提升。