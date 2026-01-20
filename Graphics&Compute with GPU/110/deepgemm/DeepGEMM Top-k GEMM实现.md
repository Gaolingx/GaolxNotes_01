# DeepGEMM Top-k GEMM实现

### 🧑‍💻 User

[deepseek-ai-deepgemm-8a5edab282632443.txt](./deepseek-ai-deepgemm-8a5edab282632443.txt)

在deepseek-ai的deepgemm（https://github.com/deepseek-ai/DeepGEMM）中，这具体是怎么实现top-k个专家的fused gemm的？

---

### 🤖 Assistant

好的，我们来深入分析 DeepGEMM 中为 Top-K 专家（MoE 模型）设计的 Fused GEMM 是如何实现的。

其核心思想是将多个小型的 GEMM 操作融合成一个大的、单一的 CUDA Kernel 来执行，从而避免多次 Kernel 启动的开销，并最大化 GPU 的利用率。这主要是通过一种称为 **M-Grouped Contiguous GEMM** 的策略来实现的。

### 1. 核心概念：M-Grouped Contiguous GEMM

传统的 Grouped GEMM (例如 CUTLASS 中的) 通常处理一批形状各异的矩阵乘法。然而，在 MoE 模型中，所有专家的权重矩阵形状通常是相同的。DeepGEMM 利用了这一特性。

它的策略不是为每个专家启动一个独立的 GEMM，也不是使用传统的 Batched GEMM，而是：

1.  **数据重排 (Data Layout)**：
    *   **输入激活 (Activation) `A`**：将所有被路由到这 `k` 个专家的 tokens 的激活向量，按照专家分组、连续排列，形成一个大的 `A` 矩阵。其形状为 `[M_total, K]`，其中 `M_total` 是所有 `k` 个专家要处理的 token 总数。
    *   **专家权重 (Weights) `B`**：将 `k` 个专家的权重矩阵堆叠（stack）起来，形成一个 `B` 张量。其形状为 `[num_experts, N, K]`。
    *   **输出 (Output) `D`**：输出张量的形状与输入 `A` 类似，为 `[M_total, N]`。

2.  **路由信息 (`grouped_layout`)**：为了在融合的 Kernel 中知道哪个 token 应该使用哪个专家的权重，需要一个额外的索引张量，在 DeepGEMM 中被称为 `grouped_layout`。
    *   这是一个一维整数张量，形状为 `[M_total]`。
    *   `grouped_layout[i]` 的值代表第 `i` 个 token 应该使用的专家索引（expert index）。

3.  **关键约束**：为了实现高效的内存访问和计算，DeepGEMM 施加了一个重要的约束：**所有在同一个计算块（`BLOCK_M`）内的 tokens 必须被路由到同一个专家**。这意味着在将 tokens 送入 DeepGEMM 之前，需要对它们进行排序和填充（padding），以确保每个专家的 token 数量是 `BLOCK_M` 的整数倍。这个对齐大小可以通过 `deep_gemm.get_mk_alignment_for_contiguous_layout()` 获取。

### 2. 代码实现追踪

下面我们从 Python API 到 CUDA Kernel 层面，追踪这一实现。

#### 步骤 1：Python API 和 C++ 绑定

用户从 Python 中调用类似 `m_grouped_fp8_gemm_nt_contiguous` 的函数。
`deep_gemm/__init__.py`:
```python
from ._C import (
    # ...
    m_grouped_fp8_gemm_nt_contiguous,
    # ...
)
```
这个调用会进入 `csrc/python_api.cpp` 中注册的 C++ 函数，最终调用到 `csrc/apis/gemm.hpp` 中的 `m_grouped_fp8_fp4_gemm_nt_contiguous` 函数。

`csrc/apis/gemm.hpp`:
```cpp
static void m_grouped_fp8_fp4_gemm_nt_contiguous(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d,
    const torch::Tensor& grouped_layout, // 核心：路由信息
    // ...
) {
    // 形状检查: A 是 [M, K], B 是 [G, N, K], d 是 [M, N]
    // G (num_groups) 是专家数量
    const auto [m , k ] = check_ab_fp8_fp4(a.first, ...);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, ...);
    const auto [m_, n_] = get_shape<2>(d);

    // grouped_layout 检查
    // ...
    const auto& [m__] = get_shape<1>(grouped_layout);
    DG_HOST_ASSERT(m == m__); // 确认 grouped_layout 的长度等于 token 总数 M

    // ... 省略了缩放因子(scaling factor)和架构相关的准备代码

    // 分发到具体架构的实现
    if (arch_major == 9 ...) {
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, grouped_layout, ...);
    } else if (arch_major == 10 ...) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, grouped_layout, ...);
    }
}
```
这个 C++ 函数主要做形状检查、准备数据，然后根据 GPU 架构（SM90 或 SM100）调用更底层的实现。

#### 步骤 2：Kernel 启动和 JIT 编译

我们以 SM100 (Hopper 架构) 为例，进入 `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp` (FP8的逻辑类似)。

```cpp
static void sm100_m_grouped_bf16_gemm_contiguous(
    // ...
    const torch::Tensor& grouped_layout,
    // ...
) {
    // 1. 使用启发式模型选择最佳配置 (块大小, 流水线阶段数等)
    const auto& config = get_best_config<SM100ArchSpec>(
        GemmType::MGroupedContiguous, // << 指定GEMM类型
        ...);

    // 2. 创建TMA描述符，用于硬件加速的内存拷贝
    const auto& tensor_map_a = make_tma_a_desc(...);
    const auto& tensor_map_b = make_tma_b_desc(..., num_groups, ...); // b的描述符会考虑专家数量
    const auto& tensor_map_cd = make_tma_cd_desc(...);

    // 3. 准备启动参数
    const SM100BF16GemmRuntime::Args& args = {
        // ...
        .gemm_config = config,
        .grouped_layout = grouped_layout.data_ptr(), // 传递 grouped_layout 的指针
        // ...
    };

    // 4. JIT编译并启动Kernel
    const auto& code = SM100BF16GemmRuntime::generate(args);
    const auto& runtime = compiler->build("sm100_bf16_m_grouped_gemm_contiguous", code);
    SM100BF16GemmRuntime::launch(runtime, args);
}
```
这一层是“主机端(Host)”代码，它不执行 GEMM 计算，而是配置、编译并启动将在 GPU 上运行的 CUDA Kernel。

#### 步骤 3：CUDA Kernel 内部实现

真正的魔法发生在 CUDA Kernel 内部，例如 `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` 中的 `sm100_bf16_gemm_impl`。

这个 Kernel 的核心是 `Scheduler` 类 (`deep_gemm/include/deep_gemm/common/scheduler.cuh`)，它负责为每个 CUDA 块（CTA）分配工作。

```cpp
// deep_gemm/include/deep_gemm/common/scheduler.cuh (概念性简化)

template <GemmType kGemmType, ...>
struct Scheduler {
    // ...
    int* grouped_layout;

    __device__ __forceinline__ explicit Scheduler(..., int* grouped_layout = nullptr) {
        // ...
        if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            this->grouped_layout = grouped_layout;
        }
    }

    template <bool kWithGroupOffset, ...>
    __device__ __forceinline__ uint32_t get_global_idx(...) {
        if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            // 如果需要组偏移 (例如，访问权重B时)
            if (kWithGroupOffset) {
                // 读取 grouped_layout 张量来获取专家索引
                // __ldg 是只读缓存加载指令 (load via texture cache)
                // m_block_idx * BLOCK_M 定位到当前token块的起始位置
                const auto offset = cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M));
                // 返回: expert_idx * N + n_offset
                return offset * shape_dim + block_idx * block_size;
            } else {
                 return block_idx * block_size;
            }
        }
        // ... 其他 GEMM 类型的逻辑
    }
};
```

**工作流解释**:

1.  **Kernel 启动**: GPU 启动一个大的 Grid，包含许多 CUDA 块 (CTAs)。
2.  **任务分配**: 每个 CTA 进入一个 `while` 循环，通过 `scheduler.get_next_block(m_block_idx, n_block_idx)` 获取它要处理的 `D` 矩阵的块坐标。`m_block_idx` 对应于 `A` 矩阵的行块，`n_block_idx` 对应于 `B` 矩阵的列块。
3.  **加载权重 (Fused 的关键)**:
    *   当一个 CTA 需要加载其对应的 `B` 矩阵（专家权重）的瓦片（tile）时，它会调用 `scheduler.get_global_idx` 来计算内存地址。
    *   `get_global_idx` 函数会执行 `__ldg(grouped_layout + m_block_idx * BLOCK_M)`。因为之前已经保证了一个 `BLOCK_M` 内的所有 tokens 属于同一个专家，所以只需要读取这个块的第一个 token 对应的专家索引即可。
    *   这个索引（`offset`）被用来计算 `B` 张量 `[num_experts, N, K]` 的偏移量，确保 CTA 从正确的专家权重中加载数据。
    *   例如，要加载第 `i` 个专家的权重，地址偏移就是 `i * N * K`。
4.  **加载激活**: 加载 `A` 矩阵的瓦片则比较直接，仅使用 `m_block_idx` 计算行偏移。
5.  **计算与存储**: CTA 加载了正确的 `A` 瓦片和 `B` 瓦片后，执行矩阵乘法，并将结果写回 `D` 矩阵的相应位置。

### 总结

DeepGEMM 实现 Top-K 专家 Fused GEMM 的方法可以概括为：

-   **数据预处理**: 将 tokens 按专家排序，聚合成一个大的激活矩阵 `A`。
-   **单一 Kernel 启动**: 启动一个单一的、大规模的 M-Grouped GEMM Kernel，而不是多个小 Kernel。
-   **智能索引**: 在 Kernel 内部，通过一个 `grouped_layout` 辅助张量，在运行时动态地为每个计算块（CTA）选择正确的专家权重矩阵。
-   **硬件利用**: 这种方式将大量并行的计算任务打包到一个 Kernel 中，通过 `Scheduler` 在 SM（Streaming Multiprocessor）之间高效分发，充分利用了 GPU 的并行计算能力，并减少了启动开销。

这种设计巧妙地将在 MoE 路由中产生的、离散的、小规模的计算任务，转换成了适合 GPU 大规模并行架构的、连续的、大规模的计算任务。