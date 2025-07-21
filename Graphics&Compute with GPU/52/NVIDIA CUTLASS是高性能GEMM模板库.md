# NVIDIA CUTLASS是高性能GEMM模板库

### 🧑‍💻 User

cutlass是现成的gemm模板库吗？

---

### 🤖 Assistant

是的，**NVIDIA CUTLASS 是一个高性能的 GEMM（通用矩阵乘法）模板库**，但它不仅仅是一个“现成的”黑盒库，而是一个**高度模块化、基于 C++ 模板的构建块集合**，专门用于在 NVIDIA GPU 上实现高效的 GEMM 和相关计算内核（如卷积）。

以下是关于 CUTLASS 的关键点，帮助你理解它是否是“现成的”：

1.  **核心是模板库：**
    *   CUTLASS 提供的是**源代码**（主要是头文件），包含大量精心设计的 C++ 模板类。
    *   这些模板定义了数据移动策略、计算操作（如 Tensor Core 指令）、线程块/线程束的组织方式、分片策略等 GEMM 所需的组件。
    *   **你需要根据自己的需求（数据类型 `FP32`, `FP16`, `BF16`, `INT8`, `TF32`等，矩阵尺寸，布局 `Row-major/Column-major`，是否使用 Tensor Cores，目标 GPU 架构）来实例化这些模板，组合成一个具体的 GEMM 内核。**

2.  **“现成”体现在哪里？**
    *   **提供高质量实现组件：** CUTLASS 提供了实现高效 GEMM 所需的所有关键、经过高度优化的构建模块。你不需要从零开始写复杂的 CUDA 内核来利用共享内存、流水线、Tensor Cores 等。
    *   **提供参考实现/示例：** CUTLASS 包含了大量预定义的、针对特定场景（如特定数据类型组合和架构）**实例化好的 GEMM 内核示例**（例如 `cutlass::gemm::device::Gemm` 的各种特化）。你可以直接编译和运行这些示例，它们通常就能提供接近 cuBLAS 的性能。
    *   **易于定制和扩展：** 如果你想微调性能或实现非标准 GEMM 变体（如批处理 GEMM、带偏移的 GEMM、结构化稀疏 GEMM），基于 CUTLASS 的模板进行修改和组合比从头开始写要容易得多、快得多。
    *   **性能有保障：** NVIDIA 持续优化 CUTLASS，其预定义内核的性能通常与高度优化的 cuBLAS 库处于同一水平，有时甚至能超越（尤其是在特定问题尺寸或新硬件特性上）。它本身就是 cuBLAS 和其他 NVIDIA 库的后端实现基础之一。

3.  **与 cuBLAS 的区别（“现成”的对比）：**
    *   **cuBLAS：** 是 NVIDIA 提供的**预编译、二进制闭源**的加速库。你通过链接库和调用 API 函数（如 `cublasGemmEx`）来执行 GEMM。它开箱即用，用户不需要关心内核实现细节，但定制性极低。
    *   **CUTLASS：** 是**开源**的**模板库/工具包**。你需要**编写代码**来实例化或组合模板（或者使用其提供的示例实例化），然后**编译**生成最终的可执行内核。它提供了极高的灵活性和定制能力，但需要用户对 GEMM 算法和 CUDA 编程模型有更深的理解。

**总结：**

*   **是 GEMM 模板库吗？** **绝对是。** 它是专门为构建高性能 GEMM 内核而设计的模板库。
*   **是“现成的”吗？** **既是也不是。**
    *   **不是开箱即用的二进制库：** 你不能像调用 cuBLAS 函数那样直接动态调用一个 CUTLASS 函数。需要编译步骤。
    *   **提供“现成的”高质量组件和参考实现：** 它提供了所有必要的、优化好的积木块，以及大量预配置好的、可以直接编译使用的 GEMM 内核示例。对于许多常见场景，编译运行这些示例几乎就是“现成可用”的高性能 GEMM 实现。
    *   **目标是让你更容易构建“现成的”高性能内核：** CUTLASS 极大地降低了开发定制化、高性能 GEMM 内核的门槛和开发时间。

**简单来说：如果你想找一个像 cuBLAS 那样直接调用的 API 库，CUTLASS 不是。但如果你想获得接近 cuBLAS 性能的 GEMM 实现源码，或者你需要高度定制化的 GEMM 内核，那么 CUTLASS 提供了最强大、最高效的“现成”模板和构建模块来帮助你达成目标。** 它是构建高性能 GEMM 的利器，而不是一个直接调用的黑盒函数。

---

### 🧑‍💻 User

可以把cutlass当成可定制的cublas吗？那我是否也可以享受cublas开箱即用的便利呢？

---

### 🤖 Assistant

你的理解非常准确！**可以把 CUTLASS 看作是一个“可定制化的 cuBLAS”**，这个比喻抓住了 CUTLASS 的核心价值：**它提供了接近 cuBLAS 性能的潜力，同时赋予了你极大的灵活性。**

**但是，关于“享受 cuBLAS 开箱即用的便利”，答案更复杂一些：**

### 🛠️ CUTLASS 作为“可定制的 cuBLAS”
1.  **性能对标：** CUTLASS 的设计目标之一就是提供与 cuBLAS 同等甚至在某些情况下更好的性能。NVIDIA 内部也在使用 CUTLASS 来构建和优化 cuBLAS 的部分实现。
2.  **深度定制：** 这是 CUTLASS 的核心优势。你可以：
    *   为特定问题尺寸（尤其是非标准尺寸）优化内核。
    *   轻松支持 cuBLAS 可能不直接支持或支持不好的新数据类型、混合精度组合或特殊布局。
    *   实现非标准的 GEMM 变体（如批处理 GEMM 的特殊访存模式、带固定缩放/偏移的 GEMM、结构化稀疏 GEMM）。
    *   精确控制计算流水线、数据预取策略、共享内存分配等底层细节。
    *   针对特定 GPU 架构（如 Ampere, Hopper）进行极致优化。
3.  **模块化构建：** 使用 CUTLASS 就像用高级乐高积木搭建 GEMM 内核，你选择不同的计算策略、数据移动策略、分块策略等模块进行组合。

### 📦 能否享受 cuBLAS 的开箱即用便利？
**直接答案：不能完全达到 cuBLAS 级别的开箱即用便利，但 CUTLASS 提供了不同层次的“便利性”，可以满足不同需求。**

| **特性**         | **cuBLAS**                     | **CUTLASS**                                   | **说明**                                                                 |
| :--------------- | :----------------------------- | :-------------------------------------------- | :----------------------------------------------------------------------- |
| **形式**         | **预编译二进制库**             | **源代码模板库**                              | cuBLAS 直接链接调用；CUTLASS 需实例化模板并编译生成内核代码               |
| **API 调用**     | **直接函数调用**               | **需编写实例化代码或封装**                    | cuBLAS 一行 API 调用；CUTLASS 需编写模板实例化代码或封装成函数            |
| **编译**         | **无需编译库本身**             | **必须编译实例化后的内核**                    | CUTLASS 每次调整参数或内核结构都需重新编译                                |
| **预置内核**     | **海量预置优化内核**           | **提供常见示例，但需手动编译**                | CUTLASS 示例覆盖常见场景，编译后接近开箱即用                              |
| **定制灵活性**   | **极低**                       | **极高**                                      | cuBLAS 参数固定；CUTLASS 可深度定制内核结构                               |
| **学习曲线**     | **低**                         | **中到高**                                    | CUTLASS 需理解 GEMM 原理、模板元编程和 CUDA 优化                          |

### 🎯 CUTLASS 提供的“便利性”层级（如何接近开箱即用）
1.  **使用预定义的示例 (`examples/` 目录)：**
    *   CUTLASS 提供了大量预配置好的示例程序（如 `00_basic_gemm`, `10_planar_complex`, `45_hopper_gemm_with_kernel_schedule` 等）。
    *   这些示例针对常见的数据类型、布局和架构进行了实例化。
    *   **做法：** 找到最接近你需求的示例，编译并运行它。这非常接近“开箱即用”。性能通常非常好。
    *   **限制：** 如果示例的参数（如数据类型、尺寸范围、布局、是否使用 Tensor Core）不完全匹配你的需求，你需要**修改示例代码中的模板参数并重新编译**。这比从头写容易，但比 cuBLAS 的 API 调用麻烦。

2.  **使用高级封装接口 (`tools/library`):**
    *   CUTLASS 提供了一个 `tools/library` 目录，其中包含一个**运行时库接口的雏形**。
    *   它定义了一组类似于 cuBLAS 的 C++ API (如 `GemmUniversal`, `GemmBatched`)，并在内部使用 CUTLASS 模板进行实例化和调度。
    *   **做法：** 你可以链接这个库，并调用它的 API。这提供了比直接使用底层模板更方便的接口。
    *   **现状：** 这个库接口不如 cuBLAS 成熟和全面。它支持的算子种类、参数选项、自动调优能力通常不如 cuBLAS。它更像是一个“如何封装 CUTLASS 内核”的参考实现。你可能需要基于它进行扩展才能满足复杂需求。

3.  **自己封装：**
    *   你可以基于 CUTLASS 模板，为你特定的应用场景编写一个封装层。
    *   例如，实例化一个固定的 `cutlass::gemm::device::Gemm` 对象，然后写一个简单的 C++ 函数调用它，隐藏内部的模板复杂性。
    *   **优点：** 一旦封装好，对于**该特定配置**，调用起来就非常方便（接近 cuBLAS）。
    *   **缺点：** 封装工作需要开发时间。如果你的应用需要多种不同的配置（不同数据类型、不同尺寸范围），你需要封装多个版本，或者实现更复杂的运行时调度（这又接近开发一个微型 cuBLAS 了）。

### 📌 结论
1.  **追求绝对开箱即用、最小化集成工作量：选 cuBLAS。** 一行 API 调用搞定，性能有保障，支持广泛。
2.  **需要极致性能定制、支持新特性/非标准操作、或愿意为特定场景封装一次以获得长期便利：选 CUTLASS。**
    *   对于常见场景，**编译运行预定义示例**是接近开箱即用的选择。
    *   对于固定需求，**自行封装一次**，之后调用封装好的接口，也能获得类似 cuBLAS 的便利（但仅限于你封装的那些配置）。
    *   **`tools/library` 提供了中间路线**，但其成熟度和覆盖度目前可能不如 cuBLAS，需要评估是否满足你的需求。

**简单说：CUTLASS 给了你打造一把专属高性能“瑞士军刀”🧰（GEMM 内核）的能力和材料，但你需要自己动手组装（编译）甚至设计（修改模板）。cuBLAS 则直接卖给你一把功能强大、开箱即用的成品瑞士军刀🔪，但你无法改变它的结构和功能。** 是否能享受“开箱即用”，取决于你选择 CUTLASS 的哪一层抽象以及你愿意提前做多少封装工作。

---

### 🧑‍💻 User

我不希望研究cute等底层细节，那又要如何快速上手cutlass？

---

### 🤖 Assistant

理解你的需求！不想深入 CUTLASS 底层（尤其是 CUTE 这类抽象），只想**快速上手并利用其高性能 GEMM 能力**是完全可行的。核心策略就是：**最大程度利用 CUTLASS 提供的预编译示例和高级封装，避免直接操作底层模板。**

以下是为你量身定制的快速上手步骤，避开底层细节：

### 🚀 快速上手策略：使用预编译示例 & 高级接口

#### 📦 1.  **直接编译和运行官方示例 (最推荐！)**
    *   **核心思想：** CUTLASS 的 `examples/` 目录下提供了大量**预配置好、可直接编译运行**的高性能 GEMM 实现。这些就是“开箱即用”的宝藏。
    *   **步骤：**
        1.  **获取代码：** `git clone https://github.com/NVIDIA/cutlass`
        2.  **进入示例目录：** `cd cutlass/build`
        3.  **配置和生成构建系统 (推荐 CMake)：**
            ```bash
            cmake .. -DCUTLASS_NVCC_ARCHS=<你的 GPU SM 架构> # 例如 -DCUTLASS_NVCC_ARCHS=80 或 90
            make -j
            ```
        4.  **查找并运行目标示例：**
            *   生成的示例程序在 `tools/examples/` 或 `examples/` 子目录下 (具体看 CMake 输出)。
            *   **最常见、最基础的 GEMM 示例：**
                *   `00_basic_gemm`：最基本 FP32 GEMM。
                *   `01_cutlass_utilities`：展示工具函数。
                *   `02_dump_reg_shmem`：调试用。
                *   `04_tensor_op_mma`：使用 Tensor Core 的混合精度 GEMM (FP16 累加 FP32, BF16 等)。**重点关注这个！**
                *   `10_planar_complex`：复数 GEMM。
                *   `13_fused_two_gemms`：融合两个 GEMM。
                *   `14_ampere_tf32_tensorop_gemm`：Ampere TF32 GEMM。
                *   `45_hopper_gemm_with_kernel_schedule`：Hopper 架构 GEMM。
            *   **运行示例 (例如 FP16 Tensor Core GEMM)：**
                ```bash
                ./examples/04_tensor_op_mma/tensorop_mma --m=1024 --n=1024 --k=1024
                ```
            *   **关键参数：** 运行程序通常带 `--help` 查看支持的参数。最常用的是 `--m`, `--n`, `--k` (矩阵维度)，`--alpha`, `--beta` (标量系数)。示例程序会自动选择合适的内核。
    *   **优点：**
        *   **完全避开底层：** 你不需要理解任何 CUTLASS 模板或 CUTE。
        *   **高性能：** 这些示例使用的内核是 NVIDIA 精心优化过的，性能通常接近 cuBLAS。
        *   **开箱即用：** 编译后直接运行，像使用一个命令行工具。
        *   **验证与基准测试：** 快速测试不同尺寸、数据类型的 GEMM 在你 GPU 上的性能，并与 cuBLAS 比较 (`--reference-check` 通常开启验证)。
    *   **如何“定制化”？**
        *   如果预编译示例的参数（如 `--m=1024`, `--n=1024`, `--k=1024`, 数据类型 FP16）正好满足你的需求，那就直接用！
        *   **修改示例代码 (轻度定制)：** 如果示例 *接近* 你的需求（比如你需要 BF16 而不是 FP16），直接修改对应示例的 `.cu` 文件中的**模板参数**通常很简单。例如在 `04_tensor_op_mma.cu` 中找到 `using Gemm = cutlass::gemm::device::Gemm<...>` 这行，修改里面的 `ElementA`, `ElementB`, `ElementC`, `ElementAccumulator`, `OpClassTensorOp`, `ArchTag` 等类型定义即可。**这是最常见的“有限定制”方式，无需懂底层。** 修改后重新 `make`。

#### 🧩 2.  **使用 `tools/library` 接口 (类 cuBLAS API)**
    *   **核心思想：** CUTLASS 提供了一个正在发展的**运行时库接口** (`tools/library`)，旨在模仿 cuBLAS 的调用方式。
    *   **步骤：**
        1.  **编译库：** 在 `build` 目录执行 `make cutlass_lib` (或根据 CMakeLists 找到构建 `cutlass_library` 的目标)。这会生成一个 `libcutlass_library.a` (静态库) 或 `.so` (动态库，如果配置了)。
        2.  **包含头文件：** 在你的应用程序中包含 `cutlass/library/library.h`, `cutlass/library/handle.h` 等。
        3.  **链接库：** 链接编译好的 `libcutlass_library` 和 CUDA 运行时等。
        4.  **编写调用代码：** 使用类似 cuBLAS 的 API 进行 GEMM 调用。
    *   **示例代码片段：**
        ```cpp
        #include <cutlass/library/library.h>
        #include <cutlass/library/handle.h>
        #include <cutlass/library/operation_table.h>

        // ... (初始化 CUDA, 分配设备内存 d_A, d_B, d_C, 设置 M, N, K, alpha, beta, lda, ldb, ldc 等)

        // 创建库句柄
        cutlass::library::Handle handle;

        // 配置 GEMM 操作描述符 (非常类似 cuBLAS)
        cutlass::library::GemmUniversalDescriptor op_desc;
        op_desc.problem_size = {M, N, K}; // 问题尺寸
        op_desc.A = {d_A, cutlass::library::LayoutTypeID::kColumnMajor, lda}; // A 矩阵指针/布局/leading dim
        op_desc.B = {d_B, cutlass::library::LayoutTypeID::kColumnMajor, ldb};
        op_desc.C = {d_C, cutlass::library::LayoutTypeID::kColumnMajor, ldc};
        op_desc.alpha = &alpha; // scalar pointer
        op_desc.beta = &beta;
        op_desc.kind = cutlass::library::GemmUniversalKind::kArray; // 或 kGemm (取决于参数传递方式)
        op_desc.batch_count = 1; // 批处理大小

        // 设置计算类型 (例如 FP32 累加的 FP16 GEMM)
        cutlass::library::NumericTypeID compute_type = cutlass::library::NumericTypeID::kF32;
        cutlass::library::NumericTypeID scale_type = cutlass::library::NumericTypeID::kF32;

        // 查找并执行最优内核
        cutlass::Status status = handle.gemm_universal_operation(
            op_desc, // 操作描述符
            compute_type, // 计算累加类型
            scale_type, // alpha/beta 标量类型
            cutlass::library::NumericTypeID::kF16, // A 元素类型
            cutlass::library::NumericTypeID::kF16, // B 元素类型
            cutlass::library::NumericTypeID::kF16, // C 元素类型
            cutlass::library::NumericTypeID::kF16, // D 元素类型 (通常=C)
            nullptr, // 可选: 偏好某个内核的 tag
            nullptr // 可选: 调优结果输出
        );

        // 检查状态
        if (status != cutlass::Status::kSuccess) { ... }
        ```
    *   **优点：**
        *   **API 类似 cuBLAS：** 调用方式更接近你熟悉的 cuBLAS `cublasGemmEx`。
        *   **运行时调度：** 库内部根据问题尺寸、数据类型、GPU 架构自动选择最优预编译内核，无需你手动实例化。
        *   **避开模板实例化：** 你写的应用代码是标准的 C++/CUDA，不涉及 CUTLASS 模板。
    *   **缺点/现状：**
        *   **不如 cuBLAS 成熟稳定：** 功能覆盖度、错误处理、API 稳定性可能不如 cuBLAS。
        *   **支持的算子有限：** 主要聚焦在标准 GEMM 和批处理 GEMM。复杂变体 (如带 epilogue 融合) 支持可能不完善。
        *   **性能调度器可能不如 cuBLAS 智能：** 自动选择的内核不一定总是最优。
        *   **需要额外编译链接库：** 比直接运行示例程序稍复杂。
    *   **适用场景：** 当你需要在应用程序中以 **API 调用方式** 嵌入 CUTLASS GEMM，且预编译示例不能满足集成需求时。**这是最接近 cuBLAS “开箱即用” 体验的方案。**

### 📌 总结：如何快速上手而不碰底层

1.  **首选：运行预编译示例 (`examples/`)**
    *   编译整个 `build` 目录 (`cmake .. && make -j`)。
    *   找到并运行最接近你需求的示例程序 (如 `04_tensor_op_mma`, `14_ampere_tf32_tensorop_gemm`)。用 `--help` 看参数。
    *   轻度定制：直接修改示例 `.cu` 文件中的模板参数 (数据类型、ArchTag 等)，然后重新编译该示例。**这是上手最快、最实用的方式。**
2.  **次选：使用 `tools/library` API**
    *   编译 `cutlass_library` (`make cutlass_lib`)。
    *   在你的 C++ 代码中包含 `library.h` 等，链接 `libcutlass_library`。
    *   使用 `Handle::gemm_universal_operation` 等 API 调用 GEMM。**这提供了类似 cuBLAS 的编程体验。**
3.  **明确避免：**
    *   阅读 `include/cutlass/gemm/device/` 下的复杂模板代码 (除非必要)。
    *   直接使用 `Gemm` 等 device 级模板并自行实例化 (除非示例和库接口都无法满足)。
    *   研究 `CUTE` (Cutlass Universal Tensor Expressions) 抽象层。

**一句话：把你的“快速上手”集中在编译和运行 `examples/` 下的程序，以及探索 `tools/library/` 的 API 上。95% 的需求可以通过这两种方式满足，完全不需要触碰 CUTLASS 的底层模板魔方或 CUTE。** 把这当成一个提供高性能可执行程序和类 cuBLAS 库的宝藏项目来用！