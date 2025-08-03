# Vulkan 协作矩阵加速计算

### 🧑‍💻 User

vulkan shader中的cooperative vector有什么用途？

---

### 🤖 Assistant

好的，我们来详细探讨一下 Vulkan着色器中的“协作矩阵”（Cooperative Matrix），这在技术上通常被称为 `Cooperative Matrix` 而非 `Cooperative Vector`。这可能是您想了解的核心概念，因为它是在现代Vulkan和GLSL中引入的一项重要特性，旨在加速机器学习和科学计算等领域的矩阵运算。

这个功能通过 `GL_KHR_cooperative_matrix` GLSL扩展和Vulkan的 `VK_KHR_cooperative_matrix` 扩展来启用。

### 什么是协作矩阵 (Cooperative Matrix)？

协作矩阵是一种新的变量类型，它允许一个着色器线程组（Subgroup）中的多个线程（invocations）共同协作来处理一个更大的矩阵。这个矩阵被分割成小块，每个线程负责处理其中的一小部分。关键在于，这些操作是在硬件级别上进行优化的，能够利用现代GPU中专门为矩阵乘法和累加（Matrix Multiply-Accumulate, MMA）设计的硬件单元，例如NVIDIA的Tensor Cores或AMD的Matrix Cores。

您可以将其想象成一个团队任务：一个大矩阵的乘法任务被分配给一个线程小组（Subgroup），小组中的每个成员（线程）都领取一小块任务，然后硬件会以一种高度并行和高效的方式协调它们，最终将结果组合在一起。

### 协作矩阵的主要用途和优势

协作矩阵的核心用途是**在着色器中以极高的效率执行矩阵乘法累加（MMA）操作**，即 `D = A * B + C` 这样的运算。这在以下领域至关重要：

1.  **机器学习（Machine Learning）推断**：

      * 神经网络（尤其是深度神经网络，DNNs）的核心计算就是大量的矩阵乘法。例如，卷积层和全连接层都可以分解为一系列的矩阵乘法操作。
      * 在游戏中实时运行AI模型，如风格迁移、DLSS（深度学习超级采样）或高级的敌人AI，都需要在GPU上进行高效的推断。协作矩阵让游戏开发者可以直接在图形渲染管线或计算管线中利用硬件加速来实现这些功能。

2.  **科学与高性能计算（HPC）**：

      * 许多物理模拟、信号处理（如傅里叶变换）、流体动力学等科学计算问题都依赖于大规模的矩阵运算。
      * 在GPU上直接利用其专用的MMA硬件，可以显著加速这些计算任务。

3.  **高级图像处理**：

      * 复杂的图像滤波、计算机视觉算法（如特征提取）和实时光线追踪中的降噪算法，往往也涉及到矩阵运算。协作矩阵可以为这些任务提供硬件加速。

#### 优势总结：

  * **极致的性能**：通过利用专门的硬件单元（如Tensor Cores），矩阵运算的速度可以比使用传统的通用ALU（算术逻辑单元）快一个数量级。这带来了更高的吞吐量和更低的功耗。
  * **简化编程模型**：虽然需要理解其概念，但相比于手动通过共享内存（Shared Memory）和复杂的同步来管理和优化矩阵分块，协作矩阵提供了一个更高层次、更抽象的API。开发者只需定义矩阵的类型、大小和布局，然后调用内建的`cooperativeMatrixMulAdd`函数即可。
  * **与图形和计算管线的无缝集成**：开发者可以在现有的Vulkan计算着色器（Compute Shader）甚至图形着色器（如Fragment Shader）中直接使用这些功能，从而轻松地将AI/ML加速集成到渲染流程中。

### 工作原理简述

使用协作矩阵通常遵循以下步骤：

1.  **启用扩展**：在创建Vulkan实例和设备时，需要启用 `VK_KHR_cooperative_matrix` 扩展。

2.  **查询硬件支持**：查询物理设备支持哪些矩阵大小、类型（如FP16, FP32, INT8）和作用域（Scope）。

3.  **在GLSL中声明**：

      * 使用 `#extension GL_KHR_cooperative_matrix : require` 来启用GLSL扩展。
      * 使用 `coopmat` 关键字声明协作矩阵变量。你需要指定其数据类型、作用域（通常是`subgroup`）、行数和列数。
      * 例如：`coopmat<float, gl_ScopeSubgroup, 16, 16> myMatrix;`

4.  **加载数据 (Load)**：

      * 矩阵数据通常从输入缓冲区（Storage Buffer）或共享内存中加载。
      * 由于整个矩阵分布在Subgroup的多个线程中，每个线程只负责加载它自己需要处理的那部分数据。这通常通过 `cooperativeMatrixLoad` 函数完成。

5.  **执行计算 (Multiply-Accumulate)**：

      * 调用核心的 `cooperativeMatrixMulAdd(A, B, C)` 函数。
      * `A`、`B` 和 `C` 都是之前加载或计算得到的协作矩阵。
      * 硬件会自动协调Subgroup中的所有线程，高效地执行矩阵乘法和加法。

6.  **存储结果 (Store)**：

      * 计算完成后，结果矩阵 `D` 同样分布在Subgroup的各个线程中。
      * 使用 `cooperativeMatrixStore` 函数将每个线程持有的结果片段写回到输出缓冲区或共享内存中。

### 示例代码片段 (GLSL)

这是一个概念性的计算着色器代码，展示了其基本用法：

```glsl
#version 460
#extension GL_KHR_cooperative_matrix : require

// 假设Subgroup大小为64，硬件支持16x16的FP16矩阵运算
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// 定义协作矩阵类型
// M, N, K 是矩阵维度，这里使用硬件支持的尺寸
const uint M = 16;
const uint N = 16;
const uint K = 16;

// 使用半精度浮点数 (FP16)
// 作用域为subgroup
// A是 M x K, B是 K x N, C和D是 M x N
using matA_t = coopmat<float16_t, gl_ScopeSubgroup, M, K>;
using matB_t = coopmat<float16_t, gl_ScopeSubgroup, K, N>;
using matC_t = coopmat<float16_t, gl_ScopeSubgroup, M, N>;

layout(binding = 0) readonly buffer MatrixA { float16_t dataA[]; };
layout(binding = 1) readonly buffer MatrixB { float16_t dataB[]; };
layout(binding = 2) writeonly buffer MatrixD { float16_t dataD[]; };

void main() {
    // 1. 声明协作矩阵变量
    matA_t A;
    matB_t B;
    matC_t C;

    // 2. 加载数据
    // 每个线程从全局内存加载自己负责的矩阵片段
    // 需要根据gl_SubgroupInvocationID等计算正确的偏移量
    // 假设矩阵是行主序存储的
    cooperativeMatrixLoad(A, dataA, /* 偏移量 */, M, false);
    cooperativeMatrixLoad(B, dataB, /* 偏移量 */, K, false);
    
    // 初始化累加矩阵C为0
    C = matC_t(float16_t(0.0));

    // 3. 执行核心计算 D = A * B + C
    matC_t D = cooperativeMatrixMulAdd(A, B, C);

    // 4. 存储结果
    // 每个线程将自己持有的结果片段写回全局内存
    cooperativeMatrixStore(D, dataD, /* 偏移量 */, M, false);
}
```

### 总结

Vulkan中的**协作矩阵**是释放现代GPU专用硬件（如Tensor Cores）强大计算能力的关键。它通过提供一个高级抽象，让着色器线程组能够高效协作，共同完成大规模的矩阵乘法累加运算。这极大地推动了在实时图形应用和通用计算中集成高性能机器学习推断和其他矩阵密集型算法的可能性。

---

### 🧑‍💻 User

支持该拓展的硬件最低要求是什么， Kompute库中能否使用这个拓展，用于加速大规模矩阵乘法计算？

---

### 🤖 Assistant

当然，我们来逐一解答您的这两个问题。

### 1. `VK_KHR_cooperative_matrix` 的最低硬件要求

`VK_KHR_cooperative_matrix` 扩展的目的是暴露GPU内部专门用于加速矩阵运算的硬件单元。因此，硬件支持是关键。以下是主要GPU厂商的硬件支持情况：

* **NVIDIA**:
    * **硬件要求**: 需要支持**张量核心（Tensor Cores）**的GPU。
    * **最低架构**: **Turing 架构**（`GeForce RTX 20系列` / `Quadro RTX` 系列）或更新的架构（Ampere, Ada Lovelace等）。
    * **简而言之**: 如果您的NVIDIA显卡是 `RTX` 系列，那么它就支持该扩展。

* **AMD**:
    * **硬件要求**: 需要支持 **Wavefront Mixed-Precision Multiply-Accumulate (WMMA)** 指令的GPU。
    * **最低架构**: **RDNA 3 架构**（`Radeon RX 7000` 系列）或更新的架构。
    * **注意**: 在Linux上，Mesa的RADV驱动从Mesa 23.3版本开始为RDNA 3及更新的GPU提供此扩展支持。

* **Intel**:
    * **硬件要求**: 支持 `DPAS` (Dot Product Accumulate Systolic) 或 `DP4A` 指令的集成/独立显卡。
    * **最低架构**:
        * 在Linux上，Mesa的ANV驱动为 **Intel Gen9**（Skylake时代）及更新的显卡提供了支持。
        * 对于性能最佳的硬件加速（使用DPAS指令），需要 **Xe-HPG** 架构（`Intel Arc` 显卡，如DG2）或更新的架构。
        * 较早的架构（如Tigerlake）会使用 `DP4A` 指令进行8位整数矩阵乘法加速，而更早的平台则会通过模拟（使用subgroup操作）来实现，性能上会有折扣。

**总结**:
要获得该扩展带来的真正硬件加速性能，您至少需要 **NVIDIA RTX 20系列**、**AMD Radeon RX 7000系列** 或 **Intel Arc** 系列的显卡。虽然一些较旧的Intel硬件在技术上可能暴露此扩展，但其性能提升将非常有限，因为它依赖于模拟或功能较弱的指令集。

### 2. Kompute库与`VK_KHR_cooperative_matrix`

关于在Kompute库中使用此扩展，情况如下：

**Kompute本身并不直接提供对 `VK_KHR_cooperative_matrix` 的高级抽象或原生操作。**

Kompute是一个旨在简化Vulkan计算工作流程的通用框架。它的核心设计理念是“自带Vulkan”（Bring-your-own-Vulkan），允许与现有的Vulkan应用无缝集成。它提供了管理资源（Tensors）、序列（Sequences）和操作（Operations）的便捷工具，但它并不限制您在着色器中可以使用的Vulkan/GLSL功能。

这意味着：

* **可以间接使用**: 您完全可以编写一个使用 `GL_KHR_cooperative_matrix` 扩展的GLSL计算着色器，然后使用Kompute来管理和调度这个着色器的执行。Kompute负责处理创建Vulkan管线、分配缓冲区（Tensors）、记录命令和同步等繁琐工作。
* **需要手动编写着色器**: 您需要自己编写利用协作矩阵的GLSL代码（如上一回答中的示例所示）。Kompute的 `OpAlgo` 或自定义操作（Custom Operations）可以用来加载和执行这个自定义的着色器代码。
* **没有内置的 `OpCooperativeMatrixMul`**: Kompute的标准操作集（如 `OpMult`）是通用的，不会自动利用 `VK_KHR_cooperative_matrix`。您需要自己实现这个逻辑。

**如何结合使用？**

1.  **编写GLSL着色器**: 创建一个 `.comp` 文件，在其中使用 `GL_KHR_cooperative_matrix` 扩展来执行您的大规模矩阵乘法。
2.  **在Kompute中加载着色器**: 您可以把编译后的SPIR-V代码加载到Kompute的 `kp::Algorithm` 中。
3.  **创建Kompute操作**: 使用 `kp::OpAlgo` 将您的输入/输出张量（`kp::Tensor`）与这个着色器算法绑定起来。
4.  **执行**: 将这个操作添加到Kompute序列（`kp::Sequence`）中并执行。

**结论**:
是的，您**可以**在Kompute工作流中利用 `VK_KHR_cooperative_matrix` 来加速大规模矩阵乘法，但这需要您**手动编写和管理使用该扩展的GLSL着色器代码**。Kompute作为框架，将极大地简化与该着色器交互的Vulkan后端工作，但它本身不提供一个“一键式”的协作矩阵操作。

---

### 🧑‍💻 User

相较于手动循环展开+矩阵分块+warp原语实现matmul kernel，使用cooperative matrix有哪些优势？

---

### 🤖 Assistant

这是一个非常棒的问题，直击了现代GPU编程模型演进的核心。将使用`cooperative_matrix`扩展与手动优化（循环展开+矩阵分块+Warp原语）进行比较，可以清晰地看出新一代图形API在易用性和性能之间所做的权衡与进步。

总的来说，**`cooperative_matrix` 的核心优势在于它以极低的编程复杂度，提供了接近甚至超越手动优化所能达到的性能，同时还具备更好的可移植性和前向兼容性。**

下面我们从几个关键维度进行详细对比：

---

### 1. 编程复杂度和生产力 (Programming Complexity & Productivity)

* **手动优化**:
    * **极其复杂**。开发者需要手动处理所有细节：
        * **矩阵分块 (Tiling)**: 计算每个线程块（Workgroup）和每个线程（Invocation/Warp）负责的数据块。
        * **共享内存管理 (Shared Memory)**: 手动将数据从全局内存加载到共享内存（LDS/SLM）以减少延迟，并精确计算访问偏移量。
        * **同步**: 在线程块内使用 `barrier()` 进行严格的同步，确保数据在计算前已完全加载。
        * **Warp级计算**: 使用Warp原语（Subgroup intrinsics）在Warp内部进行数据交换和计算，这本身就非常复杂。
        * **循环展开和寄存器管理**: 手动展开内层循环，并小心翼翼地管理寄存器使用，防止溢出到本地内存。
    * 代码量巨大，难以阅读、调试和维护。一个小小的索引错误就可能导致难以追踪的bug。

* **Cooperative Matrix**:
    * **极其简洁**。整个复杂过程被抽象为几个高级API调用：
        1.  `cooperativeMatrixLoad()`: 从内存加载数据到协作矩阵。
        2.  `cooperativeMatrixMulAdd()`: 执行核心的矩阵乘法累加。
        3.  `cooperativeMatrixStore()`: 将结果写回内存。
    * 开发者无需关心数据如何在Subgroup的线程间分布，也无需管理共享内存或进行手动同步。所有底层的复杂性都由驱动程序和硬件处理。
    * 代码更具声明性（"我要做什么"）而非指令性（"我该如何一步步做"），生产力得到极大提升。

**优势归属**: **Cooperative Matrix (压倒性优势)**

---

### 2. 性能 (Performance)

* **手动优化**:
    * **理论上限高，但实践难度大**。一个由GPU架构师或资深驱动工程师编写的、针对特定硬件（例如NVIDIA Ampere）高度优化的手动kernel，其性能*可能*是理论上的天花板。
    * **但对于绝大多数开发者而言，很难达到理想性能**。容易遇到各种性能陷阱，如共享内存的银行冲突（bank conflicts）、不恰当的分块策略导致的缓存未命中、过多的同步开销等。最终写出的代码很可能比编译器自动优化的版本还要慢。

* **Cooperative Matrix**:
    * **直接访问专用硬件**。这是最关键的一点。`cooperativeMatrixMulAdd` 函数会直接映射到GPU内部的 **MMA（Matrix Multiply-Accumulate）硬件单元**，如NVIDIA的Tensor Cores或AMD的Matrix Cores。
    * **由专家预先优化**。驱动程序的实现是由硬件厂商的工程师编写的，他们最了解如何高效地驱动自己的硬件。使用这个API，相当于免费获得了厂商级别的优化。
    * **性能稳定且非常高**。它避免了手动优化中的各种陷阱，确保了数据流能够最高效地喂给MMA单元。对于支持该扩展的硬件，其性能通常会远超开发者用通用ALU手写的Matmul Kernel。

**优势归属**: **Cooperative Matrix (对99%的开发者而言性能更高且更可靠)**

---

### 3. 可移植性和前向兼容性 (Portability & Forward Compatibility)

* **手动优化**:
    * **可移植性极差**。为一个架构（如NVIDIA Turing）优化的分块大小、Warp原语用法和同步策略，在另一个架构（如AMD RDNA 3）上可能效率低下甚至完全错误。跨厂商移植几乎等于重写。
    * **前向兼容性差**。当GPU更新换代时（例如从Ampere到Ada Lovelace），Warp大小、缓存结构、指令延迟都可能改变，之前手写的Kernel可能不再是最优的，需要重新调优。

* **Cooperative Matrix**:
    * **高度可移植**。代码本身是厂商中立的。只要目标硬件（无论是NVIDIA, AMD还是Intel）的Vulkan驱动支持 `VK_KHR_cooperative_matrix` 扩展，同一份GLSL代码就可以在上面运行。
    * **优秀的前向兼容性**。当新一代GPU推出时，开发者无需修改着色器代码。厂商会更新驱动程序，将`cooperativeMatrixMulAdd`调用映射到新的、可能更强大的MMA硬件上。您的代码将自动享受到新硬件带来的性能提升。

**优势归属**: **Cooperative Matrix (压倒性优势)**

---

### 总结与类比

| 特性 | 手动优化 (循环展开+分块+Warp原语) | Cooperative Matrix | 优势方 |
| :--- | :--- | :--- | :--- |
| **编程复杂度** | 极高，易出错，维护困难 | 极低，代码简洁，易于维护 | **Cooperative Matrix** |
| **性能** | 理论上限高，但实践中难以达到 | 稳定地接近硬件理论性能，风险低 | **Cooperative Matrix** |
| **可移植性** | 极差，与特定硬件架构深度绑定 | 优秀，跨厂商、跨代际兼容 | **Cooperative Matrix** |
| **核心机制** | 手动编排数据流，使用通用ALU计算 | 抽象化声明，直接调用专用MMA硬件 | **Cooperative Matrix** |

**一个绝佳的类比是汽车的变速箱**:

* **手动优化** 就像是 **专业赛车手使用的手动序列式变速箱**。他们通过对引擎转速、赛道状况的精确判断，进行完美的换挡，可以榨取出赛车最后0.01秒的圈速。但如果普通人来开，很可能会频繁熄火、换挡顿挫，速度远不如自动挡。
* **Cooperative Matrix** 就像是现代高性能跑车上的 **双离合自动变速箱（PDK/DCT）**。它由顶尖工程师设计，换挡速度比任何人类手动操作都要快，并且能在各种工况下都保持最佳表现。你只需要踩下油门（调用`cooperativeMatrixMulAdd`），它就能为你提供极致的加速性能。

对于GPU编程来说，`cooperative_matrix` 正是那个让广大开发者都能轻松驾驭“矩阵运算”这台性能猛兽的“双离合自动变速箱”。