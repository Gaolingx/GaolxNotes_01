# 静态编译可摆脱cudart依赖

### 🧑‍💻 User

静态编译可以实现不依赖cudart cudnn dll吗？

---

### 🤖 Assistant

简短的回答是：**对于 `cudart` 完全可以，对于 `cudnn` 理论上可以但实际操作非常困难且不推荐，而对于底层的 CUDA 驱动 (`nvcuda.dll`) 则绝对不可能。**

下面为您详细拆解各项依赖的静态编译可行性及实现方法。

### 1. CUDA Runtime (`cudart`)：完全可以静态编译

NVIDIA 官方原生支持将 CUDA Runtime 静态链接到您的程序中。这样您的程序在目标机器上运行时，就不再需要 `cudart64_xxx.dll`。

<details>
<summary><b>点击展开：如何静态链接 cudart</b></summary>

*   **使用 nvcc 编译：**
    在编译时加上 `-cudart static` 参数即可（实际上，较新版本的 nvcc 默认就是静态链接 cudart 的）。
    ```bash
    nvcc my_program.cu -o my_program.exe -cudart static
    ```
*   **在 CMake 中配置：**
    如果是使用 CMake 的 `FindCUDAToolkit`，可以链接到静态目标：
    ```cmake
    target_link_libraries(MyTarget PRIVATE CUDA::cudart_static)
    ```
*   **在 Visual Studio 中：**
    确保链接器输入中包含的是 `cudart_static.lib`，而不是 `cudart.lib`。

</details>

### 2. cuDNN：理论上可行，但极度不推荐

虽然 NVIDIA 提供了 cuDNN 的静态库（如 `cudnn_static.lib` 或 `libcudnn_static.a`），但将其静态编译到程序中会面临巨大的挑战：

*   **依赖地狱：** cuDNN 依赖于 cuBLAS。如果您要静态链接 cuDNN，就必须同时静态链接 `cublas_static.lib`。而静态链接 NVIDIA 的数学库还需要额外链接辅助库，如 `culibos.lib`（用于处理多线程等操作系统调用）。
*   **体积庞大：** cuDNN 和 cuBLAS 的静态库非常巨大。如果全部静态链接，您的可执行文件（`.exe`）体积可能会膨胀到几百 MB 甚至上 GB。
*   **JIT 编译和动态特性丢失：** 现代的 cuDNN 使用了大量的动态启发式算法和 JIT (Just-In-Time) 编译内核。某些最优的卷积算法可能依赖于运行时的 NVRTC (NV Runtime Compilation) 动态生成。静态链接可能会导致这部分功能失效，或者强迫您必须附带其他的 DLL（如 `nvrtc.dll`）。

<details>
<summary><b>点击展开：如果硬要静态链接 cuDNN 的大致方法</b></summary>

您需要在链接器中指定所有的静态依赖项。以 Windows/CMake 为例，大概需要链接以下库（顺序很重要）：
```cmake
target_link_libraries(MyTarget PRIVATE 
    cudnn_static.lib
    cublas_static.lib
    cublasLt_static.lib
    culibos.lib # 必须，NVIDIA 的 OS 抽象层
    cudart_static.lib
)
```
*注意：即便如此，在某些使用了特定算法（如 RNN 或较新的 Transformer 算子）的代码中，仍可能在运行时报错找不到特定的动态库。*

</details>

### 3. 绝对无法摆脱的依赖：CUDA Driver (`nvcuda.dll`)

无论您如何静态编译，任何 CUDA 程序最终都需要和显卡硬件通信。这层通信是通过 **CUDA Driver API** 实现的。
*   在 Windows 上表现为 `nvcuda.dll`。
*   在 Linux 上表现为 `libcuda.so`。

**这个文件绝对不能静态链接。** 它是随着用户的**显卡驱动程序**一起安装在操作系统中的，因为它必须与显卡的底层硬件和系统内核态驱动严格匹配。

### 总结与最佳实践

如果您是为了方便分发软件，避免用户安装庞大的 CUDA Toolkit，业界的**标准做法（最佳实践）**是：

1.  **静态链接 `cudart`**（使用 `cudart_static.lib`），消除对 CUDA 运行时的依赖。
2.  **动态链接 `cudnn` 和 `cublas`**。
3.  **打包分发 DLL**：将编译好的 `.exe` 文件与它所需的 `cudnn64_x.dll`、`cublas64_x.dll` 等文件放在同一个文件夹中直接打包发给用户。

只要用户的电脑上安装了足够新版本的**NVIDIA 显卡驱动**（提供 `nvcuda.dll`），您的程序解压后就可以直接运行，无需用户配置任何 CUDA 环境变量或安装 Toolkit。