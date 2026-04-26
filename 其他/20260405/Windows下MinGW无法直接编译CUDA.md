# Windows下MinGW无法直接编译CUDA

### 🧑‍💻 User

Windows上用mingw怎么调用cuda toolkit的nvcc编译.cu文件？

---

### 🤖 Assistant

在Windows系统上，**官方明确规定 `nvcc` 不支持将 MinGW (GCC) 作为宿主编译器 (Host Compiler)**。

在Windows环境下，`nvcc` 强依赖于 Microsoft Visual C++ (MSVC，即 `cl.exe`) 来编译主机端的 C/C++ 代码。如果你尝试使用 `nvcc -ccbin g++` 强制指定 MinGW，编译会由于 CUDA 头文件中大量使用了 MSVC 专属的宏（如 `_MSC_VER`）和内联汇编而报错失败。

虽然不能直接用 `nvcc` + MinGW 编译同一个文件，但你有以下几种切实可行的替代方案：

<details>
<summary><b>方案一：混合编译（最正规的 Windows 做法）</b></summary>

你可以将 CUDA 代码（`.cu`）和普通的 C/C++ 代码（`.cpp`）分离开来。用 `nvcc`（底层调用 MSVC）编译 CUDA 部分并生成动态链接库（DLL）或静态库，然后用 MinGW 编译你的主程序并链接该库。

**注意：** 为了避免 MSVC 和 MinGW 之间的 C++ ABI 不兼容问题（比如 `std::vector` 或 `std::string` 在两者间结构不同），**两者之间的接口必须使用纯 C 接口 (`extern "C"`)。**

**步骤：**

1. **编写 CUDA 文件 (`kernel.cu`)**
   提供一个纯 C 接口：
   ```cuda
   // kernel.cu
   #include <cuda_runtime.h>
   #include <stdio.h>

   __global__ void addKernel(int *c, const int *a, const int *b) {
       int i = threadIdx.x;
       c[i] = a[i] + b[i];
   }

   // 使用 extern "C" 导出纯 C 接口，供 MinGW 调用
   extern "C" __declspec(dllexport) void launch_add(int *c, const int *a, const int *b, int size) {
       int *dev_a = 0, *dev_b = 0, *dev_c = 0;
       cudaMalloc((void**)&dev_c, size * sizeof(int));
       cudaMalloc((void**)&dev_a, size * sizeof(int));
       cudaMalloc((void**)&dev_b, size * sizeof(int));

       cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
       cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

       addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

       cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

       cudaFree(dev_c);
       cudaFree(dev_a);
       cudaFree(dev_b);
   }
   ```

2. **使用 nvcc (MSVC) 编译为 DLL**
   在已安装 Visual Studio 的环境下，打开 x64 Native Tools Command Prompt，或者确保环境变量里有 `cl.exe`：
   ```bash
   nvcc -shared -o cudamodule.dll kernel.cu
   ```
   这会生成 `cudamodule.dll` 和对应的导入库 `cudamodule.lib`。

3. **编写主程序 (`main.cpp`)**
   ```cpp
   // main.cpp
   #include <iostream>

   // 声明 DLL 中的函数
   extern "C" void launch_add(int *c, const int *a, const int *b, int size);

   int main() {
       const int arraySize = 5;
       int a[arraySize] = { 1, 2, 3, 4, 5 };
       int b[arraySize] = { 10, 20, 30, 40, 50 };
       int c[arraySize] = { 0 };

       launch_add(c, a, b, arraySize);

       for (int i = 0; i < arraySize; i++) {
           std::cout << c[i] << " ";
       }
       std::cout << std::endl;
       return 0;
   }
   ```

4. **使用 MinGW 编译并链接**
   打开你的 MinGW 终端：
   ```bash
   g++ main.cpp -L. -lcudamodule -o main.exe
   ```
   *(注：MinGW 可以直接链接 `.dll` 或 `.lib` 文件)*
</details>

<details>
<summary><b>方案二：使用 Clang 替代 nvcc 编译 CUDA (全 MinGW 工具链)</b></summary>

LLVM/Clang 原生支持编译 CUDA 代码，并且 Clang 在 Windows 上可以完美配合 MinGW 环境工作。这样你就可以完全抛弃 `nvcc` 和 MSVC。

**步骤：**

1. 在 MSYS2/MinGW 环境中安装 Clang：
   ```bash
   pacman -S mingw-w64-x86_64-clang
   ```
2. 确保系统已安装 NVIDIA CUDA Toolkit，且环境变量配置正确。
3. 使用 `clang++` 直接编译 `.cu` 文件。你需要告诉 Clang 这是一个 CUDA 文件，并指定你的显卡架构（例如 RTX 30 系列是 `sm_86`）：
   ```bash
   clang++ -x cuda kernel.cu main.cpp -o app.exe --cuda-gpu-arch=sm_86 -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64" -lcudart
   ```
   *注意：具体的 CUDA 路径和 `sm_XX` 版本请根据你的实际环境替换。*
</details>

<details>
<summary><b>方案三：使用 WSL2 (推荐的终极方案)</b></summary>

如果你只是习惯使用 GCC/MinGW 的命令行体验，强烈建议放弃在 Windows 宿主机上折腾 CUDA + MinGW，转而使用 **WSL2 (Windows Subsystem for Linux)**。

在 WSL2 Ubuntu 环境下：
1. `nvcc` 官方完美支持 `gcc` 作为宿主编译器。
2. WSL2 支持 GPU 直通 (GPU Pass-through)，你在 Linux 子系统里跑出来的 CUDA 性能与 Windows 宿主机几乎没有区别。

**步骤：**
1. 安装 WSL2 和 Ubuntu。
2. 在 Ubuntu 中安装 Linux 版的 CUDA Toolkit：`sudo apt install nvidia-cuda-toolkit`
3. 直接使用经典的 Linux 编译命令：
   ```bash
   nvcc main.cu -o main
   ./main
   ```
</details>

### 总结
如果你必须在 Windows 原生环境下工作，请使用 **方案一**（界面分离）或 **方案二**（改用 Clang）。如果你只是想要 GCC 的编译体验，请使用 **方案三**（WSL2）。**不要尝试使用 `nvcc -ccbin gcc`**，在 Windows 下这是行不通的。