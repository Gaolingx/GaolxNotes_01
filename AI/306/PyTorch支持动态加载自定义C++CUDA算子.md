# PyTorch支持动态加载自定义C++CUDA算子

### 🧑‍💻 User

pytorch可以做到不重新编译的情况下插入新的自定义c++/cuda算子源码吗？

---

### 🤖 Assistant

**是的，PyTorch 完全支持在不重新编译 PyTorch 自身，也不需要手动运行 `setup.py build` 的情况下，动态地“插入”并编译新的 C++/CUDA 算子。**

这主要通过 **JIT (Just-In-Time) 编译** 机制实现，核心工具是 `torch.utils.cpp_extension` 模块中的 `load` 和 `load_inline` 函数。

### 核心机制

当你运行脚本时：
1.  PyTorch 会在后台自动调用系统编译器（如 `g++` 或 `nvcc`）。
2.  它将你的 C++/CUDA 源码编译成一个动态链接库（`.so` 或 `.pyd`）。
3.  该库会被缓存到磁盘（默认在 `~/.cache/torch_extensions`）。
4.  **下次运行**：如果源码没有修改，PyTorch 会直接加载缓存的库，跳过编译步骤（速度很快）。如果不一致，则重新编译。

---

### 方法一：直接在 Python 字符串中写 C++/CUDA (`load_inline`)

这是最灵活的方法，适合快速原型开发。你可以把 C++ 和 CUDA 代码作为字符串直接写在 Python 文件里。

```python
import torch
from torch.utils.cpp_extension import load_inline

# 1. 定义 C++ 源码 (用于 PyTorch 绑定)
cpp_source = """
std::vector<torch::Tensor> my_add_forward(torch::Tensor input1, torch::Tensor input2);
"""

# 2. 定义 CUDA 源码 (内核实现)
cuda_source = """
__global__ void my_add_kernel(const float* __restrict__ input1,
                              const float* __restrict__ input2,
                              float* __restrict__ output,
                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input1[idx] + input2[idx];
    }
}

std::vector<torch::Tensor> my_add_forward(torch::Tensor input1, torch::Tensor input2) {
    auto output = torch::zeros_like(input1);
  
    const int threads = 256;
    const int blocks = (input1.numel() + threads - 1) / threads;
  
    my_add_kernel<<<blocks, threads>>>(
        input1.data_ptr<float>(),
        input2.data_ptr<float>(),
        output.data_ptr<float>(),
        input1.numel());
      
    return {output};
}
"""

# 3. JIT 编译并加载
# 第一次运行会花几秒钟编译，之后会直接加载缓存
my_module = load_inline(
    name='inline_cuda_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['my_add_forward'],
    verbose=True, # 设置为 True 可以看到编译过程
    with_cuda=True
)

# 4. 使用算子
x = torch.randn(10, device='cuda')
y = torch.randn(10, device='cuda')
z = my_module.my_add_forward(x, y)

print(f"Result match: {torch.allclose(z, x + y)}")
```

---

### 方法二：加载独立的源码文件 (`load`)

如果你的 C++/CUDA 代码比较长，建议放在单独的文件中（如 `.cpp` 和 `.cu`），然后通过 `load` 函数加载。

假设文件结构如下：
*   `main.py`
*   `my_op.cpp` (C++ 接口定义)
*   `my_op_kernel.cu` (CUDA 内核实现)

<details>
<summary>点击展开查看 C++/CUDA 源码文件示例</summary>

**my_op.cpp**
```cpp
#include <torch/extension.h>

torch::Tensor my_square_add(torch::Tensor input, torch::Tensor other);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("square_add", &my_square_add, "My Square Add (CUDA)");
}
```

**my_op_kernel.cu**
```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void square_add_kernel(const float* x, const float* y, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * x[idx] + y[idx];
    }
}

torch::Tensor my_square_add(torch::Tensor input, torch::Tensor other) {
    auto output = torch::zeros_like(input);
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;

    square_add_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        other.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
    return output;
}
```
</details>

**main.py 中的加载代码：**

```python
import torch
from torch.utils.cpp_extension import load

# 这里会在运行时编译 my_op.cpp 和 my_op_kernel.cu
# 并返回一个包含算子的 Python 模块
custom_op = load(
    name="custom_square_add",
    sources=["my_op.cpp", "my_op_kernel.cu"],
    verbose=True
)

# 使用
a = torch.tensor([1., 2., 3.], device="cuda")
b = torch.tensor([4., 5., 6.], device="cuda")
c = custom_op.square_add(a, b) # $1^2+4=5, 2^2+5=9...$

print(c)
```

---

### 重要提示与先决条件

1.  **环境要求**：
    *   你的机器上必须安装了与 PyTorch 编译版本兼容的 C++ 编译器（Linux 下通常是 `g++`，Windows 下是 MSVC）。
    *   如果是 CUDA 算子，必须安装 `nvcc` (CUDA Toolkit)。
    *   可以通过 `nvcc --version` 检查。

2.  **构建工具 Ninja**：
    *   强烈建议安装 `ninja` 构建系统（`pip install ninja`）。
    *   PyTorch 使用 Ninja 进行并行编译，这能显著加快 JIT 编译的速度。

3.  **缓存位置**：
    *   编译后的二进制文件通常存储在 `~/.cache/torch_extensions/` (Linux) 或 `%LOCALAPPDATA%\torch_extensions\` (Windows)。
    *   如果遇到奇怪的链接错误，删除该目录下的对应文件夹可以强制重新编译。

4.  **性能影响**：
    *   **第一次运行**：慢（因为要编译）。
    *   **第二次运行**：快（直接加载 `.so` 动态库）。
    *   **运行时性能**：与原生编译的 PyTorch 算子完全一致，没有任何 Python 解释器开销（因为核心是编译好的 C++/CUDA 二进制）。