# PyTorch调用手写CUDA内核方法

### 🧑‍💻 User

pytorch中如何调用手写的cuda kernel？

---

### 🤖 Assistant

# 在PyTorch中调用手写的CUDA内核

在PyTorch中调用手写的CUDA内核通常有几种方法，下面我将介绍最常用的几种方式：

## 方法1：使用PyTorch的CUDA扩展API

PyTorch提供了`torch.utils.cpp_extension`模块来简化CUDA内核的集成：

```python
import torch
from torch.utils.cpp_extension import load

# 直接加载CUDA扩展
cuda_module = load(
    name='custom_kernel',
    sources=['kernel.cu'],
    extra_cflags=['-O2'],
    verbose=True)

# 调用内核
def custom_function(input_tensor):
    return cuda_module.custom_operation(input_tensor)
```

对应的`kernel.cu`文件示例：

```cpp
#include <torch/extension.h>

__global__ void custom_kernel(const float* input, float* output, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;  // 示例操作
    }
}

torch::Tensor custom_operation(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_operation", &custom_operation, "Custom CUDA operation");
}
```

## 方法2：使用NVRTC即时编译

对于更动态的场景，可以使用NVRTC进行运行时编译：

```python
import torch
from torch.utils.cpp_extension import load_inline

# CUDA内核代码字符串
cuda_src = '''
__global__ void custom_kernel(const float* input, float* output, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor custom_operation(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    custom_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel());
    
    return output;
}
'''

# 即时编译
module = load_inline(
    name='inline_kernel',
    cpp_sources='',
    cuda_sources=cuda_src,
    functions=['custom_operation'])

# 调用
output = module.custom_operation(input_tensor)
```

## 方法3：使用pybind11和setuptools构建扩展

对于更复杂的项目，可以创建完整的扩展模块：

1. 创建`setup.py`:

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_cuda',
    ext_modules=[
        CUDAExtension('custom_cuda', [
            'custom_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

2. 安装扩展：
```bash
python setup.py install
```

3. 在Python中使用：
```python
import torch
import custom_cuda

output = custom_cuda.custom_operation(input_tensor)
```

## 注意事项

1. **内存管理**：PyTorch张量已经处理了内存管理，通常不需要手动分配/释放内存
2. **同步/异步**：CUDA内核调用默认是异步的，必要时使用`cudaDeviceSynchronize()`
3. **错误检查**：内核启动后检查错误`cudaError_t err = cudaGetLastError()`
4. **类型安全**：确保张量类型与内核期望的类型匹配
5. **性能**：优化块大小和网格大小以获得最佳性能

## 示例：向量加法

完整示例 - 向量加法：

`vector_add.cu`:
```cpp
#include <torch/extension.h>

__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    auto c = torch::zeros_like(a);
    
    int threads = 256;
    int blocks = (a.numel() + threads - 1) / threads;
    
    vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         c.data_ptr<float>(),
                                         a.numel());
    
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector add with CUDA");
}
```

Python端：
```python
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name='vector_add', sources=['vector_add.cu'])

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = cuda_module.vector_add(a, b)
```

以上方法都可以在PyTorch中成功集成和调用手写的CUDA内核。选择哪种方法取决于项目的复杂性和需求。