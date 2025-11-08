# PyTorch调用CUDA算子实现指南

### 🧑‍💻 User

如何用torch.utils.cpp_extension实现pytorch里面调用cuda算子？

---

### 🤖 Assistant

# 使用torch.utils.cpp_extension实现PyTorch中调用CUDA算子

`torch.utils.cpp_extension` 是PyTorch提供的一个强大工具，允许用户编写C++/CUDA代码并将其编译为Python可调用的模块。下面我将详细介绍如何使用它来自定义CUDA算子。

## 基本步骤

1. 编写CUDA内核代码
2. 编写C++包装函数
3. 编写Python绑定代码
4. 使用JIT编译或setup.py方式编译

## 示例：实现一个简单的加法算子

### 1. 编写CUDA内核代码

首先创建一个名为`add_kernel.cu`的文件：

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> b,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> out) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.size(0)) {
    out[idx] = a[idx] + b[idx];
  }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
  // 检查输入张量
  TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

  // 创建输出张量
  auto out = torch::empty_like(a);

  // 设置CUDA线程网格
  const int64_t num_elements = a.numel();
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  // 根据数据类型调用不同的内核
  AT_DISPATCH_ALL_TYPES(a.type(), "add_cuda", ([&] {
    add_cuda_kernel<scalar_t><<<num_blocks, block_size>>>(
        a.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        b.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
  }));

  return out;
}
```

### 2. 编写C++包装函数

创建一个名为`add.cpp`的文件：

```cpp
#include <torch/extension.h>

// 声明CUDA函数
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

// CPU版本的实现（可选）
torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
  return a + b;
}

// 分发函数
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  // 根据张量设备类型选择实现
  if (a.device().is_cuda()) {
    return add_cuda(a.contiguous(), b.contiguous());
  } else {
    return add_cpu(a, b);
  }
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two tensors (CUDA)");
}
```

### 3. 使用JIT编译方式

创建一个Python脚本`test_add.py`：

```python
import torch
from torch.utils.cpp_extension import load

# 使用JIT编译
cuda_add_module = load(
    name="cuda_add",
    sources=["add.cpp", "add_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=["-O2", "--use_fast_math"],
    extra_cflags=["-O2"]
)

# 测试函数
if __name__ == "__main__":
    # 使用CPU
    a_cpu = torch.randn(10)
    b_cpu = torch.randn(10)
    result_cpu = cuda_add_module.add(a_cpu, b_cpu)
    print("CPU result:", result_cpu)
    print("CPU check:", torch.allclose(result_cpu, a_cpu + b_cpu))
  
    # 使用CUDA
    if torch.cuda.is_available():
        a_cuda = a_cpu.cuda()
        b_cuda = b_cpu.cuda()
        result_cuda = cuda_add_module.add(a_cuda, b_cuda)
        print("CUDA result:", result_cuda)
        print("CUDA check:", torch.allclose(result_cuda, a_cuda + b_cuda))
```

### 4. 使用setup.py方式（更正式的项目）

创建`setup.py`文件：

```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="cuda_add",
    ext_modules=[
        CUDAExtension(
            name="cuda_add",
            sources=["add.cpp", "add_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
```

然后通过以下命令安装：

```
python setup.py install
```

## 高级技巧

### 1. 自定义编译选项

```python
torch.utils.cpp_extension.load(
    name="custom_extension",
    sources=["source1.cpp", "source2.cu"],
    extra_cuda_cflags=["-arch=sm_70", "-O3", "--use_fast_math"],
    extra_cflags=["-O3", "-march=native"],
    extra_ldflags=["-L/usr/local/cuda/lib64"],
    build_directory="./build",  # 自定义构建目录
    verbose=True  # 打印编译信息
)
```

### 2. 条件编译

可以检查是否为Windows或CUDA可用等条件：

```cpp
#ifdef _WIN32
#  define API extern "C" __declspec(dllexport)
#else
#  define API extern "C"
#endif
```

### 3. 调试技巧

在CUDA代码中使用：

```cuda
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename scalar_t>
__global__ void kernel(...) {
  CUDA_KERNEL_LOOP(i, n) {
    // 内核代码
  }
}
```

### 4. 性能优化技巧

- 使用向量化加载/存储
- 合并全局内存访问
- 避免线程分歧
- 使用共享内存减少全局内存访问
- 使用常量内存存储只读数据

## 常见问题与解决方案

1. **编译错误：找不到头文件**
   - 确保正确设置了`CXX`和`CUDA_HOME`环境变量
   - 使用`torch.utils.cpp_extension.CUDAExtension`而不是普通的Extension

2. **运行时错误：无效的设备函数**
   - 检查CUDA架构是否匹配（如sm_70）
   - 确保设备支持的CUDA版本与代码兼容

3. **性能不佳**
   - 使用`nvprof`或`nsight`分析性能瓶颈
   - 优化内存访问模式
   - 检查是否充分利用了GPU并行性

通过这种方式，你可以将高性能的CUDA代码无缝集成到PyTorch中，充分利用GPU的计算能力。