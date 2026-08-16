# PyTorch扩展通过动态注册实现算子拓展

### 🧑‍💻 User

pytorch extension是如何拓展torch算子而不修改源码的（这在vllm/sglang/trt等框架很常见）？

---

### 🤖 Assistant

**PyTorch Extension 通过动态库加载 + 算子注册（Dispatcher / TORCH_LIBRARY）机制扩展算子，无需修改 PyTorch 源码。** 这正是 vLLM、SGLang、TensorRT-LLM 等框架广泛使用的方式。

### 核心机制

PyTorch 的算子系统是可扩展的：

1. **Dispatcher（调度器）**：所有算子调用都经过统一的 Dispatcher。它根据算子 schema、设备（CPU/CUDA）、布局等路由到具体实现。
2. **算子注册表**：全局注册表允许在运行时注册新算子或为已有算子添加新后端实现。
3. **动态共享库（.so）**：C++/CUDA 代码编译成 `.so`，加载时执行静态初始化代码（`TORCH_LIBRARY` 等宏），把算子注册进 PyTorch 的全局表。
4. **Python 侧暴露**：通过 `torch.ops.xxx` 或 `pybind11` 绑定调用。

因此，只需：
- 写内核 + 注册代码
- 编译成扩展
- `import` 或 `torch.ops.load_library` 加载

PyTorch 核心代码完全不用改。

### 常用实现路径

#### 1. 现代推荐方式：`TORCH_LIBRARY` + `cpp_extension`（vLLM 等常用）

**C++ / CUDA 侧示例**（`my_ops.cpp` + CUDA 内核）：

```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>

// 简单 CUDA 内核声明（实际实现放 .cu）
at::Tensor my_add_cuda(const at::Tensor& a, const at::Tensor& b);

// 定义 schema（算子签名）
TORCH_LIBRARY(myops, m) {
  m.def("my_add(Tensor a, Tensor b) -> Tensor");
}

// 为 CUDA 后端注册实现
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("my_add", &my_add_cuda);
}

// 也可以注册 CPU 实现
// TORCH_LIBRARY_IMPL(myops, CPU, m) { ... }
```

**Python 构建与加载**（`setup.py` 或 JIT）：

```python
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, load
from setuptools import setup

# 方式 A：setuptools 正式构建（生产常用）
setup(
    name="myops",
    ext_modules=[
        CUDAExtension(
            name="myops._C",          # 生成 myops/_C*.so
            sources=["my_ops.cpp", "my_ops_cuda.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# 方式 B：JIT 加载（开发方便）
# myops = load(name="myops", sources=["my_ops.cpp", "my_ops_cuda.cu"], verbose=True)
```

**使用**：

```python
import torch
import myops._C   # 或 torch.ops.load_library("path/to/myops.so")

a = torch.randn(10, device="cuda")
b = torch.randn(10, device="cuda")
c = torch.ops.myops.my_add(a, b)   # 直接调用自定义算子
```

加载 `.so` 时，`TORCH_LIBRARY` 宏里的静态代码自动执行注册，Dispatcher 就能找到实现。

#### 2. 纯 Python 自定义算子（PyTorch 2.x+，适合较简单逻辑）

```python
import torch
from torch import Tensor
from torch.library import custom_op, triton_op  # 或 register_fake 等

@custom_op("myops::my_add", mutates_args=())
def my_add(a: Tensor, b: Tensor) -> Tensor:
    return a + b   # 或调用你的 CUDA 内核

# 提供 meta / fake tensor 实现（torch.compile / export 需要）
@my_add.register_fake
def _(a, b):
    return torch.empty_like(a)
```

然后同样通过 `torch.ops.myops.my_add` 调用。复杂高性能内核仍建议 C++/CUDA。

#### 3. 传统 pybind11 模块方式（很多老代码 / 简单函数）

```cpp
#include <torch/extension.h>

at::Tensor my_add(at::Tensor a, at::Tensor b) { ... }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("my_add", &my_add);
}
```

Python 侧变成普通函数调用，而不是 `torch.ops`。功能弱一些（对 TorchScript、Dispatcher、compose 支持较差），但仍广泛使用。

### 在 vLLM / SGLang / TensorRT 等框架中的典型用法

- **vLLM**：大量自定义 CUDA 内核（PagedAttention、FlashAttention 变体、量化 GEMM、RoPE、RMSNorm 融合等）。通过 `CUDAExtension` 编译，注册成 `torch.ops.vllm.*` 或内部 C 扩展，再包一层 Python 模块。加载时自动注册，无需改 PyTorch。
- **SGLang**：类似，自定义 attention、radix attention、融合算子等走 extension。
- **TensorRT-LLM / TensorRT**：更多是 TensorRT Plugin + PyTorch 绑定，或通过类似扩展把 TRT engine 包装成可调用算子；原理同属“动态注册 + 外部库”。
- 共同点：高性能内核写在独立 `.cu` / 库里 → 编译成扩展 → 运行时注册进 PyTorch Dispatcher → 上层 Python 代码像调用原生算子一样使用，支持 autograd（需额外注册 backward）、CUDA Graph、部分 `torch.compile` 支持。

### 完整工作流小结

| 步骤 | 内容 | 工具 |
|------|------|------|
| 1. 写内核 | CUDA kernel + 包装函数 | `.cu` / `.cpp` |
| 2. 注册算子 | `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL` | 定义 schema 与实现 |
| 3. 构建 | `CUDAExtension` / `CppExtension` | `setup.py` 或 `load()` |
| 4. 加载 | `import xxx._C` 或 `torch.ops.load_library` | 触发静态注册 |
| 5. 调用 | `torch.ops.namespace.op(...)` | 走 Dispatcher |
| 6. （可选） | 注册 backward、meta、vmap 等 | 支持 autograd / compile |

### 关键注意事项

- **Autograd**：简单情况用 `torch.autograd.Function` 包装；完整支持需在 C++ 注册 `Autograd` 后端或用 `torch.library` 的 Python 装饰器。
- **torch.compile / Inductor**：需要提供 `register_fake`（meta 实现）和正确的 schema，否则可能 fallback 或失败。
- **多后端**：同一 schema 可为 CPU、CUDA、XPU、Meta 等分别 `TORCH_LIBRARY_IMPL`。
- **覆盖已有算子**：可以给现有 namespace（如 `aten`）注册新实现（需谨慎，版本兼容性风险）。
- **构建依赖**：需要匹配的 CUDA toolkit、PyTorch 头文件、C++ ABI 一致。生产环境常用 `torch.utils.cpp_extension` + ninja。
- **调试**：`TORCH_SHOW_DISPATCH_TRACE=1` 可看 Dispatcher 路由；`nm -D xxx.so` 检查符号。

### 为什么这套设计强大

- **零源码修改**：PyTorch 只提供注册钩子和 Dispatcher。
- **高性能**：直接写 CUDA，绕过 Python 开销，还能做 kernel fusion。
- **可组合**：自定义算子可被 `torch.compile`、TorchScript、导出、分布式等系统部分支持。
- **隔离性**：框架（vLLM 等）把定制内核放在自己的 extension 里，升级 PyTorch 通常只需重新编译扩展。

如果你需要针对某个具体场景（例如写一个带 backward 的 fused attention 扩展、如何对接 vLLM 的自定义 op 风格、或 `torch.compile` 兼容写法）的完整可运行示例，可以告诉我更多细节，我可以给出对应模板代码。