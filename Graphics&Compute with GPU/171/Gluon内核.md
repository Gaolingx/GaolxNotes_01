# Gluon内核

## 🧑‍💻 User

pytorch gluon kernel是什么？

---
## 😀 默认助手 (7)

I'll research this for you. Let me first see what tools are available.

I found fetch and browser tools. Let me search the web to get accurate, current information about this.

The search points to Triton's Gluon. Let me fetch the official documentation for precise details.

I have authoritative details now. Let me grab the Gluon overview page for additional context on its design rationale.

## PyTorch Gluon kernel 是什么？

严格来说，"**PyTorch Gluon**" 这个名字略有歧义 —— 它指的**不是** PyTorch 官方一个叫 "Gluon" 的独立框架，而是指 **用 Gluon 语言编写的 GPU kernel**。Gluon 是 **Triton** 项目（`triton-lang`）里新推出的一个**低层级 GPU 编程语言**，而 Triton 正是 PyTorch 生态中编写自定义算子的核心编译栈（`torch.compile`、各种自定义 kernel 都基于它，Triton 现由 PyTorch 基金会/Linux 基金会维护）。

所以一句话概括：

> **Gluon kernel = 用 Gluon（Triton 的低层方言）写的高性能 GPU kernel，用来为 PyTorch 张量做自定义加速算子。**

---

### Gluon 的定位

Gluon 和 Triton **共用同一套编译器栈**，也共用同一套基于 tile 的 SPMD 编程模型和 Python DSL 前端（`@gluon.jit`）。但两者最大的区别在于**抽象层次**：

| 维度 | Triton | Gluon |
|------|--------|-------|
| 抽象层级 | 高（编译器替你决定一切） | 低（细节交给你掌控） |
|  tile 布局 (layout) | 编译器自动推导 | **用户显式指定** |
| 共享内存分配 | 编译器管理 | 用户手动管理 |
| 数据搬运 / 异步 | 编译器处理 | 用户写 async copy / TMA |
| 目标 | 易用、广泛场景自动生成高效代码 | 极致性能、手写调优 |

核心设计动机：Triton 编译器虽然对大部分 kernel 都生成得不错，但会被**手工调优的低层代码**打败。而一旦被打败，用户在 Triton 里几乎没有修改余地，因为这些细节全被隐藏了。**Gluon 把这些细节全部暴露出来**，让你可以精细控制，从而写出更快的 kernel —— 代价是需要对 GPU 硬件有更深理解。

---

### 代码长什么样

导入路径（注意是 `experimental`）：

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

一个最简单的 Gluon kernel（标量拷贝），写法和 Triton 几乎一样：

```python
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

def copy_scalar(input, output):
    grid = (1,)
    copy_scalar_kernel[grid](input, output, num_warps=1)
```

> 和 Triton 一样，**PyTorch tensor 传给 Gluon kernel 时会自动转成 global memory 指针**，grid 的指定方式也完全相同。

---

### Gluon 能控制哪些底层细节

Gluon 教程覆盖了现代 GPU 的一整套高级特性，这些在 Triton 里通常不直接暴露：

- **Tensor Layouts**：显式定义 tile 在寄存器/线程上的分布
- **Async Copy**（`cp.async`）与 **TMA**（Tensor Memory Accelerator）
- **Warp-Group MMA**（Hopper 的 `wgmma`）
- **TCGen05**：第 5 代 Tensor Core 指令（Blackwell）
- **Persistent Kernels**（持久化 kernel）
- **Warp Specialization**（warp 专门化）
- **Multi-CTA / Cluster Launch Control**、TMA gather/scatter、im2col 卷积等

---

### 什么时候该用它

<details>
<summary>展开：适用场景与注意事项</summary>

**适合：**
- 对性能极度敏感、Triton 编译器优化不到位的算子（如复杂 GEMM、Attention 变体、卷积）
- 需要精确控制数据搬运、共享内存、异步流水线的场景
- 想榨干新硬件（Hopper/Blackwell）特性时

**注意：**
- Gluon 目前仍是 **experimental**（`triton.experimental.gluon`），API 可能变动
- 学习门槛高于 Triton，需要扎实的 GPU 体系结构知识
- 主机端（launcher）代码基本不变，主要变化在设备端 kernel 写法
</details>

---

**一句话总结**：所谓 "PyTorch Gluon kernel"，就是借助 Triton 的低层语言 **Gluon** 编写的 GPU 内核 —— 它让你像写 Triton 一样用 Python，却能像写 CUDA 一样掌控布局、显存和异步搬运，从而为 PyTorch 写出接近硬件极限的高性能算子。

---
## 🧑‍💻 User

对比triton/cutedsl有什么区别？

---
## 😀 默认助手 (7)

Good question — this is a three-way comparison. Let me pull authoritative details on NVIDIA's **CuTe DSL** first, then build the comparison.

The toolset changed. Let me use the fetch tools directly.

DuckDuckGo is blocked and that doc path 404'd. Let me try the correct NVIDIA doc paths and Bing.

Let me get the CUTLASS docs index and GitHub repo directly.

I now have authoritative details from both NVIDIA's CUTLASS docs and the Triton Gluon docs. Here's the comparison.

---

# Triton vs. Gluon vs. CuTe DSL

## 一句话定位

| 语言 | 一句话 |
|------|--------|
| **Triton** | 高层的 tile 级 GPU 语言，把布局/显存/流水线**全交给编译器**，追求"多数场景自动就够快" |
| **Gluon** | Triton 的**低层方言**，同一个编译器栈，但把上面那些细节**全部暴露给用户**，用于榨干最后一点性能 |
| **CuTe DSL** | NVIDIA 在 **CUTLASS 4.x** 里推出的 Python DSL，用 Python 复刻 CuTe C++ 的 layout/tensor/atom 抽象，**NVIDIA 专属**的高性能 kernel 编写方式 |

核心关系可以用一句话概括：

> **Gluon 和 CuTe DSL 是同一场"战争"里来自两个阵营的对手** —— 它们都为了解决"Triton 管得太死、CUDA C++ 写得太累"这个中间地带。
> - Gluon = **PyTorch/Triton 阵营**的答案（Triton 的高层在底层 `truncated`，暴露细节）
> - CuTe DSL = **NVIDIA/CUTLASS 阵营**的答案（把 CUTLASS C++ 模板元编程搬进 Python）

---

## 核心对比总表

| 维度 | **Triton** | **Gluon** | **CuTe DSL (CUTLASS Python)** |
|------|-----------|-----------|-------------------------------|
| **出身 / 维护** | OpenAI 起家（Tillet），现属 PyTorch 基金会 | Triton 项目（`triton.experimental`） | **NVIDIA**，CUTLASS 4.x |
| **生态** | PyTorch / 通用 | PyTorch / Triton 生态内 | CUTLASS / NVIDIA |
| **抽象层级** | 高（编译器托管） | 低（用户托管） | 低（用户托管，CuTe 风格）|
| **布局 (layout)** | 编译器自动推导 | **用户显式指定** | **用户显式指定**（layout algebra）|
| **共享内存 / 异步** | 编译器自动管理 | 用户手写 async copy / TMA | 用户手写（cp.async / TMA / Pipeline）|
| **编程模型** | tile-based SPMD，Python DSL | 同样的 tile SPMD，DSL 同前端 | CuTe 抽象：Layout / Tensor / Atom / TiledMma |
| **编译器栈** | Triton（MLIR → LLVM → PTX）| **与 Triton 同一个栈** | **CUTLASS Python compiler** → CUDA toolchain/NVCC |
| **与 Triton 关系** | — | 同源，可无缝混用、共用前端/JIT | 独立体系，概念对应 CUTLASS C++ |
| **目标硬件** | NVIDIA **+ AMD ROCm**，跨厂商 | 主要暴露 **NVIDIA** 特性 | **仅 NVIDIA**（CUDA）|
| **语言入口** | `@triton.jit` | `@gluon.jit` | Python DSL（`cute` 命名空间）|
| **成熟度** | 成熟、生产中广泛使用 | **experimental**，API 可能变 | CUTLASS 4.x 新推出，快速演进 |
| **与 PyTorch** | 深度集成（`torch.compile`、inductor 后端）| 通过 Triton 间接集成 | 有 PyTorch 集成指南（FFI 等）|
| **典型用途** | 通用自定义算子、Attention 变体 | 手工调优到极限的 kernel | NV 上的极限 GEMM / Attention |

---

## 分维度详解

<details>
<summary><b>1. 抽象层级与控制力 —— 三者是一条光谱</b></summary>

```
  编译器全包                   细节全交给你
 ←──────────────────────────────────────────→
   Triton        Gluon          CuTe DSL / CUDA C++
   (自动布局)   (显式布局)      (显式布局 + CuTe 代数)
```

- **Triton**：`tl.load/tl.store` 处理 tile，layout、shared memory、software pipelining、异步全部由编译器决定。用户写起来极简，但**被编译器打败时无从下手**。
- **Gluon**：同样的 tile SPMD、同样的 `gl.load/gl.store`，但要你**自己选 layout、自己分配 shared memory、自己写 async copy / warp specialization**。官方原话：*"Gluon changes how device code is written, and only changes host-side code insofar as Gluon kernels may have more hyperparameters."* —— 主机端几乎不变，变化在设备端。
- **CuTe DSL**：走 CuTe 抽象路线，`Layout`、`Tensor`、`Atom`（MMA/copy 原子操作）、`TiledMma`、`TiledCopy`、`Pipeline`。这是 CUTLASS C++ `cute::` 命名空间的 Python 翻版，控制粒度最接近"手写 CUTLASS"。

</details>

<details>
<summary><b>2. 编译器栈 —— 同源 vs 异源（这是最本质的区别）</b></summary>

- **Triton 和 Gluon 共用同一套前端和 JIT 基础设施**（同一个 MLIR/LLVM 编译栈）。这意味着 Gluon 不是另起炉灶，而是 Triton 的"低层模式"，两者可以**混用**、共享 autotune、共享 `triton.testing` 等工具。
- **CuTe DSL 是独立的一套**：CUTLASS Python compiler stack，最终经 **NVIDIA CUDA toolchain（NVCC / CUBIN）** 生成 CUDA 代码。它属于 CUTLASS 4.x 体系，与 Triton 无关。

> 结果：Gluon 跟 Triton 是"父子/兄弟"关系；CuTe DSL 跟 Triton 是"竞品/对手"关系。

</details>

<details>
<summary><b>3. 硬件可移植性</b></summary>

- **Triton**：**跨厂商**，NVIDIA 之外还有 AMD ROCm 后端（以及社区的 Intel 等尝试）。这是它被大量框架采用的重要原因。
- **Gluon**：本身在 Triton 里，但教程和 API 主要针对 **NVIDIA 硬件特性**（TMA、wgmma、tcgen05 等）暴露细节。
- **CuTe DSL**：**纯 NVIDIA**，且按架构分层覆盖：
  - Ampere / Ada → warp-level MMA
  - Hopper → warpgroup MMA（wgmma）+ TMA
  - Blackwell → `tcgen05` MMA + TMEM

如果你要跑 AMD，基本只能在 Triton 路线里选；如果你只吃 NVIDIA 且要极限性能，三者都能用，但 CuTe DSL 是最"原生"的。

</details>

<details>
<summary><b>4. 成熟度与稳定性</b></summary>

- **Triton**：最成熟，生产环境（`torch.compile`、vLLM、SGLang 等）大量使用。
- **Gluon**：明确的 **experimental**（导入路径就是 `triton.experimental.gluon`），API 可能变动。
- **CuTe DSL**：CUTLASS 4.x 的新主力方向，NVIDIA 明确表示它**不是 CUTLASS C++ 的替代品**，而是"高生产率的 kernel 编写框架"，且"生成的 kernel 目标是与 CUTLASS C++ 持平性能，但可能存在差距"。仍在积极演进。

</details>

---

## 怎么选（决策指南）

| 你的情况 | 推荐 |
|----------|------|
| 写通用算子、要跨 NVIDIA/AMD、要快但不必极致 | **Triton** |
| 用 Triton 写完了，但性能卡在编译器手里，要精细调 layout/流水线 | **Gluon** |
| 只做 NVIDIA、要 CUTLASS 级极限 GEMM/Attention、想摆脱 C++ 模板地狱 | **CuTe DSL** |
| 需要 CUTLASS 库/Profiler 的完整功能 | CUTLASS C++（CuTe DSL 目前不含完整 profiler/库接口）|
| 要 pyTorch 里即插即用的自定义算子 | Triton（生态最顺）或 Gluon |

---

## 心智模型（最好记的版本）

> 把 GPU kernel 编程想成一个"控制权旋钮"：
>
> - **Triton** 把旋钮拧到"自动挡"——省心，但遇到瓶颈你无能为力；
> - **Gluon** 是同一条 Triton 流水线上的"手动挡"——同一台车（同一编译器），你亲自换挡；
> - **CuTe DSL** 是 NVIDIA 自己造的一台"手动挡赛车"——用的是 CuTe/CUTLASS 那套零件，只在 NVIDIA 赛道上跑，但最贴硬件。

<details>
<summary><b>补充：三者的导入 / 入口对照（代码）</b></summary>

```python
# Triton
import triton, triton.language as tl
@triton.jit
def kernel(...): ...

# Gluon（Triton 的低层方言，注意是 experimental）
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
@gluon.jit
def kernel(...): ...

# CuTe DSL（CUTLASS 4.x，NVIDIA）
# 使用 cute 的 Layout / Tensor / TiledMma / TiledCopy / Pipeline 等抽象，
# 经 CUTLASS Python compiler → CUDA toolchain JIT 成 CUDA 代码
```

</details>

---

**一句话总结**：**Triton** 是"自动挡通用车"，**Gluon** 是同一辆车的"手动挡版"（属 Triton/PyTorch 阵营），**CuTe DSL** 是 NVIDIA 用 Python 重写的 CUTLASS 手动挡赛车（NVIDIA 专属）。后两者都在抢"比 Triton 更可控、比 CUDA 更好写"这同一个位置。
