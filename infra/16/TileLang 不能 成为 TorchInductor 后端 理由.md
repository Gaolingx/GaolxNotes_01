# TileLang 不能 成为 TorchInductor 后端 理由

## 🧑‍💻 User

tilelang为什么不能作为torchinductor后端

---
## 😀 默认助手 (7)

# TileLang 为什么不能当 TorchInductor 的后端

**一句话结论**：不是"理论上做不到"，而是 **Inductor 的 codegen 后端插槽是按 device type 硬注册的固定枚举、CUDA 那个槽已经被 Triton 占死**，加上 **TileLang 的抽象粒度（tile 级 `@T.prim_func`）和 Inductor 输出物（逐元素 loop nest）不在同一层**，以及编译/缓存/autotune 语义冲突、收益重叠、上游维护成本高。但它**确实已经被做成了一个 Inductor 后端——在昇腾 NPU 上**。

下面把"插槽层 / IR 与抽象层 / 编译模型层 / 生态层"拆开讲，最后给出实际可用的三条路径。

---

## 0. 先把"后端"这个词拆清楚

"Inductor 后端"有两个完全不同的含义，混在一起就会得到错误结论：

| 含义 | 机制 | TileLang 能不能用 |
|---|---|---|
| `torch.compile(backend=...)` 的**编译器后端** | `torch._dynamo.backends.registry.register_backend`，FX Graph → callable | 可以。你可以写个 backend 包一层或直接不用 Inductor |
| Inductor **内部的 codegen 后端**（Triton 那个位置） | `torch._inductor.codegen`，`device_codegens[device_type]` | CUDA 上不行（理由见下），NPU 上已经有人在用 |

用户真正想问的、也是"不能"的部分，是**第二种**。

---

## 1. 插槽层：CUDA 的位置是"硬编码枚举"，没有第三方 DSL 的入口

这是最直接、也最容易被忽略的原因。看 PyTorch main 的实际代码：

`torch/_inductor/codegen/common.py`：

```python
class DeviceCodegen:
    scheduling: SchedulingConstructor
    wrapper_codegen: WrapperConstructor
    cpp_wrapper_codegen: WrapperConstructor | None = None
    fx_wrapper_codegen: WrapperConstructor | None = None

device_codegens: dict[str, DeviceCodegen] = {}   # 按 device type 索引

def register_backend_for_device(device, device_scheduling, device_wrapper_codegen,
                               device_cpp_wrapper_codegen=None,
                               device_fx_wrapper_codegen=None, ...):
    device_codegens[device] = DeviceCodegen(...)
```

`_init_builtin_backend_registration()` 里的 CUDA 注册：

```python
if get_scheduling_for_device("cuda") is None:
    cuda_backends = {
        "triton": CUDACombinedScheduling,   # 默认
        "halide": HalideScheduling,         # 实验
        "pallas": PallasScheduling,         # 实验
    }
    register_backend_for_device(
        "cuda",
        lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
        PythonWrapperCodegen, CppWrapperGpu, WrapperFxCodegen,
    )
```

`torch/_inductor/config.py`：

```python
cuda_backend: Literal["triton", "halide", "pallas"] = "triton"
cpu_backend:  Literal["cpp", "triton", "halide", "pallas"] = "cpp"
```

关键点有三条：

1. **粒度是 device type，不是"DSL 名字"**。一个设备类型只有一个 `DeviceCodegen`，选择余地只有枚举里那几个内置类。想要 `cuda_backend="tilelang"`，必须**给上游打补丁**改 `Literal` 和注册表，不能像注册 custom op 那样从外部挂进去。
2. **注册是进程级全局、只跑一次**（`if get_scheduling_for_device("cuda") is None`）。同一个进程里没法按用户偏好对某次编译换一套 codegen。
3. **对外真正开放的扩展点是给"非内置 device type"的**：`_init_builtin_backend_registration()` 之后有一段 PrivateUse1 探测 —— 如果 `torch.<backend>._inductor_backend_init` 存在，就调这个 vendor hook，由厂商自己 `register_backend_for_device`。**昇腾正是走的这条路**（见第 4 节）。

> 换句话说：CUDA 是"别人的地盘"，NPU/自定义加速器是"空地"。

---

## 2. IR / 抽象层：TileLang 的入口和 Inductor 的出口不是同一层

一个 codegen 后端要实现的契约是这样（以 CUDA 为例）：

```python
class TritonScheduling(SIMDScheduling):
    kernel_type = TritonKernel
    backend_features = OrderedSet([BackendFeature.FOREACH, BackendFeature.SORT,
                                   BackendFeature.SCAN, BackendFeature.TUPLE_REDUCTION,
                                   BackendFeature.TRITON_TEMPLATES, ...])
```

它要求你：

- 接住 `SchedulerNode / FusedSchedulerNode / Loops` 这种**循环级** IR；
- 逐句生成"每个元素"的代码：索引算术（`ModularIndexing`/`FloorDiv`）、mask/bound 判断、对齐与向量化、`tl.load`/`tl.store`、归约（Welford）、scatter/atomic、随机数、CSE、`dtype` 规则……
- 用 `BackendFeature` **显式声明自己支持哪些能力**，不支持的由 Inductor 回退（fallback 到 ATen eager）。

而 TileLang 的入口是 `@T.prim_func` 这种 **tile 级**程序：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
    T.copy(A[by*block_M, k*block_K], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

`T.copy` / `T.gemm` / `T.Pipelined` / layout inference / warp specialization，全都建立在**"你先知道 tile 形状、缓冲区整块存在"**这个前提上。它是"设计一个 kernel"，不是"把一个 loop nest 逐句翻译成代码"。

所以要接上，必须**先把 Inductor IR 重新规划成 tile 程序**：捕获 IR → 语义化 operation graph → kernel plan / region plan → 再 emit `@T.prim_func`。这不是胶水活，而是一整套新编译器前端。昇腾那个 PR 的做法就是这句话的直译：

> "增加 `TileLangKernel` 与 `TileLangScheduling`，复用 NPU Triton 的 range tree、轴分析、tiling 和 scheduler 基础能力。将 Inductor IR 捕获结果整理为语义化 operation graph，再构建 kernel plan 与 region plan，最后由独立 emitter 生成 TileLang `@T.prim_func`。" —— Ascend/pytorch PR #40597

另外还有一个"所有权"问题：TileLang 是**从 host stub 到 device 代码整条链路自持**的（自带 launch/wrapper、stream 处理、workspace、cache 目录）；而 Inductor 要求 host 侧 wrapper 由它自己生成（`PythonWrapperCodegen` / `CppWrapperGpu`），codegen 只负责出 device 代码 + 元数据。这两边的边界是冲突的。

---

## 3. 编译模型层：缓存、autotune、AOT、动态 shape，全是摩擦点

| 维度 | Inductor 的假设 | TileLang 的现实 |
|---|---|---|
| 代码生成确定性 | 生成源码参与 `FxGraphCache` key，要求可复现、可 CSE | 有 autotune（多候选编译 + profiling），生成结果依赖搜索过程 |
| 编译时间预算 | 秒级，放在 `compile_worker` 子进程里异步做 | 首次 per-shape JIT，加 autotune 很容易变成几十秒 |
| 缓存 | Inductor 自己的 FX graph cache + `codecache` | 自己的 `$HOME/.tilelang/cache`，两份缓存要一起失效 |
| 动态 shape | SymInt + range/guard 体系 | `T.dynamic("m")` + `T.ceildiv` + mask（有支持，但覆盖面与符号化能力比 Triton 窄） |
| 部署 | AOTInductor 把生成的 kernel 打包进 `.so` | 需要 tilelang 运行时 / 预编译产物管理，跨机器 ABI 更重 |

这些不是"不可调和"，但每一条都要专门写适配，收益却不明朗（见下）。

---

## 4. 但它其实**能**——已经在昇腾上做出来了（重点）

这是回答这个问题时最该讲的反例：

- **[Ascend/pytorch PR #40597] `[feat] Add TileLang Inductor backend for Ascend NPU`** —— 为昇腾 NPU 的 Inductor 增加一个 **TileLang codegen 后端**，实现 codegen 数据模型、kernel 规划与源码生成链路，并配套测试、benchmark、文档；把 pointwise / reduction / broadcast / loop region / 数据搬运 / 函数签名分开规划，避免在 CSE 捕获阶段直接拼源码。不支持的 op/dtype、tuple reduction、atomic add、无法映射的 layout/indexing 会**报错并回退**。
- 为什么昇腾可以：**`npu` 这个 device 插槽是空的**，Triton-Ascend 生态弱，TileLang 在这个区间是净收益；而且走的是上一节说的 PrivateUse1 vendor hook，不用改上游。
- 反过来，CUDA 上那个槽是 Triton，且 Inductor 已经在 CUDA 上把 Triton 打磨到了"模板 + CUTLASS/cuBLAS extern + C++ wrapper + AOTInductor"整套体系，**替换而不是新增**等于重写 Inductor 的 CUDA 路径。

另外两个旁证，说明社区对 TileLang 的定位：

- PyTorch RISC-V 路线图（issue #180975）Phase 4 的措辞是：**"Define the Triton / TileLang integration interface for RISC-V (inductor template backend or external kernel provider)"** —— 默认定位是"模板后端 / 外部 kernel 提供者"，不是通用 codegen 后端。
- TileLang 侧：issue #87 `[Feature Request] Torch Inductor Support`（2025-02 提，2025-12 关闭）、roadmap #79 里的 `Integrate TileLang with Torch Inductor` 长期未打勾；而在 #905 里 TileLang 作者 LeiWang1999 的说法很直接：

> "Currently, Tilelang is being used to write some specific kernels. **I think combining it with torch.compile doesn't bring much benefit at the moment.** My suggestion is to write more complex kernels using Tilelang, while letting torch.compile automatically optimize the other kernels with Triton."
> —— tile-ai/tilelang issue #905

- PyTorch 侧的官方 RFC（#152032, msaroufim）也在同一层意思上：Triton 是"PyTorch-first"的、直接吃 `torch.Tensor` 和 `torch.cuda.Stream`；对 tilelang / cutile / cuda-python 这类新 DSL，"what the interop will look like ... remains to be seen"，并给出 AOTInductor 现有的四条接入途径：`register_external_matmul`、`ExternKernel`、`triton_template_registry`、CMake 打包自定义 kernel。

**生态位的本质**：TileLang 的卖点是"手写复杂 kernel"（FlashAttention/MLA/MoE/dequant GEMM）；而这类 kernel 在 `torch.compile` 里本来就是以 **extern / custom op** 的形式存在的，Inductor 不需要"生成"它们。Inductor 需要生成的是海量 pointwise/reduction，那正是 Triton 的舒适区。两者**覆盖区间重叠很小、互补关系很清楚**——这正是"不能也不划算"的核心。

顺带说，Inductor 里塞多一条 codegen 路线的维护成本有先例：`torch/_inductor/codegen/halide.py`（`HalideScheduling` / `HalideKernel`）到今天还躺在 main 里，但它是 `cuda_backend`/`cpu_backend` 的"experimental"选项，基本没有生产用户——这就是"新增 codegen 后端在上游的命运"的样本。

---

## 5. 那实际想用 TileLang + torch.compile，怎么做？

<details>
<summary><b>路径 A（最常用、最推荐）：把 TileLang kernel 包成 custom op，让 Inductor 当 extern 处理</b></summary>

```python
import torch, tilelang
import tilelang.language as T

@torch.library.custom_op("mylib::flash_fwd", mutates_args=())
def flash_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _tl_flash_kernel(q, k, v)   # 内部调用 tilelang 编译好的 kernel

@flash_fwd.register_fake
def _(q, k, v):
    return torch.empty_like(q)
```

Inductor 会把它当黑盒（`ExternKernel` / fallback），其余图照旧交给 Triton。想让它参与调度/autotune，就配 `register_external_matmul` 或 `triton_template_registry`。
</details>

<details>
<summary><b>路径 B：写一个自定义 torch.compile backend，在 Graph 层替换子图</b></summary>

```python
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx

@register_backend
def tilelang_inductor(gm, example_inputs):
    gm = my_pattern_rewrite_to_tilelang_ops(gm)  # 把 attention/MoE 子图换成 TileLang op
    return compile_fx(gm, example_inputs)        # 其余仍走 Inductor+Triton
```
好处是**不用碰 Inductor 内部**，坏处是只能做图级替换，拿不到 codegen 级的融合。
</details>

<details>
<summary><b>路径 C：干脆不用 torch.compile，TileLang + CUDA Graph 手工编排</b></summary>

这也是 vLLM / SGLang 现在的做法：主干用 TileLang 写融合 kernel，用 `torch.cuda.CUDAGraph` 或自研 runtime 编排，绕开"谁当谁的 codegen 后端"这个问题。
</details>

<details>
<summary><b>路径 D（只在非 CUDA 设备上有意义）：复制昇腾的做法</b></summary>

如果你的目标是自研加速器/NPU 类 device：走 PrivateUse1 的 `torch.<backend>._inductor_backend_init` vendor hook，自己 `register_backend_for_device`，在里面挂 `TileLangScheduling + TileLangKernel`，并用 `BackendFeature` 声明支持范围，不支持的 op 主动抛错回退。**这条路 CUDA 走不通，因为槽位被 Triton 占了。**
</details>

---

## 6. 一句话总结

TileLang 当不了 TorchInductor 的 CUDA codegen 后端，**不是因为技术上被"禁止"，而是因为 (1) `device_codegens["cuda"]` 这个槽是进程级硬注册、枚举里只有 triton/halide/pallas；(2) TileLang 是 tile 级 DSL，没有 Inductor 需要的"loop nest → 逐元素代码"契约，接上去得先重写一套 IR→tile plan 的前端；(3) JIT + autotune + 自有 cache/ABI 与 Inductor 的确定性 codegen、FxGraphCache、AOT 打包语义相冲；(4) 它和 Triton 的覆盖区间互补而非替代，TileLang 作者本人也认为收益不大。** 一旦换到"插槽空着、Triton 生态弱"的设备（昇腾 NPU），立刻就有团队把它做成了真正的 Inductor 后端（Ascend/pytorch PR #40597）——这恰好证明限制在**插槽与生态位**，而不在"能不能编译"。
