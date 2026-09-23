# HF transformers核函数自动分发

## 🧑‍💻 User

hf transformers的`use_kernel_func_from_hub_with_fallback`如何使用？（自动将算子分发到高性能实现上）

---
## 🤖 Assistant

这个 PR 是关于 CPU dtype 安全的，不是目标函数。让我直接搜索代码。

Google 无法返回结果。直接尝试拉取 transformers 源码中可能包含该函数的文件：

找到了源码。让我再确认一下实际模型中的使用示例：

我刚才从 transformers `main` 分支的源码里找到了这个函数的完整实现和使用范例（`src/transformers/integrations/hub_kernels.py`，以及 `modeling_qwen3_next.py` 中的真实用法）。下面是详细说明。

# `use_kernel_func_from_hub_with_fallback` 使用指南

## 一、它是什么

这是 transformers v5.x 中 **Hub Kernels 集成体系**里的一个**函数装饰器**（针对自由函数，而非 `nn.Module` 类），定义在 `transformers/integrations/hub_kernels.py` 中：

```python
def use_kernel_func_from_hub_with_fallback(
    func_name: str,               # 算子/kernel 的函数名
    package: str,                 # 可选的"原厂"高性能包，如 "fla"、"causal_conv1d"、"mamba_ssm"
    internal_path: str | None = None,  # 函数在包内的子模块路径，如 "ops.gated_delta_rule"
):
```

官方 docstring 的核心一句话：

> The same as `use_kernel_forward_from_hub` but with the optional fallback to an original package if it exists, e.g., FLA for Gated Delta Rule, mamba-ssm for mamba2.

它的作用是给一个"纯 PyTorch 参考实现"打上标记，让该算子在被调用时**按优先级自动分发到最高性能的可用实现**：

| 优先级 | 实现来源 | 生效条件 |
|---|---|---|
| 1️⃣ | **HF Hub Kernel**（如 `kernels-community/fla` 的 Triton kernel） | 安装了 `kernels` 库，且加载模型时传入 `use_kernels=True` |
| 2️⃣ | **原厂优化包**（如 `flash-linear-attention`、`causal-conv1d`、`mamba-ssm`） | 该 pip 包已安装 |
| 3️⃣ | **纯 PyTorch 参考实现**（被装饰的函数体） | 兜底，永远可用；会 `warning_once` 提示"正确但慢一个数量级" |

## 二、作为终端用户：怎么让它生效

**你通常不需要自己写这个装饰器**——transformers 内置模型（Qwen3-Next、Qwen3.5、Mamba2、KDA 等线性注意力模型）已经大量使用它。你要做的是"喂给"分发链条件：

```bash
# 1. 安装 kernels 库（版本需在 transformers 要求区间内，见报错提示）
pip install kernels

# 2.（可选）安装原厂包作为第 2 级 fallback
pip install flash-linear-attention causal-conv1d
```

```python
from transformers import AutoModelForCausalLM

# 3. 加载时显式开启 Hub kernels
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    use_kernels=True,   # ← 关键：触发 kernelize，把装饰过的算子替换为 Hub kernel
)
```

加载时 transformers 会：
1. 调用 `register_kernel_mapping_transformers()` 注册默认 kernel 映射表（`_KERNEL_MAPPING`）；
2. 调用 `kernelize(model)`，按你的 **设备**（`cuda`/`rocm`/`xpu`/`mps`/`npu`/`cpu`，含 SM capability 条件）和 **模式**（`Mode.TRAINING` / `Mode.INFERENCE` / `Mode.TORCH_COMPILE`）挑选匹配的 Hub kernel 并替换函数调用；
3. 没有匹配项 → 自动退回第 2/3 级，**不会报错**。

<details>
<summary><strong>⚙️ 进阶：自定义 kernel 映射（KernelConfig）</strong></summary>

```python
from transformers.utils.kernel_config import KernelConfig

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_kernels=True,
    kernel_config=KernelConfig(
        kernel_mapping={"RMSNorm": "kernels-community/rmsnorm:RMSNorm"},  # "repo_id:layer_name"
        use_local_kernel=False,   # True 时从本地路径加载（LocalLayerRepository）
        inherit_mapping=True,     # 是否叠加 transformers 默认映射
    ),
)
```

- 非官方 `kernels-community` 之外的 repo 需要 `trust_remote_code=True`，或用上下文管理器：
```python
  from transformers.integrations.hub_kernels import allow_all_hub_kernels
  with allow_all_hub_kernels():
      model = ...from_pretrained(..., use_kernels=True)
  ```
- 全局开关：环境变量 `USE_HUB_KERNELS=NO` 可彻底禁用所有 Hub-kernel 装饰器（退化为 no-op）。

</details>

## 三、作为模型/库作者：怎么给自己的算子接上分发链

以 `modeling_qwen3_next.py` 中的**真实源码**为例（这就是标准用法）：

```python
from transformers.integrations import (
 use_kernel_forward_from_hub,
 use_kernel_func_from_hub_with_fallback,
 use_kernelized_func,
)

# 第 1 步：给纯 torch 参考实现打标记
# "chunk_gated_delta_rule" = kernel 名（须存在于 kernel 映射表中，映射到 kernels-community/fla）
# package="fla" = 第 2 级 fallback：flash-linear-attention 包
@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
def torch_chunk_gated_delta_rule(query, key, value, g, beta, ...):
 # 这里是纯 PyTorch 实现（优先级 3），逐 token/chunk 的参考逻辑
 ...

# 另一个例子：package 里函数不在根命名空间时，指定 internal_path
@use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")
def causal_conv1d_update(hidden_states, conv_state, weight, ...):
 ... # F.conv1d 的纯 torch 版

# 第 2 步：在调用这些函数的 Module 上，用 use_kernelized_func 登记它们，
# 使 kernels 库在 kernelize 时能把"函数调用"替换为 Hub kernel
@use_kernelized_func(
 [torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
 causal_conv1d_fn, causal_conv1d_update]
)
class Qwen3NextGatedDeltaNet(nn.Module):
 def forward(self, ...):
 ...
 # 正常调用装饰过的函数即可，分发全自动
 core_attn_out, state = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, ...)
```

### 三个参数的具体含义

| 参数 | 作用 | 示例 |
|---|---|---|
| `func_name` | kernel 函数名，**必须**能在 kernel 映射表里找到对应 `LayerRepository`；同时也是包内函数名 | `"chunk_gated_delta_rule"` |
| `package` | 第 2 级 fallback 的 pip 包名（import 名） | `"fla"`、`"causal_conv1d"`、`"mamba_ssm"` |
| `internal_path` | 函数在包内的子模块路径。**注意**：内置映射表 `_KERNELS_INTERNAL_PATH_MAPPINGS` 优先于你手动传入的值 | `"ops.gated_delta_rule"`（→ `fla.ops.gated_delta_rule.chunk_gated_delta_rule`） |

内置的 `internal_path` 自动映射（无需手动传）：

```python
_KERNELS_INTERNAL_PATH_MAPPINGS = {
 "chunk_kda": "ops.kda",
 "fused_recurrent_kda": "ops.kda",
 "chunk_gated_delta_rule": "ops.gated_delta_rule",
 "fused_recurrent_gated_delta_rule": "ops.gated_delta_rule",
 "mamba_split_conv1d_scan_combined": "ops.triton.ssd_combined",
 "selective_state_update": "ops.triton.selective_state_update",
 "mamba_chunk_scan_combined": "ops.triton.ssd_combined",
 "mamba_inner_fn": "ops.selective_scan_interface",
 "selective_scan_fn": "ops.selective_scan_interface",
}
```

## 四、运行时分发逻辑（源码级解析）

<details>
<summary><strong>🔍 展开：装饰器内部做了什么</strong></summary>

```python
def use_kernel_func_from_hub_with_fallback(func_name, package, internal_path=None):
 # ① 外层套上 hub-kernel 包装（use_kernels=True 时由 kernelize 替换）
 kernel_wrapper_decorator = use_kernel_forward_from_hub(func_name)
 internal_path = _KERNELS_INTERNAL_PATH_MAPPINGS.get(func_name, internal_path)
 full_func_path = func_name if internal_path is None else f"{internal_path}.{func_name}"

 def decorator(torch_function):
 # ② 装饰时立刻尝试 import 原厂包，解析出真正的实现
 try:
 module = importlib.import_module(package)
 implementation = resolve_internal_import(module, full_func_path)
 # 兼容 FLA 这类不从包根暴露子模块的库
 if implementation is None and full_module_path != package:
 implementation = getattr(importlib.import_module(full_module_path), func_name, None)
 except Exception:
 implementation = torch_function
 finally:
 implementation = torch_function if implementation is None else implementation

 applicable_params = tuple(inspect.signature(implementation).parameters) # 固化签名
 is_new_implementation = implementation is not torch_function

 @functools.wraps(torch_function)
 def wrapped(*args, **kwargs):
 # ③ torch.export 时永远走纯 torch 路径（部分包不兼容 export）
 if is_new_implementation and is_torchdynamo_exporting():
 return torch_function(*args, **kwargs)
 # ④ 落到参考实现时给出一次性警告
 if not is_new_implementation and not is_torchdynamo_compiling():
 logger.warning_once(
 f"`{func_name}` is falling back to its reference PyTorch implementation because "
 f"`{distribution}` is not installed. This is correct but much slower; install "
 f"`{distribution}` for the optimized kernel.")
 # ⑤ 按"实现签名"过滤 kwargs，屏蔽 torch 实现与原厂包之间的签名差异
 kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
 return implementation(*args, **kwargs)

 return kernel_wrapper_decorator(wrapped)
 return decorator
```

几个值得注意的行为细节：
- **kwargs 白名单过滤**：原厂包（如 FLA）和 torch 参考实现的参数签名不完全一致，装饰器只把"实现所接受的参数"传下去，多余参数被静默丢弃；
- **`torch.compile` 感知**：处于 dynamo 编译追踪时不触发警告日志；
- **`torch.export` 保护**：export 场景强制走纯 torch；
- **失败静默降级**：原厂包 import 失败 → 直接用 torch 实现，永不 raise。

</details>

## 五、常见问题与注意事项

- **版本要求**：这是 transformers v5.x 的 API（`use_kernel_func_from_hub` 这个不带 fallback 的旧版在 v5.16 已被标记弃用，请用 `use_kernel_forward_from_hub` / `use_kernel_func_from_hub_with_fallback`）。`kernels` 库版本需落在 transformers 声明的区间内，否则报错提示 `pip install kernels==<min_version>`。
- **import 路径**：从 `transformers.integrations` 导入，不在顶层命名空间：
```python
from transformers.integrations import use_kernel_func_from_hub_with_fallback
```
- **设备/模式覆盖**：Hub kernel 只有在映射表中存在对应 `设备 + Mode` 条目时才被替换。例如 `RMSNorm` 映射了 `cuda`/`rocm`/`xpu`/`mps`/`npu`；`chunk_gated_delta_rule` 只映射了 `cuda`。在其他设备上自动落到第 2/3 级。
- **Attention kernel 是另一套入口**：attention 类 kernel 不走这个装饰器，而是通过 `attn_implementation="kernels-community/flash-attn2"` 这种 `repo_id[@rev][:kernel_name]` 格式的字符串由 `load_and_register_attn_kernel` 加载。
- **区分两个姊妹装饰器**：
 - `use_kernel_forward_from_hub("名字")` → 用于 **`nn.Module` 类**（替换其 `forward`），如 `@use_kernel_forward_from_hub("RMSNormGated")` 装饰 `Qwen3NextRMSNormGated`；
 - `use_kernel_func_from_hub_with_fallback(...)` → 用于**自由函数**（op 函数），并额外提供"原厂包 → torch"的降级链。

**一句话总结**：你只需 `pip install kernels` + `from_pretrained(..., use_kernels=True)`，transformers 就会自动把被打标的算子分发到 Hub 上针对你当前硬件和训练/推理模式优化的 kernel 实现；缺什么就逐级降级（Hub kernel → 原厂包 → 纯 PyTorch），保证永远能跑。
