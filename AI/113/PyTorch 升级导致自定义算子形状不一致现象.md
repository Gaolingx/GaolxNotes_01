# PyTorch 升级导致自定义算子形状不一致现象

## 🧑‍💻 User

什么情况下升级pytorch我的自定义算子shape有可能不一致

---
## 😀 默认助手 (7)

# 升级 PyTorch 后自定义算子 shape "不一致" 的常见原因

先给结论：**绝大多数"shape 不一致"不是你的 kernel 算错了，而是 shape 推断路径（meta / FakeTensor / SymInt / out= 变体）在新版本里走了不同的分支，或者你依赖了不稳定的内部 API。** 只有少数情况是下游算子公式/广播规则本身变了。

---

## 一、最高频：Meta / Fake / 符号化推断路径缺失或变化

只要你的算子进入 `torch.compile`、`torch.export`、`fake tensor` 或 `meta device`，PyTorch 就不再执行真实 kernel，而是调用"abstract"实现来推 shape。这里最容易跨版本翻车：

| 情况 | 升级后可能的表现 |
|---|---|
| 只注册了 CPU/CUDA kernel，没注册 `register_fake` / meta | 旧版可能 fallback 到 CPU 真张量跑一遍来猜 shape；新版可能报错，或改用别的推断路径 |
| meta 实现里用了 `int` 而不是 `SymInt` | `torch.compile(dynamic=True)` 下 shape 被特化成一个固定值，与真实执行不符 |
| 数据依赖的输出 shape（如 `nonzero` 类）没声明动态维 | fake 阶段返回具体常量，真实运行时是动态值 |
| meta 实现里做了 `.item()` / `.tolist()` / `int(shape)` | 引入 guard 或特化，新旧版本 guard 行为不同 |

正确写法：

```python
@torch.library.custom_op("mylib::fused", mutates_args=())
def fused(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

@fused.register_fake
def _(x, y):
    return x.new_empty(x.shape[0], x.shape[1] + y.shape[1])

# 数据依赖的维度
@torch.library.register_fake("mylib::nonzero_like")
def _(x):
    ctx = torch.library.get_ctx()
    n = ctx.new_dynamic_size()      # 不要用具体 int
    return x.new_empty(n)
```

> 注意 `register_fake`（FakeTensor，给 compile 用）和 `register_meta`（meta device）是**两个不同**的注册点，升级后要求可能更严格。

---

## 二、依赖了不稳定内部 API

以下东西**跨版本没有兼容保证**，升级后 shape 推导改变非常正常：

- `at::native::*`、`at::native::cpu_kernel`、THNN/THC 遗留实现
- `TensorIterator` 的内部 shape 计算与 broadcast 细节
- `set_output` 系列（`set_output_raw_strided` / `resize_output`）在结构化 kernel 里的行为
- `torch._C._nn`、`torch._C._VariableFunctions` 等私有入口
- `torch._subclasses`、`torch._dynamo`、`torch._functorch` 的私有接口

**尤其注意 `out=` 变体**：早期版本对传入的 `out` 张量会自动 `resize_`，新版可能改为要求 shape 一致或报 warning/error。如果你的自定义算子有 out= 重载，升级后 shape 行为最容易突变。

---

## 三、广播 / 类型提升 / 视图语义变化

- **零尺寸维度的广播**：PyTorch 曾多次向 NumPy 对齐（例如 `(0,3)` 与 `(1,3)` 的广播），若你用 TensorIterator 或手工广播，结果 shape 会变。
- **类型提升规则**：不同 dtype 组合提升到不同结果 dtype，进而影响 dispatch 到哪个 kernel、以及该 kernel 的 shape 逻辑。
- **视图/别名语义**：`view` vs `reshape`、`expand`、`squeeze`、`unfold`、`as_strided` 在编译路径下的 view/rebase 处理变化，可能表现为 metadata/shape 不一致。
- **SymInt 溢出检查**：大尺寸场景新旧版本对 `int64` 溢出的处理不同。

---

## 四、内部调用的下游算子公式变了

如果自定义算子内部调用了会改变输出尺寸的 ATen 算子，它的公式一变，你的 shape 就变：

- `interpolate`（`recompute_scale_factor`、`align_corners` 相关默认值曾在版本间变化）
- 卷积 / 池化在 `padding='same'`、`ceil_mode`、`output_padding`、`count_include_pad` 下的输出公式：
  $$H_{out}=\left\lfloor \frac{H_{in}+2p-d(k-1)-1}{s}\right\rfloor+1$$
- `pad` 的 `reflect`/`replicate` 对超大 padding 的约束
- `nonzero`、`unique`、`sort`、`bincount`、`split`/`chunk` 的边界与稳定性语义修 bug
- autograd 的 view/rebuild 行为（影响 `.grad` 的 shape，而非 forward）

---

## 五、Tensor 子类与 dispatch 层

- `__torch_function__` / `__torch_dispatch__`：新版本对子类、mode、decomposition 的介入时机变化，可能导致输出落到不同实现上，shape 随之不同。
- `CompositeImplicitAutograd` / `CompositeExplicitAutograd` 注册方式会影响 PyTorch 是"追踪实现"还是"当作黑盒"。当 trace 你的 composite 实现时，里面任何广播规则变化都会传导到输出 shape。
- 动态 shape 的 guard 逻辑在 `torch._dynamo` 中跨版本调整，可能让同一份代码在某版本被特化、另一版本不特化。

---

## 六、后端 / 构建差异（不是版本本身，但常一起变）

- PrivateUse1 新后端、`TORCH_LIBRARY_IMPL` 注册优先级变化
- CPU vs CUDA build、不同 BLAS/cuDNN 版本导致算法选择不同（一般不改 shape，但配合 `padding='same'` 等实现可能改）
- `torchvision` / `torchtext` / `triton` 与 torch 版本不匹配
- ONNX exporter（旧 exporter vs dynamo exporter）走不同的 `symbolic` 推断

---

## 快速定位表

| 症状 | 先查这里 |
|---|---|
| 只有 compile/export 下 shape 错 | `register_fake` / `register_meta` / `SymInt` 使用 |
| 动态 batch 被固定成常量 | meta 里用了 `int`、`.item()`、`new_dynamic_size` 缺失 |
| 带 `out=` 的重载报错或 shape 怪异 | `out` 的 resize 语义变化 |
| 零尺寸 / 空张量下 shape 不同 | 广播规则、`TensorIterator` |
| 内部调 conv/pool/interpolate 的算子 | 下游输出公式与默认参数 |
| 子类/自定义 backend 上出错 | dispatch 注册类型、`__torch_dispatch__` |

排查命令：

```python
from torch.library import opcheck
opcheck(torch.ops.mylib.fused.default,
        (x, y),
        test_utils=["test_schema", "test_faketensor", "test_autograd"])

# 对比真实执行与 fake 执行
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode() as mode:
    fx = mode.from_tensor(x)
    print(torch.ops.mylib.fused(fx, mode.from_tensor(y)).shape)
print(torch.ops.mylib.fused(x, y).shape)
```

---

## 预防清单

- **给每个自定义算子补 `register_fake`（或 meta）**，并且只用 `SymInt`/动态维，不要返回硬编码 int。
- **只用公开 API**（`at::empty` / `at::Tensor` 语义），远离 `at::native` 与 `_C._nn`。
- **显式声明 `out=` 行为**，不要依赖自动 resize。
- **CI 里跑 `torch.library.opcheck` + `torch.compile(dynamic=True)` 的 shape 对比测试**，锁死跨版本的 shape 契约。
- 升级时读 release notes 的 **"BC-breaking changes"**，重点看 `TensorIterator`、type promotion、`FakeTensor`、`torch.library`、结构化 kernel 那几节。
- 固定 `torch` / `torchvision` / `triton` 的配套版本，避免混搭。

<details>
<summary><b>一句话记忆</b></summary>

真实 kernel 的 shape 通常不会因升级而变；**变的是"不执行 kernel 时 PyTorch 怎么猜 shape"**。所以自定义算子只要补全 meta/fake 推断、不碰内部 API、显式处理 `out=`，升级时 shape 不一致的概率会大幅降低。

</details>
