# Muon优化器支持3DMoE权重

### 🧑‍💻 User

torch.optim.Muon 只支持 2D 参数，但 MoE 模型的 expert 权重是 3D 的（[num_experts, dim1, dim2]），怎么解决？

---

### 🤖 Assistant

在处理 Mixture of Experts (MoE) 模型时，专家权重通常是 3D 张量 `[num_experts, in_features, out_features]`。由于 Muon (Momentum Orthogonalizer) 优化器的核心是基于 Newton-Schulz 迭代的正交化操作，原始实现通常硬编码了对 2D 矩阵的断言（`assert p.ndim == 2`）。

要解决这个问题，有两种主要方法。**最推荐的方法是微调 Muon 优化器的代码使其支持 Batched 3D 运算**，这样可以在数学上完美保持“每个专家独立正交化”的特性。

以下是具体的解决方案：

### 方法一：修改 Muon 源码支持 Batched 3D（最推荐）

Muon 的正交化本质上是矩阵乘法。在 PyTorch 中，`@` 运算符（或 `torch.matmul`）原生支持批量矩阵乘法（Batched Matrix Multiplication）。因此，只需将转置操作从 `.T` 改为 `.mT`（最后两个维度转置），并放宽维度检查即可。

#### 1. 修改正交化函数
找到 Muon 源码中的 Newton-Schulz 迭代函数，将其修改为兼容 3D：

```python
import torch

def zeropower_via_newtonschulz5(G, steps=5):
    # 断言支持 2D (常规层) 和 3D (MoE层)
    assert len(G.shape) in [2, 3], "G must be 2D or 3D"
  
    # 计算缩放系数，针对最后两个维度计算 Frobenius 范数
    # sum(dim=(-1, -2)) 适用于 2D 和 3D
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz 迭代
    for _ in range(steps):
        # 使用 .mT (Matrix Transpose) 而不是 .T
        # 这样对于 3D 张量 [E, D1, D2]，.mT 会变成 [E, D2, D1]，天然支持批量矩阵乘法
        A = X @ X.mT 
        B = b * A + c * A @ A
        X = a * X + B @ X
      
    return X.to(G.dtype)
```

#### 2. 修改优化器的 step 函数
在 Muon 优化器的 `step()` 方法中，将维度限制放宽：

```python
# 在 Muon 的 __init__ 或 step 中寻找类似这样的代码并修改：
for p in group['params']:
    if p.grad is None:
        continue
    # 将原来的 p.ndim == 2 改为：
    if p.ndim not in [2, 3]:
        continue # Muon 仅处理 2D (Linear) 或 3D (MoE) 参数
      
    # 后续操作保持不变，因为上述的 zeropower 函数已经支持 3D
    grad = p.grad
    # ...
```

**数学原理解释：**
对于 MoE，我们希望每个专家的梯度 $G_i$ 独立进行正交化，得到 $O_i$。通过将 $G$ 视为批量矩阵，PyTorch 的批量乘法 $G @ G^T$ 实际上并行计算了所有 $i \in \{1, \dots, E\}$ 的 $G_i G_i^T$。这完全符合 Muon 的数学定义。

---

<details>
<summary><b>方法二：在模型 Forward 中 Reshape（不修改优化器源码）</b></summary>
<p>

如果你使用的是第三方库（如 `bitsandbytes` 或 `apex`）中封装好的 Muon，不方便修改其底层源码，你可以通过修改 MoE 模型的定义来“欺骗”优化器。

Muon 优化器要求传入的是 2D 的 `nn.Parameter`。你可以将权重定义为 2D 并在 `forward` 过程中动态 `view` 成 3D。

**模型修改示例：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
      
        # 1. 将参数定义为 2D: [num_experts * in_features, out_features]
        # 这样就可以安全地传入原版 Muon 优化器
        self.weight = nn.Parameter(
            torch.empty(num_experts * in_features, out_features)
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, routing_weights):
        # 2. 在前向传播时，将其 view 恢复为 3D: [num_experts, in_features, out_features]
        # 注意：view 不会破坏计算图和梯度回传
        w_3d = self.weight.view(self.num_experts, self.in_features, self.out_features)
      
        # 继续你原来的 MoE 前向逻辑
        # ...
        pass
```

**⚠️ 注意事项（数学差异）：**
将 `[E, D1, D2]` 展平为 `[E * D1, D2]` 后交由原版 Muon 处理，意味着优化器将**所有专家的参数合并为一个超大矩阵**进行正交化。
数学上，这不再是计算块对角矩阵的正交化，而是计算整个大矩阵的近似正交。虽然在实际深度学习训练中通常也能收敛，但这与“每个专家独立更新”的直觉稍有偏差，可能会对收敛速度产生微小影响。因此，**方法一始终是更优解**。

</p>
</details>

<details>
<summary><b>方法三：ParameterList 拆分（不推荐）</b></summary>
<p>
你可以将 3D 参数彻底拆分为 `num_experts` 个 2D `nn.Parameter`。

```python
self.experts_weight = nn.ParameterList([
    nn.Parameter(torch.randn(in_features, out_features)) 
    for _ in range(num_experts)
])
```
这种方法对 Muon 友好，但**极其不推荐**。因为它会破坏前向传播时利用 `torch.bmm`（批量矩阵乘法）带来的计算效率优势，导致 MoE 的计算退化为 Python 层面的 for 循环，严重拖慢训练速度。
</p>
</details>

### 总结
对于 3D MoE 权重，最优雅的做法是**修改 Muon 源码中的正交化步骤**。利用 PyTorch 原生支持批量矩阵运算的特性，将转置操作 `.T` 替换为 `.mT`，并允许 `ndim == 3` 即可完美解决。

---

### 🧑‍💻 User

感觉moe层不用muon，有些喧宾夺主，实际像megatron torchtitan训练框架是怎么处理这种问题的？（无法修改torch源码情况下）

---

### 🤖 Assistant

你提到的“MoE 层不用 Muon 是喧宾夺主”非常准确。在当前的大模型架构中，MoE 层的参数量通常占据了总参数量的 70% 以上，如果核心的优化算法无法应用在 MoE 上，那确实失去了使用 Muon 的最大意义。

由于 Muon 是一个相对较新的优化器（目前主要以 Python 脚本的形式在社区流传，尚未合并入 PyTorch C++ 核心底层），在大型训练框架（如 Megatron-LM 或 TorchTitan）中，如果遇到这种**“模型结构（3D）与优化器接口（2D）不匹配，且无法修改优化器源码”**的情况，通常会采用以下两种工程架构级的处理方式：

### 1. 采用 Grouped GEMM 与 2D 参数列表（Megatron-LM 的主流做法）

大型框架在追求极致性能时，并不会在 Python 层面直接使用一个 `[num_experts, in_features, out_features]` 的 3D Tensor，而是通过底层的 **Grouped GEMM（分组矩阵乘法）** 来实现 MoE。

**框架是如何处理的：**
在 Megatron-LM 等框架中，MoE 的多个专家在 Python 层的定义实际上就是**多个独立的 2D `nn.Parameter`**（通常放在一个 `nn.ModuleList` 中）。

*   **对优化器（Muon）而言**：它看到的完全是标准的 2D 张量，因此 `assert p.ndim == 2` 会完美通过，Muon 可以对每个专家独立进行正交化计算，数学上完全严谨。
*   **对前向传播而言**：框架不会使用低效的 Python `for` 循环，而是通过自定义的 CUDA 算子（如 CUTLASS 提供的 Grouped GEMM 或 MegaBlocks），将这 $E$ 个 2D 张量的内存指针（Pointers）收集成一个数组，直接在 GPU 底层以类似 3D Batched GEMM 的超高效率完成计算。

**代码概念示例：**
```python
# 框架层面的 MoE 权重定义（对优化器完全透明、友好）
self.experts_weight = nn.ParameterList([
    nn.Parameter(torch.empty(in_features, out_features)) # 标准 2D
    for _ in range(num_experts)
])

def forward(self, x, routing_indices):
    # 底层调用 Grouped GEMM CUDA 算子，传入 2D 参数指针
    return grouped_gemm_cuda_forward(x, routing_indices, self.experts_weight)
```
这种设计既保全了算力效率，又天然兼容所有传统的 2D 优化器逻辑。

---

### 2. 优化器包装器（Optimizer Wrapper）动态切片映射

如果在 TorchTitan 或特定的 FSDP 架构中，模型强制使用了 3D 参数矩阵，且你**绝对不能修改 Muon 的 Python 源码**，框架工程师通常会写一个 **Optimizer Wrapper**。

这个 Wrapper 会拦截 `optimizer.step()`，在传入黑盒优化器之前，在内存原地（In-place）伪装维度，算完后再还原。

**处理逻辑：**
既然 PyTorch 优化器要求叶子节点（Leaf Tensor），且不能传入 3D 张量的切片（View 且非叶子节点会报错），我们可以通过遍历 3D 张量的第一维，逐个提取 `p.data` 和 `p.grad` 构造成临时的 2D 叶子节点，送入 Muon 的单步计算。

**代码概念示例：**
```python
class MoECompatibleMuon(torch.optim.Optimizer):
    def __init__(self, params, muon_cls, **kwargs):
        # muon_cls 是原始的、无法修改的 Muon 优化器类
        defaults = kwargs
        super().__init__(params, defaults)
        self.muon_cls = muon_cls
        self.muon_kwargs = kwargs

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
              
                # 如果是正常的 2D 参数，直接调用原版 Muon
                if p.ndim == 2:
                    self._apply_muon_2d(p, group)
                  
                # 【核心解决逻辑】：如果遇到 3D MoE 权重
                elif p.ndim == 3:
                    num_experts = p.size(0)
                    for i in range(num_experts):
                        # 构造临时的 2D 独立参数和梯度（共享底层显存）
                        temp_p = torch.nn.Parameter(p.data[i])
                        temp_p.grad = p.grad[i]
                      
                        # 调用原版 Muon 计算单个 expert
                        self._apply_muon_2d(temp_p, group)
                      
                        # 将更新后的数据写回 3D Tensor
                        p.data[i].copy_(temp_p.data)

    def _apply_muon_2d(self, temp_p, group):
        # 临时构造一个仅包含单个 2D 参数的优化器实例并执行 step
        # (实际框架中会复用 state 字典以避免重复初始化开销)
        temp_opt = self.muon_cls([temp_p], **self.muon_kwargs)
        # 这里需要把外部的 optimizer state 映射进来
        temp_opt.state[temp_p] = self.state[p] # 省略复杂的状态路由代码
        temp_opt.step()
```

### 总结
在无法修改 `Muon` 源码的工业界训练框架中，绝不是简单地跳过 MoE，而是通过 **底层算子配合（Grouped GEMM）** 或 **Optimizer Wrapper 维度映射** 来解决的。

如果是你自己做实验，由于 Muon 本身就是一个开源的 Python 函数，**直接修改 Muon 的脚本（把 `.T` 改成 `.mT`，并放宽维度限制为 2 和 3）是成本最低、最 Pythonic 的做法**，千万不要为了追求“不改源码”而牺牲了训练效率。