# 大MLP前向溢出解决

### 🧑‍💻 User

如何解决训练过程中fp16mixed遇到mlp shape特别大的时候fwd出现精度溢出导致NaN的问题？（已知gradscaler不work）

---

### 🤖 Assistant

### 解决训练过程中fp16 mixed precision下MLP大shape前向溢出NaN的问题

当MLP的shape特别大时，fp16的最大表示范围仅为 $\pm 65504$，导致在前向传播（FWD）过程中累加溢出产生Inf/NaN。鉴于 `GradScaler` 仅作用于反向传播的梯度缩放，对前向计算无效，需要从**底层算子、计算精度、网络结构及数值稳定性**四个维度进行修复。

以下是几种解决方案的对比与实施细节：

<details>
<summary><b>方案一：启用底层BLAS的TF32与BF16计算路径（推荐，无需改代码）</b></summary>

最有效的解决方式是让框架底层的 cuBLAS 在处理大型矩阵乘法时使用具有更大动态范围的格式。

- **BF16 矩阵乘法**：BF16 具有与 FP32 相同的动态范围（约 $10^{\pm 38}$），可以直接避免 FP16 的溢出问题。现代 GPU（如 A100, H100）原生支持 BF16。
- **TF32 矩阵乘法**：如果硬件不支持 BF16 加速，可以使用 TF32（NVIDIA Ampere+ 架构支持），其动态范围与 FP32 相同，精度介于 FP16 和 FP32 之间。

**PyTorch 启用方法**：
```python
import torch
# 启用 TF32 矩阵乘法 (针对 Ampere+ 架构)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 或者直接强制大矩阵乘法使用 BF16 而非 FP16 
# torch.set_float32_matmul_precision("medium") # 或 "high"
```
*注意：此方法可能在底层将 fp16 输入转换并提升为 bf16 或 tf32 进行 GEMM 计算。*
</details>

<details>
<summary><b>方案二：使用 BF16 替代 FP16 进行混合精度训练</b></summary>

既然 `GradScaler` 不 work，说明问题出在前向。最直接的架构级替换是将 `torch.cuda.amp.autocast` 的数据类型从 `fp16` 改为 `bf16`。

BF16 提供了 $8$ 位指数（同 FP32）和 $7$ 位尾数，完全不存在 $\pm 65504$ 的溢出上限问题。

**PyTorch 实现方式**：
```python
# PyTorch >= 1.10 支持 BF16 混合精度
scaler = None # 使用 BF16 时不需要 GradScaler

with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

# loss.backward() 直接反向传播即可
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
**前提条件**：训练硬件必须支持 BF16（NVIDIA A100/H100 或 更新的 GPU；Akilibaba 平崗/平头哥等）。
</details>

<details>
<summary><b>方案三：Fused Kernel 替换与自定义算子（针对极深/极宽MLP）</b></summary>

如果框架原生的 Linear 层在极端 shape 下依然有计算中间态溢出，可以考虑替换底层算子。

1. **使用 `xops` 或 `flash-attention` 类似的高效算子**：
   比如使用 `torch.nn.functional.linear` 底层的优化版本，或者采用 `apex` 的 `FusedLayerNorm` / `FusedMLP`。这些算子通常在累加时使用 FP32 寄存器，最后再截断回 FP16。
 
2. **手写 Triton 算子控制累加精度**：
   如果瓶颈在于 $Y = XW^T$ 这种大型 GEMM 的局部累加溢出，可以使用 Triton 自定义控制累加器精度。
   ```python
   import triton
   import triton.language as tl

   @triton.jit
   def fused_mlp_kernel(
       x_ptr, w_ptr, y_ptr,
       N, M, K,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
   ):
       # 关键：在 K 维度（内部 reduce 维度）累加时，强制使用 tl.float32
       acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
       for k in range(0, K, BLOCK_K):
           x = tl.load(...) # 加载 fp16
           w = tl.load(...) # 加载 fp16
           # tl.dot 支持指定 output_dtype
           acc += tl.dot(x, w, out_dtype=tl.float32) 
     
       # 截断回 fp16 保存
       y = acc.to(tl.float16)
       tl.store(...)
   ```
</details>

<details>
<summary><b>方案四：网络结构层面的数值隔离（不需切回纯 FP32）</b></summary>

如果由于硬件限制（如 V100 不支持 BF16，也不能用 TF32）必须使用 FP16，可以通过调整网络结构来限制激活值的范围。

1. **给大型 MLP 添加自定义前向 Hook 进行截断**：
   在前向传播中，对可能溢出的大 shape 层的输出进行限幅，限制在 $[-65504, 65504]$ 之内，拦截 NaN。
   ```python
   def clamp_hook(module, input, output):
       # 限制在 fp16 最大安全范围，比如稍微留点余量
       return torch.clamp(output, min=-65000, max=65000)

   large_mlp_layer.register_forward_hook(clamp_hook)
   ```

2. **分块计算**：
   将一个巨大的 MLP 层拆分为两个较小的层，中间插入 GELU/ReLU 或者残差连接。降低单次 GEMM 的隐层维度 $N$：
   $$ \text{Linear}_{in \to big} \rightarrow \text{Linear}_{big \to out} $$
   修改为：
   $$ \text{Linear}_{in \to mid} \rightarrow \text{GELU} \rightarrow \text{Linear}_{mid \to out} $$
 
3. **对输入进行细粒度 Norm**：
   大量累加溢出通常是因为输入 $X$ 的方差极大导致的。在进入极大 Linear 层之前，强制做一次 FP32 范围的 Normalization：
   ```python
   with torch.autocast(enabled=False):
       x_fp32 = x.float()
       x_fp32 = (x_fp32 - x_fp32.mean(dim=-1, keepdim=True)) / (x_fp32.std(dim=-1, keepdim=True) + 1e-6)
       x = x_fp32.half()
   output = large_mlp(x)
   ```
</details>

### 总结与优先级建议

建议按以下优先级排查和解决问题：

1. **第一步验证**：首先尝试替换为 BF16 混合精度（前提是显卡支持），这能 100% 解决前向溢出问题且几乎不损失训练速度。
2. **第二步验证**：若不支持 BF16，开启 PyTorch 算子层面的 `TF32` 支持。
3. **第三步**：若由于某些原因必须死守 FP16，则结合**网络结构前向 Clamp 截断**与**强制输入 FP32 归一化**，从数学上控制激活值绝对值不超过 $\pm 65504$。

---

### 🧑‍💻 User

经过debug发现溢出主要发生在部分layer的mlp matmul超过65504，修改arch 添加swiglu clamp能从根本解决这个问题吗？

---

### 🤖 Assistant

### 结论先行

**不能。** 在 SwiGLU 的 MLP 上加 clamp 属于"止损/封堵"，不是根因修复。它能阻止 NaN 向下游扩散、让训练"看起来还能跑"，但溢出发生在同一层的内部 GEMM 上，clamp 输出根本管不到那个位置。

---

### 一、先定位溢出到底发生在 SwiGLU MLP 的哪个环节

LLaMA 式 SwiGLU MLP 的计算链是：

$$h_{gate} = x W_{gate},\quad h_{up} = x W_{up},\quad h_{gate} = \text{SiLU}(h_{gate}),\quad h = h_{gate} \odot h_{up},\quad y = h W_{down}$$

可能的溢出点有三个：

| 位置 | 溢出原因 | 对输出做 clamp 能否修复 |
|---|---|---|
| $x W_{up}$ / $x W_{gate}$ 的**输出**（GEMM 结果本身 > 65504） | 输入 $x$ 或权重 $W$ 幅值过大，或 $K$（hidden）很大导致累加和超大 | **不能**（这是本层内部，clamp 在更后面） |
| $h_{gate} \odot h_{up}$ 元素乘 | $h_{up}$ 单元素就超过 65504 | **不能**（乘出来的 Inf 已经产生） |
| $h W_{down}$ 的累加（$K$ = 4× 隐藏维，极大） | 累加溢出 | **不能**（同样在本层内部） |

关键数学事实：**SiLU 的输出是有界的**

$$\text{SiLU}(z) = z \cdot \sigma(z) \in [-0.278,\ 1.278]$$

所以 SwiGLU 的门控乘法的幅值满足：

$$|h_{gate} \odot h_{up}| \le 1.278 \cdot |x W_{up}|$$

也就是说门控**最多只能把 up 投影的幅值缩小/放大 1.278 倍**。如果 $x W_{up}$ 本身超过 $\approx 51253$（即 $65504 / 1.278$），门控之后的 $h$ 依然溢出；如果 $x W_{up}$ 已经算出了 Inf，那 $h$ 里就是 Inf，clamp 拦不住——**Inf 在被 clamp 之前就已经进入了元素乘**。而 down 投影的输入 $h$ 是 MatMul 的中间结果，clamp 作用在 $h$ 上（如果你 clamp 的是 MLP 输出）则连 down 投影内部累加都碰不到。

---

### 二、为什么 clamp "看起来有效"

它确实能挡掉一类情况：溢出产生 Inf/NaN 后，NaN 会**逐层传染**（进 attention QKV、进下一层、反向传播回权重、再污染下一次前向，形成死循环）。如果你 clamp 在溢出层**之后**的输出上，等于把 NaN 掐断在源头附近，下游不再被污染，loss 曲线恢复"正常"。

但这是**隔离，不是治愈**：

1. **本层内部仍然每次都在溢出**，数值失真和梯度信息丢失持续存在；
2. **clamp 会杀掉梯度**：硬 clamp 在饱和区 $\partial y/\partial x = 0$，容易造成死神经元或训练不稳；
3. **掩盖真实问题**：正常训练的 fp16 MLP 输出是**不应该**稳定触到 65504 的。会稳定触顶通常意味着输入分布、初始化、学习率或梯度处理已经有问题——clamp 只是把表象盖住了。

---

### 三、真正的根因与对应修法（按优先级）

**第 0 步：先分清是"真实大数值"还是"上游污染"。** 用 hook 检查权重里是否已经有 NaN/Inf：

```python
for name, p in model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        print("weight poisoned:", name)
```

如果权重已经是 NaN，那问题在**反向/梯度缩放**而不是前向，先修那边的数值（如 NaN 时的 grad 清零、clip grad norm、检查 loss 里是否有 inf）。GradScaler "不 work" 有时就是因为梯度里进了 NaN。

**第 1 步：确认 GEMM 是否真的在 fp32 累加。** 多数 cuBLAS fp16 GEMM 默认 fp32 累加（65504 溢出通常发生在**结果写回 fp16** 那一刻，而不是累加过程中）。确认你的 kernel 没有走 fp16 累加的快速路径。若已是 fp32 累加，说明是**真实幅值**超过了 65504，往下看。

**第 2 步（最根本）：换计算精度，绕开 65504 天花板。**

- **BF16**：指数位 8 位与 fp32 相同，动态范围 $\approx 10^{\pm38}$，完全消除溢出问题（A100/H100 原生支持）；
- **TF32**：Ampere+ 支持，指数范围与 fp32 相同，GEMM 精度介于 fp16 和 fp32 之间；

```python
torch.backends.cuda.matmul.allow_tf32 = True
# 或对 MLP 单独用 BF16 autocast
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    ...
```

这是**唯一能"从根本上"消除 fp16 动态范围限制**的办法。

**第 3 步：如果必须死守 fp16（硬件限制），把 clamp 放在正确的位置 + 用软 clamp。**

- 位置：clamp **up/gate 投影的输出**（GEMM 结果写回 fp16 之前），而不是 clamp SwiGLU 输出；
- 函数：用可微的软 clamp 代替硬截断，保住梯度：

$$\text{softclamp}(z,\ c) = c \cdot \tanh(z/c)$$

```python
def soft_clamp(z, c=40000.0):
    return c * torch.tanh(z / c)

# 放在 up/gate 投影之后、gating 之前
h_up   = soft_clamp(x @ W_up)      # 留出 ×1.278 的门控余量
h_gate = soft_clamp(x @ W_gate)
h = torch.nn.functional.silu(h_gate) * h_up
y = h @ W_down
```

注意这里 $c$ 必须满足 $c \cdot 1.278 < 65504$，否则门控后仍可能溢出。

**第 4 步（架构级、更 principled）：在 MLP 之前强制归一化输入。** 正常 LLM 里 MLP 前有 RMSNorm/LayerNorm，把 $x$ 的幅值钉在 $O(1)$，这是大 shape MLP 在 fp16 下不溢出的**正解**。如果你的架构恰好缺了这一步（或 norm 被 fp16 计算污染），补上它对数值稳定性的收益远大于 clamp：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x
```

**第 5 步：per-tensor forward scaling。** 相当于给前向也做一个"loss scaling"：对 up/gate 投影的输出按层除以一个统计缩放因子（用 EMA 估计的 max|·|），再做 gating，最后 down 投影时乘回来。比 clamp 温和，但要小心乘回来的精度。

---

### 四、一句话总结

| 方案 | 是否根本解决 |
|---|---|
| SwiGLU 输出加 clamp | ❌ 只是隔离 NaN 传播 |
| up/gate 投影前加软 clamp + 前置 Norm | ⚠️ 缓解，工程可用但不根本 |
| 换 BF16 / TF32（或 fp32 累加） | ✅ 根本解决（消除 65504 上限） |
| 修复上游 NaN（梯度/权重污染） | ✅ 若根因在此则是正解 |

**建议行动顺序**：先查权重是否被 NaN 污染 → 确认 GEMM 是 fp32 累加 → 能换 BF16/TF32 就换（一劳永逸）→ 若必须 fp16，则把 soft-clamp 放到 up/gate 投影**之前**并补上 MLP 前的 RMSNorm，而不是 clamp SwiGLU 的输出。