# dwconv1d融合

## 🧑‍💻 User

卷积中的dwconv1d可以和哪些常见op fused，试写出triton代码

---
## 😀 默认助手 (7)

# Depthwise Conv1d 的融合机会与 Triton 实现

卷积融合的核心洞察是：**dwconv1d 每个通道独立计算，因此它的输入 tile 可以在寄存器里直接被"生产者"算子生成，它的输出 tile 可以在寄存器里直接喂给"消费者"算子**，从而省掉中间张量的 DRAM 往返。下面先列出常见的融合清单，再给出可运行的 Triton 代码。

## 一、可融合的常见 op 清单

<details open>
<summary><b>1. Epilogue（卷积之后，最自然、最划算）</b></summary>

| 融合 op | 说明 |
|--------|------|
| **BatchNorm** | 推理期直接 fold 进卷积 `weight`/`bias`（`w'=w·γ/√(σ²+ε)`，`b'=β-γ·μ/√(σ²+ε)`），**零成本** |
| **Bias add** | 逐通道 bias，直接加在累加器上 |
| **Activation** | ReLU / ReLU6 / SiLU(Swish) / GELU / HardSwish（MobileNetV3、Mamba 常用 SiLU）|
| **Residual / Skip add** | `y = act(conv(x)) + x`，注意通道数与长度对齐 |
| **Scalar / clamp** | 量化的 `clamp`、dequant scale |
| **RMSNorm / LayerNorm / GroupNorm** | 放在 DW 之后（ConvNeXt 风格），需要沿通道维度 reduce |

</details>

<details>
<summary><b>2. Producer（卷积之前）</b></summary>

| 融合 op | 说明 |
|--------|------|
| **Pointwise conv (1×1) / Linear** | Depthwise-Separable 的 `PW → DW`，PW 输出直接在寄存器里当 DW 输入 |
| **Split / Gate** | 门控卷积把输出拆成两半做 `a * σ(b)`（GLU，音频模型常见）|
| **上一层 dwconv** | 堆叠 depthwise（中间带 activation）|

</details>

<details>
<summary><b>3. 结构性融合（整个 block）</b></summary>

- **MBConv 1D**：`PW1 → DW → SiLU → PW2 → +skip`（MobileNetV2 倒残差块的 1D 版）
- **Gated Conv**：`x → split → (conv, σ(conv)) → 相乘`
- **SSM/Mamba 前置**：`Linear proj → causal dwconv1d → SiLU`（causal 意味着 `PAD_L = K-1`，仅左侧 padding）
- **Conv → 下采样**：`stride` 融合进卷积本身

</details>

---

## 二、主 Kernel：`dwconv1d + bias + activation + residual`

这是覆盖度最高的融合形式（BN 已在外部 fold 成 bias）。支持 `stride`、`dilation`、任意 `padding`、多种激活。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _act(x, ACT: tl.constexpr):
    if ACT == 1:        # ReLU
        return tl.maximum(x, 0.0)
    elif ACT == 2:      # SiLU / Swish
        return x * tl.sigmoid(x)
    elif ACT == 3:      # GELU (tanh 近似: 0.5x(1+tanh(z)) = x·σ(2z))
        z = 0.7978845608 * (x + 0.044715 * x * x * x)
        return x * tl.sigmoid(2.0 * z)
    elif ACT == 4:      # ReLU6
        return tl.minimum(tl.maximum(x, 0.0), 6.0)
    else:
        return x


@triton.jit
def dwconv1d_fused(
    x_ptr, w_ptr, b_ptr, r_ptr, y_ptr,
    B, C, L, Lout,
    sx_b, sx_c, sx_l,
    sr_b, sr_c, sr_l,
    sy_b, sy_c, sy_l,
    STRIDE: tl.constexpr,
    DILATION: tl.constexpr,
    PAD_L: tl.constexpr,
    K: tl.constexpr,
    ACT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_bc = tl.program_id(0)          # over B*C  (B*C 并行，通道互不依赖)
    pid_l = tl.program_id(1)           # over L blocks

    b = pid_bc // C
    c = pid_bc % C

    x_base = x_ptr + b * sx_b + c * sx_c
    y_base = y_ptr + b * sy_b + c * sy_c

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    out_mask = offs_l < Lout

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # ---- depthwise 卷积：逐 k unroll，input halo 用 mask 处理边界 ----
    for k in tl.static_range(K):
        in_l = offs_l * STRIDE + k * DILATION - PAD_L
        m = (in_l >= 0) & (in_l < L) & out_mask
        xv = tl.load(x_base + in_l * sx_l, mask=m, other=0.0)
        wv = tl.load(w_ptr + c * K + k)          # weight 视图为 [C, K]
        acc += xv * wv

    # ---- epilogue: bias + activation ----
    acc = acc + tl.load(b_ptr + c)
    acc = _act(acc, ACT)

    # ---- residual add ----
    if HAS_RESIDUAL:
        r_base = r_ptr + b * sr_b + c * sr_c
        rv = tl.load(r_base + offs_l * sr_l, mask=out_mask, other=0.0)
        acc = acc + rv

    tl.store(y_base + offs_l * sy_l, acc, mask=out_mask)
```

Python 调用封装（含 BATCH NORM FOLD 与 causal 支持）：

```python
_ACT_MAP = {"none": 0, "relu": 1, "silu": 2, "gelu": 3, "relu6": 4}


def fold_bn(weight, bn):
    """把 BatchNorm fold 进 depthwise 权重：推理期零成本。"""
    w = weight.float()
    gamma, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    scale = gamma / torch.sqrt(var + eps)
    w = w * scale.view(-1, 1)
    b = beta - gamma * mean / torch.sqrt(var + eps)
    return w, b


def dwconv1d_fused_forward(
    x, weight, bias=None, residual=None,
    stride=1, dilation=1, padding=None,
    activation="silu", causal=False, block_l=128,
):
    B, C, L = x.shape
    K = weight.shape[-1]
    if padding is None:
        padding = K - 1 if causal else (K - 1) * dilation // 2  # causal 只左 padding
    Lout = (L + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    y = torch.empty((B, C, Lout), device=x.device, dtype=x.dtype)
    if bias is None:
        bias = torch.zeros(C, device=x.device, dtype=x.dtype)
    has_res = residual is not None
    r = residual if has_res else x                      # 占位，HAS_RESIDUAL=False 不会读

    assert weight.is_contiguous(), "weight 需为 [C, K] 或 [C, 1, K] 连续"
    grid = (B * C, triton.cdiv(Lout, block_l))
    dwconv1d_fused[grid](
        x, weight, bias, r, y,
        B, C, L, Lout,
        x.stride(0), x.stride(1), x.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        STRIDE=stride, DILATION=dilation, PAD_L=padding,
        K=K, ACT=_ACT_MAP[activation], HAS_RESIDUAL=has_res,
        BLOCK_L=block_l,
    )
    return y
```

> 关键点：`stride_xc`/`stride_yc` 让 kernel 兼容 `[B,C,L]`（NCL）与 `[B,L,C]` 的通道优先布局——只要外部传对应 stride 即可；若要 NLC 上更好的合并访存，可把 `BLOCK_L` 放到最后一维并将 `C` 作为并行轴。

---

## 三、结构性融合：`PW1 → DW → SiLU → PW2 (+skip)` 的写法思路

1D 的 MBConv 块无法在单个 DW kernel 里塞进大通道矩阵乘，常见做法是**两级融合**：

```text
Kernel A:  PW1(1x1) + bias              -> 写出中间张量 t  [B, Cmid, L]
Kernel B:  DW(t) + bias + SiLU + PW2 + skip   -> y
```

其中 Kernel B 把 DW 的 halos 加载后，在 epilogue 里做 `PW2` 的缩减（沿 `Cmid` 维做 `tl.dot` 或 `tl.sum`），再叠加 skip。核心代码骨架：

```python
@triton.jit
def dwconv_pw_fused(...):
    # 1) DW: 沿 L 取 [BLOCK_L]，逐 k unroll，得到 dw 输出 [Cmid_tile, BLOCK_L]
    # 2) act: SiLU
    # 3) PW2: 对 Cmid 全量用 tl.dot(dw_out, W2[c_out, :].T) -> [COUT_tile, BLOCK_L]
    # 4) skip: tl.load(residual) + acc，store
    ...
```

若 `Cmid` 不大（≤256），可以把 `Cmid` 整维放进一个 program 内，**真正把整块融成 1 个 kernel**，省掉 `t` 的显存读写——这是 MobileNet 类模型在推理引擎里最常见的优化点。

**Gated Conv** 同理：DW 输出一次性算出 `2·Cout`，前半走 activation、后半走 sigmoid gate，寄存器内相乘即可。

---

## 四、实践注意事项

- **BN fold 优先**：只在训练期/`eval` 前做一次 fold，比在 kernel 内算 BN 便宜得多。
- **`K` 用 `tl.static_range` unroll**：depthwise 的 `K` 通常 3~7，展开后指令级并行好，边界 mask 也便于常量折叠。
- **边界 mask 代价**：`mask` 在 `L` halo 区域会引入分支；若 `L` 足够大且 padding 很小时，可把 body/edge 拆成两个 kernel 或直接用 `tl.load` 的 `other=0.0` 掩盖。
- **Causal（Mamba/音频）**：`PAD_L = K-1`，只在序列左侧补零；此时 grid 的 L 维不做中央 padding。
- **布局**：`stride` 参数化让同一 kernel 同时适配 NCL / NLC，避免额外 transpose kernel（transpose 本身也常和 DW 融合掉）。
- **调参**：`BLOCK_L` 一般取 128/256，配合 `triton.autotune` 在 `num_warps`（2/4/8）上搜索。
- **数值**：累加器用 `fp32`，store 前 cast 回 `fp16/bf16`，特别是 `[B,C,L]` 长序列时。

如果需要，我可以把 **整块 MBConv（PW1→DW→PW2→skip）单 kernel** 的完整 Triton 实现也写出来。
