# Kimi K3 SiTU-GLU与SwiGLU核心区别解析

## 🧑‍💻 User

kimi k3的SiTU和常规的SwiGLU有什么区别？

---
## 😀 默认助手 (7)

## 先说结论

Kimi K3 用的 **SiTU-GLU** 本质仍是 GLU 家族（三个矩阵 × 一个门控乘一个 up 分支），但它把 SwiGLU 里的 **SiLU（$x\sigma(x)$，单侧无界）换成了一个双侧有界的“Sigmoid-Tanh 门”**，并且对 up 分支额外做了一次 **软裁剪（soft-clip）**。也就是说：**SwiGLU = 线性门控 × 线性通路；SiTU-GLU = 饱和门控 × 软饱和通路。**

依据（不是传闻）：K3 模型卡明写 `Activation Function: SiTU-GLU`，`config.json` 里是 `"hidden_act": "situ"`、`activation_situ_beta: 4.0`、`activation_situ_linear_beta: 25.0`，HF 官方 `modeling_kimi_linear.py` 和 vLLM 的 `situ_and_mul` kernel 都给了逐行实现，三处完全一致。

---

## 公式层面的区别

**SwiGLU（常规）**，设 $g=\text{gate}(x),\,u=\text{up}(x)$：

$$
\text{SwiGLU}(x)=\underbrace{g\,\sigma(g)}_{\text{SiLU}(g)}\cdot u
$$

**SiTU-GLU（K3）**，$\beta=4,\ \beta_{\text{lin}}=25$：

$$
\text{SiTU}(g)=\beta\tanh\!\left(\frac{g}{\beta}\right)\sigma(g)
$$

$$
\text{SiTU-GLU}(x)=\text{SiTU}(g)\cdot\Big[\beta_{\text{lin}}\tanh\!\left(\frac{u}{\beta_{\text{lin}}}\right)\Big]
$$

对应官方实现（简化自 HF `SituAndMul`）：

```python
gate_up = torch.cat([w1(x), w3(x)], dim=-1)   # 一次 GEMM 出两半
g, u = gate_up.chunk(2, dim=-1)
situ_a = beta * torch.tanh(g / beta) * torch.sigmoid(g)   # 门：双侧有界
u = linear_beta * torch.tanh(u / linear_beta)             # up：软裁剪 ≈ 恒等
out = situ_a * u
```

注意 $\beta$ 是**固定常数**（来自 config），不是可学习参数，所以不增加任何参数或 FLOPs。

---

## 四处实质差异

| 维度 | SiLU / SwiGLU | SiTU-GLU（K3） |
|---|---|---|
| 正半轴 | $x\sigma(x)\to x$，**线性无界** | $\to \beta=4$，**饱和到常数** |
| 负半轴 | $\to 0^-$，最小值约 $-0.28$ | $\to 0^-$，最小值约 $-0.27$（形状几乎一样） |
| 近零区 | $x\sigma(x)$ | $\beta\tanh(x/\beta)\approx x-\frac{x^3}{3\beta^2}$，**几乎与 SiLU 重合** |
| 导数谱 | 峰值 $1.0998$ @ $x\approx2.40$ | 峰值 $0.9264$ @ $x\approx1.49$（更“收缩”） |
| up 分支 | 恒等 $u$ | 软裁剪到 $\pm25$（$u{=}10\to9.5$，$u{=}50\to24.1$，$u{=}100\to24.98$） |

要点浓缩：

1. **单侧无界 → 双侧有界。** SiLU 让大激活通道无限放大（$x{=}12$ 时值 12），SiTU 在 $x{\gtrsim}3\beta$ 后就基本“归 4”。门控从“增益”变成“**带最大开度的阀门**”，单个 FFN 通道的输出上界约为 $4\times25=100$。
2. **活跃区几乎等价，只压缩长尾。** $|x|\lesssim2$ 时两者偏差只有百分之几（$x{=}2$：$1.762$ vs $1.628$），负半轴的“小凹陷”几乎一样。所以它不是把激活改成完全不同的东西，而是**只重写了分布尾部**——这也是它能“免费换装”的原因。
3. **梯度侧更温和。** 门控导数峰值从 1.10 降到 0.93、且左移；负半轴极小值略深（$-0.105$ vs $-0.100$）。整体是更收缩的映射，深层堆叠时对残差流方差的放大更小。
4. **up 分支是软 clamp 而非硬 clamp。** 这点常被混淆：gpt-oss 的 “SwiGLU-OAI” 是 **硬截断** $\text{clamp}(\cdot,\pm7)$ 且 $(u+1)$；K3 用 $\tanh$ **光滑压缩**，$\beta_{\text{lin}}{=}25$ 很大，$\pm10$ 内几乎无感，只在真正的大离群值上起作用。

<details>
<summary>数值对照（含导数，$x$ 为门控输入）</summary>

| $x$ | SiLU | SiTU($\beta$=4) | dSiLU | dSiTU |
|---|---|---|---|---|
| -4 | -0.072 | -0.055 | -0.053 | -0.046 |
| -2 | -0.238 | -0.220 | -0.091 | -0.100 |
| -1 | -0.269 | -0.264 | 0.072 | 0.060 |
| 0 | 0 | 0 | 0.500 | 0.500 |
| 1 | 0.731 | 0.716 | 0.928 | 0.880 |
| 2 | 1.762 | 1.628 | 1.091 | 0.887 |
| 4 | 3.928 | 2.992 | 1.053 | 0.466 |
| 8 | 7.997 | 3.855 | 1.002 | 0.072 |
| 12 | 12.000 | 3.980 | 1.000 | 0.010 |

</details>

---

## 为什么要这么改（动机与代价）

- **量化友好的激活范围。** K3 从 SFT 起就做 **MXFP4 权重 / MXFP8 激活** 的量化感知训练。SwiGLU 的无界通道是低精度训练里最典型的离群值来源；把门和 up 都做成有界后再乘，激活动态范围被结构性压住，MXFP8 的 block scale 更好用——这和它 2.8T/104B 激活、“Stable LatentMoE”那套稳定性叙事是同一路线。
- **收益/成本**：结构、参数量、计算量都没变（仍是 `gate`、`up`、`down` 三个矩阵），只是多一个 `tanh`，工程上被 fused kernel 吃掉（vLLM 有 `situ_and_mul` 的 CUDA/XPU 融合算子，直接写 `out`，无 fp32 中间张量）。
- **一个实现细节**：`gate`/`up` 被拼成一个 `gate_up` 张量走单次 GEMM 再切分，这是所有 fused GLU kernel（`silu_and_mul` 也一样）的通用布局，**不是** SiTU 的语义差异，别当成架构创新读。

---

## 需要注意的两点

1. 严格说这是**两处**改动：门控激活换成 SiTU + up 分支软裁剪。只谈“SiTU vs SiLU”会漏掉后者。
2. 我核对的是模型卡、官方 `config.json` / `modeling_kimi_linear.py`、vLLM 实现三处一手代码；K3 技术报告（arXiv:2607.24653）正文里的消融数字我没能读到（HTML 页面超出抓取体积上限），所以上面关于“为什么这么设计”的动机部分是**基于实现的推断**，不是报告原文的表述。若你要引用具体增益（比如某 benchmark 上 SiTU 相对 SwiGLU 的差值），需要去报告正文的 activation 消融表里取数。

---
## 🧑‍💻 User

那SiTU-GLU能不能从根源上解决transformer训练过程中residual stream growing和channel异常值的问题？

---
## 😀 默认助手 (7)

## 直接回答

**不能从根本上解决。** 它切断的是残差增长链条中的**一个放大器**，而且是在结构上正确的一次切断；但 `residual stream growing` 和 `channel 异常值` 的根因不在 FFN 的激活函数里，所以界限虽然存在，却几乎不会是真正起作用的约束（non-binding）。

下面从数学上区分“它改变了什么”和“它没有改变什么”。

---

## 一、先看根因：增长与 outlier 来自哪里

Pre-LN 残差：$h_{\ell+1}=h_\ell+f_\ell(\mathrm{RMSNorm}(h_\ell))$，于是

$$
\|h_L\|^2=\|h_0\|^2+\sum_{\ell}\|f_\ell\|^2+2\sum_{\ell<\ell'}\langle f_\ell, f_{\ell'}\rangle
$$

如果各子层输出近似正交，$\|h_L\|\approx\Theta(\sqrt L\cdot\overline{\|f\|})$。**这种 $\sqrt L$ 量级的增长是恒等路径的设计目的**（信息不被 $L$ 层反复覆盖），不是缺陷。

真正危险的情形是**齐次放大**：若 $\|f_\ell(h)\|$ 在高激活区**与 $\|h\|$ 成正比**且比例 $\lambda>0$，则 $\|h_L\|\approx(1+\lambda)^L\|h_0\|$，指数爆炸。而 SwiGLU 在高激活区**恰好是 1 次正齐次的**：$t\to\infty$ 时 $\mathrm{SiLU}(tg)\to t\,\mathrm{SiLU}(g)$，所以 $f(t h)\approx t f(h)$。这是一个**尺度共变、没有有界吸引子**的映射；它对尺度是中性或边际不稳定的。

outlier 的根因更不在这里，而在这三条：
1. **softmax 的指数敏感性**：只要 QK 在某方向上对齐，就能得到几乎饱和的注意力权重，也就是 attention sink。支付一次“大常数”就能换来稳定的 sink，性价比极高。
2. **RMSNorm 全局尺度不变性**：残差的整体量级在数学上几乎“免费”，损失对 $\|h\|$ 的梯度上没有惩罚，因此没有压力去约束它。
3. **用权重编码近常数偏置更便宜**：massive activation 是**与输入无关**的常数偏置，由某个 `down_proj` 通道施加。这条路径的“承载能力”由**权重范数**决定，跟激活上界无关。

---

## 二、SiTU 真正做到的（这部分有实质价值）

**1. 破坏正齐次性，把指数或线性累积换成有常数上界的逐层增量。**

$$
\lim_{|g|\to\infty}\text{SiTU}(g)=\begin{cases}4 & g\to+\infty\\ 0 & g\to-\infty\end{cases}
$$

所以 $\|\text{SiTU-GLU}(x)\|_\infty \le 4\times 25 = 100$，与输入尺度无关。于是

$$
\|h_L\|_\infty \le \|h_0\|_\infty + L\cdot 100\cdot\max_j\|W^{\text{down}}_{j\cdot}\|_1
$$

增长从“可能指数”退化为“**至多线性，且每一项都有与输入无关的常数帽**”。这是一个真实的结构性改变，SwiGLU 给不出这样的帽。

**2. 给 outlier 通道“断粮”（梯度侧）。** $\text{SiTU}'$ 的峰值只有 $0.926$（$x\approx1.49$），且在 $x=12$ 时降到 $0.0099$（SiLU 是 $1.0001$）。一个大激活单元一旦进入饱和区，**它自己的梯度会掉两个数量级**，于是它既不会再继续变大，也不会把 `gate_proj` 的权重推得更大。这是**自阻尼**，而不是硬裁剪，因此比 clamp 更温和。

**3. 对 MXFP8 的 block scale 是真正的约束（binding）。** 这是唯一“硬兑现”的收益：每 32 个元素共享一个 scale，而 SiLU 的右尾可以单点跑到 $10^3$，把一个 block 里其余 31 个元素的有效位宽破坏到接近零；SiTU 把每个元素确定地压到 $\pm100$ 内，block scale 的可预测性直接决定量化精度。**K3 从 SFT 起做 MXFP4/MXFP8 QAT，这个收益是刚需。**

---

## 三、SiTU 做不到的（为什么说“不能从根源”）

| 逃逸路径 | 为什么 SiTU 管不着 |
|---|---|
| `down_proj` 是线性映射 | 有界输入 $\times$ **无界权重** = 无界输出。$\lvert y_j\rvert\le 100\sum_i\lvert W_{ji}\rvert$，d=3072、$\sigma\!\sim\!0.02$ 时右端仍在 $10^3\sim10^4$——**与观测到的 massive activation 同量级，这个界限根本不 binding** |
| outlier 迁移到权重 | 网络要“大常数”的需求没变。激活侧被堵住，最省力的解是**把 `down_proj` 那一行整体放大**。于是 residual outlier 照旧出现，而 MXFP4 的**权重**异常值反而更重 |
| attention 侧完全不管 | sink / QK outlier 是 softmax 结构决定的，SiTU 在 MLP 里，与之正交 |
| 逐元素界限不等于范数界限 | 每元素 $\le100$ 但 $d$ 个通道同向时 $\|f\|\le100\sqrt d\approx5.5\times10^3$，没有任何实质约束 |
| 有界不等于有惩罚 | RMSNorm 下残差量级仍近乎“免费”；没有显式正则（z-loss、spectral norm、branch-output norm）时，增长只被“变得不方便”而非“被禁止” |
| 代价 | 饱和区梯度 $\sim10^{-2}$ → 高幅特征可能变成**死通道**，容量损失；极端情况下稳定性换来的是表达力的浪费 |

**一句话**：SiTU 把 outlier 的**出口**收窄了，但没有减少对 outlier 的**需求**，也没有管住**其他出口**（权重、attention、down_proj 行）。

---

## 四、K3 自己的答案：它是组合拳里的一环，不是银弹

看 `config.json` + `modeling_kimi_linear.py` 能直接读出来，Moonshot 在残差流上至少叠了四层：

- **Attention Residuals（AttnRes）**——`attn_res_block_size: 12`，在 block 边界缓存 prefix sum，用 `softmax` 对历史残差做**归一化重加权读取**。这才是针对 “residual stream 稀释与增长” 的**根治型机制**：它把“累积”换成“带归一化的、可学习的加权和”，从原理上限制了 $\|h_L\|$ 的无界累积，而不只是限制增量。
- **`latent_moe_use_norm: true`**——对 routed expert 在 3584 维 latent 空间输出做 RMSNorm。这比 SiTU 更直接：**直接把分支输出归一化到固定尺度**，再 `up_proj` 回 7168 维。
- **MLA output gate**（`mla_use_output_gate`，sigmoid）——给 attention 输出装了门；**KDA `gate_lower_bound: -5.0`**（`safe_gate`）——给线性注意力的对数衰减率设下界。
- **MXFP4/8 QAT + `qk_l2norm`**——量化与 QK 归一化在数值侧兜底。

换句话说：**官方把“增长”交给 AttnRes 和 branch-norm 治理，把“局部数值放大器”交给 SiTU 治理。** 只看到 SiTU 就以为找到了根治方案，是把工程分工读成了因果。

---

## 五、如果要更接近“根治”，缺的是这些

1. **分支输出归一化/尺度约束**（最有效）：$f_\ell$ 后接 RMSNorm 或 spectral norm 约束，直接杀掉齐次放大项，把它变成 $\sqrt L$ 增长而非 $(1+\lambda)^L$。
2. **残差流的显式归一化读**（AttnRes 类）：把“累积”改成“加权读取”。
3. **attention 侧**：QK-norm、logit soft-cap（Gemma-2 式）、显式 sink/register token。
4. **权重侧**：weight decay 的非对称处理、muP 式 spectral scaling——因为 outlier 会逃到 `down_proj`。
5. **承认 $\sqrt L$ 增长是特性**：目标是“有界可控”，不是“归零”。把 residual norm 压成常数会直接削掉深层网络的可用深度。

<details>
<summary>一个可检验的预测（如果你要验证 SiTU 是否真的起作用）</summary>

若 SiTU 的自阻尼机制生效，应当同时观察到：

- **FFN 预激活右尾被削平**：`gate` 分支分布尾部在 $|x|\gtrsim8$ 处快速衰减，而不是 SiLU 那样的长尾；
- **`down_proj` 权重行范数右尾变重**：outlier 从激活迁移到权重；
- `gate_proj` 中饱和单元的权重更新幅度显著小于非饱和单元（梯度差 $\sim100\times$）。

如果第 2 条不出现、第 1 条出现，说明 SiTU 确实在转移瓶颈；如果两条都出现不了，说明 SiTU 在这套规模下主要只兑现了**量化友好**这一个收益。

</details>

---

**结论**：SiTU-GLU 是一次**方向正确、边界诚实的局部修复**——它消除了 SwiGLU 在高激活区的 1 次正齐次性（这是可以严格写出来的结构差异），给每层 FFN 增量装上了与尺度无关的常数帽，并顺带解决了 MXFP8 的 block-scale 问题。但 residual growing 的根因是恒等路径 + 尺度不变性，channel outlier 的根因是 softmax 需求 + 线性映射 + 权重自由度；两者都能绕过 SiTU。**它是约束，不是根治。**

---
## 🧑‍💻 User

为什么配合Attention Residuals（AttnRes）可以一定程度缓解？

---
## 😀 默认助手 (7)

## 一句话机制

**SiTU 限制的是“每层写进去多少”，AttnRes 限制的是“每层读出来多少”，并把跨层聚合从“求和”换成“凸组合（≈取最大）”。** 残差增长是“写”和“聚合”两件事一起造成的，只堵一边必漏；两个一起堵，增长的量级才能从 $O(L\cdot B)$ 降到 $O(\text{常数})$。

下面按官方 `modeling_kimi_linear.py` 里的实现讲，不靠想象。

---

## 一、先看 AttnRes 实际做了什么

核心函数只有十几行：

```python
def _apply_attn_res(prefix_sum, block_residual, proj, norm):
    # block_residual: (T, K, H)  —— 每 12 层存一个“块摘要”
    # prefix_sum:     (T, H)     —— 当前块内累积
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)  # (T, K+1, H)
    k = v * rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)   # 逐候选 RMSNorm（不含 γ）
    scores = (k * (norm.weight * proj.weight.squeeze(0))).sum(-1)  # 打分，尺度不变
    probs  = scores.softmax(-1)                            # 归一化 → 凸组合权重
    return torch.matmul(probs.unsqueeze(1), v).squeeze(1)   # 混合“原始”候选
```

而 decoder 层的残差流被改成了这样：

```python
prefix_sum = hidden_states                       # 进入层时
if layer_idx % 12 == 0:                          # 块边界：快照 + 清零
    block_residual = cat([block_residual, prefix_sum], dim=1)
    prefix_sum = None
hidden_states = read(prefix_sum, block_residual) # 读：凸组合
hidden_states = attn(input_layernorm(hidden_states))
prefix_sum = prefix_sum + hidden_states           # 写：局部累加
hidden_states = read(prefix_sum, block_residual) # 读
hidden_states = mlp(post_layernorm(hidden_states))
prefix_sum = prefix_sum + hidden_states
return prefix_sum, block_residual                 # 返回的是块内局部和
```

请注意三点：读取出来的结果**不参与累加**；`prefix_sum` **每 12 层清零**；跨块的信息只通过 softmax 凸组合传递。

---

## 二、五个具体的缓解机理

**1. 聚合算子从 $\Sigma$ 变成 softmax 凸组合 ⇒ 上界从“求和”变成“取最大”**

普通 pre-LN：$\|h_L\|_\infty$ 是 $L$ 次写入的**和**。AttnRes：

$$
x=\sum_{i=0}^{K}p_i v_i,\quad \sum_i p_i=1,\ p_i\ge0\ \Longrightarrow\ \|x\|_\infty\le\max_i\|v_i\|_\infty
$$

一个通道的幅度**不能被放大、只能被稀释**：如果 $p_i$ 小，那个候选里的 outlier 就被按比例压掉。而普通残差里的 outlier 通道是**只增不减的滑动和**，一旦产生就跟随到最后一层。这是“异常值不那么粘”的根本原因。

**2. 每 12 层强制截断 ⇒ 上界与深度无关**

$h_\ell$ 的累积被切成 12 层一段（24 次子层写入），跨段不再相加。配合 SiTU 的每写有界 $B\approx 4\times25=100$：

$$
\|h\|_\infty \;\lesssim\; \|h_0\|_\infty + 24B \quad\text{（与 }L\text{ 无关）}
$$

$$L=93 \Rightarrow
\begin{cases}
\text{纯 pre-LN + SwiGLU：无上界（且高激活区 1-齐次，可指数）}\\
\text{pre-LN + SiTU：}O(L\cdot B)\approx 10^{4}\cdot\|W^{\text{down}}\|\\
\text{SiTU + AttnRes：}O(24B)\ \text{，depth-independent}
\end{cases}$$

这才是“一定程度”的准确含义：**它没有把 outlier 变没，而是把“随深度累积”这一项删掉了。**

**3. 打分在 RMSNorm 之后 ⇒ 尺度不变，删掉了“膨胀范数换路由”的畸变激励**

$s_i=\langle \mathrm{RMSNorm}(v_i),\ \gamma\rangle$ 只看**方向**。如果直接在原始幅值上打分，模型会发现“把某个块的范数做大”是赢得路由最省力的方式，就会形成正反馈，这正是 softmax attention 需要 QK-norm 的同一病理。AttnRes 的读路径相当于**残差流的 QK-norm**。因此它不会为 outlier 提供新的生存激励，而 SiTU 单独存在时，网络仍可通过放大 `down_proj` 行来“买”表达力。

**4. 需求侧：给“深度信息保真”提供了一条不依赖幅度的通道（这是补上 SiTU 缺的那一半）**

这是最关键的一条：残差之所以会长，很大程度是**模型需要跨层保存信息**，而“大常数偏置或大振幅”是它最便宜的编码方式。AttnRes 让第 10 层的信息到第 90 层只需**一次凸组合跳转**（$K{+}1\approx9$ 个候选的 softmax），不必穿透 80 次加法。于是：

> **SiTU 减少“供给”（放大器被限幅）；AttnRes 减少“需求”（不必用振幅当记忆）。**

两者在因果链的两端同时作用。SiTU 单独用会有一个副作用：饱和单元的梯度掉约 $100\times$（$x{=}12$ 时 $\text{SiTU}'\approx0.0099$ vs $\text{SiLU}'\approx1.0$），高幅特征容易变成“死通道”，容量损失；而 AttnRes 恰好给了这些信息另一条通路，容量的损失被缓和。

**5. 信息稀释被消除（一个常被忽略但直接的收益）**

普通 pre-LN 里 $\|h_\ell\|$ 随深度增长，而每层写入的 $f_\ell$ 幅值不变 ⇒ 新写入对残差的**相对贡献逐层衰减**，深层近似“只读”。AttnRes 让读到的输入幅值被拉回 $\max_i\|v_i\|$ 量级且有界，所以**每层的有效更新幅度不再随深度衰减**。这直接改善深层优化，也间接降低了“用极端幅值才能引起注意”的动机。

> 附带一句工程细节：`_apply_attn_res` 里 `v.float()` 把混合放到 fp32 做，再转回 bf16。所以即便某个候选含 $10^3$ 级通道，凸组合的累加**不会在 bf16 里溢出**——混合这一步本身就不放大 outlier。这是有意的数值设计。

---

## 三、协同的严格表述

设每层写入有界 $B$（由 SiTU 给出），块内最多 $2\cdot12=24$ 次写入，候选数 $K+1$：

$$
\underbrace{\|x\|_\infty\le\max_i\|v_i\|_\infty}_{\text{AttnRes：聚合不放大}}
\le\underbrace{24B}_{\text{SiTU：写入不无界}}
+\underbrace{\text{块首值}}_{\text{有界}}
$$

- 只有 SiTU：$\|h\|$ 有界但**随深度线性累积**，且 outlier 一旦写入就永久保留（$10^4$ 量级的界，实际上不 binding）。
- 只有 AttnRes：聚合不放大，但**单次写入可以无界**，一个病态的块摘要会污染所有读它的人（凸组合会以 $p\!\approx\!1$ 把它原样传出去）。
- 两者结合：**深度无关、且每项有常数帽**。这才是 K3 在 93 层、2.8T 参数、MXFP4/8 训练下能把激活范围控住的组合原因。

---

## 四、它仍然没有解决什么（避免过度乐观）

| 残留漏洞 | 说明 |
|---|---|
| 凸组合保留 max | 若 softmax 在某个候选上饱和（$p_i\to1$），读路径**退化为普通残差**。代码里对 score 幅值没有任何上界，衰减是“学出来的软约束”，不是硬 clamp |
| 块内仍是无界求和 | 12 层内的 24 次写入仍是加法，块内 blow-up 依然可能；只是被限制在 12 层尺度内 |
| 新增一个 outlier 劫持点 | 打分是 $\gamma=\gamma^{\text{norm}}\odot a$ 的**秩-1 读出**（每层仅 $2H$ 参数）。某个残差通道极大 + $\gamma_j$ 大 ⇒ 路由塌缩到单一块，构成新的失真模式 |
| 不约束 attention 侧 QK | sink / QK outlier 与 AttnRes 正交；那部分由 MLA output gate（sigmoid）与 KDA 的 `gate_lower_bound=-5.0` 兜 |
| 不约束权重 | `down_proj` 行的 L1 范数仍自由；outlier 的需求可以迁移到权重侧（MXFP4 下代价转移，不是消失） |

---

## 结论

**AttnRes 之所以能“一定程度缓解”，是因为它把残差流的代数结构从“加法累积”换成了“块内截断 + 尺度不变打分 + 凸组合读取”。** 于是：幅度**不能跨块累积**（depth-independent ceiling）、outlier **能被稀释而非永久携带**（$\max$ 而非 $\Sigma$）、路由**没有范数畸变激励**（RMS 归一化打分）、信息保真**不必依赖振幅**（减少 SiTU 治不了的“需求侧”）。

它与 SiTU 是**一个管供给、一个管需求 + 一个管聚合**的分工：SiTU 让每次写入有界，AttnRes 让写入不被累积、也更少被需要。但两者都是**软约束**——凸组合可以被饱和的 softmax 架空，`down_proj` 仍能把 outlier 搬到权重上，attention 侧完全不受益。所以准确说法是：**它把“随深度发散”这一类失败模式从架构上排除了，但把“outlier 存在”从架构上排除了则没有做到。**
