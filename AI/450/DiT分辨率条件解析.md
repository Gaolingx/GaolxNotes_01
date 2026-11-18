# DiT分辨率条件解析

## 🧑‍💻 User

为什么对于DiT来说分辨率也是一种condition signal？

---
## 😀 默认助手 (7)

## 先澄清一件事：原版 DiT 其实**没有**把分辨率作为显式条件

在 `DiT`(Peebles & Xie, 2022) 的正文里，条件只有两个：timestep $t$ 和 class label $c$，通过 adaLN-Zero 注入：

$$y=\text{MLP}\big(\text{emb}(t)+\text{emb}(c)\big),\qquad (\gamma,\beta,\alpha)=\text{Linear}(\text{SiLU}(y))$$

但他们是在 256 和 512 上**各训练一个独立模型**（Table 4 里两个 `DiT-XL/2`，Gflops 分别为 118.6 与 524.6），论文中也没有任何 resolution conditioning。因此严格说，分辨率在原始 DiT 里是一个**隐式条件**；把它**显式化成 condition signal** 的是后续的 DiT 系工作（SD3、PixArt、SDXL、各类视频 DiT）。

所以你的问题实质是：**为什么 DiT 这类架构（在混合分辨率训练/推理时）不得不把分辨率当成一种 condition？** 下面五条理由按重要性排序。

---

## 一、架构上：分辨率不是输入的一部分，而是决定输入形状的东西

Patchify 之后：

$$T=\Big(\frac{H}{p}\Big)\Big(\frac{W}{p}\Big),\qquad \text{token}_i \leftarrow \text{token}_i + \text{PE}(x_i,y_i)$$

分辨率一变，token 数 $T$ 变了，sincos 位置编码的集合也变了。但关键矛盾在于：

- 这些信息**只对 attention 可见**（attention 本身是 permutation-equivariant，几何信息全靠 pos-emb）；
- 而 DiT 的**全局调制通路**（adaLN 的 $\gamma,\beta,\alpha$）只看 $t$ 和 $c$，**对分辨率没有输入通道**；
- DiT 没有 global pooling，也没有 UNet 式的多尺度金字塔，因此**没有其他地方能把“目标尺度”这个全局标量传进去**。

换句话说，要想让“整块网络的行为随分辨率而变”（细节预算、锐度、纹理统计），唯一每层都存在的全局入口就是 conditioning 向量 $y$。这是 DiT 把分辨率塞进 condition 的**结构性原因**。

## 二、统计上：$p(x|c)$ 在混合分辨率下变成多峰分布

同一语义内容在 256 和 1024 下都出现，且互为近似重复样本（近乎“同一张图的两个版本”）。此时

$$p(x\mid c)=\sum_{r} p(r\mid c)\,p(x\mid c,r)$$

采样时模型看到的是混合分布的 score：$\nabla_x\log p(x\mid c)=\sum_r w_r(x)\nabla_x\log p(x\mid c,r)$，它**不是任何一个单一分辨率数据分布的 score**。当几个分量对同一像素位置给出冲突的高频内容时，加权平均的结果就是高频能量被削减，表现为整体发糊、细节不足、纹理糊成一片（mode averaging）。加上 $r$ 之后，$p(x\mid c,r)$ 重新变回单峰，这是一个非常“干净”的建模理由。

## 三、噪声/SNR 上：同一个 $t$ 在不同分辨率含义不同

如果把低分辨率图看成高分辨率的下采样，噪声也会一起被下采样：

$$\text{AvgPool}_{2\times2}\big(\epsilon\big)\sim\mathcal N\!\left(0,\frac{\sigma^2}{4}\right)\quad\Rightarrow\quad \text{std}=\frac{\sigma}{2}$$

所以在**同一个前向时刻 $t$ / 同一个 $\sigma$**（比如 EDM、rectified flow 的连续时间框架）下，“每单位像素能保留多少结构”与分辨率耦合：$t$ 单独**无法确定**去噪难度。这是 $t$ 与 $r$ 在信息上的直接纠缠，也是 SD3 里“shift 应当随分辨率变化”的同一件事（分辨率越高越需要把 shift 调大）。

## 四、频谱上：模型必须知道“高频带宽预算”

自然图像功率谱近似 $S(f)\propto 1/f^{2}$，不同分辨率的数据分布在高频段的能量占比完全不同（重采样会截断或混叠频谱）。扩散模型本质上在做**谱匹配**；如果不知道目标带宽，它只能选一个折中，结果要么不够锐、要么过度锐化。这也解释了为什么分辨率条件在实践中可以被当作“锐度或细节旋钮”使用：推理时把 $r$ 条件调高或偏低，就能得到更锐或更软的图。

## 五、工程上：单模型多分辨率训练已经成了默认配置

多桶（bucketing）训练、任意宽高比、超高分辨率外推、t2i 与 SR 统一到一个模型，以及视频 DiT 中把分辨率、时长和帧率当作同族元数据条件，都需要模型**显式知道目标尺寸**，否则无法在 bucket 之间稳定共享权重。附带好处是：对 $r$ 做条件 dropout（类似 CFG 的 null 条件），还能顺便获得分辨率外推能力。

---

<details>
<summary><b>为什么 DiT 比 UNet 更“依赖”显式分辨率条件（对比表）</b></summary>

| | U-Net（ADM/LDM） | DiT |
|---|---|---|
| 尺度信息的位置 | 下采样层级、每级通道数，尺度与输入尺寸强绑定 | 只有 token 数和 pos-emb |
| 是否有多尺度结构 | 有（分辨率金字塔天然“知道”自己在第几级） | 无，所有权重在所有 token 上共享 |
| 全局统计 | GroupNorm/BN 的统计量随输入尺寸变化，隐式携带尺度 | LayerNorm 逐 token 计算，**不带任何全局尺寸信息** |
| 全局调制 | adaGN（$t$/$c$ → scale/shift） | adaLN（$t$/$c$ → $\gamma,\beta,\alpha$）——**唯一全局通路，但输入里没有 $r$** |
| 结论 | 部分靠归纳偏置“感知”分辨率 | 必须显式把 $r$ 送进调制向量 |

还有一个常被忽略的实现细节：如果位置编码用的是**归一化坐标**（$u\in[0,1]$）而非像素坐标，那么 $32\times32$ 与 $64\times64$ 的 pos-emb 集合在连续意义上“长得一样”，模型几乎无从推断分辨率，此时显式条件的必要性更高；反之用像素坐标，模型只能从 pos-emb 的绝对尺度**隐约**推断，既不可靠，还与“细节预算”这种全局决策不匹配。

</details>

---

<details>
<summary><b>具体怎么加（adaLN 路线，SD3 的做法）</b></summary>

```python
class ResEmbedder(nn.Module):
    """Fourier features of (h, w) -> additive conditioning vector"""
    def __init__(self, dim, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(4 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, hw):                      # hw: (B, 2) float [h, w]
        x = hw[:, :, None] * self.freqs[None, None]      # (B,2,F)
        x = torch.cat([x.sin(), x.cos()], dim=-1)        # (B,2,2F)
        return self.mlp(x.flatten(1))

# DiT block 内
y = t_emb + c_emb + res_emb + ar_emb + crop_emb          # 全部相加后统一进 adaLN
shift, scale, gate = modulation(y).chunk(3, dim=-1)      # adaLN-Zero
x = x + gate * attn(modulate(norm1(x), shift, scale))
x = x + gate * mlp(modulate(norm2(x), shift, shift))

# 训练时对 r 做 dropout，推理可外推 / 当锐度旋钮
if random() < p_drop: res_emb = null_emb
```

要点：
- **和 $t$ 相加**，而不是单独一路：这样 adaLN 的 MLP 能自动学出 $t$ 与 $r$ 的交互，正好对应“同一 $t$ 在不同分辨率下 SNR 不同”这一耦合。
- 用 **Fourier/正弦编码**而非连续标量，因为 $r$ 是离散 bucket，正弦编码在 bucket 间更平滑，便于外推。
- 别忽略 **aspect ratio 与 crop 参数**：SDXL 证明了 size 与 crop 元数据一起条件化对美学质量提升巨大，它们和分辨率属于同一族的“形状元数据”。

</details>

---

### 一句话总结

分辨率在 DiT 里之所以是（也应当是）condition signal，是因为它是**唯一无法被架构本身表达的全局属性**：token 数和 pos-emb 只能让 attention“局部地”看到尺寸，而细节预算、频谱目标、有效 SNR 这些都需要**全局调制通路**来决策；而 DiT 恰好把这条通路全部押在了 adaLN 的 conditioning 向量上。混合分辨率训练又让 $p(x\mid c)$ 变成多峰分布，使得不加分辨率条件时，模型只能给出多个分布的加权平均，表现为整体发糊。
