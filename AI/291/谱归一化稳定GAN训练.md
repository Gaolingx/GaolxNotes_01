# 谱归一化稳定GAN训练

### 🧑‍💻 User

为什么在卷积层上使用 Spectral Normalization 限制判别器的 Lipschitz 常数作为正则化手段对gan 训练有益？

---

### 🤖 Assistant

# Spectral Normalization 在 GAN 判别器中的作用

## 1. 核心动机：Lipschitz 连续性与训练稳定性

GAN 训练的本质是一个极小极大博弈：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

训练不稳定的根本原因之一在于：**判别器如果不受约束，其梯度可以任意大**，导致：

| 问题 | 后果 |
|------|------|
| 梯度爆炸 | 生成器参数更新剧烈震荡 |
| 判别器过强 | 提供给生成器的梯度信号退化（vanishing gradients） |
| 模式坍塌 | 生成器只学到少数几种输出 |

---

## 2. 理论基础：Wasserstein 距离与 Lipschitz 约束

从 WGAN 的理论出发，最优判别器（critic）需要满足 **1-Lipschitz 约束**：

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

这里 $\|f\|_L \leq 1$ 意味着：

$$\|f(x_1) - f(x_2)\| \leq \|x_1 - x_2\|, \quad \forall x_1, x_2$$

即使在**标准 GAN（非 WGAN）**中，控制判别器的 Lipschitz 常数同样有助于稳定训练，因为它本质上控制了判别器函数的**平滑度**。

---

## 3. Spectral Normalization 的具体机制

### 3.1 对单层线性变换的谱范数

对于权重矩阵 $W$，其**谱范数**定义为最大奇异值：

$$\sigma(W) = \max_{h: h \neq 0} \frac{\|Wh\|}{\|h\|} = \sigma_{\max}(W)$$

Spectral Normalization 将权重归一化为：

$$\bar{W} = \frac{W}{\sigma(W)}$$

归一化后，该层作为映射的 Lipschitz 常数恰好为 1：

$$\|\bar{W}h_1 - \bar{W}h_2\| \leq \sigma(\bar{W}) \|h_1 - h_2\| = \|h_1 - h_2\|$$

### 3.2 多层网络的 Lipschitz 常数

判别器通常是多层复合函数：

$$D(x) = a_L(W_L \cdot a_{L-1}(W_{L-1} \cdots a_1(W_1 \cdot x)))$$

其中 $a_l$ 是激活函数（如 ReLU，Lipschitz 常数为 1）。由 Lipschitz 常数的**链式法则**：

$$\|D\|_{\text{Lip}} \leq \prod_{l=1}^{L} \sigma(W_l) \cdot \prod_{l=1}^{L} \|a_l\|_{\text{Lip}}$$

当每层都做 Spectral Normalization 后（$\sigma(\bar{W}_l) = 1$）且激活函数的 Lipschitz 常数为 1 时：

$$\boxed{\|D\|_{\text{Lip}} \leq 1}$$

### 3.3 卷积层的情况

对于卷积层，卷积操作可以表示为 Toeplitz 矩阵（或分块循环矩阵）乘法。设卷积核为 $w$，对应的等效矩阵为 $W_{\text{conv}}$，则：

$$\sigma(W_{\text{conv}}) = \max_k |\hat{w}(k)|$$

其中 $\hat{w}(k)$ 是卷积核的傅里叶变换。实际实现中，通常用**幂迭代法（power iteration）** 来高效近似最大奇异值，而不需要显式构造 Toeplitz 矩阵。

---

## 4. 为什么 Spectral Normalization 优于其他方法？

### 与其他 Lipschitz 约束方法的对比

```
方法                  |  计算开销  |  约束精确性  |  对容量的影响
─────────────────────|────────── |────────────|────────────
Weight Clipping      |  极低     |  过于粗暴    |  严重限制（rank collapse）
Gradient Penalty     |  高       |  仅在采样点  |  较好但计算昂贵
Spectral Norm        |  低       |  全局上界    |  温和且有效
```

<details>
<summary><b>Weight Clipping 的问题（点击展开）</b></summary>

WGAN 原始论文采用的 weight clipping 将权重硬截断到 $[-c, c]$：
- 导致权重趋向于 $\pm c$ 的两极分布
- 判别器容量被严重压缩，倾向于学习极其简单的函数
- 对超参 $c$ 极其敏感

</details>

<details>
<summary><b>Gradient Penalty 的问题（点击展开）</b></summary>

WGAN-GP 通过惩罚项约束梯度：
$$\lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

- 需要额外计算梯度（反向传播中的反向传播），**计算代价高**
- 仅在采样的插值点 $\hat{x}$ 上施加约束，不是全局保证
- 无法与某些架构（如 BatchNorm）兼容

</details>

### Spectral Normalization 的核心优势

1. **计算高效**：幂迭代每步只需 $O(mn)$，一次迭代即可（实践中足够）
2. **全局约束**：对整个输入空间提供 Lipschitz 上界保证
3. **最小化容量损失**：只约束最大奇异值，不改变奇异向量的方向结构
4. **实现简单**：几行代码即可

---

## 5. 直觉理解：为什么控制平滑度有用？

```
未正则化的判别器:                  Spectral Norm 正则化后:
                                  
  D(x)                              D(x)
   │    /\                            │      ___
   │   /  \    ← 陡峭梯度              │    /     \    ← 平滑梯度
   │  /    \                           │  /        \
   │ /      \                          │/           \
   └──────────── x                     └──────────── x
       ↑                                    ↑
  梯度信号极不稳定                      梯度信号平稳有意义
  真假边界附近梯度爆炸                  生成器能获得持续有效的学习信号
```

**关键直觉**：

- 不受约束的判别器可以在真实数据与生成数据之间形成**极陡峭的决策边界**
- 此时生成器的梯度 $\nabla_\theta \mathbb{E}[\log(1-D(G(z)))]$ 要么消失（远离边界）要么爆炸（在边界上）
- Spectral Normalization 迫使判别器**不能变化太快**，从而在整个空间中提供**信息丰富且稳定的梯度信号**

---

## 6. 代码实现示例

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # 每个卷积层都应用 spectral_norm
            spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),  # Lip = 1

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
        )
        # 注意：不使用 BatchNorm（它会破坏 Lipschitz 约束）

    def forward(self, x):
        return self.main(x)
```

> **注意**：`spectral_norm` 内部通过幂迭代（默认 1 步）估算 $\sigma(W)$，并在每次 `forward` 时动态归一化权重。

---

## 7. 总结

$$\boxed{\text{Spectral Normalization 有益于 GAN 训练} \iff \text{以最小代价控制判别器平滑度，稳定对抗博弈}}$$

核心因果链：

> **谱归一化** $\xrightarrow{\sigma(\bar{W})=1}$ **每层 Lipschitz $\leq 1$** $\xrightarrow{\text{链式法则}}$ **整体 Lipschitz 有界** $\xrightarrow{}$ **判别器函数平滑** $\xrightarrow{}$ **梯度信号稳定** $\xrightarrow{}$ **生成器有效学习** $\xrightarrow{}$ **训练稳定 + 高质量生成**