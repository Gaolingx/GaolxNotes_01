# Diffusion Model 原理深度解析

## 🧑‍💻 User

能帮我介绍一下diffusion model的原理吗？

---
## 😀 默认助手 (7)

# 扩散模型（Diffusion Model）原理

## 一、核心直觉

扩散模型的基本思路可以用一句话概括：**先学会怎么把数据一步一步加噪，直到它变成纯高斯噪声；然后把这个过程反过来学，就能从噪声一步一步还原出数据。**

它由两个过程组成：

| 过程 | 记号 | 方向 | 是否学习 |
|---|---|---|---|
| 前向扩散 | $q(x_t \mid x_{t-1})$ | 数据 → 噪声 | 否（固定公式） |
| 反向去噪 | $p_\theta(x_{t-1} \mid x_t)$ | 噪声 → 数据 | 是（神经网络） |

其中 $t = 1, \dots, T$，$T$ 通常取几百到一千步。$x_0$ 是真实数据，$x_T$ 近似标准高斯 $\mathcal{N}(0, I)$。

---

## 二、前向过程：逐步加噪

### 1. 单步转移

每一步都加入少量高斯噪声：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

等价地：

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
$$

$\beta_t \in (0,1)$ 是噪声调度（noise schedule），通常从 $\beta_1 \approx 10^{-4}$ 缓慢增长到 $\beta_T \approx 0.02$。注意系数 $\sqrt{1-\beta_t}$ 的设计是让方差保持大致不变（variance-preserving），否则信号会过快被淹没。

### 2. 关键性质：可一步到位

由于高斯分布的性质，任意时刻 $t$ 的边缘分布有封闭形式。记 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$，则

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
$$

也就是可以直接采样：

$$
\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon}
$$

**这一条是整个方法能高效训练的关键**：训练时不需要真的跑 $t$ 步，随机抽一个 $t$，一次就能造出 $x_t$。

由此也看出：$\sqrt{\bar\alpha_t}$ 是"信号保留比例"，$\sqrt{1-\bar\alpha_t}$ 是"噪声比例"。当 $\bar\alpha_T \to 0$ 时 $x_T \approx \mathcal{N}(0,I)$。

---

## 三、反向过程：学习去噪

反向过程也被建模为高斯马尔可夫链：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)
$$

实践中方差 $\Sigma_\theta$ 常固定为 $\sigma_t^2 I$（如 $\sigma_t^2 = \beta_t$ 或 $\tilde\beta_t$），只让网络预测均值 $\mu_\theta$。而均值通过"预测噪声"来实现：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)
$$

其中 $\epsilon_\theta(x_t,t)$ 是神经网络（U-Net / Transformer），输入"噪声图 + 时间步 $t$"，输出"这一步加了什么噪声"。

> 时间步 $t$ 必须作为条件输入，因为不同 $t$ 的去噪难度和噪声量级完全不同。

---

## 四、训练目标是怎么来的

<details>
<summary><b>点开：从 ELBO 到最简单的 MSE（推导细节）</b></summary>

### 1. 变分下界

最大化对数似然 $\log p_\theta(x_0)$，用变分下界（ELBO）：

$$
\log p_\theta(x_0) \ge \mathbb{E}_q\left[\log p_\theta(x_0\mid x_1) - \sum_{t=2}^{T} D_{KL}\big(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big) - D_{KL}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)\right]
$$

- 最后一项不含参数，可忽略（因为 $q(x_T|x_0)\approx\mathcal{N}(0,I)$）；
- 第一项是"重建项"；
- 中间每一项是 KL 散度，衡量"真实后验"和"模型预测"的差距。

### 2. 真实后验也是高斯的

前向过程的后验 $q(x_{t-1}\mid x_t, x_0)$ 也是高斯，均值和方差可以解析写出：

$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

两个高斯之间的 KL 散度有闭式解，所以整个损失就是"两个均值之间的加权 L2 距离"。

### 3. 重参数化为预测噪声

把 $x_0$ 用 $x_t$ 和 $\epsilon$ 表示：$x_0 = \dfrac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$，代入后可以发现：**预测均值等价于预测噪声 $\epsilon$**。

于是损失简化为

$$
L_{\text{simple}} = \mathbb{E}_{t\sim U(1,T),\ x_0,\ \epsilon}\left[\left\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\right\|^2\right]
$$

也就是：**随机取一张图、随机取一个时间步、随机取一个噪声，加噪后让网络把噪声猜出来。** 这就是 DDPM（Ho et al., 2020）最核心的一行公式。

DDPM 论文发现，去掉 ELBO 里每项的权重系数（即用 $L_{\text{simple}}$ 而不是完整 ELBO），生成质量反而更好——这是一个经验性的、但被广泛验证的取舍。

</details>

### 训练算法（伪代码）

```python
for x0 in dataloader:
    t = randint(1, T)                          # 随机时间步
    eps = randn_like(x0)                       # 随机噪声
    xt = sqrt(abar[t]) * x0 + sqrt(1 - abar[t]) * eps
    loss = mse(eps, model(xt, t))              # 预测噪声
    loss.backward(); optimizer.step()
```

---

## 五、三个等价视角（为什么它有效）

这是理解现代扩散模型的关键——同一个目标有三种语言描述：

| 视角 | 核心对象 | 代表工作 |
|---|---|---|
| **变分推断** | ELBO、KL 散度 | DDPM |
| **分数匹配（Score Matching）** | 数据分布的梯度场 $\nabla_x \log q(x_t)$ | Song & Ermon 2019 |
| **随机微分方程（SDE / ODE）** | 连续时间扩散、概率流 | Song et al. 2021 |

### 1. Score 视角

定义分数函数 $s(x_t) = \nabla_{x_t}\log q(x_t)$。对加噪分布求导可得

$$
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

所以"预测噪声"等价于"估计分数"，差一个已知的缩放因子：

$$
\nabla_{x_t}\log q(x_t) \approx -\frac{1}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)
$$

这就是**去噪分数匹配（denoising score matching）**，说明扩散模型的训练目标本质上是在学数据分布的对数密度梯度场。

### 2. SDE 视角

把离散步骤取极限 $T\to\infty$，前向过程变成一个 Itô SDE：

$$
\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}w
$$

其反向时间过程为（Anderson, 1982）：

$$
\mathrm{d}x = \big[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\big]\mathrm{d}t + g(t)\,\mathrm{d}\bar w
$$

只要网络学会了 $\nabla_x\log p_t(x)$，就能数值求解这个 SDE 从噪声生成样本。去除随机项后得到**概率流 ODE**，是 DDIM、flow matching 等方法的基础，也是可实现"确定性采样 + 潜空间插值"的原因。

两个经典选择：

- **VP-SDE**（variance preserving）：$f = -\tfrac12\beta(t)x$，$g=\sqrt{\beta(t)}$，对应 DDPM；
- **VE-SDE**（variance exploding）：$f = 0$，$g = \sqrt{\mathrm{d}\sigma^2(t)/\mathrm{d}t}$，对应 NCSN。

---

## 六、采样（推理）

### 1. DDPM 祖先采样

从 $x_T \sim \mathcal{N}(0,I)$ 出发，对 $t = T, \dots, 1$：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z,\qquad z\sim\mathcal{N}(0,I)
$$

最后一为 $z=0$。总共 $T$ 次前向，所以**慢**——这是扩散模型最主要的工程痛点。

### 2. DDIM：确定性与加速

DDIM 把采样改写为非马尔可夫的确定性映射（$\eta$ 控制随机性，$\eta=0$ 完全确定）：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\hat x_0}{\sqrt{1-\bar\alpha_t}} + \sigma_t z
$$

其中 $\hat x_0 = \dfrac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}$。

好处：**可以用几十步甚至十几步的子序列采样**，且因为确定性，同一噪声输入总得到同一样本，支持语义插值。

### 3. 进一步的加速

| 方法 | 思路 |
|---|---|
| DPM-Solver / Heun | 高阶 ODE 求解器，10–20 步 |
| 蒸馏（Progressive / Consistency / LCM） | 把多步教师压成 1–4 步学生 |
| Rectified Flow / Flow Matching | 学更直的轨迹，减少步数 |
| 潜空间扩散（LDM） | 在 VAE 潜空间扩散，单步计算量大幅下降 |

---

## 七、条件生成与引导（Guidance）

文本到图像必须能"控制"生成内容，即建模 $p(x \mid c)$。

### 1. Classifier Guidance

用贝叶斯公式：

$$
\nabla_x \log p(x_t\mid c) = \nabla_x\log p(x_t) + \nabla_x \log p(c\mid x_t)
$$

需要一个额外的分类器 $p(c\mid x_t)$。问题是：要为每个条件（如每段文本）训练分类器，不现实。

### 2. Classifier-Free Guidance（CFG）—— 主流做法

训练时以一定概率把条件 $c$ 置为空（$\varnothing$），让同一个网络同时学会"有条件"和"无条件"预测。采样时外推：

$$
\tilde\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s\cdot\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right)
$$

$s$ 是 guidance scale：

- $s=1$：普通条件生成；
- $s>1$：更强地贴合条件，但多样性下降、易过饱和/失真（常见"CFG 烧伤"）。

这是 Stable Diffusion、DALL·E 2/3、Imagen 等全部采用的机制。

---

## 八、从像素扩散到潜空间：Latent Diffusion

直接在 $512\times512\times3$ 上扩散开销巨大。**Latent Diffusion（LDM / Stable Diffusion）** 的做法：

1. 训练一个 VAE，把图像压到潜空间 $z = \mathcal{E}(x)$（如 $64\times64\times4$，空间缩小 8 倍）；
2. 在 $z$ 上做扩散；
3. 采样后 $x = \mathcal{D}(z)$ 解码回像素。

同时把条件（文本、分割图等）通过 cross-attention 注入 U-Net/DiT 的每个 block。这一步让扩散模型第一次能在消费级 GPU 上训练和推理。

近年架构上 U-Net 正被 **DiT（Diffusion Transformer）** 取代（Sora、SD3、Flux 等），scaling 行为更好。

---

## 九、优缺点与横向对比

<details>
<summary><b>点开：与其他生成模型比较</b></summary>

| | GAN | VAE | 归一化流 | 扩散模型 |
|---|---|---|---|---|
| 训练稳定性 | 差（模式崩塌、震荡） | 好 | 好 | **好** |
| 样本质量 | 高 | 偏糊 | 高 | **很高** |
| 多样性/覆盖 | 差 | 好 | 好 | **好** |
| 采样速度 | **1 步** | 1 步 | 1 步 | 慢（需蒸馏） |
| 精确似然 | 无 | 下界 | **有** | 下界（可算） |
| 潜空间结构 | 无 | 有 | 有 | 有（ODE 视角） |

**优点**：训练稳定、目标简单（就是回归 MSE）、模式覆盖好、条件控制灵活（CFG、inpainting、ControlNet 都是自然延伸）。

**缺点**：采样慢（需多步迭代）、不能像 GAN 一样一步出图（除非蒸馏）、完整 ELBO 的似然估计偏松。

</details>

---

## 十、一句话总结

> 扩散模型 = **一个固定的加噪过程** + **一个学习去噪的回归网络**；数学上它优化的是变分下界，等价于学习数据分布的对数密度梯度（score），而采样就是沿这个梯度场反向求解一个 SDE/ODE。它的优雅之处在于：**把"生成"这个困难问题，转化成了"去噪"这个简单的监督回归问题。**

---

如果你想继续深入，我可以再展开其中任一块，比如：
- ELBO 的完整逐步推导
- Score matching 与 SDE 的严谨对应
- DDIM / DPM-Solver 的数值原理
- 如何写一个最小的 DDPM（MNIST/CIFAR 级别，可跑）
