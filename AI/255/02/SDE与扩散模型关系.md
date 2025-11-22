# SDE与扩散模型关系

### 🧑‍💻 User

SDE → Fokker-Planck → 热方程 → 扩散模型之间的联系？

---

### 🤖 Assistant

下面我把 **SDE → Fokker–Planck → 热方程 → 扩散模型** 之间的数学与概念联系，从最底层的物理/概率论原理到现代生成模型的结构，串成一条清晰的链路。

---

# 总览：扩散模型其实就是“热方程 + 逆热方程”的机器学习版本

扩散模型的本质：

* **前向过程（破坏数据）**：数据逐渐加噪 → 满足 **SDE** → 其密度演化遵从 **Fokker–Planck 方程** → 在特殊情形下就是 **热方程**。
* **后向过程（生成数据）**：逆向模拟 Fokker–Planck / 热方程 → 有漂移项（score） → 对应 **逆SDE** 或 **概率流 ODE**。

扩散模型做的事：
**把“热扩散”逆过来学会恢复数据分布。**

---

# 1. SDE（随机微分方程）：扩散模型的前向噪声过程

扩散模型前向加噪定义为一个伊藤 SDE：

[
dx_t = f(x_t, t)dt + g(t) dW_t
]

常用的 VP SDE（DDPM 风格）：

[
dx_t = -\tfrac{1}{2}\beta(t) x_t dt + \sqrt{\beta(t)}, dW_t
]

常用的 VE SDE（NCSN 风格）：

[
dx_t = \sqrt{2\alpha(t)}, dW_t
]

SDE 的作用：
**描述数据点如何被不断加噪并趋近高斯分布**。

---

# 2. Fokker–Planck方程（密度演化 PDE）：SDE 的分布演变

给定 SDE：
[
dx_t = f(x,t)dt + g(t)dW_t
]

对应的密度 (p_t(x)) 满足 Fokker–Planck / Kolmogorov 前向方程：

[
\frac{\partial p_t(x)}{\partial t}
= -\nabla \cdot ( f(x,t) p_t(x) )

* \frac{1}{2}\nabla^2 (g^2(t) p_t(x)).
  ]

解释：

* **漂移项 f** → 产生概率流动
* **扩散系数 g** → 产生扩散（加噪）

因此扩散模型所谓的 “扩散” 其实是：

> **SDE 的扩散项 (g(t)) 使得密度满足一个热扩散 PDE。**

---

# 3. 热方程 = Fokker–Planck 在特殊情形下的简化

典型热方程：

[
\frac{\partial p}{\partial t}
= D \nabla^2 p.
]

当 SDE 是纯扩散（无漂移）且扩散系数常数：

[
dx_t = \sqrt{2D}, dW_t
]

其 Fokker–Planck 就退化成热方程。

扩散模型中的 **VE SDE**：

[
dx_t = \sigma(t), dW_t
]

当 (\sigma(t)=const) 时 → 正是热方程
当 (\sigma(t)) 随时间变化 → 热方程的 time-inhomogeneous 扩展

所以：

> **VE 前向扩散 = 非均匀热扩散方程**
> **VP 前向扩散 = 热扩散 + 收缩（Ornstein–Uhlenbeck）**

---

# 4. 扩散模型 = “逆热方程” + “得分函数（score）”

热方程性质：

* 正向：平滑函数，信息变少
* 逆向：不适定（unstable），需要梯度信息 “推回去”

逆 Fokker–Planck 对应的逆 SDE 为：

[
dx_t
= \left[f(x,t) - g^2(t)\nabla_x \log p_t(x) \right] dt

* g(t) d\bar W_t.
  ]

其中：

[
s_\theta(x,t) \approx \nabla_x\log p_t(x)
]

就是扩散模型所学习的 **score function**。

这就是 DSM/score matching 获得名义基础的地方。

---

# 串起来：扩散模型的前向–后向链条

### （1）前向：加噪

SDE → Fokker–Planck → 密度向高斯扩散（热方程行为）

### （2）后向：逆扩散

逆 Fokker–Planck → 逆 SDE
需要 score（数据对数密度的梯度）

### （3）神经网络学习 score

训练目标：

[
s_\theta(x_t,t) \approx \nabla_x \log p(x_t|x_0)
]

或等价的 ε-pred, v-pred, x0-pred。

### （4）采样：模拟逆 SDE 或 probability-flow ODE

最终生成数据。

---

# 各概念之间的直观关系图

```
SDE（前向加噪）
    |
    v
Fokker–Planck（密度演化 PDE）
    |
    | 特殊情形
    v
热方程（Gaussian smoothing）
    |
    v
扩散模型：学习逆热方程（逆SDE）
    |
    v
score network（学习密度梯度）
    |
    v
逆SDE / ODE 生成样本
```

---

# 为什么深度学习界把它叫“扩散模型”？

因为：

1. 前向 SDE 的密度演变就是 **热扩散过程**
2. 它模拟的是 **从数据到高斯的扩散**
3. 逆向采样就是 **逆热扩散**（数学上 ill-posed，所以要用 score）

因此扩散模型 = 逆热方程的可学习近似。

---

# 如果你想，我可以继续解释：

* 逆热方程为什么不适定（ill-posed）
* Score 的几何意义（为什么能逆扩散）
* Probability Flow ODE 与逆 SDE 的等价性
* DDIM/score-SDE/score-ODE 三者的统一框架
* 为什么 x₀-pred、ε-pred、v-pred 都是 score 的不同参数化

你想继续哪一部分？
