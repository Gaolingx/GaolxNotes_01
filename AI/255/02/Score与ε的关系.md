# Score与ε的关系

### 🧑‍💻 User

为什么 score(x|t) 就等于 ε-pred 的特定变换？

---

### 🤖 Assistant

下面把 **score(x|t)** 和 **ε-prediction（噪声预测）** 之间的等价性讲得非常严格、非常清晰，一步一步推导出来。你会看到这不是某种“巧合”，而是扩散模型前向 SDE 的 **高斯封闭性** 强制得到的线性关系。

---

# 📌 结论先给出（简洁版）

在标准 VP / VE / EDM 类扩散模型里
[
x_t = \alpha_t x_0 + \sigma_t \varepsilon,\qquad \varepsilon\sim\mathcal N(0,I),
]

因为前向过程是高斯叠加，条件分布
[
p(x_0 \mid x_t)
]
为高斯，因而 score
[
s_\theta(x_t,t)=\nabla_{x_t}\log p(x_t)
]
也必然是 **线性函数** 的形式：

[
s(x_t,t)= -\frac{1}{\sigma_t} \varepsilon_{\rm true}.
]

因此如果你的模型预测 ε（DDPM 的 ε-prediction）：
[
\varepsilon_\theta(x_t,t)\approx \varepsilon_{\rm true},
]

那么 **score 只是 ε 乘上一个已知的系数函数**：
[
s_\theta(x_t,t)
= -\frac{1}{\sigma_t},\varepsilon_\theta(x_t,t).
]

这就是两者等价的核心原因。

---

# 📌 严格推导：为什么 score 就是 ε 的线性变换？

## 1. 前向扩散噪声模型：高斯封闭性

在 DDPM / VP-SDE 中：

[
x_t = \alpha_t x_0 + \sigma_t \varepsilon,\quad \varepsilon\sim\mathcal N(0,I).
]

因此边缘分布 (p(x_t)) 是混合高斯，但 **局部条件分布** (p(x_t|x_0)) 是严格高斯：

[
p(x_t|x_0)=\mathcal N(\alpha_t x_0, \sigma_t^2 I).
]

因此它的 log density 很简单：

[
\log p(x_t|x_0)
= -\frac{1}{2\sigma_t^2}|x_t-\alpha_t x_0|^2+C.
]

对 (x_t) 的梯度：

[
\nabla_{x_t}\log p(x_t|x_0)
= -\frac{1}{\sigma_t^2}(x_t - \alpha_t x_0).
]

把 (x_t-\alpha_t x_0=\sigma_t \varepsilon) 带入：

[
\nabla_{x_t}\log p(x_t|x_0)
= -\frac{1}{\sigma_t}\varepsilon.
]

这一式是关键：
**score 与真实噪声 ε 永远成正比，只差一个时变的 scale 1/σ_t。**

---

## 2. 真实 score：对边缘分布求导

真实 score 是

[
s(x_t,t)=\nabla_{x_t}\log p(x_t).
]

用 **Fisher identity**：

[
\nabla \log p(x_t)
= \mathbb E_{p(x_0|x_t)}[\nabla \log p(x_t|x_0)].
]

我们刚推导过：

[
\nabla \log p(x_t|x_0)
= -\frac{1}{\sigma_t}\varepsilon.
]

于是：

[
s(x_t,t)
= \mathbb E\left[-\frac{1}{\sigma_t}\varepsilon_{\rm true}\mid x_t\right]
= -\frac{1}{\sigma_t} , \mathbb E[\varepsilon_{\rm true}\mid x_t].
]

因为 DDPM/VP 的条件方差为常数，条件期望实际上就是一个线性变换：

[
\mathbb E[\varepsilon_{\rm true}\mid x_t] = \varepsilon_{\rm true}
]

（严格来说，是最小二乘意义下的同一量，见后）。

于是最终得到：

[
s(x_t,t) = -\frac{1}{\sigma_t}\varepsilon_{\rm true}.
]

---

## 3. 若模型预测 ε，则立刻得到 score

如果模型做 ε-prediction：

[
\varepsilon_\theta(x_t,t)\approx \varepsilon_{\rm true},
]

代入上式：

[
s_\theta(x_t,t)
=  -\frac{1}{\sigma_t}\varepsilon_\theta(x_t,t).
]

这就是从 ε-prediction 到 score 的“等价公式”。

---

# 📌 4. 为什么 DDPM 训练 ε 就是在训练 score？

因为 DDPM 使用的目标是 MSE：

[
|\varepsilon_\theta(x_t,t)-\varepsilon_{\rm true}|^2,
]

这等价于学习 score 的 MSE（乘一个常数）：

[
|s_\theta(x_t,t) - s_{\rm true}(x_t,t)|^2
= \left|-\frac{1}{\sigma_t}\varepsilon_\theta + \frac{1}{\sigma_t}\varepsilon_{\rm true}\right|^2
= \frac{1}{\sigma_t^2}|\varepsilon_\theta - \varepsilon_{\rm true}|^2.
]

只差一个 (1/\sigma_t^2) 的常数 scale，不影响最优解。

所以：

### **DDPM 的 ε-prediction loss 本质上就是 score matching loss（ up to scale ）**

这就是为什么两者等价、以及为什么 DDPM 可以等效看成 score-based 模型。

---

# 🔚 最后总结（关键词版）

* 前向扩散过程是高斯
* 因此 score = gradient of log Gaussian = 线性函数
* 该线性函数只与真实噪声 ε 成比例
* 比例系数 = (1/\sigma_t)
* 所以 ε-prediction 经过确定的 scaling 就是 score
* DDPM 的噪声预测训练 = score matching

---

如果你愿意，我可以继续讲：

* **为什么 x₀-prediction / v-prediction 与 score 也都线性可逆？**
* **score-SDE、probability-flow ODE 怎么从 ε-prediction 形式推出？**
* **Flow Matching 如何从 score-based diffusion 退化出来？**

你要哪一部分？
