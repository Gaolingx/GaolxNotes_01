# ε预测推导SDE与ODE

### 🧑‍💻 User

score-SDE、probability-flow ODE 怎么从 ε-prediction 形式推出？

---

### 🤖 Assistant

这是一个非常核心的问题，它连接了 **DDPM（离散时间、预测噪声）** 和 **Score-based Generative Models（连续时间、预测分数）** 两个体系。

要从 $\epsilon$-prediction（噪声预测）形式推导出 Score-SDE 和 Probability-flow ODE，关键在于建立 **分数（Score）与 噪声（$\epsilon$）** 之间的数学关系。

以下是详细的推导步骤：

### 1. 核心桥梁：Score 与 $\epsilon$ 的关系

在 DDPM 中，前向扩散过程（Forward Process）在任意时刻 $t$ 的边缘分布 $p_t(\mathbf{x}_t | \mathbf{x}_0)$ 是一个高斯分布：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其概率密度函数为：
$$
p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

我们想要计算的是对数概率密度的梯度（即 Score Function）：$\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t | \mathbf{x}_0)$。

根据高斯分布的对数密度公式：
$$
\log p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = -\frac{1}{2(1 - \bar{\alpha}_t)} \|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + C
$$

对 $\mathbf{x}_t$ 求梯度：
$$
\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1 - \bar{\alpha}_t}
$$

回到 $\mathbf{x}_t$ 的定义，我们知道 $\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0 = \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$。将其代入分子：

$$
\nabla_{\mathbf{x}_t} \log p_{0t}(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

**结论：** 神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 预测的噪声，本质上是在预测缩放后的 Score。真实数据的 Score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ 可以通过下式近似：

$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx \mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

---

### 2. 设定 SDE 框架 (以 VP-SDE 为例)

DDPM 的连续时间极限对应于 **Variance Preserving (VP) SDE**。

**前向 SDE (Forward SDE):**
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)dt + g(t)d\mathbf{w}
$$
对于 VP-SDE（对应 DDPM）：
*   漂移项 (Drift): $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$
*   扩散项 (Diffusion): $g(t) = \sqrt{\beta(t)}$

这里 $\beta(t)$ 是连续时间的噪声方差调度。

---

### 3. 推导 Score-SDE (Reverse SDE)

根据 Anderson 定理，任何前向 SDE 都有一个对应的 **逆向 SDE (Reverse SDE)**，用于生成样本。逆向 SDE 的形式为：

$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}
$$
*(注：这里 $dt$ 代表时间反向流动，$d\bar{\mathbf{w}}$ 是反向布朗运动)*

**推导步骤：**

1.  将 VP-SDE 的 $\mathbf{f}$ 和 $g$ 代入：
    $$
    d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\mathbf{x} - (\sqrt{\beta(t)})^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}
    $$
2.  将第 1 步得到的 **Score 与 $\epsilon$ 的关系**代入 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$：
    $$
    \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)}{\sqrt{1 - \bar{\alpha}_t}}
    $$
3.  整理方程：
    $$
    d\mathbf{x} = \left[-\frac{1}{2}\beta(t)\mathbf{x} - \beta(t) \left( -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)}{\sqrt{1 - \bar{\alpha}_t}} \right)\right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}
    $$

**最终 Score-SDE ($\epsilon$-prediction 形式):**

$$
d\mathbf{x}_t = \left[ -\frac{1}{2}\beta(t)\mathbf{x}_t + \frac{\beta(t)}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right] dt + \sqrt{\beta(t)} d\bar{\mathbf{w}}
$$

这就是我们在代码中实现 Euler-Maruyama 采样时所使用的公式基础。

---

### 4. 推导 Probability-Flow ODE (PF-ODE)

Song Yang 等人证明，对于任意扩散 SDE，都存在一个 **确定性的常微分方程 (ODE)**，其边缘分布 $p_t(\mathbf{x})$ 与 SDE 完全一致。这个 ODE 被称为 Probability Flow ODE。

PF-ODE 的通用公式为：
$$
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt
$$
*(注意：对比 Reverse SDE，这里的 Score 系数是 $\frac{1}{2}g^2$，且没有随机噪声项 $d\bar{\mathbf{w}}$)*

**推导步骤：**

1.  代入 VP-SDE 的 $\mathbf{f}$ 和 $g$：
    $$
    d\mathbf{x} = \left[ -\frac{1}{2}\beta(t)\mathbf{x} - \frac{1}{2}\beta(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt
    $$
2.  再次代入 **Score 与 $\epsilon$ 的关系** ($-\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}_t}}$)：
    $$
    d\mathbf{x} = \left[ -\frac{1}{2}\beta(t)\mathbf{x} - \frac{1}{2}\beta(t) \left( -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)}{\sqrt{1 - \bar{\alpha}_t}} \right) \right] dt
    $$

**最终 PF-ODE ($\epsilon$-prediction 形式):**

$$
d\mathbf{x}_t = \left[ -\frac{1}{2}\beta(t)\mathbf{x}_t + \frac{\beta(t)}{2\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right] dt
$$

---

### 总结

从 $\epsilon$-prediction 形式出发，我们只需要利用核心转换关系 $\text{Score} \propto -\epsilon$，就可以将通用的 SDE/ODE 框架具体化为我们常见的形式：

1.  **转换关系**：
    $$ \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}, t)}{\sqrt{1 - \bar{\alpha}_t}} $$

2.  **Reverse SDE** (用于随机采样，如 DDPM/SDE solver)：
    $$ d\mathbf{x}_t = \left( -\frac{1}{2}\beta(t)\mathbf{x}_t + \frac{\beta(t)}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta \right) dt + \sqrt{\beta(t)} d\bar{\mathbf{w}} $$

3.  **Probability Flow ODE** (用于确定性采样，如 DDIM/ODE solver)：
    $$ d\mathbf{x}_t = \left( -\frac{1}{2}\beta(t)\mathbf{x}_t + \frac{\beta(t)}{2\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta \right) dt $$