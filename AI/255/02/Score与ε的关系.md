# Score与ε的关系

### 🧑‍💻 User

为什么 score(x|t) 就等于 ε-pred 的特定变换？

---

### 🤖 Assistant

在扩散模型（Diffusion Models）尤其是 DDPM（Denoising Diffusion Probabilistic Models）中，**Score Function**（分数函数）与 **$\boldsymbol{\epsilon}$-prediction**（噪声预测）之间的等价关系是理解该模型为何有效的核心。

简单来说：**Score 指示了数据密度的梯度方向（如何让图片变清晰），而预测出的噪声 $\boldsymbol{\epsilon}$ 正好是相反的方向（是什么让图片变模糊的）。**

下面通过严格的数学推导来解释为什么 $\text{score}(\mathbf{x}_t) \propto -\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。

### 1. 定义前向扩散过程

在 DDPM 中，给定初始数据 $\mathbf{x}_0$，任意时刻 $t$ 的加噪数据 $\mathbf{x}_t$ 服从高斯分布。根据重参数化技巧（Reparameterization Trick），我们可以直接写出 $\mathbf{x}_t$ 的条件分布 $q(\mathbf{x}_t | \mathbf{x}_0)$：

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

其中：
*   $\bar{\alpha}_t$ 是直到 $t$ 时刻的噪声明细表累乘。
*   均值 $\boldsymbol{\mu} = \sqrt{\bar{\alpha}_t}\mathbf{x}_0$。
*   方差 $\sigma_t^2 = 1 - \bar{\alpha}_t$。

我们也可以把 $\mathbf{x}_t$ 写成：
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### 2. 推导条件分数的梯度 (Conditional Score)

Score Function 的定义是对数概率密度的梯度：$\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$。
首先，我们看**给定 $\mathbf{x}_0$ 时**的 Score，即 $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)$。

高斯分布的对数概率密度函数为：
$$
\log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{1}{2(1 - \bar{\alpha}_t)} \|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + C
$$

对 $\mathbf{x}_t$ 求导：
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1 - \bar{\alpha}_t}
$$

### 3. 将 $\boldsymbol{\epsilon}$ 代入梯度公式

这一步是连接 Score 和 Noise 的关键。
回想刚才的重参数化公式：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$。
我们可以通过移项把 $\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0$ 替换掉：

$$
\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0 = \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}
$$

将这个式子代入上面的导数公式中：

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) &= -\frac{\sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} \\
&= -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
\end{aligned}
$$

**结论：** 如果我们知道 $\mathbf{x}_0$（即知道真实的噪声 $\boldsymbol{\epsilon}$），那么 Score 仅仅就是噪声的负缩放版本。

### 4. 从条件 Score 到 边缘 Score (Marginal Score)

在实际采样中，我们不知道 $\mathbf{x}_0$，我们想求的是边缘分布的 Score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$。

根据 **Denoising Score Matching (DSM)** 理论（Vincent, 2011），如果我们需要训练一个神经网络 $s_\theta(\mathbf{x}_t, t)$ 来估计真实的 Score $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$，我们可以通过最小化以下目标函数来实现：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_t} \left[ \| s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) \|^2 \right]
$$

也就是说，**为了学习真实数据的 Score，我们只需要让神经网络去拟合条件分布的 Score（即加上去的噪声）即可。** 在最优情况下，神经网络 $s_\theta(\mathbf{x}_t, t)$ 会收敛到 $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$。

既然我们已经推导出 $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$，那么我们可以定义神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 来预测 $\boldsymbol{\epsilon}$，关系如下：

$$
s_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

所以，真实的 Score 近似为：
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
$$

### 5. 直观理解 (Intuition)

1.  **物理意义**：
    *   **$\boldsymbol{\epsilon}$ (噪声)**：是从清晰图像 $\mathbf{x}_0$ 指向噪声图像 $\mathbf{x}_t$ 的向量（扩散方向）。
    *   **$-\boldsymbol{\epsilon}$ (去噪)**：是从噪声图像 $\mathbf{x}_t$ 指回清晰图像 $\mathbf{x}_0$ 的向量（生成方向）。
    *   **$\nabla \log p(\mathbf{x})$ (Score)**：是指向数据密度更高区域的向量。数据密度最高的地方就是无噪的自然图像流形。

2.  **为何相等**：
    要让概率密度变大（Score 的方向），你就得去掉噪声。因此，Score 的方向与噪声向量的方向正好相反。

### 总结

Score 等于 $\boldsymbol{\epsilon}$-pred 的特定变换，是因为：
1.  扩散核是**高斯分布**。
2.  高斯分布的对数梯度（Score）形式上正比于 $(\mathbf{x} - \mu)$。
3.  在扩散过程中，$(\mathbf{x} - \mu)$ 正好就是**噪声项**。
4.  通过 **Denoising Score Matching** 理论，训练网络预测噪声 $\boldsymbol{\epsilon}$ 等价于训练网络估计 Score。