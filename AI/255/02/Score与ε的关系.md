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

---

### 🧑‍💻 User

为什么 x₀-pred、ε-pred、v-pred 都是 score 的不同参数化？

---

### 🤖 Assistant

这是一个非常深刻且触及扩散模型本质的问题。

简单来说，**$\mathbf{x}_0$-pred、$\boldsymbol{\epsilon}$-pred 和 $\mathbf{v}$-pred 之所以都是 Score 的参数化，是因为它们三者之间通过一个简单的线性方程组互相锁定。** 只要知道了其中一个，结合当前的输入 $\mathbf{x}_t$ 和时间 $t$，就能精确算出另外两个，进而算出 Score。

它们本质上是在预测同一个高维空间中的**同一个向量**，只是选取的**坐标系基底**不同。

---

### 1. 核心方程：线性约束

一切的根源在于扩散模型的前向过程公式（我们使用简化的 $\alpha, \sigma$ 符号，对应 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1-\bar{\alpha}_t}$）：

$$
\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}
$$

其中：
*   $\mathbf{x}_t$ 是网络输入（已知）。
*   $\alpha_t, \sigma_t$ 是预设的超参数（已知，满足 $\alpha_t^2 + \sigma_t^2 = 1$）。
*   **$\mathbf{x}_0$ 和 $\boldsymbol{\epsilon}$ 是未知的变量**。

这是一个二元一次方程（虽然是向量），有两个未知数。虽然我们无法仅凭 $\mathbf{x}_t$ 解出 $\mathbf{x}_0$ 和 $\boldsymbol{\epsilon}$，但**一旦神经网络预测了其中任意一个，另一个就被唯一确定了**。

而我们之前已经推导过，**Score Function** 直接由 $\boldsymbol{\epsilon}$ 决定：
$$
\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}}{\sigma_t}
$$

因此，无论网络预测什么，只要能推导出 $\boldsymbol{\epsilon}$，就是在预测 Score。

---

### 2. 三种参数化的推导

我们定义神经网络的输出为 $f_\theta(\mathbf{x}_t, t)$，看看它如何转化为 Score。

#### A. $\boldsymbol{\epsilon}$-prediction (预测噪声)
这是最常见的方式（DDPM 默认）。网络直接输出 $\hat{\boldsymbol{\epsilon}}$。

*   **关系**：$\text{Score} \propto -\hat{\boldsymbol{\epsilon}}$
*   **公式**：
    $$ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{f_\theta(\mathbf{x}_t, t)}{\sigma_t} $$
*   **直观**：直接预测“也就是什么东西把图搞糊了”。

#### B. $\mathbf{x}_0$-prediction (预测原图)
网络输出对清晰原图的估计 $\hat{\mathbf{x}}_0$。
我们可以通过移项前向公式来反求隐含的噪声 $\hat{\boldsymbol{\epsilon}}$：

$$
\hat{\boldsymbol{\epsilon}} = \frac{\mathbf{x}_t - \alpha_t \hat{\mathbf{x}}_0}{\sigma_t}
$$

*   **关系**：Score 是 $\mathbf{x}_t$ 和预测原图 $\hat{\mathbf{x}}_0$ 的线性组合。
*   **公式**：
    $$ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{\mathbf{x}_t - \alpha_t f_\theta(\mathbf{x}_t, t)}{\sigma_t^2} $$
*   **直观**：如果我知道原图长什么样，我就知道现在的图里包含了多少噪声，也就知道了去噪方向。

#### C. $\mathbf{v}$-prediction (预测速度)
这种参数化方式源自 *Progressive Distillation* (Salimans & Ho, 2022)。
如果你把扩散过程看作是 $\mathbf{x}_0$ 和 $\boldsymbol{\epsilon}$ 在一个圆上的**旋转**（因为 $\alpha_t = \cos \phi, \sigma_t = \sin \phi$），那么 $\mathbf{x}_t$ 是位置向量，而 $\mathbf{v}_t$ 就是与其垂直的**速度向量**。

定义为：
$$
\mathbf{v}_t \equiv \alpha_t \boldsymbol{\epsilon} - \sigma_t \mathbf{x}_0
$$

利用 $\mathbf{x}_t$ 和 $\mathbf{v}_t$ 的定义以及 $\alpha_t^2 + \sigma_t^2 = 1$，我们可以完美解出 $\boldsymbol{\epsilon}$：
$$
\boldsymbol{\epsilon} = \alpha_t \mathbf{v}_t + \sigma_t \mathbf{x}_t
$$

*   **关系**：Score 是预测速度 $\hat{\mathbf{v}}$ 和当前输入 $\mathbf{x}_t$ 的线性组合。
*   **公式**：
    $$ \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) \approx -\frac{\alpha_t f_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{x}_t}{\sigma_t} $$

---

### 3. 几何视角的统一 (The Geometric Intuition)

想象一个二维平面坐标系，横轴是 $\mathbf{x}_0$（信号），纵轴是 $\boldsymbol{\epsilon}$（噪声）。

1.  **扩散轨迹**：随着 $t$ 从 0 到 $T$，状态 $\mathbf{x}_t$ 是从横轴（纯信号）慢慢旋转到纵轴（纯噪声）的一个向量（假设 Variance Preserving）。
2.  **$\mathbf{x}_t$ 向量**：$\mathbf{x}_t = \cos(\phi) \mathbf{x}_0 + \sin(\phi) \boldsymbol{\epsilon}$。
3.  **$\mathbf{v}_t$ 向量**：$\mathbf{v}_t = -\sin(\phi) \mathbf{x}_0 + \cos(\phi) \boldsymbol{\epsilon}$（这是 $\mathbf{x}_t$ 旋转 90 度后的向量）。

此时，Score 本质上就是要把 $\mathbf{x}_t$ 推回数据流形。这三个预测目标相当于把同一个向量投影到了不同的轴上：

*   **$\mathbf{x}_0$-pred**：网络预测该向量在 X 轴的投影。
*   **$\boldsymbol{\epsilon}$-pred**：网络预测该向量在 Y 轴的投影。
*   **$\mathbf{v}$-pred**：网络预测该向量在“切线方向”的分量（也就是旋转速度）。

### 4. 为什么要区分？(数值稳定性)

既然数学上等价，为什么还要分这么多种？因为在计算机里，**$0$ 不能作除数，且浮点数有精度问题**。即 **Signal-to-Noise Ratio (SNR)** 问题。

*   **$\boldsymbol{\epsilon}$-pred 的问题**：
    *   在 $t=0$ 附近（生成刚结束），$\mathbf{x}_t \approx \mathbf{x}_0$，噪声 $\sigma_t \approx 0$。
    *   此时 Score 公式里分母是 $\sigma_t$，预测 $\boldsymbol{\epsilon}$ 需要非常精准，否则除以一个极小的数会导致 Score 爆炸。
    *   但在大多数扩散模型（如 Stable Diffusion）中，$t=0$ 处其实不采样，所以 $\boldsymbol{\epsilon}$-pred 表现很好。

*   **$\mathbf{x}_0$-pred 的问题**：
    *   在 $t=T$ 附近（纯噪声），$\mathbf{x}_t \approx \boldsymbol{\epsilon}$，信号 $\alpha_t \approx 0$。
    *   此时想从一堆纯噪声里预测出原图 $\mathbf{x}_0$ 极其困难，网络收敛很慢。

*   **$\mathbf{v}$-pred 的优势**：
    *   它在整个区间（从 $t=0$ 到 $t=T$）都比较平滑。
    *   当 $\alpha_t \to 0$ 时，$\mathbf{v}_t \approx \boldsymbol{\epsilon}$（像 $\boldsymbol{\epsilon}$-pred）。
    *   当 $\sigma_t \to 0$ 时，$\mathbf{v}_t \approx -\mathbf{x}_0$（像 $\mathbf{x}_0$-pred）。
    *   所以在**蒸馏（Distillation）**或者需要**零信噪比（Zero-SNR）**训练时，$\mathbf{v}$-pred 是最佳选择，因为它避免了端点的数值不稳定。

### 总结

$\mathbf{x}_0, \boldsymbol{\epsilon}, \mathbf{v}$ 三者就像是同一个物理对象（去噪所需的梯度场）在不同视角下的投影。
*   **数学上**：它们通过线性变换互相等价。
*   **工程上**：它们决定了神经网络在不同噪声强度下的**学习难度**和**数值稳定性**。