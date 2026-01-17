# 扩散模型与SDE

### 🧑‍💻 User

diffusion model为什么可以用随机微分方程描述？

---

### 🤖 Assistant

Diffusion Model（扩散模型）之所以可以用**随机微分方程（Stochastic Differential Equations, SDEs）**来描述，本质上是因为扩散模型通常被定义为一个离散的马尔可夫链（Markov Chain），而当我们将这个离散过程的步数推向无穷大（$N \to \infty$）、步长趋近于零时，离散的加噪过程就收敛为了一个连续时间的随机过程。

宋飏（Yang Song）等人在论文《Score-Based Generative Modeling through Stochastic Differential Equations》中系统地阐述了这一观点，将DDPM（Denoising Diffusion Probabilistic Models）和SMLD（Score Matching with Langevin Dynamics）统一在了SDE的框架下。

以下是详细的数学推导和直观解释：

### 1. 从离散过程到连续过程的极限

#### 离散的前向过程 (DDPM)
在DDPM中，前向过程是一个向数据逐渐添加高斯噪声的过程。对于时间步 $i=1, \dots, N$，状态 $x_i$ 的更新公式通常写作：
$$
x_i = \sqrt{1-\beta_i} x_{i-1} + \sqrt{\beta_i} z_{i-1}, \quad z_{i-1} \sim \mathcal{N}(0, I)
$$
其中 $\beta_i$ 是方差调度参数。

#### 连续的极限 (SDE)
如果我们让总的时间步数 $N \to \infty$，使得每一步的变化变得非常微小，我们可以引入一个连续的时间变量 $t \in [0, T]$。此时，离散的迭代就变成了连续的微分形式。

一个一般的SDE形式如下：
$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) dt + g(t) d\mathbf{w}
$$
其中：
*   $\mathbf{f}(\mathbf{x}, t)$ 是**漂移系数（Drift Coefficient）**，控制数据的确定性变化趋势（例如，向原点收缩）。
*   $g(t)$ 是**扩散系数（Diffusion Coefficient）**，控制注入噪声的强度。
*   $\mathbf{w}$ 是标准的**维纳过程（Wiener Process）**，也就是布朗运动，$d\mathbf{w}$ 代表无穷小的随机噪声。

DDPM的前向过程对应的一个特定SDE（称为 Variance Preserving SDE, VP-SDE）可以写为：
$$
d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x} dt + \sqrt{\beta(t)} d\mathbf{w}
$$
这里 $\mathbf{f}(\mathbf{x}, t) = -\frac{1}{2}\beta(t)\mathbf{x}$ 负责将数据拉向均值0（对应 $\sqrt{1-\beta}$ 缩小幅值），$g(t) = \sqrt{\beta(t)}$ 负责注入噪声。

### 2. 时间反转：生成的关键

扩散模型的核心在于**生成**，即如何从纯噪声（$t=T$）还原回数据（$t=0$）。在SDE框架下，这对应于求解**逆向SDE**。

根据 Anderson (1982) 的著名理论，如果一个前向过程由上述SDE描述，那么其**逆向时间过程**也是一个扩散过程，并且满足以下逆向SDE：

$$
d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt + g(t) d\bar{\mathbf{w}}
$$

这里的 $dt$ 是反向的时间流，$d\bar{\mathbf{w}}$ 是反向时间的布朗运动。

#### 为什么这一点至关重要？
观察上面的公式，为了通过反向SDE生成数据，我们需要知道 $\mathbf{f}(\mathbf{x}, t)$ 和 $g(t)$（这是我们自己定义的），唯一未知项是：
$$
\nabla_\mathbf{x} \log p_t(\mathbf{x})
$$
这就是**分数函数（Score Function）**，即概率密度函数对数关于数据的梯度。

这解释了扩散模型训练的目标：**我们训练一个神经网络（U-Net）来估计这个分数函数**（Score Matching）。一旦网络学会了 $\nabla_\mathbf{x} \log p_t(\mathbf{x})$，我们就可以把它代入逆向SDE，通过数值求解器（如Euler-Maruyama方法）从噪声中还原出图像。

### 3. SDE视角带来的好处

用SDE描述扩散模型不仅仅是数学上的优雅，它带来了实际的工程优势：

1.  **统一框架**：它将基于分数的生成模型（SMLD/NCSN）和概率扩散模型（DDPM）统一在同一个数学框架下，证明它们只是同一个SDE的不同离散化方式。
2.  **灵活的采样**：
    *   因为是一个连续方程，我们可以使用任何现成的SDE数值求解器（Solver）。
    *   我们可以根据需要调整采样步数，在生成质量和速度之间权衡，而不需要重新训练模型。
3.  **概率流 ODE (Probability Flow ODE)**：
    *   SDE还有一个对应的确定性常微分方程（ODE），它与SDE具有相同的边缘概率分布 $p_t(x)$。
    *   $$ d\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})] dt $$
    *   这意味着我们可以把生成过程看作是一个平滑的流形变换，允许我们使用高级ODE求解器（如Runge-Kutta）来加速采样，或者进行精确的似然估计（Likelihood computation）。

### 总结

扩散模型之所以可以用SDE描述，是因为：
1.  **物理本质**：扩散过程本质上就是粒子在介质中受到随机力（噪声）和确定力（漂移）作用的连续运动，这正是SDE研究的对象。
2.  **数学极限**：当离散的加噪步数趋于无穷时，DDPM的迭代公式收敛于伊藤积分（Itô Calculus）描述的SDE。
3.  **逆向能力**：SDE理论提供了严谨的时间反转公式，明确指出了生成过程需要学习的核心量——**Score Function（分数函数）**。