# Flow Matching 与 Score-based Diffusion 关系

### 🧑‍💻 User

Flow Matching 如何从 score-based diffusion 退化出来？

---

### 🤖 Assistant

Flow Matching (FM) 与 Score-based Diffusion Models (SBDM) 之间有着深刻的数学联系。简而言之，**Diffusion Model 是 Flow Matching 的一种特殊情况**。

当我们将 Flow Matching 中的概率路径（Probability Path）限制为由扩散过程（SDE）诱导的高斯路径，并适当地加权损失函数时，Flow Matching 训练出的向量场（Vector Field）就完全等价于 Score-based Diffusion 中的概率流 ODE（Probability Flow ODE）。

下面通过数学推导详细展示 Flow Matching 如何退化为 Score-based Diffusion。

---

### 1. Score-based Diffusion 的基础 (Probability Flow ODE)

在 Score-based Diffusion 中，前向过程由随机微分方程（SDE）定义：

$$
d\mathbf{x} = f(t)\mathbf{x} dt + g(t) d\mathbf{w}
$$

其中 $f(t)$ 是漂移系数，$g(t)$ 是扩散系数。Song et al. (2020) 证明了存在一个对应的常微分方程（ODE），其边缘分布 $p_t(\mathbf{x})$ 与上述 SDE 完全一致。这个 **Probability Flow ODE** 定义为：

$$
d\mathbf{x} = \left[ f(t)\mathbf{x} - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right] dt
$$

这里的括号项就是驱动粒子运动的**向量场** $v_t(\mathbf{x})$：
$$
v_t^{\text{diff}}(\mathbf{x}) = f(t)\mathbf{x} - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$
这就是我们希望从 Flow Matching 中推导出的目标形式。

---

### 2. Flow Matching 的设定

Flow Matching 的目标是直接回归一个向量场 $v_t(\mathbf{x})$，使得它生成的流 $\phi_t$ 能够将噪声分布 $p_0$ 映射到数据分布 $p_1$（注意：这里使用 $t=0$ 为噪声，$t=1$ 为数据的 FM 惯例，与 Diffusion 的时间方向通常相反，但原理互通）。

Flow Matching 使用 **Conditional Flow Matching (CFM)** 目标函数来训练：
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, q(\mathbf{x}_1), p_t(\mathbf{x}|\mathbf{x}_1)} \left[ || v_\theta(t, \mathbf{x}) - u_t(\mathbf{x}|\mathbf{x}_1) ||^2 \right]
$$
其中 $u_t(\mathbf{x}|\mathbf{x}_1)$ 是**条件向量场**，它生成了条件概率路径 $p_t(\mathbf{x}|\mathbf{x}_1)$。

---

### 3. 推导：从 Flow Matching 到 Diffusion

要将 FM 退化为 Diffusion，我们需要显式地构造一个符合 Diffusion 定义的高斯概率路径。

#### 步骤 1：定义 Diffusion 路径
Diffusion 的扰动核（Perturbation Kernel）通常是高斯的。假设我们将数据 $\mathbf{x}_1$（对应 Diffusion 中的 $\mathbf{x}_0$）加噪到 $t$ 时刻：

$$
p_t(\mathbf{x}|\mathbf{x}_1) = \mathcal{N}(\mathbf{x}; \mu_t(\mathbf{x}_1), \sigma_t^2 \mathbf{I})
$$

在经典 Diffusion（如 VP-SDE）中，通常有 $\mu_t(\mathbf{x}_1) = \alpha_t \mathbf{x}_1$。因此样本可以表示为：
$$
\mathbf{x} = \alpha_t \mathbf{x}_1 + \sigma_t \mathbf{\epsilon}, \quad \text{其中 } \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

#### 步骤 2：计算条件向量场 $u_t(\mathbf{x}|\mathbf{x}_1)$
这是 Flow Matching 的关键。我们需要找到生成上述高斯路径的流的速度场。
对 $\mathbf{x}$ 关于时间 $t$ 求导：

$$
\frac{d\mathbf{x}}{dt} = \dot{\alpha}_t \mathbf{x}_1 + \dot{\sigma}_t \mathbf{\epsilon}
$$

我们需要将右边的 $\mathbf{x}_1$ 和 $\mathbf{\epsilon}$ 替换为 $\mathbf{x}$ 的函数。
由 $\mathbf{x} = \alpha_t \mathbf{x}_1 + \sigma_t \mathbf{\epsilon}$ 可知 $\mathbf{\epsilon} = \frac{\mathbf{x} - \alpha_t \mathbf{x}_1}{\sigma_t}$。代入上式：

$$
\begin{aligned}
u_t(\mathbf{x}|\mathbf{x}_1) &= \dot{\alpha}_t \mathbf{x}_1 + \dot{\sigma}_t \left( \frac{\mathbf{x} - \alpha_t \mathbf{x}_1}{\sigma_t} \right) \\
&= \frac{\dot{\sigma}_t}{\sigma_t} \mathbf{x} + \left( \dot{\alpha}_t - \frac{\dot{\sigma}_t \alpha_t}{\sigma_t} \right) \mathbf{x}_1
\end{aligned}
$$

这就是**高斯条件向量场**。

#### 步骤 3：计算边缘向量场 (Marginal Vector Field)
Flow Matching 训练的最优解 $v_{opt}(\mathbf{x})$ 近似于边缘向量场 $u_t(\mathbf{x})$，它是条件向量场的期望：
$$
u_t(\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}_1|\mathbf{x})} [u_t(\mathbf{x}|\mathbf{x}_1)]
$$
将步骤 2 的结果代入期望：
$$
u_t(\mathbf{x}) = \frac{\dot{\sigma}_t}{\sigma_t} \mathbf{x} + \left( \dot{\alpha}_t - \frac{\dot{\sigma}_t \alpha_t}{\sigma_t} \right) \mathbb{E}[\mathbf{x}_1 | \mathbf{x}]
$$

这里出现了 $\mathbb{E}[\mathbf{x}_1 | \mathbf{x}]$（后验均值）。这可以通过 **Tweedie's Formula** 与 Score Function 联系起来。

#### 步骤 4：引入 Score Function (Tweedie's Formula)
对于高斯分布 $p_t(\mathbf{x}|\mathbf{x}_1) = \mathcal{N}(\mathbf{x}; \alpha_t \mathbf{x}_1, \sigma_t^2 \mathbf{I})$，Score Function $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ 满足 Tweedie 公式：

$$
\mathbb{E}[\mathbf{x}_1 | \mathbf{x}] = \frac{1}{\alpha_t} \left( \mathbf{x} + \sigma_t^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right)
$$

#### 步骤 5：最终替换与对比
将 Tweedie 公式代入步骤 3 中的边缘向量场方程：

$$
\begin{aligned}
u_t(\mathbf{x}) &= \frac{\dot{\sigma}_t}{\sigma_t} \mathbf{x} + \left( \dot{\alpha}_t - \frac{\dot{\sigma}_t \alpha_t}{\sigma_t} \right) \frac{1}{\alpha_t} \left( \mathbf{x} + \sigma_t^2 \nabla_\mathbf{x} \log p_t(\mathbf{x}) \right) \\
&= \frac{\dot{\sigma}_t}{\sigma_t} \mathbf{x} + \left( \frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\sigma}_t}{\sigma_t} \right) (\mathbf{x} + \sigma_t^2 \nabla \log p_t(\mathbf{x})) \\
&= \left( \frac{\dot{\sigma}_t}{\sigma_t} + \frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\sigma}_t}{\sigma_t} \right) \mathbf{x} + \sigma_t^2 \left( \frac{\dot{\alpha}_t}{\alpha_t} - \frac{\dot{\sigma}_t}{\sigma_t} \right) \nabla \log p_t(\mathbf{x}) \\
&= \frac{\dot{\alpha}_t}{\alpha_t} \mathbf{x} + \left( \frac{\sigma_t^2 \dot{\alpha}_t}{\alpha_t} - \sigma_t \dot{\sigma}_t \right) \nabla \log p_t(\mathbf{x})
\end{aligned}
$$

现在，我们将其与 Diffusion 的 Probability Flow ODE 系数进行匹配。
在 SDE $d\mathbf{x} = f(t)\mathbf{x}dt + g(t)d\mathbf{w}$ 中，其对应的均值和方差演化满足：
*   $f(t) = \frac{\dot{\alpha}_t}{\alpha_t}$
*   $g^2(t) = \frac{d(\sigma_t^2)}{dt} - 2f(t)\sigma_t^2 = 2\sigma_t \dot{\sigma}_t - 2\frac{\dot{\alpha}_t}{\alpha_t}\sigma_t^2$

如果我们将 $u_t(\mathbf{x})$ 中的 Score 系数整理一下：
$$
\text{Score Coeff} = \frac{\sigma_t^2 \dot{\alpha}_t}{\alpha_t} - \sigma_t \dot{\sigma}_t = -\frac{1}{2} \left( 2\sigma_t \dot{\sigma}_t - 2\frac{\dot{\alpha}_t}{\alpha_t}\sigma_t^2 \right) = -\frac{1}{2} g^2(t)
$$

因此，Flow Matching 推导出的边缘向量场为：
$$
u_t(\mathbf{x}) = f(t)\mathbf{x} - \frac{1}{2}g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})
$$

这与 **Step 1** 中的 Score-based Diffusion Probability Flow ODE **完全一致**。

---

### 总结

Flow Matching 从 Score-based Diffusion 退化出来的逻辑链条如下：

1.  **通用性：** Flow Matching 允许任意定义的条件概率路径 $p_t(\mathbf{x}|\mathbf{x}_1)$。
2.  **约束：** 如果我们将路径**强制约束**为 Diffusion SDE 所诱导的高斯路径（即满足 $\mu_t, \sigma_t$ 的特定微分方程关系）。
3.  **结果：** 此时 Flow Matching 试图回归的目标向量场 $v_t(\mathbf{x})$，在数学上精确等同于 Score-based Diffusion 的 Probability Flow ODE 的漂移项。
4.  **本质区别：** Diffusion 间接通过学习 Score ($\nabla \log p$) 来构建向量场；而 Flow Matching 直接回归向量场。当路径选定为扩散路径时，两者殊途同归。

**代码视角的对应关系：**

```python
# Score Matching (Diffusion) loss roughly looks like:
loss_diff = || score_model(x, t) * sigma_t + epsilon ||^2

# Flow Matching loss roughly looks like:
# Target v is derived from the path x_t = alpha_t * x_1 + sigma_t * x_0
target_v = alpha_dot * x_1 + sigma_dot * x_0
loss_fm = || vector_field_model(x, t) - target_v ||^2

# 此时，vector_field_model 学习到的就是 ODE 的漂移项。
# 如果路径参数 alpha, sigma 满足 SDE 关系，vector_field_model 就隐含了 score。
```