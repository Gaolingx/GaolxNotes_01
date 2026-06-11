# Muon无需阻尼因正交化自动归一

### 🧑‍💻 User

为什么muon优化器不需要像sgd一样要dampening参数？

---

### 🤖 Assistant

Muon 优化器（Momentum Orthogonalizer）不需要像传统的 SGD（随机梯度下降）那样使用 `dampening`（阻尼）参数，根本原因在于 **Muon 的核心更新机制——矩阵正交化（Orthogonalization）——具有尺度不变性（Scale Invariance）。**

简单来说，SGD 依赖梯度的**绝对数值大小**来决定更新步长，而 Muon 只提取梯度的**结构方向**，完全丢弃了梯度的缩放比例。

以下是详细的原因拆解：

### 1. SGD 中 `dampening` 的作用是什么？

在带有动量（Momentum）的 SGD 中（以 PyTorch 的实现为例），更新公式通常是：

$$v_t = \beta v_{t-1} + (1 - \text{dampening}) g_t$$
$$\theta_t = \theta_{t-1} - \text{lr} \cdot v_t$$

如果没有 `dampening`，当 $\beta$ 很大时，梯度累积会导致动量 $v_t$ 的绝对数值变得非常大，从而引起训练不稳定。`dampening` 的主要作用是**抑制新进入梯度的缩放比例**，防止动量缓冲区的值爆炸。它本质上是一个**尺度控制（Scale Control）**工具。

### 2. Muon 的核心机制：正交化（Newton-Schulz Iteration）

Muon 优化器专门针对深度学习中的 2D 权重矩阵（如全连接层）设计。它的核心思想是：不直接使用梯度或动量矩阵来更新权重，而是先将其**正交化**。

对于动量矩阵 $M$，Muon 使用 Newton-Schulz 迭代来寻找最接近的近似正交矩阵 $O$。从奇异值分解（SVD）的角度来看，如果 $M = U \Sigma V^T$，Muon 会丢弃奇异值矩阵 $\Sigma$（将所有奇异值设为 1），只保留 $U V^T$。

最终的权重更新公式大致为：
$$\theta_t = \theta_{t-1} - \text{lr} \cdot \text{Orthogonalize}(M_t)$$

### 3. 为什么 Muon 不需要 Dampening？

因为**正交化操作会完全抹除任何标量缩放因子**。

如果我们在 Muon 中强行加入一个类似 `dampening` 的参数，将当前的梯度缩小为一个常数倍 $c$（例如 $c = 1 - \text{dampening}$），这只会导致累积的动量矩阵 $M$ 按比例缩放为 $cM$。

但是，正交化函数具有以下数学性质：
$$\text{Orthogonalize}(cM) = \text{Orthogonalize}(M)$$

<details>
<summary><b>点击展开：尺度不变性的数学证明</b></summary>

矩阵 $M$ 的理想正交化（Polar Decomposition 的正交部分）可以表示为：
$$O = M(M^T M)^{-1/2}$$

假设我们因为应用了 `dampening`，使得动量矩阵变成了一个缩放后的矩阵 $cM$（其中 $c$ 是标量）：

$$O_{scaled} = (cM) ((cM)^T (cM))^{-1/2}$$
$$O_{scaled} = cM (c^2 M^T M)^{-1/2}$$
$$O_{scaled} = cM \cdot \frac{1}{\sqrt{c^2}} (M^T M)^{-1/2}$$
$$O_{scaled} = \frac{c}{c} M (M^T M)^{-1/2} = M (M^T M)^{-1/2} = O$$

可以看到，标量 $c$ 在分子和分母中被完全抵消了。
</details>

### 总结

*   **SGD** 需要 `dampening`，因为它把梯度当成“长度+方向”的向量，必须控制新加进来的“长度”以免动量过大。
*   **Muon** 不需要 `dampening`，因为它强制把更新矩阵变成一个正交矩阵（可以理解为高维空间中的“单位向量”）。无论你输入多大的梯度，经过 Muon 内部的 Newton-Schulz 迭代后，更新步长的规模（Scale）都会被重置为 1，然后严格由学习率（`lr`）来控制。

因此，在 Muon 中引入 `dampening` 参数不仅是多余的，而且在数学上没有任何实际效果。