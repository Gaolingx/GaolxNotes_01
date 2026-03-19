# 为何不用二阶导数替代Adam和RMSProp

### 🧑‍💻 User

Adam/RMSProp scale the individual elements of the gradient vector based on a heuristic that comprises the computation of running mean and variance of the gradient vectors over several time steps.
An alternative would be to scale the elements of the gradient vector based on the second order derivative of that particular element with respect to the loss function. This is equivalent to Newton's method with diagonal entries. This can be computed recursively similar to backpropagation. Why is this not preferable to the other heuristics like Adam and RMSProp?

---

### 🤖 Assistant

Scaling the gradient by the inverse of the diagonal of the Hessian matrix is a very intuitive idea. As you noted, it is equivalent to a **Diagonal Newton's Method**, where the update rule for a parameter $\theta_i$ would look like:

$$ \theta_i^{(t+1)} = \theta_i^{(t)} - \eta \left( \frac{\partial^2 L}{\partial \theta_i^2} \right)^{-1} \frac{\partial L}{\partial \theta_i} $$

While mathematically elegant for convex problems, this approach is generally not preferred over first-order adaptive heuristics like Adam or RMSProp in deep learning for several fundamental reasons.

Here is a breakdown of why Adam and RMSProp remain the standard over diagonal second-order methods:

<details open>
<summary><b>1. The Computational Complexity Myth</b></summary>

You mentioned that the diagonal of the Hessian can be computed recursively similar to backpropagation. **This is actually a common misconception.** 

While you can compute a *Hessian-vector product* (HVP) efficiently in $O(N)$ time using Pearlmutter's trick (two passes of backprop), you **cannot** extract the exact diagonal of the Hessian in a single forward/backward pass. 

To compute the exact diagonal of the Hessian $H$, you would theoretically need to compute $N$ Hessian-vector products (where $N$ is the number of parameters, often in the millions or billions), making the complexity $O(N^2)$. 
To get around this, we have to use randomized algorithms (like Hutchinson's estimator) to approximate the diagonal:
$$ \text{diag}(H) \approx \mathbb{E}[z \odot (Hz)] $$
where $z$ is a random vector with Rademacher or Gaussian entries. This requires multiple forward-backward passes per optimization step to get a low-variance estimate, which is vastly more expensive than Adam's $O(1)$ memory and compute overhead per parameter.
</details>

<details open>
<summary><b>2. Non-Convexity and Negative Curvature</b></summary>

Deep neural network loss landscapes are highly non-convex, meaning they are riddled with saddle points and regions of negative curvature. 

In Newton's method, if the second derivative $\frac{\partial^2 L}{\partial \theta_i^2}$ is negative, the update direction is flipped. Instead of moving towards a minimum, the optimizer will actively move towards a **maximum** (ascent instead of descent). 

To fix this, you have to use a pseudo-Newton method where you take the absolute value of the Hessian diagonal, or add a damping term:
$$ H_{ii}' = |H_{ii}| + \lambda $$
However, near saddle points, $H_{ii}$ approaches zero. This causes the step size to explode precisely in regions where the gradient is also close to zero, often catapulting the weights out of good regions. Adam avoids this because the denominator $\sqrt{v_t}$ is strictly positive and bounded by the historical magnitude of the gradients, not the curvature.
</details>

<details>
<summary><b>3. Mini-Batch Noise and Stochasticity</b></summary>

In deep learning, we almost never compute the true loss $L$; we compute a stochastic estimate based on a mini-batch. 
Second-order derivatives are significantly more sensitive to noise than first-order derivatives. If you compute the diagonal Hessian on a small mini-batch, the variance of that estimate will be enormous. 

Adam and RMSProp inherently solve this stochasticity problem by taking an **Exponential Moving Average (EMA)** of the squared gradients over time:
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $$
This acts as a low-pass filter, smoothing out the noise of individual mini-batches. While you *could* take an EMA of the Hessian diagonal, the underlying noise of the second derivative often makes the signal-to-noise ratio too poor to be useful without massive batch sizes.
</details>

<details>
<summary><b>4. The Generalization Gap (Sharp vs. Flat Minima)</b></summary>

Empirically, second-order methods (and their diagonal approximations) tend to converge extremely fast to the *closest* local minimum. Because they adapt perfectly to the curvature, they easily dive into very sharp, narrow valleys in the loss landscape.

First-order methods (like SGD) and pseudo-first-order methods (like Adam) have a regularization effect. Their "imperfect" steps and inherent noise cause them to bounce out of sharp minima and settle into **flat minima**. In deep learning theory, flatter minima correlate strongly with better generalization to unseen data. Second-order methods often achieve near-zero training loss but suffer on validation metrics.
</details>

### The Hidden Connection: Adam *is* an approximation of the Hessian
It is worth noting that Adam and RMSProp are actually closely related to second-order methods. 

Under certain conditions (specifically, when the model is near a minimum and we are using log-likelihood losses), the uncentered variance of the gradients—which is exactly what RMSProp and Adam compute in their denominator—approximates the diagonal of the **Empirical Fisher Information Matrix**. 

The Fisher Information Matrix $F$ is asymptotically equivalent to the Hessian $H$ (this is known as the Gauss-Newton approximation). Therefore, scaling by $\frac{1}{\sqrt{v_t}}$ in Adam can be mathematically interpreted as taking a step pre-conditioned by the square root of the diagonal of the Fisher Information Matrix. 

**Summary:**
Adam and RMSProp give us the best of both worlds. They approximate the geometry of the loss landscape (similar to the diagonal Hessian) using only first-order gradients, guaranteeing positive-definite scaling, avoiding the $O(N)$ computational bottleneck of second derivatives, and inherently smoothing out mini-batch noise.