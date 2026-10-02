# Softmax 与 Softplus 区别

## 🧑‍💻 User

softmax和softplus有什么区别？

---
## 🤖 Assistant

# Softmax vs. Softplus

They are **not** substitutes for each other — different input shapes, different purposes — but they are closely related mathematically.

## Definitions

**Softmax** — takes a **vector** $\mathbf{x} = (x_1, \dots, x_n)$ and returns a probability distribution:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Outputs are positive and sum to 1: $\sum_i \text{softmax}(x_i) = 1$. Used as the final layer of a **multi-class classifier**.

**Softplus** — takes a **scalar** and returns a positive scalar:

$$
\text{softplus}(x) = \ln(1 + e^x)
$$

It's a smooth approximation of $\text{ReLU}(x) = \max(0, x)$. Used as an **activation function** in hidden layers (and to guarantee positive outputs).

## Key Differences

| Property | Softmax | Softplus |
|---|---|---|
| Input | Vector | Scalar (element-wise) |
| Output | Probability vector | Positive scalar |
| Output sums to 1? | Yes | No |
| Range | $(0,1)$ each, sum $=1$ | $(0, \infty)$ |
| Main use | Output layer, multi-class | Hidden activation, smooth ReLU |
| Shape-preserving | No (couples elements) | Yes (element-wise) |

## The Mathematical Connection

They are two faces of the same object — the **log-sum-exp** (LSE) function:

$$
\text{LSE}(\mathbf{x}) = \ln\!\sum_j e^{x_j}
$$

- **Softmax is the gradient of LSE:**
$$
\frac{\partial}{\partial x_i}\,\text{LSE}(\mathbf{x}) = \text{softmax}(x_i)
$$

- **Softplus is the 1-D / "0-anchored" case of LSE:**
$$
\text{softplus}(x) = \ln(1 + e^x) = \text{LSE}(0, x)
$$

In short: **softplus is the scalar version of log-sum-exp, softmax is its derivative.** That's why softplus is sometimes called the "smooth maximum" for a single value, just as softmax is the smooth version of `argmax`.

## Relation to Sigmoid

A useful bridge: the derivative of softplus is the **sigmoid**:

$$
\frac{d}{dx}\text{softplus}(x) = \frac{e^x}{1+e^x} = \sigma(x)
$$

And **softmax is the multivariate generalization of the sigmoid** (with 2 classes, softmax reduces exactly to sigmoid). So the analogy is:

$$
\text{sigmoid} \;\longleftrightarrow\; \text{softmax}, \qquad \text{softplus} \;\longleftrightarrow\; \text{log-sum-exp}
$$

## Practical Note

Softplus avoids the "dead ReLU" gradient problem but is slightly more expensive to compute. It's also the standard way to parameterize positive quantities (e.g., variance $\sigma^2 = \text{softplus}(x)$). Softmax, by contrast, is almost always paired with cross-entropy loss for classification.
