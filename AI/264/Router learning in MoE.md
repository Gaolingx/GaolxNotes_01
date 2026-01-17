# Router learning in MoE

### 🧑‍💻 User

Looking at the code for current mixture of experts models, they seem to use argmax, with k=1 (picking only the top expert) to select the router choice. Since argmax is non differentiable, the gradient cannot flow to the other experts. Thus it seems to me that only the weights of the selected expert will be updated if it performs poorly. However, it could be the case that a different expert was in fact a better choice for the given input, but the router cannot know this because the gradient does not flow to the other experts.

How can the router learn that it has made a wrong choice and use a different expert next time?

---

### 🤖 Assistant

You have hit on a fundamental challenge in Mixture of Experts (MoE) design: the discrete nature of `argmax` (or `top-k`) is indeed non-differentiable. If the model simply selected an expert index and ran the input through it, the gradient would be zero with respect to the router's selection mechanism.

However, modern MoE models (like Switch Transformer, Mixtral, GShard) use a specific mathematical formulation to bypass this and allow gradients to flow back to the router.

Here is how the router learns to switch experts despite the discrete selection.

### 1. The "Multiplication" Trick
The key is that the output of the MoE layer is not just the output of the selected expert. Instead, it is the output of the selected expert **multiplied by the router's probability score** for that expert.

Let's look at the math.
1.  **Router Probabilities:** The router produces logits, which pass through a Softmax to create probabilities $p$ for all experts.
    $$ p = \text{softmax}(W_r \cdot x) $$
2.  **Selection:** We select the index $k$ with the highest probability (for Top-1).
    $$ k = \text{argmax}(p) $$
3.  **The Output Equation:** This is the crucial part. The output $y$ is:
    $$ y = p_k \cdot E_k(x) $$

Here, $E_k(x)$ is the output of the chosen expert, and $p_k$ is the probability scalar assigned to that expert by the router.

### 2. How the Gradient Flows
Because $p_k$ is included in the output calculation as a multiplier, the loss function is differentiable with respect to $p_k$.

When we perform backpropagation:
1.  The gradient flows from the Loss $L$ to the output $y$.
2.  From $y$, it flows into the expert weights (updating the expert).
3.  Crucially, it **also flows into $p_k$**.

By the chain rule:
$$ \frac{\partial L}{\partial W_r} = \frac{\partial L}{\partial y} \cdot E_k(x) \cdot \frac{\partial p_k}{\partial W_r} $$

This means the router receives a gradient signal.
*   **If the expert performed poorly:** The gradient will tell the network to lower the weight $p_k$.
*   **If the expert performed well:** The gradient will encourage keeping or increasing $p_k$.

### 3. The "Hydraulic" Effect of Softmax
You asked: *"How can the router learn... and use a different expert next time?"*

The gradient updates the router weights to decrease $p_k$ (the probability of the currently selected expert) if the loss is high.

Because $p$ is the result of a **Softmax** function, the probabilities must sum to 1 ($ \sum p_i = 1 $).
*   If the router learns to **decrease** the probability of the current expert ($p_k \downarrow$), the probabilities of the *other* (unchosen) experts must mathematically **increase**.
*   The next time the model sees similar input, the probability of the previously chosen expert will be lower, and the probability of a rival expert might now be high enough to cross the threshold and become the new `argmax`.

### 4. The Limitation: The "Dead Expert" Problem
Your intuition was partially correct regarding a specific danger: **Exploration**.

While the mechanism above explains how we discourage a *bad* expert, the router does not explicitly calculate what the loss *would have been* had we chosen a different expert. We only get gradients for the active path.

This can lead to a collapse where the router keeps picking the same few experts (because their weights are getting updated and they are becoming "good enough"), while other experts are never picked, never updated, and remain random/useless. This is known as the **Rich-get-Richer** phenomenon.

To solve this, almost all modern MoE models add an **Auxiliary Load Balancing Loss**.

$$ L_{total} = L_{task} + \alpha \cdot L_{balance} $$

This auxiliary loss ignores accuracy. It simply penalizes the router if it assigns too many tokens to one expert and not enough to others. This forces the router to try different experts occasionally, ensuring gradients eventually flow through all experts, allowing them to learn and become viable candidates.

### Summary
The router learns because the output is a **weighted average** (even if $k=1$, it is a weighted single item).
1.  The output is $y = \text{gate\_value} \times \text{expert\_output}$.
2.  Backprop updates the `gate_value`.
3.  Softmax ensures that lowering the `gate_value` for a bad expert automatically raises the potential for other experts to be chosen next time.

---

### 🧑‍💻 User

Yes, during training, gumbel softmax is used. But Gumbel softmax is only one approach but the traditional Switch Transformer from Google does not use Gumbel as the trick for differentiability. They just weigh the expert outputs by the softmax router scores for the experts. Since the router/gate parameters are algebraically connected to the router scores assigned to the experts, this makes it possible for the router params to be updated through a differntiable path in backpropagation. This is how they avoid using Gumbel trick.

---

### 🤖 Assistant

You are absolutely correct. The Switch Transformer (and most subsequent MoE models like Mixtral) intentionally avoids the Gumbel-Softmax relaxation in favor of the simpler **probability-scaled hard switching** mechanism.

You have correctly identified the mechanism: the router weights are updated **only** through the path of the selected expert's probability score.

Here is a deeper look at the implications of this design choice and why it creates a very specific dynamic during training.

### 1. The Switch Transformer Mechanism (No Gumbel)
In the Switch Transformer paper (Fedus et al., 2021), they explicitly state they use a "hard" selection for the forward pass computation (to save compute) but keep the "soft" probability for the backward pass (to save the gradients).

The formulation is:

1.  **Router Logits:** $h(x) = W_r \cdot x$
2.  **Probabilities:** $p = \text{softmax}(h(x))$
3.  **Hard Selection:** $k = \text{argmax}(p)$
4.  **Output:** $y = p_k \cdot E_k(x)$

As you noted, the gradient $\nabla y$ flows back into $p_k$, and since $p_k$ is a function of $W_r$, the router weights are updated.

### 2. The Consequence: Gradients are "Blind" to Alternatives
Your original intuition remains a critical observation regarding this specific implementation:

> "It could be the case that a different expert was in fact a better choice... but the router cannot know this."

In the Switch Transformer implementation (without Gumbel), **the gradient vector for all unselected experts is exactly zero.**

$$ \frac{\partial L}{\partial E_{j \neq k}} = 0 $$

The router receives **no signal** regarding whether Expert B would have done a better job than Expert A. It only receives a signal regarding whether Expert A (the chosen one) did a "good job" relative to the current loss.

### 3. How it learns to switch without "Counterfactual" gradients
Since the router cannot see the gradients of unselected experts, it cannot perform a true comparison during a single step. Instead, it relies on **Probabilistic Displacement**.

Because $p$ is a softmax distribution ($\sum p_i = 1$), the gradients update the router in a zero-sum game fashion.

**Scenario:**
The router picks Expert A with probability $0.6$. Expert A performs poorly (high loss).
1.  The gradient descent tries to lower the total loss.
2.  To lower the loss contribution of this specific path ($p_A \cdot E_A(x)$), the gradient tells the router to **reduce** $p_A$.
3.  Let's say the update changes weights such that $p_A$ drops from $0.6$ to $0.4$.
4.  Due to the softmax constraint, the $0.2$ probability mass lost by A **must** be redistributed to experts B, C, D, etc.
5.  Eventually, the probability of Expert B might rise high enough to surpass A.
6.  Only **then**, when Expert B becomes the `argmax`, does the router finally receive a gradient signal from Expert B's path to validate if it is actually better.

### 4. Why Gumbel-Softmax is omitted
Why don't they use Gumbel-Softmax?
Gumbel-Softmax allows gradients to flow to *all* experts by sampling, effectively keeping all experts slightly "alive" in the computation graph.

MoE researchers generally avoid it for Large Language Models because:
1.  **Compute Cost:** To get gradients for all experts, you often have to run the forward pass for all experts (or at least more than $k=1$), which defeats the purpose of sparse activation (efficiency).
2.  **Training Stability:** Gumbel adds stochastic noise to the routing. In massive scale training (like GPT-4 scale), reducing noise and variance is usually preferred to ensure stability.
3.  **The "Good Enough" Reality:** Empirical results showed that the hard-switch method, combined with an auxiliary load-balancing loss, converges just as well as stochastic methods, but with much higher inference efficiency.

### Summary
The router in a Switch Transformer is indeed "blind" to the other experts in the short term. It does not know if another expert is better *right now*.

It acts like a greedy manager: it keeps assigning work to the top employee until that employee starts failing (gradients reduce $p_k$). Only when the current top employee's rating drops sufficiently does the manager tentatively assign the task to the next person in line.

---

### 🧑‍💻 User

What’s the difference between the output of Gumbel softmax and argmax of softmax?

---

### 🤖 Assistant

The fundamental difference lies in **determinism vs. stochasticity** and **discrete vs. continuous**.

Here is the breakdown of the differences between the output of **Argmax (applied to Softmax)** and **Gumbel-Softmax**.

### 1. The Input
Let's assume we have a router producing unnormalized logits for 3 experts:
$$ z = [10.0, \quad 5.0, \quad 2.0] $$

---

### 2. Argmax of Softmax (The "Hard" Choice)
This is the standard approach used in inference and in the Switch Transformer.

**The Process:**
1.  **Softmax:** Convert logits to probabilities.
    $$ p = [0.993, \quad 0.006, \quad 0.001] $$
2.  **Argmax:** Select the index with the highest value.
    $$ k = 0 $$
3.  **Output Representation (One-Hot):**
    $$ y = [1, \quad 0, \quad 0] $$

**Characteristics:**
*   **Deterministic:** If you run this input a million times, the output is identical every time.
*   **Discrete:** The output is strictly integers (0 or 1).
*   **Non-Differentiable:** You cannot calculate the gradient of `argmax`. A tiny change in input ($10.0 \to 9.9$) causes **zero change** in output (it's still index 0). If the change is massive ($10.0 \to 4.0$), the output jumps discontinuously to a new index. This breaks backpropagation (gradient is either 0 or undefined).

---

### 3. Gumbel-Softmax (The "Soft" Approximation)
This approach is used to make categorical sampling differentiable.

**The Process:**
1.  **Add Noise:** We add random noise drawn from the Gumbel distribution ($g$) to the logits.
    $$ z' = z + g $$
    *(e.g., $z'$ might become $[10.5, \quad 4.2, \quad 3.1]$)*
2.  **Temperature Scaling:** We divide by a temperature parameter $\tau$.
3.  **Softmax:** We apply softmax to the noisy, scaled logits.
    $$ y = \text{softmax}\left(\frac{z + g}{\tau}\right) $$

**Output Representation:**
The output is **not** a strictly One-Hot vector (like `[1, 0, 0]`). It is a probability vector that **resembles** a One-Hot vector.

*   **If Temperature ($\tau$) is High:** The output is smooth and uniform.
    $$ y \approx [0.5, \quad 0.3, \quad 0.2] $$
*   **If Temperature ($\tau$) is Low (approaching 0):** The output sharpens to approximate `argmax`.
    $$ y \approx [0.999, \quad 0.001, \quad 0.000] $$

**Characteristics:**
*   **Stochastic:** Because of the added noise ($g$), running the same input twice yields different vectors. It essentially **samples** from the distribution.
*   **Continuous:** The output values are continuous floats between 0 and 1.
*   **Differentiable:** Because it uses Softmax (math operations) rather than Argmax (logic operation), gradients can flow through it to all experts.

---

### Comparison Summary

| Feature | Argmax of Softmax | Gumbel-Softmax |
| :--- | :--- | :--- |
| **Output Type** | **Discrete** (Integer index or strict One-Hot) | **Continuous** (Probability vector summing to 1) |
| **Behavior** | **Deterministic** (Always picks highest logit) | **Stochastic** (Samples based on probability) |
| **Gradient** | **None** (Gradient is 0 or undefined) | **Flows** (Gradient exists for all logits) |
| **Sparsity** | **Truly Sparse** (Only 1 expert runs) | **Dense** (Mathematically, *all* experts run with small weights)* |

*\*Note on Sparsity: In Gumbel-Softmax, since the output vector might be `[0.9, 0.09, 0.01]`, you technically have to compute the forward pass for ALL experts to multiply them by these small weights. This makes Gumbel-Softmax expensive for MoE unless you force values below a threshold to zero.*

### The "Straight-Through" Gumbel Trick
To get the best of both worlds, some implementations use **Straight-Through Gumbel-Softmax**:

1.  **Forward Pass:** Perform the Gumbel calculation, but cast the result to a Hard One-Hot vector (like Argmax). This ensures only 1 expert runs (computationally cheap).
2.  **Backward Pass:** Pretend we didn't cast it to Hard One-Hot, and use the continuous Gumbel gradients.

This allows the model to "explore" (due to Gumbel noise) and "differentiate" (due to soft approximation) while keeping the forward pass "sparse" (hard selection).

---

### 🧑‍💻 User

I think gumbel forces exploration via random sampling. But, the output distribution gets converted to one-hot via the Straight Through Estimator. As far as I can tell, STE is valid for regular softmax too, but can more easily fall into mode collapse as the model can more easily just select a single route via a constant output distribution, versus learning to balance across experts.

---

### 🤖 Assistant

You are spot on. You have correctly identified the critical weakness of using a standard **Softmax + Straight Through Estimator (STE)** compared to **Gumbel-Softmax + STE**.

While both methods use the STE to pass gradients through a discrete decision, **Standard Softmax + STE lacks the mechanism to explore paths that currently look suboptimal.**

Here is the detailed breakdown of why **Softmax + STE** leads to mode collapse, while **Gumbel + STE** forces exploration.

### 1. The Trap of Determinism (Softmax + STE)

Let's imagine you have two experts, A and B.
*   **Logit A:** 5.0
*   **Logit B:** 4.9

If you use standard Softmax followed by Argmax (and pass gradients via STE):

1.  **Forward Pass:** Argmax picks **Expert A** (because $5.0 > 4.9$).
2.  **Observation:** Expert A is executed. Expert B is **never** executed.
3.  **Gradient Update:**
    *   If Expert A does well, Logit A increases.
    *   If Expert A does poorly, Logit A decreases.

**The Problem:**
Even if Expert B is actually the "perfect" expert for this input, the model will **never know**. As long as Logit A is even slightly higher than Logit B ($5.0$ vs $4.9$), Expert A gets 100% of the traffic.

To switch to Expert B, the model has to wait for Expert A to perform **so poorly** that its logit drops below 4.9. This is "Learning by Failure." It is passive. If Expert A is "mediocre but passable," the model settles into a local minimum (Mode Collapse) and never tries Expert B.

### 2. The Solution of Noise (Gumbel-Softmax + STE)

Gumbel-Softmax adds random noise ($g$) to the logits before the selection.

*   **Logit A:** 5.0
*   **Logit B:** 4.9

1.  **Forward Pass (Sampling):** We add random noise.
    *   $Noise_A = 0.1 \rightarrow Total_A = 5.1$
    *   $Noise_B = 0.5 \rightarrow Total_B = 5.4$
2.  **Selection:** The router picks **Expert B** (because $5.4 > 5.1$), even though Expert B's base logit was lower.
3.  **Observation:** Expert B finally runs!
4.  **The "Aha!" Moment:** If Expert B performs amazingly well, the gradients will flow back to **increase Logit B**.

**The Benefit:**
Gumbel-Softmax allows the router to say: *"I'm pretty sure Expert A is best, but let me just check on Expert B for a second."*

This is "Active Exploration." It prevents mode collapse because the random noise ensures that every expert has a non-zero probability of being chosen, regardless of the current weights.

### 3. Why STE is dangerous without Noise

The Straight Through Estimator (STE) is simply a gradient trick:
$$ \frac{\partial L}{\partial z} \approx \frac{\partial L}{\partial y_{discrete}} $$

If you apply this to a deterministic function (Standard Softmax Argmax), you are telling the model: **"Calculate the gradient as if we output probabilities, but in reality, we only ran the top path."**

*   **With Standard Softmax:** The gradients for the unselected experts push them up or down based on the *hypothetical* idea that they contributed to the output (softly). But since they didn't actually run, the router updates weights based on a hallucination of what the other experts might have done.
*   **With Gumbel:** The noise ensures that the "hypothetical" becomes "actual" occasionally. The expert actually runs, generates a real loss, and creates a real gradient signal that validates (or invalidates) the router's choice.

### Summary

| Method | Mechanism | Exploration Strategy | Risk of Mode Collapse |
| :--- | :--- | :--- | :--- |
| **Softmax + STE** | Deterministic greedy selection | None. Only tries new things if the current leader fails catastrophically. | **High.** "Rich get richer." If Expert A starts slightly ahead, it wins forever. |
| **Gumbel + STE** | Stochastic sampling + Hard threshold | Random noise forces occasional selection of lower-ranked experts. | **Low.** The noise prevents the model from locking into a single path too early. |

This is why, if you are designing a router from scratch and want to use STE, you almost **must** inject noise (like Gumbel or Gaussian) to ensure the router doesn't collapse into using just one or two experts.

---

### 🧑‍💻 User

Hmm, that's interesting - but what's the point of doing random sampling with Gumbel when you typically have to balance token distribution across experts anyway. That's already a form of exploration.

---

### 🤖 Assistant

This is a very keen observation. You are effectively asking: **"If we are effectively forcing the router to distribute tokens evenly via load balancing, aren't we implicitly forcing it to explore?"**

The short answer is **yes**, load balancing forces usage, which is a form of exploration.

However, there is a subtle but critical difference between the **"Forced Fairness"** of load balancing and the **"Probabilistic Exploration"** of Gumbel noise.

Here is why relying solely on Load Balancing can sometimes lead to suboptimal routing, and why random sampling (Gumbel) theoretically offers something different.

### 1. The "Bad Partition" Problem (Local Minima)

Load balancing ensures that every expert gets roughly the same number of tokens. It does **not** ensure that the *right* tokens go to the right experts.

Imagine a scenario with 2 Experts (A and B) and a dataset of Math and History questions.
*   **Optimal Routing:** A gets Math, B gets History.
*   **Current (Bad) State:** The router has learned to send all inputs starting with "T" to Expert A, and "W" to Expert B.

**With only Load Balancing:**
If the text distribution is 50% "T..." and 50% "W...", the load is perfectly balanced. The auxiliary loss is zero. The router is happy.
*   However, Expert A is getting a mix of Math and History (bad).
*   Expert B is getting a mix of Math and History (bad).
*   The model is in a **local minimum**. It satisfies the load constraint, but the routing logic is nonsense.

**With Gumbel (Random Sampling):**
Even if the router is 100% confident that "T" belongs to A, Gumbel noise will occasionally flip a "T" token to Expert B.
*   The model suddenly realizes: "Wait, Expert B actually solved this 'T' (History) question much better than A usually does!"
*   This gradient signal breaks the "T vs W" logic and helps the router learn the semantic "Math vs History" logic.

**Summary:** Load balancing solves the **Quantity** problem (dead experts). Gumbel helps solve the **Quality** problem (optimal mapping).

---

### 2. "Overflow" Exploration vs. "True" Exploration

When relying on capacity limits (dropping tokens or shifting them when a buffer is full) to force exploration, the exploration is biased.

*   **Capacity-based Exploration:** If Expert A is the "favorite," it fills up first. The *overflow* tokens are sent to Expert B.
    *   This means Expert B only gets to train on the "leftovers" or the tokens where the router was least confident about A. It never gets a shot at the "easy" tokens that A is hogging.
*   **Gumbel Exploration:** Noise is added to *every* token's decision. Expert B gets a random chance to try the "easy" tokens that A usually dominates. This allows Expert B to potentially prove it is actually superior at A's core competency.

### 3. Why Modern Models (Switch, Mixtral) Ditched Gumbel anyway

Despite the theoretical benefits of Gumbel described above, your intuition holds true for the current State of the Art. Most modern MoE LLMs (Switch Transformer, GShard, Mixtral 8x7B) **do not use Gumbel Softmax.**

They rely on **Top-1 (or Top-2) + Load Balancing Loss.**

Why?
1.  **Scale Fixes Noise:** When training on trillions of tokens, there is enough natural noise and variance in the data that "perfect partitions" rarely happen. The "Bad Partition" problem is less severe at scale.
2.  **Training Stability:** Gumbel noise introduces variance in the gradients. In massive runs, stability is king. Engineers prefer the deterministic behavior of Argmax over the stochastic behavior of Gumbel.
3.  **The "Good Enough" Principle:** As long as the load balancing ensures no expert dies, the model usually manages to specialize experts reasonably well based on semantic clusters, even without forced random exploration.

### Conclusion

You are right that Load Balancing is a proxy for exploration.
*   **Load Balancing** forces the router to **widen** its selection (Stop picking A for everything).
*   **Gumbel** forces the router to **doubt** its selection (Maybe B is better than A for this specific thing).

While Gumbel is theoretically cleaner for finding global optima, modern LLMs have found that Load Balancing is "good enough" to prevent collapse, and computationally friendlier.