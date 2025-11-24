# Router learning in MoE

### 🧑‍💻 User

Looking at the code for current mixture of experts models, they seem to use argmax, with k=1 (picking only the top expert) to select the router choice. Since argmax is non differentiable, the gradient cannot flow to the other experts. Thus it seems to me that only the weights of the selected expert will be updated if it performs poorly. However, it could be the case that a different expert was in fact a better choice for the given input, but the router cannot know this because the gradient does not flow to the other experts.

How can the router learn that it has made a wrong choice and use a different expert next time?

---

### 🤖 Assistant

Short answer: routers **are** trained so they can learn from mistakes even when you use a hard top-1 at inference — you just make the routing *differentiable or learnable* during training (or give it a learning signal via auxiliary losses / policy gradients). Common tactics are **noisy soft/top-k gating**, **straight-through / Gumbel relaxations**, **auxiliary load/importance losses**, or (less common) **policy-gradient** methods. Those let gradient (or a low-variance surrogate) tell the router that it picked the “wrong” expert.

Below is a compact explanation of the main approaches, why they work, and short pseudo-code you can map to typical MoE implementations.

# Why argmax/top-1 looks bad but can be fixed

Argmax/top-1 is non-differentiable, so if you literally forward only one expert and backprop only through that discrete path, the router parameters get no gradient that says “you should have chosen expert B instead of A.” But we can *train* the router so it *knows* when it made a mistake by either:

1. Making the gating **soft/differentiable** during training (so gradients flow to all experts and the gate), or
2. Using a **hard forward** but a differentiable **backward surrogate** (straight-through), or
3. Treating routing as a discrete action and using **policy gradient / REINFORCE** (usually high variance), or
4. Adding **auxiliary losses** that give explicit learning signals to the router (e.g., load-balancing / importance losses).

# Common techniques (practical, used in Switch/GShard-style MoE)

1. **Noisy Top-K Gating (soft + top-k)**

   * Compute router logits (z = W_g x). Add noise: (\tilde z = z + \mathrm{noise}).
   * Compute softmax over (\tilde z) → probabilities (p_j).
   * Select the top-k experts by value for efficiency, but normalize the probabilities over the top-k.
   * During backward you use gradients through the softmax (or through the expected output). Noise encourages exploration so router can discover better experts.
   * This is the practical training regime in many MoE papers.

   Pseudo:

   ```
   logits = Wg @ x
   logits += random_noise(scale=sigma)    # exploration
   probs = softmax(logits)
   topk_idx = top_k_indices(logits, k)   # hard choice for routing
   gated_probs = renormalize(probs[topk_idx])
   output = sum_j gated_probs[j] * Expert[j](x)
   ```

   Gradients flow back into `Wg` because `probs` were computed by softmax (and noise gives gradients that encourage trying different experts).

2. **Straight-Through / Gumbel-Softmax (hard forward, soft/backward)**

   * Forward: use hard top-1 (argmax) for efficiency.
   * Backward: pretend the gating was soft (use softmax/Gumbel-softmax gradients) — i.e., use the continuous relaxation as the backward pass (straight-through estimator).
   * This gives a low-bias, low-cost surrogate gradient so router gets signals about alternative experts.

3. **Fully Soft Mixture (k>1 or fractional routing)**

   * Route fractions of the token to multiple experts: (y = \sum_j p_j , E_j(x)).
   * This is fully differentiable: every expert and the router receive gradients proportional to (p_j). But it is expensive (computational & memory) so often used only during training or for small models.

4. **Auxiliary Load / Importance Losses**

   * Even if the gating is (mostly) hard, you add explicit losses that push the router to *explore* and *balance* expert usage. Typical form: penalize variance in expert usage / importance so the router can’t collapse to one expert.
   * These losses produce gradients that flow into the router parameters and cause it to change its assignments if its distribution is bad. (E.g., encourage uniform load across experts or penalize very skewed importance.)

5. **Policy-gradient (REINFORCE or critic)**

   * Treat routing as discrete actions and train the router with REINFORCE using the downstream loss as a reward. Works but has high variance; usually used with baselines/critics and variance reduction techniques.

# Intuition for “how it learns it was wrong”

* If you use **soft gating** (or straight-through / Gumbel), the router’s logits determine a probability distribution over experts and the loss gradient flows back through those probabilities. If a different expert would have produced lower loss, the gradient will push the router logits to increase that expert’s probability next time.
* If you use **noise + auxiliary load/importance loss**, the router is encouraged to try alternatives and not collapse; poor performance from the selected expert will (through the differentiable surrogate) reduce the router’s score for it and increase others.
* If you use **policy gradients**, negative reward from a bad expert reduces the probability of picking it in future, by the standard RL gradient.

# Practical recipe used in practice (what many large MoE implementations do)

* Train with **noisy top-k gating** and a **softmax-based probability**; use top-1 for efficiency at inference.
* Add a **load-balancing/importance** auxiliary loss so router doesn’t collapse and explores other experts.
* Optionally use **straight-through / Gumbel relaxations** if you must use hard routing during training for memory reasons.
* Avoid naive REINFORCE unless you have a good variance baseline.

# Short code sketch (training-friendly)

```python
# logits = Wg x
logits = Wg @ x
logits += torch.randn_like(logits) * noise_scale   # noisy gating
probs = softmax(logits)                            # differentiable
topk = probs.topk(k)                               # choose top-k (for efficiency)
gated_probs = renormalize(probs[topk.indices])
# expected output (differentiable w.r.t probs and Wg)
y = sum(gated_probs[j] * Expert[topk.indices[j]](x) for j in range(k))
loss = task_loss(y, target) + lambda_bal * load_loss(probs)
loss.backward()
```

# Tradeoffs and practical tips

* Soft routing → low variance gradients but expensive.
* Straight-through/Gumbel → good cost/benefit tradeoff (hard forward, soft backward).
* Noise scale and auxiliary loss weight are hyperparameters — tune them to get exploration without destabilizing experts.
* The load-balancing term is essential in practice to avoid expert collapse.

---

If you want, I can:

* show a small concrete PyTorch example of noisy top-1 gating + load loss, or
* summarize how Switch / GShard implement the auxiliary losses (with simple formulae). Which would help you most?


---

### 🧑‍💻 User

Yes, during training, gumbel softmax is used. But Gumbel softmax is only one approach but the traditional Switch Transformer from Google does not use Gumbel as the trick for differentiability. They just weigh the expert outputs by the softmax router scores for the experts. Since the router/gate parameters are algebraically connected to the router scores assigned to the experts, this makes it possible for the router params to be updated through a differntiable path in backpropagation. This is how they avoid using Gumbel trick.

---

### 🤖 Assistant

Exactly — that’s right, and your understanding is on point.

Short version: Switch-style routers make the router parameters algebraically affect the *scalar gate scores* (softmax probs) which multiply the expert outputs during training. Because the final output is (gate * expert_output), the loss gradient flows back through the gate probabilities into the router logits and hence into the router weights — so the router *can* learn that it picked the wrong expert even though the forward pass uses a hard top-1 dispatch for efficiency.

Below I’ll step through the key math and practical implications, then give a compact PyTorch-style sketch of how Switch / noisy top-1 gating is implemented.

# How gradients actually flow

Let (z_j = w_j^\top x) be the router logits for experts (j). Switch-style training computes
[
p_j = \mathrm{softmax}(z)*j = \frac{e^{z_j}}{\sum_k e^{z_k}}.
]
If we do top-1 dispatch but still include the gate weight, the output for that token (assuming top-1 chosen is (j^*)) is
[
y = p*{j^*},E_{j^*}(x).
]
The scalar (p_{j^*}) depends algebraically on every (z_j) (and therefore on the router parameters). So by chain rule
[
\frac{\partial \mathcal L}{\partial w} = \frac{\partial \mathcal L}{\partial y}\frac{\partial y}{\partial p_{j^*}}\frac{\partial p_{j^*}}{\partial z}\frac{\partial z}{\partial w},
]
which gives a nonzero learning signal to the router weights (w). Concretely, if the chosen expert produces a large loss, the gradient can push down (p_{j^*}) (and alter other (p_j)), making other experts more likely next time.

# What this *doesn't* do

* If you only compute and backprop through the single expert (E_{j^*}) (no soft mixture), then **other experts' parameters** receive no direct gradient from this token because they were never executed. That is, the router learns, but non-executed experts do not get updated by this token.
* The router must have mechanisms to *explore* alternatives and not collapse to a few experts. That’s why Switch-style training usually includes noise in logits and an auxiliary load/importance loss.

# Practical training tricks that make this work

1. **Noisy logits**: add small noise to (z) before softmax so the router explores different experts during training (helps find better assignments).
2. **Use the softmax probabilities in the backward path** (even if forward dispatch is top-1), as you described — that provides gradients to the router.
3. **Auxiliary load / importance losses**: penalize imbalanced expert usage (router learns to spread tokens and try alternatives).
4. (Optional) **Soft mixtures in training**: route to multiple experts with fractional weights for full differentiability (more expensive).

# Compact PyTorch-like sketch (Switch-style / noisy top-1)

```python
# x: [batch, d]
# Wg: gating matrix -> logits per expert
logits = x @ Wg.T                      # [batch, num_experts]
logits += torch.randn_like(logits) * noise_scale   # noisy gating
probs = torch.softmax(logits, dim=-1)             # differentiable probs

# top-1 choices for efficiency (forward computes only those experts)
top1_idx = probs.argmax(dim=-1)        # [batch]

# compute expert outputs only for selected experts
expert_outputs = dispatch_to_experts(x, top1_idx)  # list or tensor per token

# weigh the chosen expert outputs by their softmax probs
gated = probs.gather(1, top1_idx.unsqueeze(1)).squeeze(1) * expert_outputs

y = rest_of_model(gated)
loss = task_loss(y, targets) + lambda_bal * load_balance_loss(probs)
loss.backward()
```

# Intuition summary

* Router parameters are learnable because (p_j) is a differentiable function of the router logits; multiplying the expert output by (p_j) creates a path for gradients back into the router.
* Non-executed experts don’t get token-specific gradients if you physically skip their forward computation — which is why exploration (noise) and balancing losses are crucial so the router will *try* other experts and thus cause them to receive gradients on some tokens.
* At inference you can switch to pure hard top-1 for speed; the router has already learned useful assignments during training.

If you want, I can:

* show a full minimal PyTorch example that implements the load-balancing loss used in practice, or
* derive the exact gradient term (\partial p_{j^*}/\partial z_k) so you can see how the router logits for other experts are affected by a loss on the chosen expert. Which would be more useful?


---

### 🧑‍💻 User

What’s the difference between the output of Gumbel softmax and argmax of softmax?

---

### 🤖 Assistant

**Short, crisp answer:**

* **Argmax(softmax(logits))** returns a **one-hot**, **discrete**, **non-differentiable** choice.
* **Gumbel-Softmax(logits, τ)** returns a **continuous**, **differentiable**, **soft** probability vector that *approaches* one-hot as τ→0.
* With τ→0 and using **straight-through**, Gumbel-Softmax behaves like “hard argmax in the forward pass, soft probabilities in the backward pass.”

Below is the clean, precise comparison.

---

# 1. Output of `argmax(softmax(logits))`

Given logits (z), the normal softmax is:
[
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}.
]
Argmax gives:
[
\text{argmax}(p) = \text{index of largest } p_i.
]

**Properties**

* **Discrete** output: e.g. `[0,1,0,0]`.
* **Not differentiable** w.r.t logits.
* Gradient is **zero everywhere** except undefined at ties.
* No stochasticity unless you add exploration by hand.
* In MoE, this is used for *hard routing*, but cannot be trained through.

---

# 2. Output of **Gumbel-Softmax**

Gumbel-softmax samples from:
[
\tilde z_i = z_i + g_i,\quad g_i \sim \text{Gumbel}(0,1)
]
[
y_i = \frac{\exp(\tilde z_i/\tau)}{\sum_j \exp(\tilde z_j/\tau)}
]

**Properties**

* Output is a **real‐valued probability vector** in the simplex, e.g. `[0.02, 0.89, 0.07, 0.02]`.
* **Differentiable** w.r.t logits.
* **Stochastic**, so it naturally explores different experts.
* Temperature τ controls softness:

  * Large τ → almost uniform
  * Small τ → nearly one-hot
* In the τ→0 limit, softmax becomes “spiky”, approximating argmax, but still has a gradient.

---

# 3. How they differ in practice

### ▸ **Argmax(softmax)**

* Hard, non-differentiable, deterministic.
* No gradient flows to router from alternative experts.
* Only one expert gets activated.
* Good for inference, not directly trainable.

### ▸ **Gumbel-Softmax**

* Soft + differentiable (full gradient to router logits).
* Stochastic (explores experts).
* Temperature allows control of hardness.
* Can use **straight-through Gumbel-Softmax**:

  * Forward pass uses **hard argmax** (boolean mask).
  * Backward pass uses **soft y** (continuous relaxation).
  * This is the closest differentiable approximation to hard routing.

---

# 4. Key idea: Gumbel-Softmax is a *relaxation of argmax*

Mathematically:
[
\lim_{\tau \to 0} \text{GumbelSoftmax}(z, \tau) \rightarrow \text{one-hot distributed exactly like } \text{argmax}(z+g).
]
This means:

* Same *discrete distribution* as argmax over noisy logits.
* But with differentiability during training.

**Argmax(softmax)** cannot give you this — it’s just a deterministic, non-differentiable pick.

---

# 5. Visual intuition

| Method                  | Output                         | Differentiable? | Stochastic? | Approximates argmax?   |
| ----------------------- | ------------------------------ | --------------- | ----------- | ---------------------- |
| `argmax(softmax)`       | One-hot                        | ❌ No            | ❌ No        | Yes, exactly           |
| Gumbel-Softmax (τ=1)    | Soft                           | ✔️ Yes          | ✔️ Yes      | No                     |
| Gumbel-Softmax (τ→0)    | Nearly one-hot                 | ✔️ Yes          | ✔️ Yes      | Yes (soft version)     |
| Straight-through Gumbel | One-hot forward, soft backward | ✔️ Yes          | ✔️ Yes      | Yes (trainable argmax) |

---

If you'd like, I can also explain:

* Why Gumbel-Softmax approximates the **argmax of logits plus Gumbel noise**.
* Why Switch Transformer does *not* need Gumbel-Softmax (because it uses differentiable softmax gating).
* How straight-through works mathematically.

Just tell me which direction you want to go.


---

### 🧑‍💻 User

I think gumbel forces exploration via random sampling. But, the output distribution gets converted to one-hot via the Straight Through Estimator. As far as I can tell, STE is valid for regular softmax too, but can more easily fall into mode collapse as the model can more easily just select a single route via a constant output distribution, versus learning to balance across experts.

---

### 🤖 Assistant

You’re thinking exactly along the right lines. Short answer: **yes** — Gumbel + STE helps *because* the sampling injects per-token stochasticity (exploration), while STE alone applied to a deterministic softmax argmax can much more easily collapse to a constant routing policy. Below I’ll explain why that happens, what the concrete differences are in practice, and give a checklist of effective mitigations you can use if you see mode-collapse with softmax+STE.

---

## Why Gumbel+STE resists collapse while softmax+STE can collapse

1. **Stochastic perturbation vs deterministic selection.**

   * Gumbel-softmax samples `z + g` (Gumbel noise) per token, so the *forward* choices vary across tokens and minibatches even if the base logits are similar. That per-sample randomness forces the optimizer to see alternative expert assignments and gives a signal about their downstream loss.
   * Plain softmax + STE (hard forward, soft backward) can be *deterministic* in the forward pass: if the router learns logits that always pick expert A, there is no exploration and the model never sees losses from routing to other experts — so it can get stuck.

2. **Gradient signal / bias-variance tradeoff.**

   * STE is a biased estimator (it fakes a continuous gradient for a discrete forward). That bias exists whether you use Gumbel or not. What differs is the *data the gradient sees*. Gumbel ensures many different discrete forwards across training, so the biased gradient has richer information. Softmax+STE without added noise only sees one discrete forward and so the biased gradient is weaker at discovering better assignments.

3. **Temperature & annealing.**

   * Gumbel-softmax with a higher temperature yields more soft probabilities (more exploration), with lower temperature closer to hard. With annealing you can explore early, then harden. Softmax+STE lacks that natural per-sample randomness unless you inject noise or explicitly schedule temperature/noise.

---

## Practical consequences (what you’ll observe)

* With Gumbel+STE: counts of top-1 selections across a batch are more diverse early in training; experts get trained on more varied tokens; router learns to separate responsibilities.
* With plain softmax+STE: you’ll often see one or a few experts dominate (expert collapse); those experts get most updates; unused experts stay dead.

---

## Strong practical fixes (used in papers/implementations)

If you like the computational efficiency of hard routing but want to avoid collapse, combine several of these:

1. **Inject noise in logits** (GShard / Switch noisy gating): add Gaussian or Gumbel noise to logits before softmax. This gives stochasticity without changing infrastructure much.
2. **Gumbel-softmax (with/without STE)**: do soft Gumbel during backward; optionally use hard forward + soft backward (straight-through).
3. **Auxiliary load / importance regularizers**: add a small loss that encourages balanced usage of experts (e.g. penalize variance of per-expert load or encourage expected usage to match). This is extremely effective at preventing collapse.
4. **Entropy or KL regularization on router probs**: encourage the router distribution to keep some entropy (avoid becoming a delta too early).
5. **Expert dropout or random routing jitter**: randomly disable an expert or reroute a small fraction of tokens to other experts during training to force exploration.
6. **Capacity factors and per-expert buffers**: allow slightly larger capacity (so overloaded experts don’t drown useful gradient signal) and vary capacity occasionally to let other experts get tokens.
7. **Temperature scheduling**: start with a higher temperature (so gates are softer/stochastic) then anneal to lower temperature for more deterministic routing later.
8. **Diagnostics & monitoring**: track per-expert top-1 counts, mean gate probability (`importance`), coefficient of variation (std/mean of counts). If CV is high, collapse is happening.

---

## A compact recipe that works well in practice

* Compute logits `z = Wg x`.
* Add noise `z' = z + eps` where `eps ~ Gumbel()` (or Gaussian).
* Compute `probs = softmax(z'/tau)`.
* Use **top-k** (often k=1) for dispatch in the forward (hard forward for efficiency) but multiply chosen expert outputs by the corresponding `probs` value so backward sees the soft path (straight-through style).
* Add a small load-balancing penalty (or variance penalty) on expert usage.
* Optionally anneal `tau` down over training.

This combines the best of both worlds: cheap hard forward, differentiable/ informative backwards, and exploration.

---

## Example diagnostics you should log

* `counts_j = number of tokens assigned top-1 to expert j` (per-batch histogram)
* `importance_j = mean over batch of probs[j]` (soft importance)
* `CV = std(counts)/mean(counts)` (want CV small-ish)
* `avg_entropy = mean(-sum(probs * log probs))` per token (shouldn’t collapse to zero too quickly)

If `counts` is extremely skewed and `avg_entropy` is tiny early in training, add noise / increase λ for load loss / increase temperature.

---

## Bottom line

* **Gumbel helps** because it *forces exploration* at the forward-sample level, so even a biased STE gradient sees alternative expert outcomes and can unstick the router.
* **STE on softmax only** can work, but it is brittle: without extra stochasticity or explicit balancing, it more easily mode-collapses to a single route.
* Use noise (Gumbel or Gaussian), load-balancing losses, temperature scheduling, and small amounts of random routing/dropout to get a stable, diverse expert assignment.

---

If you want, I can:

* give a short PyTorch snippet that implements noisy top-1 + STE + a simple load-variance penalty, or
* show quick plotting code to track the diagnostics above so you can see collapse happen and fix it.

Which would you prefer?


---

### 🧑‍💻 User

Hmm, that's interesting - but what's the point of doing random sampling with Gumbel when you typically have to balance token distribution across experts anyway. That's already a form of exploration.

---

### 🤖 Assistant

Great question — you’ve nailed the tension: load-balancing is *global* encouragement to spread tokens, whereas Gumbel sampling is *per-token* stochastic exploration. They’re not redundant; they solve different problems and work best together. Below I’ll explain why, with examples, failure modes if you rely on balancing alone, and practical recipes.

# Why Gumbel sampling still helps even with load-balancing

1. **Different signal types: global vs per-token.**

   * **Load loss** (importance / variance penalty) is a *global* regularizer: it nudges the router’s distribution across the dataset/batch toward uniformity. It doesn’t tell you *which specific token* should go to a different expert to reduce task loss.
   * **Gumbel noise** injects *per-example* randomness so the optimizer can observe what happens when *that particular token* is routed elsewhere. That direct per-token outcome is crucial for correct credit assignment.

2. **Discovery vs enforcement.**

   * Load-balancing enforces usage quotas; it doesn’t *discover* good specializations. Gumbel lets tokens occasionally “try out” other experts and reveal that e.g. token *t* actually gets lower loss under expert B. That discovery is how experts specialize properly.

3. **Richer gradient / exploration dynamics.**

   * When you sample, gradients reflect the actual losses experienced under alternative discrete choices (through the relaxation). Balance alone might only push bulk statistics and can suppress learning signals that would recommend a particular reroute.

4. **Avoiding pathological equilibria.**

   * With only a balancing penalty, the optimizer can satisfy the auxiliary objective by assigning tokens quasi-randomly or by routing based on spurious, high-level heuristics that don't minimize task loss. Gumbel-driven sampling reveals which reassignments help task loss and which are merely balancing noise.

5. **Helps with rare / fine-grained specializations.**

   * Rare token types or niche inputs need occasional trials on alternative experts to find a fit. Global balancing might give them quota, but without per-token trials you won’t learn the right mapping for those rare cases.

6. **Smoother optimization / escaping local minima.**

   * Per-sample noise perturbs the effective optimization landscape and helps escape local minima where router logits are stuck in a bad routing decision that satisfies load but hurts task loss.

# Concrete failure modes if you rely on balancing only

* **Mode collapse to semantically wrong mapping:** you might get a uniform distribution but experts learn to specialize on *dataset artifacts* (e.g., token position or frequent prefixes) rather than true semantic clusters, because there was never targeted sampling to test alternatives for individual tokens.
* **Oscillation / instability:** balancing pushes traffic around but if a reroute increases task loss, the router may oscillate between satisfying load and reducing loss, never converging to a stable specialization.
* **Starvation of informative gradients:** unused experts remain untrained for the *right* tokens; balancing might send them tokens but not the *ones that reveal better performance*, so those experts don't become useful.

# How to think about combining them (practical recipe)

* **Use both.** Noise (Gumbel or Gaussian) for per-token exploration + a modest load/importance loss to avoid extreme collapse. They complement each other.
* **Anneal exploration.** Start with stronger Gumbel noise / higher τ so the router explores, then reduce noise/τ as experts specialize.
* **Keep load-loss small but meaningful.** If it’s too large it can overwhelm task loss and force suboptimal routing; typical practice is a small multiplier (tune it).
* **Monitor per-expert task loss and counts.** If counts are balanced but per-expert losses differ widely, you may need more per-token sampling (increase noise) so tokens can find better experts.
* **Optionally use straight-through:** forward can be hard (fast), backward uses the soft relax — combined with sampling, this gives low-cost training with effective exploration.

# Quick intuitive example

Suppose expert A is overloaded and gets most tokens; load-loss pushes some tokens away but doesn’t reveal *which* tokens would actually be better on expert B. With Gumbel you sometimes send a token to B, see its loss drop, and the router increases p_B for that token’s feature region. Over time B specializes on the tokens where it’s actually better — balancing alone wouldn’t discover that mapping reliably.

---

**Bottom line:** load-balancing prevents pathological collapse at the population level; Gumbel (per-token sampling) *discovers* better token→expert assignments and provides the local learning signal the router needs. Use both, tune the balance, and anneal exploration as training progresses.

Want a tiny PyTorch sketch that shows noisy-top-k + small load-loss together, or some diagnostic plots to run while training? I can drop code you can drop into your MoE loop.
