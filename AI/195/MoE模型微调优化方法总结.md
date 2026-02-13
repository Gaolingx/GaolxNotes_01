# MoE模型微调优化方法总结

### 🧑‍💻 User

Compared to dense LLMs, fine-tuning the MoE model is significantly more difficult. As far as I know, mixtral QLoRA experiments are showing fairly bad results, with people reporting [odd behaviours on models](https://old.reddit.com/r/LocalLLaMA/comments/18gz54r/llm_comparisontest_mixtral8x7b_mistral_decilm/) or plainly seeing loss curves explode:
This is the kind of thing you’d expect in advance for a variety of reasons – most of which are covered in papers like [ST-MOE](https://arxiv.org/abs/2202.08906) – including:

- lack of auxiliary/balancing/z-loss. people who chuck in MixtralForCausalLM into normal dense model trainers will end up with [no auxiliary loss](https://github.com/huggingface/transformers/blob/680c610f9733561fb09e21c0e6ed513d69564f56/src/transformers/models/mixtral/modeling_mixtral.py#L1243) – `output_router_logits` is [disabled by default](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commit/3de0408ae8b591d9ac516a2384925dd98ebc66f4) – much less additional stabilisers like [router z-loss](https://152334h.github.io/blog/mixtral-vs-oss/):
  ![](imgs/Pasted%20image%2020231213170124.png)
- lack of MoE-training-specific hyperparameters like Capacity Factor, Expert Dropout, routing jitter, etc.
- Batch sizes being too small. The [HF MoE Blog](https://huggingface.co/blog/moe) notes that the ST-MOE paper recommends “small” batch sizes…
  ![](imgs/Pasted%20image%2020231213173422.png)
  …but “small” in this case refers to a token batch size of 65k tokens (or a sequence batch size of **~128**, since this is for T5 which has `ctxlen==512`). Most consumer LoRA runs do not have a batch size that large.
- router being trained in half precision at all.
- having a LoRA adapter over the routing layers at all. Each `layers.[0..32].ffn.gate` layer in mixtral is a `Linear(dim,8)`; training it a LoRA with $r>=8$ creates an adapter *bigger than the gate itself:*
  ![](imgs/Pasted%20image%2020231213171556.png)
  I am not quite sure what the effect of this is yet, but I suspect it is not a good thing to be doing.

All of these issues make fine-tuning the moe model extremely challenging. And most of these issues are engineering issues that can be solved with time, and there is a large space of dumb ideas that may “just work”, e.g. freezing the routers, or forcing expert randomization, or throwing compute at hparam search, or…In general, having a learned bias against unreplicated academic papers that say “This is the way things should be done”. What methods can address these problems and make fine-tuning easier?

---

### 🤖 Assistant

The observation that fine-tuning Mixtral (and MoEs in general) via QLoRA is unstable is widely shared in the community. You have correctly identified the primary culprits: precision issues, lack of auxiliary losses, and the mathematical redundancy of applying LoRA to the routing gates.

Here are concrete, engineering-focused methods to address these problems, ranging from configuration fixes to custom training loops.

### 1. The "Golden Rule": Freeze the Router (Do not QLoRA the Gate)

The single most effective fix for Mixtral QLoRA instability is to **exclude the gating layers from the LoRA adapters.**

As you noted, the gating layer in Mixtral is a projection of `dim` $\to$ `num_experts` (4096 $\to$ 8).
If you apply LoRA with rank $r=8$ (or higher), you are not creating a low-rank approximation; you are training a full-rank adapter alongside the original weights. This destabilizes the delicate pre-trained routing logic without the safeguards (z-loss) used during pre-training.

**Why this fixes the problem:**
1.  **No Balancing Loss Needed:** If the router weights are frozen, the expert assignment for a given input token remains deterministic (based on the pre-trained knowledge). You do not need to add auxiliary load-balancing loss because the router cannot "collapse" to a single expert during training.
2.  **Preserves Pre-trained Knowledge:** Mixtral already knows how to route tokens. Fine-tuning usually aims to change the *content* or *format* of the response (the experts), not the semantic routing logic.

**Configuration:**
When configuring `LoraConfig`, explicitly set `target_modules` to the attention and FFN layers, but **omit** the gate.

```python
# Recommended Target Modules for Mixtral
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "w1",
    "w2",
    "w3",
    "lm_head" # Optional, helps with chat formats
]

# Explicitly do NOT include "gate" or "block_sparse_moe.gate"
```

### 2. High-Precision Routing (Cast Gate to float32)

If you are using 4-bit quantization (QLoRA), the model is cast to `float16` or `bfloat16`, and the weights are `nf4`. However, the Softmax calculation in the router is extremely sensitive to precision. In `float16`, large logit variations can cause overflows or underflows, leading to `NaN` loss or expert collapse.

**The Fix:**
Ensure the gating layer execution (not necessarily storage, but calculation) happens in `float32`. Most modern HF implementations try to handle this, but you can force it via upcasting hooks or ensuring your LoRA config doesn't quantize the gate (which is naturally solved by **Method 1**, as the base model gate remains in its loaded precision).

If you are seeing exploding gradients, verify your compute dtype:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # Try bfloat16 if hardware supports it
    # ...
)
```
*Note: `bfloat16` has the same dynamic range as `float32` and is significantly more stable for MoE routing than `float16`.*

### 3. Solving the "Small Batch Size" Issue

The ST-MOE paper recommends large batch sizes (approx 128 sequences or 65k tokens) to average out expert utilization. In a local QLoRA setup with a batch size of 1 or 2, the noise in the gradient estimation is high.

**The Fix: Massive Gradient Accumulation.**
You cannot fit 128 sequences in VRAM, but you can delay the optimizer step.

If your GPU fits a micro-batch of 2, and you want to simulate the "stable" MoE batch size of 128:
$$ \text{gradient\_accumulation\_steps} = \frac{128}{\text{micro\_batch\_size}} = 64 $$

Set `gradient_accumulation_steps=64` (or as high as you can tolerate waiting). This smoothes out the gradients and prevents a single "weird" token from violently shifting the expert weights.

### 4. Implementing Auxiliary Loss (If you MUST train the router)

If you insist on training the gate layers (not recommended for simple finetunes), you **must** re-introduce the auxiliary loss. The standard Hugging Face `Trainer` ignores the `aux_loss` returned by the Mixtral model forward pass.

You need to subclass the Trainer to add this back in.

<details>
<summary><b>Click to expand: Custom MoE Trainer Code</b></summary>

```python
from transformers import Trainer

class MixtralMoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
      
        # Standard task loss (CrossEntropy)
        loss = outputs.get("loss")
      
        # Add auxiliary load balancing loss if available
        # The coeff 0.02 is standard for Mixtral/ST-MOE experiments
        if "aux_loss" in outputs:
            loss += 0.02 * outputs["aux_loss"]
          
        return (loss, outputs) if return_outputs else loss
```
</details>

**Note:** You must also ensure `output_router_logits=True` is passed to the model config, though the internal `MixtralModel` implementation usually calculates `aux_loss` automatically if the config attributes `router_aux_loss_coef` is set in the model configuration.

### 5. Conservative Hyperparameters

Sparse models have different convergence properties than dense models.

*   **Learning Rate:** MoEs often prefer lower learning rates or longer warmups. If your loss explodes, drop your LR by $5\times$ or $10\times$. A typical QLoRA LR for Llama-2 might be `2e-4`; for Mixtral, start conservatively at `1e-5` or `5e-5`.
*   **Gradient Clipping:** This is essential for MoE training to catch exploding gradients from the router. Set `max_grad_norm` to 0.5 or 1.0.

### 6. The "Dumb Idea": Deterministic / Top-K Hardening

A more "engineering hack" approach is to reduce the stochasticity of the model during training.

The ST-MOE paper suggests "Router Jitter" (adding noise). However, for **fine-tuning** (where we want to retain capabilities), we often want the opposite: stability.

If you are writing a custom training loop, you can modify the temperature of the router softmax. By lowering the temperature (sharpening the distribution), you force the model to commit to experts more strongly. This can prevent the "drift" where the model tries to learn to use all experts equally for a specific new task, which ruins its general capabilities.

### Summary Checklist for Stable Mixtral QLoRA

1.  **Target Modules:** `['k_proj', 'q_proj', 'v_proj', 'o_proj', 'w1', 'w2', 'w3']`. **Exclude `gate`.**
2.  **Precision:** Use `bfloat16` for compute if possible.
3.  **Batch Size:** Use high `gradient_accumulation_steps` (target effective batch size of 64+).
4.  **Learning Rate:** Lower than dense models (start `5e-5`).
5.  **Grad Clip:** Enable `max_grad_norm=1.0`.

By simply removing the gate from the trainable parameters (Strategy 1), you bypass the need for z-loss and load balancing, converting the problem back into a standard (albeit distinct) Sparse-Linear-Layer fine-tuning task.