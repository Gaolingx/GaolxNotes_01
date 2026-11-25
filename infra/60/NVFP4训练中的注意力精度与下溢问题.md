# NVFP4训练中的注意力精度与下溢问题

## 🧑‍💻 User

https://thinkingmachines.ai/blog/end-to-end-4-bit-training/
Native NVFP4 forward pass across the full pretraining trajectory is huge for memory bandwidth. Did you find it necessary to keep the attention softmax accumulation in FP32/FP16, or does the scale factor clipping manage underflow during long-context RL rollouts?

---
## 😀 默认助手 (7)

## Short answer

I'm not one of the authors — Jack Cook and Thinking Machines Lab wrote that post, and I'm reading it the same way you are — so treat this as "what the write-up actually claims," not inside information. And the write-up doesn't support either branch of your question:

- **Attention/softmax precision is never discussed**, because attention was never inside the 4-bit path.
- **There is no "scale factor clipping" mechanism** doing underflow management in their recipe.

## What the recipe actually quantizes

The native NVFP4 forward pass is applied to **MoE expert GEMMs only** (`w13`, `w2`). Everything else — attention, norms, softmax, residuals — stays in BF16/MXFP8 by construction. Their "end-to-end" claim is about *scheduling*, not *coverage*:

| Stage | Precision |
|---|---|
| Pretrain/SFT/RL forward, expert layers | NVFP4 (activations + weights) |
| Backward (DGRAD / WGRAD) | MXFP8, with SR re-quantization |
| Last 16.6% of layers | MXFP8 (not BF16) |
| MoE combine output | BF16 by default (or RHT + block-scaled S2F6, 8.25 bits) |

So "did they *need* FP32/FP16 softmax accumulation" is a question the paper doesn't answer, because they never put softmax in a 4-bit accumulator to begin with. The forward pass being identical from pretraining through final rollout means *the config* didn't change; it doesn't mean attention got quantized.

## Where underflow risk does show up, and what they use

<details>
<summary><b>Expand: the four actual underflow mitigations in the post</b></summary>

1. **Block granularity, not clipping.** NVFP4 uses 1×16 blocks. They explicitly *rejected* NVIDIA's 16×16-tile single-scale approach, saying it was "too coarse, hurting CE beyond what we were willing to accept" — precisely because one outlier among 256 values flushes the rest to zero. The fix is narrower blocks, not clipping the scale.
2. **Two-level outer scaling.** Per-value reconstruction is $x_i^{\text{FP4}} \times x_{i/16}^{\text{FP8}} \times x^{\text{FP32}}$, with the FP32 outer factor shifting the representable window ($[2^{-10}, 2688]$) up or down per tensor. That's range *shift*, not clipping.
3. **Four Over Six (activations only, not weights).** Two candidate block scales, picks the lower-error one — this is the closest thing to your "clipping" intuition, but it's adaptive scale selection, not a clamp.
4. **Expectation-preserving re-quantization.** Their transpose problem is solved by dequant → transpose in high precision → re-quantize to MXFP8 with **stochastic rounding**, so $\mathbb{E}[w_q] = \mathbb{E}[(w_q^T)^T]$ rather than requiring bitwise identity. SR is what keeps small values from being deterministically zeroed on the backward side.

</details>

## On long-context RL rollouts specifically

Relevant to your worry, but not via softmax accumulation:

- Their **per-token outer scales for activations** are chosen partly because they give batch-invariance and "no worry about causality leakage" — i.e. a token's quantization doesn't depend on other tokens in the sequence. That's the property you actually want for 10k+ token rollouts, and it's about *scale assignment*, not accumulator width.
- The RL speedup (20%) is attributed to **weight-loading bandwidth** in sampling and to NVLink/combine tail reduction, not to softmax arithmetic.
- They report "no discernible difference in reward" in RL, and flag FP4 *backward* as still-open future work — a tacit admission that the numerically fragile parts were kept out of FP4 for now.

**Bottom line:** if you're asking whether their scale-factor scheme is a substitute for FP32/FP16 softmax accumulation over long contexts — the post gives no evidence either way, because that path was never quantized. Don't cite it for that claim. If you want the real failure modes they hit, they're the 16×16 tile coarseness and the row↔column transpose mismatch, and both were fixed by block shape and stochastic rounding.
