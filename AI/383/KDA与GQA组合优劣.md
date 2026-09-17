# KDA与GQA组合优劣

### 🧑‍💻 User

kda+gqa+nope is better than gqa only?

---

### 🤖 Assistant

I'll research this. Let me search for information about these attention mechanisms.

Let me dig deeper into these sources.

Bing returns noise. Let me fetch specific sources directly.

Let me get the raw README and more context on the specific hybrid comparison.

This is very relevant. Let me get the Arcee blog which directly compares a KDA+NoPE hybrid against the original GQA model.

Let me get the benchmark results and conclusion from the Arcee blog.

Let me check one more relevant source on NoPE hybrids.

## Short answer

**Not as a blanket statement.** A "KDA + full-attention + NoPE" hybrid is clearly **better on efficiency**, and it is **≈ or slightly better on quality when trained from scratch at scale** — but it is **worse than a plain GQA model when it's a post-hoc distillation of a fixed GQA model**, and the **NoPE part can actively hurt long-context behavior**. So "better" depends entirely on *which axis* and *how it was produced*.

Also, one nit: the headline architecture you're probably thinking of ("KDA + NoPE") pairs KDA with **MLA**, not GQA — Kimi Linear's full-attention layers are Multi-head Latent Attention. More below.

<details>
<summary><strong>What the acronyms mean here</strong></summary>

- **KDA (Kimi Delta Attention)** — a linear-attention / finite-state RNN module from Moonshot's *Kimi Linear* paper (arXiv:2510.26692). It refines Gated DeltaNet with **channel-wise, fine-grained decay**. State update:
  $$S_t = \big(I - \beta_t k_t k_t^\top\big)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$$
  where $\mathrm{Diag}(\alpha_t)$ gives each feature dimension its own forgetting rate.
- **GQA (Grouped Query Attention)** — the standard *full/global softmax attention* with grouped KV heads; this is what a "GQA-only" model uses in every layer.
- **NoPE (No Positional Encoding)** — RoPE is *removed* and not replaced in the full-attention layers; the model offloads all positional/recency information to the KDA layers.

</details>

## The three regimes that matter

### 1. Efficiency — the hybrid wins decisively ✅
Kimi Linear's 3:1 interleave (3 KDA layers : 1 full-attention layer) reduced **KV-cache memory by up to ~75%**, and at 1M context reached **~6× higher decode throughput** and **~2.9× faster prefill** than full MLA attention. Arcee's distillation hit ~8,000 tokens/s/H100 in training. On this axis there is no contest.

### 2. Quality from scratch at scale — hybrid ≈/ ≥ full attention ✅
This is Kimi Linear's core claim: the KDA+full-attention hybrid **matches or outperforms full attention under matched training**, across short-context, long-context and RL scaling. Reported examples: RULER 84.3 vs 81.3 for MLA; better MMLU/GSM8K/coding. Ablations found the **3:1 ratio** was the sweet spot (better than 1:1 or 7:1). So *when both are trained under the same recipe*, adding KDA layers does not cost you quality — and buys large efficiency.

### 3. Quality via post-hoc distillation — the GQA teacher still wins ❌
Arcee's experiment is the most direct test of *your* exact question. They distilled `AFM-4.5B-Base` (a GQA model) into three students:
- **A) KDA-NoPE hybrid** (interleaved 3 KDA + 1 NoPE full-attn)
- **B) KDA-FLP** (full attention front-loaded)
- **C) KDA-Only** (no full attention)

Results:
- The **teacher (original GQA model) "reigns supreme"** — no student beat it, because the teacher had ~8T tokens vs ~1B for the students.
- The **hybrid recovered dramatically more** than pure-KDA, especially on math (GSM8K) where all-KDA dropped more. Knowledge benchmarks were "within statistical spitting distance."
- **KDA-FLP was unexpectedly terrible** — just reordering which layers keep full attention mattered a lot.

So in a distillation setting, the answer is **no**: KDA+GQA+NoPE does not beat the original GQA model — it *approximates* it while being much faster.

## The NoPE caveat (this is the subtle part)

On RULER multi-key NIAH, Arcee found:
- Everyone got 100% on single-needle retrieval up to 65K.
- **Pure-KDA degraded smoothly**; the **KDA+NoPE hybrid crashed off a cliff past 32K to ~zero**.

Their inference: the **NoPE full-attention layers are the culprit** — KDA generalizes to longer sequences natively, but NoPE global-attention layers need to have *seen* sequences at that length or they fall apart. So dropping RoPE is **not** a free win for extrapolation; it trades RoPE-length-tuning hassle for a hard length-generalization dependency. (For balance, the *"Rope to Nope and Back Again"* paper, arXiv:2501.18795, finds hybrid RoPE/NoPE layouts can beat full RoPE attention on both long and short tasks — so it's a design tradeoff, not a universal law.)

## Two accuracy caveats in the phrasing itself

- **"KDA + GQA"** is slightly off: Kimi Linear pairs KDA with **MLA**, not GQA. And KDA has **no GQA-equivalent** — Arcee noted there's no grouped-KV support for KDA, so they had to **repeat heads out to MHA shape** (ballooning 4.5B → ~5B params). So a literal "KDA + GQA" hybrid isn't the canonical config; it's KDA (MHA) + GQA-full-attn.
- The QINGYI-KDA project claims the first **GQA+RoPE-preserving** hybrid conversion, deliberately *not* using NoPE — further evidence that NoPE is a choice with real tradeoffs, not a strict upgrade.

## Bottom line

| Comparison | Verdict |
|---|---|
| KDA+NoPE hybrid **vs GQA-only** on **memory/speed** | ✅ Hybrid much better |
| ...when **trained from scratch, matched recipe** | ✅ ≈ or slightly better quality |
| ...when **distilled from the GQA model** | ❌ GQA teacher still best; hybrid ≈ approximate |
| ...on **long-context (>~32K) multi-key recall** | ❌ NoPE layers can crash; pure-KDA degrades more gracefully |

So: **"KDA + full-attention + NoPE" is a better *efficiency* architecture and a competitive (sometimes better) *quality* one at scale — but it is not automatically better than GQA-only, and it's actually worse in the two situations you'd most likely be asking about: distilling an existing GQA model, and long-context multi-key retrieval.**

If you tell me the specific source/model you saw making the claim (e.g., a particular benchmark table), I can pin down exactly which regime it's asserting and whether it holds.