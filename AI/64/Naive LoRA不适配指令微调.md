# Naive LoRA不适配指令微调

### 🧑‍💻 User

why naive LoRA isn't a great fit for instruct-tuning from a base model. (It's not that it can't be done, but you need to think more carefully about how you do and don't do various things, and it requires a fairly advanced understanding of the characteristics of various training methods to get a good result, or a significant amount of trial and error.)

---

### 🤖 Assistant

You have hit on a highly nuanced and often misunderstood aspect of Large Language Model (LLM) training. Many practitioners assume that because LoRA (Low-Rank Adaptation) works beautifully for teaching an *already instruct-tuned* model a specific task (like medical Q&A or a specific JSON format), it will work just as well for turning a raw, base completion model into an instruct model. 

As you noted, it *can* be done, but a "naive" out-of-the-box LoRA configuration usually fails or produces a severely degraded model. Here is the theoretical and practical breakdown of why naive LoRA is a poor fit for base-to-instruct tuning.

### 1. The Low-Rank Assumption vs. Fundamental Paradigm Shifts
LoRA relies on the hypothesis that the change in weights required to adapt a model has a low "intrinsic dimension." Mathematically, instead of updating the full weight matrix $W_0$, LoRA freezes $W_0$ and trains a low-rank decomposition:
$$ W_{new} = W_0 + \Delta W = W_0 + BA $$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with the rank $r \ll \min(d, k)$.

*   **When Naive LoRA works:** If you are teaching an instruct model a new domain, the model already knows *how* to converse. You are just slightly steering its existing conversational manifold. This is a low-rank update.
*   **Why Base-to-Instruct fails:** A base model is a pure document-continuation engine. Instruction tuning requires a massive behavioral paradigm shift: it must learn to recognize instructions, separate user input from its own output, learn refusal mechanisms, and adopt a helpful persona. Forcing this massive, unstructured behavioral shift through a narrow, low-rank bottleneck (e.g., $r=8$) often results in a model that "forgets" how to form coherent sentences while desperately trying to overfit to the chat format.

### 2. The Embedding Layer and Special Tokens
Instruction tuning almost always requires introducing new special tokens to structure the conversation (e.g., `<|user|>`, `<|assistant|>`, `<|im_start|>`, `<|im_end|>`). 

Naive LoRA scripts typically freeze the embedding layer and the language modeling head to save VRAM. If you freeze the embeddings, the model literally cannot learn what these new structural tokens mean. It will treat them as random noise, leading to bleeding context, endless generation (failing to output the EOS token), or complete structural collapse.

### 3. Module Targeting: Attention vs. MLPs
The original LoRA paper primarily focused on applying the adapters to the Query ($W_q$) and Value ($W_v$) projection matrices in the self-attention mechanisms. Many naive LoRA scripts still default to this.

*   **The Attention layer** acts as a router, moving information between tokens.
*   **The MLP (Feed-Forward) layer** acts as the model's knowledge base and reasoning engine (comprising roughly two-thirds of the model's parameters).

Instruction tuning requires the model to re-wire how it accesses its own internal knowledge to answer direct questions. If you only apply LoRA to the attention layers, the model learns to *look* at the right tokens, but lacks the plasticity in its MLPs to process the reasoning required by the instruction. 

### 4. The "Alignment Tax" and Hyperparameter Brittleness
Because naive LoRA restricts the model's capacity to learn, the model suffers a severe "alignment tax" during base-to-instruct tuning. 
If the learning rate is too high, the model quickly collapses into mode-collapse (giving the exact same generic response to every prompt). If the learning rate is too low, the model ignores the instruction format and reverts to base-model continuation (e.g., answering a question with more questions). Finding the exact Goldilocks zone for the learning rate, rank ($r$), and scaling factor ($\alpha$) requires exhaustive hyperparameter sweeps.

***

<details>
<summary><b>How to do it correctly (Advanced LoRA Techniques)</b></summary>

If you *must* use parameter-efficient fine-tuning (PEFT) to turn a base model into an instruct model, you have to abandon the "naive" approach and implement several advanced techniques:

1.  **Target All Linear Layers:** You must apply LoRA to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. This drastically increases the trainable parameters but gives the model the plasticity it needs.
2.  **Train the Embeddings and LM Head:** You must unfreeze the `embed_tokens` and `lm_head` so the model can learn the new chat template tokens.
    ```python
    # Example using PEFT config
    config = LoraConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"], # Crucial for special tokens
        ...
    )
    ```
3.  **High Rank and Alpha:** Instead of $r=8$, you often need $r=64$ or even $r=128$, with $\alpha$ set to $1\times$ or $2\times$ the rank, to widen the information bottleneck.
4.  **Use Variants like DoRA or RS-LoRA:** 
    *   **DoRA (Weight-Decomposed Low-Rank Adaptation):** Separates the magnitude and direction of the weight updates, which has been shown to mimic full fine-tuning much closer than standard LoRA.
    *   **RS-LoRA (Rank-Stabilized LoRA):** Scales the adapter by $\frac{\alpha}{\sqrt{r}}$ instead of $\frac{\alpha}{r}$, which prevents gradient collapse when using the high ranks required for instruct-tuning.
5.  **Multi-Stage Training:** First, train on a highly curated, massive dataset to teach the format. Then, do a second LoRA pass for specific complex reasoning tasks.

</details>

### Summary
Naive LoRA is fundamentally a **knowledge-steering** tool, whereas base-to-instruct tuning requires a **behavior-rewiring** tool. While full fine-tuning (FFT) effortlessly rearranges the model's internal representations to accommodate this new behavior, naive LoRA fights against its own mathematical bottlenecks, resulting in a fragile, underperforming model unless heavily modified.