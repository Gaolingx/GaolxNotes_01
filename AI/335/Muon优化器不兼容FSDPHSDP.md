# Muon优化器不兼容FSDP/HSDP

### 🧑‍💻 User

Is torch Muon optimizer compatible with FSDP/HSDP?

---

### 🤖 Assistant

Out of the box, the **Muon optimizer is conceptually and technically incompatible with standard FSDP (Fully Sharded Data Parallel) and HSDP (Hybrid Sharded Data Parallel)** in PyTorch. 

While it can be made to work with significant custom engineering, you cannot simply drop `Muon` into an FSDP training script like you would `AdamW`.

Here is a breakdown of why this incompatibility exists and what it takes to work around it.

### The Fundamental Conflict

The incompatibility stems from how Muon calculates its updates versus how FSDP manages memory.

1. **How Muon Works:** 
   Muon (Momentum Orthogonalizer) operates on 2D parameters (like Linear layer weights). To compute the update, it uses a Newton-Schulz iteration to orthogonalize the gradient matrix $G$. This requires performing matrix multiplications (e.g., $G \times G^T$) on the **full, 2D gradient matrix**.
2. **How FSDP/HSDP Works:** 
   To save memory, FSDP flattens the parameters and gradients of your model into 1D arrays and shards (splits) them across GPUs. During the `optimizer.step()`, each GPU only possesses a seemingly random, flattened 1D chunk of the original weights and gradients.

**The Clash:** You cannot perform a 2D matrix orthogonalization (Newton-Schulz iteration) on an arbitrary 1D, flattened chunk of a matrix. The math simply breaks. 

<details>
<summary><b>Mathematical Details of the Conflict</b></summary>

For a weight matrix $W$ and its gradient $G$, Muon applies a Newton-Schulz iteration to approximate the orthogonalization of $G$. A standard step looks like this:

$$X_0 = \frac{G}{\|G\|_F}$$
$$X_{k+1} = X_k + \frac{1}{2} X_k (I - X_k^T X_k)$$

In standard PyTorch, $G$ is a full tensor. In FSDP, what is passed to the optimizer is $G_{sharded}$, which is a flat 1D vector of length $\frac{N}{world\_size}$. The operations $X_k^T X_k$ are impossible to compute locally because rank $i$ does not have the rest of the rows/columns required for the dot products.
</details>

### Can it be made to work?

Yes, but it requires writing a custom implementation rather than using standard PyTorch APIs. To make Muon compatible with FSDP, you have to intercept the training loop at specific points:

* **Unsharding for the Optimizer Step:** You would need to use FSDP's communication hooks to `all_gather` the gradients back into full 2D matrices right before the optimizer step.
* **Applying Muon:** Once the full 2D gradient matrix is reconstructed on the GPU (or across GPUs using Tensor Parallelism), you perform the Newton-Schulz iteration.
* **Resharding:** After computing the update, you must flatten and reshard the updated weights/momentum states back across the GPUs.

**The Problem with the Workaround:** Doing this completely defeats the primary purpose of FSDP. FSDP is designed to keep optimizer states and gradients sharded to save memory. Reconstructing full gradients and states to compute the Muon step will cause massive memory spikes, likely leading to Out-Of-Memory (OOM) errors on large models.

### Recommended Alternatives for Distributed Muon

If you want to scale up a model using the Muon optimizer, standard FSDP is not the right tool. Instead, consider:

1. **DDP (Distributed Data Parallel):** Since DDP replicates the model across all GPUs, the optimizer has access to the full 2D weight matrices. Muon works perfectly with DDP.
2. **Tensor Parallelism (TP):** TP splits matrices along rows or columns. Because the underlying structure is still 2D and matrix math can be mathematically distributed across GPUs (using Megatron-LM style column/row parallel linear layers), Muon can be adapted to work with TP.
3. **ZeRO-1 (Optimizer State Sharding only):** If you use DeepSpeed ZeRO-1 or PyTorch DDP with a sharded optimizer, you *might* be able to adapt Muon, provided the gradients themselves are not sharded when the Newton-Schulz iteration is called. However, ZeRO-2 and ZeRO-3 (which shard gradients and parameters, similar to FSDP) will suffer from the exact same issues as FSDP.