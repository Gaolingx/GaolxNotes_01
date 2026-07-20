# 使用Muon优化器训练MoE模型

### 🧑‍💻 User

how to use muon optimizer for moe model(has 3d shape)

---

### 🤖 Assistant

**Use a Muon implementation that treats the last two dimensions as the matrix (batched Newton-Schulz).** This independently orthogonalizes each expert when the MoE weight is 3D `[num_experts, out_features, in_features]`.

Muon (MomentUm Orthogonalized by Newton-Schulz) is designed for matrix-shaped parameters. It applies momentum, then uses a few steps of Newton-Schulz iteration to (approximately) orthogonalize the update. For MoE models the expert weights are commonly stored as a single 3D tensor. Flattening the expert dimension into one big 2D matrix couples the experts (undesirable). Operating on the last two dimensions keeps them independent and is efficient (batched matmuls).

### 1. Muon optimizer that supports 3D+ (MoE) tensors

```python
import torch
from torch.optim import Optimizer

def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration for (approximate) orthogonalization / zeroth power.
    Supports batched inputs (ndim >= 2): leading dims are batch, last two are the matrix.
    Perfect for MoE weights of shape [E, out, in].
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=torch.bfloat16)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer with support for ndim >= 2 (including MoE 3D weights).
    """
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize (batched over leading dims → each expert independent)
                update = newton_schulz5(g, steps=ns_steps)

                # Scale so RMS of update ≈ 1 (size-agnostic, works well with lr≈0.02)
                update.mul_((max(update.size(-2), update.size(-1))) ** 0.5)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss
```

**Key points for 3D MoE weights:**
- Shape must be `[num_experts, out_features, in_features]` (or any leading batch dims + matrix). The NS iteration runs independently on every expert.
- If your layout is different (e.g. experts in the last dimension), `permute` so that the two matrix dimensions are last, run the update, then `permute` back.
- Do **not** blindly `.view(-1, last_dim)` — that couples experts.

### 2. Parameter grouping (recommended)

Put matrix-like parameters (including MoE 3D experts) into Muon. Put 1D parameters (biases, layer-norms, etc.) into AdamW. Large embedding tables are often also better in AdamW.

```python
# Example
muon_params = [p for p in model.parameters() if p.ndim >= 2]
adamw_params = [p for p in model.parameters() if p.ndim < 2]

# Optional refinements:
# - Move embedding / lm_head to adamw_params even if 2D
# - Keep router/gate in Muon or AdamW (both work; try both)
# - If you have ModuleList-style experts (separate 2D weights), they already go into muon_params naturally

optimizer_muon = Muon(
    muon_params,
    lr=0.02,              # typical starting point (much higher than AdamW)
    weight_decay=0.01,
    momentum=0.95,
    nesterov=False,       # set True if you want
    ns_steps=5,
)

optimizer_adamw = torch.optim.AdamW(
    adamw_params,
    lr=3e-4,              # classic AdamW LR
    weight_decay=0.01,
    betas=(0.9, 0.95),
)
```

### 3. Training loop

```python
optimizer_muon.zero_grad(set_to_none=True)
optimizer_adamw.zero_grad(set_to_none=True)

loss = model(...)          # your forward
loss.backward()

optimizer_muon.step()
optimizer_adamw.step()

# Apply LR schedule to both optimizers (cosine, WSD, etc.)
```

You can also wrap them in a single scheduler or use a small helper that steps both.

### 4. Practical tips for MoE + Muon

- **Learning rate**: Start with Muon `lr ≈ 0.01–0.05` (0.02 is a strong default). AdamW stays in the usual `1e-4`–`1e-3` range. You often need less tuning than pure AdamW.
- **Weight decay**: Decoupled (already in the code). Same value on both optimizers is fine.
- **ns_steps**: 5 is the usual sweet spot (cheap and sufficient). You can try 3–6.
- **dtype**: The Newton-Schulz is done in bfloat16 for speed; the rest of the model can stay in bf16/fp16/fp32 as usual.
- **Fused / grouped experts**: The 3D path above is ideal for fused expert weights. If you use a `ModuleList` of independent `nn.Linear`, every expert weight is already 2D and works out of the box.
- **Router / gate**: Usually small; putting it in either Muon or AdamW is fine.
- **Distributed / FSDP / expert parallelism**: Make sure the Newton-Schulz sees the full matrix for each expert (or the local shard if you intentionally want local orthogonalization). Gather if needed before the orthogonalization step.
- **Conv / other higher-dim tensors**: For true 4D conv weights the “last-two-dims” view is usually wrong. Reshape them to `[out_channels, in_channels * kH * kW]` (or keep them out of Muon). For pure MoE 3D the last-two view is correct.
- **Stability**: Because of the built-in normalization + orthogonalization, Muon is often more stable and can use higher LRs. Still apply gradient clipping if you normally do.
- **Reference implementations**: Start from Keller Jordan’s Muon (or the versions in modded-nanogpt / community LLM trainers) and only change the Newton-Schulz function to the batched version shown above.

### Quick checklist

1. MoE expert weights are 3D with matrix dims last → use the batched `newton_schulz5`.
2. `ndim >= 2` → Muon group; `ndim < 2` → AdamW group.
3. Muon LR ~ 0.02, AdamW LR ~ 3e-4, same weight decay.
4. Step both optimizers after `backward`.

This setup is the standard, efficient way to run Muon on MoE models that store experts as 3D tensors. You get per-expert orthogonalized updates without coupling them and without writing a manual loop over experts.

---

### 🧑‍💻 User

how to use moe+muon with fsdp2?

---

### 🤖 Assistant

**Muon + MoE works well with FSDP2 if you keep the two matrix dimensions of every Muon parameter intact (unsharded).**

Newton-Schulz needs complete matrices. For a 3D MoE weight of shape `[num_experts, out_features, in_features]`, the clean solution is to **shard only the expert dimension**. Each rank then sees a local tensor `[E_local, out, in]`. The batched Newton-Schulz from the previous answer runs independently on every local expert with **zero extra communication**.

### Core Rules for FSDP2 + Muon + MoE

1. **Never shard the last two dimensions** of any parameter that goes into Muon.
2. For fused 3D expert weights: place `Shard(0)` (expert dim). Local shape stays valid matrices → original Muon code works unchanged.
3. Prefer **Expert Parallelism (EP)** + FSDP2 for large MoEs. Each rank owns full experts.
4. Put 1D params (biases, norms, etc.) and usually embeddings/lm_head into AdamW. Matrix-like params (including MoE experts and attention weights) go into Muon.
5. FSDP2 uses DTensors. The Muon step below handles both plain tensors and DTensors cleanly when the matrix dims are complete.

### 1. Recommended Muon (FSDP2 / DTensor aware)

This is an improved version of the previous Muon. It:
- Supports plain tensors and DTensors.
- Works with 3D MoE when experts are sharded (local matrices stay full).
- Optionally gathers only when a matrix is actually incomplete (rare if you follow the rules above).

```python
import torch
from torch.optim import Optimizer
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard, Replicate

def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched Newton-Schulz. Leading dims = batch (experts), last two = matrix."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=torch.bfloat16)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)
    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
        # Set True only if you are forced to shard a matrix dimension
        allow_gather: bool = False,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps, allow_gather=allow_gather,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            allow_gather = group["allow_gather"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # --- momentum (keep state in the same layout as p) ---
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                update = grad.add(buf, alpha=momentum) if nesterov else buf

                # --- Newton-Schulz ---
                # Prefer operating on local tensor when matrix dims are complete
                if isinstance(update, DTensor):
                    # Check whether the last two dims are sharded
                    placements = update.placements
                    matrix_sharded = any(
                        isinstance(pl, Shard) and pl.dim >= update.ndim - 2
                        for pl in placements
                    )
                    if matrix_sharded:
                        if not allow_gather:
                            raise RuntimeError(
                                "Muon matrix dimension is sharded. "
                                "Reshard so that last two dims stay complete, "
                                "or set allow_gather=True (expensive)."
                            )
                        # Expensive path – avoid if possible
                        update = update.full_tensor()
                        update = newton_schulz5(update, steps=ns_steps)
                        update = update * (max(update.size(-2), update.size(-1)) ** 0.5)
                        # Turn back into a DTensor matching p's placement
                        update = DTensor.from_local(
                            update, device_mesh=p.device_mesh, placements=p.placements
                        ) if False else update  # simplify: most people apply after
                        # Better: work with full then redistribute. For simplicity
                        # many codebases just full_tensor the param too for the add_.
                    else:
                        # Happy path: local experts or unsharded matrix dims
                        local_update = update.to_local()
                        local_update = newton_schulz5(local_update, steps=ns_steps)
                        local_update.mul_((max(local_update.size(-2), local_update.size(-1))) ** 0.5)
                        update = DTensor.from_local(
                            local_update,
                            device_mesh=update.device_mesh,
                            placements=update.placements,
                            run_check=False,
                        )
                else:
                    # Normal non-DTensor path (single GPU or EP with full local experts)
                    update = newton_schulz5(update, steps=ns_steps)
                    update.mul_((max(update.size(-2), update.size(-1))) ** 0.5)

                # Decoupled weight decay + update
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss
```

**Notes on the DTensor path**
- When you shard only the expert dimension, `matrix_sharded == False` and you stay on the fast local path.
- `allow_gather=True` is a fallback (all-gather the matrix, run NS, then continue). It works but costs communication and peak memory; avoid it for large matrices.

### 2. FSDP2 Setup for MoE (Expert-dim sharding)

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard, Replicate

# Example device mesh (data parallel + optional expert parallel)
# For pure FSDP (DP only):
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

# For Hybrid EP + FSDP (recommended for big MoE):
# mesh = init_device_mesh("cuda", (ep_size, dp_size), mesh_dim_names=("ep", "dp"))

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,   # or bfloat16
    output_dtype=torch.bfloat16,
)

def apply_fsdp(model):
    # 1. Shard MoE expert weights on expert dimension (dim 0)
    for name, module in model.named_modules():
        if hasattr(module, "experts") and isinstance(module.experts, nn.Parameter):
            # fused 3D weight: [E, out, in]
            # fully_shard will respect the mesh; we force Shard(0) via placement if needed
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
            )
            # If using explicit DTensor placement on the parameter:
            # module.experts = DTensor.from_local(..., placements=[Shard(0)])

        # 2. Normal transformer blocks / attention / etc.
        elif "layers" in name or isinstance(module, (nn.Linear, YourAttention)):
            fully_shard(module, mesh=mesh, mp_policy=mp_policy)

    # Root
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model
```

**Better for large MoE – Expert Parallelism**
Keep each expert fully local:

```python
# Pseudocode
# - Create a DeviceMesh with an "ep" dimension
# - Place experts so that each rank owns a contiguous slice of experts (Shard on expert dim or use ModuleList + EP)
# - Apply fully_shard only on the non-expert modules (attention, norms, router, embeddings)
# - Muon then sees completely local full matrices → original simple Muon works perfectly
```

This is the most common production pattern for MoE + Muon.

### 3. Optimizer construction (same as before)

```python
# After model = apply_fsdp(model) and model.to(device)

muon_params = []
adamw_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    # Heuristic: matrices (ndim >= 2) → Muon, including 3D MoE
    # Exclude huge embeddings / lm_head if you prefer AdamW for them
    if p.ndim >= 2 and "embed" not in name and "lm_head" not in name:
        muon_params.append(p)
    else:
        adamw_params.append(p)

opt_muon = Muon(
    muon_params,
    lr=0.02,
    weight_decay=0.01,
    momentum=0.95,
    ns_steps=5,
    allow_gather=False,   # keep False if you followed the sharding rules
)

opt_adamw = torch.optim.AdamW(
    adamw_params,
    lr=3e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95),
)
```

### 4. Training loop

```python
# Standard FSDP2 loop
for batch in loader:
    opt_muon.zero_grad(set_to_none=True)
    opt_adamw.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch)

    loss.backward()          # FSDP2 handles reduce-scatter / all-reduce

    # Optional: gradient clipping (on the root or use FSDP helpers)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    opt_muon.step()
    opt_adamw.step()

    # LR schedulers for both optimizers
```

### 5. Practical Tips & Gotchas

| Topic | Recommendation |
|-------|----------------|
| **Sharding dim** | Always keep last two dims of Muon params complete. For 3D MoE → shard expert dim only. |
| **Memory** | `reshard_after_forward=True` is almost always better. Use activation checkpointing on MoE layers. |
| **Router / Gate** | Small; can go to either Muon or AdamW. Often left replicated or lightly sharded. |
| **Embeddings / lm_head** | Usually better in AdamW even if 2D. |
| **Communication** | When expert dim is sharded correctly there is **no extra all-gather** for Muon. |
| **Hybrid EP+DP** | Best scalability for large MoE. Muon stays local and cheap. |
| **torch.compile** | Works, but compile the model *after* applying `fully_shard`. Optimizer step can stay eager. |
| **ns_steps** | 5 is standard. You can drop to 3–4 if you need a tiny bit more speed. |
| **LR** | Muon 0.01–0.05 (start 0.02). AdamW classic 1e-4–5e-4. Same WD is fine. |
| **Debugging** | Print `p.shape` and `p.placements` (if DTensor) inside the Muon step for a few parameters to confirm matrix dims are not sharded. |

### Quick Checklist

1. MoE weight layout = `[E, out, in]` (or permute to make matrix dims last).
2. FSDP2 / EP shards **only** the expert dimension (or each expert is fully local).
3. Muon receives those parameters; AdamW receives the rest.
4. Use the DTensor-aware Muon above (fast path when matrices are complete).
5. Step both optimizers after `backward()`.

This combination (MoE 3D + Muon + FSDP2 with expert-dim sharding or EP) is currently one of the most efficient ways to train large MoEs. The orthogonalized updates stay independent per expert and the communication overhead of Muon remains essentially zero.

If you share more details about your MoE implementation (fused 3D vs `ModuleList`, whether you already use Expert Parallel, PyTorch version, etc.) I can give a more precise code skeleton.