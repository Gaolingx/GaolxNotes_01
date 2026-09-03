# DTensor梯度裁剪

### 🧑‍💻 User

torch.nn.utils.clip_grad_norm_ support dtensor to get param_l2_norm in FSDP2?
https://docs.pytorch.org/docs/2.13/generated/torch.nn.utils.clip_grad_norm_.html

---

### 🤖 Assistant

Let me start by fetching the documentation page and also searching for any relevant source code and discussions about this topic.

Now let me look at the actual source code and search for any DTensor-specific handling:

Both v2.13.0 and main are identical — no DTensor handling. Let me now search for FSDP2-specific gradient clipping and any GitHub issues on this topic:

Let me dig deeper into the FSDP2-specific `clip_grad_norm_` and DTensor interactions:

Excellent! Found two critical issues. Let me dig into the DTensor `clip_grad_norm_` PR and the FSDP2 CPU offload issue:

Now I have all the key pieces. Let me do one final check to see how `_group_tensors_by_device_and_dtype` handles DTensors:

Now I have a complete picture. Here's the comprehensive analysis:

---

## Answer: Yes — But With Important Nuances

`torch.nn.utils.clip_grad_norm_` **does work with DTensor gradients in FSDP2**, but there is **no special DTensor-aware code path** in the function itself. It relies entirely on DTensor's operator dispatch mechanism to make things work transparently.

---

### How It Works Under the Hood

Looking at the source of `_get_total_norm` (the internal norm-computation function called by `clip_grad_norm_`):

```python
# In _get_total_norm (simplified)
norms = []
for (device, _), ([device_tensors], _) in grouped_tensors.items():
    if foreach_path:
        norms.extend(torch._foreach_norm(device_tensors, norm_type))
    else:
        norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_tensors])

total_norm = torch.linalg.vector_norm(
    torch.stack([norm.to(first_device) for norm in norms]), norm_type
)
```

**For DTensor gradients, the foreach path is skipped** because `_has_foreach_support` checks `type(t) in [torch.Tensor]` — and `DTensor` is not `torch.Tensor`. It falls through to the per-tensor path:

1. **`torch.linalg.vector_norm(g, norm_type)` on a DTensor** → dispatched to DTensor's `vector_norm_single_dim_strategy`, which computes the local norm on the shard and uses `_NormPartial` as output placement → triggers an **all-reduce per gradient tensor**.

2. **`torch.stack(...)` of DTensor norms** → stacks the (now replicated) scalar norms.

3. **Final `torch.linalg.vector_norm(...)`** → another DTensor dispatch → final all-reduce → returns a **replicated scalar DTensor** as `total_norm`.

---

### The PR That Made It Work: [#121945](https://github.com/pytorch/pytorch/pull/121945)

This PR (merged for PyTorch 2.4+) specifically:

| Fix | Detail |
|-----|--------|
| **2D Mesh Support** | Added proper handling for TP + DP (2D `DeviceMesh`) scenarios |
| **`_partition_value` Fix** | Fixed norm partitioning for different `norm_type` values (1, 2, general p) — previously only `norm_type=2` was correct |
| **PyTree Support** | Added `needs_pytree` support in `OpSchema` for `_foreach_norm`/`_foreach_powsum` so DTensor can handle grouped tensor inputs |

The critical fix in `_partition_value`:

```python
# Before PR #121945 (wrong for norm_type != 2):
return tensor / mesh.size(mesh_dim=mesh_dim)

# After PR #121945 (correct for all p-norms):
if self.norm_type == 1:
    return tensor / mesh.size(mesh_dim)
# For general p-norm:
return tensor / math.pow(mesh.size(mesh_dim), 1 / self.norm_type)
```

Without this fix, `clip_grad_norm_` with `norm_type=1` or `norm_type=3` (etc.) on DTensors would return a **wrong** `total_norm` (off by a factor).

---

### Communication Overhead

The current implementation is **functionally correct but communication-inefficient**: each gradient tensor triggers its own all-reduce during `torch.linalg.vector_norm(g, norm_type)`. For a model with $N$ parameters, you get $N+1$ all-reduces instead of the ideal single all-reduce.

In practice, this is often acceptable because:
- Many models use `foreach=True` which batching on the non-DTensor path
- For very large models, this overhead may be noticeable

---

### Known Issues in FSDP2 Context

| Issue | Status |
|-------|--------|
| **[#148532](https://github.com/pytorch/pytorch/issues/148532)** — CPU offload + `clip_grad_norm` | Closed fixed — gradients must be on GPU when calling `clip_grad_norm_` |
| **[#171517](https://github.com/pytorch/pytorch/issues/171517)** — Mixed-dtype gradients | Closed — FSDP's own `clip_grad_norm_` wrapper had dtype issues |

---

### Practical Recommendation

For FSDP2, you can safely use:

```python
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=1.0, norm_type=2.0
)
```

- The returned `total_norm` will be a **replicated DTensor** (same scalar value on all ranks).
- If you need just the norm without clipping, use `torch.nn.utils.get_total_norm()`.
- Ensure all gradients reside on GPU (not CPU-offloaded) when calling.
- For `norm_type=2` (L2, the default), the computation is correct in all PyTorch versions ≥ 2.4. For other norm types, you need the fix from PR #121945 (PyTorch ≥ 2.4).