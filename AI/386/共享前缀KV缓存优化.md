# 共享前缀KV缓存优化

### 🧑‍💻 User

Currently the server slot's KV-cache is available only if the slot that served a request is now idle. But there are many situations when the slot is in use and another request with the same prompt prefix is received. For example, if a few users work with a knowledge base that is relatively big (hundreds of thousands of tokens) then when the first user starts the processing of the knowledge base's text, which could take minutes, or even tens of minutes, the other users can not reuse the processed prompt's kv-cache and must reprocess all the text entirely, spending minutes or even tens of minutes. But it is even worse if we add the generation stage, it again could take minutes or even tens of minutes. During the generation the slot is still used, then its kv-cache is still unavailable for reuse. In total it could take tens of minutes until the kv-cache of the first slot will be released and become available for reuse. Then, during an half of hour, for example, every new user should start his own prompt processing task which is put into a queue and waits until the other users will get their prompts processed, and it is beside of the time it takes to process the user's prompt again. So, in total it could take more than an hour of absolutely wasted user and GPU time.

The feature request is about to increase the efficiency of the same prompt processing. For some model architectures the current implementation of llama.cpp already saves a state of kv-cache per each predefined number of processed tokens. Then for new requests for the processing of the same text it is enough just to include in the longest prefix search the saved states. If such state is found it could be copied into a new slot and there will be no additional processing and related inefficiency. For the architectures that do not require the state saving it is possible to introduce such practice of saving intermediary state, unified with the architectures where the state is already saved. As a result the first user will always process the large prompt "for all", meaning every next user could start the work with the knowledge base without waiting for the reprocesing of the whole large text and, additionally, not wasting time of other users who now must wait until GPU will finish the completely wasted reprocessing effort.

And if we remember that there could be a lot of scenarios when the same prefix is required by many users/agents then the need for such optimization becomes even more demanding.

So, It's necessary to solve for llama.cpp kv cache manage?

---

### 🤖 Assistant

I'll start by discovering what tools are available through the hub, in case there are code/GitHub tools that can ground this analysis in the actual llama.cpp implementation.

The hub gives me browser/fetch tools. Let me ground this in the actual current llama.cpp implementation rather than answer purely from memory.

I now have solid grounding. Let me pull a few more specifics on the existing checkpoint/prompt-cache mechanisms so my analysis maps to the exact current code.

## Verdict

Yes — the problem is real and important, but the way you framed the solution is only half right, and the good news is that a large part of what you're asking for **already landed in llama.cpp master during 2025–2026**.

The correct framing is:

> This is primarily a **shared-prefix registry** problem, not a "snapshot the KV every $N$ tokens" problem — for dense full-attention transformers you want to **share/copy-on-write KV cells**, not duplicate states. Snapshots (checkpoints) matter only where the memory is *non-reconstructible* (SWA / recurrent / hybrid).

Below is what already exists, where the genuine gap is, and a concrete design.

---

## 1. What already exists in current master

I checked the current `tools/server/README.md`, `src/llama-kv-cache.h`, and the merged PRs. Your request maps onto several existing mechanisms:

| Your requirement | Existing mechanism | PR / flag |
|---|---|---|
| Reuse the prefix when the serving slot is **not** idle | Host-side **prompt cache** decoupled from slots (`--cache-ram`, default 8 GiB; `--cache-idle-slots` saves idle slots into it on a new task) | #16391 |
| Periodically saved intermediate states | **Context checkpoints** per slot (`--ctx-checkpoints`/`--swa-checkpoints`, default 32; `--checkpoint-min-step`, default 8192) | #15293, #16382 |
| "Architectures that don't need state saving" | SWA/hybrid/recurrent checkpoints via a flags API; dense attention needs no snapshot because the full KV is present | #15293, #16382 |
| Cheap prefix copy into a new slot | Unified KV across sequences (`--kv-unified`, `n_stream == 1`) + `seq_cp` | core |
| Explicit save/restore of a slot | `--slot-save-path` + `POST /slots/{id}?action=save|restore` | server |

The low-level plumbing you'd need is already there in `src/llama-kv-cache.h`:

- `seq_cp`, `seq_rm`, `seq_keep`, `seq_add`, `seq_div`
- `state_write` / `state_read` / `state_read_sinfo`
- `stream_copy_info` (pending stream copies applied at next `update`)
- extended sequence-state API: `llama_state_seq_{get_size,get_data,set_data}_ext(...)` with `LLAMA_STATE_SEQ_FLAGS_*`
- an explicit TODO in the code: **`[TAG_KV_CACHE_SHARE_CELLS]`** — *"temporary until we refactor to be able to share the same cells between 2 kv caches."*

Crucially, `src/llama-kv-cells.h` stores, per cell, `pos` plus a `std::bitset` of owning sequences. **Sharing a KV cell between sequences is O(1) metadata**, not a data copy. That is the whole reason `--kv-unified` enables "efficient prefix sharing."

> Historical note: the exact concern you're describing was raised by user `kripper` in **discussion #13606** in early 2026 — non-optimal LCP slot selection, no cross-slot prefix sharing, agents needing to branch from a checkpoint. Worth reading before filing a new issue.

---

## 2. Where the genuine gap still is

So what *is* still unsolved? Three things, and they're exactly the hard part of your request:

1. **Checkpoints are produced on slot *transition*, not *in flight*.** The SWA checkpoint (PR #15293) is created *"upon finishing the processing of a prompt."* So while a slot is chewing through a 200k-token prefill for minutes, and then generating for more minutes, no *newly produced* reusable state is exposed to other requests — except the coarse `--checkpoint-min-step` ticks. Your "half an hour of wasted GPU" scenario is specifically about this window.
2. **Cross-slot prefix sharing is not the default scheduler behavior.** Slot selection is similarity/LCP-based and slot-local; two clients with a shared 50k-token prefix on *different* slots do **not** automatically share the KV. (`--kv-unified` makes it *possible*, not *automatic*.)
3. **No reference-counted / copy-on-write view of a busy sequence's prefix.** `seq_cp` marks cells as jointly owned, but a busy slot that keeps appending is still mutating the tail; there is no "freeze prefix $[0,p)$, share it read-only, let the source keep generating" abstraction.

The `[TAG_KV_CACHE_SHARE_CELLS]` TODO is literally the missing primitive for (2) and (3).

---

## 3. Why "save a state every $N$ tokens for every architecture" is the wrong first move

Do the math. Llama-3-8B, f16 KV:

$$2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times 2\ \text{bytes} = 2 \times 32 \times 8 \times 128 \times 2 \approx 128\ \text{KB/token}$$

- A **200k-token** full-attention snapshot ≈ **25 GB**.
- Storing 25 such snapshots (every 8k tokens) ≈ **640 GB**. Absurd.
- An **SWA snapshot** (window ≈ 1024) is $1024 \times 128\text{ KB} \approx 128\text{ MB}$ — *feasible*, which is exactly why checkpoints were introduced for SWA/hybrid/recurrent first.

So the architecture distinction in your request is real, but it points the opposite way from "unify by snapshotting everything":

| Memory type | Reconstructible? | Right mechanism |
|---|---|---|
| Dense full attention | Yes (KV is all there) | **Cell sharing / COW** (`seq_cp`), *not* snapshots |
| SWA | No (old KV is discarded) | **Checkpoints** (small) ✅ already done |
| Recurrent (Mamba/RWKV/DeltaNet) | No | **Checkpoints** ✅ already done (#16382) |
| Hybrid | Partly | **Checkpoints** for the recurrent/SWA part ✅ |

Trying to "introduce checkpointing practice" for dense attention would be the memory blow-up above. Copying/sharing cells is the cheap path.

---

## 4. What the actual feature request should say

If you file this upstream, frame it as **shared prefix state across slots, independent of slot occupancy**, with four deliverables:

<details>
<summary><b>Concrete design sketch (expand)</b></summary>

1. **Global prefix/state registry** (extends `--cache-ram`).
   Index every reusable state by `(model, token-prefix-hash) → {seq_state | cell-ranges, pos, refcount}`. Populate it incrementally *during* prefill at `--checkpoint-min-step` boundaries, not only at prompt completion.

2. **Reference-counted cell sharing (implement `[TAG_KV_CACHE_SHARE_CELLS]`).**
   A new request with the same prefix does `seq_cp` from the producer sequence's frozen $[0,p)$ range into the new sequence. `llama_kv_cells` already stores a `bitset` per cell, so this is metadata-only. Add COW semantics so the busy producer can append to $[p,\infty)$ without disturbing sharers (or bump cell refcounts and let the ring buffer evict only when refcount hits 0).

3. **Read-only prefix snapshot for in-flight slots.**
   Expose the current prefix of a *busy* sequence as a shareable range. This is the piece that removes your "wait until the slot is idle" wait.

4. **Best-prefix scheduler, not best-similarity slot.**
   Change slot assignment to maximize **reusable prefix length** across *all* registered states, not slot-local LCP similarity (the `kripper` complaint). The `tokens_cached` / `cache_n` counters already expose the payoff per request.

</details>

The `llama_state_seq_*_ext` flags API from #15293 is the right extensibility hook: add a `LLAMA_STATE_SEQ_FLAGS_PREFIX_ONLY` alongside `SWA_ONLY` / `PARTIAL_ONLY`, and let the server register prefix states without copying K/V tensors.

---

## 5. Practical advice today

- Use `--kv-unified` (default when slots are auto) so prefix sharing is *possible*, `--cache-ram` sized to your RAM, and `--cache-idle-slots`.
- For SWA/hybrid models set `--swa-full` **off** and raise `--ctx-checkpoints`; `--swa-full` on removes the need for checkpoints at the cost of a much larger cache.
- Tune `--slot-prompt-similarity` and, for deterministic behavior, assign `id_slot` client-side (as the #13606 tutorial does with `--slot-save-path`).
- Understand the current limitation: shared prefixes only pay off when they land on **the same slot** or within one unified context. That's exactly the gap to push upstream.

---

## Bottom line

**Yes, it's necessary — but it's largely a solved-in-progress problem, and the remaining work is cross-slot / in-flight *cell sharing*, not universal checkpointing.** SWA, recurrent, and hybrid state checkpoints already exist (#15293, #16382), and host-RAM prompt caching decouples reuse from slot idleness (#16391). What's genuinely still missing — and worth solving — is: (a) exposing a *busy* sequence's prefix, (b) automatic, refcounted, cross-slot prefix sharing (the `[TAG_KV_CACHE_SHARE_CELLS]` refactor), and (c) making the scheduler maximize reusable prefix length rather than slot-local similarity. For dense attention, sharing cells is the correct, cheap solution; snapshotting full KV would explode memory by orders of magnitude.