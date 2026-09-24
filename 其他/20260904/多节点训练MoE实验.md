# 多节点训练MoE实验

## 🧑‍💻 User

dgx spark multi node support torchlightning training?

---
## 🤖 Assistant

I'll research this question using web search and fetching.

Let me fetch the most relevant pages for details.

Let me get more specific details from the forum thread and playbook sources.

## Short answer

**Yes — PyTorch Lightning multi-node training on DGX Spark is supported and works, but it's not plug-and-play.** The framework layer (Lightning) is the easy part; the hard part is the DGX Spark interconnect + NCCL/RDMA configuration, and there are known rough edges you should expect.

<details>
<summary><b>What NVIDIA officially supports (2–4 nodes)</b></summary>

NVIDIA ships DGX Spark playbooks for multi-node setups:

| Nodes | Interconnect playbook | Notes |
|-------|----------------------|-------|
| 2 | **Connect Two Sparks** | Direct **200 Gb/s QSFP56** cable between them |
| 3 | **Connect Three Sparks** | Ring topology |
| 4 | **Connect Multiple Sparks through a Switch** | Switched fabric |
| — | **NCCL for Multiple Sparks** | Builds/validates NCCL for 2–4 Sparks |
| — | **Fine-tune with PyTorch** | Has an explicit **"Run on two Sparks"** distributed path |

The NCCL playbook explicitly frames this as enabling *"multi-node distributed training on two, three, or four DGX Spark systems with Blackwell architecture"* and includes helper scripts (`setup.sh`, `launch.sh`) plus an `all_gather` test to validate bandwidth. So NVIDIA's supported multi-node ceiling for Spark is **4 nodes** via the documented playbooks.

</details>

<details>
<summary><b>Does PyTorch Lightning itself work? (yes, framework-level)</b></summary>

Lightning is architecture-agnostic Python and rides on top of `torch.distributed`. Lightning's own docs state:

> *"If your script uses the PyTorch Lightning Trainer, your code is already compatible and optimized for multi-GPU and multi-node training. No code changes required!"*

So the same `Trainer` that works single-node works multi-node. You just need the standard launcher contract:

- `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `WORLD_SIZE`
- one process per node (`--nproc_per_node=1` on Spark, since each Spark has **1 GPU** — the GB10 SoC)

Minimal pattern (via `torchrun` on each node):

```bash
# On node 0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=192.168.100.1 --master_port=29500 train_lightning.py

# On node 1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=192.168.100.1 --master_port=29500 train_lightning.py
```

Or let Lightning configure from env automatically:

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=1,          # 1 GPU per Spark
    num_nodes=2,        # set per your cluster
    strategy="ddp",     # "fsdp" for larger models (128 GB unified mem helps)
)
```

</details>

<details>
<summary><b>Real-world evidence + known problems</b></summary>

**Confirmed working, with caveats:**

1. **NVIDIA Developer Forums — "PyTorch Lightning on 2 DGX Sparks crashes connected by QSFP"** (a ControlNet workload, "just PyTorch-Lightning"):
 - NCCL initialized, all-reduce ran (`NCCL INFO AllReduce: ... Algo RING proto LL`), and training *started on both nodes*.
 - It then **crashed at the end of the epoch** with a **collective timeout**:
 ```
 [Rank 0] Watchdog caught collective operation timeout:
 WorkNCCL(SeqNum=2407, OpType=ALLREDUCE, NumelIn=14749440, ...)
 ```
 - This is the classic **collective-mismatch / rank-divergence** signature — ranks issuing collectives in a different order or count (common in Lightning when validation/logging/early-stopping behavior is rank-dependent, or when a rank hits an error and stops issuing collectives).

2. **arXiv:2608.07226 — "Dual-Node NVIDIA DGX Spark over Tailscale"** (Aug 2026):
 - Successfully ran **2-node distributed pretraining** with **PyTorch `torchrun` + DDP + NCCL**, 200 Gb/s QSFP56 direct fiber link, 1 process/node.
 - Sustained ≈ **69.4 s/step, ≈ 1,890 tokens/s**, ~**653M tokens over 4 days** (depth-20 model, 128 GB unified memory per node).
 - Explicitly documents *"a step-zero evaluation bug that triggered NCCL timeouts"* — the same class of issue as the forum thread. Note their result is a **feasibility proof, not a scaling-efficiency claim** (single-node baseline was estimated, not measured matched).

So: multi-node Lightning *runs*, but you'll likely spend real debugging time on NCCL timeouts.

</details>

<details>
<summary><b>The practical gotchas to plan for</b></summary>

- **No NVLink between Sparks.** Inter-node traffic goes over **200 GbE / RoCE (ConnectX-7)**, not NVLink. Expect far lower inter-node bandwidth than a GB200 NVL system — DDP all-reduce overheads are real, and FSDP/Tensor-parallel across nodes is bandwidth-hungry.
- **ARM64 (aarch64)** platform. Most things work, but some wheels/packages need ARM compatibility checks.
- **NCCL must be right.** The playbook builds **NCCL v2.30.7-1 with Blackwell support**. You'll need correct interface binding (e.g. `NCCL_SOCKET_IFNAME`, RoCE/GID index) or you'll get hangs — a forum post documents an "RDMA gotcha with network host" on 2× Spark.
- **Timeouts from rank divergence.** In Lightning, avoid:
 - rank-dependent `if self.global_rank == 0:` around anything that changes collective counts
 - validation/sanity-check loops that don't run symmetrically
 - the known "step-zero evaluation" pitfall
 - tooling help: set `TORCH_NCCL_TRACE_BUFFER_SIZE` (FlightRecorder) and often `TORCH_NCCL_ASYNC_ERROR_HANDLING` / `TORCH_NCCL_BLOCKING_WAIT` to get stack traces on hang.
- **Container vs host networking.** The playbooks use Docker + NVIDIA Container Toolkit + (for some workflows) Docker Swarm. Container-native multi-node NCCL without host build is achievable but networking is the friction point.

</details>

### Bottom line

- **2 nodes:** Well-trodden path (official playbook + independent paper). Start here.
- **3–4 nodes:** Supported by playbooks, less battle-tested.
- **>4 nodes:** Not a documented Spark multi-node topology — you're off the paved road.
- **Lightning:** No code changes needed in principle; the work is all in interconnect + NCCL + avoiding collective divergence.

If you tell me your node count, interconnect (direct QSFP vs switch), and whether you're inside Docker, I can give you a concrete config (`NCCL_*` env vars + Lightning `Trainer` settings + a minimal smoke test) tailored to it.

---
## 🧑‍💻 User

could we can pretrain moe llm on the mini cluster?

---
## 🤖 Assistant

I'll research this properly — MoE pretraining has very different bottlenecks than dense training. Let me find the relevant tools.

Let me pull the full paper details and DGX Spark specs, plus MoE-specific data points.

Key finding: there's a **"Run NVFP4 Pretraining with Megatron Bridge"** playbook. Let me dig into that and the training playbooks.

Critical findings emerging. Let me verify MoE-specific support and check what the Spark multi-node playbooks actually target.

# Pretraining a MoE LLM on a DGX Spark mini-cluster

**Short answer: You can — but only at toy/research scale (roughly ≤200M–1B *total* params for a from-scratch pretraining run), and MoE is the *worst-case* workload for this cluster's weakest link.** For anything resembling a real MoE LLM, the answer is no. For fine-tuning a large MoE, the answer is a comfortable yes.

Here's the breakdown, grounded in the actual Spark numbers.

---

<details>
<summary><b>⚠️ First, clear up NVIDIA's "200B / 405B / 700B parameters" claims</b></summary>

NVIDIA's marketing says DGX Spark supports **"models up to 200B parameters"**, **405B for dual-Spark**, and ConnectX enables **"up to four Sparks... up to 700B parameters."**

These are **inference / running** figures, not training figures. They assume low-bit weight-only residency (e.g. gpt-oss-120b ships in MXFP4 at ~57–63 GB, which fits a single 128 GB Spark).

Training requires **weights + gradients + optimizer state**, roughly **16 bytes/param** with bf16 mixed precision + AdamW:

$$2\,(\text{bf16 w}) + 2\,(\text{bf16 g}) + 4\,(\text{fp32 master}) + 4\,(m) + 4\,(v) = 16\ \text{bytes/param}$$

That's an ~8–16× higher memory tax than inference. Don't map the marketing numbers onto pretraining.

</details>

---

<details>
<summary><b>📏 Memory ceiling — how big a MoE can you actually hold?</b></summary>

Per node: **128 GB unified** (LPDDR5X, **273 GB/s**). Cluster: up to **4 nodes = 512 GB**.

Reserving ~15–20% for activations, CUDA context, comm buffers, fragmentation:

| Setup | Usable mem | @16 B/param (AdamW bf16) | @10 B/param (8-bit optim) | Plausible MoE (total / active) |
|---|---|---|---|---|
| 2 nodes (officially supported) | ~215 GB | ~13B total | ~21B total | ~10B / ~2–3B active |
| 4 nodes | ~430 GB | ~27B total | ~43B total | ~20B / ~4–5B active |

**But** that ceiling assumes near-perfect ZeRO-3/FSDP sharding and negligible activations. Realistically, budget **60–75% of those numbers** → a practical from-scratch MoE on 4 Sparks is **~10–20B total params**, and on 2 Sparks **~7–12B total params**.

> **Critical MoE caveat:** if all experts fit in *one* node's 128 GB, you can use plain FSDP/DP with **no expert parallelism**. Once experts must be **split across nodes**, every MoE layer becomes an **all-to-all over 200 GbE** — see below.

</details>

---

<details>
<summary><b>⏱️ Throughput — this is what actually kills it</b></summary>

Your interconnect is **200 GbE QSFP. Measured NCCL bus bandwidth ≈ 24 GB/s** (a 3-node Spark user's `nccl-tests` result; theoretical is 25 GB/s). For comparison:

| System | Inter-node link | Per-node mem BW |
|---|---|---|
| DGX Spark | 200 GbE ≈ **24 GB/s** | **273 GB/s** |
| H100 DGX | NVLink/NVSwitch ~900 GB/s | 3.35 TB/s |
| GB300 | NVLink ~1.8 TB/s | ~8 TB/s |

Measured end-to-end training throughput (real published numbers):

- **Dual DGX Spark, DDP + NCCL, 200 Gb/s direct fiber** (arXiv:2608.07226): a *small dense* model at **~69.4 s/step, ~1,890 tok/s**, 653M tokens over 4 days.
- NanoChat d24 on a Spark **cluster**: reported **~5 days** (≈ half of single-Spark time).
- Continued-pretraining an 8–9B *dense* model on a **single** Spark: **~a week** for a small domain corpus.

**Now do the token budget.** Assume 4 nodes scale sub-linearly (≈1.7× of the 2-node number) → ~**3,200 tok/s**.

$$\text{tokens in 14 days} = 3200 \times 86400 \times 14 \approx 3.9\text{B tokens}$$

At a Chinchilla-ish $20\times$ params:

$$\text{max params} \approx 3.9\text{B} / 20 \approx \mathbf{195M\ total\ params}$$

| Target MoE (total params) | Tokens needed ($\sim 20\times$) | Time @ ~3,200 tok/s |
|---|---|---|
| **200M** | 4B | **~14 days** ✅ |
| **1B** | 20B | ~72 days (~2.5 months) 🟡 |
| **10B** | 200B | ~2.1 years ❌ |
| **Mixtral-class (47B)** | ~1T | ~10+ years ❌❌ |

Modern MoE LLMs are trained on **10T–30T tokens** on thousands of GPUs. You are 4–6 orders of magnitude away on the token-throughput axis. That's the hard wall — not memory.

</details>

---

<details>
<summary><b>🚧 MoE-specific bottlenecks (beyond the generic cluster limits)</b></summary>

1. **Expert parallelism = all-to-all over the slowest link.** With 1 GPU per Spark, any EP>1 *must* cross the 24 GB/s QSFP link, twice per MoE layer (dispatch + combine). TP across nodes is even worse (2 all-reduces/layer). **Avoid cross-node TP/EP for training.**
2. **Spark is memory-bandwidth-bound** at 273 GB/s (~12× less than H100). MoE training is heavily weight-load-bound, so per-node throughput is intrinsically low.
3. **No NVLink between Sparks.** Intra-node CPU↔GPU is NVLink-C2C, but *inter-node* is plain Ethernet/RoCE via ConnectX-7.
4. **Ecosystem lag:** Spark is **aarch64 + sm_121 + CUDA 13.0**. Community reports describe this as "significant challenges... the broader ecosystem hasn't caught up." DeepSpeed-MoE and many fused/quantized training kernels may lack ARM64/sm_121 builds.
5. **Framework status is telling:**
 - Megatron Bridge **has** Expert Parallelism + MoE recipes (and an NVFP4 mixed-precision pretraining recipe at ~1.68× BF16 speed) — but the Spark playbook for it targets **DGX Station / GB300, single-node**.
 - The **NanoChat training playbook explicitly removed DGX Spark support** ("not ready yet") and is Station-only.
 - Spark playbooks that *do* claim multi-node training are **fine-tuning** playbooks (PyTorch SFT/LoRA/QLoRA with Accelerate+FSDP, 2 nodes) — not pretraining.
 - The official **MoE on Spark material is all inference** (gpt-oss-120b on one Spark; multi-node MoE uses TP=2 for *serving*).

</details>

---

<details>
<summary><b>✅ What actually works — recommendation matrix</b></summary>

| Goal | Feasible? | Setup |
|---|---|---|
| **MoE architecture research** (routing, load balancing, ablations) — ~50–300M total params | ✅ **Yes** | From-scratch pretraining, 2–4 nodes, FSDP2, EP=1 (keep experts replicated) |
| **~1B total-param MoE** | 🟡 Yes, patient | ~2.5 months; Megatron Bridge or FSDP2 |
| **Full pretraining a competitive MoE LLM** | ❌ **No** | Physically token-bound |
| **Continued pretraining / domain adaptation of an existing MoE** | ✅ Yes | Much better ROI than from-scratch; small corpus |
| **MoE SFT / LoRA / QLoRA** (e.g. gpt-oss-120b) | ✅ **Yes, comfortable** | Works on **one** Spark; 2 nodes with Accelerate/FSDP |
| **Multi-node MoE inference** | ✅ Yes | TP=2 across 2 Sparks (vLLM / SGLang / Ray) |

**If you proceed, the correct architecture is:**
- **Size the model so all experts fit on one node** (avoid EP across the QSFP link entirely). Use **FSDP2 / ZeRO-3** for sharding, EP=1.
- Prefer **8-bit or FP8 optimizer states** to push the memory ceiling up ~1.6×.
- Use **activation checkpointing** aggressively.
- Use **NeMo / Megatron Bridge** (MoE + EP aware) rather than vanilla Lightning for MoE; **native PyTorch Lightning has no first-class MoE/EP support** — you'd hand-roll it.
- Set up via **NVIDIA Sync Cluster Assistant** (handles ConnectX-7 + inter-device SSH), then validate with `nccl-tests` before training.

</details>

---

<details>
<summary><b>Bottom line</b></summary>

The cluster is **memory-capacious but bandwidth-starved**, and MoE training punishes exactly that dimension — both the 273 GB/s per-node memory BW and the 24 GB/s inter-node link.

- **From-scratch MoE pretraining on Spark is viable only as a research toy** (~a few hundred million total params for a couple weeks). "Pretrain an MoE LLM" in any competitive sense: **not feasible.**
- **Fine-tuning / continued-pretraining a large MoE is genuinely useful** and officially supported — that's where the Spark cluster earns its keep for MoE work.
- If your real goal is MoE *training* at meaningful scale, this hardware is the wrong tool; if your goal is **MoE research, adaptation, and serving**, it's a reasonable fit.

If you tell me your node count and target (research ablation vs. domain adaptation vs. serving) and total/active param target, I can give you a concrete sharding plan, precision recipe, and a go/no-go on the token budget.
