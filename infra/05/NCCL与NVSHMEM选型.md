# NCCL与NVSHMEM选型

### 🧑‍💻 User

I am a heavy user of NCCL, but recently I am aware of a new toolkit named NVSHMEM, which allows different gpu devices to directly communicate with each other using one-side rdma-like verbs. I am wondering if the functionality of the two tools is overlapped? Could you please give me some instructions about when should I use NCCL instead of NVSHMEM? In another word, under what scenarios will NCCL outperform NVSHMEM?

---

### 🤖 Assistant

I've gathered substantial material from NVIDIA's official documentation, the NVSHMEM developer page, the NCCL overview, the NVSHMEM best practices guide, and the key blog post about NVSHMEM leveraging NCCL. Here is a comprehensive analysis.

---

## NCCL vs NVSHMEM: A Comprehensive Comparison

### 1. Are They Overlapping? — Yes, Partially

There **is** meaningful overlap, but the overlap is actually by design — NVSHMEM **embeds and depends on NCCL** for part of its functionality. Since NVSHMEM 2.0, the host and on-stream collective APIs in NVSHMEM are backed by NCCL under the hood. From the official blog post:

> *"For every NVSHMEM team, NVSHMEM constructs a corresponding NCCL communicator. NVSHMEM calls NCCL whenever there is a corresponding collectives API available in NCCL."*

And from the NVSHMEM Best Practices Guide:

> *"When available, NVSHMEM will use NVIDIA Collective Communication Library (NCCL) for on-stream collectives. These collectives are highly optimized for host-submission and are recommended for applications composing existing NCCL-optimized workloads with pt-to-pt NVSHMEM operations."*

So the relationship is best understood as **complementary, layered, with NCCL serving as the optimized collective backend for NVSHMEM**.

---

### 2. Fundamental Design Philosophy Difference

| Dimension | NCCL | NVSHMEM |
|-----------|------|---------|
| **Programming Model** | MPI-like, two-sided collectives + p2p | PGAS (Partitioned Global Address Space), one-sided RDMA-like |
| **Initiator** | Host (CPU) launches ops on CUDA streams | GPU threads *themselves* can initiate communication from inside CUDA kernels |
| **Global Address Space** | No — explicit send/recv buffers | Yes — symmetric heap accessible across all PEs via `nvshmem_ptr()` |
| **Synchronization** | Implicit in collectives; explicit for p2p | Explicit: `nvshmem_barrier`, `nvshmem_quiet`, `nvshmem_fence` |
| **Primary Target** | Deep learning training (AllReduce) | HPC / scientific computing with irregular communication patterns |

---

### 3. When NCCL Outperforms NVSHMEM (Use NCCL)

#### ✅ Collective-Heavy Workloads (Especially Large Messages)

NCCL's collectives are the gold standard. NVSHMEM's *native* collectives used a recursive-doubling algorithm optimized only for small-message latency. The blog post benchmarked `nvshmem_int_sum_reduce_on_stream`:

- **2 GPUs (DGX A100):** NCCL-backed was **1.5×–10× faster**
- **16 GPUs (2 nodes):** NCCL-backed was **up to ~260× faster** for large (1 MiB) messages
- **Small 4-byte messages:** Still **1.5×–1.7× faster** with NCCL

This is why NVSHMEM itself defers to NCCL for collectives.

#### ✅ Deep Learning Training

NCCL's `AllReduce` is the backbone of distributed data-parallel training in PyTorch DDP, Horovod, TensorFlow, JAX, etc. NCCL is deeply integrated into these frameworks. The topology-aware ring/tree algorithms are tuned specifically for the gradient synchronization pattern.

#### ✅ When You Want Automatic Algorithm Selection

NCCL automatically selects the best algorithm (Ring, Tree, CollNet, NVLS) and protocol (LL, LL128, Simple) based on message size and topology. You get this for free. With NVSHMEM's native point-to-point, you must manually orchestrate the communication pattern.

#### ✅ Two-Sided Communication Patterns

When both sender and receiver need to coordinate — e.g., the receiver needs to know the incoming message size, or when flow control matters — NCCL's `ncclSend`/`ncclRecv` model is more natural and safer than NVSHMEM's one-sided `put`/`get`.

#### ✅ Ease of Adoption

NCCL's API closely follows MPI collectives. If you're porting an MPI application or using a DL framework, NCCL is a drop-in. NVSHMEM requires adopting the PGAS mental model — symmetric heap allocation, manual `barrier`/`quiet` calls, and explicit remote address calculation.

---

### 4. When NVSHMEM Outperforms NCCL (Use NVSHMEM)

#### ✅ GPU-Initiated, In-Kernel Communication

This is NVSHMEM's killer feature. Individual GPU threads can call `nvshmem_put()`, `nvshmem_get()`, `nvshmem_atomic_add()`, etc., *directly from inside a CUDA kernel* — no CPU involvement, no kernel launch overhead:

```c
__global__ void halo_exchange_kernel(float *data, ...) {
    int my_pe = ...;
    int neighbor_pe = ...;
    // Directly put data to neighbor's GPU memory from within the kernel!
    nvshmem_float_put(neighbor_dest, my_src, chunk_size, neighbor_pe);
}
```

NCCL has a newer **Device API** (`ncclPutSignal`/`ncclWaitSignal`) for one-sided operations, but NVSHMEM's PGAS model is more mature here.

#### ✅ Strong Scaling with Irregular Communication

When you have fine-grained, irregular communication interleaved with computation (e.g., halo exchanges in stencil codes, unstructured mesh applications, particle methods), NVSHMEM's one-sided model eliminates the CPU-GPU synchronization that would otherwise bottleneck strong scaling. From the developer page:

> *"NVSHMEM enables long-running kernels that include both communication and computation, reducing overheads that can limit an application's performance when strong scaling."*

#### ✅ HPC Applications with Proven Wins

- **QUDA (Lattice QCD):** NVSHMEM-enabled QUDA showed significant strong-scaling efficiency improvements on DGX SuperPOD (180 vs 141 TFlops at 256 GPUs, 346 vs 253 TFlops at 1024 GPUs)
- **LBANN (Deep Learning):** NVSHMEM-based halo exchange was up to **~2× faster** than MPI-based on Sierra at 32 GPUs
- **GROMACS:** Massively improved multi-node GPU scalability

---

### 5. Decision Matrix Summary

| Scenario | Recommendation |
|----------|---------------|
| Distributed DL training (DDP, FSDP) | **NCCL** — framework native, optimized AllReduce |
| Collective operations (AllReduce, AllGather, etc.) | **NCCL** — NVSHMEM defers to NCCL for these anyway |
| In-kernel, fine-grained p2p from GPU threads | **NVSHMEM** — its core strength |
| Irregular / random access communication patterns | **NVSHMEM** — one-sided PGAS is the right model |
| Halo exchange / stencil computations | **NVSHMEM** — proven strong-scaling wins |
| MPI-like two-sided orchestrated communication | **NCCL** — more natural API match |
| Need both collectives AND GPU-initiated p2p | **Use both** — NVSHMEM's on-stream collectives are NCCL-backed; use NVSHMEM device API for in-kernel p2p |
| Porting existing MPI code | **NCCL** — minimal API friction |
| Porting existing OpenSHMEM / PGAS code | **NVSHMEM** — designed for this |

---

### 6. Key Takeaway

The two libraries are **not competitors — they are collaborators**. NVSHMEM literally builds on NCCL for collectives. Think of it as:

- **NCCL:** The best-in-class *collective communication engine*
- **NVSHMEM:** A *PGAS programming model* that exposes GPU-initiated one-sided RDMA-like operations, and delegates collectives to NCCL

If your workload is collective-heavy (like DL training), stick with NCCL. If you need GPU threads to initiate fine-grained, one-sided data movement without CPU involvement, NVSHMEM is the right tool. And in many HPC applications, the optimal design uses both — NVSHMEM for in-kernel point-to-point and NCCL (via NVSHMEM's on-stream API) for bulk collectives.

---

### 🧑‍💻 User

We  know that there is a bit of overlap, but NVSHMEM is more like a lower-level API, providing put/get (load/store, one-sided) semantics at the CUDA level, while NCCL provides two-sided, bulk operations, launched at the CPU level.

It's easier to use NCCL since you can just add NCCL operation as part as your CUDA stream flow, and synchronization are taken care of by the two-sided semantics.

Using NVSHMEM is trickier as it happens inside your CUDA code and you have to write the synchronization code yourself (GPU A writes data to GPU B, sync, GPU B reads the data).

After reading the documents of NVSHMEM, I notice that its API seems to require direct communication between each pair of gpus. Is that to say, when there are two gpus that are connected with 2-hop nvlink (gpu1 and gpu3), NCCL can still utilize the nvlink while NVSHMEM cannot?

---

### 🤖 Assistant

This is an excellent and very precise question. Let me dig deeper into the NVSHMEM transport layer to give you a definitive answer.

Let me fetch the technical documentation and the blog post about NVSHMEM's transport layer to get a definitive answer on this.

Now let me fetch the NVSHMEM Best Practices Guide which should have critical details about the transport layer.

Let me fetch the specific pages that discuss the transport layer and device API behavior.

Now I have enough information from the official documentation to give you a precise, evidence-backed answer. Let me synthesize everything.

---

## Short Answer: You Are Largely Correct

**NVSHMEM's raw point-to-point `put`/`get` operations do NOT perform multi-hop NVLink routing.** They require direct P2P connectivity between the source and destination GPU. NCCL, by contrast, explicitly builds topology-aware ring/tree algorithms that can route through intermediate GPUs. This is a fundamental architectural difference with real performance implications.

---

## The Evidence

### 1. `NVSHMEM_TEAM_SHARED` Reveals the Direct-Connection Requirement

From the NVSHMEM 2.0 blog post by the NVSHMEM lead developer:

> *"`NVSHMEM_TEAM_SHARED` — The PEs whose symmetric heap is **directly accessible** through memory load/store/atomic using the `nvshmem_ptr` operation."*

And critically, on the DGX-1 V100 topology (which has **two NVLink islands** — GPUs 0–3 and GPUs 4–7, each fully connected within their group but NOT across groups):

> *"On DGX-1 V100, the `NVSHMEMX_TEAM_NODE` team contains all PEs running on the same node but on each node, **there are two `NVSHMEM_TEAM_SHARED` teams**."*

This means that GPUs 0–3 form one `NVSHMEM_TEAM_SHARED`, and GPUs 4–7 form another. GPUs **across** these two NVLink islands (e.g., GPU1 and GPU5) **cannot** be in the same shared team because there is no direct NVLink path between them.

In contrast, on the fully-connected DGX A100:

> *"On systems like an NVIDIA DGX A100 where **all GPUs are directly connected** to all other GPUs via NVLink, the `NVSHMEMX_TEAM_NODE` teams contain the **same PEs** as the `NVSHMEM_TEAM_SHARED` teams."*

### 2. The Transport Layer Confirms No Multi-Hop Routing

From the Best Practices Guide, NVSHMEM has three transport paths for device APIs:

| Transport | Mechanism | When Used |
|-----------|-----------|-----------|
| **P2P Transport** | GPU SMs directly push data over NVLink/PCIe | Intra-node, direct P2P available |
| **Proxy-Based Transport** | CPU proxy thread relays data via NIC | Inter-node or when direct P2P unavailable |
| **GDAKI Transport** | GPU directly drives NIC (IBGDA/GPUNetIO) | Inter-node with capable hardware |

There is **no "NVLink forwarding" or "multi-hop routing" transport layer** mentioned anywhere in the documentation. The P2P transport uses the GPU's SMs to push data directly over NVLink — it does not route through intermediate GPUs.

### 3. What Actually Happens on a 2-Hop Topology

When GPU1 wants to `nvshmem_put()` data to GPU3 and they are in different NVLink islands (e.g., DGX-1 V100):

- **NVSHMEM cannot use NVLink directly.** The P2P transport requires a direct NVLink or PCIe P2P path.
- It falls back to either **PCIe P2P** (slower, if the PCIe topology allows it) or in the worst case, the **proxy/network transport** which goes through CPU memory and the NIC — even for intra-node communication.
- **NCCL, by contrast, builds a ring** that goes GPU1 → GPU2 → GPU3, fully utilizing the NVLink bandwidth of both hops.

This is precisely why NCCL's collectives are up to **260× faster** than NVSHMEM's native collectives for large messages across 16 GPUs (from the blog post). NVSHMEM's native recursive-doubling cannot exploit multi-hop paths.

---

## Architectural Summary

```
┌─────────────────────────────────────────────────────────┐
│  NCCL: Topology-Aware Routing                            │
│                                                         │
│  GPU1 ──NVLink── GPU2 ──NVLink── GPU3                   │
│   ↑                                  ↓                  │
│   └────────── Ring / Tree ───────────┘                  │
│                                                         │
│  ✓ Can use intermediate GPUs as hops                    │
│  ✓ Automatic ring/tree/collnet topology discovery       │
│  ✓ Optimized for both latency and bandwidth             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  NVSHMEM: Direct P2P Only (put/get)                     │
│                                                         │
│  GPU1 ──NVLink── GPU2     GPU3                          │
│   │                  │                                  │
│   └──────────────────┘  ← No direct path!               │
│       Falls back to PCIe P2P or proxy/NIC               │
│                                                         │
│  ✗ No multi-hop NVLink routing for raw put/get          │
│  ✓ Collectives (on-stream) delegate to NCCL internally  │
└─────────────────────────────────────────────────────────┘
```

---

## Practical Implications

| Scenario | NCCL | NVSHMEM (raw put/get) |
|----------|------|-----------------------|
| DGX A100 / H100 (fully connected NVSwitch) | ✓ Full bandwidth | ✓ Full bandwidth (all GPUs in same `TEAM_SHARED`) |
| DGX-1 V100 (two NVLink islands) | ✓ Rings across islands | ✗ Falls back to PCIe/proxy across islands |
| Multi-node | ✓ Tree/ring over IB | ✓ Proxy or GDAKI over IB |
| GPU-initiated, in-kernel | ✗ Limited (new Device API) | ✓ **Core strength** |
| DL training (AllReduce) | ✓ **Gold standard** | ✗ Use NCCL-backed on-stream collectives |

---

## The Nuance: NVSHMEM Collectives Now Use NCCL

Since NVSHMEM 2.0, the **on-stream and host collective APIs** (`nvshmem_*_reduce_on_stream`, etc.) are **backed by NCCL** when built with `NVSHMEM_USE_NCCL=1`. So when you call an NVSHMEM collective, you *are* getting NCCL's topology-aware routing under the hood. The multi-hop limitation applies specifically to the **raw device-side `put`/`get`/`p` operations** that you call from inside CUDA kernels.

This creates a practical design pattern: use NVSHMEM's device API for in-kernel point-to-point between directly-connected GPUs, and use NVSHMEM's on-stream API (NCCL-backed) for bulk collectives across the full topology.

---

### 🧑‍💻 User

The flexibility, the beauty of it really, of NVSHMEM is that you could write code to, for example, do GPU-based forwarding in that scenario, such that whenever GPU1 wants to talk to GPU3, it goes through a 2-hop NVLink connection rather than the NIC. Of course, NCCL does this heavy-lifting already, I believe, through its ring algorithms as in the DGX topology, for example, there are actually three [parallel rings](https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf) encompassing all 8 GPUs. Please explain it with code.

---

### 🤖 Assistant

This is a fantastic deep-dive. The core insight is exactly right: NVSHMEM gives you the *primitives* (one-sided put/get from within CUDA kernels), and you can compose them into multi-hop forwarding patterns — essentially building the same ring/tree algorithms that NCCL provides pre-built. Let me show this concretely with code.

---

## The Topology Problem, Visualized

On the DGX-1 V100, GPUs are arranged in a cube-mesh with two NVLink islands. But there are *some* cross-island NVLink connections, and the three parallel rings exploit these:

```
DGX-1 V100 NVLink Topology (simplified):

 Island A (fully connected):          Island B (fully connected):
   GPU0 ═══ GPU1                        GPU4 ═══ GPU5
    ║  ╲  ╱  ║                          ║  ╲  ╱  ║
    ║   ╳   ║                          ║   ╳   ║
    ║  ╱  ╲  ║                          ║  ╱  ╲  ║
   GPU3 ═══ GPU2                        GPU7 ═══ GPU6

 Cross-island links (partial):
   GPU0 ←→ GPU4    GPU1 ←→ GPU5    GPU2 ←→ GPU6    GPU3 ←→ GPU7

Ring 1: GPU0 → GPU1 → GPU3 → GPU2 → GPU6 → GPU7 → GPU5 → GPU4 → GPU0
Ring 2: GPU0 → GPU2 → GPU1 → GPU3 → GPU7 → GPU5 → GPU6 → GPU4 → GPU0
Ring 3: GPU0 → GPU3 → GPU1 → GPU2 → GPU6 → GPU4 → GPU7 → GPU5 → GPU0
```

NCCL uses these rings. A naive `nvshmem_put(gpu3_dst, gpu1_src, size, 3)` from GPU1 to GPU3 **cannot** use NVLink if they lack a direct NVLink connection — and falls back to PCIe or NIC/proxy.

But with NVSHMEM, you can **build the ring yourself** from inside a CUDA kernel.

---

## Part 1: The "Naive" Approach (Fails on 2-Hop Topologies)

Here's what happens when you try a direct put to a non-neighbor GPU:

```c
// This kernel runs on GPU1, trying to write directly to GPU3's memory
__global__ void naive_halo_exchange(float *my_data, int dst_pe) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    // Allocate from the symmetric heap
    float *my_sym_buf = (float*)nvshmem_malloc(sizeof(float) * N);
    float *dst_sym_buf; // We need this on GPU3
    int dst_offset;
  
    nvshmem_float_put_nbi(dst_sym_buf + dst_offset,
                           my_sym_buf, N, dst_pe);
    nvshmem_quiet(); // Flush outstanding puts
  
    // ⚠️ PROBLEM: If GPU1 and GPU3 are not directly NVLink-connected:
    //
    //   - P2P transport: FAILS or falls back to PCIe P2P (slower)
    //   - If PCIe P2P also unavailable: falls back to PROXY transport
    //     (CPU proxy thread → NIC → remote CPU → GPU), which is SLOW
    //
    //   The NVLink bandwidth is wasted because there's no automatic routing.
}
```

---

## Part 2: Manual GPU-Based Forwarding with NVSHMEM

Now let's implement the forwarding pattern. The idea: GPU1 puts data to GPU2 (direct NVLink), and GPU2's kernel forwards it to GPU3 (direct NVLink). Both hops run at full NVLink speed.

### Step 1: Set Up Forwarding Buffers and Signaling

```c
#include <nvshmem.h>
#include <cuda/atomic>

#define CHUNK_SIZE (1024 * 1024)  // 1 MiB chunks

// Per-PE symmetric state for the forwarding ring
typedef struct {
    float *data_buf;          // My local data in symmetric heap
    float *fwd_buf;           // Forwarding buffer (for data passing through me)
  
    // Synchronization: signal/wait pairs for each "downstream" neighbor
    // We use NVSHMEM's signal operations for cross-GPU notification
    uint64_t *signal_recv;    // Signal I write TO (for my downstream neighbor to read)
    uint64_t *signal_send;    // Signal I read FROM (written by my upstream neighbor)
  
    int pe_left;              // My upstream neighbor in the ring
    int pe_right;             // My downstream neighbor in the ring
} ring_state_t;

ring_state_t *g_state;  // Allocated from symmetric heap
```

### Step 2: Initialize the Ring Topology

```c
// Ring order: GPU0 → GPU1 → GPU3 → GPU2 → GPU6 → GPU7 → GPU5 → GPU4 → GPU0
// (One of the three DGX-1 V100 rings)
//
// Key point: each step in this ring is a DIRECT NVLink hop!

int ring_order[8] = {0, 1, 3, 2, 6, 7, 5, 4};

void init_ring_state(int my_pe, int npes) {
    g_state = (ring_state_t*)nvshmem_malloc(sizeof(ring_state_t) * npes);
  
    ring_state_t *my = &g_state[my_pe];
  
    // Allocate data and forwarding buffers in symmetric heap
    my->data_buf = (float*)nvshmem_malloc(sizeof(float) * CHUNK_SIZE);
    my->fwd_buf  = (float*)nvshmem_malloc(sizeof(float) * CHUNK_SIZE);
  
    // Allocate signal space (one uint64 per PE pair, in symmetric heap)
    my->signal_recv = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
    my->signal_send = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
  
    // Find my position in the ring and set neighbors
    int my_pos = -1;
    for (int i = 0; i < npes; i++) {
        if (ring_order[i] == my_pe) { my_pos = i; break; }
    }
  
    my->pe_left  = ring_order[(my_pos - 1 + npes) % npes];
    my->pe_right = ring_order[(my_pos + 1) % npes];
  
    // IMPORTANT: The signal I write TO must be mapped to my downstream
    // neighbor's signal_send. This is what makes the forwarding chain work.
    // We use nvshmem_ptr() to get the remote address.
  
    *my->signal_recv = 0; // Initialize
    *my->signal_send = 0;
}
```

### Step 3: The Forwarding Kernel

```c
// ──────────────────────────────────────────────────────────
// Forwarding Kernel (runs on EVERY GPU in the ring)
// ──────────────────────────────────────────────────────────
//
// This kernel implements one "hop" of a multi-hop forwarding chain.
// It waits for data from the left neighbor, then forwards it to the right.
//
// The beauty: every GPU runs THE SAME kernel. The ring emerges from
// the neighbor configuration, not hard-coded logic.

__global__ void ring_forward_chunk(ring_state_t *state, int my_pe, int chunk_id) {
    ring_state_t *my = &state[my_pe];
  
    // ── Step 1: Wait for upstream neighbor's data ────────────
    //
    // The upstream neighbor (pe_left) will nvshmem_put data into my->fwd_buf,
    // then nvshmem_signal my signal_recv location.
    //
    // We spin-wait here. In production, use nvshmem_uint64_wait_until()
    // or cooperative groups for more efficient waiting.
  
    uint64_t *my_recv_signal = my->signal_recv;
    uint64_t expected = chunk_id;  // Each chunk has a unique signal value
  
    // This is a GPU-side wait. The SM spin-polls on NVLink.
    // nvshmem_uint64_wait_until(my_recv_signal, NVSHMEM_CMP_EQ, expected);
    // (Simplified: busy-wait in this example)
    while (nvshmem_uint64_g(my_recv_signal, my_pe) != expected) {
        // On modern hardware, use nanosecond sleeps or cooperative groups
        __nanosleep(100);
    }
  
    // ── Step 2: Forward the data downstream ─────────────────
    //
    // Now my->fwd_buf contains the data from upstream.
    // Forward it to my downstream neighbor's fwd_buf using direct NVLink put.
  
    ring_state_t *right_state = &state[my->pe_right];
  
    // Direct one-sided put: GPU-to-GPU, zero CPU involvement
    // If my_pe and pe_right share a direct NVLink, this runs at NVLink bandwidth
    nvshmem_float_put_nbi(
        right_state->fwd_buf,   // remote address (on pe_right's GPU)
        my->fwd_buf,            // local source
        CHUNK_SIZE,             // size in floats
        my->pe_right            // destination PE
    );
  
    // ── Step 3: Signal the downstream neighbor ──────────────
    //
    // After the put completes, update the downstream neighbor's
    // signal so they know data is ready.
  
    nvshmem_quiet();  // Ensure all puts are visible before signaling
  
    uint64_t next_signal = chunk_id + 1;  // Increment for next hop
    nvshmem_uint64_p(
        right_state->signal_recv,  // remote signal location
        next_signal,               // new value
        my->pe_right               // destination PE
    );
  
    nvshmem_quiet();
}
```

### Step 4: The Initiator Kernel (First GPU in the Chain)

```c
// This kernel runs on the GPU that OWNS the data and initiates the chain.
// It's a special case of the forwarding kernel — upstream is "self."

__global__ void ring_initiate_chunk(ring_state_t *state, int my_pe, int chunk_id) {
    ring_state_t *my = &state[my_pe];
  
    // ── Step 1: Prepare data in my fwd_buf ──────────────────
    // (Copy from my data buffer to the forwarding buffer)
  
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < CHUNK_SIZE; i += blockDim.x * gridDim.x) {
        my->fwd_buf[i] = my->data_buf[i];
    }
    __syncthreads();
  
    // ── Step 2: Put to right neighbor ───────────────────────
  
    ring_state_t *right_state = &state[my->pe_right];
  
    nvshmem_float_put_nbi(
        right_state->fwd_buf,
        my->fwd_buf,
        CHUNK_SIZE,
        my->pe_right
    );
  
    nvshmem_quiet();
  
    // ── Step 3: Signal right neighbor ───────────────────────
  
    nvshmem_uint64_p(right_state->signal_recv, chunk_id, my->pe_right);
    nvshmem_quiet();
}
```

### Step 5: Putting It All Together — The Full Pipeline

```c
void run_ring_forwarding_chain(int src_pe, int dst_pe, int npes) {
    // Determine which GPUs participate in the chain between src and dst
    // For a 2-hop: GPU1 → GPU2 → GPU3, all three launch kernels
  
    // Every GPU in the chain launches:
    //   - src_pe:       ring_initiate_chunk
    //   - intermediate: ring_forward_chunk
    //   - dst_pe:       ring_forward_chunk (receives and consumes)
  
    // This is done with per-GPU kernel launches:
  
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Launch on ALL GPUs in the ring simultaneously
        // (using CUDA streams + per-device scheduling)
      
        for (int pe = 0; pe < npes; pe++) {
            cudaSetDevice(pe);
          
            if (pe == src_pe) {
                ring_initiate_chunk<<<grid, block, 0, stream[pe]>>>(
                    g_state, pe, chunk);
            } else if (is_on_path(pe, src_pe, dst_pe)) {
                ring_forward_chunk<<<grid, block, 0, stream[pe]>>>(
                    g_state, pe, chunk);
            }
            // else: not on path, skip
        }
      
        // Synchronize across all GPUs between chunks
        for (int pe = 0; pe < npes; pe++) {
            cudaSetDevice(pe);
            cudaStreamSynchronize(stream[pe]);
        }
    }
}
```

---

## Part 3: Ring-Based AllReduce — NCCL's Algorithm, NVSHMEM's Primitives

Now let's implement the full ring AllReduce — this is what NCCL does internally:

```
Round 0:  GPU0→GPU1→GPU3→GPU2→GPU6→GPU7→GPU5→GPU4  (each stores partial reduction)
Round 1:  Same ring, but now each GPU forwards what it received in Round 0
...
Round N-1: After N-1 rounds, all GPUs have the full sum
```

```c
// ──────────────────────────────────────────────────────────
// NVSHMEM Ring AllReduce Kernel
//
// This implements NCCL's ring algorithm using NVSHMEM's one-sided
// put/get/signal primitives, running entirely on the GPU.
// ──────────────────────────────────────────────────────────

typedef struct {
    float *reduce_buf;     // My contribution + accumulation buffer
    float *recv_buf;       // Buffer for incoming data from left neighbor
    float *send_buf;       // Buffer for outgoing data to right neighbor
  
    uint64_t *ready_flag;  // Written by left neighbor when data is ready
  
    // Direct pointers to neighbors' buffers (obtained via nvshmem_ptr)
    float *right_recv_buf_ptr;  // Pointer to my right neighbor's recv_buf
    uint64_t *right_ready_ptr;  // Pointer to my right neighbor's ready_flag
  
    int pe_left, pe_right;
    int n_chunks;
} ring_allreduce_state_t;

__global__ void ring_allreduce_kernel(
    ring_allreduce_state_t *state, 
    int my_pe, 
    int round,
    int n_pes)
{
    ring_allreduce_state_t *my = &state[my_pe];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
    // ── Phase 1: Wait for data from left neighbor ──────────
  
    if (round > 0) {
        uint64_t expected = (uint64_t)(round); // Unique per round
      
        // Spin on NVLink: efficient GPU-side polling
        // This is a direct NVLink load from remote GPU's memory
        while (nvshmem_uint64_g(my->ready_flag, my_pe) != expected) {
            __nanosleep(1000);
        }
      
        __syncthreads();
    }
  
    // ── Phase 2: Reduce received data with local data ──────
  
    if (round > 0) {
        // my->recv_buf now contains neighbor's partial reduction from round-1
        for (int i = tid; i < my->n_chunks; i += blockDim.x * gridDim.x) {
            my->reduce_buf[i] += my->recv_buf[i];  // Accumulate
        }
        __syncthreads();
    }
  
    // ── Phase 3: Send my accumulated result to right neighbor ──
  
    if (round < n_pes - 1) {
        // Copy my reduction buffer to the send buffer
        for (int i = tid; i < my->n_chunks; i += blockDim.x * gridDim.x) {
            my->send_buf[i] = my->reduce_buf[i];
        }
        __syncthreads();
      
        // One-sided put to right neighbor's recv_buf via NVLink
        nvshmem_float_put_nbi(
            my->right_recv_buf_ptr,   // remote: neighbor's recv_buf
            my->send_buf,             // local: my send_buf
            my->n_chunks,             // count
            my->pe_right              // destination PE
        );
      
        nvshmem_quiet();  // Ensure put is complete before signaling
      
        // Signal right neighbor: data is ready for next round
        uint64_t next_expected = (uint64_t)(round + 1);
        nvshmem_uint64_p(
            my->right_ready_ptr,      // remote: neighbor's ready_flag
            next_expected,            // value
            my->pe_right              // destination PE
        );
      
        nvshmem_quiet();
    }
}

// ── Host-side orchestration ─────────────────────────────────

void ring_allreduce_host(int my_pe, int n_pes) {
    ring_allreduce_state_t *state = (ring_allreduce_state_t*)nvshmem_malloc(
        sizeof(ring_allreduce_state_t)
    );
  
    ring_allreduce_state_t *my = &state[my_pe];
  
    // Set up ring neighbors (using one of the three DGX rings)
    int ring[8] = {0, 1, 3, 2, 6, 7, 5, 4};
    int pos = find_pos(ring, my_pe, n_pes);
  
    my->pe_left  = ring[(pos - 1 + n_pes) % n_pes];
    my->pe_right = ring[(pos + 1) % n_pes];
  
    // Allocate buffers
    my->reduce_buf = (float*)nvshmem_malloc(my->n_chunks * sizeof(float));
    my->recv_buf   = (float*)nvshmem_malloc(my->n_chunks * sizeof(float));
    my->send_buf   = (float*)nvshmem_malloc(my->n_chunks * sizeof(float));
    my->ready_flag = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
  
    *my->ready_flag = 0;
  
    // Get direct pointers to neighbor buffers (the key NVSHMEM primitive!)
    //
    // nvshmem_ptr() returns a LOCAL pointer that maps to REMOTE GPU memory.
    // Reads/writes through this pointer go over NVLink transparently.
    // This is what makes the ring efficient: no CPU proxy, no kernel launch
    // overhead per transfer — just pointer dereferences.

    ring_allreduce_state_t *right_state = 
        (ring_allreduce_state_t*)nvshmem_ptr(state, my->pe_right);
  
    my->right_recv_buf_ptr = right_state->recv_buf;
    my->right_ready_ptr    = right_state->ready_flag;
  
    // Launch N-1 rounds of the ring algorithm
    for (int round = 0; round < n_pes; round++) {
        ring_allreduce_kernel<<<grid, block>>>(state, my_pe, round, n_pes);
    }
}
```

---

## Part 4: The Synchronization Challenge — Where NCCL Wins

The code above illustrates the concept, but there's a **real synchronization problem** that NCCL solves and you'd have to handle manually:

```c
// ──────────────────────────────────────────────────────────
// THE PROBLEM: How does GPU2 know when to forward?
//
// In the ring_allreduce_kernel above, we used a simple spin-loop
// on a remote flag. But there are subtle issues:
// ──────────────────────────────────────────────────────────

// ISSUE 1: Deadlock with CUDA stream ordering
//
// If GPU2 launches ring_forward_chunk BEFORE GPU1 has finished
// putting data, GPU2 spins forever. You need careful orchestration
// of kernel launch order across GPUs — NCCL handles this with
// internal CUDA events and stream management.

// ISSUE 2: Progress guarantees
//
// If GPU2's SMs are all busy spinning on the flag, there might be
// no SM available to process NVLink completions. NCCL uses dedicated
// proxy threads and CUDA cooperative groups to guarantee progress.

// ISSUE 3: Bandwidth efficiency of spin-waiting
//
// Spin-polling a remote memory location over NVLink consumes
// NVLink bandwidth. This is the classic polling vs. notification
// tradeoff. NCCL uses adaptive algorithms.

// ──────────────────────────────────────────────────────────
// NVSHMEM's solution: nvshmem_signal and block-style waits
// ──────────────────────────────────────────────────────────

// Better approach: use NVSHMEM signal operations
__global__ void ring_forward_chunk_v2(
    ring_state_t *state, int my_pe, int chunk_id) 
{
    ring_state_t *my = &state[my_pe];
  
    // nvshmem_uint64_wait_until is a more efficient wait primitive
    // that can be implemented with reduced NVLink traffic
    nvshmem_uint64_wait_until(
        my->signal_recv,           // location to poll
        NVSHMEM_CMP_EQ,            // comparison
        (uint64_t)chunk_id,        // expected value
    );
  
    // Forward data...
    ring_state_t *right = &state[my->pe_right];
    nvshmem_float_put_nbi(right->fwd_buf, my->fwd_buf, CHUNK_SIZE, my->pe_right);
    nvshmem_fence();  // Ensure ordering
  
    // Signal with atomic update
    nvshmem_uint64_atomic_add(right->signal_recv, 1, my->pe_right);
    nvshmem_quiet();
}
```

---

## The Big Picture: What NCCL Gives You For Free

Here's a side-by-side of what you'd have to implement manually vs. what NCCL provides:

```
┌──────────────────────────────────────────────────────────────────┐
│                   NCCL vs DIY NVSHMEM Ring                        │
├────────────────────────────────┬─────────────────────────────────┤
│         NCCL (1 line)          │     NVSHMEM DIY (~200+ lines)    │
├────────────────────────────────┼─────────────────────────────────┤
│ ncclAllReduce(send, recv,     │ • Topology discovery code        │
│   count, type, op, comm,     │ • Ring construction (3 parallel  │
│   stream);                     │   rings on DGX-1)               │
│                                │ • Buffer management             │
│ ✓ Auto topology discovery     │ • Chunking / pipelining logic   │
│ ✓ 3 parallel rings on DGX-1  │ • Kernel launch orchestration    │
│ ✓ Ring + Tree hybrid for      │ • Progress guarantees           │
│   large messages              │ • Deadlock avoidance             │
│ ✓ Adaptive algorithm selection│ • Signal protocol design        │
│ ✓ NVSwitch multicast (A100+) │ • Multi-rail bandwidth balancing │
│ ✓ IB/RoCE multi-rail         │                                  │
│ ✓ Automatic chunking/pipelining│                                 │
│ ✓ CUDA graph integration      │                                  │
│ ✓ Tested on every topology    │                                  │
└────────────────────────────────┴─────────────────────────────────┘
```

---

## When You'd Actually Do This

So when would you write NVSHMEM forwarding code? Three scenarios:

### 1. In-Kernel Collectives Interleaved with Computation

NCCL collectives are launched from the **host**. If you have a long-running GPU kernel that needs mid-kernel reduction, you can't call NCCL. With NVSHMEM you can:

```c
__global__ void iterative_solver(float *x, float *b, ring_allreduce_state_t *ring, int my_pe) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ... compute local residual ...
      
        // Mid-kernel AllReduce — NO KERNEL EXIT, NO CPU INVOLVEMENT!
        ring_allreduce_kernel(ring, my_pe, 0, N_PES);  // Round 0
        ring_allreduce_kernel(ring, my_pe, 1, N_PES);  // Round 1
        // ... N-1 rounds ...
      
        // Now all GPUs have the reduced residual — continue iterating
        // without ever returning to the CPU!
        // ... update x ...
    }
}
```

This eliminates $N_{\text{iterations}} \times N_{\text{rounds}}$ kernel launch overheads.

### 2. Custom Communication Patterns

When your communication pattern doesn't fit any standard collective:

```c
// Example: irregular halo exchange for unstructured mesh
__global__ void unstructured_halo_exchange(...) {
    int elem_id = threadIdx.x + blockIdx.x * blockDim.x;
    int neighbor_elem = adjacency[elem_id];
    int neighbor_pe  = partition[neighbor_elem];
  
    // Direct put to whichever GPU owns the neighbor element
    // Could be any GPU — not just ring neighbors
    nvshmem_float_put_nbi(
        remote_buf + neighbor_offset[neighbor_elem],
        local_buf + elem_id,
        1,
        neighbor_pe  // Could be 1-hop or 2-hop away
    );
}
```

For multi-hop neighbors here, you'd use the forwarding pattern shown above.

### 3. Overlapping Communication with Computation

NVSHMEM lets you pipeline compute and communication at the warp level:

```c
__global__ void pipelined_stencil(float *grid, ring_state_t *ring, int my_pe) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
  
    // Issue non-blocking put to send halo to right neighbor
    nvshmem_float_put_nbi(right_halo, my_edge, HALO_SIZE, my->pe_right);
  
    // Compute interior points WHILE halo is in flight
    compute_interior(grid, x);
  
    // Now wait for left neighbor's halo
    nvshmem_uint64_wait_until(my->signal_recv, NVSHMEM_CMP_EQ, expected_val);
  
    // Compute boundary using received halo
    compute_boundary(grid, x, my->recv_buf);
}
```

---

## Bottom Line

> **NCCL is the expertly engineered, topology-aware, production-hardened ring/tree implementation. NVSHMEM gives you the one-sided primitives to build your own rings, trees, or arbitrary forwarding graphs from within CUDA kernels.**

The beauty of NVSHMEM is exactly as you described: you *can* write GPU-based forwarding code to exploit multi-hop NVLink paths. Whether you *should* depends on whether you need in-kernel communication (use NVSHMEM), or can tolerate host-launched collectives (use NCCL and save months of debugging deadlocks).