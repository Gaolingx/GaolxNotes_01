# 现代MoE解决令牌丢弃与填充问题

### 🧑‍💻 User

Prior MoE implementations either dropped tokens from computation or waste computation and memory on padding. What are the differences between this and modern moe models?

---

### 🤖 Assistant

The historical bottleneck of Mixture of Experts (MoE) architectures—such as the Switch Transformer—was the reliance on **fixed expert capacities**. Because the model couldn't predict exactly how many tokens would be routed to each expert, it allocated a fixed memory and compute capacity per expert. 

If too many tokens were routed to an expert, the excess tokens were **dropped** (passed to the next layer without transformation via a residual connection). If too few tokens were routed, the remaining capacity was filled with **padding** (wasting FLOPs and memory).

Modern MoE models (like Mixtral 8x7B, Grok-1, and Google's latest architectures) solve this through a combination of **algorithmic innovations** and **system-level kernel optimizations**. 

Here are the primary differences that allow modern MoE models to avoid dropping tokens and wasting computation:

### 1. System-Level Innovations: Block-Sparse Operations (MegaBlocks)
Modern open-weights MoEs heavily rely on frameworks like **MegaBlocks** (developed alongside models like Databricks' DBRX). Instead of treating experts as separate dense matrix multiplications with padded inputs, MegaBlocks formulates the MoE computation as a **block-sparse matrix multiplication**.

*   **Dynamic Computation:** The framework dynamically allocates computation based on the exact number of tokens sent to each expert. 
*   **No Padding or Dropping:** By sorting tokens and grouping them continuously in memory, custom GPU kernels process the jagged (uneven) lists of tokens without requiring padding arrays. No token is ever dropped because there is no arbitrary "capacity" limit enforced by the matrix shape.

### 2. Algorithmic Innovations: Expert Choice Routing
Google introduced **Expert Choice Routing** to solve the load-balancing problem algorithmically. 
*   **Legacy (Token Choice):** Each token chooses its top-$k$ experts. This leads to unbalanced loads.
*   **Modern (Expert Choice):** Each expert chooses its top-$k$ tokens from the sequence. The capacity $C$ for each expert is fixed, but instead of tokens fighting for space, experts pull the most relevant tokens. This guarantees perfect load balancing across experts, eliminating padding. Furthermore, a token might be selected by zero, one, or multiple experts, removing the need for tokens to be dropped due to artificial capacity limits.

### 3. Custom Fused Kernels (Triton/CUDA)
Modern inference engines (like `vLLM` or `TensorRT-LLM`) use highly optimized, fused kernels written in Triton or CUDA to handle routing on the fly. They use a technique called **Token Sorting**.

Instead of creating a dense padded tensor of shape $B \times S \times E \times D$, these kernels:
1. Compute routing probabilities.
2. Sort the tokens based on their assigned expert.
3. Keep track of the offset (start and end index) for each expert.
4. Launch a single grouped GEMM (General Matrix Multiply) that processes all experts simultaneously without a single padding token.

---

### Detailed Breakdowns

<details>
<summary><strong>Expand: The Math Behind the Legacy "Capacity" Problem</strong></summary>

In legacy systems, expert capacity $C$ was defined as:
$$C = \frac{T \times k}{E} \times c$$
Where:
*   $T$ is the total number of tokens in the batch.
*   $k$ is the number of experts each token selects (usually 1 or 2).
*   $E$ is the total number of experts.
*   $c$ is the capacity factor (e.g., $1.25$ to allow for a $25\%$ imbalance).

If an expert received $N$ tokens:
*   If $N > C$: The system drops $N - C$ tokens.
*   If $N < C$: The system adds $C - N$ padding tokens, wasting compute.

Modern systems discard the capacity factor $c$ entirely during inference, relying on dynamic shapes.
</details>

<details>
<summary><strong>Expand: Conceptual Code for Modern Token Sorting</strong></summary>

Modern frameworks bypass padding by flattening and sorting tokens. Instead of padded tensors, they use continuous arrays and offset trackers.

```python
# Conceptual representation of modern MoE unpadded routing
def modern_moe_forward(tokens, routing_weights, experts):
    # tokens shape: [batch_size * seq_len, dim]
  
    # 1. Get expert assignments (e.g., top-1)
    expert_indices = get_top_k(routing_weights, k=1)
  
    # 2. Sort tokens by expert to group them contiguously
    sorted_indices = torch.argsort(expert_indices)
    sorted_tokens = tokens[sorted_indices]
  
    # 3. Calculate how many tokens go to each expert (no padding!)
    expert_counts = torch.bincount(expert_indices, minlength=len(experts))
  
    # 4. Process chunks without padding using grouped/block-sparse GEMM
    outputs = []
    start_idx = 0
    for i, expert in enumerate(experts):
        count = expert_counts[i]
        if count > 0:
            # Process exactly 'count' tokens, no dropping, no padding
            expert_in = sorted_tokens[start_idx : start_idx + count]
            outputs.append(expert(expert_in))
            start_idx += count
          
    # 5. Unsort back to original sequence order
    return unsort_tokens(outputs, sorted_indices)
```
</details>

<details>
<summary><strong>Expand: Soft MoE (Fully Differentiable Routing)</strong></summary>

Another modern alternative to solve dropping is **Soft MoE**. Instead of hard-routing tokens to specific experts (which creates discrete, unbalanced groups), Soft MoE computes a weighted blend of tokens to send to each expert, and a weighted blend of expert outputs to return to the sequence. 

Because every expert processes a mathematical combination of all tokens, the matrix dimensions are perfectly static. There is zero padding, zero dropping, and the routing mechanism is fully differentiable, though it comes at the cost of dense computation at the routing layer.
</details>

---

### Summary Comparison

| Feature | Legacy MoE (e.g., Switch Transformer) | Modern MoE (e.g., Mixtral, DBRX) |
| :--- | :--- | :--- |
| **Routing Mechanism** | Token Choice (Tokens pick top-$k$ experts) | Token Choice (with dynamic sorting) or Expert Choice |
| **Memory Allocation** | Fixed capacity per expert | Dynamic allocation (Block-Sparse) |
| **Handling Overflows** | Drops excess tokens | Processes all tokens (no dropping) |
| **Handling Underflows** | Pads with zeros (wastes FLOPs) | Exact compute allocation (no padding) |
| **Software Backend** | Standard PyTorch/XLA padded tensors | Custom Triton/CUDA Kernels (MegaBlocks, Grouped GEMM) |

---

### 🧑‍💻 User

How MegaBlocks allows for a viable number of tokens per expert?

---

### 🤖 Assistant

To understand how **MegaBlocks** makes any number of tokens per expert computationally "viable," we have to understand why uneven token distributions were a problem for GPUs in the first place.

GPUs are designed to process perfectly rectangular, dense blocks of data. If Expert A receives 10 tokens and Expert B receives 1,000 tokens, standard GPU operations force you to either process them sequentially (which leaves the GPU mostly idle during Expert A's turn) or pad Expert A with 990 fake tokens to make a rectangular matrix (which wastes memory and compute).

MegaBlocks makes variable token counts viable by abandoning rectangular matrices altogether. It translates the MoE routing problem into a **Block-Sparse Matrix Multiplication (SpMM)** problem. 

Here is exactly how MegaBlocks achieves this under the hood:

### 1. Embracing "Jagged" Tensors
Instead of enforcing a fixed capacity $C$, MegaBlocks allows the routing algorithm to assign tokens naturally. If $N_i$ is the number of tokens sent to expert $i$, the total number of tokens processed is exactly $\sum_{i=1}^{E} N_i$. 

This creates a "jagged" (or ragged) tensor, where the lists of tokens for each expert are of different lengths. MegaBlocks flattens these jagged lists into a single, continuous 1D array in memory, keeping track of where each expert's tokens start and end.

### 2. Block-CSR Format
To make GPUs understand this jagged array, MegaBlocks represents the MoE layer as a sparse matrix using a format called **Block-CSR (Compressed Sparse Row)**. 
*   In this matrix, the "blocks" are the tokens assigned to specific experts.
*   The empty space in the matrix represents the experts that tokens were *not* routed to.
*   By formulating the problem this way, MegaBlocks can use sparse matrix mathematics to multiply the inputs by the expert weights.

### 3. High-Performance Block-Sparse Kernels
The true magic of MegaBlocks is its custom GPU kernels (built using `CUTLASS`). Instead of launching a separate matrix multiplication for each expert (which incurs massive overhead), MegaBlocks launches a **single GPU kernel** that performs a Block-Sparse GEMM (General Matrix Multiply).

This kernel is smart enough to look at the Block-CSR metadata, assign different thread blocks on the GPU to different experts, and process the exact number of tokens each expert has. Because it all happens in one kernel, the GPU remains fully saturated and highly utilized, even if the tokens are wildly imbalanced.

---

### Deep Dive into MegaBlocks Mechanics

<details>
<summary><strong>Expand: The Block-Sparse Matrix Math</strong></summary>

Let $X$ be our input token matrix of shape $T \times D$, where $T$ is the number of tokens and $D$ is the hidden dimension. Let $W$ be a stacked tensor of all expert weights.

In a dense MoE, we would try to compute $Y = X \times W$, but $W$ is massive, and most tokens only need a tiny slice of $W$.

MegaBlocks defines a sparse topology matrix $A$. If token $t$ is routed to expert $e$, there is a block of 1s in $A$; otherwise, it is 0. 
MegaBlocks computes:
$$ Y = \text{SpMM}(A, X, W) $$

Because $A$ is block-sparse, the custom MegaBlocks kernel only performs the multiplication $X_t \times W_e$ where $A$ dictates. It skips all the zeros entirely. There is no padding to compute, and no capacity limit $C$ forcing tokens to be dropped.
</details>

<details>
<summary><strong>Expand: Grouped GEMM vs. Batched GEMM</strong></summary>

*   **Batched GEMM (Legacy MoE):** Requires all matrices in the batch to be the exact same size. This is why legacy MoEs forced padding: so they could use fast batched operations like `torch.bmm()`.
*   **Grouped GEMM / Block-Sparse (MegaBlocks):** Allows passing a list of matrices of *different sizes* to the GPU, along with their dimensions. The GPU scheduler dynamically assigns Streaming Multiprocessors (SMs) to the different multiplications based on their size. If Expert B has $100\times$ more tokens than Expert A, the hardware simply assigns $100\times$ more SMs to Expert B's math, finishing them at roughly the same time.
</details>

<details>
<summary><strong>Expand: Conceptual MegaBlocks Code Flow</strong></summary>

While MegaBlocks uses complex C++/CUDA under the hood, conceptually, it operates like this in Python/PyTorch:

```python
import megablocks.ops as ops

def megablocks_moe_forward(x, router_probs, expert_weights):
    # x: [batch * seq_len, dim]
  
    # 1. Get routing indices and topology
    # topology contains the Block-CSR metadata (row offsets, column indices)
    expert_indices, topology = compute_topology(router_probs)
  
    # 2. Permute/Sort tokens into a continuous jagged array
    # No padding is added here. Size remains [batch * seq_len, dim]
    x_sorted = ops.sort(x, expert_indices)
  
    # 3. Block-Sparse Matrix Multiplication (The MegaBlocks Kernel)
    # This single kernel processes all experts based on the topology
    y_sorted = ops.sddmm(topology, x_sorted, expert_weights)
  
    # 4. Unsort back to original token order
    y = ops.unsort(y_sorted, expert_indices)
  
    return y
```
</details>

### Summary: Why it makes uneven routing "Viable"

MegaBlocks decouples **algorithmic routing** from **hardware execution**. 

Before MegaBlocks, ML researchers had to cripple their routing algorithms (forcing tokens to be dropped or routing to sub-optimal experts just to balance the load) to keep the GPU happy. MegaBlocks built a software bridge that allows the GPU to efficiently process unbalanced loads. Therefore, the routing algorithm is free to assign any number of tokens to any expert—making highly specialized, unbalanced experts practically viable for the first time.