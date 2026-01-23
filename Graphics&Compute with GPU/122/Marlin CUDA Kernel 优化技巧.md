# Marlin CUDA Kernel 优化技巧

### 🧑‍💻 User

[ist-daslab-marlin-8a5edab282632443.txt](./ist-daslab-marlin-8a5edab282632443.txt)

能否帮我解析下该cuda kernel（marlin_cuda_kernel.cu）的计算过程、Tile划分、计算调度、Tile计算（IO处理、数据重排、MatMul、unpack & 反量化）、Reduce、写回操作，最后总结一下使用了哪些高性能CUDA kernel常用的技巧，如访存合并，异步访存，pipeline，Bank Conflict Free等。

---

### 🤖 Assistant

这份代码（`marlin_cuda_kernel.cu`）是一个针对 **FP16 x INT4** 矩阵乘法（MatMul）高度优化的 CUDA Kernel，专门用于大语言模型（LLM）推理加速。它的核心目标是在 Ampere (SM80) 及更新架构的 GPU 上，通过最大限度利用 L2 Cache、Shared Memory 和 Tensor Cores 来克服 Weight-Only Quantization 场景下的访存瓶颈。

下面我将分模块解析其计算过程，并总结其使用的高性能技巧。

---

### 一、 Marlin Kernel 计算过程解析

Kernel 的入口函数是 `Marlin`，这是一个模板函数，通过模板参数定义了 Tile 的大小（`thread_m_blocks` 等）和 Pipeline 的级数（`stages`）。

#### 1. Tile 划分与问题定义 (Grid & Block Partitioning)
代码采用了一种特殊的 **"Striped" Partitioning（条纹划分）** 策略，而不是传统的简单的 Block 映射。

*   **问题规模：** 输入 $A (m \times k)$，权重 $B (k \times n)$，输出 $C (m \times n)$。
*   **Grid 映射：** 
    *   传统的 GEMM 通常将 $m$ 和 $n$ 划分给 grid 的 x, y 维度。
    *   Marlin 计算每个 SM（Block）需要处理的 "Tiles" 总数 (`iters`)。
    *   代码逻辑：`int slice_row = ...`, `int slice_col = ...`。
    *   **Striped 策略：** 每个 Block 负责 $B$ 矩阵的一条“条纹”（Stripe）。这个条纹可能跨越多个列切片（Column Slice）。这种方式可以在 $m$ 较小或形状不规则时，保证所有 SM 的负载均衡，并最大化 $A$ 矩阵在 L2 Cache 中的复用。
*   **Block 内部：** 每个 Thread Block 包含 256 个线程 (`THREADS=256`)，这是为了配合 8 个 Warp 的调度。

#### 2. 计算调度 (Scheduling & Loop)
核心是一个基于 **Pipeline** 的主循环。

*   **Pipeline 启动：** `start_pipes()` 函数预先加载前几级数据（Global $\to$ Shared $\to$ Register）。
*   **主循环：** `while (slice_iters)`。
    *   循环内部展开了 `stages` 次（通常为 4 次）。
    *   每一级 Pipeline 中，交错执行 **计算** (`matmul`) 和 **数据预取** (`fetch_to_registers`, `fetch_to_shared`)。

#### 3. Tile 计算细节

这是 Kernel 最核心的部分，包含从内存加载到 Tensor Core 计算的全流程。

##### A. IO 处理 (Global $\to$ Shared)
*   **异步复制 (Async Copy):** 使用 `cp_async4_pred` (对应 PTX `cp.async.cg.shared.global`)。这允许数据从 Global Memory 搬运到 Shared Memory 的过程中不阻塞 CUDA Core 的计算线程。
*   **L2 Cache 策略:**
    *   **矩阵 A (Activation):** 正常加载，因为 A 会被多个 $B$ 的 Tile 复用，需要驻留在 L2 Cache。
    *   **矩阵 B (Quantized Weights):** 使用 `cp_async4_stream`，并在 PTX 中指定了 `.L2::cache_hint` 为 `evict_first`。因为权重 $B$ 在推理时通常只读一次（对于 batch size = 1），读完即丢，避免污染 L2 Cache，把空间留给 $A$ 和 $C$。

##### B. 数据重排 (Data Rearrangement / Swizzling)
为了让后续的 `ldmatrix` 指令无 Bank Conflict 地读取 Shared Memory，数据在写入 Shared Memory 时进行了地址重映射。
*   **代码：** `transform_a` lambda 函数。
*   **逻辑：** 使用 XOR 操作 (`row ^ col` 风格) 计算偏移量。这是为了适配 NVIDIA GPU Shared Memory 的 Bank 映射规则，确保同一个 Warp 的线程访问不同的 Bank。

##### C. Shared $\to$ Registers (LDSM)
*   **指令：** 使用 `ldsm4` (对应 PTX `ldmatrix.sync.aligned.m8n8.x4.shared.b16`)。
*   **作用：** 一次性将 Shared Memory 中的数据直接加载到 Tensor Core 所需的寄存器布局中（`FragA`），绕过普通的寄存器文件读写，提高带宽并降低指令数。

##### D. Unpack & 反量化 (Dequantization)
权重 $B$ 是 INT4 格式，但 Tensor Core 计算通常需要 FP16。Marlin 在寄存器层面进行极其高效的“即时反量化”。
*   **存储格式：** $B$ 的布局经过离线重排（Marlin Format），使得加载后的 `int4` 数据刚好对应 Tensor Core 的 Lane。
*   **SIMD 位操作：**
    *   **LOP3 指令：** 代码使用 `lop3` (3-input logic operation) 提取 INT4 数据。这是 NVIDIA GPU 极快的位操作指令。
    *   **转换：** `dequant(int q)` 函数将 packed int 转换为 FP16 (`half2`)。
    *   **公式：** $W_{fp16} = (W_{int4} - ZeroPoint) \times Scale$。代码利用了 `__hsub2` 和 `__hfma2` 指令高效完成。

##### E. MatMul (Tensor Core)
*   **指令：** `mma` (对应 PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`)。
*   **精度：** 输入 FP16，累加器（Accumulator）使用 FP32 (`FragC`) 以保证精度。
*   **循环展开：** `matmul` lambda 中包含 `#pragma unroll`，强制编译器展开循环，减少循环开销并允许编译器进行指令重排以掩盖延迟。

#### 4. Reduce (归约)

由于一个 Block 可能只计算了 $K$ 维度的一部分（Split-K），或者 Block 内部使用了多个 Warp 累加，需要两级归约。

*   **Block 内归约 (Shared Memory Reduce):**
    *   `thread_block_reduce` 函数。
    *   不同 Warp 计算的部分和写入 Shared Memory，然后同步，再累加。
*   **跨 Block 归约 (Global Reduce):**
    *   `global_reduce` 函数。
    *   **锁机制：** 使用 Global Memory 中的 `locks` 数组实现自旋锁 (`barrier_acquire` / `barrier_release`)。
    *   **直接累加到输出：** 拿到锁的 Block 读取 Global Memory 中的 $C$（作为临时累加缓冲区），加上自己的结果，再写回。这里为了节省带宽，中间结果会暂时转为 FP16 存入 L2 Cache（因为 L2 比 Global Memory 快得多，且该位置会被迅速再次读取）。

#### 5. 写回操作 (Write Back)
*   **代码：** `write_result`。
*   当所有 $K$ 维度的计算完成后，最后一个持有锁的 Block 负责最终的写回。
*   将 FP32 累加器转回 FP16。
*   如果使用的是 per-column scaling (非 group)，在这里应用最终的 scale。
*   写入 Global Memory $C$。

---

### 二、 高性能 CUDA Kernel 技巧总结

Marlin 几乎使用了所有现代 CUDA 编程的高级优化技巧，是教科书级别的优化案例：

1.  **异步数据拷贝 (Asynchronous Data Copy / `cp.async`):**
    *   **作用：** 将 Global Memory 到 Shared Memory 的数据搬运指令发出后，无需等待完成即可执行其他计算指令。
    *   **应用：** 完美掩盖了 Global Memory 的高延迟（Latency Hiding）。

2.  **多级流水线 (Software Pipelining / Multi-stage Buffering):**
    *   **作用：** 代码定义了 `STAGES=4`。Shared Memory 被划分为 4 个 buffer。
    *   **应用：** 当 GPU 计算第 $N$ 级 Tile 时，DMA 引擎正在加载第 $N+1, N+2...$ 级的数据。时刻保持 Tensor Core 满载。

3.  **Shared Memory 无冲突访问 (Bank Conflict Free via XOR Swizzling):**
    *   **作用：** 避免多个线程同时访问同一个 Shared Memory Bank（导致串行化）。
    *   **应用：** `transform_a` 函数计算了一个基于 XOR 的地址重排，专门配合 `ldmatrix` 指令的访存模式，实现 100% 吞吐。

4.  **LDSM (Load Matrix) 指令:**
    *   **作用：** 专用硬件路径，从 Shared Memory 直接加载数据到 Tensor Core 寄存器，不经过通用寄存器文件。
    *   **应用：** 减少寄存器压力，提高数据吞吐。

5.  **寄存器级数据重用与 Double Buffering:**
    *   **应用：** `FragA frag_a[2][thread_m_blocks]`。在寄存器层面也做了双缓冲，计算当前寄存器数据的同时，加载下一组数据到备用寄存器。

6.  **自定义数据布局 (Custom Layout / Pre-shuffling):**
    *   **应用：** 权重 $B$ 和 Scale $S$ 在 CPU 端预处理时被重排（Marlin Format）。这使得 Kernel 内部读取时，不需要做复杂的 Shuffle 操作，直接加载即可满足 Tensor Core 的 Lane 需求。

7.  **L2 Cache Residency Control (Cache Eviction Policy):**
    *   **技巧：** `cp.async.cg.shared.global.L2::cache_hint` 配合 `evict_first`。
    *   **应用：** 显式告诉 GPU 权重数据是一次性的，读完不仅不要保留，还应该优先通过置换算法踢出，从而保护激活值 $A$（需要高频复用）不被挤出 L2 Cache。

8.  **高效位操作 (LOP3 & Fast Math):**
    *   **应用：** 使用 `lop3` 指令在一条指令周期内完成复杂的位掩码和移位操作，用于快速从 `int4` 包中提取权重。

9.  **Striped Partitioning (条纹划分):**
    *   **应用：** 解决大模型推理中常见的“长瘦”矩阵或不规则形状导致的 SM 负载不均（Tail Effect）问题。

10. **编译期常量展开 (`#pragma unroll` & Templates):**
    *   **应用：** 大量的循环和索引计算依赖模板参数（如 `thread_k`）。编译器在编译时就能计算出具体的内存偏移量（Immediate Offsets），减少了运行时的整数指令计算开销。

通过这些技术的组合，Marlin 能够在 Batch Size 较小（16-32）的情况下，依然让 GPU 的 Tensor Core 接近打满，实现了 4-bit 量化下接近理想的 4 倍加速。

---

### 🧑‍💻 User

随着GPU引入越来越多DSA的特性，为了达到最佳的SOL性能，需要引入更复杂的pipeline来尽可能隐藏延迟和打满计算单元的吞吐。marlin kernel都采取了哪些方式来实现这些目标？

---

### 🤖 Assistant

针对现代 GPU（特别是 NVIDIA Ampere 及之后的架构）的 DSA（Domain Specific Architecture）特性，Marlin Kernel 构建了一个深度的、多级流水线系统，旨在将内存延迟（Global Memory）、共享内存延迟（Shared Memory）以及计算指令延迟（Instruction Latency）完全“掩盖”在 Tensor Core 的强大算力之下。

为了达到最佳的 SOL（Speed Of Light）性能，Marlin 在代码层面采取了以下几种核心方式来构建 Pipeline 和隐藏延迟：

### 1. 全局内存到共享内存的流水线 (Async Copy Pipeline)

这是最外层的 Pipeline，用于隐藏最大的瓶颈——显存（HBM/GDDR）访问延迟。

*   **DSA 特性利用：** 使用 `cp.async` 指令（CUDA 的 Async Copy Engine）。这是一个专用硬件单元（DMA），负责将数据从 Global Memory 搬运到 Shared Memory，**无需**占用 CUDA Core 的计算资源。
*   **多级缓冲 (Multi-Stage Buffering)：**
    *   代码中定义了 `const int STAGES = 4;`。这意味着 Shared Memory 被划分成了 4 个缓冲区（Circular Buffer）。
    *   **Pipeline 逻辑：**
        *   当 Tensor Core 正在计算 **第 N 级** 数据时。
        *   寄存器正在加载 **第 N+1 级** 数据。
        *   DMA 引擎（`cp.async`）正在从显存预取 **第 N+2 级** 甚至 **第 N+3 级** 的数据。
*   **代码体现：**
    *   `fetch_to_shared(...)`: 发起异步拷贝指令。
    *   `wait_for_stage()`: 使用 `cp_async_wait<stages - 2>()`。这里只等待最旧的一级数据就绪，从而让后续的拷贝请求继续在后台执行。

### 2. 共享内存到寄存器的流水线 (LDSM & Double Buffering)

这是中间层 Pipeline，用于隐藏 Shared Memory 的读取延迟和 Bank 冲突风险。

*   **DSA 特性利用：** 使用 `ldmatrix` (Load Matrix) 指令。这是 Tensor Core 的配套指令，能够绕过常规的寄存器文件读写路径，直接将 Shared Memory 数据以矩阵块的形式加载到 Warp 寄存器中。
*   **寄存器双缓冲 (Register Double Buffering)：**
    *   代码中声明了 `FragA frag_a[2][thread_m_blocks];` 和 `I4 frag_b_quant[2];`。这里的 `[2]` 就是为了双缓冲。
    *   **逻辑：** 在计算当前迭代 `k` 的同时，利用空闲的 Load/Store 单元预取下一次迭代 `k+1` 的数据到备用寄存器中。
*   **代码体现：**
    *   在主循环 `while` 内部的 `fetch_to_registers(k + 1, ...)` 调用位于 `matmul(k)` 之前。即：**先发起下一次的读，再进行当前的算**。

### 3. 计算指令与整数指令的指令级并发 (Instruction Level Parallelism)

Marlin 是量化 Kernel，必须在计算前进行**反量化**。这是一个额外的开销，如果处理不好会阻塞 Tensor Core。

*   **流水线设计：** Marlin 利用 GPU 上 **INT 单元（整数计算）** 和 **Tensor Core 单元** 可以并行发射指令的特性。
*   **操作交错 (Interleaving)：**
    *   **反量化（INT Pipe）：** 使用 `lop3` (位操作) 和 `dequant` (整数转浮点) 处理权重 $B$。
    *   **矩阵乘（Tensor Pipe）：** 使用 `mma` 指令处理 $A \times B$。
    *   **隐藏策略：** 当 `mma` 指令在 Tensor Core 中执行时（通常需要多个周期），CUDA Core 的 ALU 单元正好利用这段时间去计算下一组数据的地址偏移、执行 `lop3` 提取下一组 INT4 权重。
*   **代码体现：**
    *   在 `matmul` 函数中，反量化操作 (`dequant`, `scale`) 与 `mma` 紧密交织。循环被 `#pragma unroll` 展开后，编译器会自动进行指令重排（Instruction Reordering），将这些不同类型的指令混合在一起以填满指令流水线。

### 4. L2 Cache 驻留策略 (L2 Persistence Pipeline)

为了最大化吞吐，Marlin 对 L2 Cache 进行了精细的控制，这可以看作是一种隐式的“数据生存周期管理”流水线。

*   **DSA 特性利用：** `cp.async` 的 Cache Hint 功能。
*   **策略：**
    *   **矩阵 A (Activations)：** 需要被反复复用（因为一个 A tile 要和所有的 B tile 相乘）。Marlin 让 A 自然驻留在 L2 中。
    *   **矩阵 B (Weights)：** 在推理（Batch Size=1）场景下，权重只会被读取一次。
    *   **优化：** 代码中对 B 的加载使用了 `createpolicy.fractional.L2::evict_first`。这告诉 GPU 硬件：“这个数据读完就立刻丢掉，不要占用 L2 空间”。
*   **效果：** 这种策略防止了巨大的权重矩阵将较小的激活矩阵挤出 L2 Cache，确保了 A 矩阵的加载延迟极低（接近 L1 速度）。

### 5. 编译期地址计算流水线 (Static Offset Calculation)

为了不让运行时的地址计算（Pointer Arithmetic）打断计算流水线，Marlin 极度依赖编译期计算。

*   **策略：** 大量使用 `constexpr` 和 `#pragma unroll`。
*   **效果：**
    *   所有 Shared Memory 的读写偏移量 (`a_sh_wr_trans`, `a_sh_rd_trans`) 在编译后都变成了**立即数 (Immediate values)**。
    *   这意味着 GPU 运行时不需要执行 `ADD`, `MUL` 等指令来计算地址，不仅节省了指令发射槽位，也减少了寄存器占用，让更多的寄存器可以用于掩盖延迟。

### 总结图示

Marlin 的 Pipeline 可以抽象为以下并行层级：

```text
[时间轴 Time] -------------------------------------------------------------------->

Stage 1 (DMA):      [ Global -> Shared (Batch N+2) | cp.async ]
                         | (无需 CUDA Core 参与)
Stage 2 (LDSM):          [ Shared -> Reg (Batch N+1) | ldmatrix ]
                              | (占用 Load/Store 单元)
Stage 3 (ALU):               [ Dequantize (Batch N) | lop3, cvt ]
                                   | (占用 INT/FP32 单元)
Stage 4 (Tensor):                  [ MatMul (Batch N) | mma.sync ]
                                        (占用 Tensor Core)
```

**Marlin 成功的核心在于：它确保了上述四个 Stage 在同一时刻都在工作。** 只要 `mma.sync`（最耗时的部分）从未断档，GPU 就达到了理论上的 SOL 性能。