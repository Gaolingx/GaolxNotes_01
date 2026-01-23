# Marlin提升TensorCore利用率

### 🧑‍💻 User

[ist-daslab-marlin-8a5edab282632443.txt](./ist-daslab-marlin-8a5edab282632443.txt)

marlin gptq kernel是如何解决AutoGPTQ在大bs下tensorcore利用率低问题的？

---

### 🤖 Assistant

根据提供的文件 `ist-daslab-marlin-8a5edab282632443.txt`（包含 README 和 CUDA 源码），Marlin kernel 通过以下几个核心技术方案解决了 AutoGPTQ 在大 Batch Size（BS）下 Tensor Core 利用率低的问题。

AutoGPTQ 等现有内核在大 BS 下性能下降的主要原因是：随着计算量的增加，未能有效平衡**全局内存（Global Memory）带宽**、**L2 缓存争用**以及**计算流水线**的延迟，导致 Tensor Core 经常处于等待数据的状态。

Marlin 通过以下机制实现了高达 16-32 batch size 的近乎理想的加速比（4x）：

### 1. 优化的 L2 缓存管理策略 (L2 Cache Management)
这是 Marlin 最关键的优化之一。在大 BS 下，激活值（Activations, Matrix A）的数据量变大，同时权重的读取量也非常大。如果管理不当，权重的加载会将需要频繁复用的激活值从 L2 缓存中挤出（Eviction），导致严重的内存抖动和带宽浪费。

*   **激活值常驻 L2：** Marlin 组织计算的方式确保激活值（Activations）几乎总是从 L2 缓存中获取，并在寄存器中多次复用。
*   **权重立即驱逐 (Evict-First Policy)：** 由于权重矩阵（Matrix B）在一次推理中每个权重只被读取一次（不复用），Marlin 使用了特殊的 CUDA 异步加载指令，并带有**缓存提示（Cache Hint）**。
*   **代码证据 (`marlin_cuda_kernel.cu`)：**
    在 `cp_async4_stream` 函数中，使用了内联 PTX 汇编 `createpolicy.fractional.L2::evict_first.b64`。这意味着告诉 GPU：“加载这个数据后，立即将其从 L2 缓存中驱逐”。这样可以防止“用完即弃”的权重数据污染 L2 缓存，从而为需要高频复用的激活值和输出累加器保留宝贵的 L2 空间。

```cpp
// marlin_cuda_kernel.cu
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  // ...
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;" // <--- 关键：立即驱逐策略
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}
```

### 2. 高效的流水线与异步加载 (Global Memory & Shared Memory Pipelining)
为了不让 Tensor Core 等待数据，Marlin 极力隐藏内存延迟。

*   **异步全局加载 (Async Global Loads)：** 所有的全局内存读取（权重）都与计算操作以及激活值的加载异步执行。
*   **多阶段流水线 (Multi-stage Pipeline)：** 源码中定义了 `STAGES = 4`，使用共享内存作为缓冲区。
*   **双缓冲 (Double Buffering)：** 共享内存的加载通过双缓冲进行，以重叠（Overlap）数据传输和计算。

### 3. "条纹状"分区方案 (Striped Partitioning Scheme)
在大 BS 或不同的矩阵形状下，传统的块平铺（tiling）可能导致某些 SM（流多处理器）负载过重，而其他 SM 空闲（Tail effect），或者导致过多的全局归约（Global Reduction）开销。

*   **跨列切片 (Column Slices)：** Marlin 实现了一种“条纹”分区，每个 SM 处理的 tile 段可以跨越多个列切片。
*   **优势：** 这种方案在大多数矩阵形状下都能保持极高的 SM 利用率，同时最大限度地减少了需要的全局归约步骤。
*   **L2 中的归约：** 当必须进行全局归约时，Marlin 直接在输出缓冲区（保持在 L2 缓存中）进行，避免了不必要的读写。

### 4. 离线权重重排 (Offline Weight Reshuffling)
AutoGPTQ 的部分开销在于运行时需要将权重解包或重排以适应 Tensor Core 的输入要求。

*   **预处理 (`Layer.pack` in `__init__.py`)：** Marlin 将 4-bit 量化权重和缩放因子（scales）在离线阶段就重排成特定的布局。
*   **直接解量化到 Tensor Core：** 这种布局允许内核在运行时直接将权重解量化为 Tensor Core 所需的寄存器组织形式，消除了运行时的 Shuffle 开销。

### 5. 提升 Warp 并发度 (Multi-warp Partial Results)
为了提高延迟隐藏能力，需要在一个 Threadblock 中运行足够多的 Warp。

*   **问题：** 如果单纯增加 Output Tile Size 来增加 Warp，会使得在处理实际模型中常见的中小矩阵时难以分割（Partitioning）。
*   **解决方案：** Marlin 让 Threadblock 中的多个 Warp 计算**同一个输出 Tile 的部分结果**（Partial Results）。这样可以在不增加输出 Tile 大小的情况下增加 Warp 数量，从而最大化计算能力和延迟隐藏。

### 6. 指令级优化 (Instruction Level Optimizations)
*   **最大化向量长度：** 所有的加载指令都使用最大向量长度（如 `cp.async.cg.shared.global` 处理 16 字节），以达到峰值内存效率。
*   **无冲突共享内存访问：** 通过布局转换（XOR-based layout in `marlin_cuda_kernel.cu`），保证所有的共享内存读写都没有 Bank Conflict。
*   **静态偏移：** 循环展开（Unrolling）使得绝大多数内存偏移量在编译时就是静态的，减少了运行时的索引计算开销。

### 总结
Marlin 解决大 BS 下 Tensor Core 利用率低的核心在于：**它不再将内存带宽视为一个简单的瓶颈，而是通过精细的 L2 缓存控制（Evict-first）和异步流水线，确保了即使在权重数据流巨大的情况下，Tensor Core 所需的激活值依然能快速命中缓存，从而让计算单元保持饱和。**

---

### 🧑‍💻 User

marlin_cuda_kernel.cu整个load->dequant->matmul->st pipeline是怎么样的？

---

### 🤖 Assistant

基于 `marlin_cuda_kernel.cu` 的代码，Marlin Kernel 的执行流水线（Pipeline）设计得非常紧凑且高度并行。它采用了 **多阶段（Multi-stage）软件流水线** 结合 **异步内存拷贝（Async Copy）** 技术，以此来掩盖全局内存访问的延迟。

整个流程可以概括为四个主要层级的数据流动：
**Global Memory (DRAM/L2) -> Shared Memory (SRAM) -> Registers (RF) -> Tensor Cores**

以下是该 Pipeline 的详细拆解：

### 1. 核心架构概览

*   **Stages (阶段数):** 代码中定义 `STAGES = 4`。这意味着共享内存（Shared Memory）被划分为 4 个缓冲区，形成一个循环队列。
*   **Warp 分工:** 不需要专门的 Warp 做数据搬运（这是老式做法），而是所有 Warp 协同工作，利用 `cp.async` 指令让 DMA 引擎在后台搬运数据，Warp 本身专注于计算。
*   **数据类型:**
    *   **A (Input):** FP16
    *   **B (Weights):** INT4 (打包在 int32 中)
    *   **s (Scales):** FP16

---

### 2. 详细 Pipeline 步骤

#### 阶段一：Global Memory -> Shared Memory (异步加载)
**负责函数:** `fetch_to_shared`

这是流水线的最前端。Kernel 会预取未来需要计算的数据块（Tile）。

1.  **A 矩阵加载:**
    *   使用 `cp_async4_pred` (即 `cp.async.cg.shared.global`)。
    *   **关键优化:** 为了避免 Bank Conflict，地址经过了 XOR 变换 (`transform_a` lambda 函数)，实现了 Swizzling 布局。
2.  **B 矩阵 (权重) 加载:**
    *   使用 `cp_async4_stream`。
    *   **关键优化 (L2 Cache Policy):** 这里使用了 `createpolicy.fractional.L2::evict_first`。告诉 GPU 加载完 B 矩阵数据后**立即从 L2 缓存中驱逐**。因为权重只用一次，这样可以保护 A 矩阵和 C 矩阵（累加器）常驻 L2。
3.  **s (Scales) 加载:**
    *   如果是分组量化（Grouped Quantization），每隔 `groupsize` 也会异步加载一次 Scales。

#### 阶段二：Shared Memory -> Registers (双缓冲加载)
**负责函数:** `fetch_to_registers`

当 Tensor Core 正在计算当前子块（Sub-tile）时，Warp 会从 Shared Memory 加载**下一个**子块的数据到寄存器。

1.  **A 矩阵:**
    *   使用 `ldsm4` (即 `ldmatrix.sync.aligned.m8n8.x4.shared.b16`)。
    *   这是 Tensor Core 专用指令，直接将 Shared Memory 中的矩阵块加载到寄存器中符合 Tensor Core 要求的布局（Fragment）。
2.  **B 矩阵 (INT4):**
    *   直接加载为 `int4` 类型到寄存器 (`frag_b_quant`)。

#### 阶段三：寄存器内解量化与计算 (Dequant -> MMA)
**负责函数:** `matmul`

这是计算的核心。数据在寄存器中准备好后，立即进行处理。

1.  **解量化 (Dequantization):**
    *   **函数:** `dequant`
    *   **逻辑:** 并没有将 INT4 转换回 Global Memory，而是利用位运算 (`lop3`) 和快速数学指令，在寄存器内将 `int32` 包裹的 INT4 数据实时解压并转换为 FP16 (`half2`)。
    *   **Symmetric Zero Point:** 将 -8 的零点偏移融合进了加减法操作中。
2.  **缩放 (Scaling):**
    *   **函数:** `scale`
    *   如果开启了分组量化，解量化后的 FP16 权重会立即乘以对应的 Scale。
3.  **矩阵乘 (MatMul):**
    *   **函数:** `mma`
    *   指令: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
    *   输入 FP16，累加在 FP32 (`frag_c`) 中。

---

### 3. 主循环逻辑 (The Pipeline Loop)

代码中的主循环 (`while (slice_iters)`) 通过精细的指令排布实现了上述操作的重叠（Overlap）。

```cpp
// 伪代码流程分析
#pragma unroll
for (int pipe = 0; pipe < stages;) { // 遍历流水线阶段
    #pragma unroll
    for (int k = 0; k < b_sh_wr_iters; k++) { // 遍历当前 Stage 内的子块(Sub-tiles)
      
        // 1. 预加载：从 Shared Memory 加载 [k+1] (下一个子块) 到寄存器
        //    为下一次 matmul 做准备，掩盖 Shared Memory 延迟
        fetch_to_registers(k + 1, pipe % stages);

        // 2. 触发异步拷贝：如果当前 Stage 快算完了，就开始从 Global Memory 
        //    请求 [未来] 的 Stage 数据填入 Shared Memory
        if (k == b_sh_wr_iters - 2) {
            fetch_to_shared((pipe + stages - 1) % stages, ...);
            pipe++; 
            wait_for_stage(); // 确保即将写入的 Shared Memory 已经被消费完
        }

        // 3. 计算：对 [k] (当前子块) 进行解量化和矩阵乘
        //    这是最耗时的部分，这期间 DMA 引擎在后台搬运数据 (Global->Shared)
        matmul(k); 
    }
}
```

**流水线特点：**
*   **计算掩盖传输:** `matmul` 的计算时间掩盖了 `fetch_to_shared` (Global->Shared) 的延迟。
*   **双缓冲掩盖读取:** `fetch_to_registers` 总是预取下一个子块，掩盖 Shared->Register 的延迟。

---

### 4. 阶段四：Epilogue (归约与写回)

当所有 Tile 计算完成后，进入收尾阶段：

1.  **Thread Block Reduce (块内归约):**
    *   **函数:** `thread_block_reduce`
    *   由于为了提高并行度，不同的 Warp 可能计算了同一个输出位置的部分和。
    *   使用 Shared Memory 进行跨 Warp 的并行归约（Logarithmic reduction）。
2.  **Global Reduce (全局归约 - 若需要):**
    *   **函数:** `global_reduce`
    *   如果 Slice 跨越了多个 Thread Block（Marlin 的 Striped Partitioning 特性），Block 之间需要同步。
    *   使用 `barrier_acquire` (自旋锁) 等待其他 Block。
    *   直接在 L2 Cache 中（即直接读写 C 矩阵地址）将 FP32 累加器转换为 FP16 进行加法，然后写回。
3.  **Write Result (写回):**
    *   **函数:** `write_result`
    *   如果是最后一个负责该 Slice 的 Block，将最终结果通过 Shared Memory 重排（以保证合并写 Coalesced Access），然后写入 Global Memory C。

### 总结图示

```text
Time --->
[Global Fetch Stage 0] ...
                       [Global Fetch Stage 1] ...
                                              [Global Fetch Stage 2] ...
-----------------------------------------------------------------------
(Wait Stage 0)
[Shared Load SubTile 0]
      |
      +---> [Dequant & Matmul SubTile 0] 
            [Shared Load SubTile 1]  <-- 预取下一块
                  |
                  +---> [Dequant & Matmul SubTile 1]
                        [Global Fetch Stage 3 (Async)] <-- 触发下一轮全局加载
                        ...
```

这种设计确保了 Tensor Core (MMAPipe) 几乎从不空闲，始终有寄存器数据可供计算，从而在大 Batch Size 下实现了极高的利用率。