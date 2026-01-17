# 优化matmul kernel分块参数

### 🧑‍💻 User

结合hopper架构特点，说说优化matmul kernel的时候，如何根据架构细节推算出理论上能达到峰值的分块参数

---

### 🤖 Assistant

在NVIDIA Hopper (H100/H800) 架构上优化 Matrix Multiplication (MatMul) Kernel，与之前的 Ampere (A100) 或 Volta 架构有显著不同。核心差异在于引入了 **TMA (Tensor Memory Accelerator)** 和 **WGMMA (Warpgroup Matrix Multiply Accumulate)** 指令。

要推算理论峰值的分块参数（Tiling Parameters），必须遵循“**数据搬运掩盖计算延迟**”的核心原则，并严格适配 Hopper 的硬件限制。

以下是基于 Hopper 架构细节推算最佳分块参数的逻辑步骤：

---

### 1. 理解 Hopper 的核心架构约束

在推导参数前，需要明确几个关键的硬件常数（以 FP16/BF16 输入，FP32 累加为例）：

*   **SM 资源**: 每个 SM 拥有 228KB 的 Shared Memory (SMem)。
*   **WGMMA 指令**: Hopper 引入了 Warpgroup (128 线程，4 个 warp) 概念。WGMMA 指令直接读取 Shared Memory 进行计算，**不需要**像以前那样先将 $A/B$ 矩阵载入寄存器文件（Register File）。
    *   典型 WGMMA 形状 ($M \times N \times K$): $64 \times N \times 16$ (N 可为 8 to 256)。
*   **TMA**: 异步数据拷贝引擎，支持多维张量拷贝。TMA block size 最好是 128 Bytes 的倍数（对应 L2 cache line 和 DRAM burst）。
*   **Register File**: 每个线程最多 255 个寄存器，但为了高 Occupancy，通常限制在 128-160 以内。在 Hopper 上，寄存器主要用于存储矩阵 $C$ 的累加结果 (Accumulators)。

---

### 2. 分块层级定义

我们需要确定以下三个层级的分块大小：
1.  **Grid Level (CTA Tile)**: $B_M \times B_N \times B_K$ (一个 Thread Block 处理的大小)。
2.  **Warpgroup Level (WG Tile)**: $W_M \times W_N \times W_K$ (一个 Warpgroup 处理的大小)。
3.  **Instruction Level**: $I_M \times I_N \times I_K$ (一条 WGMMA 指令的大小)。

---

### 3. 推算步骤与公式

#### 第一步：确定 CTA Tile ($B_M, B_N$) —— 由寄存器压力决定

在 Hopper WGMMA 模式下，$A$ 和 $B$ 矩阵驻留在 SMem 中，只有 $C$ 矩阵驻留在寄存器中。因此，Block 的大小主要受限于**累加器寄存器的大小**。

假设一个 Warpgroup (128 threads) 处理 $W_M \times W_N$ 的块：
*   每个线程持有的元素数量 = $\frac{W_M \times W_N}{128}$
*   若使用 FP32 累加器（每个 4 Bytes），寄存器占用量为：
    $$ \text{Regs}_{acc} = \frac{W_M \times W_N}{128} \times \frac{4 \text{ Bytes}}{4 \text{ Bytes/Reg}} = \frac{W_M \times W_N}{128} $$

**推导：**
为了防止寄存器溢出（Spill），我们通常希望每个线程的 Accumulator 占用不超过 64-128 个寄存器（留出空间给标量计算、TMA 指针等）。
*   若目标 $\text{Regs}_{acc} = 64$: $W_M \times W_N = 64 \times 128 = 8192$。
    *   可能的组合: $64 \times 128$ 或 $128 \times 64$。
*   若目标 $\text{Regs}_{acc} = 128$: $W_M \times W_N = 128 \times 128 = 16384$。
    *   可能的组合: $128 \times 128$。

**架构偏好：**
Hopper 的 WGMMA 指令对 $M=64$ 优化最好。通常选择 **Block Size ($B_M \times B_N$) 为 $128 \times 128$ 或 $128 \times 256$**。如果是 $128 \times 256$，通常分配 2 个 Warpgroups (256 线程)。

#### 第二步：确定 CTA K-维度 ($B_K$) 与流水线级数 (Stages) —— 由 SMem 容量决定

这是 Hopper 优化的关键。我们需要使用 TMA 构建多级流水线（Multistage Pipeline），通常是 3 到 5 级。

*   **SMem 占用公式**:
    $$ \text{SMem}_{\text{usage}} = (B_M \times B_K + B_N \times B_K) \times \text{dtype\_size} \times \text{Stages} $$
*   **约束**: $\text{SMem}_{\text{usage}} < \text{SMem}_{\text{capacity}}$ (228KB, 保守取 200KB 以留出 barrier 空间)。

**推导示例 (FP16, 2 bytes):**
假设 $B_M=128, B_N=256$ (使用 2 个 Warpgroups)。
$$ \text{One Stage Size} = (128 \times B_K + 256 \times B_K) \times 2 = 768 \times B_K \text{ Bytes} $$

我们需要选择 $B_K$。
*   **指令对齐**: $B_K$ 必须是 16 (FP16) 的倍数，WGMMA 建议 $K=64$ 或更高以分摊开销。
*   试算 $B_K = 64$:
    *   单级大小: $768 \times 64 = 49,152 \text{ Bytes} \approx 48 \text{ KB}$。
    *   若 Stage = 3: $48 \times 3 = 144 \text{ KB}$ (Fit ✅)。
    *   若 Stage = 4: $48 \times 4 = 192 \text{ KB}$ (Fit ✅，且能更好掩盖延迟)。
*   试算 $B_K = 128$:
    *   单级大小: $96 \text{ KB}$。
    *   Stage = 2: $192 \text{ KB}$ (Fit，但级数太少，流水线气泡风险大)。

**结论**: 对于 $128 \times 256$ 的 Block，推荐 $B_K = 64$，Stage = 3 或 4。

#### 第三步：计算密度与掩盖延迟 (Arithmetic Intensity)

验证选定的参数是否能达到计算受限（Compute Bound）而非访存受限（Memory Bound）。

*   **计算量 (Math)**: $2 \times B_M \times B_N \times B_K$ (FLOPs per pipe stage).
*   **数据量 (Bytes)**: $(B_M + B_N) \times B_K \times \text{dtype\_size}$.
*   **H100 比例**: H100 SXM FP16 峰值约 1000 TFLOPS，HBM3 带宽约 3.35 TB/s。
    $$ \text{Ratio}_{\text{hw}} \approx \frac{1000}{3.35} \approx 300 \text{ FLOPs/Byte} $$

**验证我们的参数 ($128 \times 128 \times 64$):**
*   Math: $2 \times 128 \times 128 \times 64 = 2,097,152$ FLOPs.
*   Bytes: $(128 + 128) \times 64 \times 2 = 32,768$ Bytes.
*   Arithmetic Intensity: $\frac{2,097,152}{32,768} = 64$ FLOPs/Byte.

**分析**: 64 远小于 300。这看起来是 Memory Bound？
**其实不然**。这里的 Calculation 是针对单次 GEMM 迭代。实际 Kernel 中，数据 $A$ 和 $B$ 是从 HBM 读取一次，然后在寄存器中与 $C$ 累加多次（平铺整个 K 维度）。
但是，对于主循环（Main Loop）的流水线，我们需要确保 **WGMMA 计算时间 > TMA Copy 时间**。

*   **WGMMA 吞吐**: Hopper 每个 SM 每周期吞吐量巨大。
*   **TMA 带宽**: 接近 L2/DRAM 峰值。

为了掩盖延迟，必须最大化 $B_K$ 和 Stages。在 Hopper 上，**WGMMA 指令非常快，通常瓶颈在于 Shared Memory 的 Bank Conflict 或者是 TMA 的加载速度**。因此，Swizzle 模式（数据在 SMem 的排列）至关重要。

---

### 4. 针对 Hopper 的“黄金参数”推荐

结合 Cutlass 和 FlashAttention 的优化经验，Hopper 上的 MatMul 最佳参数通常如下：

#### 场景 A: 大矩阵 FP16 GEMM (高吞吐)
*   **Block Shape ($B_M \times B_N \times B_K$)**:
    *   $128 \times 256 \times 64$ (需要 2 个 Warpgroups，即 256 线程)。
    *   或者 $256 \times 128 \times 64$。
*   **Pipeline Stages**:
    *   **3** (如果 SMem 紧张)。
    *   **4 或 5** (推荐，如果 SMem 允许，能最大程度掩盖 HBM 延迟)。
*   **Warpgroup Shape**:
    *   使用 `wgmma.mma_async.sync.aligned.m64nNk16`。
    *   每个 WG 负责 $64 \times N \times K$ 的子块计算。
*   **TMA**:
    *   启用 TMA Multicast (如果计算 $Q \times K^T$ 这种 Head 广播场景)。
    *   Swizzle: 128B Swizzle (对应 `cp.async.bulk.tensor` 的 swizzle 模式)。

#### 场景 B: 小 Batch 或 FlashAttention (SMem 受限)
*   **Block Shape**: $128 \times 128 \times 64$ (1 个 Warpgroup, 128 线程)。
*   **Pipeline Stages**: 3。
*   **优势**: 只需要 128 个线程，SM 占用率高，寄存器压力较小。

---

### 5. 总结：推算流程表

| 步骤 | 决策依据 | Hopper 特有考量 | 典型值 |
| :--- | :--- | :--- | :--- |
| **1. Block $M, N$** | 寄存器文件大小 (Accumulators) | WGMMA 不需要 A/B 进寄存器，只考虑 C | $128 \times 128$ 或 $128 \times 256$ |
| **2. Block $K$** | WGMMA 效率 vs. SMem 开销 | WGMMA 指令深度通常为 16，$B_K$ 应较大以分摊开销 | 64 (FP16), 32 (FP32) |
| **3. Stages** | SMem 总容量 (228KB) | TMA 需要多缓冲来掩盖 DRAM 延迟 | 3, 4, 或 5 |
| **4. Grid Size** | GPU SM 数量 | 确保 Grid 数量能填满 132 个 SM (H100) | 动态调整 Block Size 以适配 Problem Size |
| **5. Warps** | Warpgroup 架构 | 必须是 4 Warps (128 threads) 的倍数 | 1 WG (128 th) 或 2 WG (256 th) |

### 6. 代码实现层面的微调

在确定了 $128 \times 256 \times 64$, 4-stage 这种理论参数后，实际达到峰值还需要：
1.  **SMem Layout**: 使用 `swizzle_128B` 避免 WGMMA 访问 SMem 时的 Bank Conflict。
2.  **Instruction Issue**: 连续发射 WGMMA 指令，中间不要插入非必要的标量指令。
3.  **TMA Descriptor**: 在 Host 端预先构建好 TMA Descriptor，Kernel 运行时只更新指针。
4.  **L2 Cache Residency**: 设置 `L2 Cache Persistence` (SetAside) 策略，如果是复用率高的数据（如 Weight Stationary 推理中的权重）。

通过上述推算，你可以将计算密度与 Hopper 巨大的 SMem 带宽和 WGMMA 算力匹配，从而接近 90%+ 的理论峰值性能。

---

### 🧑‍💻 User

我记得需要根据寄存器文件大小，shared memory size/带宽/延迟，各级cache及global memory的带宽/延迟，有一套方法论，确认矩阵乘分块的tile m/n/k及软流水的级数。能不能展开说说？

---

### 🤖 Assistant

这是一个非常专业的问题。你提到的这套方法论，通常被称为 **"Roofline Model with Resource Constraints" (基于资源约束的 Roofline 模型)** 或者 **"Occupancy-aware Latency Hiding" (基于占用率的延迟掩盖分析)**。

在 Hopper 架构上，这套方法论依然适用，但因为 WGMMA（在 Shared Memory 直接计算）和 TMA（异步搬运）的引入，**约束条件的权重发生了变化**。

下面我将详细拆解如何利用这套方法论，通过计算推导 $B_M, B_N, B_K$ 以及流水线级数（Stages）。

---

### 核心逻辑框架

我们的目标是求解一个优化问题：**最大化吞吐量**，同时满足以下三个硬性约束：
1.  **寄存器约束 (Register Wall)** $\rightarrow$ 决定 $B_M, B_N$
2.  **SMem 容量约束 (Capacity Wall)** $\rightarrow$ 决定 $B_K$ 和最大 Stage 数
3.  **延迟掩盖约束 (Latency Wall)** $\rightarrow$ 决定最小 Stage 数

---

### 第一步：确定 $B_M, B_N$ (由寄存器文件决定)

在 Hopper WGMMA 模式下，$A$ 和 $B$ 矩阵驻留在 SMem，$C$ 矩阵（Accumulators）驻留在寄存器。因此，$B_M$ 和 $B_N$ 的大小直接决定了每个线程需要多少寄存器来存放结果。

**已知参数：**
*   $R_{file}$: 每个线程可用的最大寄存器数（Hopper max 255，但在高优化的 Kernel 中，为了保持高 Occupancy，通常控制在 **128 ~ 160** 之间）。
*   $N_{threads}$: 一个 Warpgroup (WG) 有 128 个线程。
*   $S_{acc}$: 累加器数据类型大小 (FP32 = 4 Bytes)。

**计算公式：**
每个线程需要的 Accumulator 寄存器数 $R_{acc}$ 为：
$$ R_{acc} = \frac{B_M \times B_N}{N_{threads} \times \text{Num\_WGs}} \times \frac{S_{acc}}{4 \text{ Bytes/Reg}} $$

**推导过程：**
为了防止寄存器溢出到 Local Memory，我们要求 $R_{acc} + R_{other} < R_{limit}$。假设 $R_{other}$ (指针、标量、TMA描述符) 需要 40 个寄存器，目标是将总寄存器控制在 128 以内，则 $R_{acc} \approx 80 \sim 90$。

*   **Case 1: 1个 WG (128 threads)**
    $$ \frac{B_M \times B_N}{128} \approx 80 \implies B_M \times B_N \approx 10240 $$
    最接近的规整块是 $128 \times 64 = 8192$ 或 $64 \times 128$。
    *如果要上到 $128 \times 128 (16384)$，单线程需 128 个寄存器给 $C$，总寄存器会飙升到 160+，导致 Occupancy 下降，但在 Hopper 上这通常是可接受的（因为 Tensor Core 极快，Latency hiding 靠 TMA）。*

*   **Case 2: 2个 WG (256 threads)**
    为了处理更大的块，通常合并两个 WG。
    $$ \frac{B_M \times B_N}{256} \approx 80 \implies B_M \times B_N \approx 20480 $$
    *   **黄金组合**: $128 \times 128$ (64 Regs/thread) 或 **$128 \times 256$** (128 Regs/thread)。
    *   Hopper 推荐：对于大矩阵，使用 2个 WG 处理 $128 \times 256$ 的 Tile，此时寄存器压力适中，计算密度极大。

**结论 1**: 选定 $B_M=128, B_N=256$ (使用 256 线程) 或 $128 \times 128$ (使用 128 线程)。

---

### 第二步：确定 $B_K$ 和 Stages (由 SMem 和 带宽延迟积决定)

这是最复杂的平衡。我们需要 $B_K$ 足够大以利用 WGMMA 效率，又需要 Stage 足够多以掩盖 Global Memory 延迟。

#### 1. 计算单级流水线的计算时间 ($T_{math}$)
这是我们掩盖延迟的“本钱”。
$$ T_{math} = \frac{\text{FLOPs per Tile}}{\text{SM Peak FLOPS}} = \frac{2 \times B_M \times B_N \times B_K}{\text{Tensor Core Peak Rate}} $$
*   H100 SM FP16 峰值 $\approx 1980$ TFLOPS (含 Sparsity) / 132 SMs $\approx$ 15 TFLOPS/SM (非稀疏)。
*   注意：这里使用的是 SM 的**有效峰值**，通常取理论值的 80%-90%。

#### 2. 计算单级流水线的数据搬运时间 ($T_{load}$)
$$ T_{load} = \frac{\text{Bytes per Tile}}{\text{Effective Mem BW}} = \frac{(B_M + B_N) \times B_K \times \text{dtype\_size}}{\text{HBM BW per SM}} $$
*   HBM3 带宽 3.35 TB/s 分摊到 132 个 SM，约 25 GB/s/SM。

#### 3. 延迟掩盖分析 (Latency Hiding)
要实现完全的延迟掩盖，必须满足：**在计算当前数据的同时，下一块数据已经准备好了。**
但是，HBM 的访问延迟 ($L_{mem}$, Latency) 很高，大约 500~800 cycles。TMA 虽然带宽大，但从发出指令到数据到达 SMem 有固定延迟。

我们需要的 Stage 数量 ($N_{stage}$) 必须满足：
$$ (N_{stage} - 1) \times T_{math} > L_{mem} + T_{load} $$
*   解释：除了正在计算的这一级（Stage 0），剩余的 ($N-1$) 级提供的计算时间总和，必须能够覆盖从 HBM 拉取新数据所需的 延迟 + 传输时间。

**推算 $B_K$：**
*   **WGMMA 限制**: $K$ 维度通常固定为 16 (FP16)，为了减少指令发射开销，软件流水通常将 $K$ 展开。
*   **带宽利用率**: 更大的 $B_K$ 意味着单次 TMA 指令搬运更多数据，更能吃满带宽。
*   通常试探性选择 $B_K = 32, 64, 128$。
    *   $B_K=32$: 计算量太小，$T_{math}$ 太短，难以掩盖 $L_{mem}$。
    *   $B_K=128$: 单个 Stage 消耗 SMem 太多，导致 Stage 数量受限。
    *   **黄金值**: Hopper 上通常选择 **$B_K = 64$** (FP16)。

---

### 第三步：SMem 容量校验 (Capacity Check)

选定了 $B_M, B_N, B_K$ 和预估的 $N_{stage}$ 后，必须验证 SMem 是否装得下。

**公式：**
$$ \text{SMem}_{\text{req}} = (B_M \times B_K + B_N \times B_K) \times \text{dtype\_size} \times N_{stage} $$
**约束：**
$$ \text{SMem}_{\text{req}} + \text{Barrier/Descriptor Overhead} < 228 \text{ KB} $$

**实战演算 (FP16, $128 \times 256 \times 64$):**
*   单级数据量: $(128 + 256) \times 64 \times 2 \text{ Bytes} = 384 \times 128 = 49152 \text{ Bytes} \approx 48 \text{ KB}$。
*   H100 SM SMem = 228 KB。保留 4KB 给系统开销，可用 $\approx 224$ KB。
*   最大 Stage 数: $224 / 48 = 4.66$。
*   **结论**: 我们可以开 **3 Stage** 或 **4 Stage**。

---

### 第四步：综合决策 (Final Determination)

现在我们有两个选择：3 Stage 还是 4 Stage？这就回到了 **延迟掩盖** 的公式。

假设 $B_M=128, B_N=256, B_K=64$。
*   **Math Time**: $2 \times 128 \times 256 \times 64 = 4.2$ MFLOPs。
    *   假设 SM 跑在 1000 TFLOPS (保守)，单 SM $\approx 7.5$ TFLOPS。
    *   $T_{math} \approx 4.2 \times 10^6 / (7.5 \times 10^{12}) \approx 0.56 \mu s$。
    *   转化为时钟周期 (1.8GHz): $\approx 1000$ Cycles。
*   **Mem Latency**: HBM Latency $\approx 600$ Cycles。

**分析 Stage:**
*   **3 Stages**: 缓冲池有 2 个 Stage 的计算时间来掩盖加载。
    *   掩盖能力: $2 \times 1000 = 2000$ Cycles。
    *   需求: Latency (600) + Transmission (取决于带宽)。
    *   2000 Cycles 通常足够掩盖 HBM 延迟。
*   **4 Stages**: 缓冲池有 3 个 Stage。
    *   掩盖能力: 3000 Cycles。
    *   优势: 抗抖动能力更强，当由 DRAM 刷新或路由拥塞导致延迟波动时，Pipeline 不会断流。

**决策路径:**
1.  如果 SMem 允许，**优先选择更多 Stage** (4 or 5)。
2.  在 Hopper 上，$128 \times 256 \times 64$ @ 4 Stages 消耗 $192$ KB SMem，是一个非常激进但高效的配置，正好接近 228KB 的极限。

---

### 总结：推算方法论 Check List

当你面对一个新的矩阵乘问题时，按以下顺序填空：

1.  **Look at Hardware**: 确认 SM SMem 大小 (228KB) 和 Max Regs (255)。
2.  **Determine Tile $M, N$ (Regs)**:
    *   目标：让 $\frac{B_M \cdot B_N}{\text{Threads}}$ 对应的寄存器数 $\approx 128$。
    *   结果：$128 \times 256$ (256 threads) 或 $128 \times 128$ (128 threads)。
3.  **Determine Tile $K$ (Efficiency)**:
    *   目标：至少 64 (FP16) 以摊薄 TMA 头和 WGMMA 指令开销。
4.  **Calculate Single Stage Size (SMem)**:
    *   $S_{stage} = (B_M + B_N) \cdot B_K \cdot \text{size}$。
5.  **Calculate Max Stages (Capacity)**:
    *   $N_{max} = \lfloor \text{SMem\_Total} / S_{stage} \rfloor$。
6.  **Verify Latency Hiding (Bandwidth)**:
    *   计算 Arithmetic Intensity (AI) = $\frac{2 B_M B_N B_K}{S_{stage}}$。
    *   如果 AI > 机器的 FLOPs/Byte 比率，理论上只需 2-3 Stages 即可掩盖。
    *   Hopper FP16 Tensor Core 算力极高，实际上往往处于 **Memory Latency Bound** 的边缘，因此倾向于 **Max Stages**。

**最终 Hopper FP16 推荐配置：**
*   **Block**: $128 \times 256 \times 64$
*   **Warps**: 8 Warps (256 threads)
*   **Stages**: 3 或 4 (取决于是否需要额外的 SMem 做 Epilogue 或 bias)
*   **Pipeline**: TMA Async Copy + WGMMA Pipelining