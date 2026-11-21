# NPU与MoE动态调度

## 🧑‍💻 User

说起来 大部分移动端的NPU在2023年以前基本上都是固化Conv算法ASIC类似物
现在的端侧NPU发展的怎么样了
对于moe fa这类需要细粒度数据流依赖 动态dispatch的算子支持情况怎么样？
和像groq、tpu这类专门针对llm infer的npu差距在哪里

---
## 😀 默认助手 (7)

# 端侧 NPU：从“Conv ASIC”到“能跑 Transformer”，以及 MoE/FA 的关键缺口

## 0. 先校准“固化 Conv 算法 ASIC”这个前提

这个判断对一半，而且偏差正好在要害上。

2023 年前的端侧 NPU 真正“固化”的不是 Conv 算法本身，而是三件事：

| 固化的是 | 后果 |
|---|---|
| **执行模型**：逐层整层执行（layer-by-layer），算子边界有全量 DRAM 往返 | 无法跨算子融合，FlashAttention 这种“必须跨算子融合才有意义”的优化天然不兼容 |
| **形状**：编译期静态 shape，AOT 编译成不可变的指令流/blob | 任何 `if`、可变长度、动态 batch 都要靠“多编译几份 + host 选一份” |
| **计算单元构成**：只有 MAC 阵列，**没有可编程向量单元** | softmax、LayerNorm、argmax、mask 只能落回 CPU → 图被切碎成几十段 → 每段有提交开销 |

第三点是最致命的、也最常被忽略的。Conv 网络的算子栈（conv/bn/relu/pool/concat）几乎全是“MAC + LUT 激活”，所以纯 MAC 阵列 + 查表激活就够了，不需要通用向量处理。但 Transformer 的算子栈一半是**非 matmul 的 elementwise + reduction**（softmax、RMSNorm、RoPE、residual add、mask、top-k），纯 MAC 阵列对这些算子几乎无能为力。

所以 2023 年前的典型情况是：NPU 负责 70% 的时间里在算 GEMM，但每层都要把 feature map 写回 DRAM 交给 CPU 做 LayerNorm，再交回去。**计算不是瓶颈，数据搬运和 kernel launch 次数才是**。

> 补充一个反例，避免过度概括：Hexagon 的 HVX（2016 起）本来就是可编程向量 DSP，华为 Da Vinci（2018 起）从 Kirin 980 起就是 `3D Cube（矩阵）+ Vector（向量）+ Scalar（标量）` 三单元架构——Da Vinci 的 Cube 是 16×16×16 的通用 MatMul 单元，不是 Conv 专用。真正“纯 Conv ASIC”的是早期三星/联发科 APU、EdgeTPU、Movidius 那一类，以及 NNAPI HAL 时代那一批。

---

## 1. 2023 年之后发生了什么：四条主线

### 1.1 计算单元：向“Da Vinci 化”收敛

现在的端侧 NPU 基本都变成了 **tensor core + 可编程 vector unit + scalar unit** 三件套：

- tensor core 吃 GEMM（含 Conv 转 implicit GEMM）
- vector unit 吃 softmax/LayerNorm/RoPE/mask——**这是 FA 能不能落地的物理前提**
- scalar unit 吃索引计算和分支，让“控制流留在 NPU 上”成为可能

这条线是 2023 年后最重要的变化。它把“图被 CPU 切碎”的问题从架构上解掉了：一层 Transformer 现在可以是一个完整的 submission。

### 1.2 存储层级：on-chip SRAM 涨到 MB 级

- Hexagon 的 VTCM 到 8 MB 量级（并且程序员可见、可显式管理，这是它能手写融合 kernel 的原因）
- Apple ANE / 各家 APU 的专用 SRAM 也在同量级或更大

8 MB 意味着什么：一个 128×128 的 Q tile 加 4K context 的 K/V tile（FP16, GQA 下）大致刚好能驻留。**这正好是“能不能做单 pass online softmax”的分水岭**。

### 1.3 精度：向 LLM 量化格式对齐

INT8 per-tensor → **INT4 block-wise / per-group scale**，部分支持 FP8、MXFP4/MXINT8（microscaling）。这一条比 TOPS 重要得多，因为：

$$
T_{\text{token}} \approx \frac{B_{\text{active weights}} + B_{\text{KV cache}}}{BW_{\text{eff}}}
$$

端侧 decode 是**纯带宽 bound**，算力基本闲置。所以“原生支持 4-bit 权重布局”带来的收益是 2× 的 token 速率，而“TOPS 翻倍”带来的收益接近 0。

### 1.4 软件：从 AOT 静态图到“LLM runtime”

这一层的进展其实比硬件大：
- **KV cache 变成带动态 seq 维的外部 I/O tensor**（不再需要为每个长度重编译）
- **dynamic shape** 的有限支持（主要是 batch / seq 维）
- **图内分支算子**（if/switch/while）开始被接受
- **host 侧编排 runtime**（Qualcomm Genie、MediaTek NeuroPilot、Apple CoreML + Foundation Models、Meta ExecuTorch + 各家 delegate）
- **CPU 侧微内核**：Arm KleidiAI、i8mm/dotprod、SME/SME2、Apple AMX——CPU 现在是合格的 prefill 协处理器，不再是“只做 LayerNorm 的杂役”

<details>
<summary>各家现状（参考性，具体数字请以官方材料为准）</summary>

- **Qualcomm**：Snapdragon 8 Elite 一代把 Hexagon 的 tensor/vector/scalar 做了融合，并引入面向 attention 的 micro-tile 式推理与更低比特权重支持；PC 侧 Snapdragon X 系列 NPU 到几十 TOPS 级、LPDDR5X 128-bit，已经能跑 20B 级模型。软件侧 QNN/QAIRT + Genie 是目前端侧最完整的一套。
- **Apple**：ANE 架构不公开，但 CoreML 的 compute unit 自动切分 + 统一内存 + 相对成熟的 Metal/simdgroup 路径，让“NPU 做 prefill、GPU 做 decode”这条路走得最顺；Foundation Models framework 把端侧 3B 级模型做成了系统 API。
- **MediaTek**：APU 从 2023 起主打“Generative AI Engine”，宣传重点是 INT4、LoRA 端侧微调、speculative decoding，以及最近的超低比特（1.x-bit ternary / BitNet 类）与 MoE 演示。
- **华为**：Da Vinci 架构本来就是 MatMul 优先，CANN/MindSpore 的算子覆盖度在国内是最好的之一，但受制程与生态限制。
- **Google**：Tensor 系列 NPU 与 Gemini Nano 垂直整合最深，是“模型为硬件裁剪”的极端案例。
- **Arm**：Ethos 系列没赢下手机主 NPU，但 KleidiAI + SME2 让 CPU 路线重新变得有竞争力。

</details>

---

## 2. FlashAttention 类算子：能用，但是“编译期版本”

### 已经普遍支持的

- **定长 chunk 的 fused attention**：Q·Kᵀ → mask → softmax → ·V 在一个 kernel 内完成，中间的 attention matrix 不回 DRAM
- **causal mask / GQA / MQA / RoPE**：基本是标配
- **KV cache 作为外部 tensor、seq 维动态**：2024 年后的主流 runtime 都有了

### 仍然缺失或不完整的

| 缺失项 | 为什么端侧 NPU 难做 |
|---|---|
| **任意 / 持续增长的 seq len** | 必须 pad 到预编译的 bucket（512/1K/2K/4K…），跨 bucket 要切换子图 |
| **online softmax 的数值稳定性** | 需要 running max + rescale + exp + 除法。per-tensor INT8 做不了；多数 NPU 走 FP16 attention 路径，或 exp 用 LUT 近似，长 context 下累积误差明显 |
| **PagedAttention / 非连续 KV** | NPU 的权重和激活是**离线 tiling 好的连续布局**，页表式间接寻址需要 DMA gather，而端侧 NPU 的 DMA 描述符能力很弱 |
| **sliding window + attention sink、prefix cache 复用** | 都需要在 kernel 内部做数据依赖的地址跳转 |
| **真正的 tile 级融合** | 需要 Q tile + K/V tile 同时驻留 SRAM；SRAM 不够就退化成多 pass，online softmax 的 rescale 要反复写回 DRAM，收益归零 |

### 工程上的现实解法

行业事实上收敛到了**混合流水线**：

```
prefill  (matmul-bound, 形状固定) ──→ NPU（fusion 收益最大）
decode   (bandwidth-bound, 形状动态) ──→ GPU / CPU
router / 控制流                      ──→ CPU
```

原因很朴素：prefill 阶段 attention 是 $O(n^2 d)$ 计算密集，融合收益巨大；decode 阶段 attention 是 $O(nd)$、纯读 KV cache，NPU 相对 GPU 没有优势，反而被动态形状拖累。所以“NPU 跑 decode 更快”这个直觉在长 context 下基本是错的。

---

## 3. MoE：这才是真正的硬骨头

MoE 比 FA 难，是因为**三个约束同时出现**，而 FA 只踩了第一个：

1. **数据依赖的控制流**（top-k routing）
2. **间接寻址的权重访问**（只读被选中的 k 个 expert）——需要 gather，而 NPU 权重是离线 tiling 的固定布局
3. **batch=1 时 dispatch 固定开销无法摊薄**

第 3 点是端侧独有的。算一笔账：单次 graph submission 在 hexagon 上是几十 µs 到 ~1 ms 量级。一个 32 层、每层 2~4 次 expert 切换的模型，如果走“CPU 跑 router + 每个 expert 一个子图”的朴素实现：

$$
T_{\text{dispatch}} \approx 32 \times 3 \times 0.3\,\text{ms} \approx 30\,\text{ms/token}
$$

这已经吃掉了整个 token 预算（30 tok/s 就是 33 ms/token）。**所以“host 编排 + 子图提交”这条路在 decode 场景直接判死刑**，只能在 prefill（batch 大、摊销得开）用。

### 四种 workaround 及其代价

| 方案 | 做法 | 代价 |
|---|---|---|
| **Dense 化** | 所有 expert 都算，路由结果当 mask | 计算量 × $n_{\text{total}}/n_{\text{active}}$，只在 active 占比 >1/4 时划得来 |
| **Static bucket + capacity factor** | 每 expert 固定容量 C，超出丢弃或溢出到 CPU | 需要 pad，padding 浪费随路由熵增；GPU 上的标准做法，端侧要预编译成固定 shape |
| **图内条件分支 + 间接 DMA** | NPU 原生支持 if/switch 与描述符级地址生成，一次提交内完成路由和数据搬运 | 需要硬件/编译器支持，目前只有部分平台支持，且调度器要能处理变长 |
| **权重常驻 DRAM + expert paging** | 类似 mmap 的按需读入 | 破坏离线 tiling 布局，DMA 效率掉一个档，但**可行且通用** |

### 但这里有个反直觉的正面结论

从 roofline 看，**MoE 在端侧其实是最友好的架构**，只是 dispatch 的实现成熟度没跟上。

batch=1 decode 是纯带宽 bound，MoE 的全部价值就是“减少每 token 要读的权重字节数”：

$$
T_{\text{token}} \propto \frac{n_{\text{active}}}{n_{\text{total}}} \cdot B_{\text{total,4bit}}
$$

一个 8B total / 1B active 的 MoE，4-bit 下每 token 只需读 ~0.5 GB；而 7B dense 要读 ~3.5 GB。**7× 的带宽优势**，在 77 GB/s 的 LPDDR5X 上就是 6.5 ms vs 45 ms 的差别。

这就是为什么 2025 年各家发布都在大力宣传 MoE 和超低比特：**在带宽受限设备上，MoE + 1~4 bit 是唯一能把“模型能力”和“token 速率”同时推上去的组合**。缺口在 dispatch 的实现，不在数学。

---

## 4. 和 Groq / TPU 的差距在哪

先给结论：**差距的主要来源是带宽、batch、可编程性，动态性不是主因。而且 Groq/TPU 并没有“解决”动态 dispatch——它们是从问题定义里把 dispatch 消掉了。**

| 维度 | 端侧 NPU | Groq LPU | TPU (Trillium/Ironwood) |
|---|---|---|---|
| **内存** | LPDDR5X ~77–120 GB/s，容量 12–64 GB（共享） | 230 MB **SRAM**，无 DRAM、无 cache | HBM3e，TB/s 级，容量 100+ GB |
| **batch 假设** | 1（硬约束） | 1（但靠 SRAM 常驻 + 确定性全网） | 大 batch（几十到上千） |
| **调度** | 编译期为主，运行时有图切换 | 纯静态 VLIW，周期精确 | XLA 静态，SPMD 分片 |
| **动态形状** | 有限支持（bucket） | **零支持**，编译期展开 | 静态，靠 batch 摊平 |
| **MoE dispatch** | 主要瓶颈 | 编译期把 expert 全排进 SRAM/网络 | **SparseCore 专门做 all-to-all shuffle** |
| **可编程性** | 黑盒 compiler，无 kernel 语言 | 黑盒，但接口是“整模型编译” | XLA/Pallas，相对开放 |
| **功耗** | 2–5 W，burst 型 boost | 数百 W | 数百 W |

### 逐条拆

**① 带宽差了约两个数量级，这是能量的差距，不是微架构的差距。**
HBM3e 到 ~8 TB/s，LPDDR5X 到 ~0.1 TB/s。每 bit 搬运的能量在封装内 DRAM 和片外 LPDDR 之间差 1~2 个数量级。**这条决定了端侧必须用 4-bit、必须小模型、必须 KV 量化——所有其他设计选择都是它的推论。**

**② Groq 的“动态性”其实是最差的，但它不需要。**
LPU 是 SIMD/VLIW + 静态调度 + 无 cache + 无 DRAM，任何数据依赖的分支都必须在编译期展开。它的做法是把**权重全部驻留 230 MB SRAM、靠确定性片上网络把 expert 静态分布到各芯片**——不是运行时 dispatch，而是编译期就把所有可能路径铺好。所以 Groq 在 MoE 上表现好，恰恰是因为它**放弃了动态性**，用“容量 + 确定性”换取“不需要动态性”。

代价就是：模型必须能被切成小片塞进 230 MB，且批次内 token 的路由分布要被静态容量兜住。

**③ TPU 的 SparseCore 是“跨芯片 all-to-all”问题的解，和端侧不同构。**
Trillium 起加 SparseCore 是因为 expert parallelism 下 token 要在 chip 间 shuffle，这是网络问题。端侧 MoE 是**单芯片内跨 DRAM bank 的 gather**，没有网络，但要求细粒度间接寻址和极低延迟——问题形态完全不同，TPU 的解法搬不过来。而且 TPU 的 MoE 友好性很大程度来自 batch=512 时 padding 浪费可以忽略；batch=1 时不成立。

**④ 真正的差距是软件，不是硅。**
端侧 NPU 没有 CUDA 级的 kernel 语言、没有可靠的 profiler、每个 SoC 一套私有图和布局。“能不能手写一个 fused attention kernel”这个能力，直接决定了 FA 在端侧到底是硬件能力还是纸面规格。

唯一接近“可编程”的是 Hexagon——HVX intrinsics + 可见 VTCM，这也是为什么 llama.cpp 的 QNN/HTP backend 是目前最成熟的端侧 LLM 路径之一。**同一个 SoC 上，手写融合 kernel 和跑通用 compiler，性能能差 2~3 倍**，这个差距比任何 TOPS 差距都真实。

**⑤ 功耗包络不同。**
Groq/TPU 可以持续满载；端侧是 2–5 W 且靠 boost 撑峰值，而 decode 是长时间稳态负载 → 热降频。**你测到的 20 tok/s 通常是前 10 秒的，5 分钟后可能只有 12 tok/s。**

---

## 5. 我的判断

<details>
<summary>短期（1–2 年）：混合流水线是唯一答案</summary>

NPU 做 prefill、GPU/CPU 做 decode 与 attention、CPU 做 router 与控制流，靠 ExecuTorch / CoreML / Genie 这类图分段框架编排。别指望单一引擎跑完整条 LLM 流水线。

</details>

<details>
<summary>中期（2–4 年）：需要三件事同时到位</summary>

1. **原生 MXFP4/MXINT4 + 更细粒度的 block-wise scale**（对齐 GGUF/GPTQ 布局，省掉重排）
2. **图内控制流 + 描述符级间接寻址 DMA**（把 MoE dispatch 收进单次提交）
3. **更大 on-chip SRAM + LPDDR6 / 近存计算**（把有效带宽往上推）

这三件事里，第 2 件是纯软件/编译器工作，也是投入产出比最高的——**同样的硅，MoE token 速率可能有 3–5× 差距**。

</details>

<details>
<summary>评估这类 NPU 时该看什么（别信 TOPS）</summary>

- **固定 context 下的 4-bit 实际 tok/s**，且区分 **prefill TPS** 和 **decode TPS**（两个数经常差 20 倍，宣传只报前者）
- **graph submission 开销**：跑 32 层的模型需要几次提交？
- **KV cache 是否是可寻址的外部 tensor**，seq 维动态到什么程度，跨 bucket 要不要切换子图
- **能否复用 prefix cache**（多轮对话场景直接决定体感）
- **持续负载 5 分钟后的降频曲线**
- **能不能自己写融合 kernel**（有没有 intrinsics / 可见 scratchpad）

</details>

最后一句可能有点反共识：**端侧 NPU 走向“通用化”未必是对的路。** 长期可能是 SoC 级异构赢——CPU 靠 SME2/i8mm/KleidiAI 做小 batch GEMM、GPU 做带宽 bound 的 decode、NPU 只吃 prefill 的大 GEMM，再叠加内存压缩和近存计算。把所有算子都塞进一个“通用 NPU”，等于同时继承了 ASIC 的僵化和 GPU 的低效，而两者都不占。

---
## 🧑‍💻 User

那megakernel、batched prefill 这种动态 dispatch、动态路由、变长序列、稀疏计算，以及运行时负载不均衡的工作负载 npu天生就不适合吗

---
## 😀 默认助手 (7)

# 不是“天生不适合”，是“第一代硬件把可预测性卖得太狠了”

先把结论摆出来：**这六类 workload 里，只有一类是 NPU 结构上真不擅长的，其余五类是工具链和少数几个硬件特性缺位，不是物理不可能。**

而且有个讽刺的地方：**megakernel 这个技术方向，本质上是 GPU 世界在向 NPU 的执行模型靠拢**，不是反过来。

---

## 1. 先建立一个判据：动态性的“粒度 × 可枚举性”

“动态”不是二元属性。决定 NPU 能不能承受的，是三个量：

| 维度 | NPU 友好 | NPU 敌对 |
|---|---|---|
| **粒度**（一次动态决策覆盖多少计算） | 每 expert / 每 block / 每 64 token | 每 token / 每元素 |
| **可枚举性**（运行时可能的分支数） | 8–64 个可枚举分支 | 开放集合、数据依赖递归 |
| **绑定时刻**（决策何时可知） | 单次提交内可知（base address、trip count） | 需要重编译或重新提交图 |

关键洞察：**NPU 不怕“数据依赖”，怕的是“细粒度的数据依赖”。** 因为地址生成、分支判定、队列管理这些开销在 NPU 上无法用 TLP 摊薄——它故意没做 latency hiding，这是它能效的来源（下面第 6 节展开）。

用这个判据把这六类重排一下：

```
                    ┌─ 计算密集 ─────────────┬─ 带宽密集 ──────────────┐
  静态/可枚举        │  dense prefill   ★最舒服 │  dense decode      ★舒服 │
                    │  batched prefill ★舒服   │  KV 读取           ★舒服 │
                    ├─────────────────────────┼─────────────────────────┤
  动态/粗粒度        │  block-sparse attn  ○可以│  MoE decode        ○可以│
                    │  MoE prefill        ○可以│  sparse attn decode○可以 │
                    ├─────────────────────────┼─────────────────────────┤
  动态/细粒度        │  unstructured sparse GEMM │  token-level dynamic    │
                    │  ✗ 真不适合              │  ✗ 真不适合              │
                    └─────────────────────────┴─────────────────────────┘
```

**只有左下和右下（细粒度动态 + 计算密集）是真不适合，而且这一格对 GPU 也同样糟糕。**

---

## 2. Megakernel：NPU 才是这个思路的原生地主

Megakernel 要解决的是“kernel launch 开销 + 层间气泡”，做法是把整模型塞进一次提交，用 persistent block + 软件调度循环。它的四个必要条件：

| 条件 | GPU | 端侧 NPU |
|---|---|---|
| (a) 全设备范围的持久化 block | 需要 grid 恰好占满 SM | **天然满足**——核数就 1–16 个，全占满不用调 |
| (b) 设备级 barrier / 核间同步 | 有 cooperative groups / mbarrier | **弱**，很多平台没有 kernel 内设备级 barrier |
| (c) 工作队列（动态部分靠它拉活） | 有 atomic work-stealing | **基本没有**——这是最大缺口 |
| (d) 廉价的块间通信 | L2 相干 + 高带宽 | **差**——只能走 DRAM scratch，且带宽是瓶颈 |

所以 NPU 在 megakernel 上是 **(a) 大赢、(c)(d) 大亏**。

但这里有个非常关键的推论：**因为 NPU 核少而大，它的最优 megakernel 形态根本不是“多 block 抢任务”，而是“空间切分的层流水线”**——

```
core0: layer 0-7  → core1: layer 8-15 → core2: layer 16-23 → core3: layer 24-31
         ↕ ring buffer (DRAM)          ↕                    ↕
```

这是个**纯静态调度**，恰好是 NPU 的主场；而且 decode 阶段 batch=1 并行度极低，流水线是唯一能喂饱多核的方式。换句话说：**decode 的 megakernel 在 NPU 上是“天然的”，在 GPU 上反而要费劲用 persistent kernel 去模拟。**

**真正把这条路堵死的是 (c)。** 没有 work queue，MoE 路由、varlen batch 就只能靠在编译期把所有分支展开成一张巨大的静态图——这正是 Groq 干的事（230 MB SRAM 常驻 + 全确定性网络 + 编译期铺好所有 expert 路径）。**Groq 不是解决了动态 dispatch，它是把 dispatch 从运行时删掉了。** 代价是 230 MB 容量上限和极端的模型形态约束。

反向的存在性证明是 **Tenstorrent**：每个 tile 一个 RISC-V 小核 + 可编程 NoC，动态路由和稀疏是软件问题。它证明了“NPU 式的大 MAC 阵列 + 可编程数据流”完全可以做动态负载——代价是程序员要自己写数据流，也就是回到了 CUDA 的痛点上。

---

## 3. Batched prefill / 变长序列：这是最不该担心的一项

Prefill 是 **NPU 最舒服的场景**：大 GEMM、算术强度高、形状可预测。

变长序列的三条现实出路，都不需要硬件改动：

| 方案 | 代价 |
|---|---|
| **Bucket + mask**（按 64/128/256 分桶，pad 到桶上界 + attention mask） | 浪费 $\approx$ 桶粒度 / 平均长度，典型 10–20%，可接受 |
| **Sort-by-length batching**（同长度优先成批） | 调度复杂度，尾批退化 |
| **Chunked prefill**（固定 chunk 长度切分长序列） | 需要跨 chunk 的 KV 拼接，但形状恒定 |

**Continuous batching（iteration-level scheduling）在 NPU 上的标准解法是“固定 batch bucket + 空槽 mask”**：永远按 batch=4 跑，实际只有 2 个有效序列时空槽被 mask 掉。代价最多 2×，换来的是形状恒定、零重编译。这是所有端侧 LLM runtime 的实际做法。

真正会痛的只有一种：**varlen attention 不 pad 直接跑**（FlashAttention 的 `cu_seqlens` 风格）。这需要 kernel 内的数据依赖 loop trip count，NPU 的循环边界是编译期常量。但在 batch 不大时，“按桶 pad”几乎总是更划算——**这是工程上被解决了的**。

---

## 4. 动态路由（MoE）：硬件要求比看起来低得多

上一轮说过 dispatch 的开销问题，这里补一个更本质的点：

**如果权重按 expert-major 连续布局，MoE 路由对硬件的全部要求就是“一个 base address 可变、运行时可计算的 DMA 描述符”。**

不是任意 gather。只是：

```
descriptor.expert_base = router_output → expert_id_table[k]   // 一个标量
descriptor.count       = tokens_per_expert[e]                  // 一个标量
```

这是 NPU 上**最便宜的那类硬件特性**——比 vector unit 便宜，比 in-graph branch 便宜。很多 NPU 的 DMA 硬件其实能读运行时寄存器算地址，只是软件层没暴露。

真正的坑在这三处：

1. **token scatter/permute**（batch>1 时要把 token 按 expert 分组）——需要在提交内做一次 sort/permute。CPU 或 vector unit 都能做，但要在 kernel 内、不能跳出去。
2. **capacity padding 的浪费**。用容量因子 $C$ 截断后：
$$
\eta_{\text{pad}} = \frac{\sum_{e} \left\lceil t_e / M_{\text{tile}} \right\rceil M_{\text{tile}}}{T \cdot k}
$$
其中 $M_{\text{tile}}$ 是 tensor core 的 M 维 tile（16 或 64）。$t_e$ 小时 $\eta$ 迅速恶化——**expert 越多、batch 越小，padding 越浪费**。这就是为什么小 MoE + batch=1 是最坏组合。
3. **权重布局与离线 tiling 的冲突**——NPU 的权重 blob 是离线排好序的，如果 runtime 要按 expert 重排，就得放弃预 tiling。

一旦这三个解决（本质是“把 expert 当 N 维分块的大 GEMM”），MoE 在 NPU 上就是**带宽 roofline 上的最优解**，因为 batch=1 decode 本来就是纯带宽 bound，MoE 的全部价值就是减少每 token 权重字节。

---

## 5. 稀疏计算：必须分清三种，混在一起谈是错的

这是最容易被“NPU 不支持稀疏”这句话糊弄的地方。

| 类型 | NPU 可行性 | 说明 |
|---|---|---|
| **Unstructured weight sparsity（细粒度剪枝）** | ✗ 死路 | 稠密 MAC 阵列不会因为操作数是 0 而少走时钟周期。**这一条对 GPU 同样成立**——Ampere/Hopper 的 2:4 稀疏标称 2×，实测 1.3–1.5×，且模型质量掉点。不是 NPU 特有问题 |
| **Structured / block sparsity（2:4、16×16 block、1:2）** | ○ 可以 | 硬件零跳过 + 硬件 gating 能拿到 1.5–2×，前提是 MAC 阵列支持 operand gating |
| **Sparse attention（block-sparse、sliding window + sink、NSA 类）** | ○ 可以 | 关键是**按 block 粒度 gather**——粗粒度 gather = 便宜。不需要 element 级间接寻址 |
| **Token-level dynamic sparsity / early exit / per-token 跳层** | ✗ 最痛 | 每个 token 工作量不同 → 直接变成负载不均衡问题 |

所以“稀疏计算不适合 NPU”这个说法，只在**细粒度、逐元素**这个区间成立，而那个区间本来就没有硬件能做好。

---

## 6. 负载不均衡：这是唯一一条“结构性”的短板

前面五项都能靠工程解掉，这一项是真正的架构取舍。

**GPU 靠 oversubscription 吸收不均衡**：132 个 SM、几万个 warp，硬件 scheduler 从队列里拉活，任何一个 warp 卡住就有别的顶上来。不均衡被调度器吃掉了。

**NPU 没有这个能力，而且是故意的。** 它的能效正来自“不做 latency hiding”：

- 不做寄存器堆超配（register file 小一个数量级）
- 不做大量 in-flight 线程（concurrent context 通常 1–2 个）
- 不做 scoreboard 重排 / miss buffer
- 静态调度，编译器把延迟算准，硬件不猜

Little's law 说得很直接——要掩盖延迟 $L$ 达到吞吐 $R$，需要

$$
N_{\text{in-flight}} \geq R \times L
$$

GPU 用几千个 warp 把这个 $N$ 堆到很大；NPU 的 $N$ 小，所以**任何不可预测的停顿都直接暴露成周期浪费**。

这就是全部问题的根源：**动态工作负载需要的正是“可预测性之外”的冗余并行度，而 NPU 恰恰是靠删除这个冗余换来能效的。** 这是设计取舍，不是疏忽。

### 但负载不均衡可以“转化”掉

关键技巧：**把“时间维度”的不均衡转成“M 维的 padding”**。

MoE 的 `tokens_per_expert` 不均衡，但因为 GEMM 的 M 维本来就要 pad 到 $M_{\text{tile}}$，只要不均衡幅度小于 tile 粒度，**代价是零**。所以：

- 不均衡落在 **M 维**（每 expert 的 token 数）→ 便宜，被 tile padding 吸收
- 不均衡落在 **kernel 时长**（一个 persistent kernel 内某些核要比别人多跑一倍）→ 昂贵，因为核数少，没有第三方来填空闲

**这就是为什么“按 expert 分组 + 每个 expert 一个 GEMM”的 MoE 在 NPU 上是可行的，而“按 token 动态跳过某些层”是不可行的。** 前者把不均衡塞进了 M 维，后者没法塞。

代价公式（$C$ 个核，$n_e$ 个 expert 的任务量）：

$$
\text{有效利用率} \;\approx\; \frac{\sum_e n_e}{C \cdot \max_e n_e}
$$

$n_e$ 越不均、$C$ 越少，越接近 $\max$ 那里的 1/C。所以**端侧多核 MoE 的尾效应是实打实的成本**，只能靠容量因子截断把分布压平（用质量换利用率）。

---

## 7. 两边正在收敛，但带宽不会

把 GPU 和 NPU 这几年的新增特性列在一起看：

| 能力 | GPU 的加法 | NPU 的加法 |
|---|---|---|
| 描述符式异步拷贝 | TMA / `cp.async.bulk` | descriptor DMA（有，但不给用户用） |
| 软件管理 scratchpad | distributed shared memory / cluster | VTCM / on-chip SRAM（**这件事 NPU 一直领先**） |
| 生产者-消费者流水 | warp specialization | vector unit + 双缓冲 |
| 控制流 | CUDA Graph 条件节点 / device-side graph launch | in-graph if/switch |
| 动态形状 | 重新 capture graph | dynamic shape（bucket 级） |
| 稀疏 | 2:4 structured | 部分 block sparsity |

**收敛点很清楚：软件管理的 scratchpad + 描述符异步拷贝 + 可编程数据流 + 恰好够用的并发。** 两边都在往这个中间点走。

而**不会收敛的是**：HBM 到 ~8 TB/s vs LPDDR 到 ~0.1 TB/s 的带宽差、容量差、以及每 bit 搬运的能量差。这才是端侧 NPU 永久性的约束，也是所有设计选择（4-bit 权重、MoE、KV 量化、小模型）的根源。

所以正确的表述是：

> **动态性差距是“工具链 + 3~4 个便宜的硬件特性”的问题，正在关闭；带宽差距是物理的，不会关闭。**

---

## 8. 一个“动态友好”的 NPU 需要什么（按性价比排序）

1. **描述符级、运行时可计算 base/stride 的 DMA，且能由 kernel 内部发起** ← 性价比最高，几乎纯软件
2. **kernel 内 device-wide barrier / producer-consumer 同步** ← megakernel 的前提
3. **可编程 sequencer（每 MAC 阵列一个标量核）** ← Hexagon 有，这是它最像 LLM 加速器的原因
4. **固定 batch bucket + 空槽 mask 的 runtime 支持** ← 纯软件，continuous batching 的答案
5. **多 context（2–4 个并发图）** ← 补回最基础的 latency hiding，代价是会牺牲一点能效
6. **operand gating / 零周期跳过** ← structured sparsity 的前提
7. **work queue / persistent kernel 模型** ← 最贵，也是唯一真正需要改微架构的一条

第 1、2、4 条加起来，就能吃掉 MoE 和 varlen 的绝大部分痛点，而这几乎是纯软件工作量。

<details>
<summary>一段可能反直觉的收尾判断</summary>

**“NPU 不适合动态 workload”这句话，在 2026 年大致等价于“端侧 NPU 的编译器还不成熟”。**

真正结构性的短板只有两条：负载不均衡（因为放弃了 latency hiding）和细粒度间接寻址（因为 DMA 由编译器而非 kernel 驱动）。其余都能靠 bucket + mask + 粗粒度 gather + M 维 padding 转成有界浪费。

反过来说，**如果端侧只做“稠密、静态、大批量 prefill”，NPU 已经赢了**。所有争论都集中在 batch=1 的 decode 和 MoE 路由上——而这恰好是“能效优先”和“动态性优先”这两个设计目标正面相撞的地方。选哪个，取决于你认为端侧的瓶颈是 5W 的功耗墙，还是 30 tok/s 的体验墙。目前看，业界的选择是**先撞功耗墙**，所以 NPU 继续赢，动态性靠软件硬扛。

</details>
