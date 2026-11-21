# 端侧NPU对MoE支持与专用芯片差距

## 🧑‍💻 User

说起来 大部分移动端的NPU在2023年以前基本上都是固化Conv算法ASIC类似物
现在的端侧NPU发展的怎么样了
对于moe fa这类需要细粒度数据流依赖 动态dispatch的算子支持情况怎么样？
和像groq、tpu这类专门针对llm infer的npu差距在哪里

---
## 😀 默认助手 (7)

## 简要结论

- **架构上**，移动 NPU 过去两年已经不再只是“Conv ASIC 类似物”了，实际上变成了**「可编程 2D MAC 阵列 + 向量单元 + 可编程 DMA」的小型张量加速器**，微码层面能表达 GEMM、softmax、gather。但**调度模型仍然是编译期静态的**，这是后面所有问题的根源。
- **MoE**：能运行（Qwen3-30B-A3B 这个量级已能在手机、AI PC 的 NPU 上运行），但动态 dispatch 基本由 **CPU/DSP 代替执行 + 静态化补齐**，并非硬件原生能力。代价是每 token 几十到几百次 host 往返。
- **FlashAttention**：关键限制不在“动态”，而在**融合**（epilogue exp、跨迭代状态、KV tile 常驻 SRAM）。
- **和 Groq/TPU 的差距**：核心限制不是算力，而是 ① SRAM/FLOP 比 ② 调度模型 ③ 编译栈表达力 ④ 带宽与功耗限制。而且 Groq 和 TPU 自身在 MoE 动态性上也有明显弱点，“专用 LLM 硬件”并不必然占优。

---

## 一、移动 NPU 这几年实际的变化

大致分三代：

| 阶段 | 形态 | 代表 |
|---|---|---|
| 2017–2020 | 固化流水线：Conv→BN→Act→Pool 的 fixed-function chain，Winograd、line buffer | 早期达芬奇 / 寒武纪 IP、A11 ANE、Hexagon HVX |
| 2021–2023 | 可编程张量核，支持任意 MatMul，但编译器仍是 ONNX 分区 + 静态 tiling | 8G1/8G2 的 HTA、APU 6xx/7xx |
| 2023–2025 | 为 LLM 重构：更大片上 SRAM、原生 INT4 MAC、attention fusion、动态 shape、KV cache 进 runtime | 8 Elite、D9400/9500、A18/M4 ANE、Tensor G5 |

**但有三个条件没有改变**，它们决定了后续所有结论：

1. **片上 SRAM 很小**：几 MB 到几十 MB。TPU 的 VMEM 是几十到上百 MB，Groq 是 230 MB SRAM + 无 DRAM——差一到两个数量级。
2. **带宽是共享的**：LPDDR5X 大约 60–85 GB/s，要和 CPU/GPU/ISP/显示一起争抢。
3. **调度是静态的**：编译器把图切成 tile、分配 scratchpad、生成 DMA 描述符链，运行期基本不重排。

先算一下 decode 的上限：

$$T_{\text{decode}} \approx \frac{W_{\text{bytes}}}{BW_{\text{eff}}}$$

4B 模型 INT4 ≈ 2 GB，$BW_{\text{eff}}$ 取 40 GB/s（已经能跑到理论值的六成）→ **约 20 tok/s**。这就是当前旗舰机 demo 数字的天花板，和 NPU 标称的 40–50 TOPS 几乎无关。**移动端 LLM 是纯带宽问题，不是算力问题。**

---

## 二、MoE：能运行，但 dispatch 是关键弱点

### 核心矛盾

router 是 data-dependent 的——选哪些专家、每个专家收几个 token，运行期才知道；而 NPU 的 tiling、scratchpad 布局、DMA 链全是编译期写死的。这正好是“静态数据流硬件 vs 动态稀疏计算”的经典冲突。

### 现在实际在用的四种做法

**1. dense 化 / padding（最简单）**
所有专家都算，用 mask 屏蔽无关专家。top-2/8 就是 4× 无效 MAC。但注意：在 decode 场景**权重反正要全读**，所以带宽上并不额外浪费，只有 prefill 阶段吃亏。很多 runtime 第一版就是这个。

**2. host orchestration（目前主流）**
CPU/DSP 算 router → argsort → 按专家分桶 → 逐专家发 GEMM 给 NPU。问题很直接：

- 每次 launch 有 µs 级往返 + 同步 stall；24 层 × top-2 ×（gather/scatter/GEMM）轻松到几百次/token → **ms 级固定开销**
- batch=1 时每个专家的 GEMM 只有 1–2 行，阵列利用率掉到个位数百分比
- 所以真正缺的原语是 **grouped GEMM / variable-M GEMM**，这几年厂商在补，但成熟度参差

**3. 片上动态控制流 / 动态 shape**
新一代 NPU 加了 conditional branch、indirect DMA、动态 shape，能让“选专家”这个决策留在设备上。但粒度粗——通常是**层级别**的分支，不是 token 级别。离真正的 token-wise 动态 dispatch 还差得远。

**4. 异构分工（Qualcomm 的思路最能说明问题）**
Hexagon 里 **HVX（SIMD，可跑任意控制流）做 routing / gather，张量核做 GEMM**，靠共享片上 SRAM 传数据。本质上就是“拿 DSP 当 dispatch 引擎用”。这也解释了为什么移动端 MoE 的可行性高度依赖 SoC 里有没有一块通用向量核。

### 一个反直觉但重要的点

在手机这种 batch=1、带宽受限的场景，**MoE 的收益其实比在服务器上更直接**：每 token 只读 active expert 的权重。30B-A3B 每 token 读约 1.5–3 GB，而 30B dense 要读 15 GB——**这是一个数量级的带宽优势**。

所以厂商有强动机做这件事。难点不在“算力浪费”（移动端本来就不缺算力），而在于：

- dispatch 开销能不能压到可忽略
- 有没有办法**真的不碰 inactive expert 的权重**（否则带宽优势就没了）

### 真正缺的能力

**token 级细粒度、跨核的专家并行**。TPU 靠 all-to-all + expert capacity padding 解决，但那是几千芯片规模的 collective；单 die 上这套根本不成立。移动端只能退化成“分桶串行 + 一堆小 GEMM”，专家并行的好处只能吃到一部分。

---

## 三、FlashAttention：问题在于融合，动态度不是主要障碍

FA 对硬件的真实要求是四条：

1. **tile 级 Q/K/V 常驻 SRAM**，$S = QK^\top$ 分块后不落到 DRAM
2. **exp / max / sum 能在 MMA 输出离开阵列之前完成**（epilogue fusion）——否则一次 DRAM 往返就白做了
3. **循环携带标量状态 $(m, l)$ + 在线 rescale 累加器**，这是 loop-carried dependency，需要流水线支持
4. 任意 `seq_len` / `head_dim` 的 tiling，以及长 context 下 KV cache 的分页/共享

移动 NPU 的架构传统恰好卡在第 2 条：早期把 softmax 放在独立 fixed-function 或向量单元，中间结果必须 materialize。2023 年之后各家都在补 fused attention 算子，但现状是：

- 多数是**厂商手写、针对固定 shape 模板**的 kernel，不是编译器从 $QK^\top \to \text{softmax} \to PV$ 自动 fuse 出来的
- 长 context（>4–8k）仍容易退化成分块 + 落 DRAM，或者直接 fallback 给 GPU
- INT8 的 $QK^\top$ 动态范围难处理，很多实现退回 FP16 attention、只量化权重

**对比**：TPU 有紧挨 MXU 的独立向量单元、VMEM 够放 tile、XLA/Pallas 能做软件流水，FA 是“顺理成章”；Groq 的 SRAM 带宽 80 TB/s，230 MB 三微秒读完，attention 根本不是瓶颈。移动端则是“能拼出来，但每一块都要人工调”。

---

## 四、和 Groq / TPU 的差距在哪

| 维度 | 移动 NPU | Groq LPU | TPU (v4/v5p) |
|---|---|---|---|
| 调度模型 | 编译期静态 tiling + 运行期 host 编排 | 全静态、cycle-deterministic | 静态 + XLA 全局优化 + 动态 shape（padding） |
| 片上存储 | 几 MB ~ 几十 MB | 230 MB SRAM，**无 DRAM** | 数十 MB VMEM + 32–95 GB HBM |
| 带宽 | 60–85 GB/s 共享 LPDDR | 80 TB/s SRAM | 1.2–2.8 TB/s HBM |
| 算力 | 30–50 TOPS INT8，2–3 W | ~750 TOPS INT8，~300 W | 275–459 TFLOPS BF16，200–400 W |
| 精度 | INT8/INT4 为主，FP16 受限 | INT8/FP16 | BF16/FP8 原生 |
| 动态性 | 粗粒度分支；gather 靠 DMA 描述符 | 无 cache、无动态调度，MoE 靠 predication | all-to-all + capacity padding |
| 编译栈 | ONNX 分区 + 闭源 SDK | 自有编译器，全图静态调度 | XLA/JAX，开源可改 |

**差距的本质是三个比值：**

1. **$\text{SRAM}/\text{FLOP}$**。Groq 和 TPU 都是“存储富裕”型，FA、长 KV、MoE 权重缓存天然舒服；移动端是“存储贫乏”型，一切都要为 tiling 让步。
2. **$\dfrac{\text{dispatch 延迟}}{\text{单 op 时长}}$**。batch=1 decode 的单个 GEMM 只有几十 µs，host 往返 10 µs 就是 20–30% 开销。Groq 用编译器把整图排成一条流水线，甚至可以说**消除了 dispatch**；TPU 用 XLA 把整个 step 编成一个 program（近似 CUDA-graph 式 launch）。
3. **编译栈表达力**：能否表达 dynamic control flow、scatter/gather、persistent kernel、跨 op 大融合。这是移动 NPU 最大的短板，而且**是软件问题，不是硬件问题**。

**但要公平**：Groq/TPU 在 MoE 上也不舒服。Groq 的全静态调度遇到 token 级路由，只能 predict + 广播 + 空转；TPU 的解法依赖大规模 collective，单 die 上 capacity padding 的浪费照样存在。所以“专用 LLM 硬件”的优势集中在 **dense、大 batch、context 固定**的场景——恰好是手机最不具备的三个条件。

---

## 五、容易被忽略的一点：移动 NPU 的真正对手是 GPU

手机上 LLM decode，**GPU 目前经常超过 NPU**——Adreno/Mali/Apple GPU 有统一内存、成熟 kernel（llama.cpp/MLC）、能自然表达动态控制流。

NPU 的优势是能效（TOPS/W），但要靠厂商 SDK 把**每个模型手工调优**才能兑现，泛化性差。真实格局是：

> **NPU 跑“认证过的”模型，GPU 运行“其它一切”**——MoE、长 context 这类新结构，通常先落到 GPU 上验证，再等 SDK 跟进。

所以讨论“NPU 能不能跑 MoE”时，实际在问的是“**厂商有没有为这个具体模型写 kernel**”，而不是“硬件是否支持”。

---

<details>
<summary><b>六、接下来 1–2 年值得关注的（点击展开）</b></summary>

- **persistent kernel / megakernel**：整层甚至整模型只 launch 一次，把 dispatch 开销压到接近零——这是补 MoE 最现实的路径，也最不需要改硬件
- **原生 grouped / variable-M GEMM** + 片上 gather-scatter 单元
- **expert 权重预取**：router logits 一出来立刻 prefetch top-k 专家到 SRAM，与当前计算重叠
- **per-expert 混合精度**：不重要的专家用 2–3 bit，甚至三值化（BitNet，D9500 已经在推这个方向）
- **编译器出现"带控制流的图 IR"**（类似 IREE 的 stream/flow dialect），而不是 ONNX 分区那一套——这是结构性变化
- **KV cache 压缩 + 极端量化（3.5-bit）**：因为**带宽才是那堵墙**，不是调度

</details>

---

**一句话总结**：计算单元本身没问题，移动 NPU 已经算“像样的张量加速器”了；真正没跟上的是**运行时和编译栈——它们还活在静态图世界里**。MoE 和 FA 难的不是算不动，而是硬件缺乏表达“数据相关的数据移动”的原语，因此只能靠主机端反复往返，把开销一层层堆上去。而这个差距，恰好是 Groq/TPU 通过“更大的 SRAM + 更静态但更全局的调度 + 能改的编译栈”绕过去的。
