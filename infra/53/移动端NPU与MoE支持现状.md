# 移动端NPU与MoE支持现状

## 🧑‍💻 User

说起来 大部分移动端的NPU在2023年以前基本上都是固化Conv算法ASIC类似物
现在的端侧NPU发展的怎么样了
对于moe fa这类需要细粒度数据流依赖 动态dispatch的算子支持情况怎么样？
和像groq、tpu这类专门针对llm infer的npu差距在哪里

---
## 😀 默认助手 (7)

# 端侧 NPU 在 2026 年的真实位置

先给出结论，再逐项说明细节：

> 端侧 NPU 这几年的进步，主要在于**可编程性**，而不是**峰值算力**。架构上它基本演变成了“标量核 + 向量核 + 矩阵核 + DMA + 片上 scratchpad”。但它保留了一个致命的设计假设：**批量密集矩阵乘**。而 LLM 的 decode 是 batch=1 的**带宽饥渴 + 稀疏访存**，这两件事在 roofline 上相差两个数量级。所以 MoE / FlashAttention 在端侧 NPU 上的支持，本质上是“用编译期技巧把动态性抹平成静态的”，而不是“硬件真的支持动态”。

---

## 一、先修正一个前提

你说“2023 年以前基本是固化 Conv 算法 ASIC 类似物”，这个判断**在编程模型层面是对的，在硅片层面不准确**。

<details>
<summary>展开：为什么“ASIC”这个词要打个折</summary>

真正的“Conv 硬连线 ASIC”是 2016 年那一代 IP（寒武纪 DianNao 系、早期 Movidius Myriad 的 SHAVE、以及一堆只支持 `conv/pool/relu` 有限算子集的小 NPU）。但主流移动 NPU 从 2017 年起就**不是硬连线**了：

- **Apple ANE**（A11 起）一直是可重构的数据流引擎 + 16 个 core + 大 SRAM，CoreML 编译的是任意 DAG。
- **Qualcomm Hexagon** 从来就是 VLIW DSP + HVX 向量协处理器，本质是可编程的。
- **Arm Ethos / MediaTek APU** 是 MAC 阵列 + 有限指令集。

所以更准确的说法是：**它们是“算子集固化的、以 Conv 形状为优化目标的 MAC 阵列 + 闭源编译器”**。硅片可编程，但**编译器只愿意为你生成它见过的形状**：NCHW、固定 H/W、无动态 shape、无 KV cache、无自回归。你从编译器那一侧感受到的“固化”，是对的。
</details>

转折点是 2024–2025：LLM 端侧落地的实际需求，迫使 NPU 从“视觉图加速器”改造成“能跑 autoregressive 的可编程加速器”。改造清单包括：动态 shape、KV cache 驻留、int4 weight-only 量化、大 M 变化的 GEMM、以及（关键的）允许**自定义 kernel**。

---

## 二、现在的端侧 NPU 到底变成了什么

架构上，2026 年的端侧 NPU 已经收敛成固定的四部分：

| 部件 | 作用 | 对应 LLM 里的什么 |
|---|---|---|
| 标量/控制核 | control flow、间接寻址、循环 | router 的 argtopk、index 计算、permute |
| 向量核 (SIMD/VPU) | elementwise、reduce | softmax、LayerNorm、RMSNorm、激活、量化/反量化 |
| 矩阵核 (MAC array) | 稠密 GEMM | QKV/O/FFN 的 projection |
| DMA / 张量搬运 | 显式 double-buffer | 权重、KV cache 的 load |

编程模型分三档，能力差距非常大：

<details>
<summary>展开：三档编程模型的差别（这是选型的真正分水岭）</summary>

1. **纯闭源算子库**：`CoreML` + ANE、`QNN/HTP`、`NeuroPilot`。你只能跑 op set 覆盖到的图。遇到不支持的算子，就整段 fallback 到 CPU，触发一次**张量往返 DRAM**，代价常常超过那个算子本身。
2. **半开放：可写自定义 op**：QNN 的 `op package`（要自己按 VTCM 容量切 tile、自己写 HVX intrinsics）、Arm 的 Vela（编译期固定 shape）、Apple 的 **Core AI** / MIL。
3. **事实上的赢家：绕过 NPU，改用 GPU**：llama.cpp 的 `ggml-hexagon` backend 直接调用 HVX，Apple 的 MLX 直接调用 Metal。

第 3 档不是权宜之计，而是**方向**。最硬的证据是 Apple 自己在 M5 上的选择：给**每个 GPU core 塞一个 Neural Accelerator**（片上矩阵单元）。结果是一个 2026 年的公开实验测出来的（M5 Pro，Qwen3-4B，fp16，输入 500–16K）：

- GPU 在**所有**测试长度上都比 ANE 快：decode 快 $2.0\text{–}3.9\times$，prefill 快 $2.9\text{–}15.9\times$；
- 而能耗并没有按速度反比变化。

也就是说：**在被自己的专用 NPU 服务了八年的场景上，Apple 用“带 tensor 单元的通用 GPU”赢了**。这个信号比任何 TOPS 数字都重要。
</details>

### 真正的瓶颈：端侧 NPU 的算力带宽比根本不适合 LLM

这是我认为最值得记住的一条。定义机器平衡点：

$$\text{Machine Balance} = \frac{\text{峰值算力}}{\text{内存带宽}}\quad [\text{op/byte}]$$

而 batch=1 的 decode，每读一个权重参数只做 $2\text{ FLOP}$（一次乘、一次加，只服务一个 token），所以算术强度：

$$AI_{\text{decode}} = \frac{2P}{(b/8)\,P} = \frac{16}{b}\ \text{FLOP/byte}\quad(b=\text{每参数量化位数})$$

int8 时 $AI=2$，int4 时 $AI=4$。现在把三者的平衡点摆在一起：

| 平台 | 算力 | 带宽 | 平衡点 | 与 $AI_{\text{decode}}$ 的差距 |
|---|---|---|---|---|
| 手机 NPU (int8) | ~40 TOPS | ~77 GB/s | **~520** | 差 **260×** → 99.6% MAC 空转 |
| TPU v7 Ironwood (FP8) | 4614 TFLOPS | 7.37 TB/s | ~626 | 差 ~150×（靠 batch 拉到几百才回本） |
| Groq LPU (int8) | ~750 TOPS | ~80 TB/s 片上 | **~9.4** | **几乎刚好**，可支撑 batch 1–4 |

这张表一次性解释了三件事：

1. **手机 NPU 的 TOPS 对 LLM decode 毫无意义**。40 TOPS 的真实含义是“在访存受限的负载上，40 TOPS 里能用上不到 1%”。端侧 LLM 的唯一有效指标是 `GB/s × 有效率`。
2. **Groq 不是“专用 LLM 芯片”这么模糊的说法，而是“平衡点被精确调到 batch-1 自回归”的芯片**。它牺牲一切换片上带宽，是因为它算准了 decode 的 $AI\approx 2\text{–}4$。
3. **TPU 的平衡点是给大 batch 训练/prefill 调的**，所以它必须靠极致的 batch 和 grouped GEMM 才能不浪费。

所以正确的架构判断是：**decode 应交给高带宽、低算力的路径，prefill 应交给高 MAC 密度、高算力的路径**。PowerInfer-2 的设计（prefill 交给 NPU、decode 交给 CPU、UFS 流式加载）恰好符合这个判断，并非巧合。

---

## 三、MoE：五个卡点，以及端侧怎么暂时应对

MoE 对硬件的需求可以精确地拆成五件事，端侧 NPU 一个都不擅长。

### 1. Router 的 gather/scatter：端侧没有 SparseCore

top-k 路由之后要把 token 重排成 $(\text{kn},\, d)$ 的密集 batch，这一步是**纯 scatter**：index 在标量核上算，靠 DMA 做间接 copy，每个小 copy 都是 $\sim d\times 2$ byte 的碎块。在 128-bit 总线、有最小 burst 长度的 DRAM 上，这种访问的有效带宽经常掉到峰值的 $1/3$。

**端侧的实际做法**：多数时候**干脆不做 dispatch**。三个退让方案：
- **dense-over-all-experts + mask**：算所有 $E$ 个专家再 mask 掉未选中的。算力 $\times E$，换掉所有 scatter。$E=8$ 时还勉强能接受。
- **top-k 定长打包**：不管每个 expert 收到几个 token，都补到固定的 $M$。
- **shared expert 加大占比**：很多“端侧友好 MoE”其实是 1 个大 shared expert + 一堆小专家，实际变成“dense + 一点点稀疏”。

### 2. 动态 shape：用编译期桶化模拟

端侧 NPU 编译器要靠静态 shape 决定 VTCM 分配和 tile 切分。而 MoE 的 $M$（每个 expert 实际的 token 数）只有运行时才知道。

**实际做法**：编译 $\{1,2,4,8,16,32,64\}$ 个 shape 桶，运行时选桶 + padding。或者每个 expert 编译一个固定 shape kernel（$E$ 份 code，指令 cache 压力）。这是**用编译期多版本编码来模拟运行时动态**，本质上是把动态性推给了二进制体积。

### 3. 负载不均衡：静态调度的最慢者决定阶段耗时

端侧只有一个 die，没有 expert parallelism 可调。只能 capacity factor + drop/pad。这条在 batch=1 时反而没那么严重（因为本来就几乎没有可并行的 token）。

### 4. 权重搬运：这才是决定性的瓶颈

算一下：3B 激活参数在 int4 下每 token 要读 $3\times10^9 \times 0.5\text{ B} = 1.5\text{ GB}$。手机有效带宽按 $77\text{ GB/s} \times 0.6$ 算：

$$t_{\text{token}} \approx \frac{1.5}{46} \approx 32\text{ ms} \Rightarrow \sim 31\text{ tok/s}\ (\text{理论上限})$$

对比一个 dense 8B int4（4 GB/token）：同样带宽下只有 $\sim 11.5\text{ tok/s}$。

**这就是 MoE 对端侧的真正吸引力：质量是 30B 级，速度还比 8B dense 快。** 但障碍在于**总权重 15 GB 装不进手机的 12–16 GB DRAM**，必须从 UFS 流式加载（顺序 ~3–4 GB/s，随机访问更差）。

于是端侧 MoE 的问题在物理上等价于：**如何在 UFS 带宽下，让每 token 只读 1.5 GB 的“活跃专家子集”，并且命中率高。**

这正是 PowerInfer-2 的思路（手机，TurboSparse-Mixtral-47B，11.68 tok/s，2024）：把矩阵算子分解成 **neuron cluster**，按激活的幂律分布分出热/冷 neuron，热的常驻、冷的走 UFS，用极细粒度的 I/O-compute pipeline 把加载藏在计算后面。

### 5. 量化敏感度

Router 是 MoE 里最不能量化的部分：int4 下 top-k 排序会翻转，导致整个 token 走错专家，误差被放大而非平均掉。所以端侧 MoE 的实现几乎都保留 router 为 fp16/bf16，只量化 expert 的权重。这会额外引入一次精度切换和格式转换。

---

## 四、FlashAttention：端侧 NPU 上“能跑，但不是 FA”

FA 的本质，是把 softmax 的数值稳定性搬运到**跨 tile 的串行递归**里：

$$m^{(j)} = \max\left(m^{(j-1)},\, \tilde m^{(j)}\right),\qquad \ell^{(j)} = e^{m^{(j-1)}-m^{(j)}}\,\ell^{(j-1)} + \tilde\ell^{(j)}$$

$$O^{(j)} = \operatorname{diag}\!\left(e^{m^{(j-1)}-m^{(j)}}\right) O^{(j-1)} + e^{\tilde m^{(j)}-m^{(j)}}\, \tilde P^{(j)} V^{(j)}$$

对静态调度的 NPU，这个式子有四个导致问题的特征：

1. **tile 之间的循环依赖**：上一 tile 的 $m,\ell$ 决定下一 tile 的 rescale 系数。NPU 的 double-buffer 流水是按“tile 之间完全独立”设计的，这条依赖会让 pipeline 断流，DMA 预取也失效。
2. **rescale 是 elementwise 读改写整个 acc tile**：$d_v$ 宽的累加器要反复在寄存器/片上 buffer 里重读重写，向量吞吐被 rescale 吃掉的比例在 tile 小时非常高。
3. **causal mask 让有效 tile 数随 $q$ tile 线性减少**：静态调度的前提是“每个 tile 耗时相同”，而三角结构直接破坏这个前提。要么按三角编译不同长度的 kernel 序列（指令体积爆炸），要么全算（浪费 50%）。
4. **decode 阶段 FA 完全没优势**：$seq_q=1$ 时没有 tile 复用，退化成 GEMV + softmax。此时 paged KV + 普通 attention 更好。

**端侧的实际做法**：
- 固定 tile、**不做 online rescale** 的 fused SDPA（当 $d$ 和 chunk 能塞进片上 SRAM 时，chunk 内不需要 rescale）；
- split-KV + 两遍 softmax（更接近 FlashDecoding 而非 FlashAttention）；
- **prefill 交给 GPU**（simdgroup MMA + 大 L1），NPU 只做 projection；
- **改用混合线性注意力**。这一点可能更重要：**在硬件跟不上时修改模型**。

<details>
<summary>展开：为什么“混合线性注意力”是端侧真正的解法</summary>

2025 年之后，端侧友好的模型架构明显转向 $O(L)$ 的状态递归（Gated DeltaNet / Mamba-2 系 / RWKV 系）。这类算子的硬件友好度对端侧 NPU 是**降维打击**：

- 状态是**固定大小**的（与序列长度无关）→ 编译期 shape 完全确定；
- 无跨 tile 的数值递归 → 不需要 online softmax 那套 rescale；
- 天然可做成 streaming kernel，和 KV cache 一样常驻片上。

代价是表达能力（长上下文精确检索），所以主流是**混合**：大部分层线性注意力 + 少数层全注意力。

值得注意的是：这类算子的端侧 kernel 目前**基本不存在于任何主流加速框架里**。2026 年能跑 Gated DeltaNet 的加速路径只有 CUDA 和极少数手写的 Hexagon HVX kernel，ONNX / QNN / CoreML / TensorRT 全都不支持。这本身就是对“端侧 NPU 对新算子支持有多慢”的最好注脚。
</details>

---

## 五、和 Groq / TPU 的差距在哪

先纠正一个常见混淆：**Groq 和 TPU 是两种完全不同的“专用”**，差距也不是同一类。

### Groq：专用在“确定性”，不在“只支持稠密”

无 cache、无乱序执行、无 DRAM 依赖（230 MB 片上 SRAM，~80 TB/s）。编译器做**完全静态的指令调度**，运行时是确定性的，没有 jitter。它的目标函数是**单请求延迟**（TTFT 极低）。

有意思的是：**Groq 的设计哲学和端侧 NPU 是同一条路**，都是“把不确定性消灭在编译期”。区别只是规模差 100 倍，以及 Groq 把带宽做到了 $10^3$ 倍，因此恰好踩中 batch-1 decode 的平衡点。**端的 MoE 问题和 Groq 的 MoE 问题在物理上是同一个：模型的权重放不进片上，必须流式加载。** 只是 Groq 的“慢存储”是别的芯片，端侧的“慢存储”是 UFS。

### TPU：专用在“用 SparseCore 吸收不规则性”

Ironwood (v7) 的关键不是 4614 TFLOPS，而是**每 chiplet 两个、每芯片四个 SparseCore**：一个专门处理不规则 gather/scatter 的协处理器，有自己的标量/向量核、自己的 SPMEM scratchpad、能直接访 HBM、跑自己的 program（sort / partition / gather / scatter / all-to-all），把 TensorCore 从 index 计算里彻底解放出来。

它最初是为推荐系统的 embedding 查找做的，但访问模式和 MoE 路由**结构上同构**：按 index 取不连续的向量，然后做 reduce。

> **一个非常说明问题的细节**：即使有专门的硬件，SparseCore 的配置参数仍然是 `max_ids_per_partition`、`max_unique_ids_per_partition` 这种**编译期常量**，用来静态 sizing SPMEM buffer；运行时超出上限，就只能 `allow_id_dropping=True` 静默丢弃 ID，或者直接报错、终止。
>
> 连 Google 都只能用“编译期上限 + 越界丢弃”来处理数据依赖的基数。这是理解“动态 dispatch 在静态硬件上”最好的注脚：**动态能力的上限，是编译器愿意预先分配多少 buffer。**

### 差距的真正结构：谁来吸收不规则性

这是我认为唯一有解释力的框架：

| | 吸收不规则性的位置 | 代价 |
|---|---|---|
| **Groq** | 编译器（静态调度，运行时零不确定性） | 必须编译期知道一切；模型必须 fit 或流式 |
| **TPU** | 编译期限制声明 + grouped GEMM + SparseCore 分担 | 必须能批量化、能给出 shape bound |
| **GPU** | 硬件运行时（warp divergence、cache、乱序）优雅降级 | dense 效率损失 20–40% |
| **端侧 NPU** | ❌ 无处可去 | 既没有 TPU 的规模与 SparseCore，也没有 GPU 的运行时弹性 |

端侧 NPU 现在的处境是：**二流的静态调度 + 没有降级路径**。这比“算力小”严重得多。

### 量化对比

| 维度 | 端侧 NPU | TPU v7 Ironwood | Groq LPU |
|---|---|---|---|
| 片上 SRAM / scratchpad | 4–64 MB (VTCM/TCM) | 192 GB HBM3e 作为后备 | 230 MB |
| 内存带宽 | 50–140 GB/s (LPDDR5X) | 7.37 TB/s | ~80 TB/s (片上) |
| 算力 | 40–80 TOPS INT8；FP8/BF16 刚起步 | 4614 TFLOPS FP8 | ~750 TOPS INT8 |
| 平衡点 (op/byte) | ~520 | ~626 | **~9.4** |
| 不规则访存硬件 | 无（标量核 + DMA 拼） | **SparseCore ×4** | 无（靠片上带宽掩盖） |
| 编译栈 | 闭源 op 库，静态 shape，有限自定义 op | XLA + Pallas，可写任意 kernel | 静态编译器，可写任意 kernel |
| scale-out | 单 die | 9216 芯片/pod（ICI + OCS） | wafer 级 |
| 功耗预算 | 3–15 W（整 SoC） | 数百 W ~ 1 kW | ~kW 级机架 |

**但要注意：上表里最不重要的就是算力和带宽的倍率。** 真正的差距是**编译栈的表达能力**：XLA/Pallas 允许你为 MoE 手写一个 `fused_ep_moe` kernel，把 all-to-all、permute、grouped GEMM 全部融合进一个 pipeline；Groq 的编译器允许你把整个模型编译成一张确定性的指令表。端侧 NPU 的编译器**不允许你做这件事**，你只能在它给的算子里拼图。

---

## 六、我的判断

1. **端侧 NPU 不会赢 LLM 主战场。** 它会赢 always-on 小模型、多模态前端、语音/视觉常驻、省电。LLM 主战场正在迁到“带矩阵单元的 GPU”（Apple M5 的 GPU Neural Accelerator 是最明确的信号，ANE 被自己家 GPU 全面超越）。这不是 NPU 做得不好，而是**平衡点错了**。

2. **MoE 在端侧的唯一出路是“存储层级”，不是“算力”。** 如果手机出现更高带宽的 DRAM（LPDDR6）或小规模 3D 堆叠 SRAM，MoE 立刻可行；否则只能靠“热 expert 常驻 + UFS 流式 + 极强预取”。PowerInfer-2 的方向是对的，但它要求模型侧配合（幂律稀疏性），工程复杂度极高，至今没变成产品。

3. **“dense 引擎 + 不规则协处理器”会成为标配。** MoE 已是默认架构，SparseCore 那种形态迟早被复制。端侧最可能先落地的不是完整 SparseCore，而是一个**能跑 sort / permute / all-to-all 的小型 index 引擎**（HVX 其实已经在做这件事，只是没有专门的编程模型）。

4. **静态 shape 会被 shape-polymorphic kernel 取代。** CoreML 的 flexible shapes、QNN 的 dynamic shape、Arm 新 IP 都在往这走。这也是端侧追上来的**唯一可行路径**：不改变平衡点，但把“编译期逼你静态化”这个约束放宽。

5. **模型架构会继续为硬件让路。** 混合线性注意力不是为了效果更好，而是为了在端侧能跑。这类“硬件驱动模型设计”的循环，在端侧会比在云端更明显。

---

⚠️ **一点诚实的说明**：上面关于架构机制、roofline 计算、SparseCore 的编译期限制、MoE/FA 的算子特征，都是可靠的。但**具体到 2026 年各家的 SoC 型号、TOPS 数字、宣传口径**（比如高通 X2 一代对 MoE 的支持程度、联发科 APU 的具体能力），我的信息可能有滞后或误差。如果要落到选型决策，这些数字请以官方 spec 和实测为准备。相比之下，**“算力带宽比 vs decode 的算术强度”这个判据是稳定的**：你可以拿任何一颗芯片的 spec 自己算，比看宣传可靠得多。

---
## 🧑‍💻 User

那megakernel、persistent kernel（比如fused moe kernel常常会让 GPU 上的 SM 持续运行一个 scheduler：获取待处理 token->读取 expert ID->根据 expert ID 分配计算任务->将任务放入对应的工作队列->从队列中获取 GEMM 工作->完成计算后进行结果聚合，这里需要构建一个运行时数据流调度器） 这种动态 dispatch、动态路由、变长序列、稀疏计算，以及运行时负载不均衡的工作负载 npu天生就不适合吗

---
## 😀 默认助手 (7)

# 拆开看：这是三个东西，答案完全不同

这个问题容易被笼统化。`megakernel`、`persistent kernel`、`dynamic dispatch` 对硬件的需求集合**不重叠**，混在一起会得出“NPU 什么都不行”的错误结论。

| 负载特征 | NPU 适合吗 | 真实原因 |
|---|---|---|
| **Megakernel**（整模型融合、片上复用、省掉中间张量往返 DRAM） | **恰好擅长** | 静态数据流编译器**本质上就是在生成 megakernel**。这是 NPU 相对 GPU 的结构性优势 |
| **Persistent kernel**（in-kernel scheduler、跨 block 同步、work queue） | **结构性不适合** | 无 forward progress 保证、无“能放队列的地方”、无原子/低延迟同步 |
| **Dynamic dispatch / grouped GEMM**（运行时 M、运行时 expert 表） | **部分不适合** | 无 load/store 间接寻址、无运行时 sized buffer。但这是**最容易被绕过**的一条 |
| 变长序列 | 基本不是问题 | 本质是地址算术；DMA 支持动态长度就够了 |
| 结构化稀疏（2:4） | 可以做 | 静态稀疏 = dense kernel + metadata，NPU 的 tile 译码器天然能吃 |
| 运行时非结构化稀疏 | 不适合 | 需要 stream compaction，即 scan，也就是 gather/scatter 单元 |
| 运行时负载不均衡 | 不适合 | 无 work stealing；但**GPU 上的生产系统也普遍放弃**（capacity factor + drop） |

有意思的是，**真正不兼容的只有 bullet 2 和 bullet 6/7，而它们恰好是端侧 batch=1 时最没用的那两个。**

下面逐条拆。

---

## 一、Megakernel 恰恰是 NPU 的主场

这里是反直觉的地方。Megakernel 想解决的是**每层之间的 DRAM 往返**，在 GPU 上，这要靠 `__shared__` + 跨 kernel 的 L2 命中率来绕开。

而一个静态数据流编译器在做的事就是：**给你一整个模型，输出一条固定指令序列，所有中间张量在 VTCM 里手工排布，DMA 双缓冲**。这已经是 megakernel 了，而且是编译器免费给你的。

在 dense 模型上，编译良好的 NPU graph 的融合程度经常**优于** GPU，因为它不依赖 cache 命中，而是**显式分配**片上 buffer。这是端侧 NPU 唯一一个比 GPU 强的 LLM 场景。

**所以“NPU 做不了 megakernel”是错的。NPU 做不了的是“带数据依赖控制流的 megakernel”。** 加个“动态”两个字，性质全变了。

---

## 二、Persistent kernel：这才是真的结构性不兼容

把你们描述的那个 scheduler（取 token → 读 expert ID → 分配 → 入队 → 取 GEMM 工作 → 聚合）拆成硬件需求，逐条打分：

| 需求 | 为什么需要 | GPU | 端侧 NPU |
|---|---|---|---|
| ① **共享、可寻址、低延迟的队列** | 队列要有地方放，且所有计算单元可见 | L2 / global memory，几十 MB | **✗** VTCM 是 private scratchpad，**只能被 DMA 寻址，不能被计算单元 load/store**；跨引擎无共享地址空间 |
| ② **原子操作 + 内存序** | work-stealing 的 CAS、计数器 | `atomicAdd` / `mbarrier` / fence | **✗**（Hexagon 是例外，它有 atomics；Ethos/MAC-array 系没有） |
| ③ **Forward progress 保证** | scheduler 等 B 算完，必须保证 B 会被调度到，否则死锁 | cooperative launch 保证 `blocks ≤ 常驻容量`，`grid.sync()` 合法 | **✗** 没有 block 概念，没有“全常驻”语义，单一静态调度里不存在并发实体 |
| ④ **运行时 trip count 的循环** | “从队列取下一个任务”次数编译期未知 | SASS 分支随便跳 | **△** 标量核能跳，但**流水线的 DMA 预取需要静态 trip count**，一跳就断流 |
| ⑤ **运行时 sized buffer** | 每 expert 的 M 运行时才知道 | dynamic shared memory，超额申请几乎免费（227 KB SMEM / 64K reg / L2） | **✗** VTCM 只有 2–16 MB，且 buffer plan 是**全局编译期优化**，不能“保险起见申请两倍” |
| ⑥ **间接寻址 gather** | 按 index 表取权重/激活 | load with computed address，一行代码 | **△** 只有 DMA 描述符，而**描述符数量是静态的** |
| ⑦ **抢占 / 时间片** | scheduler 卡住时能被踢掉；UI 不能冻 | warp scheduler + preemption | **✗** |
| ⑧ **空间异构**：scheduler 核与 GEMM 核并行 | 控制面不占算力 | SM 之间天然并行 | **✓** 标量核 + 向量核 + 矩阵核本来就是分离的、异步的 |

**4 个 ✗、2 个 △、1 个 ✓。**

第 ③ 条是最根本的。它跟“性能”无关，跟**可终止性**有关：没有 block 级并发模型，就没有“全常驻”这个契约，也就无法证明 `grid.sync()` 不会死锁。而手机上一个跑飞的内核没有抢占可以踢掉：**watchdog 重启 → 整机卡顿**。这是移动端独有的产品约束，比性能约束硬得多。

### 但请把这条结论的边界收紧

**Groq 比任何 NPU 都更“静态”，它照样跑 MoE。** Cerebras 更极端，整个 wafer 的数据流是编译期固定好的，也跑 MoE。

所以**“静态”本身不是障碍**。障碍是组合条件：

$$\text{静态调度} \;+\; \text{权重放不下片上} \;+\; \text{无运行时降级路径}$$

三条同时成立才会出问题。Groq 缺第三条（它的编译器可以在编译期把每一条路径都展开成确定指令）；GPU 缺第一、二条（有 warp divergence 兜底）。NPU 三条全中。

---

## 三、关键：真正的 MoE kernel 也没有做“真动态 dispatch”

这一点被讨论得很少，但它把整个问题重新定位了。

看两个最先进的实现：

**MegaBlocks（Databricks，dropless MoE 的经典实现）**：它的做法是每个 block 分到一个 expert，每 expert 的容量用一个**上界** `max_M = ceil(T·k / E · capacity_factor)`，然后 padding + mask。**它在编译期就知道 M 的上界。**

**TPU SparseCore**：配置项是 `max_ids_per_partition` / `max_unique_ids_per_partition`，**编译期常量**，用来静态 sizing SPMEM；运行时超出就 `allow_id_dropping=True` 静默丢 ID，或者直接报错终止。

连 Google 的专用不规则协处理器，答案都是“**编译期上界 + 越界丢弃**”。

所以行业里从来没有“真动态 dispatch”，只有“**有界 dispatch + padding**”。所有人的算法是一样的，**区别只在于谁付代价**：

- **GPU 付冗余算力**。它便宜，因为服务端 batch 大、算力相对带宽充裕（平衡点差 150–600 倍），多算 30% 的 padding 无所谓。而且动态 shared memory 申请过一点几乎免费。
- **NPU 付 VTCM**。它贵，因为我们上一轮算过：端侧 NPU 是**带宽受限 260 倍**，算力本来就是过剩的，多算不要钱；**但 VTCM 只有 4 MB 左右，padding 到 worst case 直接把张量 tile 的预算吃光**。

> **这才是硅片层面的真原因，不是“NPU 生来僵硬”，而是“NPU 的片上 SRAM 太贵，付不起保险式的超额分配”。**

这个结论可操作得多：想改善，就想办法**降低为动态性预留的 buffer 开销**（虚拟 extent、按需分配、描述符生成器），而不是空谈“支持动态 shape”。

---

## 四、而端侧最大的那个负载，压根不需要 scheduler

这是我认为最该说的一条。用上一轮的框架算笔账。

### Decode（batch=1）：调度器比工作还长

30B 总参 / 3B 激活，top-2，int4，手机上有效带宽 46 GB/s：

$$t_{\text{mem}} = \frac{1.5\ \text{GB}}{46\ \text{GB/s}} \approx 32\ \text{ms},\qquad t_{\text{compute}} = \frac{2\times 3\times10^9}{40\times10^{12}} \approx 0.15\ \text{ms}$$

**计算只占 0.47%。**

更关键的是任务数：batch=1、top-2 意味着需要派发的 (token, expert) 对**总共 2 个**。

也就是说：**这个 token 的 dispatch 就是 2 条 DMA 描述符**。没有队列，没有不均衡，没有 work stealing，没有可并行的东西——一个 GEMV 的输出维度就那么多，填不满任何阵列。

**scheduler 的全部前提（把不规则的活摊到很多计算单元上）在这里是空的。** 你在一个 32 ms 的纯 DMA 等待里插一个 2 µs 的队列循环，除了增加不可终止性风险，收益是 **0**。

### Prefill（4K prompt）：真需要 dispatch，但问题是 shape 不是调度

4096 tokens × top-2 = **8192 个 (token, expert) 对，64 个 expert 各拿 ~128 个 token**。这时候：

$$t_{\text{compute}} = \frac{2 \times 3\times10^9 \times 4096}{40\times10^{12}} \approx 615\ \text{ms},\qquad AI = \frac{24.6\ \text{TFLOP}}{15\ \text{GB}} \approx 1640\ \text{FLOP/B} > 520$$

**变成 compute-bound 了。** 而 $M\approx 128$ 对 tile GEMM 是**极好的**形状——远好于 batch=1 的 $M=1$。

所以端侧 prefill 的真需求不是 persistent scheduler，而是：

> **一个能接受运行时 $M$ 的 grouped GEMM（$M$ 从内存里读，而不是编译期烧进 kernel），外加一个硬件 permute（按 index 表重排 token）。**

$M=128$ 这个量级你根本不需要 work stealing 去平衡，只需要 64 个 shape 不同的 GEMM 拼在一起。**这比“运行时数据流调度器”小了两个数量级的工程问题。**

这也顺便解释了 PowerInfer-2 为什么把 prefill 放 NPU、decode 放 CPU：**因为只有 prefill 需要 NPU 的矩阵吞吐，而 decode 需要的是 CPU 的灵活寻址 + 大内存。**

---

## 五、其余几项的精确判定

<details>
<summary>变长序列：被高估的难点</summary>

varlen 本身是**纯地址算术**（`cu_seqlens[i] + off`），任何有 load/store 的 ISA 都免费支持。GPU 上 `flash_attn_varlen` 的优雅之处是：**host 端算好精确的 grid 尺寸，launch 一次，零浪费。**

真正的问题在组合效应：attention 的 tile 总数是

$$\sum_{i} \left\lceil \frac{L_i}{64} \right\rceil \cdot \left\lceil \frac{L_i}{64} \right\rceil$$

这是运行时才知道的。所以 varlen + paged KV + 动态 batch 三者叠加，才逼出 persistent kernel。

在 NPU 上，这个组合的代价是 **pad 到 max_len**（浪费），**或者**为每个可能的总 tile 数编译一个图（体积爆炸）。这是个 **shape 问题**，不是调度问题。
</details>

<details>
<summary>稀疏：要区分“静态稀疏”和“运行时稀疏”</summary>

- **2:4 结构化稀疏**：模式编译期已知，等价于 dense kernel + metadata，NPU 的 tile 译码器能吃。**没问题。**
- **激活稀疏**（ReLU 后大量 0）：需要“检测 0 → 跳过”，在固定节拍的流水线里没有便宜的做法（这正是 systolic array 的经典缺陷）。要做得先做 stream compaction，那是 scan，需要 gather/scatter 单元。**做不了。**
- **绕过办法**：PowerInfer 系列用 **online predictor 离线/预测地判定每层哪些 neuron 是热的**，把 runtime 稀疏**转化成 compile-time 稀疏**。这是用软件把动态性前移，代价是预测误差会损质量，而且 predictor 得跑在 CPU 上。

这条路线其实比 megakernel 更适合端侧，因为它**主动消除动态性**而不是要求硬件容忍它。
</details>

<details>
<summary>负载不均衡：GPU 上大家也在“放弃”</summary>

dropless MoE 在 GPU 上依然昂贵，绝大多数生产系统用的是 capacity factor + token dropping。所以“运行时负载均衡”这条，**不是 NPU 独有的失败**，是整个行业的普遍妥协。NPU 只是少了一个逃生口：GPU 至少可以选择 dropless（用冗余算力换质量），NPU 连这个选项都没有（VTCM 付不起）。
</details>

---

## 六、产业实际选了哪条路（很说明问题）

面对同一个缺口，三条路都有人走，而且**成本排序很清楚**：

1. **往 GPU 里补矩阵单元**（最便宜）。Apple M5 每 GPU core 一个 Neural Accelerator——GPU 已经有队列、atomic、forward progress、抢占、统一地址空间，**缺的只是矩阵吞吐**。补这一块远便宜于“往 NPU 里补一整套调度语义”。M5 上的实测（GPU 在所有长度都赢 ANE，decode $2.0$–$3.9\times$、prefill $2.9$–$15.9\times$）就是这个选择的验证。手写 Metal 4 cooperative tensor kernel 跑 fused MoE expert GEMM / prefill attention（BaseRT 那类工作）拿到 $6\times$+ 于 llama.cpp，也是这个方向。
2. **往加速器旁边加不规则协处理器**（中等）。SparseCore 路线。贵，但它把“dense 引擎不该干的活”隔离掉了。端侧目前**没有对应物**。
3. **改模型，让动态性消失**（最便宜，也最被低估）。静态/平衡路由（expert-choice routing、hash routing、aux-loss-free 均衡）、top-1、大 shared expert 占比、混合线性注意力（$O(L)$ 状态递归 → shape 完全固定、无跨 tile 数值递归）。这类改动的收益是**让硬件问题不存在**，而不是让硬件容忍它。

**端侧的现实是 3 主导、1 缓慢跟进、2 缺席。**

---

## 七、如果只给端侧 NPU 加一块硬件，加什么

按 ROI 排序，我认为是这样：

1. **一个 descriptor generator（小型 gather 引擎）**。能读一张 index 数组，**硬件生成 DMA 描述符列表**，顺便做 stream compaction。这一个单元干掉 MoE dispatch 的 80%——permute、expert 权重 gather、varlen 的 tile 表，全是它。本质上就是 “SparseCore-lite”，硅面积很小（一个小型向量核 + 一个小 SRAM 端口），**不需要** atomic、不需要 forward progress、不需要抢占。这是“最低成本换最大表达力”的点。
2. **虚拟 extent 的 VTCM**。编译期分配 max，运行时用一部分，剩下的还给别的 op。这**主要是编译器问题**，不是硅问题——但它直接命中“padding 到 worst case 太贵”这个根因。
3. **运行时 M 的 tile GEMM**。$M$ 从内存读，尾块 mask 掉。有了 1 和 2，这一条几乎自动成立。prefill 阶段的 MoE 立刻可用。
4. **（可选）一条从 UFS 到 VTCM 的、可被设备程序主动发起的分页路径**。没有它，权重流式加载必须由 CPU 编排，做不出真正的 producer/consumer 重叠。

注意这四条的**共同点**：**没有一条需要给 NPU 加“调度器语义”**。加完之后 NPU 依然是静态调度的，但它能接受“有界动态”——而这恰好覆盖了 MegaBlocks 和 SparseCore 所依赖的全部能力。

---

## 结论

- **“NPU 天生不适合 megakernel”是错的**——megakernel 正是静态数据流编译器的天然产物，是它的强项。
- **“NPU 天生不适合 persistent kernel”是对的，且是结构性的**：缺的不是算力，是**一个能放队列的共享地址空间** + **forward progress 契约** + **可抢占性**。三条都在 ISA 和内存模型层面，不在 datapath 层面。
- **但端侧最重的负载（batch=1 decode）根本不需要它**：dispatch 就是 2 条描述符，compute 占 0.47%，scheduler 无活可干。
- **端侧真正缺的是“有界动态”**：运行时 $M$ 的 grouped GEMM + 硬件 permute + 便宜的尾部 padding。这比调度器小两个数量级，且**这才是 B300/GPU 上 MegaBlocks 也在做的同一件事**——所有人都在做“有界 dispatch + padding”，只是 GPU 的 padding 便宜、NPU 的 padding 贵。
- **最后的判据不是“动态 vs 静态”，而是“不规则性在哪一层被吸收，以及谁付账”**。Groq 用编译器吸收（付编译期知识）、GPU 用冗余算力吸收（付利用率）、TPU 用 SparseCore 隔离（付硅面积）、NPU 目前无处可去。

⚠️ 一点边界说明：Hexagon 是个例外——它是真正的 VLIW DSP，有 load/store、atomics、VTCM、多线程 RTOS（QuRT），所以 `ggml-hexagon` 能写出 DeltaNet 这种任何加速框架都不支持的 kernel。这说明 **DSP 血统的 NPU 在“可表达性”上并不受限，受限的是性能**；而 Ethos/MAC-array 血统的 IP 连 load/store 单元都没有，“不适合”是针对它们说的。谈端侧 NPU 时必须区分这两条血统，否则结论会错。

---
## 🧑‍💻 User

既然simt=fgmt+simd+自由scatter gather，Scatter/Gather能力为什么对于动态dispatch能力也有很大影响呢，为什么它需要一定程度的硬件支持（这也是为什么 GPU 的 SIMT + 自由 Scatter/Gather 对 MoE、稀疏计算、变长序列特别有价值）

---
## 😀 默认助手 (7)

# 先把“自由 scatter/gather”这件事说准

你的公式 `SIMT = FGMT + SIMD + 自由 S/G` 是对的，但有个隐含偏差：**它把 S/G 写成了“免费项”，而 S/G 恰恰是 GPU 相对向量机最大的面积开销来源。**

对比一下历史：

| | 一条 gather 指令的含义 | 吞吐 | 需要什么硅 |
|---|---|---|---|
| Cray 类向量机 | 按索引向量取 $n$ 个元素 | ~1 元素/cycle（微码循环） | 一个索引寄存器 + 一个 AGU |
| SIMT | 每个 lane 按自己的地址取 1 个元素 | $W$ 元素/cycle（$W$ = warp 宽） | **$W$ 个并行 AGU + $W$ 份地址寄存器** |

所以“自由”指的是：**地址解析的吞吐 = SIMD 宽度，因此地址解析永远不是瓶颈**。这个 $W\times$（32–128×）的差距，就是 GPU 与“向量 DSP”的本质分界线。

而且“自由”只在**表达**层面成立。执行层面，一个完全散开的 warp 访问要付：

$$\text{transactions} = \min\left(W,\ \#\text{distinct 32B sectors}\right) \le W$$

最坏情况是 32 条独立事务代替 1 条。所以准确说法是：**S/G 在 ISA 层免费，在内存系统层极贵。** 记住这句话，后面全靠它。

---

## 一、把"scatter/gather 能力"拆成三个独立能力

这三件事常被混为一谈，但要的硅完全不同，对 dispatch 的影响也完全不同。

### S1：间接寻址（address = f(data)）

一个能接受寄存器作为地址的 AGU。成本很便宜：一个加法器 + 一个寄存器堆端口。

**但关键问题在于它位于哪里。** NPU 的标量核*有*这个能力（Hexagon 是完整 load/store ISA），可是标量核既够不到 DRAM 带宽，也喂不饱 MAC 阵列。所以有和没有差不多。

### S2：变长枚举（loop over 一张 index list）

这是真正贵的那一半，也最常被忽略。

设路由产生一个映射 $\pi: [Tk] \to [E]$，每个 expert 的 token 数 $n_e = |\pi^{-1}(e)|$ 是**运行时才知道的基数**。要执行 expert $e$ 的 GEMM，你必须：

$$\tilde X[p] = X[\iota(p)] \quad\text{（gather，长度 } n_e\text{ 未知）}$$

$$\tilde Y[p] = f_e(\tilde X[p]) \quad\text{（grouped GEMM，} M = n_e\text{ 未知）}$$

$$Y[\iota(p)] \mathrel{+}= \tilde Y[p] \quad\text{（scatter + 聚合）}$$

注意这里两层动态性叠在一起：**地址是数据的函数（S1），而迭代次数也是数据的函数（S2）。**

- 在 SIMT 上，S2 **根本不存在**。线程 $i$ 做 `dst[i] = src[idx[i]]`——你不需要循环，因为 $n_e$ 个元素由 $n_e$ 个线程并行处理。基数变成了并行度，而并行度是免费的。
- 在 DMA/描述符机器上，描述符是 `(src, dst, len)`。要表达"按索引表 gather $n_e$ 行"，**引擎本身必须能解引用那张索引表并循环**——也就是说，DMA 引擎里要装一个微型 AGU 状态机。这不是加一条指令，这是加一个引擎。

> **这就是你问的核心答案的第一半**：SIMT 之所以"有" S/G，是因为它把地址生成器复制了 $W$ 份并塞进了本来就存在的数据通路。没有这份复制的机器，只能专门造一个引擎去循环索引表。

### S3：容错性（能不能扛住这个访问模式）

性能由**同时在飞的请求数**决定，用 Little 定律：

$$N_{\text{inflight}} = B_{\text{bandwidth}} \times L_{\text{latency}}$$

这是硬件资源（MSHR 数、DMA 队列深度、TLB 条目），不是指令。

有意思的是，**在端侧这一条反而不太紧**：

- GPU：(3 TB/s) × (600 ns) ≈ 1.8 MB in-flight
- 手机 NPU：(77 GB/s) × (300 ns) ≈ **23 KB** in-flight

差 80 倍。所以**端侧的 S/G 瓶颈不在"请求数不够"，而在下面两件事**：

1. **描述符生成速率**——标量定序器 ~1 描述符/几个 cycle，vs SIMT 的 $W$ 地址/cycle
2. **DRAM row buffer 命中率**——15 GB 工作集 + 128 B 随机读，几乎每次都要 `activate` 新行（$t_{RC}\approx45\text{ ns}$ vs $t_{CAS}\approx15\text{ ns}$），有效带宽直接掉到 $\sim 1/3$

第 2 条把上一轮的账改写了：$1.5\text{ GB} / (46 \times 1/3\text{ GB/s}) \approx 96\text{ ms} \Rightarrow \sim 10\text{ tok/s}$。**这正好是手机上 MoE 的实测数量级。** 模型对上了。

---

## 二、核心：dispatch 的输出本身就是地址模式

这是"为什么 S/G 和 dispatch 高度相关"的真正答案——它们**不是相关，是同一件事**。

看上一节那三个公式。dynamic dispatch 的产物是什么？**一个置换（permutation）**。而置换的执行器是什么？**gather 和 scatter。**

$$\text{dispatch} \equiv \text{permute} \equiv \text{gather} + \text{scatter}$$

所以：

> **没有 S/G，就没有 dispatch，只有 dense 化。**

因为除了"按索引表重排"，你只剩一个办法让 access pattern 在编译期可表达：**不做重排，把所有 expert 都算一遍，用 mask 掉**。这就是 why NPU 上 MoE 的默认实现是 dense-over-all-experts——**这不是实现者的偷懒，而是没有 S/G 时唯一的可表达形式。**

代价是算力 $\times E$。而这个代价在设备上是否致命，取决于你处在 decode 还是 prefill（见第四节）。

还有一个容易被忽略的点：**scatter 比 gather 更难**，因为它有**写冲突**。$k>1$ 时多个 packed 位置映射回同一个 token，需要 atomic 或 sort-then-reduce。GPU 用 atomic / 分段归约；NPU 没有 atomic，只能**保证无冲突**——即不聚合，把 top-k 写成 $k$ 份独立拷贝，最后再加。这又是一份 $\times k$ 的带宽。

---

## 三、为什么"必须付硅"：解析位置只有三个

这是最干净的论证。**任何执行引擎都必须把"work → location"这个映射在某个地方解析掉。只有三个地方可以解析：**

| 解析位置 | 谁在做 | 硅成本 | 能力边界 | 代表 |
|---|---|---|---|---|
| **编译期** | 编译器 | 0 | 只能处理**静态**路由 | 静态 shape NPU 图、Groq |
| **运行时，在 lane 上** | 每 lane 的 AGU | **高**（AGU × $W$ + cache + MSHR + TLB，整颗芯片 50–100× NPU 面积） | 任意动态，但散开时慢 32× | GPU |
| **运行时，在专用引擎里** | 索引循环引擎 | **低**（一个小单元，但要串行化） | 任意动态，但吞吐受限于一个定序器 | TMA gather、**SparseCore** |

**路由是 learned 的 ⇒ 第一个位置永久不可用。** 所以你必须付后两者之一的硅。

这就是"为什么需要硬件支持"的完整答案：**因为映射必须在运行时解析，而运行时解析必须有一个物理执行者。你可以选择把它放在 lane 上（贵但并行），或者放在引擎里（便宜但串行），但你不能选择"不付"。**

### 顺便：Google 事实上选了第三个位置

vLLM 的 TPU 后端里，SparseCore 干的活叫 **"MoE preprocessing"**——activation all-gather、metadata 通信、scatter。**GEMM 跑在 TensorCore 上，SparseCore 干的是 dispatch 那一段。** 这直接确认了：S/G 能力 = dispatch 能力，是一个专门的硬件单元。

---

## 四、为什么不能"给标量核加 load/store 就完事"

这是最容易踩的坑。Hexagon 已经有完整 load/store + atomics，为什么还是不行？四个耦合：

### (a) 吞吐差 $W\times$，且这是结构性的

标量定序器生成 $n$ 个地址要 $O(n)$ 个 cycle，SIMT 是 $O(n/W)$。**这一个因子与工艺无关、与代际无关，是 SIMD 宽度的直接后果。** 而且它不是一个可以优化的常数——你没法让一个 in-order 标量核并行执行 32 个独立的地址计算。

不过要诚实：**这一条在今天的形状下还不足以致命**。8192 个 token-expert 对，每对 10 cycle ≈ 82k cycle ≈ 82 µs，对比 prefill 的 615 ms 是 0.01%。**所以 (a) 不是主因。**

### (b) 一个 in-flight dispatch vs 很多个 —— 这才是致命的

这一条是核心，而且几乎没人讲。

看那条依赖链：

$$\text{router} \to \text{top-}k \to \textbf{barrier} \to \text{gather} \to \textbf{barrier} \to \text{GEMM} \to \textbf{barrier} \to \text{scatter}$$

每一步的输出是下一步的**地址**。所以：

- 你不能预取（地址还不存在）；
- 每次依赖边都要暴露一次完整的 DRAM 延迟；
- **静态数据流的定义就是"编译器证明了执行顺序"，所以它按构造只允许一条链在飞。** 这不是实现缺陷，这是这个范式的定义。

而真实负载里有**成千上万条彼此独立的链**（不同 token、不同 expert、不同 batch 项）。要让它们重叠，硬件必须能同时 hold 住 $N$ 条独立的、进度互不干扰的上下文。

**这就是 FGMT 的真正作用，而它常被误以为是"为了隐藏延迟"。** 更准确地说：**FGMT 提供了"$N$ 条独立数据依赖链同时在飞"的能力，而这正是动态 dispatch 的吞吐前提。** S/G 只是让每条链可以有不同的地址模式——**没有 FGMT，S/G 只是把一个串行问题变成另一个串行问题。**

一句话：**S/G 制造了运行时的、不可预测的延迟；FGMT 提供了容忍这些延迟的并发度。两者缺一不可，而 NPU 两个都缺。**

### (c) DMA 路径上没有 MMU

这是个非常具体、常被忽略的耦合：GPU 的 scatter 是**虚拟地址**，TLB + page walk 帮你映射。移动 NPU 的 DMA 描述符里通常是**物理地址**（简化 + 安全）。

后果是：**buffer 必须是驱动预先 pin 住、物理连续、大小固定的一大块。** 于是"按运行时 $n_e$ 分配"这件事在物理上不成立——这直接闭合了上一轮说的 `max_ids_per_partition` 现象。

### (d) 安全：device-side DMA = 提权面

这条是非技术性的，但**可能是移动端最硬的一条**。让设备端程序自己生成 DMA 描述符，等于把任意物理地址读写暴露给一个可被攻击的进程。IOMMU 能缓解，但移动 SoC 为了功耗/面积/认证，普遍**故意保持描述符由 host 生成**。

所以"NPU 不支持动态 dispatch"里，有相当一部分比例不是"做不到"，而是"**不愿承担这个攻击面 + 认证成本**"。

---

## 五、代价的三种付法，NPU 一种都不付

这是我认为最有解释力的收束。**在不规则访问上，吞吐机器永远不可能免费——你只能从三种货币里挑一种付：**

1. **付排序（restore locality）**：把 scatter 变成 sorted-then-coalesced。这就是所有 MoE kernel 里 `moe_align_block_size` / grouped GEMM 预处理在做的事——你在**花一次 scan 的钱，买回 row buffer 的命中率**。这也是 SparseCore 最初为推荐系统做的三件事：**sort / dedup / partition**。
2. **付事务数**：让 coalescer 失效，付 $32\times$。
3. **付硅**：堆 MSHR / cache / AGU，让硬件自己吸收随机性。

关键的洞察在这里：

> **S/G 硬件支持的价值，不在于"让随机访问变快"，而在于"让'把随机变成顺序'这个转换本身变便宜"。**
>
> 因为 sort / partition / compact **全都是 scan**，而 scan 需要宽并行的地址生成——也就是需要 $W$ 份 AGU。GPU 用 warp 级 scan 免费做；NPU 用标量核做就是串行；SparseCore 就是**专门做 scan 的硬件**。

所以端侧真正该补的不是"scatter 指令"，而是**一个能做 compaction / partition 的小引擎**——这和我上一轮说的 "descriptor generator" 是同一个东西，只是现在有了理论依据：**它补的是 scan 吞吐，不是 gather 吞吐。**

顺带解释了 PowerInfer 的 hot/cold neuron clustering：它压根没有加速 gather，它在**改变数据的物理布局**，让 gather 变得 row-buffer friendly。这是"付货币 1"的软件版做法。

---

## 六、接回上一轮：S/G 什么时候才真正成为瓶颈

现在可以给出一个干净的判据了。dense-over-all-experts 的代价是算力 $\times E$，而这个代价的相对大小随情形剧烈变化：

**Decode（$T=1$，compute 占 0.47%）**

$$0.47\% \times 8 = 3.8\% \quad\text{—— 无所谓}$$

没有 S/G 硬件带来的代价，**完全被带宽掩盖**。所以 NPU 在 decode 上"用稠密化替代排列"这个偷懒是**成立的**——这也解释了我上一轮说的"decode 根本不需要 scheduler"：它连 dispatch 都不需要。

**Prefill（$T=4096$，compute-bound）**

$$615\ \text{ms} \times 8 = 4.9\ \text{s} \quad\text{—— 直接死掉}$$

**这就是 S/G 硬件从"可有可无"变成"卡脖子"的全部原因：不是负载变动态了，是计算变贵了。**

$$ \text{S/G 的必要性} \;\propto\; \frac{\text{compute 时间}}{\text{memory 时间}} $$

在端侧，这个比值在 decode 时是 $1/200$，在 prefill 时是 $2$。**只有跨过 1 之后，你才被迫从"稠密化"转向"真 dispatch"，而一旦转向，就必须有 S/G 硬件。**

---

## 收束成一句话

> **Dynamic dispatch 在数学上就是一个置换，而置换的执行器就是 gather/scatter。** S/G 之所以需要"一定程度的硬件支持"，是因为路由是 learned 的 ⇒ 地址模式只能在运行时解析 ⇒ 而运行时解析必须有一个物理执行者。你可以选择把它复制到 $W$ 条 lane 上（贵，但并行，且天然带 FGMT 来容忍它制造出来的延迟），或者塞进一个专用引擎（便宜，但串行）。**选择"不付"的唯一后果，就是被迫用 $\times E$ 的稠密计算去模拟置换——而这条路只在 compute 不值钱的时候成立。**

端侧的现实是：decode 阶段 compute 确实不值钱（所以偷懒成立），prefill 阶段不值钱的幻觉破灭（所以偷懒直接崩掉），而恰好 prefill 才是用户感知最强的那一段（TTFT）。这就是为什么端侧 NPU 在 MoE 上的体验会呈现"要么还行，要么完全不可用"的悬崖式分布。
