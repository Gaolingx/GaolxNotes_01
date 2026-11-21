# NPU动态调度瓶颈

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

# 核心：scatter/gather 是一条指令

你的 `SIMT = FGMT + SIMD + 自由 scatter/gather` 这个拆法是对的。但第三条常被误解成“搬运能力”。它的真正作用是把**一个数据依赖的不规则访存压缩成一条指令**，从而让“动态”不必升级成“控制流 + 同步”。

这个区别是决定性的，因为**静态调度的编译器唯一编译不了的东西，就是“带屏障的、trip count 由数据决定的循环”**。scatter/gather 硬件做的事，就是让这种循环根本不需要出现。

---

## 一、把 dynamic dispatch 完全拆开：里面没有一个分支

你们描述的那个 scheduler（取 token → 读 expert ID → 分配 → 入队 → 取 GEMM → 聚合），拆成最底层的原语：

| 步骤 | 真实计算 | 原语 |
|---|---|---|
| 读 router logits，top-k | 归约 + 排序 | SIMD reduce |
| 统计每 expert 收到多少 token | `H[I[j]] += 1` | **scatter-with-add**（histogram） |
| 求每 expert 的起始偏移 | 前缀和 | scan |
| 生成重排后的 token 表 | `P[pos[e]++] = j` | **scatter**（materialize permutation） |
| 按 index 取 token 行 / 权重行 | `A[g[i]]` | **gather** |
| 按逆置换把结果写回 token 位置、乘 gate | `O[i] += w·Y[inv[i]]` | **gather + reduce** |

**整条链路里没有一个分支。**

因为“不同 expert”跑的是**同一个 kernel，只是基地址不同**。所谓 dispatch，从头到尾是**地址生成**，不是控制流。

$$\text{dynamic dispatch} \;=\; \text{histogram} + \text{scan} + \text{scatter} + \text{gather}$$

这一步想通了，后面的问题就都清楚了：**MoE 只有在被误写成“if expert==3: ... else if ...”的时候才需要控制流。写对了，它只需要地址。**

---

## 二、为什么必须“在数据通路里”，而不是“在旁边”

这是全部答案的关键。同样一件事，两种表达：

**路径 A（有 scatter/gather ISA）**

```
ld.global.v4  r, [addr + idx*4]     // 一条指令，地址在寄存器里
```

动态性被**封装在指令内部**。对编译器而言，这是直行代码：没有分支、没有变长循环、没有跨引擎同步。它可以被静态调度、被双缓冲、被流水线化。

**路径 B（NPU 现状）**

```
1. 向量核算出 index 数组  → 写进 VTCM
2. 标量核从 VTCM 读出每个 index
3. 标量核把 index 填进 DMA 描述符
4. 标量核敲 doorbell，DMA 启动
5. 等 barrier，DMA 完成
（重复 N 次，N 由数据决定）
```

问题不在第 1–5 步慢，而在于：**第 3–5 步是一个“trip count 动态 + 含屏障”的循环**。

- 静态数据流编译器要把 barrier 放进指令流，就必须知道循环展开几次，否则**无法确定同步点在指令流中的位置**；
- DMA 描述符是**程序的一部分**（编译期烧进指令流），不是数据的一部分；
- 于是 N 必须静态。这正是 SparseCore 的 `max_ids_per_partition` 存在的理由：**用编译期上界冒充运行时 N，越界就丢数据。**

所以差别可以用一句话概括：

> **在 GPU 上，不规则访存是“一条会延迟的指令”；在 NPU 上，不规则访存是“一个跨引擎的协议”。**
>
> 前者对调度器不可见，后者是调度器唯一的噩梦形态。

顺便说清一个常见误解：**NPU 的 DMA 其实也是 scatter/gather 引擎**（多维描述符能表达任意 stride）。真正的区别是**地址从哪来**：

| | 地址来源 | 属于 | 动态？ |
|---|---|---|---|
| GPU | 寄存器 | **数据** | ✅ 运行时任意 |
| NPU DMA | 描述符字段 | **程序** | ❌ 编译期定死 |

一字之差：`addr = reg` vs `addr = imm`。**这正是静态世界与动态世界的分界线。**

---

## 三、为什么“每 lane 独立”也不够，规模还必须够大

即使你给 NPU 加了 per-lane AGU，还会撞上 Little's Law：

$$N_{\text{outstanding}} = B \times L_{\text{latency}}$$

以端侧为例，$B = 46\ \text{GB/s}$，行大小 64 B，DRAM 延迟 $L \approx 200\ \text{ns}$：

$$N = \frac{46\times 10^9}{64} \times 200\times10^{-9} \approx 144$$

也就是说：**要维持端侧带宽下的随机访问，需要约 144 个并发呈递中的独立地址请求。** 每个都要有自己的 MSHR 项、返回数据的路由标签、冲突重放（replay）逻辑。这是**一块实打实的硅**，标量核有几个 scoreboard 槽？

再算服务器：HBM3e、3.35 TB/s、32B sector、600 ns → 需要**数万**个 outstanding。而 H100 只有几千个 MSHR。

> **这就是为什么连 GPU 也撑不住真正的随机访问**：随机访问只能发挥峰值带宽的 5–10%。这也是**为什么所有 MoE kernel 都要先做一遍 permute，再做连续的 grouped GEMM**，而不是做“scattered GEMM”。

于是 scatter/gather 的真实定位浮出来了：

> **它是一个“转导原语”（transduction primitive）**：花一次带宽，把不规则布局转成规则布局，让后面的稠密引擎（tensor core / MAC array）永远只看到规则 tile。
>
> **不规则性被集中到一个便宜、带宽受限的数据搬运环节，而不是让昂贵的计算引擎去适应它。**

而 NPU 的 DMA 是**保布局**引擎（strided/rectangular，输出布局由输入布局决定），不是**转导**引擎。**这一句话就是端侧 NPU 和 GPU 在这个问题上的全部差距。**

---

## 四、NPU 特有的第三个坑：gather 落地必须在 scratchpad

这条最少被讨论，但可能是最本质的。

**在 GPU 上，gather 的目标是寄存器，而寄存器是每 lane 私有的。** 每个 lane 拿到自己的数据，地址随便乱，没有布局契约。这就是“自由”的物质基础。

**在 NPU 上没有 per-lane 私有存储**，所有数据都是在**共享 scratchpad（VTCM）里由编译器预先分配好的地址**。所以“gather 一行到某处”实际是：

```
写 VTCM[compiler_fixed_addr + runtime_offset]
```

这已经是**一个 scatter**（对共享存储的、地址按 lane 计算的写）。而共享存储的 scatter 立刻带来 **bank conflict**：两个 lane 命中同一个 bank 就得串行。

于是出现一个很别扭的事实：

> **NPU 要做 gather，必须先有 scatter 硬件。** 而且 scatter 完的数据还要满足 MAC array 的 swizzle 契约，往往需要**再 repack 一遍**，等于付两倍带宽。而带宽恰恰是端侧唯一缺的东西。

（顺带解释一个现象：GPU 上的 on-chip permute kernel 也受 bank conflict 折磨，所以大家用 padding 技巧。这不是 NPU 特有的失败，是**共享存储 scatter 的普遍代价**。）

---

## 五、回到 SIMT 三足：它们是协同设计，缺一崩

你说 SIMT = FGMT + SIMD + 自由 scatter/gather。这三条正好构成“不规则负载也免费”的完整闭环：

| 支柱 | 单独提供什么 | 拿掉会怎样 |
|---|---|---|
| **SIMD** | 规则数据的吞吐 | 变成 MIMD，稠密效率崩 |
| **自由 scatter/gather** | 不规则负载**不改变控制流** | 动态性升级成变长循环 + 屏障 → 静态调度失效 |
| **FGMT** | 把 gather 的**延迟**换成**吞吐** | 一个 gather 的 200 ns 就硬生生停顿，MSHR 里的请求等不到 warp 来填 |

关键在于第三点：**scatter/gather 的代价是延迟，而 FGMT 通过 oversubscription 把延迟摊薄。** 48 个 warp 同时各发一个 gather，144 个 outstanding 就被填满了。

所以“给 NPU 加 scatter/gather 单元”不是加一个模块，它会拖进来：per-lane 寄存器文件、scoreboarding、MSHR 阵列、地址合并器、以及一整套内存一致性模型。**这在很大程度上就是一个 GPU。** 这也再次解释了 Apple 的选择：与其给 ANE 补这套，不如给 GPU 补矩阵单元，**补短板比补长板便宜得多**。

### 但产业正在长出一条“NPU 能抄”的路

Hopper 的 TMA 有个 **gather4 模式**：不是 per-lane 地址，而是一次 bulk 异步搬 4 行，行号来自一张**运行时索引表**。Blackwell 往上继续扩。

这恰好是 NPU 可承受的形态：

- 异步（无 per-lane 寄存器压力）
- bulk（描述符一次覆盖多行）
- 但**行地址来自数据，不是来自程序**

也就是说，硬件演化正在把“自由 scatter/gather”**降维成“带运行时索引的描述符列表”**：丢掉了 per-lane 的极端自由度，但保住了“动态性不进入控制流”这个关键性质。**这正是上一轮我说的那个 descriptor generator，只是它有了 TMA 这个现成的血统和专利路径。**

<details>
<summary>为什么这条路线对端侧 ROI 最高</summary>

它同时满足四个条件：

1. **不需要 FGMT / 抢占 / forward progress**，纯粹是数据通路增强，不破坏 NPU 的静态调度模型；
2. **不需要原子操作**（histogram 可以留给标量核，N 在端侧本来就小）；
3. 它把 MoE 的 permute、expert 权重 gather、varlen 的 tile 表三件事**用同一个单元解决**；
4. 它不改变“静态调度”这件事，只是让**有界动态**变得可表达；而 MegaBlocks 和 SparseCore 证明有界动态就够了。
</details>

---

## 六、但要诚实：端侧的账和服务器完全不同

我上一轮的 roofline 分析在这里要收紧一下。**在端侧，缺 scatter/gather 的“代价”其实不是百分比，而是断崖。**

先算百分比，你会发现它小得惊人：

- Permute 4096 token 的激活：16 MB 读 + 16 MB 写 = 32 MB，占 prefill 权重流量 15 GB 的 **0.2%**；
- 用标量核模拟 descriptor 生成：8192 条 × ~30 周期 ≈ 12 万周期 ≈ **60 µs**，占 prefill 615 ms 的 **0.01%**；
- 32K 上下文 decode 的 paged KV gather：32 层 × 512 页 ≈ 1.6 万条描述符 ≈ **0.2 ms**，占 decode 32 ms 的 **0.6%**。

**所以在端侧，即使把 scatter/gather 全部用标量核模拟，性能损失也在 1% 量级。** 这一点跟服务器完全相反（服务器上 routing + permute 能占 MoE 端到端延迟的 10–20%）。

那真正的代价在哪？**在“能不能编译出来”。**

<details>
<summary>断崖是怎么形成的：一次 fallback 放大 5 倍带宽</summary>

端侧 NPU 遇到不支持的算子只能 fallback 到 CPU/GPU，而 fallback 的边界会**把 KV cache 一起拖过 DRAM**。

Llama-3-8B 级模型，32K 上下文，int8 KV：

$$\text{KV} = 32\ \text{层} \times 32\text{K} \times 8\ \text{head} \times 128 \times 2(\text{K,V}) \times 1\ \text{B} = 2\ \text{GB}$$

每 token 的权重流量约 1.5–4 GB。**如果每层都有一个 fallback 点，KV 就要往返一次，等于每 token 额外 2–4 GB，是权重流量的 1–3 倍。**

这已经不是“慢 20%”，而是“能不能用”的问题。所以端侧的核心矛盾是：

> **不是 scatter/gather 本身太慢，而是“没有它”这件事会把算子推出加速器，而每次推出都会拖着 KV cache 走一遍 DRAM。**

这就是为什么端侧要的是 **expressibility（能表达）**，而不是 **performance（跑得快）**。也是为什么“编译期桶化 + 有界 dispatch”这条看起来笨的路，对端侧其实是对的：**它牺牲的算力和带宽都是过剩的，换来的是“不出加速器”。**
</details>

---

## 收束

用一句话回答你的问题：

> **Scatter/gather 之所以是动态 dispatch 的枢纽，是因为它把“数据依赖的不规则性”封闭在一条指令内，使动态负载能以直行代码的形式被编译。**
>
> **它之所以必须硬件化，不是因为标量核算不出地址，而是因为：**
> 1. **只有它能把动态性挡在控制流之外**；否则会变成“变长循环 + 跨引擎屏障”，而静态调度器无法为这种结构放置同步点；
> 2. **地址在寄存器里（数据）还是描述符里（程序）**，决定了它属于运行时还是编译期；
> 3. **Little's Law 要求的并发 outstanding 数（端侧 ~144，HBM 数万）**，只有硬件队列能提供，标量核的 scoreboard 差三个数量级。

而 NPU 的处境可以概括成 **“SIMD + DMA，缺了第三条腿”**：

- 缺 **scatter/gather** → 动态性升级成控制流 + 同步 → 编译期必须静态化（`max_ids`）；
- 缺 **FGMT** → 即使补上 scatter/gather，延迟也换不成吞吐，所以补它就等于在造 GPU；
- 缺 **per-lane 私有存储** → gather 落地变成 scratchpad scatter，还要付 bank conflict 和 repack 的双倍带宽。

**三条都不是“性能不足”，而是“表达力缺失”。** 而表达力缺失在端侧表现为断崖（fallback + KV 放大），在服务器端才表现为可量化的带宽税。这是同一件事在两个尺度上的两种面孔。
