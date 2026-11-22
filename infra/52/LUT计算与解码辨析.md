# LUT计算与解码辨析

## 🧑‍💻 User

现在的fused densegemm kernel无非就是那一套
访存/预处理 -> mat/vec/sfpu运算 -> 写回/后处理
这种特征是
比如说你有一些很复杂的quant 像是一些vq-like或LUT based或NF4这种的
你可以访存的core一边读quant weight一边dequant到L1
compute core就只要当成普通的matmul
所以我可以同一套matmul kernel改访存代码就能测试几十种不同的quant

如果计算访存解耦话这么干完全没问题，tenstorrent上话
我是把dequant拆成基础lut + vector work
因为他compute core是三个CPU + 一堆vec/mat后端
所以我可以一个core负责matmul另一个core负责vector work dequant
还有就是可以搞core to core
不用走dram直接noc写到别的tensix core

不过目前主要问题是因为这种作法对GPU来说也很难崩 所以学术研究很少
得找到一个效果好的baseline先
方法本身能行 但是比较没有现成的quant算法能用

---
## 😀 默认助手 (7)

看完你的描述，我认为卡点不在“缺少 quant 算法”，而在于**你还没有把“解耦后的 dequant”写成一个可以优化代价的目标函数**。一旦写出来，候选算法其实不少，只是它们没被归到“LUT quant”这个类名下面。

<details>
<summary><b>先确认一下我们对硬件的共同前提（可展开）</b></summary>

我查了 tt-metal 的 `tech_reports/data_formats` 和 tt-llk 的文档，确认几个关键点，后面要用：

- Tensix 的 unpacker 原生支持 **BFP8 / BFP4 / BFP2**（`bfloat8_b` / `bfloat4_b` / `bfloat2_b`），语义是**每 16 个元素共享一个指数**，元素内是 sign + mantissa（mantissa 的 MSB 是隐含的 1）。
  → 所以 `bfloat4_b` ≈ **1 sign + 2 bit 小数**（4 个幅值档），`bfloat8_b` ≈ 1 sign + 6 bit 小数，`bfloat2_b` 退化成纯符号。这点很重要，见第 4 节。
- MX 格式（MXFP8/4）在 tt-llk 里已有专门处理，和 BFP 共享指数那套是两条路径。
- 你说“三个 CPU + vec/mat 后端”是 TRISC0/1/2 + BRISC/NCRISC 的简化说法。**这里有个需要你自己拍板的点**：BRISC/NCRISC 是标量 RISC-V，能不能碰 SFPU 决定了你的“访存 core 做 dequant”是*真解耦*还是*名义解耦*（如果真的只能在 TRISC1 上跑 dequant，那它还是和 matmul 抢发射槽）。
</details>

## 1. 真正的目标函数

你的 kernel 形状是 $f(\text{code}) \to w$，decoder 的代价只有两项：

$$
\text{cost}(B\text{-tile}) = \underbrace{\frac{c_d}{M_{\text{block}}}}_{\text{dequant ALU / MAC}} \;+\; \underbrace{\frac{b_{\text{out}}}{b_{\text{in}}}}_{\text{L1/NoC 写放大}}
$$

其中 $c_d$ 是每元素 dequant 指令数，$M_{\text{block}}$ 是这个 B tile 在 core 里被复用的 M 分块大小。4-bit 解成 bf16 意味着 $b_{\text{out}}/b_{\text{in}} = 4$。

**这个式子的两个直接推论**：

- $M_{\text{block}} = 1$（decode 阶段）时，$c_d$ 完全无法被摊薄，解耦纯亏。所以你的结果必须画成**随 $M$ 变化的曲线**，而这恰好就是你论文里最有说服力的图：「格式 × 复用度的交叉点」。
- 你现在“一套 matmul 改访存就能测几十种 quant”，本质是把 $c_d$ 变成一个可插拔参数。那真正的研究问题就是：**在 $c_d \le c_{\max}$ 的约束下，最大化 rate–distortion**。文献里没人这么提，因为 GPU 上 $c_d$ 几乎被 MMA 掩盖掉了，这个约束是松的。

## 2. 把“LUT + vector work”拆成四类（这是选型的坐标系）

| 类别 | decode 形式 | 每元素代价 | 代表 | 在你的框架里 |
|---|---|---|---|---|
| **A. 逐元素 LUT** | $w = s_g \cdot T[c]$ | 1 unpack + 1 permute + 1 FMA | NF4, IQ4_NL, MXFP4, NVFP4, 任意非均匀标量量化 | ✅ 几乎免费，SFPU 的主场 |
| **B. 多级 LUT 相加** | $w = \sum_k s_k T_k[c_k]$ | $k\times$ A | AQLM (2×8), GGUF IQ2/IQ3 grid, 残差 LUT | ✅ 代价线性，可接受 |
| **C. 结构化码本 VQ** | E8 格点 = 若干次整数加/移位 | 1–3 次加 + shift，**无 gather** | QuIP# (E8P), HIGGS | ✅✅ 你最该关注的一类 |
| **D. 真 gather VQ / PQ** | 按 code 取 $d$ 维向量，需变步长 byte permute | 高，且 SIMD 不友好 | GPTVQ, VPTQ, 传统 PQ | ⚠️ 只能靠 core-to-core + 共享码本才划算 |

**关键洞见：C 类才是你的甜点区。** 它的质量接近 VQ，但解码退化成“查小表并做算术”，不需要数据重排。而现有 GPU 论文里 C 类很少，因为大家默认 A 类（per-group affine）已经够用；但在你这里，A 类的 $c_d$ 已经压到 3–4 条指令，**再往上提质量只靠 $c_d$ 是不够的，必须换码本结构**。这就是空白的来源。

## 3. 我建议的 baseline 阶梯（可以直接开测）

按“先建可信度、再冲天花板”排：

1. **`bfloat8_b` / `bfloat4_b` 硬件路径** — 你必须先有一个“$c_d = 0$”的点。如果灵活 dequant 打不过硬件 BFP4，整个框架没有存在理由。顺带说：Xylolabs 已经有 `tensix-bfp-gptq` 把 LLM 的 linear 用 GPTQ round 到 BFP4，HF 上也有 `Qwen3-8B-Tensix-BFP4` 这类 checkpoint —— 你的 GPTQ+BFP4 质量 baseline **今天就能拿到**，不用自己训。
2. **NF4 + double quant**，和 **IQ4_NL**（llama.cpp 的非线性 4-bit LUT，per-32 scale）。两者都是 A 类、$c_d \approx 3$，是“标准 4-bit”质量线。
3. **MXFP4 (E2M1+E8M0 per-32) / NVFP4 (E4M3 per-16)**。必须放进去，因为这是审稿人唯一认识的 4-bit 标准，而且和 BFP4 正好构成“指数预算 vs 尾数预算”的对照。
4. **AQLM 2×8** — B 类压力测试，$c_d \approx 6$，质量天花板之一。
5. **QuIP# E8P** — C 类，用来证明你的框架能吃下“非平凡码本”。
6. **VPTQ** — D 类，用来测 gather 代价的上界；也是你 core-to-core + 共享码本那条路唯一有意义的对手。

> 顺便纠正一个判断：**“没现成的 LUT quant”不成立。** llama.cpp 的 IQ 系列（IQ1_S / IQ2_XXS / IQ3_XXS / **IQ4_NL**）就是字号意义上的 LUT quant —— 权重解成“符号 × 若干 grid LUT 查表 + 块内缩放”，而且附带 importance-matrix 的校准流程。ID 里那个 `NL` 就是 non-linear LUT。这是 2–3 bit 档位除 AQLM/QuIP# 外质量最好的公开方案，代码可读，直接抄格式。

## 4. 一个可能被你低估的硬 baseline：BFP4 是“免费”的 3-bit

`bfloat4_b` = 1 sign + 2 bit 小数 + **每 16 元素共享指数**。对比 MXFP4 = 1 sign + 2 bit 指数 + 1 bit 小数 + per-32 缩放。两者都是 4 bit/元素，但预算分配相反：

- BFP4 **块内动态范围差**（16 个权重共用一个指数），但**块内精度好**（2 bit 小数）。
- MXFP4 相反（每元素 2 bit 指数，但只有 1 bit 小数）。

对权重这种已经做过 per-channel 归一化的张量，BFP4 的先验其实更合适。所以真正的“成本为零”门槛不是 4-bit 通用质量线，而是 **BFP4+GPTQ** —— 这个点比大家想象的更靠右。你要做的格式如果只是“NF4 级别的 LUT + 一点点优化”，很可能被它按在地上打，而且它 $c_d=0$。这一点我认为值得单独写一页，它本身就是个不错的结论。

## 5. 真正值得做的格式（研究空缺）

<details>
<summary><b>三个候选，按我下注的顺序（展开）</b></summary>

**① 逐元素可学习 LUT + GPTQ 目标（最稳）**
$c_d$ 和 NF4 一样，但 $T[\cdot]$ 的 16 个值按层/按组由 GPTQ 式 block-wise 误差最小化学出来（不是分位数初始化）。这是“Lloyd–Max 加上 GPTQ 目标”，本质是 1-D 学习码本。已知做得接近的只有 NF4 的分位网格和一些 learnable-clip 工作，**“每层学一个 16 项 LUT”这个正中间的点反而空着**。实现成本极低，因为你框架里 LUT 就是个指针。这是我最推荐的第一个自研格式。

**② 残差/位平面双 LUT（bit-plane LUT）**
$w = s\,(T_{\text{hi}}[c_{\text{hi}}] + \alpha\, T_{\text{lo}}[c_{\text{lo}}])$，hi 取 3 bit、lo 取 1–2 bit。表达能力超过 4-bit 均匀量化（非均匀、非对称），代价是 $c_d$ 翻倍。它和 GGUF IQ2/IQ3 的 grid 是同一个思路，但你可以把两级表**分开在两个 core 上算**（hi 表常驻、lo 表广播），这是解耦架构独有的自由度。

**③ 可编程 BFP：块级 LUT 覆盖尾数映射（最 architecture-specific，也最新颖）**
既然硬件 BFP4 已经免费展开尾数，那就让**每个 16-元素块的 4 个幅值档位由一个 per-block 的小 LUT 重映射**（$b_{\text{out}}$ 不变，只加一次 per-block 的 permute）。这把“块内共享指数”从约束变成自由度：指数管动态范围，LUT 管分布形状。我没见过公开工作做这个，而且它只有在你这种“访存 core 上跑 vector work”的设计里才自然——纯 fused 路径上这个 permute 会直接进 matmul core 的热路径。**如果你要一个“只有解耦才能做”的卖点，这个比“能测 50 种格式”强得多。**

</details>

## 6. 三个可能让这件事失败的风险（我认为你必须提前回答）

1. **它是 issue-bound 还是 bandwidth-bound？** 解耦只在“fused 版本被 unpack/发射槽卡住”时才赢。如果 fused 版本的瓶颈其实是 L1 读，那你把 4-bit 展开成 bf16 写进 L1，就是净亏 4× 写带宽。**这个测量应该先做，而且它决定整个工作的成色。**
2. **GPU 上的“难崩”你怎么回答。** 我理解你的意思是 GPU 上解耦收益太小（register 里的 dequant 被 MMA 掩盖），所以没人做。那叙事必须是 “near-memory / chiplet / 未来架构的代价模型”，而不是“我们比 GPU 快”。否则这就是一份 Tenstorrent 的工程报告。
3. **core-to-core 的真正理由是码本预算，不是搬运。** 解耦最不能自圆其说的地方是“我只是把 work 搬了个地方”。但它能自圆其说的地方是：**fused 设计里码本必须塞进每个 core 的 L1；解耦后可以有一个“码本 server core”，把码本预算放大一个数量级** —— 大码本 VQ 在 fused kernel 里根本不可能，在你这儿是 trivial。把这条当作核心 motivation，$c_d$ 只是次要论据。

## 7. 我会怎么排实验

1. 把 kernel 里的 dequant 抽成 `dequant_spec(code_bits, group_size, LUT_ptr, stages_k, scale_fmt)` —— 这是你唯一的工程投入，别再多做。
2. 先只测 **BFP4 vs NF4 vs IQ4_NL**（三个 A 类，$c_d \in \{0, 3, 4\}$），扫 $M \in \{1, 16, 32, 128, 512\}$。**这张图就能判断项目生死。**
3. 加上 AQLM 2×8 和 E8P，画 **quality vs $c_d$/elem**，BFP4 画成一条竖线。
4. 再上自研格式（我押 ① 和 ③）。
5. 统一用同一个 GPTQ 式目标去训所有格式 —— 否则格式比较会被“目标函数差异”污染，这是这类工作最常见的被拒理由。

如果你想，我可以把第 2 步那个 $M$–$c_d$ 交叉点的分析式（含 L1 读写、NoC 带宽、matmul 吞吐的符号化）推出来，直接给你一个“什么时候解耦才可能赢”的判据，省得你靠 profiling 撞。

---
## 🧑‍💻 User

现在LUT TensorCore更多是lut解码吗？有用lut做计算的吗

---
## 😀 默认助手 (7)

这问题需要拆成两条完全不同的路线；而且恰好相反：**现在叫“LUT Tensor Core”的那批工作，绝大多数是用 LUT 做计算，并非用 LUT 做解码。**

先说结论，再给依据。

## 1. 两条路线，语义完全不同

| | **LUT 做解码 (LUT-as-decode)** | **LUT 做计算 (LUT-as-compute / LUT-mpGEMM)** |
|---|---|---|
| 表里存什么 | **权重值**：$T[c] \to w$ | **激活的部分和**：$T[c] \to \sum a_i w_i$ |
| 索引是什么 | 权重的码 | 权重的码（同一套码） |
| LUT 之后干什么 | 普通 MMA 做乘加 | **没有乘加，只剩查表 + 加法** |
| 消除的目标 | 反量化开销 | **乘法，以及 dequant 本身** |
| 复用维度 | B tile 被 M 行复用 | 表被 N 个输出通道复用 |
| 代表 | NF4、GGUF IQ 系列、AQLM、QuIP#、VPTQ | **LUT-GEMM、T-MAC、LUT Tensor Core**、MADDNESS |

**数学上是同一个式子的两种因式分解方式。** 对逐元素量化，单靠交换求和顺序就能从 A 变成 B：

$$
y_{m,n}=\sum_{i=0}^{G-1} a_{m,i}\, w_{n,i} = \sum_{g}\sum_{i\in g} a_{m,i}\, s_g T[c_{n,i}]
\;\Longrightarrow\;
y_{m,n} = \sum_{g} \underbrace{\Big(\textstyle\sum_{i\in g} a_{m,i}\, s_g T[\,\cdot\,]\Big)}_{\text{预计算成一张 }2^{kG}\text{ 项的表}}\big[\,c_{n,g}\,\big]
$$

也就是说，**表里装的是 $\sum a_i \cdot \text{deq}(c)$，它按激活 group 现场生成，被 N 个输出通道复用**。乘法和 dequant 同时消失了。

<details>
<summary><b>LUT Tensor Core (ISCA'25, arXiv 2408.06003) 到底做了什么 —— 可展开</b></summary>

作者是 Mo / Wang / Wei / Cao / Ma / Xue / Fan Yang / Mao Yang，微软亚研 + Imperial + PKU + UW，和 T-MAC 同一批人。

它的出发点是一句批评：**"a conventional LUT implementation fails to achieve the promised gains"** —— 也就是说，LUT-GEMM 那条思路理论上省乘法，实际上表预计算开销和表存储把收益吃光了。它的三个贡献全部围绕这一点：

1. **软件侧**：operator fusion 压缩表预计算开销；**table symmetrization + weight reinterpretation** 压表存储（利用符号/对称性把 $2^k$ 砍下来）。
2. **硬件侧**：LUT-based Tensor Core，**elongated tiling shape**（把 M 拉长）来最大化表的复用 —— 因为表的复用度就是 N，tile 形状必须让一张表服务尽可能多的激活行；**bit-serial 设计**支持各种 weight/activation 精度组合。
3. **新 ISA + 编译优化**。

结论：比纯软件 LUT 实现快很多，比之前的 LUT 加速器 compute density / energy 高 $1.44\times$。**注意它从来没有提 dequant —— 它的整个卖点就是"绕过 mpGEMM 没有原生支持"这件事。**

另外 T-MAC（EuroSys'25）在 CPU 上的说法更直白：*"directly supports mpGEMM without dequantization, while simultaneously eliminating multiplications"*。
</details>

## 2. "表爆炸"在哪里形成，稳定区间又在哪

LUT-as-compute 的可行性只有一个约束：

$$
\underbrace{2^{k}}_{\text{表项数}} \ll \underbrace{N}_{\text{复用度}}, \qquad 2^{k}\times \text{entry\_size} < \text{L1}
$$

$k$ 是**一次查表吃掉的权重位数**（$k = b\times G$，$b$ 是每元素 bit，$G$ 是 group 长度）。$k=8$ 是 256 项（还行），$k=16$ 就是 64K 项 × 2B = 128KB，L1 塞不下，并且生成这张表的加法数是 $2^{16}$。**这就是"conventional LUT fails"的真实原因：瓶颈是存储和生成带宽，并非算力。**

所以在向量 ISA 上，真正的稳定区间是 $k=4$，最多 8：

- $k=4$ 的 16 项表可以用 **1 次 byte permute** 或 4 级 select tree 完成查找；
- $k=8$ 的 256 项表就要 8 级 select（每元素 8 条指令），比 dequant 的 3 条还贵。

**而 NF4 / IQ4_NL / 学习型 LUT 的那张 16 项表，正好是 $k=4$。** 这不只是巧合：它指向下面这个对你最有用的结论。

## 3. 对你项目的三个判断

**(a) 非均匀标量量化在两条路线里是同一张表 —— 你可以零成本双后端。**
如果你的格式是 $w = s_g T[c]$（NF4、IQ4_NL、我上一轮说的"可学习 LUT"），那么：
- dequant 路线：`load → permute(T) → FMA(s) → MMA`
- LUT-as-compute 路线：`permute(T) 装进表 → 生成部分和 → 查表 + 加`，**matmul 直接没有了**

两条路共用同一个 `T`。所以你的 `dequant_spec` 抽象可以天然多挂一个 backend，不用为它写第二套格式代码。这对你"一套 kernel 测几十种 quant"的目标是免费的额外收益。

**(b) 但两条路的复用维度不同，适用区间正交 —— 这是最关键的一点。**

$$
\text{dequant 的摊薄因子} = M,\qquad \text{LUT-as-compute 的摊薄因子} = N
$$

- LUT-as-compute 的成本只跟 $N$ 有关，**和 $M$ 完全无关** → 天生适合 **decode**（$M$ 小、memory-bound、向量核本来闲着）。这正是 T-MAC / LUT Tensor Core 全部盯着 decode latency 的原因。
- 你的解耦 dequant 要 $M$ 大才划算 → 天生适合 **prefill**。

所以两条路是**同一个 kernel 里按 $M$ 切换的两个后端**，并非竞争关系。这个"按算术强度切换 mpGEMM 实现"的说法，本身就是个站得住的论文骨架。

**(c) 别指望 LUT-as-compute 替代 matmul。** 它有硬件前提：**目标机器上没有能吃的 tensor core**。T-MAC 之所以成立，是因为 CPU 没有 tensor core；FPGA 那批（TeLLMe、PD-Swap、MADDNESS 系）同理。一条查表+加法的 MAC 大约要 1–2 条向量指令，一个 tensor core 的 MAC 吞吐比这高一到两个数量级。**在 Tensix 上有 FPU 的情况下，LUT-as-compute 只在 DRAM 带宽已经打满的时候才赢**（那时你用的是本来闲置的向量 ALU）。而你的主力论点是 prefill 且 $M$ 大 —— 那条路你必须走 dequant + MMA，别被 LUT-as-compute 带跑。

## 4. 顺便清一下术语歧义（容易踩坑）

<details>
<summary><b>“LUT” 在文献里至少有四个意思（展开）</b></summary>

1. **LUT-as-compute（查表代替乘法）** — LUT-GEMM / T-MAC / LUT Tensor Core / TeLLMe / MADDNESS。**你问的就是这一支，是主流。**
2. **LUT-as-decode（码本查表）** — NF4 / IQ4_NL / IQ2_XXS / AQLM / QuIP# / VPTQ。通常不被叫 "LUT"，所以你以为它少 —— 其实它是 4-bit 量化的默认形态。
3. **LUT 做非线性函数**（sigmoid/GELU/exp 用表近似）—— 这个在 Tensix 上是既有的（SFPU 侧），和 mpGEMM 无关，但**同一个 LUT 硬件单元可能可以复用**。建议你去 `tt-isa-documentation` 查一下 SFPU 的 LUT 指令到底是几项表、怎么寻址，这直接决定上面 $k$ 的稳定区间到底是 4 还是 8。
4. **FPGA 的 LUT 原语当神经元** — LUTNet / LogicNets / PolyLUT。这是"用 FPGA 的查找表单元实现布尔函数"，和上面三种毫无关系，搜的时候会被污染。

</details>

## 5. 一句话回答你

- **"LUT Tensor Core 更多是 LUT 解码吗？"** —— 恰好反过来：以它为代表的一整条线（LUT-GEMM → T-MAC → LUT Tensor Core）全是 **LUT 做计算**，它们的卖点明确写着 "without dequantization"。
- **"有用 LUT 做计算的吗？"** —— 有，而且这是 LUT 相关的硬件工作里唯一活跃的一支；但全部集中在 **memory-bound / 无 tensor core** 的场景（CPU 边端、FPGA 边端、decode 阶段），$k \le 8$、靠 $N$ 摊薄。

对你的具体建议：去把 LUT Tensor Core 的 **table symmetrization** 那节读了 —— 它做的事情本质是"用码本的对称性把 $2^k$ 压下来"。你要做的 VQ/学习 LUT 码本，**码本本身就是非对称的、学习出来的**，所以那套压缩用不上，但这反过来是一个空位：*能不能设计一个码本，让它在 $k \le 4$ 的同时又能承载 VQ 级的表达能力* —— 也就是我上轮说的"结构化码本"。这个位置现在真没人占。
