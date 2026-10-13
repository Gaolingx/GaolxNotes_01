# 隐藏状态直接到TTS解码的可行性探讨

## 🧑‍💻 User

为什么tts没有人想到把所有复杂度吞进离散 Tokenizer 里，token all in one，让codec根据llm输出的hiddenstate流经多层codebook重建出波形，无论如何都比gptsovits这种级联架构好得多吧，而且更好scaling还能保留llm预训练知识，甚至还能借助mtp预测多层codebook降低ttft

---
## 🧑‍💻 User

为什么tts没有人想到把所有复杂度吞进离散 Tokenizer 里，token all in one，让codec根据llm输出的hiddenstate流进多层codebook直出波形，无论如何都比gptsovits这种级联架构好得多吧

---
## 🧑‍💻 User

为什么tts没有人想到把所有复杂度吞进离散 Tokenizer 里，token all in one，让codec根据llm输出的hiddenstate流经多层codebook直出波形，无论如何都比gptsovits这种级联架构好得多吧，而且更好scaling还能保留llm预训练知识，甚至还能借助mtp预测多层codebook降低ttft

---
## 😀 默认助手 (7)

## 结论先放前面

这个想法**不是没人想到，而是已经被做出来了不止一次**——但真正落地的形态和你的设想有一个关键分歧：**没人让 LLM 的 hidden state 直接去驱动 codec 重建波形，而是让 hidden state 去驱动一个「小的声学生成器」**。Sesame 的 CSM、Kyutai 的 Moshi 基本就是你说的一半;CosyVoice2 / F5 / E2-TTS 是你说的另一半。你的直觉方向（统一、保留 LLM 先验、多码本并行降延迟）是对的,但**"把复杂度全吞进离散 tokenizer"这个具体形式被三个硬约束挡住**。

先把你混在一起的三个技术点拆开,否则讨论会糊:

| 你的设想要素 | 对应现实工作 | 现实结论 |
|---|---|---|
| 复杂度吞进离散 tokenizer | VALL-E, AudioLM, EnCodec/SNAC/Mimi | ✅ 已做,但速率-失真有硬墙 |
| LLM hidden → codec 直接重建 | 基本没人这么做 | ❌ 被"随机性从哪来"和"hidden 不是声学充分统计量"挡住 |
| hidden → 小声学头 → 波形 | CosyVoice2, F5-TTS, E2-TTS, CSM, GLM-4-Voice | ✅ 这是当前主流 |
| 多层码本 + MTP 并行降 TTFT | Moshi depth transformer, MusicGen delay pattern, Kyutai TTS | ✅ 已做,但质量税和 TTFT 瓶颈位置和你想的不同 |

---

## 一、离散 token 采样的本质是"随机性的注入场所"

TTS 是典型的一对多映射 $p(\text{audio}\mid\text{text})$,条件熵极高(韵律、音色、时长、呼吸)。任何**确定性**的 $h \to \text{codec} \to \text{wave}$ 路径,都在最小化 $\mathbb{E}\|x-\hat x\|^2$,最优解是条件均值——**听起来就是闷、糊、机器人腔**(这也是为什么 L1/L2 vocoder 被 GAN/diffusion vocoder 淘汰)。

所以生成建模的"采样"必须发生在某处:

- **采样离散 token**(VALL-E/CosyVoice):随机性在 LM 的 categorical 采样里,codec decoder 是确定性的。
- **连续随机生成**(flow matching / diffusion head):随机性在 head 里。
- **你说的 hidden 直连 codec**:两条路都堵死了——hidden 本身是采样 token 的确定性函数,codec decoder 又通常是确定性的。**你想省掉的那次采样,恰恰是 TTS 的命根子。**

这就是为什么所有"统一架构"最终都长成 **backbone(出 hidden)× 声学头(做随机生成)**,而不是"hidden → codec 直接重建"。

---

## 二、hidden state 不是声学的充分统计量

要让 codec 从 $h$ 完美重建波形,需要互信息意义上的

$$I(h;\,x_{\text{audio}}) \;\ge\; H(x_{\text{audio}}) - \epsilon$$

但 LLM 的 $h$ 是在 **next-token CE** 下训出来的:它富含语义/内容信息($I(h;\text{semantic})$ 高),声学细节(齿音、气息、微韵律、音色频谱)几乎不在其表示里。于是只剩两条路:

<details>
<summary><b>路 A:冻结 backbone,新学声学头(CSM/Moshi 的选法)</b></summary>

- 优点:复用全部预训练知识,算力最省。
- 代价:**级联并没有消失**。你只是把一个 100M 的 acoustic decoder 塞进了同一个仓库/同一张图里,它依然要 AR 地预测 Mimi 的 8 层 RVQ。你说"token all in one",但实际上是 **backbone + 第二级 token 预测器**,这正是 Sesame CSM 的架构。

</details>

<details>
<summary><b>路 B:联合训练 codec + LM,让 hidden 真的携带声学信息</b></summary>

- 这就等于**端到端训练**,要穿过量化做梯度,需要 straight-through / Gumbel / FSQ,量化误差与梯度噪声一起进 LM。
- 更致命的是:**你会破坏"复用预训练知识"这个前提**。端到端从头训,数据需求爆炸,且 text 预训练的先验在声学残差上几乎帮不上忙——这正是 AudioLM 当年发现"语义 token 和声学 token 必须分层"的原因。

</details>

一句话:**你想同时要"复用预训练"和"把全部复杂度吞进 tokenizer",这两件事在信息论上是打架的。**

---

## 三、速率-失真的算术比直觉残酷

codec 的码率是

$$R \;=\; f \cdot K \cdot \log_2 N \quad (\text{bit/s})$$

Mimi 就是个好例子:$f=12.5\,\text{Hz},\;K=8,\;N=1024$,得 $R = 12.5\times8\times10 = 1000\ \text{bps}$。

而 LLM 要建模的**序列长度**是 $L = f\cdot T$ **乘以 $K$**(如果拉平)。60 秒音频:

- 单码本 12.5 Hz → 750 token
- 8 码本拉平 → **6000 token**

同一段文本大约 150–200 token。也就是说语音序列比文本长 **4× 到 30×**。你想"all in one"意味着**这些 token 的熵全部由 LM 承担**,而文本预训练的先验对声学残差几乎无用——误差随序列指数累积(exposure bias)。

要降序列长度就得降 $f$ 或降 $K$,但降 $f$ 就是降重建保真(单码本低频典型如 WavTokenizer 的质量折衷),降 $K$ 就丢掉残差细节。**没有免费午餐,只有把复杂度搬到哪一层的选择。**

---

## 四、多码本并行 + MTP:降的是吞吐,不是 TTFT

这是你设想里最接近"已实现"的部分,但因果要理清:

- **层间强依赖**:RVQ 第 $k$ 层是第 $k-1$ 层的残差,$p(c_k \mid c_{<k})$ 依赖极强。纯并行预测会掉质量。
- 现实方案分别是:
  - **AR + NAR 精修**(VALL-E)
  - **delay pattern**(MusicGen、Kyutai TTS):错位并行,保留因果但一次前向出多码本
  - **depth transformer**(Moshi):backbone 出一个帧隐状态,小 transformer 逐层展开 8 码本
  - **并行迭代细化**(SoundStorm、MaskGCT):置信度驱动的非 AR
- **MTP(DeepSeek-V3 式)本质是给模型加"预判未来若干 token"的辅助头**,收益主要在**训练信号密度**和推理时的 speculative decoding **吞吐**(tokens/s),它改善的是 **TPOT** 远多于 **TTFT**。
- TTFT 的真实瓶颈是:(1) prefill 长度,(2) AR 逐帧串行,(3) codec decoder 是否严格流式。**降 TTFT 靠的是 chunk 级并行 + 流式 causal vocoder + 更短 prefill**,MTP 只是其中一环。

---

## 五、那为什么 GPT-SoVITS 还能活着?

因为"比它好得多"这个判断只在上限维度成立:

| 维度 | 级联(GPT-SoVITS 类) | 统一 LLM+声学头(CSM/CosyVoice2 类) |
|---|---|---|
| 上限/自然度 | 中 | **高** |
| 数据效率 | **高**(小数据可训) | 低(吃海量语音) |
| 可调试性 | **每级独立** | 全黑盒 |
| 保留 LLM 先验 | 否 | **是** |
| 12GB 卡能跑 | **能** | 难 |

顺带说,GPT-SoVITS 本身也不是级联的最先进形态,它甚至不算"典型语音识别级联",它是 **SSL 语义 token + VITS decoder**。拿它当靶子会低估现代级联(如 NaturalSpeech 3 的 factorized codec + diffusion)。

---

## 六、如果你要真做这个方向,正确的形态是什么

<details>
<summary><b>我建议的架构(其实就是把"级联"内化到一层里)</b></summary>

1. **Tokenizer**:选单流 + 高保真(FSQ / 单码本高帧率)压缩,别用 8 层 RVQ 拉平,序列长度是 scaling 的命门。
2. **backbone**:冻结或轻量 LoRA 的文本预训练 LLM,输出 hidden。
3. **声学头**:**flow matching / diffusion**,而不是"hidden → codec 确定性重建"。这是随机性的唯一合法出口。
4. **多码本并行**:若保留 RVQ,用 delay pattern 或 depth transformer;MTP 作为辅助训练目标提吞吐。
5. **训练目标**:$\mathcal{L} = \mathcal{L}_{CE/CD} + \lambda\,\mathcal{L}_{recon} + \mu\,\mathcal{L}_{flow}$ 联合,防止 hidden 退化成纯语义表示。

</details>

**一句话总结**:你的方向判断是对的(统一、保留先验、并行码本),但"LLM hidden 流经码本直接重建波形"之所以不是主流,是因为它同时要求"消除采样随机性"和"复用与声学无关的文本先验",这在信息论上自相矛盾。现实的最优解是 **backbone(保留知识)+ 专用随机声学头(承担复杂度)**——**级联没有消失,只是被内化进了一个模型里**。

---
## 🧑‍💻 User

也不是说直接让llm出语音token，我是说能不能在大llm基础上先训个独立小AR负责在 llm 的 hidden states 上生成语音码本，同时这个AR的 backbone 本身预测第 0 层码本，MTP（Multi-Token Prediction）模块一次性生成后面 N-1 层残差码本....不知道qwen3tts/qwen3omni是怎么做的

---
## 🧑‍💻 User

也不是说直接让llm出语音token，我是说能不能在大llm基础上先训个独立小AR负责在 llm 的 hidden states 上生成语音码本，同时这个AR的 backbone 本身预测第 0 层码本，MTP（Multi-Token Prediction）模块一次性生成后面 N-1 层残差码本

---
## 😀 默认助手 (7)

## 一句话结论

**你描述的架构，Qwen3-Omni / Qwen3-TTS 已经这么做了，而且细节几乎和你设想的一模一样**——独立小 AR 吃大 LLM 的 hidden state、backbone 出第 0 层码本、MTP 模块出残差码本、轻量 codec 渲染波形。唯一和你想的不一样的是:**MTP 不是"一次性并行出 N-1 层"，而是固定步数 AR 串行出 15 层**，而且这 15 层恰恰是整条流水线 94% 的串行开销。

下面是我实际查到的架构（Qwen3-Omni 技术报告 [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)、Qwen2.5-Omni 技术报告 [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)，以及 WhisperKit 团队逆向 Qwen3-TTS 权重后的 [arXiv:2609.37007](https://arxiv.org/abs/2609.37007)）。

---

## 一、Qwen3-Omni 的 Talker：就是你说的"独立小 AR"

报告原文（§2.4 Speech Generation）:

> "our approach departs from Xu et al. (2025) by operating directly on **RVQ tokens**. The Talker employs a **hierarchical prediction scheme**: the backbone ingests the aggregated codebook features of the current frame and uses a **linear head to predict the zeroth codebook**, after which a **multi-token prediction (MTP) module generates all residual codebooks**... waveform reconstruction is simplified to a **lightweight causal ConvNet (Code2Wav)**"

模块表（Table 1）:

| 模块 | 架构 | 参数 | 流式 |
|---|---|---|---|
| Thinker | MoE Transformer | 30B-A3B | ✓ |
| **Talker** | MoE Transformer | **3B-A0.3B** | ✓ |
| **MTP** | Dense Transformer | **80M** | ✓ |
| **Code2wav** | **ConvNet** | 200M | ✓ |

端到端首包延迟 **234 ms**（冷启动），码率 **12.5 Hz**。

这就是你的方案:一个 30B 的 LLM 冻结着出 hidden，一个 3B 的小 AR 在 hidden 上生成语音码本（Talker），一个 80M 的 MTP 出残差码本，一个 200M 的 ConvNet 出波形。**连参数配比都符合"小 AR + 极小 MTP"的直觉。**

---

## 二、Qwen3-TTS 的精确数据流：16 层 RVQ，1+15 次调用

WhisperKit 团队拆解了 Qwen3-TTS 的实际权重，给出了非常干净的两级循环:

> "Qwen3-TTS uses **two nested loops** over RVQ codes: the **outer loop** runs a text-conditioned **CodeDecoder** once per **80 ms** audio frame to emit the coarse code $c_0$, and the **inner loop** runs the **MultiCodeDecoder 15 times autoregressively** to emit the 15 residual codes $c_1,\dots,c_{15}$"

- **CodeDecoder**（28 层，1 次/帧）→ 出 $c_0$
- **MultiCodeDecoder**（15 次固定步 AR）→ 出 $c_1..c_{15}$
- 每帧共 **16 个 RVQ 码**，合成帧 embedding $\{c_0,\dots,c_{15}\}$，再和文本 token $t_i$ 相加（$\Sigma$）去条件下一帧
- **Speech Decoder** 吃 RVQ 码流 → 波形

> "$\sim$1 s of audio is represented with $\sim$200 tokens: **13 outer loop CodeDecoder calls and 187 inner loop MultiCodeDecoder calls**"

$$\text{calls/s} = \underbrace{12.5}_{\text{帧率}} \times \underbrace{(1 + 15)}_{c_0 + \text{残差}} = 200,\qquad \frac{187}{200} \approx 94\% \text{ 来自 MTP}$$

<details>
<summary><b>关键细节：MultiCodeDecoder 是"按位置多头"结构</b></summary>

> "the Qwen3-TTS MultiCodeDecoder is **multiheaded by construction**: each of the 15 residual codebook positions has its **own dedicated input embedding table and its own LM head**... the previously committed code at position $i$ is looked up in table $i$ and the hidden state at position $i$ is projected by $\text{LMH}_i$"

也就是说:**15 个残差层不是共享一个头,而是各有独立的 embedding 表和 LM head**。这从工程上直接否掉了"一次前向并行出所有残差层"——每层的输出分布不同,而且第 $k$ 层要把**已采样定下的** $c_{k-1}$ 查进自己的表里才能算。这是链式依赖,不是可并行分支。

另一个独立工作（SCIC, [arXiv:2609.39088](https://arxiv.org/abs/2609.39088)）通过分析 Qwen3-TTS 的码本发现:"**Energy 集中在早期残差码本,而 Pitch 横跨更深的码本前缀**"——残差层之间语义分工明确,这也解释了为什么必须 per-position head。

</details>

---

## 三、和你的设想的三处关键分歧

### 分歧 1：MTP 不是"一次性生成"，而是固定步数 AR

Qwen3-Omni 报告里对 MTP 的描述:

> "The MTP Module is an **ultra-lightweight fixed-step autoregressive dense transformer**... Its fixed-step autoregressive inference mechanism allows it to effectively leverage a **fixed KV cache memory space** for acceleration"

而 WhisperKit 那篇论文明确说了为什么不做并行:

> "Non-speculative alternatives such as **delay pattern interleaving** and **masked parallel generation** also reduce sequential RVQ calls, **but require retraining the backbone and change its output distribution**, which is undesirable for an already deployed model."

即:delay pattern（MusicGen 那套）和 masked 并行（SoundStorm 那套）**确实能省串行步数**，但要重训 backbone、且会改变输出分布。所以他们选择保留 AR 链，**从另一头解决延迟**。

### 分歧 2：降 TTFT 的真实杠杆是投机解码，不是并行出码本

既然 94% 的串行调用来自 MTP 内层循环，他们就直接对**内层循环**做 speculative decoding:

- 复用**已有的 per-position LM heads**当 drafter（因为结构天然是 multihead），只加 **3072 个可训练参数**，不加新头、不改输出分布
- 树状 attention 提升接受率：draft 深度 $K$、每位置 $m$ 个候选时，树有 $N$ 个节点但要在一个长度 $q=N+1$ 的 query 里验证
- 实测 **2.47 accepted tokens/call**，把 **200 → 88 次串行调用/秒**，RVQ 生成 **2–2.2× 加速**

所以：**你想用 MTP 降 TTFT 的方向是对的，但真正的杠杆在"让 MTP 自己可以被投机解码打穿"，而不是"让 MTP 一步出完所有层"。** 后者会牺牲残差层的链式依赖。

### 分歧 3：小 AR 的输入不是纯 hidden，而是 hidden + 上一帧码本

$t_i$（文本 token）**与上一帧 16 个 RVQ 码的 embedding 求和**后，才去条件当前帧的 CodeDecoder。小 AR 不是"纯吃 hidden state"，hidden 只是其中一个条件源。

---

## 四、Qwen 官方原话：为什么不能让 hidden 直接驱动重建

这条最值钱，出自 Qwen2.5-Omni 报告（讲 qwen-tts-tokenizer 那一节）:

> "Thinker's representations primarily express **semantic similarity** in the representational space rather than **phonetic similarity**. Consequently, even phonetically distinct words may have very similar high-level representations, **necessitating the input of sampled discrete tokens to eliminate such uncertainty**."

**这就是上一轮我说的"hidden 不是声学充分统计量"的官方版本。** LLM hidden 在语义空间里近，在声学空间里不近；离散采样那一步不是多余的中间层，而是**把语义歧义消解成声学确定性**的必要操作。这也再次说明:随机性必须有个采样场所。

---

## 五、Qwen2.5-Omni → Qwen3-Omni 的演进，正是你设想的路线

这个演进本身就是对你直觉的背书:

| | Qwen2.5-Omni | Qwen3-Omni |
|---|---|---|
| Talker | 单轨（单码本） | **多轨多码本** |
| 码本 | 单码本 | **多层 RVQ**（报告称"increased capacity supports faithful modeling of diverse voices, paralinguistic cues"） |
| 波形阶段 | 滑窗 block-wise **DiT + Flow Matching → mel → BigVGAN** | **轻量 causal ConvNet** |
| 码率 | — | **12.5 Hz**，单帧即时合成 |
| 首包 | 需等足够 block context（滑窗回看 2 block、前瞻 1 block） | **234 ms**，生成 1 个 token 即可出波形 |

Qwen3 报告对这次替换的说法:

> "Leveraging the **representational capacity of these codebooks**, we replace computationally intensive **block-wise diffusion with a lightweight causal ConvNet**, enabling streaming from the **first codec frame**"

**你说的"多码本容量够大时，波形重建可以退化成轻量确定性渲染器"，Qwen3 就是这么做的，而且声称保真度优于 DiT 声码器、FLOPs 更低。** 你的判断在这点上完全正确。

同时也要注意:**随机性仍然 100% 在离散 token 采样里**——每帧 16 次 categorical 采样。Code2Wav 是确定性的。所以"把生成建模收进 tokenizer"这个说法准确，但收进去的是**采样 + 信息容量**两件事，波形解码器只是确定性的 renderer。

---

## 六、你的设想 vs Qwen3 实际实现

| 你的设想 | Qwen3 的做法 | 是否吻合 |
|---|---|---|
| 大 LLM 基础上训独立小 AR 吃 hidden | Thinker 30B-A3B → Talker 3B-A0.3B | ✅ 完全一致 |
| 该 AR 的 backbone 预测第 0 层码本 | linear head 出 $c_0$ | ✅ 完全一致 |
| MTP 一次性生成后 N−1 层 | MTP 80M **固定步 AR 出 15 层** | ⚠️ 改为串行，因为链式依赖 + per-position 分布 |
| codec 根据多层 codebook 重建波形 | Code2Wav 200M causal ConvNet | ✅ 一致，且比 DiT 更轻 |
| 保留 LLM 预训练知识 | Thinker 复用，Talker 只 3B-A0.3B | ✅ 一致 |
| MTP 降低 TTFT | 改用 **speculative decoding** 打穿 MTP 内层循环 | ⚠️ 方向对，杠杆点不同 |

---

## 七、结论

你的方案不是"没人想到"，而是**Qwen3 系列的官方设计**。真正值得从 Qwen 这套里学到的是那个反直觉的教训:

> **多码本方案的代价不在 backbone，而在残差深度。** 16 层码本把每帧的串行调用放大了 16 倍（200 calls/s 里 187 次是 MTP），所以"MTP 出残差码本"这个模块既是让方案成立的功臣，也是延迟的最大瓶颈。把它做成一次性并行，会丢掉 RVQ 的链式依赖和 per-position 分布（Qwen 团队明确说那需要重训 backbone 且改变输出分布）；正确的破法是**让 MTP 结构天然可投机解码**。

而"hidden 直接驱动 codec 重建"之所以被放弃，Qwen 自己给的理由最干净:**hidden 表达的是语义相似性，不是语音相似性**——所以那一步离散采样不能省。
