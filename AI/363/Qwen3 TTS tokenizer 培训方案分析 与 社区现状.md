# Qwen3 TTS tokenizer 培训方案分析 与 社区现状

## 🧑‍💻 User

Qwen3 tts tokenizer 12hz 用的是 lightweight causal ConvNet 解码器。要做到 16 层码本、近透明重建、且因果可流式，需要百万小时级音频、多尺度谱损失 + 对抗损失、RVQ 的 codebook collapse 处理，还要把 SSL 语义蒸馏进第一层码本（否则 LM 根本学不动——这正是 Qwen3-Omni 专门做 "AuT pretraining" 的原因）。自研这一层比训 LM 本身更难。 帮我研究下这部分比较好的training recipe是什么，为什么社区没人训练这种tts，gptsovits为什么更容易训练

---
## 😀 默认助手 (7)

我先去查证一下 Qwen3-Omni/Qwen3-TTS tokenizer 的公开细节，避免在配方层面给你错的东西。

DuckDuckGo 拦了。换 arXiv 官方 API 和别的搜索源。

搜索引擎基本被封，arXiv API 通。直接抓一手仓库和技术报告。

核心事实对上了。现在拉技术报告全文（tokenizer 那一节）和 GPT-SoVITS 的架构说明。

拿到 GPT-SoVITS 的一手说明了。现在读技术报告全文里 tokenizer 那几节。

关键信息已到位。再补 Qwen3-Omni 的 AuT / 语音生成 / 预训练细节，以及 Qwen3-TTS 的 tokenizer 评测节。

一手材料够了。再核对几件事：Mimi/开源 12.5Hz 路线、RVQ collapse 处理、以及 12Hz tokenizer 的实际配置。

配置拿到了，非常具体。再确认 Mimi 这条开源同规格路线的规模。

最后确认一下 Mimi 的配置，用来对比"社区其实训过同规格 codec"这一点。

先说结论：**你的技术直觉基本是对的，但有三处前提需要修正，而且其中一处修正会直接改变你的路线选择**——Qwen3-TTS-Tokenizer-12Hz 的 **encoder 配置和 Kyutai 的 Mimi 逐字段完全一致**，它不是从零自研，而是在 Mimi 架构上做 scale + 语义蒸馏 + 更好的 GAN recipe。

---

## 0. 前提校正

| 你的说法 | 核实结果 | 依据 |
|---|---|---|
| 12Hz tokenizer 用 lightweight causal ConvNet 解码器 | ✅ 对。但有细节：不是纯 ConvNet，是 **conv 主干 + 8 层带滑窗注意力的层**（滑窗 72 帧 ≈ 5.76 s） | 论文原文 + HF `config.json` |
| 16 层码本 | ✅ 对。16 层 RVQ，每层 2048（**第 0 层语义码本 4096**），12.5 Hz，24 kHz | 同上 |
| 近透明重建 | ✅ 对。LibriSpeech test-clean：PESQ-WB **3.21** / STOI **0.96** / UTMOS **4.16** / SIM **0.95** | 论文 Table 4 |
| 需要百万小时级音频 | ⚠️ 低估了。**500 万+ 小时**，10 语种 | 论文 Abstract / S1 |
| 把 SSL 语义蒸馏进第一层码本 | ✅ 对。**WavLM 作 teacher** 引导第一层语义码本 | 论文 §2.2 |
| 这正是 Qwen3-Omni 做 "AuT pretraining" 的原因 | ❌ **这里混了**。AuT 是**理解侧**的音频编码器（650M，2000 万小时监督音频，12.5 Hz，attention encoder-decoder），用来替换 Whisper，不是 codec。codec 侧叫 Code2Wav（200M ConvNet）+ MTP（80M） | Qwen3-Omni 报告 §2.2 / Table 1 |
| 社区没人训练这种 TTS | ⚠️ 不成立。**Mimi 就是同规格开源实现** | 见下 |

**关键发现**——把两份 config 并排看：

```
Mimi (kyutai/mimi)                    Qwen3-TTS-Tokenizer-12Hz (encoder_config)
frame_rate: 12.5                      _frame_rate: 12.5
codebook_size: 2048                   codebook_size: 2048
codebook_dim: 256                     codebook_dim: 256
num_semantic_quantizers: 1            num_semantic_quantizers: 1
compress: 2, upsampling_ratios:[8,6,5,4]   compress: 2, upsampling_ratios:[8,6,5,4]
use_causal_conv: true                 use_causal_conv: true
hidden_size: 512, layers: 8, heads: 8 hidden_size: 512, layers: 8, heads: 8
kernel_size: 7, dilation_growth_rate: 2    ……逐字段一致
```

**完全一致**，连 `initializer_range: 0.02`、`layer_scale_initial_scale: 0.01`、`trim_right_ratio: 1` 都一样。Qwen 改的是：量化器 **8 层 → 16 层**，decoder 换成 `latent_dim=1024 / decoder_dim=1536 / 16 heads / sliding_window=72`，参数量 96M → **170.6M**，并且把 WavLM 蒸馏写进了论文。

这直接改变了答案：这不是"没人做过的圣杯"，而是**"Mimi 的架构 + 2 倍比特率 + 大规模 GAN 调参 + 500 万小时 LM"**。

---

## 1. Qwen 实际用的配方

<details>
<summary><b>展开：可直接照抄的阶段化 recipe（含损失项与超参建议）</b></summary>

### 阶段 0：先算比特率预算，别急着训

$$R = f_s\left(\log_2 C_0 + \sum_{k=1}^{K-1}\log_2 C_k\right) = 12.5 \times (12 + 15 \times 11) = 2212.5\ \text{bps}$$

对照 Mimi：$12.5 \times 8 \times 11 = 1100\ \text{bps}$（与官方 card 的 1.1 kbps 吻合）。

**这就是"为什么能用 lightweight ConvNet 解码"的全部秘密**：每帧 177 bit 的容量（vs 单码本 15 bit/帧）已经足够，decoder 不需要 diffusion 去补细节。**近透明重建是拿 2 倍比特率换来的，不是架构更聪明。** 所以第一步先跑 1/2/4/8/16 层的 RD 曲线，确认 16 层真的是最优性价比点。

### 阶段 1：语义码本（Layer 0）——先做，别和声学一起做

这是 LM 学得动与否的分水岭，也是 Mimi 和 Qwen 共同的做法。

- 冻结 **WavLM-Large** 作 teacher，取其 6/9/12 层加权特征作回归目标
- Layer 0 输出（Qwen：`semantic_codebook_size=4096`, `codebook_dim=512`）用 cosine + L1 对齐 teacher 特征
- 建议先单独训 layer 0 到收敛（此时只有重建 + 蒸馏两个损失），**再解冻全部 RVQ**
- 验收指标：对该层 token 做 ASR（S3-tokenizer 式的探针），WER 应该接近学得动的水平。Qwen 的 25Hz 版本层 1 达 C.V. EN 7.51，层 2 退化到 10.40，他们明确说这是"用语义辨识度换声学丰富度"的自觉取舍

### 阶段 2：声学 RVQ + 端到端 GAN

架构照 Mimi/Qwen 抄即可（SEANet 因果编码器：`num_filters=64, kernel=7, dilation_growth_rate=2, strides=[8,6,5,4]×compress 2` → 1920× 下采样 → 12.5 Hz）。

损失组合（这是真正决定成败的地方）：

$$\mathcal{L}_{\text{mel}} = \sum_{s\in\mathcal{S}} \left\| \mathrm{Mel}_s(x) - \mathrm{Mel}_s(\hat{x}) \right\|_1,\quad \mathcal{S}=\{2048,1024,512,256,128,64\}$$

$$\mathcal{L}_{\text{VQ}} = \left\|\mathrm{sg}[z]-e\right\|_2^2 + \beta\left\|z-\mathrm{sg}[e]\right\|_2^2$$

- **多尺度 mel**：窗长 2048→64，hop = 窗长/4。Qwen 明文写"multi-scale mel-spectrogram reconstruction loss enforces time–frequency consistency"
- **多尺度 STFT 判别器 + 多周期判别器**，hinge GAN，$\lambda_{\text{adv}}\approx 1$，feature matching 权重 2~10
- **commitment** $\beta \in [0.25, 1]$，DAC 风格 online k-means 码本更新（比 EMA 稳）
- **speaker SIM 损失**：在 $\hat x$ 和 $x$ 上用 WavLM-based speaker verifier 算 cosine。Qwen 报 SIM 0.95，这一项几乎必不可少
- **训练调度**：先无 GAN 跑 100k~200k step 学频谱，再开 GAN；判别器 lr 单独 warmup；**上 adaptive loss balancer（DAC 那套）**——这是 GAN codec 唯一能收敛的原因，手调权重必崩

### 阶段 3：RVQ codebook collapse

Qwen 没公开细节，但低码率 codec 的公认手段：

- **低维码本**（$\text{dim}=256$）：BigCodec 明确指出低维 VQ 是 1 kbps 下高利用率的关键
- **code balancing / online clustering**：ERVQ 的 intra-codebook（在线聚类 + 平衡损失）+ inter-codebook（惩罚相邻层量化特征相似度），报告可做到 **100% 利用率**
- **随机重启**：未使用码字用当前 batch 的 encoder 输出替换（VQ-VAE-2 / SoundStream 老招，必留）
- 监控：**每层 perplexity 与 usage entropy**，任何一层利用率 < 10% 就说明配方坏了

### 阶段 4：因果化与流式

- encoder/decoder 全 causal（`use_causal_conv: true`），不做 look-ahead
- Qwen 选 **4 token = 320 ms 一个 packet**（每 token 80 ms），decode TPP **4 ms**；这样避免小包调度开销
- **必做的回归测试**：逐帧 encode/decode 与整段 batch 推理输出数值对比，误差应 < 1e-4。训练时用整段 STFT 损失，推理时用因果卷积，**这是 latent 的 train/inference mismatch 高发区**

### 阶段 5：才轮到 LM

Qwen3-TTS 的 LM 是 3 阶段预训练 + 3 阶段后训练：

- S1 通用（5M 小时）→ S2 高质量 CPT（治 S1 噪声导致的幻觉）→ S3 长上下文（8,192 → 32,768）
- 后训练：DPO（人工偏好对）→ GSPO（规则奖励）→ speaker 微调

注意 Qwen 自己的一句话：*"successfully maximizes perceptual quality without overfitting to ASR-related metrics"*——**PESQ 高 ≠ LM 好训**，这是下面第 2 节的关键。

</details>

---

## 2. 为什么社区（几乎）不训这套

先把"没人训"这个前提拆开：

**训 codec 的人很多**——Mimi（96M，12.5 Hz，1.1 kbps，CC-BY，权重+代码全开）、BigCodec（159M，1.04 kbps）、ZipCodec（2026，6.25 Hz / 0.80 kbps / 842M，大规模 WavLM 蒸馏 + 流式 decoder）、SpeechTokenizer / XCodec2 / XY-Tokenizer / ERVQ / ESC，以及 Fish-Speech、CosyVoice3、Higgs Audio v2、Step-Audio 各家自研 tokenizer。

**罕见的是"训出一个能撑起 1.7B LM 做 SOTA zero-shot TTS 的 codec"**。真实瓶颈有四条：

**① 前置依赖强，且反馈链路太长。**
codec 好不好，**要训练完 LM 才知道**。学术 codec 论文优化 PESQ/STOI/UTMOS，但 Qwen 明说这些指标和 LM 可学性是两回事。你缺一个"codec 质量 → LM 收敛速度"的代理指标，所以没法做有效的 ablation 循环。

**② GAN codec 是"全或无"的训练。**
loss balancer、判别器 lr、warmup 时序，任何一项错了就是崩。而且**训练到一半的 codec 完全不可用**——没有"70% 完成度的中间收益"。一次失败 = 几万 GPU 小时。这跟 LM 训练可以随时看 loss 曲线完全不同。

**③ 算力差 1000 倍。**
Qwen3-TTS **5,000,000 小时**；GPT-SoVITS v2 **5,000 小时**（README 明确写 2k→5k）。Mimi 是个 7B 全双工对话模型的副产品，Kyutai 专门发论文分析过 Moshi 的算力/碳排放——**他们要算的是失败实验和 ablation 的 GPU 时间**。

**④ 能力栈错配。**
做 codec 的人通常没有大 LM；有 LM 的人不想碰 GAN codec。Qwen 能做是因为它同时持有 Qwen3 LM、Qwen2-Audio、WavLM 级 teacher，以及"从 LM 侧回流改 codec"的迭代闭环。

---

## 3. GPT-SoVITS 为什么好训（这是本题最实用的一问）

一句话直觉：**GPT-SoVITS 把语音拆成了三个成熟件 + 一个很窄的映射；Qwen 的路线是把这三块全部压进一个 2.2 kbps 离散码本，让 LM 自己去学。**

$$ \underbrace{\text{冻结 SSL}}_{\text{内容}} + \underbrace{\text{参考音频}}_{\text{音色}} + \underbrace{\text{BigVGAN}}_{\text{波形}} \quad\Longrightarrow\quad \text{只学 } \text{text}\rightarrow\text{语义token} $$

对比 Qwen：

$$
\text{text} \longrightarrow \underbrace{\text{16 层} \times \text{12.5 Hz 离散码本}}_{2.2\ \text{kbps，内容+音色+韵律+环境全在里面}} \longrightarrow \text{ConvNet}
$$

具体拆成 7 条：

| # | GPT-SoVITS | Qwen3-TTS-12Hz |
|---|---|---|
| 1 | **完全不训 tokenizer**，直接复用 ContentVec/CNHubert（别人花千万小时训好的 SSL） | tokenizer 本体 170.6M 参数 GAN，必须自己训 |
| 2 | 重建走**连续** SSL 特征 → flow → BigVGAN，没有离散瓶颈 | 16 层离散 token 是**唯一**信息通道，必须近透明 |
| 3 | **音色由参考音频给**，LM 只建模语义+韵律 | 音色**必须在码本里**（否则无法 zero-shot 克隆） |
| 4 | 单层 1024/4096 语义 token，冗余极高、近乎单调对齐 | 16 层 × 12.5 Hz，层间强依赖，第 0 层外全是残差细节 |
| 5 | s1 / s2 **解耦可独立调试**，一个崩不拖累另一个 | tokenizer→LM→Code2Wav 串联，tokenizer 分布直接决定 LM 能否收敛 |
| 6 | 预训练 5,000 小时 | 5,000,000 小时 |
| 7 | 产品目标是"1 分钟数据微调"，**容易训就是设计目标** | 目标是 zero-shot SOTA，tokenizer 是要被复用的基础设施 |

所以本质上是**两种不同的工程哲学**：GPT-SoVITS 把难度留在推理侧的级联里（换来好训），Qwen 把难度提前内化进 tokenizer（换来 97 ms 首包和 single-model 可控性）。**"自研 codec 比训 LM 更难"这句话是对的，但难的不是数学，是它没有中间反馈、失败成本不可分摊。**

---

## 4. 给你的路线建议

<details>
<summary><b>展开：按目标选路线（含"不要做什么"）</b></summary>

**路线 A（强烈推荐）：别自研 tokenizer。**
`Qwen/Qwen3-TTS-Tokenizer-12Hz` 是 **Apache 2.0** 开源的，170.6M，`transformers` 原生支持，下载即用。自研 tokenizer 的边际收益几乎为零——你很难做出比"5M 小时 + WavLM 蒸馏 + 大规模 GAN 调参"更好的东西，而这正是你上一段自己说的"比训 LM 更难"。**把创新放在 LM 侧。**

**路线 B：真要做 codec，也从 Mimi 起步。**
`kyutai/mimi` 是 **CC-BY**，config 与 Qwen encoder 逐字段一致，且 Moshi 仓库的 codec 训练代码是开的。你的工作 = 量化器 8→16 层 + 补 WavLM 蒸馏损失 + 换 decoder + 重调 loss balancer。这比从零开始省掉 80% 的工作量，而且**有一个已知能收敛的起点**。

**路线 C：如果目标只是"能用的 TTS + 少量数据"，别碰 codec。**
GPT-SoVITS / F5-TTS / CosyVoice3 微调就是最优解，1 分钟到几小时数据、单卡、小时级迭代。

**必须建立的三个监控指标（无论走哪条路）：**
1. **每层码本 usage entropy / perplexity**（collapse 预警，< 10% 即坏）
2. **小 LM 探针**：固定算力训一个 tiny LM，测 NLL 收敛速度——这是唯一能代理"LM 学不学得动"的指标
3. **因果一致性测试**：逐帧 vs 整段推理的数值误差（应 < 1e-4）

</details>

---

**一句话总结**：Qwen3-TTS 12Hz 不是"没人敢做的新物种"，而是 **Mimi 架构 + 16 层量化 + WavLM 蒸馏 + 2.2 kbps 比特率 + 500 万小时**的工程化结果；社区不是训不了 codec，而是**训不出"撑得起大 LM"的 codec**——因为它缺乏中间反馈、失败不可分摊、且需要同时持有 LM 和大规模数据。GPT-SoVITS 好训，是因为它用"级的联"换掉了这个最难环节。

**参考来源（均为本次实际抓取的一手材料）**：Qwen3-TTS Technical Report（arXiv 2601.15621，§2.2 / Table 4）、Qwen3-Omni Technical Report（arXiv 2509.17765，§2.2 / §2.4 / §2.5）、`Qwen/Qwen3-TTS-Tokenizer-12Hz` 与 `kyutai/mimi` 的 `config.json` 与 model card、Moshi 论文（arXiv 2410.00037）、GPT-SoVITS README、ERVQ（arXiv 2410.12359）、BigCodec（arXiv 2409.05377）、ZipCodec（arXiv 2609.11642）。

> 说明：Qwen 未公布 tokenizer 自身的训练小时数与损失权重，阶段 2/3 中的具体权重和调度是我基于 EnCodec/DAC/Mimi/SoundStream 惯例给的工程建议，不是论文原文，请以小规模 ablation 为准。
