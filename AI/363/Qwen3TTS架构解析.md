# Qwen3TTS架构解析

### 🧑‍💻 User

qwen3 tts vocoder实现是怎么样的？

---

### 🤖 Assistant

# Qwen3-TTS 的 Vocoder / Detokenizer 实现

Qwen3-TTS **有两套不同的波形重建（vocoder/detokenizer）方案**，对应两个 tokenizer。目前正式开源主推的是 **12Hz** 版本。

---

## 总体结论

| 方案 | Tokenizer | Vocoder / Detokenizer | 特点 |
|------|-----------|------------------------|------|
| **25Hz** | 单 codebook，25 Hz | **DiT (Flow Matching) → Mel → BigVGAN** | 语义强，但延迟更高 |
| **12Hz**（当前主推） | 16 层 multi-codebook，12.5 Hz | **轻量因果 ConvNet（非 DiT）** | 超低延迟，首包约 97ms |

README 里说的：

> *high-speed, high-fidelity speech reconstruction through a **lightweight non-DiT architecture***

主要就是指 **12Hz tokenizer 的 decoder**。

---

## 1. Qwen-TTS-Tokenizer-12Hz（当前主力）

### 设计思路
- 受 **Mimi / SpeechTokenizer** 启发
- **语义 + 声学解耦量化**
  - 第 1 层 codebook：语义
  - 后续 15 层 RVQ：声学细节、韵律等
- 共 **16 层 codebook**，每层 size **2048**
- 帧率：**12.5 Hz**（每帧约 80ms 音频）

### Vocoder / Decoder 实现
- **不是 DiT**
- 使用 **fully causal ConvNet decoder**
- 编码器/解码器都是 **因果（causal）**，无 look-ahead
- 训练：
  - GAN（generator 直接处理 waveform）
  - multi-scale mel reconstruction loss
  - 语义侧用 **WavLM** 做 teacher

### 为什么快
- 有了多 codebook 的高容量表示后，**不需要复杂扩散模型**就能重建高质量波形
- 也不需要额外 speaker vector 抽取
- 支持 **token 一到就能立刻 decode**
- 实测：
  - 首包延迟：**97ms（0.6B）/ 101ms（1.7B）**
  - 单包 decode 时间约 **4–5ms**

这就是官方说的 **lightweight non-DiT architecture**。

---

## 2. Qwen-TTS-Tokenizer-25Hz（另一套）

### 设计思路
- 基于 **Qwen2-Audio** 两阶段训练
- **单 codebook**，25 Hz，codebook size 较大（32768）
- 更偏语义 token，方便和 audio LLM 打通

### Vocoder / Streaming Detokenizer
这是更“传统”的两阶段重建：

1. **DiT + Flow Matching**
   - code → mel-spectrogram
   - sliding-window block attention
   - 每个 token 只看有限上下文：当前块 + 3 个 lookback + 1 个 lookahead

2. **BigVGAN**
   - mel → waveform
   - 也做 chunked streaming
   - 还有额外 right-context look-ahead（约 130ms）

### 流式代价
因为 DiT 需要 look-ahead：
- chunk size = 8
- 首包前至少要等 LM 生成 **16 个 token**
- 25Hz 下首包延迟明显高于 12Hz（约 138–150ms 起步，并发更高更慢）

所以 25Hz 更适合和语义系统结合，**不是超低延迟首选**。

---

## 3. 在整条 TTS 流水线中的位置

Qwen3-TTS 整体是：

```text
Text → Qwen3 LM (dual-track) → Speech Tokens → Code2Wav(Vocoder) → Waveform
```

### 12Hz 路径
```text
LM backbone 预测第 0 层 codebook
     ↓
MTP (Multi-Token Prediction) 预测剩余 residual codebooks
     ↓
12Hz causal ConvNet decoder 直接出 waveform
```

### 25Hz 路径
```text
LM 预测 single-level speech tokens
     ↓
chunk-wise DiT (Flow Matching) → mel
     ↓
BigVGAN → waveform
```

---

## 4. 为什么官方更推 12Hz

从技术报告看：

1. **重建质量更高**
   - PESQ_WB **3.21**
   - STOI **0.96**
   - UTMOS **4.16**
   - SIM **0.95**
   - 超过 Mimi / SpeechTokenizer / X-codec 等

2. **延迟更低**
   - pure left-context，无需 future context
   - 首包就能发

3. **架构更统一、更轻**
   - 多 codebook 足够表达声学细节
   - 不需要 DiT + BigVGAN 这种重链路

4. **工程友好**
   - decode 轻、可 batch
   - 高并发下 RTF 更稳

---

## 5. 代码位置（仓库结构）

官方仓库：

- `qwen_tts/core/tokenizer_12hz/`
  - `modeling_qwen3_tts_tokenizer_v2.py`
  - `configuration_qwen3_tts_tokenizer_v2.py`
- `qwen_tts/core/tokenizer_25hz/`
  - `modeling_qwen3_tts_tokenizer_v1.py`
  - `vq/`
  - 以及 DiT / BigVGAN 相关实现

如果你关心的是“**真正把 token 变成波形的那部分**”：

- **12Hz**：看 `tokenizer_12hz` 的 decoder / Code2Wav（因果卷积）
- **25Hz**：看 DiT + BigVGAN 的 streaming detokenizer

---

## 一句话总结

**Qwen3-TTS 的 vocoder 不是单一方案：**

- **25Hz** = `DiT(Flow Matching) + BigVGAN`
- **12Hz（当前主推）** = **轻量因果卷积 decoder（non-DiT）**，靠 16 层 multi-codebook 表达能力，实现超低延迟、高保真重建

如果你想，我可以继续帮你拆：
1. **12Hz decoder 的网络结构细节**（从源码角度）
2. **和 CosyVoice / Mimi / BigVGAN 的对比**
3. **如何自己替换/加速这个 vocoder**

---

### 🧑‍💻 User

llm在其中发挥了什么作用？画出整个模型架构图

---

### 🤖 Assistant

# Qwen3-TTS 中 LLM 的作用 & 整体架构图

Qwen3-TTS 的核心不是“传统声学模型”，而是：

> **把 TTS 变成一个以 Qwen3 为 backbone 的离散多码本语言模型问题**

也就是：

```text
Text / Instruction / Speaker Prompt
        ↓
     Qwen3 LLM  (dual-track AR)
        ↓
   Speech Tokens (12Hz 多码本 / 25Hz 单码本)
        ↓
     Code2Wav
        ↓
     Waveform
```

---

## 一、LLM 在其中到底做什么？

Qwen3-TTS 的 LLM 不是“辅助模块”，而是**主生成器**。它负责：

| 作用 | 说明 |
|------|------|
| **1. 文本理解** | 继承 Qwen3 的语义理解能力，处理复杂文本、标点、口语、多语言 |
| **2. 文本 → 语音 token 映射** | 直接自回归预测离散 speech tokens，而不是 mel |
| **3. 指令控制** | 根据自然语言指令控制音色、情绪、语速、韵律 |
| **4. 音色克隆 / 设计** | 结合 speaker embedding 或 ICL 参考音频，实现 3 秒克隆 / 文生音色 |
| **5. 流式双轨生成** | Dual-Track：文本 token 一到就立刻预测对应 speech tokens |
| **6. 多码本分层预测（12Hz）** | Backbone 预测第 0 层语义码本，MTP 预测剩余 15 层声学码本 |
| **7. 长上下文稳定性** | 训练到 32k，支持 10 分钟以上连贯长语音 |

一句话：

> **LLM = 大脑 + 导演**
> Tokenizer/Vocoder = 耳朵/嘴
> LLM 决定“说什么、怎么说、用什么声音说”，Code2Wav 只负责“把 token 变成波形”。

---

## 二、整体模型架构图（总览）

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                            Qwen3-TTS 总架构                               │
└──────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
  │   Text Input │   │  Instruction   │   │ Reference Audio    │
  │  (待合成文本) │   │ (风格/情绪指令) │   │ (可选：克隆/ICL)    │
  └──────┬───────┘   └───────┬────────┘   └─────────┬──────────┘
         │                   │                      │
         ▼                   ▼                      ▼
  ┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
  │ Qwen Tokenizer│   │ ChatML Format  │   │ Speaker Encoder    │
  │ (文本分词)    │   │ 指令拼接       │   │ (可学习 speaker emb)│
  └──────┬───────┘   └───────┬────────┘   └─────────┬──────────┘
         │                   │                      │
         └───────────┬───────┴──────────┬───────────┘
                     ▼                  ▼
           ┌─────────────────────────────────────┐
           │         Dual-Track Input            │
           │  Text tokens  ⊕  Speech tokens      │
           │   (沿 channel 拼接，流式双轨)         │
           └─────────────────┬───────────────────┘
                             ▼
           ┌─────────────────────────────────────┐
           │           Qwen3 LLM Backbone        │
           │      (0.6B / 1.7B Transformer)      │
           │                                     │
           │  功能：                               │
           │  - 理解文本语义                       │
           │  - 遵循自然语言指令                   │
           │  - 融合 speaker / prompt 条件         │
           │  - 自回归预测 speech tokens           │
           └─────────────────┬───────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
   ┌─────────────────────┐       ┌─────────────────────┐
   │  Qwen3-TTS-12Hz     │       │  Qwen3-TTS-25Hz     │
   │  (当前主推)          │       │  (语义更强/可对接)   │
   └──────────┬──────────┘       └──────────┬──────────┘
              │                             │
              ▼                             ▼
   Backbone → 第0层 code           Backbone → single speech token
              │                             │
              ▼                             ▼
        MTP 模块预测                 Chunk-wise DiT
      剩余 15 层 residual codes      (Flow Matching → Mel)
              │                             │
              ▼                             ▼
     16 层 multi-codebook            BigVGAN (mel→wav)
              │                             │
              ▼                             ▼
     Causal ConvNet Decoder          Streaming Detokenizer
              │                             │
              └──────────────┬──────────────┘
                             ▼
                      ┌─────────────┐
                      │  Waveform   │
                      │  音频输出    │
                      └─────────────┘
```

---

## 三、更精确的 12Hz 架构（官方主推）

这是当前开源主力方案。

```text
                    ┌─────────────────────────────────────────┐
                    │              Conditioning               │
                    │  Text + Instruct + Speaker Emb / ICL    │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Dual-Track AR Generation                         │
│                                                                          │
│   t1  t2  t3  t4  t5 ...          ← 文本流（可流式输入）                   │
│   │   │   │   │   │                                                      │
│   ▼   ▼   ▼   ▼   ▼                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     Qwen3 Transformer Backbone                     │  │
│  │              输入：text feat + 已生成 speech codes 聚合              │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│                    Predict Codebook-0 (语义层)                            │
│                                  │                                       │
│                                  ▼                                       │
│                    ┌──────────────────────────┐                          │
│                    │   MTP (Multi-Token Pred) │                          │
│                    │  预测 residual codebooks │                          │
│                    │   CB1, CB2, ..., CB15    │                          │
│                    └─────────────┬────────────┘                          │
│                                  │                                       │
│   s1  s2  s3  s4 ...  ← 每帧 16 codes（12.5Hz，约 80ms/帧）               │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   Code2Wav (non-DiT)     │
                    │  Fully Causal ConvNet    │
                    │  无 look-ahead，可立即解码 │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                             Waveform (首包 ~97ms)
```

### 12Hz 关键点
1. **LLM 只预测离散 codes**，不直接出 mel/wav
2. **分层预测**：
   - Backbone → 语义码本（第 0 层）
   - MTP → 声学细节（后 15 层 RVQ）
3. **Dual-Track**：
   - 文本 token 一到，立刻开始出 speech tokens
   - 支持“边读边说”的真正流式
4. **Code2Wav 很轻**：
   - 因为 16 层码本表达能力够强
   - 不需要 DiT，也不需要额外 speaker vector 抽取

---

## 四、25Hz 架构（另一条线）

```text
Text / Instruct / Speaker
            │
            ▼
     Qwen3 LM Backbone
            │
            ▼
  预测 single-level speech tokens (25Hz)
            │
            ▼
   Sliding-window DiT (Flow Matching)
   - chunk size = 8
   - 当前块 + 3 lookback + 1 lookahead
            │
            ▼
          Mel spectrogram
            │
            ▼
         BigVGAN
            │
            ▼
          Waveform
```

### 和 12Hz 的核心差异

| 项目 | 12Hz | 25Hz |
|------|------|------|
| Token | 16 层 multi-codebook | 单 codebook |
| LLM 输出 | CB0 + MTP residual | 单层 speech token |
| 波形重建 | 轻量因果 ConvNet | DiT + BigVGAN |
| 首包延迟 | ~97–101ms | 更高（需 look-ahead） |
| 特点 | 超低延迟、高保真 | 语义更强，更易对接 audio LLM |

---

## 五、LLM 在不同任务中的角色

### 1) Custom Voice / 预置音色
```text
[Speaker ID] + [可选 instruct] + [text]
        ↓
   Qwen3 LLM 条件生成 speech tokens
        ↓
     Code2Wav
```
LLM 负责：在固定音色下，按指令调整情绪/语速/风格。

### 2) Voice Design（文生音色）
```text
[自然语言音色描述] + [text]
        ↓
   Qwen3 LLM（强文本理解 + thinking pattern）
        ↓
  直接生成匹配描述的 speech tokens
```
LLM 负责：把“萝莉音、低沉男声、紧张、撒娇”等描述映射成声学 token 序列。

### 3) Voice Clone（3 秒克隆）
两种条件注入方式：

**A. Speaker Embedding 模式**
```text
ref_audio → Speaker Encoder → speaker emb
                +
             target text
                ↓
            Qwen3 LLM
```

**B. In-Context Learning 模式**
```text
(ref_text, ref_audio_tokens) 作为 prompt
                +
           target text
                ↓
            Qwen3 LLM 模仿参考段的音色/韵律
```
ICL 通常对韵律保留更好；x-vector only 更快但质量可能略降。

---

## 六、为什么是“LLM 做 TTS”，而不是“TTS 挂一个 LLM”？

传统 TTS：
```text
文本分析 → 声学模型 → 声码器
```
各模块串行，信息瓶颈大，错误会级联。

Qwen3-TTS：
```text
统一成一个 AR LM：
Text/Instruction/Speaker → Speech Codes → Wav
```

论文明确强调：

> **discrete multi-codebook LM architecture**
> 实现 full-information end-to-end speech modeling
> 绕开传统 **LM + DiT** 的信息瓶颈与级联误差

所以 LLM 在这里不是“理解文本后交给 TTS”，而是：

- 直接建模语音的离散序列
- 同时承担语义、风格、音色、流式对齐

---

## 七、训练视角：LLM 学到了什么

| 阶段 | 目标 |
|------|------|
| **S1 General** | 500 万小时多语言，建立 text→speech 单调映射 |
| **S2 High-Quality** | 高质量数据 CPT，降幻觉、提自然度 |
| **S3 Long-Context** | 上下文扩到 32k，稳住长语音 |
| **Post: DPO** | 对齐人类偏好（自然度、表达力） |
| **Post: GSPO/规则奖励** | 提升任务稳定性 |
| **Speaker SFT** | 固定/定制音色，进一步提升可控性 |

LLM 最终学到的是：

1. 文本语义 → 语音内容
2. 指令语义 → 风格/情绪/韵律
3. speaker 条件 → 音色身份
4. dual-track 对齐 → 低延迟流式输出

---

## 八、一句话总结

### LLM 的作用
**Qwen3 LLM 是 Qwen3-TTS 的核心生成引擎。**
它负责：
- 理解文本与指令
- 融合说话人条件
- 自回归生成离散 speech tokens
- 支持流式双轨、音色克隆、音色设计、风格控制

### Vocoder / Tokenizer 的作用
只是把 LLM 预测出的 codes **高效、低延迟地还原成波形**。

---

## 九、最简心智模型

```text
Qwen3-TTS ≈

  Qwen3 (语言模型能力)
      +
  Dual-Track AR (文本/语音同步流式)
      +
  Multi-codebook Speech Tokens (12Hz) 或 Semantic Tokens (25Hz)
      +
  Code2Wav (ConvNet 或 DiT+BigVGAN)
```

如果你愿意，我可以继续给你画两张更工程向的图：

1. **推理时序图**（首包 97ms 是怎么算出来的）
2. **12Hz MTP 模块内部结构图**（backbone 和 residual heads 怎么接）