# VITS与新范式的社区TTS发展

## 🧑‍💻 User

为什么社区的tts都离不开vits，fishaudios2pro这种为什么不需要也没人训练

---
## 😀 默认助手 (7)

I'll research this — the "FishAudio S2 Pro" line and the current architecture landscape need verification before I answer.

Let me pull primary sources directly.

Good — primary source secured. Now the technical reports and the community side.

## 先给结论

你的前提有一半是错的，而且错得很有意思：

**FishAudio 不是"不需要 VITS 的外来者"——它就是 VITS 社区的核心成员。** `github.com/fishaudio/Bert-VITS2` 就是他们家的仓库，fish-speech 的 README Credits 里还留着 `VITS2` / `Bert-VITS2` / `GPT VITS` / `MQTTS` / `GPT-SoVITS`——这是它的出身证明。

所以真实的故事不是"社区 A 用 VITS，公司 B 跳过了 VITS"，而是：**同一批人，把自己从 VITS 里升级出去了**。理解这条线，你的两个问题就都能回答。

---

## 一、VITS 赢在哪：它是"小数据 + 单卡"约束下的最优解

VITS（KAIST, 2021, arXiv:2106.06103）的结构可以一句话概括：**条件 VAE + 归一化流 + 单调对齐搜索（MAS）+ 随机时长预测器 + HiFi-GAN 声码器，端到端对抗训练**。

它当年打败 Tacotron2 / FastSpeech2 + 独立声码器这套两段式方案，靠的是三件事：

1. **单模型端到端**：文本直接出波形，不用单独训声码器。
2. **自带对齐归纳偏置**：MAS 自己搜索单调对齐，不需要外部强制对齐器（MFA）。对齐是内嵌在目标函数里的：

$$A^* = \arg\max_A \log p_\theta(z \mid c_{\text{text}}, A), \qquad \mathcal{L}_{kl} = \log q_\phi(z \mid x) - \log p_\theta(z \mid c_{\text{text}}, A)$$

3. **小数据友好**——这一条决定了社区的一切。

第 3 点是关键。**MAS 这个"文本长度 ↔ 音频长度"的手工先验，让模型不必从数据里学对齐**。结果是：

- 100 小时能训，10 小时能训，**5 分钟微调也能出可用音色**；
- 6–12GB 显存的消费级显卡就能跑微调；
- 模型只有 ~100M 参数，能导出 ONNX，能上树莓派和手机（Piper 就是 VITS）。

**低数据场景下，强归纳偏置是优势。** 这一点后面会反转。

---

## 二、社区"离不开"VITS：技术只占一半，另外一半是社会工程

### 技术侧
- 微调门槛低（几十分钟音频 + 单卡一天内出结果）
- 推理极便宜：GPT-SoVITS v2ProPlus 在 4060Ti 上 RTF 0.028，4090 上 0.014
- 时长预测器 → 显式控制语速；speaker embedding → 多音色
- 中文文本前端被社区做透了：`g2pW`、`pypinyin`、`CJK 分词`、`G2PWModel` 多音字消歧

### 社会学侧（这才是真正的原因）
- **工具链全是按 VITS 的形状搭的**：UVR5 分离 → audio-slicer 切片 → FunASR/faster-whisper 打标 → SubFix 校正 → `.list` 数据集格式（`路径|说话人|语言|文本`）→ 训练 → ONNX 导出。
- **预训练底模是社区真正依赖的资产**：Bert-VITS2 的底模被扩展从 2k 小时到 5k 小时再到更大规模。社区做的是*微调*，不是*从零训*。你换架构，底模就没了。
- **许可证**：GPT-SoVITS 是 MIT。而 Fish Audio S2 Pro 用的是 *Fish Audio Research License*，明确只允许非商业研究使用，商用要单独授权。**社区生态没法长在一个非商业许可的底座上。**
- 需求场景对口：二次元配音、游戏语音、AI 翻唱——需求就是"把某个角色音色复刻出来"，这恰好是 VITS 微调范式的主场。

> 一句话：**社区产出的单位不是"架构"，而是"微调流程 + 角色音色包"。**

---

## 三、Fish Audio S2 为什么不需要 VITS：范式变了

现代 LLM-TTS 把问题拆成三件套：

$$\underbrace{\text{文本}}_{\text{字符/字节}} \xrightarrow{\ \text{LLM}\ } \underbrace{\text{RVQ 音频 token}}_{\text{离散码本}} \xrightarrow{\ \text{codec decoder}\ } \text{波形}$$

S2 Pro 的具体配置（README + 技术报告 arXiv:2603.08823）：

- **Slow AR（4B）**：沿时间轴预测主语义码本
- **Fast AR（400M）**：每个时间步生成剩余 9 个残差码本
- RVQ 编解码器：10 个码本，~21 Hz 帧率
- 10M+ 小时数据，80+ 语言，**不需要音素、不需要语言特定预处理**（直接吃字符）
- GRPO 做 RL 对齐；SGLang 推理，H200 上 RTF 0.195、TTFA < 100ms

### 逐组件对照：VITS 的每一块都被替换掉了

| VITS 组件 | 为什么 VITS 需要它 | LLM 范式里由什么替代 |
|---|---|---|
| MAS 单调对齐搜索 | 自己学文本↔音频对齐 | cross-attention / 自回归，隐式学出来 |
| 随机时长预测器 | 非自回归，必须显式预测每音素时长 | 时长 = 自回归生成的 token 数，是副产物 |
| 归一化流（prior） | 把简单分布映射到复杂隐空间 | 不需要，直接建模离散 token 分布 $P(a_t \mid a_{<t}, \text{text})$ |
| HiFi-GAN 声码器 | 声学特征 → 波形 | RVQ codec decoder（FF-GAN） |
| G2P / 音素表 | 文本前端，逐语言写规则 | 字符直接进 LLM，多音字由上下文解决 |
| speaker embedding 查表 | 说话人身份烘焙进模型 | 10–30 秒参考音频 in-context 零样本克隆 |

**为什么现在能做到？** 因为归纳偏置从**架构**转移到了**数据 + 算力**：

- 10M 小时数据下，MAS 这种手工先验不再是优势，反而成了上限；
- 强对齐先验在低数据时才值钱，数据足够时，让模型自己学对齐更优；
- 目标也变了：从"为一个说话人训一个模型"→"用 prompt 抄音色 + 自然语言指令控制情绪"（`[whisper]`、`[excited]`… 支持 1.5 万+ 自由文本标签）。

VITS 的**根本瓶颈**正在这里：说话人身份是查表烘焙进去的，**克隆新音色必须微调**。LLM 范式靠 in-context learning，不需要微调。

---

## 四、"为什么没人训练"其实是三个不同的问题

<details>
<summary><b>Q1：为什么社区不自己训一个 S2 这种模型？</b></summary>

**训不动，而且不划算。** 门槛从"一张 4090 + 100 条语音"变成了：

- 4B 参数 + 10M 小时数据吞吐 → 几万到几十万 GPU 小时量级
- **数据没开源**：S2 开源了权重 + 微调代码 + 推理引擎，但训练数据没有。真正的护城河是数据管道（视频字幕、语音字幕、音质评估、reward model、去重切分），不是架构。论文里架构写得清清楚楚，你照着写也复现不出来。

**架构不是壁垒，数据和算力是。** 所以"S2 不用 VITS 且没人训"不是因为没人想，而是因为**开源了权重和微调代码 ≠ 开源了复现能力**。
</details>

<details>
<summary><b>Q2：社区为什么不搞非 VITS 的新架构？</b></summary>

其实**已经有很多非 VITS 的开源 TTS 了**：F5-TTS（flow matching + DiT）、CosyVoice 2（LLM + flow matching）、Kokoro（StyleTTS2 系，82M）、IndexTTS 等。

但它们的诞生模式是**"实验室/大厂首发 → 社区微调封装"**。社区原创新架构的 ROI 极低：

- 训底模要算力，做新架构要论文级工作量；
- 最后大概率打不过已有底模；
- 而"做 WebUI + 做音色包 + 写教程"立刻就有用户。

所以社区的理性选择是**不原创架构，只消费架构**。
</details>

<details>
<summary><b>Q3：那为什么大家还在用 VITS 而不转向新范式？</b></summary>

因为真实需求没变，而 VITS 在三个点上仍然最省心：小数据、消费级显卡、成熟中文文本前端。

但**注意一个反直觉的事实**：GPT-SoVITS 是 **"LLM 的前半段 + VITS 的后半段"**——它把"文本 → 语义 token"换成了 AR GPT，但 SoVITS 解码器仍然是 VITS 血统。

> **社区的演进方向是把 VITS 越缩越后，最后只剩解码器那一步。**
> FishAudio 只是**把最后一步也换掉了**（RVQ codec + FF-GAN 替代 SoVITS），于是彻底不需要 VITS 了。

另外别忘了部署现实：VITS 家族可以在 CPU 上实时跑；S2 Pro 是 4.4B 参数，要 H200 才能到 RTF 0.195。本地/边缘/实时场景，VITS 领先不止一个数量级。
</details>

---

## 五、总结

- **VITS 不是"最好"的架构，而是"约束下的最优解"**：小数据、单卡、中文前端、ONNX/CPU 部署。它在低数据 regime 下用 MAS 和时长预测器换来了别的架构给不了的东西。
- **社区的路径依赖是理性的**：换架构的收益（零样本克隆）大多已被大厂开源模型免费提供，而成本（丢底模、丢工具链、丢工作流、丢 MIT 许可）全落在自己头上。
- **S2 不需要 VITS，本质是把归纳偏置从架构搬到了数据和算力**：10M 小时学出来的对齐和韵律，就不需要 MAS 和时长预测器这类手工先验了。
- **"没人训练"≠ 没人想要**，而是门槛已经变成几千张卡 + 千万小时 + 一整套数据清洗流水线。而 FishAudio 自己就是从 Bert-VITS2 走出来的——**他们是跑出这个社区的人，不是绕过它的人。**
