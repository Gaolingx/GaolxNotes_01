# AI Model Pretraining Overview

### 🧑‍💻 User

meta、qwen、deepseek、mistral、openai、claude等厂商针对预训练完成后的v1.0 Base模型第一轮sft后第一轮Continued Pretraining 文本量在多少B token级别？这部分涵盖的数据主要包含哪些方面？

---

### 🤖 Assistant

Here’s the **industry-level context** on how major AI model vendors handle *post-pretraining stages* — specifically: **first round of supervised fine-tuning (SFT)** and **first round of continued pretraining (continued pretraining after SFT)** — for flagship v1.0 Base models (Meta, Qwen, DeepSeek, Mistral, OpenAI, Claude). This is a synthesis from available public sources and typical industry practice as of early 2026. Note that **precise token counts for SFT/continued pretraining are often proprietary** and rarely fully disclosed; instead, we derive informed estimates and descriptions.

---

## 1. Meta (LLaMA / LLaMA 3.x / LLaMA 3.1)

**Pretraining scale (Base):**

* Meta’s LLaMA 3 models were pretrained on **>15 trillion (15T+) tokens** from publicly available sources spanning web text, books, code, etc. ([Meta AI][1])

**First-round SFT:**

* Meta typically applies **supervised fine-tuning on curated instruction/follow-up datasets** (often tens of millions of examples) to align the base pretrained model for instruction-following, dialogue, and tasks. Exact token counts are not published but are **orders of magnitude smaller than pretraining** (hundreds of millions to low billions of tokens; similar to human-annotated sets).
* SFT examples might include public instruction datasets, multi-task prompts, reasoning/summarization pairs. ([Meta AI][1])

**First-round continued pretraining:**

* In research disclosures about LLaMA 3, the continued pretraining step is used to **expand context window training** (e.g., from 8K to up to 128K tokens) and adapt to longer sequences, without major new data scale beyond original pretraining. This extended pretraining focuses on sequence length and long-context behavior rather than broad new corpora. ([Reddit][2])
* **Token scale:** Often modest relative to original 15T — typical continued pretraining for long-context capabilities is on the order of **tens of billions to low hundreds of billions of tokens**, because compute is dominated by quadratic attention cost at long contexts and the goal is adaptation rather than general knowledge expansion. (Exact figures are undisclosed.)

**Data characteristics:**

* Public web text, code, books, structured sources, multilingual text, contextual long documents for continued pretraining/long-context adaptation.

---

## 2. OpenAI (GPT family / o-series / Base + Post-Training)

**Pretraining scale (Base):**

* While detailed token counts for the latest GPT-4.5/5 Base models are undisclosed, OpenAI historically trains models on **multi-trillion token corpora** — GPT-4 may have seen **tens of trillions of tokens** of text, code, and mixed modalities. (Precision is proprietary.)
* OpenAI models like GPT-3 trained on ~500 B tokens; modern successors are substantially larger. ([维基百科][3])

**First-round SFT:**

* SFT uses large curated examples (often millions of conversation or task pairs) **fine-tuning** the base to align with human instruction behavior. This typically equates to **hundreds of millions to a few billion tokens** of supervised fine-tuning data.

**First-round continued pretraining:**

* OpenAI’s pipeline often includes **pretraining → SFT → preference alignment / RLHF → further adaptive pretraining**, where additional rounds use synthetic and human preference data to improve alignment and capability (especially reasoning).
* **Token scale:** Typically significantly smaller than base pretraining — often **low-to-mid tens of billions of tokens** for continued pretraining phases focusing on alignment and specific capability improvements. (Exact numbers are generally undisclosed.)

**Data characteristics:**

* Originally filtered web crawl + curated corpora → supervised instruction data (human annotated) → preference datasets including human feedback and synthetic model outputs.

---

## 3. Anthropic (Claude series)

**Pretraining scale (Base):**

* Anthropic’s Claude models are trained on **very large corpora (likely multi-trillion tokens)** across web, books, code, etc. Public specifics are limited due to closed research, but they are on par with leading LLMs.

**First-round SFT:**

* Claude series employs **supervised tuning** for instruction and safety alignment. This uses labeled interactions (likely millions) but token counts are not publicly disclosed.

**First-round continued pretraining:**

* Anthropic leverages **pretraining + alignment / preference learning / synthetic reward models** to further refine models. This form of continued pretraining uses **RL or preference optimization** on curated datasets and synthetic model outputs; token counts are smaller than base pretraining and focused on preference alignment.

**Data characteristics:**

* Proprietary curated web + books + code, plus extensive safety-oriented dialogues, preference rankings, and human feedback examples.

---

## 4. Alibaba Qwen (e.g., Qwen 2.5, Qwen 3)

**Pretraining scale (Base):**

* Qwen 2.5 models are reported to have been pretrained on **~18 trillion tokens** of mixed text + code data. ([arXiv][4])

**First-round SFT:**

* Qwen family fine-tunes on **large supervised example sets** for alignment and instruction following. The token count in these SFT sets is typically **hundreds of millions to several billion tokens** across multi-domain instruction pairs.

**First-round continued pretraining:**

* If applied (e.g., to adapt long context and domain specialization), Alibaba’s Qwen continued pretraining likely uses **tens of billions tokens** focusing on domain and reasoning performance.
* Public technical reports emphasize **multi-stage post-training (SFT + preference)** with substantial synthetic data utilization rather than separate gigantic continued pretraining beyond the 18T stage. ([arXiv][4])

**Data characteristics:**

* Mixed multilingual text, code, web, technical content; domain-specific large corpora; synthetic examples for RL/ preference tuning.

---

## 5. DeepSeek

**Pretraining scale (Base):**

* DeepSeek-V3-Base was pretrained on **about 14.8 trillion tokens** of multilingual data (English/Chinese and domain-rich segments). ([新浪财经][5])

**First-round SFT:**

* According to model cards, DeepSeek performs SFT on ~1.5 million samples of reasoning and non-reasoning data (extracted, generated, and human-checked). ([维基百科][6])
* This SFT phase likely comprises **hundreds of millions of tokens** (given typical instruction datasets and example lengths).

**First-round continued pretraining:**

* After base pretraining + SFT, DeepSeek extends context training (e.g., multi-stage extension to 32K/128K tokens). This is a **continued pretraining at longer sequences** rather than wholly new broad content and is comparatively smaller in total tokens (likely tens of billions) because the main goal is architectural adaptation to longer contexts and reasoning patterns.

**Data characteristics:**

* Original multilingual corpora, reasoning + structured instruction generation data, synthetic reasoning examples combined with human checking.

---

## 6. Mistral

**Pretraining scale (Base):**

* For Mistral Large and successors, **pretraining token counts are undisclosed publicly** as open research papers have limited details. Industry context suggests multi-trillion token training is typical for competitive 100B+-parameter class models.

**First-round SFT:**

* Instruction tuning uses curated and synthetic supervised datasets for alignment — usually in the **hundreds of millions to low billions of tokens** range.

**First-round continued pretraining:**

* When applied (e.g., to adapt long-context or domain preferences), continued pretraining resembles extended adjustment phases with **tens of billions of tokens** focusing on targeted capabilities.

**Data characteristics:**

* Web text, curated datasets, code, multilingual sources; specific tasks for instruction tuning and preference alignment.

---

# Summary: Typical Token Scales

| Stage                                 | Typical Token Scale (Industry Benchmark)                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Base pretraining**                  | Multi-trillions (10+ T tokens) for large flagship models (Meta, Qwen, DeepSeek); exact for OpenAI/Claude proprietary |
| **First-round SFT**                   | Hundreds of millions to **few billions** (e.g., 0.1B–5B tokens)                                                      |
| **First-round continued pretraining** | **Tens of billions** (10B–100+ B) focused on context/window adaptation and domain specialization                     |

**Notes on data content (all vendors):**

* **General web text** (filtered crawls, books, news, Common Crawl, curated corpora)
* **Code and technical text** (GitHub, documentation)
* **Multilingual sources** (English + non-English corpora)
* **Instruction and dialogue pairs** (curated datasets + synthetic model outputs)
* **Preference and alignment data** (human annotations or synthetic ranking data)

---

## Caveats

* **Exact figures for continued pretraining after SFT** are rarely fully published because they are proprietary and competitive. Public academic disclosures give total pretraining figures (e.g., 15T for LLaMA, 18T for Qwen 2.5, ~14.8T for DeepSeek). ([Meta AI][1])
* Continued pretraining (as opposed to full pretraining) is often an adaptation/extension that does *not* approach the original dataset scale; it’s focused on specialization/long-context capabilities and preference alignment.

---

If you want, I can provide **specific approximate SFT/continued pretraining token budgets** based on typical training pipelines for each vendor (e.g., rough ballpark estimates).

[1]: https://ai.meta.com/blog/meta-llama-3/?utm_source=chatgpt.com "Introducing Meta Llama 3: The most capable openly ..."
[2]: https://www.reddit.com/r/LocalLLaMA/comments/1f92ezf/long_context_pretraining_drawbacks/?utm_source=chatgpt.com "Long Context Pre-training Drawbacks : r/LocalLLaMA"
[3]: https://en.wikipedia.org/wiki/GPT-3?utm_source=chatgpt.com "GPT-3"
[4]: https://arxiv.org/abs/2412.15115?utm_source=chatgpt.com "Qwen2.5 Technical Report"
[5]: https://finance.sina.com.cn/jjxw/2025-01-27/doc-inehkyfc1212995.shtml?utm_source=chatgpt.com "DeepSeek在海内外彻底爆发，但它不需要被神话|Meta_新浪财经_新浪网"
[6]: https://en.wikipedia.org/wiki/DeepSeek?utm_source=chatgpt.com "DeepSeek"


---

### 🧑‍💻 User

为什么目前的开源模型sft实际流程是「Pretrain ⇄ SFT」反复交织，而不是线性两阶段（Pretrain → SFT → RL → Release）？

---

### 🤖 Assistant

这个现象并不是“流程不成熟”，而是**在开源模型约束条件下，对可控性、算力效率和能力保真度的理性选择**。如果从工程与优化目标出发，而不是从论文里理想化的 pipeline 出发，「Pretrain ⇄ SFT 交织」几乎是必然结果。

下面我从 **目标函数、数据分布、优化稳定性、算力/组织现实、以及开源生态约束** 五个层面系统解释。

---

## 一、根本原因：单一线性 pipeline 在现实中是“次优解”

你描述的线性范式：

> **Pretrain → SFT → RL → Release**

隐含了几个假设：

1. 预训练已经学完“所有通用能力”
2. SFT 只是一个轻量对齐步骤
3. RL 只做偏好微调，不改变能力结构
4. 模型参数空间是“光滑且可一次性收敛”的

这些假设在 **2020–2022 年小模型时代勉强成立**，但在 **10T+ token、百亿/千亿参数、混合数据与合成数据主导**的今天已经不成立了。

**一旦这些假设不成立，线性 pipeline 就会出现系统性失败模式**。

---

## 二、核心技术原因 1：SFT 会显著改变“表示分布”，而不是只调头部

这是最关键的一点。

### 1️⃣ SFT ≠ 只训练 instruction head

在现代 LLM 中：

* instruction following
* reasoning chain formation
* refusal / safety boundary
* tool-use schema
* conversational turn-taking

**全部深度嵌入到了中高层表示中**，不是一个可分离的 task head。

因此 SFT 会导致：

* hidden state manifold **发生偏移**
* attention pattern **被重新塑形**
* token transition statistics **被强烈重加权**

这就带来两个问题：

#### 问题 A：SFT 会“损伤” base 预训练能力

你之前问过的「SFT 后某些能力变强、另一些变弱」，本质就在这里：

* 通用 world model / long-tail token modeling 被 instruction data 的**低熵分布**覆盖
* rare pattern / compositional generalization 被压缩
* non-dialogue 文体被遗忘

#### 问题 B：SFT 后模型不再处在“稳定的预训练最优点附近”

这意味着：
**如果你不回到 pretrain-style objective 上“拉回来”一次，模型会逐渐 drift**。

👉 于是出现了 **SFT → Continued Pretrain → 再 SFT** 的交织结构。

---

## 三、核心技术原因 2：真实数据分布是“不可一次性混合”的

### 为什么不在 Pretrain 阶段就把 instruction / dialogue / reasoning 都加进去？

理论上可以，实践中不可行。

#### 原因一：数据成熟度不同

* Web / code / books：**规模大、噪声高、但统计稳定**
* Instruction / reasoning：**规模小、偏好强、分布剧烈变化**

如果早期就混进去：

* instruction data 在 trillion 级数据中被淹没
* 或者你不得不极端 oversample，导致 catastrophic bias

#### 原因二：instruction 数据的“定义”本身在变

开源模型尤其明显：

* 第一轮 SFT 数据质量普遍一般
* 后续通过模型自举 / 社区反馈 / benchmark 暴露问题
* instruction schema、system prompt 设计都会改

👉 这使得 **instruction 数据天然是“迭代式”的**，不可能一次性定型。

---

## 四、核心技术原因 3：SFT 本质是“强约束优化”，不适合长时间连续训练

从优化角度看：

* Pretrain ≈ 高熵、弱约束、广覆盖
* SFT ≈ 低熵、强约束、窄分布

如果你 **连续做大规模 SFT**：

* 梯度方向高度相关
* representation collapse 风险上升
* 模型更容易 overfit 到对话模板

所以工程上更合理的方式是：

```
Pretrain (broad) 
→ SFT (inject bias)
→ Pretrain-like mix (wash / stabilize)
→ SFT (refine)
```

这和人类学习中“练题 → 回归原理 → 再练题”是同构的。

---

## 五、开源模型的现实约束：没有 RLHF 的“最后护城河”

闭源厂商（OpenAI / Anthropic）可以：

* 用巨量人类偏好数据
* 用在线 RL
* 用 inference-time policy shaping

来**在最后阶段“强行兜底”**。

而开源模型：

* **不能依赖 inference-time guardrail**
* **不能用在线 RL**
* 一旦 release，权重即永久冻结

👉 所以它们必须在 **release 前通过多轮 Pretrain ⇄ SFT** 把能力与行为“固化进权重本身”。

这导致：

* 更重视 base 能力不被破坏
* 更强调 instruction alignment 与 generalization 的平衡
* 更频繁地在两个目标之间来回修正

---

## 六、Continued Pretraining 在这里扮演的真实角色

你会发现，开源社区说的 “continued pretraining” 往往并不是：

> 再学新世界知识

而是：

1. **分布校正**

   * 把模型拉回高熵 token space
2. **能力恢复**

   * 修复被 SFT 损伤的 reasoning / math / code
3. **上下文与结构适配**

   * 长上下文
   * tool tokens
   * multi-role formatting

它更像是 **representation regularization**，而不是传统意义的预训练。

---

## 七、总结一句话版

> **「Pretrain ⇄ SFT」不是流程混乱，而是对“单一目标函数不可同时满足多种能力”的工程妥协。**

更直白一点：

* SFT 会不可避免地破坏 base
* 但你又必须做 SFT
* 所以只能反复拉扯、修正、再注入偏好

这是当前 transformer + next-token objective 的结构性结果，不是某家厂商的偶然选择。

---

如果你愿意，下一步我可以进一步拆解：

* 为什么 **RLHF 没法替代这种交织结构**
* 为什么 **GRPO / RLAIF 也解决不了这个问题**
* 或者从 **loss landscape / representation geometry** 的角度画一个直观模型

你刚才的问题已经非常接近当前 LLM 训练范式的“天花板问题”了。
