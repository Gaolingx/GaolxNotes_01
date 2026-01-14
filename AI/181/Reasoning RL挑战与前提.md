# Reasoning RL挑战与前提

### 🧑‍💻 User

为什么llm微调的sft是强信号，rl是弱信号？

---

### 🤖 Assistant

这是一个**训练信号信息密度与约束强度**的问题，而不是“算法先进性”的问题。用一句话概括：

> **SFT 给的是“你应该输出什么”的直接监督；RL 给的是“你做得好不好”的间接评价。**

下面从信息论、优化视角、以及 LLM 实际训练形态三个层面系统解释为什么 **SFT 是强信号，而 RL 是弱信号**。

---

## 一、从监督信息量看：token-level vs trajectory-level

### 1. SFT：逐 token 的高分辨率监督（dense signal）

在 SFT 中，训练目标通常是：

[
\mathcal{L}*{\text{SFT}} = - \sum_t \log p*\theta(y_t \mid x, y_{<t})
]

特点是：

* **每一个 token 都有明确的 target**
* 每一步梯度都指向“正确 token 的概率应该更大”
* 梯度信号：

  * 密集（dense）
  * 低噪声
  * 可直接反向传播到 embedding / attention / MLP

信息量角度看：

* 一条长度为 (T) 的样本，提供 (O(T)) 个监督信号
* 信号与模型参数的对应关系是明确的（credit assignment 几乎是“逐点”的）

---

### 2. RL：整段输出一个标量奖励（sparse signal）

典型 RLHF / RLVR 优化目标：

[
\nabla_\theta \mathbb{E}*{y \sim \pi*\theta} [R(y)]
\approx
\sum_t \nabla_\theta \log \pi_\theta(y_t) \cdot (R - b)
]

特点是：

* **一整段 response 只有一个 reward**
* 每个 token 的“好坏”是**事后推断**
* 信号：

  * 稀疏（sparse）
  * 高噪声（reward variance 大）
  * credit assignment 极其困难

信息量角度看：

* 长度为 (T) 的输出，只提供 **1 个标量**
* 要把这个标量“摊派”到所有 token

这是 RL 天然弱信号的第一性原因。

---

## 二、从优化约束看：SFT 在“参数空间里拉”，RL 在“策略空间里推”

### 1. SFT：强约束、强投影

SFT 本质是：

> 把模型参数 **直接投影到某个条件分布附近**

它在做的是：

* 明确规定：

  * 看到这种 prompt
  * **你就应该产生这种 token 分布**
* 优化是凸近似的（至少局部稳定）
* 不需要探索

这在高维参数空间中是**强收缩算子**。

---

### 2. RL：弱约束、依赖探索

RL 的约束形式是：

> 在“还能被 KL 约束住”的前提下，**稍微往 reward 高的方向挪一点**

现实中的 RLHF 几乎都是：

[
\max_\theta \mathbb{E}[R] \quad \text{s.t. } \text{KL}(\pi_\theta | \pi_{\text{ref}}) < \epsilon
]

结果是：

* 步子必须很小（否则语言能力直接崩）
* reward 只决定“方向”，不决定“形状”
* 无法告诉模型：

  * “这句话哪里错了”
  * “应该怎么改才更好”

所以它**只能做微调（shaping），而不是重塑（rewriting）**。

---

## 三、从能力形成角度：RL 无法注入“新知识结构”

这点与你之前提到的 **RLVR 无法激发新能力** 是同一个根因。

### 1. SFT 可以“写入行为模板”

因为 SFT 给的是完整输出序列，它可以：

* 教会模型新的：

  * 推理格式
  * 工具调用协议
  * 对话风格
* 本质是在**教语言结构本身**

这相当于在预训练分布上“加新 mode”。

---

### 2. RL 只能重排已有 mode 的权重

RL 的 reward 函数通常只能表达：

* 更安全 / 更有礼貌
* 更符合人类偏好
* 答案看起来更完整

但它**无法表达复杂结构**，例如：

* “第 3 步要先列假设”
* “先枚举再归纳”
* “遇到不确定性要反问”

因为 reward 是低带宽信道。

所以 RL 的真实作用是：

> **在已有能力空间里做 re-weighting，而不是扩展空间本身**

---

## 四、为什么工业界的训练流程是：Pretrain ⇄ SFT + 少量 RL

把以上几点合在一起，你会发现一个非常清晰的工程结论：

* **能力上限**：由 Pretrain + 高质量 SFT 决定
* **行为对齐与稳定性**：由 RL 负责
* **RL 永远是最后一公里，而不是地基**

这也是为什么：

* 大厂 SFT 数据极度珍贵
* RL 步数很少、lr 很小
* reward model 不敢太“聪明”

---

## 五、一句话总结（工程视角）

> **SFT 是高带宽、低噪声、强约束的“写入信号”；
> RL 是低带宽、高噪声、弱约束的“偏好修正信号”。**

因此在 LLM 微调中：

* SFT 决定“你会不会”
* RL 决定“你像不像人”

如果你愿意，下一步我可以结合 **MoE / 长推理模型 / GRPO**，具体分析为什么 RL 在 reasoning 上几乎只能“扶正姿态”，而无法“创造能力”。


---

### 🧑‍💻 User

我们都知道sft可以让模型学会预训练阶段不会的东西，但是rl很难。目前的sota普遍带有cot，但是pertrain+instruct sft阶段的模型显然是不具备think step by step能力的，可能导致后续的reasoning RL阶段不稳定，为了让instruct模型能有一个good start，业界主流的做法是什么？

---

### 🤖 Assistant

你这个问题已经切到**当前一线大模型训练里最“脏”、但也最关键的一层工程现实**了。结论先行：

> **业界并不是指望 RL 去“点燃”CoT / reasoning，而是想尽一切办法在进入 RL 之前，就把模型预热到“已经会慢慢想”的状态。**

换句话说：
**RL 从来不负责“教会思考”，它只负责“把已经存在的思考推到更稳定、更可控”。**

下面按真实工业流程拆开说。

---

## 一、核心矛盾：instruct SFT 的目标函数天然压制 CoT

你观察得非常准确：
**Pretrain + 普通 instruct SFT 的模型，确实不具备稳定的 step-by-step 推理能力。**

原因并不神秘：

### 1. Instruct SFT 的隐式偏置

典型 instruct SFT 数据是：

```
User: 问题
Assistant: 简短、直接、完成度高的回答
```

其损失函数会强烈奖励：

* 早停（short answer）
* 低 token 数
* 高表面相关性

这在梯度层面等价于：

> **把“中间思考 token”当成噪声压掉**

因此：

* 模型即便在 pretrain 中见过 CoT
* 也会在 instruct SFT 中被**系统性遗忘 / 抑制**

---

## 二、业界的共识：必须在 RL 之前“注入推理轨道”

目前几乎所有 SOTA 模型，都采用了 **“Reasoning warm-start”** 思路，只是叫法不同。

### 核心原则

> **进入 RL 之前，模型已经是一个“会慢慢想、能走多步、不怕 token 长度”的模型。**

否则 RL 会出现你说的：

* reward collapse
* policy oscillation
* 学会投机而非推理

---

## 三、主流做法一：显式 CoT SFT（最核心、最普遍）

### 1. 数据形态已经发生变化

SOTA instruct 数据早已不是：

```
Q → A
```

而是：

```
Q → [Reasoning / Analysis] → A
```

或者显式标注：

* `Let's think step by step`
* `Chain of Thought`
* `Explanation:`

### 2. 工业界的真实做法

* **不是把所有 instruct 都换成 CoT**
* 而是：

  * 普通对话：短回答
  * 复杂任务：**强制带推理轨道**

这相当于在模型内部**铺一条低阻抗路径**：

> “遇到复杂问题 → 可以合法地写很多 token”

---

## 四、主流做法二：Reasoning-only SFT / Continued Pretrain

这是近两年非常关键、但论文里写得很轻描淡写的一步。

### 1. 独立的 reasoning SFT 阶段

在正式 instruct SFT 之前或中间：

* 使用 **纯 reasoning 数据**

  * 数学
  * 代码推理
  * 逻辑证明
* 输出极长
* 不追求“像不像助手”，只追求“想得对”

这一步的目标是：

> **恢复甚至强化 pretrain 中的“长序列推理惯性”**

---

### 2. 为什么不是直接 RL？

因为：

* RL 无法告诉模型：

  * 哪一步推理错了
* 但 SFT 可以**逐 token 对齐中间状态**

---

## 五、主流做法三：格式控制 + 隐式 CoT（为安全与产品妥协）

你肯定也注意到了一个现象：

> SOTA 模型“会想”，但不一定“给你看它怎么想”。

这是因为：

### 1. 工程上区分两件事

* **internal reasoning token**（训练/推理时存在）
* **exposed CoT**（用户看到的）

### 2. 典型做法

* SFT 时：允许 / 强化显式 CoT
* RL 时：reward 正确性 + 简洁性
* 最终：

  * 内部仍走多步
  * 输出只给 concise answer / summary

这也是为什么很多模型：

* 不再公开完整 CoT
* 但 reasoning 能力反而更强

---

## 六、主流做法四：让 reward “不惩罚长思考”

这是 RL 阶段的一个**关键工程细节**。

### 常见坑

* reward model 偏好短答案
* 或隐含 length penalty

结果：

* 模型学会“赌答案”
* 推理直接塌掉

### 工业界的解决方案

* reward 只看：

  * 最终正确性
  * 结构完整性
* **明确移除对 token length 的负向信号**
* 有时甚至：

  * 对“合理长度”给正奖励

---

## 七、把真实流程串起来（工业级）

一个典型 SOTA reasoning 模型的路径更接近：

```
Pretrain
  ↓
Reasoning-heavy Continued Pretrain / SFT
  ↓
Instruct SFT（混合：短答 + CoT）
  ↓
Reasoning-aware RL（GRPO / PPO）
  ↓
Output-style 对齐（隐藏 CoT）
```

而不是论文里常画的那条简化直线。

---

## 八、关键结论（与你的问题一一对应）

> **业界不是用 RL 来“点亮 CoT”，而是用 SFT 把“慢思考模式”先焊死在模型里。**

因此：

* Pretrain + 普通 instruct ≠ reasoning-ready
* CoT 能力是**被精心保留下来的，不是自然涌现的**
* RL 只是：

  * 防止胡思乱想
  * 提高收敛与稳定性

---

如果你愿意，下一步我可以具体拆：

* **DeepSeek / Qwen / OpenAI 在 reasoning warm-start 上的差异**
* **为什么 GRPO 比 PPO 更适合 reasoning**
* **MoE 模型中 reasoning expert 是如何被“激活”的**

你这个问题，已经完全站在“训练配方设计”的视角了。


---

### 🧑‍💻 User

通过阅读deepseek-r1/qwen3 Technical Report，我们发现在ReasoningRL阶段之前，会先进行个Long-CoT Cold Start的sft步骤，通过构建长思维链数据和采用特定的数据收集与处理方法（可以是reasoning模型合成推理路径+人工筛选），为模型注入基础的推理模式。这个是否是比较成熟的做法？

---

### 🤖 Assistant

是的，你对 **DeepSeek-R1 / Qwen3 这类 reasoning 模型训练流程的理解基本上是准确的**，而且这确实已经成为当前 **一类较为成熟、被多方采用的工程实践**。核心在于：

> **为了让在进入推理引导的强化学习（Reasoning RL）之前模型已经具备稳定的 long-CoT 能力，必须在 RL 之前用专门构造的 SFT 数据“预热”（cold start）模型。** ([CSDN博客][1])

下面我把关键点按技术细节与成熟度做清晰说明：

---

## 一、什么是 Long-CoT Cold Start SFT

所谓 **cold start SFT**，核心是：

* 用 **少量但质量非常高的推理链条数据（long chain of thought）** 去 finetune 基座模型，
* 数据来源一般包括：

  * few-shot prompt 引导生成
  * 目标生成带有推理过程的答案
  * 人工筛选和清洗
* 数据格式通常明确标注 **中间推理过程 + 最终总结/答案**。 ([53AI][2])

这样的 SFT 有两个本质作用：

1. **注入稳定的推理轨迹模式**
   让模型学会在输出中包含合理论证步骤，而不是直接给出答案。

2. **提高 RL 的训练稳定性与效率**
   避免 RL 一开始就从完全 “不会思考” 的模型出发，导致梯度噪声大、探索困难、容易学坏。 RL 在有基础能力的模型上 converge 更快、更稳定。 ([CSDN博客][1])

---

## 二、为什么这是工业界主流做法

从多个公开的技术资料和社区总结来看：

### 1. DeepSeek-R1 的训练流程

DeepSeek-R1 在其 pipeline 中明确设定了 **冷启动阶段 Cold Start SFT**，这是整个训练体系的第一步：

* 对 base model 用少量 cold start SFT 数据 fine-tune
  → 得到初版推理基础模型（例如 R1-Zero）。
* 再对 R1-Zero 进行 RL 训练提升 reasoning 能力。
* RL 收敛后，用拒绝采样从训练 checkpoint 生成大量高质量推理数据用于再次 SFT。
* 再做 new SFT + 进一步 RL。 ([53AI][2])

这个结构清晰体现了 **SFT → RL → 数据扩增 → SFT → RL** 的循环，而不是简单的 Pretrain + RL。

---

### 2. Community 与工程实践也证实该路径

从多个社区资料总结：

* DeepSeek 团队明确用了 **几千条长推理冷启动数据做初期 SFT**。 ([CSDN博客][1])
* 冷启动数据强调整体结构格式，对 readability 和 reasoning 结构都有专门格式约束。 ([53AI][2])
* 同时有信源指出 **Cold Start SFT 帮助显著提升后续 RL 效果**。 ([Reddit][3])

---

## 三、这做法是否成熟

是的，这种组合做法已经不仅仅是 “零散的实验性尝试”，而是 **多个领先模型都在实操中采用或验证过的训练策略**：

### 成熟性的体现

1. **不是单一实验室的探索，而是多个组织/社区共识**
   DeepSeek 系列、Light-R1 等训练管线都在实践 cold start SFT + RL 组合。 ([arXiv][4])

2. **有明确的工程收益**
   比纯 RL 或单纯 SFT 效果更稳定、更高效、更易收敛。 ([53AI][5])

3. **也有文献/开源代码记录了同样思路**
   多篇社区解读与开源专案都在构造显式 CoT 数据用于冷启动或者长期循环生成训练数据。 ([DeepSeek技术社区][6])

---

## 四、为什么它有效（核心机理层面）

### 原因 1：强监督 vs 弱监督

Cold Start SFT **给出了完整的中间推理路径监督信号**，这比 RL 的单一 reward 要强得多，有效解决了：

* 低层次 token 逻辑错误
* 推理 chain structure collapse
* reward credit assignment 不明确

这直接提升模型的 reasoning 内部能力。

---

### 原因 2：降低 RL 的探索难度

在启用 RL 时：

* 如果模型完全没有 CoT 能力，RL 只能通过奖励强化输出长度，而不是真正的思考步骤；
* 有一个 good cold-start 模型，RL 可以进一步提高**质量**而不是从零开始“教会推理”。

---

## 五、对比没有 Cold Start 的 alternative

行业早期也尝试过：

* **zero-SFT + pure RL**（DeepSeek R1-Zero 就是这个思路）
  → 常出现 training instability。 ([CSDN博客][1])

* **直接大规模生成并蒸馏**
  → 效果很好，但对生成质量要求非常高，否则蒸馏数据本身有噪声。 ([DeepSeek技术社区][6])

Cold Start SFT 的优势是：

* 数据量小，但信号强
* 快速提升模型的 core reasoning behavior
* 为后续策略学习打好基础

所以在实际大规模训练体系里这是更“务实”的工程方案。

---

## 结论

**Long-CoT Cold Start SFT 在进入 RL 之前给模型注入基本推理模式，确实是当前工业界主流且成熟的做法之一。**

它的功效主要在于：

* 用高质量推理链数据稳定初始化模型
* 提升 RL 的收敛速度与稳定性
* 避免只用 RL 诱导 reasoning 时的 brittle behavior

因此，如果你的目标是训练一个稳定、强 reasoning 能力的 LLM，这个做法是**业界验证过、可直接采用的实践路径**。 ([CSDN博客][1])

如果你愿意，我也可以进一步细化：

* 典型数据构造方法与筛选策略
* 为什么冷启动数据不必太大但要格式化
* RL + SFT 混合循环训练为何更稳定与高效

[1]: https://blog.csdn.net/2401_84494441/article/details/152041771?utm_source=chatgpt.com "大模型数据构建实战指南：Qwen3/Deepseek/Kimi等技术全解析，建议收藏！_kimi、deepseek、qwen-CSDN博客"
[2]: https://www.53ai.com/news/finetuning/2025012520749.html?utm_source=chatgpt.com "谈谈对DeepSeek-R1的一些理解 - 53AI-AI知识库|企业AI知识库|大模型知识库|AIHub"
[3]: https://www.reddit.com/r/singularity/comments/1io9odx?utm_source=chatgpt.com "SYNTHETIC-1 finished generating 2 Million Reasoning samples using DeepSeek-R1 with decentralized compute - Next step is to use those samples to cold start a base model"
[4]: https://arxiv.org/abs/2503.10460?utm_source=chatgpt.com "Light-R1: Curriculum SFT, DPO and RL for Long COT from Scratch and Beyond"
[5]: https://www.53ai.com/news/finetuning/2025041607826.html?utm_source=chatgpt.com "三问一图万字拆解DeepSeek-R1：训练之道、实力之源与市场之变 - 53AI-AI知识库|企业AI知识库|大模型知识库|AIHub"
[6]: https://deepseek.csdn.net/6842d8497e10b149bf1ed362.html?utm_source=chatgpt.com "一篇详解！以DeepSeek-R1、Qwen3为例搞懂如何微调推理大模型_DeepSeek_Android老皮-DeepSeek技术社区"
