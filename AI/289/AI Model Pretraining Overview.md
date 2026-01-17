# AI Model Pretraining Overview

### 🧑‍💻 User

meta、qwen、deepseek、mistral、openai、claude等厂商针对预训练完成后的v1.0 Base模型第一轮sft后第一轮Continued Pretraining 文本量在多少B token级别？这部分涵盖的数据主要包含哪些方面？

---

### 🤖 Assistant

针对目前主流大模型厂商（Meta, DeepSeek, Qwen, Mistral等）的训练范式，你所提到的“第一轮SFT后”或“预训练末期”的 **Continued Pretraining (CPt)** 阶段，通常被称为 **"Annealing" (退火)** 阶段或 **"Long-context Extension" (长窗口扩展)** 阶段。

严格来说，大多数厂商并不是在“第一轮SFT完成之后”再回去做Continued Pretraining，而是在**Base模型预训练的最后阶段（SFT之前）**或者**SFT与RLHF的迭代循环中（Iterative SFT）**加入这部分高质量数据。

以下是针对各大厂商在该阶段的 Token 量级与数据构成的详细分析：

### 1. Token 量级概览

在 Base 模型预训练结束后的“退火”或“持续预训练”阶段，数据量通常占原始预训练数据量的 **0.1% ~ 5%** 左右，但其权重极高。具体的量级通常在 **40B ~ 1.5T Tokens** 之间。

| 厂商/模型 | 阶段名称 | 估算 Token 量级 | 备注 |
| :--- | :--- | :--- | :--- |
| **Meta (Llama 3/3.1)** | Annealing / Long-context | **800B Tokens** | Llama 3.1 在预训练最后阶段使用了800B数据进行128K长窗口和高质量数据退火。 |
| **DeepSeek (V2/V3)** | MTP / Context Extension | **~1.5T Tokens** | DeepSeek-V3 在预训练末期引入 Multi-Token Prediction (MTP) 和长窗口数据，约1.5T。 |
| **Qwen (2.5)** | Post-training Refinement | **未公开 (预计 100B+)** | 强调在 Post-training 阶段使用了海量合成数据，量级巨大。 |
| **Mistral** | Context Extension | **未公开 (预计 ~50B)** | 针对 Mistral Large/Nemo 的长窗口微调。 |
| **OpenAI (GPT-4/o1)** | Post-training RL | **N/A (闭源)** | o1 (Strawberry) 引入了极大规模的推理数据（RL tokens），量级可能达到 Base 训练的 10%。 |

### 2. 这部分涵盖的数据主要方面

这部分数据（Continued Pretraining / Annealing）与第一阶段大规模预训练（Pretraining）的“清洗互联网数据”不同，它追求极高的**信息密度**和**逻辑质量**。主要包含以下三类：

#### A. 核心能力数据 (STEM & Reasoning) - 占比最高 (~60-70%)
这是为了解决 Base 模型“这就好比读了万卷书但不会做题”的问题。
*   **数学 (Math):** 包含数学竞赛题、高数教材、带有详细推导步骤（Chain of Thought）的数学解题数据。
*   **代码 (Code):** 高质量的 GitHub 仓库（过滤掉低质代码）、LeetCode 题解、API 文档、合成的“Text-to-Code”数据。
*   **逻辑推理 (Logic):** 哲学论证、科学论文推导、合成的逻辑谜题。

#### B. 长上下文数据 (Long Context) - 占比次之 (~20%)
为了让模型从 4K/8K 窗口扩展到 128K 或 1M，必须在这一阶段引入长文本。
*   **书籍 (Books):** 完整的长篇小说、非虚构类书籍。
*   **财报与法律文档:** 具有超长跨度依赖关系的专业文档。
*   **代码库:** 跨文件的整个 Repository 代码，要求模型理解全局依赖。

#### C. 高质量合成数据 (Synthetic Data) - 关键增量
现代模型（尤其是 Qwen 2.5 和 Llama 3.1）在这一阶段大量使用由强模型（如 GPT-4 或上一代最强模型）生成的合成数据。
*   **Rejection Sampling (拒绝采样):** 对同一个问题生成多个回答，通过 Reward Model 筛选出最好的回答作为训练数据。
*   **合成教科书:** 类似于微软 phi 系列的思路，用模型重写互联网上的低质内容，将其转化为“教科书风格”的文本。

### 3. 具体案例深度解析

#### Meta (Llama 3 / 3.1) 的策略
Meta 在 Llama 3 技术报告中明确提到，他们在预训练的最后阶段（Pre-training Annealing）使用了 **800B (8000亿)** Token。
*   **目的：** 主要用于训练 128K 的长上下文能力，并巩固通用能力。
*   **数据：** 这一阶段的数据配比与原始预训练完全不同，大幅增加了**代码**和**数学**的权重，剔除了低质量的网页爬取数据。

#### DeepSeek (V3 / R1) 的策略
DeepSeek 展示了另一种路径，特别是 R1 (Reasoning) 模型。
*   **Base -> Cold Start SFT:** 首先用少量高质量数据（几千条）做冷启动。
*   **Reasoning CPt (RL阶段):** 实际上 DeepSeek R1 的核心在于 Base 模型之后的 **RL (强化学习)** 阶段。虽然这不叫 Continued Pretraining，但在这一阶段，模型自我生成的 Token 量（用于训练自身）是巨大的。
*   **数据内容：** 纯粹的 CoT（思维链）数据，强制模型在输出最终答案前生成 `<think>...</think>` 过程。

#### Qwen (2.5) 的策略
Qwen 2.5 能够霸榜的关键在于其 **Post-training** 阶段的数据规模。
*   他们不仅在 Base 阶段使用了 18T tokens，更在 SFT/CPt 阶段引入了**数百万条**合成的指令数据（Instruction Data）。
*   **重点：** 这里的重点不是“自然语言文本”，而是“Q-A 对”和“代码执行轨迹”。

### 总结

对于你提到的“第一轮 SFT 后”或“Base 模型完成后”的 Continued Pretraining：
1.  **数据量级：** 通常在 **40B 到 1T Token** 之间。
2.  **核心特征：** 这是一个 **"Quality over Quantity" (质量大于数量)** 的阶段。
3.  **数据构成：**
    *   $$ \text{Code} + \text{Math} + \text{Logic} \approx 70\% $$
    *   $$ \text{Long Context Books} \approx 20\% $$
    *   $$ \text{General High-Quality Text} \approx 10\% $$

各大厂商目前普遍认为，Base 模型的知识主要来自前 10T+ 数据，而模型的**智力（Reasoning）和遵循指令的能力**，主要取决于这最后几百 B Token 的高质量“退火”。

---

### 🧑‍💻 User

为什么目前的开源模型sft实际流程是「Pretrain ⇄ SFT」反复交织，而不是线性两阶段（Pretrain → SFT → RL → Release）？

---

### 🤖 Assistant

这是一个非常敏锐且切中要害的问题。你观察到的现象准确描述了 2024 年以后高性能大模型（如 Llama 3、Qwen 2.5、DeepSeek-V3/R1）的核心训练范式转变。

传统的 **「Pretrain → SFT → RL → Release」** 这种线性流水线模式（主要流行于 GPT-3/ChatGPT 早期时代）已经被一种 **「迭代式（Iterative）」** 或 **「螺旋上升」** 的模式所取代。

造成这种转变的核心原因可以概括为：**模型本身成为了生成和筛选高质量数据的工具**。

以下是具体的四大技术动因，解释了为什么现在是 Pretrain 和 SFT（及 RL）反复交织：

### 1. 合成数据飞轮 (The Synthetic Data Flywheel)
这是最根本的原因。在早期，SFT 数据主要靠人类标注。但随着模型能力提升，人类标注的数据在**量**和**质**上都无法满足需求。

*   **线性模式的局限：** 模型只能学到人类标注员写好的数据。
*   **交织模式的优势：**
    1.  先训练一个 **v1版本的 SFT 模型**。
    2.  利用 v1 模型针对大量 Prompt 生成回答（Synthetic Data）。
    3.  利用 Reward Model 或强模型（如 GPT-4）对生成的回答进行评分和筛选（Rejection Sampling，拒绝采样）。
    4.  **回流：** 将筛选出的高质量合成数据，加入到下一轮的 Pretrain 或 SFT 中。
    5.  训练出更强的 **v2 模型**，重复上述过程。

**案例：** **Llama 3** 和 **Qwen 2.5** 均明确表示，其后期训练数据中有数百万甚至上千万条是由上一轮模型生成的合成数据。如果不进行“交织”，就无法利用模型自身的能力来滚雪球。

### 2. 缓解“对齐税”与灾难性遗忘 (Mitigating Alignment Tax)
SFT（有监督微调）本质上是在改变模型的概率分布，使其符合人类对话格式。但过度的 SFT 会导致模型**变笨**（SFT 导致 Base 模型的分布偏移，丢失了 Pretrain 阶段的通用知识），这被称为“对齐税”。

*   **混合策略（Data Replay）：**
    在进行 SFT 时，厂商往往会混入（Replay）一部分 Pretrain 阶段的高质量数据（如数学、代码、科学论文）。
*   **交织流程：**
    如果发现 SFT 后模型在某个领域（比如化学）能力下降，工程师会立即停止，**回到类似 Pretrain 的状态**，针对性地加强该领域的文本训练（Continued Pretraining），然后再切回 SFT。这种动态调整打破了严格的阶段界限。

### 3. 利用 Checkpoint 进行数据清洗 (Model-Based Filtering)
这是 Meta 和 DeepSeek 非常推崇的策略。**最好的数据清洗器，就是你刚刚训练出来的那个模型。**

*   **流程：**
    1.  **Pretrain (Early Stage):** 训练一个中间检查点模型（Base Model checkpoint）。
    2.  **Training for Classification:** 对这个 Base 模型做轻量级 SFT，使其具备识别“高质量文本”和“垃圾文本”的能力。
    3.  **Data Filtering:** 用这个微调后的模型去清洗剩下的几十 TB 的原始语料。
    4.  **Continue Pretrain:** 用清洗后的更纯净数据，继续训练 Base 模型。

这种 Pretrain 和 SFT 能力的交互，保证了预训练数据的质量随着模型变强而不断提升。

### 4. DeepSeek-R1 模式：推理能力的特殊需求
DeepSeek-R1 (DeepSeek-V3 的推理版本) 的出现彻底打破了线性流程。其流程是典型的交织状：

$$ \text{Base} \rightarrow \text{SFT (Cold Start)} \rightarrow \text{RL (Reasoning)} \rightarrow \text{SFT (Distillation)} \rightarrow \text{RL (Final)} $$

*   **为什么要交织？**
    DeepSeek 发现，如果直接在 Base 模型上做大规模 SFT，会限制模型的思维发散性。
    1.  **第一轮 SFT (Cold Start):** 只是为了让模型学会由 `<think>` 标签开始思考，数据量极小。
    2.  **第一轮 RL:** 让模型自己探索思维链，产生大量成功的推理路径。
    3.  **第二轮 SFT:** **这是关键点。** 他们将第一轮 RL 产生的高分样本（成功的思考过程）收集起来，变成 SFT 数据，去微调 Base 模型（或者 V3 模型）。这相当于把 RL 探索出的“智力”固化下来。
    4.  **第二轮 RL:** 在固化智力的基础上做对齐。

### 总结：从“流水线”到“炼丹炉”

*   **旧模式 (Linear):** 像汽车组装线。零件（数据）是现成的，组装（Pretrain），喷漆（SFT），出厂。
*   **新模式 (Iterative):** 像炼钢或炼丹。
    *   **Pretrain** 是熔炼矿石。
    *   **SFT** 是锻打成型。
    *   发现杂质（能力缺陷）？**回炉**（Continued Pretrain）重炼。
    *   需要更高强度（推理能力）？**加入催化剂**（合成数据/RL）再锻打。

目前的 SFT 实际上更像是一个 **"Post-training Refinement Loop" (后训练精炼循环)**。在这个循环中，预训练数据、人类指令数据、模型合成数据被统一看待，根据模型当前的能力短板，动态地调整配比并反复训练。