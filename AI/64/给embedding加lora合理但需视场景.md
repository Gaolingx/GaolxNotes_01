# 给embedding加lora合理但需视场景

### 🧑‍💻 User

hf transformers peft中，给embedding加lora adaptor合理吗？

---

### 🤖 Assistant

在 Hugging Face 的 `transformers` 和 `peft` 库中，**给 Embedding 层添加 LoRA Adapter 是合理的，但并非所有场景都必需。** 它取决于你的具体任务需求。

总的来说，给 Embedding 加 LoRA 属于一种**“进阶的微调策略”**。在大多数常规的指令微调（Instruct Fine-tuning）中，仅对 Attention 层和 MLP 层加 LoRA 就足够了；但在涉及领域自适应、多语言扩展或词表扩充时，对 Embedding 层加 LoRA 就显得非常有意义。

以下是详细的分析与使用建议：

### 1. 什么时候给 Embedding 加 LoRA 是合理的？

<details>
<summary><b>点击展开：适用场景详细说明</b></summary>

*   **领域自适应 (Domain Adaptation) / 持续预训练 (Continual Pre-training)**
    如果你在进行特定垂直领域的微调（例如医疗、法律、代码），很多常见词汇在特定领域的语义会发生改变（例如“苹果”在通用语境下是水果，在科技语境下是公司）。此时，调整 Embedding 层有助于模型从源头上修正词汇的语义表示。
*   **多语言微调 (Multilingual Fine-tuning)**
    如果基础模型在某种语言上能力较弱，你想通过微调增强该语言的能力。由于不同语言的 Token 分布和语义空间差异较大，微调 Embedding 层能显著加速模型对新语言的适配。
*   **输入特征的全局性偏移**
    Embedding 是模型接收输入的第一层，如果你的任务输入分布与预训练数据差异极大，仅靠深层（Attention/MLP）的 LoRA 可能难以拉回这种底层特征的偏差。
</details>

### 2. 什么时候不需要或不建议给 Embedding 加 LoRA？

<details>
<summary><b>点击展开：不适用场景详细说明</b></summary>

*   **常规的对话/指令微调 (General Instruct-Tuning)**
    如果你的任务只是教模型如何以特定的格式回答问题（例如 QA 任务、总结任务），且没有引入大量生僻词或跨领域知识，此时基础模型的 Embedding 已经足够优秀。增加 Embedding 的 LoRA 只会徒增显存占用和训练时间，收益甚微。
*   **只增加了极少数的新 Token**
    如果你只是在词表末尾添加了几个特殊 Token（如 `<|im_start|>`, `<pad>`），更常见的做法是调整 `Embedding` 矩阵的大小，并将 `Embedding` 层和 `lm_head` 层加入到 `modules_to_save` 中（即对这些层进行全量微调），而不是给整个极其庞大的 Embedding 矩阵加上 LoRA。
</details>

### 3. 数学原理与底层逻辑

Embedding 层本质上是一个没有偏置（bias）的线性层（Lookup Table）。
假设词表大小为 $V$，隐藏层维度为 $d$，则原 Embedding 矩阵 $W_0 \in \mathbb{R}^{V \times d}$。

当对其应用 LoRA 时，更新矩阵 $\Delta W$ 的计算方式为：
$$ \Delta W = B \cdot A $$
其中降维矩阵 $B \in \mathbb{R}^{V \times r}$，升维矩阵 $A \in \mathbb{R}^{r \times d}$，且秩 $r \ll \min(V, d)$。

前向传播时，对于输入的 One-hot 向量 $x$：
$$ h = x W_0 + x (B \cdot A) \cdot \alpha/r $$

**优势：** 相比于全量微调整个 $V \times d$ 的矩阵（词表往往高达几万甚至十几万），LoRA 只需要训练 $(V \times r + r \times d)$ 个参数，极大降低了显存消耗。

### 4. 在 `peft` 中如何实现？

在 `peft` 中，你可以通过配置 `LoraConfig` 中的 `target_modules` 来将 LoRA 应用于 Embedding 层。不同模型的 Embedding 层名称不同（例如 LLaMA 中叫 `embed_tokens`，Qwen 中叫 `wte`）。

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LLaMA 模型的层命名示例
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens",  # 给 Embedding 层加 LoRA
        "lm_head"        # 通常如果动了 Embedding，lm_head 也会一起加 LoRA 或全量微调
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

### 总结建议

1.  **优先尝试不加：** 先只对 Attention 和 MLP 加 LoRA。如果 Loss 降不下去，或者在专业词汇、特定语言上表现糟糕，再考虑给 Embedding 加 LoRA。
2.  **联动 `lm_head`：** 在语言模型中，Embedding（输入词向量）和 `lm_head`（输出分类器）通常具有强相关性（有些模型甚至共享权重）。如果你决定给 Embedding 加 LoRA，强烈建议同时也给 `lm_head` 加 LoRA。
3.  **区分扩充词表：** 如果你的主要目的是**扩充词表**，首选方案是将 `Embedding` 设为 `modules_to_save` 进行全量微调，因为新初始化的 Token 向量需要较大的梯度更新，LoRA 的低秩特性可能会限制新 Token 的学习能力。

---

### 🧑‍💻 User

只扩充了2个vocab但没训练embed token/lm head会有影响吗？

---

### 🤖 Assistant

**有非常大的影响，通常会导致微调失败或产生严重的格式错乱。**

简单来说：**如果你扩充了词表，就必须训练对应的 Embedding 和 LM Head。** 

以下是具体原因和不同场景下的影响分析：

### 1. 核心原因：随机初始化的“噪声”

当你向 Tokenizer 扩充了 2 个新 Vocab 时，模型的词表大小从 $V$ 变成了 $V+2$。为了匹配这个新的词表，你必须调整模型底层 `embed_tokens` 和顶层 `lm_head` 的矩阵形状（通常使用 `model.resize_token_embeddings()`）。

此时，矩阵中对应这 2 个新 Token 的权重向量（即新增的那 2 行）是**随机初始化的**（或者是基于现有权重的均值初始化的，具体取决于代码实现）。

如果你在 PEFT (LoRA) 微调时冻结了这些层（没有将它们设为可训练），会出现以下致命问题：

*   **输入端 (`embed_tokens`)**：当模型读取到这两个新 Token 时，它提取出的是毫无意义的随机向量。这相当于给模型输入了乱码，模型无法理解这两个 Token 代表什么含义。
*   **输出端 (`lm_head`)**：当模型尝试预测下一个 Token 时，计算这两个新 Token 概率的权重也是随机的。模型永远无法正确地学会何时该输出这两个 Token。

### 2. 影响程度（取决于这 2 个 Vocab 的用途）

这 2 个新 Token 的具体用途决定了灾难的严重程度：

<details open>
<summary><b>场景 A：作为对话模板的控制符（如 <code>&lt;|im_start|&gt;</code>, <code>&lt;|im_end|&gt;</code>）—— 致命影响</b></summary>

如果你添加的是用于区分 System、User、Assistant 或用于标记回答结束的特殊字符，并且没有训练它们：
1. **无法停止生成：** 模型算不出 `lm_head` 中 `<|im_end|>` 的正确概率，导致模型生成完答案后不知道停止，会一直胡言乱语直到达到 `max_length`。
2. **无法理解指令边界：** 模型把 `<|im_start|>` 当成随机噪声，无法区分哪里是用户的提问，哪里是系统的预设，导致指令遵循能力崩塌。
</details>

<details>
<summary><b>场景 B：纯粹作为 Padding Token（如 <code>&lt;pad&gt;</code>）—— 影响相对较小，但仍是隐患</b></summary>

如果基础模型没有 `<pad>`，你强行加了一个用于对齐 Batch 数据，且**严格保证了**：
1. 在输入时通过 `attention_mask = 0` 屏蔽了 `<pad>`。
2. 在计算 Loss 时通过 `labels = -100` 忽略了 `<pad>` 的预测。

在这种极端的严格控制下，模型的前向传播和损失计算不会直接受到这个随机向量的污染，可能勉强能训练。**但这也是非常不规范的做法**，极易在推理阶段产生不可预知的 Bug。
</details>

### 3. 正确的解决办法：使用 `modules_to_save`

在只增加极少量 Token（如 2 个）的情况下，**不要用 LoRA 去拟合 Embedding**，因为 LoRA 的低秩特性不利于从头学习全新的 Token 向量。

最标准的做法是：**将 `embed_tokens` 和 `lm_head` 进行全量微调，而模型的其他层依然使用 LoRA。**

在 `peft` 的 `LoraConfig` 中，你可以通过 `modules_to_save` 参数轻松实现这一点。配置如下：

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 你的常规 LoRA 目标层
    # 核心配置：将 Embedding 和 LM_Head 设为全参微调并保存
    modules_to_save=["embed_tokens", "lm_head"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**代码背后的数学逻辑：**
使用 `modules_to_save` 后，`embed_tokens` 矩阵 $\mathbb{R}^{(V+2) \times d}$ 会被从冻结状态唤醒，其梯度 $\nabla W$ 将正常计算并更新。由于大部分原本的 Vocab 在微调数据中出现频率较低，它们的权重基本保持不变，而那 2 个新增 Token 的随机权重会在微调过程中迅速收敛到正确的语义空间中。

---

### 🧑‍💻 User

还有一种情况，没有扩充vocab 但如果是预训练完全没见过tool_call这种spec token但sft阶段用到了不训练会有影响吗？

---

### 🤖 Assistant

**结论是：完全没有严重影响，模型依然能够正常学习并掌握这个能力。**

这是业界非常标准且常用的一种做法。只要你不主动调用 `add_tokens` 或 `add_special_tokens` 来扩充词表，仅仅是构造包含了 `<tool_call>` 这种字符串的 SFT 数据，即使完全冻结 Embedding 和 LM Head（只训练 Attention 和 MLP 的 LoRA），也是可行的。

以下是底层的运行逻辑和利弊分析：

### 1. 为什么不训练也能学会？（核心原理：子词切分）

因为你**没有扩充词表**，所以当 Tokenizer 遇到 `<tool_call>` 这个它没见过的“整体”时，它不会将其作为一个全新的、随机初始化的独立 Token 处理，而是**将其切分成它认识的已有 Token（Subwords）**。

例如，对于 LLaMA 的 Tokenizer，`<tool_call>` 可能会被切分成类似这样的序列（具体切分视不同模型而定）：
`['<', 'tool', '_', 'call', '>']` 或者 `['<t', 'ool', '_call', '>']`

1.  **输入端 (Embedding)**：切分出来的 `<`、`tool`、`call`、`>` 这些基础 Token 在预训练阶段都是见过无数次的，它们的 Embedding 向量是非常稳定且充满语义的。
2.  **中间层 (LoRA in Attention/MLP)**：模型通过深层的 Attention 机制，能够“看到”这几个基础 Token 按顺序组合在了一起。你加在 Attention 和 MLP 上的 LoRA 权重，会通过 SFT 数据的梯度更新，**学会识别这个特定的组合序列**，并赋予它“调用工具”的抽象语义。
3.  **输出端 (LM Head)**：当模型需要输出 `<tool_call>` 时，深层网络计算出的 Hidden State 会引导冻结的 `lm_head` 依次输出 `<`、`tool`、`_`、`call`、`>` 的高概率。因为它本来就会输出这些基础字符，现在只是 LoRA 告诉它“在这个特定语境下，请按顺序把这几个字符输出来”。

### 2. 这种做法的优势

*   **极度稳定，不会崩坏：** 因为底层 Embedding 没有任何随机噪声，模型不会出现生成乱码或不停止的问题。
*   **节省显存：** 不需要设置 `modules_to_save=["embed_tokens", "lm_head"]`，避免了全量微调这两个庞大矩阵带来的显存和显存带宽开销。

### 3. 这种做法的潜在缺点（相比于将其添加为独立的新 Token）

虽然能 work，但把特殊标记当作普通字符串（切分成多个 Token）处理，相比于将其注册为单行独立 Token（即上一问中扩充 Vocab 的做法），有几个小劣势：

<details open>
<summary><b>点击展开：潜在劣势分析</b></summary>

1.  **Context Window 占用更多：**
    如果 `<tool_call>` 被切分成 4 个 Token，那么模型每次生成或读取这个标记，都要消耗 4 个 Token 的上下文长度和算力。如果把它加进词表作为一个独立 Token，它就只占 1 个位置。对于频繁调用工具的 Agent 任务，积少成多会浪费不少上下文。
2.  **生成稳定性和“幻觉”风险：**
    输出 1 个独立的特殊 Token 是一步到位的。但输出 4 个 Token（比如 `<`、`tool`、`_`、`call`、`>`），在自回归生成时，模型中间有可能“跑偏”。比如它可能输出 `<tool_calls>` 或者 `<tool-call>`，导致你的后处理正则表达式匹配失败。不过，经过充分的 SFT，这种跑偏的概率非常低。
3.  **无法利用特殊 Token 的强制机制：**
    如果你将其添加为 `eos_token` 之一，模型生成它时可以直接触发底层的停止逻辑。如果是普通字符串组合，往往需要靠外层的 `stopping_criteria` 来截断。
</details>

### 总结

如果你不想折腾 Embedding 和 `lm_head` 的形状改变和全参微调，**直接把 `<tool_call>` 当成普通字符串写进 prompt 里，并只用常规的 LoRA 微调（不训练 Embedding），是完全合法且能达到很好效果的。** 

只有当你极其追求 Token 效率，或者对生成的绝对格式有着极其严格的强迫症时，才建议走“扩充词表 + 全量微调 Embedding/LM_head + 冻结其他层加 LoRA”的复杂路线。