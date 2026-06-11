# SFT中Prompt不参与Loss计算

### 🧑‍💻 User

为什么sft训练中，prompt部分不参与loss计算？

---

### 🤖 Assistant

在大型语言模型（LLM）的监督微调（Supervised Fine-Tuning, SFT）中，不计算 Prompt（提示/指令）部分的 Loss，是基于**模型训练目标**和**实际推理场景**的严谨考量。

简单来说：**我们希望模型学习的是“如何根据问题回答”，而不是“如何提出问题”。**

以下是具体的核心原因：

### 1. 训练目标的本质：条件概率 vs 联合概率

从数学和算法的角度来看，LLM 的基础结构是自回归（Autoregressive）模型，基于前面的 Token 预测下一个 Token。

*   **如果不 Mask 掉 Prompt 的 Loss**：模型实际上在学习整个序列（Prompt + Response）的联合概率分布 $P(X, Y)$。这意味着模型不仅在学习如何生成高质量的回答，还在花精力学习用户通常会怎么提问、提问的语法结构等。
*   **Mask 掉 Prompt 的 Loss 后**：模型的训练目标变成了纯粹的**条件概率** $P(Y \mid X)$，即在给定输入 $X$（Prompt）的条件下，生成输出 $Y$（Response）的概率。这正是对话模型在实际应用中需要的行为。

### 2. 节省模型容量（Model Capacity）与计算焦点

模型的参数容量是有限的。在 SFT 阶段，数据的核心价值在于**高质量的回答（Response）**。
用户的 Prompt 往往是千奇百怪、口语化甚至包含错别字的。如果让模型计算 Prompt 的 Loss：
*   模型会浪费大量算力和参数去拟合用户千奇百怪的输入分布。
*   这会分散模型的注意力（Focus），导致其对高质量 Response 的学习不够充分，从而降低指令遵循（Instruction Following）的能力。

### 3. 防止推理时的“自言自语”（幻觉与重复）

在实际推理（Inference）时，我们提供 Prompt，模型生成 Response。
如果模型在训练时习惯了“生成 Prompt”，可能会导致一种常见的负面现象：**模型在回答完用户的问题后，自己又生成了一个新的提问，然后再自己回答。**
通过不计算 Prompt 的 Loss，模型会明确地知道：Prompt 是先决条件（只读），Response 才是需要我生成的内容（可写）。

---

<details>
<summary><b>🛠️ 展开查看：技术实现方式（PyTorch 中如何屏蔽 Prompt Loss）</b></summary>

在实际的代码实现中（例如使用 PyTorch 和 Hugging Face `transformers` 库），通常通过将 Prompt 对应的标签（Labels）设置为 `-100` 来实现忽略。

PyTorch 的 `CrossEntropyLoss` 默认有一个参数 `ignore_index=-100`。在计算 Loss 时，所有 Label 为 `-100` 的位置都会被直接跳过，不参与梯度回传。

**数据构造示例：**

```python
# 假设词表 ID 如下，并且我们要训练的数据是: 
# Prompt: "你好" (ID: 10, 11)
# Response: "你好，我是AI" (ID: 20, 21, 22, 23)

# 模型的输入 (Input IDs)
input_ids = [10, 11, 20, 21, 22, 23]

# 对应的标签 (Labels)
# Prompt 部分被替换为 -100，Response 部分保持原样
labels =    [-100, -100, 20, 21, 22, 23]
```
在这种构造下，前向传播时，模型依然能看到 Prompt 的前置 Context（因为注意力机制会看到前面的 `input_ids`），但在计算 Loss 并反向传播时，`-100` 的位置不会产生梯度更新。
</details>

<details>
<summary><b>💡 展开查看：有没有计算 Prompt Loss 的特例？</b></summary>

虽然标准 SFT 不计算 Prompt Loss，但在某些特定场景下，也会打破这个规则：

1.  **无监督预训练（Pre-training）或继续预训练（Continual Pre-training）：** 这个阶段没有明确的 Prompt 和 Response 的界限，是一整段纯文本，因此计算所有 Token 的 Loss，学习全局的语言分布。
2.  **某些特殊的统一架构训练：** 一些研究（如某些混合目标训练任务）为了让模型同时具备极强的续写能力和对话能力，可能会按照一定比例混合带有 Prompt Loss 的数据和不带 Prompt Loss 的数据。但这属于高级调优技巧，非标准 SFT 流程。
</details>

---

### 🧑‍💻 User

SFT 的样本有长有短，一些短的样本会存在很长的 padding 字符，这样对算力也是一种巨大的浪费，尤其是mask label完之后，要如何解决？

---

### 🤖 Assistant

这是一个非常敏锐且直击 LLM 训练痛点的问题。

在 Transformer 架构中，自注意力机制的计算复杂度是序列长度 $L$ 的平方即 $O(L^2)$。如果一个 Batch 中有一条长达 2048 的数据，而其他数据只有 128，那么短数据必须 Padding 到 2048。这些 Padding 字符虽然在 Loss 计算时被 Mask 掉了（Label 为 `-100`），但**它们依然参与了模型前向和反向传播中所有的矩阵乘法（QKV映射、注意力计算、FFN层等）**，这会造成极其严重的算力与显存浪费。

为了解决这个问题，目前工业界和学术界主要采用以下三种核心解决方案：

### 1. 数据拼接 (Sequence Packing / Multipack) —— 工业界最通用的方案

这是目前主流框架（如 Hugging Face TRL, LLaMA-Factory, Megatron-LM）最常用的做法。
**核心思想**：既然单条数据太短，那就把多条短数据“首尾相连”拼成一条长数据，直到达到 `max_seq_length`。

*   **做法**：假设 `max_seq_length` 为 2048。有三条长度分别为 500, 800, 700 的短样本。我们不再对它们各自 Padding，而是将它们拼接成一条长度为 2000 的序列，最后只加上 48 个 Padding 字符。
*   **隔离机制**：为了防止拼接在同一个序列中的不同样本产生“上下文污染”（即样本 B 看到了样本 A 的内容），通常有两种处理方式：
    1.  **简单粗暴版（依靠 EOS 截断）**：在每条样本末尾加上特定的 `<eos>` token，使用标准的因果掩码（Causal Mask）。模型会学习到遇到 `<eos>` 就重置上下文。
    2.  **严谨版（Block Diagonal Attention Mask）**：修改 Attention Mask 矩阵，使其变成块状对角矩阵。样本 A 的 Token 只能看到样本 A，样本 B 只能看到样本 B。

### 2. 变长注意力机制 (Unpadded / Variable Length FlashAttention) —— 底层算子级别的解决

这是从硬件和底层算子层面直接消灭 Padding 的高级方案。
**核心思想**：干脆在输入到模型前，把所有的 Padding 字符全部删掉，将整个 Batch 展平成一个一维的超长张量。

*   **做法**：利用 FlashAttention 等优化的注意力算子（具体使用的是 `flash_attn_var_len_func`）。我们需要传给算子两个东西：
    1.  **一维张量**：所有真实 Token 拼接在一起的序列（完全没有 Padding）。
    2.  **位置索引表 (`cu_seqlens`)**：告诉算子每一条数据在这一维张量中的起止位置（例如 `[0, 128, 256, 1024]`）。
*   **优势**：算子在计算 Attention 时，会严格按照 `cu_seqlens` 划定的边界计算，彻底零 Padding，极致压榨算力。

### 3. 按长度分组 (Length Grouping / Dynamic Batching) —— 数据加载器层面的优化

如果在某些场景下无法使用上述两种方法，可以在构建 Batch 的阶段进行优化。
**核心思想**：把长度相近的样本放在同一个 Batch 里。

*   **做法**：在每个 Epoch 开始前，根据样本长度对整个数据集进行排序或分桶（Bucketing）。在组装 Batch 时，确保同一个 Batch 内的样本长度差异尽可能小。
*   **效果**：虽然依然需要 Padding，但由于 Batch 内最长和最短的差距被缩小了，Padding 的总量会大幅度降低。例如，全都是 100~150 长度的样本分在一起，最多只需要 Padding 50 个 Token。

---

<details>
<summary><b>🛠️ 展开查看：Sequence Packing (数据拼接) 的直观对比</b></summary>

**传统 Padding 方式构建的一个 Batch (假设 max_len=10)：**
`[样本1, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]`
`[样本2, 样本2, 样本2, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]`
`[样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3]`
*这里产生了大量的 `<pad>` 计算浪费。*

**Sequence Packing 方式构建的一个 Batch (假设 max_len=10)：**
`[样本1, <eos>, 样本2, 样本2, 样本2, <eos>, 样本4, 样本4, 样本4, <eos>]`
`[样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3, 样本3]`
*短样本被塞进了同一个 sequence 中，`<pad>` 几乎消失，GPU 利用率拉满。在 Hugging Face 的 `SFTTrainer` 中，只需设置 `packing=True` 即可一键开启此功能。*

</details>

<details>
<summary><b>💡 展开查看：Length Grouping 带来的副作用与解决</b></summary>

虽然 **Length Grouping（按长度分组）** 很容易实现（Hugging Face `Trainer` 中只需设置 `group_by_length=True`），但它有一个不可忽视的副作用：

**破坏了数据的独立同分布（I.I.D）特性。**
因为模型会连续更新好几步全都是“极短对话”的梯度，然后又连续更新好几步全都是“极长文章”的梯度。这可能导致训练过程中的 Loss 震荡，甚至影响最终收敛效果。

**缓解方案**：
通常不会对全局数据进行绝对严格的排序，而是采用“局部打乱（Local Shuffling）”或“分桶后在桶间随机采样”的策略，在**减少 Padding** 和**保持数据随机性**之间取得平衡。
</details>