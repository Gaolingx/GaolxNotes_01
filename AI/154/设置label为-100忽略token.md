# 设置label为-100忽略token

### 🧑‍💻 User

让模型忽略某个token可以怎么做，将label设置成-100是怎么实现的？（常见于sft全监督/半监督）

---

### 🤖 Assistant

在大型语言模型（LLM）的监督微调（SFT）中，让模型“忽略”某些token（通常是Prompt部分或Padding部分）最标准的做法，就是**在计算损失（Loss）时将这些token的标签（label）设置为 `-100`**。

以下是关于如何实现以及底层原理的详细解析：

### 1. 为什么是 `-100`？底层是怎么实现的？

将 label 设置为 `-100` 的底层逻辑完全依赖于深度学习框架（主要指 **PyTorch**）中损失函数的底层实现机制。

在 PyTorch 中，LLM 最常用的损失函数是 `torch.nn.CrossEntropyLoss`。这个函数内部有一个参数叫做 `ignore_index`，**它的默认值刚好就是 `-100`**。

#### 数学与底层逻辑
标准的交叉熵损失函数公式为：
$$ \text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i, y_i}) $$
其中 $N$ 是 token 的总数，$y_i$ 是真实的标签ID，$p_{i, y_i}$ 是模型预测该标签的概率。

当 PyTorch 的底层的 C++/CUDA 代码遇到 $y_i = -100$ 时，它会触发以下两个关键动作：
1. **跳过损失计算：** 对于 $y_i = -100$ 的位置，直接不计算 $-\log(p_{i, y_i})$，该位置的损失视为 $0$。因此，模型在反向传播时，**不会针对这些 token 计算梯度**，权重也就不会根据这些 token 进行更新。
2. **调整平均分母：** 如果 `reduction='mean'`（默认求平均），PyTorch **不会**把 `-100` 的 token 计入分母 $N$ 中。分母会变成有效 token 的数量 $M$。
   实际计算公式变为：
   $$ \text{Loss} = -\frac{1}{M} \sum_{i=1, y_i \neq -100}^{N} \log(p_{i, y_i}) $$
   *如果分母依然是 $N$，会导致 Loss 被严重稀释（被一堆 0 拉低），从而影响学习率和收敛。PyTorch 底层巧妙地处理了这一点。*

---

### 2. SFT 中如何构建包含 `-100` 的 Labels？

在 Causal LM（因果语言模型）的 SFT 过程中，我们的目标是：**只让模型学习如何生成回答（Response），而不去拟合用户的指令（Prompt）。**

具体构建 `input_ids` 和 `labels` 的步骤如下：

```python
import torch

# 假设词表ID如下：
# <pad>: 0, [User]: 10, 你好: 11, [Bot]: 20, 很高兴: 21, 认识你: 22

# 1. 拼接完整的输入 (Prompt + Response)
input_ids = [10, 11, 20, 21, 22]  # "[User] 你好 [Bot] 很高兴 认识你"

# 2. 初始化 labels，通常先完全复制 input_ids
labels = input_ids.copy()         # [10, 11, 20, 21, 22]

# 3. 找到 Prompt 部分的长度，将其对应的 label 替换为 -100
prompt_length = 3                 # "[User] 你好 [Bot]" 占3个token
for i in range(prompt_length):
    labels[i] = -100

print("Input IDs:", input_ids)
# Input IDs: [10, 11, 20, 21, 22]

print("Labels:   ", labels)
# Labels:    [-100, -100, -100, 21, 22]
# 模型只会在预测 "很高兴" (21) 和 "认识你" (22) 时产生 Loss 和梯度。
```

---

### 3. 代码验证 `-100` 的效果

我们可以用一段简单的 PyTorch 代码来验证 `-100` 是如何工作的：

```python
import torch
import torch.nn as nn

# 假设 vocab_size=5, sequence_length=3
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

# 模型的 logits 输出 (未经过 softmax)，形状: (batch=1, seq_len=3, vocab_size=5)
logits = torch.tensor([[[ 1.0,  2.0,  0.5, -1.0,  0.0],  # Token 1 预测
                        [-1.0,  0.0,  3.0,  1.0,  0.5],  # Token 2 预测
                        [ 0.0, -0.5,  0.0,  0.0,  2.5]]])# Token 3 预测

# 情况A：所有 Token 都计算 Loss (比如预训练阶段)
labels_all = torch.tensor([[1, 2, 4]]) 
loss_all = loss_fct(logits.view(-1, 5), labels_all.view(-1))
print(f"Loss (All tokens): {loss_all.item():.4f}")

# 情况B：忽略前两个 Token (比如 SFT，只学最后一个 Token)
labels_masked = torch.tensor([[-100, -100, 4]]) 
loss_masked = loss_fct(logits.view(-1, 5), labels_masked.view(-1))
print(f"Loss (Masked tokens): {loss_masked.item():.4f}")

# 情况C：手动只计算最后一个 Token 的 Loss (等价验证)
loss_manual = loss_fct(logits[:, 2, :], labels_masked[:, 2])
print(f"Loss (Manual single token): {loss_manual.item():.4f}")
```
**输出结果将证明 情况B 和 情况C 的 Loss 完全一致**，证明 `-100` 完美忽略了前两个 token 且正确调整了分母。

---

<details>
<summary><b>补充：另一种忽略 Token 的方式（Attention Mask）</b></summary>

除了在 Loss 层面设置 `-100`，还有一种“忽略”是在**前向传播（Forward）**阶段实现的，这就是 `attention_mask`。

*   **Labels = -100** 决定了模型**不因该 token 被惩罚（不更新梯度）**。但是，模型在生成后面的回答时，**依然可以“看到”并理解**这些 token（因为需要根据 Prompt 生成 Response）。
*   **Attention Mask = 0** 决定了模型在计算自注意力（Self-Attention）时，**完全看不到这些 token**。这通常用于批量处理时的 `<pad>` (填充) 字符。

在标准的 SFT 中：
*   Prompt 部分：`attention_mask = 1`, `label = -100`（能看到，但不计算 Loss）。
*   Response 部分：`attention_mask = 1`, `label = 真实ID`（能看到，且计算 Loss）。
*   Padding 部分：`attention_mask = 0`, `label = -100`（看不到，也不计算 Loss）。

</details>

---

### 🧑‍💻 User

如果seq全被mask掉了没做异常处理会发生什么？

---

### 🤖 Assistant

如果在数据预处理或训练时没有做异常处理，导致一个 batch 内某条序列（甚至整个 batch）的 `labels` 全部变成了 `-100`，会引发一场**“灾难性的连锁反应”**，最终通常会导致**模型崩溃（Model Collapse）**。

以下是具体的发生过程和底层原理：

### 1. 第一张多米诺骨牌：Loss 变成 `NaN`

正如前面提到的，当 `CrossEntropyLoss` 计算平均损失（`reduction='mean'`）时，它会自动忽略 `-100` 的 token，并且**不在分母中计入这些 token**。

如果整个序列（或整个 Batch）的 label 全部是 `-100`：
*   有效 token 的数量（分母）变为 $0$。
*   累计的 Loss 值（分子）也是 $0$。
*   在底层 C++ 计算时，触发 $0.0 / 0.0$ 的除法。
*   **结果：该步的 Loss 返回 `NaN`（Not a Number）。**

---

### 2. 连锁反应：从 `NaN` Loss 到模型崩溃

一旦 Loss 变成了 `NaN`，如果不加干预，会按照以下顺序毁掉你的训练：

1.  **梯度（Gradients）全部变成 `NaN`：**
    当调用 `loss.backward()` 时，反向传播算法会基于 `NaN` 的 Loss 计算梯度。由于任何数与 `NaN` 的数学运算结果都是 `NaN`，模型所有参与计算的参数梯度瞬间全部变成 `NaN`。
2.  **权重（Weights）被污染：**
    当执行 `optimizer.step()` 时，优化器会用这些 `NaN` 梯度去更新模型的权重。这导致模型的参数（Weights）直接变成 `NaN`。
3.  **模型彻底“死亡”：**
    在下一个 Step 时，由于模型权重已经是 `NaN`，前向传播（Forward）计算出的 Logits 也会全都是 `NaN`。从此以后，不论输入什么正常数据，Loss 永远是 `NaN`，模型彻底失去预测能力，无法自动恢复。

<details>
<summary><b>分布式训练中的“放大效应”（点击展开）</b></summary>

如果你在使用多卡分布式训练（如 DDP, DeepSpeed, Megatron）：
由于在计算完梯度后，各个 GPU 之间会进行梯度同步（AllReduce）。**只要有一张卡上的某一条数据引发了 `NaN` 梯度，这个 `NaN` 就会像病毒一样在同步过程中传染给所有 GPU**。结果就是，整个集群训练的模型瞬间全部崩溃。

</details>

---

### 3. 混合精度（AMP）环境下的特殊情况

现在训练大模型基本都会使用混合精度（FP16/BF16）和梯度缩放器（`GradScaler`）。这种情况下，表现会有所不同，但依然致命：

*   `GradScaler` 在执行 `optimizer.step()` 之前，会检查梯度中是否包含 `inf` 或 `NaN`。
*   如果发现了 `NaN`，`GradScaler` 会**跳过本次权重更新（Skip step）**，并将缩放因子（Scale Factor）减半。
*   **后果：** 如果这种“全 -100”的异常数据频繁出现，`GradScaler` 会不断减半缩放因子，最终缩放因子趋近于 $0$。这会导致正常的梯度也变成 $0$，**模型直接停止学习（Loss 不降）**，也就是常说的“假死”状态。

---

### 4. 常见的防范与异常处理方法

为了避免这种情况，在构建 DataLoader 或者 Loss 计算时，通常必须加入以下异常处理：

**方法一：在数据预处理阶段过滤（最推荐）**
在将数据送入训练集之前，检查每条数据的 labels。如果发现所有元素都是 `-100`，直接丢弃该样本。
```python
# 过滤掉无效数据
valid_datasets = []
for item in datasets:
    if not all(label == -100 for label in item['labels']):
        valid_datasets.append(item)
```

**方法二：在计算 Loss 时加安全锁（兜底方案）**
在自定义的 Loss 计算代码中，检查如果分母为 0，则直接返回 0 并且停止反向传播，或者手动给 Loss 赋 0.0 并保留梯度图。
```python
def safe_cross_entropy_loss(logits, labels):
    # 将 -100 变成 false, 其他变成 true
    valid_mask = (labels != -100)
  
    # 检查是否全军覆没
    if valid_mask.sum() == 0:
        # 返回一个带有梯度的 0 损失，防止后续流程报错
        return logits.sum() * 0.0 
  
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
```

---

### 🧑‍💻 User

哪些不当的dataset设置会导致这种全mask情况出现？

---

### 🤖 Assistant

在 SFT（监督微调）数据预处理时，如果整个序列的 label 全部变成了 `-100`，通常是由**数据集本身的缺陷**或**预处理脚本（Tokenization & Masking）的逻辑漏洞**导致的。

以下是引发“全序列 `-100`”的最常见场景，按发生频率从高到低排序：

### 1. 截断策略不当（最常见）

在大模型训练中，通常会设置一个最大序列长度 `max_seq_length`（如 2048 或 4096）。当输入数据超过这个长度时，需要进行截断。

**错误场景：**
如果一条数据中，**Prompt（指令部分）本身非常长**（例如让模型阅读一篇超长文章并总结），其长度已经超过或等于 `max_seq_length`。
默认的截断策略通常是 `truncation='longest_first'` 或从右侧截断（丢弃尾部）。这会导致**回答（Response）部分被完全切掉**，序列中只剩下了 Prompt。
由于代码逻辑会把 Prompt 对应的标签全部设置为 `-100`，截断后剩下的 token 就全变成了 `-100`。

```python
# 假设 max_length = 5
# 原始: Prompt (长为6) + Response (长为2)
# 截断后只剩前5个 token，全是 Prompt
labels = [-100, -100, -100, -100, -100] 
```

### 2. 模板匹配（Template Matching）失败

为了区分哪部分是 Prompt 哪部分是 Response，预处理脚本通常会搜索特定的“分隔符”或“触发词”（例如 `<|im_start|>assistant` 或 `[Bot]:`）。找到这个位置后，将它前面的所有 token label 置为 `-100`。

**错误场景：**
*   **格式不统一：** 数据集中部分数据的格式写错了（比如原本该有 `Assistant:` 的地方写成了 `assistant:` 或漏掉了空格）。
*   **Tokenizer 的贪婪合并（BPE）：** 字符串级别的匹配和 Token 级别的匹配不一致。比如 `" Assistant"` 和 `"Assistant"` 在分词后可能是完全不同的 Token ID。如果脚本按 Token ID 去查找边界，可能会找不到。

如果找不到边界，很多暴力脚本的默认处理逻辑可能是：**把整句话当作 Prompt，全部赋予 `-100`**。

### 3. 数据集存在“空回复”脏数据

在抓取、清洗或格式化数据时，可能会漏掉一些异常样本，导致数据的输出目标（Target/Output）本身就是空的。

**错误场景：**
```json
{
  "instruction": "请把以下句子翻译成英文：你好。",
  "input": "",
  "output": ""  // 致命错误：回复为空
}
```
经过 `Prompt + Output` 的拼接后，整个序列实际上只有 Prompt，最终 label 必然全部是 `-100`。

### 4. 数据填充（Padding）与批处理（Collate）的 Bug

由于同一个 Batch 内的文本长度必须对齐，较短的文本会被填充 `<pad>` token。按照标准逻辑，`<pad>` 对应的 label 也要设为 `-100`。

**错误场景：**
如果 DataLoader 的 `collate_fn` 存在 bug，或者在构建分布式训练的虚拟 Batch 时，为了凑齐 Batch Size 塞入了**纯填充序列**（一整行全是 `<pad>`），那么这行数据的 label 也会全是 `-100`。

---

<details>
<summary><b>🛠️ 如何排查并修复这些问题？（排查指南）</b></summary>

要彻底杜绝这种情况，建议在你的 Dataset 或 DataLoader 代码中加入以下几重保险：

**1. 加入有效性断言（Assertion）**
在 map 函数（预处理函数）的最后，强制检查 label：
```python
def preprocess_function(example):
    # ... 前面的 tokenization 和 mask 逻辑 ...
  
    # 检查: 如果没有一个 token 的 label 大于等于 0 (即全为 -100)
    if not any(l >= 0 for l in labels):
        # 打印出来看看是哪条数据作妖
        print(f"警告: 发现无效数据, 文本为: {example['text']}")
        # 或者直接抛出异常，提前终止
        # raise ValueError("Found a sequence with all -100 labels!")
      
    return {"input_ids": input_ids, "labels": labels}
```

**2. 优化截断策略（Left Truncation for Prompt）**
如果遇到超长 Prompt，最好的做法不是截掉 Response，而是**从左侧截断 Prompt**（保留 Prompt 的尾部指令和完整的 Response）。
```python
# 伪代码：优先保留 Response
if len(prompt_ids) + len(response_ids) > max_length:
    # 计算需要切掉多少个 prompt token
    excess = len(prompt_ids) + len(response_ids) - max_length
    prompt_ids = prompt_ids[excess:] # 从左侧切掉多余的 prompt
```

**3. 数据清洗脚本**
在训练前，写个简单的脚本扫一遍你的 JSON/JSONL 数据集，把 `output` 字段为空格、长度为 0 或只包含换行符的样本全部过滤掉。

</details>