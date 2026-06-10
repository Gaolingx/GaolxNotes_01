# 高阶LoRA逼近全参微调效果

### 🧑‍💻 User

对于一个4b参数的llm，rank 128 alpha 256 dropout 0 all linear+embedding+lm head的标准lora和全参数微调差异有多大，对于基模的首轮sft来说

---

### 🤖 Assistant

对于一个 4B 参数的基础模型进行**首轮 SFT（监督微调）**，在你给定的这套极其“豪华”的 LoRA 配置下（$r=128$, $\alpha=256$, `dropout=0`, 且作用于 `all linear + embedding + lm head`），**它与全参数微调（FFT, Full Fine-Tuning）在最终模型能力上的差异微乎其微，但在显存占用和训练成本上差异巨大。**

可以说，这套配置是一种**“无限逼近全参数微调”的高阶参数高效微调（PEFT）方案**。

以下是具体的差异对比和深度解析：

### 1. 核心结论：能力差异极小

在首轮 SFT 中，基础模型（Base Model）的主要任务是**完成行为范式的转换**（从“文本接龙”转变为“指令遵循”），这需要模型内部表征发生较大的偏移。
*   传统的低秩 LoRA（如 $r=8$，仅作用于 Attention 层）在首轮 SFT 中通常会明显逊色于 FFT。
*   但你的配置使用了极高的秩 $r=128$，并且覆盖了所有的线性层（Attention的 QKV/O，MLP的 Gate/Up/Down），**最关键的是你加入了 Embedding 和 LM Head**。这赋予了模型极大的参数表达空间。在大多数主流 Benchmark 和实际体感上，这套配置产出的模型与 FFT 产出的模型表现差异基本在 $\pm 1\%$ 到 $2\%$ 之间，甚至在某些容易过拟合的数据集上表现由于正则化效应而更好。

### 2. 关键维度差异对比

| 维度 | 高阶 LoRA ($r=128$, 全模块) | 全参数微调 (FFT) | 差异分析 |
| :--- | :--- | :--- | :--- |
| **可训练参数量** | 约占总参数的 5% - 8% (约 200M - 300M) | 100% (4B) | LoRA 参数量足够大，足以拟合复杂的指令遵循模式。 |
| **显存占用 (VRAM)** | **极低** (单张 24G/40G 显卡即可跑) | **极高** (通常需要多卡 + DeepSpeed Zero-3) | FFT 需要保存完整的 4B 优化器状态和梯度，而 LoRA 只需要保存那小部分参数的状态。 |
| **对新词汇/特殊Token的适应** | **优秀** (因为放开了 Embedding 和 LM Head) | **优秀** | 首轮 SFT 通常会引入 `<\|im_start\|>` 等特殊 Token，不微调 Embedding/LM head 会导致性能崩塌。你的配置完美避开了这个坑。 |
| **训练速度** | 较快（反向传播计算量少） | 较慢（完整的反向传播） | LoRA 节省了大量权重更新的时间和多卡通信开销。 |

<details>
<summary><strong>👉 展开查看：关于首轮 SFT 为什么要带上 Embedding 和 LM Head 的深度解释</strong></summary>

在首轮 SFT 中，基础模型第一次见到对话模板（Chat Template）。以 ChatML 为例，模型需要理解 `<|im_start|>`、`<|im_end|>` 等全新 Token 的语义。
如果采用标准 LoRA（不训练 Embedding 和 LM Head），这些新 Token 的表征向量是随机初始化或未充分学习的，模型只能靠中间层的 LoRA 权重去“硬拗”逻辑，这会导致：
1. 模型容易出现无法停止生成的现象（学不会输出 EOS 或 im_end）。
2. 格式遵循能力极差。
你的配置将这两个模块纳入训练，等同于打通了输入和输出的瓶颈，这是让 LoRA 逼近 FFT 效果的最关键一步。
</details>

<details>
<summary><strong>👉 展开查看：显存占用的数学估算</strong></summary>

假设使用 AdamW 优化器，以 16-bit (BF16/FP16) 精度训练一个 4B 模型：
对于全参数微调（FFT）：
*   模型权重：$4 \text{B} \times 2 \text{ bytes} = 8 \text{ GB}$
*   梯度：$4 \text{B} \times 2 \text{ bytes} = 8 \text{ GB}$
*   优化器状态 (AdamW, FP32)：$4 \text{B} \times 8 \text{ bytes} = 32 \text{ GB}$
*   **基础占用至少 48 GB**（还不包括激活值，实际可能需要 80GB 以上或切分）。

对于你的 LoRA：
*   冻结模型权重：$8 \text{ GB}$
*   LoRA 权重（假设 300M）：$300\text{M} \times 2 \text{ bytes} = 0.6 \text{ GB}$
*   LoRA 梯度：$0.6 \text{ GB}$
*   LoRA 优化器状态：$300\text{M} \times 8 \text{ bytes} = 2.4 \text{ GB}$
*   **基础占用仅需约 11.6 GB**，单张 RTX 4090 或 A10 即可轻松搞定。
</details>

### 3. 训练动态与超参数差异

虽然最终能力相似，但在**训练过程**中，你需要注意两者在超参数设置上的不同：

*   **学习率 (Learning Rate):**
    *   **FFT:** 对于 4B 模型，FFT 的学习率通常设得极低，例如 $1 \times 10^{-5}$ 到 $5 \times 10^{-6}$。
    *   **LoRA:** 由于缩放系数 $\Delta W = \frac{\alpha}{r} AB$，在你的配置中 $\frac{256}{128} = 2$。LoRA 的学习率需要比 FFT 大 1 到 2 个数量级。建议从 $1 \times 10^{-4}$ 到 $3 \times 10^{-4}$ 开始尝试。
*   **Dropout ($0$ 的影响):**
    *   在 $r=128$ 这样高的秩下，参数量不小。如果你用于 SFT 的数据集非常大（例如 $>50k$ 条高质量指令），`dropout=0` 是完全没问题的，可以最大化模型的拟合能力。
    *   如果你的 SFT 数据集很小（例如只有两三千条），这么高的 $r$ 加上 $0$ Dropout 可能会有**过拟合**的风险（模型变得像复读机）。此时建议略微开启 Dropout（例如 $0.05$）或者降低 Epoch 数量。

### 总结建议

对于 4B 级别的基础模型首轮 SFT，**强烈推荐使用你提出的这套 LoRA 配置而不是全参数微调**。

它能在节省 70% 以上显存、大幅降低多卡通信成本和硬件门槛的前提下，达到 98% 以上的 FFT 效果。你只需要确保在 `peft` 库中正确配置 `target_modules`：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention
        "gate_proj", "up_proj", "down_proj",    # MLP
        "embed_tokens", "lm_head"               # 核心：词表与输出头
    ],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
```
*注：具体 module 名称需根据你使用的基础模型架构（如 LLaMA, Qwen 等）进行适配。*

---

### 🧑‍💻 User

该rank下训练参数量一般在多少m params？（含embedding和lmhead）

---

### 🤖 Assistant

对于一个 4B 参数级别的模型，在 $r=128$ 且作用于 `all linear + embedding + lm head` 的配置下，训练参数量通常在 **240M 到 300M** 之间（约占总参数量的 **6% - 7.5%**）。

这个数值并不固定，因为它受到基础模型具体架构（主要是**隐藏层维度**、**层数**和**词表大小**）的直接影响。

为了让你有更清晰的概念，我们可以拆解一下这 200多M 参数是怎么算出来的。假设我们以一个典型的 4B 模型（参考 Qwen1.5-4B 或缩小的 LLaMA 架构）为例：

### 核心架构假设（典型 4B 模型）
*   **隐藏层维度 (Hidden Size, $h$)**: $2560$
*   **MLP 中间维度 (Intermediate Size, $d_{ff}$)**: $6912$
*   **模型层数 (Layers, $L$)**: $40$
*   **词表大小 (Vocab Size, $V$)**: $152,000$ (以 Qwen 为例，词表较大)
*   **LoRA Rank ($r$)**: $128$

对于任何一个线性层，LoRA 引入的参数量计算公式为：**$r \times (\text{输入维度} + \text{输出维度})$**

---

### 参数量详细拆解

#### 1. 注意力层 (Attention: q, k, v, o)
在标准多头注意力（MHA）下：
*   **q_proj / k_proj / v_proj / o_proj**: 输入和输出维度基本都是 $h=2560$。
*   单层单个投影矩阵的 LoRA 参数量 = $128 \times (2560 + 2560) = 655,360$ (约 0.65M)
*   单层 Attention (4个矩阵) = $0.65\text{M} \times 4 \approx 2.62\text{M}$
*   **40层 Attention 总计**: $2.62\text{M} \times 40 \approx$ **104.8 M**
*(注：如果模型使用了 GQA 组查询注意力，k_proj 和 v_proj 的参数量会更小，这部分的实际参数量可能会降到 80M 左右。)*

#### 2. 前馈神经网络层 (MLP: gate, up, down)
MLP 包含输入到中间层的扩张（gate, up）和中间层到输出的收缩（down）：
*   **gate_proj / up_proj**: 输入 $2560$，输出 $6912$
    *   参数量 = $128 \times (2560 + 6912) \approx 1.21\text{M}$
*   **down_proj**: 输入 $6912$，输出 $2560$
    *   参数量 = $128 \times (6912 + 2560) \approx 1.21\text{M}$
*   单层 MLP (3个矩阵) = $1.21\text{M} \times 3 \approx 3.63\text{M}$
*   **40层 MLP 总计**: $3.63\text{M} \times 40 \approx$ **145.2 M**

#### 3. 词表嵌入与输出头 (Embedding & LM Head)
这是该配置下非常吃参数的两块，**受到词表大小的绝对影响**。
*   **Embedding (`embed_tokens`)**: 词表映射到隐藏层。
    *   参数量 = $128 \times (152000 + 2560) \approx 19.78\text{M}$
*   **LM Head (`lm_head`)**: 隐藏层映射回词表概率。
    *   参数量 = $128 \times (2560 + 152000) \approx 19.78\text{M}$
*   **词表部分总计**: $19.78\text{M} \times 2 \approx$ **39.5 M**

---

### 汇总结果

| 模块 | 估算参数量 (Millions) | 占比 (相对于LoRA总参数) |
| :--- | :--- | :--- |
| **Attention 层** | ~ 100 M | 34% |
| **MLP 层** | ~ 145 M | 50% |
| **Embedding + LM Head** | ~ 40 M | 16% |
| **总计可训练参数** | **~ 285 M** | **100%** |

### 总结
在你的配置下，模型将会有 **~285M** 的可训练参数。

*   **如果你的基础模型词表很小**（例如 Phi-3 的词表只有 $32,000$ 左右），Embedding 和 LM Head 的参数量会骤降到只有 ~9M，此时总参数量会在 **240M** 左右。
*   **如果你的模型是深度极深、宽度较窄的设计**，MLP 层的参数占比还会进一步提高。

无论具体是 240M 还是 300M，这个级别的可训练参数量（几亿规模）对于首轮 SFT 来说已经拥有极其强大的表达能力，完全足够拟合高质量的指令遵循数据，同时依然能保持单卡（如 24G 显存）的轻松训练。