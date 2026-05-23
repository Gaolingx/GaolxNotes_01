# 不推荐4bit qlora用于基座首轮sft

### 🧑‍💻 User

bitsandbytes qlora 4bits推荐用于基座的第一轮大规模sft吗（rank 128/256+）？

---

### 🤖 Assistant

**结论先行：不强烈推荐。** 

对于基座模型（Base Model）的**第一轮大规模 SFT**（Supervised Fine-Tuning），除非你受到极其严格的显存（VRAM）限制，否则**不建议**使用 bitsandbytes 4-bit QLoRA，即使你将 LoRA 的 Rank 设置得很高（$r=128$ 或 $256+$）。

第一轮 SFT 决定了模型从“续写机器”向“对话助手”转变的基础底色，对模型的权重更新要求极高。以下是详细的原因分析以及替代建议：

### 为什么不推荐 4-bit QLoRA 用于首轮大规模 SFT？

**1. 4-bit 量化带来的不可逆精度损失**
4-bit 量化（如 NF4）会强制压缩基座模型的原始权重。虽然 QLoRA 证明了在很多场景下它能达到与全量微调（Full SFT）接近的效果，但在面对**第一轮大规模、高质量的数据灌入**时，量化误差（Quantization Error）会限制模型对复杂模式和新知识的吸收能力。

**2. 第一轮 SFT 的特殊性**
基座模型在第一轮 SFT 中需要学习的东西最多：
*   **指令遵循与格式对齐**（Chat Template, System Prompt 响应）。
*   **语气与价值观对齐**。
*   **领域知识注入**（如果是垂直领域基座）。
这种大规模的基础行为重塑，需要模型具有完整的表征能力。4-bit 权重在反向传播计算梯度时，会导致梯度更新的精度下降，影响模型收敛到最优解。

**3. 高 Rank ($r \ge 128$) 的“伪替代”**
增加 LoRA 的 Rank 确实增加了可训练参数量（Trainable Parameters），但这**无法完全弥补基座权重处于 4-bit 状态所带来的信息瓶颈**。
*   根据多项经验测试，当 Rank 超过 $64$ 后，QLoRA 的能力提升存在明显的**边际效应递减**。
*   $r=256$ 的 4-bit QLoRA，其最终效果通常依然**打不过** $r=32$ 或 $64$ 的 16-bit (bf16/fp16) 标准 LoRA，更打不过 bf16 的全参微调。

---

### 推荐的替代方案（按优先级排序）

对于第一轮大规模 SFT，建议优先考虑以下方案：

1.  **全参数微调 (Full Parameter SFT - bf16)**
    *   **适用场景**：算力充足（多张 A100/H100）。
    *   **优势**：效果上限最高，模型对指令的服从度和通用能力保留最好。配合 DeepSpeed Zero-3 或 FSDP 可以有效降低单卡显存。
2.  **16-bit 标准 LoRA (bf16 base weights + bf16 LoRA)**
    *   **适用场景**：算力中等。
    *   **优势**：基座模型保持 16-bit 精度，无量化损失。将 $r$ 设置为 $64$ 或 $128$，并 Target 所有线性层（`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`），效果非常接近全参微调。
3.  **8-bit QLoRA**
    *   **适用场景**：显存略显紧张。
    *   **优势**：Int8 量化的精度损失远小于 4-bit，作为妥协方案比 4-bit 更适合打地基。

---

<details>
<summary><b>🛠️ 展开查看：如果只有条件做 4-bit QLoRA，该如何优化？</b></summary>

如果你受限于硬件（例如只有消费级显卡如 RTX 4090 或单张 A6000），**必须**使用 4-bit QLoRA 进行 SFT，请务必采取以下措施来拉高模型上限：

1.  **覆盖所有线性层 (Target All Linear Layers)**
    不要只对 Attention 层（Q/V）做 LoRA。必须覆盖 MLP 层。
    ```python
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    ```

2.  **调整 Alpha 值**
    LoRA 的缩放因子是 $\frac{\alpha}{r}$。如果你的 $r=128$，通常建议 $\alpha$ 设置为 $r$ 的 1 倍或 2 倍（即 $\alpha=128$ 或 $256$），以保证梯度的有效更新。

3.  **使用双重精度与分页优化器**
    开启 bitsandbytes 的嵌套量化（Double Quantization）和计算精度保持：
    ```python
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # 关键：计算时使用 bf16
    )
    ```
    搭配优化器：`optim="paged_adamw_32bit"` 防止显存 OOM 并保持动量精度。

4.  **引入 NEFTune 噪声**
    在 SFT 阶段为 Embedding 添加微小噪声，可以有效缓解 QLoRA 容易出现的过拟合问题，提高指令遵循的泛化性。
    ```python
    neftune_noise_alpha=5.0
    ```
</details>

<details>
<summary><b>💡 展开查看：什么时候 4-bit QLoRA 是好选择？</b></summary>

虽然不推荐用于**基座的第一轮 SFT**，但在以下场景中 4-bit QLoRA 极其优秀：

*   **第二/第三轮的垂直领域微调**：如果基座模型已经被官方（如 Llama-3-Instruct, Qwen-Chat）很好地做过了 SFT 和 RLHF，你只是想让它适应你的私有数据（比如医疗问答、特定代码库），4-bit QLoRA 性价比极高。
*   **快速概念验证 (PoC)**：在投入大量算力进行全参微调前，用 QLoRA 快速验证数据质量和数据配比。
*   **极度低资源环境**：在单卡 24G 显存上微调 13B/14B 模型，或者在双卡上微调 30B+ 模型，这是唯一可行的方案。
</details>

### 总结

对于**第一轮大规模基座 SFT**，模型的“地基”最重要。**4-bit 带来的精度损失是高 Rank 无法完全弥补的**。如果有条件，请务必选择 **Full SFT (bf16)** 或 **16-bit LoRA (Target all linear)**。如果只能用 QLoRA，请将其视为算力妥协的产物，并对所有线性层注入 LoRA 参数。

---

### 🧑‍💻 User

显存有限情况下，能否使用高rank 标准lora并为所有模块添加adaptor？（target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, modules_to_save: lm_head, embed_tokens）（考虑到我们模型只做过pretrain）

---

### 🤖 Assistant

**结论先行：在显存有限的情况下，你提出的方案（标准 16-bit LoRA + 高 Rank + 全模块 + 训练 Embedding/LM Head）是一个“显存悖论”，极大概率会直接 OOM（Out Of Memory，显存溢出）。**

这个配置的思路在**理论上非常正确**（尤其针对纯 Pretrain 模型的第一轮 SFT，覆盖全模块和输入输出层是最佳实践），但在**显存有限的工程实践中完全不可行**。

以下是具体的显存瓶颈剖析以及优化建议：

### 为什么会 OOM？寻找显存杀手

你的配置中有三个主要的显存增长点，其中最后一个是致命的：

**1. 基础权重占用（16-bit）**
既然是标准 LoRA，基座模型需要以 16-bit（`bfloat16` 或 `float16`）加载。
*   以一个 7B/8B 模型为例，仅加载基座权重就需要约 **$14\text{ GB} \sim 16\text{ GB}$** 的显存。

**2. 全线性层 + 高 Rank 的梯度和优化器状态**
对所有线性层（`q/k/v/o/up/down/gate`）开启 LoRA，且 $r \ge 128$。
*   可训练参数量会暴增到 $200\text{ M} \sim 500\text{ M}$。
*   如果使用标准的 AdamW 优化器（32-bit 状态），这部分参数的梯度和优化器状态会额外占用 **$3\text{ GB} \sim 8\text{ GB}$** 显存。

**3. 致命瓶颈：`modules_to_save: lm_head, embed_tokens`**
**这是显存爆炸的直接原因。** `modules_to_save` 意味着这些层不使用 LoRA 降维，而是**全参数训练**。
*   现代大模型的词表（Vocabulary Size）非常大。例如 Llama-3/Qwen 的词表在 $100,000 \sim 150,000$ 级别，隐藏层维度（Hidden Size）约 $4096$。
*   `embed_tokens` 和 `lm_head` 的参数量之和可能高达 **$1\text{ Billion}$**（10 亿）参数！
*   对这 10 亿参数进行全参训练，需要保存 32-bit 的动量和方差（AdamW），以及计算梯度。仅这两个模块的训练，就会**额外生吞 $12\text{ GB} \sim 16\text{ GB}$ 的显存**。

加起来，在一张 24G 显卡（如 RTX 3090/4090）上，这个配置在开始训练的前几步就会崩溃。

---

### 显存有限时的破局方案

既然模型是纯 Pretrain 的，做第一轮 SFT，我们必须在“效果”和“显存”之间做精明的交易。建议按照以下顺序进行调整：

#### 核心调整 1：放弃全参训练词表，改用 Embedding LoRA（或只训练新增 Token）
对于纯 Pretrain 模型，如果没有引入大量新的专业词汇，**不需要全参数训练 `embed_tokens` 和 `lm_head`**。
*   **如果为了对齐格式加入了 Chat Template 特殊 Token**（如 `<|im_start|>`）：
    *   绝大多数框架（如 Llama-Factory, axolotl）支持**只训练新增 Token 的 Embedding**，或者系统会自动将新 Token 的梯度冻结解除，而不必把整个几万词表的参数放进优化器。
*   **如果不加新 Token**：
    *   直接去掉 `modules_to_save` 这一项。靠全线性层的 LoRA 完全能够完成 Chat 对齐。

#### 核心调整 2：降低 Rank，保留全模块
对于第一轮 SFT，**覆盖所有层（Target All Modules）的优先级 远远高于 高 Rank**。
*   与其使用 $r=128$ 但只能微调 Attention 层，不如使用 **$r=32$ 或 $r=64$ 并覆盖所有 7 个线性层**。
*   经验表明，$r=64$ 配合全层 LoRA，已经能保留基座模型 $95\%$ 以上的全参微调效果，且显存占用大幅下降。

#### 核心调整 3：极限压榨显存的工程 Trick
必须开启以下配置：
1.  **Gradient Checkpointing（梯度检查点）**：牺牲约 $20\%$ 的计算速度，换取极大幅度的显存节省。这是必选项。
2.  **8-bit 优化器**：使用 `optim="adamw_8bit"`，可以将优化器状态占用的显存减少约 $75\%$，对最终模型精度几乎无影响。

---

<details>
<summary><b>🛠️ 展开查看：推荐的折中 Config 代码示例</b></summary>

如果你只有单卡 24G 显存，微调 7B 级别模型的第一轮 SFT，建议的 PEFT/Training 配置如下：

```python
# PEFT Config (LoRA)
peft_config = LoraConfig(
    r=64,                # 降级 Rank，64 已经足够打底
    lora_alpha=128,      # alpha 通常设为 r 的 2 倍
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # 移除 modules_to_save=["lm_head", "embed_tokens"]
)

# Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # 累计梯度，等效 batch size = 8
    gradient_checkpointing=True,   # 必开！
    optim="adamw_8bit",            # 显著省显存的优化器
    bf16=True,                     # 必须用 bf16
    learning_rate=2e-4,            # 全模块 LoRA 建议学习率
    # ... 其他参数
)
```
</details>

<details>
<summary><b>💡 展开查看：如果一定要训练 LM Head 怎么办？</b></summary>

如果你有强烈的理由必须微调 `lm_head`（例如从头教模型说一种全新语言，导致输出分布完全改变），并且显存有限，你可以：

1.  **使用 QLoRA 作为底层**：回到你上一个问题，将基座加载为 4-bit NF4。这时候即便你把 `lm_head` 加入 `modules_to_save`，由于基座权重只占极小显存，你或许能在一张 24G 显卡上跑通。但这会付出 4-bit 的精度代价。
2.  **多卡分布式并行 (ZeRO-2/3 或 FSDP)**：如果你有 2-4 张小显存卡（如 $4 \times 3090/4090$），使用 DeepSpeed ZeRO-3 可以将优化器状态和梯度切片分摊到多张卡上，从而完美跑通你最初提出的“16-bit Base + 高 Rank + LM Head”方案。
</details>

**总结建议**：去掉 `modules_to_save: lm_head, embed_tokens`，将 Rank 适当降到 $64$，保留 16-bit 基座和全线性层注入，开启梯度检查点。这是在有限显存下单卡做纯 Pretrain 模型首轮 SFT 最具性价比的方案。