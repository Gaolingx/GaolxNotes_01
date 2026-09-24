# 16GB显存QLoRA

## 🧑‍💻 User

16g显存怎么训练3.7b参数的模型？

---
## 🤖 Assistant

# 16GB 显存训练 3.7B 模型

**核心结论先讲:16GB 显存几乎不可能做「全参数微调」,但有两条成熟可行路线——首推 QLoRA(4-bit 量化 + LoRA),其次是 LoRA(半精度)。** 全参训练在单卡 16GB 上基本无解(除非用 ZeRO-3 + CPU/NVMe offload,但速度会慢到不实用)。

---

## 1. 先算清楚显存账

3.7B 参数在不同精度下**光权重**就占:

| 精度 | 权重显存 | 说明 |
|------|---------|------|
| FP32 | 14.8 GB | 已占满整个显存 |
| BF16/FP16 | 7.4 GB | 还要再放梯度、优化器 |
| INT8 | 3.7 GB | |
| **INT4** | **≈1.85 GB** | QLoRA 的基础 |

<details>
<summary><b>全参微调为什么 16GB 放不下(展开看计算)</b></summary>

以混合精度 + AdamW 为例,除了权重和梯度,优化器还要为每个参数存 FP32 主权重、一阶动量、二阶动量:

$$
\text{Total} \approx \underbrace{7.4}_{\text{权重 bf16}} + \underbrace{7.4}_{\text{梯度 bf16}} + \underbrace{14.8 + 14.8 + 14.8}_{\text{FP32 主权重 + Adam } m + v} \approx 59\ \text{GB}
$$

即使换成 8-bit Adam,优化器部分仍需 $\approx 30$ GB。**这还没算激活值。** 所以全参微调至少要 4×24GB 或 8×A100,单卡 16GB 无解。

</details>

---

## 2. 三条路线对比

| 方案 | 显存占用 | 可行性(16GB) | 训练质量 | 速度 |
|------|---------|--------------|---------|------|
| **QLoRA(4-bit 基座 + LoRA)** | ~6–11 GB | ✅ 轻松 | 接近全参,95%+ | 快 |
| **LoRA(bf16 基座)** | ~12–15 GB | ⚠️ 勉强(需小 batch/短序列) | 好 | 中 |
| 全参 + ZeRO-3 offload | >60GB(靠 CPU 顶) | ❌ 极慢,不推荐 | 最好 | 极慢 |

---

## 3. 推荐方案:QLoRA 具体配置

要点:
- 基座 **4-bit NF4 量化**(double quant 再省约 0.4GB)
- 只训练 **LoRA 适配器**,优化器状态只覆盖适配器(极小)
- 开启 **gradient checkpointing** 牺牲约 30% 速度换大量激活显存
- 用 **paged 8-bit AdamW**(bitsandbytes),峰值显存不满时自动换页
- Flash-Attention 2 + 序列 packing

### 3.1 关键超参建议

| 参数 | 推荐值 | 备注 |
|------|--------|------|
| `r`(LoRA rank) | 16–64 | 越大越强也越占显存,16GB 建议 32 以内 |
| `lora_alpha` | 2×r | |
| `target_modules` | q,k,v,o + gate,up,down | 覆盖全线性层效果更好 |
| `lora_dropout` | 0.05 | |
| `max_seq_len` | 1024–2048 | 序列长度对激活显存影响最大 |
| `per_device_batch_size` | 1–4 | 配合梯度累积 |
| `grad_accum_steps` | 8–16 | 凑出等效大 batch |
| 优化器 | `paged_adamw_8bit` | |
| 精度 | bf16 | |

### 3.2 代码(transformers + peft + trl)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1) 4-bit 量化配置
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,   # 再省约 0.4GB
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "your-3.7B-model",
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained("your-3.7B-model")

# 2) 为 k-bit 训练做准备(会开 gradient checkpointing 等)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 3) LoRA 配置
peft_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()   # 通常 <1%

# 4) 训练参数
args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=3,
    bf16=True,
    optim="paged_adamw_8bit",       # 关键
    gradient_checkpointing=True,    # 关键
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=dataset,
    tokenizer=tok,
    max_seq_length=2048,
    packing=True,                   # 提升吞吐
)
trainer.train()
model.save_pretrained("./lora_out")   # 只存适配器,几十 MB
```

<details>
<summary><b>不想写代码?用现成框架(点开看)</b></summary>

- **LLaMA-Factory**:一条 yaml 就跑,内置 QLoRA,适合快速上手
- **Unsloth**:对 16GB 尤其友好,宣称省 70% 显存、快 2 倍
- **Axolotl**:配置式,社区活跃

LLaMA-Factory 示例 yaml:
```yaml
model_name_or_path: your-3.7B-model
stage: sft
finetuning_type: lora
lora_rank: 32
lora_target: all
quantization_bit: 4          # 开启 QLoRA
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 2.0e-4
bf16: true
gradient_checkpointing: true
cutoff_len: 2048
```
运行:`llamafactory-cli train your_config.yaml`

</details>

---

## 4. 显存不够时的调优顺序

按「性价比」依次尝试:

1. **减小 `max_seq_len`**(激活显存对它近乎平方敏感,—最有效)
2. **`per_device_batch_size=1` + 增大梯度累积**
3. **开启/确认 gradient checkpointing**
4. **降 LoRA rank**(32 → 16)
5. **换 8-bit / paged 优化器**
6. **Flash-Attention 2**(省激活显存又提速)
7. 用 **Unsloth** 替换原生实现

> 经验值:上述 QLoRA 配置在 3.7B、seq 2048、batch 2 下,峰值通常在 **8–11 GB**,16GB 绰绰有余。

---

## 5. 什么时候该考虑全参微调?

只有当 QLoRA 效果明显不足(需要模型学全新知识、大幅改变分布)时,才考虑:
- 多卡 **FSDP/DeepSpeed ZeRO-3 + CPU offload**,需要 **≥64GB 系统内存 + NVMe** 做参数卸载
- 或用云 GPU(A100×4 / 4090×8)

单卡 16GB 环境,老老实实用 QLoRA 是性价比最高的答案。

---

需要的话,告诉我你的**具体模型**(如 Phi-3-mini / Qwen2.5-3B)、**数据集大小**和**框架偏好**,我可以给一份可直接跑的完整配置。
