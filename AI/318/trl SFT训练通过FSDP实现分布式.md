# trl SFT训练通过FSDP实现分布式

### 🧑‍💻 User

trl 框架中 SFT 训练如何通过 FSDP 实现分布式训练？

---

### 🤖 Assistant

在 `trl` 框架中，`SFTTrainer` 直接继承自 Hugging Face 的 `Trainer`。因此，通过 FSDP（Fully Sharded Data Parallel）实现分布式训练的底层逻辑完全依赖于 `transformers` 和 `accelerate` 库的 FSDP 集成。

要实现 FSDP 分布式 SFT 训练，主要有两种配置方式：**使用 Accelerate 配置文件（推荐）**或**在代码中通过 `TrainingArguments` 配置**。

以下是完整的实现指南。

### 方法一：使用 Accelerate 配置文件（推荐）

这是最标准且侵入性最小的方法。你不需要在 Python 代码中写死 FSDP 的参数。

**1. 生成配置文件**
在终端中运行以下命令：
```bash
accelerate config
```
按照提示进行选择：
*   Which compute environment are you running? **This machine**
*   Which type of machine are you using? **multi-GPU**
*   How many different machines will you use? **1** (如果是单机多卡)
*   Do you wish to optimize your script with torch dynamo? **No**
*   Do you want to use DeepSpeed? **No**
*   Do you want to use FullyShardedDataParallel? **Yes**
*   What should be your sharding strategy? **FULL_SHARD** (切分参数、梯度和优化器状态)
*   Do you want to use offload? **No** (除非你的显存极小，否则不要选 Yes，会严重拖慢速度)
*   What should be your auto wrap policy? **TRANSFORMER_BASED_WRAP** (针对 Transformer 模型优化)
*   Do you want to use a custom layer class? **Yes** -> 输入你的模型的 Transformer 层名称 (例如 LLaMA 是 `LlamaDecoderLayer`，Qwen 是 `Qwen2DecoderLayer`)。

**2. 编写 SFT 代码 (`sft_train.py`)**
在代码中，你**不需要**写任何 FSDP 相关的逻辑，只需正常实例化 `SFTTrainer`。

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

model_id = "Qwen/Qwen2-7B"

# 注意：使用 FSDP 时，不要在这里使用 device_map="auto"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16
)

dataset = load_dataset("imdb", split="train[:1000]")

training_args = TrainingArguments(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    logging_steps=10,
    max_steps=100,
    bf16=True, # 强烈建议使用 bfloat16
    save_strategy="steps",
    save_steps=50,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()
```

**3. 启动训练**
使用 `accelerate launch` 启动脚本，它会自动读取刚才生成的配置文件并启用 FSDP：
```bash
accelerate launch sft_train.py
```

---

### 方法二：在代码中通过 `TrainingArguments` 配置

如果你希望将配置固定在代码中，可以直接在 `TrainingArguments` 中传入 `fsdp` 和 `fsdp_config` 参数。

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# ... 前面的模型和数据集加载逻辑同上 ...

# 定义 FSDP 参数
fsdp_config = {
    "fsdp_transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"], # 必须替换为你模型的 Layer 名字
    "fsdp_backward_prefetch": "backward_pre",
    "fsdp_forward_prefetch": False,
    "limit_all_gathers": True,
    "use_orig_params": True # 如果结合 LoRA 训练，通常需要设为 True
}

training_args = TrainingArguments(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    bf16=True,
    # FSDP 核心配置
    fsdp="full_shard auto_wrap",
    fsdp_config=fsdp_config,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()
```

使用 `torchrun` 启动（假设有 4 张显卡）：
```bash
torchrun --nproc_per_node=4 sft_train.py
```

---

<details>
<summary><b>🛠️ 进阶配置与避坑指南 (点击展开)</b></summary>

#### 1. FSDP 与 LoRA / QLoRA 结合
默认情况下，FSDP 用于**全量参数微调 (Full-Parameter Tuning)** 非常高效。如果你要在 FSDP 中使用 PEFT (LoRA)：
*   在 `fsdp_config` 中必须设置 `"use_orig_params": True`。
*   **关于 QLoRA (4-bit)**：传统的 FSDP 不支持与 `bitsandbytes` 4-bit 量化共存，因为 FSDP 无法切分量化后的权重。目前社区的解决方案是使用 **FSDP + QLoRA (HuggingFace 最新支持)**，你需要确保 `transformers >= 4.38.0` 并且 `peft >= 0.8.0`，或者使用针对显存优化的 DeepSpeed Zero-3 代替。

#### 2. Auto Wrap Policy (`transformer_layer_cls_to_wrap`)
这是 FSDP 避免 OOM（内存溢出）的关键。如果不设置自动包装策略，FSDP 可能会把整个模型当作一个大块来切分，导致在 Forward/Backward 时需要把整个模型重构到单个 GPU 上，从而 OOM。
*   **LLaMA 系列**: `LlamaDecoderLayer`
*   **Qwen 系列**: `Qwen2DecoderLayer`
*   **Mistral 系列**: `MistralDecoderLayer`

#### 3. 数学显存估算
使用 FSDP `FULL_SHARD` 策略时，每张显卡上的显存占用大致为：
$$ \text{Memory per GPU} \approx \frac{M_{params} + M_{grads} + M_{optim}}{N_{gpus}} + M_{activations} $$
其中 $N_{gpus}$ 是 GPU 数量。这意味着模型参数、梯度和优化器状态都被切分了，但前向传播时的激活值 $M_{activations}$ 没有被切分（可以通过开启 Gradient Checkpointing 来降低激活值显存）。

#### 4. 模型保存注意事项
使用 FSDP 训练结束后，模型权重是分布在不同 GPU 上的。Hugging Face `Trainer` 默认会在保存时将权重聚合（Gather）到主节点（Rank 0）然后保存为正常的 Hugging Face 格式。请确保主节点的 CPU 内存足够放下完整的模型权重，否则保存时会发生 CPU OOM。
</details>