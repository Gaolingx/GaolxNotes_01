# VLM与纯文本SFT区别及trl训练方法

### 🧑‍💻 User

vlm的sft微调和纯文本有什么区别，trl中怎么训练？

---

### 🤖 Assistant

视觉语言模型（VLM）的监督微调（SFT）与纯文本大语言模型（LLM）的SFT在底层逻辑上相似（都是基于自回归的交叉熵损失），但在**模型架构、数据处理、微调策略和算力消耗**上存在显著差异。

以下是详细对比，以及如何使用 Hugging Face 的 `trl` 库进行 VLM SFT 的实战指南。

---

### 一、 VLM SFT 与纯文本 SFT 的核心区别

#### 1. 输入模态与数据结构
*   **纯文本：** 数据是一维的 Token 序列。格式通常是 `Prompt + Response`。
*   **VLM：** 数据包含图像（或视频）和文本。图像需要经过视觉编码器（Vision Encoder）处理成视觉 Token，并插入到文本 Token 序列中（通常使用 `<image>` 等占位符）。
*   **数学表示区别：**
    纯文本 SFT 的目标是最大化似然估计：
    $$ \mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid X_{text}, y_{<t}) $$
    VLM SFT 的目标函数中加入了图像特征 $V$：
    $$ \mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid V, X_{text}, y_{<t}) $$

#### 2. 模型架构与可训练参数
*   **纯文本：** 只有 LLM 主干网络（Transformer）。微调时要么全参数更新，要么在注意力层和全连接层加 LoRA。
*   **VLM：** 包含三个部分：**视觉编码器**（如 CLIP、SigLIP）、**对齐投影层**（Projector/MLP）和 **LLM 主干**。
*   **微调策略区别：** 在 VLM SFT 中，通常**冻结（Freeze）视觉编码器**，只训练**投影层（全参数）** 和 **LLM 主干（全参数或 LoRA）**。如果图像域与预训练差异极大（如医疗影像），才会解冻视觉编码器的最后几层。

#### 3. 计算开销与显存 (VRAM)
*   图像会被切分成 Patch。高分辨率图像会产生大量的 Token（例如一张 $336 \times 336$ 的图像在 CLIP 中会产生 576 个 Token；在 LLaVA-NeXT 中可能会产生近 3000 个 Token）。
*   这导致 VLM 的上下文长度急剧增加，训练时的 KV Cache 和激活显存占用远高于同级别的纯文本模型。

---

### 二、 如何在 `trl` 中训练 VLM

Hugging Face 的 `trl` 库（特别是 `SFTTrainer`）近期已经原生支持了 VLM 的微调。关键在于**正确构造数据集格式**并**使用多模态 Data Collator**。

以下是使用 `trl` 对 `LLaVA-1.5` 进行 LoRA SFT 的完整代码示例。

#### 1. 环境准备
确保安装了最新版本的库：
```bash
pip install transformers trl peft datasets accelerate bitsandbytes
```

#### 2. 训练代码实战

```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. 加载模型和处理器
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)

# 为了节省显存，可以加载 4bit 或 8bit 模型，或者使用 bf16
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. 配置 LoRA (只微调 LLM 部分和投影层)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "multi_modal_projector"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 准备数据集
# VLM 数据集通常需要包含图像对象和对话列表
# 数据格式要求类似 Hugging Face 的 standard chat template
def format_dataset(example):
    # 构建符合 processor.apply_chat_template 的格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["prompt"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": example["response"]}
            ]
        }
    ]
    return {"messages": messages, "images": [example["image"]]}

# 假设你的数据集有 'image', 'prompt', 'response' 三列
dataset = load_dataset("your_dataset_path", split="train")
dataset = dataset.map(format_dataset, remove_columns=["prompt", "response", "image"])

# 4. 自定义 Data Collator (处理图像和文本合并)
def collate_fn(examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"][0] for example in examples]
  
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  
    # 构建 labels用于计算 Loss (忽略 padding 的 token)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch

# 5. 配置 TrainingArguments
training_args = TrainingArguments(
    output_dir="./vlm-sft-results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    max_steps=500,
    save_steps=100,
    gradient_checkpointing=True, # 必须开启，否则显存必爆
)

# 6. 初始化 SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
    dataset_text_field="messages", # trl v0.8.0+ 支持自动解析 messages
    tokenizer=processor.tokenizer,
)

# 开始训练
trainer.train()

# 保存 LoRA 权重
trainer.model.save_pretrained("./vlm-lora-final")
```

---

<details>
<summary><b>💡 展开查看：VLM 训练的进阶技巧与踩坑指南</b></summary>

1. **Loss 掩码（Label Masking）：**
   在标准的 SFT 中，我们只计算 Assistant 回复部分的 Loss。在 `collate_fn` 中，你需要找到用户 Prompt 和 `<image>` Token 的位置，并将它们在 `labels` 中的值设置为 `-100`。上面的基础代码简化了这一步，但在生产环境中建议使用 `DataCollatorForCompletionOnlyLM` 的变体或手动 masking。

2. **显存优化策略 (VRAM Optimization)：**
   * VLM 对显存极其敏感。强烈建议使用 **Flash Attention 2**。
   * 确保开启了梯度检查点：`model.gradient_checkpointing_enable()`。
   * 如果仍然 OOM，可以减小图像分辨率（如果你有自定义的 processor 设置），或者使用 `bitsandbytes` 进行 QLoRA (4-bit) 训练。

3. **投影层（Projector）的处理：**
   如果你在做一个全新的任务，通常建议**全参数解冻投影层**（Multi-modal Projector），而对 LLM 使用 LoRA。在上面的 `target_modules` 中加入投影层的名字（如 LLaVA 中的 `multi_modal_projector`）可以实现这一点。

4. **纯文本数据的混合：**
   在微调 VLM 时，为了防止模型发生**灾难性遗忘**（Catastrophic Forgetting，即学会了看图但丧失了原本的纯文本对话能力），通常需要在 SFT 数据集中混入 $10\% \sim 20\%$ 的纯文本指令微调数据。

</details>

---

### 🧑‍💻 User

vision encoder需要微调吗？

---

### 🤖 Assistant

**通常情况下不需要（且不建议）微调视觉编码器，但在特定场景下（如领域跨度极大的任务）必须微调。**

在 VLM 的训练中，视觉编码器（如 CLIP、SigLIP）通常是在海量图文对上预训练好的，已经具备了极强的泛化视觉特征提取能力。是否微调它，取决于你的**目标任务域**与**预训练数据域**的差异。

以下是具体的决策指南：

### 一、 默认情况：冻结（Freeze）视觉编码器
在 $90\%$ 的 SFT 场景下，我们都会选择**冻结视觉编码器**，只训练投影层（Projector）和 LLM（全参或 LoRA）。

*   **适用场景：** 通用视觉问答（VQA）、通用图像描述、日常物体识别等。
*   **为什么不微调？**
    1.  **防止灾难性遗忘：** 视觉编码器已经学会了极其丰富的通用特征。如果在少量 SFT 数据上微调，很容易让它“遗忘”原有的泛化能力（即发生过拟合）。
    2.  **节省算力：** 视觉编码器（如 ViT-L）参数量通常在 300M 左右，高分辨率下产生的激活显存（Activation VRAM）极大。冻结它可以大幅降低显存占用和训练时间。

### 二、 必须微调的场景（Unfreeze）
如果你的任务图像与自然图像（猫、狗、风景、人物）差异极大，预训练的视觉编码器根本“看不懂”这些图，那么仅仅靠微调后面的 LLM 是救不回来的，这时候必须微调视觉编码器。

*   **典型适用场景：**
    1.  **医疗影像诊断：** X 光片、CT、MRI 扫描图（CLIP 预训练时很少见过这些特征）。
    2.  **遥感与卫星图像：** 俯视视角的地图、地理特征提取。
    3.  **细粒度文档理解（OCR）：** 密集的文字扫描件、复杂的财务报表。
    4.  **工业缺陷检测：** 电路板划痕、零部件微小瑕疵。

### 三、 如果要微调，应该怎么做？（最佳实践）

如果你决定要微调视觉编码器，**千万不要直接全参数解冻并且使用统一的学习率**，这极易导致模型崩溃（Mode Collapse）。推荐以下三种策略：

#### 策略 1：只解冻最后几层（最常用）
视觉特征是分层的：浅层提取边缘和纹理，深层提取高级语义。对于新领域，通常底层的通用纹理特征是通用的，只需要改变深层的高级语义特征。
*   **做法：** 冻结视觉编码器的前 80% 的层，只解冻最后 2 到 4 个 Transformer Block。

#### 策略 2：对视觉编码器使用 LoRA
如果显存有限，或者想保留大部分预训练权重，可以在视觉编码器的注意力层（`q_proj`, `v_proj`）也加上 LoRA。
*   **做法：** 在 `peft` 配置中，将视觉编码器的模块名加入 `target_modules` 中。

#### 策略 3：使用差异化学习率（Differential Learning Rates）
即使全参解冻视觉编码器，也要使用比 LLM 和 Projector 小得多的学习率。
*   **经验值：** 如果 LLM/Projector 的学习率是 $2 \times 10^{-5}$，那么视觉编码器的学习率应该设置在 $1 \times 10^{-6}$ 到 $5 \times 10^{-6}$ 之间。

---

<details>
<summary><b>💻 展开查看：在代码中如何解冻/应用 LoRA 到视觉编码器</b></summary>

**1. 解冻视觉编码器的最后几层（PyTorch 原生写法）：**
```python
# 假设 model.model.vision_tower 是视觉编码器
# 1. 首先全部冻结
for param in model.model.vision_tower.parameters():
    param.requires_grad = False

# 2. 获取 vision encoder 的所有层
vision_layers = model.model.vision_tower.vision_model.encoder.layers

# 3. 只解冻最后 2 层
num_layers_to_unfreeze = 2
for layer in vision_layers[-num_layers_to_unfreeze:]:
    for param in layer.parameters():
        param.requires_grad = True
```

**2. 在 PEFT (LoRA) 中同时覆盖 LLM 和 Vision Encoder：**
在使用 `trl` 和 `peft` 时，只要把 Vision Encoder 中的注意力矩阵名字加到 `target_modules` 里即可。
```python
from peft import LoraConfig

# 注意：不同模型的模块命名不同，这里以 LLaVA 架构为例
# LLaVA 中 LLM 投影层通常叫 q_proj, v_proj
# CLIP Vision Encoder 的层通常叫 q_proj, v_proj 或者在 vision_model 层级下
lora_config = LoraConfig(
    r=16,
    target_modules=[
        # LLM 的模块
        "q_proj", "k_proj", "v_proj", "o_proj", 
        # Projector
        "multi_modal_projector",
        # 匹配 Vision Encoder 中的线性层 (正则匹配或写全名)
        "vision_model.encoder.layers.*.self_attn.q_proj",
        "vision_model.encoder.layers.*.self_attn.v_proj"
    ],
    task_type="CAUSAL_LM"
)
```

</details>

**总结法则：**
如果你做的是通用图文对话 $\rightarrow$ **绝对不微调（冻结）**。
如果你做的是医疗/遥感/复杂图表，且发现 Loss 降不下去，模型一本正经胡说八道 $\rightarrow$ **尝试微调（只解冻最后两层 或 加 LoRA），并降低学习率**。