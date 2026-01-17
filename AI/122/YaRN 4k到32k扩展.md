# YaRN 4k到32k扩展

### 🧑‍💻 User

如何用yarn rope scaling将4k上下文扩展到32k？

---

### 🤖 Assistant

使用 YaRN (Yet another RoPE extension) 将大语言模型（如 LLaMA 2, LLaMA 3, Mistral 等使用 RoPE 的模型）的上下文窗口从 4k 扩展到 32k，主要涉及**计算缩放因子**、**修改模型配置**以及（通常建议的）**微调**。

以下是具体的实施步骤和技术细节：

### 1. 核心概念与计算

YaRN 是一种改进的旋转位置编码（RoPE）插值方法。与线性插值不同，它通过混合“NTK-aware”插值和直接外推，并针对不同频率（维度）调整注意力熵，从而在扩展上下文的同时保持短文本的性能。

要将上下文从原始的 $L_{orig} = 4096$ (4k) 扩展到目标 $L_{target} = 32768$ (32k)，我们需要计算缩放因子 $s$：

$$s = \frac{L_{target}}{L_{orig}} = \frac{32768}{4096} = 8$$

### 2. 在 Hugging Face Transformers 中实施

Hugging Face `transformers` 库原生支持 YaRN。你只需要在加载模型时修改配置，或者直接修改模型的 `config.json` 文件。

#### 方法 A：修改 `config.json` (永久生效)

在模型文件夹中的 `config.json` 文件里，找到或添加 `rope_scaling` 字段。

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "max_position_embeddings": 32768, 
  "rope_scaling": {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 4096
  },
  ...
}
```

*   **`max_position_embeddings`**: 设置为你想要的目标长度 **32768**。
*   **`type`**: 设置为 `"yarn"`。
*   **`factor`**: 设置为 **8.0** ($32k / 4k$)。
*   **`original_max_position_embeddings`**: 原始模型的长度 **4096**（这对于 YaRN 准确计算高频和低频部分至关重要）。

#### 方法 B：代码中动态加载 (推理/微调时)

如果你不想修改原始文件，可以在 Python 代码中动态传入配置：

```python
from transformers import AutoConfig, AutoModelForCausalLM

model_id = "your-base-model-4k"

# 加载原始配置
config = AutoConfig.from_pretrained(model_id)

# 修改 RoPE 配置以启用 YaRN
config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,  # 32k / 4k
    "original_max_position_embeddings": 4096
}
config.max_position_embeddings = 32768

# 使用修改后的配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    trust_remote_code=True
)

# 现在模型可以处理更长的上下文，但未微调前效果可能有限
```

### 3. YaRN 的高级超参数 (针对微调)

如果你是在自己实现 YaRN 算法或者使用像 `Axolotl` 或 `LLaMA-Factory` 这样的框架进行微调，你可能需要配置更具体的参数（通常由论文推荐）：

*   **$\beta_{fast}$ (beta_fast)**: 通常设为 **32**。这是高频部分不需要插值的阈值。
*   **$\beta_{slow}$ (beta_slow)**: 通常设为 **1**。这是低频部分完全插值的阈值。
*   **$mscale$**: 这是一个用于修正注意力熵的系数。
    *   对于 LLaMA 2 模型，YaRN 论文建议设置 $mscale = 0.1 \cdot \ln(s) + 1$。
    *   当 $s=8$ 时，近似值为 $0.1 \cdot \ln(8) + 1 \approx 1.2079$。但在 HF 的实现中，这通常是自动处理的，或者默认为 1。

### 4. 关于微调 (Fine-tuning) 的必要性

虽然 YaRN 声称在“免微调”（Zero-shot）扩展方面比线性插值好，但要从 4k 直接跳到 32k 并保持高质量输出，**必须进行微调**。

**微调建议：**
1.  **数据集**: 使用长文本数据集（如 PG19, ArXiv, 或专门构建的长指令集）。
2.  **步数**: 不需要全量训练。通常只需要对原始预训练步数的 **0.1% 到 0.5%** 进行微调即可让模型适应新的长度。
3.  **显存优化**: 32k 上下文训练极其消耗显存。
    *   使用 **Flash Attention 2** (必须)。
    *   使用 **Gradient Checkpointing**。
    *   使用 **Deepspeed Zero3** 或 **FSDP**。
    *   考虑使用 LoRA 进行参数高效微调。

### 5. 常见问题排查

*   **PPL (困惑度) 激增**: 检查 `original_max_position_embeddings` 是否填写正确。如果没有指定原始长度，YaRN 可能会错误地计算频率，导致模型崩溃。
*   **Flash Attention 报错**: 确保你的 Flash Attention 版本支持这种缩放配置，并且在调用模型时 `use_flash_attention_2=True`。
*   **推理时输出乱码**: 确保推理时的 `rope_scaling` 配置与微调时的配置完全一致。

### 总结

将 4k 扩展到 32k 的核心在于：将 `config.json` 中的 `rope_scaling` 类型设为 `"yarn"`，`factor` 设为 `8.0`，并确保 `original_max_position_embeddings` 设为 `4096`。为了获得可用的模型，请务必在长文本数据上进行少量的继续预训练（Continual Pre-training）。

---

### 🧑‍💻 User

但是如果模型本身没有进行过后训练长上下文微调 直接在config.json里面添加yarn参数 外推还是会掉点，针对original model的ft要怎么做？

---

### 🤖 Assistant

你说得很对。**直接在 `config.json` 里修改参数而不进行微调（Zero-shot），对于从 4k 扩展到 32k（8倍扩展）这种跨度，效果通常很差。** 模型会出现严重的“困惑度暴涨”（PPL explosion）或者在该输出长文本时胡言乱语。

YaRN 的作者在论文中也明确指出，虽然 YaRN 的外推能力比线性插值好，但要达到目标长度的良好性能，**必须进行简短的继续预训练（Continual Pre-training / Fine-tuning）**。

针对 Original Model（基座模型）进行长上下文微调（Long-Context Fine-Tuning），以下是标准且经过验证的流程：

### 1. 数据准备 (Data Preparation)

你不能只用简短的对话数据来微调长上下文。你需要真正的长文本数据。

*   **数据集选择**：
    *   **PG19**: 古腾堡计划书籍，天然的长文本。
    *   **SlimPajama / RedPajama**: 其中的 `book` 或 `arxiv` 子集。
    *   **Pile-CC**: CommonCrawl 的清洗版本，提取长文章。
    *   **LongAlpaca**: 专门构建的长指令数据集（如果你是做指令微调）。
*   **混合策略 (关键)**：为了防止模型“遗忘”短上下文的能力（性能退化），强烈建议在长文本数据中**混合一部分原始长度（4k以下）的高质量数据**。比例可以是 80% 长文本 + 20% 短文本。

### 2. 训练框架与配置

目前最方便进行此类微调的工具是 **LLaMA-Factory** 或 **Axolotl**。这里以通用的逻辑（适用于 HF Trainer）说明配置重点。

#### 核心配置清单

1.  **RoPE 配置 (必须与推理时一致)**:
    在训练开始前，必须载入开启了 YaRN 的模型配置。
    ```python
    # 伪代码示例
    config.rope_scaling = {
        "type": "yarn",
        "factor": 8.0, 
        "original_max_position_embeddings": 4096
    }
    config.max_position_embeddings = 32768
    ```

2.  **序列长度 (Sequence Length)**:
    设置 `max_seq_length` (或 `block_size`) 为 **32768**。

3.  **数据打包 (Packing)**:
    **一定要开启 `packing=True`** (在 LLaMA-Factory 中叫 `packing`，Axolotl 中叫 `sample_packing`)。
    *   这意味着如果一条数据只有 10k token，训练器会自动把下一条数据拼在后面，直到填满 32k。这不仅利用了显存，更重要的是**强迫模型学习长距离的 attention 依赖**。

4.  **学习率 (Learning Rate)**:
    不要使用原始预训练的学习率。应使用较小的 LR。
    *   建议范围：$1 \times 10^{-5}$ 到 $5 \times 10^{-5}$。
    *   LR Scheduler：通常使用 `cosine` 或 `constant_with_warmup`。

5.  **训练步数 (Steps)**:
    YaRN 的收敛速度非常快。你不需要跑完整个 epoch。
    *   通常只需要 **200 到 1000 个 step** (取决于 batch size) 就可以让模型适应新的 RoPE 频率。
    *   观察 Loss 曲线，当 Loss 稳定下降并趋于平缓时即可停止。

### 3. 显存优化 (Memory Optimization)

训练 32k 上下文对显存要求极高。如果直接 Full Fine-tune (全量微调)，通常需要 8张 A100 80G 或更多。对于资源有限的情况，方案如下：

#### 方案 A：LoRA (推荐资源受限者)
如果显存不够全量微调，可以使用 LoRA，但在长上下文扩展场景下有特殊技巧：
*   **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (即 `all-linear`)。
*   **Rank**: 建议稍大一点，如 `r=64` 或 `r=128`。
*   **关键点**: 虽然不是必须，但在大幅度扩展上下文时，将 `embed_tokens` 和 `lm_head` 加入 `modules_to_save` (即参与训练) 有助于模型适应新的位置编码分布，但这会显著增加显存。**通常仅 LoRA attention 层配合 YaRN 就足够了。**

#### 方案 B：Gradient Checkpointing (必须开启)
必须开启梯度检查点 (`gradient_checkpointing=True`)，这能用计算换显存，将显存占用降低约 50%-70%。

#### 方案 C：Flash Attention 2 (必须开启)
在 32k 长度下，没有 Flash Attention 2 的训练速度是不可接受的。

### 4. 实战：使用 LLaMA-Factory 的配置示例

如果你使用 LLaMA-Factory，可以参考以下 `yaml` 配置思路进行微调：

```yaml
### model
model_name_or_path: /path/to/llama-2-7b-hf

### method
stage: pt           # 预训练模式 (Continual Pre-training)
do_train: true
finetuning_type: lora # 或者 full
lora_target: all

### dataset
dataset: wiki_demo  # 替换为你的长文本数据集
cutoff_len: 32768   # 关键：切分长度设为 32k
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/llama-2-7b/yarn-32k
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8 # 累积梯度以模拟大 Batch
learning_rate: 2.0e-5
num_train_epochs: 1.0
max_steps: 500       # 只需要少量步数
lr_scheduler_type: cosine
warmup_steps: 20

### rope scaling (关键部分)
# LLaMA-Factory 可能会自动识别 config，但建议显式指定或修改 config.json
# 如果通过参数传入，这取决于框架版本，通常修改 config.json 最稳妥。

### runtime
fp16: true
flash_attn: fa2      # 开启 Flash Attention 2
gradient_checkpointing: true
```

### 5. 验证与测试

微调完成后，如何验证有没有成功？
1.  **PPL Test**: 在长文本测试集上计算困惑度（Perplexity）。在 32k 长度内，PPL 应该保持平稳，不会随长度增加而剧烈上升。
2.  **大海捞针 (Passkey Retrieval)**: 这是一个经典的测试。将一个随机 Passkey 藏在 30k 长度的文本中间，问模型 Passkey 是什么。经过 YaRN 微调的模型应该能达到接近 100% 的准确率。

### 总结

针对 Original Model 的操作路径：
1.  修改 `config.json` 开启 YaRN (`factor=8`)。
2.  准备长文本数据集（如书籍）。
3.  开启 **Packing** 和 **Flash Attention 2**。
4.  进行 **200-500 步** 的继续预训练（LR 约为 $2e-5$）。
5.  这样得到的模型就是真正的 32k 长上下文模型。