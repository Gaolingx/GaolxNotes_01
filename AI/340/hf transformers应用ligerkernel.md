# hf transformers应用ligerkernel

### 🧑‍💻 User

hf transformers如何apply ligerkernel

---

### 🤖 Assistant

在 Hugging Face `transformers` 中应用 [Liger Kernel](https://github.com/linkedin/Liger-Kernel) 是一个非常有效的优化手段。Liger Kernel 是 LinkedIn 开源的一套专门针对大语言模型（LLM）训练进行优化的 Triton 内核，能够显著降低显存占用量（通常可降低 $20\% \sim 60\%$）并提升训练吞吐量。

目前有两种主流方式将 Liger Kernel 应用到 `transformers` 中：**通过 Trainer API 直接启用**（最简单，官方推荐）和**手动 Patch 模型**（适用于自定义训练循环）。

### 第一步：安装依赖

首先，确保你的环境满足要求并安装了必要的库：

```bash
pip install torch triton
pip install transformers>=4.44.0  # 建议使用最新版本
pip install liger-kernel
```

---

### 方法一：通过 `TrainingArguments` 启用（官方推荐，最简单）

如果你使用的是 Hugging Face 的 `Trainer` 或者 `trl` 库的 `SFTTrainer`，由于 `transformers` 已经原生集成了对 Liger Kernel 的支持，你只需要在 `TrainingArguments` 中将 `use_liger_kernel` 参数设置为 `True` 即可。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

# 核心设置：在训练参数中启用 use_liger_kernel
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    use_liger_kernel=True, # <--- 开启 Liger Kernel
    bf16=True,             # Liger Kernel 对 bf16/fp16 支持最好
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=..., # 替换为你的数据集
)

trainer.train()
```

---

### 方法二：手动 Patch 替换模型组件（适用于自定义训练循环）

如果你不使用 `Trainer`，而是手写训练循环（如纯 PyTorch 或 Accelerate），你需要手动调用 Liger Kernel 提供的 API 对模型进行“Monkey Patch”（即在运行时替换掉原生的 PyTorch 算子）。

#### 方式 2.1：针对特定模型进行 Patch
Liger Kernel 提供了针对不同模型系列的 patch 函数：

```python
import torch
from transformers import AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_llama

model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 将原生的 RoPE, RMSNorm, SwiGLU, CrossEntropy 等算子替换为 Liger Triton 算子
apply_liger_kernel_to_llama() 

# 如果只想替换部分算子，可以传入参数
# apply_liger_kernel_to_llama(rope=True, cross_entropy=False)

# ... 接下来写你的自定义训练循环 (Accelerate 等) ...
```

#### 方式 2.2：使用 `AutoLigerKernelForCausalLM`（通用自动 Patch）
类似于 `AutoModel`，Liger Kernel 提供了一个自动包装器，可以自动识别你的模型结构并应用对应的内核：

```python
import torch
from transformers import AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 自动应用 Liger Kernel
model = AutoLigerKernelForCausalLM.from_pretrained(model, model_id)

# ... 你的训练循环 ...
```

---

<details>
<summary><strong>💡 进阶信息：支持的模型列表与注意事项（点击展开）</strong></summary>

#### 支持的模型架构 (截至目前)
Liger Kernel 支持了市面上绝大多数主流的开源模型架构，包括但不限于：
*   **Llama 系列** (Llama 2, Llama 3)
*   **Mistral / Mixtral**
*   **Qwen 2 / Qwen 2.5**
*   **Gemma / Gemma 2**
*   **Phi-3**

你可以通过对应的函数（如 `apply_liger_kernel_to_qwen2`）或者自动包装器来应用它们。

#### 注意事项
1.  **数据类型**：Liger Kernel 主要是为了混合精度训练设计的，强烈建议模型权重和训练使用 `bfloat16`（即参数配置为 `torch.bfloat16` 或 `bf16=True`），在纯 `float32` 下可能无法发挥性能优势或报错。
2.  **不兼容性**：如果在推理（Inference/Generation）阶段，建议**不要**使用 Liger Kernel，或者在使用后重启环境。Liger Kernel 是专门为**前向传播 + 反向传播（训练）**优化的。
3.  **Flash Attention**：Liger Kernel 可以与 `Flash Attention 2` 完美配合。建议在加载模型时同时开启 Flash Attention：
    ```python
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
    ```

</details>