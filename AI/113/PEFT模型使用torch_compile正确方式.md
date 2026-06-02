# PEFT模型使用torch_compile正确方式

### 🧑‍💻 User

对于hf transformers，对peft模型使用torch.compile正确方式是什么？

---

### 🤖 Assistant

在 Hugging Face `transformers` 和 `peft` 库中使用 PyTorch 2.x 的 `torch.compile` 来加速模型，其正确的方式取决于你使用的训练/推理框架。

总体原则是：**先使用 `get_peft_model` 包装模型，然后再应用 `torch.compile`**。

以下是具体的正确实现方式：

### 1. 使用 Hugging Face `Trainer`（最简单且推荐）

如果你使用的是官方的 `Trainer`，你不需要手动调用 `torch.compile`。Hugging Face 已经在底层的 `TrainingArguments` 中为你处理好了兼容性问题。

```python
from transformers import TrainingArguments, Trainer

# 1. 加载基础模型和 PEFT 配置
# model = AutoModelForCausalLM.from_pretrained(...)
# model = get_peft_model(model, peft_config)

# 2. 在 TrainingArguments 中开启 torch.compile
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    # 开启 torch.compile
    torch_compile=True, 
    # 可选：指定 backend 和 mode
    torch_compile_backend="inductor", 
    torch_compile_mode="default", # 或者 "reduce-overhead", "max-autotune"
    # 其他参数...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### 2. 自定义训练循环或纯推理（手动编译）

如果你编写自己的训练循环，或者仅用于推理，你需要手动编译模型。**关键顺序是：基础模型 -> PEFT 包装 -> 编译。**

```python
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained("your-model-id", device_map="auto")

# 2. 包装为 PEFT 模型 (例如 LoRA)
peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)

# --- 关键配置：梯度检查点兼容性 ---
# 如果你使用 gradient checkpointing (显存优化必备)，必须设置 use_reentrant=False
# 否则 torch.compile 会报错或产生图断裂 (Graph Break)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 3. 使用 torch.compile 编译模型
# dynamic=True 很重要，因为 NLP 任务的序列长度 (sequence length) 通常是变化的
compiled_model = torch.compile(model, backend="inductor", dynamic=True)

# 4. 开始训练或推理
# outputs = compiled_model(**inputs)
```

---

<details>
<summary><b>💡 进阶避坑指南与最佳实践（点击展开）</b></summary>

### 1. 梯度检查点（Gradient Checkpointing）冲突
在 PyTorch 2.1 之前，`torch.compile` 和梯度检查点是严重冲突的。如果你使用的是较新的 PyTorch（>= 2.1），**必须**像上面的代码一样，显式指定 `use_reentrant=False`。因为默认的旧版重入机制（reentrant）与编译器的 AOT Autograd 不兼容。

### 2. 动态形状（Dynamic Shapes）
在 NLP 中，每个 Batch 的 `seq_len` 可能不同。如果不加 `dynamic=True`，每次遇到新的序列长度，`torch.compile` 就会**重新编译**（Recompilation），导致不仅没有加速，反而慢得令人发指。

### 3. 环境版本要求
要让 PEFT 和 `torch.compile` 完美配合，请确保你的库足够新：
*   **PyTorch**: $\ge 2.1$ （强烈建议 $2.2$ 或 $2.3$，对动态图的支持更好）
*   **Transformers**: $\ge 4.36.0$
*   **PEFT**: $\ge 0.7.0$

### 4. 推理时的合并优化
如果你只是做推理（不需要训练），最高效的做法是**先合并 LoRA 权重，再编译**，而不是编译带有 LoRA 适配器的模型：

```python
# 推理时的最佳做法
model = AutoPeftModelForCausalLM.from_pretrained("your-lora-path")
# 1. 将 LoRA 权重合并回基础模型
model = model.merge_and_unload() 
# 2. 编译合并后的纯模型
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 5. 避免图断裂（Graph Breaks）
如果你在模型中注入了自定义的代码逻辑（如特殊的 print、依赖 CPU 数据的控制流），会导致编译器退回到 Eager 模式。你可以通过设置环境变量 `TORCH_LOGS="graph_breaks"` 来调试查看编译是否被中断。

</details>