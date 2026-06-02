# from_pretrained传torch_dtype

### 🧑‍💻 User

AutoModelForCausalLM.from_pretrained到底传dtype还是torch_dtype？

---

### 🤖 Assistant

在 Hugging Face 的 `transformers` 库中，调用 `AutoModelForCausalLM.from_pretrained` 时，**必须传入 `torch_dtype`**。

千万不要传 `dtype`，否则会报 `TypeError: unexpected keyword argument 'dtype'` 的错误。

### 正确的代码示例

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name-or-path",
    torch_dtype=torch.float16,  # 正确写法
    device_map="auto"
)
```

<details>
<summary><b>💡 为什么是 torch_dtype 而不是 dtype？（点击展开）</b></summary>
Hugging Face `transformers` 是一个支持多种深度学习框架（PyTorch, TensorFlow, JAX）的库。

- 在原生 PyTorch 中，创建张量时确实使用的是 `dtype`（例如 `torch.zeros(5, dtype=torch.float16)`）。
- 但是，为了在跨框架 API 中明确区分这是 PyTorch 的数据类型参数，Hugging Face 将这个参数统一命名为了 `torch_dtype`。如果使用的是 TensorFlow 版本（如 `TFAutoModel.from_pretrained`），则对应的参数可能是框架相关的逻辑。
</details>

### `torch_dtype` 常用的传参选项

通常我们在加载大语言模型（LLM）时，为了节省显存，会使用以下几种参数：

1. **`torch_dtype="auto"`**（最推荐）：
   模型会读取自身的 `config.json` 文件，自动以模型训练或保存时的精度（通常是 fp16 或 bf16）来加载权重。这能最大程度保证模型性能和精度的原汁原味。
 
2. **`torch_dtype=torch.float16`**：
   强制以半精度（FP16）加载模型，显存占用比默认的 FP32 减少一半，适用于大多数较老的 GPU（如 T4, V100）。

3. **`torch_dtype=torch.bfloat16`**：
   强制以 BF16 加载模型。如果你的显卡是 NVIDIA Ampere 架构及以上（如 RTX 30系列、40系列、A100、H100），强烈推荐使用 BF16，它的数值稳定性比 FP16 更好，不容易溢出。

4. **`torch_dtype=torch.float32`**：
   单精度（FP32）。通常不推荐手动设置这个，因为大语言模型以 FP32 加载会占用极大的显存（例如 7B 模型需要约 28GB 显存）。