# 配置bitsandbytes跳过关键层量化

### 🧑‍💻 User

bitsandbytes 4bit量化如何编写模型config.json中quantization config字段避免关键层（例如mlp.gate）被量化？

---

### 🤖 Assistant

在使用 `bitsandbytes` 进行 4-bit 量化时，如果希望在 `config.json` 中配置跳过特定关键层（如 `mlp.gate` 或 `lm_head`）以保留其高精度（通常为 fp16 或 bf16），你需要使用 **`llm_int8_skip_modules`** 字段。

这是一个历史遗留的命名问题：尽管字段名包含 `int8`，但 Hugging Face `transformers` 库**在 4-bit 量化时同样使用这个字段**来决定哪些模块不被转换为量化线性层。

以下是具体的编写方法和相关说明：

### 1. 修改 `config.json` 中的 `quantization_config`

在你的模型根目录下的 `config.json` 文件中，找到（或创建）`quantization_config` 字典，并添加 `llm_int8_skip_modules` 列表：

```json
{
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "llm_int8_skip_modules": [
      "lm_head",
      "gate_proj",
      "mlp.gate"
    ]
  }
}
```

**关键字段说明：**
*   `llm_int8_skip_modules`: 这是一个字符串列表。在加载模型时，如果某基层的名称**包含**列表中的任意字符串（子串匹配），该层就不会被 4-bit 量化，而是保持 `torch_dtype`（通常是 fp16/bf16）。

---

### 2. Python 代码中的等效写法

如果你不想直接修改 `config.json`，在加载模型时通过 Python 代码动态传入 `BitsAndBytesConfig` 也是非常推荐的做法：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 定义量化配置，指定要跳过的层
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["gate_proj", "mlp.gate", "lm_head"] # 在此处跳过
)

model = AutoModelForCausalLM.from_pretrained(
    "your-model-path",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

### 补充指南

<details>
<summary><b>如何确定准确的层名称？</b></summary>

不同的模型架构（如 LLaMA, Qwen, ChatGLM）对 MLP 层的命名不同。例如：
*   **LLaMA / Mistral**: 通常叫 `gate_proj`, `up_proj`, `down_proj`
*   **Qwen**: 通常叫 `mlp.w1`, `mlp.w2`, `mlp.c_proj`

在填写 `llm_int8_skip_modules` 之前，建议先以非量化方式（或查阅源码）打印模型结构：
```python
# 打印模型结构查看具体的 layer 名字
for name, module in model.named_modules():
    print(name)
```
只要你的列表中包含目标层的**名字片段**（如 `"gate_proj"`），所有包含该片段的层（如 `model.layers.0.mlp.gate_proj`）都会被跳过。

</details>

<details>
<summary><b>为什么要跳过特定的 MLP 层？</b></summary>

在一些大语言模型（如使用 SwiGLU 激活函数的模型）中，MLP 的 `gate` 层负责控制信息的流通比例，其权重分布往往存在明显的离群值（Outliers）。
强行将其压缩为 4-bit 可能会导致激活值失真，从而引起模型输出乱码或困惑度（Perplexity, $PPL$）显著上升。保留这些层为 16-bit 可以在几乎不增加显存消耗的情况下，大幅挽回量化带来的性能损失。
</details>

<details>
<summary><b>验证是否跳过成功</b></summary>

加载模型后，你可以遍历模型的层，检查它们的类型（`dtype`）和所属类，以确认关键层是否仍为普通线性层而非 `Linear4bit`：

```python
for name, module in model.named_modules():
    if "gate" in name:
        print(f"{name}: {type(module)}")
        # 预期输出: <class 'torch.nn.modules.linear.Linear'> 而不是 bitsandbytes 的 Linear4bit
```
</details>

---

### 🧑‍💻 User

处于fp32精度的norm层会自动跳过量化吗？

---

### 🤖 Assistant

**是的，处于 fp32（或 fp16/bf16）精度的 Norm 层（如 `LayerNorm`、`RMSNorm`）会自动跳过 4-bit 量化。**

你不需要在 `llm_int8_skip_modules` 中手动将它们添加进去。具体原因如下：

### 1. 模块类型的限制 (最主要原因)
`bitsandbytes` 的 4-bit/8-bit 量化逻辑在 Hugging Face `transformers` 库中，默认**仅针对 `torch.nn.Linear`（线性全连接层）** 进行替换。

归一化层（如 `torch.nn.LayerNorm` 或各大模型自定义的 `RMSNorm`）属于完全不同的神经网络模块类型。量化脚本在遍历模型层时，遇到 Norm 层会自动忽略，不会将其替换为 4-bit 的量化层（如 `Linear4bit`）。

### 2. 参数量与收益的权衡
Norm 层的参数量极小（通常只有一维的 `weight` 和 `bias`），对它们进行量化几乎节省不了任何显存。相反，大语言模型对 Norm 层的精度极其敏感。如果将 Norm 层量化为 4-bit，会导致严重的数值截断误差，极易引起模型输出跑偏或直接输出 NaN（Not a Number）。因此，底层框架设计之初就排除了对 Norm 层的量化。

---

### 补充说明：Norm 层的最终精度由什么决定？

虽然 Norm 层不会变成 4-bit，但它加载到显存时的具体精度（是 fp32 还是 fp16/bf16），取决于你加载模型时传入的 `torch_dtype` 参数或 `config.json` 中的设置：

*   **默认情况**：如果你在 `AutoModelForCausalLM.from_pretrained` 中指定了 `torch_dtype=torch.float16`，那么未被量化的层（包括 Norm 层和 `llm_int8_skip_modules` 里的层）都会被转换为 fp16。
*   **保持 FP32**：在很多现代大模型（如 LLaMA 及其变体）的训练和推理中，为了保证数值稳定性，即使主体计算使用 fp16/bf16，**Norm 层的权重也会被强制保持在 fp32**，并在 fp32 下进行前向计算。`transformers` 库能够很好地处理这种混合精度状态。

### 如何验证？
加载模型后，你可以运行以下简单的代码来验证 Norm 层的状态：

```python
for name, module in model.named_modules():
    # 假设模型的 norm 层名字里包含 'norm'
    if "norm" in name.lower():
        # 获取该模块的权重参数
        if hasattr(module, "weight") and module.weight is not None:
            print(f"Layer: {name} | Type: {type(module).__name__} | Dtype: {module.weight.dtype}")
```
**预期输出示例：**
你会看到它们仍然是原生的 Norm 类，并且数据类型通常是 `torch.float32` 或 `torch.float16`，绝不会是 `bitsandbytes` 的 `uint8` 伪装类型。

```text
Layer: model.layers.0.input_layernorm | Type: LlamaRMSNorm | Dtype: torch.float32
Layer: model.layers.0.post_attention_layernorm | Type: LlamaRMSNorm | Dtype: torch.float32
...
```