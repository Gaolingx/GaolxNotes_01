# hf模型传入generationconfig方法

### 🧑‍💻 User

hf 模型automodel初始化时候怎么把generation config传入model对象？

---

### 🤖 Assistant

在 Hugging Face 的 `transformers` 库中，为 `AutoModel`（通常是用于文本生成的 `AutoModelForCausalLM` 或 `AutoModelForSeq2SeqLM`）传入或配置 `GenerationConfig` 主要有以下三种标准方式。

最推荐的方式是**在初始化时通过 `from_pretrained` 直接传入**，或者**在模型加载后直接修改对象的属性**。

### 方法一：在 `from_pretrained` 初始化时传入（推荐）

你可以先独立加载或创建一个 `GenerationConfig` 对象，进行修改后，通过 `generation_config` 参数直接传给模型的 `from_pretrained` 方法。

```python
from transformers import AutoModelForCausalLM, GenerationConfig

model_id = "gpt2"

# 1. 加载默认的生成配置（或者创建一个全新的）
gen_config = GenerationConfig.from_pretrained(model_id)

# 2. 修改你需要的生成参数
gen_config.max_new_tokens = 100
gen_config.temperature = 0.8
gen_config.top_p = 0.95
gen_config.do_sample = True
gen_config.repetition_penalty = 1.2

# 3. 在模型初始化时传入该配置
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    generation_config=gen_config
)

print(model.generation_config.temperature) # 输出: 0.8
```

### 方法二：在模型初始化后直接修改属性

如果在初始化时忘记传入，或者想在运行时动态修改，可以直接操作 `model.generation_config` 属性。Hugging Face 的模型在加载时会自动实例化一个默认的 `GenerationConfig` 挂载在模型对象上。

```python
from transformers import AutoModelForCausalLM, GenerationConfig

model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)

# 方式 A：直接修改现有配置的属性
model.generation_config.max_new_tokens = 50
model.generation_config.temperature = 0.7
model.generation_config.do_sample = True

# 方式 B：用一个全新的 GenerationConfig 对象完全替换它
new_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.9,
    do_sample=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id
)
model.generation_config = new_config
```

### 方法三：在调用 `generate()` 时临时覆盖

如果你不想把配置永久绑定在模型对象上，而只是想在某次生成时使用特定的参数，可以将 `GenerationConfig` 传入 `generate()` 方法。这会临时覆盖模型自带的配置。

```python
from transformers import AutoModelForCausalLM, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 创建临时配置
temp_config = GenerationConfig(
    max_new_tokens=20,
    temperature=2.0,
    do_sample=True
)

# 生成时传入（假设 inputs 已经准备好）
# outputs = model.generate(**inputs, generation_config=temp_config)
```

---

<details>
<summary>💡 进阶提示：为什么 Hugging Face 引入了 GenerationConfig？以及如何保存它（点击展开）</summary>

**背景原因：**
在旧版本的 `transformers` 中，生成参数（如 `temperature`, `top_p`）是直接混写在模型的主配置 `config.json` 中的。这导致了模型结构参数（如层数、注意力头数）和推理生成参数的混乱。
因此，Hugging Face 引入了独立的 `GenerationConfig` 类和对应的 `generation_config.json` 文件，实现了**模型结构**与**生成策略**的解耦。

**如何保存你的自定义配置：**
当你修改好了一个非常棒的生成配置，并希望以后直接使用它，可以将它与模型一起保存。

```python
# 你的模型和修改好的配置
model.generation_config.temperature = 0.6

# 保存到本地目录
model.save_pretrained("./my-custom-model")

# 这将在 "./my-custom-model" 目录下生成一个 generation_config.json 文件。
# 下次使用 AutoModelForCausalLM.from_pretrained("./my-custom-model") 时，
# 它会自动读取这个配置，无需再次手动传入。
```
</details>

---

### 🧑‍💻 User

如果初始化传入后其他地方怎么获取？

---

### 🤖 Assistant

在模型初始化并传入 `GenerationConfig` 之后，在代码的其他任何地方，只要你能访问到这个 `model` 对象，就可以直接通过 **`model.generation_config`** 属性来获取它。

它被挂载为模型实例的一个普通属性。以下是具体的获取和使用方式：

### 1. 直接获取和读取配置

你可以直接访问该属性来查看当前的生成参数：

```python
# 假设 model 已经在某处被初始化
current_gen_config = model.generation_config

# 查看参数
print(f"当前温度 (Temperature): {current_gen_config.temperature}")
print(f"最大生成长度 (Max New Tokens): {current_gen_config.max_new_tokens}")
print(f"是否采样 (Do Sample): {current_gen_config.do_sample}")

# 甚至可以直接打印整个配置对象，查看所有参数
print(current_gen_config)
```

### 2. 获取后动态修改（In-place 修改）

获取到这个对象后，你可以随时在代码的其他地方修改它的属性。修改后，下一次调用 `model.generate()` 时就会自动生效（前提是你调用 `generate` 时没有传入新的临时配置）：

```python
def change_generation_strategy(model, strategy_type):
    # 获取模型的生成配置
    config = model.generation_config
  
    if strategy_type == "creative":
        # 变得更有创造性
        config.temperature = 1.2
        config.top_p = 0.9
        config.do_sample = True
    elif strategy_type == "strict":
        # 变得更严谨（贪心搜索）
        config.temperature = 0.0
        config.do_sample = False
      
    print(f"已切换到 {strategy_type} 策略，当前温度: {model.generation_config.temperature}")

# 在代码其他地方调用
change_generation_strategy(model, "creative")
# 后续的 model.generate(...) 将使用修改后的配置
```

### 3. 获取后复制为副本 (避免污染原配置)

有时候你可能想获取当前的配置，稍微修改一下给某个特定的生成任务用，但**不想改变模型默认的配置**。你可以通过 `copy()` 方法获取它的副本：

```python
import copy

# 1. 获取当前配置的深度拷贝
temp_config = copy.deepcopy(model.generation_config)

# 2. 修改副本
temp_config.max_new_tokens = 500

# 3. 仅在这次生成中使用修改后的副本
# outputs = model.generate(**inputs, generation_config=temp_config)

# 模型的默认配置 model.generation_config 保持不变
```

**总结：**
无论你在哪里，只要有 `model` 变量，直接调用 `model.generation_config` 即可获取、读取或修改它。

---

### 🧑‍💻 User

如果传入了generation配置，save pretrained会自动导出generation config.json吗？

---

### 🤖 Assistant

**是的，会自动导出。**

只要你的模型对象（`model`）附带了 `generation_config` 属性（不论是初始化时传入的，还是初始化后动态修改的），当你调用 `model.save_pretrained(...)` 时，Hugging Face 的 `transformers` 库会自动将这个配置序列化，并在指定的目录中生成一个 `generation_config.json` 文件。

### 具体行为演示

```python
from transformers import AutoModelForCausalLM, GenerationConfig

model_id = "gpt2"

# 1. 创建并自定义生成配置
custom_gen_config = GenerationConfig.from_pretrained(model_id)
custom_gen_config.temperature = 0.55
custom_gen_config.max_new_tokens = 128

# 2. 初始化时传入模型
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    generation_config=custom_gen_config
)

# 3. 保存模型
save_directory = "./my_custom_model"
model.save_pretrained(save_directory)
```

执行上述代码后，如果你查看 `./my_custom_model` 目录，你会看到如下文件结构：

```text
my_custom_model/
├── config.json               # 模型的主结构配置
├── generation_config.json    # ✅ 自动导出的生成配置（包含 temperature=0.55 等设定）
├── model.safetensors         # 模型权重文件
└── ...
```

### 重新加载时的行为

当你下次从这个本地目录加载模型时，**不需要**再手动传入 `generation_config`。`from_pretrained` 会自动检测目录下的 `generation_config.json` 并加载它：

```python
# 下次加载时，直接读取本地目录
loaded_model = AutoModelForCausalLM.from_pretrained("./my_custom_model")

# 它会自动继承你之前保存的配置
print(loaded_model.generation_config.temperature) 
# 输出: 0.55
print(loaded_model.generation_config.max_new_tokens) 
# 输出: 128
```

<details>
<summary>⚠️ 注意事项（点击展开）</summary>

1. **适用模型类型**：自动保存 `generation_config.json` 的前提是该模型支持生成任务（即 `model.can_generate()` 返回 `True`，例如 `AutoModelForCausalLM` 或 `AutoModelForSeq2SeqLM`）。如果是纯粹的特征提取模型或分类模型（如普通的 `AutoModel` 或 `AutoModelForSequenceClassification`），通常不会保存此文件。
2. **单独保存配置**：如果你只想保存生成配置，而不想保存庞大的模型权重，可以单独调用配置的保存方法：
   ```python
   model.generation_config.save_pretrained("./only_gen_config_dir")
   ```
</details>