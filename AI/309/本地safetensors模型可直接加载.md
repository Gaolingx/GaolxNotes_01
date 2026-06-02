# 本地safetensors模型可直接加载

### 🧑‍💻 User

hf的AutoModelForCausalLM.from_pretrained中model_name_or_path必须是hf hub的模型路径吗，可以是本地safetensors模型目录吗？

---

### 🤖 Assistant

**不需要必须是 Hugging Face Hub 的模型路径**。它完全**可以是一个本地的文件夹目录**，并且完美支持加载本地的 `safetensors` 模型。

Hugging Face 的 `transformers` 库在执行 `.from_pretrained()` 时，会首先检查传入的字符串是否是一个存在的本地路径。如果该本地路径存在且包含正确的基础配置文件，它就会直接从本地加载，根本不会去请求网络。

### 如何加载本地 `safetensors` 模型目录

你只需要将 `model_name_or_path` 参数设置为你存有模型文件的**本地文件夹路径**（绝对路径或相对路径都可以）。

#### 示例代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 你的本地模型目录路径 (相对路径或绝对路径)
local_model_path = "/path/to/your/local/safetensors_model_dir"

# 1. 加载本地的分词器 (通常分词器文件和模型文件在同一个目录下)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 2. 加载本地的模型
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",         # 可选：自动分配 GPU/CPU 内存
    trust_remote_code=True,    # 可选：如果模型自带了自定义的 Python 脚本则需要开启
    use_safetensors=True       # 可选：强制使用 safetensors（默认情况下如果目录下有 safetensors 就会优先使用）
)
```

### 本地目录需要包含哪些文件？

为了能让 `from_pretrained` 成功加载，你的本地目录 `local_model_path` 下通常需要包含以下核心文件：

1.  **`config.json`**：模型的配置文件（必须有，库需要靠它知道模型架构、层数、隐藏层维度等）。
2.  **权重文件**：例如 `model.safetensors` 或者分片的权重文件如 `model-00001-of-00002.safetensors` 以及配套的 `model.safetensors.index.json`。（必须有）
3.  **分词器文件**（如果要用 `AutoTokenizer` 加载）：通常包括 `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt` 等（具体视模型而定）。
4.  **`generation_config.json`**：（可选）用于生成任务的配置。

**总结**：只要你用工具（比如 `huggingface-cli` 或者迅雷/IDM）把 Hugging Face 上的那个模型仓库里的所有文件下载到了本地的一个文件夹里，你就可以直接把那个文件夹的路径传给 `model_name_or_path`，这样不仅加载速度快，而且完全支持断网离线使用。

---

### 🧑‍💻 User

huggingface中对于自定义架构但实现了PreTrainedModel类的模型且在transformers库外部，如何将其先注册到auto model中，然后走auto model自动加载？

---

### 🤖 Assistant

在 Hugging Face 的 `transformers` 库中，如果你在外部自定义了模型架构（继承了 `PreTrainedModel`）和配置（继承了 `PretrainedConfig`），你可以通过 **`register()` 方法** 将它们注册到 `AutoConfig` 和相应的 `AutoModel` 类中。

一旦注册完成，你就可以像使用官方模型一样，直接使用 `AutoModelForCausalLM.from_pretrained()` 来自动加载你的自定义模型。

以下是完整的步骤和代码示例：

### 核心步骤

1. **定义 Custom Config**：继承 `PretrainedConfig`，并必须指定一个独一无二的 `model_type` 字符串。
2. **定义 Custom Model**：继承 `PreTrainedModel`，并将 `config_class` 属性指向你刚才定义的 Config 类。
3. **注册 Config**：使用 `AutoConfig.register(model_type, CustomConfigClass)`。
4. **注册 Model**：使用 `AutoModelForCausalLM.register(CustomConfigClass, CustomModelClass)`。

---

### 完整代码示例

```python
import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig, 
    PreTrainedModel, 
    AutoConfig, 
    AutoModelForCausalLM
)

# ==========================================
# 1. 定义自定义配置 (Config)
# ==========================================
class MyCustomConfig(PretrainedConfig):
    # 【关键】定义一个独特的 model_type
    model_type = "my_custom_model"

    def __init__(self, vocab_size=32000, hidden_size=256, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


# ==========================================
# 2. 定义自定义模型 (Model)
# ==========================================
class MyCustomModelForCausalLM(PreTrainedModel):
    # 【关键】绑定对应的 Config 类
    config_class = MyCustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 这里随便写一个简单的网络结构作为示例
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        # 简单的 forward 逻辑
        hidden_states = self.embed(input_ids)
        logits = self.lm_head(hidden_states)
      
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
          
        return {"loss": loss, "logits": logits}


# ==========================================
# 3. 将自定义类注册到 Auto 体系中
# ==========================================
# 注册 Config：将 model_type 字符串映射到 Config 类
AutoConfig.register("my_custom_model", MyCustomConfig)

# 注册 Model：将 Config 类映射到 Model 类
AutoModelForCausalLM.register(MyCustomConfig, MyCustomModelForCausalLM)


# ==========================================
# 4. 测试：保存和使用 AutoModel 自动加载
# ==========================================
if __name__ == "__main__":
    local_save_path = "./my_local_custom_model"

    # 实例化并保存模型到本地
    print("Initializing and saving custom model...")
    config = MyCustomConfig(vocab_size=1000, hidden_size=128)
    model = MyCustomModelForCausalLM(config)
  
    # 因为继承了 PreTrainedModel，所以天然支持 save_pretrained，默认也会保存为 safetensors
    model.save_pretrained(local_save_path)
  
    # ------------------------------------------
    # 现在，见证奇迹的时刻：使用 AutoModel 加载
    # ------------------------------------------
    print("Loading via AutoModelForCausalLM...")
  
    # 由于前面已经执行了 register，transformers 现在认识 "my_custom_model" 了
    loaded_model = AutoModelForCausalLM.from_pretrained(local_save_path)
  
    print(f"Loaded model class: {type(loaded_model).__name__}")
    print(f"Loaded config type: {loaded_model.config.model_type}")
```

### 关键细节说明

<details>
<summary><b>1. <code>register</code> 方法的参数对应关系（容易填错）</b></summary>

*   `AutoConfig.register` 的第一个参数是 **字符串**（即你的 `model_type`）。
*   `AutoModelForxxx.register` 的第一个参数是 **配置类本身**（即你的 `MyCustomConfig`），而不是字符串。
</details>

<details>
<summary><b>2. 为什么要使用这种方式？</b></summary>

如果你只是自己本地做实验，写死类名实例化（`model = MyCustomModelForCausalLM.from_pretrained(...)`）当然没问题。
但如果你使用这种 `register` 机制，你的自定义模型就可以完美兼容那些**强依赖 `AutoModel` 的下游工具库**（比如 `PEFT` 做 Lora 微调、`vLLM`、`DeepSpeed`、`trl` 等），因为这些工具内部通常都是调用 `AutoModel.from_pretrained`。
</details>

<details>
<summary><b>3. 如果我想发布到 Hugging Face Hub 给别人用呢？（关于 <code>trust_remote_code</code>）</b></summary>

上面的 `register` 方法只在**当前运行的 Python 进程中有效**。如果你把模型文件夹发给别人，别人直接跑 `from_pretrained` 会报错。

如果想让代码和模型绑定，方便别人直接用，你需要重写模型的类方法将其转为 "Remote Code" 模式（即用户只需加 `trust_remote_code=True` 即可自动下载你的 Python 文件并加载）。
这需要使用 `push_to_hub`，具体做法是在保存前调用：
```python
MyCustomConfig.register_for_auto_class()
MyCustomModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
# 然后 push_to_hub 会自动把你的 python 源码一起传上去
model.push_to_hub("your-username/your-repo")
```
</details>

---

### 🧑‍💻 User

如果我们选择走trust_remote_code=True加载自定义模型（已经自动生成了modeling_*，configuration已实现），但是modeling_*代码和safetensors检查点权重在不同位置怎么办，怎么分别指定他们的path？

---

### 🤖 Assistant

在 Hugging Face 的 `transformers` 库中，如果使用 `trust_remote_code=True` 来加载模型，**原生 API 并不支持分别指定代码路径和权重路径**。

这是因为 `AutoModel.from_pretrained(model_name_or_path)` 在底层执行时，会从传入的 **同一个** `model_name_or_path` 目录中读取 `config.json`，解析出 `auto_map`，然后**强制**在这个相同的目录下寻找对应的 `modeling_*.py` 和 `configuration_*.py` 文件。如果找不到，就会报错。

既然代码和权重分离，你有两种最优雅的解决方案来解决这个问题：

### 方案一：使用软链接（Symlink）“欺骗”框架（最推荐、最简单）

你不需要修改任何 Python 代码，只需要在操作系统层面，将代码文件所在目录下的关键文件（`.py` 文件和 `config.json` 等）**软链接**到你的权重目录中。这样 `transformers` 就会认为它们在同一个目录。

<details open>
<summary><b>操作步骤（以 Linux/macOS 为例）</b></summary>

假设你的目录结构如下：
*   代码目录：`/path/to/code/` (包含 `modeling_my.py`, `configuration_my.py`, `config.json`)
*   权重目录：`/path/to/weights/` (包含 `model.safetensors`)

在终端中执行以下命令，将代码文件链接到权重目录：

```bash
# 进入权重目录
cd /path/to/weights/

# 创建软链接 (按需修改文件名)
ln -s /path/to/code/modeling_my.py ./modeling_my.py
ln -s /path/to/code/configuration_my.py ./configuration_my.py
ln -s /path/to/code/config.json ./config.json
```

完成之后，你的 Python 代码依然可以像标准格式一样加载：

```python
from transformers import AutoModelForCausalLM

# 直接指向包含软链接的权重目录即可
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/weights/",
    trust_remote_code=True
)
```

*(注：Windows 用户可以在 CMD 中使用 `mklink` 命令，或者在 PowerShell 中使用 `New-Item -ItemType SymbolicLink` 来创建软链接。)*

</details>

---

### 方案二：通过 `sys.path` 导入并结合 Auto Register（代码级解决）

如果你不方便修改文件系统的结构（例如在某些受限的 Docker 容器或云环境中），你可以放弃依赖 `trust_remote_code=True` 的动态加载机制，而是手动将代码目录加入 Python 的环境变量中，导入模型类后，将其注册到 Auto 类，然后再加载权重。

<details>
<summary><b>具体实现代码</b></summary>

```python
import sys
from transformers import AutoConfig, AutoModelForCausalLM

# 1. 明确你的两个路径
code_dir = "/path/to/code"
weights_dir = "/path/to/weights"

# 2. 将代码所在目录动态加入 Python 环境变量
if code_dir not in sys.path:
    sys.path.append(code_dir)

# 3. 正常导入你自定义的配置和模型类
# 假设你的代码文件名叫 configuration_my.py 和 modeling_my.py
from configuration_my import MyCustomConfig
from modeling_my import MyCustomModelForCausalLM

# 4. 手动注册它们到 Auto 体系 (替代 trust_remote_code 的作用)
# 注意："my_custom_model" 需要与你 config 中的 model_type 一致
AutoConfig.register("my_custom_model", MyCustomConfig)
AutoModelForCausalLM.register(MyCustomConfig, MyCustomModelForCausalLM)

# 5. 直接从权重目录加载 (此时不再需要 trust_remote_code=True)
# 库会读取 weights_dir 下的 config.json，发现 model_type 是 "my_custom_model"，
# 就会自动调用你刚才注册的 MyCustomModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(weights_dir)

print("模型加载成功！")
```

</details>

### 方案三：直接实例化模型加载权重（绕过 AutoModel）

如果你其实并不强制要求走 `AutoModelForCausalLM`（即你的下游代码可以接受直接实例化的模型对象），你可以用最直接的 `from_pretrained` 方式：

<details>
<summary><b>直接加载代码</b></summary>

```python
import sys

code_dir = "/path/to/code"
weights_dir = "/path/to/weights"

if code_dir not in sys.path:
    sys.path.append(code_dir)

from modeling_my import MyCustomModelForCausalLM

# 所有的 PreTrainedModel 子类都有 from_pretrained 方法，可以直接指定权重目录
model = MyCustomModelForCausalLM.from_pretrained(weights_dir)
```
</details>

**总结**：
因为 `trust_remote_code=True` 的底层安全与寻址机制严格绑定了单一路径，所以要么通过**软链接**在物理路径上把它们合二为一（方案一），要么通过 Python 级别的 **`sys.path` 导入 + 注册** 来接管动态加载的过程（方案二）。