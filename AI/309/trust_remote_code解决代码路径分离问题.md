# trust_remote_code解决代码路径分离问题

### 🧑‍💻 User

hf transformers的`AutoModelForCausalLM.from_pretrained`方法的`trust_remote_code=True`如何解决`modeling_*/configuration_*`代码与model.safetensors不在相同路径的情况？

---

### 🤖 Assistant

在 Hugging Face `transformers` 库中，当启用 `trust_remote_code=True` 时，底层依赖的是 `config.json` 中的 `auto_map` 字段来动态加载本地或远程的自定义 `.py` 文件（如 `modeling_xxx.py` 和 `configuration_xxx.py`）。

默认情况下，`from_pretrained` 会严格假设**配置文件 (`config.json`)、自定义代码文件 (`.py`) 和权重文件 (`.safetensors`) 都在同一个路径下**。如果你的代码和权重被硬性分离在了不同的目录，直接调用会报错（通常提示找不到 module）。

以下是解决代码与权重不在同一路径的几种有效方法：

### 方法一：分步加载 Config 和 Model（官方推荐做法）

这是最优雅的在代码层面解决的方法。你可以先从**代码所在路径**加载 `AutoConfig`，然后再将这个实例化的 `config` 传给 `AutoModelForCausalLM` 去**权重所在路径**加载模型。

```python
from transformers import AutoConfig, AutoModelForCausalLM

code_dir = "/path/to/your/custom_code_dir"  # 包含 modeling_*.py, configuration_*.py 和 config.json
weights_dir = "/path/to/your/weights_dir"   # 包含 model.safetensors

# 1. 从代码路径加载 Config，这会触发 trust_remote_code 并在内存中注册自定义类
config = AutoConfig.from_pretrained(
    code_dir, 
    trust_remote_code=True
)

# 2. 将加载好的 config 传入，同时指定权重路径
model = AutoModelForCausalLM.from_pretrained(
    weights_dir,
    config=config,
    trust_remote_code=True # 依然需要设为 True
)
```
**原理**：当 `AutoConfig.from_pretrained` 带有 `trust_remote_code=True` 执行时，`transformers` 会把自定义的代码编译并注册到系统的动态模块中。后续加载权重时，直接使用内存中已经注册好的类结构去填充权重。

---

### 方法二：系统级别的软链接 (Symlink)

如果你不想修改加载模型的 Python 代码，最简单粗暴且不出错的方法是在操作系统层面建立软链接，欺骗 `transformers` 让它以为它们在同一个文件夹里。

```bash
# 假设当前在 weights_dir 目录下
cd /path/to/your/weights_dir

# 将代码目录下的必要文件软链接过来
ln -s /path/to/your/custom_code_dir/*.py .
ln -s /path/to/your/custom_code_dir/config.json .
```
之后你只需要按照常规方式加载即可：
```python
model = AutoModelForCausalLM.from_pretrained("/path/to/your/weights_dir", trust_remote_code=True)
```

---

<details>
<summary><b>方法三：手动修改 sys.path 并导入 (Hack 做法 - 点击展开)</b></summary>

如果在某些特殊环境下你既不能分步加载，也不能做软链接，你可以通过修改 Python 的 `sys.path`，手动把自定义代码引入环境变量中。

前提：`weights_dir` 中必须要有 `config.json`，且其中的 `auto_map` 指向你的自定义文件名。

```python
import sys
import os
from transformers import AutoModelForCausalLM

code_dir = "/path/to/your/custom_code_dir"
weights_dir = "/path/to/your/weights_dir"

# 强行将代码目录加入系统环境变量
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# 此时 transformers 内部的 import 机制能直接从 sys.path 找到相关的 .py 文件
model = AutoModelForCausalLM.from_pretrained(
    weights_dir, 
    trust_remote_code=True
)
```
*注意：这种方法可能会引起命名空间污染，通常只在快速调试时使用。*

</details>

<details>
<summary><b>深入理解：trust_remote_code 是如何工作的？ (点击展开)</b></summary>

当你设置 `trust_remote_code=True` 时，`transformers` 的内部处理流程如下：

1. **读取 config.json**：寻找 `auto_map` 字段。例如：
   ```json
   "auto_map": {
     "AutoConfig": "configuration_custom.CustomConfig",
     "AutoModelForCausalLM": "modeling_custom.CustomModel"
   }
   ```
2. **定位模块**：库内的 `dynamic_module_utils.py` 会去 `pretrained_model_name_or_path` (即你传入的第一个参数) 下寻找 `configuration_custom.py` 和 `modeling_custom.py`。
3. **缓存与加载**：
   - 如果是远程 Hugging Face 仓库，它会把这些 `.py` 文件下载到 `~/.cache/huggingface/modules/` 下。
   - 如果是本地路径，它会将该本地路径作为一个临时的 Python package 动态导入。
4. **实例化**：使用导入的 `CustomModel` 类，并尝试读取该目录下的 `model.safetensors` 或 `pytorch_model.bin` 来加载权重。

因此，如果代码和权重分离，核心矛盾在于**步骤 2 找不到代码** 或 **步骤 4 找不到权重**。这就是为什么“方法一”通过分开指定 Config 路径和 Model 路径能完美解决这个问题的原因。

</details>

---

### 🧑‍💻 User

AutoModelForCausalLM.from_pretrained里边传入config=AutoConfig.from_pretrained(...)可以吗？需要分别如何编写auto_map字段？

---

### 🤖 Assistant

**结论是可以的**，你可以直接在 `AutoModelForCausalLM.from_pretrained` 中传入实例化的 `config`。

关于 `auto_map` 的编写，**你不需要（也不能）在里面写绝对路径**，它的写法与标准情况完全一致。但因为 `transformers` 底层源码的逻辑，直接传入 `config` 会踩到一个“隐藏的坑”。下面我将为你详细说明如何编写 `auto_map` 以及如何避坑。

### 1. `auto_map` 应该如何编写？

`auto_map` 字段存放在代码目录 (`code_dir`) 的 `config.json` 中。无论你的权重文件在哪里，`auto_map` 的格式都是固定的 `"文件名.类名"`：

```json
{
  ...
  "auto_map": {
    "AutoConfig": "configuration_custom.CustomConfig",
    "AutoModelForCausalLM": "modeling_custom.CustomModelForCausalLM"
  },
  ...
}
```
*注：这里的 `configuration_custom.py` 和 `modeling_custom.py` 必须存在于你的 `code_dir` 目录下。*

---

### 2. 隐藏的坑：为什么直接传 `config` 可能依然会报错？

如果你按照下面的常规思路写代码：

```python
# 这样写很大可能会报错！
config = AutoConfig.from_pretrained(code_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    weights_dir,          # 模型权重目录
    config=config,        # 传入刚刚加载的 config
    trust_remote_code=True
)
```

**报错原因**：虽然 `config` 是从 `code_dir` 成功加载的，但当你调用 `AutoModelForCausalLM.from_pretrained(weights_dir, ...)` 时，`transformers` 底层逻辑会读取 `config.auto_map` 中的 `"modeling_custom.CustomModelForCausalLM"`，**然后尝试去 `weights_dir` (你传入的第一个参数) 下寻找 `modeling_custom.py`**。因为代码和权重分离，`weights_dir` 下没有这个文件，程序就会抛出 `EnvironmentError` 或 `ModuleNotFoundError`。

---

### 3. 终极解决方案：动态提取模型类

为了完美解决上述问题（既不修改环境变量 `sys.path`，也不做软链接），你可以使用 `transformers` 内部的动态加载工具，先从 `code_dir` 提取出真正的模型类，再用这个类去加载 `weights_dir` 的权重。

这是最符合代码逻辑的写法：

```python
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

code_dir = "/path/to/custom_code_dir"
weights_dir = "/path/to/weights_dir"

# 1. 从代码目录加载 Config
config = AutoConfig.from_pretrained(code_dir, trust_remote_code=True)

# 2. 从 auto_map 中获取模型类的引用字符串 (例如: "modeling_custom.CustomModelForCausalLM")
class_ref = config.auto_map["AutoModelForCausalLM"]

# 3. 强制从代码目录 (code_dir) 动态加载这个类！
# 这一步是核心，它告诉 transformers 去 code_dir 找 modeling_custom.py
CustomModelClass = get_class_from_dynamic_module(class_ref, code_dir)

# 4. 使用加载好的类，去权重目录 (weights_dir) 读取 .safetensors 权重
model = CustomModelClass.from_pretrained(
    weights_dir,
    config=config
)

print("模型加载成功！")
```

<details>
<summary><b>附录：底层原理解析 (点击展开)</b></summary>

在 `transformers/models/auto/auto_factory.py` 源码中，`AutoModelForCausalLM` 加载远程代码的核心逻辑如下：

```python
if has_remote_code and trust_remote_code:
    class_ref = config.auto_map[cls.__name__]
    model_class = get_class_from_dynamic_module(
        class_ref, pretrained_model_name_or_path, ...
    )
```

因为你调用 `AutoModel` 时传入的 `pretrained_model_name_or_path` 是 `weights_dir`，所以源码强制去 `weights_dir` 找代码。

我们上面的“终极解决方案”，本质上就是**手动接管了这段逻辑**，把 `pretrained_model_name_or_path` 替换成了真正的 `code_dir`，从而获取到 `CustomModelClass`。一旦获取到具体的类实例，调用它的 `from_pretrained` 方法就只会去关注权重文件（如 `model.safetensors`）了。

</details>

---

### 🧑‍💻 User

对于以下两种项目结构分别怎么处理？
case 1:
```
/project/
  ├── config_dir/
  │   ├── config.json
  │   ├── configuration_my_model.py  # 或放在这里
  │   └── modeling_my_model.py
  └── weights_dir/
      └── model.safetensors          # 权重文件
```
case 2:
```
/project/
  ├── config_dir/
  │   └── config.json
  ├── code_dir/                # 自定义代码在这里
  │   ├── configuration_my_model.py
  │   └── modeling_my_model.py
  └── weights_dir/
      └── model.safetensors
```

---

### 🤖 Assistant

针对这两种代码结构，核心原则是：**打破 `transformers` 默认的“配置文件、代码文件、权重文件必须在一起”的假设，手动接管类的加载过程。**

以下是分别针对这两种情况的具体处理方法和代码实现。

---

### Case 1: 代码与 Config 在同级目录，权重独立

```text
/project/
  ├── config_dir/     <-- config.json, configuration_*.py, modeling_*.py
  └── weights_dir/    <-- model.safetensors
```

在这种结构下，`AutoConfig` 可以正常解析同目录下的 `.py` 文件。我们只需用 `get_class_from_dynamic_module` 提取模型类，再指向权重目录即可。这是最标准的拆分加载方式。

#### 核心实现代码
```python
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

config_dir = "/project/config_dir"
weights_dir = "/project/weights_dir"

# 1. 从 config_dir 正常加载 config（它会自动读取同目录的 configuration_my_model.py）
config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)

# 2. 提取模型类的引用字符串 (例如："modeling_my_model.MyModelForCausalLM")
model_class_ref = config.auto_map["AutoModelForCausalLM"]

# 3. 强制从 config_dir 动态加载模型类（读取同目录的 modeling_my_model.py）
MyModelClass = get_class_from_dynamic_module(model_class_ref, config_dir)

# 4. 实例化模型并加载 weights_dir 中的权重
model = MyModelClass.from_pretrained(
    weights_dir,
    config=config
)
print("Case 1: 模型加载成功")
```

---

### Case 2: Config、代码、权重三者完全分离

```text
/project/
  ├── config_dir/     <-- config.json
  ├── code_dir/       <-- configuration_*.py, modeling_*.py
  └── weights_dir/    <-- model.safetensors
```

这种情况稍微复杂。如果你直接调用 `AutoConfig.from_pretrained(config_dir)`，程序会去 `config_dir` 下找 `configuration_my_model.py`，因为没找到而报错。
因此，我们需要**彻底绕过 `Auto` 类的文件查找机制**，手动读取 JSON，然后直接从 `code_dir` 加载 Config 和 Model 的类。

#### 核心实现代码
```python
import json
import os
from transformers.dynamic_module_utils import get_class_from_dynamic_module

config_path = "/project/config_dir/config.json"
code_dir = "/project/code_dir"
weights_dir = "/project/weights_dir"

# 1. 手动读取 config.json 为字典
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

# 2. 动态加载自定义的 Config 类
config_class_ref = config_dict["auto_map"]["AutoConfig"]
MyConfigClass = get_class_from_dynamic_module(config_class_ref, code_dir)

# 3. 使用字典实例化 Config
config = MyConfigClass.from_dict(config_dict)

# 4. 动态加载自定义的 Model 类
model_class_ref = config_dict["auto_map"]["AutoModelForCausalLM"]
MyModelClass = get_class_from_dynamic_module(model_class_ref, code_dir)

# 5. 实例化模型并加载权重
model = MyModelClass.from_pretrained(
    weights_dir,
    config=config
)
print("Case 2: 模型加载成功")
```

<details>
<summary><b>关于 `auto_map` 的编写规范要求 (点击展开)</b></summary>

无论是 Case 1 还是 Case 2，你的 `config.json` 中的 `auto_map` 都不需要包含路径信息，只需指定 **`模块名.类名`** 即可。

例如，针对上面的两个 Case，`config.json` 中的 `auto_map` 必须写成这样：
```json
{
  "auto_map": {
    "AutoConfig": "configuration_my_model.MyCustomConfig",
    "AutoModelForCausalLM": "modeling_my_model.MyCustomModelForCausalLM"
  }
}
```
`get_class_from_dynamic_module` 函数的第二个参数（我们传入的 `config_dir` 或 `code_dir`）会决定程序去哪个物理路径下寻找 `configuration_my_model.py` 和 `modeling_my_model.py`。
</details>