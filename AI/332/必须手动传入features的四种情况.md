# 必须手动传入features的四种情况

### 🧑‍💻 User

hf map_dataset，什么情况下必须手动传入features参数？

---

### 🤖 Assistant

在使用 Hugging Face `datasets` 库的 `Dataset.map()` 方法时，默认情况下，底层的数据引擎（PyArrow）会通过检查 `map` 函数返回的前几个样本来**自动推断**新的数据结构（Schema）。

但是，自动推断并不总是完美的。在以下 **4 种核心情况**下，你**必须**手动传入 `features` 参数来明确指定数据结构：

<details open>
<summary><b>1. 引入了 Hugging Face 特有的数据类型（如 ClassLabel, Image, Audio）</b></summary>

PyArrow 只能推断基础的数据类型（如整数、字符串、字节流）。如果你通过 `map` 生成了类别标签、图像或音频，底层只会把它们识别为普通的 `int64`、`bytes` 或字典。你必须手动指定 `features`，Hugging Face 才能正确解析它们。

*   **`ClassLabel`：** 如果你的 `map` 函数将文本标签转换为了数字ID，如果不指定 `features`，它只会被当做普通的整数列，你将失去整数到字符串的映射关系（`id2label`）。
*   **`Image` / `Audio`：** 如果你生成了图像字节流或音频数组，必须指定为 `Image()` 或 `Audio()` 特征，否则在取出数据时，它们不会被自动解码为 PIL 图像或 NumPy 数组。

</details>

<details open>
<summary><b>2. 需要严格控制张量的数据精度（如 float32 vs float64）</b></summary>

在深度学习中，数据类型（dtype）非常关键。Python 默认的浮点数推断通常是 `float64`。如果你正在对数据进行归一化、特征提取等操作，并且准备将其送入 PyTorch 或 TensorFlow 模型，你通常需要 `float32` 甚至 `float16`。
如果不手动传入 `features` 强制指定 `Value("float32")`，你后续在转为 Tensor 时可能会遇到数据类型不匹配的错误，或者占用双倍的内存。

</details>

<details open>
<summary><b>3. 生成了复杂嵌套结构或固定形状的多维数组</b></summary>

当你的 `map` 函数返回多维列表或复杂的嵌套字典时，自动推断很容易出错：
*   如果你的列表长度是固定的（例如词向量、图像像素矩阵），自动推断通常只能识别为不定长的普通列表。你需要显式使用 `Sequence(length=...)` 或 `Array2D`、`Array3D` 等来优化存储和读取性能。
*   如果列表内的数据类型混合或存在 `None`，PyArrow 在推断时极易抛出 `ArrowInvalid` 异常，必须手动定义结构。

</details>

<details open>
<summary><b>4. Map 操作后存在“空样本”或首批样本不具代表性</b></summary>

因为推断是基于前几个样本进行的：
*   如果你的 `map` 函数包含过滤逻辑（虽然通常推荐用 `filter`），导致前几个返回的样本中某个字段是空列表 `[]` 或 `None`，PyArrow 将无法猜出这个字段里面到底装的是什么类型的数据，从而报错。
*   如果某个特征在不同样本中表现形式有差异，导致前几个样本无法代表整体结构。

</details>

---

### 如何手动传入 `features`？

你需要导入 `Features` 模块，并组合使用 `Value`, `ClassLabel`, `Sequence` 等类型。这里是一个典型的代码示例：

```python
from datasets import load_dataset, Features, Value, ClassLabel, Sequence

dataset = load_dataset("imdb", split="train[:100]")

# 假设原始 dataset 的 features 是:
# {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}

# 我们定义一个新的 features 结构
new_features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=['negative', 'positive']), # 保留 ClassLabel 类型
    'input_ids': Sequence(Value('int32')),               # 明确指定长列表内的数据类型为 int32
    'attention_mask': Sequence(Value('int8'))            # mask 通常只需 int8 即可，节省内存
})

def tokenize_function(examples):
    # 这里模拟 tokenizer 的输出
    return {
        "input_ids": [[101, 2023, 102] for _ in examples["text"]],
        "attention_mask": [[1, 1, 1] for _ in examples["text"]]
    }

# 调用 map 时，手动传入 features
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    features=new_features
)
```

**总结：** 只要你对 `map` 后的数据类型精度有严格要求，或者涉及 Hugging Face 特有的高阶类型（Label、多媒体、固定维度数组），就应该养成手动定义并传入 `features` 的习惯，这能避免 90% 以上难以排查的 schema 报错。

---

### 🧑‍💻 User

对于sft ds预处理中常见的包含多轮tool/toolcall/content/reasoning content结构且部分稀疏字段（部分样本无tool/toolcall）的json是必要的吗？

---

### 🤖 Assistant

**是的，在这种情况下，手动传入 `features` 几乎是必须的。** 

对于包含复杂多轮对话（Multi-turn）、工具调用（Tool Calls）、推理内容（Reasoning Content）且存在**稀疏字段**的 SFT 数据集，如果你依赖底层（PyArrow）的自动推断，极大概率会遇到 `ArrowInvalid` 或 `SchemaMismatch` 报错崩溃。

以下是必须手动定义 `features` 的三个核心原因：

<details open>
<summary><b>1. 致命的“首批样本推断陷阱”（解决稀疏字段报错）</b></summary>

PyArrow 在执行 `map` 时，只会根据**前几个样本（或第一个 batch）**来推断整个数据集的数据结构。
*   **缺失字段：** 如果你的数据集前几行刚好全是普通的闲聊对话，没有触发任何 `toolcall`，也没有产生 `reasoning_content`，推断出的 Schema 将**完全不包含**这两个字段。当 `map` 处理到后面几万条包含工具调用的样本时，由于当前 Schema 不认识 `toolcall` 字段，程序会直接报错崩溃。
*   **Null 类型冲突：** 即使所有样本的字典里都写了这些键，但前几个样本里 `toolcall` 是 `None` 或空列表 `[]`，PyArrow 会将该列类型推断为 `Null` 或无类型的空列表。当后续样本出现真实的工具调用数据（如嵌套字典）时，无法写入 `Null` 类型的列中。

</details>

<details open>
<summary><b>2. 深度嵌套列表与字典的类型混乱</b></summary>

SFT 中的 Tool Call 结构通常非常深，例如：
`Message -> List[Dict] -> tool_calls -> List[Dict] -> function -> Dict(name, arguments)`

在如此深的层级下，如果 `arguments` 有时是 JSON 字符串，有时是解析好的字典，或者某些参数缺失，PyArrow 的自动推断极易因为类型不统一而失败。手动定义 `features` 可以强制统一这些嵌套字典的结构，把不一致的数据强转或过滤。

</details>

<details open>
<summary><b>3. 避免内存和存储空间的浪费</b></summary>

自动推断通常为了安全会分配宽泛的类型（例如把所有的文本标记为通用的字符串/对象）。而在 SFT 预处理阶段，你可能需要将对话结构扁平化或直接 Tokenize 成 `input_ids`。通过手动指定 `features`，你可以精确控制 `role` 字段使用 `ClassLabel` 或者定长类型，从而大幅减小处理后的缓存（Cache）文件体积，提升训练加载速度。

</details>

---

### 如何为这种复杂的 SFT 数据定义 Features？

建议在 `load_dataset` 或者 `map` 阶段，使用 `datasets.Features` 将结构严格固定下来。针对你描述的结构，代码示例如下：

```python
from datasets import load_dataset, Features, Value, Sequence

# 定义严格的 SFT 多轮对话 Schema
# 即使某些样本没有 reasoning_content 或 tool_calls，只要结构定义好了，缺失的部分会自动填补 None 或 []
sft_features = Features({
    "messages": Sequence({
        "role": Value("string"),
        "content": Value("string"),
        # 推理内容（针对 DeepSeek-R1 这类模型），稀疏字段
        "reasoning_content": Value("string"), 
        # 工具调用，稀疏字段，通常是一个列表
        "tool_calls": Sequence({
            "id": Value("string"),
            "type": Value("string"),
            "function": {
                "name": Value("string"),
                "arguments": Value("string") # 建议将参数作为 JSON 字符串存储，避免内部嵌套过于复杂
            }
        }),
        # 工具的返回结果，也是稀疏字段
        "tool_call_id": Value("string")
    })
})

# 假设我们在进行一个预处理 map 操作
def preprocess_sft(example):
    # 这里处理你的复杂逻辑
    # 哪怕你的 example["messages"] 中某一项缺少 "tool_calls"，
    # 因为 Features 已经声明了它，底层也会安全处理。
    return example

# 在 map 时强制注入 features
processed_dataset = dataset.map(
    preprocess_sft,
    batched=False,
    features=sft_features # 必须手动传入
)
```

**总结：** 面对带有稀疏字段的复杂 JSON 结构，**不要信任自动推断**。花两分钟写一个清晰的 `Features` 定义，不仅可以彻底根除 `map` 过程中的 Schema 报错，还能作为你数据集格式的“契约”，让后续调试和送入模型的代码更加可靠。