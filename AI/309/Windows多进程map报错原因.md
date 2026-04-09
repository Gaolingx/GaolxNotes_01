# Windows多进程map报错原因

### 🧑‍💻 User

`datasets.Dataset.map()`在Windows上多workers为什么报错？

---

### 🤖 Assistant

在 Windows 上使用 Hugging Face 的 `datasets.Dataset.map()` 并开启多进程（`num_proc > 1`）时报错，是一个非常经典的问题。

导致这个问题的**根本原因在于 Windows 和 Linux 在创建多进程时的底层机制不同**：
*   **Linux** 使用 `fork` 机制。子进程会直接继承父进程的内存状态，速度快且不需要重新加载代码。
*   **Windows** 没有 `fork`，只能使用 `spawn` 机制。它会启动一个全新的 Python 解释器，并**重新导入（import）一遍你的主脚本**，然后通过 `pickle` 序列化来传递数据。

基于这个底层差异，通常会引发以下几种报错，以下是对应的解决方案：

### 1. 最常见报错：缺少 `__main__` 保护
**报错信息通常包含：** `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.`

**原因：** 因为 Windows 的 `spawn` 会重新执行主脚本，如果你的 `map()` 调用暴露在最外层，子进程在启动时又会执行到 `map()`，从而无限递归创建子进程，导致程序崩溃。

**解决方法：** 必须将所有执行代码放入 `if __name__ == '__main__':` 块中。

**错误写法 ❌**
```python
from datasets import load_dataset

def process_data(example):
    example["new_col"] = len(example["text"])
    return example

dataset = load_dataset("imdb", split="train")
# 在 Windows 上直接这样写会报错！
dataset = dataset.map(process_data, num_proc=4) 
```

**正确写法 ✅**
```python
from datasets import load_dataset

# 函数定义可以放在外面
def process_data(example):
    example["new_col"] = len(example["text"])
    return example

if __name__ == '__main__':
    # 执行逻辑必须放在 main 保护块里面
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.map(process_data, num_proc=4)
    print(dataset)
```

---

<details>
<summary><b>2. 报错：使用了 Lambda 匿名函数 (AttributeError: Can't pickle local object)</b></summary>

**原因：** Windows 必须使用 `pickle` 将处理函数发送给子进程。但是 Python 标准的 `pickle` **无法序列化 lambda 函数**或嵌套在函数内部的局部函数。

**解决方法：** 不要使用 lambda 函数，将其改为标准的全局 `def` 函数。

**错误写法 ❌**
```python
if __name__ == '__main__':
    dataset = dataset.map(lambda x: {"len": len(x["text"])}, num_proc=4)
```

**正确写法 ✅**
```python
def get_length(example):
    return {"len": len(example["text"])}

if __name__ == '__main__':
    dataset = dataset.map(get_length, num_proc=4)
```
</details>

<details>
<summary><b>3. 报错或卡死：Tokenizer 导致的死锁</b></summary>

**原因：** Hugging Face 的 Tokenizer 底层是 Rust 编写的，自带多线程并行。当 Python 的多进程（`num_proc`）和 Rust 的多线程同时开启时，在 Windows 上极易发生死锁（程序不报错，但进度条一直卡住不动）。

**解决方法：** 在代码最开头禁用 Tokenizer 的内部并行。

```python
import os
# 必须在导入 transformers 之前或代码最开头设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from datasets import load_dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("imdb", split="train")
    # batched=True 通常能提升 tokenizer 的处理速度
    dataset = dataset.map(tokenize_function, batched=True, num_proc=4) 
```
</details>

<details>
<summary><b>4. 报错：无法 Pickle 复杂的全局对象（如数据库连接、大型模型实例）</b></summary>

**原因：** 如果你在 `map()` 的处理函数中引用了外部无法被 `pickle` 的对象（例如数据库连接池、某些特定的 C++ 绑定对象），Windows 的多进程传递会失败。

**解决方法：** 
1. 尽量让传入 `map` 的函数保持“纯净”，只处理字典/文本。
2. 如果必须在 `map` 中使用复杂对象，请在函数内部进行初始化（这会导致每个批次初始化一次，性能会有损耗）。
3. 如果实在无法解决，建议在 Windows 上妥协，设置 `num_proc=1`（或者直接删掉 `num_proc` 参数）单进程运行，配合 `batched=True`，速度通常也可以接受。
</details>

### 总结检查清单
如果在 Windows 上使用 `Dataset.map(num_proc=N)`，请确保：
1. [x] 运行代码被包裹在 `if __name__ == '__main__':` 中。
2. [x] 传入的函数是顶层定义的 `def` 函数，而不是 `lambda`。
3. [x] 如果处理逻辑包含分词（Tokenization），设置了环境变量 `TOKENIZERS_PARALLELISM=false`。