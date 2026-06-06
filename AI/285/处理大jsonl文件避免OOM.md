# 处理大jsonl文件避免OOM

### 🧑‍💻 User

数百g的预训练jsonl到底要怎么编辑，直接vscode打开会oom

---

### 🤖 Assistant

处理数百GB的预训练 `.jsonl`（JSON Lines）文件时，**绝对不能使用普通的文本编辑器（如 VS Code, Notepad++, Sublime Text）直接打开**。这些编辑器会尝试将整个文件加载到内存（RAM）中，必然会导致 OOM（Out of Memory）崩溃。

`.jsonl` 格式的优势在于**每一行都是一个独立的 JSON 对象**。这意味着你完全不需要把整个文件读入内存，而是可以**“流式”（Streaming）**或**“逐行”（Line-by-line）**处理。

针对你的需求（查看、局部修改、批量修改），以下是标准的处理方案：

### 1. 仅仅为了“查看”和“采样”（不修改）
如果你只是想看看数据长什么样，检查字段结构，请使用命令行工具。它们只会读取文件的开头部分或当前屏幕的内容，内存占用几乎为 0。

*   **查看前 10 行：**
    ```bash
    head -n 10 dataset.jsonl
    ```
*   **交互式浏览（可上下翻页，按 `q` 退出）：**
    ```bash
    less dataset.jsonl
    ```
*   **提取特定的几行保存到新文件（用于在 VS Code 中安全打开并编写测试代码）：**
    ```bash
    head -n 1000 dataset.jsonl > sample.jsonl
    ```

---

### 2. 批量编辑/清洗数据（最推荐：Python 流式处理）
对于数百GB的数据，真正的“编辑”通常是写一段脚本，边读边改边写。Python 处理这种任务非常简单且内存安全。

**核心思想：读一行 -> 解析 JSON -> 修改 -> 写入新文件 -> 释放内存。**

```python
import json

input_file = 'large_dataset.jsonl'
output_file = 'cleaned_dataset.jsonl'

# 逐行读取和写入，内存占用极小（只有当前行的大小）
with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
  
    for line_number, line in enumerate(fin):
        try:
            # 1. 解析当前行
            data = json.loads(line)
          
            # ==========================================
            # 2. 在这里进行你的“编辑”操作
            # 例如：删除某个不需要的字段
            if 'meta_data' in data:
                del data['meta_data']
              
            # 例如：过滤掉文本太短的数据
            if len(data.get('text', '')) < 50:
                continue # 跳过这一行，不写入新文件
              
            # 例如：修改某个字段的值
            data['text'] = data['text'].replace('旧词', '新词')
            # ==========================================
          
            # 3. 将修改后的字典转回 JSON 字符串并写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
          
        except json.JSONDecodeError:
            print(f"警告: 第 {line_number} 行 JSON 格式错误，已跳过。")
            continue
```
*提示：跑全量数据前，先用前面提到的 `sample.jsonl` 测试这段脚本。*

---

### 3. 如果非要用图形界面 (GUI) 手动修改
如果你确实需要像在 VS Code 里那样手动滚动、查找并修改某几行，你必须使用专门为超大文件设计的文本编辑器。

*   **Windows 平台王者：EmEditor**
    *   **特点：** 专门为处理大数据和日志文件设计。它可以轻松打开 TB 级别的文件而不会 OOM。
    *   **原理：** 它自己实现了一套高效的内存管理和磁盘分页机制。
    *   **使用体验：** 打开几百GB的文件只需要几秒钟，可以直接按 `Ctrl+F` 搜索，手动修改并保存。
*   **Linux / macOS 平台：Vim / Neovim (需要配置)**
    *   Vim 可以打开大文件，但前提是禁用语法高亮、交换文件和撤销树等消耗内存的功能。不推荐新手尝试处理超过 10GB 的文件。

---

<details>
<summary><b>点击展开：4. 进阶处理（命令行高级工具与分布式库）</b></summary>

如果你需要进行复杂的过滤或快速的文本替换，可以使用以下工具：

**1. `sed` 和 `awk` (纯文本替换，极速)**
如果只是简单的字符串替换（比如全局替换某个敏感词），用 `sed` 是最快的，完全不需要解析 JSON：
```bash
# 将所有的 "old_word" 替换为 "new_word" 并输出到新文件
sed 's/old_word/new_word/g' dataset.jsonl > new_dataset.jsonl
```

**2. `jq` (命令行 JSON 处理神器)**
如果是依据 JSON 结构进行提取，用 `jq`。
```bash
# 提取所有 JSON 行中 key 为 "text" 的内容
cat dataset.jsonl | jq '.text' > only_texts.txt
```

**3. Hugging Face `datasets` 库 (用于机器学习工程)**
如果你是在做大模型预训练，推荐使用 HuggingFace 的 datasets 库，它的底层基于 Apache Arrow，使用内存映射（Memory-Mapping）技术，几百GB的数据也不会 OOM。
```python
from datasets import load_dataset

# 使用 streaming=True 开启流式加载，不占用内存
dataset = load_dataset("json", data_files="dataset.jsonl", split="train", streaming=True)

# 定义处理函数
def process_data(example):
    example["text"] = example["text"].upper()
    return example

# 映射处理
mapped_dataset = dataset.map(process_data)

# 遍历查看前 5 条
for i, data in enumerate(mapped_dataset):
    print(data)
    if i == 4:
        break
```
</details>

### 总结建议
1. 先用 `head` 提取 1000 行存成小文件。
2. 在 VS Code 中打开这个小文件，观察数据结构，编写并测试 Python 脚本。
3. 确认无误后，用 Python 脚本去**流式读取**处理那几百 GB 的大文件。