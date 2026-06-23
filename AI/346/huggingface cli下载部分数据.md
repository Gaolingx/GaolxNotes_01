# huggingface cli下载部分数据

### 🧑‍💻 User

huggingface cli如何只下载部分数据？

---

### 🤖 Assistant

使用 `huggingface-cli` 下载部分数据时，需要理解一个核心概念：**CLI 工具是基于“文件（File）”级别操作的**。你不能用 CLI 只下载一个文件的“前 100 行”，但你可以通过**指定文件名**或**使用通配符**来只下载特定的文件（例如只下载某个分片或只下载验证集）。

以下是具体的操作方法：

### 1. 下载指定的单个文件
如果你明确知道只需要数据集中的某一个特定文件，可以直接在命令中指定文件名。

```bash
huggingface-cli download <dataset_repo_id> <file_name> --repo-type dataset --local-dir <local_path>
```

**示例：** 只下载 `wikitext` 数据集中的特定 Parquet 文件：
```bash
huggingface-cli download wikitext wikitext-103-v1/train-00000-of-00002.parquet --repo-type dataset --local-dir ./my_data
```

### 2. 使用通配符过滤文件 (`--include` 和 `--exclude`)
如果数据集包含成百上千个文件，你可以使用 `--include` 或 `--exclude` 参数结合 Glob 模式（通配符）来按规则下载部分文件。

**只下载特定格式的文件（如 .parquet）：**
```bash
huggingface-cli download tiiuae/falcon-refinedweb --repo-type dataset --include "*.parquet" --local-dir ./data
```

**只下载验证集 (Validation split)：**
假设文件命名中包含 `val` 或 `validation`：
```bash
huggingface-cli download <dataset_repo_id> --repo-type dataset --include "*val*" --local-dir ./data
```

**排除特定文件（例如排除大体积的 .bin 或 .safetensors 文件）：**
```bash
huggingface-cli download <dataset_repo_id> --repo-type dataset --exclude "*.safetensors" --local-dir ./data
```

### 3. 下载特定分支或版本 (`--revision`)
如果数据集有多个分支（例如不同语言版本放在不同分支），可以通过 `--revision` 指定。

```bash
huggingface-cli download <dataset_repo_id> --repo-type dataset --revision <branch_name> --local-dir ./data
```

---

<details>
<summary><b>💡 进阶补充：如果你想精确控制“行数”（需要使用 Python 代码）</b></summary>

如果你面临的情况是：**数据集只有一个巨大的文件，而你只想下载其中的前 10%，或者只读取前 1000 条数据。** 
这种情况下，`huggingface-cli` 无法做到，你需要使用 Python 的 `datasets` 库。

**方法 A：使用切片语法下载部分（Slicing）**
```python
from datasets import load_dataset

# 只下载/加载训练集的前 10%
dataset = load_dataset("imdb", split="train[:10%]")

# 只下载/加载前 1000 行
dataset = load_dataset("imdb", split="train[:1000]")
```

**方法 B：使用流式读取（Streaming）**
流式读取不会将整个数据集下载到硬盘，而是用多少下多少，非常适合超大型数据集。
```python
from datasets import load_dataset

# 开启 streaming
dataset = load_dataset("c4", "en", split="train", streaming=True)

# 仅获取前 10 条数据
for i, row in enumerate(dataset):
    print(row)
    if i == 9:
        break
```
</details>

**注意：** 如果你下载的是需要授权许可的数据集（Gated Dataset），请确保你已经在网页端同意了协议，并在终端运行过 `huggingface-cli login` 进行登录授权。