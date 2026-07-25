# hf cli获取token上传模型

### 🧑‍💻 User

hf cli如何获取hf token并上传模型？

---

### 🤖 Assistant

# Hugging Face CLI 获取 Token 并上传模型指南

## 一、获取 Hugging Face Token

### 1. 注册/登录 Hugging Face
- 前往 [huggingface.co](https://huggingface.co) 注册账号

### 2. 创建 Access Token
进入 **Settings → Access Tokens**，点击 **New token**：
- 选择权限：`write`（上传需要）或 `read`
- 保存生成的 token

### 3. 配置 Token 到本地环境

**方式一：命令行配置**
```bash
huggingface-cli login
# 按提示粘贴 token
```

**方式二：环境变量**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```
或添加到 `~/.bashrc` / `~/.zshrc` 永久生效。

**方式三：直接配置文件**
编辑 `~/.huggingface/token`，将 token 写入该文件。

---

## 二、上传模型到 Hugging Face

### 步骤 1：确保已安装并登录
```bash
pip install huggingface_hub
huggingface-cli login  # 输入你的 token
```

### 步骤 2：创建模型仓库（可选）
```bash
huggingface-cli repo create your-model-name --type model
```

### 步骤 3：上传模型

**方式一：使用 `upload` 命令（推荐）**
```bash
huggingface-cli upload your-username/your-model-name ./local_model_dir/ --repo-type model
```

**方式二：Python API**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./local_model_dir",
    repo_id="your-username/your-model-name",
    repo_type="model"
)
```

**方式三：Git 推送（传统方式）**
```bash
git lfs install
git clone https://huggingface.co/your-username/your-model-name
cd your-model-name
# 复制模型文件到该目录
cp -r /path/to/your/model/* .
git add .
git commit -m "Upload model"
git push
```

---

## 三、常用参数与技巧

| 参数 | 说明 | 示例 |
|------|------|------|
| `--repo-type` | 仓库类型（model/dataset/space） | `--repo-type model` |
| `--revision` | 分支名 | `--revision main` |
| `--include` | 包含的文件模式 | `--include "*.bin" "*.json"` |
| `--exclude` | 排除的文件模式 | `--exclude "*.pt"` |
| `--private` | 创建私有仓库 | `--private` |

**上传特定文件示例：**
```bash
# 只上传模型权重和配置文件
huggingface-cli upload your-username/your-model-name ./output/ \
    --include "pytorch_model.bin" "config.json" "tokenizer.json"
```

**查看上传状态：**
```bash
huggingface-cli whoami  # 确认登录状态
```

---

## 四、实际案例

```bash
# 完整流程
pip install huggingface_hub -q
huggingface-cli login --token hf_xxxxxxxxxxxx

# 创建仓库并上传
huggingface-cli repo create my-bert-model --type model --private
huggingface-cli upload my-username/my-bert-model ./bert_output/ \
    --include "*.bin" "config.json" "*.txt"
```

---

> **注意**：大文件（>5GB）建议使用 `huggingface-cli lfs-enable-largefiles` 启用大文件支持。上传过程中确保网络稳定，可通过 `--verbose` 参数查看详细日志。

---

### 🧑‍💻 User

对于已有的模型仓库且能access怎么上传？

---

### 🤖 Assistant

# 上传到已有可访问的模型仓库

既然已经能 access 该仓库，只需直接上传即可：

## 方法一：CLI 命令行（最推荐）

```bash
# 直接上传整个目录
huggingface-cli upload your-username/existing-repo ./local_model_dir/

# 或上传特定文件
huggingface-cli upload your-username/existing-repo ./local_model_dir/pytorch_model.bin
huggingface-cli upload your-username/existing-repo ./local_model_dir/config.json
```

**常用参数：**
```bash
# 排除缓存和日志文件
huggingface-cli upload your-username/existing-repo ./output/ \
    --exclude "*.cache" "*.log" ".git*" "__pycache__/*" \
    --repo-type model

# 指定分支（默认 main）
huggingface-cli upload your-username/existing-repo ./model/ \
    --revision v2.0
```

---

## 方法二：Python API（适合需要控制逻辑的场景）

```python
from huggingface_hub import HfApi

api = HfApi()

# 上传整个文件夹
api.upload_folder(
    folder_path="./model_output",
    repo_id="your-username/existing-repo",
    repo_type="model",
    ignore_patterns=["*.cache", "*.log", "*.pt"]  # 排除不需要的文件
)

# 或上传单个文件
api.upload_file(
    path_or_fileobj="./model_output/pytorch_model.bin",
    path_in_repo="pytorch_model.bin",
    repo_id="your-username/existing-repo",
    repo_type="model"
)
```

---

## 方法三：Git 方式（适合版本控制敏感场景）

```bash
# 克隆已有仓库
git clone https://huggingface.co/your-username/existing-repo
cd existing-repo

# 复制新模型文件
cp -r /path/to/new/model/* .

# 查看改动
git status

# 提交并推送
git add .
git commit -m "Upload updated model v2"
git push
```

**注意：** 如果之前不是用 git 方式提交的，可能需要先 pull 同步：
```bash
git pull origin main --allow-unrelated-histories
```

---

## 四、常见场景示例

### 场景 1：增量更新（只上传新文件）
```bash
# 只上传新增的 tokenizer 文件
huggingface-cli upload your-username/my-model \
    ./new_tokenizer/ --include "tokenizer*"
```

### 场景 2：覆盖特定文件
```bash
# 只更新 config.json
huggingface-cli upload your-username/my-model \
    ./updated_config/config.json \
    --path-in-repo config.json
```

### 场景 3：多文件分批上传
```bash
# 先上传权重文件
huggingface-cli upload your-username/my-model \
    ./model/pytorch_model.bin

# 再上传配置文件
huggingface-cli upload your-username/my-model \
    ./model/config.json ./model/tokenizer.json
```

### 场景 4：Python 脚本批量处理
```python
from huggingface_hub import HfApi, logging
import os

logging.set_verbosity_debug()  # 开启调试日志

api = HfApi()
repo_id = "your-username/existing-repo"
local_dir = "./model_output"

# 遍历文件夹上传
for file in os.listdir(local_dir):
    file_path = os.path.join(local_dir, file)
    if os.path.isfile(file_path):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model"
        )
```

---

## 五、验证上传结果

```bash
# 查看仓库文件结构
huggingface-cli list-repo-files your-username/existing-repo

# 或直接访问网页确认
# https://huggingface.co/your-username/existing-repo
```

---

> **重要提示：**
> - 确保 token 有 **write** 权限
> - 上传大文件时建议使用 `--verbose` 参数查看进度
> - 如果遇到 `HTTP 403` 错误，重新 `login` 再试
> - 模型文件超过 5GB 需启用 LFS：`huggingface-cli lfs-enable-largefiles`

---

### 🧑‍💻 User

--include arg可以使用通配符吗（例如*.safetensors）

---

### 🤖 Assistant

# 是的，`--include` 支持通配符！

你可以直接使用 `*.safetensors` 这样的通配符模式来匹配文件。

## 实际示例

### 1. 只上传 `.safetensors` 文件
```bash
huggingface-cli upload your-username/existing-repo ./local_model/ \
    --include "*.safetensors"
```

### 2. 上传所有 `.safetensors` 和 `.json` 文件
```bash
huggingface-cli upload your-username/existing-repo ./local_model/ \
    --include "*.safetensors" "*.json"
```

### 3. 排除特定格式，包含所有其他文件
```bash
# 上传除了 .bin 和 .pt 以外的所有文件
huggingface-cli upload your-username/existing-repo ./local_model/ \
    --exclude "*.bin" "*.pt"
```

### 4. 组合使用 include 和 exclude
```bash
# 只上传 safetensors，但排除 shard 文件
huggingface-cli upload your-username/existing-repo ./local_model/ \
    --include "*.safetensors" \
    --exclude "model-00001-of-*.safetensors"
```

## 通配符模式说明

| 模式 | 匹配规则 | 示例 |
|------|----------|------|
| `*.safetensors` | 所有 `.safetensors` 文件 | `model.safetensors` |
| `model-*.safetensors` | 以 `model-` 开头的 safetensors | `model-00001-of-00005.safetensors` |
| `**/*.json` | 所有子目录中的 `.json` 文件 | `subdir/config.json` |
| `*` | 所有文件 | `所有文件` |
| `?` | 匹配单个字符 | `model-?????.safetensors` |

## 完整示例：只上传 safetensors 权重
```bash
huggingface-cli upload your-username/my-model ./llama-output/ \
    --include "*.safetensors" \
    --repo-type model \
    --verbose
```

## Python API 同样支持
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./model_output",
    repo_id="your-username/my-model",
    repo_type="model",
    allow_patterns=["*.safetensors", "*.json"],  # 只上传这些
    ignore_patterns=["*.bin", "*.pt"]            # 排除这些
)
```

> **注意：** `--include` 和 `--exclude` 使用 glob 模式匹配，支持 `*`、`?`、`**` 等通配符。多个模式用空格分隔。

---

### 🧑‍💻 User

如果要移除/覆盖部分文件怎么做？

---

### 🤖 Assistant

# 移除和覆盖文件的操作方法

## 一、覆盖文件（最简单）

直接上传同名文件即可覆盖：

```bash
# 直接覆盖 config.json
huggingface-cli upload your-username/existing-repo ./new_files/config.json

# 覆盖多个文件
huggingface-cli upload your-username/existing-repo ./new_files/ \
    --include "config.json" "tokenizer.json"
```

## 二、删除/移除文件

### 方法 1：CLI 命令（推荐）
```bash
# 删除单个文件
huggingface-cli delete your-username/existing-repo config.json

# 删除整个目录
huggingface-cli delete your-username/existing-repo old_model/

# 删除指定分支的文件
huggingface-cli delete your-username/existing-repo model.bin --revision v1.0
```

### 方法 2：Python API
```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/existing-repo"

# 删除单个文件
api.delete_file(
    path_in_repo="old_model.bin",
    repo_id=repo_id,
    repo_type="model"
)

# 删除文件夹下的所有文件
api.delete_folder(
    path_in_repo="old_checkpoints/",
    repo_id=repo_id,
    repo_type="model"
)
```

### 方法 3：Git 方式
```bash
# 克隆仓库
git clone https://huggingface.co/your-username/existing-repo
cd existing-repo

# 删除文件
rm old_model.bin
git rm old_model.bin

# 或者 git rm 后删除（更推荐）
git rm -r old_checkpoints/

# 提交删除
git commit -m "Remove old model files"
git push
```

## 三、实用场景

### 场景 1：更新模型权重（先删后传）
```bash
# 1. 删除旧的 safetensors
huggingface-cli delete your-username/my-model model.safetensors

# 2. 上传新版本
huggingface-cli upload your-username/my-model ./new_model/model.safetensors
```

### 场景 2：批量删除特定文件
```bash
# Python 方式批量删除
api = HfApi()
repo_id = "your-username/my-model"

# 获取所有文件列表
files = api.list_repo_files(repo_id)

# 删除所有 .bin 文件
for file in files:
    if file.endswith('.bin'):
        api.delete_file(path_in_repo=file, repo_id=repo_id)
        print(f"Deleted: {file}")
```

### 场景 3：清空文件夹中的所有旧模型
```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/my-model"

# 列出并删除文件夹内容
folder_files = api.list_repo_files(repo_id)
for file in folder_files:
    if file.startswith("old_models/"):
        api.delete_file(path_in_repo=file, repo_id=repo_id)
```

### 场景 4：替换整个文件夹
```bash
# 1. 先删除旧的模型文件夹
huggingface-cli delete your-username/my-model model_weights/

# 2. 上传新的完整文件夹
huggingface-cli upload your-username/my-model ./new_weights/ \
    --include "*.safetensors" "*.json"
```

## 四、一键覆盖式上传（最省事）

如果你想完全替换某些文件而不关心旧文件：

```bash
# 直接上传会覆盖同名文件，不会删除不相关的文件
huggingface-cli upload your-username/my-model ./new_model_output/ \
    --include "*.safetensors" "config.json" "tokenizer.*"
```

如果**只想保留新上传的文件**，删除所有其他文件：

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/my-model"

# 获取当前所有文件
current_files = api.list_repo_files(repo_id)

# 指定要保留的文件
keep_files = [
    "config.json",
    "tokenizer.json",
    "model.safetensors"
]

# 删除不在保留列表中的文件
for file in current_files:
    if file not in keep_files:
        api.delete_file(path_in_repo=file, repo_id=repo_id)
        print(f"Removed: {file}")

# 然后再上传新文件
api.upload_folder(
    folder_path="./new_model",
    repo_id=repo_id
)
```

## 五、注意事项

> **重要：**
> - 删除操作**不可逆**，建议先确认文件列表
> - 使用 `huggingface-cli list-repo-files` 查看当前文件
> - Git 方式的删除可以回滚（通过 `git revert`）
> - CLI 删除需要 token 有 **write** 权限
> - 大文件删除后，存储空间会很快释放

---

### 🧑‍💻 User

举个例子 我想**删除**model-00001-of-00046.safetensors、model-00002-of-00046.safetensors...model-000046-of-00046.safetensors，
但是**保留**model-00001-of-00020.safetensors、model-00002-of-00020.safetensors、model-00003-of-00020.safetensors...model-000020-of-00020.safetensors，
需要如何编写通配符命令？

---

### 🤖 Assistant

# 删除特定编号范围的文件

这个需求因为通配符无法直接表达"46个分片"这种逻辑，需要用脚本方式处理。

## 方法一：Python 脚本（最灵活）

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/my-model"

# 获取所有文件
files = api.list_repo_files(repo_id)

# 删除 model-*-of-00046.safetensors
for file in files:
    if "of-00046.safetensors" in file:
        api.delete_file(path_in_repo=file, repo_id=repo_id)
        print(f"Deleted: {file}")
```

## 方法二：一次性生成删除列表

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/my-model"

# 生成所有 46 片文件的路径
files_to_delete = [
    f"model-{i:05d}-of-00046.safetensors" 
    for i in range(1, 47)  # 1 到 46
]

# 批量删除
for file in files_to_delete:
    try:
        api.delete_file(path_in_repo=file, repo_id=repo_id)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Failed to delete {file}: {e}")
```

## 方法三：用 Bash + CLI（循环删除）

```bash
# 删除所有 46 片的文件
for i in $(seq -w 1 46); do
    huggingface-cli delete your-username/my-model "model-${i}-of-00046.safetensors"
done
```

或者一行命令：
```bash
for i in {1..46}; do
    printf -v num "%05d" $i
    huggingface-cli delete your-username/my-model "model-${num}-of-00046.safetensors"
done
```

## 方法四：Git 方式批量删除

```bash
# 克隆仓库
git clone https://huggingface.co/your-username/my-model
cd my-model

# 删除所有 46 片的文件（保留 20 片的）
rm model-*-of-00046.safetensors

# 提交删除
git add .
git commit -m "Remove 46-shard files, keep 20-shard files"
git push
```

## 方法五：使用 include 反向思维（Git Bash 环境）

如果你在 Windows Git Bash 中，可以这样：

```bash
# 先列出要删除的文件看看对不对
ls model-*-of-00046.safetensors

# 确认无误后，用 git rm 删除
git rm model-*-of-00046.safetensors
git commit -m "Remove old 46-shard files"
git push
```

## 验证结果

删除后验证文件结构：
```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("your-username/my-model")

print("Remaining safetensors files:")
for f in sorted(files):
    if ".safetensors" in f and "model-" in f:
        print(f"  ✓ {f}")
```

预期输出：
```
✓ model-00001-of-00020.safetensors
✓ model-00002-of-00020.safetensors
...
✓ model-00020-of-00020.safetensors
```

> **总结：** 通配符无法表达"删除 46 片"这种复杂逻辑，需要借助循环或 Python 脚本。推荐使用**方法一**的 Python 脚本，简单可靠。