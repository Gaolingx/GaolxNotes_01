# HuggingFace命令行上传数据集指南

### 🧑‍💻 User

hf cli如何创建一个dataset并上传目录所有文件？

---

### 🤖 Assistant

# 使用 HF CLI 创建数据集并上传整个目录

## 1. 安装与登录

```bash
pip install -U huggingface_hub
hf auth login          # 新版命令
# 或者（旧写法，等价）
huggingface-cli login
```

粘贴你的 Hugging Face Token（可在 https://huggingface.co/settings/tokens 创建）。

## 2. 创建数据集仓库

```bash
hf repo create my-dataset --type dataset
```

- 会创建 `https://huggingface.co/datasets/<你的用户名>/my-dataset`
- 如果是组织仓库：`hf repo create my-org/my-dataset --type dataset`

## 3. 上传目录中的所有文件

```bash
hf upload <用户名>/my-dataset ./本地目录 . --repo-type=dataset
```

示例：

```bash
hf upload alice/my-dataset ./data . --repo-type=dataset
```

- 第一个参数 `./data` 是本地目录
- 第二个参数 `.` 表示上传到仓库根目录（也可写成子目录名，如 `images/`）
- `--repo-type=dataset` 指明这是数据集仓库（重要，默认是 model）

## 4. 常用选项

```bash
# 只上传某些文件（glob 匹配）
hf upload alice/my-dataset ./data . --repo-type=dataset --include="*.json"

# 排除某些文件
hf upload alice/my-dataset ./data . --repo-type=dataset --exclude="*.tmp" --exclude=".DS_Store"

# 自定义提交信息
hf upload alice/my-dataset ./data . --repo-type=dataset --commit-message="add raw data v2"
```

## 备选方案

### 旧版命令（`huggingface-cli`，仍可用）

```bash
huggingface-cli repo create my-dataset --type dataset
huggingface-cli upload alice/my-dataset ./data . --repo-type dataset
```

### 传统 Git 方式（适合需要版本管理/大仓库）

```bash
git clone https://huggingface.co/datasets/alice/my-dataset
cd my-dataset
cp -r /本地/数据目录/* .
git add .
git commit -m "add dataset files"
git push
```

### Python API（脚本自动化）

```python
from huggingface_hub import HfApi

api = HfApi()

# 创建仓库（已存在会报错，可用 exist_ok=True 忽略）
api.create_repo("my-dataset", repo_type="dataset", exist_ok=True)

# 上传整个目录
api.upload_folder(
    folder_path="./data",            # 本地目录
    repo_id="alice/my-dataset",
    repo_type="dataset",
    path_in_repo=".",                # 上传到仓库根目录
    commit_message="upload all files",
)
```

## 几点说明

| 事项 | 说明 |
|------|------|
| `--repo-type` | 必须为 `dataset`，否则会当成模型仓库创建/上传 |
| 大文件（>10MB） | 自动走 Git LFS 存储，无需手动处理 |
| 已存在文件 | 会自动覆盖（相同路径）；`hf upload` 是按文件夹同步 |
| 私有仓库 | 创建时加 `--private`：`hf repo create my-dataset --type dataset --private` |
| 空目录 | Git 不追踪空文件夹，空目录不会被上传 |

如果你想实现“目录内容变化后增量同步”，`hf upload` 本身就会跳过未变动的文件（按内容哈希去重），可以放心重复执行。