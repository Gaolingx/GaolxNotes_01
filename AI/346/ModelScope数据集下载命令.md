# ModelScope数据集下载命令

### 🧑‍💻 User

modelscope 下载数据集的命令（单文件/多文件/特定文件夹）是什么？

---

### 🤖 Assistant

在 ModelScope 中下载数据集主要使用 `modelscope` 命令行工具（需先安装：`pip install modelscope`）。
以下命令适用于 **ModelScope >= 1.9.0** 版本，支持按需下载单文件、多文件和特定文件夹。

---

### 1. 基础命令结构
```bash
modelscope download --dataset <数据集名称/ID> [选项] [文件路径...]
```
- `--dataset`：必选，指定数据集（例如 `damo/ms_dataset_test`）
- `--local-dir`：下载到指定的本地目录（省略则保存到当前目录下以数据集名命名的文件夹）
- `--revision`：指定版本或分支（可选）

---

### 2. 单文件下载
直接在数据集 ID 后写出**完整的远端文件路径**（包括子目录）：
```bash
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    README.md
```
如果文件位于子目录中，需写完整路径：
```bash
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    subdir/config.json
```

---

### 3. 多文件下载
用**空格**分隔多个文件路径即可：
```bash
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    README.md images/pic1.jpg labels/annotations.json
```

---

### 4. 特定文件夹（含所有文件）下载
使用 `--include` 配合 **glob 通配符**来匹配文件夹下的所有文件：
```bash
# 下载 images 文件夹下的所有文件（不递归子文件夹）
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    --include "images/*"

# 递归下载 images 文件夹下所有内容（含子文件夹）
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    --include "images/**"

# 同时匹配多个模式（可多次使用 --include）
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    --include "images/*" \
    --include "labels/*.txt"
```

如果下载时想**排除**某些文件/文件夹，可以结合 `--exclude`：
```bash
modelscope download --dataset damo/my_dataset \
    --local-dir ./data \
    --include "images/**" \
    --exclude "images/private/**"
```

---

### 5. 补充说明
- 如果只写 `--dataset` 而不接任何文件名或 `--include`，会**下载整个数据集**。
- 某些旧版本需要使用 `--repo-type dataset` 显式声明，等价写法：
  ```bash
  modelscope download --repo-type dataset damo/my_dataset README.md
  ```
- 查看完整帮助：`modelscope download --help`