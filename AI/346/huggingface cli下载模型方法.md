# huggingface cli下载模型方法

### 🧑‍💻 User

hf hub cli怎么下载模型？

---

### 🤖 Assistant

使用 Hugging Face Hub CLI 下载模型非常方便，支持断点续传、并行下载以及指定文件下载等功能。以下是详细的使用指南：

### 1. 安装与更新 CLI

首先，你需要确保安装了最新版本的 `huggingface_hub` Python 库，其中包含了 CLI 工具。

```bash
pip install -U "huggingface_hub[cli]"
```

### 2. 基本下载命令

使用 `huggingface-cli download` 命令下载模型。如果不指定本地目录，模型默认会下载到 Hugging Face 的缓存目录（通常是 `~/.cache/huggingface/hub`）。

```bash
huggingface-cli download <模型ID>
```

**示例：**下载 `bert-base-uncased` 模型并保存到当前目录下的 `models/bert-base-uncased` 文件夹中。

```bash
huggingface-cli download bert-base-uncased --local-dir models/bert-base-uncased
```

### 3. 常用参数详解

<details>
<summary><b>点击展开高级下载参数</b></summary>

*   **`--local-dir`**: 指定下载文件保存的本地路径。如果不使用此参数，文件将被存放在缓存中并创建符号链接。
*   **`--include`**: 使用 glob 模式筛选需要下载的特定文件。
    *   *示例:* 仅下载 `.safetensors` 权重文件和 `config.json`
        ```bash
        huggingface-cli download bert-base-uncased --include "*.safetensors" "config.json" --local-dir ./bert-model
        ```
*   **`--exclude`**: 排除符合特定模式的文件。
    *   *示例:* 排除 `.bin` 文件（通常用于排除 pytorch 原生格式的权重，只保留 safetensors）
        ```bash
        huggingface-cli download bert-base-uncased --exclude "*.bin" --local-dir ./bert-model
        ```
*   **`--resume-download`**: 虽然新版本的 CLI 默认支持断点续传，但在遇到网络问题时可以尝试显式加上此参数。
*   **`--token`**: 如果要下载受限或私有模型，需要提供 Access Token。
    *   *示例:*
        ```bash
        huggingface-cli download meta-llama/Llama-2-7b-hf --token hf_YOUR_TOKEN_HERE --local-dir ./llama
        ```
*   **`--revision`**: 指定要下载的模型版本（分支名、标签名或 commit hash）。

</details>

### 4. 网络问题与镜像站配置（针对国内网络）

如果你在中国大陆地区，直接下载可能会遇到连接超时或速度极慢的问题。强烈建议在使用 CLI 之前设置环境变量 `HF_ENDPOINT` 指向国内的镜像站（如 `hf-mirror.com`）。

**在 Linux/macOS 中设置镜像：**

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir ./bert-base-uncased
```

**在 Windows PowerShell 中设置镜像：**

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download bert-base-uncased --local-dir ./bert-base-uncased
```

### 5. 使用简写命令 `hf` (可选)

在较新版本的 `huggingface_hub` 中，除了 `huggingface-cli`，你也可以使用更简短的 `hf` 命令，功能完全相同：

```bash
hf download bert-base-uncased --local-dir ./bert-base-uncased
```

### 总结建议

如果你希望将模型直接下载到指定的文件夹以便于管理和加载，推荐使用以下组合命令模板：

```bash
# 设置镜像 (可选)
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型并指定目录
huggingface-cli download <模型ID> --local-dir <本地保存路径>
```