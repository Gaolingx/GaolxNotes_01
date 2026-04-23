# 只下载HuggingFace部分文件

### 🧑‍💻 User

huggingface hub如何只下载部分路径的文件而不是下载整个仓库？

---

### 🤖 Assistant

在使用 Hugging Face Hub 时，如果你只想下载仓库中的特定文件或特定文件夹（部分路径），而不想下载整个庞大的仓库，可以通过 Python API 或命令行工具（CLI）轻松实现。

以下是几种最常用的方法：

### 方法一：下载单个特定文件 (`hf_hub_download`)

如果你只需要仓库里的某**一个**具体文件，可以使用 `hf_hub_download`。

```python
from huggingface_hub import hf_hub_download

# 下载特定文件
file_path = hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",  # 仓库名称
    filename="v1-5-pruned-emaonly.safetensors", # 文件完整路径/名称
    local_dir="./my_models"                     # 保存在本地的目录（可选）
)

print(f"文件已下载到: {file_path}")
```
*注：如果文件在一个子目录中，`filename` 应该包含路径，例如 `filename="unet/diffusion_pytorch_model.bin"`。*

### 方法二：下载特定文件夹或按规则下载 (`snapshot_download`)

如果你想下载某个**特定文件夹**下的所有文件，或者只想下载**特定格式**（如 `.safetensors`）的文件，可以使用 `snapshot_download` 结合 `allow_patterns` 参数。

`allow_patterns` 支持 Glob 通配符语法。

```python
from huggingface_hub import snapshot_download

# 示例 1：只下载 "unet" 文件夹下的所有文件
snapshot_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    allow_patterns="unet/*",            # 只允许下载 unet 目录下的文件
    local_dir="./my_models/sd-unet"
)

# 示例 2：只下载特定扩展名的文件（如配置文件和 safetensors 权重）
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    allow_patterns=["*.json", "*.safetensors"], # 支持传入列表
    ignore_patterns=["*.bin"],                  # 排除特定文件（可选）
    local_dir="./my_llama"
)
```

---

<details>
<summary><b>方法三：使用命令行工具 (CLI) 实现部分下载</b></summary>

如果你更喜欢在终端/命令行中操作，可以使用官方提供的 `huggingface-cli`。

首先确保安装了 CLI 工具：
```bash
pip install -U "huggingface_hub[cli]"
```

**1. 下载单个文件：**
```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 unet/config.json --local-dir ./my_model
```

**2. 下载特定文件夹（使用 `--include`）：**
```bash
# 只下载 unet 目录及其内容
huggingface-cli download runwayml/stable-diffusion-v1-5 --include "unet/*" --local-dir ./my_model
```

**3. 下载特定格式文件：**
```bash
# 只下载 safetensors 和 json 文件
huggingface-cli download meta-llama/Llama-2-7b-hf --include "*.safetensors" "*.json" --local-dir ./my_llama
```
</details>

<details>
<summary><b>💡 额外提示：访问私有模型或需要授权的模型</b></summary>

如果你下载的模型是私有的（Private）或者是像 Llama 3 这样需要同意协议（Gated）的模型，你需要提供你的 Hugging Face Token。

**在 Python 中：**
添加 `token` 参数：
```python
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    allow_patterns="*.safetensors",
    token="hf_你的专属Token"  # 从 HuggingFace 设置中获取
)
```

**在 CLI 中：**
先运行登录命令，然后再执行下载命令：
```bash
huggingface-cli login
```
</details>