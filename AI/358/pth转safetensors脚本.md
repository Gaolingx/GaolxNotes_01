# pth转safetensors脚本

### 🧑‍💻 User

编写一个pth转safetensors的脚本，要求走huggingface的hf导出接口

---

### 🤖 Assistant

以下是一份使用 Hugging Face 官方 `safetensors` 库将 `.pth` 权重文件转换为 `.safetensors` 格式的脚本，依赖最小且通用性强。

## 脚本 `convert_pth_to_safetensors.py`

```python
#!/usr/bin/env python3
"""
将 PyTorch 的 .pth (或 .bin) 权重文件转换为 safetensors 格式。
使用 Hugging Face 的 safetensors 库 (`safetensors.torch.save_file`) 完成导出。
"""

import argparse
import torch
from safetensors.torch import save_file


def convert(pth_path: str, safetensors_path: str) -> None:
    """加载 .pth 文件并将其张量保存为 .safetensors 格式。"""
    # 加载 checkpoint（始终加载到 CPU，避免显存浪费）
    checkpoint = torch.load(pth_path, map_location="cpu")

    # 从可能的数据结构中提取张量字典
    if isinstance(checkpoint, dict):
        # 常见情况：字典里包含 'state_dict' 或 'model' 键
        if "state_dict" in checkpoint:
            tensors = checkpoint["state_dict"]
        elif "model" in checkpoint:
            tensors = checkpoint["model"]
        else:
            # 假设整个字典本身就是 {name: tensor} 的映射
            tensors = checkpoint
    else:
        # 可能是完整模型对象，取其 state_dict
        tensors = checkpoint.state_dict()

    # 使用 Hugging Face 的 safetensors 导出接口保存
    save_file(tensors, safetensors_path)
    print(f"✅ 转换完成：{safetensors_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将 .pth 权重转换为 .safetensors (Hugging Face safetensors 接口)"
    )
    parser.add_argument("input", help="输入的 .pth 或 .bin 文件路径")
    parser.add_argument("output", help="输出的 .safetensors 文件路径")
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
```

---

## 使用方式

1. **安装依赖**

```bash
pip install torch safetensors
```

2. **运行转换**

```bash
python convert_pth_to_safetensors.py model.pth model.safetensors
```

如果输入的 `.pth` 文件是完整的训练 checkpoint（例如 `{'epoch': 10, 'state_dict': {...}}`），脚本会自动提取 `state_dict`。

---

## 关键点说明

- **Hugging Face 导出接口**：使用 `safetensors.torch.save_file()`，这是 Hugging Face 官方维护的 safetensors 库提供的保存函数，速度快且安全。
- **自动键检测**：兼容常见的 PyTorch 保存格式：
  - 纯权重字典 `{name: tensor}`
  - 包含 `state_dict` / `model` 键的训练 checkpoint
  - 完整的 `nn.Module` 模型对象
- **CPU 加载**：始终使用 `map_location="cpu"`，避免在 GPU 上浪费显存，也兼容无 GPU 环境。
- **元数据保留**：`save_file` 可以传递 `metadata` 参数，如需保存额外信息（如 epoch）可自行扩展。

---

## 扩展：转换整个 Hugging Face 模型目录

如果你的 `.pth` 对应一个 Hugging Face `transformers` 模型，并且希望连带 `config.json` 等文件一同转换成 safetensors 格式，可以在加载完整模型后调用 `model.save_pretrained(..., safe_serialization=True)`，示例如下：

```python
from transformers import AutoModel, AutoConfig

model = AutoModel.from_pretrained("path/to/pth_dir")
model.save_pretrained("path/to/output_dir", safe_serialization=True)
```

但以上脚本聚焦于单个权重文件的轻量转换，适用于大多数场景。