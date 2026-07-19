# pth转safetensors最简脚本

### 🧑‍💻 User

编写一份符合huggingface的pth转safetensors脚本，要求最简实现，直接加载pth然后走hf的模型save接口

---

### 🤖 Assistant

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最简 PTH → Safetensors 转换脚本
直接 torch.load 加载 pth，然后调用 Hugging Face 模型的 save_pretrained 接口
"""

import os
import sys
import torch


def convert(pth_path: str, output_dir: str):
    print(f"Loading {pth_path} ...")
    model = torch.load(pth_path, map_location="cpu")

    if not hasattr(model, "save_pretrained"):
        raise TypeError(
            "加载结果没有 save_pretrained 方法。\n"
            "请确保 .pth 里保存的是完整的 transformers 模型实例（torch.save(model, ...)），\n"
            "而不是 state_dict。若是 state_dict，请先实例化对应模型再 model.load_state_dict()。"
        )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving to {output_dir} (safe_serialization=True) ...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pth2safetensors.py <model.pth> <output_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
```

### 使用说明
```bash
python pth2safetensors.py model.pth ./output_model
```

### 前提
1. `.pth` 必须是用 `torch.save(model, "model.pth")` 保存的**完整模型对象**（且该模型是 `transformers.PreTrainedModel` 子类）。
2. 运行前需要能正确 import 出模型对应的类（否则 `torch.load` 反序列化会失败）。
3. 依赖：`torch`、`transformers`、`safetensors`。

转换完成后，`output_dir` 下会生成 `model.safetensors`（以及 `config.json` 等 HF 标准文件），可直接被 `from_pretrained` 加载。