# Transformers原生支持量化模型加载

### 🧑‍💻 User

hf transformers from pretrained接口原生支持哪些量化模型加载？（awq/gptq/hqq/bnb/SmoothQuant/mxfp8/nvfp4...）

---

### 🤖 Assistant

## Hugging Face Transformers `from_pretrained` 原生支持的量化加载

根据最新版 (main / v5.14.0) 官方文档，以下是针对你提到的几种量化方法的结论：

### 逐一回答

| 量化方法 | `from_pretrained` 原生支持？ | 说明 |
|:---|:---:|:---|
| **AWQ** | ✅ 支持 | 需 `pip install autoawq`，通过 `quantization_config=AwqConfig(...)` 或自动检测 `config.json` 中的 `quant_method: "awq"` |
| **GPTQ** | ✅ 支持 | 后端已从 AutoGPTQ 迁移到 **GPT-QModel**（`pip install gptqmodel`），通过 `GPTQConfig` 传递。支持 Marlin 内核加速 |
| **HQQ** | ✅ 支持 | 需 `pip install hqq`，通过 `HqqConfig` 传递。支持 1~8 bit，无需校准数据，兼容 torch.compile |
| **bitsandbytes (bnb)** | ✅ 支持 | 需 `pip install bitsandbytes`，通过 `BitsAndBytesConfig` 传递。支持 LLM.int8() (8-bit) 和 QLoRA/NF4 (4-bit)，支持 GPU/CPU/XPU/HPU |
| **MXFP4** | ✅ 支持 | 内置支持，自动从 `kernels-community/triton_kernels` 拉取 Triton 内核。**当前仅限 OpenAI GPT-OSS 120B/20B 模型** |
| **SmoothQuant** | ❌ 不支持 | 不在 Transformers 原生支持的量化方法列表中。通常通过 TensorRT-LLM、vLLM 等框架使用 |
| **nvfp4** | ❌ 不支持 | 不在 Transformers 量化文档中。NVFP4 是 NVIDIA Blackwell 的专有格式，尚未纳入 Transformers 生态 |

---

### 完整原生支持列表（共 22 种）

根据官方 Overview 页面的 "🤗Transformers Support" 列，以下量化方法均可通过 `from_pretrained(... quantization_config=...)` 原生加载：

| 方法 | Bit 范围 | On-the-fly | 需额外安装 |
|:---|---:|:---:|:---|
| **AQLM** | 1/2 | ❌ | `aqlm` |
| **AutoRound** | 2/3/4/8 | ❌ | `auto-round` |
| **AWQ** | 4 | ❌ | `autoawq` |
| **BitNet** | 1.58 | ❌ | 内置 |
| **bitsandbytes** | 4/8 | ✅ | `bitsandbytes` |
| **compressed-tensors** | 1~8 | ❌ | `compressed-tensors` |
| **EETQ** | 8 | ✅ | `eetq` |
| **FBGEMM_FP8** | 8 | ✅ | 内置 (PyTorch) |
| **Fine-grained FP8** | 8 | ✅ | 内置 |
| **Four Over Six** | 4 | ✅ | `fouroversix` |
| **FP-Quant** | 4 | ✅ | `fpquant` |
| **GPTQ (GPT-QModel)** | 2/3/4/8 | ❌ | `gptqmodel` |
| **HIGGS** | 2/4 | ✅ | `flute` |
| **HQQ** | 1~8 | ✅ | `hqq` |
| **Metal (Apple)** | 2/4/8 | ✅ | 内置 (MLX) |
| **MXFP4** | 4 | ✅ | 内置 (Triton kernels) |
| **Quanto** | 2/4/8 | ✅ | `optimum-quanto` |
| **Quark** | 2/4/6/8/9/16 | ❌ | `quark` |
| **torchao** | 4/8 | ✅ | `torchao` |
| **SpQR** | 3 | ❌ | `spqr` |
| **VPTQ** | 1~8 | ❌ | `vptq` |
| **SINQ** | 2/3/4/6/8 | ✅ | `sinq` |

> **注**：GGUF 在表中标记为 "See Notes"（部分支持），需通过 `llama.cpp` 路径加载，并非标准 `from_pretrained` 流程。

---

### 典型加载模式

所有原生支持的方法都遵循统一接口，以 AWQ 为例：

```python
from transformers import AutoModelForCausalLM, AwqConfig

# 方式1：自动检测 config.json 中的 quantization_config
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    device_map="auto"
)

# 方式2：显式指定 quantization_config
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    quantization_config=AwqConfig(bits=4, do_fuse=True),
    device_map="auto"
)
```

识别模型是否量化：查看 `config.json` 中是否有 `"quantization_config": {"quant_method": "xxx"}` 字段。