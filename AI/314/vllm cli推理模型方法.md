# vllm cli推理模型方法

### 🧑‍💻 User

vllm cli如何推理模型

---

### 🤖 Assistant

**vLLM 主要通过两种方式进行模型推理：离线 Python API（批量推理最常用）和 CLI 启动 OpenAI 兼容服务（在线服务）。**

### 1. 安装
```bash
pip install vllm
```
需要 NVIDIA GPU + 兼容 CUDA。安装后可直接使用 `vllm` 命令。

---

### 2. 离线推理（推荐，无需启动服务）
使用 `LLM` 类直接加载模型并生成：

```python
from vllm import LLM, SamplingParams

# 加载模型（支持 HuggingFace 模型名或本地路径）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",   # 或 "/path/to/local/model"
    # tensor_parallel_size=2,            # 多卡张量并行
    # dtype="auto",                      # 或 "float16"/"bfloat16"
    # trust_remote_code=True,            # 部分模型需要
    # gpu_memory_utilization=0.9,
    # max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    # stop=["<|im_end|>", "</s>"] 等
)

prompts = [
    "你好，请介绍一下自己。",
    "解释一下什么是注意力机制。"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 50)
```

**说明**：
- 支持批量 prompts，吞吐量高。
- 首次运行会自动下载模型（可设置 `HF_ENDPOINT` 使用镜像）。
- 本地模型直接写路径即可。

---

### 3. CLI 启动服务（在线推理 / OpenAI 兼容）
这是真正的 **vLLM CLI** 用法：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 1
```

常用参数：
- `--tensor-parallel-size N`：多卡并行
- `--quantization awq` / `gptq` / `fp8` 等（模型需支持）
- `--enforce-eager`：关闭 CUDA Graph（调试时常用）
- `--trust-remote-code`
- `--served-model-name my-model`：自定义返回的模型名
- `--api-key token-xxx`：可选鉴权
- 本地模型：`vllm serve /path/to/model`

查看完整参数：
```bash
vllm serve --help
```

启动后服务地址默认为 `http://localhost:8000`。

---

### 4. 调用服务进行推理

#### 方式 A：curl（Chat 推荐）
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手。"},
      {"role": "user", "content": "你好，介绍一下 vLLM"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

Completions 接口示例：
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

#### 方式 B：OpenAI Python 客户端（最方便）
```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"          # 没设置 --api-key 时随便填
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "用一句话解释 Transformer"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
```

---

### 实用技巧
- **指定 GPU**：`CUDA_VISIBLE_DEVICES=0,1 vllm serve ...`
- **需要登录的模型**（如 Llama）：`export HF_TOKEN=hf_xxx`
- **国内下载慢**：`export HF_ENDPOINT=https://hf-mirror.com`
- **查看服务状态**：`curl http://localhost:8000/v1/models`
- **性能相关**：适当调高 `--gpu-memory-utilization`（默认 0.9），根据显存调整 `--max-model-len`。
- 量化模型示例：`vllm serve TheBloke/xxx-AWQ --quantization awq`

---

**总结推荐**：
- 做批量/离线评测/脚本推理 → 用 `LLM` + `generate`（离线方式）。
- 做 API 服务、对接前端/其他工具 → 用 `vllm serve` + OpenAI 兼容接口。

官方文档：https://docs.vllm.ai/

如果需要针对特定模型（如 Qwen、Llama、DeepSeek）、多卡配置、量化或报错排查，可以告诉我具体场景。