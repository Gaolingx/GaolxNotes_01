# 使用Conda创建虚拟环境运行7B模型

### 🧑‍💻 User

如何用conda创建虚拟环境，安装vllm运行一个7b模型？

---

### 🤖 Assistant

要使用 Conda 创建虚拟环境并安装 vLLM 运行 7B 模型，请按以下步骤操作：

---

### **步骤 1：创建 Conda 虚拟环境**
```bash
conda create -n vllm_env python=3.9 -y  # 推荐 Python 3.9
conda activate vllm_env
```

---

### **步骤 2：安装 PyTorch (CUDA 版本)**
```bash
# 根据你的 CUDA 版本选择安装命令 (示例为 CUDA 11.8)
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

> 查看 CUDA 版本：`nvidia-smi`（右上角显示版本，如 `12.x` 需对应安装 `cu121`）

---

### **步骤 3：安装 vLLM**
```bash
pip install vllm  # 自动安装依赖
```
> 验证安装：`python -c "from vllm import LLM; print('Success')"`

---

### **步骤 4：运行 7B 模型**
#### **方式 1：使用 Hugging Face 模型（如 `mistralai/Mistral-7B-v0.1`）**
```python
from vllm import LLM, SamplingParams

# 初始化模型和采样参数
llm = LLM(model="mistralai/Mistral-7B-v0.1")  # 首次运行自动下载模型
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# 输入提示
prompts = ["Explain what AI is in 3 sentences:"]
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}\n")
```

#### **方式 2：使用本地模型**
1. 提前下载模型到本地：
   ```bash
   huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./mistral-7b
   ```
2. 修改代码加载路径：
   ```python
   llm = LLM(model="./mistral-7b")  # 指向本地目录
   ```

---

### **步骤 5：执行脚本**
```bash
python run_vllm.py  # 保存代码为 run_vllm.py
```

---

### **常见问题解决**
1. **CUDA 版本不匹配**：
   - 确保 `nvidia-smi` 显示的驱动版本支持安装的 CUDA 版本。
   - 重新安装匹配的 PyTorch（如 CUDA 11.8 用 `cu118`）。

2. **内存不足**：
   - 7B 模型需要约 **15GB GPU 显存**。
   - 启用量化（修改 `LLM` 初始化）：
     ```python
     llm = LLM(model="mistralai/Mistral-7B-v0.1", quantization="awq")  # 需提前转换模型
     ```

3. **下载模型失败**：
   - 使用国内镜像源：
     ```bash
     export HF_ENDPOINT=https://hf-mirror.com
     huggingface-cli download --resume-download mistralai/Mistral-7B-v0.1 --local-dir ./mistral-7b
     ```

---

### **完整示例代码**
```python
# run_vllm.py
from vllm import LLM, SamplingParams

# 配置模型和生成参数
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",  # 或本地路径 "./mistral-7b"
    tensor_parallel_size=1,            # 多 GPU 并行数量
)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)

# 生成文本
prompts = [
    "法国的首都是哪里？",
    "How to bake a chocolate cake?",
]
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {output.prompt}")
    print(f"Result {i+1}: {output.outputs[0].text.strip()}\n{'='*50}")
```

---

### **备注**
- **硬件要求**：NVIDIA GPU（推荐显存 ≥ 16GB），需安装 NVIDIA 驱动。
- **模型选择**：替换 `mistralai/Mistral-7B-v0.1` 为其他 7B 模型（如 `meta-llama/Llama-2-7b-chat-hf`，需登录 Hugging Face）。
- **性能优化**：多卡用户可增加 `tensor_parallel_size`（如 2 卡设为 `2`）。

通过以上步骤，即可在 Conda 虚拟环境中高效运行 7B 模型。