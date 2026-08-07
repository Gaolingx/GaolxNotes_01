# PyTorch model.generate加速方案

### 🧑‍💻 User

pytorch native的model.genrate速度很慢怎么办？

---

### 🤖 Assistant

`model.generate` 慢通常不是“PyTorch 本身”的问题，而是**默认参数、硬件利用不充分、解码策略**等原因。以下是系统性的提速思路，从低代价到高收益排列。

---

## 1. 先检查这 5 个常见拖慢元凶

### ① 没有使用 `past_key_values` (KV Cache)
HuggingFace `generate` 默认会启用，但如果你自己写了循环或传了奇怪参数，可能被关掉。没有 KV Cache 会导致每个 token 都重新计算全序列，速度呈 $O(N^2)$ 增长。

**验证/修复：**
```python
outputs = model.generate(
    input_ids,
    use_cache=True,          # 确保为 True
    ...
)
```

---

### ② 使用了 Eager attention / 没开 Flash Attention
如果你的 transformers 版本较新 + PyTorch 2.0+，Flash Attention 可立即减少 30%-70% 的推理时间，尤其在长序列下。

**启用方法：**
```python
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",   # 强制使用 Flash Attention
    device_map="auto",
)
```

> 注意：需要 `pip install flash-attn --no-build-isolation`，且 GPU 架构 ≥ Ampere (GTX 30 系列)。

---

### ③ 生成参数过于“贪心”或“重复计算”
- `num_beams > 1` 时每条 beam 都要维护一段序列，beam search 比 greedy/sample 慢数倍。
- **尽量使用 `do_sample=True` + `temperature` 代替 `num_beams`**，或者用 `num_beams=1`。
- 如果只是想要长度，`min_length` / `max_new_tokens` 设得太大也会白白多跑很多步。

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=128,          # 尽可能小
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_beams=1,                 # 单条搜索最快
)
```

---

### ④ `pad_token_id` 未设置，导致逐条处理
当你 `model.generate(batch_tokens)` 时，如果 `tokenizer.pad_token` 是 None，模型可能跑在一条一条的循环里。

```python
tokenizer.pad_token = tokenizer.eos_token   # 或新增一个 pad_token
```

**观察是否真正批处理：** 如果 GPU 利用率很低 → 大概率在串行。

---

### ⑤ 模型类型 / 精度问题
- 运行在 **FP32** 上会比 FP16/BF16 慢 2~5 倍。
- CPU 推理极慢（除非用 `llama.cpp` 类方案），GPU 几乎必有。

```python
model = model.half().to("cuda")   # 确保 FP16 + GPU
```

---

## 2. 中级优化：让每步都跑满

### `torch.compile` – 简单一行，通常提速 20%-40%
PyTorch 2.0+ 的 `torch.compile` 对 Transformer decode 特别有效。

```python
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
# 或只包装模型主体：
# model = torch.compile(model, mode="reduce-overhead")
```

第一次预编译需要热身，之后的 `generate()` 会快很多。

---

### 使用 `StaticCache` (transformers ≥ 4.43)
默认的 `DynamicCache` 会在每一步动态拼接 KV，存在显存碎片和开销。静态缓存预分配固定长度，减少运行时开销。

```python
from transformers import StaticCache

model.generation_config.cache_implementation = "static"
# 或者
outputs = model.generate(
    input_ids,
    cache_implementation="static",
    max_cache_length=256,   # 与你 max_new_tokens 匹配
)
```

---

### 关闭不需要的 logits 计算（如果你只要生成 token）
如果你不需要每个位置的 logits 做后处理，可以让 generate 直接返回序列。

`model.generate` 默认可能计算全词表 logits，你用 `output_scores=False` 可省一些算力，但默认就是不返回，保持 `False` 即可。

---

### 使用 `assistant_model` / 投机采样（Speculative Decoding）
当需要生成长文本时，速度可提升 2-3 倍，但需要一个小草稿模型。

```python
outputs = model.generate(
    input_ids,
    assistant_model=draft_model,   # 小模型
    max_new_tokens=256,
)
```

> 若没有现成小模型，可跳过。

---

## 3. 进阶：脱离 HuggingFace 的 `generate` 循环瓶颈

HuggingFace `generate` 虽然方便，但每一步都有很多 Python 逻辑（采样、停止条件等），在 batch=1 时 GPU 利用率依然不高。极致优化需要：

### `torch.nn.functional.scaled_dot_product_attention` + 手写 decode loop
手动维护 KV Cache 并使用 `SDPA`，可以完全消除框架开销。示例框架：

```python
past_key_values = None
for step in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_ids, past_key_values=past_key_values, use_cache=True
        )
    logits, past_key_values = outputs.logits, outputs.past_key_values
    next_token = sample(logits[:, -1, :])
    input_ids = next_token.unsqueeze(1)
    # 拼接输出并检测 eos
```

很多推理框架（vLLM, TGI）就是这么做的，只是它们加上了 continuous batching、PagedAttention 等。

---

## 4. 硬件与环境检查

- **CUDA >= 11.8、PyTorch >= 2.1**，否则 Flash Attention 和 `torch.compile` 不可用。
- **NVIDIA A10/A100/4090** 对比 **T4/V100**，同样是 FP16，显存带宽差异巨大。
- 用 `nvidia-smi` 看 **GPU 利用率**，如果低于 60%，说明受限于 CPU / 数据加载 / Python loops。

---

## 5. 如果仍然慢：该换专用推理引擎了

当你需要高吞吐或低延迟时，PyTorch 原生 generate 很难击败：

| 工具 | 优势 | 是否仍保持 PyTorch 原生？ |
|------|------|---------------------------|
| **vLLM** | PagedAttention, continuous batching | 内部基于 PyTorch，但非 `generate()` |
| **TensorRT-LLM** | 极致 Kernel 优化，最高吞吐 | 需要转模型 |
| **llama.cpp** | CPU/GPU hybrid，低资源场景 | C++，与 PyTorch 无关 |
| **HF Optimum + ONNX Runtime** | 图优化，稳定提升 | 导出 ONNX |

如果项目允许，**vLLM 一行代码替换**即可获得 10x 以上的并发吞吐：

```python
from vllm import LLM, SamplingParams
llm = LLM(model="your-model")
outputs = llm.generate(["你的 prompt"], SamplingParams(temperature=0.8))
```

---

## 总结优先级

1. ✅ 开启 `use_cache`，用 Flash Attention 2，换 FP16，设 `pad_token`。
2. ✅ 生成参数：`num_beams=1`，`max_new_tokens` 尽量小。
3. ✅ 加 `torch.compile` + `StaticCache`。
4. ✅ 如果对延迟要求极高，手写 decode 循环或上 vLLM。

这些组合下来，通常可以从“一秒几个 token”提升到“一秒几十甚至上百 token”。

---

### 🧑‍💻 User

但是根据torch profiler发现绝大多数时间都花在了torch的调度开销而不是计算/访存，怎么办？

---

### 🤖 Assistant

你看到的「PyTorch 调度开销占大头」非常典型，尤其在**小模型、batch=1、短序列**时，每一步的计算量太小，GPU kernel 启动（CPU → GPU）的固定开销就凸显出来了。
好消息是：这个问题**完全可以解决**，而且很多方案都是“一行代码 + 一点样板”就能兑现。

---

## 1. 先定位你的调度开销来自哪里

在 profiler 的 trace 里重点关注：

- **大量小 kernel**：一次 forward 调了数十上百个 kernel（不用 FlashAttention 会更糟）
- **PyTorch 的 host 端函数调用**：原生 Python 循环 + `model(...)` 的 dispatcher 开销
- **CPU 与 GPU 间的隐式同步**：比如 `print(logits)` 引起的 `cudaDeviceSynchronize`

**原则**：要么减少 kernel 数量，要么把 kernel launch 开销一次性“录制”掉。

---

## 2. 最小成本的提速：`torch.compile` + `StaticCache`

`torch.compile` 不仅能融合算子，还能在 decode 阶段自动生成 **CUDA Graph**，彻底消除 kernel launch 开销。

### 只需两步
```python
# 1. 编译模型
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

# 2. 生成时用静态缓存，避免形状动态变化破坏图
from transformers import StaticCache

outputs = model.generate(
    input_ids,
    max_new_tokens=128,
    cache_implementation="static",
    max_cache_length=256,
    do_sample=True,
    temperature=0.7,
    num_beams=1,
    use_cache=True,
)
```

- `mode="reduce-overhead"` 会让 PyTorch 自动为 decode 步骤录制 CUDA Graph
- `StaticCache` 保证每次 forward 的输入形状固定，图不会被不断重新编译

**实测效果**：小模型（如 GPT-2 124M）batch=1 下，token 生成速度可以从 20-30 tok/s 跃升到 200+ tok/s。

> 注意：需要 `transformers >= 4.43` 且 `PyTorch >= 2.1`。

---

## 3. 终极杀器：手动 CUDA Graph（如果 `torch.compile` 仍有瓶颈）

如果 `torch.compile` 的自动 graph 依然不够理想（比如你的模型有动态控制流），可以手动捕获一个完整的 decode step。

### 代码示例：一个极简的 CUDA Graph decode 循环

```python
import torch

# 假设有 model, tokenizer, input_ids (batch_size=1, 已经 padding)
model = model.eval().cuda().half()

# 第一次前向，拿到初始 KV cache
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    logits, past_key_values = outputs.logits, outputs.past_key_values

# 定义每一步的输入/输出占位（形状固定）
next_token = torch.zeros(1, 1, dtype=torch.long, device='cuda')
# 假设 past_key_values 是 tuple of tuple，这里需要静态化，很繁琐
# 简化：直接使用模型的 past_key_values 接口

# 捕获 graph
g = torch.cuda.CUDAGraph()

# 静态输入：只需要 input_ids (下一个 token)
# 需要预热一次
sample_input = next_token
with torch.cuda.graph(g):
    static_outputs = model(input_ids=sample_input, past_key_values=past_key_values, use_cache=True)
    # 注意：这里不能有 dynamic shape 或 control flow

# 之后在每步中：
for step in range(max_new_tokens):
    # 更新输入 token (从采样得到)
    sample_input.copy_(next_token)
    g.replay()  # 重放图，几乎没有 CPU 开销
    logits = static_outputs.logits
    past_key_values = static_outputs.past_key_values   # 图内已更新
    # 采样（轻量级，也可以在 graph 里做，但需要自定义）
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    # 检查 eos ...
```

然而 HuggingFace 的 KV cache 是动态嵌套 tuple，手动 graph 非常脏。正因如此，**vLLM 等框架已经替你封装好了这些复杂度**。

---

## 4. 当你已经受够了 PyTorch 原生 generate：换框架

`model.generate()` 本身就是 Python heavy 的，即便你用了 graph，采样逻辑、停止判断、logits 处理器等仍在 CPU 上跑。
如果调度开销仍然无法接受（比如你需要 <1ms per token），就该上专门为低延迟解码设计的推理引擎了：

| 方案 | 调度开销 | 实现难度 | 额外收益 |
|------|----------|----------|----------|
| **vLLM** | CPU 开销极低（C++/CUDA 实现 decode 循环 + graph） | 一行 `pip install vllm`，几行调用 | PagedAttention, 连续批处理 |
| **TensorRT-LLM** | 几乎全 GPU kernel，无调度开销 | 模型转换 + 构建引擎 | 极致的 kernel 融合 |
| **llama.cpp** (GPU offload) | 中央循环在 C++，无 Python | 需转换格式，但使用简单 | 低资源也能跑 |

用 vLLM 替换原生 generate 大概长这样：
```python
from vllm import LLM, SamplingParams

llm = LLM(model="your-model", dtype="float16", enforce_eager=False)  # 默认启用 CUDA Graph
sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=128)
outputs = llm.generate(["你的 prompt"], sampling_params)
```
它内部已对 decode 做了 CUDA Graph 捕获，并把采样逻辑也移到了 GPU，**几乎零调度开销**。

---

## 5. 其他可能踩的坑

- **PyTorch 的 CUDA Graph 限制**：捕获期间不能有动态形状、不能有 GPU-CPU 同步、不能有打印。如果你在 `generate` 里加了自定义的 `logits_processor` 可能包含 Python 侧的条件，graph 会因为 re-capture 而失效。
- **`torch.compile` + HuggingFace**：有些模型（如 Llama）在 `fullgraph=True` 下可能失败，可以先用默认的 `mode="reduce-overhead"`（它会自动用 CUDA graph 但允许部分回退）。
- **Profile 时留意 `cudaDeviceSynchronize`**：`generate` 本身不会乱加同步，但你在调试时如果 `time()` 之前调用了 `.cpu()` 或 `print()`，便会引入大量隐式同步，导致看起来像“调度开销”，其实是同步等待。

---

## 总结行动清单

1. ✅ 开 `torch.compile(mode="reduce-overhead")` + `cache_implementation="static"`
2. ✅ 如果还不行，观察 profiler trace，确认 kernel launch 仍然密集 → 尝试 **手动 CUDA Graph** 或 **vLLM**
3. ✅ 确保整个 pipeline 里没有不必要的 `torch.cuda.synchronize()` 或 `.item()/.cpu()` 调用
4. ✅ 如果模型太小，考虑直接换成 vLLM，把调度、缓存管理都交给专业引擎

这些步骤之后，「调度开销」会从 80% 降到几乎可忽略，你的 GPU 终于能跑满计算/访存了。

---

### 🧑‍💻 User

为什么apply ligerkernel不能加速？

---

### 🤖 Assistant

`Liger Kernel` 确实能在训练中大幅加速，但如果你“apply”后没有看到速度提升，甚至变慢了，通常逃不出以下几个原因。请逐一排查。

---

## 1. Liger Kernel 主要针对**训练**加速，不解决推理的调度开销

Liger 的核心卖点是 **Fused Linear + 损失函数（CrossEntropy）**，让层的前向+后向在单个 Triton/CUDA kernel 里完成，节省内存带宽并减少 kernel launch。
它在训练 loop 中能显著降低峰值显存和训练时间。但如果你是调用 `model.generate()`（推理），后面的情况是：

- **没有 loss 计算，没有反向传播** → Liger 最核心的“前向+损失融合”完全用不上
- 即使你用的 `LigerLayerNorm` 或 `LigerRMSNorm`，单独一个 norm 的融合并不能对推理延迟带来质变（因为 bottleneck 往往是 attention/linear matmul，而非 norm kernel 的小开销）

👉 **结论：** 若你在推理场景（`model.generate`）下应用 Liger，收益微乎其微，无法解决你前面遇到的“调度开销占大头”问题。

---

## 2. 模型/层类型不支持替换

Liger 目前支持融合的模型有特定列表（如 Llama、Mistral、Gemma 等），如果你运行的模型不在其支持范围，使用 `apply_liger_kernel(model_name)` 可能**静默失败**——没有替换任何层，自然不会加速。

验证方法：
```python
from liger_kernel.transformers import apply_liger_kernel

model = AutoModelForCausalLM.from_pretrained("your-model")
original_forward = model.lm_head.forward  # 示例
apply_liger_kernel(model, model_type="your-model-type")
# 检查 forward 函数是否真的变化了
print(model.lm_head.forward.__module__)
```
如果仍然是 `transformers` 原生实现，说明替换未生效。你可能需要手动指定 `model_type` 或显式调用 `liger_kernel.transformers.fuse_*`。

---

## 3. 瓶颈不在融合的层上

即使替换成功，提速也需要被替换的层在总耗时中占有足够比例。
如果你用 **小模型（如 GPT-2 124M）**、**极短序列**，或主要瓶颈在 attention（`FlashAttention` 已经优化得很好），而 Liger 只替换了 `Swiglu`/`GeGLU`/`Linear + Loss`，那节省的部分相对整体耗时几乎看不出来。

用 `torch.profiler` 确认真正耗时部分：
- 如果 **attention matmul** 或 **KV cache 读写** 占比 > 70%，那么优化 MLP 或 norm 收益有限
- 如果 **交叉熵损失** 不在调用链中（推理），则没有任何收益

---

## 4. 未使用正确的 dtype / 环境

Liger Kernel 的 Triton kernel 通常在 `bf16` 或 `fp16` 下才能发挥优势；如果你的模型是 `fp32` 运行，Triton 可能降级回较慢的实现，甚至根本未触发。

检查：
```python
model = model.half()  # 或 model.bfloat16()
```

同时确保安装正确：
```bash
pip install liger-kernel
# 确保 triton 已安装（建议 nightly）
```

---

## 5. 与 `torch.compile` 等优化冲突

Liger 的 fused kernel 可能与 `torch.compile` 存在兼容性问题，导致：
- 融合后的函数无法被 `torch.compile` 进一步优化（因为自定义 kernel 是黑盒）
- 两者同时使用可能引起 re-compilation 开销，反而更慢

建议**二选一**进行实验：
- 只开 Liger（禁用 `torch.compile`）
- 只开 `torch.compile`（不用 Liger）

通常在小 batch 解码时，`torch.compile` + CUDA Graph 带来的 kernel launch 消除效果远好于单个算子融合。

---

## 6. 训练时的加速，也可能因显存带宽掩盖了

如果是**训练**却没加速，可能是：
- **全局 batch size 很小**，GPU 未能饱和，kernel 融合的带宽节省体现不出来
- **模型已经很小**，Liger 的改进被 Python 侧数据加载/CPU 预处理遮盖
- **未设置 `torch.backends.cuda.matmul.allow_tf32 = True`**，导致无法利用 Tensor Core
- **未使用 `torch.cuda.amp`** 混合精度，Liger 的很多 flash 实现依赖 fp8/bf16

---

## 7. 最简单的验证：直接测 `FusedLinearCrossEntropy`

用 Liger 自带的 benchmark 脚本：
```python
from liger_kernel.chunked_loss import LigerFusedLinearCrossEntropyLoss
# 对比原生 CrossEntropyLoss 的速度
```
如果这个基准测试本身就没提升，说明你的环境（GPU 架构、Triton 版本、CUDA 版本）不兼容。

---

## 8. 最终建议：回到你的核心痛点

你之前反馈“绝大多数时间花在 torch 的调度开销上”——这个问题属于 **decode 阶段的 kernel launch overhead**，Liger 解决不了。正确的方向应该是：

- `torch.compile(mode="reduce-overhead")` + `StaticCache`（自动捕获 CUDA Graph）
- 手动 CUDA Graph 的 decode loop
- 使用专门为推理设计的框架（vLLM、TensorRT-LLM）
- 如果是长训练，Liger 才真正发光，融合 cross-entropy 和 chunked loss

**总结：**

| 你的动作 | 预期效果 |
|----------|----------|
| 只在推理时 apply Liger | 几乎无加速，瓶颈不在 loss/MLP |
| 替换 norm 层或 SwiGLU | 微小加速，但调度开销仍占主导 |
| 训练时应用并配合 amp/大 batch | 可能 20-30% 提升 |
| 保留 Liger，但同时加上 torch.compile + static cache | 可能有效（需要测试兼容性） |
| 迁移到 vLLM 或 TensorRT-LLM | 调度开销消失，生成速度翻数倍 |

先明确你的场景是训练还是推理，再针对性选择武器。如果仍是推理且追求低延迟，那请果断放弃 Liger 这种训练工具，走向 `torch.compile` / CUDA Graph / vLLM。