# vLLM重写模型用优化内核

### 🧑‍💻 User

我听说vllm好像都不是直接拿hf trans库的model推，里面会换成框架优化过的东东，是这样吗？

---

### 🤖 Assistant

# vLLM 的模型实现机制

是的，你说得完全正确。vLLM **并不是直接调用 HuggingFace Transformers 库的模型来做推理**，而是有一套自己的模型实现。下面详细展开：

---

## 1. vLLM 做了什么替换？

vLLM 在内部**重新实现了主流模型的架构**，主要替换/优化了以下几个层面：

| 层面 | HF Transformers 原版 | vLLM 替换后 |
|---|---|---|
| **Attention** | 标准的 `torch` 实现 | 使用 **PagedAttention** 内核（自研 CUDA kernel） |
| **KV Cache 管理** | 无（每次重新计算或简单缓存） | 基于**分页虚拟内存**思想的 KV Cache 管理 |
| **Linear 层** | `nn.Linear` | 可替换为 **量化内核**（AWQ, GPTQ, FP8 等） |
| **Attention Kernel** | PyTorch 原生 / `F.scaled_dot_product_attention` | **FlashAttention** / **FlashInfer** / **xFormers** / 自研 kernel |
| **Tensor Parallel** | 不原生支持 | 内建 **Tensor Parallelism**（Megatron 风格的列/行并行 Linear） |
| **RoPE 等位置编码** | Python/PyTorch 实现 | 融合的 CUDA kernel 实现 |
| **LayerNorm / RMSNorm** | PyTorch 标准实现 | **融合 kernel**（Triton 或 CUDA） |
| **激活函数 (SiLU/GELU)** | PyTorch 标准 | 与 Linear 融合的 kernel（如 `SiluAndMul` 融合算子） |

---

## 2. 代码结构一览

在 vLLM 源码中，你可以在 `vllm/model_executor/models/` 目录下找到**重写的模型文件**：

```
vllm/model_executor/models/
├── llama.py          # LLaMA / LLaMA-2 / LLaMA-3 重写
├── qwen2.py          # Qwen2 重写
├── chatglm.py        # ChatGLM 重写
├── mixtral.py        # Mixtral (MoE) 重写
├── gpt2.py           # GPT-2 重写
├── ...
```

以 LLaMA 为例，对比一下：

### HF Transformers 中的 Attention（简化）
```python
# transformers/models/llama/modeling_llama.py
class LlamaAttention(nn.Module):
    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, ...):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # 标准的 matmul + softmax + matmul
        attn_output = torch.matmul(attn_weights, value_states)
        ...
```

### vLLM 中的 Attention（简化）
```python
# vllm/model_executor/models/llama.py
class LlamaAttention(nn.Module):
    def __init__(self, ...):
        # 使用 vLLM 的量化感知 Linear
        self.qkv_proj = QKVParallelLinear(...)  # 合并 QKV + 张量并行
        self.o_proj = RowParallelLinear(...)     # 行并行
        self.attn = Attention(...)               # vLLM 的 PagedAttention 封装

    def forward(self, positions, hidden_states, kv_cache, attn_metadata):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(...)
        # 调用 PagedAttention kernel，自动管理分页 KV Cache
        output = self.attn(q, k, v, kv_cache, attn_metadata)
        output = self.o_proj(output)
        return output
```

关键区别：
- **`QKVParallelLinear`**：将 Q、K、V 三个投影合并为一次矩阵乘法，同时支持张量并行
- **`Attention` → PagedAttention**：使用自研的分页注意力 CUDA kernel
- **`attn_metadata`**：携带调度信息（哪些是 prefill、哪些是 decode、每个序列的长度等），支持 **continuous batching**

---

## 3. 但权重还是从 HuggingFace 加载的

vLLM 虽然重写了模型**计算图/前向逻辑**，但是：

> **模型权重（parameters）仍然从 HuggingFace Hub 的 checkpoint 加载。**

流程大致是：

```
HF Hub 下载权重文件 (safetensors / bin)
        │
        ▼
vLLM 的 weight loader 读取并映射到 vLLM 模型的参数名
        │
        ▼
加载到 vLLM 自己重写的模型结构中
        │
        ▼
用 vLLM 优化过的 kernel 做推理
```

每个模型类里通常会有一个 `load_weights()` 方法来处理权重名称的映射：

```python
class LlamaForCausalLM(nn.Module):
    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # 处理 HF 权重名 → vLLM 参数名 的映射
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

---

## 4. 为什么要这样做？

核心目的是**高吞吐、低延迟的 serving**：

1. **PagedAttention**：避免 KV Cache 的显存浪费（原版预分配连续大块显存，vLLM 按需分页分配），显存利用率提升数倍
2. **Continuous Batching**：请求随到随处理，不需要等一个 batch 全部完成
3. **融合 Kernel**：减少 kernel launch overhead 和显存读写
4. **Tensor Parallelism**：原生多卡并行，HF 原版需要借助 `accelerate` 等外部工具
5. **量化支持**：直接在 Linear 层面集成 AWQ/GPTQ/FP8 量化 kernel

---

## 5. 总结

```
┌──────────────────────────────────────────────┐
│              vLLM 推理栈                      │
│                                              │
│   HF 权重文件  ──加载──▶  vLLM 自己的模型实现  │
│                          (重写的 forward)      │
│                               │               │
│                      ┌────────┴────────┐      │
│                      ▼                 ▼      │
│              PagedAttention      融合算子      │
│              (CUDA kernel)    (RMSNorm,SiLU)  │
│                      │                 │      │
│                      └────────┬────────┘      │
│                               ▼               │
│                     Continuous Batching        │
│                     Scheduler + Engine         │
└──────────────────────────────────────────────┘
```

**一句话总结**：vLLM 用 HuggingFace 的**权重**，但用**自己重写的模型结构和高度优化的 CUDA kernel** 来做推理，所以才能实现比直接用 `transformers` 库高得多的推理吞吐量。