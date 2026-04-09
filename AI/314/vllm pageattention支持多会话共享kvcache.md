# vllm pageattention支持多会话共享kvcache

### 🧑‍💻 User

vllm pageattention可以实现在多会话共享kvcache吗？

---

### 🤖 Assistant

**是的，vLLM 的 PagedAttention 完全支持且非常擅长在多会话（Multi-session）中实现 KV Cache 的共享。**

事实上，这是 PagedAttention 算法的核心优势之一。这种机制在 vLLM 中被称为 **Automatic Prefix Caching (APC)**。

以下是关于它如何工作以及如何使用的详细解析：

### 1. 核心原理：PagedAttention 如何实现共享

PagedAttention 的灵感来源于操作系统的虚拟内存管理。它将 KV Cache 分为一个个固定大小的块（Blocks）。

*   **物理块与逻辑块：** 就像 OS 将逻辑页面映射到物理页框一样，vLLM 将 token 序列的逻辑块映射到 GPU 显存中的物理块。
*   **多对一映射：** 多个不同的逻辑序列（不同的用户会话）可以映射到同一个物理块上。

#### 共享机制（Copy-on-Write）
当多个会话拥有相同的“前缀”（Prefix）时，vLLM 不需要为每个会话重复计算和存储这部分的 KV Cache。

1.  **System Prompt 共享：** 假设你有 10 个并发请求，它们都使用了相同的 System Prompt（例如：“你是一个有用的助手...”）。vLLM 只会计算一次 System Prompt 的 KV Cache，并将其存储在物理块中。
2.  **引用计数：** 这 10 个请求的 Block Table（页表）都会指向这一组相同的物理块。
3.  **写时复制（Copy-on-Write）：** 当这 10 个请求开始生成各自不同的后续 token 时，vLLM 会为每个请求分配新的物理块来存储差异化的数据，而公共的前缀部分依然保持共享。

### 2. 具体应用场景

这种共享机制在以下场景中带来了巨大的性能提升：

*   **复杂系统提示词（System Prompts）：** 如果所有用户都使用同一个长 System Prompt，显存占用会显著下降。
*   **多轮对话（Multi-turn Conversation）：** 在同一个会话中，第二轮对话的输入包含了第一轮的问答。vLLM 可以直接复用第一轮已经生成的 KV Cache，而无需重新计算。
*   **少样本学习（Few-shot Learning）：** 如果你在 Prompt 中提供了很长的示例（Examples），这些示例可以被所有请求共享。
*   **RAG（检索增强生成）：** 如果多个查询检索到了相同的文档块作为上下文，这部分上下文的 Cache 也可以被共享。

### 3. 如何在 vLLM 中启用

在 vLLM 的较新版本中，你需要显式开启 **Automatic Prefix Caching (APC)** 功能来实现跨请求（跨会话）的 KV Cache 共享。

#### Python 代码方式

在初始化 `LLM` 引擎时，设置 `enable_prefix_caching=True`。

```python
from vllm import LLM, SamplingParams

# 初始化 LLM 时启用前缀缓存
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True  # <--- 关键参数
)

# 定义两个共享相同长前缀的 Prompt
common_prefix = "You are an AI assistant that explains physics concepts clearly."
prompts = [
    f"{common_prefix} Explain gravity.",
    f"{common_prefix} Explain relativity."
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

# vLLM 会自动检测公共前缀并共享 KV Cache
outputs = llm.generate(prompts, sampling_params)
```

#### API Server 方式

如果你使用的是 vLLM 的 OpenAI 兼容服务器模式，可以通过命令行参数启用：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --enable-prefix-caching
```

<details>
<summary><strong>点击展开：关于 Block Size 对共享效率的影响</strong></summary>

PagedAttention 的共享粒度是基于 **Block（块）** 的。

*   **Block Size：** 默认为 16。这意味着只有当两个序列的公共前缀长度填满整个 Block 时，该 Block 才能被共享。
*   **对齐问题：** 如果公共前缀长度是 20，Block Size 是 16。
    *   第一个 Block (0-15 tokens) **会被共享**。
    *   剩余的 4 个 token (16-19) 会占据第二个 Block 的一部分。由于该 Block 后续可能会写入不同的 token，vLLM 通常无法跨请求共享这种“部分填充且即将分叉”的 Block（除非哈希策略允许，目前 vLLM 的 APC 是基于 Block 哈希的）。
*   **建议：** 在设计 Prompt 时，不需要刻意去凑 Block Size，vLLM 的处理效率通常足够高。但在极端优化场景下，确保公共前缀覆盖尽可能多的完整 Block 是有益的。

</details>

### 4. 性能收益

根据 vLLM 官方的基准测试，启用 Prefix Caching 后：
*   **首个 Token 延迟（TTFT）**：显著降低，因为不需要重新计算前缀的 Attention。
*   **吞吐量（Throughput）**：在长 System Prompt 或长文档问答场景下，吞吐量可提升数倍。

### 总结

vLLM 的 PagedAttention 不仅可以在**单请求的 Parallel Sampling**（如 Beam Search）中共享 KV Cache，通过 `enable_prefix_caching` 参数，它还能完美实现**多会话/多请求**间的 KV Cache 共享（前缀共享）。这是 vLLM 相比传统 HuggingFace 实现的一大核心优势。