# Rollout阶段避免torchcompile重编译

### 🧑‍💻 User

在做rl rollout（grpo/ppo）时候，对policy模型开启torch.compile会导致频繁的图编译（例如调用`model.generate(...)`），主流训练框架是怎么解决这个问题的？

---

### 🤖 Assistant

在强化学习（如 PPO、GRPO）的 Rollout 阶段，对 Policy 模型调用 `model.generate(...)` 并开启 `torch.compile` 确实会导致灾难性的频繁重编译（Recompilation）。

导致这个问题的**根本原因**是：LLM 的生成过程（自回归解码）包含了大量的**动态形状（Dynamic Shapes，如 KV Cache 逐次增长、输入 prompt 长度不一）**和**复杂的控制流（Control Flow，如 while 循环、遇到 EOS 提前停止、Beam Search 等）**，这恰恰是 `torch.compile` 的死穴。

主流训练框架（如 OpenRLHF, Hugging Face TRL, Megatron-LM 等）主要通过以下几种核心策略来解决或绕过这个问题：

### 1. 核心范式：计算与推理引擎解耦 (Offloading to Inference Engines)
这是目前业界做大规模 RLHF 最主流、最成熟的做法（例如 **OpenRLHF** 的做法）。

框架不会死磕用 PyTorch 原生的 `generate` 函数配合 `torch.compile`，而是**将 Rollout 阶段完全交给专门的推理引擎（如 vLLM、TensorRT-LLM）**。

*   **工作流：** 
    1. 训练引擎（如 DeepSpeed/FSDP）负责更新权重（这一步完美支持 `torch.compile`）。
    2. 权重通过共享显存（如 Ray）同步给推理引擎（vLLM）。
    3. vLLM 负责执行 `generate` 进行 Rollout。vLLM 内部使用了 PagedAttention 和高度优化的 CUDA Kernel，根本不需要依赖 `torch.compile` 就能达到极高的吞吐量。
*   **优势：** 彻底避开了 `torch.compile` 处理自回归生成的痛点，Rollout 速度通常比原生 PyTorch 快几倍。

---

### 2. 原生 PyTorch 路线：静态化一切 (Static Shapes & Static Cache)
如果你必须在纯 PyTorch 环境下使用 `generate` 并开启 `torch.compile`（例如在 **Hugging Face TRL** 等框架中），主流的做法是**消除所有动态元素**。

#### A. 使用静态 KV Cache (Static KV Cache)
在传统的生成中，KV Cache 的形状是 `[batch_size, current_seq_len, hidden_dim]`，随着每生成一个 Token，`current_seq_len` 都在变化，导致每一步都在触发编译。
*   **解决方案：** 预先分配一个最大长度的固定大小张量作为 KV Cache（例如 `[batch_size, max_seq_len, hidden_dim]`），然后通过一个标量 `position_id` 去索引和更新这个静态张量。
*   **框架支持：** Hugging Face `transformers` 已经引入了 `StaticCache` 类。当你在开启 `torch.compile` 时，配合 `model.forward(..., past_key_values=StaticCache(...))`，形状就变成了完全静态的，只会在第一次生成时编译一次。

#### B. Prompt Padding 到固定长度 (Left Padding to Max Length)
即使解决了生成时的 KV Cache 动态变化，不同请求的 prompt 长度不同也会导致第一次 forward（Prefill 阶段）触发编译。
*   **解决方案：** 将同一个 Batch（甚至所有 Batch）内的 prompt 统一 Left Pad 到几个固定的 Bucket 长度（例如 512, 1024, 2048）。虽然这浪费了一些计算量，但彻底避免了 Prefill 阶段的重编译。

#### C. 固定生成步数 (Padding to Max Generation Length)
为了避免循环中的动态结束（遇到 EOS 停止）导致控制流图断裂，有时会强制模型生成固定的步数，然后通过 Attention Mask 或者后处理将 EOS 之后的 Token 丢弃。

---

### 3. 只编译 Forward 算子，不编译 Generate 控制流
`model.generate(...)` 是一个包含了复杂 while 循环的高层函数。如果你直接 `torch.compile(model.generate)`，图追踪器（Dynamo）会被循环和条件分支搞崩溃。

主流框架（如 HF `transformers`）的底层逻辑是：**绝不编译整个 `generate` 函数，而是只编译模型单步的 `forward` 方法。**

```python
# 错误做法：编译整个生成逻辑
compiled_generate = torch.compile(model.generate) 
output = compiled_generate(inputs) # 灾难性的重编译

# 正确做法：只编译底层的 forward（框架内部自动处理）
model.forward = torch.compile(model.forward)
# generate 依然在 Python 层面运行 while 循环，但循环内部的单次 forward 是编译过的
output = model.generate(inputs) 
```
即便如此，依然需要配合上述的 **Static KV Cache** 才能保证循环内部的 `forward` 不会因为形状改变而重编译。

---

<details>
<summary><b>点击展开：4. 框架相关的 torch.compile 高级参数调优</b></summary>

当框架必须硬扛动态形状时，会通过调整 `torch.compile` 的底层参数来缓解重编译：

*   **`dynamic=True`**：
    框架会强制开启 `torch.compile(model, dynamic=True)`。这会告诉底层编译器（Inductor）尝试生成支持动态形状的 Triton kernel。然而，目前 PyTorch 对极端的动态形状支持仍有瑕疵，它能减少重编译次数，但往往不能完全消除，且牺牲了一部分极致性能。
*   **编译限制策略**：
    可以通过设置 `torch._dynamo.config.suppress_errors = True` 和限制缓存大小 `torch._dynamo.config.cache_size_limit` 来防止由于无限重编译导致的内存溢出（OOM），迫使模型在形状变化过于频繁时退回 Eager 模式运行。
*   **CUDAGraphs 集成**：
    在确保了静态形状（Static Cache + Padding）的前提下，框架会开启 `mode="reduce-overhead"`，这会在底层启用 CUDA Graphs，大幅降低 Python launch overhead。这对于 PPO/GRPO 中频繁的小 Batch 生成极为关键。

</details>

### 总结与建议

如果你正在开发或优化 RLHF 流程，遇到此类问题，建议采取以下策略：

1.  **首选方案 (大规模/生产级)：** 放弃用 PyTorch 原生的 `generate` 做 Rollout。采用 **Ray + vLLM** 架构（参考 OpenRLHF）。Actor 模型在 vLLM 中做高速 Rollout，然后把数据喂给开启了 `torch.compile` 的 PyTorch 模型做 PPO Forward/Backward 更新。
2.  **次选方案 (单体框架/研究型)：** 如果非要在 `transformers` 中做，必须更新到最新版，并使用 **`StaticCache` + Prompt Padding**。
    ```python
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained("...")
    model.forward = torch.compile(model.forward, mode="reduce-overhead")
  
    # 在生成时必须指定使用静态缓存
    model.generate(..., cache_implementation="static")
    ```