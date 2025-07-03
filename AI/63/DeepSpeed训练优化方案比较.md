# DeepSpeed训练优化方案比较

### 🧑‍💻 User

在deepspeed训练llm中，为了解决显存资源有限的情况，但是又要尽可能减少通信开销，能否使用autoTP+zero1代替zero3？

---

### 🤖 Assistant

这是一个很好的问题，涉及到在有限显存和通信开销之间做权衡。答案是**通常可以，并且 `autoTP + Zero-1` 确实是 DeepSpeed 中一种非常流行且推荐的配置，特别适合在显存受限但通信带宽或延迟成为瓶颈的场景下替代 `Zero-3`**。

让我们分析一下原因：

1.  **Zero-3 (Optimizer State + Gradient + Parameter Partitioning) 的优缺点：**
    *   **优点：** 显存节省最彻底。优化器状态、梯度、模型参数都进行了切分，每个 GPU 只需要存储自己负责的那部分。这对于训练超大模型至关重要。
    *   **缺点：** 通信开销最大。
        *   **前向传播：** 需要“收集”不在本地的参数 (`gather`)，计算完成后需要释放 (`release`)。
        *   **后向传播：** 同样需要`gather`参数计算梯度，计算完梯度后需要`release`。梯度本身也需要在`reduce`操作后进行切分 (`partition`)。
        *   **参数更新：** 需要`gather`参数进行更新，然后再次`release`。
        *   这些频繁的 `gather`/`release` 操作会产生大量的点对点通信，尤其是在模型层数很多时，通信延迟会成为显著瓶颈。

2.  **Zero-1 (Optimizer State Partitioning) 的优缺点：**
    *   **优点：** 通信开销最低。
        *   优化器状态被切分。
        *   梯度和参数更新只需要在 `reduce-scatter` 和 `all-gather` 通信原语上进行一次全局通信（在 `backward` 结束时进行梯度规约和平均，在 `step` 结束时广播更新后的完整参数）。没有 `gather/release` 的开销。
    *   **缺点：** 显存节省不如 Zero-3。每个 GPU 需要存储完整的模型参数和完整的梯度。这对于非常大的模型来说，显存消耗可能仍然是难以承受的。

3.  **Tensor Parallelism (`autoTP`) 的作用：**
    *   **显存节省：** TP 将模型的单个层（如 Linear, Embedding）的参数和计算切分到多个 GPU 上。**这是 `autoTP + Zero-1` 组合能替代 `Zero-3` 的关键。** TP 有效地将模型参数和计算所需显存分摊到了多个 GPU 上。
    *   **通信开销：** TP 的通信发生在层内计算过程中（例如 Linear 层的前向/后向需要 `all-reduce` 或 `all-gather` + `reduce-scatter`）。虽然 TP 引入了额外的层内通信，但这些通信通常是高度优化的，并且与模型的计算步骤重叠得比较好。更重要的是，**TP 的通信开销通常是可预测且相对固定的（与模型层数和结构相关），不像 `Zero-3` 的 `gather/release` 通信那样随模型深度线性增长且更易受延迟影响。**

4.  **`autoTP + Zero-1` vs `Zero-3`：**
    *   **显存对比：**
        *   `Zero-3`： 参数、梯度、优化器状态都切分，显存节省最极致。
        *   `autoTP + Zero-1`： **参数显存** 通过 TP 切分大幅降低（每个 GPU 只存 1/TP_degree 的参数）。**梯度显存** 每个 GPU 需要存完整的梯度（Zero-1 特性）。**优化器状态显存** 通过 Zero-1 切分降低。
        *   **结论：** `autoTP + Zero-1` 组合通过 TP 解决了最占空间的**参数显存**问题，同时利用 Zero-1 解决了**优化器状态显存**问题。主要的显存差距在于**梯度**。如果 TP 切分后，每个 GPU 存储完整梯度的显存需求是可以接受的（这取决于模型大小和 TP degree），那么 `autoTP + Zero-1` 就能在显存上满足要求。对于许多大型模型，使用足够大的 TP degree (e.g., 4, 8)，`autoTP + Zero-1` 的显存占用是可以控制在合理范围内的。
    *   **通信开销对比：**
        *   `Zero-3`： 高开销，大量的 `gather/release` 点对点通信，通信次数与模型深度强相关，对延迟敏感。
        *   `autoTP + Zero-1`： **Zero-1 部分**通信开销很低，只有全局的梯度规约和参数广播。**TP 部分**通信开销相对固定（与层数相关），但通常是集合通信 (`all-reduce`, `all-gather`, `reduce-scatter`)，在现代互连上效率较高，且更容易与计算重叠。
        *   **结论：** `autoTP + Zero-1` 的**总体通信开销通常显著低于 `Zero-3`**，尤其是在模型层数很多或者节点间互连带宽有限或延迟较高时。TP 带来的层内通信开销通常是值得付出的代价，因为它避免了 `Zero-3` 的昂贵通信模式。

5.  **为什么 `autoTP + Zero-1` 是推荐的替代方案？**
    *   **平衡点：** 它在显存节省（通过 TP 解决参数显存 + Zero-1 解决优化器状态）和通信开销（Zero-1 的低全局通信 + TP 的可管理层内通信）之间找到了一个非常好的平衡点。
    *   **避免 `Zero-3` 的通信瓶颈：** 完全避开了 `Zero-3` 最昂贵的 `gather/release` 通信模式。
    *   **兼容性：** DeepSpeed 明确支持 `autoTP` 与 `Zero-1` 的组合。`autoTP` 自动处理了 TP 的复杂切分和通信逻辑，用户只需配置 TP degree。
    *   **实际效果：** 在大量实际训练场景中（尤其是百亿到千亿参数模型），`autoTP(degree=4/8) + Zero-1` 被证明比 `Zero-3` 能提供更高的训练吞吐量（TFLOPs/s），尤其是在通信受限的环境下。

**总结与建议：**

*   **可以替代：** 在显存资源有限但通信开销成为主要瓶颈的场景下，使用 `autoTP + Zero-1` 替代 `Zero-3` **是一个常见且推荐的做法**。
*   **关键条件：** 能否替代成功的关键在于，你选择的 **TP degree** 是否足够大，使得 `autoTP + Zero-1` 组合下每个 GPU 的显存占用（特别是**完整的梯度** + 1/TP_degree 的参数 + 1/DP_degree 的优化器状态）能够满足你的 GPU 显存容量。如果模型极大，即使 TP=8 也无法容纳完整梯度，那么可能仍需考虑 `Zero-3` 或 `Zero-3 + Offload`（但通信和Offload开销更大）。
*   **优势：** 这种组合通常能提供比 `Zero-3` **更高的训练吞吐量**，因为它大幅降低了通信开销。
*   **劣势：** 需要更多的 GPU 来承载 TP（TP degree 至少为 2）。梯度显存占用是主要弱点。
*   **强烈建议：** 在你的具体模型和硬件环境上，**进行实际的基准测试**！比较 `autoTP(不同degree) + Zero-1` 和 `Zero-3` 的：
    *   每个 GPU 的显存占用
    *   训练吞吐量 (samples/sec, tokens/sec, TFLOPs/s)
    *   系统稳定性
    *   使用工具（如 `nsys`）分析通信开销的分布。

**结论：是的，在显存允许的前提下（通过选择合适的 TP degree 实现），`autoTP + Zero-1` 通常是解决显存限制并同时最小化通信开销、替代 `Zero-3` 的优选方案。**

---

### 🧑‍💻 User

模型训练代码中，deepspeed如何分别配置autoTP+zero1或者zero3呢，给出具体代码？

---

### 🤖 Assistant

在 DeepSpeed 中配置 `autoTP + Zero-1` 或 `Zero-3` 主要通过配置文件实现。以下是具体配置示例和训练代码：

### 1. AutoTP + Zero-1 配置 (推荐用于通信敏感场景)

```json
// ds_config_autoTP_zero1.json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,                 // 启用 Zero-1
    "reduce_bucket_size": 5e8   // 可调整
  },
  "tensor_parallel": {          // AutoTP 配置
    "enabled": true,
    "tp_size": 4,               // 张量并行度 (必须匹配 GPU 数量)
    "tp_group_size": 4,          // 通常等于 tp_size
    "verbose": false
  },
  "flops_profiler": {
    "enabled": false,
    "profile_step": 1
  }
}
```

### 2. Zero-3 配置 (最大显存节省)

```json
// ds_config_zero3.json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,                 // 启用 Zero-3
    "offload_optimizer": {      // 可选 CPU Offload
      "device": "none"          // 或 "cpu" 节省更多显存
    },
    "offload_param": {
      "device": "none"          // 或 "cpu"
    },
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e7,
    "reduce_bucket_size": 1e6
  },
  "flops_profiler": {
    "enabled": false
  }
}
```

### 训练代码示例 (PyTorch + DeepSpeed)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# 初始化模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"  # 以LLaMA2为例
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

# DeepSpeed 初始化
deepspeed.init_distributed()

# 选择配置文件
ds_config = "ds_config_autoTP_zero1.json"  # 或 "ds_config_zero3.json"

# 配置引擎参数
engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config,
    dist_init_required=True
)

# 训练循环示例
for batch in dataloader:
    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True).to(engine.device)
    
    # 前向传播
    outputs = engine(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # 后向传播
    engine.backward(loss)
    
    # 参数更新
    engine.step()
    
    # 监控显存使用
    if torch.distributed.get_rank() == 0:
        mem = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"Step: {step} | Loss: {loss.item():.4f} | GPU Mem: {mem:.2f}GB")
```

### 关键注意事项：

1. **GPU 数量要求**：
   - AutoTP：需要 `tp_size` 的倍数 GPU（如 tp_size=4 需要 4/8/12...个 GPU）
   - Zero-3：任意 GPU 数量

2. **启动命令区别**：
   ```bash
   # AutoTP + Zero-1 启动 (需 MPI 支持)
   deepspeed --num_gpus 4 --num_nodes 1 --master_port 29500 \
     --module training_script.py --deepspeed ds_config_autoTP_zero1.json

   # Zero-3 启动 (标准启动)
   deepspeed --num_gpus 4 training_script.py --deepspeed ds_config_zero3.json
   ```

3. **显存监控技巧**：
   ```python
   # 添加在训练循环中
   if engine.global_rank == 0:
       print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
   ```

4. **性能调优参数**：
   - `reduce_bucket_size`：增大可提升通信效率（需适配显存）
   - `stage3_prefetch_bucket_size`（Zero-3）：预取参数大小
   - `offload_optimizer`：CPU offload 可进一步节省显存

### 推荐验证步骤：

1. 先用小规模测试：
```python
# 测试配置有效性
engine, _, _, _ = deepspeed.initialize(
    model=torch.nn.Linear(10,10),  # 微型模型
    config=ds_config
)
```

2. 对比显存使用：
```bash
# 监控显存
watch -n 1 "nvidia-smi --query-gpu=memory.used --format=csv"
```

3. 通信开销分析：
```bash
# 使用 DeepSpeed 分析器
deepspeed --profile <训练命令>
```

根据实际硬件条件（特别是 GPU 显存大小和节点间网络带宽），通常：
- **NVIDIA A100/H100 集群**：`autoTP(4/8) + Zero-1` 性能最佳
- **消费级 GPU 或低速网络**：`Zero-3 + offload` 可能更稳定

建议最终通过实际训练吞吐量 (tokens/sec) 和显存占用综合评估最佳配置。