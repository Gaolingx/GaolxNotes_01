# DeepSpeed训练优化配置指南

### 🧑‍💻 User

在deepspeed训练llm中，为了解决显存资源有限的情况，但是又要尽可能减少通信开销，能否使用autoTP+zero1代替zero3，那如何在llama-factory配置他？

---

### 🤖 Assistant

在 DeepSpeed 中，为了在显存有限的情况下减少通信开销，使用 **AutoTP (Tensor Parallelism) + ZeRO Stage 1 (ZeRO-1)** 替代 **ZeRO Stage 3 (ZeRO-3)** 是一个**非常合理且推荐**的策略。你的理解完全正确。

### 为什么 AutoTP + ZeRO-1 是 ZeRO-3 的良好替代方案（在显存和通信开销之间权衡）？

1.  **显存节省对比：**
    *   **ZeRO-3:** 提供最强的显存节省。它将优化器状态、梯度和**模型参数**都进行切分。每个 GPU 只保存它负责计算的那部分参数的完整状态（优化器状态、梯度、参数本身）。其他参数在需要时通过通信获取。
    *   **AutoTP + ZeRO-1:** 也提供显著的显存节省。
        *   **ZeRO-1:** 仅切分**优化器状态**。每个 GPU 保存完整的模型参数和梯度，但只保存自己负责更新的那部分参数的优化器状态。这节省了优化器状态（通常是显存大户）和相应的参数更新通信。
        *   **AutoTP:** 在模型层内部（如 Linear, Embedding, Attention）进行**模型并行**。将大的权重矩阵切分到多个 GPU 上。这直接减少了**每个 GPU 上模型参数和计算激活值**的显存占用。同时，TP 的计算是并行的。
    *   **总结：** ZeRO-3 > (AutoTP + ZeRO-1) > 纯数据并行。但 AutoTP+ZeRO-1 通常比 ZeRO-3 占用略多显存（因为每个 GPU 存有完整参数副本），但远少于纯数据并行。

2.  **通信开销对比：**
    *   **ZeRO-3:** 通信开销**最大**。因为它需要在正向传播（获取远程参数）、反向传播（获取远程参数计算梯度）、优化器步骤（收集/切分梯度、更新参数）中进行大量精细粒度的通信（`all_gather`, `reduce_scatter`）。尤其是在跨节点（慢速网络）时，开销显著。
    *   **AutoTP + ZeRO-1:** 通信开销**显著低于 ZeRO-3**。
        *   **ZeRO-1:** 只在梯度计算完成后，进行**一次全局的梯度 `allreduce`**（用于计算平均梯度，因为数据并行）。在优化器步骤中，每个 GPU 独立更新自己负责的那部分参数，**不需要参数通信**。`allreduce` 是带宽优化过的集合通信操作。
        *   **AutoTP:** 在模型层内部进行通信（例如 `allreduce`, `all_gather`）。这些通信通常发生在计算密集的算子内部，并且通常是**在单个节点内的 GPU 之间**（通过 NVLink/PCIe，带宽高，延迟低）。TP 的通信开销相对固定，且通常比 ZeRO-3 的细粒度通信模式更高效，尤其是在节点内。
    *   **总结：** ZeRO-3 > (AutoTP + ZeRO-1) > 纯数据并行。AutoTP+ZeRO-1 的通信模式更粗粒度、更集中，并且 TP 通信在高速节点内进行，整体开销远小于 ZeRO-3。

3.  **适用场景：**
    *   **ZeRO-3:** 当模型**极大**，即使使用 TP 也无法装入单节点所有 GPU 的显存总和时是必要的。牺牲通信效率换取最大的模型容量。
    *   **AutoTP + ZeRO-1:** 当模型可以**通过 TP 在单节点（或少量节点）的 GPU 显存内放下**时，这是**更优的选择**。它在提供可观显存节省的同时，保持了相对较低的通信开销，训练效率更高。特别是对于节点内 NVLink 连接良好的环境，TP 通信非常快。

### 如何在 LLaMA-Factory 中配置 AutoTP + ZeRO-1

LLaMA-Factory 内部集成了 DeepSpeed，配置主要通过 `arguments` (`src/llmtuner/hparams/ds_train_args.py` 和相关命令行参数) 和 DeepSpeed 配置文件 (`ds_config.json` 或 `ds_config_zero*.json`) 来实现。

以下是配置 AutoTP + ZeRO-1 的关键步骤：

1.  **启用 DeepSpeed 并选择 ZeRO Stage 1:**
    *   确保在启动训练命令时传递 `--trainer_backend deepspeed`。
    *   你需要使用或创建一个 DeepSpeed 配置文件指定 `"zero_optimization"` 的 `"stage"` 为 `1`。
    *   通常可以复用或修改 LLaMA-Factory 提供的 ZeRO-1 配置文件，例如 `ds_config_zero1.json`。检查该文件确保：
        ```json
        "zero_optimization": {
            "stage": 1, // 核心配置项，设置为1
            ... // 其他 ZeRO-1 相关参数（通常默认即可）
        }
        ```

2.  **启用并配置 Tensor Parallelism (AutoTP):**
    *   **关键点：** LLaMA-Factory 通过 `--template` 参数间接支持 TP，但其核心 TP 配置依赖于 DeepSpeed 配置和启动方式。
    *   在 DeepSpeed 配置文件中，**启用** `"tensor_parallel"` 模块并设置并行度 `"tp_size"`：
        ```json
        "tensor_parallel": {
            "enabled": true, // 启用 TP
            "tp_size": 2, // 设置 TP 并行度，例如 2 表示在 2 个 GPU 上切分张量。必须能被总 GPU 数整除。
            ... // 其他 AutoTP 参数（通常默认即可）
        }
        ```
    *   **并行度 (`tp_size`) 选择：** 这个值必须能整除你用于训练的 **总 GPU 数量**。例如，你有 4 个 GPU：
        *   `"tp_size": 4`： 4 路 TP。数据并行度为 1 (4/4)。此时没有数据并行，只有模型并行。
        *   `"tp_size": 2`： 2 路 TP。数据并行度为 2 (4/2)。形成 2 个 TP 组，每个组包含 2 个 GPU 进行模型并行，组间进行数据并行。
        *   `"tp_size": 1`： 禁用 TP。纯数据并行（4 路）。

3.  **设置 `--deepspeed` 参数指向配置文件：**
    *   在启动训练命令时，使用 `--deepspeed` 参数指定你修改好的 DeepSpeed 配置文件路径（包含 ZeRO-1 和 TP 配置的）。
    *   例如： `--deepspeed ds_config_zero1_tp2.json`

4.  **调整 `--quantization_bit` 和 `--model_max_length` (可选但推荐)：**
    *   为了进一步节省显存，可以考虑使用 `--quantization_bit 4` (QLoRA) 进行 4 位量化训练。
    *   根据你的显存情况，适当调整 `--model_max_length` 以控制序列长度，这对激活值显存影响很大。

### 示例启动命令 (假设 4 个 GPU)

```bash
# 核心参数:
# --trainer_backend deepspeed  : 使用 DeepSpeed
# --deepspeed ds_config_zero1_tp2.json : 指定包含 ZeRO-1 和 TP=2 的配置文件
# --template deepspeed : 通常选择 deepspeed 模板 (确保 hf_parser 正确处理了 TP 参数)
# --quantization_bit 4 : 使用 4 位 QLoRA (可选，强烈推荐节省显存)
# --model_max_length 512 : 设置最大序列长度 (根据显存调整)
# --dataset your_data : 你的数据集
# --model_name_or_path /path/to/llama-2-7b : 模型路径
# --output_dir /path/to/output : 输出目录

torchrun --nnodes 1 --nproc_per_node 4 \
    src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/your/base_model \
    --dataset your_dataset \
    --template deepspeed \  # 或根据模型选择如 llama2, 但确保支持 TP
    --finetuning_type lora \
    --lora_target q_proj,v_proj \ # 根据模型调整
    --output_dir /path/to/save \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \ # 根据显存调整，这是 *每个 GPU* 的 batch size
    --gradient_accumulation_steps 4 \ # 根据总 batch size 调整
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16 \ # 或 bf16
    --trainer_backend deepspeed \
    --deepspeed ds_config_zero1_tp2.json \
    --quantization_bit 4 \ # 如果使用 QLoRA
    --model_max_length 512
```

### 关键配置文件和参数解释 (`ds_config_zero1_tp2.json`)

```json
{
  "train_batch_size": "auto", // 由命令行 per_device_train_batch_size * gradient_accumulation_steps * num_gpus 自动计算
  "train_micro_batch_size_per_gpu": "auto", // 由命令行 per_device_train_batch_size 自动填充
  "gradient_accumulation_steps": "auto", // 由命令行同名参数自动填充
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto", // 由命令行 learning_rate 自动填充
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "auto" // 由命令行 lr_scheduler_type 自动匹配
  },
  "fp16": { // 或 "bf16": { ... } 根据命令行 --fp16/--bf16 自动选择
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 1,  // 核心：启用 ZeRO Stage 1 (优化器状态切分)
    "reduce_bucket_size": "auto", // 通常保持 auto 或根据网络调整
    "allgather_bucket_size": "auto" // 通常保持 auto 或根据网络调整
  },
  "tensor_parallel": { // 核心：启用并配置 Tensor Parallelism
    "enabled": true,
    "tp_size": 2, // 关键：设置 Tensor Parallel 并行度为 2 (在 2 个 GPU 上切分张量)
    // 以下参数通常保持默认即可，DeepSpeed 的 AutoTP 会自动处理切分策略
    "tp_consecutive_ranks": true, // 在连续 rank 上做 TP (对于单节点总是true)
    "pp_partition_method": null, // TP 不关心 PP
    "tp_linear_layer_row_parallelism": true,
    "tp_linear_layer_col_parallelism": true,
    "tp_embedding_layer_col_parallelism": true,
    "tp_attention_row_parallelism": true,
    "tp_attention_col_parallelism": true,
    "tp_mlp_row_parallelism": true,
    "tp_mlp_col_parallelism": true
  },
  "activation_checkpointing": { // 可选：梯度检查点，进一步节省显存，会增加计算时间
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false
  },
  "gradient_clipping": "auto", // 由命令行 max_grad_norm 自动填充
  "steps_per_print": 10, // 可调整
  "wall_clock_breakdown": false, // 性能分析时开启
  "comms_logger": { // 通信日志，调试时开启
    "enabled": false,
    "verbose": false,
    "prof_all": false,
    "debug": false
  }
}
```

### 注意事项

1.  **TP 兼容性：** 确保你的模型结构（如 Transformer 层）支持 TP。主流的 Llama、GPT 等架构通常都支持。TP 会将模型层内的权重矩阵切分。注意模型参数（如注意力头数 `num_attention_heads`, `num_key_value_heads`）需要能被 `tp_size` 整除。
2.  **`tp_size` vs `world_size`:** `tp_size` 必须能整除总 GPU 数 (`world_size`)。剩下的维度就是数据并行度 (`dp_size = world_size / tp_size`)。
3.  **节点内通信：** TP 的通信发生在负责同一模型部分的 GPU 之间（即同一个 TP 组内）。为了获得最佳性能，**强烈建议将同一个 TP 组配置在同一个物理节点内**（利用 NVLink/PCIe 高速互联）。跨节点的 TP 通信（通过较慢的网络）会严重降低性能。LLaMA-Factory/DeepSpeed 默认会在连续 rank 上创建 TP 组 (`"tp_consecutive_ranks": true`)，这通常能保证组内 GPU 在同一个节点。
4.  **检查点兼容性：** 使用 TP 训练后保存的模型检查点是分片的（每个 TP rank 保存自己负责的那部分）。要合并成一个完整的模型进行推理，需要使用 DeepSpeed 提供的合并工具 (`deepspeed.engine.module.DeepSpeedEngine.save_checkpoint` 的特定方式或转换脚本），或者直接使用 DeepSpeed Inference。LLaMA-Factory 的导出/合并脚本通常需要适配 TP。
5.  **调试：** 首次运行时，密切关注日志。DeepSpeed 会打印 TP 和 ZeRO 的初始化信息。如果遇到错误（如形状不匹配），通常是 `tp_size` 设置与模型结构不兼容。
6.  **性能监控：** 使用 `nvidia-smi`, `dcgm` 或 DeepSpeed 的日志 (`"comms_logger"`, `"wall_clock_breakdown"`) 监控 GPU 利用率、显存占用和通信时间，以确认配置达到了预期效果（显存降低、通信开销可控）。

### 总结

在显存受限且希望最小化通信开销的场景下，使用 **`--trainer_backend deepspeed` + `--deepspeed ds_config_zero1_tpN.json`**（其中 `ds_config_zero1_tpN.json` 配置了 `"zero_optimization": {"stage": 1}` 和 `"tensor_parallel": {"enabled": true, "tp_size": N}`）是 LLaMA-Factory 中配置 AutoTP + ZeRO-1 替代 ZeRO-3 的标准方法。这能有效平衡显存占用和训练效率，尤其适合在 NVLink 连接良好的单节点或多节点（节点内 TP）环境下训练中等规模的大语言模型。记得根据你的总 GPU 数量 (`world_size`) 合理选择 `tp_size` (N)。