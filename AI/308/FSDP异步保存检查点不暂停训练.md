# FSDP异步保存检查点不暂停训练

### 🧑‍💻 User

用pytorch FSDP分布式训练时候如何在保存ckpt时尽量不暂停训练？

---

### 🤖 Assistant

在 PyTorch FSDP (Fully Sharded Data Parallel) 训练中，保存检查点（Checkpoint, ckpt）通常是一个耗时的阻塞操作，因为它涉及 GPU 到 CPU 的数据拷贝、跨节点的通信（如果汇聚权重）以及磁盘 I/O。

要实现“尽量不暂停训练”（即**异步保存**或**极速保存**），最核心的策略是：**使用分布式检查点 (DCP) + Sharded State Dict + 异步 I/O 线程**。

以下是具体的实现方案和代码示例，按推荐程度排序。

---

### 方案一：使用 PyTorch Distributed Checkpoint (DCP) + 异步线程

这是目前最推荐的方案（PyTorch 2.0+）。
它的核心思想是：利用 FSDP 将权重保持在分片（Sharded）状态，只将当前分片快速拷贝到 CPU 内存中，然后启动一个后台线程将 CPU 内存中的数据写入磁盘。这样，GPU 可以立即恢复训练，而不必等待磁盘写入完成。

#### 核心步骤：
1.  **State Dict Type**: 设置为 `SHARDED_STATE_DICT`（不汇聚到 Rank 0，每个 GPU 存一部分，极大减少通信）。
2.  **Offload**: 配置将 State Dict 卸载到 CPU。
3.  **Threading**: 主线程拿到 CPU 上的数据引用后，立刻开启子线程进行磁盘写入。

#### 代码实现

```python
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import threading

# 全局变量保存正在进行的保存线程
_SAVE_THREAD = None

def save_checkpoint_async(model, optimizer, step, checkpoint_dir):
    global _SAVE_THREAD
  
    # 1. 确保上一次保存已经完成（避免内存爆炸）
    if _SAVE_THREAD is not None:
        if _SAVE_THREAD.is_alive():
            print("Warning: Previous checkpoint save is still running. Skipping this save or waiting...")
            _SAVE_THREAD.join() # 这里你可以选择跳过或者等待
  
    # 2. 配置 FSDP 以 Sharded 方式导出，并卸载到 CPU
    # offload_to_cpu=True 是关键，它只阻塞 GPU->CPU 的拷贝时间（很快），不阻塞磁盘 I/O
    sharded_osd_config = ShardedStateDictConfig(offload_to_cpu=True)
  
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_osd_config):
        # 获取模型和优化器的状态字典
        # 此时数据已经在 CPU 内存中了，但还没写入磁盘
        cpu_state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer),
            "step": step
        }

    # 3. 定义后台写入函数
    def _write_to_disk(state_dict, path):
        # 使用 PyTorch DCP 进行分布式保存
        # 这里的 I/O 操作在后台线程进行，不会阻塞主训练循环
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=path,
            # dist_cp.FileSystemWriter 负责写文件
        )
        print(f"Async checkpoint finished for step {step}")

    # 4. 启动后台线程
    save_dir = f"{checkpoint_dir}/step_{step}"
    _SAVE_THREAD = threading.Thread(target=_write_to_disk, args=(cpu_state_dict, save_dir))
    _SAVE_THREAD.start()
  
    print(f"Training resumed immediately. Saving checkpoint {step} in background...")

# 在训练循环中使用
# ...
# output = model(input)
# loss.backward()
# optimizer.step()
# if step % save_interval == 0:
#     save_checkpoint_async(model, optimizer, step, "checkpoints/")
```

<details>
<summary><strong>为什么这个方案快？</strong> (点击展开)</summary>

1.  **无通信 (No Gather)**: 传统的 `FULL_STATE_DICT` 会把所有 GPU 的参数汇聚到 Rank 0，这会导致巨大的网络开销和 OOM 风险。`SHARDED_STATE_DICT` 每个 GPU 只处理自己那部分参数。
2.  **隐藏 I/O 延迟**: 磁盘写入速度通常远慢于内存拷贝。通过 `threading`，训练进程在内存拷贝（ms 级或秒级）完成后立即继续，磁盘写入（可能几十秒）在后台并行发生。

</details>

---

### 方案二：PyTorch 2.1+ 的 `async_save` (实验性功能)

在较新的 PyTorch 版本中，`torch.distributed.checkpoint.state_dict_saver` 引入了对异步保存的原生支持（尽管底层机制与方案一类似，但 API 更封装）。

你可以检查你的 PyTorch 版本是否支持 `async_save` 参数（通常在 FileSystemWriter 或 save 函数中）。

```python
import torch.distributed.checkpoint as dcp

def save_dcp(model, dir_path):
    # 使用 Sharded 模式
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = model.state_dict()
      
        # 注意：这里的 API 随 PyTorch 版本变化较大
        # PyTorch 2.x 推荐的标准写法，DCP 内部会对 I/O 进行优化
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=dir_path,
            # 部分版本支持在 Writer 中开启异步
            # storage_writer=dcp.FileSystemWriter(dir_path, thread_count=4) 
        )
```

**注意**：目前方案一（手动线程）通常比依赖库内部的异步标志更可控，因为你可以精确控制何时释放 CPU 内存。

---

### 关键配置对比

为了最大化速度，请务必避免旧式的保存方法。

| 特性 | ❌ 传统方法 (torch.save) | ✅ 推荐方法 (FSDP + DCP) |
| :--- | :--- | :--- |
| **State Dict 类型** | `FULL_STATE_DICT` | `SHARDED_STATE_DICT` |
| **通信行为** | Gather 到 Rank 0 (极慢，阻塞所有卡) | 各 Rank 独立处理 (快) |
| **内存消耗** | Rank 0 需要 100% 模型大小的显存/内存 | 内存消耗均匀分布 |
| **I/O 模式** | 单点写入 (Rank 0 写文件) | 并行写入 (每个 Rank 写分片文件) |
| **训练暂停时间** | 通信 + 序列化 + 磁盘写入 | 仅 CPU 内存拷贝时间 (如果使用异步线程) |

### 恢复 (Loading) 的注意事项

如果你使用了 `SHARDED_STATE_DICT` 保存（生成的是一个文件夹，里面有很多 `.distcp` 文件），加载时也必须使用 DCP：

```python
def load_checkpoint(model, optimizer, checkpoint_dir):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": FSDP.optim_state_dict(model, optimizer) # 必须使用 FSDP 的 optim API
        }
      
        # 此时 state_dict 中的 tensor 是空的或者是 placeholder
        # dcp.load 会原地填充这些 tensor
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )
      
        # 加载回模型
        model.load_state_dict(state_dict["model"])
        # 加载优化器 (需要先 shard 优化器状态)
        optim_state = FSDP.optim_state_dict_to_load(model, optimizer, state_dict["optimizer"])
        optimizer.load_state_dict(optim_state)
```

### 总结建议

1.  **首选**：使用 **Sharded State Dict** 配合 **PyTorch Distributed Checkpoint (DCP)**。
2.  **优化**：在保存函数中，利用 Python 的 `threading` 模块将磁盘写入操作（DCP save）放入后台，主进程只负责将数据从 GPU `offload` 到 CPU。
3.  **资源监控**：使用异步保存时，会暂时占用双倍的 CPU 内存（一份用于训练，一份在后台等待写入）。确保你的系统内存（RAM）足够大，否则会导致 Swap 交换，反而卡死训练。