# Windows下无法高效使用TP

### 🧑‍💻 User

Windows下不支持nccl能开启pytorch tp（ColwiseParallel/RowwiseParallel/SequenceParallel）吗？

---

### 🤖 Assistant

**简短回答**：
**理论上可以，但实际中完全不可用（强烈不建议）。**

在 Windows 下，由于不支持 `nccl`，你可以强制使用 `gloo` 后端来初始化分布式环境并开启 Tensor Parallelism (TP)。但是，由于 TP 在每次前向和反向传播中都需要进行极其频繁且庞大的张量通信（如 `all_reduce`、`all_gather`），使用 `gloo` 跑 GPU 通信会导致**性能灾难**，速度甚至远不如单张显卡运行。

以下是详细的原理解释、如何在没有 NCCL 的情况下强行开启的代码示例，以及 Windows 下的推荐解决方案。

---

### 1. 为什么“理论上可以”？
PyTorch 的分布式张量并行 API（如 `ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`）是建立在 `DeviceMesh` 和底层的 `torch.distributed` 进程组之上的。
`torch.distributed` 在 Windows 上原生支持 `gloo` 后端。`gloo` 不仅支持 CPU，也支持 GPU 之间的通信。因此，只要你用 `gloo` 初始化了进程组并构建了 `DeviceMesh`，这些并行策略在代码层面上是可以跑通的。

### 2. 为什么“实际上不可用”？
* **通信频率与带宽要求极高**：Tensor Parallelism 不是将不同数据分给不同 GPU（那是 DDP），而是把**同一个矩阵的计算**拆分到不同 GPU 上。例如，全连接层切分后，必须在每一层计算完后立刻执行 `all_reduce` 或 `all_gather` 来拼接结果。
* **Gloo vs NCCL**：`nccl` 是 NVIDIA 专门为 GPU 通信（利用 PCIe 和 NVLink）优化的底层库，速度极快。而 `gloo` 主要是为 CPU 通信设计的，它在处理 GPU 显存到显存的数据交换时效率极其低下，且不支持 NVLink 等高速互联技术。
* **结果**：如果用 `gloo` 跑 TP，GPU 会把 99% 的时间花在等待数据传输上，导致训练或推理直接卡死或慢到无法忍受。

---

<details>
<summary><b>点击展开：如何强行使用 Gloo 开启 TP（仅供代码测试/调试）</b></summary>

如果你只是想在单机多卡（或多进程模拟多卡）的环境下调试代码逻辑，而不关心性能，可以通过以下方式使用 `gloo`：

```python
import os
import torch
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel
)
from torch.distributed.device_mesh import init_device_mesh

def setup():
    # 在 Windows 下，强制使用 gloo 而不是 nccl
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=0, world_size=1) # 实际测试需要使用 torchrun 启动多个进程

def main():
    setup()
  
    # 建立 DeviceMesh，指定使用 cuda 设备
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
  
    # 假设你有一个简单的模型
    model = torch.nn.Linear(1024, 1024).cuda()
  
    # 强行应用并行策略
    parallelized_model = parallelize_module(
        model,
        device_mesh,
        ColwiseParallel() # 或者 RowwiseParallel / SequenceParallel
    )
  
    print("TP 模型初始化成功！")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```
*注意：在 Windows 上运行分布式代码，建议使用 `if __name__ == '__main__':` 保护，以避免多进程衍生问题。*

</details>

---

### 3. Windows 用户的正确解决方案 (强烈推荐)

如果你在 Windows 物理机上，并且必须使用 Tensor Parallelism（例如运行大语言模型 LLaMA、Qwen 等），**唯一的标准解法是使用 WSL2 (Windows Subsystem for Linux)**。

**WSL2 的优势：**
1. **完美支持 NCCL**：现在的 WSL2 已经完美支持 NVIDIA GPU 直通。在 WSL2 的 Ubuntu 系统中，你可以像在原生 Linux 中一样直接安装 PyTorch，`nccl` 可以直接满血运行。
2. **性能损失极小**：计算性能与原生 Linux 几乎没有差异。
3. **生态兼容**：绝大多数主流的分布式训练框架（Megatron-LM, DeepSpeed, vLLM 等）都只在 Linux 环境和 NCCL 下测试过，使用 WSL2 可以避免各种奇怪的 Windows 兼容性报错。

**如何在 Windows 下快速获取 NCCL 支持：**
1. 在 Windows 终端运行 `wsl --install` 安装 Ubuntu。
2. 重启电脑，进入 Ubuntu 终端。
3. （无需在 WSL 内安装显卡驱动，只需 Windows 宿主机有最新 NVIDIA 驱动即可）。
4. 在 WSL 内安装 CUDA Toolkit 和 PyTorch（Linux 版本）。
5. 在 WSL 内运行你的分布式代码，此时 `backend="nccl"` 将完全可用，TP (`ColwiseParallel`, `RowwiseParallel` 等) 也能正常发挥其应有的性能。