# 83M SDVAE张量并行策略

### 🧑‍💻 User

83m的2/2/2/4/5/3/3/3含attention的sdvae怎么编写torch张量并行策略？

---

### 🤖 Assistant

针对一个83M参数量、包含Attention机制、且具有特定block深度结构（如2/2/2/4/5/3/3/3）的SD-VAE（Stable Diffusion Variational Autoencoder），编写张量并行（Tensor Parallelism, TP）策略需要特别注意。

在现代PyTorch（2.0及以上版本）中，推荐使用原生的 `DTensor` (Distributed Tensor) 和 `DeviceMesh` 来实现张量并行，这比手动编写通信原语（如Megatron-LM早期的做法）要简洁得多。

以下是具体的张量并行策略和代码编写指南。

### 核心并行策略设计

VAE包含两种主要结构：**2D卷积（ResNet Blocks）** 和 **自注意力机制（Attention Blocks）**。为了减少跨GPU的通信（All-Reduce），我们通常采用“列并行+行并行”的配对策略。

1.  **对于 Attention 模块 (Transformer/Self-Attention)**
    *   **Q, K, V 投影矩阵**：使用**列并行** (Column Parallelism)。按输出通道切割，各GPU独立计算，无需通信。
    *   **输出投影矩阵 (Out Proj)**：使用**行并行** (Row Parallelism)。按输入通道切割，计算后进行一次 All-Reduce 聚合结果。
2.  **对于 ResNet 模块 (Conv2d)**
    *   **第一层卷积 (`conv1`)**：使用**列并行**。按输出通道切割 (`out_channels`)。
    *   **第二层卷积 (`conv2`)**：使用**行并行**。按输入通道切割 (`in_channels`)，计算后进行 All-Reduce。
3.  **对于归一化层 (GroupNorm)**
    *   由于特征通道被切分，GroupNorm在切分通道上计算均值和方差会非常麻烦。**通常保持GroupNorm在所有GPU上复制（Replicated），不在通道维度切分。**

---

### PyTorch 原生 TP 代码实现

以下代码基于 PyTorch 2.x 的 `torch.distributed.tensor.parallel` 编写。

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel
)

# 假设你的VAE定义包含了ResNetBlock和AttentionBlock
# 这里用伪代码代表你的SD-VAE组件
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # skip connection omitted for brevity

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1) # VAE的attention常使用1x1卷积代替Linear
        self.proj_out = nn.Conv2d(channels, channels, 1)

def apply_tp_to_sdvae(model: nn.Module, tp_size: int = 2):
    """
    为 SD-VAE 应用张量并行策略
    """
    # 1. 初始化设备网格 (Device Mesh)
    # 假设我们只做 1D 的张量并行
    device_mesh = init_device_mesh("cuda", (tp_size,))
  
    # 2. 定义具体的切分计划 (Parallel Plan)
    # 遍历模型中的 2/2/2/4/5/3/3/3 结构
    for name, module in model.named_modules():
      
        # 处理 ResNet 模块
        if isinstance(module, ResNetBlock):
            parallelize_module(
                module,
                device_mesh,
                {
                    "conv1": ColwiseParallel(), # 沿 out_channels 切分
                    "conv2": RowwiseParallel(), # 沿 in_channels 切分
                }
            )
          
        # 处理 Attention 模块
        elif isinstance(module, AttentionBlock):
            parallelize_module(
                module,
                device_mesh,
                {
                    "qkv": ColwiseParallel(),       # 沿输出切分，生成分布式的 Q, K, V
                    "proj_out": RowwiseParallel(),  # 沿输入切分，最后进行 All-Reduce
                }
            )
          
    return model

# --- 使用示例 ---
# if __name__ == "__main__":
#     # 必须在 torchrun 或 distributed 环境下运行
#     # model = MySDVAE83M()
#     # tp_model = apply_tp_to_sdvae(model, tp_size=2)
```

---

<details>
<summary><b>⚠️ 进阶考量：针对 83M 模型的特殊建议 (点击展开)</b></summary>

1. **参数量过小的 trade-off**：
   83M 的参数量对于现代 GPU 来说非常小。张量并行（Megatron-style）会引入大量的 All-Reduce 通信（每个 ResNet 和 Attention block 至少一次）。对于 83M 的模型，**通信延迟往往会远大于计算加速**，导致使用 TP 后模型反而变慢。

2. **VAE 的内存瓶颈是激活值 (Activations)，不是参数**：
   VAE 处理的高分辨率图像（如 $1024 \times 1024$ 的隐变量生成）会产生巨大的特征图。
   如果你的目的是为了解决显存溢出（OOM），切分通道（TP）不如**切分空间维度（Spatial/Sequence Parallelism）**。

3. **推荐替代方案：DeepSpeed Ulysses 或 空间并行 (Spatial Parallelism)**：
   与其切分 `Conv2d` 的 channel，不如把输入图像张量 $X \in \mathbb{R}^{B \times C \times H \times W}$ 在 $H$ 或 $W$ 维度上切分。
   * 卷积层：在各 GPU 上独立计算（边缘需要通信/halo exchange，或者直接使用能容忍边缘误差的平铺 VAE 技术，如 Tiled VAE）。
   * Attention 层：使用 Ring-Attention 或 DeepSpeed Ulysses 将 Sequence（$H \times W$ 展平后）切分给不同 GPU 进行注意力计算。
</details>

<details>
<summary><b>针对 2/2/2/4/5/3/3/3 结构的循环处理逻辑 (点击展开)</b></summary>

你的网络有类似 `[2, 2, 2, 4]` (Encoder) 和 `[5, 3, 3, 3]` (Decoder) 的 block 深度。在使用上述 `apply_tp_to_sdvae` 函数时，`named_modules()` 会自动递归遍历所有的子模块。

只要你的 VAE 代码是标准面向对象编写的（即 `nn.ModuleList` 包含了这些深度的 block），上述策略会自动对每一个 block 进行张量并行切分，不需要手动按数字去硬编码切分逻辑。

需要注意的是，最底层的 Attention（通常发生在下采样到最小分辨率的那一层）的特征维度 $C$ 必须能够被你的 TP 大小（如 2, 4, 8）整除。如果 $C = 512$，`tp_size = 4`，那么每个 GPU 负责 128 个 channel。
</details>