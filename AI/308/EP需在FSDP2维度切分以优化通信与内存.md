# EP需在FSDP2维度切分以优化通信与内存

### 🧑‍💻 User

为什么ep需要在fsdp2的维度（fully shard包裹layers）上进行切分？devicemesh上怎么划分正交并行矩阵？fsdp的fwd/bwd prefetch和reshared又该如何写？

---

### 🤖 Assistant

这是一个非常硬核且前沿的分布式训练系统问题。结合 PyTorch 最新的 FSDP2（基于 `torch.distributed._composable` 架构）以及 DeviceMesh 的设计理念，我将为你从架构原理到代码实现进行系统性的拆解。

---

### 一、 为什么 EP（专家并行）需要在 FSDP2 的维度上进行切分？

在混合专家模型（MoE）中，EP（Expert Parallelism）和 FSDP 的结合是不可避免的。FSDP2 通过 `fully_shard` 将 Module 包装起来，管理参数的生命周期。EP 必须与 FSDP2 在维度和切分逻辑上对齐，核心原因有以下三点：

1. **参数生命周期与通信域的冲突隔离**
   FSDP 的本质是数据并行，它在 Forward 前执行 All-Gather，在 Backward 后执行 Reduce-Scatter。而 EP 的本质是将不同的 Expert 放置在不同的 Device 上，通过 All-to-All 传递 Token。
   如果不对 MoE 层做特殊的 FSDP 隔离（即按 FSDP 层级切分），FSDP 会尝试去 All-Gather 整个 MoE 层的参数。这不仅破坏了 EP “专家参数驻留本地”的内存优势，还会导致 All-Gather（参数通信）和 All-to-All（Token 通信）在同一个通信流中死锁或严重降速。
2. **内存复用与状态机解耦**
   FSDP2 是按模块（Module）构建 FlatParameter 的。将 Dense 层包装为 FSDP，而将 MoE 层的 Expert 包装为 EP，可以实现内存解耦。EP 维度的参数不需要参与 FSDP 的 Reshard 动作，从而节省了频繁的参数显存分配和释放开销。
3. **计算与通信重叠（Overlap）**
   在 FSDP 维度上切分，意味着 Router 的计算（通常是 Replicated 或 FSDP sharded）可以和 Expert 参数的预取/通信隔离开来。当 Token 正在进行 EP 的 All-to-All 路由时，系统可以并行地去 prefetch 下一层 FSDP 包装的 Dense 层的参数。

---

### 二、 DeviceMesh 上怎么划分正交并行矩阵？

`DeviceMesh` 是 PyTorch 提供的高维并行抽象。划分正交并行矩阵的核心原则是：**将通信带宽最高的维度留给通信最密集的并行方式（TP > EP > FSDP/DP > PP）**。

#### 1. DeviceMesh 的初始化与切分策略

假设我们有一个包含 64 张 GPU 的集群（8 台机器，每台 8 张卡），我们要构建一个 3D 混合并行矩阵 `(FSDP, EP, TP)`。

```python
import torch
from torch.distributed.device_mesh import init_device_mesh

# 假设总 GPU 数量为 64
num_nodes = 8
gpus_per_node = 8

# 定义并行维度大小: FSDP=8, EP=4, TP=2 (8 * 4 * 2 = 64)
# TP 放在最后一维，保证在一个 Node 内部 (NVLink 范围内)
# EP 放在中间维，尽量在 Node 内部或跨 Node 的高速链路上
mesh_shape = (8, 4, 2)
mesh_dim_names = ("fsdp", "ep", "tp")

# 初始化 3D DeviceMesh
device_mesh = init_device_mesh("cuda", mesh_shape, mesh_names=mesh_dim_names)

# 获取局部的 Mesh 用于不同层的切分
fsdp_mesh = device_mesh["fsdp"]
ep_mesh = device_mesh["ep"]
tp_mesh = device_mesh["tp"]
```

#### 2. 物理拓扑映射 (正交矩阵的意义)

在上述 3D 矩阵中，张量 $X$ 的切分状态可以通过 `DTensor` (Distributed Tensor) 严格定义。不同并行的正交性体现在它们作用在不同的 Mesh 维度上，互不干扰：

*   **TP (Tensor Parallel):** 作用在 `tp_mesh`，切分张量本身的行列（Row/Col Linear），如 $W = [W_1, W_2]$。
*   **EP (Expert Parallel):** 作用在 `ep_mesh`，Token 根据路由概率分布到特定的 Rank，专家参数 $E_i$ 固定在 `ep_mesh` 的某个节点。
*   **FSDP (Data Parallel):** 作用在 `fsdp_mesh`，切分 Batch Size（Token 维度）和模型参数/梯度。

<details>
<summary><b>点击展开：MoE 层如何在 DeviceMesh 上应用 DTensor</b></summary>

在构建 MoE 模型时，我们可以这样应用正交矩阵：
```python
from torch.distributed._tensor import shard, replicate

# 对于非 MoE 层 (Dense Layers)，使用 FSDP + TP
# 参数在 FSDP 维度切分 (shard)，在 TP 维度切分 (shard)
# module = fully_shard(module, mesh=fsdp_mesh) 

# 对于 MoE 层中的专家 (Experts)
# 参数在 EP 维度切分 (每个 Rank 拿不同的 Expert)
# Token 在 EP 维度执行 All-to-All
```
</details>

---

### 三、 FSDP 的 Fwd/Bwd Prefetch 和 Reshard 应该怎么写？

FSDP 的核心性能来源是 **Overlap（通信掩盖计算）**。这就要求我们在当前层 $i$ 计算时，预取（Prefetch）层 $i+1$ 的参数，并在计算完层 $i$ 后立即释放（Reshard）层 $i$ 的参数。

在底层实现上，这依赖于 **CUDA Streams** 和 **CUDA Events** 的精妙配合。

#### 1. 核心状态机与 CUDA Stream 设计

FSDP 通常维护两个 Stream：
1.  **Compute Stream (默认流):** 负责执行矩阵乘法等计算。
2.  **Comm Stream (通信流):** 负责执行 All-Gather (Prefetch) 和 Reduce-Scatter (Grad Sync)。

参数的状态机转移为：$Sharded \xrightarrow{AllGather} Unsharded \xrightarrow{Compute} Unsharded \xrightarrow{Reshard} Sharded$

#### 2. Forward Prefetch 与 Reshard 的底层逻辑 (伪代码)

```python
import torch

class FSDPLayer(torch.nn.Module):
    def __init__(self, layer, comm_stream):
        super().__init__()
        self.layer = layer
        self.comm_stream = comm_stream
        self.prefetch_event = torch.cuda.Event()
        self.compute_event = torch.cuda.Event()

    def prefetch(self, next_layer):
        """在前向传播中异步拉取下一层的参数"""
        with torch.cuda.stream(self.comm_stream):
            # 执行 All-Gather
            next_layer.unsharded_params = all_gather(next_layer.sharded_params)
            # 记录通信完成的事件
            next_layer.prefetch_event.record(self.comm_stream)

    def reshard(self):
        """释放当前层的参数"""
        # 确保计算已经完成
        self.compute_event.synchronize()
        # 释放高精度 full params 显存
        self.unsharded_params = None 
        torch.cuda.empty_cache() # 实际底层使用内存池机制，不会直接 empty_cache

    def forward(self, x, next_layer=None):
        # 1. 等待当前层参数 All-Gather 通信完成 (阻塞 Compute Stream)
        torch.cuda.current_stream().wait_event(self.prefetch_event)

        # 2. 触发下一层参数的 Prefetch (在 Comm Stream 异步进行)
        if next_layer is not None:
            self.prefetch(next_layer)

        # 3. 执行前向计算
        out = self.layer(x)

        # 4. 记录计算完成的事件，准备 Reshard
        self.compute_event.record(torch.cuda.current_stream())
      
        # 5. 立即释放当前层显存 (Reshard)
        self.reshard()

        return out
```

#### 3. Backward Prefetch 的特殊性 (PyTorch Hook 机制)

在反向传播中，执行顺序是倒序的。FSDP2 是通过在 Autograd 引擎注册 `torch.autograd.Function` 或 Module 的 `register_full_backward_pre_hook` 来实现反向预取的。

*   **Backward Pre-Hook:** 当 Autograd 准备计算层 $i$ 的梯度时触发。此时在通信流上发起层 $i-1$ 的 All-Gather (因为反向传播马上要用到 $i-1$)。
*   **Backward Post-Hook:** 当层 $i$ 的反向计算完成后触发。此时发起层 $i$ 梯度的 Reduce-Scatter，并执行层 $i$ 参数的 Reshard。

<details>
<summary><b>点击展开：PyTorch FSDP `BackwardPrefetch` 策略说明</b></summary>

在 PyTorch 原生 API 中，可以通过枚举配置 Backward Prefetch 策略：
*   `BackwardPrefetch.BACKWARD_PRE`: 在计算当前层的反向之前，请求上一层的参数。**重叠度最高，但峰值显存占用较大**（因为当前层和上一层的 Unsharded 参数可能同时存在）。
*   `BackwardPrefetch.BACKWARD_POST`: 在完成当前层反向，且释放当前层参数（Reshard）之后，才请求上一层的参数。**显存占用极小，但重叠度较差**。

在 FSDP2 的自研中，通常采用类似 BACKWARD_PRE 的机制，但通过精细控制 CUDA Event，严格保证只有在 `layer_i` 的显存被标记为可回收后，才分配 `layer_i-1` 的显存。
</details>

### 总结

要写好一个支持 MoE + FSDP2 的底层框架：
1.  首先要用 `DeviceMesh` 划定好物理边界，确保 `fully_shard` 只包裹 Dense 部分，EP 控制 Expert 路由。
2.  必须实现双 Stream 架构，将参数获取（$W_{gather}$）从前向传播计算（$Y = X \cdot W$）中剥离。
3.  巧妙利用 Autograd Hook，精准控制反向传播生命周期中的 `prefetch_event.wait()` 和 `reshard()`，实现极限的显存和通信效率。