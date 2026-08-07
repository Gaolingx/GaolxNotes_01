# distribute_tensor与from_local区别

### 🧑‍💻 User

pytorch中，distribute_tensor和DTensor.from_local有什么区别？

---

### 🤖 Assistant

在 PyTorch 的分布式张量（DTensor）体系中，`distribute_tensor` 和 `DTensor.from_local` 都可以用来创建 `DTensor`，但它们的**输入假设**和**内部行为**有本质区别。理解这一点对正确编写分布式 SPMD 代码至关重要。

---

## 1. `DTensor.from_local`

**定位**：一个低级、直接的构造函数。
**输入**：一个已经属于当前设备的**本地分片**（local shard）。
**行为**：直接将该本地张量包装成 `DTensor`，**不做任何切片或通信**，只是为其附加全局形状、步长和设备网格（`DeviceMesh`）等元信息。

```python
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh

# 假设有2个设备，mesh为[0, 1]
mesh = DeviceMesh("cuda", [0, 1])
# rank 0 上本地已经存放了第一半数据
local_tensor = torch.randn(4, 4, device="cuda")

# 直接包装，表示按第0维分片，当前本地是完整 DTensor 本地分片
dt = DTensor.from_local(local_tensor, device_mesh=mesh, placements=[Shard(0)])
```

- **要求**：`local_tensor` 的形状必须与由 `placements` + `mesh` 推导出的本地分片形状严格匹配，否则会报错。
- **典型场景**：你通过其他方式（如自定义分片逻辑、分布式 checkpoint 加载）已经拥有了本地分片，只需将其“升级”为带全局语义的 `DTensor`。

---

## 2. `distribute_tensor`

**定位**：一个高层的、面向 SPMD 模式的便利函数。
**输入**：一个**全局（完整）张量**——在每一个设备上都存在相同的副本（通常每个 rank 都加载了相同的模型权重）。
**行为**：根据当前 rank 和 `placements` 自动从全局张量中**切片**出属于自己的本地分片，然后内部调用 `from_local` 包装为 `DTensor`。它同样**不执行任何设备间通信**，因为它假定所有进程已然持有相同的全局输入。

```python
from torch.distributed.tensor import distribute_tensor, Shard

# 每个 rank 都拥有相同的全局张量（例如刚加载的完整模型参数）
global_tensor = torch.randn(8, 4)
# 按第0维分片到2个设备
dt = distribute_tensor(global_tensor, device_mesh=mesh, placements=[Shard(0)])
# 在 rank 0 上，dt 自动持有前4行；rank 1 上持有后4行。
```

- **要求**：所有 rank 都调用该函数，且传入**完全相同的全局张量**。
- **典型场景**：模型初始化后立即将参数分片、或加载完整 checkpoint 后分发参数，是 SPMD 程序中最常用的分发接口。

---

## 3. 核心区别对比表

| 对比维度               | `distribute_tensor`                           | `DTensor.from_local`                         |
|------------------------|-----------------------------------------------|----------------------------------------------|
| **输入期望**           | 全局（完整）张量，每个 rank 都有一份          | 本地分片张量，每个 rank 持有自己的那一部分   |
| **是否执行切片**       | 是，自动根据 rank 和 placements 切片          | 否，直接使用传入的本地张量                   |
| **通信开销**           | 无（所有进程独立完成切片）                    | 无（纯元信息包装）                           |
| **使用时机**           | SPMD 初始化：所有 rank 有全量数据，要分片     | 已有分片数据时（如手动拆分、恢复本地 checkpoint） |
| **出错条件**           | 输入形状与 placements 不兼容时会报错          | 本地张量形状与推导出的分片形状不匹配时会报错 |

---

## 4. 代码示例对比

假设有 2 个 GPU，mesh 为 `[0, 1]`，对形状为 `(8, 4)` 的张量按第 0 维进行 Shard。

### 使用 `distribute_tensor`（SPMD 惯用）

```python
# 每个 rank 都执行相同代码
full = torch.randn(8, 4)
dt = distribute_tensor(full, mesh, placements=[Shard(0)])
# rank0: dt.to_local().shape -> torch.Size([4, 4])   (行0-3)
# rank1: dt.to_local().shape -> torch.Size([4, 4])   (行4-7)
```

### 使用 `from_local`（显式提前切片）

```python
# rank 0 手动准备自己的分片
if rank == 0:
    local = full[0:4, :]
elif rank == 1:
    local = full[4:8, :]

dt = DTensor.from_local(local, mesh, placements=[Shard(0)])
```

---

## 5. 重要注意事项

- **未弃用，各有用途**：`distribute_tensor` 并未被移除。在 PyTorch 2.2+ 的官方 FSDP/TP 示例中，两种方式都会出现。`distribute_tensor` 因简洁性更受欢迎，尤其是在使用 `DTensor` 进行模型并行化时。
- **局部性保证**：二者都只在本地操作，不触发广播或 all‑gather。如果需要从 rank 0 单独分发全量数据到所有设备，需要使用 `scatter` 等通信原语，再配合 `from_local`。
- **反向操作**：`DTensor.to_local()` 可以从 `DTensor` 中取出本地分片；`DTensor.full_tensor()` 则需要通信收集出全局张量。

理解以上区别后，你可以根据当前数据存放的状态灵活选用：
- 大家都在同一时刻拥有完整数据 → `distribute_tensor`
- 每个人已经有了自己的那块本地数据 → `DTensor.from_local`