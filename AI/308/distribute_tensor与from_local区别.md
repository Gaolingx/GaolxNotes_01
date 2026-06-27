# distribute_tensor与from_local区别

### 🧑‍💻 User

pytorch中，distribute_tensor和DTensor.from_local有什么区别？

---

### 🤖 Assistant

在 PyTorch 的分布式张量（DTensor）框架中，`distribute_tensor` 和 `DTensor.from_local` 是两个核心 API，它们的主要区别在于**初始数据的来源**以及**是否涉及显式的通信/分发过程**。

简单来说：`distribute_tensor` 是“**由整化零**”（从全局视角出发），而 `DTensor.from_local` 是“**由零成整**”（从局部视角出发）。

---

### 1. 核心区别对比

| 特性 | `distribute_tensor` | `DTensor.from_local` |
| :--- | :--- | :--- |
| **逻辑起点** | **全局张量** (Global Tensor) | **本地分片** (Local Shard) |
| **数据状态** | 调用前，用户通常持有一个完整的张量。 | 调用前，每个 Rank 已经拥有了自己那部分数据。 |
| **通信开销** | **有**。会将全局张量分发（Scatter/Broadcast）到各个 Rank。 | **无**（通常）。只是对现有本地张量的包装。 |
| **主要用途** | 将现有的单机模型参数或数据转换为分布式状态。 | 在自定义并行逻辑中，将已经算好的本地分片组合成 DTensor。 |
| **典型场景** | 模型加载、初始化。 | 自定义算子实现、从已有的分布式检查点加载。 |

---

### 2. 深入理解

<details>
<summary><b>distribute_tensor (全局视角)</b></summary>

这个函数的作用是将一个逻辑上的“大张量”按照指定的 `DeviceMesh` 和 `Placement`（切分方式）分发到集群中。

*   **工作流程**：
    1.  你提供一个完整的张量（通常只在 Rank 0 上有效，或者在所有 Rank 上都有一份相同的副本）。
    2.  PyTorch 根据 `placements`（如 `Shard(0)` 或 `Replicate()`）计算每个 Rank 应该持有的部分。
    3.  自动执行通信，将数据发送到对应的设备。
*   **示例代码**：
    ```python
    import torch
    from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard

    mesh = DeviceMesh("cuda", torch.arange(4)) # 假设有4个GPU
    # 逻辑上的全局张量，形状为 (8, 4)
    big_tensor = torch.randn(8, 4, device="cuda") 

    # 将其沿第0维切分到 mesh 上
    # 每个 rank 最终会得到一个 (2, 4) 的本地张量
    dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])
    ```
</details>

<details>
<summary><b>DTensor.from_local (局部视角)</b></summary>

这个方法假设数据**已经在**各个 Rank 的内存中了。它只是给这些散落在各处的“本地张量”套上一个 DTensor 的壳子，告诉 PyTorch：“虽然你现在看到的是一堆小张量，但它们在逻辑上组成了一个大的分布式张量”。

*   **工作流程**：
    1.  每个 Rank 创建自己的 `local_tensor`。
    2.  调用 `from_local`，传入 `DeviceMesh` 和 `Placement` 信息。
    3.  PyTorch 会校验这些本地张量的形状是否符合全局逻辑，如果符合，则返回一个 DTensor。
*   **数学约束**：
    假设 `DeviceMesh` 大小为 $n$，本地张量形状为 $S_{local}$。如果采用 `Shard(0)`，则全局张量形状 $S_{global}$ 为：
    $$S_{global} = [S_{local}[0] \times n, S_{local}[1], \dots]$$
*   **示例代码**：
    ```python
    import torch
    from torch.distributed.tensor import DeviceMesh, DTensor, Shard

    mesh = DeviceMesh("cuda", torch.arange(4))
    rank = mesh.get_rank()

    # 每个 rank 直接创建自己的局部数据 (2, 4)
    local_shard = torch.randn(2, 4, device="cuda")

    # 告诉 PyTorch：这些 (2, 4) 的片组合起来是一个 (8, 4) 的全局张量
    dtensor = DTensor.from_local(local_shard, mesh, [Shard(0)])
    ```
</details>

---

### 3. 何时使用哪一个？

#### 场景 A：从单机迁移到分布式
如果你有一段现成的单机代码，或者刚从磁盘加载了一个完整的 `state_dict`，你想把这些参数部署到 8 张显卡上。
*   **选择**：`distribute_tensor`。它会自动帮你处理“怎么切”以及“怎么发”的问题。

#### 场景 B：高性能自定义数据加载
如果你在写一个 Data Loader，为了效率，你让每个进程只读取文件的一部分。此时，每个 Rank 拿到的已经是“切好的”数据了。
*   **选择**：`DTensor.from_local`。因为数据已经到位，你不需要再次分发，只需要告知 DTensor 它的全局拓扑结构即可。

#### 场景 C：算子开发
当你实现一个复杂的并行算子（如自定义的 Attention 切分），你可能在本地对张量做了一系列运算，最后想把结果转换回 DTensor 以便进行后续的分布式自动求导。
*   **选择**：`DTensor.from_local`。

---

### 总结

*   **`distribute_tensor`**：是 **Data Movement** + **Metadata Creation**。它会真的搬运数据。
*   **`DTensor.from_local`**：主要是 **Metadata Creation**。它赋予本地数据一个全局的语义，前提是数据已经按照预期的布局存在于各个 Rank 上了。