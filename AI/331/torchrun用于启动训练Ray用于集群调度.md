# torchrun用于启动训练Ray用于集群调度

### 🧑‍💻 User

torch distribution中的ray和torchrun有什么区别？

---

### 🤖 Assistant

在 PyTorch 分布式训练（Distributed Training）中，`torchrun` 和 `ray`（特指 Ray Train）都是非常核心的工具，但它们的**定位、抽象层级和功能范围**有着本质的区别。

一句话总结：**`torchrun` 是 PyTorch 官方原生的分布式任务启动器（Launcher），而 `ray` 是一个庞大的通用分布式计算操作系统/框架。**

以下是它们的详细对比：

### 1. 核心定位与功能范围

*   **`torchrun` (PyTorch Native Launcher):**
    *   它是 PyTorch 1.10 引入的官方启动工具（替代了旧版的 `torch.distributed.launch`）。
    *   **唯一目的**：启动并管理 PyTorch 的分布式进程（如 DDP、FSDP）。它负责设置环境变量（如 `MASTER_ADDR`、`RANK`、`WORLD_SIZE`），并在节点上拉起相应的 Python 进程。
    *   **范围极窄**：它不管数据预处理，不管超参数搜索，不管模型部署。它只管“把训练脚本在多台机器/多个 GPU 上跑起来”。
*   **`ray` (Ray Train):**
    *   Ray 是一个通用的分布式计算框架。其中的 `Ray Train` 模块专门用于分布式深度学习。
    *   **底层机制**：Ray Train 底层**依然会调用** PyTorch 的分布式后端（NCCL/Gloo），它并不会替代 PyTorch 的 DDP 机制，而是对其进行了更高层的封装。
    *   **范围极广**：Ray 提供了完整的 AI 生命周期支持。你可以在同一个脚本中完成：用 `Ray Data` 做分布式数据预处理 $\rightarrow$ 用 `Ray Train` 做分布式训练 $\rightarrow$ 用 `Ray Tune` 做分布式超参搜索 $\rightarrow$ 用 `Ray Serve` 部署模型。

### 2. 集群管理与启动方式差异

这是开发者在实际使用中感受最明显的区别：假设你有 $N$ 个节点。

*   **使用 `torchrun`：**
    你必须**在每一个节点上**分别执行 `torchrun` 命令（通常借助于 Slurm、Kubernetes 或自己写的 SSH 脚本）。
    ```bash
    # 节点 0 运行
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="10.0.0.1" train.py
    # 节点 1 运行
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="10.0.0.1" train.py
    ```
*   **使用 `ray`：**
    Ray 会接管整个集群的资源（CPU、GPU 统一调度）。你只需要在主节点（Head Node）上提交**一次** Python 脚本，Ray 会自动把任务分发到各个 Worker 节点上的 GPU 并建立通信。
    ```python
    # 只需要在主节点运行 python train.py
    from ray.train.torch import TorchTrainer
  
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=ScalingConfig(num_workers=16, use_gpu=True) # 自动分配到 2台机器 x 8卡
    )
    trainer.fit()
    ```

### 3. 详细对比速查表

| 特性 | `torchrun` | `Ray Train` |
| :--- | :--- | :--- |
| **开发者/生态** | Meta (PyTorch 官方) | Anyscale (第三方开源) |
| **安装成本** | 零（自带于 `torch`） | 需额外安装 (`pip install ray[train]`) |
| **集群资源调度** | 无，需依赖 Slurm 或 K8s (Volcano/MPI) | 自带资源调度（也可运行在 K8s/Slurm 之上） |
| **启动方式** | 需在所有节点执行 CLI 命令 | 主节点执行 Python 脚本，自动分发 |
| **弹性与容错** | 支持 Worker 级别重启（基于 Torch Elastic） | 支持节点级别容错，结合云平台可自动扩缩容 |
| **超参数搜索** | 无内置，需外接 Optuna 等并手动处理分布式 | 深度集成 `Ray Tune`，一行代码开启分布式搜参 |
| **数据加载** | 依赖 PyTorch `DataLoader` (受限于单机 CPU) | 深度集成 `Ray Data` (支持跨机器 CPU 处理和流式加载) |

---

<details>
<summary><b>点击展开：何时选择 `torchrun`？</b></summary>

**推荐场景：**
1. **纯 PyTorch 玩家**：你的工作流非常纯粹，只有模型训练，没有复杂的分布式预处理或超参搜索需求。
2. **已经有成熟的集群调度系统**：你的实验室或公司已经有了完善的 Slurm 集群或 Kubernetes (Kubeflow, TorchOperator) 环境，这些系统已经帮你处理好了多节点启动的问题。
3. **追求最底层的控制**：你希望脚本里没有第三方框架的黑盒封装，方便 debug PyTorch 的底层通信问题。
4. **轻量化需求**：不想引入 Ray 这个庞然大物，不想维护 Ray 集群。

</details>

<details>
<summary><b>点击展开：何时选择 `Ray` (Ray Train)？</b></summary>

**推荐场景：**
1. **端到端的 AI 工作流**：你的任务包含“海量数据清洗 $\rightarrow$ 训练 $\rightarrow$ 评估 $\rightarrow$ 部署”，使用 Ray 生态可以避免在 Spark、PyTorch、Celery 之间来回倒腾数据。
2. **大规模超参数搜索**：你需要用多台机器的 GPU 来并行寻找最优超参，`Ray Tune` + `Ray Train` 是目前业界最成熟的方案之一。
3. **异构资源调度**：你需要复杂的资源分配。例如：用 100 个纯 CPU 节点做数据预处理，同时源源不断地喂给 8 个 GPU 节点做训练。PyTorch 的 `DataLoader` 无法跨机器，而 `Ray Data` 可以轻松实现。
4. **云原生弹性**：在 AWS/GCP 上，Ray 可以根据任务需求自动购买和释放 Spot 实例（竞价实例），容错能力更强，帮你省钱。

</details>

### 总结

`torchrun` 是一个**“启动工具”**，负责把 PyTorch 进程在多台机器上拉起来；
`Ray` 是一个**“分布式操作系统”**，它不仅能调用 PyTorch 进行训练，还能管理整个集群的 CPU/GPU 资源、调度数据、处理容错和超参搜索。

如果你刚开始做多卡/多机训练，建议先学习 `torchrun` 掌握 DDP 的基本原理；如果你的工程规模变得庞大，涉及复杂的数据流和超参调度，再引入 `Ray`。