# 1F1B重叠是手写PP关键

### 🧑‍💻 User

为什么用torch手写pp的时候，必须考虑1f1b overlap？

---

### 🤖 Assistant

在 PyTorch 中手写流水线并行（Pipeline Parallelism, 简称 PP）是一项极具挑战性的工作。当你深入到调度策略时，**1F1B (One Forward One Backward) 及其附带的 Overlap（计算与通信重叠）是必须考虑的核心机制**。

如果不使用 1F1B overlap，你的 PP 实现大概率会面临**显存爆炸 (OOM)、死锁 (Deadlock) 或极低的 GPU 利用率**。

以下是必须考虑 1F1B overlap 的核心原因：

### 1. 突破显存瓶颈（最致命的原因）

在流水线并行中，为了提高 GPU 利用率，我们会将一个大 Batch 拆分成 $M$ 个 Micro-batch。

如果你不使用 1F1B，而是采用朴素的 **GPipe 调度（All-Forward-All-Backward）**，GPU 需要先执行完所有 $M$ 个 Micro-batch 的前向传播（Forward），然后再依次执行反向传播（Backward）。

*   **GPipe 的灾难：** 在反向传播开始前，GPU 必须在显存中缓存所有 $M$ 个 Micro-batch 的激活值（Activations）。这意味着显存占用与 $M$ 成正比，即 $O(M)$。通常 $M$ 的取值在 32 到 128 之间，这会导致显存瞬间爆炸（OOM）。
*   **1F1B 的解法：** 1F1B 调度的核心思想是**尽早执行反向传播，从而尽早释放激活值显存**。一旦某个阶段完成了第一个 Forward，且后续阶段传回了梯度，它就立刻执行 Backward。
    *   在 1F1B 的稳定阶段（Steady state），GPU 交替执行 1 个 Forward 和 1 个 Backward。
    *   这使得单个 GPU 上最多只需要缓存 $P$（流水线阶段数/GPU数）个 Micro-batch 的激活值，显存复杂度从 $O(M)$ 骤降到 $O(P)$。

### 2. Overlap：掩盖 P2P 通信延迟

流水线并行的相邻阶段之间存在频繁的点对点通信（P2P Communication）：
*   前向传播时：发送激活值 $X$ 给下一个 GPU。
*   反向传播时：发送梯度 $\nabla X$ 给上一个 GPU。

如果你使用同步的阻塞式通信（例如 PyTorch 的 `dist.send` 和 `dist.recv`），GPU 在等待网络传输时会处于完全空闲的状态，导致极大的 Pipeline Bubble（流水线气泡）。

**1F1B Overlap 的精髓在于异步通信：**
当 GPU 正在计算第 $i$ 个 Micro-batch 的 Forward 时，可以通过后台（异步网络流）**同时接收**第 $i+1$ 个 Micro-batch 的激活值，或者**同时发送**第 $i-1$ 个 Micro-batch 的前向结果。
这就要求必须使用 `dist.isend`（非阻塞发送）和 `dist.irecv`（非阻塞接收），将通信时间隐藏在计算时间之下，极大提升吞吐量。

### 3. 避免死锁 (Deadlock)

在手写 PP 的环形或双向 P2P 通信中，调度逻辑极易发生死锁。

假设 GPU 0 想要发数据给 GPU 1，同时 GPU 1 也想发数据给 GPU 0。如果双方都调用了阻塞式的 `send`，它们都会永远等待对方调用 `recv`，从而引发死锁。

1F1B 调度结合 `batch_isend_irecv` (PyTorch 提供的原语) 可以完美解决这个问题。在 1F1B 的交替阶段，我们需要同时发送 Forward 的结果和 Backward 的梯度，使用 Overlap 机制不仅是为了性能，更是为了保证计算图能顺利流转而不会互相卡死。

---

<details>
<summary><b>点击展开：GPipe 与 1F1B 的数学直觉与时间线对比</b></summary>

假设我们有 4 张 GPU ($P=4$)，拆分了 8 个 Micro-batch ($M=8$)。

*   **GPipe 时间线 (F: Forward, B: Backward):**
    GPU 0: `F1 F2 F3 F4 F5 F6 F7 F8` (等待...) `B8 B7 B6 B5 B4 B3 B2 B1`
    *显存峰值：需要缓存 8 个 F 的中间变量。*

*   **1F1B 时间线:**
    GPU 0: `F1 F2 F3 F4` (预热阶段) `B1 F5 B2 F6 B3 F7 B4 F8` (1F1B 稳定阶段) `B5 B6 B7 B8` (收尾阶段)
    *显存峰值：最多只缓存 4 个 F 的中间变量（即 $P$ 个）。一旦 B1 算完，F1 的显存就被释放，空出来的空间用来存 F5。*

虽然两者的**总流水线气泡大小**在数学上是一样的（都是 $(P-1)$ 个时间步），但 1F1B 解决了显存问题，使得我们可以无脑增大 $M$，从而无限压缩气泡占比。

气泡占比公式为：
$$ \text{Bubble Fraction} = \frac{P - 1}{M + P - 1} $$
当 $M \gg P$ 时，气泡占比趋近于 0。GPipe 因为显存限制无法增大 $M$，而 1F1B 可以。

</details>

<details>
<summary><b>点击展开：PyTorch 手写 1F1B Overlap 的代码实现提示</b></summary>

在 PyTorch 中手写 1F1B overlap 时，不要使用基础的 `dist.send` 和 `dist.recv`。强烈建议使用 `torch.distributed.batch_isend_irecv` 或者自己维护一组 `dist.isend` / `dist.irecv` 的 futures 对象。

```python
import torch.distributed as dist

# 伪代码：1F1B 阶段的 Overlap 通信
def step_1f1b(micro_batch_fwd, micro_batch_bwd):
    # 准备要发送的激活值和接收梯度的 tensor
    send_act = compute_forward(micro_batch_fwd)
    recv_grad = torch.empty_like(send_act)
  
    # 构建异步通信操作列表
    ops = [
        dist.P2POp(dist.isend, send_act, peer=next_rank),
        dist.P2POp(dist.irecv, recv_grad, peer=next_rank)
    ]
  
    # 如果是 backward pass，还需要发送梯度给上一个阶段
    if micro_batch_bwd is not None:
        send_grad = compute_backward(micro_batch_bwd)
        ops.append(dist.P2POp(dist.isend, send_grad, peer=prev_rank))
  
    # 批量执行异步 P2P 通信，避免死锁，并重叠网络时间
    reqs = dist.batch_isend_irecv(ops)
  
    # 在等待网络传输的同时，可以做一些不需要依赖网络数据的本地计算
    # local_compute_overlap() 
  
    for req in reqs:
        req.wait() # 确保通信完成
      
    return recv_grad
```
</details>

### 总结

手写 PyTorch PP 时，**1F1B 调度是将“玩具代码”变成“工业级可用代码”的分水岭**。没有 1F1B，你无法处理真正的大模型和大 Batch Size；没有 Overlap 异步通信，你的 1F1B 会频繁陷入死锁，且被网络延迟拖垮 GPU 算力。如果你正在手写 PP，建议直接参考 Megatron-LM 的 `pipelining` 模块实现。

---

### 🧑‍💻 User

为什么fsdp2+tp实现难度通常低于fsdp2+pp？

---

### 🤖 Assistant

在分布式大模型训练中，将 **FSDP2 (Fully Sharded Data Parallel v2)** 与 **TP (Tensor Parallelism)** 结合的实现难度，往往呈指数级低于将其与 **PP (Pipeline Parallelism)** 结合。

这个现象的根本原因在于：**FSDP2 和 TP 都是对“空间（Tensor 数据）”的切分，遵循标准的单次前向/反向控制流；而 PP 引入了对“时间（执行调度）”的切分，彻底颠覆了标准的计算流程。**

以下是导致 FSDP2 + PP 实现难度飙升的核心技术原因：

### 1. `DeviceMesh` 的正交性 vs. 调度流的破坏

**对于 FSDP2 + TP：天作之合**
PyTorch 2.x 引入的 FSDP2 高度依赖 `DTensor` (Distributed Tensor) 和 `DeviceMesh`。FSDP2 和 TP 可以被完美抽象为一个二维网格：$$M = D_{DP} \times D_{TP}$$。
*   在 $D_{TP}$ 维度上，执行 TP 操作（权重被切分，使用 All-Reduce 聚合）。
*   在 $D_{DP}$ 维度上，执行 FSDP 操作（权重/梯度被切分，使用 All-Gather 和 Reduce-Scatter）。
*   **代码视角：** 你只需要将模型放在这个 2D Mesh 上，应用标准的 `model(x)` 和 `loss.backward()`。两者在**同一个数学算子（如 Linear 层）的内部和谐共处**，互不干扰。

**对于 FSDP2 + PP：水火不容**
PP 不是简单的张量切分，它切分的是**模型的层 (Layers)**，并且强行将一个大的 Global Batch 拆分成多个 Micro-batch，采用类似 1F1B 的交替执行策略。
*   FSDP 假设整个模型（或包裹的层）是一次性执行完 Forward，然后一次性执行完 Backward。
*   PP 却要求：先执行 Layer 1 的 Forward (Micro-batch 1)，接着执行 Layer 2 的 Forward... 等等。
*   **这种“时间流”的打断，使得 FSDP2 的底层逻辑完全失效**，因为 FSDP2 无法预知下一个到达当前 GPU 的到底是哪个 Micro-batch 的 Forward，还是某个历史 Micro-batch 的 Backward。

### 2. Autograd Hook 机制的崩溃

FSDP (包括 v1 和 v2) 的核心运行机制是高度依赖 PyTorch 的 Autograd Hook 的：
*   **Forward 前：** 触发 Pre-forward hook，执行 `All-Gather` 收集完整权重。
*   **Backward 后：** 触发 Post-backward hook，执行 `Reduce-Scatter` 切分并同步梯度，同时释放全量权重。

当遇到 PP 时：
在 PP 的 1F1B 调度中，Forward 和 Backward 是高度重叠且交替的。FSDP 的 Hook 会被以意想不到的顺序疯狂触发。如果你不魔改底层引擎，FSDP 极大概率会在执行 PP 的 Backward 时，因为找不到对应的全量权重而直接抛出异常（或者在错误的 Micro-batch 上累加了梯度）。

### 3. 梯度累加 (Gradient Accumulation) 语意冲突

*   **FSDP 默认行为：** 在 `loss.backward()` 结束时，立刻触发 `Reduce-Scatter` 跨 GPU 同步梯度。
*   **PP 强需求：** PP 由于存在多个 Micro-batch，本质上自带梯度累加特性。在 $M$ 个 Micro-batch 全部计算完之前，**绝对不能**进行梯度的跨节点同步。

在 FSDP2+PP 中，你必须极其小心地管理 FSDP 的 `no_sync()` 上下文管理器。你必须告诉 FSDP：“在前面的 $M-1$ 个 Micro-batch 的反向传播中，不要触发 Reduce-Scatter 通信，仅仅在本地累加；只有在最后一个 Micro-batch 的反向传播时，才允许通信。” 这在与 PP 的 P2P 调度结合时，极易写出引发死锁的代码。

---

<details>
<summary><b>点击展开：NCCL 通信原语碰撞导致的“死锁地狱”</b></summary>

FSDP2 + PP 会在网络通信层（NCCL）引发灾难性的死锁风险。

*   **FSDP 使用的是 Collective 通信：** `All-Gather`, `Reduce-Scatter`。这些操作要求参与通信的所有 GPU 必须同时到达集合点。
*   **PP 使用的是 P2P 通信：** `Send`, `Recv`, `isend`, `irecv`。这些操作发生在一对相邻的 GPU 之间。

**死锁场景再现：**
假设在 1F1B 阶段：
1. GPU 0 正在为 PP 执行 `isend`（将激活值发给 GPU 1）。
2. 同时，GPU 0 的 FSDP 引擎因为某个层的计算结束，悄悄触发了底层的 `Reduce-Scatter`（需要与 GPU 2 交互）。
3. 如果 NCCL 内部的执行流管理不当，GPU 0 的通信流会被 FSDP 的集合通信阻塞，导致它永远无法完成给 GPU 1 的 P2P 发送。而 GPU 1 在傻等 GPU 0 的数据，进而整个集群陷入死锁。

解决这个问题需要深度定制 PyTorch 的 `ProcessGroup` 通信流（Streams），为 P2P 和 Collectives 分配不同且互不阻塞的 CUDA Streams，这对绝大多数开发者来说难度极高。
</details>

<details>
<summary><b>点击展开：如何缓解 FSDP2 + PP 的实现难度？(业界方案)</b></summary>

目前业界如果必须同时使用 Data Parallel 和 Pipeline Parallel，通常**不会直接使用 FSDP2 原生版本**，而是采用以下妥协或替代方案：

1.  **ZeRO-1 (Optimizer State Sharding) + PP：** 
    最常见的做法。不切分模型权重和梯度（放弃 FSDP/ZeRO-2/ZeRO-3），只切分优化器状态。这样 Forward/Backward 期间没有任何通信，完美兼容 PP，只有在 Optimizer Step 时才通信。Megatron-LM 等主流框架多采用此方案。
2.  **Megatron-LM 分布式优化器 (Distributed Optimizer)：**
    Megatron 内部实现的 DP 策略与 PP 的调度引擎是深度耦合且一起设计的，避免了通用框架（如 PyTorch FSDP）带来的 Hook 冲突。
3.  **HuggingFace `accelerate` / DeepSpeed：**
    DeepSpeed 内部实现了 ZeRO-3 与 PP 的结合调度器，但即便如此，它在应对复杂的 PP schedules (如 Interleaved 1F1B) 时依然存在许多已知 Bug 和限制。
</details>

### 总结

简而言之：
*   **FSDP2 + TP** 就像在一张桌子（同一层）上把蛋糕（Tensor）横切竖切，大家都在同一个时间点吃，通过 `DTensor` 的抽象，这只是一个简单的数学分解问题。
*   **FSDP2 + PP** 则是要求流水线上的工人不仅要按顺序传递零件（P2P调度），还要在传递的间隙突然停下来，和隔壁车间的工人对齐账本（FSDP 集合通信）。这种时空交织的复杂控制流，需要直接修改底层调度引擎，远超一般算法工程师手写代码的范畴。