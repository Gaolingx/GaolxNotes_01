# MoE训练MFU过低原因分析

### 🧑‍💻 User

32节点 8*h200 141g(per node 8*cx7 400g NIC) 使用pytorch lightning的新fsdp2(enable fwd/bwd prefetch)+tp（devicemesh tp8+dp32）+flash attention 2，bs64/perbs 4096 token/disable cpu offload/disable grad ckpt 训练100b moe模型mfu不足10% 大量gpu利用率低于50%是正常的吗，可能是什么原因导致的？

---

### 🤖 Assistant

**这绝对是不正常的。** 在 32 节点 8*H200 (141GB) 搭配 8*400G 网卡的顶级算力集群上，即使是 MoE 模型，合理的 MFU（Model FLOPs Utilization）至少应该在 **35% - 50%** 之间，GPU 利用率应该持续在 90% 以上。

MFU 不足 10% 且大量 GPU 利用率低于 50%，说明计算单元（Tensor Cores）处于严重的“饥饿”状态，主要时间都消耗在了通信阻塞、显存调度或 CPU 瓶颈上。

结合你使用的技术栈（FSDP2 + TP8 + MoE 100B + 禁用 Grad Ckpt + Flash Attention 2），以下是导致该问题的几个核心原因及排查方向：

### 1. MoE 模型与 FSDP 的“死亡组合” (最可能的原因)
在 MoE 模型中，大部分参数集中在专家（Experts）层。如果你对整个 100B 模型直接应用标准 FSDP2（DP=32），会导致灾难性的网络通信：
*   **全参数 All-Gather 噩梦**：FSDP 会在每次前向/反向传播时，通过节点间的网络（DP rank 之间）All-Gather 所有专家权重。100B 模型的参数量级意味着巨大的通信量，即使是 8*400G 网卡也会被撑爆。
*   **计算与通信无法掩盖**：MoE 层的激活（Tokens）只会路由到少部分激活的参数（例如 100B 模型，激活参数可能只有 15B）。这意味着**你的通信量是 100B 的规模，但计算量只有 15B 的规模**。FSDP2 的 fwd/bwd prefetch 机制在这种极端的“重通信、轻计算”场景下会完全失效。
*   **解决思路**：MoE 模型通常需要引入 **Expert Parallelism (EP)**。标准的 FSDP 对 MoE 极度不友好。你需要配置 FSDP 策略，**不对专家权重进行 FSDP 分片**（或者仅在节点内分片），而是使用专用的 MoE 路由通信（All-to-All）来传递 Token。

### 2. MoE 路由导致的微批次 (Micro-batch) 极小，Tensor Core 饥饿
H200 需要极大的矩阵维度（大 GEMM）才能跑满算力。我们来算一笔账：
*   Global BS = 64，DP = 32 $\rightarrow$ 每个 DP rank 的 local BS = 2。
*   每个 DP rank 的总 Tokens 数 = $2 \times 4096 = 8192$ tokens。
*   **碎片化**：假设你有 8 个或 16 个专家。这 8192 个 token 会被路由给这些专家。假设均匀分布，每个专家只分到 $8192 / 8 = 1024$ 个 token。
*   **结论**：对于 H200 的 Tensor Core 来说，$M=1024$ 的矩阵乘法太小了，根本无法填满流处理器（SMs），导致 GPU 利用率低下。
*   如果还存在**负载不均衡（Load Imbalance）**，某些专家分到的 token 更少甚至为 0，对应的 GPU 就会直接闲置（Utilization < 50% 的典型症状）。

### 3. 显存隐式溢出 (UVM Fallback)
你提到 **未开启 Gradient Checkpointing (disable grad ckpt)**，且 seq_len = 4096。
*   100B 模型的优化器状态、梯度、参数在 FSDP(DP=32) 下每张卡占用约 15-20GB。
*   但是，不开启 grad ckpt 意味着**所有的激活值（Activations）都要保存在显存中**用于反向传播。4096 长度的 100B 模型激活值极其巨大。
*   **隐患**：虽然你 `disable cpu offload` 且没有报 OOM (Out of Memory)，但 PyTorch 内存分配器在逼近 141GB 极限时，可能会触发底层 CUDA 驱动的 **Unified Virtual Memory (UVM) Page Migration**（隐式将显存 swap 到系统内存）。一旦发生频繁的 PCIe 显存交换，GPU 利用率会呈断崖式下跌。
*   **排查**：使用 `nvidia-smi dmon` 观察，如果显存使用率死死卡在 140GB/141GB，极大可能发生了 Thrashing。尝试开启 Grad Ckpt 对比性能。

---

<details>
<summary><b>点击展开：其他次要但可能致命的基础架构原因 (NCCL, PyTorch Lightning等)</b></summary>

**4. NCCL 与 RDMA 网络未正确激活**
在 8*CX7 400G 的环境中，如果 NCCL 没有正确识别网卡，回退到了普通以太网甚至 PCIe 上通信，会导致跨节点 FSDP 彻底卡死。
*   **检查环境变量**：确保设置了正确的 NCCL 参数，例如：
    ```bash
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=^mlx5_0:1 # 根据实际网卡名称配置
    export NCCL_NET_GDR_LEVEL=2
    ```
*   **验证**：在训练前跑一下 `nccl-tests` 的 `all_reduce_perf`，确保 32 节点间的 AllReduce 带宽能跑到 300GB/s 以上。

**5. PyTorch Lightning 的 FSDP2 + TP 封装开销**
*   DeviceMesh (TP8 + DP32) 是 PyTorch 2.x 的新特性，Lightning 对 FSDP2 的包装可能存在 Bug，特别是在与 `torch.compile` 或自定义 MoE 算子结合时。
*   如果 FSDP2 的 prefetch 流 (CUDA Streams) 与 TP 的通信流 (All-Reduce) 发生锁竞争，会导致 GPU 经常处于 Wait 状态。

**6. DataLoader 瓶颈**
*   虽然不常见，但在算力如此强大的集群上，如果你的 DataLoader worker 数量不够，或者 CPU 到 GPU 的数据搬运存在瓶颈，GPU 也会等待。检查 CPU 利用率是否打满。

</details>

---

### 建议的 Debug 路径 (Action Items)

1.  **开启 PyTorch Profiler 分析 Timeline (最重要的一步)**:
    抓取 2-3 个 step 的 trace，使用 Chrome Tracing 或 TensorBoard 查看。
    *   如果看到大面积的 `ncclAllGather` 且 GPU 处于 Idle，说明是 FSDP 分片了 MoE 权重导致的通信瓶颈。
    *   如果看到细碎的 `gemm` 且中间有大量空隙，说明是 Batch Size 被专家路由切得太碎。
    *   如果看到 `cudaMalloc` 或 `Memcpy DtoH/HtoD` 占据大量时间，说明是显存逼近极限触发了 UVM。
2.  **验证显存与 Grad Ckpt**:
    临时开启 Gradient Checkpointing (`enable grad ckpt`)。如果开启后 MFU 突然从 10% 提升到 30% 以上，说明之前确实是显存碎片化或隐式 Swap 拖垮了性能。
3.  **修改 MoE 策略**:
    强烈建议不要对 MoE 的 Expert 层使用 FSDP。尝试使用 `torch.distributed.tensor.parallel` 中的 EP (Expert Parallel) 结合 TP，并仅对稠密层 (Dense layers/Attention) 使用 FSDP。
4.  **增大局部 Batch Size**:
    尝试将 Global Batch Size 提升到 128 或 256（配合 Grad Ckpt），增加每个 Expert 分配到的 token 数量，提高 Tensor Core 的计算效率。