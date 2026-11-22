# NCCL GIN未取代NVSHMEM

## 🧑‍💻 User

nccl gin有没有取代nvshmem

---
## 😀 默认助手 (7)

# 结论先说

**在“库”的层面：没有。** NCCL GIN 和 NVSHMEM 不是同一层级的东西，NVIDIA 也没有要废弃 NVSHMEM（它 2026-09 刚发布 3.8.0，还在加新特性）。

**在“具体应用实现”的层面：确实已经发生过一次真实的替代。** DeepEP 从 V1（NVSHMEM 后端）切到 V2 的 **NCCL Gin backend**，到 V2.5 直接把 NVSHMEM 依赖整个删掉了。

| | NVSHMEM | NCCL GIN |
|---|---|---|
| 定位 | OpenSHMEM 规范的 GPU 实现，**完整 PGAS 编程模型**（独立库） | NCCL Device API 的三个模块之一（LSA / Multimem / **GIN**），**挂在 NCCL communicator + window 上的单边原语** |
| 内存模型 | `nvshmem_malloc` 对称堆，“分配一次、全局可寻址” | 必须 `ncclMemAlloc` + `ncclCommWindowRegister` 显式注册 window（MPI RMA window 模型） |
| 原语 | put/get、**全套远程原子**（add/fetch-add/CAS/swap/逻辑运算）、collective、teams、host+device 双接口、nvshmem4py | `put`/`get`/`putValue`/`signal`(strong·weak)/`waitSignal`/counter/`flush`/`flushAsync`/`wait`、`ncclGinBarrierSession`。**没有通用远程原子，没有 OpenSHMEM collective** |
| 后端 | IBGDA / GPUNetIO / IBRC / libfabric(UCX, EFA) / NVLS | PROXY（lock-free GPU→CPU 队列 + verbs）、GDAKI（DOCA GPUNetIO）；2.31 起还有 **CFT**（Blackwell + CUDA 13.3） |
| 首次可用 | 2.x 起持续演进 | Device API 自 NCCL **2.28**，**GIN 自 2.28.7**；`get`/`flushAsync` 2.30.3；strong/weak signal 2.30.5；CFT 2.31 |

---

# 为什么说“实际替代”发生了

DeepEP 官方 README 写得非常明确：

> **V2 release**: A complete refactoring of expert parallelism … with the lightweight **NCCL Gin backend** … Able to reuse existing NCCL communicators
>
> **V2.5**: Fully remove V1, including its APIs, **NVSHMEM backend**, and legacy documentation. **NVSHMEM is no longer a dependency**

NVIDIA 自己的论文（arXiv:2511.15076, *GPU-Initiated Networking for NCCL*, 2025-11）也是拿 DeepEP 作为落地验证，并强调 GIN 的价值在于“**在 NCCL 统一 runtime 内做 device-initiated 通信**”——即低延迟 + 直接复用 NCCL 的 collective 算法和生产设施。

所以真实动机不是“GIN 比 NVSHMEM 快”，而是：

1. **少一层依赖**：DeepEP 本来就要 NCCL 做常规 collective，不必再引入 NVSHMEM + 它的 bootstrap（MPI/PMIx）。
2. **复用 communicator**：`EP_REUSE_NCCL_COMM=1`，直接吃 PyTorch 的 `_comm_ptr`。
3. **性能持平**即可接受：对比结论是 GIN 与 NVSHMEM(IBGDA) 路径“性能基本一致”，GIN 的小消息延迟略高一点。

---

# 但 NVSHMEM 远没有被取代

- **仍在高强度开发**：`NVSHMEM 3.8.0-0` 于 **2026-09-22** 发布，新增 host/device RMA regions、counted-signal、NVLS multicast logical endpoints、**CFT counted writes / logical endpoints**、CUDA 13.4 / Vera Rubin 支持、Rust bindings、TMA 等。文档站更新到 2026-09。
- **两者在向同一硬件方向收敛**：NVSHMEM 3.8 也在做 CFT / logical endpoint；NCCL 侧 GIN 之后是 CFT。说明这是“跟着网卡（CX8/Rubin）走”，不是“一个吃掉另一个”。
- **NVSHMEM 的能力面仍然更宽**：远程原子（DeepEP 的 low-latency combine 就依赖 RDMA atomic，README 里还专门建议 `mlciofnig PCI_ATOMIC_MODE=4`）、PGAS 任意寻址、OpenSHMEM collective、host+device 统一 API、非 NVIDIA 网络（EFA/Slingshot via libfabric）。

<details>
<summary><b>选型建议（点开）</b></summary>

**用 NCCL GIN 更合适：**
- MoE / EP 的 all-to-all、put + signal 类通信，且已经在 PyTorch + NCCL 栈里
- 想复用已有 NCCL communicator、不想引 NVSHMEM bootstrap
- 网卡没那么新（CX4+ 可退化成 PROXY 后端仍能跑）
- 需要和 NCCL collective 混在同一个 runtime / 同一个 stream 语义下

**仍应选 NVSHMEM：**
- 需要细粒度远程原子（fetch-add / CAS / swap），或大量不规则小消息
- 想要 PGAS “allocate once, address anywhere”，不想逐个 buffer 注册 window
- 已有 OpenSHMEM 代码 / 需要 host + device 统一接口
- 跑在 libfabric（EFA、Slingshot）、UCX 或非标准 RDMA 网络上
- 需要 NVSHMEM 的 collective 与 teams 语义

**新项目看 Blackwell+：** 直接评估 **NCCL CFT**（2.31+，需要 CUDA 13.3 / driver ≥ 610.43.02），DeepEP 的 `nvDev` 分支已经在用它换小 token 场景的延迟。
</details>

<details>
<summary><b>GIN 目前工程上的坑（点开）</b></summary>

- **跨版本不保证兼容**：官方明说 “Kernels utilizing GIN are currently **not backwards compatible** and need to be recompiled when NCCL is upgraded”（LSA/multimem 已支持）。NVSHMEM 侧则是 3.5.21 起 IBGDA 的 QP 内部布局变更会破 ABI，各有各的痛。
- **要拿到低延迟必须走 GDAKI**：需 rdma-core ≥ 44、CX6+ / BlueField、`nv_peer_mem` 或 DMA-BUF、kernel ≥ 6.1（虚拟化场景）；否则退化到 CPU proxy，SM 占用和延迟特性完全不同。
- **window 语义更“手动”**：每个跨节点访问的 buffer 都要显式注册；2.28 时还不支持各 rank 注册不同大小的 buffer。
- **完成语义要自己捋清**：`flush` 只保证**本地**发送缓冲可复用，不保证远端收到；signal 分 strong（隐含前序 put 可见）和 weak（只保证自己这一笔）；GIN 操作乱序，只有同一 ctx 内的 put+signal 保序。
</details>

---

**参考来源**
- NCCL 官方 *Device-Initiated Communication*（GIN 自 2.28.7，含 requirements、跨版本兼容说明）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
- NCCL *Device API – GIN* 参考（put/get/signal/counter/barrier 全集）：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html
- NVIDIA, *GPU-Initiated Networking for NCCL*, arXiv:2511.15076：https://arxiv.org/abs/2511.15076
- DeepEP README（V2 用 Gin 取代 NVSHMEM，V2.5 移除 NVSHMEM 依赖）：https://github.com/deepseek-ai/DeepEP
- NVSHMEM 3.8.0 Release Notes（2026-09-22）：https://github.com/NVIDIA/nvshmem/releases/tag/v3.8.0-0
- 中文笔记《NCCL-GIN 特性介绍》（后端对比、与 IBGDA 的异同）：https://fanyizhen1995.github.io/2025/12/29/nccl-gin/
