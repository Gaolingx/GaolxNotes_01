# NVSHMEM可用于EP A2A

### 🧑‍💻 User

nvshmem能否用于编写ep的a2a kernel？

---

### 🤖 Assistant

这是一个很好的问题，涉及NVSHMEM在MoE（Mixture of Experts）专家并行场景下的应用。让我先搜索相关的技术资料。

让我搜索一些NVSHMEM与MoE all-to-all相关的技术资料。

让我通过浏览器搜索相关内容。

基于以上调研，以下是详细的分析：

---

## 答案：可以，而且已经被多个生产级项目实现

NVSHMEM **完全可以**用于编写 Expert Parallelism (EP) 的 All-to-All (a2a) kernel。事实上，业界已经有多个成熟的开源项目基于 NVSHMEM 实现了 MoE dispatch/combine 的 a2a 通信。

---

### 1. 已知的基于 NVSHMEM 的 EP A2A 实现

| 项目 | 组织 | 说明 |
|------|------|------|
| **pplx-kernels** | Perplexity AI | 最早的 NVSHMEM-based MoE kernel，仅依赖 **3个** NVSHMEM 函数，配合自定义同步/fencing 逻辑（现已迁移至 pplx-garden） |
| **DeepEP V1** | DeepSeek | 完整的 EP 通信库，基于 NVSHMEM + IBGDA，提供高吞吐/低延迟的 dispatch/combine kernel，支持 FP8 |
| **NCCL EP** | 学术界 | 论文中提到 pplx-kernels 使用 NVSHMEM with IBGDA 实现 device-initiated MoE 通信 |

---

### 2. 为什么 NVSHMEM 适合写 EP A2A Kernel？

NVSHMEM 的核心优势在于 **GPU-initiated（设备发起）通信**，这对 EP 场景至关重要：

```
传统 NCCL 模式：  GPU → CPU 发起 → NIC/RDMA → 远端
NVSHMEM 模式：    GPU kernel 内直接 → NVLink/RDMA → 远端 GPU
```

| 特性 | 对 EP A2A 的意义 |
|------|------------------|
| **单边 Put/Get** | Token dispatch 本质是"把数据写到目标 expert 所在 GPU"，单边 `put` 天然匹配 |
| **GPU-initiated RDMA** | 省去 CPU 参与，延迟更低，适合 decode 阶段的低延迟需求 |
| **Signal/Fence 原语** | GPU 间精细同步，避免全局 barrier，适合 token 数量不均衡的场景 |
| **IBGDA 支持** | 跨节点时 GPU 直接通过 InfiniBand 做 RDMA，无需 CPU bounce buffer |
| **零/低 SM 占用** | 可以用 copy engine 或极少量 SM 完成通信，SM 可全部留给计算 |

Perplexity 的博客明确提到：**"While NVSHMEM provides a wide range of primitives, our kernels rely on only 3 functions, building all synchronization and fencing upon them."** — 这证明 NVSHMEM 的原语集足够精简且强大。

---

### 3. 典型 EP A2A 流程（Dispatch + Combine）

```
┌─────────────────────────────────────────────────────┐
│                   MoE Dispatch (A2A)                  │
│                                                       │
│  GPU 0 (tokens)  ──nvshmem_put──▶  GPU 1 (Expert 0) │
│                  ──nvshmem_put──▶  GPU 2 (Expert 1) │
│                  ──nvshmem_put──▶  GPU 3 (Expert 2) │
│                                                       │
│  GPU 1 (tokens)  ──nvshmem_put──▶  GPU 0 (Expert 3) │
│                  ──nvshmem_put──▶  GPU 2 (Expert 4) │
│                  ...                                  │
│                                                       │
│  完成后再 nvshmem_signal 通知接收方                    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   MoE Combine (A2A)                   │
│                                                       │
│  各 GPU 上的 expert 计算完成后，通过 nvshmem_put      │
│  将结果写回 token 原始所在的 GPU，再做 reduce         │
└─────────────────────────────────────────────────────┘
```

---

### 4. 当前趋势与演进

值得注意的是，虽然 NVSHMEM 是可行的，但业界也在探索替代方案：

| 方向 | 代表项目 | 动机 |
|------|---------|------|
| **NCCL Gin backend** | DeepEP **V2** | 更轻量，可复用现有 NCCL communicator，无需额外安装 NVSHMEM |
| **RDMA TransferEngine** | pplx-garden (**fabric-lib**) | 更精细的 P2P RDMA 控制，Rust 实现 |
| **纯 NCCL + Custom Kernel** | NCCL EP (UCCL-EP) | 统一接口，跨平台兼容 |

DeepEP V2 从 NVSHMEM 迁移到 NCCL Gin 的 README 中明确说：
> *"V2 has also switched from the NVSHMEM backend to the more lightweight NCCL Gin backend."*

但这不意味着 NVSHMEM 被"淘汰"——DeepEP V2 仍然依赖 NVSHMEM 提供 legacy 支持。

---

### 5. 结论

| 问题 | 答案 |
|------|------|
| NVSHMEM 能用于 EP A2A kernel？ | ✅ **可以** |
| 有生产级案例？ | ✅ pplx-kernels、DeepEP V1 |
| 适合什么场景？ | 低延迟 decode、跨节点 RDMA、零 SM 通信 |
| 需要多少 NVSHMEM API？ | 核心只需 `put`/`get`/`signal` 等 3~5 个函数 |
| 是否有替代方案？ | ✅ DeepEP V2 (NCCL Gin)、pplx-garden (fabric-lib) |

如果你正在自己实现 EP a2a kernel，NVSHMEM 是一个成熟、可靠的选择；如果追求更轻量的依赖和更简单的部署，可以考虑参考 DeepEP V2 的 NCCL Gin 路线。

---

### 🧑‍💻 User

工业界为什么目前也有在往NCCL Gin迁移的趋势？

---

### 🤖 Assistant

这个趋势的核心驱动力可以归纳为：**NVSHMEM 在功能上"能做"，但在工程上"太重"**。以下是工业界迁移的深层原因：

---

## 一、NVSHMEM 的工程痛点

### 1. 部署与依赖的"重量级"

```
NVSHMEM 部署栈：
  NVSHMEM 库 → MPI (OpenMPI/MVAPICH) → libfabric/UCX → IB Verbs → 硬件

NCCL Gin 部署栈：
  NCCL (已随 PyTorch 安装) → IB Verbs → 硬件
```

| 痛点 | 具体表现 |
|------|----------|
| **MPI 依赖** | NVSHMEM 需要 MPI 或 bootstrapper 启动，而绝大多数深度学习训练集群没有 MPI 运行时 |
| **容器化不友好** | K8s 环境下初始化 MPI 多进程通信域极其痛苦，NCCL 的 `ncclUniqueId` 引导方式远比 `mpirun` 灵活 |
| **双通信域管理** | 训练已有 NCCL communicator（TP/DP/PP），NVSHMEM 需要额外建立一套完全独立的通信域，两套拓扑发现、两套内存注册、两套故障处理 |
| **版本兼容地狱** | NVSHMEM + MPI + libfabric + CUDA Driver 的版本矩阵兼容性问题远多于 NCCL |

### 2. 内存模型的不匹配

NVSHMEM 的核心抽象是 **Symmetric Heap（对称堆）**：

```
// NVSHMEM 模式：所有 GPU 预分配相同大小的对称内存
void *sym_heap = nvshmem_malloc(SYMMETRIC_SIZE);  // 每个 GPU 都必须分配！

// NCCL Gin 模式：直接用 PyTorch Tensor 的显存
at::Tensor buffer = torch::empty({...});  // 灵活的、已存在的显存
```

在 MoE EP 场景中，不同 expert 的 token 负载极不均衡，**对称内存要么浪费显存，要么不够用导致溢出**。NVSHMEM 解决这个问题需要额外的流控逻辑，而 Gin 模式可以直接复用 PyTorch 的动态显存管理。

---

## 二、NCCL Gin 的本质优势

### "Gin" 是什么？

NCCL Gin 不是公开 API，而是 NCCL 内部的一个**轻量级传输抽象层**。DeepEP 等团队通过逆向或合作拿到了访问权限。它的核心能力是：

```
用户 GPU Kernel
      │
      ▼
  NCCL Gin 传输 API     ← 极薄的一层（几十行代码）
      │
      ▼
  NCCL Transport Layer (IB Verbs / NVLink / ...)
      │
      ▼
      硬件
```

对比 NVSHMEM：

```
用户 GPU Kernel
      │
      ▼
  NVSHMEM API
      │
      ▼
  NVSHMEM Internal (对称堆管理、PE 映射、同步原语)
      │
      ▼
  MPI / Bootstrapper
      │
      ▼
  libfabric / UCX
      │
      ▼
      硬件
```

**Gin 的本质是把通信栈中不需要的部分全部砍掉。**

---

## 三、迁移动机的四个维度

### 1. 运维成本

```
问题场景：一个 256 GPU 的 EP 训练任务

NVSHMEM 方案：
  1. 确保所有节点安装了正确版本的 MPI
  2. 配置 mpirun 或 pmix 引导
  3. 处理每节点多进程的 PE 编号映射
  4. 与 PyTorch elastic launch 打架时需要手动协调
  5. 出问题时两个通信域的 log 互相污染

NCCL Gin 方案：
  1. PyTorch 分布式已初始化好 NCCL
  2. 直接复用，零额外配置
  3. torchrun 一行启动
```

这就是为什么 DeepEP V2 的 README 中说 **"switched to the more lightweight NCCL Gin backend"**——"lightweight" 首先指的不是代码体积，而是**运维心智负担**。

### 2. 统一通信资源管理

```
NVSHMEM 方案（双轨制）：
  ┌──────────────────────┐
  │  NVSHMEM Domain      │──▶ EP All-to-All (IB/RDMA)
  │  (独立 IB QP、独立内存)│
  └──────────────────────┘
  ┌──────────────────────┐
  │  NCCL Domain          │──▶ TP/DP/PP AllReduce (NVLink/IB)
  │  (独立 IB QP、独立内存)│
  └──────────────────────┘

  问题：两个域争抢 IB 带宽，需要手动 QoS 分配

NCCL Gin 方案（统一）：
  ┌──────────────────────┐
  │  NCCL Domain          │──▶ 所有通信统一调度
  │  (统一 QP、统一拓扑)   │     NCCL 内部做带宽仲裁
  └──────────────────────┘
```

### 3. 可以"偷" NCCL 的能力

复用 NCCL communicator 意味着免费获得：

| NCCL 能力 | 对 EP A2A 的价值 |
|-----------|-----------------|
| **拓扑感知路由** | 自动选择 NVLink vs IB，自动 rail-local 优化 |
| **NVLS (NVLink Sharp)** | In-network reduction for combine |
| **Tuner 自适应** | 自动为当前拓扑选择最优传输参数 |
| **Profiling 工具链** | nsys/Nsight 可直接追踪 |
| **Net Plugin 插件** | 未来支持新硬件（如 CX8）无需改代码 |

这些是 NVIDIA 投入上百人年打磨的能力，NVSHMEM 方案要么需要自己实现，要么根本享受不到。

### 4. 跨平台与供应商锁定

NVSHMEM 是 NVIDIA 专属的。虽然 EP 当前主要跑在 NVIDIA GPU 上，但工业界（尤其是大厂）倾向于：

- 通信层与硬件解耦
- 避免单一供应商锁定
- 为未来可能的 AMD/Intel GPU 留后路

NCCL 虽然是 NVIDIA 的，但已经有 **RCCL (AMD's NCCL fork)** 作为兼容层。基于 NCCL 抽象写的 Gin kernel，理论上更容易迁移到 ROCm 生态。

---

## 四、但这不意味着 NVSHMEM 被淘汰

NVSHMEM 在某些场景仍有不可替代的优势：

| 场景 | 为什么仍然选 NVSHMEM |
|------|---------------------|
| **HPC 传统负载** | MPI 生态完整，NVSHMEM 天然适配 |
| **极低延迟（<1μs）** | NVSHMEM 的 `put_signal` 组合比 Gin 路径更短 |
| **非 DL 的 GPU-to-GPU 通信** | 不存在 NCCL communicator 时，NVSHMEM 是唯一选择 |
| **原子操作需求** | NVSHMEM 的 `amo` (atomic memory ops) 比 Gin 暴露的原子操作更丰富 |

DeepEP V2 也保留了 NVSHMEM 作为 **legacy backend**，说明团队承认两种方案各有适用场景。

---

## 五、总结：迁移的本质

```
工业界的迁移逻辑：

NVSHMEM:  "我能做所有的事"     → 通用、完整、但重
    ↓
NCCL Gin: "我只做 EP A2A 需要的" → 专用、极简、但依赖 NCCL

迁移不是因为 NVSHMEM "做不到"，
而是因为 NVSHMEM "做了太多不需要做的事"。
```

这和工业界"微服务 → 单体"的回摆、以及"PyTorch 吞噬 ML 生态"的趋势一致：**当专用场景足够明确时，更薄、更专一的方案总是胜出。** NCCL Gin 提供了"恰好够用"的 GPU-initiated P2P 通信能力，同时继承了 NCCL 多年积累的拓扑优化、错误处理和生产稳定性——这是在 EP A2A 这个具体问题上对 NVSHMEM 的"降维打击"。