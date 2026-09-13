# DeepEP换引擎

### 🧑‍💻 User

use websearch, why deepseek deepepv1 use nvshemem api?

---

### 🤖 Assistant

## DeepEP V1 为何直接使用 NVSHMEM 内部 API（而非公开 API）

DeepEP V1 没有调用 NVSHMEM 的标准公开 API（如 `nvshmemx_putmem_nbi_warp` 等），而是直接深入使用了 NVSHMEM 的 **IBGDA 内部接口**（`nvshmemi_ibgda_*`），这是 DeepSeek 团队刻意为之的架构决策。核心原因可以从 **GitHub Issue #18** 中两位 DeepSeek 核心开发者（**LyricZhao** 和 **sphish**）的官方回复中找到答案：

---

### 1. 微秒级延迟的性能要求

> *"For microsecond-level performance, the public APIs are not very good."*

公开 NVSHMEM API 为了保证通用正确性，内部会执行大量的 **同步（sync）和原子操作（atomic operations）**，这些额外开销在微秒级别的通信场景下变得不可接受。DeepEP 的低延迟 kernel 需要在约 **77μs–194μs** 内完成一次 dispatch/combine，任何多余的同步屏障都会显著拖慢整体速度。

### 2. 缺乏细粒度的保序操作

> *"One such limitation is the lack of support for fine-grained order-preserving operations."*

在 MoE dispatch 中，存在一个经典需求：**传输完数据后，紧接着发送一个 flag 信号通知对方**。使用公开 API 时，唯一能保证顺序的方法是调用 **`nvshmem_quiet()`**——这是一个非常重的全局同步原语。DeepEP 通过直接操控底层 QP 和 IBGDA 原语，可以在不需要全局静默的情况下实现轻量级的顺序保证。

### 3. 每个 Expert 独占 QP（Queue Pair）的需求

这是 DeepEP 架构中最关键的设计假设之一：

- **不同 expert 之间没有数据依赖** → 使用不同的 QP 可以最大化并行度
- **同一 expert 的数据必须保序** → 同一 expert 的所有通信必须在同一个 QP 内处理

> *"Our requirement to ensure each expert process exclusively uses a dedicated QP, which cannot be achieved through NVSHMEM's native interfaces."*

公开 NVSHMEM API 的 QP 分配是黑盒的，开发者无法精确控制哪个 QP 服务于哪个 expert。DeepEP 直接操作内部接口，实现了 **expert ↔ QP 的一对一绑定**。

### 4. Adaptive Routing（AR）的效率问题

> *"The public API is not convenient for AR enabled — maybe you should do a quiet operation, which is slow."*

启用 InfiniBand 的自适应路由时，公开 API 同样需要昂贵的 quiet 操作，而内部接口允许团队 **协同设计（co-design）** 通信模式，绕过这些瓶颈。

### 5. NVSHMEM Patch 中的额外优化

DeepEP 并不直接使用原版 NVSHMEM，而是依赖其 **修改过的 NVSHMEM（modified NVSHMEM / NVSHMEM patch）**，其中包含：

- **QP 索引重排（shuffling QP indices）**：优化物理 QP 与 NIC 端口之间的映射，降低拥塞
- 其他面向 MoE 通信模式的定制优化

> *"There are other optimizations in the NVSHMEM patch, e.g. shuffling the QP indices, all these can improve performance."*

---

### 总结

| 维度 | 公开 NVSHMEM API | DeepEP 直接调用内部 API |
|------|:---:|:---:|
| 同步开销 | 大量隐式 sync/atomic | 最小化同步，按需精确控制 |
| 保序机制 | 只能靠 `nvshmem_quiet()` 全局静默 | 轻量级 per-QP 顺序保证 |
| QP 控制 | 黑盒分配 | Expert ↔ QP 一对一精确绑定 |
| AR 支持 | 依赖 costly quiet | 协同设计，零额外开销 |
| QP 索引 | 默认 | 重排优化以降低拥塞 |

> **补充**：DeepEP V2 已完全放弃了 NVSHMEM 后端，转而使用更轻量的 **NCCL Gin 后端**，同时将 SM 占用从 24 个降低到 4–6 个，性能反而提升最高 1.3 倍——这也从侧面印证了 NVSHMEM 在 MoE 场景下的局限性。

---

### 🧑‍💻 User

why DeepEP V2 is now moving away from NVSHMEM toward NCCL GIN?

---

### 🤖 Assistant

我已经收集到了足够的资料。以下是对 DeepEP V2 从 NVSHMEM 迁移到 NCCL Gin 的全面分析。

---

## DeepEP V2 为何放弃 NVSHMEM，转向 NCCL Gin？

DeepEP V2 的 README 开篇就明确声明了这一架构变更：

> *"V2 has also switched from the NVSHMEM backend to the more lightweight **NCCL Gin backend**."*

这不是一个"修修补补"式的升级，而是 **DeepEP V2 整体重构的核心支柱之一**。答案可以从三个角度理解：**NVSHMEM V1 的痛点、NCCL Gin 是什么、以及为什么 Gin 是正确答案。**

---

### 一、回顾：NVSHMEM V1 的"原罪"

从上一次讨论（Issue #18）中，LyricZhao 和 sphish 已经清楚阐述了 V1 使用 NVSHMEM 内部 API 的原因，但这些"变通方案"本身就埋下了技术债务：

| V1 的 NVSHMEM 策略 | 代价 |
|---|---|
| 直接调用 `nvshmemi_ibgda_*` 内部 API | 绕过公开 API = 依赖私有实现，升级 NVSHMEM 版本极易 break |
| 维护自定义 NVSHMEM patch（QP 索引重排等） | 额外的维护负担，patch 与上游版本绑定 |
| Expert ↔ QP 一对一绑定 | 缺少抽象，扩展性差 |
| 靠 `quiet` / busy-polling 做同步 | 昂贵的同步开销，占用大量 SM |

更重要的是，V1 的 low-latency (LL) 路径采用了**双缓冲（ping-pong）设计**，通过 NVLink 上的 in-kernel atomic/flag polling 来 overlap 通信与计算。这个设计虽然巧妙，但**内存消耗大、L2 cache 压力高、NVLink 原子轮询导致拥塞**——全部是瓶颈。

---

### 二、NCCL Gin 是什么？

**NCCL Gin** 是 NVIDIA 在 NCCL 中引入的 **Device-Initiated Communication API**（从 NCCL 2.12+ 开始引入，2.29U1 后正式支持对称 GIN kernel：`NCCL_SYM_GIN_KERNELS_ENABLE`）。

它的核心理念是：

> **允许 GPU kernel（device code）直接通过 NCCL 发起 RDMA 通信，而无需 NVSHMEM 这个中间层。**

关键特性：

- **Header-only & lightweight** —— 只需 include 头文件，不需要额外的 runtime 库依赖
- **复用现有 NCCL communicator** —— DeepEP 用户可以复用已有的 `ncclComm_t`，不需要再创建独立的 NVSHMEM 上下文
- **零额外 SM 占用（0 SM Engram/PP/CP）** —— 通信可以由 copy engine 或 RDMA 引擎完成，不消耗 CUDA core
- **与 PyTorch 生态天然集成** —— PyTorch 已经深度依赖 NCCL，Gin 是 NCCL 的一部分

用 NCCL 官方文档的话说，Gin 提供了 **"Building Blocks for Custom Communication Kernels"**——包括 `ReduceSum`、`Copy`、`ReduceSumCopy` 以及 Lambda-based 自定义布局操作。

---

### 三、为什么 NCCL Gin 是 V2 的正确答案？

从 DeepEP V2 的性能数据和设计文档中，可以归纳出以下关键动机：

#### 1. 消除对 NVSHMEM patch 的依赖

V1 依赖一个自定义的 NVSHMEM patch（QP 索引重排等），这导致：
- 与 NVSHMEM 上游版本强绑定，升级困难
- 社区用户难以部署（大量 Issue 都在问 NVSHMEM 版本兼容性问题，如 #707、#691）

V2 只依赖 **NCCL ≥ 2.30.4**（通过 `pip install nvidia-nccl-cu13` 即可），远更轻量。

#### 2. SM 占用从 24 降到 4–6（最高 4x 节省）

这是 V2 最引人注目的改进：

> *"For V3-like legacy training, SM usage reduced from 24 to 4–6 while maintaining equivalent or better performance."*

NVSHMEM V1 的 LL 路径需要大量 SM 来做 atomic polling、flag 同步、以及双缓冲管理。NCCL Gin 将这些同步下推到更底层的硬件机制（RDMA atomic、copy engine），释放了大量 SM 给实际计算使用。

#### 3. 性能反而提升 1.3x

> *"V2 achieves up to 1.3x peak performance, while saving up to 4x SM count."*

这似乎违反直觉——怎么去掉了 ping-pong overlap 还更快？关键在于：

- **NCCL Gin 的底层实现经过了 NVIDIA 专业优化**，包括对 InfiniBand Adaptive Routing、PCI Relaxed Ordering、GDR 等特性的深度利用
- **不再需要 expensive `quiet` 操作** —— 同步开销极小
- **解析式 SM/QP 计算**（analytical SM & QP count calculation）取代了 auto-tuning，在最优配置下运行

#### 4. 更大的 scale-up & scale-out 域（EP2048）

V1 的 Expert ↔ QP 一对一绑定模型，在 EP 规模增大时 QP 数爆炸。NCCL Gin 的抽象允许更灵活的 QP 复用和路由策略，支持到 EP2048。

#### 5. Fully JIT，零 CUDA 编译安装

V2 的所有 kernel 都是运行时 JIT 编译的。这意味着：
- `pip install` 即可，不需要预编译 CUDA 代码
- 自动适配目标架构（SM90/SM100）
- 缓存目录 `$HOME/.deep_ep` 避免重复编译

这与 NCCL Gin 的 header-only 设计天然契合——Gin 的接口就是头文件 + 运行时链接 NCCL。

---

### 四、总结对比

| 维度 | DeepEP V1 (NVSHMEM) | DeepEP V2 (NCCL Gin) |
|---|---|---|
| **依赖** | NVSHMEM + 自定义 patch | NCCL ≥ 2.30.4（pip 安装） |
| **SM 占用** | 24 SMs | 4–6 SMs（节省 4x） |
| **性能** | 基线 | 最高 1.3x |
| **扩展性** | EP 受限 | 支持 EP2048 |
| **安装** | 需编译 CUDA + NVSHMEM patch | Fully JIT，pip install |
| **QP 管理** | Expert ↔ QP 手动一对一绑定 | 解析式自动计算 |
| **同步机制** | 双缓冲 + atomic polling | 轻量 pre/post barrier |
| **生态集成** | 独立通信库 | 复用 PyTorch NCCL communicator |

> **一句话总结：NVSHMEM V1 的方案是为了极致性能而"铁轨上开赛车"——快，但定制化太重，扩展性差。NCCL Gin 让 DeepEP V2 得以在标准化的高速公路上用更少的 SM 跑到更快的速度。**