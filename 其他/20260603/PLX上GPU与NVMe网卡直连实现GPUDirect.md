# PLX上GPU与NVMe网卡直连实现GPUDirect

### 🧑‍💻 User

当nic/nvme固态硬盘与gpu位于同一个plx上时，怎么实现gpu direct， 即允许连接到 PCIe 总线的设备（如 GPU、NVMe SSD、网卡）直接互相发送数据，而无需全部挤在 CPU 的根联合体（Root Complex）上。

---

### 🤖 Assistant

当 NIC（网卡）、NVMe 固态硬盘与 GPU 连接在同一个 PCIe Switch（如 PLX 芯片）下时，这为实现真正的端到端 PCIe Peer-to-Peer (P2P) 通信提供了完美的硬件拓扑。在这种拓扑下，数据可以直接在 PLX 芯片内部完成交换（Routing），完全不需要上行到达 CPU 的 Root Complex (RC) 和系统内存。

为了实现这一点，你需要配置 NVIDIA 的 **GPUDirect RDMA**（针对网卡）和 **GPUDirect Storage (GDS)**（针对 NVMe）。

以下是实现这一目标的完整指南：

### 1. 硬件拓扑与 BIOS 准备工作

要让 PLX 芯片能够直接路由数据，必须确保系统硬件和 BIOS 允许 P2P 通信。

<details>
<summary><b>点击展开：必要的 BIOS 与系统配置</b></summary>

1.  **开启 Above 4G Decoding 和 Resizable BAR (SRIOV)**：
    GPU 的显存需要映射到系统的 PCIe 地址空间中，使得网卡和 NVMe 能够通过物理地址直接寻址 GPU 显存。
2.  **IOMMU 与 ACS (Access Control Services) 设置**：
    *   默认情况下，PCIe ACS 可能会强制所有 PCIe 事务上行到 CPU Root Complex 进行 IOMMU 权限检查，这会破坏 PLX 的直通优势。
    *   **解决方法**：在 Linux 内核启动参数（grub）中，你可能需要添加 `pcie_acs_override=downstream,multifunction` 或禁用 IOMMU 严格模式 `iommu=pt` (Passthrough)，以允许 PLX 芯片直接进行 P2P 转发。
</details>

---

### 2. 验证硬件拓扑状态

在进行软件配置前，先确认 GPU 是否识别到了 PLX 交换机。运行以下命令：

```bash
nvidia-smi topo -m
```

在输出的矩阵中，寻找 GPU 与 NIC/NVMe 之间的连接状态：
*   **PIX**：表示设备通过同一个 PCIe Switch（PLX）连接。**这是你当前架构的理想状态。**
*   **PXB**：表示通过多个 PCIe Switch 连接，但未跨越 CPU。
*   **NODE/SYS**：表示跨越了 NUMA 节点或 CPU Root Complex（未实现完美的 P2P）。

---

### 3. 实现 GPU <-> NVMe 直通：GPUDirect Storage (GDS)

GPUDirect Storage 允许 NVMe SSD 通过 DMA 直接将数据读写到 GPU 显存，绕过 CPU 的 Bounce Buffer。

#### 配置步骤：
1. **安装 CUDA 和 NVIDIA 驱动**（建议 CUDA 11.4 以上）。
2. **安装 GDS 软件包**：
   ```bash
   sudo apt-get install nvidia-gds
   ```
3. **加载 `nvidia-fs` 内核模块**：
   ```bash
   sudo modprobe nvidia-fs
   ```
4. **配置 GDS**：
   检查配置文件 `/etc/cufile.json`，确保允许 P2P 路由。
5. **验证 GDS 状态**：
   运行 GDS 检查工具，它会测试 NVMe 到 GPU 的带宽：
   ```bash
   /usr/local/cuda/gds/tools/gdscheck -p
   ```

#### 代码实现 (C/C++ API)：
在代码中，你不能使用标准的 POSIX `read()`/`write()`。必须使用 `cuFile` API：

```cpp
#include "cufile.h"

// 1. 注册 GPU 显存
cuFileBufRegister(d_gpu_buffer, buffer_size, 0);

// 2. 打开文件
cuFileHandle_t cf_handle;
cuFileDriverOpen();
// ... (使用 cuFileHandleRegister 注册文件句柄) ...

// 3. 直接从 NVMe 读入 GPU 显存 (Zero-copy)
cuFileRead(cf_handle, d_gpu_buffer, buffer_size, offset, offset);

// 4. 清理
cuFileBufDeregister(d_gpu_buffer);
cuFileDriverClose();
```

---

### 4. 实现 GPU <-> NIC 直通：GPUDirect RDMA

GPUDirect RDMA 允许支持 RDMA（如 RoCEv2 或 InfiniBand）的网卡直接读写 GPU 显存，这在分布式训练（如 NCCL）中极其重要。

#### 配置步骤：
以最常用的 Mellanox (NVIDIA) 网卡为例：
1. **安装 MLNX_OFED 驱动**：
   这是 Mellanox 的官方驱动，包含了对 GPUDirect RDMA 的底层支持。
2. **加载 `nvidia-peermem` 模块**：
   这是连接 NVIDIA GPU 驱动和 RDMA 网卡驱动的桥梁（在较老的系统中称为 `nv_peer_mem`）。
   ```bash
   sudo modprobe nvidia-peermem
   ```
   *验证模块是否加载成功：* `lsmod | grep peermem`

#### 软件与代码层面的实现：
*   **现成框架 (NCCL, MPI)**：如果你使用的是 PyTorch/TensorFlow 等深度学习框架，底层通常使用 NCCL。只需设置环境变量，NCCL 会自动发现 PIX 拓扑并使用 GPUDirect RDMA：
    ```bash
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=PIX  # 强制要求 GPUDirect 发生在 PLX 级别
    ```
*   **底层 API (ibverbs)**：如果你自己写 C++ 网络代码，标准的 RDMA 流程保持不变，唯一的区别是你传递给 `ibv_reg_mr`（注册内存区域）的指针是通过 `cudaMalloc` 分配的 GPU 显存指针，而不是 `malloc` 分配的 CPU 内存指针。`nvidia-peermem` 模块会在内核中拦截这个注册请求，并正确映射 GPU 的物理 PCI 地址给网卡。

---

### 5. 性能理论上限

当设备位于同一个 PLX 上时，它们之间的通信带宽完全取决于 PCIe 规范。例如，如果你的 PLX、GPU、NIC 和 NVMe 都是 PCIe Gen4 x16，那么理论单向带宽 $B$ 为：

$$ B = 16 \text{ GT/s} \times 16 \text{ lanes} \times \frac{128}{130} \text{ (编码损耗)} \approx 31.5 \text{ GB/s} $$

由于流量在 PLX 处发生环回（Loopback），CPU 到 PLX 的上行链路完全空闲，你可以同时实现 GPU 和 NVMe 之间的数据传输，而不影响 CPU 计算或其他外设。

---

### 🧑‍💻 User

对于拥有大量外围设备的服务器（例如8卡h200+大量cx7网卡），pcie p2p非常重要吗？

---

### 🤖 Assistant

**绝对非常重要。对于 8 卡 H200 加上大量 CX7（ConnectX-7）网卡的顶级 AI 服务器来说，PCIe P2P（特别是 GPUDirect RDMA）不是一个“锦上添花”的功能，而是系统的“命脉”。**

如果没有 PCIe P2P，这台服务器的跨节点扩展能力将彻底瘫痪，你花费巨资购买的 H200 算力和 400G 网卡带宽将会有极大的浪费。

以下是具体的原理解析和带宽数学账：

### 1. 灾难性的 CPU Root Complex 拥塞（算一笔带宽账）

CX7 网卡通常支持 400 Gbps（NDR InfiniBand 或 400G 以太网）。8 张 H200 服务器通常采用 **Rail-Optimized（轨道优化）** 拓扑，配备 8 张 CX7 网卡（每个 GPU 专属对应一张网卡）。

我们来算一下理论带宽（单向）：
$$ \text{单张 CX7 带宽} = 400 \text{ Gbps} = 50 \text{ GB/s} $$
$$ \text{8 张 CX7 并发带宽} = 8 \times 50 \text{ GB/s} = 400 \text{ GB/s} $$

**如果没有 PCIe P2P（传统数据路径）：**
当 GPU 需要将计算结果发送到其他节点时，数据必须走：`GPU -> PCIe Switch -> CPU 内存 -> PCIe Switch -> CX7 网卡`。
这意味着 **400 GB/s** 的数据需要先写入 CPU 内存（系统内存），然后再以 **400 GB/s** 的速度被网卡读走。

*   **内存带宽瓶颈**：即使是最新的 AMD Genoa 或 Intel Sapphire Rapids 处理器，其 12 通道/8 通道 DDR5 内存的理论总带宽大约在 $300 \text{ GB/s} \sim 460 \text{ GB/s}$ 左右。8 张网卡的并发 IO 流量会**瞬间击穿整个 CPU 的内存带宽上限**。
*   **PCIe 通道瓶颈**：CPU 连接到下游 PCIe Switch 的 UPI/PCIe 链路根本无法同时承载如此庞大的双向数据流（上行+下行）。

<details>
<summary><b>点击展开：有 PCIe P2P（GPUDirect RDMA）的数据路径</b></summary>

数据路径变为：`GPU -> PCIe Switch (PLX) -> CX7 网卡`。
*   数据在 PCIe Switch 内部直接掉头（Loopback）转发。
*   **零 CPU 参与，零系统内存占用**。
*   8 张网卡的 400 GB/s 流量全部分散在底层的各个 PCIe Switch 上并发执行，完美发挥硬件极限。
</details>

---

### 2. 跨节点分布式训练（Scale-out）的刚需

在 8 卡 H200 节点内部（Intra-node），GPU 之间的数据交换使用的是 **NVLink / NVSwitch**，带宽极高（H200 双向可达 $900 \text{ GB/s}$），此时并不依赖 PCIe。

但是，现代大语言模型（LLM）的训练通常需要几十上百台这样的服务器组成集群（Inter-node）。
*   在进行 **All-Reduce** 等集合通信操作时，GPU 必须频繁地与其他服务器上的 GPU 交换梯度数据。
*   **NCCL（NVIDIA Collective Communications Library）** 高度依赖 GPUDirect RDMA（PCIe P2P）。如果不开启 P2P，网络延迟会急剧增加，导致这 8 张昂贵的 H200 经常处于“等待网络数据”的空闲状态（GPU Utilization 暴跌）。

---

### 3. 海量存储的吞吐要求 (GPUDirect Storage)

H200 拥有 141GB 的 HBM3e 显存，8 卡总显存高达 1.1TB。在训练 Checkpoint（检查点）保存/恢复，或者加载海量数据集时：
*   如果使用传统的 CPU Bounce Buffer，频繁的 CPU 拷贝会消耗大量 CPU 周期，拖慢整个训练循环。
*   通过将多块 NVMe SSD 挂载在同一个 PCIe Switch 下，并开启基于 P2P 的 **GPUDirect Storage (GDS)**，显存数据可以直接“倒”进 NVMe 阵列，峰值速度可达几十 GB/s，瞬间完成 Checkpoint 的保存。

### 总结

对于 "8卡 H200 + 8卡 CX7" 这种怪兽级服务器：
**PCIe P2P 不是用来“提升”性能的，它是用来“防止系统崩溃性降级”的。** 这类服务器的硬件拓扑设计（如 NVIDIA DGX/HGX 参考架构中的多个 PCIe Switch 级联）从第一天起，就是**完全围绕 PCIe P2P 和 RDMA 来设计的**。禁用或未正确配置 P2P，相当于买了一辆 F1 赛车却在泥泞的乡间小道上开。