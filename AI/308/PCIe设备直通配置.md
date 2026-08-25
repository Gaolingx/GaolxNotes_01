# PCIe设备直通配置

### 🧑‍💻 User

GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt pcie_acs_override=downstream,multifunction"这个命令有什么用？

---

### 🤖 Assistant

# GRUB 内核启动参数解析

这行配置是 Linux GRUB 引导加载器中的内核启动参数，主要用于 **启用 IOMMU（输入/输出内存管理单元）** 和 **PCIe 设备直通（Passthrough）** 相关功能，常用于虚拟化场景（如 KVM/QEMU、PVE）中。

---

## 各参数详解

| 参数 | 作用 |
|------|------|
| `intel_iommu=on` | 启用 Intel VT-d 硬件虚拟化中的 IOMMU 功能 |
| `iommu=pt` | 设置 IOMMU 为 **passthrough（直通）模式** |
| `pcie_acs_override=downstream,multifunction` | 强制绕过 PCIe ACS（Access Control Services）限制，解决 IOMMU 分组过大问题 |

---

## 详细说明

### 1. `intel_iommu=on`

- **作用**：开启 Intel 平台的 **VT-d**（Virtualization Technology for Directed I/O）功能。
- **意义**：IOMMU 允许将物理硬件设备安全地映射到虚拟机内存空间中，这是 **PCIe 设备直通（PCI Passthrough）** 的前提条件。
- **AMD 平台对应**：`amd_iommu=on iommu=pt`

### 2. `iommu=pt`

- **作用**：将 IOMMU 设为 **Passthrough 模式**。
- **意义**：
  - 对于宿主机（Host），直通模式让主机绕过 IOMMU 地址转换，**减少性能开销**。
  - 对于直通给虚拟机的设备，IOMMU 仍然生效，保证隔离性和安全性。
  - 简单来说：**主机不翻译，虚拟机才翻译**，兼顾性能和安全。

### 3. `pcie_acs_override=downstream,multifunction`

这是最关键的参数，用于解决 **IOMMU 分组（IOMMU Group）过大** 的问题。

#### 什么是 ACS？
ACS（Access Control Services）是 PCIe 规范中的一项功能，用于控制设备之间的 peer-to-peer 访问。只有支持 ACS 的设备才能被正确隔离到不同的 IOMMU 分组中。

#### 为什么需要 override？
许多消费级主板/设备的 PCIe 交换芯片**不支持 ACS** 或**默认未启用 ACS**，导致：
- 多个不相关的设备被分到**同一个 IOMMU Group**。
- 同一组内的设备**必须一起直通**给虚拟机，否则会报错。
- 严重限制设备直通的灵活性。

#### 两个子选项：

| 子选项 | 含义 |
|--------|------|
| `downstream` | 强制所有 **下游端口（Downstream Port）** 视为支持 ACS，从而在交换芯片处切分分组 |
| `multifunction` | 强制 **多功能设备（Multifunction Device）** 的不同功能（function）被视为支持 ACS，从而将同一设备的不同功能分到不同组 |

---

## 使用场景

```
┌─────────────────────────────────────────┐
│           宿主机 (Proxmox/Arch/Ubuntu)    │
│                                         │
│   ┌──────────┐    ┌──────────────────┐  │
│   │  VM-101  │    │     VM-102       │  │
│   │ ┌──────┐ │    │  ┌───────────┐   │  │
│   │ │ GPU  │◄┼────┼──│ 网卡直通  │   │  │
│   │ │直通  │ │    │  └───────────┘   │  │
│   │ └──────┘ │    │                  │  │
│   └──────────┘    └──────────────────┘  │
│         ↑               ↑               │
│    IOMMU Group 15   IOMMU Group 22      │
│    (ACS override    (ACS override       │
│     将GPU独立分组)    将网卡独立分组)     │
└─────────────────────────────────────────┘
```

典型应用包括：

- 🎮 **GPU 直通**（Gaming VM / VFIO）
- 🌐 **网卡直通**（专用网络设备给虚拟机）
- 💾 **HBA/磁盘控制器直通**（TrueNAS / 存储虚拟化）
- 🖥️ **PVE 单设备直通**（避免整组直通）

---

## 配置方法

```bash
# 1. 编辑 GRUB 配置文件
sudo nano /etc/default/grub

# 2. 修改 GRUB_CMDLINE_LINUX_DEFAULT（或 GRUB_CMDLINE_LINUX）
GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt pcie_acs_override=downstream,multifunction"

# 3. 更新 GRUB
sudo update-grub          # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # RHEL/CentOS/Fedora/Arch

# 4. 重启生效
sudo reboot
```

---

## 验证是否生效

```bash
# 检查 IOMMU 是否启用
dmesg | grep -e IOMMU -e DMAR

# 查看设备分组
for d in /sys/kernel/iommu_groups/*/devices/*; do
    n=${d#*/iommu_groups/*}; n=${n%%/*}
    printf 'IOMMU Group %s %s\n' "$n" "${d##*/}"
done | sort -V
```

> ⚠️ **注意**：`pcie_acs_override` 是一种**软件层面的变通方案**，它会绕过硬件 ACS 保护。在大多数消费级平台上这是安全的，但在生产环境中应优先选择**硬件原生支持 ACS** 的设备（如服务器主板、ASRock Rack 等）。

---

### 🧑‍💻 User

针对大模型训练场景pcie p2p这个设置是有必要的吗？

---

### 🤖 Assistant

# PCIe P2P 与大模型训练中 `pcie_acs_override` 的必要性分析

## 结论先行

> **取决于硬件平台**：服务器级平台（DGX/HGX）通常**不需要**；消费级/工作站多卡平台**可能需要**。

---

## 核心矛盾：ACS vs P2P

```
┌──────────────────────────────────────────────┐
│              PCIe Root Complex                │
│                    │                          │
│         ┌──────────┴──────────┐              │
│         │   PCIe Switch (PLX)  │              │
│         │   ACS enabled? ──────┼──→ YES: P2P 被阻断 ❌  │
│         │                      │   NO:  P2P 直接通行 ✅  │
│   ┌─────┴───┐           ┌─────┴───┐          │
│   │  GPU 0  │◄──P2P──►│  GPU 1  │          │
│   │ (NCCL)  │  DMA     │ (NCCL)  │          │
│   └─────────┘          └─────────┘          │
└──────────────────────────────────────────────┘
```

ACS 的本质是控制 PCIe 设备间的 **peer-to-peer 访问**。当 ACS 在 PCIe Switch 的下游端口启用时，会**阻止** GPU 之间的直接 DMA 传输，迫使数据绕行 CPU 内存（root complex），导致：

| 指标 | P2P 直连 | ACS 阻断后（绕行CPU） |
|------|----------|---------------------|
| GPU↔GPU 带宽 | ~25-32 GB/s (PCIe 4.0 x16) | ~12-16 GB/s（减半） |
| 延迟 | 低（直接 DMA） | 高（经过主机内存中转） |
| CPU 占用 | 几乎为零 | 显著增加 |
| NCCL 性能 | 最优 | 严重退化 |

---

## 分场景判断

### 场景一：服务器级平台（DGX / HGX / 服务器主板）→ ❌ 不需要

```
NVIDIA DGX A100/H100 拓扑：
┌─────────────────────────────────────────┐
│              NVSwitch (NVLink)          │  ← GPU间走NVLink
│         全互联 600-900 GB/s             │     完全绕过PCIe P2P
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐
│  │GPU│ │GPU│ │GPU│ │GPU│ │GPU│ │GPU│ │GPU│ │GPU│
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘
└─────────────────────────────────────────┘
```

- **NVLink/NVSwitch 全互联**，GPU 间通信不走 PCIe P2P
- PCIe Switch 固件已针对 P2P 优化，ACS 默认不阻断
- 硬件原生支持，无需内核参数干预

### 场景二：工作站/消费级多卡（无 NVLink）→ ⚠️ 可能需要

```
典型工作站拓扑（如 2-4x RTX 3090/4090）：
┌──────────────────────────────────────────────┐
│              CPU (Root Complex)               │
│         ┌──────────┴──────────┐              │
│         │   PCIe Switch (PLX)  │              │
│         │   ACS 可能默认启用 ──┼──→ P2P 被阻断  │
│   ┌─────┴───┐           ┌─────┴───┐          │
│   │  GPU 0  │◄─需要P2P─►│  GPU 1  │          │
│   │RTX 4090 │   NCCL     │RTX 4090 │          │
│   └─────────┘           └─────────┘          │
└──────────────────────────────────────────────┘
```

- **无 NVLink**，GPU 间通信**完全依赖 PCIe P2P**
- 消费级主板/PCIe Switch 可能默认启用 ACS
- NCCL 检测到 P2P 不可用时，fallback 到通过主机内存中转，性能大幅下降

### 场景三：混合拓扑（部分 NVLink + 部分 PCIe）→ 🔍 视情况

例如双路主板 + 混合连接：
- 同 CPU 节点下的 GPU 可能有 NVLink
- 跨 CPU 节点（跨 NUMA）的 GPU 需要 PCIe P2P（甚至跨 QPI/UPI）

---

## 如何检测当前系统是否需要

### 1. 检查 P2P 是否可用

```bash
# 使用 NVIDIA 工具检查 GPU 间 P2P 能力
nvidia-smi topo -m

# 输出示例（P2P 可用）：
#         GPU0  GPU1  GPU2  GPU3
#  GPU0    X   NV12  PIX   SYS
#  GPU1   NV12   X   SYS   PIX
#  GPU2   PIX   SYS    X   NV12
#  GPU3   SYS   PIX   NV12   X
#
# PIX = 同一PCIe Switch下，支持P2P ✅
# NV12 = NVLink 12条lane ✅
# SYS = 跨QPI/UPI，P2P受限 ⚠️
```

### 2. 检查 NCCL 是否实际使用 P2P

```bash
# 运行 NCCL 测试，观察日志
NCCL_DEBUG=INFO python -c "
import torch.distributed as dist
dist.init_process_group('nccl', init_method='env://')
# ... 训练代码
"

# 关键日志行：
# NCCL INFO Channel 00 : 0[xx] -> 1[xx] via P2P/IPC    ← P2P 正常 ✅
# NCCL INFO Channel 00 : 0[xx] -> 1[xx] via SHM        ← P2P 被阻断，走共享内存 ❌
```

### 3. 检查 ACS 状态

```bash
# 查看每个 PCIe 设备的 ACS 配置
lspci -vvv | grep -A 5 "Access Control"

# 如果看到 "Access Control Services: Enabled"
# 说明 ACS 已启用，可能阻断 P2P
```

---

## 实际影响 benchmark

以 2× RTX 3090 PCIe 4.0 x16 为例（无 NVLink）：

| 配置 | all-reduce 带宽 | 训练吞吐量（相对值） |
|------|----------------|-------------------|
| P2P 正常（ACS 关闭） | ~24 GB/s | 100% (基准) |
| P2P 被阻断（ACS 开启） | ~8 GB/s | ~75-85% |
| `pcie_acs_override` 修复后 | ~24 GB/s | ~100% |

> 💡 对于**大模型训练**，通信开销占比越高（如张量并行、流水线并行），P2P 的影响越大。纯数据并行（DP）因梯度同步占比较小，影响相对可控。

---

## 建议配置

<details>
<summary>📋 大模型训练推荐内核参数</summary>

```bash
# /etc/default/grub

# 服务器级（DGX/HGX，有NVLink）—— 最简配置
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt"

# 工作站/消费级多卡（无NVLink，依赖PCIe P2P）—— 加上ACS override
GRUB_CMDLINE_LINUX="intel_iommu=on iommu=pt pcie_acs_override=downstream,multifunction"

# 如果完全不需要设备直通（纯训练机），可以更激进
GRUB_CMDLINE_LINUX="intel_iommu=off"   # 直接关闭IOMMU，P2P无障碍
```

```bash
# 更新并重启
sudo update-grub && sudo reboot
```

</details>

---

## 总结

```
                    NVLink 全互联？
                   /              \
                 是                 否
                 |                  |
        ┌────────┘                  ┌───────────┐
        │ 不需要 override           │ 检测 P2P   │
        │ 硬件原生支持              │ nvidia-smi │
        │ NVSwitch 绕过 PCIe        │ topo -m    │
                              │            │
                          P2P 可用       P2P 不可用
                              │            │
                         不需要          ✅ 需要
                                      pcie_acs_override
```

| 平台类型 | 有无 NVLink | 需要 `pcie_acs_override`？ |
|----------|------------|--------------------------|
| DGX/H100 服务器 | ✅ NVSwitch | ❌ 不需要 |
| 服务器 + A100 (SXM) | ✅ NVLink | ❌ 不需要 |
| 服务器 + A100 (PCIe) | ❌ | ⚠️ 检测后决定 |
| 工作站 + RTX 4090 ×4 | ❌ | ✅ **建议添加** |
| 消费级主板 + 消费级 GPU | ❌ | ✅ **强烈建议** |

> **核心原则**：如果你的 GPU 之间没有 NVLink，且训练依赖 NCCL 进行多卡通信，那么确保 PCIe P2P 畅通是性能的关键，`pcie_acs_override` 在消费级/工作站硬件上是有必要的。