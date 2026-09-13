# QEMU虚拟化技术详解

### 🧑‍💻 User

解释ovmf包。解释virtio驱动和硬盘格式。除了ide，termux还提供了哪些，各有什么特点

---

### 🤖 Assistant

# 一、OVMF 包是什么

**OVMF = Open Virtual Machine Firmware**，是 TianoCore **EDK II** 项目提供的、面向虚拟机的 **UEFI 固件**实现。你可以把它理解为"虚拟机的 UEFI BIOS"。

### 它解决什么问题

QEMU 默认给 x86_64 虚拟机用的是 **SeaBIOS**（传统 Legacy BIOS）。但现代系统越来越多地要求 UEFI 启动：

- Windows 8/10/11 默认走 UEFI + GPT 分区表
- 很多 Linux 发行版也推荐 UEFI + GPT
- UEFI 支持 Secure Boot、更大的启动分区、更规范的启动流程

OVMF 就是给 QEMU/KVM 提供这套 UEFI 环境。

### 关键文件

| 文件 | 作用 |
|------|------|
| `OVMF_CODE.fd` | 固件代码（只读） |
| `OVMF_VARS.fd` | NVRAM 变量存储（可读写，保存启动项、Secure Boot 密钥等） |
| `OVMF_CODE.secboot.fd` / `OVMF_VARS.secboot.fd` | 带 Secure Boot 的版本 |

### Termux 中的用法

Termux 提供了官方 `ovmf` 包（从 Fedora 的 edk2 RPM 抽取，版本基于 edk2 20231122）：

```bash
pkg install ovmf
```

安装后固件位于 `$PREFIX/share/edk2/ovmf/`（还有 `ia32/`、`aarch64/`、`arm/`、`riscv64/` 等架构目录），并在 `$PREFIX/share/qemu/firmware/` 生成 JSON 描述供 QEMU 自动识别。

用 OVMF 启动虚拟机：

```bash
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=$PREFIX/share/edk2/ovmf/OVMF_CODE.fd \
  -drive if=pflash,format=raw,file=$PREFIX/share/edk2/ovmf/OVMF_VARS.fd \
  -hda disk.img
```

> 注意：`OVMF_VARS.fd` 是可写的 NVRAM，建议先复制一份到工作目录再使用，避免污染原始文件。装 Windows/新 Linux 基本都要用它，否则只能用 Legacy BIOS。

---

# 二、virtio 驱动 与 硬盘格式

## 1. virtio 驱动（半虚拟化驱动）

**virtio** 是 Red Hat 制定的**半虚拟化（paravirtualized）I/O 标准**（现为 OASIS 标准）。

### 核心思想

传统的全虚拟化设备（如模拟真实 IDE 硬盘、真实 e1000 网卡）需要 QEMU **完整模拟硬件行为**，Guest 每发一个 I/O 请求都要经过大量指令翻译，开销大、性能差。

virtio 的思路是：**Guest 里装一个"知道自己在虚拟机里"的驱动**，这个驱动直接和宿主机（hypervisor）约定好一种高效的内存队列（virtqueue）来交换数据，**不需要模拟真实硬件寄存器**，从而接近原生性能。

### 常见 virtio 设备

| 设备 | 用途 |
|------|------|
| `virtio-blk` | 块设备（硬盘） |
| `virtio-net` | 网卡 |
| `virtio-scsi` | SCSI 控制器 |
| `virtio-balloon` | 内存气球（动态调整内存） |
| `virtio-gpu` / `virtio-vga` | 显卡 |
| `virtio-rng` | 随机数 |
| `virtio-serial` / `virtio-console` | 串口/通信通道 |

### 驱动获取

- **Linux**：内核自带（模块 `virtio_blk`、`virtio_net`、`virtio_scsi` 等），无需额外安装。
- **Windows**：默认不带，需要安装 **virtio-win** 驱动包（Fedora/Red Hat 提供的 ISO，如 `virtio-win-*.iso`），否则 Windows 看不到 virtio 硬盘/网卡。装 Windows 时通常先挂 IDE 盘 + 加载 virtio 驱动，或直接挂驱动 ISO 选盘。

### QEMU 中启用

```bash
# 硬盘
-drive file=disk.img,if=virtio
# 或
-drive file=disk.img,if=none,id=drive0 -device virtio-blk-pci,drive=drive0
# 网卡
-netdev user,id=n0 -device virtio-net-pci,netdev=n0
```

---

## 2. 硬盘（磁盘镜像）格式

用 `qemu-img` 创建/转换，QEMU 支持多种格式：

| 格式 | 来源 | 特点 |
|------|------|------|
| **raw** | 通用 | 裸数据、无元数据，速度最快、开销最低，但**占用=声明大小**（不稀疏），无快照/压缩 |
| **qcow2** ⭐ | QEMU 原生 | **推荐**：稀疏（按需增长）、快照、压缩、加密、支持 backing file（基于母盘）、动态分配 |
| qcow | QEMU 旧版 | qcow2 的前身，已被取代 |
| vmdk | VMware | 与 VMware 互通 |
| vdi | VirtualBox | 与 VirtualBox 互通 |
| vhd / vpc | Microsoft | Virtual PC / Hyper-V 旧格式 |
| vhdx | Microsoft | Hyper-V 新格式，支持更大容量、日志 |
| qed | QEMU | Enhanced Disk，已弃用，被 qcow2 取代 |
| bochs | Bochs | Bochs 模拟器镜像 |
| cloop | Knoppix | 压缩只读回环镜像 |
| dmg | Apple | macOS 磁盘镜像 |
| parallels | Parallels | Parallels Desktop 格式 |
| luks | LUKS | 加密镜像 |

```bash
qemu-img create -f qcow2 disk.img 10G   # 建 10G 稀疏镜像
qemu-img info disk.img                  # 查看信息
qemu-img convert -f raw -O qcow2 disk.img disk.qcow2   # 格式互转
```

> Termux 的 QEMU 构建脚本（`build.sh`）里显式开启了 `--enable-bochs --enable-cloop --enable-dmg --enable-parallels --enable-qed`，即这些"非主流"格式在 Termux 里也都能读写；raw/qcow2/vmdk/vdi/vpc/vhdx/luks 是默认自带。

---

# 三、除了 IDE，Termux 的 QEMU 还提供哪些磁盘总线

> 这里把"IDE"理解为磁盘**总线/控制器**（`if=ide`）。QEMU 的 x86_64 机器支持多种磁盘连接方式，Termux 的 `qemu-system-x86-64`（完整构建）全部可用。

| 总线 | 挂载方式 | 性能 | 特点 |
|------|---------|------|------|
| **IDE (PATA)** | `-hda` / `if=ide` | 低 | 最兼容、无需驱动；但最多 **4 个**设备（2 通道 × 主从），无热插拔，性能最差 |
| **SATA / AHCI** | `-device ich9-ahci` + `-device ide-hd` | 中高 | 每控制器 **6 端口**，支持 NCQ、**热插拔**；现代机器（q35）常用，Windows/Linux 都原生支持 |
| **SCSI** | `if=scsi`（megasas / lsi53c895a） | 中 | 可挂大量磁盘，企业级语义；普通场景用得少 |
| **virtio-blk** | `if=virtio` / `-device virtio-blk-pci` | 高 | 半虚拟化，CPU 开销最低；**Linux 原生，Windows 需装 virtio 驱动** |
| **virtio-scsi** | `-device virtio-scsi-pci` | 高 | 半虚拟化 + SCSI：可挂非常多盘且支持热插拔，兼具性能和灵活性 |
| **NVMe** | `-device nvme` + `if=none` | 高 | 模拟 PCIe NVMe SSD，**低延迟、高 IOPS**，支持多个命名空间（namespace）；现代 Windows/Linux 原生支持，是近年高性能首选 |
| **USB 存储** | `-device usb-storage` | 低 | 即插即用、便携，适合挂 U 盘/移动盘场景 |

### 举例

```bash
# IDE（默认）
qemu-system-x86_64 -hda disk.img

# SATA/AHCI（推荐给 Windows 装机，兼容 + 可热插拔）
qemu-system-x86_64 \
  -drive file=disk.img,if=none,id=hd0 \
  -device ich9-ahci,id=ahci \
  -device ide-hd,drive=hd0,bus=ahci.0

# virtio-blk（Linux 性能最佳，Windows 需驱动）
qemu-system-x86_64 -drive file=disk.img,if=virtio

# NVMe（高性能/低延迟，现代系统首选）
qemu-system-x86_64 \
  -drive file=disk.img,if=none,id=nvme0 \
  -device nvme,drive=nvme0,serial=deadbeef
```

### 怎么选

- **装 Windows**：先用 **SATA/AHCI** 或 **IDE**（免驱动、兼容好），装完系统后再换 virtio/NVMe 提性能；或在安装时挂 virtio-win 驱动 ISO。
- **跑 Linux**：直接用 **virtio-blk** 或 **NVMe**，性能最好。
- **追求极致性能**：**NVMe** 和 **virtio-blk** 是首选，IDE 最慢。

### ⚠️ Termux 特有注意点

1. Termux 的 QEMU 构建虽然开了 `--enable-kvm`，但**非 root 的 Android 上通常没有 /dev/kvm**，所以实际跑在 **TCG 软件模拟**下，速度偏慢。此时 virtio/NVMe 的优势主要体现在**降低 CPU 翻译开销**，而不是原生 KVM 那种质的飞跃。
2. Termux 有两个 QEMU 包：
   - `qemu-system-x86-64-headless`（主仓库）：无 GUI，支持 VNC/SPICE/curses，**同时含 9 种架构的系统仿真 + 用户态仿真**；
   - `qemu-system-x86-64`（x11 仓库）：带 SDL/GTK/OpenGL/virgl 图形，仅系统仿真。
   - 配套还有 `qemu-common`（共享数据）、`qemu-utils`（`qemu-img`、`qemu-io` 等工具）、`ovmf`（UEFI 固件）。

如果你告诉我你具体想装什么系统（Windows/Linux、什么架构），我可以直接给你一套能跑的完整启动命令。