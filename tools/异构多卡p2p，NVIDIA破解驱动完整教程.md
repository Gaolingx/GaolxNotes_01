# 第一篇：科普与项目准备

> **目标**：理解破解驱动的原理，准备好所有需要的项目和环境。

---

## 一、你到底在装什么

NVIDIA 消费卡（RTX 20/30/40 系列）出厂**禁止** GPU 间 P2P 直连通信。数据传输必须经过 CPU 中转（~6 GB/s）。

破解驱动通过 **BAR1 P2P** 技术实现直连（~13-50 GB/s），并保留 NVLink 高速连接（~94 GB/s）。

### 1.1 NVIDIA 驱动的两层结构

```
┌──────────────────────────────────────┐
│     用户空间（Userspace）              │
│  libcuda.so, nvidia-smi, nvcc 等     │
│  预编译二进制，跟内核版本无关          │
└──────────────────────────────────────┘
                  ↕ ioctl
┌──────────────────────────────────────┐
│     内核空间（Kernel Space）           │
│  nvidia.ko, nvidia-modeset.ko 等     │
│  从源码编译，必须匹配当前内核版本      │
└──────────────────────────────────────┘
```

**安装逻辑**：先装官方 `.run` 包（提供用户空间），再编译破解源码覆盖内核模块。

### 1.2 BAR1 P2P 是什么

| 方式 | 路径 | 带宽 |
|:---|:---|:---|
| CPU 中转 | GPU A → CPU 内存 → GPU B | ~6 GB/s |
| BAR1 P2P | GPU A → PCIe → GPU B | ~13-25 GB/s |
| NVLink | GPU A → NVLink → GPU B | ~47-94 GB/s |

BAR1 P2P 要求每块 GPU 的 **BAR1 窗口 ≥ 显存大小**（32GB 即可覆盖所有消费卡）。

### 1.3 NVIDIA 设了哪些锁

| 锁 | 位置 | 破解方式 |
|:---|:---|:---|
| P2P 通信禁用 | 内核模块 | 破解源码绕过，改用 BAR1 P2P |
| BAR1 限制 256MB (2080 Ti) | STRAPS 寄存器 | 驱动加载时软件覆盖为 32GB |
| NVLink 路径被误伤 | `nv_gpu_ops.c` 等 | 回滚 3 个文件的修改 |
| 跨代 GPU P2P 禁止 | `libcuda.so` | 二进制补丁 NOP 掉检查 |

### 1.4 Resizable BAR 与 STRAPS

- **3080 Ti 及以后**：硬件直接支持 Resizable BAR，驱动自动扩展 BAR1 到 32GB
- **2080 Ti (Turing)**：STRAPS 寄存器默认 BAR1=256MB，需要驱动在加载时**软件覆盖** STRAPS 才能让 GPU 报告支持更大 BAR1

STRAPS 修改是**非持久的**——只在运行时生效，重启后 GPU 自动恢复出厂值。零风险。

---

## 二、需要哪些项目

### 2.1 必需文件

| 文件/仓库 | 来源 | 用途 |
|:---|:---|:---|
| `NVIDIA-Linux-x86_64-590.48.01.run` | [NVIDIA 官网](https://www.nvidia.com/drivers) 或 CUDA Toolkit | 官方驱动安装包，提供用户空间组件 |
| `open-gpu-kernel-modules` (分支 `590.48.01-p2p`) | [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) | 破解版内核模块源码 |
| `cuda-samples` | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | P2P 带宽测试工具 |

### 2.2 可选参考

| 仓库 | 用途 |
|:---|:---|
| `NvStrapsReBar` ([terminatorul/NvStrapsReBar](https://github.com/terminatorul/NvStrapsReBar)) | STRAPS 寄存器修改原理参考（UEFI 方案，我们用驱动层方案替代） |

### 2.3 版本对应关系

> **破解仓库的分支名必须与驱动版本完全匹配！**

| 驱动版本 | 对应分支 | CUDA 版本 |
|:---|:---|:---|
| 590.48.01 | `590.48.01-p2p` | 13.1 |

如果未来有新版本分支，按同样的方法找对应分支。

---

## 三、环境准备

### 3.1 系统要求

| 项目 | 要求 |
|:---|:---|
| 操作系统 | Ubuntu 22.04 / 24.04 |
| 内核 | 5.15+ |
| BIOS | **Above 4G Decoding** 和 **Resizable BAR** 启用 |
| IOMMU | 内核参数中设置 `iommu=pt` |

### 3.2 IOMMU 配置

编辑 `/etc/default/grub`：
```bash
# AMD 平台
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=on iommu=pt"

# Intel 平台
GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt"
```

```bash
sudo update-grub && sudo reboot
```

### 3.3 安装编译依赖

```bash
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r) dkms pkg-config libglvnd-dev
```

### 3.4 清除旧驱动

```bash
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
sudo apt purge -y 'nvidia-*' 2>/dev/null
sudo /usr/bin/nvidia-uninstall 2>/dev/null
```

### 3.5 创建工作目录并克隆项目

```bash
mkdir -p /root/projects/cuda_install && cd /root/projects/cuda_install

# 破解内核模块源码（注意分支名）
git clone -b 590.48.01-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git

# P2P 测试工具
git clone https://github.com/NVIDIA/cuda-samples.git

# STRAPS 参考（可选）
git clone https://github.com/terminatorul/NvStrapsReBar.git
```

### 3.6 下载并安装官方驱动（用户空间）

```bash
# 方法一：直接下载驱动 .run（推荐）
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/590.48.01/NVIDIA-Linux-x86_64-590.48.01.run

# 方法二：下载 CUDA Toolkit .run（包含 nvcc 等工具链）
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_590.48.01_linux.run

# 安装用户空间（静默模式，不装 OpenGL）
chmod +x NVIDIA-Linux-x86_64-590.48.01.run
sudo ./NVIDIA-Linux-x86_64-590.48.01.run --silent --no-opengl-files

# 验证
nvidia-smi
```

> **此时 nvidia-smi 应该能看到所有 GPU。官方内核模块已经被安装了，后续会被破解版覆盖。**

---

## 四、涉及修改的文件总览

后续两篇文档会详细说明每个修改。这里先给出全局视图：

| # | 文件 | 层 | 修改类型 | 是否所有场景都需要 |
|:---|:---|:---|:---|:---|
| 1 | `nv-pci.c` | 内核 | STRAPS + Bridge 窗口释放 | ✅ 必需 |
| 2 | `kernel_bif.c` | 内核 | forceP2PType → DEFAULT | 仅有 NVLink 时 |
| 3 | `nv_gpu_ops.c` | 内核 | 回滚 4 处 NVLink 有害修改 | 仅有 NVLink 时 |
| 4 | `gmmu_fmt.c` | 内核 | PEER → fldAddrPeer | 仅有 NVLink 时 |
| 5 | `libcuda.so` | 用户空间 | 二进制补丁跨代检查 | 仅有混代 GPU 时 |

→ 内核层修改见 **第二篇**，用户空间修改见 **第三篇**。
">
</invoke>


---


# 第二篇：源码修改与 Resize BAR

> **前置条件**：已完成第一篇中的环境准备和官方驱动安装。

---

## 一、修改 `nv-pci.c` — STRAPS 覆盖 + Bridge 窗口释放

**文件路径**：`kernel-open/nvidia/nv-pci.c`

这是最关键的修改，包含两个功能块，都加在 `nv_resize_pcie_bars()` 函数中。

### 1.1 STRAPS 寄存器覆盖（让旧卡支持大 BAR1）

在函数开头、`pci_rebar_get_possible_sizes()` 调用**之前**，插入以下代码：

```c
    //
    // STRAPS override: modify STRAPS registers via BAR0 MMIO to make GPU
    // advertise larger BAR1 sizes. Equivalent to NvStrapsReBar's UEFI approach.
    //
    {
        void __iomem *bar0;
        resource_size_t bar0_start = pci_resource_start(pci_dev, 0);
        resource_size_t bar0_len = pci_resource_len(pci_dev, 0);

        if (bar0_start && bar0_len >= 0x102000)
        {
            bar0 = ioremap(bar0_start, 0x102000);
            if (bar0)
            {
                u32 straps0 = ioread32(bar0 + 0x101000);
                u32 straps1 = ioread32(bar0 + 0x10100C);
                u8 bar1_part1 = (straps0 >> 14) & 0x3;
                u8 bar1_part2 = (straps1 >> 20) & 0x7;

                nv_printf(NV_DBG_INFO,
                    "NVRM: %04x:%02x:%02x.%x: STRAPS before: "
                    "STRAPS0=0x%08x (PART1=%u) STRAPS1=0x%08x (PART2=%u)\n",
                    NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                    NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn),
                    straps0, bar1_part1, straps1, bar1_part2);

                if (bar1_part1 < 3 || bar1_part2 < 7)
                {
                    // STRAPS0: BAR1_SIZE_PART1 = 3 (bits 15:14), override bit 31
                    straps0 &= ~(0x3u << 14);
                    straps0 |= (3u << 14);
                    straps0 |= (1u << 31);
                    iowrite32(straps0, bar0 + 0x101000);

                    // STRAPS1: BAR1_SIZE_PART2 = 7 (bits 22:20), override bit 31
                    straps1 &= ~(0x7u << 20);
                    straps1 |= (7u << 20);
                    straps1 |= (1u << 31);
                    iowrite32(straps1, bar0 + 0x10100C);

                    // Read back to verify
                    straps0 = ioread32(bar0 + 0x101000);
                    straps1 = ioread32(bar0 + 0x10100C);
                    nv_printf(NV_DBG_INFO,
                        "NVRM: %04x:%02x:%02x.%x: STRAPS after: "
                        "STRAPS0=0x%08x (PART1=%u) STRAPS1=0x%08x (PART2=%u)\n",
                        NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                        NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn),
                        straps0, (straps0 >> 14) & 0x3, straps1, (straps1 >> 20) & 0x7);
                }

                iounmap(bar0);
                msleep(100);
            }
        }
    }
```

**原理**：STRAPS 寄存器（BAR0+0x101000）中的 bit 15:14 和 bit 22:20 编码 BAR1 最大大小。写入时设置 bit 31=1 启用软件覆盖。GPU 立即更新 ReBAR Capability 报告。

### 1.2 Bridge 窗口释放（让 Bridge 重新分配足够大的窗口）

在释放 BAR1/BAR3 之后、`pci_resize_resource()` 之前，插入以下代码：

```c
    //
    // Release sibling prefetchable resources and bridge window to allow
    // kernel to re-allocate a larger window for the resized BAR.
    //
    {
        struct pci_dev *bridge = pci_dev->bus->self;
        struct pci_dev *sibling;

        if (bridge) {
            list_for_each_entry(sibling, &pci_dev->bus->devices, bus_list) {
                int i;
                if (sibling == pci_dev) continue;
                for (i = 0; i < PCI_BRIDGE_RESOURCES; i++) {
                    struct resource *res = &sibling->resource[i];
                    if (res->parent && (res->flags & IORESOURCE_PREFETCH)) {
                        nv_printf(NV_DBG_INFO,
                            "NVRM: Releasing sibling %s resource %d\n",
                            dev_name(&sibling->dev), i);
                        pci_release_resource(sibling, i);
                    }
                }
            }

            if (bridge->resource[PCI_BRIDGE_RESOURCES + 2].parent) {
                nv_printf(NV_DBG_INFO,
                    "NVRM: Releasing bridge %s prefetchable window\n",
                    dev_name(&bridge->dev));
                release_resource(&bridge->resource[PCI_BRIDGE_RESOURCES + 2]);
                memset(&bridge->resource[PCI_BRIDGE_RESOURCES + 2], 0,
                       sizeof(bridge->resource[PCI_BRIDGE_RESOURCES + 2]));
                bridge->resource[PCI_BRIDGE_RESOURCES + 2].flags =
                    IORESOURCE_MEM | IORESOURCE_PREFETCH | IORESOURCE_MEM_64;
            }
        }
    }
```

**原理**：2080 Ti 与 USB 控制器共用 PCIe Bridge，BIOS 分配的 Bridge 窗口仅 289MB，不够 32GB。先释放 Bridge 下所有资源，内核 `pci_assign_unassigned_bus_resources()` 会重新分配 ~48GB 的窗口。

---

## 二、修改 `kernel_bif.c` — NVLink 兼容（仅有 NVLink 时需要）

**文件路径**：`src/nvidia/src/kernel/gpu/bif/kernel_bif.c`

搜索 `forceP2PType`，找到：
```c
pKernelBif->forceP2PType = NV_REG_STR_RM_FORCE_P2P_TYPE_PCIEP2P;
```

改为：
```c
pKernelBif->forceP2PType = NV_REG_STR_RM_FORCE_P2P_TYPE_DEFAULT;
```

**为什么**：原始补丁强制所有 GPU 走 PCIe P2P，NVLink GPU 对也被迫走慢速通道。改为 DEFAULT 后自动选择最优路径。

---

## 三、修改 `nv_gpu_ops.c` — 回滚 NVLink 有害修改（仅有 NVLink 时需要）

**文件路径**：`src/nvidia/src/kernel/rmapi/nv_gpu_ops.c`

搜索所有 `GMMU_APERTURE_PEER` 出现的位置，有 4 处被破解补丁修改过，需要全部回滚：

### 模式 A：aperture 强制替换（2 处）

```diff
- if (aperture == GMMU_APERTURE_PEER) {
-     gmmuFieldSetAperture(&pPteFmt->fldAperture, GMMU_APERTURE_SYS_COH, pte.v8);
- } else {
-     gmmuFieldSetAperture(&pPteFmt->fldAperture, aperture, pte.v8);
- }
+ gmmuFieldSetAperture(&pPteFmt->fldAperture, aperture, pte.v8);
```

### 模式 B：fabricBaseAddress 和 peerIndex 注释（2 处）

去掉 `if (aperture == GMMU_APERTURE_PEER) { fabricBaseAddress = bar1BusAddr; }` 这行，
恢复被注释掉的 `nvFieldSet32(&pPteFmt->fldPeerIndex, ...)` 代码块。

**为什么**：BAR1 P2P 使用 `SYS_NONCOH` aperture，这些 `if (PEER)` 检查只影响 NVLink 路径。回滚后 NVLink 恢复满速，BAR1 P2P 不受影响。

---

## 四、修改 `gmmu_fmt.c` — PTE 地址字段修复（仅有 NVLink 时需要）

**文件路径**：`src/nvidia/src/libraries/mmu/gmmu_fmt.c`

找到 `gmmuFmtPtePhysAddrFld` 函数中的 switch-case：

```diff
  case GMMU_APERTURE_SYS_COH:
  case GMMU_APERTURE_SYS_NONCOH:
-  case GMMU_APERTURE_PEER:
      return &pPte->fldAddrSysmem;
+  case GMMU_APERTURE_PEER:
+      return &pPte->fldAddrPeer;
```

**为什么**：NVLink 使用 PEER aperture，需要 `fldAddrPeer` 来编码地址。被错误合并到 Sysmem 分支会导致 NVLink 地址编码错误。

---

## 五、编译安装

```bash
cd /root/projects/cuda_install/open-gpu-kernel-modules

# 卸载当前内核模块
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null

# 编译
make modules -j$(nproc)

# 安装
sudo make modules_install -j$(nproc)
sudo depmod -a

# 加载
sudo modprobe nvidia
sudo modprobe nvidia-modeset
sudo modprobe nvidia-drm
sudo modprobe nvidia-uvm
```

---

## 六、验证 Resize BAR

```bash
# 检查 BAR1 大小（所有 GPU 应为 32768 MiB）
nvidia-smi -q -d MEMORY

# 检查 dmesg 日志
dmesg | grep -E "STRAPS|BAR|bridge window|resize"

# 检查 ReBAR Capability（lspci 应显示 supported: ... 32GB）
lspci -s <BUS_ID> -vvv | grep -A3 "Resizable BAR"
```

**预期结果**：

| GPU | BAR1 |
|:---|:---|
| 2080 Ti | 32768 MiB ✅ |
| 3080 Ti | 32768 MiB ✅ |

→ 确认 BAR1 全部 32GB 后，进入第三篇。


---


# 第三篇：闭源驱动破译与功耗持久化

> **前置条件**：已完成第二篇的源码修改、编译安装，BAR1 全部 32GB。

---

## 一、libcuda.so 跨代 P2P 补丁

> 仅当系统中有**不同代 GPU 混合**（如 2080 Ti + 3080 Ti）时需要。

### 1.1 原理

NVIDIA 在 `libcuda.so` 中检查两块 GPU 的计算能力是否匹配（如 7.5 vs 8.6）。不匹配时返回 `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`（Error 217）。

补丁位置：偏移 `0x19c610`，原始指令为 `je 0x19c76e`（如果相等则跳转到错误处理），替换为 6 个 `NOP` 跳过检查。

> ⚠️ **偏移量 `0x19c610` 仅适用于 590.48.01 版本。换版本需要用 GDB 重新定位。**

### 1.2 操作步骤

```bash
# 1. 找到 libcuda.so
LIBCUDA=$(find /usr -name "libcuda.so.590.48.01" 2>/dev/null)
echo "libcuda 路径: $LIBCUDA"

# 2. 备份
sudo cp "$LIBCUDA" "${LIBCUDA}.bak"

# 3. 打补丁
sudo python3 -c "
f = open('$LIBCUDA', 'r+b')
f.seek(0x19c610)
data = f.read(6)
if data == b'\x0f\x84\x58\x01\x00\x00':
    f.seek(0x19c610)
    f.write(b'\x90\x90\x90\x90\x90\x90')
    print('✅ 补丁成功')
elif data == b'\x90\x90\x90\x90\x90\x90':
    print('ℹ️ 已打过补丁')
else:
    print('⚠️ 字节不匹配: ' + data.hex())
f.close()
"
```

### 1.3 验证

```bash
# 检查补丁状态
python3 -c "
f = open('/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01', 'rb')
f.seek(0x19c610)
data = f.read(6)
f.close()
print('已补丁 ✅' if data == b'\x90\x90\x90\x90\x90\x90' else '未补丁: ' + data.hex())
"
```

### 1.4 如何为新版本定位补丁偏移

如果驱动版本变了，用 GDB 定位：

```bash
# 查找 cuCtxEnablePeerAccess 中的架构比较跳转
gdb -batch -ex "disas cuCtxEnablePeerAccess" /usr/lib/x86_64-linux-gnu/libcuda.so.XXX.XX.XX

# 找到类似 "je <addr>" 的条件跳转（紧跟在 cmp/test 之后），
# 该跳转目标地址是返回 error 217 的代码块
```

---

## 二、Persistence Mode（持久模式）

### 2.1 为什么需要

GPU 驱动卸载/重载时，固件会启动保护机制导致风扇全速。开启持久模式让 GPU 始终保持初始化状态。

### 2.2 手动开启

```bash
sudo nvidia-smi -pm 1
```

### 2.3 开机自动启用（systemd 服务）

```bash
sudo tee /etc/systemd/system/nvidia-persistence.service > /dev/null << 'EOF'
[Unit]
Description=NVIDIA Persistence Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-persistence.service
```

---

## 三、功耗限制持久化

### 3.1 为什么要限制功耗

5 块 GPU 满载总功耗可达 1750W+（3×350W + 2×250W），容易导致系统宕机。通过限制功耗降低总负载：

| GPU | 默认功耗 | 限制后功耗 | 满载节省 |
|:---|:---|:---|:---|
| 2080 Ti ×2 | 250W ×2 = 500W | **180W** ×2 = 360W | 140W |
| 3080 Ti ×3 | 350W ×3 = 1050W | **250W** ×3 = 750W | 300W |
| **合计** | **1550W** | **1110W** | **440W** |

### 3.2 手动设置

```bash
# 2080 Ti = 180W（GPU 0 和 2，按 nvidia-smi 索引）
sudo nvidia-smi -i 0 -pl 180
sudo nvidia-smi -i 2 -pl 180

# 3080 Ti = 250W（GPU 1, 3, 4）
sudo nvidia-smi -i 1 -pl 250
sudo nvidia-smi -i 3 -pl 250
sudo nvidia-smi -i 4 -pl 250
```

> ⚠️ **`nvidia-smi -pl` 设置在重启后会丢失！** 必须持久化。

### 3.3 通过 PCIe Bus ID 设置（推荐，不受 GPU 索引变化影响）

```bash
# 2080 Ti = 180W
sudo nvidia-smi -i 00000000:01:00.0 -pl 180
sudo nvidia-smi -i 00000000:42:00.0 -pl 180

# 3080 Ti = 250W
sudo nvidia-smi -i 00000000:41:00.0 -pl 250
sudo nvidia-smi -i 00000000:82:00.0 -pl 250
sudo nvidia-smi -i 00000000:C1:00.0 -pl 250
```

### 3.4 开机自动设置（systemd 服务）

```bash
sudo tee /etc/systemd/system/nvidia-powerlimit.service > /dev/null << 'EOF'
[Unit]
Description=NVIDIA GPU Power Limit Configuration
After=nvidia-persistence.service
Requires=nvidia-persistence.service

[Service]
Type=oneshot
RemainAfterExit=yes

# 2080 Ti = 180W
ExecStart=/usr/bin/nvidia-smi -i 00000000:01:00.0 -pl 180
ExecStart=/usr/bin/nvidia-smi -i 00000000:42:00.0 -pl 180

# 3080 Ti = 250W
ExecStart=/usr/bin/nvidia-smi -i 00000000:41:00.0 -pl 250
ExecStart=/usr/bin/nvidia-smi -i 00000000:82:00.0 -pl 250
ExecStart=/usr/bin/nvidia-smi -i 00000000:C1:00.0 -pl 250

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nvidia-powerlimit.service
```

> **注意**：如果更换显卡或插槽位置，Bus ID 会变，需要更新此服务文件。

---

## 四、CUDA 环境变量

如果安装了 CUDA Toolkit（nvcc 等），添加到 `~/.bashrc`：

```bash
cat >> ~/.bashrc << 'EOF'

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc
```

---

## 五、最终验证

### 5.1 P2P 带宽测试

```bash
cd /root/projects/cuda_install/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest

# 如果还没编译
export PATH=/usr/local/cuda/bin:$PATH
make

./p2pBandwidthLatencyTest
```

**预期结果**（P2P=Enabled）：

| GPU 对 | 连接 | 单向 | 双向 |
|:---|:---|:---|:---|
| 3080Ti ↔ 3080Ti | BAR1 P2P | ~26 GB/s | ~50 GB/s |
| 3080Ti ↔ 2080Ti | BAR1 P2P | ~13 GB/s | ~25 GB/s |
| 2080Ti ↔ 2080Ti | NVLink | ~47 GB/s | ~94 GB/s |

### 5.2 功耗验证

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,power.draw,power.limit --format=csv,noheader
```

预期：2080 Ti 限制 180W，3080 Ti 限制 250W。

### 5.3 重启后验证

重启系统后，检查以下服务是否自动生效：

```bash
# Persistence Mode
nvidia-smi -q -d PERFORMANCE | grep "Persistence"

# 功耗限制
nvidia-smi --query-gpu=power.limit --format=csv,noheader

# BAR1 大小
nvidia-smi -q -d MEMORY | grep -A2 "BAR1"
```

---

## 六、故障速查

| 症状 | 原因 | 解决 |
|:---|:---|:---|
| P2P 测试 Error 217 | libcuda 补丁未生效 | 重新运行补丁脚本 |
| NVLink 带宽仅 4 GB/s | NVLink 兼容修复不完整 | 检查第二篇中 3 个文件的修改 |
| BAR1 仍为 256MB | STRAPS/Bridge 修改未生效 | 检查 dmesg 日志 |
| 满载宕机 | 功耗限制未设置 | 配置功耗限制 systemd 服务 |
| 重启后功耗恢复默认 | systemd 服务未启用 | `systemctl enable nvidia-powerlimit` |
| GPU 风扇满速 | Persistence Mode 未开 | `nvidia-smi -pm 1` |

---

## 七、换机器/升级速查

| 场景 | 操作 |
|:---|:---|
| 新机器 | 三篇文档从头走一遍 |
| 换显卡 | 更新功耗限制服务中的 Bus ID |
| 升级内核 | 重新编译安装内核模块（第二篇步骤五） |
| 升级驱动 | 全部重来（分支名、libcuda 偏移量都会变） |
