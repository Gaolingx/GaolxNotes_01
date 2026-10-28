# 多卡直通 PVE AI 训练 与 游戏 互不干扰

## 🧑‍💻 User

pve如何直通8张pro6000在不同虚拟机里面训练ai/打游戏（比如6卡分给ubuntu，2卡分给windows）互不干扰？

---
## 😀 默认助手 (7)

# 8 张 RTX PRO 6000 分给两个 VM：可行，但难点不在 PVE 配置

先说结论：**你要做的事在软件层面很简单**，因为 8 张卡是 8 个独立的 PCIe 设备，整卡直通（VFIO）不需要 vGPU、不需要 MIG、不需要特殊授权，每张卡绑定到哪个 VM 就是硬隔离。真正决定成败的是**硬件拓扑**和**三个坑**：IOMMU group / Resizable BAR / VM 内多卡 P2P。

---

## 一、方案总览

| 项目 | 推荐做法 | 说明 |
|---|---|---|
| 6 卡 AI 训练 | **LXC 容器**（首选）或 KVM 直通 | 见第四节，这是本文最重要的一条建议 |
| 2 卡 Windows | KVM + `x-vga=1` + Looking Glass/Sunshine | 一张显示、一张算力，或拆成两个 Windows VM |
| 隔离级别 | IOMMU 硬隔离 + CPU/Core pinning + 独立 NVMe | 见第六节 |
| 虚拟化层 | PVE 8.2+/9.x，`q35` + OVMF | 必须 q35，i440fx 撑不住多张大 BAR 卡 |

---

## 二、硬件选型（先看这一节，80% 的失败在这里）

### 2.1 供电：这是最容易被低估的

| 变体 | 单卡 TDP | 8 卡满载 | 实际建议 |
|---|---|---|---|
| RTX PRO 6000 Blackwell WS Edition | 600 W | 4.8 kW | 需 2× 240V 回路 |
| **RTX PRO 6000 Max-Q Edition** | **300 W** | **2.4 kW** | **8 卡同机的正确选择** |
| Server Edition（被动散热） | 600 W | 4.8 kW | 需服务器风道 |

整机（EPYC + 8 卡 + 内存 + 盘）峰值 **5.5–6.5 kW**，普通家用 16A 插座（3.5 kW）直接跳闸。要么买 Max-Q 版本，要么在 guest 里用 `nvidia-smi -pl 350` 限制功耗墙（`-pl` 在整卡直通下通常可用）。

### 2.2 主板与 PCIe 拓扑

推荐机型（8× PCIe 5.0 x16 双槽宽）：

- Supermicro `AS-4125GS-TNRT` / `SYS-421GE-TNRT`（4U，AMD EPYC）
- Gigabyte `G293-Z43`
- ASUS `ESC8000A-E13P`

**买之前必须向厂商确认三件事：**

1. **每个 x16 槽是否各有独立的 PCIe Root Port？** 如果 8 个槽挂在一颗 Broadcom PEX88000 switch 下面又没有 ACS，8 张卡会落进**同一个 IOMMU group**，你就没法拆分给两个 VM（只能用 `pcie_acs_override` 强行拆，安全性下降且可能不稳定）。
2. **BIOS 能否关闭 Resizable BAR？** 见 2.3。
3. **8 槽是否都是电性 x16 Gen5？** 有些板子是 x16 物理 / x8 电性。

### 2.3 Resizable BAR：多卡直通必须关掉

启用了 ReBAR 的 RTX PRO 6000，单卡 BAR1 会申请 **32 GB以上**。8 张 = 256 GB 的 MMIO aperture：

- 宿主机 BIOS 大概率分配不出来 → 开机黑屏 / 卡在 POST
- 即使宿主过了，guest 里 6 张卡 × 32 GB = 192 GB，OVMF 的 64-bit MMIO window 也放不下 → 启动报 `not enough MMIO space`

**配置：BIOS 里 `Above 4G Decoding = Enabled`，`Resizable BAR = Disabled`**。关掉后单卡 BAR1 = 256 MB，6 卡只占 1.5 GB，完全无压力。对纯 CUDA 计算吞吐几乎无影响。

### 2.4 CPU / 内存 / NUMA 布局

```
双路 EPYC：
  Socket 0 → GPU 0-5 (x16)  → 6 卡训练 VM 绑这个 socket
  Socket 1 → GPU 6-7 (x16)  → Windows VM 绑这个 socket
  剩余 lane 给 NVMe / 网卡
```
这样两个 VM 的 PCIe 流量、内存带宽、中断都落在不同 socket 上，**这是"互不干扰"的物理基础**。单路 EPYC 有 128 lane，8×16 刚好用满，NVMe 就没 lane 了，需要 x8 拆分或 PCIe switch。

内存：6 卡 96 GB × 6 训练，建议 512 GB–1 TB，按 socket 对半分。

---

## 三、宿主机配置

<details>
<summary><b>3.1 内核参数 / vfio 绑定（点开）</b></summary>

先确认 IOMMU 已开：

```bash
dmesg | grep -e DMAR -e IOMMU
```

没有输出再加参数。`/etc/default/grub`：

```bash
# AMD 平台
GRUB_CMDLINE_LINUX_DEFAULT="quiet amd_iommu=on iommu=pt initcall_blacklist=sysfb_init video=efifb:off"
# Intel 平台
GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt initcall_blacklist=sysfb_init video=efifb:off"
```

- `iommu=pt`：直通性能必需，避免全量 DMA 翻译开销
- `initcall_blacklist=sysfb_init` + `video=efifb:off`：防止宿主 framebuffer 占用你要直通的卡
- **只有在 IOMMU group 无法拆分时**才加 `pcie_acs_override=downstream,multifunction`，这是最后手段

```bash
update-grub && proxmox-boot-tool refresh
```

**黑名单宿主的 NVIDIA 驱动**（关键，否则宿主会抢占卡）：

```bash
cat >/etc/modprobe.d/blacklist-gpu.conf <<'EOF'
blacklist nouveau
blacklist nvidia
blacklist nvidia_drm
blacklist nvidiafb
EOF
```

**用 vfio-pci 抢占全部 8 张卡**。因为 8 张卡型号完全相同，直接用 vendor:device ID 一网打尽最省事：

```bash
lspci -nn | grep -i nvidia     # 拿到形如 [10de:2b8c] 的 ID
cat >/etc/modprobe.d/vfio.conf <<'EOF'
options vfio-pci ids=10de:XXXX,10de:YYYY
softdep nvidia pre: vfio-pci
softdep nouveau pre: vfio-pci
EOF
update-initramfs -u -k all && reboot
```

验证：

```bash
lspci -nnk -d 10de: | grep -A3 -i "pro 6000"
# 每张卡 Kernel driver in use 应为 vfio-pci
```
</details>

<details>
<summary><b>3.2 检查 IOMMU group（决定生死的一步，点开）</b></summary>

```bash
#!/bin/bash
shopt -s nullglob
for g in /sys/kernel/iommu_groups/*; do
  echo "=== IOMMU Group ${g##*/} ==="
  for d in "$g"/devices/*; do
    printf '  %s\n' "$(lspci -nns "${d##*/}")"
  done
done
```

**期望结果**：每个 group 里只有一张 GPU 的 `.0`（显卡）+ `.1`（HDMI/DP 音频），或者一张卡独占一组。

**坏结果**：一个 group 里出现 2 张以上显卡 → 这俩没法拆给不同 VM。先试试：

```bash
lspci -vvv -s 41:00.0 | grep -i acs     # 看 ACSCtl 是否 Supported+
```

不支持 → 要么换主板/加 PCIe switch 分区，要么用 ACS override（牺牲隔离强度，训练机打游戏机互通 DMA 理论上可行，自用可接受）。

```bash
# 检查 ACS override 生效后 group 是否拆开
cat /sys/kernel/iommu_groups/*/devices/*/... # 同上脚本再跑一遍
```
</details>

---

## 四、⚠️ 强烈建议：6 卡 AI 训练不要放 KVM，用 LXC

这是本文最核心的一条经验。原因在于**KVM 里多卡之间的 P2P DMA 基本不可用**：

> 直通设备的 BAR 地址在 guest 里是个 GPA，但这个 GPA 在 VFIO 的 IOMMU 容器中没有对应的 host 内存映射。GPU A 想直接 DMA 写 GPU B 的显存，物理 IOMMU 会直接 fault。所以 `nvidia-smi topo -m` 在 guest 里通常显示 `SYS`/`PHB`，NCCL 检测到 P2P 不可用后**回退到经由 host 内存中转（SHM 路径）**。

后果：6 卡 all-reduce 带宽从 PCIe 直连的 ~50 GB/s 掉到 ~15–20 GB/s 量级。对大模型训练，通信占比高时可能慢 20–40%。

**对照方案：**

| 方案 | 性能 | 隔离强度 | 配置难度 | 适用 |
|---|---|---|---|---|
| **LXC + 设备直通（6 卡）** | 原生，P2P 可用 | 中（共享内核，cgroup 限制设备） | 低 | ✅ AI 训练首选 |
| KVM 整卡直通（6 卡） | PCIe 通信打折 | 高（IOMMU 硬隔离） | 中 | 必须强隔离时 |
| 裸机跑训练 + VM 只给游戏 | 原生 | — | 低 | 双机方案 |

LXC 配法（PVE 上 6 卡 Ubuntu）：

```bash
# /etc/pve/lxc/200.conf  追加
lxc.cgroup2.devices.allow: c 195:* rwm      # nvidia
lxc.cgroup2.devices.allow: c 234:* rwm      # nvidia-uvm
lxc.cgroup2.devices.allow: c 237:* rwm      # nvidia-drm(可选)
lxc.cgroup2.devices.allow: c 511:* rwm      # nvidia-caps
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
# 只绑定属于 6 张训练卡的那 6 个节点
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidia1 dev/nvidia1 none bind,optional,create=file
# ... nvidia2..nvidia5
```

配合容器内 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`。**注意 `/dev/nvidiaN` 的序号不保证与 PCI 顺序稳定**，建议写一个开机脚本按 PCI BDF 用 `nvidia-smi --query-gpu=index,pci.bus_id` 生成绑定，否则重启后可能绑错卡。

> LXC 里宿主必须装 NVIDIA 驱动和 CUDA（容器与宿主共享内核模块），Windows VM 的那 2 张卡由 vfio-pci 抢占，两者互不相干——**这种混搭本身就是最好的隔离**：LXC 用的是宿主 nvidia 驱动路径，VM 用的是 vfio 路径，物理上不可能串。

---

## 五、如果坚持用 KVM：VM 配置

### 5.1 6 卡 Ubuntu VM

```bash
qm create 100 --name ai-ubuntu --ostype l26 --machine q35 --bios ovmf \
  --cpu host --sockets 1 --cores 48 --memory 524288 --balloon 0 \
  --numa 1 --hugepages 1 --scsihw virtio-scsi-single

qm set 100 --efidisk0 local-lvm:1,efitype=4m,pre-enrolled-keys=0
qm set 100 --net0 virtio,bridge=vmbr0,tag=10

# 6 张卡，每张带上音频 function，必须 pcie=1
qm set 100 --hostpci0  "0000:41:00.0;0000:41:00.1,pcie=1"
qm set 100 --hostpci1  "0000:42:00.0;0000:42:00.1,pcie=1"
qm set 100 --hostpci2  "0000:61:00.0;0000:61:00.1,pcie=1"
qm set 100 --hostpci3  "0000:62:00.0;0000:62:00.1,pcie=1"
qm set 100 --hostpci4  "0000:81:00.0;0000:81:00.1,pcie=1"
qm set 100 --hostpci5  "0000:82:00.0;0000:82:00.1,pcie=1"

# CPU 绑核（PVE 7.2+ 原生支持），避免和 Windows VM 抢核
qm set 100 --affinity 0-47
```

要点：

- `pcie=1` 每个设备独立 root port，**必须加**，否则多卡会共享一个 legacy PCI 总线
- `hugepages 1` + `balloon 0`：训练机不要气球，避免宿主动态回收内存导致卡顿
- `numa 1`：让 guest 看到 NUMA 拓扑，NCCL 和 PyTorch 会自动按拓扑分配
- 同一个 `;` 分隔的多个 function 在 shell 里**必须加引号**
- 训练机的磁盘建议**整块 NVMe 直通**（`hostpci` 一个 NVMe 控制器），不要和 Windows VM 共用同一块物理盘

进入 guest 后立刻验证：

```bash
nvidia-smi topo -m
# 记录 P2P 矩阵，预期多为 SYS/PHB（见第四节）

# 实测通信带宽，别信理论值
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make MPI=1 ...
./build/all_reduce_perf -b 8G -e 8G -f 2 -g 6
```

### 5.2 Windows 2 卡

先明确你的"2 卡给 Windows"是哪一种：

| 意图 | 做法 |
|---|---|
| 一张显示 + 一张算力（游戏 + 本地跑模型） | 单个 Win11 VM 直通 2 张，第 1 张 `x-vga=1` |
| 两个人各打各的游戏 | 拆成两个 Win11 VM，各 1 张 |
| 想 SLI 提升游戏帧数 | ❌ 不可行，SLI 已被 NVIDIA 放弃，多卡对游戏无加速 |

```bash
qm create 101 --name win-gaming --ostype win11 --machine q35 --bios ovmf \
  --cpu host,hidden=1 --sockets 1 --cores 16 --memory 65536 --balloon 0 \
  --numa 1

# 主显示卡：x-vga=1（一台 VM 只能有一个）
qm set 101 --hostpci0 "0000:c1:00.0;0000:c1:00.1,pcie=1,x-vga=1"
qm set 101 --hostpci1 "0000:c2:00.0;0000:c2:00.1,pcie=1"

qm set 101 --affinity 64-79
qm set 101 --vga none            # 关掉虚拟显卡
qm set 101 --machine q35,viommu=virtio
```

关键项解释：

- **`hidden=1`**：对 RTX PRO（专业卡）驱动一般不会报 Code 43，但加上无副作用，建议加
- **`affinity 64-79`**：绑到另一个 socket，和训练 VM 的核完全不相交
- **`--vga none`**：用物理卡做唯一输出，避免虚拟 VGA 抢主显示
- **音频**：HDMI/DP 音频走 GPU 的 `.1` function；耳机/麦克风建议**直通一个 USB 声卡**，比板载 HDA 直通稳定得多
- **输入**：直通 USB 控制器或 USB 设备（键鼠）；远程用 Sunshine + Moonlight 比 Parsec 延迟低

低延迟画面回传（不接物理显示器时）：

```bash
# Looking Glass: 加 ivshmem
qm set 101 --args '-device ivshmem-plain,memdev=ivshmem,bus=pcie.0 -object memory-backend-file,id=ivshmem,share=on,mem-path=/dev/shm/looking-glass,size=256M'
```
宿主需 `touch /dev/shm/looking-glass && chown kvm:kvm` 并调大 shm 权限。

---

## 六、"互不干扰"的四道闸

| 资源 | 手段 | 命令/配置 |
|---|---|---|
| **设备** | IOMMU group 硬隔离 | 第二节验证 |
| **CPU** | 绑核，核不重叠 | `qm set --affinity 0-47` / `64-79`，LXC 用 `lxc.cgroup2.cpuset.cpus` |
| **内存带宽** | NUMA 对齐，GPU 与 CPU 同 socket | 2.4 节的物理布局 |
| **存储 IO** | 各自独享 NVMe 控制器直通 | 别共用同一块 SSD 的同一个 pool |
| **网络** | 不同 VLAN / SR-IOV VF | 游戏机用 VF 降延迟 |
| **功耗/温度** | `nvidia-smi -pl` 限墙 | 见下 |

功耗墙的例子（在 guest 内）：

```bash
nvidia-smi -pm 1
nvidia-smi -pl 400          # 6 卡 400W = 2.4kW，给 2 卡游戏机留足余量
```

**注意**：训练机满载时整机功耗会顶到 PSU 上限，如果 PSU 不够，两边同时跑会瞬间掉电。这是"互不干扰"最容易被忽略的一环——**先算功耗预算，再谈软件隔离**。

---

## 七、已知问题清单

<details>
<summary><b>NVIDIA reset bug：VM 重启后显卡起不来</b></summary>

现象：`qm stop` 后再 `qm start`，报 `vfio: Unable to reset device` 或 guest 里 `nvidia-smi` 报 `GPU is lost`。

原因：NVIDIA 消费级/工作站卡不支持 PCIe FLR，且 HDMI 音频 function 常拖住 PM reset。

处理顺序：
1. 优先用 `qm shutdown` 让 guest 内的驱动正常卸载，别直接 `stop`
2. 宿主侧尝试 `echo 1 > /sys/bus/pci/devices/0000:41:00.0/reset`
3. 仍失败则只能重启宿主（或换用 Colovu/`vendor-reset` 类补丁，主要针对 AMD）
4. 实践中 Blackwell 比 Turing 时代好很多，但如果 6 张卡里有 1 张反复出问题，把它挪到独立 root port
</details>

<details>
<summary><b>6 卡以上 root port 不够 / MMIO 分配失败</b></summary>

报错：`pci_bridge_check_ranges: Failed to allocate memory` 或 OVMF 卡在启动。

- 确认 BIOS `Resizable BAR = Disabled`（2.3 节）
- 确认 `Above 4G Decoding = Enabled`
- 确认 `pcie=1`
- 仍不够时手动加 root port：
```bash
qm set 100 --args '-device pcie-root-port,id=rp6,bus=pcie.0,chassis=6,slot=6 -device pcie-root-port,id=rp7,bus=pcie.0,chassis=7,slot=7'
```
</details>

<details>
<summary><b>多张同型号卡需要独立 ROM 文件</b></summary>

UEFI + 现代卡通常**不需要** `romfile`。如果 guest 启动黑屏、`dmesg` 里见 `BAR 0: can't reserve`，才考虑导出 VBIOS：

```bash
echo 1 > /sys/bus/pci/devices/0000:41:00.0/rom
cat /sys/bus/pci/devices/0000:41:00.0/rom > /usr/share/kvm/gpu41.rom
echo 0 > /sys/bus/pci/devices/0000:41:00.0/rom
qm set 100 --hostpci0 "0000:41:00.0,pcie=1,romfile=gpu41.rom"
```
每张卡一个独立文件，不要共用。
</details>

<details>
<summary><b>关于 vGPU / MIG / SR-IOV 的传言</b></summary>

用户常问"能不能把一张卡切成两半"。回答：

- **GeForce / RTX 消费级**：官方不支持 vGPU/SR-IOV，MIG 只在 A100/H100 以上数据中心卡提供
- **RTX PRO 6000 Blackwell Server Edition**：NVIDIA 的 vGPU 认证列表里确实覆盖了部分 RTX PRO 型号（以 `nvidia-vgpu` 官方兼容性列表为准），但需要**付费 vGPU license**，且配置文件、驱动版本、宿主版本匹配极其挑剔
- 你的场景（6+2 整卡分给不同 VM）**完全不需要** vGPU——整卡直通更简单、更快、免费。不要为了"切卡"去折腾 vGPU
</details>

---

## 八、落地检查清单

```
硬件
[ ] BIOS: Above 4G Decoding = On / Resizable BAR = Off / VT-d 或 AMD-Vi = On
[ ] 8 个 x16 槽各自独立 root port，ACS 支持（或接受 ACS override）
[ ] PSU ≥ 6 kW，240V 回路，风道能吹透 8 张卡
[ ] 每 socket 的 GPU 分组与 VM 分配对齐

宿主
[ ] dmesg | grep -e DMAR -e IOMMU 有输出
[ ] iommu=pt 已生效
[ ] 8 张卡 Kernel driver in use = vfio-pci
[ ] IOMMU group 脚本输出符合预期（一卡一组）
[ ] initcall_blacklist=sysfb_init 已加

VM
[ ] q35 + OVMF + pcie=1 + balloon=0 + hugepages=1 + numa=1
[ ] affinity 绑核不重叠
[ ] guest 内 nvidia-smi topo -m / nccl-tests all_reduce_perf 实测
[ ] Windows: vga=none + x-vga=1 + USB 声卡 + Sunshine/LG
[ ] 关机重启 VM 三次，验证 reset 正常
[ ] 双 VM 同时满载 30 分钟，看温度与整机功率
```

---

## 九、我的最终建议

**不要把 6 卡训练塞进 KVM。** 用下面这个组合，能绕开 90% 的坑：

```
PVE 宿主（裸装 NVIDIA 驱动 + CUDA）
├── LXC 200: Ubuntu 24.04 + 6×RTX PRO 6000  → 训练，原生性能，NCCL P2P 可用
├── KVM 101: Windows 11 + 1×RTX PRO 6000  → 游戏（x-vga + Sunshine）
└── KVM 102: Windows 11 + 1×RTX PRO 6000  → 第二台游戏机 / 或合并成一个双卡 VM
```

这样"互不干扰"由两层机制保证：物理层（LXC 走宿主驱动、VM 走 vfio-pci，路径完全不同，不可能串卡）+ 资源层（cpuset 绑核、NUMA 对齐、独立 NVMe、功耗墙）。只有当你有合规或强安全隔离的硬要求（比如对外提供服务、多租户）时，才值得为 6 卡 KVM 付出 NCCL 降级的代价。

如果你告诉我具体的主板型号、CPU 型号、以及 2 张卡给 Windows 是想一台机器还是两台，我可以把 `qm` 参数和绑核方案写成可直接执行的脚本。
