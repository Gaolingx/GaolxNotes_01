# v86QEMU方案权衡

### 🧑‍💻 User

如何使用v86／wasm。已有x64 win10的iso镜像和vhdx系统盘镜像。考虑qemu方案。物理设备为arm，裸机／android可选。目标系统为x64 win10，iso重新安装和vhdx现有系统可选，vhdx优先

---

### 🤖 Assistant

我已经核实了关键事实（v86 官方 README 和 QEMU 官方文档），下面直接给你结论和可落地的方案。

---

## 一、先说结论（重要）

| 方案 | 能否跑 x64 Win10 | 现实可行性 |
|---|---|---|
| **v86 / WASM** | ❌ **完全不行** | v86 是**纯 32 位 x86** 模拟器，官方明确不支持 64-bit 扩展、不支持多核、兼容列表最高到 32 位 Win8，无 Win10 |
| **QEMU (TCG)** | ✅ 技术上可行 | ARM 上没有 x86 硬件加速，只能纯软件模拟，**极慢**（启动数十分钟～小时级），只适合验证/应急 |

**核心事实**：你的物理机是 ARM，跑 x64 Windows 没有任何硬件加速手段（KVM 在 ARM 上只能加速 ARM 客户机，无法加速 x86），只能走 TCG 动态翻译。所以这条路能通，但别指望日常使用。

你已有的 **x64 Win10 ISO + VHDX**：v86 用不了；QEMU 可以用（现代 QEMU 已支持 VHDX 读写），**VHDX 优先是对的**，比 ISO 重装快得多。

---

## 二、为什么 v86 这条路是死的（已核实）

v86 官方 README 明确写着：

- 指令集约 "Pentium 4 级别，含完整 SSE3"，缺失项里**明确列出 "64-bit extensions"**；
- "Linux works pretty well. **64-bit kernels are not supported**"；
- 兼容列表最高到 Windows XP/Vista/8（32 位、特定条件），**完全没有 Windows 10**，且它是 SeaBIOS（BIOS）环境、单核。

所以你的 x64 ISO 和 VHDX 在 v86 上连引导都过不去。v86 适合跑 Win95/98/2000、旧 Linux 等 32 位系统，用法极简（浏览器里 `new V86({...})`），但对你的目标没用，这里不展开。

---

## 三、QEMU 落地步骤（VHDX 优先）

### 第 1 步：检查并转换 VHDX

QEMU 能读 VHDX，但 qcow2/raw 更稳、支持快照，建议先转换：

```bash
# 1. 先看镜像信息（注意是不是"差异盘/子盘"，以及是否 UEFI/GPT）
qemu-img info win10.vhdx

# 2. 转成 qcow2（推荐）或 raw
qemu-img convert -f vhdx -O qcow2 -p win10.vhdx win10.qcow2
```

<details>
<summary>⚠️ VHDX 两个坑（点开看）</summary>

1. **差异盘（Differencing）**：如果 `qemu-img info` 显示有 parent（父盘），QEMU 处理不好。先在 Hyper-V 里执行 `Merge-VHD` 合并成完整盘再转。
2. **未回放日志**：动态 VHDX 若异常关机有脏日志，先在 Hyper-V 正常关机/合并后再转，否则可能读不出数据。
3. 转完再 `qemu-img info win10.qcow2` 确认 `virtual size` 和分区没问题。

</details>

### 第 2 步：判断 BIOS 还是 UEFI

这决定你用哪套固件：

- **UEFI + GPT**（现代 x64 Win10 绝大多数，Hyper-V Gen2 虚拟机都是）→ 用 **OVMF**；
- **BIOS + MBR**（Hyper-V Gen1）→ 用 SeaBIOS（QEMU 默认）。

判断方法：`qemu-img convert` 后，用 `fdisk -l win10.qcow2` 看是不是 GPT；或看你当初 Hyper-V 建的是第几代（Gen1=BIOS，Gen2=UEFI）。

### 第 3 步：裸机 ARM Linux 安装并启动

```bash
# Debian/Ubuntu (含树莓派等 ARM 板)
sudo apt update
sudo apt install -y qemu-system-x86 qemu-utils ovmf
```

<details>
<summary>✅ UEFI 启动（GPT 盘，最可能的情况）</summary>

```bash
# 每个 VM 需要一份可写的 VARS 文件
cp /usr/share/OVMF/OVMF_VARS_4M.fd ./OVMF_VARS_4M.fd

qemu-system-x86_64 \
  -machine q35 \
  -accel tcg,thread=multi \
  -cpu max \
  -smp 4 \
  -m 4096 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=OVMF_VARS_4M.fd \
  -drive file=win10.qcow2,format=qcow2,if=ide \
  -nic user,model=e1000 \
  -vga std \
  -device usb-tablet \
  -display gtk
```

</details>

<details>
<summary>✅ BIOS 启动（MBR 盘）</summary>

```bash
qemu-system-x86_64 \
  -machine pc \
  -accel tcg,thread=multi \
  -cpu max \
  -smp 4 \
  -m 4096 \
  -hda win10.qcow2 \
  -nic user,model=e1000 \
  -vga std \
  -device usb-tablet \
  -display gtk
```

</details>

参数说明（关键点）：
- `-accel tcg,thread=multi`：多线程 TCG，**利用多核是唯一的提速手段**，必须配 `-smp N`；
- `-cpu max`：给客户机最大指令集，Win10 x64 需要的 SSE2/PAE/NX/CMPXCHG16b 都在；
- 磁盘用 `if=ide`（或 `if=sata`）而不是 virtio，因为 **Win10 原生自带 IDE/AHCI 驱动**，virtio 需要额外装驱动；
- 网卡用 `e1000`（Win10 原生驱动），别用 virtio-net；
- 无显示器/手机场景把 `-display gtk` 换成 `-display none -vnc :0`，用 VNC 客户端连。

### 第 4 步：Android 上跑

<details>
<summary>方案 A：Termux（灵活，推荐折腾党）</summary>

```bash
pkg update && pkg upgrade
pkg install qemu-system-x86-64-headless qemu-utils
```

把 `win10.qcow2` 放进 Termux 可访问目录，然后：

```bash
qemu-system-x86_64 \
  -machine q35 \
  -accel tcg,thread=multi -cpu max -smp 4 -m 4096 \
  -drive if=pflash,format=raw,readonly=on,file=/path/to/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=/path/to/OVMF_VARS_4M.fd \
  -drive file=win10.qcow2,format=qcow2,if=ide \
  -nic user,model=e1000 -vga std \
  -display none -vnc :0
```

（OVMF 固件文件可从 `ovmf` 包或 edk2 项目下载，放手机里即可。）然后用 VNC Viewer 连 `127.0.0.1:5900`。

</details>

<details>
<summary>方案 B：Limbo PC Emulator（省事，有图形界面）</summary>

- 下载 **Limbo x86_64 版**（GitHub `limboemu/limbo` 的 release APK）；
- 架构选 `x86_64`，机器类型按需要选 `q35`（UEFI）或 `pc`（BIOS）；
- 挂载 `win10.qcow2`（或 raw），CPU 核数拉满，内存尽量大（4G 起）；
- 勾选 TCG，启动。

Limbo 本质就是 QEMU 的 Android 前端，配置项和上面命令一一对应。

</details>

---

## 四、ISO 重新安装（备用，不推荐）

<details>
<summary>只在 VHDX 起不来时考虑（点开）</summary>

TCG 下 Win10 安装**极慢**（可能几小时甚至卡死），所以 VHDX 优先是对的。若必须重装：

```bash
qemu-system-x86_64 \
  -machine q35 \
  -accel tcg,thread=multi -cpu max -smp 4 -m 4096 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=OVMF_VARS_4M.fd \
  -drive file=win10-new.qcow2,format=qcow2,if=ide \
  -cdrom Win10_x64.iso \
  -boot d \
  -nic user,model=e1000 -vga std -device usb-tablet -display gtk
```

装完再把驱动（virtio、显示）等补上，整体耗时和成功率都不如直接转 VHDX。

</details>

---

## 五、提速清单（能缓解，不能根治）

<details>
<summary>点开看优化项</summary>

1. **多核 + 多线程 TCG**：`-smp 4` 配 `-accel tcg,thread=multi`，这是收益最大的一项；
2. 内存给足（4G 以上，别把宿主内存吃满）；
3. 磁盘用 `raw` 或 `qcow2 + cache=none`，减少一层开销；
4. 客户机里关特效、关动画、电源设"高性能"；
5. 避免在 TCG 里做任何重负载（视频、压缩、浏览器）；
6. 树莓派/手机注意散热，TCG 会把 CPU 长期拉满。

即便如此，现实预期是：**启动 20 分钟～2 小时，进去后鼠标都卡**。它更像"能证明能跑"，不是"能用"。

</details>

---

## 六、常见坑

<details>
<summary>点开看排障</summary>

- **蓝屏 `INACCESSIBLE_BOOT_DEVICE` (0x7B)**：Hyper-V 的盘在 QEMU 里磁盘控制器变了（storvsc→IDE/AHCI），Windows 找不到启动盘。开机狂按 F8 进**安全模式**让它自动重装磁盘驱动，或提前在 Hyper-V 里注入标准 AHCI 驱动；
- **卡 Windows logo 转圈**：多等（TCG 就是慢），别以为死机；
- **激活**：硬件大改会导致重新激活，属正常；
- **`-accel tcg,thread=multi` 报错**：某些老版本 QEMU 需要显式 `-machine ...-smp` 配合，升到较新 QEMU 即可。

</details>

---

## 七、更现实的替代方案（按你的真实目的）

如果你的真实目的是"在 ARM 设备上用上 Windows"，这些比 QEMU TCG 强一个数量级：

<details>
<summary>点开看替代路线</summary>

1. **远程桌面（最实用）**：把 x64 Win10 跑在任意一台 x64 真机/服务器上，ARM 设备用 RDP/ToDesk/Parsec 连。体验流畅，几乎无性能损失。
2. **云电脑**：Windows 365 Cloud PC / AWS / Azure 的 x64 云主机，按需付费。
3. **Windows on ARM（如果你只是想要"Windows"）**：Snapdragon 设备装 Win11/10 ARM，自带 x64 应用转译层——但这是 ARM 版系统，**不能用你手里的 x64 镜像**；Apple Silicon 同理（Parallels/VMware 跑 Win11 ARM）。
4. **只求兼容验证**：接受 QEMU TCG 的慢，按上面第四节做。

</details>

---

**一句话总结**：v86 直接放弃（不支持 x64）；QEMU TCG 是唯一能"跑起来"的路，把 VHDX 转成 qcow2、按 UEFI/BIOS 选对固件、开多线程 TCG，能启动但极慢，适合验证；真要日用就上远程桌面或云电脑。