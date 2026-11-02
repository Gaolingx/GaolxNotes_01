# NVIDIA MIG 与 Hyper-V GPU-PV/SR-IOV 差异解析

## 🧑‍💻 User

nvidia gpu要在pve上实现multiinstance需要授权吗

---
## 😀 默认助手 (7)

## 直接结论

**MIG 本身不需要授权**（NVIDIA 的数据中心驱动里 `MIG` 是免费开启的硬件分区特性），**但你想在 PVE 上把 MIG 切片“分给多个 VM”，NVIDIA 官方只给了 vGPU-on-MIG 这一条路，而那就是 NVIDIA vGPU 产品线，必须买授权。**

关键在这两个不同的东西：

| 你说的事 | 是否授权 | 说明 |
|---|---|---|
| 主机上开 MIG、在宿主机上跑多容器/多进程 | ❌ 不需要 | 裸金属/容器部署，数据中心驱动自带 |
| 整块 GPU 直通给**一个** Linux VM，在 guest 里做 MIG | ❌ 不需要 | 官方支持矩阵里的“GPU pass-through to Linux guests” |
| **把单个 MIG 实例分给不同 VM**（一个 VM 一个切片） | ✅ **需要 vGPU 授权** | 官方唯一方式就是 *MIG-backed vGPU*，属 NVIDIA vGPU / AI Enterprise |
| Time-sliced vGPU（不切 MIG，时间片轮转） | ✅ 需要 vGPU 授权 | 同上 |

NVIDIA 官方文档里 MIG 的“受支持部署形态”只有三种：裸金属（含容器/K8s）、GPU 直通给 Linux 客户机、vGPU on hypervisor。**“MIG 实例直通给不同 VM”这一条根本不在支持矩阵里**——原因很实际：原生数据中心驱动不会把 MIG 实例暴露成 mdev/VFIO 设备，只有 vGPU Manager 驱动才会，所以绕不开授权。

---

<details>
<summary><b>授权到底怎么算（vGPU / AI Enterprise）</b></summary>

按 NVIDIA 官方的 *vGPU Software Licensing* 表，授权是**按部署形态**要求的：

- A/B/Q 系列 vGPU → 需要 vApps / vPC / vWS
- GPU 直通 / DDA / vDGA，用于**图形工作站级 3D** → 需要 **vWS**（软件强制校验）
- 直通用于普通 PC 级应用 → 需要 vApps，仅 EULA 强制
- **纯 CUDA / 计算用途** → 不在任何授权条目里，不需要

强制力度：软件强制类（MIG-backed vGPU 属于这类）在拿不到 license 时，VM 启动后 **20 分钟**开始降性能，**24 小时**后再降一档；不是完全不可用，但生产环境别指望。

计费方式：现在 vGPU 的授权路径已经统一到 **NVIDIA AI Enterprise 订阅（按 GPU 每年计费，含 vWS/vApps entitlement 和 vGPU Manager 软件下载权）**，具体报价要找代理商询。注意：**没有 AI Enterprise 订阅，你连 vGPU Manager 驱动都下载不到**——这是比 license 校验更硬的“门槛”。

另外提醒一句：网上流传的 `vgpu_unlock` 之类补丁脚本是绕过驱动授权校验的，一是明确违反 EULA，二是客户机驱动一升级就崩，商用环境不要碰。

</details>

<details>
<summary><b>PVE 上的现实约束（比授权更容易翻车的地方）</b></summary>

1. **硬件门槛**：MIG 只支持 Ampere 及以后的数据中心卡 —— A100、A30、H100、H200、B200、GB200、RTX PRO 6000 Blackwell（Server/Workstation）等。**GeForce/消费级卡（4090/5090）既不支持 MIG，也不被 vGPU 支持**，这点很多人踩坑。

2. **PVE 没有 MIG 的原生支持**：PVE 的 `hostpci` 只支持 `mdev=<类型名>` 这一种 GPU 虚拟化语法，GUI 里也只有 vGPU 的类型下拉框，**没有 MIG 的任何界面或 CLI 封装**。wiki 上 GPU 部分只讲整卡直通和 mediated devices(vGPU)。

3. **`mdev` 的语法对不上 MIG**：`hostpci0: 0000:81:00.0,mdev=nvidia-xxx` 里的类型名要求出现在 `mdev_supported_types` 目录下；而 MIG 实例是“已存在、带 UUID 的 mdev 设备”，不在那个列表里。所以就算你手上有 vGPU Manager，MIG-backed vGPU 在 PVE 上也基本是**手动写 UUID / 绕校验 / hookscript** 的路子，属于“能跑但没人背书”。

4. **官方认证列表里没有 PVE**：vGPU 的 validated platform 列表只有 vSphere、XenServer、RHEL-KVM、Ubuntu、Nutanix AHV 等，Proxmox 属于“Linux KVM 家族，理论可行、官方不支持”。跑不起来或出 bug 时没有 NVIDIA 支持兜底。

5. **MIG 是纯计算分区**：A100/H100 没有显示输出，MIG 实例也没有 vGPU 那套图形能力。想给 Windows 客户机做远程桌面 3D 加速的，MIG 这条线直接不成立，只能走 vGPU。

</details>

<details>
<summary><b>那在实际项目里应该怎么选</b></summary>

按“要不要花钱”排一下：

**方案 A：不想花钱，用整卡直通 + guest 内 MIG**
```bash
# 主机上正常绑 vfio-pci 整卡直通（不需要 nvidia 驱动）
qm set 100 -hostpci0 0000:81:00,pcie=1
```
客户机（Linux）里再 `nvidia-smi -mig 1` + `nvidia-smi mig -cgi ...`。这是**官方支持**的形态，但一块卡只能给**一个** VM，MIG 只在那个 VM 内部做多租户隔离。适合“一个大 K8s/训练节点 VM”的场景。

**方案 B：真要多 VM 各拿一份，走 vGPU（需授权）**
```bash
# 主机装 vGPU Manager 驱动后，类型来自 mdev_supported_types
ls /sys/bus/pci/devices/0000:81:00.0/mdev_supported_types
qm set 100 -hostpci0 0000:81:00.0,mdev=nvidia-35
```
同时在客户机装 GRID 驱动、配置 license server。A100/A30 上可以选 MIG-backed 类型（如 `nvidia-xxx` 里的 1g.10gb 对应项），隔离性最好。

**方案 C：消费卡 / 预算敏感的多用户**
PVE 官方没有免费的官方方案。现实做法就是“一块卡一个 VM”，用多张卡堆；或者接受 vGPU 授权成本。时间片轮转的官方版本（time-sliced vGPU）同样要授权。

</details>

<details>
<summary><b>如果确定要上，PVE 侧的准备清单</b></summary>

- 主板/CPU 开 IOMMU/VT-d，内核参数 `intel_iommu=on iommu=pt`（AMD 平台默认开）
- **注意**：mdev/vGPU 场景下**不要**把 GPU 绑到 `vfio-pci`，设备必须留在 host 驱动（vGPU Manager）手里；整卡直通才需要绑
- 装 vGPU Manager 驱动（不是普通 `.run` 数据中心驱动），`nvidia-smi` 能看到 `vGPU` 相关类型
- 关掉内核里的 nouveau、确保 `nvidia-vgpud`/`nvidia-vgpu-mgr` 服务正常
- license server（NLS 或旧版 FlexLM）先通，客户机里 `nvidia-smi -q` 看 License Status
- Windows 客户机记得装 GRID 版驱动，普通驱动装上去会认不到 vGPU 设备

</details>

---

**一句话总结**：MIG 这个特性不要钱，但“MIG 切片分给多台 VM”在 NVIDIA 的产品定义里就是 vGPU，要买 AI Enterprise；不买授权的话，PVE 上只能整卡直通给一台 Linux VM，然后在它里面自己做 MIG 隔离。

如果你能说一下具体是哪块卡、几个 VM、客户机是 Linux 还是 Windows、有没有图形需求，我可以直接给一个确定的选型建议。

---
## 🧑‍💻 User

但是sriov不是硬件层面特性吗，为什么hyperv的gpupv不用

---
## 😀 默认助手 (7)

## 名字先拆开——这里有个很容易混淆的坑

你说的“Hyper-V 的 GPU-PV”和微软文档里的“GPU partitioning（GPU-P）”**是两个不同的东西**，只是名字撞了：

| 特性 | 平台 | 机制 | 要不要 SR-IOV |
|---|---|---|---|
| **GPU-PV**（GPU 半虚拟化） | Win10/11 **客户端** Hyper-V、**WSL2**、Windows Sandbox | 纯软件：guest 里的虚拟 GPU 通过 **VMbus** 把命令转发给宿主机内核里的真实 GPU 驱动，由宿主驱动做调度和显存分配 | ❌ **完全不用** |
| **GPU partitioning / GPU-P** | **Windows Server 2025** | 微软原话：“uses the Single Root IO Virtualization (SR-IOV) interface” | ✅ **就是 SR-IOV VF** |
| **DDA**（离散设备分配） | Windows Server | 整卡 PCI 直通 | 不用（但也不是分区） |

所以你观察到的“GPU-PV 不用 SR-IOV”是对的——**它就是不用，而且是刻意设计的**。但微软在 Server 2025 里做的 GPU-P 恰恰**就是** SR-IOV，两个功能路线完全相反。

---

<details>
<summary><b>为什么 GPU-PV 刻意不用 SR-IOV（六条真实原因）</b></summary>

**1. 目标硬件上根本没有 SR-IOV**
GPU-PV 的目标是“让 Windows 笔记本/台式机上那块核显或 GeForce 也能被 Hyper-V 分给 VM”，因为 WSL2 要在**所有** Windows 机器上跑 CUDA/DirectML。可是消费级 GPU、核显几乎都不实现 SR-IOV——给 GPU 做 SR-IOV 需要每个 VF 独立 MMU、独立显存分区、独立 FLR 复位逻辑，这是极贵的硅片面积。厂商没动力在 4090 上做。

**2. Windows 客户端 Hyper-V 没有 DDA，只能软件共享**
DDA（整卡直通）是 **Windows Server 独占**的功能，客户端 Hyper-V 从来没给过（策略原因）。所以在客户端上做 GPU 共享，从一开始就只有“软件半虚拟化”这一条路可选。

**3. WDDM 的“单驱动管一个 GPU”模型和 VF 冲突**
Windows 显示栈是一个 display miniport driver 管整块 GPU 的显存管理器、调度器、电源管理。SR-IOV 要求每个 VF 在 guest 里跑一份**完整、独立、互不知情**的驱动实例，这和 WDDM 客户端的全局资源模型天生不匹配。微软与其改 WDDM，不如在上层加一层半虚拟化接口。

**4. SR-IOV 给不了 GPU-PV 想要的产品特性**
- 分区大小**运行时动态改**（GPU-PV 可以改显存配额；VF 是固定切分）
- **超分**（分配总和可以超过物理显存；VF 不行）
- **宿主自己也能用这张卡**（VF 分出去就锁死）
- **在线迁移**（客户端场景不需要，但 Server 2025 为了迁移，反而必须回去求 SR-IOV + IOMMU DMA bit tracking + NVIDIA vGPU v18 驱动）
- 分区数量随显卡型号变（A2 能切十几份，A40 只能切个位数）

**5. 隔离性本来就不是 GPU-PV 的目标**
GPU-PV 的命令路径是 guest → VMbus → **宿主内核 GPU 驱动**。宿主驱动要解析 guest 提交的命令缓冲区，没有 IOMMU 那样的硬件边界。所以微软文档明确不推荐把 GPU-PV 用在**不可信的多租户**场景——它面向的是“我自己信任的 VM”，比如 WSL2、开发沙箱、VDI 内部用户。真要硬件隔离边界，就得回 SR-IOV（IOMMU 挡 DMA）或 MIG（片上 crossbar/L2/显存控制器级隔离）。

**6. 历史包袱**
GPU-PV 的前身是 RemoteFX vGPU（2010），那套是把渲染放进了 hypervisor，攻击面巨大，2021 年被微软废弃。GPU-PV 换个思路：hypervisor 不进数据通路，让宿主机的厂商驱动去做事——代价就是安全边界从“硬件”降级成“驱动质量”。

</details>

<details>
<summary><b>反过来说：SR-IOV 是硬件特性，但“能用”不等于“免费”</b></summary>

这点和你上一个问题直接相关——**SR-IOV 接口本身是 PCIe 规范里免费的硬件能力，但把 GPU 分区喂给 VM 的那层软件栈不免费。**

看微软自己的文档就很有意思，Windows Server 2025 的 GPU partitioning 支持列表：

```
NVIDIA A2 / A10 / A16 / A40 / L2 / L4 / L40 / L40S
NVIDIA RTX PRO 6000 Blackwell Server Edition
AMD Radeon PRO V710
```

而且原文写着：

> To use live migration with GPU partitioning for NVIDIA GPUs, you must use the driver included in the **NVIDIA vGPU Software v18.x or later**.

也就是说：**微软做了个 SR-IOV 的框架，但底下填的坑还是 NVIDIA vGPU**。你依然要买 vGPU / AI Enterprise 授权，才能拿到那个实现了 SR-IOV PF/VF 管理的驱动。

顺带注意两个细节：
- 列表里**没有 A100 / H100**。那两张卡走的是 **MIG** 路线（芯片级分区），不是 SR-IOV VF 路线。两条硬件路线是并列的，不是替代关系。
- 分区是**自动分配**的，你没法指定某个分区给某个 VM；集群化还得上 **Datacenter** 版授权。

所以完整的图景是：

```
物理隔离强度：MIG（片上分区） > SR-IOV VF（IOMMU 边界） >> GPU-PV（驱动层多路复用）
软件/授权成本：MIG 免费(直通) / 收费(vGPU) ；SR-IOV 免费(网卡) / 收费(GPU) ；GPU-PV 免费
硬件门槛：      MIG 仅数采卡      ；SR-IOV 仅数采卡          ；GPU-PV 任何卡
```

</details>

<details>
<summary><b>回到 PVE：这件事对你的实际影响</b></summary>

**GPU-PV 在 PVE 上等于不存在**，不是“没集成”，而是**架构上不可能**：

- 它依赖 **VMbus**（Hyper-V 的 VMBus 通道）+ **WDDM 半虚拟化驱动契约**。guest 里那个“虚拟 GPU”是 WDDM 概念，Linux 没有对应物；宿主侧那套 `Add-VMGpuPartitionAdapter` 也是 Windows 内核里的东西。
- 所以 WSL2 能跑 CUDA 是 Windows 独有的福利，Linux 世界没有等价物。Linux 上“GPU 半虚拟化”的历史尝试是 Intel GVT-g（mdev），**已经废弃**，官方转向 SR-IOV。

PVE 上你实际的选项就三个：

| 做法 | 隔离 | 授权 | 适用 |
|---|---|---|---|
| `hostpci0: 0000:81:00` 整卡直通 | 硬件级（独占） | 免费 | 一块卡一个 VM，最稳 |
| `hostpci0: ...,mdev=<type>`（vGPU） | 硬件级（时间片/MIG） | **要买** | 多 VM 共享，唯一官方路径 |
| SR-IOV VF 直通（`sriov_numvfs` 后当普通 PCI 设备直通） | 硬件级（IOMMU） | **要买** | 网卡成熟；GPU 上仍卡在 NVIDIA 驱动授权 |

第三种就是 PVE wiki 里 SR-IOV 那一节讲的做法，网卡上完全免费好用（Intel/Mellanox 的 VF 驱动是开源免费的），但**搬到 GPU 上就撞上同一个墙**：VF 的 PF 管理驱动在 NVIDIA 手里，不买 vGPU 你拿不到。

</details>

---

**一句话回答你的问题**：SR-IOV 确实是硬件特性，但“硬件特性”只意味着**隔离边界免费**，不意味着**管理栈免费**；而 GPU-PV 之所以不用它，是因为它的设计目标是“让没有 SR-IOV 的消费级 GPU 也能被共享”，一旦要求 SR-IOV，WSL2 和笔记本用户就全被排除在外了——微软宁可牺牲隔离强度，换取覆盖面。

（顺带一说：Server 2025 的 GPU-P 转回 SR-IOV，恰恰证明了当微软真的需要“给多台 VM 分 GPU 且要有硬件边界”时，它也只能回到 SR-IOV 或 MIG，没有第三条路。）
