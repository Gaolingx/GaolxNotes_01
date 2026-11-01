# SLI桥不给CUDA/NCCL用；PCIe P2P驱动层封禁

## 🧑‍💻 User

gtx600/700的这个sli是不是不能cuda用，就是nccl走这个高速链路传数据啥的

---
## 😀 默认助手 (7)

## 先说结论

**不能。SLI 桥对 CUDA 来说完全不可见，NCCL 也从来不走这条链路。** 这两件事都跟 SLI 无关：`cudaDeviceCanAccessPeer` 看不到它，`nvidia-smi topo -m` 也看不到它。

---

### 1. SLI 桥是图形专用链路，不是计算互联

Kepler 时代的 SLI 桥只负责以下几件事：

- AFR（交替帧渲染）的帧分发
- 帧同步、帧节奏（frame pacing）
- SLI 抗锯齿和部分特效用到的中间数据交换

它是 NVIDIA 的**私有图形链路**，没有暴露给任何 CUDA Runtime API 或 Driver API。没有任何 API 能往 SLI 接口上发送数据，也没有 `cudaMemcpy` 路径会经过它。它本质上是个黑盒，CUDA 完全不知道它存在。

### 2. CUDA 里 GPU 之间的高速通路只有两条

| 通路 | 出现时间 | 范围 |
|---|---|---|
| **NVLink** | Pascal P100（2016）起 | 专业卡、部分 DGX；消费级到 RTX 40 才有 NVLink（4090 已取消） |
| **PCIe P2P** | Fermi/Kepler 起，同一个 root complex | 全系硬件支持，**但驱动可封** |

NCCL 的传输选择顺序是：

```
NVLink  →  PCIe P2P  →  IB/RoCE/TCP 网络  →  同机 fallback（SHM + 主机内存中转）
```

这个列表里**压根没有 SLI**。SLI 不在 NCCL 的 transport 枚举里。

### 3. 更关键的是：GeForce 上连 PCIe P2P 都被驱动封了

这才是实际导致性能受限的原因，和 SLI 没有关系：

```cuda
int can = 0;
cudaDeviceCanAccessPeer(&can, 0, 1);   // GTX 600/700 上返回 0
```

- `cudaDeviceCanAccessPeer()` 在消费级 GeForce 上恒为 0
- CUDA samples 里的 `p2pBandwidthLatencyTest` 会直接报 `Peer access is not supported between these two devices`
- 于是 NCCL 在两张 GTX 之间只能走 **shared memory + 两次 `cudaMemcpy`（host staging）**，或者退化成 socket。带宽相比 P2P 会大幅下降，还会额外占用 host 内存带宽。

### 4. 就算能用，SLI 桥也不快

Kepler 时代的 SLI 链路带宽是 **~1 GB/s 量级（单向）**，而 PCIe 3.0 x16 是 16 GB/s。它从设计上就是**低延迟小包**（同步信号、帧数据），不是高吞吐互联。拿它当“高速链路”传梯度是方向搞反了。

### 5. GTX 600/700 本身的现实约束

无论 SLI 能否使用，这代卡在 2024+ 都很难有实际用途：

- **算力架构**：GTX 680/670/660/690 = `sm_30`，GTX 780/780Ti/Titan/770/760 = `sm_35`
- **CUDA 支持**：`sm_30` 在 CUDA 11 被移除，Kepler 整体在 CUDA 12 被砍掉，因此只能用 CUDA 10.x 或更老
- **NCCL**：现代 NCCL 早就不编译 Kepler 目标，得用 2.4～2.7 那种老版本
- **显存**：2～4 GB，无 Tensor Core
- **双芯卡 690 / Titan Z**：板载 PLX 桥，在 CUDA 里是两个独立 device，P2P 一样被驱动禁掉

<details>
<summary><b>怎么自己验证（点开）</b></summary>

```bash
# 1. 拓扑：看有没有 NV#（NVLink）；PIX/PXB/PHB 是 PCIe 层级；SYS 是跨 NUMA
nvidia-smi topo -m

# 2. 官方 P2P 带宽/延迟测试（CUDA samples 里）
./p2pBandwidthLatencyTest
# 预期输出：两张 GTX 之间 "Peer access is not supported"

# 3. 看 NCCL 实际用了什么 transport
export NCCL_DEBUG=INFO
./your_nccl_program
# 日志里会打印 "NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM" 或 "via P2P/CUMEM"
```

在所有输出里，你都**不会**看到 SLI 相关的任何字段。

</details>

<details>
<summary><b>一个历史例外（跟 SLI 仍无关）</b></summary>

2013–2014 年、CUDA 5/6 时代，有零星报告说某些 Kepler GeForce 在极老驱动下能开 PCIe P2P，后来被 NVIDIA 收紧掉了。近年 geohot / tinygrad 在 RTX 4090 上打驱动补丁绕过 P2P 检查（`nvidia-p2p` 那个），也再次证实 **P2P 是软件限制而非硬件限制**。

但注意：**这些全都是 PCIe P2P，没有一个跟 SLI 桥有关**。补丁只是让 `cudaMemcpyPeer` 走 PCIe，跟 SLI 口没有关系。

</details>

---

### 想多卡跑 NCCL，实际建议

- 最低门槛：**Pascal 及以后的卡**（至少 sm_60），且能用 `nvidia-smi topo -m` 看到 `PIX`/`PXB` 那种同桥拓扑，再打 P2P 补丁
- 真正舒服：**P100/V100/A100/H100 + NVLink**，或者至少 RTX 30/40 系列（同 NUMA、同 PCIe switch）配合 `NCCL_P2P_LEVEL`
- 如果目标是 GTX 600/700 这种卡上的双卡训练：**性价比极低**，NCCL 拿不到 P2P，显存小、算力老、工具链全断，不如单卡大显存或者直接上现代卡

一句话：**SLI 桥是给显示器看的，不是给计算用的；NCCL 用的是 NVLink 和 PCIe P2P，而 GeForce 的 P2P 被驱动关了。**

---
## 🧑‍💻 User

pcie p2p能用吗

---
## 😀 默认助手 (7)

## 分层回答：不能用（驱动层封了），但硬件本身是支持的

### 第一层：官方答案 —— GTX 600/700 上 **不能用**

```cuda
int can = 0;
cudaDeviceCanAccessPeer(&can, 0, 1);
// GTX 680 / 780 / 690 / Titan Z  →  can == 0
```

NVIDIA 按产品线把 P2P 划给了 **Tesla / Quadro** 专业卡，GeForce 一律返回"不支持"。这是驱动里一个产品分类检查（`is_p2p_supported` 之类的 flag），**不是硬件能力问题**。

后果是容易误判的一点：

```cuda
cudaMemcpyPeerAsync(dst, 1, src, 0, size, stream);
```

即使 `can == 0`，这个调用**照样不报错**，但它会自动 fallback 成"拷到 host 再拷回 GPU"（host staging）。看上去是通的，实际没走 P2P。所以判断依据必须是 `cudaDeviceCanAccessPeer` 或 `p2pBandwidthLatencyTest` 的带宽数字，不能看有没有报错。

### 第二层：硬件层 —— **能**，Kepler 硅片支持

同代 Tesla K20/K40/K80、Quadro K6000 都支持 PCIe P2P，GTX 680/780 用的是同一套 Kepler 硅片和 PCIe 控制器。所以：

> **P2P 是被驱动按型号禁用的，不是做不出来。**

这也被后来的事情反复证实：2023 年 geohot / tinygrad 在 RTX 4090 上打驱动补丁，把那个检查绕掉，P2P 就通了。**是软件闸门。**

### 第三层：就算绕过驱动，还有两道坎

| 坎 | 说明 | 怎么查 |
|---|---|---|
| **拓扑** | 两块卡必须在**同一个 PCIe switch 下**（`PIX`/`PXB`）。走 root complex（`PHB`）或跨 CPU（`SYS`）时，根复合体**不转发 P2P 流量**，驱动开了也没用 | `nvidia-smi topo -m` |
| **ACS** | PCIe 的 Access Control Services 默认会拦截 P2P 事务。消费级主板/PLX 卡上常常是打开的，要 `setpci` 关掉 | `lspci -vvv \| grep ACSCtl` |

第 1 条是大多数普通桌面平台上 M.2 / PCIe 插两张卡时 P2P 失败的真因，**跟你是不是 GeForce 无关**。这也是双 3090 走 PCIe 跑 NCCL 慢、必须靠 NVLink 桥的根本原因。

<details>
<summary><b>如果你真想试（历史路径 + 现代路径）</b></summary>

**历史路径（Kepler 时代）**

2013–2014 年，CUDA 5/6 + 3xx 驱动时代，有报告称部分 Kepler GeForce 能开 PCIe P2P（当时驱动里那个检查还没收紧）。要复现得：

- 用 `nvidia-smi topo -m` 确认是 `PIX`/`PXB`
- 跑 CUDA 6.x 的 `p2pBandwidthLatencyTest`
- 基本只能靠虚拟机镜像 / 老发行版，文档极少，不能指望

**现代路径（改驱动）**

tinygrad 那套思路：把 `nvidia` 内核模块里判断"这卡是不是专业卡"的分支 patch 掉。但要注意：

- 需要能编译或二进制 patch 内核模块，配合 `nouveau`/开源模块更麻烦
- 每升级一次驱动就要重打
- **GTX 600/700 用这套毫无意义**——它的瓶颈不是 P2P，是整个平台太老

**官方能用的方案**

- 换 Tesla/Quadro 同代卡（K20/K40/K6000），P2P 开箱即用
- 或者干脆放弃这代平台

</details>

---

### 自己验证的最小步骤

```bash
# 1. 看拓扑：NV# = NVLink，PIX/PXB = 同 switch（P2P 有可能），
#    PHB = 走 root complex（P2P 基本没戏），SYS = 跨 CPU（肯定没戏）
nvidia-smi topo -m

# 2. 官方带宽/延迟测试（CUDA samples）
./p2pBandwidthLatencyTest
# 期望：GTX 上打印 "Peer access is not supported between these two devices"

# 3. 看 NCCL 究竟走了哪条路
NCCL_DEBUG=INFO ./your_prog 2>&1 | grep -i "via\|P2P"
#   via P2P     → 真 P2P
#   via SHM     → 主机内存中转（没 P2P）
```

---

### 结论

- **官方身份（GeForce GTX 600/700）：不能用**，`cudaDeviceCanAccessPeer` 恒为 0
- **硬件：能用**，Kepler 硅片支持；同代 Tesla/Quadro 就有
- **绕过驱动：理论可行**（历史上有老驱动、现代有 patch），但还要过拓扑 + ACS 两关
- **实际上不值得**：GTX 600/700 受限于 `sm_30/sm_35`（CUDA 11 起弃用 sm_30，CUDA 12 砍掉 Kepler）、2–4 GB 显存、无 Tensor Core，NCCL 也只剩老版本。**P2P 通不通已经不是这代卡的瓶颈了**——瓶颈是整个平台。

真要跑多卡 NCCL，最低门槛是 Pascal 之后 + `PIX`/`PXB` 拓扑，舒服的是 P100/V100/A100 + NVLink 那一档。
