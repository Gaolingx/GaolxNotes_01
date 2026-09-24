# 跨NUMA启用P2P

## 🧑‍💻 User

运行训练代码之前设置`NCCL_P2P_LEVEL=SYS`有什么意义？

---
## 🤖 Assistant

## 简短回答

`NCCL_P2P_LEVEL=SYS` 的作用是**把 NCCL 允许使用 GPU 之间 P2P（Peer-to-Peer）直连的"拓扑距离阈值"放宽到最大**，即：即使两块 GPU 位于**不同的 CPU socket / NUMA 节点**上、只能走"系统级"路径，NCCL 也**仍然尝试启用 P2P（PCIe GPUDirect 直连）**，而不是退回到经过主机内存的共享内存（SHM）传输。

它的默认值是 `PHB`（Host Bridge），也就是"同一颗 CPU 所在的 PCIe 域内才用 P2P"。

---

## 背景：NCCL 如何选通信路径

NCCL 给每条 GPU 间通信路径标注一个"拓扑距离等级"，数值越小越近、越快：

| 等级 | 含义 | 典型场景 |
|------|------|----------|
| `LOC` / `NVL` | 同一位置 / NVLink | 同一块板上的 SXM GPU（走 NVLink） |
| `PIX` | 同一个 PCIe Switch 下 | 同 switch 的两个 GPU |
| `PXB` | 经过多个 PCIe Switch | 同域内跨 switch |
| `PHB` | 经过 PCIe Host Bridge（同一 socket） | 同一 CPU 下的 GPU |
| `SYS` | 需要穿过 CPU 互连（UPI / Infinity Fabric 等） | **跨 socket 的 GPU** |

- 若两端距离 $\le$ `NCCL_P2P_LEVEL`，NCCL **启用 P2P**（GPU 直连，走 PCIe/NVLink）。
- 若距离 $>$ 阈值，NCCL **禁用 P2P**，改用 **SHM**（数据经主机内存中转）。

默认阈值是 `PHB`，所以**跨 socket 的 GPU 对默认不会启用 P2P**。而 `SYS` 在 NCCL 源码里被解析成一个"无穷大"值（约 `1000`），因此**任何拓扑距离都允许 P2P**。

---

<details>
<summary><b>为什么默认是 PHB？为什么需要手动设 SYS？（点开）</b></summary>

NVIDIA 的默认假设是：**跨 socket 的 PCIe P2P 往往不比走共享内存快，甚至更慢或更不稳定**。因为跨 socket 的 PCIe 事务实际要经过 CPU 互连（UPI/IF），带宽和延迟都不理想。所以默认只在"确定快"的范围内（同一 socket）才用 P2P。

但在某些平台/拓扑下，**跨 socket 的 GPUDirect P2P 仍然比 SHM 快**，或者（更常见）**NCCL 的拓扑探测"误判"了距离**，导致本该用 P2P 的链路被降级为 SHM。这时把 `NCCL_P2P_LEVEL=SYS` 强制打开 P2P，可以显著提升集合通信（AllReduce、AllGather 等）带宽。

</details>

---

<details>
<summary><b>典型使用场景 / 意义</b></summary>

1. **强制启用跨 NUMA 的 GPU 直连**
在只有 PCIe、没有 NVLink/NVSwitch 的多卡机器上，让跨 socket 的 GPU 对也走 GPUDirect P2P，避免数据在主机内存里多走一趟。

2. **绕过拓扑探测不准的问题**
有些主板/BIOS/虚拟化环境下 NCCL 会把 GPU 间距离算成 `SYS`，于是默认直接禁用 P2P。显式设为 `SYS` 可覆盖这种误判，恢复 P2P。

3. **统一多机/多卡行为、便于复现**
显式写死一个等级，避免不同节点拓扑探测差异导致通信路径不一致，从而让性能和行为可复现。

4. **配合 `NCCL_P2P_DISABLE=1` / 其他变量做 A/B 测试**
常和 `NCCL_DEBUG=INFO`、`NCCL_P2P_DISABLE` 一起排查通信瓶颈。

⚠️ 注意：在 **SXM + NVLink/NVSwitch** 的机器上，GPU 间优先走 NVLink，`NCCL_P2P_LEVEL` 主要影响 PCIe 路径，改动通常**几乎无效果**。它主要对 **PCIe 多卡（尤其跨 socket）** 有意义。

</details>

---

<details>
<summary><b>风险与代价</b></summary>

- **可能反而变慢**：跨 socket 的 PCIe P2P 若确实要穿过 CPU 互连，带宽可能低于共享内存方案。
- **可能引起挂起/报错**：某些 GPU/主板组合的跨 socket P2P 存在硬件或驱动 bug，强行开启会 hang 或报 `unhandled system error`。
- **并非万能加速**：它只是改变"用哪条路"，不增加物理带宽。
- **优先级**：`NCCL_P2P_DISABLE=1` 仍然会整体关掉 P2P；`NCCL_P2P_LEVEL` 是在未被禁用的前提下放宽范围。

实践中建议：先不设（用默认），跑一次看带宽；只有当确认跨 socket P2P 更快、或默认探测误判时，才加 `SYS`，并做对比测试。

</details>

---

## 如何验证是否生效

```bash
# 查看物理拓扑（确认哪些 GPU 跨 socket）
nvidia-smi topo -m

# 打开 NCCL 日志，观察实际选用的通道/传输
NCCL_DEBUG=INFO NCCL_P2P_LEVEL=SYS python train.py 2>&1 | grep -iE "P2P|SHM|Channel|Ring|Tree"
```

日志里若看到对应链路从 `SHM` 变成 `P2P`，说明设置起了作用。

---

**一句话总结**：`NCCL_P2P_LEVEL=SYS` = "别管跨不跨 socket，只要能 P2P 就用 P2P"，主要用于**让跨 NUMA 节点的 GPU 走 PCIe 直连、或修正 NCCL 对拓扑距离的误判**；但要小心它在某些硬件上可能更慢甚至挂起，需实测确认。
