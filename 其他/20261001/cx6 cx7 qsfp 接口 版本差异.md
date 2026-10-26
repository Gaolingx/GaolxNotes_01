# cx6 cx7 qsfp 接口 版本差异

## 🧑‍💻 User

cx6和7网卡是qsfp吗 cx7是不是有两个版本

---
## 😀 默认助手 (7)

## 结论先说

**不是"都是 QSFP"。** CX6 大部分是 QSFP 系（QSFP56 / QSFP28），但低速型号用的是 SFP28/SFP56；CX7 则是**三种接口形态并存**——OSFP、QSFP112、SFP56。

至于"CX7 是不是有两个版本"：高速型号确实是**两个物理接口家族（OSFP 和 QSFP112）**，而且每个家族内部又分 400G / 200G 两档能力。

---

## ConnectX-6 家族的接口

| 子系列 | 速率 | 接口/笼子 | 典型 OPN |
|---|---|---|---|
| ConnectX-6 VPI（HDR IB） | 200Gb/s IB、200GbE | **单 QSFP56** 或 **双 QSFP56** | MCX653105A-HDAT、MCX653106A-ECAT |
| ConnectX-6 Dx | 200GbE | **单 QSFP56**（1×200G）或 **双 QSFP56**（2×100G） | MCX623105A\*-CDAT/VDAT、MCX623106A\*-CDAT |
| ConnectX-6 Dx（低速版） | 25 / 50GbE | **双 SFP28**（2×25G）或 **双 SFP56**（2×50G） | MCX623102A\*-ADAT、MCX623102A\*-GDAT |
| ConnectX-6 Lx | 25 / 50GbE | **双 SFP28**（2×25G）或 **单 QSFP28**（1×50G） | MCX631102A\*-ADAT、MCX631105A\*-GDAT |

👉 特别注意 **ConnectX-6 Lx**：它不是 QSFP56——25G 版是 SFP28，50G 单口版是 **QSFP28**（NVIDIA 手册明确写的 Single QSFP28）。

---

## ConnectX-7 家族的接口（官方手册全量）

| 接口形态 | 速率 | 端口 | OPN |
|---|---|---|---|
| **单口 OSFP** | 400G NDR / 400GbE | 1 | MCX75310AAC/AS-NEAT、MCX75510AAS-NEAT |
| **单口 OSFP** | 200G NDR200 / 200GbE | 1 | MCX75310AAS-HEAT、MCX75510AAS-HEAT（纯 IB） |
| **单口 QSFP112** | 400G NDR / 400GbE，**支持端口拆分** | 1 | MCX715105AS-WEAT |
| **双口 QSFP112** | 2×200G NDR200 / 2×200GbE | 2 | MCX755106AC/AS-HEAT |
| **双口 QSFP112**（以太网专用） | 2×100GbE / 2×200GbE | 2 | MCX713106A\*-CEAT / -VEAT |
| **四口 SFP56** | 4×25 / 50GbE | 4 | MCX713104A\*-ADAT、MCX713114TC-GEAT |

所以"两个版本"的准确说法是：

1. **按接口分：OSFP 版 vs QSFP112 版**（这是最本质的两个版本，二者机械上不通用）；
2. **按速率分：400G（NDR/400GbE）vs 200G（NDR200/200GbE）**，OSFP 和 QSFP112 里都各有这两种档位。

<details>
<summary><b>兼容性与选型要点（点开）</b></summary>

- **QSFP112 / QSFP56 / QSFP28 是同一个机械外形**，互相可插。QSFP112 = 4×112G PAM4（可达 400G）。所以把老的 QSFP56 200G 线插到 CX7 的 QSFP112 口上能通，只是速率按线走（200G）。
- **OSFP 和 QSFP 互不兼容**，不能对插，需要 OSFP↔QSFP 转接/分支线（例如 OSFP 转 2×QSFP112）。
- CX7 的 **OSFP 卡只支持 RHS（Riding Heatsink）笼子**，换机箱/笼子时要注意。
- 选型的实用口诀：
  - 对接 **NVIDIA Quantum-2 QM9700（64×400G，QSFP112 笼子）** → 选 **CX7 QSFP112 单口 400G（MCX715105AS-WEAT）**；
  - 对接 **OSFP 形态的 400G/800G 交换机（如 QM9790 这类）** → 选 **CX7 OSFP 版**；
  - 只是想要 2×200G 以太网/IB → **MCX755106AS-HEAT（双口 QSFP112）**是性价比最高的常见型号；
  - 25/50G 场景 → CX6 Lx / CX6 Dx 的 SFP 版本，或 CX7 四口 SFP56。

</details>

<details>
<summary><b>信息来源</b></summary>

- NVIDIA《ConnectX-7 Adapter Cards User Manual》→ Specifications / Supported Interfaces（networking-docs.nvidia.com/connectx7hw/specifications）
- NVIDIA《ConnectX-6 Dx Ethernet Adapter Cards User Manual》→ Specifications
- NVIDIA《ConnectX-6 Lx Ethernet Adapter Cards User Manual》→ Specifications
- NVIDIA《ConnectX-6 VPI Adapter Cards User Manual》→ Specifications

以上均为官方文档原文（"Connector: Dual SFP28 / Single QSFP56 / Dual QSFP112 ..."）。
</details>

需要我针对你要对接的具体交换机型号（或预算）给一份 OPN 选型清单吗？
