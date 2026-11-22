# Xilinx HBM AI FPGA型号汇总

## 🧑‍💻 User

xilinx hbm aifpga有哪些

---
## 😀 默认助手 (7)

## 先说明一下

“Xilinx HBM FPGA” 现在属于 **AMD（原 Xilinx）** 产品线，公开在售且带**封装内 HBM**的器件共有两大系列、12 款芯片，另外还有 3 款基于这些芯片的 Alveo 加速卡。这两大系列分别是：16nm 的 **Virtex UltraScale+ HBM**（HBM2，460 GB/s）和 7nm 的 **Versal HBM 系列**（HBM2e，818–820 GB/s）。

---

## 一、Virtex UltraScale+ HBM（16nm，HBM2）

共 **7 款**，全部为 `P` 后缀（Package 内集成 HBM）：

| 器件 | 逻辑单元 (K) | DSP | HBM 容量 | HBM AXI 端口 | GTY/GTM (32.75/58G) | I/O |
|---|---|---|---|---|---|---|
| XCVU31P | 962 | 2,880 | 4 GB | 32 | 32 / 0 | 208 |
| XCVU33P | 962 | 2,880 | 8 GB | 32 | 32 / 0 | 208 |
| XCVU35P | 1,907 | 5,952 | 8 GB | 32 | 64 / 0 | 416 |
| XCVU37P | 2,852 | 9,024 | 8 GB | 32 | 96 / 0 | 624 |
| XCVU45P | 1,907 | 5,952 | 16 GB | 32 | 64 / 0 | 416 |
| XCVU47P | 2,852 | 9,024 | 16 GB | 32 | 96 / 0 | 624 |
| XCVU57P | 2,852 | 9,024 | 16 GB | 32 | 32 / 32（58G PAM4） | 624 |

关键点：
- 全系 **460 GB/s** HBM2 带宽，约 **7 pJ/bit**；集成 HBM 控制器 + 端口交换（any port → any address），可省约 250K LUT。
- PCIe Gen3x16 / Gen4x8 + **CCIX**。
- 典型板卡：**VCU128 评估套件（VU37P）**。

---

## 二、Versal HBM 系列（7nm，HBM2e）

共 **5 款**（VH = Versal HBM）：

| 器件 | HBM 容量 | 逻辑单元 (K) | LUT | DSP | BRAM / URAM |
|---|---|---|---|---|---|
| VH1522 | 8 GB | 3,837 | 1,753,984 | 7,392 | 89 / 366 Mb |
| VH1542 | 16 GB | 3,837 | 1,753,984 | 7,392 | 89 / 366 Mb |
| VH1582 | 32 GB | 3,837 | 1,753,984 | 7,392 | 89 / 366 Mb |
| VH1742 | 16 GB | 5,631 | 2,574,208 | 10,848 | 132 / 541 Mb |
| VH1782 | 32 GB | 5,631 | 2,574,208 | 10,848 | 132 / 541 Mb |

平台特性（全系）：
- **HBM2e 最高 819 GB/s、32 GB**，由可编程 NoC 全局访问（2.2 Tb/s 片内互连）
- 112G PAM4（GTM 20/60 路）+ 32.75G GTYP 681 路，串行带宽最高 **5.6 Tb/s**
- 2× PCIe Gen5x8（CPM5，带 DMA）+ 8× Gen5x4（PL PCIe5）
- **400G 高速加密引擎 2~3 个**、100G 多速率 MAC 4~6 个、600G EMAC、600G Interlaken（VH17x2）
- 处理子系统：双核 Cortex-A72 + 双核 Cortex-R5F + PMC

⚠️ 重要区分：**Versal HBM 系列没有 AI Engine**（它基于 Versal Premium 架构）。如果要 AI Engine + HBM，需要自行组合，或者用 **VCK5000**（AI Core VC1902，无 HBM）。

---

## 三、带 HBM 的 Alveo 加速卡（板级产品）

| 卡 | 器件 | HBM | 带宽 | 备注 |
|---|---|---|---|---|
| **Alveo V80** | Versal HBM XCV80（VH1782 级） | 32 GB HBM2e（2×16GB） | 820 GB/s | 2.6M LUT / 10,848 DSP，4×200G，Gen4x16 或 2×Gen5x8，190W，另有 DDR4 DIMM 扩展 |
| **Alveo U55C** | Virtex UltraScale+ XCU55 | 16 GB HBM2 | 460 GB/s | 1.3M LUT / 9,024 DSP，2×100G，150W，主打 HPC 集群 |
| **Alveo U50** | Virtex UltraScale+ XCU50 | 8 GB HBM2 | 316 GB/s（峰值，标称 201 GB/s） | 872K LUT，75W 单槽半高，Gen4 + 100G |
| Alveo U280（旧款） | VU37P | 8 GB HBM2 | 460 GB/s | 已逐步退市 |

---

<details>
<summary>四、选型建议 & 易混淆点（点击展开）</summary>

**按需求选：**

- **要最大容量和带宽（AI 推理、数据库、防火墙、测试仪表）** → Versal HBM（VH1582、VH1782，32 GB / 819 GB/s），或者直接买 **Alveo V80**（最快落地，含 4×200G 和 Gen5）。
- **要 58G PAM4 + HBM + CCIX，成熟 16nm 生态** → **XCVU57P**（唯一同时有 58G 和 HBM 的 UltraScale+ 器件）。
- **成本敏感、只要 4 GB HBM** → **XCVU31P**。
- **HPC 集群、软件栈成熟（Vitis）** → **Alveo U55C**。

**容易搞错的点：**

1. 并非所有 UltraScale+ 都带 HBM：`VU9P`、`VU13P`、`VU19P`、`VU23P`、`VU27P`、`VU29P`、`VU3P`、`VU5P`、`VU7P` 都是**无 HBM** 的（带 HBM 的只有上表这些 `P` 系列）。注意 `VU19P` 是最大容量 FPGA，但**没有 HBM**。
2. **Versal 其他系列均无 HBM**：AI Core、AI Edge、Prime、Premium、VP1902 都不含 HBM；唯一带 HBM 的是 **Versal HBM 系列（VHxxxx）**。
3. **Kintex / Artix / Spartan / Zynq** 全系无 HBM。
4. 带宽换算可参考：HBM2 单 stack 约 $1024\ \text{bit} \times 2\ \text{Gb/s} \approx 256\ \text{GB/s}$，双 stack 即约 460 GB/s；Versal 的 HBM2e 双 stack 提升到约 819 GB/s。
5. 开发上 HBM 需要通过 **AXI 端口（32 个）** 或 Versal 的 **NoC** 访问，不是普通 DDR 控制器；Vivado 里用 HBM IP + AXI 交换机。

</details>

---

**一句话总结**：带 HBM 的 Xilinx/AMD 芯片 = **Virtex UltraScale+ HBM 7 款（VU31P、33P、35P、37P、45P、47P、57P，HBM2 460 GB/s）** + **Versal HBM 5 款（VH1522、1542、1582、1742、1782，HBM2e 819 GB/s）**；板级对应 **Alveo V80（32GB/820GB/s）、U55C（16GB/460GB/s）、U50（8GB/316GB/s）**。

数据来源：AMD 官网 Virtex UltraScale+ HBM 产品页、Versal HBM Series 产品页及 Alveo V80、U55C、U50 产品规格页。

---
## 🧑‍💻 User

说来versal带hbm的器件还有吧？停产的老一代US+的确实停产了

---
## 😀 默认助手 (7)

## 结论：严格来说，“Versal HBM 系列”就是 5 颗，加上 Alveo 专用的 **XCV80**，算是 6 个型号

AMD 官方产品页和产品选型指南里，Versal HBM 的可订购器件是：

| 器件 | HBM2e 容量 | 逻辑单元 | LUT | DSP | HBM stack 推断 |
|---|---|---|---|---|---|
| VH1522 | 8 GB | 3,837K | 1.75M | 7,392 | 1 stack（约一半带宽） |
| VH1542 | 16 GB | 3,837K | 1.75M | 7,392 | 2 stack |
| VH1582 | 32 GB | 3,837K | 1.75M | 7,392 | 2 stack（大容量 die） |
| VH1742 | 16 GB | 5,631K | 2.57M | 10,848 | 2 stack |
| VH1782 | 32 GB | 5,631K | 2.57M | 10,848 | 2 stack |
| **XCV80**（Alveo V80 专用） | 32 GB | ≈VH1782 | 2.6M | 10,848 | 2×16 GB，820 GB/s |

- **XCV80 是真的第 6 颗**：AMD 在 Alveo V80 规格页里明确写它就是 “Versal HBM XCV80 adaptive SoC device”，2.6M LUT、10,848 DSP、32 GB HBM2e、820 GB/s，和 VH1782 同级，但它属于只能通过配套板卡获得的器件，不单独销售。
- 除此之外，**再没有第 7 颗**：没有 HBM3/HBM3e 版本的 Versal 公开型号，VP1902、Premium、AI Core、AI Edge 全系无 HBM。
- 一个容易记错的点（我上一轮强调过，这里再明确一次）：**Versal HBM 没有 AI Engine**，它的计算资源是 PL + DSP58 + 处理器子系统，走的是 Versal Premium 的血统。所以“HBM + AI Engine”在 AMD 现役产品里**没有单芯片方案**。

---

## 关于老一代 US+ HBM：你的判断基本成立，但细节有点区别

<details>
<summary>展开看 UltraScale+ HBM 这 7 颗的真实命运</summary>

官网那张表虽然还挂着 7 颗（VU31P/33P/35P/37P/45P/47P/57P），但实际供货历史很短：

- **真正上过板、量产出货的只有 3 颗**：
  - `XCVU37P` → VCU128 评估套件、Alveo U280（8 GB / 460 GB/s）
  - `XCU50`（U50 专用型号，HBM 单 stack）→ 8 GB / 316 GB/s
  - `XCU55`（U55C 专用型号）→ 16 GB / 460 GB/s
- **VU31P / VU33P / VU35P / VU45P / VU47P / VU57P 基本是“纸面器件”**：datasheet 阶段就铺开，实际几乎拿不到货，业内关于“Virtex HBM 系列流产”的说法就是从这里来的。
- 结论：**这条线不会再有新品，也没有后续 roadmap**，16 nm 的 HBM2 方案整体被 Versal HBM（7 nm、HBM2e、819 GB/s、32 GB）取代。现在 U50 / U55C / U280 属于**存量在售**，仍是可用（Vitis 生态成熟、便宜），但已经处于“卖完就完”的状态，新设计不建议再投入。

⚠️ 需要说明的边界：我能核实的是“这条线无新品、器件实际可用性极低”这些**公开事实**；但具体到某颗型号的 **PDN/EOL 编号和最后订购日期**，我没有在 AMD 生命周期数据库里查到可引用的官方公告（AMD 的 product lifecycle 页面在 2026 版站点改版后，我没有找到对应的公开入口）。所以“正式停产公告”这一层，我只能说：**没有查到，但产业现实就是你所说的那个结果**。

</details>

---

## 一个“那到底该用什么”的现实回答

如果你的目标是“HBM + 可编程逻辑做 AI/内存密集型加速”，在 AMD 现役里其实只有三条路：

1. **Versal HBM（VH15x2/VH17x2）自研板卡**：32 GB / 819 GB/s，PL 里自己搭 MAC、DSP 阵列，配合 NoC 搬运数据。
2. **Alveo V80（XCV80）**：买现成 PCIe 卡，32 GB HBM2e / 820 GB/s，4×200G，Gen4x16 或 2×Gen5x8，190 W —— 目前唯一“最新一代 HBM FPGA 卡”。
3. **Alveo U55C / U50**：老一代，便宜、生态稳，但 460/316 GB/s 且靠库存过日子。

如果你的目标是“HBM + AI Engine”，那就要换思路：AI Engine 只在 **AI Core（VC1902/VCK5000）** 上，而它们**没有 HBM**，只能靠 PL 侧 DDR4/LPDDR4 或片上 memory tile —— 这也是为什么很多人最后转向了 V80 或直接上 GPU。
