# 使用XCVU13P实现PCIe转I2C UART SPI GPIO Hub设计指南

## 🧑‍💻 User

如何用xcuv13p设计一个pcie转i2c uart spi gpio的hub？

---
## 😀 默认助手 (7)

## 先对齐两件事

**1) 型号**：AMD 的合法订货号是 **XCVU13P**（Virtex UltraScale+ VU13P），不是 "XCUV13P"。它是无 HBM 的 VU+ 里最大一款：3,780K 逻辑单元、12,288 DSP、455 Mb 片上存储、128 个 GTY、832 个 I/O，封装 2104 FCBGA，集成 **PCIe Gen3 x16** 硬核。

**2) 前提**：用一颗几千美元的器件做 I2C/UART/SPI/GPIO 这种低速总线 hub，逻辑资源占比不到 0.3%。所以真正合理的场景只有一种——

> **它是某块已有 VU13P 加速卡/采集卡上的“带内管理通道（in-band management）”，不是独立产品。** 把板上的 EEPROM、温度/电压传感器、时钟芯片（Si5341/Si570）、电源 PMBus、状态 LED、JTAG/复位控制挂到 PCIe BAR 后面，让主机 OS 不拆机就能读板子状态。这正是很多加速卡的做法。

如果目标是“新做一块 PCIe 转多路低速口的板卡”，那就别用 VU13P，直接看第 0 节的替代方案。

---

<details>
<summary><b>0. 如果不是管理通道用途：先算这笔账</b></summary>

| 方案 | 成本量级 | 说明 |
|---|---|---|
| XCVU13P 内部实现 | 器件 $5k–95k | 只在“板上本来就有 VU13P”时成立 |
| Artix‑7 XC7A35T + XDMA(AXI‑Lite 模式) | ~$30 + 配置 Flash | 最省心的自研方案，逻辑够做 4×I2C + 4×UART + 2×SPI + 32×GPIO |
| Lattice ECP5 / CertusPro | 类似 | 小封装，LUT 约 24K 足够 |
| 现成桥片 | — | 只做 UART：XR17V358 是单个芯片的 PCIe 转 8 路 16550；做 I2C/SPI/GPIO 没有通用桥片，需 FPGA |
| 外置扩展 | ~$1 | I2C 走 **TCA9548A** 做 8 路复用（`i2c-mux-pca954x` 有驱动）；GPIO 走 **PCA9555 / MCP23017**（`gpio-pca953x` 有驱动），能省掉大量 FPGA 端口 |

结论：**VU13P 只做 hub 的唯一理由是“它已经在板上，且 I/O 口有余量”。** 下面按这个前提展开。
</details>

---

## 1. 系统架构

```
Host (x86 / ARM64 server)
   │  PCIe Gen3 x8 (或 Gen3 x16)
   ▼
┌──────────────────────── XCVU13P ────────────────────────┐
│  PCIe 硬核 + XDMA (AXI4‑Lite 模式)  ← MSI‑X, 16 路 usr_irq │
│        │ axi_aclk 250 MHz / 64 bit                       │
│        ▼                                                 │
│  axi_clock_converter / register slice                    │
│        ▼                                                 │
│  AXI4‑Lite Interconnect @100 MHz ──► 地址译码 64 KB BAR   │
│   ┌────────┬────────┬────────┬────────┬───────────┐      │
│   │ I2C ×N │ UART×N │ SPI ×N │ GPIO×M │ IRQ 聚合器 │      │
│   │ bit eng│ 16550  │ CPOL/  │ 边沿   │ + 计数器   │      │
│   │ +FIFO  │ +FIFO  │ CPHA   │ 检测   │           │      │
│   └───┬────┴───┬────┴───┬────┴───┬───┴───────────┘      │
└───────┼────────┼────────┼────────┼──────────────────────┘
        ▼        ▼        ▼        ▼
   IOBUF/三态  电平转换 / ESD / 隔离 → 背板连接器、板载器件
```

两个关键点：

- **axi_aclk 是 250 MHz（Gen3 x8、64 位数据通路）**。低速外设不要直接跑在这个时钟上，用 `axi_clock_converter` 降到 100 MHz，位时序计数器会好算得多、时序也好收。
- **hub 的时钟建议独立于数据面**，用 MMCM 从 100 MHz 板载晶振派生，避免有人改数据面时钟时把管理通道一起打断。

---

## 2. IP 选型（照抄这张表就能搭起来）

| 功能 | 推荐 IP | 关键理由 / 坑 |
|---|---|---|
| PCIe 端点 | **XDMA（DMA/Bridge Subsystem for PCIe, PG195）配置成 AXI4‑Lite 模式** | 无 DMA 需求时这是最省事的：BAR → 一个 AXI4‑Lite 主口，且自带你需要的 MSI‑X 和 16 路 `usr_irq_req`。比自己写 AXI‑Lite 从机 + PCIe 硬核（PG213）省两周以上 |
| I2C | **AXI IIC（PG090，免费）** | 支持 7/16 位地址、时钟拉伸、16 字节 FIFO。每种端口一个实例，约 600 LUT。**不支持多主仲裁、不支持 SMBus PEC**，PMBus 的 block read/PEC 要在驱动里算 |
| UART | **自写 16550 兼容核**（不要用 AXI UART Lite） | AXI UART Lite（PG142）的波特率只支持基于 AXI 时钟的有限分频组合，FIFO 只有 16 字节，且无 RTS/CTS。做成 16550 兼容后 Linux/Windows 都免驱（见第 8 节） |
| SPI | **AXI Quad SPI（PG153）** | 用 **enhanced mode**（256 深 FIFO）；legacy mode 的中断语义很难用。只有需要 XIP boot 时才碰 startup mode |
| GPIO | **自写**（别用 AXI GPIO） | AXI GPIO（PG144）只有通道级电平中断，**没有边沿检测、没有按位中断向量**，做“按键/告警/中断线”必然不够用 |
| 中断 | 自写聚合器 + XDMA `usr_irq_req` | 见第 6 节 |

---

## 3. 访问模型：先定这个，再写 RTL

这是整个项目最容易返工的地方。三种选择：

| 模型 | 每次 I2C 事务的 MMIO 次数 | 适用 |
|---|---|---|
| **A. 纯寄存器**（逐字节读写数据寄存器） | 10–50+ 次 | 只做点灯/读 EEPROM，能用但慢。每次 MMIO 跨 PCIe 约 $1\text{–}2\ \mu s$，读一个 256B PMBus 块会到几百 $\mu s$ |
| **B. 描述符环 + 完成队列** | 1 次（写门铃） | 高吞吐轮询、多路并发采集。硬件复杂度最高 |
| **C. 每组端口 = 命令寄存器 + TX/RX FIFO + 门铃** | 1–2 次 | ✅ **推荐**。一次写提交整条事务，硬件跑完发中断 |

采用 C 的话，每个端口给 4 KB 地址空间就够：

| 偏移 | 寄存器 | 说明 |
|---|---|---|
| `0x000` | `ID_VER` | 只读，区分端口类型与版本，驱动用来探测 |
| `0x004` | `CTRL` | bit0 enable，bit1 soft reset，bit2 loopback |
| `0x008` | `STATUS` | busy、NACK/arbitration-lost、TX/RX FIFO 深度 |
| `0x00C` | `IRQ_STAT` | **写 1 清零（W1C）** |
| `0x010` | `IRQ_EN` | 与 `IRQ_STAT` 按位与后送中断聚合器 |
| `0x014` | `TIMING` | I2C 分频 / UART 分数分频 / SPI SCLK 分频 |
| `0x018` | `ARG` | 从机地址、字节数、flags（10 位地址、读写、PEC 使能） |
| `0x01C` | `CMD`（门铃） | **写非零即启动**，值编码 START/STOP/重复起始组合 |
| `0x100` | `TX_FIFO` | 写数据 |
| `0x200` | `RX_FIFO` | 读数据 |
| `0x300` | `FIFO_LEVEL` | 一次性读回 TX/RX 两个计数 |

几条硬性要求：

- **每个功能块 4 KB 对齐**：x86 的 MMU 页是 4 KB，用户态 `mmap` 也只能页对齐。8 个端口 → 留 **64 KB BAR**，别抠成 4 KB 然后同页内做子译码（调试时想疯）。
- BAR 设为 **MEM64 + non‑prefetchable**（有读清副作用）。
- 中断寄存器必须是 **STATUS 与 ENABLE 分离**，且 STATUS 只写 1 清零——否则电平中断会连发到把 CPU 拖死。

---

<details>
<summary><b>4. 各协议实现要点（含 I2C 位引擎代码）</b></summary>

### I2C
- 引脚必须用 `IOBUF`/`IOBUFDS`，**默认 T=1（高阻/输入）**。上电时若误把 SCL 拉低，服务器会认为总线被从机拉伸，整条 SMBus 挂死——这是最容易踩的致命 bug。
- 分频：`TIMING = f_clk / (4 * f_scl) - 1`，用 4 拍做高/低电平相位，标准模式 100 kHz、快速模式 400 kHz、FM+ 1 MHz。
- 每个 SCL 高电平期间**必须采样 SCL 实际电平**，遇到拉伸就停住计数；同时挂一个超时计数器（比如 25 ms）报 `bus_stuck`。
- 状态机：`IDLE → START → ADDR → ACK1 → DATA ×N → (ACK/NACK) → STOP`，重复起始用 `CMD` 的 flag 表达，不要用一个字节编码进数据流。

```verilog
// 单 bit 输出的三态处理（SCL/SDA 同构）
IOBUF iobuf_scl (.IO(scl_pad), .I(1'b0), .O(scl_in), .T(~scl_drive_low));
IOBUF iobuf_sda (.IO(sda_pad), .I(1'b0), .O(sda_in), .T(~sda_drive_low));

// 拉伸检测：只有 SCL 真的读到高才推进相位
always @(posedge clk)
  if (phase_cnt == 0) begin
     if (scl_in) phase_cnt <= 4'd1;      // 未被拉伸，继续
     else        stretch_cnt <= stretch_cnt + 1;   // 被拉伸，冻结
  end
```

### UART
- 16× 过采样 + **分数分频**。举例：$f_{clk}=100\text{ MHz}$、115200 bps 时 $D=\frac{100\text{e}6}{16\times115200}=54.25$，取整 54 的误差是 $+0.46\%$（可接受）；但 921600 bps 时 $D=6.78$，取整 7 的误差是 $-3.1\%$（**会丢帧**）。所以高波特率必须做小数分频，或者干脆把外设时钟提到 200 MHz。
- 收方向做中位采样 + 起始位二次确认，避免毛刺误触发。
- 若需要 RTS/CTS、RS‑485 DE、9 位地址模式，在 16550 寄存器映射里留 `MCR`/`MSR` 的位置。

### SPI
- 用 AXI Quad SPI 时，注意 **SS 线数量**：默认 1 个 SS，多从机要么用外部译码器，要么用 `SPISEL` 扩展。
- Master 模式下 `SCLK` 来自 IP 内部使能门控，**不要在 IP 外面再插组合逻辑**，否则 CPHA 相位会错。
- 想省事可以在 IP 里勾 "Loopback" 做自测，但别指望它验证时序参数，只验证数据通路。

### GPIO
建议做成两套寄存器：`DATA_OUT` / `DATA_IN` / `DIR` / `IRQ_RISE_EN` / `IRQ_FALL_EN` / `IRQ_STAT`(W1C) / `DEBOUNCE_DIV`。去抖用一个小计数器共享给整组，别每引脚一个。

</details>

<details>
<summary><b>5. 时钟域、复位与跨 SLR 布线</b></summary>

- 时钟树：
  - `pcie_refclk_100m`（slot 金手指或板载 HCSL 振荡器）；
  - `axi_aclk` 250 MHz（Gen3 x8，64 bit）→ `axi_clock_converter` → `periph_aclk` 100 MHz；
  - `periph_aclk` 分频出 `i2c_tick`（4×SCL）、`uart_tick`（16×baud）、`spi_sclk`。**全部用同一个 MMCM 输出分频**，避免异步相位导致采样点漂移。
- 复位：用 `proc_sys_reset` 生成，按 `periph_aclk` 起。**必须处理 PCIe hot reset 和 FLR**：主机重启或 VM 复位时，`user_lnk_up` 会掉，此时要把所有外设 FIFO 清空、状态机回 IDLE，否则再枚举时 hub 是有状态的僵尸。
- **VU13P 是多 SLR（SSI）器件**，跨 SLR 走线有额外延迟和 `SLR_CR` 约束。做法：确认 PCIe 硬核在哪个 SLR（看 package pinout），然后给整个 hub 用 **Pblock 约束到同一个 SLR**，把 `axi_clock_converter` 放在边界上。漏掉这条，时序报告会出现莫名其妙的 `SLR Crossing` 违例。
- 不用 `BUFG_GT` 这类高速时钟资源做 hub 的时钟，用普通 `BUFGCE` 就够，省下来给数据面。

</details>

<details>
<summary><b>6. 中断：用 MSI‑X，不要用 INTx</b></summary>

- XDMA 提供 16 路 `usr_irq_req`，每路可独立映射成一个 MSI‑X vector。分配策略：
  1. 每个 I2C 端口 1 路（慢、事件少）；
  2. 每个 UART 端口 1 路（RX 有数据、TX 空、线路状态）、
  3. 每个 SPI 端口 1 路；
  4. GPIO 按“组”分配 2–4 路，千万别“每个引脚一路”——128 个 MSI‑X vector 会让 OS 分配失败，且 `request_irq` 开销巨大。
- 中断服务程序务必 **先读 `IRQ_STAT`、再只 W1C 已处理位、最后重新使能**，顺序错了就是中断风暴。
- 现代服务器基本不给 INTx（legacy pin）路由，**不要留 INTx 后路**。
- 调试：`lspci -vvv` 看 `MSI-X: Enable+ Count=N`；`cat /proc/interrupts` 看每个 vector 的计数是否在用。

</details>

<details>
<summary><b>7. 板级硬件（hub 用不到 VU13P 的带宽，但会死在板子上）</b></summary>

| 项目 | 建议 |
|---|---|
| 电平 | FPGA I/O bank 电压可配 1.2/1.5/1.8 V，低速口通常是 3.3 V。**用一个银行专门做 3.3 V**，别指望内部电平转换 |
| 转换 | 推挽信号（UART/SPI/GPIO）用 TXS0108E/TXB0108（注意 TXS 是弱上拉、速率受限）；I2C 用 **PCA9306**（双向、便宜）或 **ISO7742/ADuM1250** 做隔离 |
| 保护 | 每条对外线加 ESD（TPD4E05U06），串 22–33 Ω 抑制过冲 |
| I2C 上拉 | 2.2 kΩ（400 kHz、总线电容 < 200 pF），1 MHz 用 ≤ 1 kΩ 并核对 $t_r = 0.8473 R C$ |
| 供电 | x8 slot 上限：3.3 V @ 3 A + 12 V @ 2.1 A。VU13P 的 0.85 V 核压大电流走板载 DC‑DC，hub 部分只吃几十 mA，**但整板功耗和散热是要单独立项的** |
| PCIe 边沿 | `PRSNT2#`、`WAKE#`（可不用）、`PERST#` 上拉、`REFCLK` ±100 ppm HCSL |

</details>

<details>
<summary><b>8. Linux 驱动：能借现成的就绝不自写</b></summary>

结构上用一个 **PCI 驱动 + mfd 分发**：probe 时 `pci_enable_device` → `pcim_iomap` → 用 `platform_device_register_full()` 或 `mfd_add_devices()` 注册 4 类子设备，每类子设备填 `platform_data`（基地址偏移、时钟频率、端口号）。

| 子系统 | 捷径 | 备注 |
|---|---|---|
| I2C | **`drivers/i2c/busses/i2c-xiic.c`** | 内核已有 AXI IIC 驱动，只要寄存器布局对齐就能直接用，省掉写 `i2c_algorithm` |
| UART | **做成 16550 兼容** → `serial8250_register_8250_port()` | 直接出 `/dev/ttySx`、支持 console、`setserial`、甚至 `minicom` 免配置。Windows 也能用通用 16550 驱动。**这是投产最快的路径** |
| SPI | **`drivers/spi/spi-xilinx.c`** | AXI Quad SPI 的官方驱动，挂上后用 `spidev` 在用户态操作 |
| GPIO | **`drivers/gpio/gpio-mmio.c`**（compatible `linux,gpio-mmio`） | 支持 set/clear 两组寄存器 + 2 条 IRQ 线。要按位边沿中断就自写 `gpio_chip`（比想简单） |
| 中断 | `pci_alloc_irq_vectors(pdev, 1, 16, PCI_IRQ_MSIX)` + `request_irq` | 注意 MSI‑X 的 affinity，别全压在一个核上 |

其他必做的：
- **FLR / remove 路径**：`pci_enable_device` 的逆序 + 清 FIFO + 停 DMA。虚拟机热迁移/复位后最常见的故障就是这里没做。
- **x86 无设备树**：用 ACPI `_DSD` 或直接 `platform_data`；ARM64 服务器可以走 DT overlay（`/sys/kernel/config/device-tree/overlays/`）。
- 用户态接口：`i2cdetect -l`、`spi-loopback-test`、`gpioinfo`、`sensors`（PMBus 走 `pmbus` 驱动）——**把 hub 造成标准接口，才有生态**。

</details>

<details>
<summary><b>9. 资源预估与验证</b></summary>

- 资源：8 端口（2×I2C + 2×UART + 2×SPI + 1×GPIO(64 位)）+ 中断聚合 ≈ **4–8K LUT、约 20 个 BRAM36**，在 VU13P 上占比 **< 0.3%**；加上 XDMA 也就 1% 出头。功耗几百 mW。
- 验证顺序（别跳）：
  1. AXI VIP + cocotb 对每个外设做寄存器读写；
  2. **内部回环**：SPI MOSI→MISO、UART TX→RX、GPIO 输出→输入、I2C 用两个端口对接并让一个当从机；
  3. 上板：`lspci -vvv` 确认 BAR 大小/类型、MSI‑X 使能；
  4. 用 `i2cdetect`、`spidev_test`、`gpioget/set` 逐口冒烟；
  5. 压力：并发跑 8 路各 10 分钟，看中断有没有丢（对比计数器与软件计数）。

</details>

<details>
<summary><b>10. 五个会真的让你项目返工的坑</b></summary>

1. **VU13P 的比特流太大，PCIe 枚举等不起。** 大器件从 Flash 配置动辄几百毫秒到数秒，而 PCIe 规范要求 `PERST#` 释放后 100 ms 内 config space 可读。**必须用 Tandem PROM / Tandem PCIe（见 PG245）**，或者板上先用 MCAP 部分加载、靠 `Tandem` 把 PCIe 硬核和必要逻辑放在第一阶段比特流里。这一条决定项目能不能过 BIOS 枚举，务必在架构阶段就定。
2. **上电把 I2C 总线拉低 → BMC 挂死。** pad 默认必须三态，配置完成前不许驱动。
3. **BAR 里同页放多个功能**，导致用户态 `mmap` 无法访问单端口 → 返工地址译码。每个功能 4 KB 对齐。
4. **用 AXI UART Lite 做产品**，然后发现波特率不对、没有 FIFO、没有流控。一开始就 16550。
5. **忘了 hot reset / FLR**，在 KVM/VMware 直通下面一重启客户机，hub 就再也不响应。

</details>

---

## 落地顺序（建议 6 周节奏）

1. **第 1 周**：定 BAR/寄存器表 + 中断向量分配（写成一份 spec，驱动和 RTL 同时照它做）。
2. **第 2–3 周**：XDMA(AXI‑Lite) + AXI Interconnect + 1 个 AXI IIC 打通，主机侧能 `ioremap` 读到 `ID_VER`。
3. **第 3–4 周**：16550 UART 核 + `serial8250` 挂上，先出 `/dev/ttyS0`；同时做 SPI/G PIO。
4. **第 5 周**：MSI‑X 全通，`/proc/interrupts` 能看到计数，热复位/FLR 测试通过。
5. **第 6 周**：Tandem 配置方案落地 + 上板长稳测试。
6. 全程并行：板级电平转换/ESD/上拉评审（这一步比 RTL 更容易翻车）。

如果你能告诉我**目标主机系统（x86 裸机 / ARM64 / VM 直通）、要几路、走板内器件还是对外连接器**，我可以把寄存器表和驱动骨架直接写成可编译的初稿。
