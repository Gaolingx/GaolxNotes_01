# 树莓派开源核移植FPGA烧录方法

## 🧑‍💻 User

树莓派pico2 rtl开源吗 怎么烧录到fpga运行

---
## 😀 默认助手 (7)

## 结论先说

| 对象 | 是否开源 | 说明 |
|---|---|---|
| **RP2350 芯片（Pico 2 的核心）的 RTL** | ❌ 不开源 | 树莓派从未发布过 RP2040 或 RP2350 的 Verilog/SystemVerilog 源码 |
| **Hazard3 RISC-V 核**（RP2350 里真正在用的那个核） | ✅ 开源，Apache-2.0 | 树莓派工程师 Luke Wren 的仓库，可综合到 FPGA，**这是你唯一能烧进 FPGA 的“Pico 2 血缘 RTL”** |
| Pico 2 开发板原理图、PCB、KiCad 设计文件 | ✅ 开源 | 板级开放硬件，跟芯片 RTL 是两回事 |
| Pico SDK、picotool、pico-examples | ✅ 开源 | 纯软件 |
| RP2350 的 BootROM | ❌ 不开源 | 掩膜 ROM，只有数据手册描述 |
| Cortex-M33 核 | ❌ 不可能开源 | ARM 授权 IP |

所以：**「把 Pico 2 的 RTL 烧到 FPGA 上跑」这件事不存在**，网上说“Pico 2 RTL 开源”的，99% 指的是 Hazard3，或者板子的 KiCad 文件。但“把 RP2350 里的 RISC-V 核跑到 FPGA 上，然后调试、跑 C 程序”是完全可行的，Hazard3 仓库自带 example SoC 和 FPGA 流程。

<details>
<summary><b>为什么整颗芯片的 RTL 逻辑上就不可能开源</b>（点开）</summary>

RP2350 是双架构芯片：两个 Cortex-M33 **或** 两个 Hazard3，启动时二选一。M33 是 ARM 的商用 IP，树莓派只有使用权，没有任何权利公开其 RTL；再加上 USB PHY、PLL、OTP 熔丝、Flash XIP 缓存这些模拟/混合信号部分，即使树莓派想开也开不了。RP2040 同理（M0+）。这也是为什么官方只给数据手册 + 《Hardware design with RP2350》+ 板级文件。

</details>

<details>
<summary><b>Hazard3 到底是什么（配置与能力）</b>（点开）</summary>

- **3 级流水 RV32I / RV32E**，可选 `M A C Zicsr Zba Zbb Zbc Zbs Zbkb Zbkx Zcb Zclsd Zcmp Zilsd`
- 机器/用户模式、PMP（最多 16 区域，NAPOT/TOR）、指令地址触发单元（硬件断点）
- **外部调试**：JTAG DTM 或 APB，配套 OpenOCD + GDB 可用
- 总线是 AHB5（`hazard3_cpu_1port.v` / `2port.v`），带全局 exclusive，支持多核
- **RP2350 用的就是这个核**：官方 README 里给出 RP2350 配置的 CoreMark = **4.15 CoreMark/MHz**；`tb_cxxrtl` 默认配置就和 RP2350 一致
- 仓库里还有：Verilator / CXXRTL 仿真台、riscv-formal 形式验证、CoreMark/Dhrystone/Embench、riscv-tests
- 许可证 Apache-2.0，默认分支 `stable`（最新 release v1.1.x）

</details>

---

## 实操：把 Hazard3 烧到 FPGA 上跑

<details open>
<summary><b>路线 A：用官方 example SoC（最快，半小时能点亮）</b></summary>

官方 `example_soc` 包含：Hazard3（单端口 RV32IMA + debug）、JTAG DTM + Debug Module、**128 kB RAM**、UART、RISC-V platform timer。支持三块板：**iCEBreaker（iCE40 UP5K）**、**ULX3S（ECP5）**、**Arty A7-100T（Artix-7）**。

**1. 克隆（必须递归，子模块里有 SoC 外设和测试代码）**

```bash
git clone --recursive https://github.com/Wren6991/Hazard3.git hazard3
cd hazard3
```

**2. 装工具链**

```bash
# RTL 综合
sudo apt install build-essential clang lld bison flex libreadline-dev gawk \
  tcl-dev libffi-dev git graphviz xdot pkg-config python3 \
  libboost-system-dev libboost-python-dev libboost-filesystem-dev zlib1g-dev

# iCEBreaker 路线：yosys + nextpnr-ice40 + iceprog（icestorm 套件）
# ULX3S 路线：yosys + nextpnr-ecp5 + ujprog
# Arty 路线：Vivado（2025.1 实测可用，免费版支持 A7-100T）

# RISC-V 编译器（32 位！不要用 arm-none-eabi）
git clone https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv/gcc15 \
  --with-arch=rv32imac_zicsr_zifencei_zba_zbb_zbkb_zbs --with-abi=ilp32
make -j$(nproc)
export PATH="$PATH:/opt/riscv/gcc15/bin"   # 需要 riscv32-unknown-elf-*
```

**3. 综合 + 烧 bitstream（三选一）**

```bash
cd example_soc/synth
make -f Icebreaker.mk prog        # iCEBreaker：综合 + iceprog 直接下载
# 或
make -f ULX3S.mk flash            # ULX3S：烧到板载 Flash

# Arty A7（Vivado 流程）
cd ../synth_vivado
make prog
```

这一步烧进去的是 **bitstream（硬件）**，不是程序。

**4. 装 OpenOCD（要 riscv 分支）并连上核**

```bash
git clone https://github.com/riscv/riscv-openocd.git
cd riscv-openocd && ./bootstrap
./configure --enable-remote-bitbang --enable-ftdi --program-prefix=riscv-
make -j$(nproc) && sudo make install

# 回到 Hazard3 目录
riscv-openocd -f example_soc/icebreaker-openocd.cfg   # 或 arty7 / ulx3s 的 cfg
```

成功时会看到 `Examined RISC-V core ... XLEN=32, misa=0x...`，说明调试链路通了。

**5. 编译固件 + 用 GDB 下载运行**

```bash
# 把测试程序编译成 elf（example_soc 没有正式软件树，先借 test/sim 下的工程练手）
cd test/sim/hellow && make       # 产出 tmp/hellow.elf / .bin

riscv32-unknown-elf-gdb
(gdb) set confirm off
(gdb) target extended-remote localhost:3333
(gdb) file test/sim/hellow/tmp/hellow.elf
(gdb) load                       # ← 这就是"下载固件"，走 JTAG 写 RAM
(gdb) compare-sections
(gdb) break _exit
(gdb) run
(gdb) info reg a0                # 退出码
```

串口那边（115200 8N1，波特率看 SoC 里 UART 分频）就能看到打印。

</details>

<details>
<summary><b>路线 B：固件写进 BRAM 初始化文件（不接调试器，开机就跑）</b></summary>

不依赖 OpenOCD 的“一次上电自启动”做法：

```bash
# 1) 编译出裸二进制
riscv32-unknown-elf-objcopy -O binary app.elf app.bin

# 2) 转成 $readmemh 格式（python 一行搞定）
python3 - <<'EOF'
data = open('app.bin','rb').read()
data += b'\0' * (-len(data) % 4)
with open('app.hex','w') as f:
    for i in range(0, len(data), 4):
        f.write(f'{int.from_bytes(data[i:i+4],"little"):08x}\n')
EOF
```

然后在 RAM 模型里加：

```verilog
initial $readmemh("app.hex", mem);
```

重新综合烧 bitstream 即可。缺点是**每次改程序都要重新跑一遍综合布线**（对大 FPGA 几分钟到几十分钟），所以日常开发还是用路线 A 的 JTAG load 更快。

</details>

<details>
<summary><b>先仿真再上板（强烈建议，省一堆调试时间）</b></summary>

Hazard3 自带两套仿真台，接口一模一样：

```bash
# CXXRTL（可出 VCD 波形，需要近期 Yosys + clang-16）
cd test/sim/tb_cxxrtl && make
cd ../../hellow && make       # 会自动调 ../tb_cxxrtl/tb --bin ...
gtkwave tmp/hellow_run.vcd

# Verilator（快，但不支持 VCD）
cd test/sim/tb_verilator && make tb
./tb --bin ../hellow/tmp/hellow.bin
```

`tb_cxxrtl` 还支持 `--port 9824` 配合 OpenOCD 的 `remote-bitbang`，**在仿真里练 GDB 调试**，跟真实 JTAG 行为一致。

</details>

<details>
<summary><b>换成自己的板子（国产 Tang Nano / 安路 / 高云等）要改什么</b></summary>

官方只给上面三块板，换板子本质是四件事：

1. **引脚约束**：照 `example_soc/synth/fpga_icebreaker.pcf` / `fpga_orangecrab_25f.lpf` / `constraints_io.xdc` 写你自己格式的约束（`.cst` / `.lpf` / `.pcf`）。需要：时钟、复位、UART TX/RX，以及 JTAG 的 TCK/TMS/TDI/TDO（如果想调试）。
2. **时钟**：`example_soc/fpga/pll_25_40.v`、`pll_25_50.v` 是现成的 PLL 原件，按你板子晶振改参数，或直接用 IP 核生成。
3. **RAM 容量**：SoC 默认 128 kB，正好塞满 iCE40 UP5K 的 SPRAM。小容量器件要么减 RAM 改 `memmap.ld`，要么用外部 SDRAM（那就要自己加控制器）。
4. **JTAG 接法**（最容易卡住的一步）：
   - Xilinx 7 系列 → `hdl/debug/dtm/hazard3_xilinx7_jtag_dtm.v`（用 `BSCANE2` 把 DTM 挂到芯片 TAP 的 DR 上，能复用板载 FT2232，速度无损）
   - ECP5 → `hazard3_ecp5_jtag_dtm.v`
   - 其它（含国产 FPGA）→ 用 `hazard3_jtag_dtm.v` 引出 4 根普通 IO，外挂一个 FT2232/CMSIS-DAP，在 OpenOCD 里用 `remote_bitbang` 或 `ftdi` 驱动
   - 完全没有调试口也行：走路线 B 的 `$readmemh`

</details>

---

## 想跑「Pico 2 上真正的程序」？这里要泼冷水

RP2350 的 RISC-V 固件依赖一堆 Hazard3 **之外**的东西：BootROM、XIP（QSPI Flash 缓存就地执行）、SIO（核间 FIFO、GPIO 原子操作）、PIO（3 块 12 状态机）、DMA、USB 1.1 PHY、OTP、SHA-256 加速器、Hazard3 的自定义 `Xh3irq` 中断控制器行为……**这些都没有开源 RTL**，也就没法照搬到 FPGA。

所以现实分三档：

| 你的目标 | 可行性 | 做法 |
|---|---|---|
| 跑纯算法/裸机 C（无外设） | ✅ 容易 | Hazard3 + `riscv32-unknown-elf-gcc`，自己写 `_start` 和链接脚本 |
| 跑带 UART/定时器/中断的裸机 | ✅ 可行 | 用 example_soc 现成的 UART + platform timer；中断用 Hazard3 自带 `hazard3_irq_ctrl` |
| 直接 `#include "pico/stdlib.h"` 编译后跑 | ❌ 不行 | Pico SDK 的 RISC-V 目标绑定 RP2350 内存映射和外设寄存器，FPGA 上没有 |
| 想在 FPGA 上“玩 Pico 生态” | 换个思路 | 用 **NEORV32 / VexRiscv / Ibex** 这类带完整 SoC 生态的开源核，或干脆在 FPGA 里塞个软核跑 RTOS |

---

## 几个容易踩的坑

- **别用 `arm-none-eabi-gcc`**，Hazard3 是 RV32，必须 `riscv32-unknown-elf-*`，且 `-march` 要和你在 `hazard3_config.vh` 里开的扩展对齐（`rv32imac_zicsr_zifencei_zba_zbb_zbkb_zbs` 是 RP2350 那套）。
- **Vivado 和 OpenOCD 抢 FTDI**：OpenOCD 连着时 Vivado 下载会失败，先关一头。
- **ULX3S 的 JTAG 是 bit-bang**（FT231X 没有真 JTAG），巨慢，且不能和 UART 同时用；iCEBreaker / Arty 的 FT2232H 才是舒服的调试体验。
- **iCEBreaker 要焊两个跳线**才能把 JTAG 引到 FT2232H，见 `fpga_icebreaker.pcf` 注释。
- **example_soc 没有软件树**，官方原话是“先读源码、改测试程序”，别指望 clone 下来就能 `make` 出个带 libc 的应用；要跑正经程序建议自己加个 linker script + `init.S`（可以照 `test/sim/common/` 抄）。
- **`make -f Icebreaker.mk prog` 里的 `prog` 只是下载 bitstream**，固件还得走 OpenOCD。

**参考入口**

- Hazard3 仓库：`https://github.com/Wren6991/Hazard3`（默认分支 `stable`）
- 手册：`https://wren.wtf/hazard3/doc`，Release 页有 PDF
- example SoC 顶层：`example_soc/soc/example_soc.v`
- RP2350 里 Hazard3 的具体配置：看 RP2350 数据手册相关章节
