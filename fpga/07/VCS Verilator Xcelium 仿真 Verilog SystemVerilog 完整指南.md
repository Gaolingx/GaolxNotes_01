# VCS Verilator Xcelium 仿真 Verilog SystemVerilog 完整指南

## 🧑‍💻 User

vcs、verilator和xcelium 怎么做verilog和systemverilog仿真

---
## 🧑‍💻 User

vcs和xcelium 怎么做verilog和systemverilog仿真

---
## 😀 默认助手 (7)

# VCS / Verilator / Xcelium 做 Verilog & SystemVerilog 仿真

先说结论：这三者**对语言的支持、仿真语义和心理模型完全不同**，尤其 Verilator 的仿真模型和时序语义与 VCS/Xcelium 完全不同。先建立正确预期，再学命令。

---

## 一、先搞清楚三者的定位

| 维度 | VCS (Synopsys) | Verilator | Xcelium (Cadence) |
|---|---|---|---|
| Verilog-2001 | ✅ | ✅（可综合子集） | ✅ |
| SystemVerilog-2012 | ✅ 几乎全部 | ⚠️ 子集 | ✅ 几乎全部 |
| UVM / class / random | ✅ | ❌ | ✅ |
| 值语义 | 4 态（0/1/X/Z） | 默认 **2 态**（X/Z 变 0/1） | 4 态 |
| `#delay` / `fork` | 原生 | 需 `--timing` | 原生 |
| 编译模型 | 编译成 `simv` 可执行文件 | 转成 **C++ 源码** 再编译 | 编译+细化两步 |
| 速度 | 中 | **最快**（常快 5–50x） | 中 |
| 测试平台 | SV / UVM | **C++ / SystemC / cocotb**（SV TB 支持很弱） | SV / UVM / e |
| 波形 | VPD(DVE) / FSDB(Verdi) | VCD / FST | SHM(SimVision) / VCD / EVCD |
| 许可证 | 商业 | Apache-2.0 | 商业 |

> **关键认知**：Verilator 并非“另一个 VCS”。它的做法是把 RTL 转换成 C++ 并编译成可执行文件，因此不能直接运行 UVM。用 Verilator 验证时，通常要把 TB 改成 C++/cocotb，或者用它做 **lint 检查加纯 RTL 快速回归**。

---

## 二、通用四步流程

```
env 准备 → (compile 编译) → (elaborate 细化) → run 仿真 → 波形/覆盖率
```

VCS 把 compile+elaborate 合成一步；Xcelium 可以一步(`xrun`)也可拆成三步；Verilator 是 `verilate → g++`。

---

## 三、VCS

### 3.1 最小可用命令

```bash
# 编译（生成 simv）
vcs -full64 -sverilog -timescale=1ns/1ps \
    -debug_access+all -kdb \
    +define+SIM -f filelist.f -top tb_top \
    -o simv -l compile.log

# 仿真
./simv +UVM_TESTNAME=smoke_test +ntb_random_seed=1 -l run.log
```

`filelist.f` 长这样：

```
+incdir+./tb
+incdir+./rtl/include
./rtl/alu.sv
./rtl/core.sv
./tb/tb_top.sv
```

### 3.2 必背选项

| 选项 | 作用 |
|---|---|
| `-full64` | 编译 64 位版本（新版默认，加上也无害） |
| `-sverilog` | **启用 SystemVerilog 语法**，忘了加就会报语法错 |
| `-timescale=1ns/1ps` | 全局时间精度，可以避免 `$time` 和延时单位混乱 |
| `-debug_access+all` | 允许波形记录、读写层次信号、UCLI 交互调试 |
| `-kdb` | 生成 Verdi 知识库（用于 `verdi -ssf` 反标） |
| `-lca` | 部分高级特性需要（`-kdb`、部分 SV 特性） |
| `-f filelist.f` | 使用文件列表 |
| `-top tb_top` | 明确指定顶层，避免多顶层歧义 |
| `-j 8` | 并行编译 |
| `-assert svaext` | 启用 SVA 扩展 |
| `-ntb_opts uvm-1.2` | 内置 UVM 库（无需手加 UVM 源码） |
| `-cm line+cond+fsm+tgl+branch` | 代码覆盖率 |
| `-Mdir ./csrc` | 增量编译目录（改一个文件只重编它） |
| `-P <novas.tab> <pli.a>` | 链接 Verdi FSDB PLI（新版可用 `-debug_access+all -kdb` 代替） |

### 3.3 波形

**方式 A：VPD + DVE（无需 Verdi）**

在 TB 里：

```systemverilog
initial begin
  $dumpfile("wave.vpd");
  $dumpvars(0, tb_top);   // 0 = 全层次
end
```

```bash
dve -vpd wave.vpd &
```

**方式 B：FSDB + Verdi（主流）**

```systemverilog
initial begin
  $fsdbDumpfile("wave.fsdb");
  $fsdbDumpvars(0, tb_top, "+all");  // +all 记录结构体/数组
  $fsdbDumpMDA();
end
```

```bash
verdi -dbdir simv.daidir -ssf wave.fsdb &
```

> FSDB 需要链接 Verdi PLI。如果编译时报 `$fsdbDumpfile` undefined，就先检查是否加了 `-P` 那一对文件，或者改用 `-debug_access+all -kdb`。

### 3.4 UVM

```bash
vcs -full64 -sverilog -ntb_opts uvm-1.2 \
    -debug_access+all -kdb \
    -f filelist.f -top tb_top -o simv
./simv +UVM_TESTNAME=my_test +UVM_VERBOSITY=UVM_LOW
```

如果已经装了独立 UVM 源码，把 `-ntb_opts uvm-1.2` 换成 `-ntb_opts uvm` 并使用 `+incdir+$UVM_HOME/src $UVM_HOME/src/uvm_pkg.sv`。

### 3.5 覆盖率

```bash
vcs ... -cm line+cond+fsm+tgl+branch -cm_dir ./cov
./simv -cm line+cond+fsm+tgl -cm_name smoke_1
urg -dir ./cov/*.vdb -report urgReport   # 或 verdi -cov -covdir ./cov
```

---

## 四、Verilator

### 4.1 先记住边界（避免浪费时间）

**不支持或受限：**

- ❌ `class`、`rand`、约束随机、UVM
- ❌ `program` 块（大部分）
- ❌ 完整 SVA（`|->` `|=>` 只有子集；`--assert` 可开立即断言）
- ❌ `real`/`shortreal` 运算有限
- ❌ DPI export
- ⚠️ X/Z 默认被优化成 0/1（**这是最大的坑**）
- ⚠️ `#delay`、`fork/join`、`wait fork` 必须加 `--timing`（5.0+）

### 4.2 当 lint 用（零成本、强烈推荐）

```bash
verilator --lint-only -Wall -Wno-DECLFILENAME \
    --top-module core -f filelist.f
```

把它塞进 CI，能提前发现大量端口位宽不匹配、未驱动信号的问题。

### 4.3 一步仿真（Verilator ≥ 5.0，最省心）

前提：TB 用 `initial` + `#delay` + `$finish` 写，**必须** `--timing`。

```bash
verilator --binary --timing --top-module tb_top \
    -Wall -j 4 \
    --trace-fst --trace-structs \
    -f filelist.f \
    -o simv
./obj_dir/simv +plusarg=1
```

| 选项 | 作用 |
|---|---|
| `--binary` | 自动生成 C++ main，一步编译成可执行文件 |
| `--timing` | 支持 `#delay`/`fork`/`wait` |
| `--trace-fst` / `--trace` | FST（推荐，文件小）/ VCD |
| `--trace-structs` | 波形显示结构体成员 |
| `--trace-depth N` | 限制记录深度，防止波形巨大 |
| `--x-assign unique` | 让 X 随机化，暴露未初始化问题（抖出 bug 的实用技巧） |
| `--coverage` | 行/翻转覆盖率（5.x） |
| `-Mdir obj_dir` | 输出目录 |
| `--public-flat-rw` | 允许从 C++ 直接读写内部信号 |

### 4.4 手写 C++ TB（传统/可控方式）

`sim_main.cpp`：

```cpp
#include "Vtb_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char** argv) {
    VerilatedContext* ctx = new VerilatedContext;
    ctx->commandArgs(argc, argv);

    Vtb_top* dut = new Vtb_top{ctx};

    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);          // 99 = 层次深度
    tfp->open("wave.vcd");

    dut->rst_n = 0;
    dut->clk   = 0;

    while (!ctx->gotFinish() && ctx->time() < 100000) {
        dut->clk = !dut->clk;
        dut->eval();
        tfp->dump(ctx->time());
        ctx->timeInc(1);          // 半周期 = 1 个时间单位
        if (ctx->time() > 20) dut->rst_n = 1;
    }

    dut->final();
    tfp->close();
    delete dut;
    delete tfp;
    delete ctx;
    return 0;
}
```

构建：

```bash
verilator --cc --exe --build --top-module tb_top \
    -Mdir obj_dir -j 4 \
    --trace -CFLAGS "-O2" -LDFLAGS "-O2" \
    sim_main.cpp ./rtl/*.v
./obj_dir/Vtb_top
```

**要点**：`#include "V<top>.h"` 的名字由 `--top-module` 决定；`eval()` 之后时间才推进；有多个时钟域时必须自己管理 `timeInc`。

### 4.5 用 cocotb（Python TB，最省力的 SV 替代）

```bash
pip install cocotb
# Makefile: SIM=verilator TOPLEVEL=tb_top VERILOG_SOURCES=...
make
```

适合验证小型模块，写 testbench 的效率远高于 C++，且支持 coroutine 风格的 `await RisingEdge(dut.clk)`。

### 4.6 波形

```bash
gtkwave wave.fst &        # FST
# 或 verilator --trace 生成 VCD，用 gtkwave / dve 打开
```

---

## 五、Xcelium

### 5.1 一步流 `xrun`（日常首选）

```bash
xrun -sv -64bit \
     -timescale 1ns/1ps \
     -f filelist.f -top tb_top \
     -access +rwc \
     -input wave.tcl \
     +UVM_TESTNAME=smoke_test \
     -l run.log
```

| 选项 | 作用 |
|---|---|
| `-sv` | **启用 SV 语法**（不写默认按 Verilog 解析） |
| `-access +rwc` | 允许 read/write/connect 层次信号（波形和 `force` 必需） |
| `-top tb_top` | 顶层单元 |
| `-elaborate` | 只细化不仿真（查 elaboration 错误） |
| `-uvmhome $UVM_HOME` | 指定 UVM 库路径 |
| `-uvm` | 使用工具自带/默认 UVM |
| `-coverage all` | 打开全类型覆盖率 |
| `-mce` | 多核细化（大设计明显加速） |
| `-j 8` | 并行编译 |
| `-gui` | 启动 SimVision 图形界面 |
| `-input file.tcl` | 启动时执行 Tcl 脚本（波形设置的标准方式） |
| `-define SIM` | 宏定义 |
| `-nowarn <code>` | 屏蔽特定告警 |

### 5.2 三步流（增量编译，大项目必备）

```bash
mkdir -p INCA_libs
# 1. 编译
xmvlog -sv -work work -f filelist.f
# 2. 细化
xmelab -work work -timescale 1ns/1ps -access +rwc tb_top
# 3. 仿真
xmsim -work work tb_top +UVM_TESTNAME=smoke_test
```

后续只改了 TB 时，重跑 `xmvlog` + `xmelab` 即可，RTL 不用重编。

### 5.3 波形（SHM + SimVision）

`wave.tcl`：

```tcl
database -open waves -shm -default
probe -create tb_top -all -depth all -shm -database waves
run
exit
```

```bash
xrun -sv -access +rwc -f filelist.f -top tb_top -input wave.tcl
simvision waves.shm &
```

需要 VCD 时用 `-input` 脚本里的 `database -open waves.vcd -vcd`，或在 TB 里 `$dumpfile`/`$dumpvars`。

### 5.4 UVM

```bash
xrun -sv -uvmhome $UVM_HOME -access +rwc \
     -f filelist.f -top tb_top \
     +UVM_TESTNAME=smoke_test +UVM_VERBOSITY=UVM_MEDIUM
```

### 5.5 覆盖率

```bash
xrun -sv -coverage all -f filelist.f -top tb_top
imc -load cov_work/scope/smoke_test &
```

---

## 六、同一份代码在三家工具里跑

<details>
<summary><b>可移植性写法：哪些语法会出问题（点击展开）</b></summary>

**会出问题的写法：**

| 写法 | 问题 | 建议 |
|---|---|---|
| `#0` 延时 | Verilator `--timing` 下语义有偏差 | 用 `#1step` 或改非阻塞 |
| 依赖 X/Z 传播 | Verilator 是 2 态，仿真通过但实际有 X 的病 | 用 `--x-assign unique` 反向探测 |
| `fork...join` 顺序依赖 | 各家调度顺序不同 | 避免依赖执行顺序 |
| `class` / UVM | Verilator 不支持 | TB 层用 `ifdef VERILATOR` 切 C++/cocotb |
| `$display("%m")` | Verilator 支持不完整 | 显式打印层次名 |
| `real` 运算 | Verilator 有限 | 用定点数或 double 转 C++ |
| 隐式位宽转换 | Verilator 更严格 | `-Wall` 全开先过 lint |
| `initial` 里的 `#delay` | 不带 `--timing` 直接被忽略 | 编译加 `--timing` |

**推荐的分支写法：**

```systemverilog
`ifdef VERILATOR
  // 2 态友好的 TB 片段，或用 DPI 调 C++ 激励
`else
  // 完整 SV/UVM TB
`endif
```

Verilator 会自动定义 `VERILATOR` 宏（值 ≥ 5000 时为 5.0+），可以直接用 `ifdef VERILATOR` 判断。

</details>

---

## 七、统一 Makefile（三家都能跑）

```makefile
SIM     ?= vcs
TOP     ?= tb_top
FLIST   ?= filelist.f
INC     ?= +incdir+./tb +incdir+./rtl

.PHONY: all compile run wave clean

all: compile run

# ---------- VCS ----------
ifeq ($(SIM),vcs)
compile:
	vcs -full64 -sverilog -timescale=1ns/1ps \
	    -debug_access+all -kdb -j 8 \
	    $(INC) -f $(FLIST) -top $(TOP) \
	    -o simv -l compile.log
run:
	./simv +UVM_TESTNAME=$(TEST) -l run.log
wave:
	verdi -dbdir simv.daidir -ssf wave.fsdb &
endif

# ---------- Verilator ----------
ifeq ($(SIM),verilator)
compile:
	verilator --binary --timing -Wall -j 4 \
	    --trace-fst --trace-structs \
	    $(INC) -f $(FLIST) --top-module $(TOP) \
	    -Mdir obj_dir -o simv
run:
	./obj_dir/simv +TESTNAME=$(TEST)
wave:
	gtkwave wave.fst &
endif

# ---------- Xcelium ----------
ifeq ($(SIM),xcelium)
compile:
	xrun -sv -64bit -elaborate -timescale 1ns/1ps \
	     -access +rwc -mce -j 8 \
	     $(INC) -f $(FLIST) -top $(TOP) -l compile.log
run:
	xrun -sv -64bit -R -timescale 1ns/1ps \
	     -access +rwc -f $(FLIST) -top $(TOP) \
	     -input wave.tcl +UVM_TESTNAME=$(TEST) -l run.log
wave:
	simvision waves.shm &
endif

clean:
	rm -rf simv* csrc *.vpd *.fsdb *.log obj_dir INCA_libs cov_work *.shm *.fst wave.vcd
```

用法：

```bash
make SIM=vcs       TEST=smoke_test
make SIM=verilator TEST=smoke_test
make SIM=xcelium   TEST=smoke_test
```

---

## 八、常见坑速查

<details>
<summary><b>编译期报错（点击展开）</b></summary>

- **VCS 报 `'typedef' is not supported` 之类** → 忘了 `-sverilog`。
- **VCS 报 timescale 相关告警** → 加 `-timescale=1ns/1ps` 统一全局。
- **Xcelium 把 SV 当 Verilog 解** → 忘了 `-sv`。
- **`$fsdbDumpvars` undefined（VCS）** → Verdi PLI 没链接，补 `-P novas.tab pli.a` 或 `-debug_access+all -kdb`。
- **Verilator 报 `%Error: Unsupported: class`** → 不能用 UVM，改用 C++/cocotb TB。
- **Verilator 报 `Unsupported: delayed assignment`** → 忘了 `--timing`。
- **Verilator 报 `Vtb_top.h not found`** → `--top-module` 名字和文件名/模块名不一致；或忘了 `--cc`/`--exe`/`--build` 三件套。

</details>

<details>
<summary><b>运行时问题（点击展开）</b></summary>

- **Xcelium 波形空 / 层次信号读不到** → 少了 `-access +rwc`。
- **Verilator 仿真“全过”但芯片回来挂** → 2 态掩盖了 X。用 `--x-assign unique` + `--x-initial unique` 提升 X 覆盖率，或者关键设计还是交给 VCS/Xcelium。
- **Verilator 波形文件巨大** → 用 `--trace-fst` 代替 `--trace`，并加 `--trace-depth 3`。
- **多顶层歧义** → `-top` / `--top-module` 一定显式给。
- **Xcelium `INCA_libs` 冲突** → 不同测试用不同 `-xmlibdirname`，或每次 `clean`。
- **覆盖率 merge** → VCS 用 `urg`，Xcelium 用 `imc -load` + merge，Verilator 的 `--coverage` 输出需自行 combine。

</details>

---

## 九、选型建议

| 场景 | 推荐 |
|---|---|
| UVM 验证、回归、覆盖率收敛 | **VCS 或 Xcelium** |
| 大 RTL 快速 smoke / 硅前软件启动（软件仿真） | **Verilator**（配合 C++/DPI 激励） |
| CI 里的 lint 检查与位宽检查 | **Verilator `--lint-only`** |
| 纯 RTL 极端速度需求（如 CPU 模型跑 bootrom） | **Verilator** |
| 混合信号 / 与 Cadence 工具链联动 | **Xcelium** |
| 开源/无许可证 | **Verilator** 或 **Icarus Verilog**（`iverilog -g2012`, 调试用） |

**最实用的组合**：日常用 **VCS/Xcelium 跑 UVM 回归**，同时在 CI 挂一条 **Verilator lint + 纯 RTL smoke**，能在早期发现很多位宽和未驱动信号问题，成本几乎为零。

---

需要的话我可以再展开：**DPI-C 让 Verilator 复用部分 SV TB**、**VCS/Xcelium 的 partition compile 加速**、或**三家工具的 X 传播 / 竞争冒险写法差异**。
