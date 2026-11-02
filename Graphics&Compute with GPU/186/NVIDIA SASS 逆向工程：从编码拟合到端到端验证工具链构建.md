# NVIDIA SASS 逆向工程：从编码拟合到端到端验证工具链构建

## 🧑‍💻 User

怎么把nvidia 把20/30/40的PTX<->SASS docs逆出来
包括寄存器state还有machinecode映射到具体寄存器的行为

---
## 😀 默认助手 (7)

# 逆向 NVIDIA SASS（Turing/Ampere/Ada）的实操方法

**一句话结论**:SASS「文档」不是读出来的,是**穷举 + 拟合**出来的。因为你手上有一个永远正确的 oracle:`ptxas` + `nvdisasm`。所以正确路径不是找泄漏的 PDF,而是**把 nvdisasm 的输出当成标注数据,用位域拟合 + 位翻转扫描 + 硬件执行验证**把表建出来。20/30/40 三代（`sm_75` / `sm_80`,`sm_86`,`sm_87` / `sm_89`）的编码 90% 一致,方法可以一次写好、每代重跑一遍自动 diff。

---

## 0. 先把「要逆什么」切成三层

| 层 | 内容 | 难度 | 主要手段 |
|---|---|---|---|
| **L1 编码** | mnemonic $\leftrightarrow$ 16 字节位模式,操作数位域,控制位 | 中 | 语料挖掘 + 位翻转,`nvdisasm` 反馈 |
| **L2 lowering** | PTX 序列 $\leftrightarrow$ SASS 序列（谁被编译成什么） | 高 | `ptxas` 差分编译 |
| **L3 状态/行为** | 寄存器/谓词/uniform/SR/barrier/控制位的**语义** | 高 | 微基准 + 真机执行 |

绝大多数人卡住是因为把三层混成一件「读懂 SASS」的事做,结果哪层都不完整。

> **关键顺序**:先挖编码表(L1),再探 lowering(L2),最后验语义(L3)。反过来做会浪费几个月。

---

## 1. 你已有的「半份官方文档」

这几样是合规、免费、且信息量被严重低估的:

- **PTX ISA 手册**(`docs.nvidia.com/cuda/parallel-thread-execution`):每个指令的精确语义、rounding mode、`lane` 语义全在里面。逆 SASS 语义时这是**参考答案**,SASS 只是 PTX 语义 + 硬件约束的 lowering,不要重新发明。
- **`nvdisasm` 本身就是规范的一部分**。它认识哪条指令、打印什么操作数形式、`-gi` 输出什么,就是 NVIDIA 对外的事实标准。
- **`cuobjdump -sass`** 和 **Nsight Compute 的 SASS 视图**:NCU 会额外告诉你每条指令的 stall reason,这是验证控制位语义的硬数据。
- **开源交叉验证**(非常重要,能省你几个月):
  - `envytools` 的 `hw/hwref` + Mesa nouveau 的 `codegen/`——Kepler/Maxwell/Volta 的完整编码表,开源可读。
  - `MaxAs`(Maxwell)、`TuringAs`(Turing)、`CuAssembler`(Volta/Turing/Ampere)——这些项目的 README 和生成脚本本身就是方法论文档。
  - Ghidra / Binary Ninja 的 SASS loader/processor 模块——可以直接拿来当反汇编器的对照。

> ⚠️ 别去逆驱动 `.so` 里的 blob,那是 EULA 明确禁止的。用 `ptxas`/`nvdisasm` 观察**自己编译的产物**是合规的;patch 自己 cubin 的 `.text` 执行处于灰区(但这是学界和上述开源项目的通行做法,自行判断)。

---

## 2. L1:编码表——两步拟合

### 2.1 语料挖掘(bootstrap,贡献 95% 覆盖率)

**核心洞察**:`nvdisasm -gi` 的输出是 $\langle \text{mnemonic}, \text{操作数列表} \rangle$,配合同行的 16 字节 hex,**这就是带标签的监督数据**。你不需要发明任何东西,只需要求解约束。

```bash
nvcc -arch=sm_86 -cubin -o k.cubin k.cu
ptxas -arch=sm_86 -O3 -o k.cubin k.ptx
nvdisasm -c -b sm_86 -gi k.cubin > k.sass     # -gi = 打印每指令的原始编码
nvdisasm -c -b sm_86 -sa k.cubin > k_sa.sass  # -sa = 打印 control info
```

（flag 名随 toolkit 版本略有变化,以 `nvdisasm --help` 为准。）

输出形如:

```
/*0040*/  IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x28] ;   /* 0x000000041c00... */
```

**语料来源优先级**:

1. 你自己 CUDA toolkit 里的全部 `libdevice`/`cublas`/`cudnn`/`cutlass` kernel——`cuobjdump -sass` 批量 dump,一天能拿到几十万个 $\langle\text{mnemonic}, \text{hex}\rangle$ 对。
2. 手写覆盖 kernel,针对语料里没出现的 mnemonic 补。
3. **注意**:有些指令 `ptxas` 从不发出(某些 `LDG`/`LD` 变体、`SETMAXNREG` 之类),只能从第三方 cubin 里找,或者靠 L1 的位翻转去扫。

### 2.2 位域求解(核心算法,20 行代码)

编码基本是**线性位域**:某个字段就是从 bit $o$ 开始的 $w$ 位,值直接等于操作数的整数值。所以对每个 `(mnemonic, 操作数位置)` 收集样本,然后穷举:

```python
def find_field(pairs, val_of, max_o=128, max_w=24):
    """pairs: [(hex128, parsed_operand), ...], 返回所有满足约束的 (offset, width)"""
    for w in range(1, max_w + 1):
        for o in range(0, max_o - w + 1):
            if all(((h >> o) & ((1 << w) - 1)) == val_of(op) for h, op in pairs):
                yield (o, w)
```

对 `R4`、`UR7`、`P2`、`c[0x0][0x160]`、`0x3f`、`gdesc[...]` 分别跑一遍,你就拿到了操作数位域。**唯一约束**是 `val_of()` 要写对(比如 `RZ`、`!P0`、`R4.H1`、`R2.64` 要分别映射到自己的编码假设)。

几条你一定会撞上的坑,提前准备好字段类型:

- **uniform 寄存器**:Turing 起 `UR0..UR63` 和 `R0..R255` 共用同一字段,另有独立标志位区分——所以拟合时要先按「是否 UR」分组,再各自求 offset,最后找那个「分组指示位」。
- **`desc[UR4][...]`**:Ampere+ 的 global memory 用 descriptor 寻址,这是新增的操作数种类,parser 必须支持,否则你会在 30/40 系上缺一大块。
- **`RZ` / `URZ` / `PT`**:通常是保留编码值,不是「寄存器 255」,别猜,用上面的方法测。
- **imm 的高位**:立即数字段可能和 modifier 共用位域(比如 `LOP3` 那个 8-bit truth table 就是 imm 直接当 LUT——这是最好的练手例子,拟合会一次成功,可以拿来验证你的 pipeline 对不对)。

### 2.3 位翻转补洞(L1 的第二半)

语料挖完后,把**每条已知指令的 16 字节逐个翻 bit**,再喂回 `nvdisasm`,看输出怎么变:

- 翻 opcode 区 → `nvdisasm` 打印出别的 mnemonic:**直接枚举出整张 opcode 表**(把 opcode 字段的 $2^n$ 个取值全扫一遍)。
- 翻操作数区 → 操作数编号变化:confirms 2.2。
- 翻到 `nvdisasm` 打印 `.word 0x...` 或报错 → **这就是未公开指令/未定义编码的位置**,价值最高。
- 翻控制位区 → `-sa` 输出里 `stall`/`yield`/`write barrier`/`read barrier`/`wait mask`/`reuse` 字段变化。

工程实现:写个「patch-re-disasm」循环,拿一个模板 cubin,从 `nvdisasm -gi` 解析出 `(vaddr, hex)`,用 `readelf -S` 拿到 `.text` 文件偏移,在 `text_off + vaddr` 处覆盖,写临时文件,再 `nvdisasm`。一次迭代几十毫秒,扫完 128 个 bit × 几千条指令是几小时的事。

**注意**:128-bit 指令必须 16 字节对齐;一条非法编码可能让 `nvdisasm` 整个函数反汇编失败,所以每次只 patch 一条、只解析那一行,并用 NOP 包裹。

---

## 3. L2:PTX ↔ SASS lowering——`ptxas` 差分编译

这里的目标不是编码,而是**语义映射**:「哪些 PTX 会变成这条 SASS」「这条 SASS 对应哪几个 PTX」。

**方法:单指令 kernel + 差分**。

1. 写一个 kernel,体内只有一个 PTX 指令加上下界和上界,参数从 global 读(`volatile`)防止被优化掉。
2. `ptxas -O0 -lineinfo` 编译,`nvdisasm -plr` 反汇编,用行号把 PTX 语句和 SASS 指令对齐。
3. **改动一个 PTX 维度**(`.s32`→`.u32`、`.rn`→`.rz`、`.sat`、`.ftz`、`.wide`、`.cc`),看 SASS 里哪个 bit / 哪个 modifier 变了 → 这就是 modifier 位。

**别做的傻事**:别指望在 PTX 层控制物理寄存器号。`.reg` 声明顺序只是强烈倾向,`ptxas` 的分配器会重映射。物理寄存器只在 SASS 层可控。

**反向(SASS → PTX)** 的实际用途:
- 给反汇编器补 `pcode`/表达式语义(IDA/Ghidra 插件)。
- 做 SASS→PTX 重编译(translation layer 场景,ZLUDA 那条路)。
- **闭环验证**:对你逆出的每条指令,写一个 PTX emitter,验证 $\text{ptxas}(\text{emit}(\text{instr}))$ 会再次产出**同类指令**。能闭环的指令 = 你真正理解了它;不能闭环的 = 还有隐藏的 modifier 没挖到。用这个当覆盖率指标。

---

## 4. 寄存器状态 & machine code → 具体寄存器

### 4.1 枚举架构状态(S2R 扫描)

SASS 有一个 `S2R`/`CS2R`/`S2UR` 指令族,操作数是 **SR 编号**。对 SR 字段做 2.3 的位翻转扫描,把 $0..255$ 全枚举一遍:

- `nvdisasm` 认识的 → `SR_TID.X` / `SR_CTAID.X` / `SR_LANEID` / `SR_CLOCKLO` / `SR_GLOBALTIMERLO` / `SR_VIRTCFG` …(清单直接拿到)。
- `nvdisasm` 打印成 `SR[0x3f]` 或不认识的 → **这是你要靠微基准测语义的那部分**:用 `S2R` 读进寄存器、`STG` 写出去、跑真机看值,反推它是什么计时器/ID/状态。

同样的方法适用于 `BAR`/`BSSY`/`BSYNC`/`WARPSYNC`/`VOTE`/`SHFL`/`DEPBAR`/`LDGDEPBAR` 的操作数字段。

完整的架构可见状态清单(你逆完应该能填满):

- 通用:$R0..R255$、$UR0..UR63$(Turing+)、$RZ$/$URZ$
- 谓词:$P0..P7$、$PT$
- 特殊寄存器:`SR_*` 一族、`barrier` 寄存器(Volta+ 有独立 barrier reg 文件)
- reconvergence 状态:Volta/Turing 是 `BSSY/BSYNC` + 每线程 PC(independent thread scheduling),Pascal 之前是 `SSY/SYNC` + 单一 reconvergence stack——**这是 20 系与更早架构最大的行为差异,逆向 divergence 语义时必须分开建模**。
- 控制位(Turing 每指令 24 bit 左右,在 128-bit 高位):`stall`(等待周期)、`yield`、`write barrier`、`read barrier`、`wait mask`、**operand reuse cache 槽位**(`.reuse`)。

### 4.2 machine code → 寄存器:两个不同的「映射」

这是最容易混淆的地方,**它们是两个独立的逆向对象**:

**(a) `ptxas` 的寄存器分配器(软件层,PTX virtual reg → SASS physical Rn)**

SASS 里的 `Rn` 已经是**物理编号**,分配是 `ptxas` 做的。逆它的方法:

- 构造受控 PTX:变量数 $N$ 递增、构造不同 live range 长度,观察 `ptxas -v` 报的 "Used N registers" 和实际分配的编号顺序。
- 测 live range 结束后的复用策略(是否立即回收、是否偏好低编号)。
- 测 `--maxrregcount` / `.maxnreg` / `__launch_bounds__` 对分配的扰动。
- 你会观察到它是「线性扫描 + 偏好」而不是全局最优着色——把这个启发式描述出来,就是「分配行为文档」。
- 实际约束:Ampere 每线程上限 255 regs,每 SM 64K regs,寄存器数决定 occupancy——分配器的行为直接喂给性能模型。

**(b) 硬件寄存器堆的物理行为(解码器/operand collector 层)**

物理编号 $R_n$ → 哪个 bank、能不能同拍读、reuse cache 怎么命中。逆法:

- **不要指望 NCU**:bank conflict / operand collector stall 没有干净的直接计数指标。用**定时微基准**:构造三源指令,系统地变化三个源寄存器编号,测 IPC,拟合 bank 函数 $b(n)$。
- 已知结论(可当你的 hypothesis 起点):寄存器按偶数/奇数或 $(n \gg 1) \,\&\, 1$ 分 bank;`RZ`/常量 bank 读不冲突;uniform 寄存器不占普通 RF 端口。
- **reuse flag 是另一个维度**:`ptxas` 会在控制位里标记 `.reuse`,让同一操作数在下一拍不必重读 RF。这批位就在你 2.3 扫出来的控制位区里,用 `-sa` 观察 + 执行时间验证。

---

## 5. 20 / 30 / 40 系的具体差异(你重跑时要重点 diff 的地方)

| 代 | arch | 关键差异 |
|---|---|---|
| 20 系 Turing | `sm_75` | uniform datapath 引入(`UR`, `R2UR`/`S2UR`);`BSSY/BSYNC`;`desc` 寻址还没普及 |
| 30 系 Ampere | `sm_80/86/87` | `LDGSTS`(cp.async)、`CP.ASYNC`、`LDSM`、`REDUX`、`ELECT`、`BMSK`;TF32/BF16 的 `HMMA` 新编码;**`desc[URx]` 全局寻址成为默认**;`STG/LDG` 操作数形式变化 |
| 40 系 Ada | `sm_89` | FP8 `QMMA`(E4M3/E5M2)、`LDSM`/`HMMA` 变体扩展;编码基本沿用 `sm_86` 表 |

**结论**:写一套「按 arch 重跑」的脚本,把 Turing 的表当 baseline,对 Ampere/Ada 只 diff 增量。三代全套手推是几周,自动化 diff 是几天。

---

## 6. 工程化与验收(别跳过这步)

1. **Schema 设计**(决定成败):`instr → {bit_ranges: opcode, operands: [{kind, offset, width, scale, signed}], control: {...}, fixed_mask}`。用 `fixed_mask` 显式表达「哪些 bit 必须是什么」,这样非法编码检测和 round-trip 都是免费的。
2. **Round-trip 验收**:你的 encoder 生成 hex → `nvdisasm` 反汇编 → **字符串 100% 匹配**原 mnemonic+操作数。对整个 toolkit 的 cubin 批跑,报告匹配率。这是唯一客观的覆盖率指标。
3. **语义验收**:patch 自己的 cubin(`cuModuleLoadData` 加载),真机执行,和 PTX 版本逐位对比输出。这是**唯一硬标准**——所有编码层面的推断,最后都要过这一关。
4. 用 `envytools` / Mesa nouveau / CuAssembler 做**交叉校验**:你的表和它们不一致的地方,要么是你错了,要么是你发现了新东西(通常是后者,因为你会覆盖 `sm_89`)。

---

<details>
<summary><b>7. 你会踩的坑(展开)</b></summary>

- **`nvdisasm` 有 bug**:它会把合法编码打印成 `.word`。别把它当完全正确的 oracle——所有 `.word` 都要用硬件执行二次确认。
- **`ptxas` 会偷偷插指令**:`-O0` 也插 `IMAD.MOV.U32`、`DEPBAR`、`BSSY`。单指令 kernel 也会被包上 prologue。做差分时必须把 prologue 剔掉。
- **`-O0` 不是关优化**:`ptxas -O0` 仍跑 optimizer 和 scheduler。想彻底关,用 `ptxas -O0 --allow-expensive-optimizations=false`,但仍不完美——所以才有「语料挖掘优先」这个顺序建议。
- **控制位不是 ISA 语义**:`stall`/`yield`/barrier 是**给硬件的 hint**,不是你理解的「指令语义」。硬件可以忽略。别把它们写进语义模型。
- **同一 mnemonic 多个编码**:`MOV` 有 `IMAD.MOV`、`MOV`(真正的)、`FSEL` 等实现,`.reuse`/bank 不同还有变体。拟合时按「助记符 + 操作数形态」分组,别按助记符分组。
- **32-bit vs 64-bit 寄存器对**:`R2.64` 表示 $R2{:}R3$,这会影响 bank 计算和 4.2 的拟合,必须显式建模。
- **不要信「寄存器 R255 = RZ」这种流传说法**,自己测。

</details>

---

## 8. 务实的时间预算

| 目标 | 工作量 |
|---|---|
| 只求「读懂 NCU 里的 SASS」 | 几天,直接用 CuAssembler 的表 |
| 完整 L1 编码表 + 控制位(三代) | 数人月 |
| 有闭环验证的 PTX↔SASS 语义模型 | 半年+ |

如果你只是要「看懂 + 能写微优化」,**别自己从零逆**:用 `CuAssembler` 的表 + `nvdisasm -sa`,把精力放在 L3(寄存器/bank/控制位的实际行为)上——那才是没有现成表、也最有价值的部分。

---

需要我针对某一层往下钻吗?比如:
- 写完整的 patch-re-disasm + 位域拟合脚本(Turing/Ampere 可直接跑)
- 展开 `desc[URx]` 寻址在 Ampere 上的编码与语义逆向
- 展开控制位 / operand reuse cache 的微基准设计

---
## 🧑‍💻 User

评估以下选项
1. 我们需要还原哪一层？
A. 指令二进制编码：opcode、位域、寄存器编号如何编码进 128-bit 指令
B. 编译器寄存器分配：PTX 虚拟寄存器如何映射到 SASS 的 R/UR/P
C. 运行时寄存器状态：每条指令执行前后寄存器值如何变化
D. 微架构状态：register file、bank、scoreboard、warp scheduler 的真实行为
E. 全部：建立端到端的逆向工具链

目标架构
A. Turing：SM75（RTX 20）
B. Ampere：SM80/86（RTX 30 / A100）
C. Ada：SM89（RTX 40）
D. 三代一起做，研究跨架构差异

硬件条件
A. 有对应 NVIDIA GPU，可以执行自定义 SASS/CUDA 微基准
B. 只有现成的 cubin、PTX 或应用二进制
C. 两者都有，希望静态分析和动态实验结合

---
这个问题，本质上是在做 NVIDIA GPU ISA 逆向工程：从 PTX 虚拟 ISA、SASS 机器指令，一直追到寄存器分配、物理寄存器状态、指令编码和硬件执行语义。

对于 Ampere（SM80/86，RTX 30 / A100）、Ada（SM89，RTX 40）和部分 Turing（SM75，RTX 20），我建议不要把目标理解成简单地“反编译 PTX”。真正有价值的是建立一套可以实验验证的 PTX → SASS → binary encoding → register dataflow / machine state 知识库。

这里有一个关键区别：

- PTX 是 NVIDIA 定义的虚拟 ISA，有相对完整的公开语义。
- SASS 是具体 GPU 架构的机器 ISA，指令语义、编码和执行细节需要结合反汇编、公开文档及实验研究。
- 寄存器分配和机器码映射主要由 `ptxas` 完成。PTX 的 `%r1` 不一定对应 SASS 的 `R1`；后者才是某个具体 kernel 中实际分配的物理寄存器编号。
- 寄存器状态还涉及谓词寄存器、特殊寄存器、寄存器生命周期、别名、溢出到 local memory、调用约定，以及指令执行时的读写依赖。这些不能仅凭 opcode 名称推断。

最有效的路线是将公开工具作为基线，再通过受控的编译实验和二进制分析补足 NVIDIA 未公开的部分。

先告诉我如何先建立工具链：公开工具能直接还原哪些东西？
然后复现 PTX → SASS → raw machine code
再回答怎么逆向 machine code 到寄存器和 state？（拆成四个独立的研究问题，而不是试图一次性还原整个 GPU。）
  - 指令编码（encoding）
    从 128-bit SASS instruction 的 raw bits 还原 opcode、目的寄存器、源寄存器、立即数、predicate、修饰位及调度控制字段。验证方法：固定一条指令，每次只改变一个操作数或 modifier，比较 cubin 的 bit-level diff。
  - 寄存器分配（register allocation）
    追踪 PTX virtual register 到 SASS physical register 的变化，识别 register coalescing、live-range splitting、寄存器复用和 spill。验证方法：构造不同生命周期和数据依赖的 PTX 程序，比较 SASS、寄存器总数及 local-memory 使用情况。
  - 运行时状态（machine state）
    不只是 `R0 = x` 这种静态映射，还要分析每个 warp 的 GPR、predicate、uniform register、特殊寄存器，以及分支和同步状态的变化。验证方法：插入可观察的写入、分支、同步和存储操作，结合设备端调试、指令级微基准与输出校验。静态反汇编本身无法读取任意运行时寄存器快照。
  - 执行语义（execution semantics）
    确认指令的读写集合、依赖延迟、吞吐率、执行单元、scoreboard 和调度行为；尤其是 predication、特殊寄存器及异步操作。验证方法：使用依赖链、独立指令链、不同 warp 数和不同调度控制的微基准，区分真实数据依赖、结构冲突与调度开销。

注意寄存器状态需要特别区分的三层：

| 层次    | 你要还原的内容                             | 能否仅凭 SASS 静态确定        |
| ----- | ----------------------------------- | --------------------- |
| 编译期   | `R0`、`R1` 等编号和指令读写集合                | 大部分可以                 |
| 架构状态  | 指令执行前后的寄存器值、predicate 和控制流状态        | 需要结合指令语义与执行路径         |
| 微架构状态 | scoreboard、执行流水线、寄存器文件 bank、调度器内部状态 | 通常需要专门实验，部分细节可能无法直接观测 |

另外，`R0` 这样的编号并不意味着它对应一个全 GPU 共享的寄存器。普通寄存器状态是按线程组织的，warp 中不同 lane 的 `R0` 可以保存不同值。寄存器分配的物理编号与底层寄存器文件的 bank、端口和调度行为也是不同层次的问题。

然后就是如何把最终产物设计成一个可查询、可验证的 ISA 数据库，而不是一本纯手工维护的文档。要求每条逆向结论都应该能追溯到具体的 cubin、raw encoding、测试程序和实验结果。

如果我们的目标是最终做到类似 CPU 上的反汇编器 + 调试器 + 指令语义数据库，能够回答“这条 SASS 的哪个 bit 选择了哪个寄存器，以及执行后每个 lane 的状态发生了什么”，那么应当优先做什么。这比直接从整个 CUDA 驱动或 GPU 固件入手更容易得到可验证的成果。

---
## 🧑‍💻 User

评估以下选项
1. 我们需要还原哪一层？
A. 指令二进制编码：opcode、位域、寄存器编号如何编码进 128-bit 指令
B. 编译器寄存器分配：PTX 虚拟寄存器如何映射到 SASS 的 R/UR/P
C. 运行时寄存器状态：每条指令执行前后寄存器值如何变化
D. 微架构状态：register file、bank、scoreboard、warp scheduler 的真实行为
E. 全部：建立端到端的逆向工具链

目标架构
A. Turing：SM75（RTX 20）
B. Ampere：SM80/86（RTX 30 / A100）
C. Ada：SM89（RTX 40）
D. 三代一起做，研究跨架构差异

硬件条件
A. 有对应 NVIDIA GPU，可以执行自定义 SASS/CUDA 微基准
B. 只有现成的 cubin、PTX 或应用二进制
C. 两者都有，希望静态分析和动态实验结合

先告诉我如何先建立工具链：公开工具能直接还原哪些东西？
然后复现 PTX → SASS → raw machine code
再回答怎么逆向 machine code 到寄存器和 state？（拆成四个独立的研究问题，而不是试图一次性还原整个 GPU。）
  - 指令编码（encoding）
    从 128-bit SASS instruction 的 raw bits 还原 opcode、目的寄存器、源寄存器、立即数、predicate、修饰位及调度控制字段。验证方法：固定一条指令，每次只改变一个操作数或 modifier，比较 cubin 的 bit-level diff。
  - 寄存器分配（register allocation）
    追踪 PTX virtual register 到 SASS physical register 的变化，识别 register coalescing、live-range splitting、寄存器复用和 spill。验证方法：构造不同生命周期和数据依赖的 PTX 程序，比较 SASS、寄存器总数及 local-memory 使用情况。
  - 运行时状态（machine state）
    不只是 `R0 = x` 这种静态映射，还要分析每个 warp 的 GPR、predicate、uniform register、特殊寄存器，以及分支和同步状态的变化。验证方法：插入可观察的写入、分支、同步和存储操作，结合设备端调试、指令级微基准与输出校验。静态反汇编本身无法读取任意运行时寄存器快照。
  - 执行语义（execution semantics）
    确认指令的读写集合、依赖延迟、吞吐率、执行单元、scoreboard 和调度行为；尤其是 predication、特殊寄存器及异步操作。验证方法：使用依赖链、独立指令链、不同 warp 数和不同调度控制的微基准，区分真实数据依赖、结构冲突与调度开销。

注意寄存器状态需要特别区分的三层：

| 层次    | 你要还原的内容                             | 能否仅凭 SASS 静态确定        |
| ----- | ----------------------------------- | --------------------- |
| 编译期   | `R0`、`R1` 等编号和指令读写集合                | 大部分可以                 |
| 架构状态  | 指令执行前后的寄存器值、predicate 和控制流状态        | 需要结合指令语义与执行路径         |
| 微架构状态 | scoreboard、执行流水线、寄存器文件 bank、调度器内部状态 | 通常需要专门实验，部分细节可能无法直接观测 |

另外，`R0` 这样的编号并不意味着它对应一个全 GPU 共享的寄存器。普通寄存器状态是按线程组织的，warp 中不同 lane 的 `R0` 可以保存不同值。寄存器分配的物理编号与底层寄存器文件的 bank、端口和调度行为也是不同层次的问题。

然后就是如何把最终产物设计成一个可查询、可验证的 ISA 数据库，而不是一本纯手工维护的文档。要求每条逆向结论都应该能追溯到具体的 cubin、raw encoding、测试程序和实验结果。

如果我们的目标是最终做到类似 CPU 上的反汇编器 + 调试器 + 指令语义数据库，能够回答“这条 SASS 的哪个 bit 选择了哪个寄存器，以及执行后每个 lane 的状态发生了什么”，那么应当优先做什么。这比直接从整个 CUDA 驱动或 GPU 固件入手更容易得到可验证的成果。

---
## 😀 默认助手 (7)

# 评估 + 工具链 + 数据库设计

## 一、三个选择题的结论

### 层次:选 **A + C**,不要选 E

| 选项 | 评估 |
|---|---|
| **A 指令编码** | ✅ **必做,而且是主干**。唯一 100% 可机械化验证的一层(`nvdisasm` 当 oracle),唯一能规模化的一层,下游所有东西都挂它上面 |
| **B 寄存器分配** | ⚠️ **降级为"经验刻画",不是"逆向"**。你永远拿不到 `ptxas` 的启发式源码,只能测出它的**单调性质**。而且——关键——**建反汇编器/语义库根本不需要它**(见下) |
| **C 运行时状态** | ✅ **这是终点站**,也是整个项目唯一真正稀缺的产出。但正确形态是**参考解释器**,不是"观测" |
| **D 微架构状态** | ⚠️ **只做可证伪的子集**。bank 函数、pipe 归属、wait 延迟可以做;scoreboard 内部结构做了也没法验证 |
| **E 全部** | ❌ **不要**。E 是 A+C 的结果,不是并列的第 5 个选项。把它当目标会毁掉项目的可交付性 |

**为什么 B 要降级**:反汇编器回答"这条指令读哪些寄存器、写哪些寄存器"——这来自**指令的读写集合**(A 的产物 + 语义表),**不是**来自分配算法。"为什么 `%r5` 变成 `R7`"对反汇编器毫无用处。B 的真实价值只有两个:①spill/occupancy 预测;②理解控制流对分配的影响。两者都是可选增量。

**如果你的目标是"反汇编器 + 调试器 + 语义库",那么 B 的优先级排在第 5 位以后。** 这是我对你方案最重要的一条修正。

### 架构:先做 **SM86 单代**,不要三代一起

理由不是"简单",而是**边际成本**:

- **SM86 = Ampere** 是生态最大的一代(RTX 30 系 + A100),CUDA toolkit 对它支持最完整,`ncu` 对它支持最完整,第三方 cubin(cuBLAS/cuDNN/CUTLASS/TensorRT)**语料最丰富**——而语料量直接决定 L1 的自动化程度。
- **SM89 = Ada 相对 SM86 是"增量"**:大多数 opcode 编号不变,新增主要集中在新数据类型(FP8 `QMMA`)和少量内存指令。等你有了 SM86 的表,**SM89 只是重跑一遍 diff,工作量 < 15%**。
- **SM75 = Turing 反而更贵**:uniform datapath 结构不同、`BSSY/BSYNC` 语义不同、CUDA 支持窗口在收窄、出货量最低。**除非客户明确要 20 系,否则放最后**。

> 推荐序:**SM86 → SM89 → SM75**。反过来做会在最贵的一代上花掉 60% 预算,拿到最少的语料。

### 硬件:必须 **C**;若只有 B,请砍掉一半项目

- 选 **C(静态 + 动态)**,而且必须先建**patch-cubin 执行环境**。原因很简单:**patch 过的 cubin 能跑起来,是唯一不可辩驳的证据**。所有编码推断最后都要过这一关,否则你只是在复述 `nvdisasm` 的行为。
- 如果只有 **B(现成 cubin/PTX/应用二进制)**:**停止 C 和 D**。你能做的只有"A 的静态部分 + 语料挖掘",产出是一个**只读反汇编器**。C 层(运行时状态)到此为止——**没有执行权就没有 state**。这必须在项目立项时就写进范围声明,否则会在第 5 个月发现自己做不出可交付物。
- 如果只有 **A**:可以做全套,但要额外注意 `ncu` 需要权限(`NVreg_RestrictProfilingToAdminUsers=0`),以及固定时钟(`nvidia-smi -lgc`)否则 D 层全是噪声。

---

## 二、公开工具基线:精确划出"免费"和"付费"的边界

这是你的第一个问题。我把边界画得比常识更细,因为**有几个工具的能力被严重低估**:

| 工具 | **直接给出** | **给不了** |
|---|---|---|
| PTX ISA 手册 | PTX 全部语义、rounding、lane 语义 | 任何 SASS |
| `ptxas -v` | regs/spill(`lmem`)/stack/cmem 用量 | 分配过程、live range |
| `nvdisasm -gi` | SASS 文本 + **128-bit 原始编码** | **位域划分**(哪几位是 Rd) |
| `nvdisasm -sa` | 控制字(stall/yield/write-barrier/read-barrier/wait-mask/reuse) | 这些位的**硬件语义**(是 hint,可被忽略) |
| `nvdisasm --print-life-ranges` ⭐ | **`ptxas` 自己的寄存器活跃区间** | 为什么这样分配 |
| `readelf -SW` / `cuobjdump -elf` | `.text` 偏移、符号、`.nv.info`、relocation | 语义 |
| `ncu` Source / InstructionStats ⭐ | **每条 SASS 的 stall reason、pipe 归属、predicated-off 比例、执行次数** | **任何寄存器值** |
| `cuda-gdb` | 断点处 per-thread/per-lane 寄存器值 | 任意时刻快照 |
| **`nvdisasm` 二进制里 strings** ⭐ | **完整 mnemonic 全集 + 操作数格式串** | 编码 |

⭐ 三项是"被低估的杠杆",重点说:

**① `nvdisasm --print-life-ranges`(flag 名以 `--help` 为准)**
如果存在,它直接输出 `ptxas` 计算的寄存器活跃区间——**这一项就干掉 B 层一半的工作**,而且不用你写任何数据流分析。先跑 `nvdisasm --help | grep -i life` 确认。

**② `smsp__inst_executed_pipe_*` 系列指标**
R4 里"执行单元/pipe 归属"看起来要靠猜,其实 `ncu` 直接给你:**逐条指令**的 pipe 命中次数(`..._pipe_fma` / `_alu` / `_lsu` / `_xu` / `_adu` / `_uniform` / `_tensor` / `_tex`)。跑一个每类指令各一条的 kernel,`ncu --metrics` 一次就能把整张 pipe 映射表**测**出来,而不是推。

**③ `nvdisasm` 可执行文件里 dump mnemonic 字符串**
不要靠枚举 opcode 空间去发现指令——**先 `strings` 拿到全集**。你因此知道"世界上一共有多少条指令",从而有一个**分母**,覆盖率从第一天起就是可量化的。

**结论:公开工具把 A 的"文本层"和 B 的"结果层"全给了你,把 D 的"可观测量"给了一大半。它们唯一不给的是:A 的位域划分、C 的全部。**——你的项目应该精确地只做这两块。

---

## 三、复现 PTX → SASS → raw machine code

这一步必须是**完全脚本化、可复现**的,因为它是所有证据的入口。

```bash
# 0. 硬性 fixed-point:版本决定编码,必须记录
nvcc --version >  meta.txt
ptxas --version >> meta.txt
nvdisasm --version >> meta.txt
sha256sum k.cu >> meta.txt

# 1. PTX
nvcc -arch=sm_86 -ptx -O0 -o k.ptx k.cu

# 2. cubin(记录全部 flag)
ptxas -arch=sm_86 -O0 -lineinfo -v -o k.cubin k.ptx

# 3. SASS + 原始编码 + 控制字 + 行号
nvdisasm -c -b sm_86 -gi -sa -plr -fun mykernel k.cubin > k.sass

# 4. ELF 布局 -> 拿到 .text 的文件偏移
readelf -SW k.cubin > k.sections.txt
cuobjdump -elf k.cubin > k.elf.txt

# 5. 原始机器码(注意 fatbin 里每个函数一个 .text.<mangled>)
objcopy -O binary --only-section=.text._Z8mykernelv k.cubin text.bin

sha256sum k.cubin >> meta.txt
```

> flag 名在不同 toolkit 版本间有漂移(`-plr` / `--print-line-info` / `--print-life-ranges`),以 `nvdisasm --help` 为准,并把 `--help` 输出也存进 `meta.txt`。

**三份对齐**(这是后续一切的基石):

```python
import re, struct
# 解析: addr -> (bits128, text, control)
pat = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\*(.*?)\*/')
def parse(sass_text):
    out = {}
    for m in pat.finditer(sass_text):
        addr = int(m.group(1), 16)
        text = m.group(2).strip()
        hexgroups = re.findall(r'0x([0-9a-fA-F]+)', m.group(3))
        # 坑: 可能打印成一整段 128-bit,也可能打印成两段 64-bit
        bits = int(''.join(reversed(hexgroups)), 16) if len(hexgroups) > 1 \
               else int(hexgroups[0], 16)
        out[addr] = (bits, text, m.group(3))
    return out
```

**三个必查的坑(每个都能让你浪费一周)**:

1. **字节序/分组**:`-gi` 的 hex 注释可能是 1 个 128-bit,也可能是 2 个 64-bit,顺序还可能和高低位相反。**用一条已知指令做 5 分钟的自检**:从 `text.bin` 在 `addr - func_start` 处切 16 字节,`int.from_bytes(chunk, 'little')`,和注释比对。对不上就换 `'big'` 或调换两个 64-bit 半段。**先对齐再动脑**。
2. **一行 ≠ 一条指令**:`nvdisasm` 会插入 `/* dataflow */` 之类的额外注释行;`-ndf` 可以关掉 dataflow 注释,让解析更干净。
3. **`.text` 不连续**:`cuobjdump -sass` 会按函数分段打印,`addr` 是**函数内偏移**,不是文件偏移。必须先按 symbol 取每个函数的 `st_value/st_size` 建映射,否则你切出来的字节全是错位的。

---

## 四、四个研究问题:对你验证方法的修正 + 可执行协议

你列的四个方向是对的,但**每个的验证方法都需要修正**。逐条:

### R1 指令编码 — 你的方法"能验证但不产生数据"

> 你的方法:固定一条指令,每次只改一个操作数,compare bit-level diff。

**缺陷**:①`ptxas` 不会让你指定"改一个操作数"——它自己重分配寄存器,一次改动往往连带若干 bit 同时变;②它**不产生数据**,只产生一个样本;③300+ mnemonic × 各种操作数形态,手推不可行。

**修正为三段式**:

| 阶段 | 手段 | 覆盖 |
|---|---|---|
| **① 语料挖掘 + 约束求解** | `nvdisasm` 的输出 = **标注数据**。对每个 `(mnemonic, 操作数槽位)` 收集样本,穷举 $(o, w)$ 使 $\text{bits}[o, o+w) \equiv \text{operand}$ | ~95%,自动 |
| **② opcode 空间位翻转枚举** | 对已知指令的 128 bit 逐个翻,喂回 `nvdisasm`,记录 mnemonic 变化 | 补全 + 发现 `.word`(未公开编码) |
| **③ 定向 PTX 扰动** | 只针对 ①② 都够不到的变体 | 剩余 5% |

约束求解就 10 行:

```python
def find_field(pairs, val_of, max_o=128, max_w=24):
    """pairs: [(bits128, parsed_operand)], 返回所有满足的 (offset, width)"""
    for w in range(1, max_w+1):
        for o in range(max_o-w+1):
            if all(((h >> o) & ((1<<w)-1)) == val_of(op) for h, op in pairs):
                yield (o, w)
```

**验收标准(关键)**:区分两个方向——

- **decode**:`bits → text` 与 `nvdisasm` **逐字符 100% 一致**。**这个必须做到 100%**,因为 oracle 就是它。做不到 100% 说明你的解析器有 bug,不是没逆出来。
- **encode**:`text → bits` 在你的 held-out 真实 cubin 语料上 **≥ 99.5% 往返成功**。剩下 0.5% 才是真正的"未解编码"。

### R2 寄存器分配 — 建议降级,原因见第一节

如果你仍要做,正确的形态是**刻画**(characterization)不是**逆向**:

- 可做的:分配顺序偏好、live range 结束后的复用延迟、coalescing 触发条件、spill 阈值、`--maxrregcount` 扰动响应。
- 交付物不是"算法",而是 `predict_regcount(ptx) → (regs, spill_bytes)` 的**模型 + 准确率报告**。
- 验收:held-out PTX 上 regs 误差 $\le 2$、spill = 0/非 0 判断准确率 ≥ 90%。
- 成本:低。价值:**中低**。建议在 C 层之后做。

### R3 运行时状态 — 正确形态是**参考解释器**,不是"观测"

> 你说"静态反汇编本身无法读取任意运行时寄存器快照"——完全正确。但由此得出的不该是"只能靠观测",而是**反解问题**:

**你无法快照,但你可以预测。** 目标产物:

$$\text{interpreter}(SASS, \text{launch config}) \rightarrow \{\text{per-lane } (R,P,UR,PC) \text{ trace}, \text{memory writes}\}$$

**三层验证阶梯**(成本递增,每层独立可交付):

1. **端到端值验证(必做)**:interpreter 跑 held-out kernel,比对**最终 global memory 输出**,要求 bit-exact(含 NaN/denormal 位模式)。这一层不需要任何特权工具。
2. **中间点验证(强)**:写"**write-then-store**"探针 kernel——在感兴趣的指令后插入 `STG` 把当时的寄存器值写出来。interpreter 预测 vs 硬件实际,逐 lane 比对。
3. **逐 lane 状态验证(最硬)**:`cuda-gdb` 断点 + `info registers`,或 `ncu` 的 `smsp__thread_inst_executed_per_inst_executed.ratio`(predicated-off 比例)验证你的 active-mask 模型。

**必须建模的隐藏难点**:Turing/Ampere 的 divergence 不是 Pascal 的 reconvergence stack,而是 **per-thread PC + `BSSY`/`BSYNC` 收敛屏障栈**。你的 interpreter 必须显式维护 `(active_mask, {thread PC}, barrier_stack)`,否则第 2 步一定对不上。

**验收**:对一套 curated kernel 集(含 warp-divergent 分支、`__shfl_sync`、`__syncwarp`、原子操作),**全局输出 bit-exact**;第 2 步在 $\ge 20$ 个程序上逐 lane 一致。

### R4 执行语义 — 必须拆成两张表,永不混

> 你的验证方法(依赖链/独立链/变 warp 数)方向对,但**结论形态错了**。

**必须拆成**:

| | ISA 语义(确定性) | 时序语义(统计性) |
|---|---|---|
| 内容 | 读写集合、值函数、predication、溢出/舍入 | pipe、latency、issue rate、结构冲突 |
| 能否精确 | **能,必须 100%** | **不能,只能是分布 + CI** |
| 表 | `semantics` | `timing` |
| 置信度 | `verified_hw` | `measured`(带 CI) |

混在一起的 DB 会在 6 个月后彻底不可维护。

**时序测量的 5 个必须控制项**:①固定时钟 `nvidia-smi -lgc`(否则全是噪声);②单 warp 一个 block 分离 issue latency 与调度;③控制 register bank 冲突(否则测到的是冲突不是延迟);④控制 `.reuse` 标志(`ptxas` 会插);⑤`ncu --replay-mode kernel` 保证可复现。

**验收**:latency 给出点估计 + 95% CI;与 CUDA C Programming Guide 的 arithmetic throughput 表、`ncu` 的 `smsp__inst_executed_pipe_*` 交叉校验,**并如实报告不一致处**。

### 你漏掉的三个东西(必须加进范围)

| # | 内容 | 为什么必须有 |
|---|---|---|
| **R5** | **控制字/调度位**(stall / yield / write-barrier / read-barrier / wait-mask / reuse) | 它决定你能不能让 patch 过的 SASS **真正跑起来**。这是"`nvdisasm` 能读懂"和"硬件能执行"的**唯一差别**。它不属于 A 也不属于 D,是全项目性价比最高的未映射区 |
| **R6** | **ABI / 调用约定**(param 在 `c[0x0][...]`,stack pointer、callee-saved、grid/block SR 映射) | 没有它,你的调试器**连 kernel 入口的寄存器状态都建立不了** |
| **R7** | **divergence 模型**(active mask + per-thread PC + `BSSY/BSYNC` 栈) | R3 的前置条件,见上 |

---

## 五、把"寄存器状态"细化为 **3 层 × 4 作用域**

你的三层表是对的,但缺了**作用域**这一维——而"`R0` 是什么"的歧义**全部来自作用域**:

| | **per-thread** | **per-warp** ⚠️ | **per-CTA** | **per-SM** |
|---|---|---|---|---|
| **编译期(静态可得)** | GPR $R0..R255$ 编号、谓词 $P0..P7$、rw 集合 | $UR0..UR63$ 编号、`.reuse` 槽位 | — | — |
| **架构状态(需语义+路径)** | GPR 值、谓词值、per-thread PC(Turing+) | **active mask**、**$UR$ 值**、收敛屏障栈 | shared mem、`bar.sync` 计数 | — |
| **微架构(需专门实验)** | RF bank/端口、operand collector | scoreboard、warp scheduler 状态 | — | L1/L2 分区、`BAR` 硬件 |

**最关键的一格**:$UR$ 是 **per-warp** 状态,不是 per-thread。$R0$ 在 32 个 lane 上可以有 32 个值;$UR0$ 在整个 warp 上**只有一个值**。这个区别搞错,整个 C 层模型就废了。

---

## 六、可查询、可验证的 ISA 数据库

### 设计原则(3 条,决定成败)

1. **A / C / D 三张表物理分离**,各自独立的 `confidence` 字段。混表是最大的失败模式。
2. **每条断言都必须能回溯到一个 artifact + 一个实验**。`evidence` 表是必需的,不是可选的。
3. **按 `(arch, toolkit_version)` 分区**。新 ptxas 会改变编码、增加 opcode——你的表必须有版本维度,否则第 3 个月就自相矛盾。

### Schema(SQLite / DuckDB 均可)

```sql
-- 可复现性锚点
CREATE TABLE manifest(
  id INTEGER PRIMARY KEY, arch TEXT, toolkit TEXT,
  ptxas_hash TEXT, nvdisasm_hash TEXT, help_dump TEXT,
  gpu TEXT, driver TEXT, clock_locked INTEGER);

CREATE TABLE cubin(
  id INTEGER PRIMARY KEY, manifest_id INTEGER,
  sha256 TEXT UNIQUE, source TEXT, ptx TEXT, ptxas_flags TEXT);

CREATE TABLE function(
  id INTEGER PRIMARY KEY, cubin_id INTEGER, name TEXT,
  text_off INTEGER, n_instr INTEGER,
  regs_used INTEGER, lmem_bytes INTEGER, stack_bytes INTEGER);

-- ===== A: 编码 =====
CREATE TABLE instr_instance(           -- 每个观测到的实例
  id INTEGER PRIMARY KEY, function_id INTEGER, addr INTEGER,
  bits128 BLOB,                        -- 16 bytes, 已归一化字节序
  mnemonic TEXT, operand_text TEXT, control_text TEXT,
  template_id INTEGER);

CREATE TABLE encoding_template(        -- 一个 mnemonic × 操作数形态
  id INTEGER PRIMARY KEY, arch TEXT, mnemonic TEXT,
  variant_key TEXT,                    -- 操作数形态签名
  fixed_mask BLOB, fixed_value BLOB,
  UNIQUE(arch, mnemonic, variant_key));

CREATE TABLE operand_slot(
  id INTEGER PRIMARY KEY, template_id INTEGER, position INTEGER,
  kind TEXT,        -- gpr|ugpr|pred|imm|cbank|desc|bar|sreg|...
  bit_off INTEGER, bit_width INTEGER,
  encoding TEXT,    -- identity|signed|scaled|bankflag|enum
  semantic TEXT);

CREATE TABLE control_field(
  id INTEGER PRIMARY KEY, arch TEXT, name TEXT,
  bit_off INTEGER, bit_width INTEGER, encoding TEXT,
  confidence TEXT);                    -- 'inferred' 为主, 少 'verified_hw'

-- ===== C: ISA 语义 =====
CREATE TABLE semantics(
  id INTEGER PRIMARY KEY, template_id INTEGER,
  reads TEXT, writes TEXT,             -- json: 槽位列表
  predication TEXT,                    -- @P 如何影响
  value_fn TEXT,                       -- interpreter 钩子名
  ptx_ref TEXT,                        -- PTX ISA 手册锚点
  confidence TEXT);

-- ===== D: 时序(独立表!) =====
CREATE TABLE timing(
  id INTEGER PRIMARY KEY, template_id INTEGER,
  pipe TEXT, latency_cycles REAL, issue_rate REAL,
  ci_low REAL, ci_high REAL,
  method TEXT, confidence TEXT);

-- ===== 证据链 =====
CREATE TABLE evidence(
  id INTEGER PRIMARY KEY, claim_table TEXT, claim_id INTEGER,
  kind TEXT,   -- roundtrip|biflip|ncu_counter|exec_diff|gdb|doc
  artifact_uri TEXT, detail TEXT, confidence TEXT);

CREATE TABLE experiment(
  id INTEGER PRIMARY KEY, purpose TEXT,
  kernel_src TEXT, build_cmd TEXT, run_cmd TEXT,
  raw_output BLOB, result_json TEXT);
```

### 覆盖率仪表盘(这是你每天要看的)

```sql
-- 按真实语料频率加权的缺口 —— 决定下一步做什么
SELECT mnemonic, SUM(freq) AS weight
FROM (SELECT mnemonic, COUNT(*) freq FROM instr_instance GROUP BY mnemonic)
WHERE mnemonic NOT IN (SELECT mnemonic FROM encoding_template)
ORDER BY weight DESC;

-- 往返通过率(唯一客观进度指标)
SELECT arch, COUNT(*) total,
       SUM(template_id IS NOT NULL) AS encoded
FROM instr_instance GROUP BY arch;
```

**这张仪表盘把"开放式逆向"变成"可度量的工程"**——这是整个项目最重要的单一设计决策。

---

## 七、优先级:如果只做一件事

> **先做"往返验证 harness"(encode → `nvdisasm` → 字符串比对)。**
>
> 不是先做 DB schema,不是先做 encoder,不是先逆 opcode。因为**它是唯一给你一个数字的东西**;有了数字,项目才能并行、才能增量、才能判断"够用了"。没有它,你会永远在"感觉还差一点"里打转。

### 阶段计划

| 阶段 | 内容 | 交付物(可验收) | 工期 |
|---|---|---|---|
| **P0** | manifest + 三份对齐 + 字节序自检 + `strings nvdisasm` 拿 mnemonic 全集 + `--help` 存证 | 可复现管线 + **指令全集分母** | 3–5 天 |
| **P1** | 语料挖掘 + 约束求解 + **往返 harness** | decode 文本 100% 匹配;encode ≥ 95%;**覆盖率仪表盘上线** | 2–3 周 |
| **P2** | opcode 空间位翻转 + 控制字拟合 | 完整 opcode 图 + 控制字段表 + **`nvdisasm` 不认识的编码清单** | 3–4 周 |
| **P3** | **patch-cubin 执行环境**:改 `.text` → `cuModuleLoadData` → 跑 | 能"自定义一条指令并成功执行"——**硬闸门** | 3–4 周 |
| **P4** | 参考解释器 + write-then-store 验证 + divergence 模型 | held-out kernel **bit-exact**;逐 lane 一致 | 6–10 周 |
| **P5** | `smsp__inst_executed_pipe_*` 建 pipe 表 + 时序表 + SM89 diff | pipe 映射表(实测)+ latency 分布 + Ada 增量 | 4–6 周 |

**P3 是项目分水岭**:过不去,你只有反汇编器;过去了,你才有调试器和语义库。

---

## 八、明确不要做的四件事

<details>
<summary><b>展开:四个陷阱</b></summary>

1. **不要从 CUDA 驱动 / GPU 固件入手。** 你的判断是对的。理由再加三条:没有 oracle(改一个 bit 没有任何反馈)、没有标签、EULA 明确禁止。而 `ptxas` + `nvdisasm` 免费给你**带标注的数据集**。这条路径的性价比差 2~3 个数量级。

2. **不要试图"逆出" `ptxas` 的分配算法。** 它是启发式,不是可恢复的规范;而且对目标产物无用。

3. **不要试图把 D 层做到"完整"。** scoreboard 深度、RF 端口仲裁这些做了也**无法验证**,只能变成猜测。D 层只收**能测出 CI 的量**。

4. **不要在 P1 之前定 schema 终稿。** 但 **P0 必须定 `manifest` 和 `evidence`**——可追溯性是架构决策,不是实现细节,改起来要重构全库。

</details>

---

## 需要我继续做哪一块?

我可以直接给出可运行代码的方向:

- **P0+P1 完整脚本**:ELF 解析 + 三份对齐 + 约束求解器 + 往返 harness + 覆盖率报表(Ampere 可直接跑)
- **P2 位翻转扫描器**:patch-re-disasm 循环 + 控制字段自动命名
- **P3 patch-cubin 执行框架**:ELF patch + `cuModuleLoadData` + 崩溃/差异检测
- **P4 interpreter 骨架**:128-bit 解码前端 + per-lane state 模型 + `BSSY/BSYNC` 收敛语义

---
## 😀 默认助手 (7)

# 结论:选项评估

先将三个轴的推荐组合确定下来,再进行拆解。

| 轴 | 推荐 | 核心理由 |
|---|---|---|
| **层级** | **A 为主线 → C 的“静态部分” → D 中可量化的部分 → B 并行推进**;E 是**最终产物的形态**,不能当作起点 | A 是唯一提供可重复生成正确结果的基准层;C 是整个工具链的基础盘;D 无法完整观测;B 不在关键路径上 |
| **架构** | **SM86 为主,SM89 / SM75 后置做 bit-diff**;但 schema 从第一天就把 `arch` 作为一等维度 | 这样能同时满足“只做一代”的收敛性和“三代对比”的目标 |
| **硬件** | **C(静态 + 动态都要)**,但“动态”必须是 **patch cubin + 真执行**,而不是只依赖 `ncu` | 没有真执行,你无法得到 SASS 语义的正确性证据,只能得到假设 |

## 为什么不选 “E + D(三代一起)”

端到端的逆向工作中,90% 的困难来自**没有可信的正确性证据**,而不是工作量。三代一起做会把“我这条 SASS 解错了吗”和“这代编码不一样吗”两个问题混在一起,导致你无法定位错误。

正确的做法是**先固定一个异构维度,再固定另一个**:先在 SM86 上把 方法 + 工具链 + 数据库 全部跑通(单架构、端到端、可验证),然后对 SM89 / SM75 只需要重跑脚本 + 人工检查 diff。跨架构差异是**自然产物**,不是第一个里程碑。

同理,A→C→D→B 顺序不是按难度排序,而是按**依赖关系**:

- **A(编码)** 解锁一切:不知道 bit 含义就无法 patch、无法反汇编、无法建立证据链。
- **C 的静态部分(读写集合 + SASS 解释器)** 解锁正确性证据:解释器是你验证 A 和后续一切的唯一可信来源。
- **D(时序)** 独立于 A/B/C,可以在有解释器之后并行推进,并且只做**可量化的子集**。
- **B(寄存器分配)** 完全独立的一条支线,不在反汇编器/调试器的关键路径上;只有当你需要做 occupancy 建模或重编译器时才必须做。

> **一句话**:A 是必须做的,C 中静态部分是必须做的,C 中真正运行时快照部分可裁剪,D 可裁剪为“只做 latency/II/pipe”的表,B 可完全后置。

---

# 1. 工具链基线:公开工具能直接恢复哪些内容

先明确哪些工具能直接给出哪些结果,哪些只是**部分结果**,哪些需要你自己做。

| 工具 | 直接给你 | 不给你(因此是你的研究内容) |
|---|---|---|
| **PTX ISA 手册** | PTX 完整语义、rounding、`lane` 语义、地址空间 | 与 SASS 的对应关系 |
| **`ptxas -v`** | 寄存器数、spill 字节数、每个 kernel 的资源占用 | 分配**规则**(为什么是这个数) |
| **`cuobjdump -sass`** | SASS 文本 | 原始 bits |
| **`nvdisasm -gi`** | SASS 文本 + **每条指令的 16 字节 raw hex** | 位域语义(哪个 bit 是谁) |
| **`nvdisasm -sa`** | 控制位的人类可读展开(stall/yield/barrier/reuse) | 这些位的位置与编码 |
| **`nvdisasm -plr` / `-lineinfo`** | `.loc` 行号,PTX 行 ↔ SASS 指令 | 精确的 lowering 规则 |
| **`nvdisasm` 内置 dataflow**(`-ndf` 可关) | 它自己算的依赖图 | 你可以启用它并与自己的实现交叉验证 |
| **`cuobjdump -elf` / `readelf`** | cubin ELF 结构、`.text` 文件偏移、`.nv.info` 元数据 | 各 `EIATTR` 的含义(部分已知) |
| **`cuda-gdb`(`-G` 编译)** | 断点处读寄存器、`$R5`、`$laneid`、active mask | 全指令级单步(慢到不可用于大 kernel) |
| **Nsight Compute** | 每指令的 stall reason、pipe 利用率、issue 率 | bank conflict、operand collector 内部状态 |
| **`clock64()` / `%globaltimer`** | 精确到周期的延时测量 | 微架构结构 |

**这里有一个被低估的关键点**:`.nv.info` 段里的 `EIATTR_*`(如 `EIATTR_REGCOUNT`、`EIATTR_MAX_THREADS`、`EIATTR_CTAIDZ_USED`、barrier 数量)是`ptxas` 自己写进去的**元数据**,它是做 B(寄存器分配)时最干净的正确性证据来源之一。它比 `ptxas -v` 更接近二进制事实。把 `.nv.info` 解析器列为 P0 工具之一。

---

# 2. 复现 PTX → SASS → raw machine code

## 2.1 四个“真相层级”

你的目标是把这四层对齐,并建立每一层的可验证映射:

```
L0  PTX 文本          ← 有官方语义(参考真值)
L1  SASS 文本         ← nvdisasm,半官方(事实标准)
L2  128-bit raw words ← 完全不公开 ← 你的主战场
L3  cubin ELF 容器    ← 半公开(.text / .nv.info / reloc)
```

`nvdisasm -gi` 直接把 **L1 ↔ L2** 对齐了(文本 + hex 在同一行)。这等价于**免费获得了到 L2 的带标注数据集**,是整个工程能开始的根本原因。

## 2.2 最小可用流水线

```bash
CUDA=12.4   # 记录版本!ptxas 版本变了分配和编码都可能变
nvcc -arch=sm_86 -ptx -o k.ptx k.cu
ptxas -arch=sm_86 -O3 -lineinfo -o k.cubin k.ptx
cuobjdump -elf k.cubin > k.elf.txt        # 拿 .text 偏移、.nv.info
readelf -SW k.cubin > k.sections.txt
nvdisasm -c -b sm_86 -gi -sa -plr k.cubin > k.sass   # L1+L2+control+lineinfo
```

产出四份文件(Kernel / PTX / cubin / anotated SASS),这就是**一条“证据记录”**。

## 2.3 把 L2 拆成可以逐个处理的单元

写一个 parser,把 `k.sass` 变成结构化记录:

```python
{
  "kernel": "add_kernel",
  "vaddr": 0x40,
  "file_off": 0x1A40 + 0x40,        # text_off + vaddr,来自 readelf
  "raw": bytes.fromhex("...32 hex..."),   # 16B
  "text": "IMAD.MOV.U32 R4, RZ, RZ, c[0x0][0x28]",
  "control": {"stall": 1, "yield": 0, "wrtdb": 0, "waitb": 0x0, "reuse": None},
  "loc": (12, "k.cu")
}
```

这个结构体就是你后面所有东西的**原子单元**。数据库的一条记录、一次 patch、一个解释器 step,都作用在它上面。

## 2.4 三个必须现在就固定下来的工程纪律

1. **版本指纹**:每条记录都存 `{ptxas_version, nvdisasm_version, cuda_version, arch}`。不同版本的分配结果不能混用,否则你会得到“不稳定的规律”。
2. **patch-execute 循环是 P0 基础能力**,先于任何逆向工作建好:
   ```python
   def patch_and_run(cubin, vaddr, new16, guard_cycles=10**7):
       off = text_off(cubin) + vaddr
       write(cubin, off, new16)
       fix_nv_info_regcount_if_needed(cubin)   # 改了寄存器用量必须同步
       return launch_with_watchdog(cubin, guard_cycles)  # 超时/看门狗必须有
   ```
   非法编码可能让 GPU hang。**必须**单独一张卡 / MIG 实例 / 硬超时。这是新手最常出的问题:一次操作可能让整台机器挂起,甚至需要重启。
3. **永远不要相信“未公开指令是非法”**:`nvdisasm` 打印 `.word 0x...` 有两种含义——(a) 它不认识,(b) 编码真的非法。区分方法只有真执行。

---

# 3. 逆向 machine code → 寄存器 / state:四个独立研究问题

强调你原文里那个正确的判断:**静态反汇编无法读任意运行时寄存器快照**。所以 C 必须拆成“可静态确定的”和“只能采样的”两半,否则会陷进去。

先把三层状态讲清楚(这是你的表,我补一列“怎么拿到”):

| 层次 | 内容 | 静态可定? | 拿到的方法 |
|---|---|---|---|
| **编译期** | R/UR/P 编号、每指令 def/use、别名 | **是(近似 100%)** | 编码表 + SASS 语义表 → DU 链 |
| **架构状态** | 执行前后的值、predicate、divergence 状态 | **否**,但可符号化 | SASS 解释器(全量)+ `cuda-gdb` 采样 |
| **微架构状态** | scoreboard、RF bank、scheduler FSM | **否**,部分不可观测 | 微基准,**只做能建表的量** |

---

## 3.1 研究问题 1:指令编码(encoding)

**范围**:128-bit → opcode / Rd / Ra / Rb / Rc / imm / predicate / 地址空间修饰 / control bits。

**方法(按性价比排序)**:

1. **语料挖掘 + 约束拟合**(贡献 ~95% 覆盖率)。`nvdisasm` 的输出就是标签,对每个 `(mnemonic, 操作数位置)` 求 `(offset, width)` 使 bit 提取值 == 操作数整数值。128×24×N 的暴力搜索,几秒出结果。
2. **单变量 delta-probing**(补洞):固定一条指令,只改一个操作数/modifier,bit-diff。这正是你原文提的方法 —— 但要注意它**在 PTX 层很难控制**(`ptxas` 会重映射寄存器),所以正确做法是**在 L2 层做**:直接改 16 字节的某个 bit,喂回 `nvdisasm`,看文本变化。你在 L2 层拥有完全控制权。
3. **控制位单独当一个问题**:它在 128-bit 高段,和操作数**不共享**解析路径。用 `-sa` 的展开值做标签,同样拟合。
4. **校准用例**:先拿 `LOP3` 的开(它 imm 字段直接就是 8-bit truth table,拟合必中)——用它验证你的整条拟合 pipeline 是对的,再铺开。

**验收标准(必须量化)**:
- 在语料上 round-trip:`encode(decode(x)) == x` 且 `nvdisasm(encode(parse(text))) == text`,**≥ 99.5%**;
- 剩下 0.5% 逐条列出并归因(`nvdisasm` bug / 未知操作数种类 / 真非法);
- 每条记录带 provenance。

---

## 3.2 研究问题 2:`ptxas` 寄存器分配

**范围**:virtual → physical 映射、coalescing、live-range splitting、reuse、spill。**不是**硬件 bank。

**方法**:
- **值打标(taint/tagging)**:PTX 里给每个虚拟寄存器一个唯一常量签名(如 `%r_k` 恒等于 `base + k`),让信息**穿过优化存活**。这样即使被 coalesce 或重映射,你也能从 SASS 的立即数反推“这个物理寄存器承载的是哪个逻辑值”。这是验证分配器最有力的手段,比读 `ptxas -v` 强得多。
- 受控 live range:用“在 `%r_k` 写入后,在 N 条指令之后再消费”这种模板扫 N,观察物理编号回收时机。
- spill 检测:SASS 里出现 `LDL/STL` + `ptxas -v` 的 local bytes + `.nv.info` 三方交叉。
- 扰动:`--maxrregcount` / `.maxnreg` / `__launch_bounds__`,观察分配策略变化。

**验收**:对合成 PTX 集合,你的模型预测的 register count 与 `ptxas -v` / `EIATTR_REGCOUNT` **完全一致**;对真实 kernel(不受 `launch_bounds` 约束的)预测误差 ≤ 小常数。

> **注意**:这一层的产物是“**启发式描述**,而非算法”。你不可能 100% 还原分配器,也不必要。目标是“可预测到能用来做 occupancy 和 spill 预警”。

---

## 3.3 研究问题 3:运行时状态(machine state)

**这是最容易做错的一题。** 不要试图做“每指令每 lane 的寄存器快照”,那既不可观测也没人需要。拆成三个交付物:

**(a) 静态 DU 链 / def-use(全量,必做)**
每条指令的读写集合是**纯静态事实**,由 3.1 的编码表直接推出。产出“在本 kernel 里,R5 的当前值来自 vaddr 0x1A4 的那条 `IADD3`”这种可查询结果。这是反汇编器/调试器真正要的东西。`nvdisasm` 自己的 dataflow(`-ndf` 那套)可作为交叉验证。

**(b) SASS 解释器(全量,必做的核心)**
这是证明 C 的唯一方式:**写一个能执行整段 cubin `.text` 的解释器**,输入 grid/block,输出 global memory 的 hash。
- 初期只覆盖整数 + 内存 + 控制流;
- 必须处理 predication(guarded write)、`RZ`、`.reuse`(可忽略)、`SHFL/VOTE/REDUX`(warp 级)、`BAR`;
- **Volta+ 的 divergence 必须建模为“每线程一个 PC”**(independent thread scheduling),`BSSY/BSYNC` 是 reconvergence 点。如果你按 Pascal 的单一 reconvergence stack 建模,在 20/30/40 上会错,这是三代共有的关键语义。
- 验收:**整数 kernel 的输出与真机逐字节一致**。这一条过了,你就拥有了完整的、可验证的指令语义库——这比任何文档都值钱。

**(c) 运行时快照采样(裁剪,后置)**
- `cuda-gdb -G` 在断点处读 `$R5`/active mask:适合小规模验证解释器,不适合全量。
- 或者用“显式 `STG` dump”+ 插入观察点,自己构造观测。

---

## 3.4 研究问题 4:执行语义(微架构)

**范围裁剪的原则**:只逆“能填进表的标量”,不逆“内部状态机”。

| 该做 | 不该做 |
|---|---|
| 每指令 latency(周期)、initiation interval | warp scheduler 的 FSM 状态数 |
| 使用哪个执行单元/pipe(`icu`/`fma`/`lsu`/`uniform`) | operand collector 内部队列深度 |
| 是否走 uniform datapath(`UR` vs `R`)的代价差 | register file bank 的物理实现细节(只要 functional model) |
| fixed-latency vs variable-latency(scoreboard 依赖) | —— |
| predication 被执行时的代价 | —— |

**方法**:依赖链(测 latency)vs 独立链(测 II);扫 warp 数;`clock64()` 差值;**用 NCU 的 pipe 利用率做归因**,不用它猜内部状态。

**验收**:一张 `(instr, latency, II, pipe)` 表,重复测量稳定性 $\pm 1$ 周期;并且能用一个简单调度模型预测随机 SASS 块的 IPC 到合理误差内。

> **明确放弃**:RF bank 冲突的精确规则、scheduler 内部状态、operand collector 行为。理由:不可直接观测、对工具链无价值、且高度依赖编译器调度——投进去的每一小时都在降低整体产出。

---

# 4. 数据库设计:可查询 + 可验证

核心原则:**没有 provenance 的结论不能进库;没有测试的结论不能进库。**

<details>
<summary><b>展开:schema 与查询接口草案(点击)</b></summary>

```yaml
# instr: sm86/IMAD.yaml
arch: sm86
mnemonic: IMAD
variants: [IMAD, IMAD.MOV, IMAD.MOV.U32, IMAD.WIDE]
encoding:
  size_bytes: 16
  fixed_bits: {0x0: "0b...", ...}          # 必须匹配的位
  opcode: {offset: 41, width: 9, value: 0x1a3}
  operands:
    - {name: d,   kind: reg,  offset: 0,  width: 8, encode: "gpr"}
    - {name: a,   kind: reg,  offset: 8,  width: 8, encode: "gpr"}
    - {name: imm, kind: imm20, offset: 16, width: 20, sign: signed}
  control:
    stall:   {offset: 105, width: 4}
    predict: {offset: 109, width: 1}
    reuse:   {offset: 110, width: 4}       # 已部分确认
semantics:
  reads:  [a, b, c]
  writes: [d]
  predicate_guarded: true
  pseudo: "d = (P ? (a*b + c) : d)"        # 注意:IMAD.MOV 是乘法恒等
  model: "sim/int_alu.py::imad"
timing:                                     # 来自实验,不是猜测
  latency: 4
  initiation_interval: 1
  pipe: fma
alloc:                                      # 来自 3.2
  emits_when: ["a*b+c", "mov", "mul"]
  ptx_hints: [multiply, mad]
evidence:
  - {kind: encoding, cubin_sha256: "..", ptxas: "12.4.131", test: "enc/0142"}
  - {kind: semantics, test: "sim/golden/int_0031", hw_hash: "..", sim_hash: ".."}
  - {kind: timing,    test: "micro/lat_imad", samples: 5, ci: 0}
status: verified          # draft | observed | verified | disputed
```

**三个接口**:

1. **查询**:`isa enc "IMAD.MOV.U32 R4,RZ,RZ,0x10" --arch sm86` → 32 hex;反向 `isa dec 0x... --arch sm86` → 文本 + operand role。
2. **diff**:`isa diff sm86 sm89 --mnemonic HMMA` → 逐位差异表。这就是“跨架构差异”的自动产出。
3. **审计**:`isa audit --status draft` → 未验证结论清单(强制清零)。

**CI(这是“可验证”的落地)**:
- `round-trip`:全语料,门槛 ≥ 99.5%;
- `sim`:golden kernel 集合,仿真输出 == 真机 hash;
- `timing`:表值漂移 > 1 周期则 fail(可能驱动/版本变了);
- **每个 commit 都要能回答:“你改了哪条结论?它的证据还在吗?”**

</details>

---

# 5. 应当优先做什么(0→1 路线)

按“解锁了什么”排序,而不是按难度:

| 阶段 | 内容 | 完成标志 | 估时 |
|---|---|---|---|
| **P0** | cubin 结构解析(`.text` 偏移 / `.nv.info`)+ **patch-execute 循环 + 看门狗** | 能改一个字节并让 GPU 跑起来 | 数天 |
| **P1** | `nvdisasm -gi/-sa` parser + 语料挖掘 + 位域拟合器 | 前 30 条高频指令的 encoding 表,round-trip ≥ 99% | 1–2 周 |
| **P2** | **整数 + 内存 + 控制流的 SASS 解释器**,真机 hash 验证 | 3 个整数 golden kernel 逐字节一致 | 2–4 周 |
| **P3** | 按**频率**扩表:统计真实 cubin 的 mnemonic 分布,逐个补齐 | 覆盖 **>95% 的静态指令实例** | 持续 |
| **P4** | DU 链 + `cuda-gdb` 采样交叉验证 | 能给任意 vaddr 回答“R5 从哪来” | 1–2 周 |
| **P5** | 时序表(latency/II/pipe) | ±1 周期稳定,IPC 模型可用 | 2–4 周 |
| **P6** | `sm89` / `sm75` bit-diff | 自动 diff 报告 + 增量修补 | 数天(有了 P1–P5) |
| **P7** | 寄存器分配研究(独立支线) | 预测 regcount / spill | 数周 |

**关于 P3 的频率驱动**:真实 kernel 里高频 mnemonic 高度集中,典型是 `IMAD`/`IMAD.MOV.U32`、`MOV`、`IADD3`、`LOP3`、`SHF`、`LEA`/`LEA.HI`、`ISETP`/`FSETP`、`SEL`/`FSEL`、`LDG.E`/`STG.E`、`LDS`/`STS`、`FADD`/`FMUL`/`FFMA`、`S2R`、`ULDC`/`R2UR`/`USHF`(uniform datapath)、`BRA`/`BSSY`/`BSYNC`/`EXIT`、`BAR`、`PRMT`、`I2F`/`F2I`、`HMMA`/`LDSM`。**前 ~50 条覆盖 >95%**,先把这 50 条做到 verified,你的工具就已经可用了。

---

## 最后:对你那句判断的回应

> “这比直接从整个 CUDA 驱动或 GPU 固件入手更容易得到可验证的成果”

**完全正确,而且理由比“容易”更强**:

| 维度 | 驱动/固件逆向 | PTX/SASS/编码层(本方案) |
|---|---|---|
| oracle | 无,行为不可复现 | `ptxas`+`nvdisasm`,确定性、可批量 |
| 验证 | 只能靠“看起来对” | 真机 hash 逐字节对比 |
| 增量 | 全或无 | 每条指令独立,天然可并行 |
| 法律 | EULA 明确禁止 | 用官方工具看自己的产物,合规 |
| 失败代价 | GPU hang / 系统崩 | 一条 `.word`,忽略即可 |

**唯一需要警惕的失败模式**是“过度膨胀”**:想一次覆盖三代全指令集、全微架构细节。请把上面表格里 P0–P2 当作不可跳过的必经路径,并且给自己设一个硬门槛:**在解释器通过三个 golden kernel 之前,不碰 P5/P6/P7。** 那道门一旦过了,后面所有东西都会变得机械而可预测。

---

要不要我下一步直接把 **P0 + P1 的代码**(cubin parser + patch-execute harness + `-gi` 解析 + 位域拟合器,面向 `sm_86`)写出来?
