# NVIDIA SASS逆向工程：从PTX到微架构状态的可验证知识库构建

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

@联网搜索 评估以下选项
1. 我们需要还原哪一层？
A. 指令二进制编码：opcode、位域、寄存器编号如何编码进 128-bit 指令
B. 编译器寄存器分配：PTX 虚拟寄存器如何映射到 SASS 的 R/UR/P
C. 运行时寄存器状态：每条指令执行前后寄存器值如何变化
D. 微架构状态：register file、bank、scoreboard、warp scheduler 的真实行为
E. 全部：建立端到端的逆向工具链

2. 目标架构
A. Turing：SM75（RTX 20）
B. Ampere：SM80/86（RTX 30 / A100）
C. Ada：SM89（RTX 40）
D. 三代一起做，研究跨架构差异

3. 硬件条件
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

---
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

@联网搜索

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

---
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

网络可用，而且搜到了关键线索：`nvdisasm` 内部有可导出的 ISA description 表，这直接改变了这些选项的可行性判断。让我继续深入查证。

搜到的东西改变了这个项目的成本结构，先说结论再逐条评估。

---

# 第一部分：三个选项的评估

## Q1. 还原哪一层？→ **不要选 E。选「一条指令打穿 A→B→C→D」的纵向切片**

先说为什么 E 是错的：E 是**结果**，不是**方法**。你不可能"先做完 A 再做 B"，因为 A 的字段语义需要 B 的数据来标注（哪个 port 是 Rd、哪个是 Rc，光看位图分不出来），而 B 的行为需要 C/D 的实验来验证。E 会让你在 A 层铺开量级、然后永远不收敛。

逐项判断：

| 选项 | 现在还要不要做 | 真实成本 | 判断依据 |
|---|---|---|---|
| **A 编码** | 要做，但**不再是研究，是提取+验证** | 数天~数周 | `nvdisasm` 自带**加密的分架构 ISA 描述表**，解出来就是 NVIDIA 自己的完整 opcode 表（SM75 有 898 个 class / 1862 条 opcode，SM86 有 999 / 2062，SM89 有 1049 / 2162）。你自己位翻转拟合的价值只剩"交叉验证" |
| **B 寄存器分配** | **最该做**，差异化最大 | 数周~数人月 | 这是 `ptxas` 的**编译器启发式**，不在 nvdisasm 的表里，也不在任何公开文档里。谁都没公开做全 |
| **C 运行时状态** | 做，但**目标要收敛** | 数周 | 「每条指令执行前后寄存器值」是不可达的；可达的是"选定 PC 上的 per-lane snapshot"。工具到了：NVBit 1.7 可以在任意 SASS 指令前后注入代码读寄存器 |
| **D 微架构** | 做**可证伪的子集** | 数周~数月 | bank 函数、依赖延迟、吞吐、pipe 占用可测；scheduler 内部仲裁细节测不出来。要提前划死边界 |
| **E 全部** | ✗ | 半年+，且大概率烂尾 | 见上 |

**推荐切片**：挑 3 条指令——`FFMA`（coupled math）、`LDG.E`（decoupled，双 scoreboard）、`ISETP`（写谓词，13 拍延迟）——把它们**纵向打穿四层**。闭环一旦建立，横向扩展到 1000 条 opcode 是机械劳动；反过来做则永远闭环不了。

---

## Q2. 目标架构？→ **SM86 主线 + SM89 做第一次 diff，Turing 只当回归测试**

关键事实：**9 个 decoder 覆盖 13 个 target**。`SM86 ≡ SM87 ≡ SM88` 字节级相同，`SM89` 独立，`SM75` 独立。

| 选择 | 评估 |
|---|---|
| A Turing SM75 | ⚠️ **不要当主线**。它是编码形态的**异类**：69 个 legacy class 用纯 12 位 opcode（bit 91 不携带信息），还有独立的 `REDIRECTABLE_FP16` 物理队列。先做它，你会误以为"特例是常态"。但它的价值是 `UR` 编码的引入点，SM75↔SM80 的 diff 最干净 |
| B Ampere SM80/86 | ✅ **主线**。一套 decoder 覆盖 3060/3070/3080/3090 + A100(部分是 sm_80，962 class) |
| C Ada SM89 | ✅ **第一个 diff 目标**。class 数最多（1049），但相对 SM86 增量小（FP8 `QMMA`/`OMMA`、`LDSM` 变体） |
| D 三代一起 | ✗ **等你有了单一格式的 DB 再说**。否则你是在三套手工 markdown 之间做 diff，那叫自虐 |

**最终建议**：`sm_86` 为主 → 用 `sm_89` 验证"我的方法能自动 diff 出新代"（这是对你方法论的第一个真实验证）→ `sm_75` 作为泛化回归集。

---

## Q3. 硬件条件？→ **必须 C。B 只能产出"反汇编器 + 假设库"，不能产出 ISA 文档**

分层看：

- **A 层（编码）**：纯静态够用。`nvdisasm` 就是 oracle，你不需要 GPU。
- **B 层（分配）**：纯静态够用。编译 + 反汇编 + `ptxas -v` 的 regcount。
- **C/D 层**：**没有真机就写不出可证伪的结论**。你会无法区分"nvdisasm 这么说"和"硬件这么做"，而这正是本项目的全部价值。

具体硬件建议：一张 **sm_86**（3060/3070/A10 都行）+ 一张 **sm_89**（4060/4090）。Turing 二手可选。A100 不必——消费卡更容易拿，且 sm_86 的 class 数比 sm_80 更全。

---

# 第二部分：先建工具链——公开工具直接给你什么

## 2.1 现状：A 层已经"半免费"了

2025–2026 年这个领域最大的变化：**`nvdisasm` 二进制里打包了加密的分架构指令描述表**（per-arch instruction table），内容包括每个指令 class 的汇编格式、二进制编码、合法性条件、调度属性。解密方案（stream cipher + LZ4）后你能拿到 NVIDIA 自己的数据，而不是猜的。

这意味着：**encoding 层从"研究项目"降级成"数据提取 + 验证项目"**。你的项目重心被迫从 A 挤到 B/C/D——而这恰好是你该做的。

> ⚠️ **一个必须说清的可信度问题**：我查到的两份最详细的中文可直接用的资料——`gh.evko.io/crucible-notes`（PTXAS Reverse Engineering Reference）和 `github.com/cicuvc/nvidia-sass-document`——前者**自己在页面顶部红条标注 "AI-GENERATED REVERSE-ENGINEERING NOTES — BEST-GUESS RECONSTRUCTION, NOT A RELIABLE SOURCE"**，后者是 8 stars 的 AI 辅助项目。
>
> 结论：**把它们当"假设生成器"和"路线图"，绝不当事实源**。它们的最大价值不是结论，而是告诉你「去 nvdisasm 里找什么」。这恰好证明了你自己定的原则是对的：**每条结论必须能追溯到 cubin + raw bits + 实验**。

## 2.2 工具 → 产物 → 边界

| 工具 | 直接给你 | **不给你** |
|---|---|---|
| `ptxas` | PTX→SASS 的 ground truth | 任何解释；`-O0` 也不真关优化 |
| `nvdisasm -c -hex/-gi` | 每条指令的 128-bit 原始编码 + 反汇编文本 | 字段语义；非法编码只打印 `.word` |
| `nvdisasm -sa` | 控制字的人类可读解码（stall/yield/barrier/reuse） | 控制字各字段的**动态**含义 |
| `nvdisasm` 内嵌指令表 | **完整 opcode 表、operand port 位域、legality、调度属性** | 动态行为、regalloc、微架构 |
| `cuobjdump -sass` | 批量 dump 语料 | hex 要配 `nvdisasm -gi` |
| `ptxas -v` | 寄存器数、spill（local memory 字节数） | 分配**策略** |
| Nsight Compute | 指令级 stall reason、pipe 利用率 | operand collector / bank conflict 的干净计数 |
| **NVBit 1.7** | **在任意 SASS 指令前后注入代码，读/写寄存器** | 全量 trace；Linux only；EULA 同 CUDA |
| CUDA-GDB | 设备端断点、读寄存器 | 大规模、自动化 |
| `CuAssembler` / `TuringAs` / `MaxAs` | 现成的编解码表（可作交叉验证） | 覆盖到 Ada |

NVBit 的官方要求（刚从仓库确认）：**SM compute capability ≥ 3.5 且 ≤ 12.1，Linux only，x86_64/aarch64，CUDA ≥ 12.0，NVBit 1.7**。→ 你的 SM75/86/89 全在范围内。这是 C 层的唯一现实工具。

---

# 第三部分：复现 PTX → SASS → raw machine code

## 3.1 命令链

```bash
# 1) CUDA C → PTX
nvcc -arch=compute_86 -ptx -o k.ptx k.cu

# 2) PTX → cubin（这步是 ptxas，是你的 oracle）
ptxas -arch=sm_86 -O3 -lineinfo -v -o k.cubin k.ptx

# 3) cubin → SASS 文本 + 原始 128-bit 编码
nvdisasm -c -b sm_86 -gi -sa -plr k.cubin > k.sass

# 4) 或者从 fatbin 里挖（真实应用）
cuobjdump -sass -arch sm_86 app.bin | head
cuobjdump -elf app.bin              # 看 section 布局
```

`-gi` 输出形如（每行尾部就是原始控制字/编码）：

```
/*0040*/  IMAD.WIDE R4, R2, R3, c[0x0][0x168] ;   /* 0x... */
```

> flag 名随 toolkit 版本变动，落地前用 `nvdisasm --help` / `nvdisasm --version` **锁版本**并把 `sha256(nvdisasm)` 记进你的证据表。

## 3.2 从 cubin 里按字节抠出 128-bit word

cubin 是 ELF。`.text` 是代码段，指令 16 字节对齐，**little-endian**，所以一个指令字是 `hi << 64 | lo`，其中 `lo` 是文件里先出现的 8 字节。**注意用 section file offset（`sh_offset`），不是 vaddr**。

```python
import struct
from elftools.elf.elffile import ELFFile   # pip install pyelftools

def read_text_words(path):
    with open(path, 'rb') as f:
        elf = ELFFile(f)
        sec = elf.get_section_by_name('.text')
        base, data = sec['sh_offset'], sec.data()
    for i in range(0, len(data) - 15, 16):
        lo, hi = struct.unpack_from('<QQ', data, i)
        yield base + i, (hi << 64) | lo         # (file_off, word128)
```

## 3.3 四联表（这是你的最小可用产物）

一定要在第一天就把产出固定成这四列，后面所有事都围绕它：

| vaddr | word128 (hex) | nvdisasm text | decoded fields (JSON) |
|---|---|---|---|

有了它，"decode/encode/explain" 三个查询就都能回答了。

---

# 第四部分：machine code → 寄存器和 state，拆成四个独立问题

## 4.1 指令编码（encoding）

**要还原**：opcode、Rd/Ra/Rb/Rc、UR 形式、谓词、立即数、const bank、修饰位、控制字。

**方法**（按性价比排序）：

1. **提取 nvdisasm 的 ISA 表** → 直接得到字段表。这是 bootstrapping，不是研究。
2. **语料拟合交叉验证**：`nvdisasm` 的文本就是标签，对每个 `(mnemonic, port)` 收集样本，穷举 $\text{offset}$ 与 $\text{width}$，要求 $\text{field} = \text{operand}$ 恒成立。20 行代码，用于**独立确认**你提取的表没被 AI 幻觉污染。
3. **位翻转补洞**：patch 一条指令的某个 bit → 重跑 `nvdisasm` → 观察输出变化。用于找"表里没有但硬件接受"的编码。
4. **形式化求解**：arXiv 上有 F2Asm（2608.20532），把 encoding 建模为 $\mathbb{F}_2$ 上的仿射映射来**学习精确 encoder**，并明确处理"nvdisasm 会把多个不同 128-bit 字反汇编成同一段文本"这个坑。这正是你要的"exact encoder + 可证伪"。

**验证（硬标准）**：`decode(word) == nvdisasm(word)` 在**所有 port** 上一致，且 `encode(decode(w)) == w` 位级 round-trip。

**已知进度（待你验证）**：

<details>
<summary>参考位图——来自 crucible-notes，标注为"AI 生成、best-guess"，仅当假设</summary>

- 字长：**128 bit**，但表头谎报 `WORD_SIZE 64`（所有 arch 的表头都自称 `ARCHITECTURE "Volta"`，是通用头 artifact）
- opcode：13 位 = $\{b_{91}\} \Vert \{b_{11..0}\}$；低 12 位自身是 `3+1+3+5` 四段拼装 $b_{11..9}\Vert b_8 \Vert b_{7..5}\Vert b_{4..0}$
- guard：`Pg@14:12`（P0..PT）+ `Pg_not@15`（取反），**每条指令都有**
- operand region：$\text{bits } 16..91$
- port：`Rd@23:16`、`Ra@31:24`、`Rb@39:32`、`Rc@71:64`、`URd@21:16`（6 位）、`Ra_URb@37:32`、`Ra_URc@69:64`、`Pu@83:81`、`Pnz@89:87`、`cop@86:84`、`SRa@79:72`、`barReg@19:16`
- 寄存器文件：GPR `R0..R254` + `RZ=255`（8 位）；UR `UR0..UR62` + `URZ=63`（6 位，塞在 GPR 字节槽的低 6 位）；谓词 `P0..P6` + `PT=7`；convergence barrier `B0..B15`（15 = 无）
- 立即数：`Sb_offset@53:40`（14 位）、`Ra_offset@63:32`（32 位）；const bank `Sb_bank@58:54`（5 位）+ offset；分支 PC 相对 **scale 4**
- 控制字 $\text{bits } 102..125$：`req_bit_set@116:121`（6 位等待掩码）、`src_rel_sb@113:115`、`dst_wr_sb@110:112`（7 = none）、`opex@{122:125}+{105:109}`、`pm_pred@102:103`（**SM90 才有**）
- $\text{opex} = \text{batch\_t} \ll 5 \mid \text{usched\_info}$；`usched_info`：0 = DRAIN(yield)，1–15 = stall n 且结束 co-issue group，17–27 = stall $n-16$ 不结束
- **坑**：目标谓词**直接编码**（`ISETP.GE P0` → raw 0），`7−N` 补码**只**适用于 load/store 的**源** guard 槽（`66:64`，取反在 67）
- **坑**：同一物理 bit 在不同指令族里角色不同（bit 90 在 `IMAD` 是 `input_reg_sz_32_dist`，在 `BSSY` 是 `Pp@not`），**不存在全局 bit→角色映射**

</details>

## 4.2 寄存器分配（`ptxas` 的虚拟→物理映射）

**这是你差异化最大的一块**，因为没有公开完成品。

**要还原**：live range 起点、coalescing、复用策略、spill 触发条件、`--maxrregcount` 的扰动、bank 偏好、以及 `.reuse` 标记的生成规则。

**方法：受控 PTX 探针矩阵**

| 探针维度 | 变什么 | 观察什么 |
|---|---|---|
| 变量数 $N$ | 1→64 | regcount 阶梯、编号分配顺序 |
| live range 长度 | 链式 vs 宽并行 | 何时开始复用同一 Rn |
| 依赖形状 | 链 / 树 / 菱形 | coalescing 是否发生 |
| 强制压力 | 人为拉长 live range | spill 阈值、spill 到 local 的编码（`STL`/`LDL`） |

```cuda
// 探针模板：单指令 kernel + 可控 live range
__global__ void probe(const float* __restrict__ in, float* out, int n) {
    // 只让 k 个中间值同时 live，其余强制长链
}
```

**验证**：
- `ptxas -v` 的 "Used N registers" 与 `nvdisasm` 里出现过的最大 `Rn` 一致。
- 你的模型能**预测**未见过的 PTX 的 regcount（±0）和是否 spill。这是可证伪的。
- 交叉验证：`RZ` 在所有 arch 上是否恒 0（用 NVBit 实测，别信文档）。

**陷阱**：`ptxas -O0` 仍跑 optimizer 和 scheduler；不要在 PTX 层试图控制物理编号。

## 4.3 运行时状态（machine state）

**目标必须收敛**：不是"每条指令的完整寄存器快照"（数据量爆炸且大部分无用），而是：

> 在**选定 PC** 上，对给定 warp 的 32 个 lane，读出**指定的一组寄存器/谓词/SR**，并与你的静态 dataflow 预测比对。

**工具**：NVBit（唯一现实选择，可在任意 SASS 指令前后注入、可读寄存器）；退路是 patch cubin + `cuModuleLoadData` + 自插 `STG` 侧信道。

**分层要分清**（你表格里的三层是对的，我加一列"验证成本"）：

| 层 | 内容 | 静态可定？ | 验证成本 |
|---|---|---|---|
| 编译期 | Rn 编号、指令读写集合 | ✅ 大部分 | 极低 |
| 架构状态 | 执行前后的值、谓词、控制流（`BSSY/BSYNC`） | ❌ 需执行路径 | 中 |
| 微架构状态 | scoreboard 计数、bank、operand collector、scheduler 内部 | ❌ 部分**不可观测** | 高 |

**关键认知**：`R0` 不是"全 GPU 共享的寄存器"。它是 **per-thread** 的，warp 里 32 个 lane 的 `R0` 各存不同值；物理上它在 4 个 sub-partition 之一的 16K 寄存器片里，再由 lane 索引 $+$ 寄存器号定位。这个"线程局部 vs 物理位置"的区别必须写进你的 DB 模型，否则后面全错。

## 4.4 执行语义（execution semantics）

**要还原**：读写集合、依赖延迟、吞吐、执行单元（pipe）、scoreboard 用法、predication 行为、异步操作语义。

**方法：微基准三件套**

1. **依赖链**：同指令串成纯依赖链，测 IPC → 得 producer→consumer 延迟。
2. **独立链**：插入 $k$ 条独立指令，扫 $k$ → 得吞吐（每拍能发几条）。
3. **变 warp 数 / 变 `.reuse` / 变 batch_t** → 区分真数据依赖、结构冲突、调度开销。

**验证**：把你的数字与 Nsight Compute 的 stall reason 分布对上；把 `req_bit_set` 与 producer 的 `dst_wr_sb` 配对，要求 100% 配得上。

**已知（待验证）**：

<details>
<summary>微架构数值——来自 crucible-notes，同样是待验证的假设</summary>

| 资源 | 值 |
|---|---|
| 寄存器文件 | **65,536 × 32-bit / SM**（256 KB），**16,384 / sub-partition** |
| 寄存器 bank | **2 个（偶/奇）**，Volta–Ampere；Kepler–Pascal 是 4 个 |
| sub-partition（scheduler） | 4 / SM |
| warp / SM | 64（Volta, sm_80）· 48（sm_86）· 32（Turing） |
| 依赖 scoreboard | **6 / warp**，SM75→SM121 不变 |
| 每线程最大寄存器 | 255，分配粒度 8 |
| ISSUE_SLOTS | **全部 arch = 1**（双发射靠 `batch_t` 在指令里表达，不是硬件 superscalar） |

观测到的依赖延迟：`IADD3→IADD3` = 4；跨 pipe（`IADD3→IMAD`）= 5；`ISETP→`谓词消费者 = **13**；`HMMA.16816→`消费者 = 11。

**这个"控制字从 Volta 到 Blackwell 字节不变"的结论如果成立，对你极其有利**：你只需把调度模型做一次，三代直接复用。

</details>

---

# 第五部分：怎么把产物做成"可查询、可验证"的 ISA 数据库

## 5.1 核心设计：claim 与 evidence 分离

这是你"每条结论可追溯"要求的唯一正确实现方式。**不要用 markdown 当数据库**——markdown 无法 diff、无法 join、无法做 CI。

```sql
-- 事实层（无观点）
CREATE TABLE cubin (
  sha256 TEXT PRIMARY KEY, path TEXT, toolchain TEXT, arch TEXT,
  ptxas_sha TEXT, nvdisasm_ver TEXT, cmdline TEXT, created TEXT);

CREATE TABLE instr_word (
  cubin TEXT, file_off INTEGER, vaddr INTEGER, word128 TEXT,  -- 32 hex chars
  PRIMARY KEY (cubin, file_off));

-- 知识层（有观点 + 有置信度）
CREATE TABLE claim (
  id TEXT PRIMARY KEY, arch TEXT, subject TEXT, predicate TEXT, obj TEXT,
  status TEXT CHECK(status IN ('observed','inferred','unverified','refuted')),
  confidence REAL, note TEXT);

-- 溯源层（把 claim 钉到证据上）
CREATE TABLE evidence (
  claim_id TEXT, kind TEXT,              -- 'nvdisasm_table'|'cubin_word'|'experiment'|'microbench'
  ref TEXT,                              -- cubin sha / tool ver / experiment id
  excerpt TEXT, checked_at TEXT);
```

三个必备视图：

```sql
-- (a) 解码：raw → 字段
CREATE VIEW v_decode AS SELECT ... ;      -- 直接由 instr_word 左连 field 表
-- (b) 溯源：这条结论凭什么
CREATE VIEW v_provenance AS
  SELECT c.*, group_concat(e.kind||':'||e.ref) AS why
  FROM claim c LEFT JOIN evidence e ON e.claim_id=c.id GROUP BY c.id;
-- (c) 漂移：nvdisasm 升级后什么变了
CREATE VIEW v_drift AS
  SELECT subject FROM claim WHERE status='observed' AND id NOT IN (
    SELECT claim_id FROM evidence WHERE ref = (SELECT max(nvdisasm_ver) FROM cubin));
```

## 5.2 三条硬规矩

1. **任何 `status='observed'` 的 claim 必须至少挂一条 `kind='cubin_word'` 或 `kind='experiment'` 的 evidence。** 挂不上的自动降级为 `inferred`。这条规矩就是防 AI 幻觉的唯一屏障。
2. **`nvdisasm_ver` 必须进 evidence 主键的一部分**。NVIDIA 每版都会改表，这样你能自动 diff 出"哪一代、哪一版、哪个字段变了"。
3. **DB 的 CI**：`round-trip`、`producer→consumer scoreboard 配对`、`opex = batch_t<<5|usched_info` 这类不变量写成测试，每次更新 DB 自动跑。测试失败 = 有 claim 是错的，自动标 `refuted`。

## 5.3 三个查询接口（这是"反汇编器+调试器+语义库"的最小骨架）

```
decode(word128, arch)        -> Instr{ mnemonic, operands[], ctrl{} }
encode(Instr, arch)          -> word128            # round-trip 必须位级相等
explain(arch, field_name)    -> { bit_range, meaning, provenance[], confidence }
```

第三条 `explain` 是你这个项目相对现有工具（CuAssembler / nvdisasm / Ghidra 插件）**唯一真正的增量**：它们能解码，但不能回答"凭什么"。

---

# 第六部分：应当优先做什么（有序 + 带 gate）

我的建议是把顺序改成"**先把闭环打通，再往里灌内容**"，而不是先铺 opcode 覆盖度。

### Phase 0（1–2 天）：锁版本 + 打通最小 decode
固定 toolkit 版本，记录 `sha256(nvdisasm)`、`ptxas --version`。写 `read_text_words` + `decode`。
> **Gate 0**：对 1000 条真实指令，你解出的 opcode/Rd/Ra/Rb/guard 与 `nvdisasm` 文本 **100% 一致**。

### Phase 1（1 周）：提取 nvdisasm 的 ISA 表 → 结构化 DB
这是**最高杠杆的一步**。拿到表之后，A 层从"研究"变成"数据导入 + 独立验证"。
> **Gate 1**：你 DB 里 `sm_86` 的 opcode 条目数与表里 `OPCODES` 总数一致；`encode(decode(w))==w` 位级 100%。

### Phase 2（1–2 周）：全链路 + 溯源
`PTX → SASS → word128 → decoded` 一键出四联表，每条记录自动带 cubin 哈希和命令。
> **Gate 2**：任意 kernel 一条命令产出可入库的四联表，且 `v_provenance` 无空 evidence。

### Phase 3（2 周）：三条指令打穿四层 —— `FFMA` / `LDG.E` / `ISETP`
对每条，产出：编码字段、bank 约束、依赖延迟、pipe、scoreboard 用法、predication 行为，**每条 claim 挂 experiment id**。
> **Gate 3**：你能不看文档、仅凭 DB 回答"这条指令的哪个 bit 选了哪个寄存器，执行后 lane 状态变了什么"。

### Phase 4（2 周）：regalloc 探针矩阵
> **Gate 4**：对**未见过的** PTX，能预测 regcount（±0）和是否 spill。

### Phase 5（2 周）：NVBit 动态层
选定 PC 上 per-lane snapshot，与 Phase 3 的静态预测比对。
> **Gate 5**：静态预测 vs 实测寄存器值一致率 ≥ 99%（不一致的地方就是你真正的发现）。

---

## 明确不要做的（这比做什么更重要）

| 不要 | 原因 |
|---|---|
| 从 CUDA 驱动 `.so` / GPU 固件入手 | EULA 明确禁止；且不可证伪 |
| 一次做全 opcode | 1000+ class，边际价值递减；先把 3 条打穿 |
| 20/30/40 三代并行 | 先在 sm_86 上把 DB 格式定死，再 diff |
| 先写 markdown 文档 | 无法 diff/join/CI，必然烂成"AI 生成的 best-guess" |
| 把 crucible-notes 当事实 | 作者自己标了 NOT A RELIABLE SOURCE |
| 追求"完整运行时 trace" | 不可达；收敛到"选定 PC 的 per-lane snapshot" |

---

## 最后一个判断

你说"这比直接从驱动/固件入手更容易得到可验证的成果"——**完全正确，而且比你想的更正确**：因为 `nvdisasm` 的加密表让 A 层几乎免费，**真正的未知已经被挤到 B（ptxas 分配启发式）和 D（微架构可观测子集）**。

所以如果你要一句话的行动指令：
> **先花 1 周把 `nvdisasm` 指令表导进 SQLite 并做到位级 round-trip（这一步建立可信底座），然后用 2 周把 `FFMA`/`LDG.E`/`ISETP` 三条打穿四层（这一步建立可证伪闭环），此后再谈覆盖度。**

需要我下一步具体做哪个？可选：
- 写 `nvdisasm` ISA 表的定位/解密/解析脚本（含格式猜测与校验）
- 写完整的 `cubin → word128 → DB` 导入器 + round-trip CI
- 设计 `ptxas` regalloc 探针矩阵（可直接跑的 CUDA 模板 + 统计脚本）
- 设计 NVBit tool 骨架（指定 PC、读 per-lane 寄存器、输出与静态预测比对）
