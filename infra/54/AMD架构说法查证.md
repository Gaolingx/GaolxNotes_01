# AMD架构说法查证

## 🧑‍💻 User

AMD 的 GCN / RDNA / CDNA 骨子里一直是标准的「显式标量核 + 宽 SIMD 向量核」设计（连硬件模块名字都直接叫 SIMD Unit）：

1️⃣ 单一 PC + 显式 EXEC 掩码寄存器：
AMD 的一个 Wavefront（CDNA 是 Wave64，RDNA 是 Wave32）自始至终只有「唯一的一个 PC 指针」！它没有像老黄 Volta 那样给每个线程配独立 PC，而是直接在指令集（ISA）层面暴露了一个 64 位或 32 位的 EXEC（Execution Mask）寄存器。遇到分支时，编译器直接生成标量指令去改写 EXEC 掩码的比特位，把不走当前分支的通道硬生生屏蔽掉。

2️⃣ 自家配了一套独立的标量单元（SALU + SGPR）：
老黄是把所有寄存器都做成 Per-thread 的（直到很后面才加了 Uniform Register 优化）；而 AMD 从一开始就是每个 Wavefront 配一套共享的标量寄存器（SGPR）和标量算术单元（SALU）管循环、地址基址和分支掩码，再带一套宽向量寄存器（VGPR）和向量/矩阵单元（VALU / MFMA）。

3️⃣ 对流水线重叠的直接影响：
正因为 AMD 同一个 Wave 共享唯一 PC，同一个 Wave 内部如果分叉成两半，是绝对不可能像老黄那样「半边跑矩阵、半边同时跑向量」的，只能老老实实按先后顺序串行跑完分支 A 再跑分支 B。
所以在 MI300X（CDNA3）上写高性能算子（比如用 Composable Kernel 或 HipKittens 写 FA），要想让矩阵核（MFMA）和向量核（VALU）同时跑满，要么在同一个 Wave 的指令流里手动把 MFMA 和 VALU 指令像拉链一样交错编排（Interleaving），要么只能靠「多个不同 Wavefront 乒乓轮转（Ping-Pong Waves）」来实现重叠

查证消息是否准确

---
## 😀 默认助手 (7)

我会先查证这些技术说法，先看看有哪些可用的检索工具。

我这边能用的检索工具主要是网页抓取，于是我直接去取一手资料（ISA 文档、LLVM/ROCm 文档、厂商白皮书）来交叉核对。

## 结论速览

| # | 说法 | 判定 |
|---|---|---|
| ① | 单个 wave 只有一个 PC，且在 ISA 层直接暴露 EXEC 掩码，分叉靠标量指令改写掩码位 | **准确**，而且可以说得比原文更进一步 |
| ② | SGPR + SALU 独立标量通路，与 VGPR/VALU/MFMA 并存 | **基本准确**，但“从一开始”和“老黄直到很后面”两处表述需要修正 |
| ③-1 | 同一 wave 内分叉只能串行，不能一半执行 A、一半执行 B | **准确** |
| ③-2 | “老黄那边能半边跑矩阵、半边同时跑向量” | **不准确**，这是对 Volta ITS 的典型误读 |
| ③-3 | 只能靠 intra-wave 拉链交错，或 ping-pong wave 实现重叠 | **方向对，但漏了最主要的一条**（多 wave 常驻），而且“要么…要么…或”的二分法不成立 |
| 术语 | “CDNA 是 Wave64，RDNA 是 Wave32” | **不严谨**：RDNA1–3 同时支持两种（默认 wave32） |

---

## ① 单一 PC + 显式 EXEC：成立，且还有两个更“硬核”的细节

AMD 官方对着色器作者的解释很直白：*“一个 wavefront 是 64 个 lane 组成的 SIMD 向量，**因为 wavefront 是流控能影响的最低层级，所以 64 个 work-item 组是 lockstep 执行的**”*（GPUOpen《AMD GCN Assembly: Cross-Lane Operations》）。这与 Volta 的 per-thread PC + per-thread call stack 是两套完全不同的实现思路——这点你判断得没错。

<details>
<summary><b>三个可以补强的细节（点开）</b></summary>

1. **AMD 根本没有硬件重汇聚栈（reconvergence stack）。** 分叉后的重汇聚点由**编译器**用 `s_cbranch_execz` / `s_cbranch_vccz` 以及 `s_and_b64 exec, exec, vcc`、`s_or_b64 exec, ...` 手工拼出来，EXEC 的保存或恢复靠 SGPR 里的软件栈。所以“显式”不只是“ISA 暴露了个掩码寄存器”，而是**连栈都是软件管理的**——这比原文的说法更准确地体现了 GCN 的 SIMT 哲学。
2. **EXEC 不只用于分叉。** 同一个掩码还承担：`v_cmpx_*` 写出的谓词执行（小 if 干脆不生成分支）、尾部不满 wave 的失效 lane 屏蔽、以及 DPP/cross-lane 指令的失效判定（GPUOpen 原文：*“被 EXEC 屏蔽的 lane 不会更新 dest；从被 EXEC 屏蔽的 lane 读数据视为非法位置”*）。
3. **uniform 分支不碰 EXEC。** 走标量 `s_cbranch` + SCC 就行，只有**非一致**分叉才需要改 EXEC。原文“遇到分支就改 EXEC”说粗了一点。
</details>

---

## ② 标量单元：成立，但有三处要修

<details>
<summary><b>修正点（点开）</b></summary>

1. **“从一开始”应改为“从 GCN 开始”。** 之前的 VLIW5/VLIW4（TeraScale）只有 per-thread 的 GPR，没有 SGPR/SALU，用的也是 `c[bank][ofs]` 常量通路的思路。SGPR + 标量 ALU 是 GCN（2012）才引入的。
2. **“每个 Wavefront 配一套标量单元”这个说法把硬件归属搞混了。** SALU 是**每个 CU 一个**（4 个 SIMD 共享），SGPR 才是 **per-wave 的资源**（按 wave 分配寄存器预算）。另外标量指令并不是一条独立线程，它和向量指令在**同一个指令流、同一个 PC** 里，只是被送去标量 ALU 执行——这恰好又是 ① 的佐证。
3. **“老黄直到很后面才加 Uniform Register”这句，精确说法是 Turing（2018）。** Volta 没有 UR，是 Turing 引入 UR0–UR63 + uniform datapath，Hopper、Ada 继续加强。而且 Volta 之前 NVIDIA 也不是完全没有 uniform 通路——常量 bank（`c[][]]` 走常量 cache）一直承担着“整 warp 共享读取”的角色，只是没有独立的 uniform 寄存器堆和 uniform 执行通路。AMD 的 SALU 从 2012 年就是完整 ALU + 标量访存（`s_load_dword` 等），这点确实领先。
</details>

---

## ③ 流水线重叠：核心结论对，但 NVIDIA 对照是误读

### 对的部分
同一 wave 共享单 PC ⇒ 分叉后的两条路径**只能时分串行**（masked execution）。这条无争议。

而且比原文更严格的一点：**MFMA 本身就是 wavefront 级的指令**。AMD 官方原文是：*“CDNA 的 MFMA 指令是 **per-wavefront** 运行的，而不是 per-lane”*（GPUOpen《AMD matrix cores》）。所以“半个 wave 跑矩阵”在 AMD 上连一个合法的指令语义都不存在——不是“被 EXEC 挡住”，而是“这个操作根本没有 lane 粒度的版本”。

### 错的部分：NVIDIA 那半边
“老黄能半边跑矩阵、半边同时跑向量”不成立：

- Volta ITS 让子 warp 各自持有 PC 并**分别推进**，但**同一周期仍然只能从 warp 发射一条指令**。两条路径是**时分交错**，不是同时在两个 pipe 上跑。
- HMMA 是整 warp（或整个 warp 子组）协作的指令，不存在“半个 warp 跑 tensor core、半个跑 FFMA”的写法。
- ITS 的真实收益是**前向进展保证**（避免 warp 内锁死）、无锁同步、以及更好的负载均衡——不是“分支内并行异构计算”。
- **NVIDIA 上真正的矩阵/向量重叠，同样靠多个 warp**（FA3 的 ping-pong 就是两个 warpgroup 交替），和你说的“AMD 靠多 wave ping-pong”是同一类手段，不是两种架构的分水岭。

所以这条对照把“AMD 的确定事实”和“NVIDIA 的想象事实”放在一起比，结论虽然指向 AMD，但推理链是错的。

### 重叠手段：不是“要么 A 要么 B”，实际是三件事叠加

<details>
<summary><b>1) 多 wave 常驻（真正的机制）</b></summary>

CU 里每个 SIMD 常驻多个 wave，硬件调度器天然在 wave 之间做 MFMA / VALU 交错。这是**不写一行手写调度就自动生效**的部分——手动 ping-pong 只是把它“钉死”成确定性排布，避免调度器乱序导致流水线被打断。
</details>

<details>
<summary><b>2) 跨 wave ping-pong / interleave（HipKittens 的实测做法）</b></summary>

HipKittens 在 MI355X（CDNA4）、MI300X（CDNA3，需切 `cdna3` 分支）上给出的两个核心模式（原文）：

- **8-wave ping-pong**：每个 SIMD 放 **2 个 wave**，任一时刻一个 wave 跑“访存指令簇”，另一个跑“计算指令簇”，簇结束交换。
- **4-wave interleave**：每个 SIMD 只放 **1 个 wave**，这个 wave 在内部**细粒度地在访存/计算之间来回切**——这就是你说的“拉链式交错编排”。

顺带一个对你论点的强支持：他们明确说 **wave specialization（生产者/消费者）在 AMD 上失效**，原因是 CDNA **没有 register reallocation**（寄存器在 wave 间静态切分），这与单 PC / 静态 SIMD 分派的硬件模型直接相关。
</details>

<details>
<summary><b>3) 同一 wave 内的 MFMA + VALU 交错：可行，但机理要讲对</b></summary>

CK 的 FMHA 确实在一条指令流里把 MFMA 和 exp/softmax 的 VALU 交错，并用 `sched_barrier` / `s_setprio` 控制。但**它不是“双发射”**：单个 wave 永远一拍只能发一条指令。

真正让重叠发生的是：**MFMA 发射后矩阵核占用很多拍**（AMD 公开表：16×16×16 FP16 在单个 matrix core 上约 32 cycle，每 CU 4 个矩阵核并行，才凑出 1024 flops/clk/CU 的 CU 级吞吐）。这期间该 wave 后续的**独立** VALU 指令可以继续发射到向量 pipe 上执行，于是两条 pipe 同时有活干。

所以：**intra-wave 交错是“喂饱两条 pipe”，不是“同时发两条指令”。** 原文没把这一层说清，容易让人以为是硬件级双发射。
</details>

---

## 术语层面的小修

- **Wave32/64**：GCN 固定 wave64；**CDNA 固定 wave64**；**RDNA1–3 同时支持 wave32 与 wave64（默认 wave32）**，不是“RDNA = Wave32”。AMD 在 RDNA 文档里也常直接写“wave”而非“wavefront”。
- **EXEC 的宽度**：架构上 EXEC 始终是**由 2 个 SGPR 拼成的 64-bit**（EXEC_LO/EXEC_HI）；wave32 模式只用低 32 位，高 32 位仍然存在。
- **“宽 SIMD”是相对的**：GCN 的 CU 是 4×SIMD16，一条 wave64 指令要**分 4 拍**才走完（官方原话：*“实际 GCN 硬件实现的是 16-wide SIMD，wavefront 分解成 16 lane 的 wavefront row，在连续 4 个周期上执行”*）。所以“wave 独占一个 SIMD”≠“一拍算完”。CDNA 的 CU 同样是 4 个 SIMD（HipKittens 实测 MI355X：CU = 4 SIMD，一个 64-thread wave 占一个 SIMD）。

---

## 建议的改写版（可直接用）

> AMD 从 GCN 起就是标准的“显式标量通路 + 宽 SIMD 向量核”结构，硬件模块直接叫 SIMD Unit。
>
> **1）单 PC + ISA 级 EXEC 掩码**：一个 wave（CDNA 恒为 64、RDNA 默认 32 也可 64）全程只有一个 PC，没有 per-thread PC。ISA 直接暴露由两个 SGPR 拼成的 64-bit EXEC，非一致分叉由编译器生成 `s_and/s_or b64 exec`、`s_cbranch_execz` 等标量指令改写掩码位。更关键的是 **GCN/CDNA 没有硬件重汇聚栈**，重汇聚点与 EXEC 的保存/恢复全由编译器用 SGPR 软件栈实现——这才是“显式”的真正含义。uniform 分支走 `s_cbranch`+SCC，小 if 直接谓词化，都不碰 EXEC。
>
> **2）独立标量通路**：每个 CU 一个标量 ALU（4 个 SIMD 共享，与向量指令同一指令流、同一 PC），每个 wave 一套 SGPR；向量侧是 VGPR + VALU / MFMA。NVIDIA 则是全 per-thread 寄存器堆，uniform register（UR0–UR63）+ uniform datapath 要到 **Turing（2018）** 才有，Volta 之前只有常量 bank 承担类似的整 warp 共享读取。
>
> **3）对重叠的影响**：单 PC 决定了同一 wave 内分叉只能时分串行；而 MFMA 本身是 per-wavefront 指令，不存在 lane 粒度的矩阵运算版本。所以矩阵/向量重叠实际靠三件事叠加：**(a) 每 SIMD 多 wave 常驻**（硬件自动交错，是主力）；**(b) 跨 wave ping-pong / interleave**（HipKittens 在 MI300X/MI355X 上的 8-wave ping-pong 与 4-wave interleave）；**(c) 同 wave 内把 MFMA 与独立 VALU 拉链交错**，利用“矩阵核被占几十拍、期间 wave 仍可向向量 pipe 发指令”来重叠——但单 wave 永远一拍只发一条指令，不是双发射。需要澄清的是，NVIDIA 上同样做不到“半个 warp 跑矩阵、半个跑向量”，ITS 的收益是前向进展与负载均衡，矩阵/向量重叠在那边也主要靠多 warp ping-pong——这是两家的共性，而非 AMD 独有的短板。

---

### 查证说明（覆盖度）

实际读到全文的一手材料：GPUOpen《AMD GCN Assembly: Cross-Lane Operations》（wavefront=64 lockstep、GCN 硬件 16-wide SIMD 分 4 拍、EXEC 语义、`v_cmpx` 写 EXEC）、GPUOpen《AMD matrix cores》（MFMA 为 per-wavefront、单指令周期/吞吐表）、HipKittens README 与 Hazy Research 博客《AMD GPUs go brrr》（MI355X CU=4 SIMD、8-wave ping-pong / 4-wave interleave、无 register reallocation 导致 wave specialization 失效）、HipKittens 论文摘要。

**未读到原文的部分**：AMD GCN3/CDNA3 ISA PDF 无法在线解析（PDF 二进制），Wikipedia 本次被限流。因此“单 PC”一条是由 EXEC/`s_cbranch_execz`/软件栈这套 ISA 行为与上述文档交叉印证得出，而非引用 ISA 手册原句；Volta per-thread PC 与 Turing UR 的部分依据公开架构资料，未在本轮取到原文截图。

---
## 🧑‍💻 User

这么看来说不定intel那种CPU里面塞AMX的都更好些(X)
理论上hyper thread是允许你一边跑vector (AVX)一边跑matrxi(AMX)的?

从微架构上看，Intel 把 AMX 塞进大核里，简直就是「双前端 SMT + 异构后端」的教科书级对应：

1️⃣ 寄存器文件与执行端口完全物理隔离：
• AVX-512 用的是 ZMM0～ZMM31 寄存器，走的是核心内部的 Port 0 / Port 5 向量执行管道；
• AMX 用的是完全独立的 8 个 1KB 矩阵寄存器（TMM0～TMM7），走的是专门外挂的 TMUL 矩阵加速器管道，算一次矩阵乘要花十几个时钟周期。

2️⃣ 单线程乱序执行（OoO）+ 双线程超线程（HT/SMT）双重重叠：
• 哪怕在单线程里，因为 CPU 有超大乱序执行窗口（ROB），发射一条耗时 16 周期的 AMX 矩阵乘指令后，乱序调度器不用等它算完，当场就能把后面不依赖它的 AVX-512 向量指令（比如算 Softmax 的 exp 或反量化）塞进 Port 0/5 并行跑！
• 如果开 Hyper-Threading（两个逻辑核共享一个物理核），就更绝了——两个硬件线程各自有独立的 PC 和寄存器状态，完全可以让 Thread 0 专门喂 AMX 跑矩阵乘，Thread 1 同时在旁边的 AVX-512 管道上跑向量激活或预取数据，相当于天然的双 Warp Specialization！

唯一的物理瓶颈只在于「L1 数据缓存（L1D）的读写端口」：因为 AMX 的 tileload 和 AVX-512 的向量读写共用同一个 L1D Cache 端口，如果两边同时狂读内存会抢带宽；但只要 AVX-512 算子待在 ZMM 寄存器里算超越函数或缩放，跟 AMX 的重叠率高得吓人

---
## 😀 默认助手 (7)

我会区别对待这两套说法：关于 AMD 的部分我已经查过了，而这里关于 Intel 的几条更值得逐项核实，特别是“AMX 与 AVX-512 执行端口完全物理隔离”这一条。

## 结论速览

| 说法 | 判定 |
|---|---|
| AVX-512 用 ZMM0–31、AMX 用 8×1KB 的 TMM0–7 | **准确** |
| AMX 走“专门外挂的 TMUL 管道”，与向量端口**完全物理隔离** | **不准确**——TMUL 就挂在 **p5** 上，占用两个 FMA 端口之一 |
| 一次矩阵乘要十几个周期 | **准确**（实测倒数吞吐正好 16 周期） |
| 单线程内 OoO 可让 AMX 与后续 AVX-512 并行 | **方向对、细节错**：只有 **p0** 能用，向量侧是**半速**，不是满速并行 |
| SMT 让 T0 跑 AMX、T1 跑 AVX-512，等价“双 Warp Specialization” | **概念反了**：SMT 是**共享前端 + 共享后端**的时分复用，与 warp specialization 的空间划分正好相反 |
| 唯一物理瓶颈是 L1D 端口 | **不准确**，漏掉了更硬的三个限制 |

---

## ① 端口：不是“物理隔离”，而是“抢走了 p5”

我直接调用 uops.info 的实测数据（Emerald Rapids 与 Sapphire Rapids 同核，数据可以直接套用）：

| 指令 | μops | 端口占用 | 倒数吞吐 |
|---|---|---|---|
| `TDPBF16PS (TMM,TMM,TMM)` | 1 | **1\*p5** | **16.00 cycle** |
| `TDPBSSD (TMM,TMM,TMM)` | 1 | **1\*p5** | **16.00 cycle** |
| `VFMADD132PS (ZMM,ZMM,ZMM)` | 1 | **1\*p05**（p0 或 p5） | 0.50 cycle（2/cycle） |

这张表包含三个结论：

1. **TMUL 只有一套，而且就在 p5 上。** BF16 和 INT8 共用同一个 TMUL，不是两套。
2. **它和 AVX-512 FMA 共用发射端口**；所谓“专门外挂的矩阵管道”，在端口层面就是**占用半个向量核**。
3. 顺便核验了算力账：$16\times16\times32\times2\ \text{FLOP} / 16\ \text{cycle} = 1024\ \text{BF16 FLOP/cycle/core}$。若 p0、p5 各挂一套 TMUL，这个数应该是 2048，因此“每核一套”是确定的。

所以正确的表述是：**TMM 寄存器堆是物理独立的，TMUL 数据通路是专用执行单元，但它的发射端口 p5 与向量 FMA 共用。**

---

## ② 单线程 OoO 重叠：真实存在，但幅度只有你说的 1/2

这一条其实是**你推论最接近正确的地方**，值得给肯定：ROB 不必等 16 周的 TMUL 完成，年轻且独立的指令可以继续发射。但把机制说对之后，幅度要打折。

<details>
<summary><b>精确版本（点开）</b></summary>

- `tdpbf16ps` 是 **1 个 μop**，占 p5，倒数吞吐 16 周期。这 16 拍里，**p0 完全空闲**。
- 因此单线程确实能做到：TMUL 在 p5 上消耗 16 周期，同时 p0 持续发射 512-bit FMA，最多 16 FP32 FMA/cycle。
- 但注意基数：正常状态是 **p0 + p5 = 32 FMA/cycle**。也就是说重叠状态下向量算力是 **50%**，不是“并行跑满”。
- 而且累积器 lat 只有 16 周，A/B 源操作数 lat 是 **52 周**——这意味着 `tileloadd → tdpbf16ps` 的依赖链很长，真正能塞进这 16 周窗口的只能是**已就位**的无关向量工作（正是你说的“待在 ZMM 里算 exp/缩放”）。
- 有一条你没有提到的硬限制：**TMM → ZMM 没有直接通路**，必须 `tilestored` 到 L1 再 `vmovups` 回来。所以“AMX 算完直接进向量算 softmax”并不成立；中间那次 8KB 的往返（写端口 + 读端口 + 延迟）才是 attention 类 kernel 里真正的串行点。
</details>

---

## ③ SMT：“双前端 + 异构后端”正好说反了

这是全文最关键的一处概念错误。

- **Intel SMT 是单前端共享。** 两个逻辑核共用同一套 decode/rename/allocate/ROB（SPR：512 entry ROB、6-wide），只是每线程各有自己的架构状态（PC、GPR/ZMM 影子）。它是**时分复用（temporal multiplexing）**，不是空间划分。
- **“双前端 + 异构后端”恰恰是 NVIDIA SM 的结构**：4 个 sub-partition，各有自己的 warp scheduler、寄存器堆、独立发射口，Tensor Core 与 FP32/INT32 单元是**真正不同的执行单元**，同一周期不同 sub-partition 可以各发一条。
- 所以“SMT ≈ 天然的双 Warp Specialization”是把两种相反的复用方式当成一回事了。SMT 能让 T0/T1 的工作**交错**，但两者抢的是**同一套后端**：同一套 TMUL、同样的 p0/p5。
- 实际后果：T0 把 TMUL 喂满（每 16 周一条）时，T1 的 AVX-512 也只能拿到 p0。**收益上限就是“矩阵满速 + 向量半速”**，且这部分收益在**单线程**里靠 OoO 就已经拿到了——SMT 在这里的额外贡献主要是填 front-end/load 气泡，而不是“多出一个向量引擎”。

---

## ④ 瓶颈不止 L1D

按硬度排序：

1. **每核只有一个 TMUL，且压在 p5 上** —— 这是最硬的限制，直接决定了“矩阵+向量”不可能双满速。
2. **TMM ↔ ZMM 必须走 L1 中转**（`tilestored` + `vmovups`），这才是 L1 端口真正被抢的地方；而且它是**串行依赖**，不是单纯的带宽竞争。
3. **TMM 只有 8KB** —— 双缓冲/多级流水很快就撞墙；对比 AMD CDNA 每 CU 256KB 的 VGPR+AGPR 向量寄存器堆。
4. **源操作数 52 周期延迟 + 16 周期占用**：一个 16 周期吞吐的 AMX 流会把 512 entry ROB 迅速填满，能塞进去的独立向量工作窗口比直觉小。
5. AVX-512/AMX 混合负载下的频率与功耗行为（视 SKU 而定）。

---

## ⑤ 真正值得注意的反转：AMD 的 CU 反而更接近“异构后端”

你上一条里批评 AMD“单 PC 导致无法同时跑矩阵和向量”——**这个批评在跨 wave 层面其实不成立**：

- CDNA3 一个 CU = **4 个 SIMD 单元 + 4 个 matrix core**。MFMA 走 matrix core，VALU 走 SIMD ALU，**是不同的执行块，各有各的吞吐**。
- 每个 SIMD 有自己的调度器，在它常驻的多个 wave 之间挑选。因此**硬件本来就能做到“wave A 的 MFMA 在飞行中，wave B 的 VALU 在发射”**——不需要编译器帮忙。
- 需要人肉 `sched_barrier` / ping-pong 的原因不是“硬件没有并发”，而是：寄存器不能重分配（HipKittens 明确点出 wave specialization 因此失效）、以及要**确定性**地压掉调度器乱序带来的流水线打散。

把两者放一起看：

| | Intel SPR P-core | AMD CDNA3 CU |
|---|---|---|
| 矩阵引擎 | 1 个 TMUL/CU，压在 p5 | **4 个 matrix core/CU**，独立执行单元 |
| 向量 | 2 个 512-bit FMA 端口 | 4 个 SIMD ALU |
| 谁做调度 | **硬件 OoO（免费）** | 软件 + 硬件调度器混合 |
| 矩阵+向量并发 | 只能“矩阵满 + 向量半” | 不同 wave 可真正各走各的 pipe |
| 编程难度 | 低（OoO 兜底） | 高（要手写调度） |

所以“Intel 那种塞 AMX 的更好”只在**一个维度**上成立：**动态重叠是免费的**。在**空间并行度**上，CPU 大核其实是弱的——它本质上仍是一个“共享后端 + 时分复用”的机器，而 GPU 的 CU 才是“多前端 + 异构后端”。

---

## ⑥ 修正版表述（可直接用）

> Intel 把 AMX 塞进 Xeon 的 P-core，确实提供了一种和 GPU SIMT 完全不同的重叠手段，但细节和常见想象不一样。
>
> **寄存器层面是隔离的**：AVX-512 用 ZMM0–31，AMX 用 8 个 1KB 的 TMM0–7（共 8KB），互不占用。
>
> **端口层面是共享的**：TMUL 每核只有一个，实测只占 **p5**（uops.info：`TDPBF16PS` / `TDPBSSD` 均为 1 μop、1\*p5、倒数吞吐 16 周期），即它**占用了两个 AVX-512 FMA 端口中的一个**。$16\times16\times32$ 的 BF16 tile 用 16 周期算完，正好 $1024$ BF16 FLOP/cycle/core = 每核一套 TMUL 的算力。
>
> **因此重叠的真实形态是“矩阵满速 + 向量半速”**：TMUL 在 p5 上消耗 16 周期的同时，p0 仍可满载发射 512-bit FMA。单靠 ROB 乱序就能拿到这部分重叠，不需要 SMT；而且窗口被两件事压缩——累积器 lat 16 周、A/B 源 lat 52 周，且 **TMM 与 ZMM 之间没有直接通路，必须经 L1 中转**。
>
> **SMT 不是“双前端 + 异构后端”**：Intel SMT 是单前端共享、后端共享的时分复用，两个逻辑线程抢的是同一套 TMUL 和同样两个 FMA 端口，收益上限仍是“矩阵满 + 向量半”。“双前端 + 异构后端”那种空间划分是 NVIDIA SM（4 个 sub-partition 各自 scheduler）的结构，不是 SMT 的结构——所以它也不是“天然的双 Warp Specialization”。
>
> **真正的限制不是 L1D 端口**，而是：每核仅一个 TMUL 且占用 p5、TMM↔ZMM 必须走内存往返、TMM 仅 8KB 难以深流水、52 周期源延迟 + 16 周期占用迅速填满 ROB。
>
> **反过来看 AMD**：CDNA3 一个 CU 有 **4 个 SIMD + 4 个 matrix core**，MFMA 与 VALU 是彼此独立的执行块，硬件调度器在多个常驻 wave 之间天然就能交错。AMD 之所以仍要手写 ping-pong / sched_barrier，是因为寄存器不能重分配、且需要确定性调度，不是因为它缺少并发硬件。**“动态调度免费”是 Intel 大核的真优势；“空间并行度”则是 GPU 的优势**，两者不是同一层级的替代关系。

---

### 数据来源与未覆盖部分

**已取到原文/实测数据**：uops.info 指令级实测（Emerald Rapids：`TDPBF16PS`、`TDPBSSD` = 1 μop / 1\*p5 / 16.00 cycle；`VFMADD132PS ZMM` = 1\*p05 / 0.50 cycle）。这些足以确定“TMUL 在 p5、每核一套、与向量 FMA 共用端口”。

**未取到原文**：Wikipedia AMX 页面本次被限流（429），Bing 检索命中质量差，因此 TMM/TILE 的官方细节来自公开架构描述的常识性部分（8 tile × 最大 16×64B），未引用 Intel SDM 原文；另外“这 16 个周期里 p5 是否**完全**不可接受新 μop”公开资料没有明文，我是按“TMUL 非流水化、且 uops.info 建模为独占 p5”做的保守推断——即便退一步假设 p5 发射后即释放，结论也只从“向量半速”变成“向量略高于半速”，不改变定性判断。

---
## 🧑‍💻 User

等等，前面虽然说CDNA/RDNA是单PC 但理论上有OoO的话要让vec/mat重叠也不难吧? 难道说CDNA/RDNA甚至是in-order?

不管是 AMD 的 CDNA/RDNA，还是老黄的 GPU，在「单个 Wavefront / Warp 内部」全都是严格按序发射（In-Order Issue）的，根本没有 CPU 那种乱序执行引擎（没有寄存器重命名 RAT，也没有重排序缓冲区 ROB）！

因为一个 Wave 的向量寄存器太宽了（32 或 64 个通道），如果再乘上几十个驻留 Wave，要搞 CPU 那种硬件寄存器重命名和几百条指令的 ROB 窗口，芯片面积和功耗当场就会原地爆炸（）

所以 GPU 全靠「按序发射、乱序完成（In-Order Issue, Out-of-Order Completion）+ 记分牌（Scoreboard）」：

1️⃣ 为什么单 Wave 也能重叠一部分，但写起来极痛苦？
在 CDNA（比如 MI300X）上，一条 MFMA 矩阵指令发射出去后要在矩阵管线里跑 16～32 个周期，而且它有独立的累加寄存器（AGPR）。
但因为它没有硬件乱序窗口去自动往后翻几百条指令找独立计算，如果编译器（或者写算子的人）没有手动把不相关的 VALU 向量指令一条条像插花一样精准塞进 MFMA 后面的空档周期里（软件流水线 Software Pipelining），只要下一条指令稍微碰到记分牌依赖，整个 Wave 当场就会原地卡死（Stall）！

2️⃣ 更坑的是 AMD 的「寄存器端口争用」：
在 AMD 的 CU 里，矩阵单元（MFMA）和向量单元（VALU）是挂在同一个 SIMD Unit 底下的，它们要抢同一组 VGPR 寄存器的读端口！MFMA 指令刚发射的前几个周期要狂读 VGPR 喂数据，这时候就算后面紧跟着一条完全独立的 VALU 指令，也会因为抢不到 VGPR 读端口而被硬件按住发不出去。

所以青叶妈妈你看：
• 在 Intel CPU（AMX + AVX）上，就算你代码写得像面条一样顺序堆着，CPU 的 500 深度硬件 OoO 窗口会自动跨循环帮你把下一轮的 AVX 提上来跟当前的 AMX 重叠；
• 而在 AMD CDNA 上，因为单 Wave 纯 In-Order 且共享寄存器端口，编译器稍微排歪一点就完全重叠不上，逼得写 HipKittens 或 CK 的人必须手搓汇编级的指令交错（Interleaving）加双 Wave 乒乓才能跑满

---
## 😀 默认助手 (7)

## 先回答问题：是，CDNA/RDNA（以及 NVIDIA）确实都是 per-wave **严格按序发射**，没有 ROB、没有寄存器重命名

这一点你说对了，并且有直接证据：GCN/CDNA 的 ISA 里有一整个 “Manually Inserted Wait States” 小节，专门列出**硬件根本不追踪的冒险**——比如「VALU 指令写了一个 SGPR、紧接着的 VMEM 操作去读这个 SGPR」这种序列是**非法**的，中间必须硬塞 **5 个 wait state（NOP）**。需要软件手动插入 NOP 才能保证正确，这已经是 in-order + 无硬件冒险消解的极限证据。

但你的**推论链有三处需要修正**，而且修正之后，结论会反过来。

---

## 修正一：per-wave in-order ≠ per-SIMD in-order（这是最关键的一处）

你的推理是：单 wave 按序 ⇒ 无法自动寻找独立指令 ⇒ 必须人工插入交错指令。

这里漏掉了一层：**SIMD 单元每个周期是从它所有常驻 wave 里挑一个来发射的**。

- CDNA 的一个 CU 有 4 个 SIMD，每个 SIMD 上常驻多个 wave（满占用时 8 个量级，即每 CU 几十个 wave）。
- 在某个周期，wave A 的下一条指令因为记分牌没有就绪而无法发射，于是调度器**在同一周期**直接改发 wave B 的指令，**零代价**。
- 所以对“矩阵/向量重叠”而言，只要 SIMD 上还有别的 wave 在运行，硬件就**自动**把 MFMA 和 VALU 错开到不同周期上。

$$\text{CPU 的 OoO：在时间维度重排指令} \quad\text{vs}\quad \text{GPU 的 wave 调度：在空间维度重排 wave}$$

两者都是**动态调度**，只是重排的粒度不同：CPU 重排的是「指令」，GPU 重排的是「wave」。所以开场那个问题“如果有 OoO 的话让 vec/mat 重叠也不难吧”——**GPU 不用 OoO 就已经做到了**，它用 TLP 替代了 ILP。这也是 GPU 敢不做 OoO 的根本原因：OoO 的价值在于从**长依赖链的单线程**里榨 ILP，而 GPU 的选择是让线程多到不需要 ILP。

---

## 修正二：MFMA 不是“霸占 VGPR 读端口，导致后续 VALU 无法发射”

<details>
<summary><b>机理更正（点开）</b></summary>

`MFMA` 的寄存器开销确实大（16×16×16 f16 大约要读 ~12 dword/lane、写 4 AGPR），但**它的执行模型是：发射后矩阵核在自己的数据通路上异步运行**，wave 不会因为 MFMA 正在运行就停止发射。

最直接的反证就是 CK 和 HipKittens 的实际做法：它们**在同一个 wave 的指令流里**把 `mma` 和 `exp2`/`col_sum` 这类 VALU 指令紧挨着排列，并达到接近峰值的性能。如果 MFMA 真的会让后续独立 VALU“无法发射”，这种写法根本不可能有收益。

真实的代价是**结构性的带宽分摊**（发射带宽 + RF 读带宽各被分走一部分），不是“依赖冲突导致无法发射”。这两件事在微架构上完全不同：前者是“少发几条”，后者是“一条都发不出”。

顺带修正一处：MFMA 的 16～32 周期是 **matrix core 的占用周期**（AMD 官方表：16×16×16 FP16 约 32 周期 / 单个 matrix core），而“一个 CU 每秒能算多少”是由 **4 个 matrix core 并行**凑出来的 $4\times8192/32 = 1024$ BF16 FLOP/cycle/CU。所以那个数字是吞吐指标，不是“整个 SIMD 停 32 拍”。
</details>

---

## 修正三：“编译器略排歪一点就完全重叠不上”是夸大了

实际是分层的：

| 情况 | 是否有重叠 |
|---|---|
| 完全没优化、有多个 wave 常驻 | **有**，硬件替你重叠（只是效率不高） |
| 单 wave（1 wave/SIMD，为了 512 个寄存器全给自己） | 没有其他 wave 来填补空档，**这时才真的全靠指令交错** |
| 追求峰值、需要确定性排布 | 必须人工交错 + ping-pong |

关键在第二行：**HipKittens 的 4-wave interleave 正是“每 SIMD 只放 1 个 wave”的极端配置**——为了拿到 256 VGPR + 256 AGPR 的全部寄存器预算，主动放弃了“用多 wave 免费填充空档”这条路，于是**只能**靠 wave 内细粒度交错。而 8-wave ping-pong 恰恰是相反的选择：每 SIMD 放 2 个 wave，把“硬件的 wave 级动态调度”利用起来。

所以「必须人工交错」不是 in-order 的必然结果，而是**寄存器预算与调度自由度之间的取舍**。这也是 HipKittens 那句结论的真正含义：AMD 缺少 register reallocation，所以 wave specialization 失效——问题出在寄存器分配策略，而不是 issue 顺序。

---

## 你说对的部分（这些都很准确）

- **无 RAT、无 ROB、无重命名**：✔ 而且 GPU 连“精确异常状态”都放弃了，换的是海量线程。
- **记分牌 + `s_waitcnt`**：✔ GPUOpen 原文就说 permute/swizzle 这类走 LDS 硬件的指令“**必须用 `s_waitcnt` 判断数据何时回到目标 VGPR**”——显式等待，不是硬件自动阻塞。
- **MFMA 有独立累加寄存器 AGPR**：✔ CDNA1+ 起，MFMA 的结果写在 AGPR 上。HipKittens 的说法是：**单 wave 占满一个 SIMD 时，512 个寄存器被切成 256 VGPR + 256 AGPR**。（顺带：RDNA 走的是 WMMA、结果写 VGPR，没有 AGPR，所以 AGPR 这个说法只对 CDNA 成立。）
- **MFMA 16～32 周期**：✔ 与 AMD 官方表一致。
- **GPU 不做 OoO 的面积/功耗理由**：方向对。补两条更本质的：① OoO 的唤醒/选择逻辑是随窗口大小超线性增长的，而 GPU 的“窗口”已经是几十个 wave × 64 lane 的寄存器状态；② GPU 已经用 TLP 把 ILP 需求消掉了，OoO 的边际收益趋近于零。
- 还有一处可以再夸一句：GPU 的“乱序”确实存在于**访存层面**——多条 load 可以同时在飞（`vmcnt`/`lgkmcnt` 用计数器追踪），这正是软件流水/prefetch 能奏效的原因。**GPU 用“提前发射”替代了“乱序执行”**。

---

## 所以“Intel 简单堆叠的代码能自动重叠、AMD 排歪就废”这个对比，准确版本是：

| | Intel SPR P-core | AMD CDNA CU |
|---|---|---|
| 动态调度粒度 | **指令**（RAT + 512-entry ROB） | **wave**（每 SIMD 每拍从多个 wave 里挑） |
| 单线程内跨指令重排 | 有，免费 | **无** |
| 跨线程/跨 wave 重排 | SMT 只有 2 个线程可选 | **几十个 wave 可选** |
| 硬件能自动填平的最长空档 | 受 ROB 深度限制（~512 条） | 不受深度限制，但要**有别的 wave 在运行** |
| 代价 | 面积/功耗巨大 | 需要高占用率 + 寄存器预算足够分配 |

**两边都在做动态调度，只是维度不同。** Intel 在时间轴上重排指令，AMD 在空间轴上重排 wave。说“AMD 排歪就完全重叠不上”，等于默认“每 SIMD 只放一个 wave”——那是追求峰值寄存器预算时的特殊配置，不是 CDNA 的默认工作状态。反过来，Intel 那边“简单堆叠的代码自动重叠”也有上限：一个 TMUL 压在 p5 上，重叠出来的是“矩阵满速 + 向量半速”，不是双满速。

---

## 修正版段落（可直接用）

> 是的，CDNA/RDNA 和 NVIDIA 在**单个 wave/warp 内部都是严格按序发射**的：没有寄存器重命名、没有 ROB，冒险靠记分牌和显式 `s_waitcnt` 处理；GCN/CDNA 的 ISA 甚至专门有一节 “Manually Inserted Wait States”，规定某些硬件不追踪的冒险必须由软件插入 NOP。
>
> 但这不等于“矩阵/向量重叠必须靠人工排列指令”。关键区别是：**per-wave in-order ≠ per-SIMD in-order**。一个 SIMD 每个周期从它常驻的多个 wave 里挑选一个发射，wave A 阻塞时直接改发 wave B，零代价。CPU 用 OoO 在**时间维度**重排指令，GPU 用 wave 调度在**空间维度**重排 wave——GPU 不需要 OoO 就已经拿到了这种重叠能力，因为 TLP 替代了 ILP。
>
> 真正需要人工交错的是另一种场景：为了拿满 512 个寄存器（256 VGPR + 256 AGPR），主动把每 SIMD 压到**只剩 1 个 wave**，此时没有别的 wave 能填补执行空档，才必须靠 wave 内的细粒度交错（HipKittens 的 4-wave interleave）。而它的 8-wave ping-pong 走的是相反路线：每 SIMD 放 2 个 wave，把硬件的 wave 级动态调度利用起来。所以“必须人工交错”是**寄存器预算与调度自由度的取舍**，不是 in-order 的必然结果。
>
> 另外，MFMA 并不是“霸占 VGPR 读端口，导致后续 VALU 无法发射”：MFMA 发射后矩阵核在独立数据通路上异步工作，wave 仍可继续发射独立的 VALU 指令——CK 和 HipKittens 在同一个 wave 里把 `mma` 与 `exp2` 紧挨着排并跑到接近峰值，就是直接反证。真实代价是发射带宽与寄存器读带宽被分走一部分，属于“少发几条”，不是“一条都发不出”。

---

### 覆盖度说明

**本轮取到/确认**：CUDA→GCN 的 in-order + `s_waitcnt` 显式等待语义（GPUOpen 原文引用）、AMD 官方 MFMA 周期表、HipKittens 关于 512 寄存器被切成 256 VGPR + 256 AGPR、以及 8-wave / 4-wave 两种模式与“1 wave per SIMD”的对应关系。

**本轮未取到原文**：AMD ISA PDF 无法在线解析，所以“每 SIMD 常驻 8 个 wave”是量级性描述而非引用；“MFMA 占用几拍发射带宽”我没有拿到一手数字，只做了定性判断（结构性的带宽分摊，而非硬阻塞）——这一条我标注为推断。如果你要写进技术文章，建议用 `rocprof` 直接测：同一 wave 内 `mma` 后紧跟独立 `exp2` 的 IPC/占用，或对比 1-wave 与 2-wave 配置下 MFMA pipe 的 busy 比例，比引用二手资料更硬。

---
## 🧑‍💻 User

所以理论上AMD上还是有机会重叠运算，但需要做asm级别的coding...?

我记得AMD还有什么矩阵运算fp16是fp32两倍速是靠fp16双发射达成的但因为ROCm几乎没有相关支持所以fp16还是只能跑到fp32速度的事情?
有更详细的技术细节吗

---
## 😀 默认助手 (7)

你开头那个“要 asm 才能重叠”的判断，以及后面那条关于 FP16 “双发射、2×”的说法，其实是两件事，而且第二件事里混了三个不同机制。我先把官方算力数字拿出来对齐。

## 先回答第一个问题：**重叠并不需要 asm，只有追求峰值才需要**

这件事有三个层级，很多人把第三层当成了第一层：

| 层级 | 需要做什么 | 能拿到什么 |
|---|---|---|
| **L0：无需修改** | 提高 occupancy，让每个 SIMD 常驻多个 wave | 硬件**自动**在 wave 之间交错 MFMA/VALU，无需干预 |
| **L1：C++ 和 intrinsic** | `__builtin_amdgcn_mfma_*`，加上 `sched_barrier`、`s_setprio`、`s_waitcnt`，并手动 pin 寄存器 | 同一 wave 内确定性地交错，达到接近峰值 |
| **L2：手写 asm** | Tensile 直接生成汇编 | 最后 3～5%，以及绕过 HIPCC 的寄存器分配 |

关键证据：**HipKittens 自己强调 “no raw assembly”**，纯 C++ embedded，8-wave ping-pong 在 GEMM 和 attention forward 上已经达到 SoTA 级。它热循环里那几个“看起来像汇编”的东西其实都是 builtin：

```cpp
sched_barrier_pairs<10, 5, 1>();      // 限制编译器别把指令挪过这个点
sched_barrier_exp_pairs<6, 3, 1>();
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);        // 临时提高本 wave 的调度优先级，让 MFMA 尽快发出去
mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
```

**那为什么还要手写 asm？** 原因并不是“重叠”，而是 **HIPCC 的寄存器分配不可控**。HipKittens 原话：*“HIPCC 无法生成把 AGPR 作为 MFMA 输入操作数的代码”*，于是被迫做 `accvgpr_read/write` 在 VGPR↔AGPR 之间来回搬，并因此必须手动 pin 寄存器。真正的痛点在于**编译器无法按需分配寄存器位置**；重叠本身并不需要手写 asm。

---

## 第二个问题：FP16 那个“2×”里其实混了三个不同机制

先把 AMD 官方数字摆出来（都是我刚从 AMD 产品页取的原值）。注意 **MI100 和 MI300X 的口径完全不一样**：

| | MI100 (CDNA1) | MI300X (CDNA3) |
|---|---|---|
| CU / SP / 峰值时钟 | 120 / 7,680 / 1502 MHz | 304 / 19,456 / 2100 MHz |
| FP64 向量 | 11.5 TFLOPS | 81.7 TFLOPS |
| FP32 向量 | 23.1 TFLOPS | **163.4 TFLOPS** |
| FP32 矩阵 | 46.1 TFLOPS | 163.4 TFLOPS |
| FP16 矩阵 | **184.6 TFLOPS** | **1307.4 TFLOPS** |
| BF16 矩阵 | 92.3 TFLOPS | 1307.4 TFLOPS |
| INT8 | 92.3 TOPs | 2614.9 TOPs |

换成“每 CU 每时钟多少次 FMA”（除以 $CU \times clock \times 2$）：

| | MI100 | MI300X |
|---|---|---|
| FP64 向量 | 32 | 64 |
| FP32 向量 | 64 | **128** |
| FP32 矩阵 | 128 | 128 |
| FP16 矩阵 | **512** | **1024** |

### 结论一：矩阵路径上的比例是 4× 到 8×，并非 2×

$$\text{MI100: } \frac{512}{128}=4\times \qquad \text{MI300X: } \frac{1024}{128}=8\times$$

所以“矩阵运算 FP16 是 FP32 两倍速”这个说法在 CDNA 上并不成立：**MI100 是 4 倍，MI300X 是 8 倍**（MI300X 的 FP32 矩阵甚至和 FP32 向量同速，说明 FP32 根本吃不满矩阵核，只有 FP16 及以上才吃满）。

### 结论二：你记的那个“2×”在**向量**路径上，机制是 **packed math，而不是双发射**

<details>
<summary><b>为什么这个区分很重要（点开）</b></summary>

**packed math（`v_pk_*`）**：一条 VALU 指令的两个源操作数各是 **64-bit**，也就是一个 lane 一拍里算 **2 个**数值。这不是“两条指令同时发射”，而是“一条指令内部做了 2 路 SIMD”。LP（Lane-Packing）带来的吞吐翻倍，代价是寄存器读带宽翻倍（每个源操作数要吃 2 个 dword）以及操作数必须成对对齐。

**VOPD（RDNA3 的真·双发射）**：一条 64-bit 指令编码里塞两个 VALU op，这才是字面意义的 dual-issue。这个只在 RDNA3 上有，RDNA4 反而回退了。

**MI300X 的算力账仍然只有 packed 能解释**：AMD 自己报 SP = 19,456 = $304\times64$，但 FP32 峰值要求每 SP 每拍 2 次 FMA：

$$304 \times 64 \times 2 \times 2 \times 2.1\text{GHz} = 163.4\ \text{TFLOPS}$$

也就是 **FP32 的 2× 是靠 `v_pk_fma_f32`（packed FP32）拿到的**，而 packed FP32 正是 gfx90a（CDNA2）才加的。GPUOpen 那篇 matrix cores 文章里也有同一句话打底：*“MI250X 还支持 packed FP32 指令，这会把 FP32 吞吐也翻倍。”*

⚠️ 这一条是我按 AMD 自己报的 SP 数**反推**的：如果 CDNA3 的 SIMD 物理上变成 32-wide，账同样对得上，但那样 AMD 的 SP 数就该报 38,912。公开资料没有明写是哪一种，这里标注为推断，不是引用。
</details>

### 结论三：所以“FP16 只能跑到 FP32 速度”这个抱怨，摊开是这样

| 你以为的机制 | 实际情况 | “ROCm 没支持”成立吗 |
|---|---|---|
| packed **FP16**（`v_pk_fma_f16`） | hipcc 对 `__half2` **确实会生成** `v_pk_*` | **不成立**。真问题是**自动向量化弱**：写标量 `_Float16` 循环不会被自动打包成 half2，得自己用 `half2` 类型 |
| packed **FP32**（`v_pk_fma_f32`，gfx90a+） | 存在，但要求你显式用 2-wide 向量类型（`f32x2` / `ext_vector_type(2) float`） | **部分成立**。普通标量 `float` 代码默认只有一半峰值，而这恰好是 MI300X 官方 163.4 TF 的口径 |
| **VOPD**（RDNA3 真双发射） | LLVM 里长期默认不生成、不完善 | **基本成立**，几乎拿不到 |
| **dot 指令**（`v_dot2_f32_f16` / `v_dot4_i32_i8`） | 必须显式 intrinsic（`__builtin_amdgcn_sdot2/4`），编译器不会从普通 FMA 自动合成 | **成立**，是典型的“必须手写 intrinsic 才吃得到” |
| **MFMA**（矩阵核） | intrinsic + rocWMMA 支持良好，CDNA 上最成熟的部分 | **不成立** |

所以更准确的说法是：**并不是“ROCm 没实现”，问题在于这些加速路径都是 opt-in（需要手动启用）**，代码不写成 2-wide 打包类型、不写 intrinsics，就只能跑到一半或四分之一。这和“编译器完全没有”是两件不同的事，但表现出来的症状一样：看 profile 会发现 VALU 吞吐只有理论值的一半。

---

## 顺手补一个容易忽略的代价

packed math 不是白捡的 2×：`v_pk_fma_f16` 每个源操作数要读 **64-bit**，一条指令的寄存器读带宽需求是标量的 2 倍。所以在**寄存器压力大**的 kernel 里（比如 attention 的 softmax 段），打包反而可能因为 RF 带宽/操作数对齐限制而拿不到收益，这也是为什么“理论上 2×”和实测之间总有差距。

---

## 修正版段落

> 在 AMD 上让 MFMA 和 VALU 重叠**不需要汇编**。三层路径：① 靠 occupancy 让每个 SIMD 常驻多个 wave，硬件自动交错，零改动；② C++ 层用 `__builtin_amdgcn_mfma_*` 配合 `sched_barrier` / `s_setprio` / `s_waitcnt` 做确定性交错——HipKittens 明确是“no raw assembly”，其 8-wave ping-pong 已在 GEMM 与 attention forward 上达到 SoTA 级；③ 只有榨最后几个百分点时才手写 asm（如 Tensile 直接生成汇编）。之所以要下到第③层，根源是 HIPCC 的寄存器分配不可控（无法让 MFMA 直接吃 AGPR），而不是重叠本身需要汇编。
>
> 关于 FP16 的“2×”，需要拆开看：**矩阵路径上 FP16 相对 FP32 是 4×（MI100）到 8×（MI300X），不是 2×**。真正的 2× 出现在**向量**路径，机制是 **packed math**（`v_pk_*`：一条 VALU 指令对 64-bit 源操作数做 2 路运算），而不是“双发射”；字面意义的双发射是 RDNA3 的 VOPD。
>
> 而 MI300X 的 2× 甚至把 FP32 自己也算进去了：AMD 报 SP = 19,456（64/CU）却给出 163.4 TFLOPS FP32 峰值，按 $304\times64\times2\times2\times2.1\text{GHz}$ 反推需要每 SP 每拍 2 次 FMA——只能靠 `v_pk_fma_f32` 达到。
>
> 至于“ROCm 不支持所以跑不到”，应当区分：`half2` → `v_pk_fma_f16` 是通的；真正的缺口是**自动向量化弱**（标量 f16/f32 循环不打包）、**dot 指令必须显式 intrinsic**、以及 **VOPD 在 LLVM 里长期默认不生成**。症状是“只有理论值的一半”，但原因是这些路径都是 opt-in，而非完全没有实现。

---

### 覆盖度说明

**本轮取到官方原值**：AMD MI300X 与 MI100 产品页的完整规格表（SP/CU/时钟/FP64/FP32/FP16/BF16/INT8 峰值），上面的所有比值和“每 CU 每时钟 FMA 数”都是基于这些数字算出来的，可自行验算。

**标注为推断的部分**：MI300X 的 128 FMA/clk/CU 到底由 packed FP32 还是物理加宽 SIMD 实现——AMD 未明写，我用 SP 数与 163.4 TFLOPS 的算术矛盾做了推断，并说明了另一种解释的可能性。

**未取到原文**：TechPowerUp 被反爬拦截、Wikipedia 再次 429，所以没有第三方交叉校验表；`v_pk_f16/f32` 的具体 ROCm 生成情况来自对 HIP/LLVM 的既有认知，未在本轮取到 LLVM 测试用例原文。如果要落成文章，建议用 `rocprof` 的 VALU busy 计数或 `llvm-objdump -d` 数 `v_pk_fma_f32` 出现次数来实测，比引述更硬。

---
## 🧑‍💻 User

看了下
1. 在 AMD 的数据中心卡 CDNA（比如 MI250 / MI300X）上，因为矩阵核（MFMA）和向量核（VALU）确实是两条物理独立的执行管线，理论上完全可以重叠。
但由于单 Wave 是按序发射（In-Order）且要抢 VGPR 读端口，LLVM 编译器自动排出来的指令往往一塌糊涂，所以逼得 AMD 生态里的高性能库（比如 Composable Kernel、HipKittens，还有 AMD 官方后来急眼了自己搞的 AITER 算子库）全都在手搓内联 AMDGCN 汇编，用 `__builtin_amdgcn_sched_barrier` 强行锁死每条 MFMA、VALU 和 LDS 访存指令的交替顺序，简直是在刀尖上绣花
2. 「双发射导致只跑到一半速度」其实是消费级的 RDNA3（7900 XTX 那代）的「双发射（Dual-Issue）与 FP16/WMMA 大坑」，具体有三个极其逆天的硬件与编译器细节：

1️⃣ PPT 上的算力翻倍，全靠苛刻到死的「VOPD 双发射」
RDNA3 每个 SIMD 单元其实塞了两组 32 通道的 ALU（共 64 个），苏妈据此宣传 FP32 算力从 RDNA2 的 23 TFLOPS 暴涨到 61 TFLOPS。
但坑爹的是，Wave32 每次只管 32 个通道！要想同时喂饱另外那 32 个 ALU，编译器必须把两条完全无关的指令打包成一条特殊的双发射指令（叫 VOPD）。
更要命的是：硬件只给了 4 个 VGPR 寄存器读端口，而两条乘加指令（FMA，a·b+c）一共需要 6 个源操作数！这就意味着：
• 两条指令必须恰好共用至少 2 个源寄存器；
• 它们的写入目标寄存器还必须「一奇一偶」（一个写偶数号 VGPR，一个写奇数号 VGPR，避免写端口 Bank 冲突）；
• 只有一小部分指令子集支持 VOPD。
结果 ROCm 的 LLVM 编译器面对这么变态的约束直接摆烂，大部分代码根本合不出 VOPD，导致另外那一半 ALU 长期处于断粮围观状态，实测算力直接腰斩！

2️⃣ 向量 FP16 的「倒退式翻倍」
在以前的 RDNA2 上，1 个 FP32 ALU 可以拆成 2 个 FP16 跑 Packed Math（`v_pk_fma_f16`），不用搞什么双发射就能轻松跑出 2 倍于 FP32 的速度。
到了 RDNA3，单发射一条 Packed FP16 指令只能喂饱第一组 ALU（刚好跑出 61 TFLOPS，跟纸面宣传的「双发射 FP32 峰值」一模一样！）。要想拿到宣传册上的 122 TFLOPS FP16 峰值，你必须把两条 Packed FP16 指令也凑成 VOPD 双发射（或者强行切回高延迟的 Wave64 模式）——一旦 ROCm 编译器没凑出双发射，你的 FP16 跑出来的 TFLOPS 就真跟纸面 FP32 一个速度了（）

3️⃣ 最绝的：RDNA3 的矩阵核（WMMA）居然是跟向量核（VALU）「共用肉身」的！
老黄的 RTX 显卡和 AMD 自家的 CDNA（MI300），矩阵单元（Tensor Core / MFMA）都是独立在向量 ALU 之外的专用硬件，所以前面咱们聊的「矩阵和向量同时重叠运行」才成立。
但 RDNA3 为了省晶圆面积，它的矩阵指令（`v_wmma_f16_16x16x16_f16`）压根没有独立的矩阵核心，而是直接征用了上面那两组常规 VALU 的乘加器（只是加了一点跨通道点积连线）！
这就导致两个惨剧：
• 第一，在 RDNA3 上跑 WMMA 矩阵乘的时候，VALU 已经被彻底占满了，物理上绝对不可能实现「一边跑 WMMA 矩阵乘、一边重叠跑 VALU 向量 Softmax」；
• 第二，早期 ROCm 连 RDNA3 的 WMMA 指令都没适配进主流库，大家跑 FP16 模型直接回退到没开双发射的普通 VALU，速度惨不忍睹，直到很后面社区和 hipBLASLt 才慢慢把 WMMA 补齐

---
## 😀 默认助手 (7)

这一轮涉及很多 RDNA3 的具体硬件细节，我先查一下公开资料，尤其想确认三件事：「SIMD 里到底有几组 ALU」「VOPD 的寄存器约束」和「WMMA 跑在什么硬件上」。

## 总判

这一版消息**骨架基本正确**，尤其 RDNA3 那三条抓住了真问题；但有 4 处需要修正：把“约束”说成了“精确规则”、把“编译器调度栅栏”说成了“硬件锁”、把三家库都算成“手搓汇编”，以及 wave64 那个括号。

---

## A. CDNA 那段（消息 1）：结论对，过程要改三处

<details>
<summary><b>✔ 成立的部分</b></summary>

MFMA 与 VALU 是两条物理独立的管线（CDNA3：304 CU × 4 matrix core = 1216 个矩阵核，与 SIMD 的向量 ALU 分开）；单 wave 按序发射；高性能库确实要靠调度手段把两者压到同一时间窗。方向完全对。
</details>

**❌ 修正一：不是“三家园子全在手搓汇编”**

| 库 | 实际情况 |
|---|---|
| HipKittens | 明确标榜 **“no raw assembly”**，是 C++ embedded + tile 抽象；只在 `s_waitcnt` / `s_barrier` 这类点上用 `asm volatile` 片段 |
| Composable Kernel | 以 C++ + `__builtin_amdgcn_mfma_*` 为主，调度靠 builtin，不是整体 asm |
| AITER | 混合体：CK kernel + Triton + 部分手写 asm，并非全 asm |
| 真正的 asm | **Tensile**（hipBLASLt 后端）直接生成汇编 kernel |

**❌ 修正二：`sched_barrier` 不是“强行锁死每条指令的交替顺序”**

它是**编译器侧的调度栅栏**（禁止调度器把指令跨过该点搬动），本质是 **hint**，不是硬件锁。而且更常用的其实是另外两个：

- `__builtin_amdgcn_sched_group_barrier(mask, count, group)` —— 按“指令类型配额”表达交错意图（“在这段里放 N 条 MFMA、M 条 VALU”），CK 的主力是它；
- `__builtin_amdgcn_s_setprio(1)` —— 临时抬高本 wave 的调度优先级，让 MFMA 尽快被发出去。

执行顺序最终仍由硬件的 wave 调度器决定，软件只是把它的选择空间压窄。

**❌ 修正三：真正逼人写 asm 的原因是寄存器分配，不是调度**

HipKittens 的原话是 **HIPCC 无法让 MFMA 直接拿 AGPR 当输入操作数**，只能被迫 `accvgpr_read/write` 来回搬并手动 pin 寄存器。**调度靠 builtin 就够，寄存器分配才需要绕过编译器。**

---

## B. RDNA3 三条：两条半对，一条存疑

### 1️⃣ VOPD 双发射：结构完全正确，但约束细节有偏差

**先给你补一个自洽性验算**（这个我很确定，也解释了“23 → 61”到底怎么来的）：

$$\underbrace{23\ \text{TF}}_{6900\text{XT}} \times \underbrace{1.2}_{\text{CU }80\to96} \times \underbrace{1.11}_{\text{2.25}\to\text{2.5GHz}} \times \underbrace{2}_{\text{双发射}} \approx 61.3\ \text{TF}$$

反向核验也对得上：官方规格里 7900 XTX 是 **96 CU / 6,144 stream processors**（Wikipedia 表格确认），而 $6144 \times 2 \times 2.5\text{GHz} = 30.7$ TF——**离 61 TF 正好差一倍**。所以“官方 SP 数 × 单发 = 一半峰值，剩下那一半必须靠 VOPD”这个判断是成立的。

**约束细节要改的地方：**

| 你的说法 | 更准确的说法 |
|---|---|
| 目标寄存器“一奇一偶” | 约束更强：两个目的 Vdst 必须是**相邻对**（Vdst / Vdst+1），因为 VOPD 的 64-bit 编码里目的字段只有一份，第二个是隐式 +1。⚠️ 这条我按 ISA/LLVM 实现的印象给出，本轮未取到 ISA 原文，建议核对 |
| 4 个读端口 vs 6 个源操作数 → 必须共用 ≥2 个源 | 推理成立，这也是 VOPD 匹配器要求 src 复用的根源 |
| 只支持一小部分指令子集 | 成立。VOPD 在汇编里长这样：`v_dual_fmac_f32 ... :: v_dual_mul_f32 ...`，只覆盖 VOPD 兼容 opcode |
| “另一半 ALU 长期断粮” | **成立**，而且这是最硬的一条：每个 SIMD 每拍只发**一条**指令，不生成 VOPD 就没有第二个发射槽，多塞 wave 也救不回来 |

**⚠️ 需要降级的一点**：LLVM “直接摆烂”说得过重。RDNA3 上市初期 LLVM 的 VOPD 成型确实不完善（后来才补上），Mesa/ACO 也是后补的；但简单 FMA 密集循环是能合出 VOPD 的。

### 2️⃣ FP16 的“倒退式翻倍”：逻辑成立，但 wave64 那个括号存疑

推导是干净的：

- **RDNA2**：32 个 ALU，一条 packed FP16 让每个 ALU 干 2 个 FP16 → 2× FP32，**不需要双发射**（23 → 46 TF）。
- **RDNA3**：单发 packed FP16 = 2 SIMD × 32 lane × 2 = 128 FP16 FMA/clk/CU，**正好等于“VOPD 双发 FP32”的 61 TF 那一档**；要摸到官方 ~123 TF 的 FP16 峰值，就得把两条 packed FP16 也凑成 VOPD。

所以“2× 从 RDNA2 的免费变成 RDNA3 的要 VOPD”这个判断是对的，症状描述（没合出 VOPD 就只有纸面 FP32 速度）也对。

⚠️ **但“或者强行切回高延迟的 Wave64 模式”这条我倾向否定**：RDNA3 上 wave64 指令在 SIMD32 上仍是按 2 拍展开执行，吞吐与 wave32 单发相同，拿不到第二组 ALU。Wikipedia 只说 RDNA “supports both 32 and 64”，没给宽度行为。这条建议你**实测确认**（写个 `a*b+c` 的 wave64 循环测 TFLOPS 最直接），我在本轮资料里找不到支持它的依据。

### 3️⃣ WMMA 与 VALU “共用肉身”：这条基本对，而且是关键洞察

- RDNA3 **没有独立矩阵核**，`v_wmma_f16_16x16x16_f16` 是靠 SIMD 的乘加阵列加跨 lane 点积连线实现的。因此 **WMMA 与 VALU 抢同一批 ALU，物理上不可能“一边 WMMA 一边 softmax”**——这个结论我认为站得住。
- 对照正确：CDNA 的 MFMA 是独立矩阵核（4/CU），NVIDIA 的 Tensor Core 也是独立单元，所以“矩阵+向量重叠”只在它们身上成立。
- “早期 ROCm 没适配、库回退到普通 VALU”也成立——ROCm 官方支持矩阵长期不含 RX 7000 系列，这是公开事实。

---

## C. 一张表把三家的“矩阵单元独立性”钉死

| | CDNA3 (MI300X) | RDNA3 (7900 XTX) | NVIDIA SM90 |
|---|---|---|---|
| 矩阵单元 | **独立 matrix core，4/CU** | **无**，复用 VALU | 独立 Tensor Core，4/SM |
| 双发射 | 无（SIMD16，每拍 1 条/wave） | **VOPD**（同拍两 op，约束极多） | 无（4 scheduler 各发 1 条） |
| 矩阵与向量能否并行 | **能**（跨 wave，硬件自动） | **不能**（同一批 ALU） | **能**（跨 warp/sub-partition） |
| 峰值算力是否需要编译器“配合” | 需要（要交错，但有 occupancy 兜底） | **强需要**（不合 VOPD 就一半） | 不需要（warp 级并行天然） |

一句话：**“矩阵/向量重叠”这个能力，CDNA 和 NVIDIA 是靠硬件结构给的，RDNA3 是靠编译器把两条指令塞进同一个发射槽换来的**——后者脆弱得多。

---

## D. 修正版段落（可直接用）

> **CDNA 侧**：MFMA 与 VALU 是物理独立的管线，重叠在硬件上可行；但把它压到峰值**不需要**手搓汇编。HipKittens 明确是 “no raw assembly”，CK 是 C++ + MFMA intrinsic，AITER 是 CK/Triton/部分 asm 的混合体；真正直接生成汇编的是 Tensile（hipBLASLt 后端）。调度靠 `sched_group_barrier`（按类型配额）和 `s_setprio`（抬优先级）这类 **builtin hint**，不是硬件锁；最终顺序仍由 wave 调度器决定。逼人下到 asm 的真正原因是 **HIPCC 的寄存器分配**（无法让 MFMA 直接吃 AGPR），不是调度本身。而且别忘了 occupancy 这条兜底：多 wave 常驻时，硬件本来就在替你做交错。
>
> **RDNA3 侧**：官方 96 CU / 6,144 SP 对应 $6144\times2\times2.5\text{GHz}=30.7$ TF，而宣传峰值是 61 TF——差额正好一倍，只能由 **VOPD 双发射**补上。约束苛刻（源操作数需复用、目的寄存器需相邻成对、指令子集受限），因此编译器合不出 VOPD 时，每 CU 的另一半 ALU 就长期空转，实测腰斩。
>
> **FP16 则是“倒退式翻倍”**：RDNA2 靠 packed FP16 白嫖 2×（每 ALU 干两个 FP16，不需要双发射）；RDNA3 上单发 packed FP16 只喂得饱一个 ALU 组，恰好等于“VOPD 双发 FP32”的 61 TF 档，要摸到 ~123 TF 的 FP16 峰值，必须把两条 packed 指令也凑成 VOPD。（所谓“切回 wave64 就能白嫖第二组 ALU”一说，我认为 RDNA3 上 wave64 仍是分两拍展开、吞吐与 wave32 单发相同，需要实测才能确认。）
>
> **最狠的一点是 WMMA 与 VALU 共用肉身**：RDNA3 没有独立矩阵核，`v_wmma_*` 直接征用 SIMD 的乘加阵列，所以在 RDNA3 上跑矩阵时 VALU 已被占满，**物理上不可能做“矩阵 + 向量 Softmax”重叠**。这正好反衬出 CDNA（4 个独立 matrix core/CU）与 NVIDIA（独立 Tensor Core）的价值——它们的重叠能力是硬件结构给的，RDNA3 的重叠能力是编译器塞双发射换来的。

---

### 覆盖度

**本轮确认到**：Wikipedia RDNA 页（96 CU / 6144 SP 的 Navi31 规格、RDNA 同时支持 wave32/wave64、GCN 每 4 拍发一条 vs RDNA 每拍一条）、AMD 官方 MI 系列规格（上一轮）、HipKittens 博客与 README 的原话。

**未能取到原文**：AMD RDNA3 ISA 参考手册（PDF，无法解析）——所以 VOPD 的寄存器约束我给的是“相邻成对”的版本，并明确标为待核对，你那条“一奇一偶”如果来自某篇二手分析，很可能是对“相邻”的误传；另外 7900 XTX 的 61 TF / 123 TF 我是用 6,144 SP 与 2.5 GHz 做算术反推（两项都能验算自洽），没有直接取到 AMD 产品页原值。

**建议实测项**：`v_dual_*` 是否被 LLVM 生成（`llvm-objdump -d` 数一下）、wave64 vs wave32 的 FP32 TFLOPS 对比、RDNA3 上 WMMA 与 VALU 混跑时的占用率。这三项一测，比任何二手资料都硬。

---
## 🧑‍💻 User

也就是说:
1. RDNA/CDNA是in order，想要全面利用它的硬体资源基本得手搓"ptx"
2. RDNA3的SIMD组数翻倍但其他资源没有翻倍，得手搓VOPD指令才能使用，并且对register要求离谱(共用两个src，且写入到不同bank的VGPR). 所以向量翻倍几乎不可达
3. RDNA3上的packed FP16也必须vopd，不然速度压根没提升
4. WMMA没有独立硬体，直接走矩阵乘加器...?

所以结合起来就是:
1. 两倍SIMD得vopd才能用(基本得手搓)
2. 两倍FP16得vopd才能用(基本得手搓)
3. 矩阵走上述路径
4. 也就是说在不手搓asm级别code的前提下，你很有可能只有1/4的标称FP16算力?
5. 官方的WMMA当然有把这部分处理好，但如果需要自己做fused kernel就很有可能惨不忍睹?

---
## 😀 默认助手 (7)

## 逐条判定

| 你的总结 | 判定 |
|---|---|
| ① in-order ⇒ 想全面利用硬件基本得手搓 “PTX” | **不成立**（两处错：in-order 不等于要手写 asm；AMD 是 AMDGCN ISA，不是 PTX） |
| ② RDNA3 SIMD 翻倍但其他资源没翻倍，得手搓 VOPD | **前半句是很好的洞察，后半句是范畴错误**：VOPD 是**后端自动生成的**，没人手写 |
| ③ packed FP16 也必须 VOPD，不然“压根没提升” | **不成立**：单发 packed FP16 相对标量 FP32 仍然是 2×；只是够不到**标称 FP16 峰值** |
| ④ WMMA 没有独立硬件 ⇒ “走矩阵乘加器” | **大方向对，表述要改**：WMMA 走 VALU + 额外点积连线，且**不走 VOPD** |
| ⑤ 不手搓就只有 1/4 标称 FP16 | **1/4 是最坏情况，典型是 1/2**；而且两档都不需要 asm |

---

## ① in-order ≠ 必须手搓

这两件事被绑在一起了，实际是三个不同的责任方：

| 事情 | 谁负责 |
|---|---|
| 指令排序、插入 `s_waitcnt`、延迟隐藏 | **编译器**（LLVM 后端 + 调度模型） |
| 填补执行空档 | **硬件 wave 调度器**（多 wave 常驻时零代价） |
| 寄存器分配、occupancy 上限 | **编译器 + 你的代码形状**（occupancy 由寄存器用量反推） |
| 才轮到人手 | 只有在“为了满寄存器而牺牲 occupancy”之后，才需要人肉交错 |

所以正确的说法是：**in-order 让“编译器的调度质量”变得比在 OoO CPU 上重要得多**，而不是“人得写汇编”。证据是库级别：HipKittens 是 “no raw assembly” 的 C++，8-wave ping-pong 已经打到 SoTA 级 GEMM / attention forward。真到 asm 的是 Tensile（hipBLASLt 后端）。

---

## ② “手搓 VOPD” 是范畴错误 —— 但你说对了一个更重要的东西

**先说对的部分**：RDNA3 只把 ALU 数量翻倍（每 SIMD 两组 32 通道），而**发射槽、VGPR 读端口、LDS、L1、wave 驻留数都没翻倍**。这个观察是准确的，而且这正是 VOPD 约束苛刻的根源——4 个读端口要喂 6 个源操作数，只能靠共用源寄存器。

**再说错的部分**：

> **VOPD 从来不建议人写，它是 LLVM / ACO 后端的指令打包（instruction packing）结果。**

汇编里它长成 `v_dual_fmac_f32 ... :: v_dual_mul_f32 ...`，由调度器把两条相容指令塞进一个 64-bit 编码。你不会为了“用上双发射”去手写 VOPD——全世界的 AMD 驱动开发者也没几个人这么干。**这是一个“编译器友好度”问题，不是“你要不要写 asm”的问题。**

所以准确的表述是：**你无法直接控制 VOPD 成型率，只能通过代码形状（源操作数复用、目的寄存器相邻、命中支持的 opcode 子集）提高它**。

另外补一条上次说过的：目的寄存器不是“写到不同 bank”，而是必须**相邻成对**（Vdst / Vdst+1），因为编码里目的字段只有一份。⚠️ 这条仍未取到 ISA 原文。

---

## ③ packed FP16 不是“没提升”，是“只到一半”

这个必须掰清楚，否则整条推理链会跑偏。以 7900 XTX（96 CU / 6,144 SP / 2.5 GHz）为例：

| 代码形态 | 实际 FP16 算力 | 占标称 ~123 TF |
|---|---|---|
| 标量 `_Float16`（不打包、不合 VOPD） | **30.7 TF** | **25%（=1/4）** |
| `half2` packed FP16，**单发**（合不出 VOPD） | **61.4 TF** | **50%（=1/2）** |
| `half2` packed FP16 + **成功合成 VOPD** | ~123 TF | ~100% |
| （参照）标量 FP32 + VOPD | 61.4 TF | — |

所以：

- **packed FP16 单发相对标量 FP32 仍然是实打实的 2×**，不是“压根没提升”；
- 它只是**相对“官方 FP16 峰值”只有一半**——这才是所谓“倒退式翻倍”的准确含义（RDNA2 上 packed FP16 单发就能顶满标称 46 TF，RDNA3 上不行了）；
- 而“1/4”恰好对应**连打包都没有**的最坏情况（标量 f16）。这是**真实存在的默认陷阱**，但它不是“必须”，因为你只要用 `__half2` 类型（普通 C++，不是 asm）就能到 1/2。

---

## ④ WMMA：对了一半，但“走上述路径（VOPD）”是错的

- ✔ **没有独立矩阵核**：RDNA3 的 `v_wmma_f16_16x16x16_f16` 确实复用 SIMD 的乘加阵列，加上跨 lane 的点积/归约连线。
- ✘ **但它不走 VOPD**：VOPD 只打包 VOP1/VOP2 类指令，WMMA 是独立的指令类，自己占多拍发射。**这两条路径互斥，不能指望 WMMA 吃双发射的红利。**
- 因此 WMMA 的含义是：用**一条指令**换掉一大堆 FMA + shuffle，省的是**指令数和寄存器压力**，而不是借用双发射。它的峰值受 ALU 数据通路限制（且实测高度依赖编译器/库）。

顺带纠正你第 3 条总结里的“矩阵走上述路径”：**矩阵和 VOPD 是两条平行路径，不是同一条。**

---

## ⑤ 所以“1/4”这个结论怎么答

**答案是：1/4 是最坏情况下的默认值，典型默认值是 1/2，而 1/2 完全不需要 asm。**

真正需要人做的是“调代码形状 + 祈祷后端合 VOPD”，不是写汇编。而且还有一层兜底：**RDNA3 的 WMMA 路径和 packed 路径都不是 VOPD 独占的**，你的 kernel 完全可以在别的维度上避开这个坑（比如走 WMMA，或干脆让 kernel 变成带宽瓶颈而不是 ALU 瓶颈——现实里绝大多数 fused kernel 都是带宽瓶颈）。

至于“官方 WMMA 处理好了、自己写 fused kernel 就惨不忍睹”——**这条对 RDNA3 反而过于乐观**：

- ROCm 官方长期不支持 RX 7000 系列，hipBLASLt 对 RDNA3 的覆盖很晚，谈不上“官方都处理好了”；
- 但“自己做就很惨”也不对——**llama.cpp 社区给 gfx1100 加 WMMA 路径后拿到的加速相当可观**，这正是“普通开发者绕开 VOPD 坑”的正面案例。

---

## 最后一句关键总结（也是 RDNA3 真正的教训）

RDNA3 的问题不是“你必须写汇编”，而是：

> **它把翻倍的算力藏在了一个“编译器必须恰好合出 VOPD 才能兑现”的编码格式里。**

对比 NVIDIA 的做法就更清楚：SM 有 4 个 warp scheduler，**每个每拍各发一条指令到各自的单元**，没有任何打包约束，编译器只要不制造依赖就自然吃到并行。RDNA3 则把“翻倍”押在单一发射槽的魔改编码上，杠杆在编译器手里，而不在程序员手里。

这也解释了为什么 **CDNA 走的是完全相反的路**：独立矩阵核（4/CU）+ 独立 packed 指令 + 高 occupancy，宁可让软件手搓调度（这是可控的），也不把性能押在一个脆弱的前端打包格式上。

---

### 覆盖度与待实测项

**本轮确认/推演**：7900 XTX 的 96 CU / 6,144 SP（Wikipedia）、RDNA 同时支持 wave32/wave64、四档算力阶梯都可以从 $6144 \times 2 \times \text{clock}$ 直接验算（30.7 / 61.4 / 123 与官方 61 / 123 自洽）。

**仍未取到原文**：RDNA3 ISA 手册（VOPD 精确约束）、AMD 官方 7900 XTX 规格页（61/123 TF 是我按 SP 数反推的）。AITER / CK 是否含手写 asm kernel 的分类来自既有认知，未逐一核对仓库。

**三条一测就清楚的实验**：
1. `llvm-objdump -d` 数 `v_dual_*` 出现次数 → VOPD 成型率；
2. 同一 FMA 循环分别用 `float` / `half2` / wave64 跑 → 直接量出四档阶梯；
3. 你的 fused kernel 里 `rocprof` 看 VALU busy 与 VGPR 用量 → 判断到底是 ALU 瓶颈还是寄存器压力导致的 occupancy 瓶颈（多数情况是后者，跟 VOPD 无关）。

---
## 🧑‍💻 User

我记得现在amdgpu的matrix core是mfma和wmma两套
一套内积一套外积 cdna是外积mfma，rdna是wmma
然后cdna4好像整了个wgmma类似物的大外积（？）
也不叫wgmma好像 是寄存器可以用多个方向上的拓展 不是wave的拓展

rocm和cuda啥区别来着
wave64 双发射 锁步执行 独立pc
甚至他自家的CDNA和RDNA的ROCm也不一样
RDNA是wave32的
CDNA有matmul但RDNA沒有 走的VALU RDNA4有3没（rdna3是走的valu，4是独立wmma cdna是外积mfma）

然后rdna4我自己用下来还有几个坑：没有带id的cta bar，没有异步拷贝，有mma但是没有ldmatrix/stmatrix
最神的是这玩意有setmaxnreg，但是没给cta bar导致想写wasp的话得用atomic手搓同步😰
mfma的m4n4k1 shape也很奇怪

---
## 😀 默认助手 (7)

这一条信息量很大，但里面混了不少记错的东西。我先获取一份按架构列出所有矩阵指令形状的资料。

## 逐条判定

| 你的说法 | 判定 |
|---|---|
| MFMA / WMMA 是“一套内积一套外积” | ❌ “内积/外积”不是这两者的区别轴 |
| CDNA 用 MFMA（独立矩阵核），RDNA3 的 WMMA 走 VALU | ✔ 准确 |
| RDNA4 有独立 WMMA 硬件 | ⚠️ 我倾向**没有独立矩阵核**，仍走 SIMD 数据通路（只是更宽 + 加低精度格式） |
| CDNA4 搞了个 wgmma 类似物，且是“寄存器多方向扩展、不是 wave 扩展” | ⚠️ 前半不成立，**后半这个描述反而很准** |
| RDNA4：没有带 ID 的 cta bar、没有异步拷贝、没有 ldmatrix/stmatrix | ✔ 前两条大概率对（待 ISA 核实），第三条对**所有** AMD 都成立 |
| RDNA4 有 setmaxnreg | ❌ 几乎肯定记错 |
| MFMA 的 m4n4k1 shape 很奇怪 | ✔ 确有其事，且确实诡异 |

---

## 一、MFMA 与 WMMA 的真实谱系

| 架构 | 矩阵指令 | 在哪执行 | 累加器 | 操作数来源 |
|---|---|---|---|---|
| GCN (gfx6–9) | 无 | — | — | — |
| **CDNA1–4** (gfx908/90a/942/950) | **MFMA** | **独立 matrix core**（4/CU） | **AGPR** | **VGRP** |
| RDNA1/2 (gfx10) | 无 | — | — | — |
| **RDNA3** (gfx11) | **WMMA** (+ SWMMAC 稀疏) | **SIMD/VALU 数据通路** | VGPR | VGPR |
| **RDNA4** (gfx12) | WMMA (+ SWMMAC + FP8) | 同上（我认为没有独立核） | VGPR | VGPR |
| NVIDIA SM90 | HMMA / wgmma | 独立 Tensor Core | 寄存器 | **wgmma 从 SMEM 直读** |

**“内积 vs 外积”这个轴不成立**：两者都是 $D=AB+C$ 的矩阵乘累加，硬件数据流本质一样（都是沿 K 累加一堆 rank-1 更新）。真正的区别是上表那四列：**谁执行 / 累加器在哪 / 形状集合 / 操作数从哪来**。

顺带说，最接近“纯外积”的反而是 CDNA 那个诡异的 `4x4x1`（K=1，等于 4 组 4×4 外积，完全没有 K 累加）——它恰恰是你后面吐槽的那个 shape，见第五节。

---

## 二、“CDNA4 那个 wgmma 类似物”

先把 **Hopper WGMMA 的定义性特征**列清，因为很多人把它简化成“大号的 MMA”：

1. **4 个 warp 协作**（warpgroup，128 线程）产出一个更大的 tile；
2. **操作数直接从 SMEM 取**，不经过寄存器文件；
3. **异步提交**（用 mbarrier 完成通知）。

**CDNA4（gfx950）实际新增的是别的东西**：更大的 MFMA 形状、FP4/FP6（MX 格式）、block scaling。它**没有** SMEM 操作数、**没有**跨 wave 协作、**没有**异步 MMA。所以“CDNA4 弄了个 wgmma 类似物”这个印象**不成立**。

但你括号里那句——“**是寄存器可以用多个方向上的拓展，不是 wave 的拓展**”——反而是全场最精准的一句：

> MFMA 从 CDNA1 起就是“**单个 wave 内的一张二维寄存器 tile**”，M/N/K 三个方向都在这张 tile 里面长；
> wgmma 是“**多个 wave 拼一个大 tile**”，用 wave 数量换尺寸。

所以 CDNA4 的路线是**把单 wave 的寄存器 tile 继续放大 + 加低精度**，而不是走向 wave 协作。这也解释了为什么 AMD 至今没做 wgmma 那种东西：它的寄存器预算模型（静态分配、每 wave 固定 VGPR/AGPR 配额）根本不支持“几个 wave 合起来算一个 tile”。

---

## 三、ROCm vs CUDA：真正的结构性差异

| 维度 | AMD (ROCm/HIP) | NVIDIA (CUDA) |
|---|---|---|
| 执行宽度 | wave64（CDNA/GCN）／wave32（RDNA 默认，也可 64） | warp32 |
| 分叉 | **EXEC 掩码 + 编译器软件栈**，无硬件重汇聚栈 | Volta+ per-thread PC + 硬件管理 |
| 通用寄存器 | **VGPR + SGPR（标量）+ AGPR（CDNA 累加器）** | 全 per-thread |
| 等待/同步 | `s_waitcnt vmcnt/lgkmcnt/expcnt`（**显式计数器**）+ `s_barrier` | scoreboard/DEPBAR + `bar.sync` + **mbarrier** |
| 异步搬运 | `buffer_load ... lds`（直入 LDS），粒度粗 | `cp.async` → **TMA**（描述符驱动） |
| 矩阵取数 | **没有 `ldmatrix`/`stmatrix`**，靠 `ds_read_b128` + 手算 swizzle | `ldmatrix`/`stmatrix` 一条到位 |
| 寄存器重分配 | **没有**（静态分配，wave specialization 因此失效） | Hopper 有 `setmaxnreg` |
| 中间表示 | 无 PTX 层，LLVM IR 直接到 ISA（**必须按 gfx target 编译**） | **PTX** 虚拟 ISA，前向兼容 |
| 队列/模型 | workgroup + LDS（64KB/CU 量级） | CTA + smem + cluster (SM90) |

**“没有 PTX 层”这一点特别值得强调**：CUDA 靠 PTX 做到“编译一次、未来架构还能跑”；ROCm 没有这层，二进制必须匹配具体 gfx 目标（fat binary / 现场重编译）。这是两个生态在工程体验上差距的一大来源，比指令集差异更致命。

---

## 四、RDNA4 那几个坑，逐条核

<details>
<summary><b>✔ 没有 ldmatrix / stmatrix —— 对，而且是全体 AMD</b></summary>

这个不是 RDNA4 的坑，是 AMD 一直以来的坑：**整个 GCN/CDNA/RDNA 都没有矩阵载入指令**。你得自己 `ds_read_b128`/`ds_write_b128` 加上手推的 XOR swizzle 才能喂 WMMA/MFMA。HipKittens 那篇博客有一整段在讲“AMD 的 layout 没法用统一 swizzle 规则，因为 `ds_read_b128` 和 `ds_write_b64` 的 phase 分组不一样”——这就是没有 ldmatrix 的直接后果。
</details>

<details>
<summary><b>✔ 没有带 ID 的 CTA barrier —— 大概率对</b></summary>

AMD 传统上 `s_barrier` 就是**整个 workgroup 一个栅栏**，不像 NVIDIA 有 16 个命名 barrier 可以做 producer/consumer 分离。gfx942 起有了 `s_barrier_signal` / `s_barrier_wait`（带 ID），**gfx12 是否具备我这轮没能核实**。如果没有，你说的“想写 warp specialization 只能拿 atomic 手搓同步”就是真实处境。
</details>

<details>
<summary><b>✔ 没有真正的异步拷贝 —— 对，但成因要说准</b></summary>

AMD 的“异步”是 `buffer_load ... lds`（直入 LDS，绕过寄存器），**粒度粗、没有描述符、没有独立完成队列**，同步还是靠 `s_waitcnt` 数计数器。它不是 TMA 的等价物，也不能像 `cp.async` 那样和 mbarrier 精细挂钩。
</details>

<details>
<summary><b>❌ 有 setmaxnreg —— 这条我强烈怀疑</b></summary>

`setmaxnreg` 是 **Hopper** 的指令，它之所以存在是因为 SM90 有 **register reallocation 硬件**（warp 之间可以让渡寄存器）。AMD 据我所知**至今没有这个能力**——而且这不是细节，是 HipKittens 明确写进论文/博客的结论：*wave specialization 在 AMD 上失效，原因就是缺少 register reallocation，寄存器在 wave 间静态切分，producer wave 用不完的寄存器 consumer 拿不到*。

如果你确实在 gfx12 上见过这个 opcode，给我全名，我去核；但更可能是和 `s_setprio`（调调度优先级）或别的指令记混了。
</details>

<details>
<summary><b>✔ m4n4k1 真的很怪 —— 而且我知道它为什么存在</b></summary>

`v_mfma_f32_4x4x1_f32` 这类 K=1 形状确实存在（CDNA1 起就有，同族还有 `16x16x1`、`32x32x1`、`4x4x4`(f16) 等）。为什么怪：

- **K=1 意味着完全没有 K 累加**，它就是 4 组独立的 4×4 外积，本质是个“向量式”操作被硬塞进矩阵指令格式；
- 它是 CDNA1 为了让**最小的 tile 也能走 MFMA 通路**而留的兼容形状，实际 GEMM 没人用——因为一条指令只算 $4\times4\times1\times2\times4=128$ flops，却要占用一整个 matrix core 的发射/访存开销，效率极低；
- 结果就是 AMD 的形状谱系又长又怪（K=1/2/4/8/16/32/64/128 混着 M/N=4/16/32），和 NVIDIA 那种“就 16x8x16、64x256x16 几个干净形状”完全两种风格。
</details>

---

## 五、修正版总结

> **矩阵指令族谱**：GCN 没有矩阵指令；CDNA1–4 用 **MFMA**，跑在**独立 matrix core**（4/CU）上，累加器是 **AGPR**，操作数在 VGPR；RDNA1/2 没有；RDNA3/4 用 **WMMA (+SWMMAC)**，跑在 **SIMD/VALU 数据通路**上（我认为 RDNA4 也没有独立矩阵核，只是更宽 + 加 FP8），累加器和操作数都在 VGPR。所以“内积 vs 外积”不是它们的分界线——两者都是 $D=AB+C$，真正的区别是**谁执行、累加器在哪、形状集合、操作数从哪来**。
>
> **CDNA4 并没有做 wgmma**。Hopper WGMMA 的三个定义性特征是“4 warp 协作 + 操作数直读 SMEM + 异步提交”，CDNA4 一个都没有；它新增的是更大形状、FP4/FP6、block scaling。但“**是寄存器在多个方向上的扩展，不是 wave 的扩展**”这个描述非常准确：MFMA 从 CDNA1 起就是单 wave 内一张二维寄存器 tile 往 M/N/K 三个方向长，而 wgmma 是用 wave 数量去换 tile 尺寸。AMD 的静态寄存器模型根本容不下后者。
>
> **ROCm 与 CUDA 的差异**里，最要害的两条不是指令集，而是：① **AMD 没有 PTX 那一层虚拟 ISA**，二进制必须匹配具体 gfx 目标；② **没有寄存器重分配**，所以 NVIDIA 靠 warp specialization 拿到的东西在 AMD 上要换别的范式（HipKittens 的 8-wave ping-pong / 4-wave interleave）。
>
> **RDNA4 的坑**：没有 `ldmatrix`/`stmatrix` 是全体 AMD 的老问题（得自己 `ds_read_b128` + 手推 swizzle）；没有带 ID 的 CTA barrier、没有真正的异步拷贝（只有 `buffer_load ... lds` + `s_waitcnt`）大概率成立；但 **setmaxnreg 几乎肯定是记混了**，AMD 至今没有寄存器重分配。至于 `m4n4k1` 那种诡异形状，是 CDNA1 为了让最小 tile 也能走 MFMA 而留下的兼容尾巴，GEMM 从来不用。

---

### 覆盖度

**本轮取到**：AMD Matrix Instruction Calculator 的 README（确认其覆盖 MFMA 与 RDNA3 的 WMMA 两条路径、可生成形状/吞吐/寄存器映射）、Wikipedia RDNA 表格（96 CU/6144 SP、wave32+64）。

**未取到原文**：CDNA4 (gfx950) 的确切新增形状清单、gfx12 的 barrier/async-copy 指令表、RDNA4 是否有独立矩阵核——这三项我都是按既有认知 + 推理给出，并已逐条标注置信度。`setmaxnreg` 那条我给了明确否定，但保留“给我 opcode 我再核”的口子。

**建议的硬验证手段**（比任何二手资料都可靠）：`llvm-mc -arch=amdgcn -mcpu=gfx1200` 直接汇编试探指令名（`s_barrier_signal`、`buffer_load_dword lds`、`v_wmma_*` 形状），或翻 `llvm/lib/Target/AMDGPU/*Instructions.td` 里的 `Predicate` 列表——gfx 特性开关在那里是一目了然的。
