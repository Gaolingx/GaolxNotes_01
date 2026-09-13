# 移动GPU架构局限

### 🧑‍💻 User

从command processor、tlb cache、fixed unit（primitive distributor/vertex fetch/idvs/dvs）、structured hazard（scoreboard、bank、port）等角度解析一下为什么mobile不适合GPU Driven rendering和密集型GPGPU负载（HPC/ML）？

---

### 🤖 Assistant

这个问题的核心在于:**移动 GPU 采用 Tile-Based（TBR/TBDR）架构 + 严格功耗预算**,而桌面 dGPU 采用 Immediate Mode（IMR）+ 充裕带宽/缓存。GPU-Driven Rendering 和密集 GPGPU（HPC/ML）都需要「大前端吞吐、低惩罚的地址翻译、可预测/高吞吐的固定单元、以及能隐藏长延迟的调度资源」——这四点正好是移动架构的结构性短板。

> 说明:我检查了当前可用工具集(浏览器、Python、Fetch 等),没有与本问题相关的检索型工具,以下基于 GPU 体系结构知识作答。

---

# 0. 两种范式与两类负载的前提

| 维度 | 桌面 dGPU (IMR) | 移动 GPU (TBR/TBDR) |
|---|---|---|
| 渲染模型 | Immediate Mode,逐 primitive 用完整屏幕 | 先 binning 写 tile list,再逐 tile 渲染 |
| 带宽 | 500 GB/s ~ 3 TB/s (GDDR6/HBM) | 50 ~ 100 GB/s (LPDDR5) |
| 命令前端 | 多硬件队列、并发 kernel、device-generated commands | 单/少队列、binning 串行化、间接命令受限 |
| 隐藏延迟的算力 | 海量 warp/线程并发 | 受 RF 与功耗约束的有限并发 |

**GPU-Driven Rendering** 要求:GPU 自己生成/消费 draw/dispatch、cluster culling、bindless 资源散落内存、meshlet/mesh shader。
**密集 GPGPU(HPC/ML)** 要求:大寄存器堆、高带宽、强 scoreboard、低 bank conflict、强原子、FP64/Tensor 吞吐。

这两类负载的共同特征:**把压力从「人写死的、可预测的图元流」转移到「运行时生成的、分散的、高并发的内存与计算流」**——而移动架构恰恰是为前者优化的。

---

# 1. Command Processor(命令处理器):binning 串行化是致命点

<details>
<summary><b>展开:移动 CP 的结构性缺陷</b></summary>

1. **Binning 强制串行** —— TBR 必须先跑完所有顶点/tiler,把 primitive 分发到各 tile 的 tile list,之后才能跑 fragment。GPU-Driven 的「compute 生成 draw → 图形消费」链路因此变成 **cull kernel → 写间接参数 → 图形 draw → 再 cull** 的串行环,无法像桌面那样在同一个 work-graph 内流水重叠。屏障与可见性每帧多次打断重叠。

2. **缺乏 device-generated commands 硬件支持** —— 多数移动 GPU 对 `VK_EXT_device_generated_commands` / `ExecuteIndirect` 支持受限或纯软件模拟,命令生成吞吐远低于 GPU 本体需求。GPU-Driven 的核心是「用 compute 喂出成千上万个小 draw/dispatch」,移动 CP 的前端(draw dispatch rate)与提交开销扛不住。

3. **无 mesh shader / work graph** —— 现代 meshlet GPU-Driven 管线依赖这些,移动端普遍缺失,只能退化成传统 vertex fetch + many small indirect draws,反而放大了前端瓶颈。

4. **异步计算/多队列受限** —— 无法用并发的 compute 队列遮盖 graphics 队列的空洞;移动常是单一时间线。

5. **能耗固定开销** —— 唤醒前端 + 跑一遍 binning 的能耗是刚性的,加重功耗墙。

**结论:** CP 的 binning 串行 + 前端吞吐不足,使 GPU-Driven 的「自生成命令流」在移动上无法闭环。
</details>

---

# 2. TLB / MMU:小页表 + 小 TLB → 地址翻译抖动

TLB 覆盖范围:

$$R_{\text{TLB}} = N_{\text{entries}} \times S_{\text{page}}$$

| | 桌面 | 移动 |
|---|---|---|
| L2 TLB 条目 | 数千 | 数百 |
| 大页 | 2 MB (可用) | 4 KB / 64 KB 为主,大页支持弱 |
| 有效覆盖 | GB 级 | 数十~数百 MB |

- **GPU-Driven 的 bindless 与稀疏资源**把描述符、indirect buffer、cluster 数据、tile list 打散在物理内存各处 → **working set ≫ TLB reach → TLB thrash**,每次 miss 触发页表遍历(多级),延迟难以隐藏。
- **TBR 的 tile list 读写**本身就在逐 tile 反复访问分散内存,翻译压力叠加。
- **统一内存 ≠ 零成本**:移动 MMU 更简单、页表缓存更小,GPU-Driven 每帧更新 buffer 导致的页表/一致性开销更痛。
- HPC/ML 的稀疏张量、embedding lookup 等**不规则访问**同样吃满 TLB。

**结论:** 地址翻译子系统是为「少量、连续、可缓存」的访问优化的,GPU-Driven 与稀疏 ML 的「海量、分散」访问正好打在其软肋上。

---

# 3. 固定单元:primitive distributor / vertex fetch / IDVS / DVS

<details>
<summary><b>展开:各固定单元的吞吐与功耗约束</b></summary>

1. **Primitive Distributor / Tiler** —— 在 tiled GPU 中,transform 后的顶点进入 primitive distributor,按 tile 归属写出 per-tile primitive list。它的**每周期图元吞吐**与 **tile list 写带宽**是硬上限。GPU-Driven 产出更多小图元、更分散的 draw,binning 成本不降反升;移动的图元率远低于桌面。

2. **Vertex Fetch (VFD)** —— 属性 fetch 槽位有限、吞吐低、每顶点能耗高。Meshlet/cluster 方案即使做了 cull,存活 cluster 仍要 vertex fetch,移动内存子系统喂不动。

3. **IDVS(Index-Driven Vertex Shading,Arm Mali)** —— 把顶点着色拆成 **position-only shading**(用于 cull/binning)与 **varying shading**(只对存活的图元),避免重复顶点着色。**前提是几何由干净的 index buffer 驱动、可预测**。GPU-Driven 用 compute 动态生成 meshlet/几何,破坏了这个假设:索引发散、两阶段收益被侵蚀,同时两遍调度给固定单元加固定压力。

4. **DVS(Deferred Vertex Shading 类延迟顶点调度)** —— 同为「延迟 + 复用」思路,收益依赖局部性与可预测性;动态生成的 GPU-Driven 几何让复用率下降,延迟调度反而增加 bookkeeping 开销。

5. **无硬件细分 / 细分吞吐低** —— 很多移动 GPU 无曲面细分单元或吞吐有限,而 GPU-Driven 常用 subdivision/meshlet 技术。

6. **Setup / Culling / Clipping 吞吐低** —— 图元洪水直接打在三角 setup、裁剪、背面剔除这些固定单元上,移动每周期处理数少。

**核心矛盾:** 这些固定单元是为「可预测、index 驱动、图元数适中的几何流」设计的优化;GPU-Driven 恰好制造「动态、发散、海量小图元」的流,把优化变成负担。
</details>

---

# 4. Structured Hazards:scoreboard / bank / port

这是 GPGPU 在移动上最致命的一环——**延迟隐藏能力**。

## 4.1 Scoreboard(记分牌 / 依赖跟踪)

隐藏内存延时的最小在途请求:

$$B_{\text{inflight}} \ge \text{BW} \times L_{\text{mem}}$$

移动 DRAM 延迟与桌面相近(~数百 cycle / ~几百 ns),但**带宽低一个数量级**且 **scoreboard 条目、warp 并发数少** → 一旦发生 miss,没有足够的独立 warp/在途 load 去填补 → 流水线饿死。桌面 SM 用大 scoreboard + 高 warp 数掩盖;移动做不到,又受功耗约束不能把所有资源拉满。

## 4.2 Bank Conflicts(共享内存银行冲突)

- 移动的 threadgroup/共享内存**容量小、bank 组织简单、缺少高级 swizzle/硬件避冲突**。
- GEMM 的 tile 加载、转置、reduction、stencil 等常见模式在移动上频繁串行化。
- 桌面 LDS 有更大的 bank 数 + 布局优化,冲突惩罚小得多。

## 4.3 Port Conflicts(端口争用)

- **LSU 发射端口少**,LDS / global / 纹理可能共享端口,load/store 与共享内存流量互相争用。
- **L1/L2 端口与带宽窄**,HPC/ML 的 memory-bound kernel(如 attention、elementwise、reduce)直接被带宽墙卡死。

用 Roofline 看:

$$P = \min\big(\pi,\ \beta \cdot I\big)$$

其中 $\pi$ 为峰值算力,$\beta$ 为带宽,$I$ 为算术强度。移动的 $\pi$ 与 $\beta$ 都低,且在大 $\pi$ 类负载(GEMM)上又被寄存器堆/共享内存容量限制 occupancy——**算力、带宽、占用率三头都不够**。

## 4.4 其他结构化瓶颈

| 资源 | 移动的约束 | 对负载的影响 |
|---|---|---|
| 寄存器堆 | 小,限制 occupancy | GEMM/ML 大寄存器 kernel 跑不起来 |
| Barrier | 吞吐低、workgroup 尺寸受限 | reduction/scan/协作 kernel 受限 |
| Atomics | 单元少、吞吐低 | GPU-Driven 原子计数、稀疏 ML 卡顿 |
| L2 | 小,端口少 | 复用性差的 kernel 全打到 DRAM |

---

# 5. 数值与功耗:压垮骆驼的最后一根稻草

- **FP64**:多数移动 GPU 无 FP64 或速率极低(1/16~1/64 或无),HPC 双精度直接出局。
- **Tensor/矩阵单元**:移动普遍缺少桌面级 HMMA,NPU 与 GPU 分离,不适合灵活 ML 训练;即便有 INT8 NPU,也难做密集 FP16/FP32 训练。
- **内存能耗**:移动的墙不是算力而是 **energy/bit**,访问 DRAM 的能量比 register 高两个数量级;GPU-Driven 与 GPGPU 都在**提高内存流量与计算密度**,直接撞功耗墙。
- **热节流**:持续负载下移动 GPU 常只能维持突发频率的 1/3 左右,而 GPU-Driven 每帧都要跑完整 cull+binning,持续压力导致降频。

---

# 6. 总结对照表

| 角度 | 桌面 dGPU 表现 | 移动 GPU 表现 | 对 GPU-Driven / GPGPU 的后果 |
|---|---|---|---|
| Command Processor | 多队列、device-generated、work graph | binning 串行、间接命令受限 | 无法闭环自生成命令流 |
| TLB/MMU | 大 TLB + 2MB 大页 | 小 TLB + 4/64KB 页 | 分散访问 TLB thrash |
| Primitive Distributor/Tiler | 高图元率 | 图元率低、tile list 带宽紧 | 小图元洪水打爆 binning |
| Vertex Fetch | 高吞吐、属性槽多 | 吞吐低、能耗高 | meshlet 存活簇喂不动 |
| IDVS/DVS | 无此机制(IMR) | 依赖可预测 index 流 | 动态几何破坏延迟收益 |
| Scoreboard | 大、高并发 | 小、warp 少 | 长延迟无法隐藏 |
| Bank Conflict | 高级 swizzle 避冲突 | 简单 bank、易冲突 | GEMM/reduce 串行化 |
| Port/BW | 端口多、带宽高 | 端口少、带宽低 | memory-bound kernel 卡死 |
| 数值/功耗 | FP64/Tensor、散热充裕 | 缺 FP64、功耗墙 | HPC/ML 训练不可行 |

**一句话结论:** 移动 GPU 的每一处「省电设计」(前端的 binning 串行、小 TLB、低吞吐固定单元、精简 scoreboard/共享内存/端口)都建立在**几何可预测、访问局部化、带宽压力可控**的假设之上;而 GPU-Driven Rendering 与密集 GPGPU 恰恰把负载推向**动态生成、内存分散、长延迟、高带宽密度**的反面,于是每一层都从「优化」变成「瓶颈」。

如果你想,我可以进一步:
- 用具体芯片参数(Adreno / Mali / Apple / NVIDIA)做定量对比;
- 或画一张 GPU-Driven 在 TBR 上的执行时间线,展示 binning 串行化如何吃掉 overlap。

---

### 🧑‍💻 User

移动端只有apple gpu的微架构稍微跟得上nv/amd；其它的老实做好cpu driven是更好的选择。
比如说adreno光一个fp指令延迟都奔着6-7 cycle走了，它ilp空间还极其有限（因为wave少，主要是wave size太大）。
mali又太少，然后前端调度又不够，ipc发射率跟不上。
就拿nvidia ampere当baseline吧（几年前发布的产品），从ipc角度来看（同等时间/频率下，执行效率），mobile和desktop/console之间的差距还很大吗？

---

### 🤖 Assistant

工具集里只有浏览器/Python/Fetch 这类通用工具,没有专门的 GPU 规格库,而这个问题是纯微架构效率分析,我直接用体系结构知识回答(若要精确到某颗芯片的实测 IPC,可以再用 Fetch 去抓 whitepaper)。

---

# 先把「IPC」定义清楚,否则比较没有意义

GPU 的 IPC 必须分层看,单纯说「IPC」会误导。真正决定等效执行效率的是下面这条链:

$$\text{有效 IPC} = \min\Big(\underbrace{W_{\text{issue}}}_{\text{发射宽度}},\ \underbrace{\frac{N_{\text{warp}} \cdot \text{ILP}}{L_{\text{op}}}}_{\text{延迟受限项}}\Big)$$

- $W_{\text{issue}}$:每周期能发射多少条 warp 指令(前端/scheduler 能力)
- $N_{\text{warp}}$:residency(驻留 warp 数)→ 由 RF、wave slot、共享内存决定
- $L_{\text{op}}$:指令延迟(尤其 FP/内存)
- $\text{ILP}$:单 warp 内可并行的独立指令数

**关键洞察:桌面 GPU 几乎永远跑在「发射受限」区(好),移动 GPU(尤其 Adreno)常掉进「延迟受限」区(坏)。** 这就是你观察到的现象的根因。

---

# 1. Ampere 作为 baseline:它是「延迟极其好藏」的设计

| 参数 | GA10x (GA102/GA104) |
|---|---|
| Scheduler / SM | 4 个,每周期各发射 1 条 warp 指令 |
| 峰值发射 | **4 warp-instr/cycle/SM** |
| Warp size | 32 |
| FP32 数据通路 | 128 lane/SM(2×64,双 FP32 发射) |
| 峰值 FP32 | 128 thread-op/cycle/SM |
| RF | 64K × 32-bit = 256 KB/SM |
| 最大驻留 | 64 warps = 2048 threads/SM |
| FMA 延迟 | **~4 cycle** |

延迟受限项:

$$\text{IPC}_{\text{lat}} = \frac{N_{\text{warp}}\cdot \text{ILP}}{L} = \frac{64 \times 1}{4} = 16 \gg 4$$

即 **Ampere 光靠 TLP 就能把发射口喂爆,有 4× 余量**。这就是为什么 Ampere 上 dense kernel 能轻松打到 60–85% 的发射利用率(实测 IPC ≈ 2.4–3.4 / 4)。4-cycle 延迟 + 64 warp 驻留,是一个「几乎不用管 ILP」的甜蜜点。

---

# 2. Adreno:你的观察完全成立,这是「延迟受限」的教科书案例

$$N_{\text{warp}}\ \text{小},\quad \text{wave}=64\!\sim\!128,\quad L_{\text{FP}}\approx 6\!-\!7$$

把数字代进延迟受限项:

$$\text{IPC}_{\text{lat}} \approx \frac{N_{\text{warp}}\cdot \text{ILP}}{7}$$

只要 $N_{\text{warp}}$ 是十几个、ILP 又因**寄存器压力 + 大 wave 调度窗口有限**而 ~1,那么 $\text{IPC}_{\text{lat}}\approx 1\!\sim\!2$,**直接压在发射宽度之下 → 发射口永远填不满**。这就是「FP 延迟趋向 6–7 cycle、ILP 空间又被 wave 数吃掉」的量化含义。

为什么会这样,是设计取舍而非失误:

<details>
<summary><b>Adreno 为什么这么设计(面积/功耗权衡)</b></summary>

1. **深流水线换频率/面积**:加流水级能降每级 critical path、提高时钟、降低电压/功耗。代价是 $L$ 变大。移动要省电 → 只能深流水 → 延迟变长。
2. **大 wave(64–128)降低调度器开销**:wave 越大,scheduler 每条指令的 bookkeeping 越省,但**驻留 wave 数(而不是 thread 数)骤减** → TLP 变少。
3. **寄存器堆相对小**:大 wave × 高寄存器 → wave slot 很快耗尽 → $N_{\text{warp}}$ 更低。
4. **省电的 scoreboard**:条目少、跟踪粒度粗 → 更难用多 warp 交错填空洞。

三重打击叠加:**延迟更长 + 驻留更少 + ILP 窗口更窄**,结果就是一个数字上「lane 数还行、但有效 IPC 只有一半」的机器。
</details>

---

# 3. Mali:问题不在「延迟」而在「前端发射率」

Mali(Valhall/Midgard)是另一条失败路径:

- **warp = 16**,极细粒度。同样的工作量需要**更多 warp 指令**(lane 数固定时指令数翻倍),对发射口压力更大。
- **每 core 的执行引擎/发射很浅**(Valhall 约 2 个 execution engine),前端与调度深度不足 → 你描述的「前端调度不够、IPC 发射率跟不上」本质是**发射受限($W_{\text{issue}}$ 本身就低)**。
- core 数量少、shader core 规模小 → 单位时钟能推进的 thread-op 本就不高。

所以 Mali 和 Adreno 是**两种相反的失败**:

| | 受限类型 | 根因 |
|---|---|---|
| Adreno | **延迟受限** | 高 $L$ + 大 wave + 少驻留,发射口喂不饱 |
| Mali | **发射受限** | 16-wide warp + 浅前端,发射宽度本身低 |

两者殊途同归:**有效 IPC 都落在 30–60% 区间**。

---

# 4. Apple:为什么它的微架构确实「跟得上」

Apple GPU 罕见地同时满足了三件事:

- **warp = 32**(和 NVIDIA 同粒度 → 调度/占用模型一致)
- **大寄存器堆 + 深驻留**(TLP 充足 → 能进发射受限区)
- **宽发射 + 统一内存 + 强前端**(命令/前端开销低)

结果它的 $\text{IPC}_{\text{lat}}$ 和 $W_{\text{issue}}$ 都接近桌面,是移动里**唯一结构上进入「发射受限区」**的。这验证了你的判断:**除 Apple 外,其它移动 GPU 的瓶颈是结构性的,不是「再优化驱动就能补上」的。**

---

# 5. 定量对比(同频率、同 SM/CU 数量下)

下面是我给出的**量级估计**(同频、归一化到各自发射峰值):

| 架构 | Warp | 峰值发射/SM | FP 延迟 | 延迟余量 | 实测/可达 IPC 利用率 | 受限类型 |
|---|---|---|---|---|---|---|
| **NVIDIA Ampere (GA10x)** | 32 | 4 | ~4 | 4× | **60–85%** | 发射受限(健康) |
| **Apple (M1/A15+)** | 32 | 宽 | 中 | ~2–3× | **50–70%** | 接近发射受限 |
| **AMD RDNA2 (console)** | 32/64 | 2/CU(64 lane) | ~4 | 2–4× | **50–75%** | 发射受限 |
| **Adreno 6xx/7xx** | 64–128 | — | 6–7 | <1× | **35–55%** | **延迟受限** |
| **Mali Valhall/G78** | 16 | 浅 | 中 | ~1–2× | **30–50%** | **发射受限** |

> 注:这些是量级估计,不是某颗芯片的实测值;具体 kernel 差异巨大(GEMM 与不规则 kernel 能差 2×)。

---

# 6. 直接回答:同频下差距还大吗?

分三层回答,才是诚实的:

<details>
<summary><b>① 纯 SIMT 核的 IPC 差距:约 1.5–2×</b></summary>

- Ampere ~60–85% 发射利用率 vs Adreno/Mali ~35–55%;
- 归一化后 **有效 IPC 差 1.5–2 倍**。
- **不是数量级差距**,而且「latch 到 ALU」的原始算力其实差不多——差的是**把 lane 喂满的能力**。
</details>

<details>
<summary><b>② 系统级执行效率差距:约 2–4×</b></summary>

把上一轮讨论的东西加进来会继续放大:
- **前端/binning 串行化**(移动)↔ 多队列并发(桌面);
- **小 TLB + 小缓存**在分散访问下的 miss 惩罚;
- **带宽差一个数量级** → memory-bound kernel 直接按带宽比吃亏;
- 这些都**不体现在「IPC」里**,但体现在真实吞吐里。
</details>

<details>
<summary><b>③ 绝对吞吐差距:更大,因为还有频率与功耗墙</b></summary>

- 移动常跑 ~0.7–1.0 GHz 可持续,Ampere boost ~1.4–1.8 GHz;
- 同架构同 IPC,频率再乘 ~1.8×;
- 叠加 ①②,真实「同时间」吞吐差距往往 **4–8×/SM** 级,再乘 SM 数量差距。
</details>

**一句话:** 如果只看「同频、同大小的 SIMT 核心内在执行效率」,移动(除 Apple)比 Ampere 差 **约 1.5–2 倍**,没有数量级那么夸张;**但这个差距是结构性的、靠驱动优化补不回来的**,而且一旦落到真实负载(带宽/前端/延迟),会扩大到 2–4 倍甚至更多。

---

# 7. 回到你的结论:为什么「CPU driven 更划算」

你的判断可以从上面的模型直接推出:

1. **对 Adreno/Mali**:GPU-Driven 和 GPGPU 都要求「高 TLP / 低惩罚延迟 / 强前端」——这正是它们的天生短板。让 CPU 去组织 draw(CPU 的 OoO 乱序引擎延迟隐藏能力远超移动 GPU 的 scoreboard),GPU 只干「喂饱就行」的简单活,**反而扬长避短**。
2. **对 Apple**:结构上进了发射受限区,才有资格接住 GPU-Driven 的负载。
3. **Console(PS5/XSX = RDNA2)**:介于两者之间——warp32 + 高带宽 eSRAM 缓解带宽,**比移动更早能玩 GPU Driven**,但比 Ampere 仍略吃亏于发射/占用。

**工程结论:** 是否上 GPU-Driven / 重 GPGPU,应该用「$N_{\text{warp}}\times\text{ILP}/L$ 是否 $\ge W_{\text{issue}}$」这个判据去卡;移动端除了 Apple,**这个不等式普遍不成立**,所以 CPU-driven 是更理性的默认选择。

---

需要的话,我可以:
- 用 `CherryFetch` 抓 **Ampere whitepaper / Adreno 白皮书 / Arm Mali 优化指南** 把某些参数坐实到官方数字;
- 或者用 Python 把上面的延迟受限模型做成一个可调参的小仿真,输入 $N_{\text{warp}}, L, W_{\text{issue}}$ 直接画出有效 IPC 曲线。