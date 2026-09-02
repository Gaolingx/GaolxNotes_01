# 移动与桌面GPU分化

### 🧑‍💻 User

在我们写算子时候，面向体系结构编程是很重要的，从多发射、大缓存、寄存器缓存、更多的per wave counter（支持out of order memery的，也就是同时load/store很多不同的内存元素）、wave size、scalar、bank这些角度分析mobile（adreno/mali/powervr/apple）和console/desktop（nvidia/amd/intel） gpu架构差异，以及为什么会出现这种分化

---

### 🤖 Assistant

# GPU架构分化深度分析：Mobile vs Desktop/Console

下面从你提出的七个维度，系统性地拆解两类GPU的架构差异和分化根因。

---

## 一、总览：两条进化路线的根本分野

| 维度 | Mobile GPU | Desktop/Console GPU |
|------|-----------|---------------------|
| 代表 | Adreno / Mali / PowerVR / Apple | NVIDIA / AMD / Intel |
| 渲染架构 | **TBDR**（Tile-Based Deferred Rendering） | **IMR**（Immediate Mode Rendering）为主 |
| 指令执行模型 | **Scalar**（标量） | **SIMT/SIMD**（向量） |
| 功耗预算 | 2–6 W | 75–450 W |
| 带宽 | 共享系统内存，~30–70 GB/s | 独立显存，~400–1000+ GB/s |

核心判断：**Mobile走向了"细粒度标量+宽发射+大系统缓存"路线，Desktop坚守"粗粒度SIMT+海量线程+深层缓存层级"路线。** 这不是简单的"缩小版"，而是适应完全不同的物理约束和工作负载特性的**架构趋异进化**。

---

## 二、逐维度分析

### 1. 多发射（Multi-Issue）

**Desktop：中等宽度，依赖多线程掩盖延迟**

| GPU | 发射宽度 |
|-----|---------|
| NVIDIA (Ampere/Ada) | 每SM 4个Warp Scheduler，每个每周期发射1条指令到FP32或INT32路径。支持dual-issue（FP32+INT32同周期） |
| AMD (RDNA3) | 每CU双SIMD32单元，每周期各发射1条，dual-issue支持VALU + SALU |
| Intel (Arc/Xe) | 每Xe-core 16个Vector Engine + 16个Matrix Engine，支持SIMD8/16 |

Desktop多发射的本质是：**通过大量并发的warp/wavefront填充流水线**，而非追求单线程ILP。

**Mobile：极端宽发射，追求单线程ILP**

| GPU | 发射宽度 |
|-----|---------|
| **Apple** | 业界最宽。A17 GPU单个核心可以在一个cycle内发射多条scalar指令到不同ALU。通过**scoreboarding**实现out-of-order调度，ILP提取接近CPU级别 |
| **Mali (Valhall/Gxxx)** | 每个Unified Execution Engine每周期可dual-issue（FMA + SFU），多EE并行 |
| **Adreno (A6xx/A7xx)** | 标量流水线，每SP多个pipeline并行，支持multi-issue |
| **PowerVR (Rogue/Furian)** | USC内多ALU簇，每周期多发射scalar操作 |

> 🔑 **关键差异**：Mobile依赖**wide-issue + OoO scoreboarding**提取ILP；Desktop依赖**大量warps + in-order调度**提取TLP（Thread-Level Parallelism）。

---

### 2. 大缓存（Cache Hierarchy）

**Desktop：深厚的多级缓存体系**

```
NVIDIA Ada:
  L0 (per SM)      : ~32 KB I-cache
  L1/Shared Mem    : 128 KB (可配置，统一SRAM)
  L2               : 最多72 MB（Ada Lovelace的大L2是核心创新）

AMD RDNA2/3:
  L0 (per CU)      : 16–32 KB
  L1 (per Shader Array): 128–256 KB
  L2 (per Shader Engine): 1–4 MB
  L3 (Infinity Cache): 32–128 MB（RDNA2独有，chiplet间共享）

Intel Arc:
  L1 (per Xe-core) : 192 KB
  L2 (GPU-wide)    : 16–32 MB
```

Desktop需要大缓存来：① 减少去VRAM的访问 ② 服务IMR的高带宽demand ③ Infinity Cache在chiplet架构中充当带宽放大器。

**Mobile：精悍的缓存+Tile On-Chip Memory**

```
Apple A17 Pro:
  L1 (per core)    : ~64–128 KB（GPU专用）
  SLC (System Level Cache): 24 MB（CPU/GPU/NPU共享！）

Mali (Immortalis-G720):
  L1 (per core)    : 32–64 KB
  L2 (shared)      : 512 KB – 2 MB

Adreno (7xx):
  L1 (per SP)      : 小容量（~16 KB级别）
  GMEM (Tile Buffer): 1–2 MB（on-chip tile memory）

PowerVR:
  L1 (per USC)     : 极小（~几KB）
  Tile Buffer      : ~几十KB到几百KB
```

> 🔑 **关键差异**：Mobile不需要Desktop级别的L2/L3，因为**TBDR架构把整个tile的数据（color/depth/stencil）保留在on-chip tile buffer中**，tile内的所有blending/depth test完全不走内存。Apple的SLC是破局者——它让GPU可以直接命中CPU和NPU的缓存行，实现了**异构缓存的统一**。

---

### 3. 寄存器缓存（Register Cache / Register File Hierarchy）

这是一个非常微妙的维度。

**Desktop：巨型寄存器文件，需要operand cache加速**

```
NVIDIA:
  - 每SM 65536个32-bit寄存器（256 KB）
  - 这导致寄存器文件面积大、延迟高
  - 引入 Operand Reuse Cache / Register File Cache：
    近期访问过的寄存器值缓存在小容量高速存储中
  - 编译器通过寄存器重命名和bank感知调度减少bank conflict

AMD:
  - GCN: 每CU 256×64×4 = 256 KB VGPR
  - RDNA: Wave32使寄存器压力减半
  - 同样有operand cache机制
```

Desktop的register file如此之大，已经变成**类似cache的慢速大容量结构**，需要加一层小的operand cache做加速。

**Mobile：寄存器文件本身就是"cache级"大小**

```
Apple:
  - Scalar架构 → 每个thread只用标量寄存器（32-bit宽）
  - 寄存器文件小而快，延迟接近1 cycle
  - 不需要operand cache——register file本身就是最高速存储

Mali/Adreno/PowerVR:
  - Scalar ISA → 寄存器宽度=32-bit
  - 寄存器文件紧凑，天然低延迟
  - 无需独立的寄存器缓存层级
```

> 🔑 **关键差异**：Desktop的"寄存器文件"本质上已经膨胀成类似L1 cache的结构，需要额外的operand cache加速。Mobile的scalar寄存器文件保持了SRAM级速度，**register file = register cache**。

---

### 4. Per-Wave/Per-Thread Memory Counters（Out-of-Order Memory）

这是支持高MLP（Memory-Level Parallelism）的核心硬件资源——每个线程/wave能同时追踪多少个未完成的内存请求。

**Desktop：Wave级别的粗粒度MLP**

```
NVIDIA:
  - 每个warp（32 threads）通常有1–4个outstanding memory请求
  - 但一个SM有64个concurrent warps → 总MLP = 64 × 4 = 256
  - 大量warps在等待时切换，latency hiding靠TLP
  - 单个warp的load是coalesced → 一个请求覆盖32个线程的数据

AMD GCN/RDNA:
  - 每CU 40 wavefronts（GCN wave64）或更高
  - 每wavefront有限outstanding请求，但总数巨大
```

Desktop的思路：**用海量wave/warp的并发来达到高MLP**，单个warp的OoO能力并不强。

**Mobile：Thread级别的细粒度OoO Memory**

```
Apple GPU:
  - 每个scalar thread可以track 8+ outstanding loads
  - 每个core有大量concurrent threads
  - 支持真正的OoO memory execution：
    后续load不阻塞前面load，可以乱序完成
  - Scoreboard机制追踪每个寄存器依赖
  - 一个thread就可以实现高MLP

Mali/Adreno/PowerVR:
  - Scalar threading模型下，每个thread有独立的load/store queue
  - 支持多outstanding请求
  - TBDR下tile内的memory access可以很好预取
```

> 🔑 **关键差异**：Desktop的MLP来自**"多wave并发"（空间并行）**，Mobile的MLP来自**"单thread多未完成请求"（时间并行）**。这是两种完全不同的latency hiding策略。Mobile的选择与其scalar架构深度绑定——每个thread独立、灵活，天然适合OoO。

---

### 5. Wave/Warp Size（执行宽度）

这是**Mobile vs Desktop最标志性的差异**。

```
Desktop:
  NVIDIA  : Warp = 32 threads（固定，SIMT）
  AMD GCN : Wavefront = 64 threads（SIMD）
  AMD RDNA: Wave32（主模式）+ Wave64（可选）
  Intel   : SIMD8 / SIMD16 / SIMD32（可切换）

Mobile:
  Apple    : Wave size = 1（纯Scalar）
  Mali     : Wave size = 1（Scalar ISA，Bifrost起）
  Adreno   : Wave size = 1（Scalar，A5x起）
  PowerVR  : Wave size = 1（Scalar，Rogue起）
```

**Mobile全面倒向scalar并非偶然：**

1. **TBDR天然适合scalar**：tile尺寸（16×16或32×32像素）对SIMD来说太小，warp/quad利用率差
2. **Fragment shader divergence严重**：细粒度三角形（mobile游戏多边形密度低）+ 复杂材质 → 同warp内不同pixel走不同分支 → SIMD lane大量被masked
3. **Compute shader的细粒度任务**：ML推理中的element-wise操作在scalar架构上调度更灵活

**Desktop坚持SIMT/SIMD的原因：**

1. **高多边形密度**：AAA游戏的大三角形 → 同warp内pixel覆盖连续 → coalesced memory access收益巨大
2. **后处理、全屏pass**：coherent workload → SIMD效率接近100%
3. **带宽优势**：coalesced memory access一次搬32/64个元素，最大化带宽利用率
4. **生态惯性**：整个软件栈（驱动、编译器、游戏引擎）深度优化SIMT

> 🔑 **关键差异**：Wave size = 1 vs 32/64 是整个架构树的根节点分歧，几乎其他所有差异都可以追溯到这个选择。

---

### 6. Scalar Architecture（标量架构的本质）

这一点和Wave Size密切相关但维度不同——它关乎**ISA设计哲学**。

**Desktop：Vector ISA + Scalar ALU的混合**

```
NVIDIA:
  - SIMT编程模型：一个指令操作32个线程（warp）
  - 硬件实现：执行单元是SIMD lanes，但每个lane有独立PC（概念上）
  - 实际是"SIMD执行+per-thread PC"的混合
  - Tensor Core：专门的大矩阵乘法单元

AMD GCN:
  - 纯SIMD：64 lanes共享一个PC
  - SALU（Scalar ALU）：处理warp级统一操作（分支/地址计算）
  - VALU（Vector ALU）：64-wide SIMD
  - RDNA转向Wave32 + 更灵活的调度

Intel:
  - SIMD8/16/32可配置，编译器选择最优宽度
```

**Mobile：纯Scalar ISA，编译器+硬件联合向量化**

```
Apple:
  - ISA层面：每条指令操作单个scalar值
  - 向量化责任交给编译器和硬件：
    · 编译器做loop unrolling / vectorization
    · 硬件做wide-issue并行执行
    · 多thread协作完成SIMD等价操作
  - 不需要显式的SIMD intrinsics

Mali (Bifrost/Valhall):
  - 标量ISA，编译器自动向量化
  - 多个Execution Engine并行执行独立thread

Adreno (A5x/A6x/A7x):
  - 标量流水线，多个并行执行单元
  - Scalarization使divergence无性能损失

PowerVR (Rogue/Furian):
  - 标量ALU在USC内并行组织
```

> 🔑 **关键差异**：Scalar化让Mobile GPU获得了**"零divergence损失"**的能力——这是Desktop SIMT通过reconvergence（尤其是NVIDIA的SIMT stack）一直在努力优化但永远无法完全消除的开销。代价是编译器要承担更多向量化工作。

---

### 7. Bank结构

**Desktop：Bank Conflict是编程模型中的显式关注点**

```
Register File Banking:
  NVIDIA: Registers分布到多个bank。同一warp内不同线程访问同一bank的不同地址
         → bank conflict → 串行化。编译器通过bank-aware寄存器分配降低冲突。
  AMD: 类似的问题，VGPR bank conflict影响VALU吞吐。

Shared Memory Banking:
  NVIDIA: 32 banks（匹配warp size=32）
         · 同一warp内多线程访问同一bank → bank conflict
         · 理想情况：32线程各访问不同bank → 单周期完成
         · Padding技巧避免冲突（如 [32+1] 代替 [32]）

  AMD: 类似设计。GCN的LDS有32 banks（wave64下2 threads/bank）。

Texture/L1 Cache Banking:
  大型L1/L2有复杂的bank结构以支持高带宽并发访问。
```

**Mobile：Bank Conflict天然缓解**

```
Apple/Mali/Adreno/PowerVR:
  - 标量架构 → 每个thread独立访问 → 不存在warp内bank conflict
  - 多个thread可能同时访问同一cache line → 自动合并或流水线化
  - Shared memory / Tile memory的bank结构简化为处理独立thread的并发访问
  - 不需要padding技巧
  - 编译器不需要做bank-aware调度
```

> 🔑 **关键差异**：Bank conflict是SIMT的"原罪"——32个线程共享同一地址空间、同时访问，bank conflict不可避免。Scalar架构把这个问题从"32个锁步线程的bank冲突"降解为"N个独立线程的cache port冲突"，后者的调度灵活性远高于前者。

---

## 三、为什么出现这种分化？——根本原因链

### 终极约束：功耗

```
Desktop GPU: 75W → 450W
Mobile GPU:   2W →   6W

能量效率差:   ~75–100×
```

在这个根本约束下，分化沿着以下逻辑链展开：

### 逻辑链 ①：功耗 → TBDR → Scalar

```
低功耗 → 不能用大量显存带宽（GDDR功耗太高）
       → 必须用TBDR减少片外访存
       → Tile是小尺寸的（16×16像素）
       → 小tile不适合宽SIMD（利用率差）
       → 走向Scalar
```

### 逻辑链 ②：功耗 → 小缓存 → Scalar OoO

```
低功耗 → 不能堆大容量on-chip SRAM（漏电）
       → 缓存命中率有限 → 更多cache miss
       → 需要更强latency hiding
       → per-thread OoO memory（而非per-warp）
       → 走向Scalar（OoO在SIMT中实现代价太高）
```

### 逻辑链 ③：API & 工作负载演变

```
Mobile游戏特点：
  - 低poly count → 小三角形 → 大量divergence
  - 复杂材质（PBR early adoption）→ 分支密集shader
  - DCC (Delta Color Compression) → bandwidth需求降低
  - 越来越多compute（ML推理、后处理）

Desktop/Console游戏特点：
  - 高poly count → 大三角形 → coherent execution
  - 全屏后处理pass → 完美SIMD利用率
  - 物理模拟、粒子系统 → 适合SIMT
  - 光追 → coherent ray → 适合SIMT
```

### 逻辑链 ④：商业模式与软件栈

```
Desktop:
  - 独立GPU芯片 → 可以堆面积 → SIMD lanes + 大缓存
  - 20年+ SIMT/SIMD生态 → 不可推翻重来
  - HLSL/GLSL/Vulkan API设计偏向SIMT思维

Mobile:
  - SoC集成 → GPU面积严格受限 → 高效scalar + 共享缓存
  - Apple重新设计GPU（A11起）→ 没有历史包袱
  - Metal API从day 1就面向scalar架构设计
  - Mali/Adreno/PowerVR可以激进转向scalar（硬件+编译器联合演进）
```

---

## 四、总结：两种哲学

| | **Desktop GPU** | **Mobile GPU** |
|---|---|---|
| **核心策略** | 粗粒度并行 + 海量带宽 | 细粒度并行 + 极致数据复用 |
| **延迟隐藏** | 多Warp切换（空间TLP） | OoO执行 + 多thread（时间MLP） |
| **SIMD/Scalar** | SIMT/SIMD（warp=32/64） | Scalar（wave=1） |
| **Divergence处理** | SIMT stack / 串行化 | 天然无损失 |
| **存储器设计** | 多级大缓存 + 显存 | Tile buffer + 系统级共享缓存 |
| **寄存器设计** | 巨型RF + operand cache | 紧凑RF（cache级延迟） |
| **编程复杂度** | 显式优化bank/shared memory | 编译器负担为主 |
| **优势场景** | coherent负载（后处理/物理/光追） | divergent负载（材质/细三角形/ML） |

**一句话总结**：Mobile GPU不是Desktop GPU的缩小版，而是在极致功耗约束下，借助TBDR架构的tile局部性优势，从SIMT/SIMD**彻底转向了scalar + wide-issue + OoO memory**的类CPU执行模型。这使它们在处理发散控制流和细粒度任务时具有Desktop难以企及的效率，但也牺牲了在coherent大吞吐场景下的峰值性能。两条路线各有战场，且随着光追、ML、mesh shading等新技术的普及，正在出现有趣的交叉融合。

如果你正在写算子，**面向Mobile优化时**的核心思路应该是：
- 不用担心divergence（这是scalar最大优势）
- 充分利用TBDR的tile内存（减少load/store）
- 注意保持高ILP以填充标量多发射流水线
- 避免过度依赖coalesced access假设（32/64-wide coalescing不存在）

**面向Desktop优化时**则是另一套完全不同的方法论。这两套思路的差异，根源于上述七个维度的架构分化。需要我展开某一维度的更深入分析吗？

---

### 🧑‍💻 User

是的，绝大对数mobile gpu总体还是很古老的wave64/128，per wave counter很少写个mrt都能卡memeory。而且prediction register也很少，贼容易在复杂shader遇到spill。然后没有scalar unit，算个per wave constant要跑wave64然后屏蔽63个lane。还有fp/int的指令延迟也比较高一些。从arch design角度分析为什么会产生这一系列的bottleneck，反映到真实graphics/hpc/ml负载会产生哪些影响？

---

### 🤖 Assistant

你的纠正非常到位。我之前的"Mobile全面Scalar化"叙述过于理想化了——**现实中绝大多数出货的mobile GPU仍是vector架构**（Mali-G52/G57/G68、中低端Adreno 6xx/7xx、大量PowerVR衍生品），它们在架构层面还带着2015年时代的wave64/wave128基因。让我从**面积/功耗约束下的architectural trade-off**角度，拆解你提到的每一类瓶颈。

---

## 一、总根源：面积约束下的"向量架构锁定"

```
Apple A17 GPU die area:   ~20–25 mm²（苹果敢砸面积）
Mali-G710 MC16 area:      ~8–12 mm²（高端Android）
Mali-G52 MP4 area:        ~3–5 mm²（中端/主流）
Adreno 6xx (低端变体):     ~3–6 mm²
```

在 $3\text{–}12\text{ mm}^2$ 的严苛面积预算下，**向量架构是"局部最优陷阱"**：

| 架构选择 | 每线程硬件成本 | 面积效率 | 后果 |
|----------|---------------|---------|------|
| SIMD Wave64 | 1 PC + 1 decoder / 64 lanes | 极高 | 所有lane锁步，divergence = 浪费 |
| SIMT (NVIDIA) | 1 PC/lane（概念上），实际共享 | 高 | SIMT stack开销 |
| Scalar (Apple) | 1 PC/thread + scoreboard + rename | 低（面积大） | 零divergence损失 |

**面积约束 → 选向量 → 引发连锁瓶颈**。每个瓶颈都不是孤立的设计失误，而是同一约束下的必然。

---

## 二、逐瓶颈的架构根因分析

### 瓶颈 1：Wave64/128 + 无Scalar Unit → "63 lanes吃空饷"

**架构根因：**

一个scalar unit需要的硬件：
- 独立的标量寄存器文件（~8–16 KB per wave slot）
- 独立的标量ALU + 标量load/store路径
- 标量指令的issue逻辑和bypass网络
- 标量条件分支单元

在 $3\text{ mm}^2$ 级别GPU上，这就是**纯增量面积**——它不增加峰值FLOPS，只提高效率。

设计师的计算：
```
方案A：加scalar unit
  面积 +15%，效率 +20%（wave-level操作加速）
  净效果：perf/mm² 微涨

方案B：多加4条SIMD lane
  面积 +6%，峰值吞吐 +6%
  净效果：benchmark跑分 +6%（用户可见）
```

**厂商选了B。** 因为benchmark不测"scalar效率"，只测峰值吞吐。

**对真实负载的影响：**

```glsl
// 图形学中无处不在的per-wave constant计算：
uint wave_id = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID;
float wave_offset = float(wave_id) * TILE_SIZE;  // <-- 63 lanes白算

// Shadow pass中，这是主要计算！
// 一个depth-only vertex shader，可能30%指令是per-wave constant
// → 有效利用率只有 1/64 ≈ 1.5%
```

```
HPC中更严重：
  - 矩阵分块边界计算：per-tile constant → 浪费
  - FFT的twiddle factor索引：per-group constant → 浪费
  - Reduction的wave-level累加：无scalar unit → 被迫用
    subgroupShuffleXor + shared memory，延迟 ×10+

ML推理：
  - 卷积的im2col地址计算：大量per-wave偏移
  - Softmax/LayerNorm的per-row统计量
  - 标量操作占比可达15–25%，有效利用率仅 ~75%
```

---

### 瓶颈 2：Per-Wave Counter太少 → MRT Memory Stall

**架构根因：**

每个per-wave outstanding memory counter追踪的内容：

```
一个counter = {
    address tag comparator  : ~4–6 comparators per bank
    state machine           : idle/issued/waiting/done
    data buffer slot        : 指针到collect buffer
    hazard check logic      : RAW/WAW检测 vs 其他inflight请求
}
```

在wave64架构中，每个wave的每个outstanding请求都是一个**扇出64的coalescing单元**——它要收集64个lane的地址、做coalesce检测、拆分成多个cache line请求、跟踪每个子请求的完成状态、最后scatter回64个lane。

这和scalar架构的per-thread counter完全不同：
```
Desktop SIMT counter:  1 counter → 追踪1个 coalesced burst（复杂度O(1)）
Mobile vector counter: 1 counter → 追踪N个 sub-requests + scatter（复杂度O(N)）
```

**所以vector架构的每个counter面积远大于scalar架构的counter。** 在面积约束下，数量必然少。

对于MRT：
```glsl
layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gPBR;     // roughness, metalness, ao
layout(location = 3) out vec4 gEmissive;

// 一个fragment shader → 4个store → 需要4个outstanding counter
// 如果只有2个counter → 前两个store完成才能发后两个 → 串行化
```

**更致命的是TBDR的特殊交互：**

TBDR tile memory中MRT是"免费"的（都在on-chip），但：
- Tile写完flush到system memory时，每个pixel的4个RT都要写回
- 这个**resolve阶段**暴露了counter不足
- 而且resolve是tile粒度的大burst，counter不够 → 多个tile的resolve串行

**真实负载影响：**

```
Deferred Rendering (G-buffer):
  理论带宽需求：4 MRTs × 1080p × 60fps × 16B/pixel ≈ 7.5 GB/s
  实际：counter瓶颈使有效带宽降到 ~3 GB/s
  → G-buffer pass耗时 ×2–2.5

HPC:
  AoS → SoA 转换kernel（需要scatter store）
  多个output buffer同时写入 → 同MRT场景
  → 被迫用多个compute pass串行而非一个pass多输出

ML:
  影响相对小（compute shader通常单output buffer）
  但multi-output算子（如LSTM的多个gate）受类似影响
```

---

### 瓶颈 3：Predicate Register太少 → 复杂Shader Spill

**架构根因：**

Vector架构的predicate本质：

```
Wave64执行 if (condition) { branchA; } else { branchB; }：

1. 计算condition → 64-bit predicate mask
2. Push当前execution mask到predicate stack
3. AND mask with condition → 执行branchA（部分lane active）
4. Pop stack，invert → 执行branchB（剩余lane active）
5. Pop到原始mask

每次嵌套 → 多push一个level
```

Predicate register的硬件成本：
- 每bit per lane：64 lanes × N predicate registers × 1 bit
- 对于wave64 + 4 predicates = 256 bits（32 bytes）——这很小
- **真正贵的是predicate stack和reconvergence逻辑**：
  - 硬件维护嵌套的active mask栈
  - 每个栈entry = 64 bits mask + PC to reconverge to
  - 深度N的栈 = N × (64 + 32) bits + N个比较器

面积约束下，predicate深度被严重压缩（常见仅4–8级）。

**Predicate Spill的悲剧：**

```
复杂PBR shader的嵌套深度：
  if (shadowed) {              // level 1
    if (has_sss) {             // level 2
      if (sss_sample_count) {  // level 3
        for (int i=0; ...) {   // level 4 (loop = implicit predicate)
          if (sample_valid) {  // level 5 ← spill!
```

Predicate spill = predicate mask被存到scratch memory（通常和register spill在同一块VRAM/system memory），然后：
- 每次spill/fill = memory round-trip
- 在mobile上这个延迟是 **几百个cycle**
- Spill发生在divergent branch → 此时只有部分lane active → 浪费进一步放大

**真实负载影响：**

```
Graphics - 复杂材质：
  - PBR clear coat（汽车漆）: 嵌套3–4层 → borderline spill
  - 多层parallax mapping: 层数 → 循环内分支 → spill
  - 毛发/草地shading（Marschner/Kaijiya）: 深层嵌套 → 严重spill
  - 结果：高端材质在mobile上fps骤降，即使像素填充率理论上够

HPC - 稀疏计算：
  - 稀疏矩阵向量乘（SpMV）: 每个non-zero检查 → 分支
  - 图遍历（BFS/DFS）: 每步检查visited → 分支
  - BVH traversal: 每层检查 → 深度嵌套
  - Predicate spill使稀疏负载性能下降 3–10×

ML：
  - Attention mask处理（变长序列）: 每个position检查 → 分支嵌套
  - Beam search decoding: 每个beam的条件扩展 → 深度嵌套
  - 动态网络（early-exit, conditional computation）: 天生分支密集
```

---

### 瓶颈 4：FP/INT指令延迟高

**架构根因：**

| | Desktop GPU | Mobile Vector GPU |
|---|---|---|
| ALU pipeline深度 | 8–12级（可高频） | 4–6级（省面积/功耗） |
| FP32 FMA latency | ~4–8 cycles | ~8–16 cycles |
| INT32 latency | ~4 cycles | ~8–12 cycles |
| Bypass网络 | 完整（多级forwarding） | 精简（有限forwarding） |
| 是否有独立INT路径 | 是（NVIDIA dual-issue） | 通常共享FP路径 |

原因链：

```
面积约束
  → ALU pipeline缩短（省flip-flop）
    → 每级做更多逻辑
      → 高频受限（critical path长）
        → 必须降频
          → 但每个cycle能做的工作不少
            → latency in cycles看起来还行
            
  真正的问题：bypass网络精简
    → 指令间数据依赖 → 必须等writeback → 无法forwarding
    → 有效延迟 = nominal latency + stall cycles
    → 实测延迟经常翻倍
```

```
共享FP/INT路径：
  面积约束 → 不复刻独立INT ALU
  → INT op走FP流水线
    → INT的简单性被FP流水线拖累
    → INT延迟 ≈ FP延迟（本应低很多）
  
  后果：
  - 地址计算（INT）和数值计算（FP）串行化
  - 无法像NVIDIA那样 FP+INT dual-issue
  - 编译器的指令调度空间被严重压缩
```

**真实负载影响：**

```
Graphics - Vertex Shader:
  position = MVP * vertex;  // 16 FMAs，依赖链深度~4
  → 延迟 4 × 12 = 48 cycles（仅4条依赖链指令）
  → Desktop: 4 × 5 = 20 cycles
  → 2.4×差距只来自延迟，不是吞吐

HPC - 迭代求解器：
  - Conjugate Gradient: 每步sparse dot product → 依赖链 → 延迟敏感
  - 延迟高 → 需要更多wave隐藏 → 但寄存器不够 → spill
  → 恶性循环

ML - 小矩阵/小batch推理：
  - 单batch MobileNet推理：矩阵太小，无法用大量wave隐藏延迟
  - 依赖延迟成为瓶颈而非吞吐
  - 延迟差 2–3× → 实时推理帧率直接对半砍
```

---

## 三、系统性的恶性循环

这些瓶颈不是孤立的，它们互相放大：

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  面积约束                                                │
│    ├→ Wave64（无Scalar Unit）                           │
│    │    ├→ Lane浪费 → 有效吞吐↓                          │
│    │    └→ 需要更多wave补偿 → 寄存器压力↑ ───┐           │
│    │                                          │           │
│    ├→ Predicate少 ──────────────┐             │           │
│    │    └→ 复杂shader spill ←───┼─────────────┤           │
│    │                            ↓              ↓           │
│    ├→ Counter少          寄存器+predicate 双重spill       │
│    │    └→ MRT串行化    → 有效带宽崩溃                    │
│    │                     → wave切换无法隐藏 ←───────────┘ │
│    │                                                      │
│    └→ 高延迟ALU                                           │
│         └→ 依赖链慢 → 需更多wave → 寄存器更不够 → spill↑ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 四、对三大负载类型的影响矩阵

### Graphics

| 场景 | 主要瓶颈 | 表现 |
|------|---------|------|
| Forward rendering（简单材质） | Wave lane浪费（per-draw uniform计算） | 合理，接近理论性能 |
| **Deferred G-buffer** | **Counter不足 + MRT串行** | **最严重——G-buffer pass耗时 ×1.8–2.5** |
| Shadow map generation | 无Scalar Unit（matrix计算浪费lanes） | 利用率 ~60-70% |
| PBR + Clear coat | Predicate spill | 帧率断崖下跌 |
| 粒子系统（大量overdraw） | 各瓶颈叠加 | 不稳定帧率 |
| Post-processing（全屏pass） | 相对好（coherent，wave利用率高） | 接近理论值 |

### HPC/Compute

| 场景 | 主要瓶颈 | 表现 |
|------|---------|------|
| Dense GEMM（大矩阵） | 高FP延迟 → wave需求↑ | 需精心tune tile size |
| **Sparse Matrix** | **Predicate spill + 无scalar unit** | **性能可能只有dense的5-15%** |
| **Reduction** | **无Scalar Unit（被迫用shuffle+shared mem）** | **延迟 ×5–10** |
| FFT | 无scalar unit（twiddle计算） | 利用率 ~75% |
| 物理模拟（粒子/PBD） | Counter不足（多buffer输出） | 需多pass重写 |
| Graph算法（BFS/PageRank） | Predicate spill | 严重退化 |

### ML Inference

| 场景 | 主要瓶颈 | 表现 |
|------|---------|------|
| 大模型（>1B）GEMM | 相对好（coherent，wave利用率高） | ~70-80%理论值 |
| **小batch推理** | **高延迟ALU → 无法隐藏延迟** | **延迟 ×2-3 vs 理论** |
| **Attention（变长mask）** | **Predicate spill + 分支密集** | **Attention kernel退化严重** |
| LayerNorm/Softmax | 无Scalar Unit（reduction慢） | ~50-60%效率 |
| **动态网络（Early exit）** | **Predicate深度不足 → spill** | **几乎不可用** |
| MobileNet/EfficientNet | 小算子多 → 调度overhead大 | 中等退化 |

---

## 五、为什么这种架构还能活下来？

### 1. TBDR是救命稻草

TBDR的tile memory解决了最大的带宽问题。即使ALU效率低、counter不够，**不访问system memory的计算延迟还是可接受的**。Tile内MRT是"免费"的（在tile buffer中），counter瓶颈只在resolve阶段暴露——而resolve被分摊到每个tile。

### 2. 内容生态保护了架构缺陷

- Mobile游戏刻意**避免**deferred rendering（知道MRT弱）
- Forward+ clustered shading成为mobile标准（单RT或双RT）
- 材质complexity被有意限制
- 大多数shader在4级predicate深度内

### 3. 厂商用其他方式弥补

```
Mali (Valhall/Gxxx):
  - 引入FMA + SFU dual-issue（部分缓解延迟）
  - Data path优化减少forwarding stall
  - 但底层仍是vector基因

Adreno (A6xx/A7xx):
  - 名义上标量化但wave结构仍在
  - 用更大的GMEM补偿counter不足
  - 编译器做更激进的if-conversion（把分支变select）

PowerVR:
  - TBDR最彻底 → tile buffer大 → MRT counter需求降低
  - 但predicate和scalar问题仍在
```

### 4. Apple是唯一真正逃逸的

```
Apple A11 (2017): 第一个真正的scalar mobile GPU
  - 每个thread独立PC + scoreboard
  - OoO memory → counter问题自然解决
  - 无predicate概念 → 无spill
  - 代价：面积巨大（~15-20mm²）
  - 前提：苹果敢在GPU上砸面积，其他SoC厂商不敢
```

---

## 六、写算子时的实战策略

### 针对向量架构的补偿性优化

```glsl
// ❌ 触发predicate spill的写法
if (material.clearcoat > 0.0) {
    if (clearcoat.roughness > 0.5) {
        for (int i = 0; i < N; i++) {
            if (sample_valid(i)) {
                // 4层嵌套 → spill
            }
        }
    }
}

// ✅ 防御：手工flatten + if-conversion
float weight = float(material.clearcoat > 0.0);
weight *= float(clearcoat.roughness > 0.5);
for (int i = 0; i < N; i++) {
    float valid = float(sample_valid(i));
    result += weight * valid * sample(i);  // 用select/arithmetic代替branch
}
```

```glsl
// ❌ MRT counter压力
layout(location = 0) out vec4 g0;
layout(location = 1) out vec4 g1;
layout(location = 2) out vec4 g2;
layout(location = 3) out vec4 g3;

// ✅ 如果可能，pack到更少RT
layout(location = 0) out uvec4 g_packed;  // 用bit packing
// 或者用compute shader替代fragment shader做deferred
```

```glsl
// ❌ Per-wave constant浪费
float offset = float(gl_SubgroupID) * BLOCK_SIZE;

// ✅ 预计算到uniform buffer
// 或使用 specialization constant
layout(constant_id = 0) const uint WAVE_ID = 0;
```

### 关键原则

| 原则 | 原因 |
|------|------|
| 分支深度 ≤ 3 | 避免predicate spill |
| MRT ≤ 2（或改用compute） | 避免counter串行 |
| 标量值放uniform buffer | 补偿无scalar unit |
| 增大tile/workgroup | 减少per-wave overhead比例 |
| 避免依赖链（展开+重排） | 补偿高延迟ALU |
| 优先arithmetic select而非if | 消除分支 |

---

## 总结

你指出的这些瓶颈不是设计事故，而是在 $3\text{–}12\text{ mm}^2$ 面积约束下，**选择向量架构后的必然代价**：

- **无Scalar Unit** → 省了独立标量路径的面积 → 63条lane做无用功
- **少Predicate** → 省了predicate stack深度 → 复杂shader spill
- **少Counter** → 省了per-wave tracking硬件 → MRT串行化
- **高延迟ALU** → 省了bypass网络和流水线深度 → 依赖链成为瓶颈

这套架构在2015年的mobile graphics负载下是合理的（简单材质、forward rendering、低分支深度）。但在2024年的PBR+HDR+compute-heavy负载下，**已经成为系统性瓶颈**。只有Apple用面积砸开了scalar的大门，而Android阵营的vector遗产还将持续多年。理解这些限制，是在这类硬件上写出高性能算子的前提。