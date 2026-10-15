# GPU架构演进要点

## 🧑‍💻 User

其实不太理解每一个架构的改进意义是什么，看nv的hopper和blackwell，我唯一的感觉就是加了个TMEM，共享内存变大了，可能我的理解还是太浅

---
## 🧑‍💻 User

其实不太理解每一个架构的改进意义是什么，看nv的hopper和blackwell，我唯一的感觉就是加了个TMEM，共享内存变大了

---
## 😀 默认助手 (7)

你这个"看不懂改进意义"的感觉，其实不是理解浅，而是**spec sheet 上根本写不出真正的改进**。NVIDIA 发布会讲的是 TFLOPS 和 HBM 带宽，而真正的架构变化全在"数据怎么搬到 tensor core 面前"这条路径上——这条路径上的东西，一个数字都体现不出来。

先纠正两个事实，因为它们是理解后面所有东西的前提。

## 一、两个事实纠正

**1. TMEM 不是 Hopper 的，是 Blackwell 的。** Hopper（H100）没有 TMEM，它的累加器还在寄存器堆里，配套机制是 `wgmma`（warpgroup MMA）+ shared memory descriptor。TMEM 是 Blackwell 的 `tcgen05` 才引入的。这两个东西解决的是同一个问题的**两种不同路线**，看清这个分叉，你就理解了半代架构。

**2. 共享内存从 Hopper 到 Blackwell 基本没变大。** A100 是 164KB/SM，H100 是 228KB/SM，B200 还是 227KB 左右。真正涨容量的是 Ampere→Hopper 那一步（也正是那一步让 wgmma 的 descriptor 方案成为可能，因为整块 B tile 得常驻 smem）。所以"smem 变大"你记的是对的，只是记在了错的代际上。

那 Hopper→Blackwell 到底变了什么？变的是**SM 的组织方式**。

## 二、北极星指标：tensor core 空转率

所有架构改进都在优化同一个目标：

$$
\text{Utilization} = \frac{\text{有效 FLOPs}}{\text{峰值 FLOPs}}
$$

而决定它的第一性约束是**机器平衡点**（peak FLOPS / memory bandwidth）：

| 代际 | FP16 tensor 峰值 | HBM 带宽 | 平衡点 |
|---|---|---|---|
| V100 | 125 TF | 0.9 TB/s | 139 FLOP/byte |
| A100 | 312 TF | 1.55 TB/s | 200 FLOP/byte |
| H100 | 989 TF | 3.35 TB/s | 295 FLOP/byte |
| B200 | ~2250 TF | 8 TB/s | ~281 FLOP/byte |

注意 B200 的平衡点**没有改善**——NVIDIA 用 HBM3e 和更多堆栈把带宽拉了 2.4 倍，才勉强跟上算力增长。这意味着：**纯堆算力的路已经走死了**。

但更关键的是：GEMM 本身从来不缺复用。一个 $4096^3$ 的 GEMM 算术强度约 $2N/3 \approx 2730$ FLOP/byte，比平衡点高一个数量级，随便写都能跑满。**问题从来不在 GEMM 上。**

问题在 LLM decode：

$$
\text{batch}=1, \text{70B 模型, FP16} \;\Rightarrow\; \frac{2 \times 70\text{e}9\ \text{FLOP}}{140\ \text{GB}} = 1\ \text{FLOP/byte}
$$

比平衡点低 **280 倍**。意思是：算力全在睡觉，你只是在搬权重。换成 FP4 权重，读到 35GB，算术强度变成 4 FLOP/byte——还是低 70 倍。

这就是为什么 Blackwell 的头条新闻是 **4-bit（NVFP4/MXFP4）和 decompression engine**，而不是 TMEM。**位宽是唯一能直接改善这个式子的杠杆**，其他所有东西都只能改善"在带宽受限的情况下别让 tensor core 更空"。

## 三、四根杠杆

所有代际改动都可以归到这四个抽屉里：

| 杠杆 | 手段 | 具体案例 |
|---|---|---|
| **容量** | 让数据多待一会儿 | smem 164→228KB、L2 40→50MB、**TMEM 256KB** |
| **带宽** | 让搬运更宽、路径更短 | `ldmatrix`、smem descriptor、TMEM load/store、DSMEM、TMA multicast |
| **重叠** | 让搬运和计算同时发生 | `cp.async` → TMA → 单线程 issue 的异步 MMA、warp specialization |
| **位宽** | 少搬比特 | TF32、2:4 稀疏、FP8、MX/NVFP4、解压引擎 |

你看到的"加了 TMEM、smem 变大"其实是**容量**这一格。但真正的架构演进量在**重叠**和**带宽**这两格里，因为它们改变的是编程模型。

## 四、四个 load-bearing 的思想

### 4.1 布局管道（layout plumbing）——最被低估的一环

Tensor core 要求操作数在 warp 内以**硬件规定的固定布局**分布。Volta 时代，你得手动 `ld.shared` + `shfl` 把数据摆成那个形状，每条 MMA 前后都有一堆非计算指令。这些指令吃掉 issue slot，吃掉寄存器。

演进路径：

- **Ampere `ldmatrix`**：一条指令从 smem 加载 8×8 tile，**直接落到 MMA 需要的 lane 布局**。省掉一批 lds+shfl。
- **Hopper smem descriptor**：`wgmma` 的 B 操作数根本不用进寄存器，硬件按 descriptor 直接从 smem 读。寄存器压力骤降。
- **Blackwell TMEM**：累加器也不在寄存器里了。

这背后是一个容易被忽略的编程模型变化：**async proxy**。TMA 和 tensor core 走在和普通 load/store **不同的内存一致性域**里，你必须 `fence.proxy.async` 才能保证可见性。这就是为什么 Hopper/Blackwell kernel 里那些 fence 看起来莫名其妙——它们标记的是两个代理之间的握手。

### 4.2 异步化 → warp specialization

看三条流水线的对比，这就是"重叠"这根杠杆的全部内容：

```cuda
// Ampere: 每个线程都参与所有阶段
for k:
    cp.async(global -> smem);        // 直连，不占寄存器
    ldmatrix(smem -> regs);           // 摆布局
    mma(regs, regs -> regs);          // 累加器在寄存器
    cp.async.wait(); barrier();

// Hopper: 角色分工，1 个线程发一整个 tensor 搬运
// ... producer warp:  tma.async.bulk.tensor(desc, smem); mbarrier.arrive();
// ... consumer wg:     mbarrier.wait(); wgmma(smem_desc, regs -> regs);

// Blackwell: MMA 自己就是引擎，单线程点火，没人等它
// ... 1 thread:  tcgen05.mma(smem_desc, tmem_accum); tcgen05.commit(mbar);
// ... 其余线程:  同时做 softmax / epilogue / 下一块
// ... 完成信号: mbarrier
```

关键变化是**每代喂饱 tensor core 所需的指令数掉了大约一个数量级**：

- Volta/Ampere：1 条 MMA 配好几条 LDS/IMAD
- Hopper：1 条 `wgmma`（128 线程协同 issue）覆盖 $64\times256\times16$
- Blackwell：1 条 `tcgen05.mma`（**1 个线程** issue）覆盖 $256\times256\times16$，完全异步

为什么"单线程 issue"是质变？因为 issue 一个指令本身要占 warp 的发射槽。wgmma 需要整个 warpgroup 128 个 lane 保持同步；而 tcgen05 只要一个线程点火，**剩下 127 个线程全部解放**——去跑 epilogue、跑 softmax、准备下一块。这就是 Blackwell 上 FP8/FP4 attention 和 MoE kernel 能跑起来的原因。

kernel 从"所有线程做所有事"变成了一个**由 mbarrier 调度的数据流图**。这才是 Hopper/Blackwell 真正的新范式，spec sheet 上一个字都没有。

### 4.3 TMEM：为什么必须把累加器搬出寄存器堆

现在回答你最关心的问题。寄存器堆是 256KB/SM，看着不少，但是：

一个 warpgroup 的 $64\times256$ FP32 累加器 = 64KB = 16384 个 FP32 = **每线程 128 个寄存器**。这只是**一个** tile、**一个** buffer。你还要放操作数、流水线状态、地址。255 寄存器的硬上限瞬间撞墙，结果就是：无法双缓冲累加器 → 无法加深流水 → tensor core 饿死。

TMEM 是 256KB/SM，128 lane × 512 个 32-bit 列，**在寄存器堆之外**，而且有一条专属的、极宽的通道直连 tensor core。它同时解决三件事：

1. **容量与 tile 尺寸解耦**：你可以同时持有好几个累加器 tile，把 K 方向流水做得极深。
2. **去掉 MMA 内环里的寄存器堆**：Hopper 的 wgmma 虽然 B 从 smem 读，但累加器的 read-modify-write 还得穿寄存器堆。TMEM 让 MMA 的整个数据流都不碰寄存器堆。
3. **让异步真正彻底**：Hopper 的 wgmma 是异步的，但它占用的寄存器在整个 MMA 期间被 async proxy 锁定。TMEM 把这些寄存器还给了你。

代价是 TMEM **不是通用内存**：不能任意寻址，只能通过 `tcgen05.ld/st` 按 lane 访问，A 操作数还得先 `tcgen05.cp` 从 smem 拷进去。这正是整个架构演进的母题——**用通用性换效率**。

### 4.4 2-SM MMA 与 thread block cluster

Blackwell 的 `tcgen05` 支持 `cta_group::2`：**两个 SM 的 tensor core 协同做一次 MMA**，一个 SM 的 TMEM 可以被配对的 SM 直接访问。

动机很直白：tile 越大，每个 FLOP 分摊的 smem 流量越小。两个 SM 拼一个更宽的 tile，就省下一半的共享操作数读取。而要让这件事成立，你就必须先有一个两边都能高速访问的暂存区——**这就是 TMEM 必须存在的另一个原因**。

cluster（Hopper 引入，最多 16 个 CTA，可移植的 8 个）是同一个思路在更粗的粒度上：CTA 之间能直接读对方的 smem（DSMEM），配合 **TMA multicast**，同一块 A tile 从 L2 只读一次就灌进多个 SM。省的是 L2 带宽。

## 五、一句话总结每代在干什么

- **Volta**：把矩阵乘变成一个指令，代价是引入了"布局"这个新麻烦。
- **Ampere**：把搬运从寄存器和 issue 槽里解放出来（`cp.async`），把布局搬运也变成指令（`ldmatrix`）。
- **Hopper**：把地址生成和搬运整个塞进硬件引擎（TMA），把 MMA 的操作数从寄存器挪走（smem descriptor），把 CTA 的边界打破（cluster/DSMEM）。**代价是 warp specialization 成为必需。**
- **Blackwell**：把 MMA 从 warp/warpgroup 里彻底拿掉（单线程 issue + 专属内存 TMEM），把 tensor core 扩到 2-SM，把位宽砍到 4-bit。
- **GB200/NVL72**：把"一台机器"的定义从 GPU 挪到机柜（72 GPU、单 NVLink 域、130TB 一致内存）。**这代最重的创新其实是封装和系统，不是 SM。**

同一个故事的更好类比是 IBM 大型机的 I/O 演进：**programmed I/O → DMA → channel processor**。SM 正在从一个"所有工作都要线程亲手做"的处理器，变成一个"带多条 I/O channel 的小型计算机"，线程退化成只负责点火和调度 channel 的驱动代码。

<details>
<summary><b>展开：Blackwell 的真正大变化可能在 SM 之外（NVL72）</b></summary>

B200 是**两颗接近光罩极限的 die**，用约 10TB/s 的 die-to-die 互连（NV-HBI）拼成"一颗" GPU。这不是架构创新，是**制造成本创新**——1000mm² 的单片的良率是灾难性的。你先得让互连快到"看不见接缝"，架构上才敢把它当一颗用。

而 GB200 NVL72 把 72 颗 GPU 放进一个 NVLink 域，加上 Grace CPU 通过 900GB/s 的 NVLink-C2C 直连。MoE 这种模型天然是"跨专家切分 + all-to-all"的负载，单卡堆算力对它毫无用处，**域的宽度才是关键**。这也解释了 NVLink 5 从 900GB/s 翻到 1.8TB/s 的战略意义：它不是为了让 kernel 更快，是为了让 GPU 的"集群"能当成一颗 GPU 来编程。

如果你只盯着 SM，这一整块都是隐形的。
</details>

<details>
<summary><b>展开：现实检验——哪些特性其实没落地（用来校准你的直觉）</b></summary>

不是每个架构特性都重要，学会分辨比记住清单更值钱：

- **2:4 结构化稀疏（Ampere 引入）**：理论 2x，实际几乎没人用。因为它要求权重按特定模式剪枝，换来的精度损失和工程负担不划算。**架构上很优雅，生态上是死的。**
- **DPX 指令（Hopper）**：基因组比对专用，极其小众。
- **TF32（Ampere）**：真香，但它是"别让我把 FP32 降成 FP16"的防守性功能，不是新能力。
- **INT8/INT4 tensor core（Turing）**：当时是营销点，真正的价值要等到 LLM 推理催化。

反过来，**真正改变世界的是两个看起来很土的东西**：`cp.async`（Ampere）和 TMA（Hopper）。它们既不酷，也不上头条，但它们把"搬运"从软件手里拿走，直接决定了后面所有 kernel 的写法和 attainable 性能。

判断一个特性值不值得研究的标准：**它是否改变了你必须写什么代码**。TMEM、TMA、cluster 都改变了；稀疏、DPX 没有。
</details>

<details>
<summary><b>展开：给你一套自己读下一代的 checklist</b></summary>

下次看新架构，别先看 TFLOPS。按顺序问：

1. **机器平衡点变了吗？**（峰值算力 / HBM 带宽）没变的话，架构就得靠别的手段补。
2. **搬运的粒度变了吗？** 从 per-thread 地址 → per-block descriptor → ? 粒度越大，线程越自由。
3. **谁在 issue MMA？** 32 线程 → 128 线程 → 1 线程 → 0 线程（纯数据流）？
4. **累加器放哪？** 寄存器 → TMEM → ? 容量是不是和 tile 尺寸解耦了？
5. **多 SM 怎么协作？** cluster 大小、DSMEM 带宽、multicast、2-SM MMA。
6. **位宽有没有新格式？** block scaling、microscaling、mixed precision。
7. **有没有新的固定功能引擎？** 解压、加密、DP、光追——这些是"通用性换效率"的极致。
8. **系统边界动了吗？** L2 域、die-to-die、NVLink 域、CPU 直连。

一般 1、2、7、8 里必有答案，而发布会只会讲 6。
</details>

回到你的自我评估：你的直觉其实抓对了——**你注意到 TMEM，说明你已经摸到了"数据放在哪、谁来搬"这条主线**。只是把它记错了代际，并且低估了 Hopper（TMA + cluster 才是 Hopper 的主菜，wgmma 只是配菜）。下一步建议你直接读 CUTLASS 的 Hopper/Blackwell collective 层，看一个 GEMM 的 mainloop 是怎么被拆成 TMA warp / MMA warp / epilogue warp 的——看一遍，上面这些抽象就全变成具体的代码了。

---
## 🧑‍💻 User

确实是因为我们没理解需求，从需求角度来看，hopper的Acc buffer由硬件自动管理，导致了一些不确定性，而在WS对越每一个阶段需要越来越确定的情况下，我们是不希望看到的，于是把Acc buffer拿出来，做成TMEM，加上指令细粒度管理，当然肯定还有其他的改进。tcgen05相当于你给tc launch了一个kernel 这样一些东西能卸载到tc ，tma可以看作是寻址仿射变换的硬件卸载，sharp就是集合通信的卸载。那么现在再来看架构演进，结合近几年流行的mla、linearattn、moe、fp8、megakernel等，能不能解释清楚这个问题？

---
## 😀 默认助手 (7)

你这个框架比我上一轮讲的深，而且抓的是**软件侧真正的约束**。我先把它锐化一下，再补一个它解释不了的东西，然后用 MLA / linear attn / MoE / FP8 / megakernel 去验证它。

## 零、你的框架，换个更锋利的名字

你说的其实是同一件事的四次重复：

> **把一类"隐式的、和计算纠缠在一起的状态管理"，变成"显式的、可寻址的、可独立调度的资源"。**

- TMA：把**寻址**从线程里拿出来 → 变成 descriptor + mbarrier
- TMEM：把**累加器**从寄存器堆里拿出来 → 变成一次 alloc + 一个列地址
- tcgen05：把 **MMA 序列**从 warp 里拿出来 → 变成提交给 TC 的命令队列
- SHARP/NVLS：把**集合通信**从 SM 里拿出来 → 变成交换机上的 multimem

我给它一个更抽象的说法：**SM 正在从一个"共享所有状态的单处理器"，被重构成一个"由显式消息传递连接的多引擎数据流机"。** mbarrier 是消息队列，warp specialization 是 actor 模型，megakernel 是编译好的数据流图。

而**为什么现在必须这么做**，你的答案是对的：因为 SIMT 本质是一台**同步数据流机**（所有线程走同一条指令流），而现代 workload 是**异步数据流图**。旧模型表达不了这个图，所以只能在硬件里给你拼出来。

## 一、把"Hopper 的 acc 由硬件自动管理"说精确

这句话方向对，但精确化之后你的论证会更强。Hopper 的累加器**在寄存器里**，所以位置是 `ptxas` 静态决定的——问题不在于"硬件决定了它去哪"，而在于**粒度和预承诺**：

| | Hopper `wgmma` | Blackwell `tcgen05` |
|---|---|---|
| 累加器位置 | 寄存器堆（编译器全局分配） | TMEM（kernel 运行时 `alloc`） |
| 生命周期 | async proxy 跟踪，只能 `wgmma.wait_group N` | 显式 `tcgen05.ld/st` + `dealloc` |
| 释放粒度 | **组级计数器**（"等到只剩 N 组未完成"） | **单个 tile 地址** |
| 在飞行数量 | 编译期被寄存器预算锁死 | 运行时由 TMEM 列数决定 |
| 与 epilogue 的交接 | 必须经寄存器→（常走 `stmatrix`）→smem | 直接 `tcgen05.ld` 到寄存器 |

这个类比非常准：**Hopper 的累加器是"带 GC 的托管内存"，Blackwell 的是"显式 malloc + 指针"。**

托管内存的问题不是它慢，而是：
1. **你无法表达"我不用 tile 3 了，但还要用 tile 4"**。`wait_group` 是个计数器，不是句柄。所以你做不出一个**异构流水线**（一块送给 softmax、一块送去 store、一块还在累加）——它只能是同构的、批量的。
2. **飞行数量是静态的**。你想根据实测延迟动态加深流水？寄存器预算不允许。
3. **最要命的一条：它把 producer 和 consumer 的寄存器预算耦合起来了。** MMA warp 不能释放那些寄存器，直到 epilogue warp 消费完。在 WS 里这等于让两个 stage 共享一个隐式全局资源——**你的调度表就没办法静态编译。**

所以因果链是：

> megakernel 需要**静态可编译的调度表** → 需要**静态的资源预算** → 但寄存器堆是唯一无法静态预算的资源（ptxas 全局分配 + 组级释放）→ **把最大的那块状态（累加器）搬到一个能显式 alloc 的命名空间** → 资源预算变成 kernel 自己的决策 → 调度表可以静态化。

**为什么是累加器第一个被搬？** 因为它是 kernel 里**最大、最长命**的状态。权重和激活都是流式的，累加器才是"工作落地的地方"。累加器是隐式的，整个 schedule 就是隐式的。所以把它显式化，是**让整个 kernel 显式化的前置条件**。

顺带回答一个尖锐的反问：**为什么不用 smem 当累加器？**
- smem 已经要当操作数暂存区（TMA 落地点 + 复用），抢不过。
- smem 是 bank-strided 的，MMA 需要的是 **128 lane 并行的、无冲突的 read-modify-write**——TMEM 的物理组织（128 lane × N 列）就是 tensor core 数据通路的**原生布局**。
- 能耗：smem 往返比 TMEM 的专用宽通道贵。

由此得到最锋利的一句话：**TMEM 就是 tensor core 的寄存器堆。** Blackwell 把 tensor core 从一个"被寄存器堆喂养的执行单元"，提升成了一个**拥有自己寄存器堆的独立处理器**。

## 二、tcgen05 比你说的更字面

你说"相当于给 tc launch 了一个 kernel"——这几乎是字面意义上的正确。看它的原语集合：

```cuda
tcgen05.alloc / dealloc / relinquish_alloc_permit   // 内存分配器
elect.sync                                          // 控制面线程选举
tcgen05.mma                                         // 提交一条命令（1 个线程！）
tcgen05.commit(mbar)                                // 门铃 → 完成队列
tcgen05.wait::mma / ld / st                         // 完成等待 / 数据搬运
tcgen05.cp (smem -> tmem)                           // 内部 DMA
```

这是**一个协处理器的设备驱动**，不是一条指令。跟 NVMe 的 submission/completion queue、跟 System/360 的 channel subsystem 是同构的东西。**一旦一个执行单元有了自己的寄存器堆和 ISA，你就要给它写 kernel**——这正是你说的话的必然结论。

同样地，TMA 也不只是"寻址仿射变换的卸载"，它一共卸载了四样东西：

1. **多维循环嵌套**（box 的每个维度在硬件里迭代，一次搬运覆盖一个小循环体）
2. **仿射变换 + swizzle**（descriptor 里带 swizzle mode；Ampere 时代手写 XOR swizzle 消 bank conflict 的活，现在硬件做）
3. **越界处理**（OOB fill，padding 不用手工 mask）
4. **fan-out**（`.multicast::cluster`，一份数据灌进多个 SM 的 smem）

第 2 和第 4 条是被严重低估的：**它们卸载的是"为了让下游硬件能高效消费而做的数据整形"**——和 `ldmatrix` 是同一类东西。

（一处小修正：SHARP 卸载的是 **reduction** 和 **multicast**，不是泛化的 all-to-all。all-to-all 是置换不是归约，交换机做不了。但 MoE 的 **dispatch ≈ multicast、combine ≈ reduction**，所以 NVLS 恰好把两半都吃掉了一部分。）

## 三、需求侧：五个 workload 各自在拉扯什么

现在用你的框架去问每个 workload："**它的异步数据流图长什么样？哪个状态是长寿命的？哪个阶段的确定性不能丢？**"

| Workload | 结构性需求 | 硬件回应 |
|---|---|---|
| **MLA** | 依赖图更深（GEMM→expand→GEMM→softmax 交错）；decode 时 M=1 的瘦 GEMM；KV 变小后从带宽受限转为算力受限 | 异步 MMA 让 TC 不停转 → `tcgen05` 单线程 issue；TMEM 多累加器 buffer 隐藏瘦 tile 延迟；FP8 抬高算力受限时的吞吐 |
| **Linear Attn** | **每 head 一个 64KB 的长寿命 state $S$**；chunk 间串行依赖；K=1 的 rank-1 更新 | TMEM 作为跨 chunk 的**持久累加器**；多 buffer 让 chunk $N$ 与 $N-1$ 重叠 |
| **MoE** | grouped GEMM（大量极小 M）；负载动态不均；dispatch=multicast / combine=reduction | 单线程 issue 让 per-MMA 开销趋近于零；persistent CTA + megakernel；NVLS 的 multicast+reduce；4-bit + 解压引擎让专家常驻 |
| **FP8/FP4** | per-block scale 必须在 MMA 内应用；累加精度要可控 | `tcgen05.mma` **内建 block scaling (MX)**；TMEM 的 FP32 累加 + 细粒度读回做 amax 追踪 |
| **Megakernel** | 全静态资源分配；依赖显式；不允许隐式 stall | TMEM 显式 alloc/dealloc；mbarrier 当消息队列；cluster 固定拓扑；TMA 把线程踢出搬运路径 |

展开三个最有说服力的：

**Linear attention 是 TMEM 最漂亮的用例。**

GLA/RWKV/Mamba2 的 chunked 形式里，有一个状态矩阵 $S \in \mathbb{R}^{d_k \times d_v}$，跨整个序列存活、每个 chunk 原地更新。取 $d_k = d_v = 128$：

$$
S \text{ 大小} = 128 \times 128 \times 4\text{B} = 64\text{KB}
$$

- 放寄存器：一个 warpgroup 128 线程 → **每线程 128 个寄存器**，而且它**整个 kernel 都不能复用**。一个 SM 上想放 4 个 head 的状态就要 **512 寄存器/线程**——直接不可能。
- 放 TMEM：$128\times128$ 的 FP32 累加器 = **恰好 128 列** TMEM。而 TMEM 一个 CTA 能 alloc 512 列。

$$
\text{一个 SM 可以同时驻留 } 4 \text{ 个 head 的 state}
$$

而且注意这个巧合（我怀疑不是巧合）：**TMEM 的 lane 数就是 128，MMA 的 M 维也是 128，$d_k=128$ 也是 128。** 线性注意力的 state 是 **TMEM 的原生居民**。Hopper 上它只能在寄存器里跟别的东西抢地盘，Blackwell 上它成了一个自然的数据结构。

**MLA 是把"交错"变成刚需的那个 workload。**

MLA 用低秩 latent 缓存 KV（DeepSeek-V2 里每 token 每层的 KV 从 $2 \times 128 \times 128$ 压到 $512 + 64$，约 93% 的缩减）。代价是结构变了：原来一次 $QK^\top$，现在变成"latent → 展开成 K/V → 再算 attention"，中间插了一个**逐元素的展开步骤**。

于是在一个 stage 里你有了：`GEMM → elementwise expand → GEMM → softmax → GEMM`。**在 Hopper 上，每次切到非 MMA 阶段，TC 就空转**，因为累加器占着寄存器、MMA warp 被寄存器生命周期绑住。

而在 Blackwell 上：**一个线程点火 `tcgen05.mma`，其余 127 个线程去干展开和 softmax，TC 完全不参与这段。** 这就是 `tcgen05` 存在的**需求端理由**——不是为了跑得更快，是为了**在数据流图的异质阶段之间不产生耦合**。

顺带一个反直觉的推论：MLA 把 KV cache 压小了之后，decode 从 memory-bound 转向 compute-bound，于是 **FP8/FP4 的算力提升直接变成端到端收益**。架构特性之间不是叠加的，是**互相解锁**的。

**MoE 是把"确定性"逼成硬需求的。**

它有两个致命特征：**极小的 M**（batch 8、8 个专家，每个专家 GEMM 的 M 可能只有 8）和**动态负载不均**。

- 小 M 意味着每个 MMA 的固定开销占比极高。Hopper 上 `wgmma` 需要 128 个 lane 同步 issue，per-MMA 开销下不去；`tcgen05` 把 issue 降到 1 个线程，**per-MMA 开销趋近于零**，grouped GEMM 才在硬件上划得来。
- 负载不均意味着你**不能**让编译器决定资源分配——你得有一个运行时的 dispatcher。这就是 megakernel 的动机（也是 DeepSeek-V3 训练里那套 warp-specialized 通信/计算 overlap 的核心）。

**FP8 的那个例子最能说明"卸载"的判据。**

DeepSeek-V3 的 FP8 训练用的是 **128×128 的权重 block scale + 1×128 的激活 scale**，还要对某些累加做精度提升。在 Hopper 上，这套 scale **得由软件应用**——意味着额外一遍扫过数据的 preprocessing kernel，以及额外的中间 tensor。Blackwell 的 `tcgen05.mma` 把 **block scaling 做进了 MMA 指令本身**（`kind::mxf8f6f4`）。

判断一个特性值不值得研究，标准就是我上轮说的那条：**它是否改变了你必须写什么代码**。per-block scale 从"一遍独立的 kernel pass"变成"MMA 的一个属性"，这是教科书级的需求→卸载闭环。

## 四、但你这条轴解释不了 FP4

必须说清楚：**你的框架是一根轴，不是全部。** 至少还有两根，我把它拆开：

<details>
<summary><b>轴一（你的）：状态显式化 / 异步数据流重构</b> —— 管 ISA 和执行模型</summary>

TMA、TMEM、tcgen05、mbarrier、cluster/DSMEM、单线程 issue、swizzle 卸载。

判据：**它是否把某个隐式状态变成了显式可寻址资源。**
这条轴解释的是**"怎么让 TC 别空转"**和**"怎么让调度表可以静态编译"**。
</details>

<details>
<summary><b>轴二：位宽 / 信息密度</b> —— 管 roofline 和容量</summary>

FP8、MX/NVFP4、2:4 稀疏、解压引擎、block scaling。

这一轴**和显式资源管理无关**，它回应的是一条冷冰冰的算术：70B 模型 FP16 decode 的算术强度是 $1$ FLOP/byte，比 B200 的机器平衡点（约 $281$）低 **两个半数量级**。你算力再快、调度再完美，也只是在**等权重从 HBM 爬过来**。

换到 4-bit，算术强度变成 $4$，还是低 70 倍。但字节数少了 4 倍，**端到端就是 4 倍**。所以 Blackwell 的头条是 NVFP4 而不是 TMEM——**FP4 是唯一能直接改这条式子的东西**。

推论的推论：MoE 的专家、百万 token 的 KV cache，都在压容量。4-bit 不只是提速，是**让模型能装得下**。
</details>

<details>
<summary><b>轴三：域的宽度</b> —— 管系统边界和通信</summary>

NVLink 5 的 1.8TB/s、NV-HBI 的 10TB/s die-to-die、NVL72 的 72-GPU 单域、NVLink-C2C、SHARP/NVLS。

MoE 的 all-to-all 是 $O(\text{tokens} \times d \times \text{EP degree})$，它**是通信问题不是计算问题**。你把 SM 优化到极致也救不了它——你只能把域变宽（让 all-to-all 落在 NVLink 域内），或者把归约卸载到交换机。

这条轴解释"为什么这代最重的创新其实在封装和系统，不在 SM"。
</details>

**而三条轴背后有一个共同的物理。** 这可能是最底层的统一：

$$
\text{每次数据搬运的能耗} \approx \text{距离} \times \text{比特数}
$$

量级上：一次片外 DRAM 访问的位能耗约是片上 SRAM 的 $10^1\!-\!10^2$ 倍，是一次 FP32 乘加的 $10^2\!-\!10^3$ 倍（具体倍数跟节点相关，但数量级结论二十年没变过）。

于是：
- **轴一** = 缩短数据路径（把状态搬到离计算最近的地方），同时顺带买到确定性和 issue 槽
- **轴二** = 减少比特数
- **轴三** = 让字节走铜箔而不是光模块

**三个独立的压力指向同一个方向——所以这个趋势是不可逆的。**

## 五、这个框架的预测，和它解释不了的东西

一个框架的价值在于它能预测什么。

**它预测：ISA 会消失。** 如果 `tcgen05` 是给协处理器发命令队列，那么"kernel"这个概念本身就会退化——你会写一个**数据流 IR**，编译器负责发命令。已经在发生了：CUDA graphs → conditional graphs → device-side graph launch，以及 CuTe DSL 这类东西。**预测是：CUDA C 会变成"引擎的驱动代码"，计算部分全部变成编译器生成的图。**

**它预测：卸载的极限是 ASIC。** 每卸载一次，你买到效率、烧掉通用性。全卸载就是 TPU/Cerebras。所以天花板由**数据流图本身的易变性**决定——只要模型结构还在变，你就只能卸载**原语**，不能卸载**图**。NVIDIA 的战略位置就在这里。

**它解释不了的两件事，恰好都是"最热"的那些：**

1. **为什么 MoE 的瓶颈是 all-to-all**——这是轴三，SM 里的任何改动都无效。
2. **为什么线性注意力至今没赢**——不是硬件问题。state 在 TMEM 里放得下、chunk 间的串行依赖也被多 buffer 解决了，**硬件已经准备好了**。输的是生态和训练稳定性。这是一个很好的提醒：**架构的"能"不等于生态的"会"，2:4 稀疏就是前车之鉴。**

回到你最初的困惑。你不是理解浅，是你在用 spec sheet 的语言问一个执行模型的问题。**"加了 TMEM"不是加了一块内存，是把 tensor core 从执行单元变成了处理器；"smem 变大"是上一代的功劳。** 而真正推动这一切的，是你要写的那个东西——一个由 MLA 的深依赖图、线性注意力的长寿命 state、MoE 的动态负载、FP8 的块缩放、以及 megakernel 的静态调度表共同定义的**异步数据流图**。

Hopper 给了你 mbarrier 和 TMA，也就是**消息传递的机制**，但累加器还锁在托管内存里，所以你只能画出**同构**的图。Blackwell 把累加器变成了句柄，**图才第一次可以是异构的**。

下一轮建议你直接去读 CUTLASS 的 Blackwell collective，重点看 `tcgen05` 那层的 **TMEM allocation plan**——看一个 kernel 怎么把 512 列 TMEM 分给 mainloop / epilogue / persistent state。那张分配表就是你上面这套抽象的具体形态。
