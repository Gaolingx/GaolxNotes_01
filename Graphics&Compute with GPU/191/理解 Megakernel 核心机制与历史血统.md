# 理解 Megakernel 核心机制与历史血统

## 🧑‍💻 User

megakernel相比于理想优化过的显式cudagraph有优势嘛
cudagraph没法一个warp做通信一个warp调mma来着
megakernel似乎能融确定性和动态的控制流
cudagraph必须确定，mega才能不确定 mega本质之一就是运行时dispatch

---
## 🧑‍💻 User

megakernel相比于理想优化过的显式cudagraph有优势嘛
cudagraph没法一个warp做通信一个warp调mma来着

---
## 😀 默认助手 (7)

## 一句话结论

**CUDA Graph 的调度单元是 grid，megakernel 的调度单元是 warp 或 task。** 在“理想图”这个前提下（launch overhead 归零、PDL/tail-launch 完全重叠、device-side launch 可用），megakernel 剩下的优势只有从这个粒度差里派生出来的那几条；其余直觉上的优势（省 launch、省 fusion 开销）基本都被图抵消掉了。因此该问的是“你的 workload 里有没有 grid 粒度强同步导致的结构性浪费”，而非普遍意义上的“谁更快”。

下面逐条校准你的三个判断。

---

## ① “cudagraph 没法一个 warp 通信一个 warp 调 mma”：一半成立

**不成立的一半**：warp specialization 是 kernel 内部的事，跟它是不是被 graph 调度毫无关系。一个 graph node 完全可以是一个 warp-specialized kernel：TMA producer warp + MMA consumer warp + epilogue warp，FlashAttention-3、DeepGEMM 都是这么写的，而且它们**经常就运行在 CUDA graph 里**。

**成立的一半**：graph 的边是 node 级 join。你无法让 node $i$ 里的一个通信 warp 直接和 node $i+1$ 里的一个 MMA warp 用 mbarrier/DSMEM 做细粒度握手而不发生 grid 级排空。跨 node 想做 warp 级耦合，唯一的办法就是把两个逻辑 phase 塞进同一个 kernel，而**那一刻你已经建了一个 megakernel**。

所以更精确的表述是：图的限制不在于 warp specialization 本身，而在于**跨 grid 边界的 warp 级耦合，以及 warp 角色能否在运行时重新分配**。warp-specialized kernel 里“几个 warp 做通信、几个做计算”是编译期或启动期固定的；megakernel 可以在运行时按负载调整通信 warp 与计算 warp 的比例。DeepEP 这类方案在图上跑得也不错，前提是切分比例静态可预测。

## ② “cudagraph 必须确定，mega 才能不确定”：这里有个概念滑移

需要把两件事分开：

| | 含义 | 图的现状 | mega 的现状 |
|---|---|---|---|
| **dynamic** | 控制流依赖运行时数据 | 有条件节点（IF/WHILE/SWITCH，CUDA 12.3+）和 device-side graph launch，**可以动态**，但决策粒度是整个 node | 可以在 warp/CTA 粒度动态 |
| **nondeterministic** | 同样输入跑两次结果不同 | 位级可重现 | 用 atomic work queue 或 work stealing 时，**通常不可重现** |

关键点在于：**你要的是 dynamic，而非 nondeterministic**。绝大多数场景（训练复现、debug、CI 对齐）把 nondeterminism 看作 bug，而不是 feature。图真正的短板在于**动态性的粒度太粗**（跳过或重放整个 node，无法进行“第 37 号 CTA 去执行别的任务”），并非“无法动态”。而 magakernel 完全可以做成**dynamic 且 deterministic**（用确定性调度，而非 atomic 抢占）。所以别把 nondeterminism 当成卖点，它是成本，并非收益。

## ③ “mega 本质是运行时 dispatch”：对，但不够 distinguishing

device-side graph launch 也是运行时 dispatch，由设备在运行时决定下一个 node。真正的差异还是**调度单元**：图在运行时 dispatch 一个 grid，mega 在运行时 dispatch 一个 warp-group、一个 CTA 或一个 task。这决定了后面所有结论。

---

## 在“理想图”之下，mega 真正剩下的三件事

**1. 依赖粒度：BSP join 变成 dataflow**

图里每个 node 边界是全体 CTA 的完全 join：

$$T_{\text{total}} = \sum_{\text{node}} \left( \max_{\text{CTA}} T_{\text{CTA}} + T_{\text{drain}} \right)$$

即使 $T_{\text{drain}}\to 0$，$\max_{\text{CTA}}$ 这一项还在，也就是说：**最慢的那个 CTA 决定整层**。在 MoE 专家倾斜、变长序列、spec decoding（接受长度随机）这些场景下，$\max/\text{mean}$ 轻松到 2–10×。megakernel 用 work queue 把整个 GPU 变成一个大工作池，并且加入了波量化：

$$T_{\text{wave}} = \left\lceil \frac{N}{B \cdot \text{CTAs/SM}} \right\rceil$$

图里每个小 node 各自向上取整，浪费的是 $\frac12$ 波 SM 时间再乘以 node 数；mega 全局只取整一次。**这是最实在的收益，而且完全不受“理想图”假设的影响。**

多了一个能力：**split-phase / 局部 barrier**。下游 CTA 只要它依赖的那块数据 ready 就能开始执行，不必等全体上游 CTA 完成。这相当于把 bulk-synchronous 换成 dataflow。用条件节点近似也可以，但粒度始终是整 grid。

**2. 片上状态跨 phase 存活**

kernel 边界会清空 register 和 smem。想做到 attention → MLP 的 intermediate 全程不落 HBM，只能靠融合。注意：**这一点不是 megakernel 的专利**，一个融合 kernel 作为图的单个节点同样能拿到。mega 独占的是：**有条件地**跨动态决策保持片上状态（dispatch 说“这个专家走完可以直接继续”就不卸载数据），这是图表达不了的。

**3. sub-grid 粒度的调度决定**

例如“本层通信推迟、先把计算补上”“这批 CTA 转去做 draft model”。图的粒度决定这件事没法做，除非你退化成“把所有可能分支都编译进一个 persistent kernel”，也就是 megakernel 本身。

---

## 但有硬代价，不能只看好处

- **Occupancy 取并集。** 整个 mega 的 register/smem 是各 phase 的 max。如果某个 phase 需要 200 reg，其余只需 40，所有 phase 都会按 200 运行。`setmaxnreg`（sm_90）只能缓解 warpgroup 级别的部分问题，不能救整体。图则允许 node 级独立设置。**这条在 LLM inference 的 megakernel 实践中经常是致命的。**
- **grid 尺寸固定。** 必须精确给到 $\text{SM数} \times \text{occupancy}$ 且常驻，占不满就要用空闲 block 顶替。
- **正确性风险。** grid sync 要求 cooperative launch，或在全常驻前提下自定义 barrier；漏一个 `__threadfence()` 就是静默数据竞争，barrier 参与集算错则无法排查的会卡死。
- **可观测性几乎为零。** Nsight 里只有一个 kernel，各 phase 的占比、停顿原因全靠自己埋点。
- **失去 free 的 kernel-boundary memory fence**，每个 phase 交界要显式 fence + barrier。通常比 kernel 排空便宜，但在细粒度 phase 上就不再免费。
- **工程/编译成本**，模板爆炸、编译时间、以及“每个 phase 独立 autotune”这件事彻底做不了（一个 kernel 只能有一套 launch bounds）。

---

## 判定表

| 场景 | 理想 graph | megakernel |
|---|---|---|
| 定 shape prefill、训练静态 batch | **基本最优**，几乎拿不到额外收益 | 纯亏（occupancy 并集） |
| batch=1 decode、算子多而小 | graph + PDL 已经接近极限 | launch 上无增量收益，只剩波量化 |
| MoE 长尾 / 变长 batch / spec decoding | 强同步 + padding，$\max$ 那一项吃满 | **真实收益所在** |
| 通信-计算的亚层重叠、比例动态 | warp-specialized node 静态切分 | 可运行时 rebalance |
| 需要跨 phase 留住片上的动态中间态 | 表达不了 | 表达得了 |
| 交付/维护/可调试性 | 赢 | 输 |

---

## 实践中的折中路线（比二选一更值得看）

1. **MPK 路线**（CMU/Mirage 组）：每个 SM 一个 task queue，**被调度的单元是 CUDA graph 子图**。这样把“mega 的调度粒度”和“图的 node 级独立配置”两个好处合并，代价是子图之间仍有 grid 语义，片上状态无法跨子图保留。
2. **图 + PDL + 细粒度 node 拆分**：把每个逻辑 phase 拆成更多小 node，用 `cudaGridDependencySynchronize` 提前重叠尾部。能抵消掉相当一部分“波量化 + 排空”损耗，剩下抵消不了的就是真正的 mega-only 区间。
3. **少量超大融合 kernel 作为图节点**：绝大多数在售的“megakernel”其实只是这个。收益来自融合（片上复用），不来自调度，别把它算到 mega 的账上。
4. **device-side graph launch 做动态 dispatch**：给图补上“运行时选分支”，粒度仍是整 grid，但工程风险远低于自建 persistent kernel。

需要给一个可操作的判据的话：**先量 $\max_{\text{CTA}}/\text{mean}_{\text{CTA}}$ 和每层的波量化浪费。如果这两个加起来不到 15%，别做 megakernel，把图优化到头就赢了；超过 30%，且 phase 间资源需求差距不到 2×，才值得上 mega。**

（以上是从架构与调度语义推的，没有你具体 workload 的 profile 数据；如果你能说明是哪类负载——训练、decode、MoE 还是 agentic 变并发——我可以把结论收窄成更明确的方案建议。）

---
## 🧑‍💻 User

block的schedule和compute task的schedule有什么区别，到底啥叫不确定的控制流啊

---
## 😀 默认助手 (7)

## 一、它们根本不是同一层的东西

区别可以一句话概括：

> **block scheduler 决定“在哪、何时”，task scheduler 决定“做什么”。**

| | 硬件 block scheduler | 软件 task scheduler |
|---|---|---|
| 调度单元 | 一个 CTA（几何形状在 launch 时就被固定） | 一个 task（tile、expert、phase 片段、子图，大小任意） |
| 由谁决定 | GigaThread Engine（硬件 work distributor） | 设备端代码（一个专门负责调度 warp 或 CTA） |
| 决定依据 | 只有**资源是否够放**：寄存器、smem、warp slot、block slot | 可以使用任意信息：数据是否就绪、优先级、负载、队列长度 |
| `blockIdx → 工作` 的映射 | **编译期静态**，写在 kernel 里 | **运行时动态**，由 pop 出来的是什么决定 |
| 有无负载均衡 | 无 | 有（work stealing、细粒度任务） |
| 数据依赖感知 | 零 | 可以做（全局 ready counter / DAG） |
| 可抢占 | 否 | 可以（检查 flag 提前退出） |
| 开销 | 流水化，基本重叠掉 | 一次 atomic + 一次 L2 往返，约几百 cycle |
| 执行顺序 | **CUDA 规范：未定义** | 由你定义（或者你故意不定义） |

关键在最后一行往上数第三行：**block scheduler 从来不做“调度决策”，它只做“资源准入 + 填槽”。** 它不关心 block 0 和 block 5 谁先跑更优，也不允许它们相互等待。所有“聪明”都在你 launch 的那个 grid 形状里，而那是静态的、launch 时定死的。

由此直接推出波量化和长尾：

$$T = \left\lceil \frac{N}{B \cdot o} \right\rceil \cdot T_{\text{block}}, \qquad \text{末波利用率} = \frac{N \bmod (B o)}{B o}$$

而硬件对这件事完全无能为力——它不能把 block 劈开，也不能让 block 3 干完一半就去干 block 7 的活。

### 一个能立刻打通概念的观察

如果你在 kernel 里写：

```cuda
while (auto t = queue.pop()) { run(t); }
```

那 block scheduler 只运行了**一次**，之后所有的调度工作都由你负责。**“block 调度” 和 “task 调度” 的区别，本质上就是 “one-shot” 和 “persistent” 的区别。** 这也正是 megakernel 与普通 kernel 的分界线。

### 顺便回答它和 CUDA Graph 的关系

**CUDA Graph 完全没有触碰 block scheduler。** 它优化的只是 *grid 的提交路径*（把 host 侧几百次 launch 压成一次 replay）。每个 node 仍然是“打一个 grid 进去，硬件照旧填槽”。所以上一轮那个 $\max_{\text{CTA}}$ 项，图优化到头也动不了——那不是提交开销，是调度语义。这也解释了为什么 megakernel 是唯一能改这一层的东西。

---

## 二、“不确定的控制流”其实不是一个有定义的概念

控制流是 $(输入, 调度)$ 的函数。你真正想说的多半是下面三件不同的事，它们的性质天差地别：

<details>
<summary><b>(a) 动态控制流 / data-dependent control flow</b> —— 这是 feature</summary>

条件依赖运行时数据：`if (token > 0)`、`while (!done)`、MoE 路由后决定走哪个专家、spec decoding 决定接受几个 token。

**给定输入，它是完全确定的。** 单线程下跑一万次，路径一模一样。CUDA Graph 的问题不是“不能动态”，而是动态的**粒度是整个 node**（条件节点只能整节点跳过/重放），做不到“第 37 号 CTA 换个活干”。
</details>

<details>
<summary><b>(b) 调度不确定 / schedule nondeterminism</b> —— 这是 cost</summary>

**独立工作项之间的执行顺序**在 run-to-run 之间变化，因为依赖 SM 之间的相对时序。

注意这里有两档强度：
- **block scheduler 的未定义顺序是“弱”的**：规范说 unspecified，但硬件没有随机源，同样的 binary 同样 occupancy 下实际顺序高度可重现。所以普通 kernel 在实践中位级可复现。
- **atomic work queue 是“强”的**：谁先抢到谁干，纯粹看时序。这才让你上一条所依赖的可重现性真正失效。
</details>

<details>
<summary><b>(c) 结果不确定 / bit-level 不可重现</b> —— 这才是多数人说的那个词</summary>

需要 (b) **并且** 有非结合运算参与：

- 跨 CTA 的浮点 `atomicAdd`：加法顺序变了，$ (a+b)+c \neq a+(b+c) $，最后一位漂移
- `atomicMax` 平局时取谁
- split-K 归约的分组方式变了

然后漂移会向下游传播，偶尔改变 argmax / 采样结果。

**而且注意：这不是 megakernel 带来的新问题。** 你在普通 CUDA 里用跨 block 浮点原子加，今天就已经在产生结果不确定了。megakernel 只是把这个东西从“藏在硬件里”变成“摆在你手上”。
</details>

### 所以正确的说法是

| 你想要的 | 对应的是 | 谁提供 |
|---|---|---|
| 细粒度、数据依赖的分支 | (a) | megakernel ✅ / 图 ❌（粒度粗） |
| 运行间位级可复现 | (c) 的否定 | 图 ✅ / 朴素 mega ❌ |

**“动态”和“不确定”是可以解耦的。** 一个任务队列完全可以是**动态且确定**的：按 task id 静态划分给 CTA、每个 warp 私有累加器、最后的合并按固定的树序进行。这样你拿到了细粒度动态调度的性能，同时保持 bit-reproducible。反过来，CUDA Graph 也是“动态（条件节点）但结果确定（顺序固定）”。

所以：

- 别把 nondeterminism 当卖点，它是**代价**，不是能力
- megakernel 的真正卖点是 **(a)**，而不是 (b)/(c)
- 做 mega 的时候，把 reducible 部分设计成确定性归约，是性价比最高的一笔工程投入

---

## 三、实操上这两件事怎么互相约束

task 必须**足够大**才能抵消 pop 的延迟（一次 L2 往返约 200–400 cycle）。这就是为什么 megakernel 的 task 粒度典型是几 µs 级而不是几 ns 级——你不能把 block scheduler 免费的 300ns dispatch 换成 300ns 的 atomic poll，收益会归零。

折中做法：
- **每个 CTA 一次 pop 一批**（batch pop），而不是每 tile 一次
- **每 SM 一个本地队列 + 周期性偷**，减少全局竞争
- **队列项预取**到 smem/register，pop 与上一个 task 的计算重叠
- **对确定性敏感的 phase 用静态分派**，只对长尾 phase 用动态队列

一句话收尾：**block scheduler 无知但免费，task scheduler 知情但收费。** 只有当“知情”能换回的 $\max/\text{mean}$ 和波量化浪费超过那笔费用时，才值得把前者换成后者。

---
## 🧑‍💻 User

能不能把cudagraph理解成diagonal fusion，horizontal的硬件特质，vertical的实现方式，diagonal的数据依赖

---
## 🧑‍💻 User

这么看来，scheduler本质就是一个cooperative yield，可以当你call scheduler时是个主动的syscall然后trap进bbk scheduler给你动态调度出一个东西，回来后就是下一个要算的东西

---
## 🧑‍💻 User

这么看来，scheduler本质就是一个cooperative yield，可以当你call scheduler时是个主动的syscall然后trap进bbk scheduler给你动态调度出一个东西，回来后就是下一个要算的东西
bbk最早也是这么抽象的，只是我觉得没必要搞得这么学究派也没提coroutine

---
## 😀 默认助手 (7)

## 判定：这框架有 60% 是对的，而且作为助记很好；但它会把 graph 的能力错误地归到一根轴上

对的部分你很准：graph 确实**既不是** horizontal 融合，**也**不是 vertical 融合，但它**同时挡住了这两者各自想解决的问题**。错的是一处关键错位，见第三节。

## 一、先把轴摆清楚（并补上你没提到的那一根）

| 轴 | 是什么 | 融合在干什么 |
|---|---|---|
| **H（horizontal）** | 空间轴：同一时刻能塞满多少独立工作 | 做**加法**：把独立 kernel 并进一个 grid，抬占用率和并行度 |
| **V（vertical）** | 时间轴或调用栈轴：依赖链上相邻 phase 的数据能否留在片上 | 做**减法**：砍掉 $B=\sum|\text{intermediate}|$ 的 HBM 往返 |
| **D（diagonal）** | 真实依赖轴：tile 级的数据流 | 沿它做**波前推进**：A 的 tile $i$ 与 B 的 tile $i-1$ 同时在跑 |
| **S（submission）** | 你漏掉的那根轴：grid 如何被提交给硬件 | 做**压缩**：$n$ 次 launch 变成 1 次 replay |

**graph 唯一真正独占工作量的，是 S 轴。** 它一个字都没改 H、V、D 的映射关系：每个 node 还是原来那个 kernel，还是原来那个 grid 形状，还是落 HBM。

## 二、逐轴核对：graph 实际参与了什么

**H：部分参与，而且是"打折的 H"。** 这个点你的直觉很好。graph 里没有依赖边的 node **确实可以并发执行**，所以它拿到了 horizontal 融合的一部分收益，即在填充硬件方面。但它拿不到另一部分：horizontal 融合是把多个 kernel **并成一个 grid**，共用一套几何形状和一套尾波；graph 是两个 grid 并发，**各自独立波量化，各自长尾，同时抢 SM**。所以是"并发"而不是"提高占用率"。

**V：完全放弃。** node 边界必然把中间结果写进 HBM 再读回来。graph 不但不做 V，它还把 V 的路径**固定下来**了。

**D：这是最关键的地方 —— 只拿到了拓扑，没拿到调度。**

## 三、错位所在：graph 是 diagonal 的拓扑 + 一维的调度

diagnonal fusion 的整个价值在于**波前（wavefront）遍历**：二维 tile 依赖空间里，$(i, j)$ 的合法执行序是一条对角线，所以你可以让 A 的 tile $i$ 和 B 的 tile $i-1$ 重叠，把依赖图"斜着切"。

graph 的依赖图**拓扑上确实是斜的**（DAG 嘛），但它的**执行语义是每个 node 边界上全 CTA 的 join**：

$$T_{\text{node}} = \max_{\text{CTA}} T_{\text{CTA}}, \qquad \text{B 开始} \iff \text{A 的全体 CTA 完成}$$

也就是说：

> **graph = diagonal 的拓扑，BSP（bulk-synchronous）的走法。**

它把"斜着的 DAG"压成了"竖直的层"。真 diagonal fusion 是沿着对角线逐个 tile 推进，graph 是沿着对角线逐层排空。**两者的依赖信息量完全不同**：graph 看到的是 node 级 DAG，是一个**保守过近似**（node 5 依赖 node 3 的全体输出，而不是某个 tile）；融合编译器看到的是 tile 级，可以更细地切。

所以"把 graph 理解成 diagonal fusion"最危险的地方在于：**你会预期它拿到波前流水线的收益，而它恰好拿不到。** 而这个问题在上一轮已经出现过两次了：$\max_{\text{CTA}}$ 项、波量化、BSP join，是同一件事的三种说法。

## 四、修正后的全景表

| 技术 | H（填满硬件） | V（省 HBM） | D（波前） | S（提交） |
|---|---|---|---|---|
| horizontal fusion | ✅ | ✗ | ✗ | 顺带 |
| vertical fusion | – | ✅ | 局部（链上） | 顺带 |
| diagonal fusion | ✅ | ✅ | ✅ 真波前 | 顺带 |
| **CUDA Graph** | 部分（node 并发，不共尾波） | ✗ | **只有拓扑，调度是 BSP** | ✅✅ 主角 |
| **Megakernel** | ✅ 动态均衡 | ✅ 片上存活 | ✅ 细粒度 dataflow | ✅ 顺带 |

## 五、这个命名错了，会付什么代价

会预测错两件事：

1. **会高估 graph 在长尾、变长、MoE 负载上的表现**，因为你会以为 D 轴已经覆盖了，实际上 D 轴上的波前从来没被覆盖。这正好解释了前两轮的结论为什么成立。
2. **会低估 fusion 的不可替代性**，因为 graph 的"diagonal"是免费的（不用重编译），fusion 的 diagonal 要重写 kernel。看到 graph 免费就以为能替代，是这类系统里最常见的误判。

## 六、如果一定要保留你的说法，我建议改成这句

> **CUDA Graph 是 D 轴上的"骨架投影" + S 轴上的完整解，H 轴打折，V 轴归零。**

或者更短：

> **graph 记下了依赖，但没有用它来调度，只是用它来提前提交。**

—— 这也顺带回答了你第一个问题里那个直觉：megakernel 和 graph 真正的分野不在"谁有 DAG"，两者都有；在于 **DAG 是拿来 `join` 的，还是拿来 `dispatch` 的**。前者是 BSP，后者才是 dataflow。

---
## 🧑‍💻 User

这么看来，scheduler本质就是一个cooperative yield，可以当你call scheduler时是个主动的syscall然后trap进bbk scheduler给你动态调度出一个东西，回来后就是下一个要算的东西
bbk最早也是这么抽象的，只是我觉得没必要搞得这么学究派也没提coroutine

我记得学界/nv最早对mega的场景是PDE算格子时，需要根据数据（局部residual）动态dispatch cta，他们不想关掉kernel再launch，于是就做了pde格子上data-dependent task enqueue & dequeue，称之为megakernel
一个大kernel里switch case完成数据依赖的动态调度就是mega kernel原始定义
不知道我说的对不对

---
## 😀 默认助手 (7)

## 一、"cooperative yield" 对，但它比 coroutine 弱，这是不该提 coroutine 的真正理由

你的抽象是对的：

```cuda
while ((t = pop()) != null) dispatch(t);   // yield point 就是 pop()
```

但把它叫 syscall 会高估三件事，把它叫 coroutine 会高估第四件：

| | 真 syscall | coroutine | mega worker |
|---|---|---|---|
| 特权级切换 | 有 | 无 | 无 |
| 上下文保存 | 硬件/内核 | 每 task 一个栈，suspend/resume | **不保存，run-to-completion** |
| 调度器是否全知 | 全知，可强制抢占 | 可 | **只知道自己被调用了** |
| 能否中途挂起 | 任意点 | 任意点 | **不能**（除非自己写检查点） |
| 返回值 | 状态码 | – | **一个新工作描述符** |

所以它不是"trap 进 scheduler"，而是**自愿把控制权交给一个 runloop，然后领一份新活**。关键差别在第二行：**mega worker 不 suspend 任务，它把任务做完**。没有 per-task 栈、没有栈切换、所有状态在寄存器和 smem 里就地复用。这正是它便宜的原因，也正是它**不能**叫做 coroutine 的原因（一叫 coroutine，别人就会问"能不能 mid-task yield"，答案是不能）。

比 syscall 更准的对照是 **libuv / 线程池 + run-to-completion**。而"cooperative"这个词你用得很准，因为 GPU 上**没有非自愿抢占**，所以：

$$\text{调度自由度} \le f(\text{你的 yield 频率})$$

一个不 yield 的 task 会永久霸占它的 worker，在 grid barrier 下直接死锁。这就是上一轮"task 要 ≥ 几 µs"那条规矩的根源：它不是性能建议，是**正确性约束的松弛版**。

还有一个你在措辞上放得很好的词，但可能没意识到它有多关键：**pop 出去的是"一个东西"而不是"一个任务索引"**。如果一个全局 MPMC 队列就是全部，那它其实没有调度器，没有 run queue、没有策略、没有优先级，只是一个**分区器（partitioner）**。有策略（locality-aware stealing、优先级、per-SM 本地队列）才叫 scheduler。这条线很多人搞混。

---

## 二、历史：机制你描述得准，"megakernel 这个词"则有三条血统

**你对的部分**：persistent kernel + data-dependent task enqueue/dequeue + 单 kernel 内 switch 分派，为的就是"不想关掉 kernel 再 launch"来响应局部 residual / 局部细化判据 —— 这是**确定无疑**的技术谱系。最经典的祖先是 **Tzeng / Patney / Davidson / Owens, "Task Management for Irregular-Parallel Workloads on the GPU" (I3D 2010)**，它的动机就是 AMR / PDE / 不规则工作负载里每个 cell 的工作量数据依赖，而且它的解法就是持久 kernel + 动态队列。AMR 里那个 **active set 压缩 + 重新 launch** 的瓶颈，是这条线上反复被引用的 motivating example。

**需要打折扣的部分**："megakernel" 这个名字至少有**三条互不相干**的来源，别把它当成单一发明：

1. **不规则 PDE / AMR / 图遍历**：就是你说的这条。核心是"工作项会**再生成**工作项"。
2. **渲染的 uber-shader**：一个 kernel 里把**所有材质/BSDF 分支全展开**，hit 到什么材质走哪个 case。这是虚拟分派在 GPU 上太贵逼出来的，历史比 GPU 通用计算还长。你第二句话"一个大 kernel 里 switch case 完成数据依赖的动态调度"**更贴这一条**，而不是 PDE 那条。
3. **现代 ML 系统**：MPK 那类框架（CMU Mirage 组）把 per-SM task queue + 子图当工作单元。

所以准确的说法是：**你描述的是这三条共享的那个"机制"**，而"megakernel"这个标签在不同社区被贴在了同一机制的不同实例上。你其实没有说错，只是把标签的唯一性说得太强了。

---

## 三、三个值得钉死的精确化

**(1) `switch` 不是修辞，它就是上一轮那个 occupancy 并集的来源。**

为什么用 switch 而不用函数指针？因为 GPU 上间接调用会阻止内联、破坏寄存器分配；用 switch 编译器能看到全部 callee，能把每个分支都内联优化。**代价正是：所有分支活在同一函数里，寄存器分配器必须满足 $\max_{\text{branch}} \text{regs}$。** 我上一轮说的"register 取并集"，机理就是这里——它不是一个笼统的工程缺陷，而是"用 switch 换内联"这个具体决策的直接后果。附带还有 I-cache 抖动。`setmaxnreg` 之所以存在，就是给这个结构兜底的。

**(2) 真正的分水岭是 "enqueue" 而不是 "dequeue"。**

只有 dequeue 的动态分区，工作总量在 kernel 启动时就确定了，那是**动态负载均衡**；能 enqueue，工作总量本身数据依赖地增长（细化一个 cell 会生出 8 个子 cell），那才是**调度器**。你说的"enqueue & dequeue"两个词都在，这点很准，而很多人只记住了 dequeue。

**(3) NVIDIA 真的做过"syscall 版"，然后输了。**

Kepler 的 Dynamic Parallelism（CDP，2012）就是硬件层面的"设备端 launch"：你在 device 上发起一次 launch，本质就是你想要的那个 syscall。它的失败正是 megakernel 至今还活着的原因：**DP 的 launch 成本比 host 侧还高**，而且它给了你"嵌套 launch"的假象却保留 grid drain。所以别把 mega 的动机只归结为"省 launch"——那个动机已经被 CDP 和 device-side graph launch 打掉一大半了。**真正活下来的是另外两条：负载均衡粒度、以及跨 phase 的片上状态存活。** 这也正是前三轮反复收敛到的那两条。

---

## 四、一句话

你的两句话分开看，第一句（cooperative yield）比 coroutine/syscall **更弱**所以更准，第二句（大 kernel 内 switch 做数据依赖分派）是三条血统共享的**机制**而不是某个单一"原始定义"。要补的一个前提是：**这个机制当年是为了"决策在 device 上产生、不想 round-trip 回 host"，而不是单纯为了省 launch 时间**——想清这一点，才能解释为什么硬件把 launch 成本打到接近零之后，megakernel 仍然没死。
