# Blackwell TMEM ld/st异步需显式wait同步

## 🧑‍💻 User

问个问题，就是blackwell架构gpu里面，同一个线程(warp)从tmem里面读取数据，在寄存器里面存储数据，再写入tmem，连续三轮，使用同一组寄存器，ld和st后到底需不需要wait?
是tcgen05.st相关的ptx

---
## 🧑‍💻 User

问个问题，就是blackwell架构gpu里面，同一个线程(warp)从tmem里面读取数据，在寄存器里面存储数据，再写入tmem，连续三轮，使用同一组寄存器，ld和st后到底需不需要wait?

---
## 😀 默认助手 (7)

## 先说结论

**需要，而且每一轮两个 wait 都不能省。** 三轮循环里，用同一组寄存器的标准写法是**每轮 2 个 wait，一共 6 个**：

```
ld  →  wait::ld  →  (改寄存器)  →  st  →  wait::st
```

如果你的代码现在在这三轮里一个 wait 都没写、结果还是“对的”，那只是碰巧：硬件通常会把数据提前送到，属于未定义行为（UB），换 shape、换 `.num`、换编译器版本或增加寄存器压力就可能出错。

---

## 为什么两个都必须有：三类 hazard

`tcgen05.ld` 和 `tcgen05.st` 都是**异步（non-blocking）**指令。指令发射出去之后，TMEM↔寄存器的数据搬运还在后台进行。PTX 明确规定：`tcgen05.wait::ld` / `tcgen05.wait::st` 会阻塞当前线程，直到**该线程此前发出的所有同类操作**完成（注意：这个等待粒度是“全部此前同类操作”，并不针对某一条指令，这一点后面讲优化时很关键）。

在这个“用同一组寄存器做三轮读-改-写”的场景里，同时踩到了三类冲突：

| 冲突 | 涉及对象 | 谁来解决 | 少了会怎样 |
|---|---|---|---|
| **RAW** | 寄存器 | `tcgen05.wait::ld` | `st`（或你后续的计算）读到的是旧值或半成品 |
| **WAR** | 寄存器 | `tcgen05.wait::st` | 下一轮 `ld` 覆盖了还在被 in-flight `st` 读取的寄存器 → 写回 TMEM 的数据被破坏 |
| **RAW/WAW** | TMEM 同一地址 | `tcgen05.wait::st` | 下一轮 `ld` 读到的是上一轮 `st` 之前的旧数据 |

这里最容易想错的是第二行：很多人以为 `tcgen05.st` 在发射的那一刻就把寄存器“抓走”了，所以下一轮 `ld` 复用同一组寄存器没有问题。事实并非如此：一次 `tcgen05.st.sync.aligned.32x32b.x32` 要从寄存器堆里读出 $32\ \text{lanes}\times 32\ \text{cells} \times 4\ \text{B} = 4\ \text{KB}$（每线程 32 个 32-bit 寄存器），不可能瞬时完成，硬件是分多拍读的。所以 **store 的源寄存器在你执行 `tcgen05.wait::st` 之前绝对不能改**——这是这一类 bug 最常见的来源。

反过来，`tcgen05.ld` 的 `.sync.aligned` 里的 `.sync` 只表示“warp 内 32 个线程一起执行这条指令”，`.aligned` 只表示“所有线程给的 `[taddr]` 必须一致”；**它们都不提供与异步完成之间的可见性**。所以不要指望 `.sync` 会顺带帮你同步。

---

## 正确的写法（内联汇编）

关键点：**把 `ld` 和 `wait::ld` 塞进同一个 asm 块，把 `st` 和 `wait::st` 塞进同一个 asm 块。**

```cpp
// 以 .32x32b.x32 为例：每线程 32 个 b32 寄存器
uint32_t r[32];
const uint32_t taddr = tmem_base + col_offset;  // 三轮同一个地址

#pragma unroll
for (int round = 0; round < 3; ++round) {
  // ---- 1) ld + wait::ld 同一个 asm 块 ----
  asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
    "{%0,%1,%2,...,%31}, [%32];\n\t"
    "tcgen05.wait::ld.sync.aligned;\n"
    : "=r"(r[0]), "=r"(r[1]), /* ... */ "=r"(r[31])
    : "r"(taddr)
    : "memory");

  // ---- 2) 现在 r[] 才有效，可以随意读写 ----
  for (int i = 0; i < 32; ++i) r[i] = f(r[i]);

  // ---- 3) st + wait::st 同一个 asm 块 ----
  asm volatile(
    "tcgen05.st.sync.aligned.32x32b.x32.b32 "
    "[%32], {%0,%1,%2,...,%31};\n\t"
    "tcgen05.wait::st.sync.aligned;\n"
    :
    : "r"(r[0]), "r"(r[1]), /* ... */ "r"(r[31]), "r"(taddr)
    : "memory");
}
```

为什么必须合在同一个 asm 块里（而不是像一些库那样拆成两条独立的 asm）：编译器的视角是“这个 asm 块把 `r[]` 写好了”，它**有权**紧接着就使用 `r[]`，或者把 `r[]` 的赋值提前到 `st` 的 asm 块之前。`asm volatile` + `"memory"` clobber 能挡住内存重排，但**挡不住寄存器级别的这种提前或延后**。合在一个块里，编译器就完全无法插进两条指令之间，这是最简单也最稳妥的做法。

---

## 什么时候可以减少同步次数（以及绝对不能省的）

<details>
<summary><b>展开：用多组寄存器做双缓冲、批量发起，能不能省 wait？</b></summary>

可以**提高吞吐**，但省不掉 wait 的“次数”级别，因为 `wait::ld` / `wait::st` 是**等待全部此前同类操作**，粒度是“全部”，没有任何字段能区分某一条。

- **单组寄存器（你现在的做法）**：完全串行，`wait::ld` 要等 ld 的数据到齐后才能开始计算，`wait::st` 要等 store 结束才能开始下一轮 → 延迟完全暴露，最慢但最安全。
- **两组寄存器轮换**：第 $i$ 轮的 `st R0` 之后立刻可以发第 $i+1$ 轮的 `ld R1`（再 `wait::ld`），此时 store 的延迟被下一轮的计算掩盖。但第 $i+2$ 轮 `ld R0` 之前仍然**必须**补一个 `wait::st`。
- **批量发起（真正能减少 wait 数量的做法）**：因为 wait 覆盖“此前全部”，可以这样做：
  ```
  ld R0; ld R1; wait::ld;      // 1 个 wait 覆盖 2 次 load
  ...计算 R0, R1...
  st R0; st R1; wait::st;      // 1 个 wait 覆盖 2 次 store
  ```
  这样 2 轮只要 2 个 wait，比每轮 2 个少一半。前提是这几组寄存器互不重叠、几个地址都算好，且寄存器压力扛得住（`x32` 就是 32 个寄存器/线程，两轮同时在线很吃寄存器）。

</details>

<details>
<summary><b>展开：这些 wait 绝对不能省（即使换寄存器、换地址）</b></summary>

- **退出循环后**：最后一个 `st` 的 `wait::st` 必须保留，否则你无法保证 TMEM 里的数据已经更新。
- **交给别的 warp/CTA 之前**：除了等你自己的 `wait`，还要走 `bar.sync` / mbarrier，并加 **`tcgen05.fence::before_thread_sync`**（写侧，同步前）和 **`tcgen05.fence::after_thread_sync`**（读侧，同步后）。这两个 fence 是跨线程顺序用的，**不能**用 `wait::ld/st` 代替，反过来也一样。
- **同一个 warp 自己 RMW**：不需要 mbarrier，也不需要 `tcgen05.commit`（`tcgen05.commit` 是给 `tcgen05.mma` 的完成通知用的，别混用）。你只需要 `wait::ld` + `wait::st`。
- **`tcgen05.dealloc` 之前**：PTX 要求所有访问这块 TMEM 的异步操作都已结束，所以必须 `wait::st` + 相应同步，之后才能释放。

</details>

<details>
<summary><b>展开：另外几个容易踩的坑</b></summary>

- **warp 能访问的 lane 窗口是固定的**：warpgroup 内 warp 0/1/2/3 分别只能碰 TMEM lane `0-31 / 32-63 / 64-95 / 96-127`。你三轮的 `taddr` 必须落在本 warp 的窗口内，跨 lane 只能换 warp（这是 `taddr` 编码在高 16 位放 lane 的原因）。
- **`.sync.aligned` 是 warp 级 collective**：32 个线程必须全部执行同一条指令、给相同的 `[taddr]`。不能在 `lane_id == 0` 分支里调它（不像 `tcgen05.alloc` 那种要单 warp 全体的情形，`ld/st` 是全体线程各拿自己那份数据）。
- **`wait` 不等 `cp` 和 `mma`**：`tcgen05.wait::ld/st` 只覆盖 `ld`/`st`。如果三轮之间你还在等 `tcgen05.mma` 写完 accumulator，那是 mbarrier + `tcgen05.commit` 的事。
- **架构 target**：需要 `sm_100a` / `sm_101a` / `sm_103a` 这类 arch-specific target，普通 `sm_100` 编不过。`cta_group` 修饰符在同一个 kernel 里必须一致（不能 `cta_group::2` 分配、`cta_group::1` 访问）。
- **`.pack::16b` / `.unpack::16b` 不改变同步要求**：它们只改数据在 TMEM 相邻 column 与寄存器之间的排列方式，hazard 一模一样。

</details>

---

## 一句话总结

三轮共用一个寄存器组、地址也可能相同，等于同时踩了“寄存器 RAW + 寄存器 WAR + TMEM RAW”，**每轮的 `wait::ld` 和 `wait::st` 都是硬性要求，共 6 个**。想快就靠“多组寄存器 + 批量发起 + 减少 wait 次数”，不要靠省略 wait；如果想验证，可以把多余的 wait 逐个去掉，看结果是否开始出现间歇性错误；这比读文档更快让你相信它确实必需。

> 参考：NVIDIA PTX ISA 中 `tcgen05.ld` / `tcgen05.st` / `tcgen05.wait` 的异步语义，以及 MLC 的 *Modern GPU Programming for MLSys* 的 “Tensor Memory（TMEM）” 一章（明确写了：ld 之后用目标寄存器前要 `tcgen05.wait::ld`，复用 store 涉及的 TMEM 位置前要 `tcgen05.wait::st`，二者分别等待此前发出的所有 `tcgen05.ld` / `tcgen05.st`。注：该章只强调了 TMEM 位置的复用，没有单独点出**寄存器 WAR** 这条，而这恰恰是“同一组寄存器连做三轮”最容易出错的地方。）

---
## 🧑‍💻 User

我问了同事，同一组线程，同一组寄存器情况下执行顺序被保证正确了
但是我看flashmla的实现是每一组里面都用了wait
这不是矛盾吗

tmem不是必须用四个warp才能读取128lane吗
原来的代码是每一组都用了wait，但是我想把x128改成四组x32，就不能用wait
因为如果硬件要wait的，这样流水线就必须重新规划 那要慢很多了

---
## 😀 默认助手 (7)

## 一、先把“矛盾”拆开：这里有两个不同层面的“顺序”

你同事说的那句话，如果被理解为“**同一 warp 下，ld 写的寄存器、st 读的寄存器，靠程序序就保证正确**”，那么它是错的；如果被理解为“**同一个 warp 对同一 TMEM 地址的 ld/st 在 TMEM 一侧由硬件保序**”，那它可能是对的。这两件事经常被混为一谈。

| 层面 | 同一 warp 程序序能保证吗 | 靠什么 |
|---|---|---|
| 寄存器 ← `tcgen05.ld` 的数据（RAW） | **不能** | `tcgen05.wait::ld` |
| 寄存器 → `tcgen05.st` 已读走（WAR） | **不能** | `tcgen05.wait::st` |
| TMEM 地址侧 st→ld 的可见性 | 可能可以，但不要依赖 | `tcgen05.wait::st` |
| 跨 warp/跨 CTA 的顺序 | **不能** | `bar`/mbarrier + `tcgen05.fence::before/after_thread_sync` |

看最后一行就能看出分工：**NVIDIA 自己的文档把“完成”和“跨线程顺序”分成两套机制**——完成用 `wait::ld`/`wait::st`，跨线程用 `fence::before/after_thread_sync`。如果同一 warp 的程序序已经保证了寄存器正确，那 `tcgen05.wait::ld` 和 `wait::st` 这两条指令就根本没有存在意义了。

### 为什么“同一 warp 程序序保证”在原理上不可能成立

拿你原来的 `.32x32b.x128` 算一下数据量：

$$32\ \text{lanes} \times 128\ \text{cols} \times 4\ \text{B} = 16\ \text{KB per warp}$$

每个线程要收 **128 个寄存器**（512 B/线程）。这个量不可能在一条指令“退役”的那一刻就全部落到寄存器堆里——它是分很多拍流回去的。同理，`st` 也不可能在发射瞬间把 128 个寄存器“快照”走。

也就是说：

- 如果硬件保证“下一条指令就能读到 ld 的结果”“st 之后立刻可以覆盖源寄存器”，那 `tcgen05.ld/st` 的吞吐就等同于同步指令，**“异步”这个设计就白做了**。
- 反过来，既然它是异步的，**寄存器的更新时刻就和指令退休解耦了**——程序序在这里管不到寄存器堆。

这跟 SM80 的 `cp.async`（要 `cp.async.wait_group`）、SM90 的 `wgmma`（要 `wgmma.wait_group`）是同一套逻辑。这三代大家都遇到过“不写 wait 也能跑对”的情况，那不是保证，只是延迟被别的指令盖住了。而且 `.x128` 这种大 shape 更危险：**前几个寄存器和后几个寄存器回来的时间差很大**，所以症状往往是“偶发、不易复现、换一批数据就错”。

---

## 二、那 FlashMLA 为什么“每组都 wait”

大概率不是因为它比硬件规范更保守，而是**结构决定的**：

1. 它拿到一组数据就马上要算（softmax / 缩放 / 转 fp16 / 求和），**消费点就在 ld 之后**，所以 RAW 必须 `wait::ld`。
2. 它用**同一组寄存器**接着 `st` 回 TMEM，所以 WAR 必须 `wait::st`。
3. 它每组只有一组寄存器，所以没有任何空间可以把 wait 往后推。

**结论：FlashMLA 每组 wait 是“该等的地方都等了”，不是“多等了”。** 你要找的不是“怎么去掉 wait”，而是“**怎么让 wait 不落在关键路径上**”——这两件事完全不同。

---

## 三、`x128` 和“四个 warp 读 128 lane”是两件不相干的事

这点你可能被绕进去了，先厘清：

- **lane 方向**：一共 128 条 lane，warpgroup 内 4 个 warp 各管 32 条（warp0→0–31，warp1→32–63，warp2→64–95，warp3→96–127）。**单个 warp 永远只能碰自己那 32 条 lane**，这是硬约束。
- **`.num` 方向**：`.32x32b.xN` 里的 `x` 是沿 **column** 方向重复，不是沿 lane 方向。`x128` = 32 条 lane × 128 个 column。

所以在 `.32x32b` 这个形状下，把 `x128` 拆成四个 `x32`，拆的是 **column 区间**（128 列→4 段各 32 列），**没有跨 lane、没有跨 warp、完全合法**。

$$\underbrace{32\times128}_{x128} \;=\; 4 \times \underbrace{32\times32}_{x32}$$

> 顺便提醒：写出 `x128` 后，你会得到 128 个寄存器/线程，这已经顶到 `.32x32b` 的 `.num` 上限，且会严重压缩 occupancy。拆成四段本身是合理的优化方向——只是理由不是“省寄存器”，见下文。

---

## 四、你的真正担心：“改成 4×x32 就必须多出 wait，流水线要重排，会慢很多”

**这是个假两难。** 关键认知是：

> **`tcgen05.wait::ld` / `::st` 的语义是“等待此前发出的全部同类操作”，它天然是批次级的，不是指令级的。**

所以 **wait 的条数取决于“你换几次寄存器批次”，跟“你发了几条 ld 指令”无关**。四条 `x32` + 一次 `wait::ld`，跟一条 `x128` + 一次 `wait::ld`，成本几乎一样。

### 三档方案对比（4 个 chunk）

**A. 全批处理（4 个 bank，128 regs）——wait 数和你的 x128 完全一样**

```
ld R0,c0; ld R1,c1; ld R2,c2; ld R3,c3;
wait::ld;                 // 1 次，覆盖 4 条
cmp R0..R3;
st R0,c0; st R1,c1; st R2,c2; st R3,c3;
wait::st;                 // 1 次，覆盖 4 条
```
**总共 2 个 wait / 迭代**，和 x128 一模一样。→ **“改成 4×x32 会不会变慢”？在这个形态下不会，一个 wait 都没多。**

**B. 单 bank 串行（32 regs）——你担心的那种，8 个 wait，确实慢**

```
ld R,c0; wait::ld; cmp; st R,c0; wait::st;
ld R,c1; wait::ld; cmp; st R,c1; wait::st;  ...
```
这是**唯一**“慢很多”的形态。它慢的原因是**只有一组寄存器，RMW 天生串行**，不是因为“拆成了四组”。如果你是为了省寄存器才这么做，那就得接受串行——**这笔账跟 x128 vs 4×x32 无关，跟 bank 数量有关**。

记个公式：设 bank 数 $B$、chunk 数 $C$，串行方案的 wait 数 $\approx 2C/B$ 量级；$B=4$ 时退化成方案 A。

**C. 跨迭代三级轮转（3 bank，96 regs）——推荐，两个 wait 都落在非关键路径**

```
// 每个 bank 扮演固定角色：X=刚发起、Y=正在算、Z=正在写回
ld  X, it(k+1)          // 异步，与下面的计算重叠
cmp Y, it(k);  st Y, it(k)
wait::ld                // 只等 ld → 与 st Y 的排空并发，几乎不阻塞
wait::st                // Z 的 st 是上一轮发的，已经排空 → 几乎不阻塞
rotate X←Y←Z
```
这两个 wait 的延迟都被“上一轮/下一轮”的活盖住了，实测吞吐接近无 wait 的理想情况，而寄存器只要 96 个（比 x128 少 32 个）。**这才是“拆成 x32”真正该换来的东西：用更少的寄存器做出跨迭代软件流水。**

---

## 五、怎么把这件事彻底确认清楚（1 小时能搞定）

<details>
<summary><b>展开：三步验证法，可直接拿去和同事对线</b></summary>

**第 1 步：先看 SASS，别猜。**
```
cuobjdump -sass your_kernel.cubin | grep -nE "TMEM|UTCLD|WAIT"
```
如果 SASS 里只有 TMEM 的 ld/st 而没有对应的 wait，那就说明**没有任何东西在保护你**——`ptxas` 不会替你插入 `tcgen05` 的 wait（它不知道你的意图）。先确认到底有没有 wait 在跑，再谈“硬件保不保序”。

**第 2 步：写个最小复现 kernel，把不变量做进去。**
先把 TMEM 填成已知值（`A[i] = i` 之类），然后做三轮 `ld → +1 → st`，最后读回校验。关键是把下列变量逐个扰动——只要有一种配置挂掉，就说明没有保证：

- shape：`.x1` / `.x8` / `.x32` / `.x128` 各跑一遍
- 寄存器压力：刻意加一些无关的活，改变调度距离
- `-O3` vs `-O0`，`--use_fast_math`，改 `maxrregcount`
- 在 ld 和消费点之间插 `__nanosleep(1000)`（变慢一定对）；再把纳秒调小（变快暴露问题）

**第 3 步：注意“偶发正确”这个陷阱。**
如果你看到的是“跑了 200 次都对，第 201 次错一个元素”，那就别再争了——这就是没 wait 的典型症状。反过来，即使 10000 次全对，也**不构成保证**，只构成“这个编译结果 + 这个硬件 + 这个 shape 恰好够快”。换 SM103、换 CUDA 版本、换 occupancy，都可能一夜之间挂掉。

</details>

---

## 六、一句话回答你的三个问题

1. **矛盾吗？** 不矛盾。`FlashMLA` 每组 wait 是**正确且必要**的（它每组都要立刻消费 + 复用寄存器）；你同事那句“程序序保证”顶多成立于 TMEM 地址侧，**不成立于寄存器侧**，而寄存器侧恰好是你这个 RMW 的致命点。
2. **必须四个 warp 才能读 128 lane 吗？** 是，**但跟 `x128` 无关**——`x128` 是沿 column 重复，单 warp 在自己的 32 lane 窗口里就能做完，拆成 4 段 `x32` 也只是拆 column，不用加 warp。
3. **改成 4×x32 是不是就“不能用 wait”、必须变慢？** 不是。**wait 是批次级的，条数由 bank 数决定**。保持 4 个 bank，你付的 wait 数和 `x128` 完全相同（2 个/迭代）；想要更快，用 3 bank 跨迭代轮转，让两个 wait 都落在非关键路径上，同时寄存器从 128 降到 96。

> 需要我把方案 C 展开成能直接编译的完整内联汇编（含 32 个操作数的 `x32` 宏、以及 ld/st 与 wait 必须合在同一个 `asm` 块里的写法）吗？告诉我你的 `dtype`、每 warp 实际要处理的 column 数、以及现在每线程的寄存器预算，我可以把三段式流水的具体代码给你排出来。
