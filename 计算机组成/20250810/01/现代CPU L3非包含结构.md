# 现代CPU L3非包含结构

### 🧑‍💻 User

不明白为什么skylakesp之后要把l3做成victim cache

---

### 🤖 Assistant

这是一个非常好的架构问题。Intel 在 Skylake-SP（Server Platform，即 Xeon Scalable 第一代）架构中，将 L3 Cache（LLC, Last Level Cache）从传统的**包含式（Inclusive）**改为**非包含式（Non-Inclusive / Victim-like）**，这是一个针对多核扩展性和数据中心应用特征的重大权衡。

简单来说，核心原因是为了**给显著增大的 L2 Cache 让路**，以及配合 **Mesh 互联架构**来提高超多核心下的总有效缓存容量。

以下是详细的架构分析：

### 1. 核心驱动力：L2 Cache 的急剧增大

在 Skylake-SP 之前的架构（如 Haswell-EP, Broadwell-EP）中：
*   **L2 Cache 很小**：每核心只有 $256\text{KB}$。
*   **L3 Cache 很大**：通常每核心对应 $2.5\text{MB}$。

因为 L2 很小，把它所有内容都在 L3 备份一份（Inclusive 策略）的代价很低。

到了 Skylake-SP：
*   **L2 Cache 暴增**：每核心增大到了 $1\text{MB}$（扩容了4倍）。
*   **L3 Cache 相对缩小**：每核心对应的 L3 缩减到了 $1.375\text{MB}$。

**如果不改策略会发生什么？**
如果你继续使用 **Inclusive** 策略，这意味着 L3 必须包含 L2 的所有数据。
$$ \text{L3 Waste} = 28 \text{ Cores} \times 1 \text{MB (L2 Copy)} = 28 \text{MB} $$
如果 L3 也是包含式的，那么 $1.375\text{MB}$ 的 L3 切片中，有 $1\text{MB}$ 必须用来存 L2 的备份，只剩下 $0.375\text{MB}$ 能存 L3 独有的数据。这简直是对昂贵的 SRAM 面积的极大浪费。

**改为 Victim Cache 的好处：**
将 L3 改为 Victim Cache（主要接收 L2 驱逐出的数据）后，L2 和 L3 的内容基本上是不重叠的。
*   **有效总容量 = L2 + L3**。
*   这使得每个核心实际上拥有 $1\text{MB} + 1.375\text{MB} \approx 2.375\text{MB}$ 的可用缓存层级，比旧架构更有效率。

### 2. 互联架构的变革：从 Ring 到 Mesh

早期的 Ring Bus（环形总线）非常依赖 Inclusive L3 来处理缓存一致性（Cache Coherency）。

*   **旧架构（Inclusive + Ring）：** 当一个核心想知道某个数据是否在其他核心的 L2 里时，它只需要查 L3。如果数据不在 L3，因为 L3 包含所有 L2 的数据，那么肯定也不在任何 L2 里。这大大减少了对核心的嗅探（Snoop）流量。
*   **新架构（Non-Inclusive + Mesh）：** Skylake-SP 为了支持 28+ 个核心，改用了 Mesh 网格互联。此时 L3 不再包含所有 L2 数据。

**问题来了：** 如果 L3 不包含 L2 数据，当发生 Cache Miss 时，CPU 如何知道数据是否藏在某个核心的私有 L2 里？难道要广播询问所有 28 个核心吗？（这会造成广播风暴，导致互联拥塞）。

**解决方案：目录与监听过滤器（Directory & Snoop Filter）**
Intel 在 L3 Cache 中集成了一个 **Snoop Filter (SF)** 目录结构。
*   即使 L3 不存这个数据本身（Data），它也会保留这个数据的**标签（Tag）和状态**。
*   通过这种方式，Skylake-SP 虽然不存数据副本，但保留了“谁拥有数据”的目录信息。这既节省了存储数据的空间，又避免了全网广播嗅探。

### 3. 应用场景的假设变化

Intel 做这个改动是基于对数据中心 workload 的观察：

1.  **私有数据局部性（Private Locality）：** 现代虚拟化和云应用中，很多数据是核心私有的。增大私有 L2 ($1\text{MB}$) 可以让核心大部分时间只访问自己的 L2，而不去挤占公共的 Mesh 总线带宽。
2.  **减少 L3 争用：** 在旧架构中，只要 L2 修改数据，L3 也得跟着改（因为包含式）。改为 Victim Cache 后，只有 L2 满了被踢出来时，才去写 L3。这大大减少了 Mesh 网络上的写流量。

<details>
<summary><strong>深入细节：Inclusive 与 Non-Inclusive 的权衡对比</strong></summary>

| 特性 | Inclusive L3 (Broadwell及以前) | Non-Inclusive L3 (Skylake-SP及以后) |
| :--- | :--- | :--- |
| **L2 大小** | 小 ($256\text{KB}$) | 大 ($1\text{MB}$ 或更大) |
| **空间利用率** | 低（L2 数据在 L3 重复存储） | **高**（L2 和 L3 数据互斥，总容量叠加） |
| **一致性嗅探** | 快且简单（查 L3 即可知 L2 状态） | 复杂（需要 Snoop Filter 目录追踪 L2） |
| **跨核心延时** | 低（核心间共享数据通过 L3 中转快） | **较高**（如果数据在 Core A 的 L2，Core B 要读，必须先去 L3 查目录，再 snoop Core A，再传给 Core B） |
| **主要优势** | 适合少量核心、频繁共享数据的场景 | **适合多核心、虚拟化、大数据集、私有数据多的场景** |

</details>

### 4. 带来的负面影响

这个改动并非没有代价。Skylake-SP 发布后，游戏玩家和某些特定应用开发者发现性能甚至不如上一代：

*   **核心间通信延迟增加**：如果是“生产者-消费者”模型（Core A 产数据，Core B 用），以前数据在 L3 里等着；现在数据可能还在 Core A 的 L2 里。Core B 去找 L3 找不到，L3 的目录发现 Core A 有，于是去 Core A 拉数据。这一来一回，延迟就高了。
*   **游戏性能**：游戏通常对延迟极其敏感，且线程间同步频繁。这就是为什么那一代的 i9-7980XE 在玩游戏时如果不超频，并没有表现出对得起价格的统治力，也是为什么消费级（Ring Bus 的 Coffee Lake/Comet Lake）在游戏上长期吊打 Mesh 架构处理器的原因之一。

### 总结

Intel 在 Skylake-SP 将 L3 做成 Victim Cache，本质上是一个**为了“大 L2”和“多核 Mesh 扩展性”而做的妥协**。

数学逻辑是：
$$ \text{Total Effective Cache} \approx \sum L2_{\text{private}} + L3_{\text{shared}} $$
通过牺牲一点核心间通信的延迟，换取了更大的每核心私有缓存 ($1\text{MB}$ L2) 和更高的整体缓存利用率，这更符合服务器领域（虚拟化、数据库、容器）吞吐量优先的需求。

---

### 🧑‍💻 User

为什么这种缓存结构下，streaming方式访问时内存与cache数据交换热点在l2 cache？

---

### 🤖 Assistant

这是一个涉及到 **缓存分配策略（Cache Allocation Policy）** 和 **数据流向** 的深层机制问题。

简单直接的回答是：在 Skylake-SP 的 Non-Inclusive 架构下，**从内存读取的新数据默认直接填入 L2 Cache，而不分配进 L3 Cache。**

这就导致了在 Streaming（流式）访问时，数据流动的特征是 **Memory $\leftrightarrow$ L2**，L3 Cache 在这个过程中经常只是“路过”或者作为“垃圾回收站”，而不是数据的主动缓存点。

以下是详细的技术原理解析：

### 1. Fill Policy：直接填充 L2 (Direct-to-L2)

在传统的 **Inclusive（包含式）** L3 架构中：
*   CPU 请求内存数据 -> 数据从内存取出 -> **先写入 L3** -> **再复制一份写入 L2**。
*   这种情况下，L3 是必经之地，也是数据驻留的第一站。

在 Skylake-SP 的 **Non-Inclusive（非包含/Victim）** 架构中：
*   CPU 请求内存数据 -> 数据从内存取出 -> **直接写入 L2**。
*   **关键点**：此时，**L3 Cache 的数据区（Data Array）是不会分配空间来存这份数据的**。L3 只是作为 Mesh 网络的一个路由节点让数据通过，但不会把数据“抄写”进自己的 SRAM 里。

因此，当你进行流式读取（Streaming Read）时，数据源源不断地从内存条流向 L2，L3 在存储层面上是“空”的。

### 2. L3 作为 Victim Cache 的被动性

既然 L3 是 Victim Cache，它只接收 **从 L2 被驱逐（Evicted）出来的** 数据。

在流式访问（Streaming）场景下，数据的特点是**“用完即弃”**（低时间局部性）。
1.  **进入**：数据 $A$ 进入 L2。
2.  **使用**：核心读取数据 $A$。
3.  **驱逐**：核心继续读取数据 $B, C, D...$ L2 容量有限（$1\text{MB}$），必须把 $A$ 踢出去。
4.  **L3 的抉择**：
    *   当 $A$ 从 L2 被踢出时，它理论上应该去 L3。
    *   但是，Intel 的缓存替换算法（Adaptive Replacement Policy）通常能够识别出这是“流式数据”（只读一次）。
    *   如果数据是 **Clean**（未修改）的：L2 直接丢弃该行，**不写入 L3**，因为内存里已经有副本了，而且预测你不会马上再用它。
    *   如果数据是 **Dirty**（已修改）的：必须回写。它会短暂进入 L3，然后最终回写到内存。

**结果：** 在纯粹的读取流（Read Stream）中，数据进了 L2，用完后直接消失（Drop），L3 全程没有分配任何 Cache Line 给这些数据。所以热点全在 L2。

### 3. 防止 L3 污染（Cache Pollution）

这是这种设计的核心目的之一。

*   **旧架构的问题**：当你跑一个遍历 10GB 数组的流式任务时，旧的 Inclusive L3 会把这 10GB 数据都在 L3 里过一遍。这会把 L3 里原本存着的、其他核心需要的“高价值热数据”全部挤出去。这就是著名的 **Cache Pollution（缓存污染）**。
*   **新架构的优势**：由于数据直达 L2，且 L2 驱逐出的流式数据往往被策略判定为不值得存入 L3。因此，**流式数据像水流一样穿过 L2，却不会弄脏 L3**。L3 可以安稳地保留那些真正被多个核心频繁共享的数据。

### 4. L2 Prefetcher（预取器）的主导地位

Skylake-SP 强化了 L2 Prefetcher 的能力。

*   当 CPU 侦测到流式访问模式时，**L2 硬件预取器** 会非常激进地直接向内存控制器请求后续的 Cache Line。
*   这些预取请求回来的数据，依然遵循 "Direct-to-L2" 原则。
*   这意味着 L2 承担了绝大部分“向内存要数据”和“给核心喂数据”的吞吐压力。

### 总结图解

为了更直观地理解，我们可以对比一下数据路径：

**Streaming Read 场景**

*   **旧架构 (Broadwell-EP):**
    $$ \text{Memory} \xrightarrow{\text{Fill}} \mathbf{L3} \xrightarrow{\text{Fill}} \mathbf{L2} \rightarrow \text{Core} $$
    *(L3 和 L2 都在疯狂吞吐数据，L3 被刷屏)*

*   **新架构 (Skylake-SP):**
    $$ \text{Memory} \xrightarrow{\text{Bypass L3 Storage}} \mathbf{L2} \rightarrow \text{Core} $$
    $$ \text{L2 Evict (Clean)} \rightarrow \text{Drop (Don't Write to L3)} $$
    *(只有 L2 在疯狂吞吐，L3 保持安静，保留了有用的共享数据)*

这就是为什么在 Skylake-SP 及其后续架构中，优化程序时需要极其关注 **L2 Cache Miss** 以及 **L2 带宽**，因为 L2 实际上承担了以前 L3 作为“内存前哨站”的很多职责。