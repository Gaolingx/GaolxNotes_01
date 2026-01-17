# 现代CPU L3非包含结构

### 🧑‍💻 User

为什么现代cpu的l3都是非包含结构（victim cache），streaming方式访问时内存与cache数据交换热点在l2 cache？

---

### 🤖 Assistant

这是一个非常深入且切中现代CPU架构设计痛点的问题。现代CPU（特别是Intel Skylake-SP/Ice Lake之后的服务器架构以及AMD Zen架构）倾向于使用**非包含式（Non-Inclusive）**或**受害者缓存（Victim Cache）**结构的L3，并且在流式（Streaming）访问时让L2承担主要的数据交换压力，这背后主要有两个核心驱动力：**有效容量利用率**和**片上互联（Mesh/Fabric）带宽优化**。

以下是详细的深度解析：

### 1. 为什么L3倾向于非包含结构 (Non-Inclusive / Victim Cache)？

在早期的CPU设计中（如Intel Sandy Bridge到Broadwell），L3通常是**包含式（Inclusive）**的。这意味着L3中必须包含L1和L2中所有数据的副本。

#### 传统的包含式 (Inclusive) 的问题：
*   **浪费空间**：随着工艺进步，私有L2缓存越来越大。例如，Intel Cascade Lake的L2从256KB增加到了1MB，Ice Lake/Sapphire Rapids甚至更大（2MB）。如果L3是包含式的，那么核心拥有的1MB L2数据必须在L3中占用同样的1MB空间。如果有28个核心，就会浪费28MB的L3空间仅仅用于存储副本。
*   **扩展性差**：当核心数量增加时，维持包含性的开销（Coherency Traffic）变大。

#### 非包含式/受害者缓存 (Victim Cache) 的优势：
*   **最大化有效容量**：
    在Victim Cache策略下，L3不强制包含L2的数据。
    *   **数据流向**：数据从内存读取时，**直接进入L2**，不进入L3。
    *   **填充L3**：只有当L2决定驱逐（Evict）一条数据，且该数据被认为是“脏”的或有价值的，它才会被写回（Evict）到L3。
    *   **容量公式**：
        $$Effective\_Cache\_Size \approx L3\_Size + \sum (L2\_Size)$$
        相比之下，包含式结构的有效容量仅为 $L3\_Size$。这意味着在现代大L2架构下，Victim L3大大增加了全芯片可缓存的数据总量。

### 2. 为什么Streaming访问时热点在L2 Cache？

理解了Victim Cache的分配策略，就能解释为什么流式访问主要给L2施加压力。

#### A. 数据填充路径 (Fill Path) 的改变
在**包含式L3**架构中，CPU读取内存的路径通常是：
$$Memory \rightarrow L3 \rightarrow L2 \rightarrow L1$$
这意味着即使是只用一次的流式数据，也会先“污染”L3，占用L3带宽和容量。

在**非包含式/Victim L3**架构中，路径变为：
$$Memory \rightarrow L2 \rightarrow L1$$
*   **L3 Bypass（绕过L3）**：当发生L2 Miss且L3 Miss时，数据直接从内存控制器（IMC）通过片上互联网络（Mesh/Infinity Fabric）加载到L2 Cache。**L3在该过程中被旁路了。**
*   **结果**：数据进入CPU的第一站是L2。因此，L2 Cache必须承担所有来自内存的入站带宽。

#### B. 驱逐路径 (Eviction Path) 与 寂静驱逐 (Silent Eviction)
当进行流式读取（Streaming Read）时，通常涉及大量的数据加载，这些数据往往遵循“用完即弃”的特性。

1.  **加载**：数据填满L2。
2.  **驱逐**：因为是流式数据，L2很快满了，需要驱逐旧数据。
3.  **判断**：
    *   如果是**Clean**（未修改）数据：在非包含式结构中，L2可以直接丢弃（Silent Evict），**不需要**写回L3。
    *   如果是**Dirty**（已修改）数据：必须写回L3。
4.  **Streaming优化**：现代CPU的预取器（Prefetcher）和指令集（如AVX-512的Non-temporal load/store）能够识别流式模式。它们会暗示Cache系统这些数据不需要在Cache中久留。
    *   在最佳情况下（纯读取流），L2不断从内存吸入数据，处理完后直接丢弃（不写回L3）。
    *   **这意味着L3的带宽占用极低，而L2与Memory Controller之间的带宽被打满。**

### 3. 这种设计的架构权衡

为什么设计者愿意让L2成为热点？

#### 1. 保护L3中的共享数据 (防止 Cache Pollution)
L3通常存放跨核心共享的数据（如锁、全局变量、指令代码段）。如果流式数据（通常是私有数据）经过L3，会将这些宝贵的共享数据挤出L3。
通过让流式数据只在L2这一层流转（L2 Fill -> L2 Use -> L2 Drop），**L3实际上变成了一个更加纯粹的“共享数据/受害者缓冲区”**，极大提高了由于上下文切换或跨核同步带来的缓存命中率。

#### 2. Mesh/Fabric 互联架构的适应
现代多核CPU（如Intel Xeon Scalable）使用网格（Mesh）互联。
*   每个Tile（节点）包含Core、L2和一小块L3切片（Slice）。
*   在非包含结构下，Core读取内存时，不需要先去查询和填充本地的L3切片（因为L3是物理分布式、逻辑统一的哈希映射）。
*   直接将数据拉入私有的L2是最短路径，延迟最低。

### 总结

现代CPU采用这种设计逻辑的本质是**分层解耦**：

1.  **L3结构是非包含/Victim**：为了在L2容量暴涨的时代，不浪费宝贵的晶体管做重复存储，实现 $$Total\_Cache = L2 + L3$$ 的最大化。
2.  **Streaming热点在L2**：是因为在Victim策略下，**新数据直接分配进L2（Bypass L3）**。
    *   这使得L3免受流式数据的“冲刷”和污染。
    *   L2虽然压力大，但它作为私有缓存，带宽设计得比L3更高，且紧邻核心，非常适合处理这种高吞吐、低重用率的数据流。