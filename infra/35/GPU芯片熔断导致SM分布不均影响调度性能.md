# GPU芯片熔断导致SM分布不均影响调度性能

## 🧑‍💻 User

想问一个问题：是不是基本每个芯片都有冗余，因为良率问题，为了保证SM数量和宣称的一样，芯片上SM其实会高出一部分，然后关掉一部分就会关了，导致cluster上sm好像是不固定的，最后导致每个芯片调度上会有细微差别，我们profiler时候发现过

---
## 😀 默认助手 (7)

你的直觉方向是对的，但因果关系需要反过来理解，而且中间还要区分几个不同层次的机制。下面逐层说明。

## 先纠正一个容易误解的因果关系

现实是：**SKU（产品料号）先定义了目标规格，测试之后才按 fuse 表熔断**。也就是说，公开宣称的 SM 数是被保证的规格，并非“多做 SM 然后随机关掉以凑数”。芯片厂做的是 **harvesting / binning**（分档利用）：同一颗 die 上做足资源，测试后用 efuse 或激光切断把不合格部分隔离，再按市场档位分配料号。因此：

- 你拿到的一片 H100 的 132 个 SM，**全部是逐条测试验证过的可用单元**；
- 关闭的 12 个里，只有一小部分是因为真实缺陷，更多是**为了对齐产品档位而主动关闭**（同一颗 die 生产出来，会按市场需要切出 SXM、PCIe、H800 等不同料号）。

所以“冗余”这个词更准确的理解是：**die 上设计了比某个料号更多的资源，缺陷可以降档，而不是为了凑数故意多做。**

## 关键点：熔断粒度决定了 SM 与 cluster 的关系并不固定

这才是你观察到的现象的根本原因。熔断的最小粒度通常是 **TPC（2 个 SM）**，也常整片 **GPC** 关掉，**很少**能精确到单个 SM。我们做一个可验证的算术检查：

| die | 满配 SM | 物理分组 | 料号 | 启用 SM | 差额 |
|---|---|---|---|---|---|
| GA100 | 128 | 8 GPC × 16 | A100 40/80GB | 108 | $-20$ |
| GV100 | 84 | 6 GPC × 14 | V100 | 80 | $-4$ |
| GH100 | 144 | 8 GPC × 18 | H100 SXM / H800 | 132 | $-12$ |
| GH100 | 144 | 8 GPC × 18 | H100 PCIe | 114 | $-30$ |
| AD102 | 144 | 12 GPC × 12 | RTX 4090 | 128 | $-16$ |

以 A100 为例：$108 = 128 - 20$。如果只整 GPC 关闭，只能得到 $112$（关 1 个）或 $96$（关 2 个），**无法得到 108**。因此必然存在某个 GPC 只启用了部分 SM；也就是说，**各 GPC 的 SM 数确实不等**。H100 同理：$132 = 144-12$，整 GPC 关闭只能得到 $126$ 或 $108$。RTX 4090 同理：$128 = 144-16$，整 GPC 关闭只能得到 $132$ 或 $120$。

结论：**总 SM 数固定不变，但“每个 cluster/GPC 里有多少 SM”不固定。** 这一点你的观察完全成立。

## 这会带来哪些真实差异

- **Thread Block Cluster 的行为**：CUDA 的 cluster 保证落在**同一个 GPC** 内。若某个 GPC 只剩 12 或 14 个 SM，那么大小为 16 的 cluster 在该 GPC 上**根本无法启动**；也就是说，这是功能差异，不只是性能差异。因此 NVIDIA 要求运行时通过 `cudaOccupancyMaxPotentialClusterSize()` 和 `cudaOccupancyMaxActiveClusters()` 查询，而不是照文档写死数字。同理，同一个 kernel、同一份配置，在不同芯片上**同时驻留的 cluster 数**可能不同，这会直接影响流水线深度和你看到的 `launch` 侧指标。
- **L2 / 内存分区的亲和性**：GPC 与 L2 slice 的对应关系，以及 HBM stack 的交叉开关，都会随之改变；地址哈希的落点因此发生细微变化，于是 bank 冲突模式、访存延迟分布和有效带宽都会出现几个百分点的漂移（memory partition camping 敏感型 kernel 更明显）。
- **功耗与频率图**：有效 SM 的物理位置变了，局部热点分布也就变了。同一个散热条件下，boost 曲线会略有差别。这是同 SKU 芯片之间 $1\%\sim3\%$ 波动的常见来源。
- **调度映射**：`%smid` 的编号和实际物理位置之间的间隔变了；大 grid 的 wavefront 切分和 tail effect 也就随之变化。
- **MIG**：`nvidia-smi mig -lgip` 列出的 profile 直接以 GPC 为划分单位，这本身就是“厂商内部清楚知道 GPC 组成不齐”的直接证据。

## 但别过度解读：差异是离散的，不是连续分布

同 SKU 芯片通常只有**有限几种 fuse pattern**，并不是每片芯片都独一无二。因为厂商会按固定的 mask 组合切档，并做 consistency validation。所以你测到的应该是**少数几种拓扑簇**，而不是连续分布。如果看到的是连续漂移，那更可能是下面这些因素：

<details>
<summary><b>排查清单：先排除非拓扑因素</b></summary>

- **时钟与功耗状态**：`nvidia-smi -q -d CLOCK,PERFORMANCE`，确认不是 throttle / 不同 power cap / 不同 VBIOS。
- **HBM 供应商与批次**：同一料号的 HBM 颗粒可能有不同 vendor，带宽特性微差。
- **ECC / MIG / 独占模式**：ECC 开关会改变可用容量与带宽；MIG 一旦开启，你就不再是全芯片的拓扑。
- **PCIe / NVLink 拓扑与 NUMA**：多卡时 `nvidia-smi topo -m`，宿主 NUMA 绑定差异远大于 fuse 差异。
- **Profiler 自身扰动**：NCU 会串行化和 replay，本身改变调度；用 NCU 对比拓扑差异时要用 `--replay-mode application` 或改用 Nsight Systems 的长时间统计。
- **驱动版本 / 固件**：调度器策略层面的差异常常被误归因到硬件。

</details>

<details>
<summary><b>怎么把拓扑差异测出来（可复现的做法）</b></summary>

**1. 直接读 cluster 容量上限**

```cpp
int maxCluster = 0;
cudaOccupancyMaxPotentialClusterSize(&maxCluster, kernel, &config);
int activeClusters = 0;
cudaOccupancyMaxActiveClusters(&activeClusters, kernel, &config);
// 同一 SKU 多片机器上比这两个数，差异即拓扑差异
```

**2. 用 `%smid` 反推实际的 SM 集合（最直接）**

```cpp
__global__ void probe(unsigned* seen, unsigned* cnt) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    atomicExch(&seen[smid], 1);
    atomicAdd(&cnt[smid], 1);
}
```

用一个刚好填满设备的 grid 跑一遍，然后收集 `seen`。**被置位的 smid 数量 = 实际可用 SM 数**，而 smid 的**空洞分布**会暴露 fuse pattern（哪些 TPC/GPC 被切过）。把多片机器的这份 bitmap 做对比，就能确认差异来自拓扑还是别的因素。

**3. Nsight Compute 侧**

看 `launch__occupancy_limit_*` 与 cluster 相关指标、以及 per-SM 的活跃分布；在 H100 上还可以从 GPC 时钟域读数交叉验证。

</details>

## 补充：被冗余/熔断的不止 SM

这一点值得一起记住，因为它解释了很多“同 SKU 不同表现”的现象：

- **HBM 容量与堆栈**：H100 80GB = 5 个 HBM3 stack（满配 6 个）。容量差异本身就是 harvesting 的直接可见结果；而 H200 的 141GB 属于 ECC/预留口径，不是熔断，不要混淆。
- **L2 slice、copy engine、NVLink 端口、PCIe lane**：都在熔断范围内。NVLink 少一路、PCIe 从 x16 降到 x8，对通信型 workload 的影响远大于 SM 拓扑。
- **CPU 同理**：8 核 die 切 6 核卖给低端料号，机制完全一致。
- **反面情形**：如果某个料号用了 100% 的 die（旗舰满配），就没有冗余空间，良率损失只能靠降低出货量来消化。

## 一句话回答

是的，**“die 上有冗余、按 fuse 表熔断”是行业通例，而且熔断粒度通常粗于单个 SM**，所以“总 SM 数固定但各 GPC/cluster 的 SM 数不齐”是真实且普遍的。你在 profiler 里看到的细微差别，**最可能就是这个机制造成的**。但要注意：宣称的 SM 数是被保证的规格，而不是凑出来的结果；同 SKU 的差异是**几种离散的拓扑 pattern**，不是每片芯片都不同。要确认的话，先从 `cudaOccupancyMaxActiveClusters` 和 `%smid` bitmap 这两项测试入手，同时排除时钟、ECC、MIG 和 profiler 自身扰动。
