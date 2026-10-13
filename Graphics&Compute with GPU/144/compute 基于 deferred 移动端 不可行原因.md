# compute 基于 deferred 移动端 不可行原因

## 🧑‍💻 User

compute based deferred + visibility buffer + cluster light culling同时面向 mobile 为什么impossible

---
## 🧑‍💻 User

compute based deferred 同时面向 mobile 为什么impossible

---
## 😀 默认助手 (7)

I'll verify a few hard spec/architecture facts before building the argument.

Search is captcha-blocked. Let me pull the authoritative spec pages directly.

Found Arm's content API. Let me pull the actual chapter text.

# 为什么「compute-based deferred + visibility buffer + cluster light culling」在 mobile 上不可能

## TL;DR

三件套各自的**收益**都建立在同一个 desktop 前提上：**IMR + 大 L2 + 高带宽 + 富余 ALU + 随机访存便宜**。
而 mobile 的架构契约正好相反：**TBDR + 硬件 HSR + 几百 KB 片上 tile memory + ~10 GB/s 带宽 + 2~4 W 功耗**。

三者组合后，每一个中间结果都被迫**离开片上 tile memory 落到 DRAM 再读回来** —— 而 mobile GPU 唯一的结构优势就是"让中间结果永远不落 DRAM"。所以这不是"慢一点"的调优问题，而是**架构契约层面的冲突**：

> 严格说：API 层面能表达；**在手机上以 60 fps、热稳态、可接受功耗跑出正收益**——不可能。

---

## §0 先明确 mobile GPU 的架构契约

| 契约 | 含义 | 为什么它是判决性的 |
|---|---|---|
| **TBDR** | 几何先 binning/tiling（Mali 固定 $16\times16$），整个 tile 在片上 tile memory 里完成光栅+着色，**只把最终像素写回 DRAM** | 省的全部是**带宽和能量** |
| **硬件 HSR** | fragment shading 被推迟到 tile 内几何确定之后，被遮挡像素**根本不跑 FS** | mobile 已经有硬件版 deferred |
| **compute 不属于 tile 流水** | compute 是独立 job，没有"当前 tile"概念 | 见下 |
| **片上中间结果只有 fragment 能读** | Vulkan 里唯一能读"上一阶段同像素结果"的机制是 input attachment / PLS / framebuffer fetch | 规范原文：`maxPerStageDescriptorInputAttachments` —— *"These are only supported for the fragment stage."* |

最后一条是**整个论证的支点**：**API 层面根本不存在让 compute 读 tile 内中间结果的手段。**
于是任何 compute pass 的两侧必然各发生一次 **全屏 tile store / load**（写回 DRAM + 重新分块）。这是 API 级保证（barrier + layout transition + render pass 结束），不是驱动实现差异。

$$\text{每次 compute dispatch 的代价} \geq \text{1 次全屏附件写回} + \text{1 次全屏附件读回}$$

---

## §1 compute-based deferred：把 mobile 唯一能省的东西全部花掉

**mobile 上真正的 deferred 长这样**（也是它唯一划算的写法）：

```
单 render pass
├─ subpass 0: 写 G-buffer（transient / lazily-allocated attachment）
└─ subpass 1: fragment 光照，用 input attachment 读 G-buffer
G-buffer 从不离开 tile memory → DRAM 流量 ≈ 0
```

**改成 compute 之后**：compute 无法读 tile → G-buffer 必须整体落 DRAM → 再整屏读回。

$$1920\times1080\times4\text{B} = 8.29\ \text{MB / 全屏 32bpp 附件}$$

- G-buffer 4 个 target：$4\times8.29 = 33\ \text{MB}$，写+读 $= 66\ \text{MB/frame}$ 纯额外
- 60 fps → $\approx 4\ \text{GB/s}$ —— 而这只是"额外"部分，已经吃掉手机可用 GPU 带宽的一大块

**收益还消失了**：deferred 的卖点是"避免对被遮挡像素重复着色"，但这件事**硬件 HSR 免费给你**。你花掉带宽，买回一台机器本来就免费提供的功能。

<details>
<summary>另一个隐蔽的硬限制：G-buffer 可能"创建不出来"</summary>

Vulkan core 下限（我已核对 spec 的 limits 表）：

| 限制 | core 下限 | Vulkan Roadmap 2022 |
|---|---|---|
| `maxColorAttachments` | **4** | 7 |
| `maxFragmentOutputAttachments` | **4** | — |
| `maxFragmentCombinedOutputResources` | **4** | 16 |
| `maxPerStageDescriptorSamplers` | 16 | 64 |

一份"正常"的 G-buffer（albedo / normal / roughness-metal / depth / velocity …）在很多设备上**连 MRT 数量都超**，只能 packing（octahedral normal、RGB10A2…）→ 额外 ALU、精度损失、resolve 更贵。
</details>

---

## §2 visibility buffer：这笔交易的"买方"在 mobile 不存在

vis buffer 的**本质是一笔交易**：

$$\underbrace{8.29\ \text{MB 的 R32\_UINT ID buffer}}_{\text{便宜}} \;\longleftrightarrow\; \underbrace{\text{resolve 时按三角形 ID 全局随机 gather 顶点数据}}_{\text{昂贵}}$$

这笔交易在 desktop 成立，是因为买方条件齐备：L2 有几 MB~几十 MB（顶点命中率高）、带宽 500 GB/s+、大 warp + 大 reorder buffer、**以及最关键的 wave-coherent 去重**。

### 决定性的一环：去重优化在 Vulkan 里**不是保证**

desktop 上 resolve 的核心技巧是：用 subgroup `ballot` / `match` 在 $8\times8$ block 内去重三角形 ID，把一个 warp 的 32 次随机 gather 压成"每个唯一三角形一次加载"。

我核对了 `VkPhysicalDeviceSubgroupProperties` 的规范文本：

> *"supportedOperations will have the `VK_SUBGROUP_FEATURE_BASIC_BIT` bit set if any of the physical device's queues support …"*
> *"If `supportedOperations` includes `VK_SUBGROUP_FEATURE_QUAD_BIT`, or the shaderSubgroupUniformControlFlow feature is enabled, subgroupSize must be greater than or equal to 4."*

也就是说：**强制只有 `BASIC`**。`VOTE` / `BALLOT` / `SHUFFLE` / `QUAD` 全是**可选 feature**。
→ 你必须 feature 探测 + 回退到**朴素逐像素 gather**：每个 lane 3 次随机顶点读 → 32 lane 落在 32 条不同 cache line 上 → **访存放大 8~32 倍**。这正好把"省下的 G-buffer 带宽"连本带利吐回去，还要额外付 ALU。

### 其余硬条件

| 需求 | 规范现状 |
|---|---|
| 单 pass depth+ID 用 $64$ bit `atomicMax` | `shaderInt64Atomics` 是**可选 feature**（core 下限 `VK_FALSE`）→ 大量移动 GPU 只能退化成 **depth prepass + ID pass 两遍**（多一轮 tile flush） |
| mega draw + 随机取 vertex/index buffer | 需要 bindless / descriptor indexing（`VK_EXT_descriptor_indexing` 非必需）；或全场顶点塞一个 SSBO，但 `maxStorageBufferRange` core 下限仅 $2^{27}=128\text{ MiB}$，**每 stage SSBO 下限只有 4 个**（Roadmap 2022 才 30） |
| 统一全场景 vertex layout | `maxVertexInputAttributes` / `maxVertexInputBindings` core 下限都只有 **16** |
| 用 input attachment 把 ID 留在 tile 内 | 可以，但 resolve 的访存是"同像素 ID → **全局随机**顶点"—— input attachment 只能给你同一像素，**贵的访存一点没省**；而放进 compute 就又是全屏 flush |

**结论**：vis buffer 在 mobile 上不是"能跑但慢"，而是**它交易的买方（L2 命中率）在 mobile 根本不存在**，所以**必然亏**。
（注意区分：*只做 depth + 4B ID* 的纯 vis buffer 在 mobile 是可行的——省掉大 G-buffer 确实赚；但它跟 deferred + cluster 拼一起就崩了。）

---

## §3 cluster light culling：算的规模比 mobile 需要的大一个数量级

clustered shading 的存在理由是 **"几千个光源 + 深度分布极不均匀"**。而 mobile 的现实是"屏幕上几十个有效光源"，且带宽喂不起每像素几百次光源循环。

$1920\times1080$、$64\times64$ tile、24 个 z-slice：

$$\underbrace{30\times17}_{=510\text{ tiles}}\times 24 \approx 12\,240\ \text{clusters}$$

- 每个 cluster 的 light index list 需要**全局 `atomicAdd` 追加** → mobile 上 DRAM/L2 全局原子延迟高、竞争重；退回"固定上限 per-cluster"则浪费 + 抖动
- light list 必须落 SSBO/DRAM → shading 阶段逐像素**非相干读** → 又一次访存放大
- 而 cluster cull 想利用深度分布，**就必须等深度/ID 写完** → 强串行依赖，无法与光栅重叠 —— 对"靠 tile 并行掩盖延迟"的架构是最坏 case
- 额外：mobile 上每次 dispatch 都是 job 切换 + pipeline flush，一帧塞 3~4 个 dispatch 的固定开销远高于 desktop

**mobile 的等效正解是 Forward+ / 2D tile light list**：在**同一个 render pass** 内用 fragment shader 读 tile 级 light list（几 KB、常驻 L2、命中率接近 1）——**用 10% 的代价拿走 90% 的收益**。

---

## §4 组合起来：冲突是**乘性**的，不是加性的

### 4.1 概念上的互斥

deferred 与 vis buffer 是**同一问题的两条替代路线**（"先记录、后着色"，粒度不同）。
硬拼起来只能变成：

```
光栅 ID  →  resolve 生成 G-buffer  →  deferred 光照
(vis 的代价)   (vis 的代价 + GB 的代价)   (GB 的代价)
```

你付了**两者的并集**，收益却取不到交集 —— 因为 vis buffer 省 G-buffer 带宽的收益，已经被"又生成了一份 G-buffer"抹掉了。

### 4.2 barrier 链

| # | Pass | 形式 | 后果 |
|---|---|---|---|
| 1 | 光栅 vis ID | render pass | 写 8.29 MB |
| 2 | vis resolve | **compute** | 全屏 store + 读 ID 8.29 + 顶点 gather 放大 → 写 G-buffer 33 MB |
| 3 | cluster cull | **compute** | 全屏 store + 读 depth + 写 light list |
| 4 | deferred 光照 | **compute / 另一 pass** | 全屏 store + 读 G-buffer 33 + 读 light list → 写色 8.29 |

**pass 数：1（TBDR 理想）→ 2 个 render pass + 2~3 个 compute dispatch，4~5 条 pipeline barrier。**
每次 barrier = 一次**全屏 tile store**（把 live attachments 全部写回 DRAM）。

### 4.3 带宽账（$1920\times1080$，60 fps，数量级估算）

$$\begin{aligned}
\text{ID 写}+\text{ID 读} &= 2\times8.29 \approx 16.6\\
\text{G-buffer 写}+\text{读} &= 2\times33 \approx 66\\
\text{顶点 gather 放大} &\approx 50\text{–}150\\
\text{cluster light list} &\approx 5\\
\text{最终色写} &\approx 8.3\\
\hline
\text{合计} &\approx \mathbf{180\text{–}280\ MB/frame}
\end{aligned}$$

$$60\ \text{fps} \Rightarrow \approx 11\text{–}17\ \text{GB/s}$$

对照 mobile 的 Forward+：

$$16.6\ (\text{depth prepass}) + 8.3\ (\text{最终色}) \approx 25\ \text{MB/frame} \Rightarrow 1.5\ \text{GB/s}$$

**差距约 10 倍** —— 11~17 GB/s 已经等于、甚至超过手机 GPU 在热稳态下能拿到的**全部**带宽。

### 4.4 功耗账

- off-chip DRAM 每字节能耗约为片上 SRAM 的 **10~100 倍**、ALU 的约 2 个数量级
- 手机 GPU 功耗上限约 **3~5 W**（整个 SoC 5~8 W，还要与 CPU / NPU / ISP / 显示分）
- **带宽 = 功耗 = 热降频**。所以即使首帧 60 fps，几秒后会掉到 30 甚至更差 —— 这不是"优化能救"的量级问题

---

## §5 "impossible" 的四种精确含义

| 维度 | 结论 |
|---|---|
| **结构不可能** | compute 永远无法访问 tile memory（规范级：input attachment 仅限 fragment stage），而这三项技术每项都要求"把 tile 内中间结果整体搬出 DRAM 再读回"→ 先天放弃 mobile 唯一的架构优势 |
| **能量/带宽不可能** | 需求 $\geq$ mobile 带宽上限，且是**持续性**需求 → 热稳态不可达 |
| **收益不可能** | deferred 的收益被硬件 HSR 免费覆盖；vis buffer 的收益被"没有大 L2"吃掉；clustered 的收益（几千光源）在 mobile 场景不存在 → **成本全额支付，收益大部分蒸发** |
| **可移植性不可能** | subgroup ballot、64-bit atomics、descriptor indexing、≥5 个 MRT 全是**可选特性** → 组合回退路径数量爆炸，且每条回退在 mobile 上更慢 |

补一句现实校核：Android CDD 对 Vulkan 的要求是 **必须完整实现 Vulkan 1.1**、**强烈建议 1.3**（我核对了 CDD 文本），API 层面能表达这套流程；但"Roadmap 2022 级"的限制（30 SSBO / 16 storage image / 7 MRT）在 Android 上并非统一保证。**API 能写 ≠ 手机上划得来。**

---

## §6 那 mobile 上应该怎么做

<details>
<summary>展开：mobile 的正确技术栈</summary>

| 目标 | mobile 正解 |
|---|---|
| 避免 overdraw 着色 | **硬件 HSR** + depth-only prepass + front-to-back 几何排序 |
| 多光源 | **Forward+**：单 render pass，2D tile（必要时 3D tile）light list，几 KB 驻 L2 |
| 需要 G-buffer | 单 render pass + **transient/lazily-allocated attachment** + **input attachment** 做光照（这才是 mobile deferred） |
| 材质/着色爆炸 | **material 分层 + stencil 分类的 multi-pass forward**，而不是 deferred |
| 需要 vis buffer？ | 只在**极高 overdraw + 极低材质复杂度**的特定 workload 下才可能赢；且只做 depth + 4B ID，不要接 deferred 与 cluster |

**什么条件下三件套才会同时划算？**
当 mobile GPU 变成 **IMR + 几 MB~几十 MB L2 + 全 bindless + 500 GB/s + 100 W 级散热预算** —— 也就是说，**当它不再是 mobile 的时候**。这三项技术（Deferred 1988/Bayoli、Clustered 2012 Olsson、Visibility Buffer 2013 Burns & Hunt）的每一个前提假设，都是桌面级内存系统。

</details>

---

## 一句话总结

> 这三项技术的共同前提是**"把中间结果摊在 DRAM 上，然后靠大 L2 和大带宽随取随用"**；
> 而 mobile GPU 的全部设计目的就是**"永远不要把中间结果摊到 DRAM 上"**。
> compute 无法访问 tile memory 是这条契约的**规范级**表现。
> 所以三件套叠在 mobile 上：**结构上互相抵消（deferred ⊥ vis buffer），带宽/功耗上超限约 10 倍，收益上被 HSR 与小 L2 吃光** —— 这就是 "impossible"。
