# 无成熟3FS用于vLLM分布式KVCache经验

### 🧑‍💻 User

使用web search tool，总结搜索结果后回答。目前是否有基于3FS集群为vLLM推理引擎提供高速前缀分布式KVCache缓存的经验吗？我本来想在Mooncake Store的基础上添加DFS/3FS集成。但Mooncake Store对于3FS和分布式文件系统的支持不太好，我个人发现有很多坑。我感觉阿里的TairedKV做的比较好，但是TairedKV的开源仓库相对缺少维护。

---

### 🤖 Assistant

## 结论先说

**有基于 3FS + 分布式前缀 KVCache 的公开实践，但成熟度分层明显：**

| 层级 | 现状 | 说明 |
|------|------|------|
| **Mooncake Store ↔ vLLM** | **较成熟** | 官方集成，有 vLLM 文档/博客，Agentic 场景有显著收益 |
| **Mooncake Store ↔ 3FS** | **有，但偏实验/演进中** | 官方 HF3FS 插件标了 Experimental；另有更完整的 DFS replica 改造在推进 |
| **3FS 直接做 vLLM 前缀缓存后端** | **几乎没有独立成熟方案** | 公开材料多是「Mooncake/Tair 管缓存语义 + 3FS 做高速存储层」 |
| **Tair KVCache（你说的 TairedKV）** | **架构更完整，但开源偏 Manager/仿真** | 明确兼容 HF3FS/Mooncake/NFS 等后端，商业产品更完整 |

你的直觉大体正确：**Mooncake 的 3FS/DFS 路径坑不少；Tair 的全局管理做得更像企业级，但开源维护重心不在完整端到端运行时。**

---

## 1. 目前是否有「3FS 集群 + vLLM 高速前缀分布式 KVCache」经验？

### 有，但通常不是「vLLM 直连 3FS」

公开可查的主流路径是：

```
vLLM (APC / Connector)
        ↓
Mooncake Store / Tair KVCache Manager
        ↓
3FS / HF3FS / NFS / 本地 SSD / DRAM
```

也就是：

- **vLLM 负责推理与 block 级前缀复用**
- **分布式 KV 系统负责跨实例共享、offload、元数据管理**
- **3FS 负责高带宽持久/近线层**

### Mooncake + vLLM：这条链路是真的有生产向经验

Mooncake 官方与 vLLM 已形成正式集成，场景包括：

1. **PD 分离**：`MooncakeConnector` 做跨节点 KV 传输（RDMA）
2. **分布式 KV 池 / 前缀共享**：`MooncakeStore` / `MooncakeStoreConnector`

官方宣称在 agentic trace 上大致有：

- 吞吐约 **3.8x**
- P50 TTFT 约 **46x** 降低
- E2E 延迟约 **8.6x** 降低
- 可扩到约 **60 GPU**，高 cache hit

文档：

- [Mooncake x vLLM Integration](https://kvcache-ai.github.io/Mooncake/getting_started/examples/vllm-integration/index.html)
- [vLLM MooncakeStoreConnector 使用指南](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
- [vLLM Blog: Serving Agentic Workloads with vLLM x Mooncake](https://vllm.ai/blog/2026-05-06-mooncake-store)

**所以：如果你问的是「有没有人用分布式前缀 KVCache 服务 vLLM」，答案是有，而且 Mooncake 是当前最公开、最完整的一条。**

### Mooncake + 3FS：有官方插件，但明确不成熟

Mooncake 有 **HF3FS Plugin（Experimental）**：

- 用 3FS 原生 **USRBIO** 做存储后端
- 编译要 `-DUSE_3FS=ON`
- 需要 `libhf3fs_api_shared.so`、`hf3fs_usrbio.h`
- master 指定 3FS 挂载点；若不是 3FS mount，会回退 POSIX

官方警告大意是：

> **Experimental / incomplete，不建议当生产方案，仅评估测试。**

文档：
[Mooncake HF3FS Plugin](https://kvcache-ai.github.io/Mooncake/getting_started/plugin-usage/3FS-USRBIO-Plugin.html)

另外，社区/仓库里还能看到：

- 已支持把 KVCache **offload 到 3FS/SSD**
- 但长期存在 **文件清理 / eviction** 不完整问题（例如 issue #952）
- 还有较新的 **DFS replica + POSIX/HF3FS adapter** 改造（如 PR #2683），说明团队自己也在补「真·分布式文件系统路径」

**这和你踩坑的感受高度一致：不是没人做，而是 3FS/DFS 集成仍在补齐。**

### 直接「3FS 集群 = vLLM 前缀缓存」的独立成熟经验：很少

搜索结果里几乎看不到：

- 某个团队把 3FS 直接当 vLLM `KVCache` backend
- 并有稳定的 block hash 索引、跨实例共享、淘汰、一致性、故障恢复完整方案

更常见的是：

- **3FS = 高速共享盘 / object-like 存储层**
- **缓存语义层仍由 Mooncake / Tair / LMCache 提供**

所以更准确的表述是：

> **有「3FS + 分布式 KVCache + vLLM」经验；几乎没有「裸 3FS 直接服务 vLLM 前缀缓存」的成熟公开经验。**

---

## 2. 你想在 Mooncake Store 上加 DFS/3FS，为什么会感觉坑多？

从公开信息看，坑主要来自这几层，不是你一个人的问题：

### A. 接口语义不匹配
- 3FS/USRBIO 偏 **高性能文件/IO 原语**
- Mooncake Store 偏 **对象级 KVCache**（PutStart / 完成通知 / lease / pin / eviction）
- 中间需要 descriptor、shard、alignment、fd-based `ReadAt/WriteAt` 等适配层

### B. 生命周期管理不完整
公开 issue 已提到：
- 能 offload 到 3FS
- 但 **file cleanup / eviction** 曾缺失或不足
这会直接导致容量膨胀、空间回收困难、命中率失真。

### C. 生产路径仍在演进
官方插件还是 experimental；同时仓库里又在做更完整的 DFS replica：
- `ReplicaType::DFS`
- shard allocator
- POSIX adapter 方便本地验证
- HF3FS adapter 做真 3FS 路径
- 目前还看到 **single-tenant** 等限制

### D. 部署复杂度高
你要同时搞定：
- 3FS 集群稳定性
- Mooncake master/client
- offload backend 配置
- vLLM connector
- 对齐/权限/挂载点/回退逻辑

任意一层抖动，都会表现为「缓存系统不稳」，而不是单纯 3FS 问题。

**结论：Mooncake + 3FS 是可行方向，但目前更像「能跑、可评估、需二次工程化」的方案，不是开箱即用。**

---

## 3. Tair KVCache（TairedKV）到底怎样？

你说的更像是 **阿里云 Tair KVCache**，开源仓库是：

- GitHub：[`alibaba/tair-kvcache`](https://github.com/alibaba/tair-kvcache)
- 产品页：[阿里云 Tair KVCache](https://www.aliyun.com/product/kvcache)

### 它的优势（你感觉「做得比较好」是有依据的）

开源 README 显示它的定位更偏 **全局 KVCache 管理系统**：

- 中心化 **Manager** 管元数据
- 支持 prefix / sliding window / KV 多种匹配
- 两阶段写（拿写地址 + 写完通知）
- 多存储后端抽象
- 明确兼容：
  - **HF3FS**
  - **Mooncake**
  - **NFS**
  - 其他存储
- 有容量/水位/回收、异步删除
- 有 Optimizer / HiSim 做命中率与容量仿真
- Client/Connector 面向多引擎：vLLM、SGLang、RTP-LLM、TRT-LLM 等

从架构上看，它比「只做 Store」更完整：

```
推理引擎
   ↓
Tair KVCache Manager Client / Connector
   ↓
Tair KVCache Manager（全局元数据 + 容量策略）
   ↓
HF3FS / Mooncake / NFS / ...
```

这正是你要的「3FS 做高速层 + 上层做分布式前缀缓存」更接近企业形态的方案。

### 它的问题（你说开源维护弱，也基本成立）

当前开源重点主要是：

- **Tair KVCache Manager**
- **HiSim 仿真系统**

并不是一个完整、开箱即用、社区高度活跃的「开源替代品」。
很多真正难的部分（完整运行时、商业存储联动、稳定性工程）更可能在云产品侧。

因此现实是：

- **架构设计：强**
- **开源可落地完整度：中等偏弱**
- **社区/持续维护：相对 Mooncake 更轻**

---

## 4. 结合你的目标，怎么选更合理？

你的目标可以拆成两件事：

1. **给 vLLM 提供跨实例高速前缀 KVCache**
2. **底层尽量吃满 3FS 集群带宽**

### 推荐优先级

### 方案 A：继续 Mooncake，但别从零硬接 3FS
**适合：你已经在 Mooncake 路径上，想尽快服务 vLLM**

建议：
1. 先把 **MooncakeStoreConnector + vLLM V1** 跑稳（DRAM/SSD 即可）
2. 再评估官方 **HF3FS experimental plugin**
3. 同时跟踪 Mooncake 的 **DFS replica / POSIX+HF3FS adapter** 相关改动
4. 自己补齐：
   - eviction / GC
   - 容量水位
   - 故障后一致性
   - 指标（hit rate、IO 延迟、空间回收）

优点：与 vLLM 集成最顺
缺点：3FS 路径仍要你承担较多工程债

### 方案 B：Tair Manager 管元数据，后端挂 3FS / Mooncake
**适合：你更在意全局管理、多后端、容量策略**

建议：
1. 用 Tair Manager 做全局索引/匹配/容量控制
2. 存储后端优先 HF3FS
3. 必要时把 Mooncake 当传输/本地缓存层，而不是唯一控制面

优点：架构更像「企业级分布式前缀缓存」
缺点：开源部分可能不够完整，工程对接量也不小

### 方案 C：不建议
- 直接把 3FS 当 vLLM KVCache 后端自己搓一套完整语义
除非你团队有很强的存储 + 推理系统工程能力，否则成本会显著高于 A/B。

---

## 5. 对你当前想法的直接建议

你说：

> 本来想在 Mooncake Store 基础上加 DFS/3FS 集成，但发现很多坑；感觉 Tair 更好，但开源维护不足。

我的判断：

### 1. 不要把「Mooncake 有 3FS 插件」理解成「生产就绪」
它是 **可评估路径**，不是成熟终态。

### 2. 如果你主战场是 vLLM agentic / 多轮共享前缀
优先保证：

- Mooncake + vLLM 的 **分布式前缀共享** 先跑通
- 3FS 先作为 **二级/三级存储**，而不是一开始就做唯一主路径

### 3. 如果你最看重 3FS 与全局缓存治理
更值得研究 Tair 的：
- Manager 元数据模型
- 两阶段写
- 多后端抽象
- 回收与水位控制

即使最终不直接用 Tair 代码，这些设计也值得抄。

### 4. 现实折中
很多团队最终会落到：

```
vLLM
  → MooncakeStoreConnector（或 Tair Connector）
    → 本地 DRAM / 节点 SSD（热）
      → 3FS（温/冷、跨节点共享）
```

而不是一上来全量 3FS 热路径。

---

## 6. 简要最终判断

- **有没有经验？**
  **有。** 主要是 **Mooncake Store + vLLM 分布式前缀缓存**；3FS 作为底层存储后端已有官方/社区探索，但偏实验。

- **有没有「3FS 集群直接给 vLLM 做高速前缀 KVCache」的成熟开源经验？**
  **基本没有成熟公开方案。**

- **Mooncake 接 3FS 坑多吗？**
  **是的，公开状态也支持这个判断：experimental、eviction/清理不足、DFS 语义还在补。**

- **Tair 是否更好？**
  **在全局管理、多后端、容量治理上更完整；但开源可维护/可落地程度弱于你对完整产品的期望。**

- **你现在最稳的路线：**
  1. 先稳 **vLLM + Mooncake 分布式前缀缓存**
  2. 再把 **3FS 当 offload/共享存储层**
  3. 同步吸收 **Tair Manager** 的元数据与容量管理设计
  4. 不要指望当前 HF3FS 插件直接等于生产级方案

如果你愿意，我可以下一步直接帮你出一版：
**「Mooncake + 3FS + vLLM 的最小可落地方案」**，包括推荐拓扑、该用哪些组件、哪些坑先规避、以及和 Tair 架构的对照表。

---

### 🧑‍💻 User

是的，现在接入3FS的方案很多，但是缺乏集群级控制的方法。现在的vllm 有接入的connector，这我明白。但是如何管理空间配额成为了新的问题，直接用的话没法管理全局空间的配额。这块有无比较好的解决方案？
我记得tairkv 同时还开了他的tair kv mgmt吧，但适配只到vllm 0.13

---

### 🤖 Assistant

你这个问题很关键：连接器有了，缺的是集群级空间配额与全局治理。我先查一下 Tair KV Mgmt、Mooncake、LMCache 等对全局配额/容量管理的公开方案和适配现状。

你的问题非常准：
**vLLM connector 解决的是“怎么接”，不是“怎么管全局空间”。**
现在真正缺的是 **cluster-level quota / capacity control plane**。

---

## 结论先说

### 目前比较好的解法，按成熟度排序：

| 方案 | 全局配额能力 | 3FS 友好度 | vLLM 适配现状 | 评价 |
|---|---|---|---|---|
| **Tair KVCache Manager (KVCM)** | **最完整** | 原生支持 HF3FS | 有 vLLM connector，但版本/维护偏旧 | **架构上最对路** |
| **自建轻量 Meta Manager + 现有 Store** | 可做成你要的 | 好 | 可控 | **工程上最务实** |
| **Mooncake Store 自身** | 弱/演进中 | 有，但偏存储层 | 好 | 不是完整配额系统 |
| **3FS 本身** | 无 LLM 语义配额 | 自身就是存储 | 无 | 不适合直接管 prefix cache 配额 |

你说的这句基本正确：

> Tair 开了 tair kv mgmt，但适配只到 vLLM 0.13

公开仓库确实是把 **Manager + Optimizer/HiSim** 开源了，并有 `py_connector/vllm`，但社区活跃度和“跟主线 vLLM 持续对齐”明显弱于 Mooncake。

---

## 1. 为什么 connector 不能解决配额问题

当前主流路径是：

```text
vLLM
  └─ KVConnector / StoreConnector
       └─ Mooncake / 3FS / NFS / 本地 SSD
```

这一层通常只负责：

- block 查找
- 读写路径
- 传输

**不负责（或只做局部）：**

- 多模型 / 多租户总配额
- 热/温/冷分层配额
- 集群级硬上限（hard quota）
- soft watermark 触发回收
- 按 Instance Group 隔离
- 版本切换时容量转移
- 全局命中率与容量 ROI 决策

所以你现在遇到的是：

> **有高速缓存池，没有企业级容量控制面。**

这在 3FS 场景尤其严重：
3FS 给你的是高带宽共享盘，不是“带 LLM 语义的配额系统”。

---

## 2. 目前最好的公开方案：Tair KVCache Manager

你记得没错，**Tair 开源的就是 Tair KVCache Manager（KVCM）**。
它不是又一个 store，而是：

```text
推理引擎 (vLLM/SGLang/...)
        │  元数据查询 / 两阶段写
        ▼
Tair KVCache Manager  ← 全局控制面
   ├─ MetaIndex（全局索引）
   ├─ Capacity Mgmt（Quota / Waterline / Eviction）
   ├─ DataStorage（HF3FS / Mooncake / NFS / ...）
   └─ Optimizer（容量仿真与参数调优）
        │
        ▼
后端存储：3FS / Mooncake / NFS / MemPool
```

### 它为什么适合你要的“全局空间配额”

官方架构里明确有 **Capacity Management (Reclaimer & Executor)**，而且是你真正需要的那种：

### 1) Instance Group 级配额（最关键）
不是给每个 vLLM 实例单独配容量，而是：

- **Instance Group 共享一套配额**
- 一个 Group 可对应：
  - 一个业务团队
  - 一个模型的多个版本
  - 重要模型独占
  - 多个长尾模型共享

### 2) 总配额 + 分后端配额
例如：

- 总 Quota = 100T
- TairMemPool Quota = 1T
- 3FS Quota = 99T

这正好对应你现在的形态：

- 本地/内存热层
- 3FS 温/冷共享层

### 3) 硬上限 + 软水位
- **Hard Quota**：超限后停止分配写地址
- **Waterline**：到水位开始 eviction
- 支持按 storage type 单独触发回收

### 4) 异步删除 / 可扩展回收
删除走后台线程池，不堵前台请求。
这对 3FS 很重要，否则 GC 一上来，IO 和命中率都会抖。

### 5) 两阶段写保证“容量记账正确”
- `startWriteCache`：先拿写位置并记账
- 本地写 3FS / Mooncake
- `finishWriteCache`：确认成功后转 serving

这比“先写盘再补登记”更适合做配额系统，因为：

- 能避免超卖
- 能处理写失败回滚
- 能避免幽灵占用

---

## 3. 你提到的痛点：Tair 适配只到 vLLM 0.13

这点基本属实，而且是当前最大现实障碍。

### 现状判断

**Tair 开源仓库强项：**
- Manager 架构完整
- 容量/配额模型清晰
- 多后端抽象（含 HF3FS、Mooncake）
- Optimizer / HiSim 有价值

**弱项：**
- 社区维护偏轻
- vLLM connector 更像“可用参考实现”，不是持续跟主线
- 对新 vLLM V1 connector API / scheduler 语义的跟进不确定
- 端到端“开箱即生产”程度一般

所以更准确的评价是：

> **Tair KVCM 是当前最好的“全局配额控制面蓝本”；**
> **但不一定是你今天就能无痛接入新版 vLLM 的完整产品。**

---

## 4. Mooncake 现在有没有全局配额？

### 有一点，但不够

Mooncake 更像：

- 高性能分布式 KV object store
- 有 eviction / offload / 本地与 SSD 容量控制
- 近期有 tenant quota RFC / 讨论

但它**不是**完整的企业级容量管理系统：

- 多模型 / 多版本 / 多租户 Instance Group 配额不成熟
- 对 3FS 文件清理 / 生命周期管理仍有坑
- 很多策略仍偏 store 层，而不是 LLM 语义控制面
- 历史上也出现过“eviction 策略留给上层框架”的边界问题

所以：

> **Mooncake 适合做高速共享缓存池，不适合单独承担全局配额治理。**

---

## 5. 真正比较好的解决思路

### 方案 A：推荐主路径
**Tair Manager 思路 + 你现有 3FS/Mooncake 存储**

不要把 Tair 当成必须完整替换现网的黑盒，而是拆成两层：

```text
控制面：Tair KVCM / 自研轻量 KVCM
数据面：3FS (+ 可选 Mooncake 热层)
接入面：vLLM connector
```

#### 你要的最小能力集（必须有）

1. **全局 block 元数据索引**
2. **Instance Group 配额**
3. **总容量 + 分介质配额**
4. **soft waterline + hard limit**
5. **两阶段写**
6. **异步 GC / eviction**
7. **按模型/版本/租户隔离**
8. **可观测：used / hit rate / reclaim rate / write reject**

如果 Tair connector 跟不上新 vLLM，可以：

- 保留 Tair Manager 做控制面
- 自己写一个薄 vLLM V1 connector
- 只对接：
  - `getCacheLocation`
  - `startWriteCache`
  - `finishWriteCache`
  - `remove/trim`

这比你从零做配额系统便宜很多。

---

### 方案 B：如果你不想绑 Tair
**自建“轻量 KV Capacity Manager”**

这其实是很多团队最终会走的路。

#### 建议最小架构

```text
vLLM Connector
   │
   ├─ Lookup(prefix/block keys)
   ├─ Reserve(bytes, group, media)
   ├─ Commit(location)
   └─ Release/Evict
          │
          ▼
KV Capacity Manager
   ├─ Redis/Valkey/etcd: meta + quota counters
   ├─ Reclaimer worker
   └─ Policy:
        - group quota
        - media quota
        - LRU/LFU/TTL
        - prefix-aware eviction
          │
          ▼
3FS / Mooncake / local SSD
```

#### 配额模型建议直接抄 Tair 这套

```text
Account / Team
  └─ InstanceGroup
       ├─ total_quota
       ├─ media_quota:
       │    dram: x
       │    ssd: y
       │    3fs: z
       ├─ waterline: 0.85
       └─ Instances:
            - model A v1
            - model A v2
```

#### 关键原则

- **配额记在 control plane，不记在 3FS 目录大小**
- 3FS 只是 backend
- 写入前先 reserve
- 成功 commit，失败 rollback
- 删除异步，元数据先标 deleting

否则你会一直遇到：

- 3FS 空间已满但元数据还在
- 元数据在但文件已被清
- 多实例同时写导致超卖
- 无法按模型公平分配

---

### 方案 C：不推荐
**继续只靠 Mooncake/3FS/vLLM connector 硬撑**

短期能跑，长期一定爆这几个问题：

1. 某个热模型把 3FS 吃满
2. 版本切换时新旧模型互相挤压
3. 长尾模型没空间
4. 无法解释“为什么命中率掉了”
5. GC 和业务写互相打架

---

## 6. 针对“3FS 接入很多，但缺集群级控制”的具体建议

你现在最该补的不是又一个 connector，而是这 4 个控制点：

### 1. Admission Control（写准入）
写之前先问控制面：

- 这个 group 还有没有 quota
- 这个 backend 是否超过 waterline
- 该写热层还是 3FS

### 2. Placement Policy（放哪）
例如：

- 高频前缀 -> 本地 DRAM / Mooncake
- 中频共享前缀 -> 3FS
- 低频/大对象 -> 3FS cold dir

### 3. Eviction Policy（怎么回收）
不要只做全局 LRU，至少：

- group 内回收优先于跨 group 抢占
- 先删 suffix 后删 prefix parent
- 先删低命中、长 TTL 过期、低价值 block
- 重要模型有 reserved floor

### 4. Observability（为什么满了）
最少要有：

- group used / limit
- media used / limit
- write reject 原因
- reclaim QPS / latency
- hit rate by model/version
- 3FS real usage vs meta usage 差

没有这些，配额系统不可运维。

---

## 7. 结合你现状，我建议这样落地

### 短期（2~4 周可验证）
1. 继续用现有 vLLM connector + 3FS/Mooncake 跑通读写
2. 加一个**外部容量网关**：
   - Redis/Valkey 记 group quota
   - 写前 reserve
   - 超限直接拒绝 offload / 降级本地
3. 先不做完美前缀感知 eviction，先做：
   - hard quota
   - waterline
   - async delete

### 中期
1. 引入 Tair KVCM 作为控制面，或按其模型自研
2. 后端继续挂 3FS
3. 可选 Mooncake 做热层
4. connector 自己维护到当前 vLLM

### 长期
1. 多租户 / 多模型版本隔离
2. Optimizer/trace 驱动容量规划
3. prefix-aware eviction
4. 热温冷自动分层

---

## 8. 直接回答你的两个问题

### Q1: 这块有没有比较好的解决方案？
**有，但不是“再找一个 3FS connector”。**
比较好的是：

> **独立的全局 KVCache Manager（配额/元数据/回收） + 3FS/Mooncake 做数据面**

目前公开方案里，**Tair KVCache Manager 最接近正确答案。**

### Q2: Tair 开了 mgmt，但适配旧，还能不能用？
**能借，不建议死等它跟主线。**

更现实的策略：

- **借 Tair 的配额模型与 Manager 设计**
- **数据面继续用你已有 3FS**
- **vLLM 适配层自己维护**
- 如果 Tair manager 服务端能跑，就优先复用服务端；connector 跟不上就自研薄封装

---

## 9. 一句话建议

你现在缺的不是“再接入一个 3FS 方案”，而是：

```text
Cluster KVCache Control Plane
= 全局元数据 + InstanceGroup 配额 + 水位回收 + 两阶段写 + 异步 GC
```

- **Mooncake/vLLM connector**：解决性能通路
- **Tair KVCM / 自研 KVCM**：解决集群配额治理
- **3FS**：继续做高速共享存储后端

---

如果你愿意，我下一步可以直接给你一版：

1. **可落地的全局配额架构图**
2. **最小 API 设计（Reserve/Commit/Lookup/Evict）**
3. **基于 3FS 的目录与记账方案**
4. **和 Tair KVCM 的对照表（哪些可复用，哪些建议自研）**