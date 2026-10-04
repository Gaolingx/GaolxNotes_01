# DSec与K8S复现

## 🧑‍💻 User

我在想能不能在K8S上复现DSec的优势。（包括但不限于元数据的本地性与可恢复性的协同管理，自动scale up的容量探测策略的配置等）
以及DSec在被从现在的RL rollout拓展到也用于驱动云Cowork时，哪些假设仍然成立，哪些假设需要调整并且会导致代码调整。

---
## 😀 默认助手 (7)

I'll start by discovering what tools are available in the hub — there may be internal docs or code search capabilities that can ground this DSec discussion.

Let me do one quick grounding check in case DSec is a documented public system.

Found it — DSec is **DeepSeek Elastic Compute**, arXiv `2609.22978`. Let me pull the paper to ground the analysis.

The arXiv HTML is JS-heavy. Let me pull the secondary summaries which are cleaner, and try the paper via a text fetch.

Good — I have the abstract and TOC. Let me pull the core architecture and RL co-design sections in detail.

Fetch seems rate-limited now. Let me switch to the browser tool.

Now I have clean text (95,750 chars). Let me extract the key architecture and co-design sections.

Let me grab the remaining sections (misbehavior mitigations, implementation, evaluation).

Let me capture the SDK/backends/lifecycle section 2, then I'll close the browser.

I have the full picture now. Let me close the browser to free resources.

# DSec 在 K8S 上复现，以及“RL rollout → 云 Cowork”的假设迁移

> 说明：以下分析基于 DeepSeek 官方论文 **DSec: DeepSeek Elastic Compute — A Sandbox Infrastructure for Effective Agentic Training at Scale**（arXiv `2609.22978`，2026-09-19）的公开内容（我实际抓取了正文并做了核对）。"云 Cowork" 我按 **面向外部租户、交互式、长会话的云端协作 Agent 产品** 来理解；如果你们的定义不同，第 2 节的结论需要按威胁模型/生命周期再校准。

---

## 0. 先对齐：DSec 的“优势”到底是什么

先把它拆成可复现的机制清单，否则"复现优势"会变成空对空。论文里真正的差异化点有 7 条：

| # | 机制 | 关键事实 |
|---|------|---------|
| M1 | 多后端统一抽象 | FnCall / container / Firecracker microVM / full VM，一个 SDK（`libdsec`） |
| M2 | 可组合环境层 | base image + workspace + toolkit 独立版本化，overlayfs 动态拼 `lowerdir`，把升级成本从 $O(m\cdot N)$ 降到 $O(m)$ |
| M3 | 按需镜像加载 + 元数据本地化 | OCI 离线转 EROFS，**元数据与数据分离，元数据预取到本地**，数据按需从 3FS 批量拉；写本地 |
| M4 | 高密度超卖 | 90% sandbox 平均 CPU <5%；单节点稳定跑到 ≥3200 container / 800 microVM |
| M5 | 内存共享与回收 | virtio-pmem+DAX（峰值内存 −40.2%）、DAMON+balloon 空闲页上报（时积分 −21.2%） |
| M6 | QoS 感知 CPU 调度 | LS/BE 分级，BE 进 `SCHED_IDLE`，LS 开 **core scheduling**（SMT 兄弟隔离），延迟膨胀 45.2% → 17.3% |
| M7 | 与 RL 框架协同 | rollout 与可抢占 GPU 训练解耦、pause/resume 保状态、`pack_diff` 环境构建、AppArmor/eBPF 反 reward hacking |

控制面结构：`apiserver`（无状态，sandbox ID 编码所属 edge）→ `placement engine`（filter+rank，**power-of-$k$**，叠加 in-flight 本地视图）→ `edge`（节点本地准入 + 快照/资源回收）→ `aether`/`chronus`（代理 + shell 会话）；`watcher` 周期性探测 edge 健康与负载；镜像层落 3FS。部署：单 scale unit ≈160 节点 / 30K core / 250TB DRAM，3M sandbox/天、380K 并发、>5000 创建/秒。云突发：利用率 >80% 时按“镜像依赖是否落在 30TB 去重镜像集（覆盖 70% 任务）”**选择性**卸载到云 VM，200 台吸收 ~30% 峰值。

记住这几个隐含前提，第 2 节会逐个对账。

---

## 1. 能否在 K8S 上复现 DSec 的优势？

**一句话结论：机制可复现，但 DSec 的“优势”不在单个功能，而在控制面里那条连续闭环（探测→放置→节点准入→回收）。搬到 K8S 意味着你把这套闭环重写成 controller / scheduler plugin / CSI，并且要接受 K8S 在控制面吞吐、缺失原语上的差距——尤其是“元数据本地性 × 可恢复性的协同”不是 K8S 原生能力。**

### 1.1 组件映射表

<details>
<summary>点开：DSec ↔ K8S 逐组件映射与缺口</summary>

| DSec 组件 | K8S 对应 | 缺口 |
|-----------|----------|------|
| `apiserver`（无状态、水平扩展） | kube-apiserver + Service | etcd 有状态，是单点吞吐瓶颈；DSec 的“无状态”被反转 |
| `placement engine`（power-of-$k$ + in-flight 叠加 + admission 兜底） | kube-scheduler + scheduler-plugins；`NodeResourcesFit` LeastAllocated | 需自写插件；in-flight 叠加、批量突发（>5000/s）默认调度器扛不住 |
| `watcher`（主动探测、无持久状态、可重建） | kubelet 上报 + metrics-server + node-exporter | 偏被动；“探测式容量发现”要自己写 |
| `edge`（本地准入、快照、eBPF 网络策略） | kubelet + CRI + CNI(Cilium eBPF) | 激进超卖的节点准入语义不同 |
| `aether`/`chronus`（多 shell 会话代理） | `kubectl exec` / attach | K8S exec 是单进程语义，多会话/异步输出要自研 |
| EROFS 按需加载 + 本地元数据（3FS） | 懒加载 snapshotter（nydus / stargz / SOCI）+ CSI | metadata/data 分离、本地元数据缓存可复现；3FS 的“大 IO 强、小随机 IO 弱”这一约束要换存储后端 |
| OverlayBD + ublk（microVM 块设备） | OverlayBD 已开源；CSI block | 需自建 CSI，或走 Kata/firecracker-containerd |
| 多后端隔离 | Kata Containers / firecracker-containerd / KubeVirt | 能覆盖“强隔离边界”，但 FnCall 那种“复用预创建容器 + 预热池”不是 K8S 原语 |
| 内存/CPU QoS（DAX、DAMON、core scheduling） | 特权 DaemonSet + cgroup v2 | **core scheduling 和 virtio-pmem 不在 K8S API 里**，只能靠 runtime hook / 宿主机调参 |
| pause/resume + 快照 | CRIU / checkpoint（alpha）、KubeVirt snapshot | 容器内状态保活不是稳定原生能力 |
| 云突发（80% 阈值 + 选择性卸载） | Karpenter / Cluster API / Karmada / virtual-kubelet | “按镜像集判定 cloud-eligible”要自写 admission webhook |
| IAM 多级 project + delegation | RBAC + Namespace + ResourceQuota（HNC 补层级） | 多级嵌套 + 委托语义需自研 |
| BGP ECMP + 多实例 | MetalLB / 云 LB + Deployment 多副本 | 基本等价 |

</details>

### 1.2 你点名的两个优势

#### (a) 元数据本地性 × 可恢复性的“协同”管理

DSec 的做法本质是一个**分区策略**：把状态分成三类，各自选择"本地"还是"远程可恢复"——

- **本地**：EROFS 元数据、overlay 可写上层的写、page cache 缓存（本地小 IO 友好）；
- **远程/可恢复**：只读镜像数据（3FS 复制保证）、不可变层（内容寻址，丢了可重建）；
- **无状态/可重建**：placement engine、watcher（重启后重新轮询即可）。

于是"可恢复性"不是靠本机耐久，而是靠**不可变 + 可重建**：节点挂了，元数据重新拉一遍即可，代价极低。这就是"本地性"与"可恢复性"能协同的根因。

**在 K8S 上：**
- 概念可复现：懒加载 snapshotter（nydus/stargz）天然是"本地元数据 + 远程 blob + 内容寻址"，节点丢失后重调度即可重建元数据，与 DSec 同构。
- **但"协同"不是 K8S 原语。** K8S 调度器只知道 PVC/topology，**不知道哪个节点缓存热**；DSec 的 placement engine 通过 watcher 掌握每 edge 的运行时状态。要在 K8S 复现，你得补四件事：
  1. 把"元数据/数据驻留"暴露成节点标签或 CRD；
  2. 写 scheduler plugin 让放置同时权衡本地性与可恢复性；
  3. 写 controller 做预取/驱逐；
  4. 用 topologySpread / PDB 表达可恢复性约束。
- **哲学反转**：DSec 控制面刻意无持久状态；K8S 的 etcd 是"持久、单源真相"。这意味着如果照搬 DSec 把元数据塞进 CRD，会撑爆 etcd；正确做法是**元数据留在 snapshotter 本地存储，etcd 只放指针**——这才是 K8S 版本"本地性 × 可恢复性"的协同关键。

#### (b) scale-up 的容量探测策略（可配置）

DSec 的语义是**探测式**而非**目标式**：watcher 周期探 edge 的健康与负载（每 edge/user/task 的 sandbox 计数），placement 拉取，edge 做本地准入兜底，>80% 触发选择性云突发。可调项大致是：探测周期、power-of-$k$ 的 $k$、水位线、冷却、eligibility 谓词。

**在 K8S 上：**

| DSec 探测语义 | K8S 映射 | 差距 |
|---|---|---|
| watcher 周期探测 | Prometheus/node-exporter + 自定义探针 controller | 需自建"容量探测"契约 |
| 探测指标驱动扩容 | **KEDA** external scaler（pollingInterval/cooldownPeriod 即探测配置） | 语义是"指标响应式"，非"容量发现式" |
| 节点扩容 | **Karpenter** / Cluster Autoscaler | 默认数分钟；DSec 靠预热池达 5000/s，K8S 需"常驻 idle pod 池 + dispatcher" |
| 80% 水位 + 选择性卸载 | 自定义 autoscaler + admission webhook | 无原生等价物 |
| 节点本地准入（edge） | kubelet eviction/admission | 激进超卖准入策略要自研 |

**结论**：**策略"可配置"能实现，但那是你自己写的探测 controller + KEDA/Karpenter 的组合，K8S 只提供伸缩执行器。** 并且 K8S 的伸缩是"根据指标反推",不是 DSec 的"探测真实剩余容量 + 本地准入"。

### 1.3 K8S 上的硬约束（复现不止是工程量问题）

1. **控制面吞吐**：5000+ 创建/秒、32K/任务的亚秒级突发，会直接压垮默认调度器与 etcd；DSec 的"无状态 + 水平扩展"是为此设计的。
2. **缺失原语**：core scheduling、virtio-pmem+DAX、本地超卖准入、FnCall 预热池、pod checkpoint/resume —— 都不在 K8S API 内，只能下沉到宿主机/特权 DaemonSet/自定义 runtime。
3. **恢复哲学不同**：DSec "re-poll 重建" vs K8S "etcd 持久 + reconcile"。

**小结**：如果你把 DSec 当"分布式沙箱平台的一种实现"来复现，K8S 能覆盖 70–80% 的机制；但它最锋利的两个点（探测闭环、本地性/可恢复性协同）恰好是 K8S 最不原生的地方，得靠自定义 controller/CSI/scheduler 补齐。若目标只是"跑起来"，K8S 够用；若目标是"复现 5000/s 与高密度超卖"，K8S 会持续成为天花板。

---

## 2. 从 RL rollout 扩展到“云 Cowork”：哪些假设成立、哪些要改

### 2.1 先从论文里抽出 RL rollout 的隐含假设

| 编号 | 假设（论文依据） | 云 Cowork 下 |
|---|---|---|
| A1 | 消费者=内部训练/评估框架；IAM 用 project 多级嵌套 + delegation；builder 是内部可信研究者 | **需大改** |
| A2 | 需求是**批式突发**，绑定训练 batch（最大 32K/任务，亚秒级） | **需调整**（突发仍在，但由用户行为驱动） |
| A3 | GPU 训练会被抢占，RL framework **主动**发 pause；sandbox 必须保状态 | **失效** |
| A4 | 会话中等寿命（中位 17.4/15.5 min，p99 >3h），TTL 到点回收 | **需大改**（会话以天计） |
| A5 | rollout **可重放/幂等化**（command log），失败可重试 | **失效且危险** |
| A6 | 单一 scale unit 共享一套 3FS；云突发复用同一 EROFS 路径 | **失效**（公有多区 + 数据驻留） |
| A7 | 90% sandbox 平均 CPU <5% → 激进超卖，密度优先 | **大体成立**（但需租户公平） |
| A8 | 不可信的是 **agent**（reward hacking / 答案泄漏），builder 可信 | **需调整**（跨租户外泄/合规） |
| A9 | 控制面无持久状态、可重建；BGP ECMP 多实例 | **部分失效**（公有云托管） |
| A10 | 环境由 agent 构建（`pack_diff`） | **失效**（环境由产品方定义） |

### 2.2 逐条展开：失效/需调整 → 代码改动

<details>
<summary>点开：A1 租户与 IAM（改动最大）</summary>

- RL：`IAM` 是 project 嵌套 + 配额委托，人与 agent 共用同一管理 API，粒度偏粗。
- Cowork：外部多租户 → 需要 **org/user/tenant 三级 + 计量 + 计费 + 审计 + SSO + 滥用防护**；delegation 语义要重写成"配额与权限不可越界"的产品化模型。
- 代码：`IAM` 主体模型、`apiserver` 鉴权链、quota/metering 中间件、审计事件流。这是**结构性改动**，不是加个字段。
</details>

<details>
<summary>点开：A3 会话生命周期（RL 耦合需拆除）</summary>

- RL：`§6.3` 由训练框架在抢占时给所有 sandbox 发 pause，用 `docker pause` + `memory.reclaim` / microVM snapshot 回收内存。
- Cowork：没有训练抢占。pause/resume 机制**仍然有用**（空闲会话回收内存），但**触发源要从"训练 preemption"换成"产品策略"**：空闲超时、scale-to-zero、成本控制、会话恢复。
- 代码：lifecycle controller 去掉 RL 耦合；新增空闲调度器、显式保留策略、前台恢复路径。
</details>

<details>
<summary>点开：A4/A5 持久性与一致性（最容易被低估）</summary>

- RL：会话寿命分钟级；失败可重试；V4.1 已让 worker container + agent sandbox 成为 rollout 的**单一真相源**（这点对 Cowork 是红利）。
- Cowork：会话以天计，用户操作**不可重放**，丢失=可见故障；多人/多 agent 并发编辑需要**强一致或冲突消解**（CRDT/OT/锁）。
- 代码：持久化与快照升为一等公民；overlay 可写上层要升级为"可协作工作区"；增加重连、迁移、跨故障域复制；`libdsec` 要暴露会话恢复/工作区卷语义。
- 注意：`§6.2` 的"command log 重放"假设在 Cowork 中**必须彻底移除**（已被单一真相源取代），但要补齐 exactly-once 语义。
</details>

<details>
<summary>点开：A6 存储与本地性（3FS → 云原生）</summary>

- RL：所有 scale unit 共享一套 **3FS**（私有、RDMA、20×15TB SSD/节点）；云突发把 30TB 去重镜像同步到云侧文件系统。
- Cowork：公有云没有 3FS；需要对象存储 + 本地二级缓存，并**按 region/数据驻留**做放置。
- 代码：存储后端抽象层；placement engine 增加区域/驻留约束；`§5.3` 的"元数据本地、数据远程、写本地、批量读"原则**保留**，但"3FS 大 IO 强/小随机 IO 弱"这一具体约束不再成立，缓存 chunk 大小/预取策略要重调。
</details>

<details>
<summary>点开：A7 超卖与公平性</summary>

- 超卖本身在 Cowork 依然成立（交互式 agent 大部分时间在等模型）。
- 但**多租户下要防 noisy neighbor**：DSec 的 LS/BE 与 per-user 计数要升级为**带租户配额的公平调度**，否则一个租户的突发会打穿他人 SLO。
- 代码：scheduler 加 fairness/quota；`watcher` 增加 per-tenant 计量。
</details>

<details>
<summary>点开：A8 安全威胁模型</summary>

- RL：防的是 agent **内部刷分/答案泄漏**（AppArmor 控文件与 socket、eBPF 控域名白名单）。
- Cowork：主威胁变成**跨租户数据外泄、prompt injection 诱导外泄、合规**。
- 代码：策略引擎从"任务级 allowlist"改为**租户作用域 + 审计 + DLP**；`§6.5` 的 AppArmor/eBPF 机制保留，但策略生成与归属要重构。
</details>

### 2.3 仍然成立的红利（这些别丢）

| 机制 | 为什么在 Cowork 更值钱 |
|------|----------------------|
| M2 可组合环境层 | 租户/toolkit 版本组合爆炸，独立版本化收益更大 |
| M3 按需加载 + 元数据本地 | 租户镜像更多样、fanout 更低，收益更大 |
| M4+M5 高密度与内存回收 | 交互会话大量空闲，回收价值高 |
| M6 LS/BE CPU QoS | "交互（LS）vs 批处理（BE）"分工天然对应 |
| M1 多级隔离（FnCall/容器/microVM/VM） | 不可信代码 + 多租户，隔离等级选择更重要 |
| 突发快速供给 | 用户行为本身突发 |
| 多级委托（概念） | 映射到 org/tenant 层级，思路成立、实现改 |
| worker+sandbox 单一真相源（V4.1） | 正是 Cowork 会话持久化的地基 |

### 2.4 一张“假设 → 代码”总表

| 假设 | 状态 | 主要代码改动 |
|------|------|-------------|
| A1 内部单信任域 | 需大改 | IAM 主体/配额/计量/审计 |
| A2 批式突发 | 调整 | 放置与预热池策略、自动伸缩驱动源 |
| A3 训练抢占驱动 pause | 失效 | lifecycle controller 去 RL 耦合，加空闲/恢复策略 |
| A4 分钟级 TTL | 需大改 | 生命周期、GC、保留策略 |
| A5 可重放 rollout | 失效 | 会话持久化、exactly-once、重连/迁移 |
| A6 单套 3FS | 失效 | 存储后端抽象、多区/驻留放置 |
| A7 激进超卖 | 大体成立 | 加租户公平与配额 |
| A8 反 reward hacking | 调整 | 租户级策略 + 审计 + DLP |
| A9 无状态控制面 | 部分失效 | 视部署目标（自建可保，K8S 则反转） |
| A10 agent 构建环境 | 失效 | 环境流水线改为产品方版本化 |

---

## 3. 把两个问题串起来 + 需要你确认

- **Q1 与 Q2 其实是同一个问题的两面**：K8S 复现考的是"**环境依赖假设**"（3FS、BGP、无状态控制面、宿主机内核特性），Cowork 迁移考的是"**工作负载依赖假设**"（批式、可重放、内部信任、训练抢占）。两者都会逼你把 DSec 的隐式契约显式化，而**显式化之处就是代码要改之处**。
- **最具性价比的路线**：先按 2.3 的"红利清单"把 M2/M3/M4/M5/M6/M1 保留；把改动集中在 IAM、lifecycle、storage 抽象、scheduler 公平性这四处（恰好也是 K8S 化时必须补齐的自研 controller）。
- **两个高风险点**：A5（会话不可重放 → 必须 exactly-once，不能靠 command log）与 A1（多租户 IAM/计量是结构性重构）。

**需要你确认以进一步收敛：**
1. "云 Cowork" 的确切定义与 SLO（延迟敏感度、会话时长、是否多人协作编辑）？
2. 目标部署是**公有云 K8S**，还是自建机房复用 3FS？（这直接决定 A6/A9 的改动面）
3. 你们更想复现的是"**吞吐/密度**"（5000+/s、3200 container/节点）还是"**弹性与成本**"？两者在 K8S 上的取舍方向不同。

你把其中任一条定下来，我可以把它细化成具体的 CRD / scheduler plugin / CSI / webhook 设计草案，以及对应的 `libdsec` API 变更清单。
