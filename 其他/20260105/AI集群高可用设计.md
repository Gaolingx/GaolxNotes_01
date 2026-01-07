# AI集群高可用设计

### 🧑‍💻 User

万卡训练集群 可能平均每几个小时就会出现掉卡情况，从基础设施（网络、存储、计算等）到上层应用（训练框架、平台监控等）是如何保证ai集群高可用的？

---

### 🤖 Assistant

在**万卡级（10k+ GPU）训练集群**中，“平均每几个小时就掉一两张卡”在工程上是**常态而非异常**。所谓**高可用（High Availability, HA）**并不是“不出故障”，而是**在持续故障背景下，保证训练吞吐、正确性和可恢复性**。这需要从**基础设施 → 系统软件 → 训练框架 → 训练策略 → 平台治理**形成一整套“容错栈”。

下面按层次系统性说明。

---

## 一、设计前提：大规模集群的故障模型

万卡集群必须接受的现实：

* GPU/网卡/光模块：**MTBF 按卡算很高，但按集群算极低**
* NCCL collective 对 **单点失败极度敏感**
* 通信拓扑（IB/NVLink）中，**一张卡等于一个 failure domain**
* 存储、etcd、scheduler、control plane 同样会抖动

因此 HA 的核心目标不是“避免失败”，而是：

> **Fail fast + Isolate + Recover + Resume with minimal lost work**

---

## 二、基础设施层（Infra）：把“掉卡”变成“可隔离事件”

### 1. 计算节点（GPU / Host）

**关键思想：节点/卡是一次性资源**

* GPU 被视为 *ephemeral compute*
* 不假设 GPU 会“陪你训练到结束”

常见手段：

* **GPU health daemon**

  * Xid error、ECC error、PCIe AER
  * NVML / DCGM 持续巡检
* **故障卡自动 cordon**

  * 标记为 `unschedulable`
  * 防止被再次调度

> 目标：**让坏卡尽快退出调度系统**

---

### 2. 网络（InfiniBand / RoCE）

网络是万卡集群的**第一大不稳定源**

常见 HA 设计：

* **多 rail IB（2~4 rails）**

  * NCCL 自动多路径
* **Leaf-Spine 全冗余**

  * 单交换机故障 ≠ 集群中断
* **自愈路由**

  * OpenSM / UFM 自动 re-route
* **端口级隔离**

  * 某个 IB port error → 局部隔离而非整机下线

关键目标：

> 把“网络抖动”限制在 **少数 ranks**，而不是全 job 死亡

---

### 3. 存储（Checkpoint / Dataset）

存储层 HA 的两个不同目标：

| 场景         | 要求                  |
| ---------- | ------------------- |
| Dataset    | 高吞吐、可 cache、可 retry |
| Checkpoint | 强一致、可恢复、低频写         |

常见方案：

* Dataset：

  * 本地 NVMe cache + 对象存储（S3 / OSS）
  * 失败可 retry
* Checkpoint：

  * 分布式 FS（Lustre / CephFS）
  * 或对象存储 + 原子 commit

**Checkpoint 必须假设：**

> “写到一半节点死了”

---

## 三、系统软件层：调度与作业生命周期管理

### 1. 调度器（K8s / Slurm / Volcano）

万卡训练必须具备：

* **Gang scheduling**

  * 要么全起，要么不起
* **抢占 / 重试**

  * job 可被 kill 并重新拉起
* **故障域感知**

  * 同一 job 的 ranks 尽量分散到不同机架

调度器不追求“稳定运行”，而是：

> **快速失败 + 快速重建**

---

### 2. Control Plane 高可用

* etcd 多副本
* Scheduler / controller 多实例
* metrics、logging 与训练解耦

> Control plane 掉 ≠ 训练立刻掉

---

## 四、训练框架层：真正的“HA 核心”

这是**决定万卡是否可用的关键层**

---

### 1. 通信容错：避免“一卡死，全 job 死”

#### 现实情况

* NCCL collective **不支持 rank 动态消失**
* 一张卡 hang，所有 ranks 阻塞

#### 工程解法（不是学术）

1. **Fail-fast NCCL**

   * 设置 timeout
   * 避免 silent hang

2. **进程级重启，而非节点级**

   * Watchdog 检测 rank 卡死
   * 直接 kill 全 job

3. **快速 resume**

   * checkpoint 足够频繁（例如 5~15 分钟）

> 结论：
> **不是不失败，而是失败成本可控**

---

### 2. 并行策略本身就是 HA 设计

不同并行方式对故障的敏感度完全不同：

| 并行方式         | 掉 1 卡影响           |
| ------------ | ----------------- |
| TP           | 直接全 job 死         |
| PP           | 某 stage 死，全 job 死 |
| FSDP         | 可重启但损失较大          |
| **EP (MoE)** | **最“抗掉卡”**        |

你之前问过 **“为什么 MoE 用 EP 而不是 TP”**，HA 正是重要原因之一：

* Expert 是天然的 failure domain
* 掉一个 expert，影响局部 token
* 可通过 re-routing / capacity over-provision 缓冲

---

### 3. 弹性训练（Elastic Training）

越来越多大厂内部都在用：

* **Elastic DDP / Elastic FSDP**
* world size 变化
* 允许临时减少 GPU 数量

代价：

* optimizer state reshape
* LR / batch size 动态调整

但收益：

> **不必每次掉卡都回滚整个 job**

---

## 五、训练策略层：接受失败，最小化损失

### 1. Checkpoint 策略是 HA 的核心指标

不是“有没有 checkpoint”，而是：

* **多久 checkpoint 一次**
* **checkpoint 写多久**
* **恢复要多久**

工业级经验值：

* 10k 卡级别：

  * checkpoint interval：5~10 min
  * restart time：< 2 min
* 否则掉卡频率 > checkpoint 频率 → 永远跑不完

---

### 2. 数据顺序与随机性可恢复

必须保证：

* global step 可恢复
* data sampler 可复现
* dropout / rng state 可恢复

否则：

> “表面恢复了，实际训练轨迹已偏移”

---

## 六、平台与监控层：让“掉卡”变成统计事件

### 1. 监控的不是“有没有故障”，而是：

* MTTR（Mean Time To Recovery）
* 有效 GPU-hour / 实际 GPU-hour
* job 成功率
* checkpoint 成功率

### 2. 自动化运维闭环

* 掉卡 → 自动下线
* job fail → 自动 resubmit
* 连续失败 → 自动迁移机架 / 网络域

最终目标不是稳定，而是：

> **系统性吞吐最大化**

---

## 七、一句话总结（非常重要）

> **万卡集群的高可用不是“系统不出问题”，而是“在持续出问题的情况下，训练仍能以可预测成本推进”。**

这也是为什么：

* 大模型训练工程 ≠ 单机 DL
* 并行策略、checkpoint、调度、infra 是同一套系统工程
* “能不能跑完”比“峰值 TFLOPs”更重要

如果你愿意，下一步我可以帮你拆解某一家（如 Meta / Google / 字节 / DeepSeek）在**万卡训练 HA 上的具体工程取舍**，或者聚焦某一层（如 Elastic Training 的实现细节）。
