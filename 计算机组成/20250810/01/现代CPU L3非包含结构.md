# 现代CPU L3非包含结构

### 🧑‍💻 User

为什么现代cpu的l3都是非包含结构，streaming方式访问时内存与cache数据交换热点在l2 cache？

---

### 🤖 Assistant

这个问题本质上涉及 **现代 CPU Cache 层级的设计目标变化**，以及 **流式（streaming）访问对数据复用位置的改变**。我分两部分回答：
1）为什么现代 CPU 普遍采用 **非包含（Non-Inclusive / Mostly Non-Inclusive）L3**
2）为什么在 streaming 访问模式下，**内存与 cache 的数据交换热点落在 L2，而不是 L3**

---

## 一、为什么现代 CPU 的 L3 采用非包含结构

### 1. 包含式 L3 的历史背景

早期多核（如 Nehalem / Sandy Bridge）大量采用 **Inclusive L3**，主要原因是：

* **简化一致性协议**

  * L3 作为“所有 L2 的超集”，天然可作为 snoop filter
  * 无需额外目录结构
* **当时核心数较少**

  * L3 容量相对大，包含冗余开销尚可接受
* **内存延迟主导**

  * L3 miss penalty 相对 DRAM 还算可控

但这些前提在现代 CPU 中已不成立。

---

### 2. Inclusive L3 在现代架构下的致命问题

#### (1) 有效容量严重浪费

Inclusive L3 必须包含 **所有 L1/L2 数据的副本**：

[
\text{Effective L3 capacity} = \text{Physical L3} - \sum \text{L2 footprints}
]

在 32–128 核服务器 CPU 上，这个冗余极其巨大。

---

#### (2) 核数提升 → L3 变成一致性瓶颈

* Inclusive L3 是 coherence 的 **强制汇聚点**
* 所有 L2 eviction 都会触发 L3 更新
* 多核 streaming / scan workload 会造成：

  * L3 fill / evict storm
  * Ring / Mesh 上的带宽压力

---

#### (3) L3 延迟已不可忽略

现代 L3：

* 分布式（Slice per core / per CCX）
* 30–50+ cycles latency
* 远端 slice 延迟更高

继续把 L3 当作“强制中转层”会显著拉低 IPC。

---

### 3. 非包含 L3 的核心优势

| 优势                 | 说明                             |
| ------------------ | ------------------------------ |
| 最大化有效容量            | L3 不再被 L2 复制污染                 |
| 减少写入/驱逐流量          | L2 eviction 不必强制写入 L3          |
| 更好 NUMA/Chiplet 适配 | 每个 slice 更像 victim / directory |
| Streaming 友好       | L3 可被旁路或仅作探测                   |

因此，Intel（Skylake-SP+）、AMD Zen、Apple M 系列都转向 **Non-Inclusive / Mostly Non-Inclusive L3**。

---

## 二、为什么 streaming 访问时，热点在 L2 而不是 L3

### 1. Streaming 的关键特性

Streaming（线性扫描）具备以下特征：

* **几乎无时间局部性**
* 空间局部性由硬件 prefetch 覆盖
* 每个 cache line：

  * 被用 1–2 次
  * 很快被淘汰

这直接决定了 **最合理的“复用点”只能是 L2**。

---

### 2. 数据在层级中的真实流动路径

以典型现代 CPU 为例（非包含 L3）：

```
DRAM → L3 (optional / victim / probe)
        ↓
       L2  ←—— 主要数据交换热点
        ↓
       L1
```

**关键点：**

* Streaming load 会：

  * 由 prefetcher 直接拉到 **L2**
  * L3 只作为探测或 victim
* L2 是：

  * 私有
  * 带宽最高
  * 延迟最低（~10–15 cycles）
  * 不参与跨核一致性

---

### 3. 为什么不是 L3？

#### (1) L3 带宽不适合作为 streaming buffer

* L3 是共享资源
* 单核 streaming 若大量占用 L3：

  * 会挤占其他核心的 working set
  * 引发 mesh / ring contention

因此硬件刻意 **避免把 streaming 数据驻留在 L3**。

---

#### (2) Prefetcher 的目标就是 L2

现代 prefetcher：

* L2 stride / streamer prefetcher
* 甚至支持 **L2-only fill**
* 典型策略：

  * DRAM → L2
  * L3 bypass 或短暂落脚

---

#### (3) 非包含 L3 的“被动角色”

在 non-inclusive 设计中：

* L3 更像：

  * victim cache
  * coherence directory
  * cross-core probe point
* 而不是数据主通道

---

### 4. L2 成为真正的“带宽放大器”

Streaming 下 L2 的作用：

| 功能               | 说明               |
| ---------------- | ---------------- |
| DRAM → Core 的缓冲  | 吸收突发流量           |
| Prefetch 队列终点    | 隐藏 DRAM latency  |
| Write-combine 聚合 | 写回更高效            |
| 降低一致性开销          | 私有 cache，无 snoop |

这也是为什么在 profiling 中你会看到：

* L2 bandwidth 接近理论峰值
* L3 hit rate 很低，但 **并非性能瓶颈**

---

## 三、一个总结性对照

| 维度           | Inclusive L3 | Non-Inclusive L3 |
| ------------ | ------------ | ---------------- |
| Streaming 数据 | 污染 L3        | 多数绕过 L3          |
| 数据交换热点       | L3           | **L2**           |
| 一致性压力        | 高            | 低                |
| 多核扩展性        | 差            | 好                |
| 实际有效容量       | 低            | 高                |

---

## 四、一句话总结

> **现代 CPU 把 L3 从“强制数据中转站”降级为“共享探测 / victim 层”，而把真正的 streaming 数据吞吐职责下放给私有 L2；非包含结构正是这一角色转变的前提。**

如果你愿意，我们可以进一步结合 **Intel Skylake-SP / Sapphire Rapids** 或 **AMD Zen4 CCX** 的具体实现，把这个结论映射到真实硬件事件（如 L2_LINES_IN、L3_LOOKUP、Snoop Filter 命中率）上。
