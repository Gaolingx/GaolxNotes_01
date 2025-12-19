# NAT映射与过滤行为

### 🧑‍💻 User

rfc5780规定的nat中的映射行为和过滤行为分别是什么意思，分别有哪些？

---

### 🤖 Assistant

RFC 5780 的核心目的，是**标准化地描述和测试 NAT（Network Address Translation）在 UDP 场景下的行为特征**，以便像 STUN / ICE 这样的协议能够**可预测地穿越 NAT**。它将 NAT 行为明确拆分为两个**正交维度**：

* **映射行为（Mapping Behavior）**：
  *NAT 如何为内部 (IP:Port) 分配外部 (IP:Port)*
* **过滤行为（Filtering Behavior）**：
  *NAT 如何决定是否允许外部报文进入该映射*

这两个维度彼此独立，一个 NAT 的“类型”本质上就是二者的组合。

---

## 一、映射行为（Mapping Behavior）

映射行为回答的是：

> 当内部主机从同一个本地端口向不同外部目的地址/端口发送 UDP 包时，
> **NAT 是否复用同一个外部端口？**

RFC 5780 定义了 **三种** 映射行为（从“最宽松”到“最严格”）：

---

### 1. Endpoint-Independent Mapping (EIM)

**端点无关映射**

**定义**
只要是同一个内部 `(IP_i, Port_i)`，不管外部目标是谁：

```
(IP_i, Port_i)  →  (IP_e, Port_e)
```

映射是固定的。

**特征**

* 外部端口只与 *内部端点* 绑定
* 与外部目的 IP / Port 无关
* 映射可被复用

**示例**

```
10.0.0.1:5000 → 8.8.8.8:3478   → 203.0.113.5:62000
10.0.0.1:5000 → 1.1.1.1:3478   → 203.0.113.5:62000  （同一个端口）
```

**影响**

* 对 P2P / WebRTC **最友好**
* Full Cone / “端口保持” NAT 的核心特征

---

### 2. Address-Dependent Mapping (ADM)

**地址相关映射**

**定义**
映射依赖于 **外部 IP**，但不依赖端口：

```
(IP_i, Port_i, IP_o) → (IP_e, Port_e)
```

**示例**

```
10.0.0.1:5000 → 8.8.8.8:3478   → 203.0.113.5:62000
10.0.0.1:5000 → 8.8.8.8:9999   → 203.0.113.5:62000
10.0.0.1:5000 → 1.1.1.1:3478   → 203.0.113.5:62001  （不同 IP，不同端口）
```

**影响**

* 同一远端 IP 可以复用端口
* 不同远端 IP 会创建不同映射
* P2P 成功率下降，但仍可 ICE 处理

---

### 3. Address-and-Port-Dependent Mapping (APDM)

**地址 + 端口相关映射**

**定义**
映射依赖于 **外部 IP 和端口**：

```
(IP_i, Port_i, IP_o, Port_o) → (IP_e, Port_e)
```

**示例**

```
10.0.0.1:5000 → 8.8.8.8:3478   → 203.0.113.5:62000
10.0.0.1:5000 → 8.8.8.8:9999   → 203.0.113.5:62001
10.0.0.1:5000 → 1.1.1.1:3478   → 203.0.113.5:62002
```

**影响**

* 每个目的端点一个映射
* 典型的 **Symmetric NAT**
* UDP Hole Punching 最困难

---

## 二、过滤行为（Filtering Behavior）

过滤行为回答的是：

> **哪些外部主机** 可以向已经存在映射的 `(IP_e, Port_e)` 发送数据并被 NAT 放行？

同样定义了 **三种** 过滤行为：

---

### 1. Endpoint-Independent Filtering (EIF)

**端点无关过滤**

**定义**
只要存在映射，**任何外部主机** 都可以向该端口发包。

**特征**

* 不校验外部 IP / Port
* 等价于 “端口全开放”

**示例**

```
任意 IP:任意 Port → 203.0.113.5:62000  （都可进入）
```

**影响**

* Full Cone NAT
* 安全性最低
* P2P 最友好

---

### 2. Address-Dependent Filtering (ADF)

**地址相关过滤**

**定义**
只允许 **内部主机曾经发过包的外部 IP** 回包，端口不限。

**示例**

```
允许： 8.8.8.8:any → 203.0.113.5:62000
拒绝： 1.1.1.1:any → 203.0.113.5:62000
```

---

### 3. Address-and-Port-Dependent Filtering (APDF)

**地址 + 端口相关过滤**

**定义**
只允许 **完全匹配的外部 (IP, Port)** 回包。

**示例**

```
允许： 8.8.8.8:3478 → 203.0.113.5:62000
拒绝： 8.8.8.8:9999 → 203.0.113.5:62000
```

**影响**

* 等价于 UDP “连接语义”
* 最严格，最安全
* 对 P2P 最不友好

---

## 三、组合关系（NAT 行为矩阵）

RFC 5780 明确指出：
**映射行为 ≠ 过滤行为**

它们可以自由组合，例如：

| 映射行为 | 过滤行为 | 常见俗称                     |
| ---- | ---- | ------------------------ |
| EIM  | EIF  | Full Cone NAT            |
| EIM  | ADF  | Restricted Cone NAT      |
| EIM  | APDF | Port-Restricted Cone NAT |
| APDM | APDF | Symmetric NAT            |
| ADM  | ADF  | 较少见                      |
| ADM  | APDF | 较少见                      |

---

## 四、RFC 5780 的工程意义

RFC 3489 的老式 “NAT 四分类” 在工程上已经不够精确。
RFC 5780 的价值在于：

1. **解耦行为维度**

   * 映射（外向）
   * 过滤（内向）

2. **可测试、可推理**

   * STUN 可以分别探测 mapping / filtering
   * ICE 可以根据组合结果调整 candidate 策略

3. **解释现实世界的“混合 NAT”**

   * 现实设备经常是：

     * `EIM + APDF`
     * `ADM + ADF`
   * 而非教科书里的“纯对称 NAT”

---

## 五、一句话总结

> **RFC 5780 把 NAT 的本质拆成两个问题：**
>
> * *我出去时，端口怎么分？*（映射行为）
> * *别人回来时，谁能进？*（过滤行为）

如果你愿意，我可以进一步结合 **STUN 探测流程** 或 **WebRTC ICE 决策树**，把这些行为如何被自动识别、以及对连通性的具体影响展开讲清楚。


---

### 🧑‍💻 User

为什么说RFC 5780 (及其前身 RFC 4787) 定义的这些行为类型比旧的 "Cone NAT" (Full, Restricted, Port Restricted) 和 "Symmetric NAT" 分类法更精确和灵活？

---

### 🤖 Assistant

这个问题的核心在于：**旧的 NAT 分类法把“一组常见实现模式”当成了“基本物理定律”，而 RFC 4787 / 5780 则直接对 NAT 的真实行为进行了解剖和参数化描述**。因此，后者在工程上明显更精确、也更灵活。

下面从设计哲学、表达能力、工程适配性三个层面系统说明。

---

## 一、旧分类法的问题本质：**把“实现模板”当成“行为定义”**

传统的 NAT 四分类：

* Full Cone NAT
* Restricted Cone NAT
* Port-Restricted Cone NAT
* Symmetric NAT

来源主要是 RFC 3489 时代的经验总结，而不是严格规范。

它们的问题并不在“是否常见”，而在于：

### 1️⃣ 分类是**整体打包的（monolithic）**

每一类同时隐含了**多条行为假设**，例如：

> Full Cone NAT =
>
> * Endpoint-Independent Mapping
> * Endpoint-Independent Filtering

但**标准本身并没有明确指出这些是假设**，只是默认你“知道”。

一旦某个实现：

* 映射是 EIM
* 过滤是 APDF

那么它**既不像 Full Cone，也不像 Port-Restricted Cone**，在旧体系里根本无处安放。

---

### 2️⃣ 现实 NAT 实现迅速“越界”

厂商实现很快就出现：

* 为了安全，**过滤更严格**
* 为了复用端口，**映射保持不变**

例如：

```
Mapping:   Endpoint-Independent
Filtering: Address-and-Port-Dependent
```

这在旧分类中：

* 不是 Full Cone（过滤太严格）
* 也不是 Port-Restricted Cone（语义又不完全一致）
* 更不是 Symmetric NAT（映射根本不是对称）

👉 **旧分类体系直接失效**

---

## 二、RFC 4787 / 5780 的关键突破：**正交分解（Orthogonal Decomposition）**

RFC 4787 / 5780 的本质思想是：

> **NAT 行为不是“类型”，而是“策略组合”**

### 1️⃣ 明确拆分两个独立维度

| 维度               | 问题        |
| ---------------- | --------- |
| 映射行为 (Mapping)   | 出口端口是否复用？ |
| 过滤行为 (Filtering) | 谁能回包？     |

这两个维度在实现上：

* 逻辑独立
* 安全目标不同
* 性能权衡不同

旧分类法**强行把它们绑在一起**。

---

### 2️⃣ 行为是“定义”，而不是“标签”

RFC 5780 直接回答：

* 映射是否依赖：

  * 外部 IP？
  * 外部 Port？
* 过滤是否依赖：

  * 外部 IP？
  * 外部 Port？

这使得 NAT 行为可以被**精确描述为规则函数**：

```
Mapping:   f(IP_i, Port_i, IP_o, Port_o)
Filtering: g(IP_o, Port_o)
```

而不是模糊的“你这是个对称 NAT”。

---

## 三、表达能力的本质差异：**从枚举到参数化**

### 旧方法：枚举有限“原型”

旧体系本质上是：

> “我们见过几种 NAT，就给它们起几个名字。”

* 可扩展性为零
* 无法描述混合行为
* 无法解释厂商差异

---

### RFC 5780：参数化描述

RFC 5780 的方法是：

> “NAT 在这两个维度上分别做什么？”

因此它能自然表达：

* EIM + EIF
* EIM + APDF
* ADM + ADF
* APDM + EIF（理论上）
* ……

👉 **不需要事先假设“常见类型”**

---

## 四、对 STUN / ICE 的工程价值：旧分类做不到的事情

### 1️⃣ 旧分类无法驱动算法决策

在 ICE 中，真正重要的问题是：

* 外部端口是否稳定？
* 不同 peer 是否能复用映射？
* 是否需要 TURN？

“你是个 Port-Restricted Cone NAT”
**对算法来说几乎是废信息**。

---

### 2️⃣ RFC 5780 的行为可直接映射到策略

例如：

| 行为   | ICE 决策        |
| ---- | ------------- |
| EIM  | 允许共享候选端口      |
| APDM | 必须为每个 peer 打洞 |
| EIF  | 被动接收可行        |
| APDF | 必须先发包         |

ICE 不关心“你叫不叫 Symmetric NAT”，
只关心**行为是否满足前置条件**。

---

## 五、精确性的另一个来源：**可测试性**

### 旧分类：结果驱动（post hoc）

* 跑 STUN
* “看起来像 Symmetric NAT”
* 分类不稳定
* 不同实现得出不同结论

---

### RFC 5780：探测驱动（behavioral probing）

* 明确规定探测包组合
* 分别测试 mapping / filtering
* 每一步有可重复、可验证的判定标准

这使 NAT 行为变成：

* 可自动识别
* 可在运行时重新评估
* 可针对变化自适应

---

## 六、一个类比（为什么旧分类在概念上就不对）

| 领域  | 旧 NAT 分类     | RFC 5780      |
| --- | ------------ | ------------- |
| CPU | “这是个高性能 CPU” | 指令集、流水线、缓存层级  |
| 网络  | “这是个企业级交换机”  | 转发表、队列模型、调度算法 |
| OS  | “这是个多任务 OS”  | 调度策略、内存模型     |

旧 NAT 分类是**营销式标签**，
RFC 5780 是**行为学建模**。

---

## 七、一句话总结

> **RFC 4787 / 5780 并没有“发明新的 NAT 类型”，而是废除了“类型”这个概念。**

它们通过：

* 正交分解映射 / 过滤
* 用规则而非标签描述行为
* 让 STUN / ICE 能基于事实而非猜测工作

因此，在精度、灵活性和工程可用性上，都**系统性地碾压**了旧的 Cone / Symmetric 分类法。


---

### 🧑‍💻 User

为什么ipv6的mapping behaviour为Direct，filtering behavior为Endpoint-Independent？如果配置了入栈防火墙，情况又会发生哪些变化？

---

### 🤖 Assistant

这个问题问到了 **RFC 4787 / RFC 5780 行为模型在 IPv6 语境下的“退化形态”**，以及 **NAT 行为与防火墙语义的严格边界**。结论先行：

> **在“无 NAT 的原生 IPv6”中：**
>
> * Mapping behavior 退化为 **Direct**
> * Filtering behavior 天然是 **Endpoint-Independent**
>
> **一旦启用入栈防火墙（stateful 或 rule-based），**
>
> * mapping 行为仍然是 Direct
> * filtering 行为将被防火墙策略“重写”，甚至可以模拟 IPv4 NAT 的任意 filtering 语义

下面分层解释。

---

## 一、为什么 IPv6 的 mapping behavior 是 **Direct**

### 1. Mapping 行为本来是为 NAT 定义的

RFC 4787 / 5780 中的 **mapping behavior** 描述的是：

> 内部 (IP, Port) 如何被 **翻译（translate）** 成外部 (IP, Port)

而在原生 IPv6 中：

* 每个终端都有 **全局可路由地址**
* **不存在地址/端口重写**
* 不存在“内部 vs 外部地址空间”的边界

因此：

```
(IP_i, Port_i) → (IP_e, Port_e)
其中：IP_e = IP_i, Port_e = Port_i
```

这是一个 **恒等映射**。

---

### 2. RFC 对 IPv6 的术语：Direct Mapping

在 STUN / ICE 的行为模型中，IPv6 明确被视为一种 **特殊但合法的 mapping 行为**：

> **Direct**
> 不存在 NAT translation，端点直接暴露其真实地址与端口。

这不是 “Endpoint-Independent Mapping”，
而是 **“根本没有映射函数”**。

---

### 3. 工程后果

* 外部端口 **100% 稳定**
* 不存在端口冲突 / 复用问题
* ICE 中：

  * IPv6 host candidate 直接可用
  * 无需 STUN 探测 mapping

---

## 二、为什么 IPv6 的 filtering behavior 是 **Endpoint-Independent**

### 1. 没有 NAT = 没有隐式过滤

在 IPv4 NAT 中：

* filtering 行为来自 NAT 的 **状态表**
* 回包是否允许，取决于是否存在匹配映射

而在 **无防火墙的 IPv6** 中：

* 没有 NAT state
* 路由器只是转发
* 不做源地址/端口校验

因此：

> 只要有路由，**任何外部端点都可以向你发包**

这在 RFC 5780 语义下，等价于：

> **Endpoint-Independent Filtering**

---

### 2. 这是 IPv6 的设计初衷

IPv6 的核心理念之一就是：

* 恢复 **端到端连通性**
* 安全由 **IPsec / 防火墙** 负责
* 而不是靠 NAT “顺便挡住”

因此，**裸 IPv6 网络 ≠ 默认受限网络**。

---

## 三、引入入栈防火墙后：发生了什么变化？

这是最关键、也是最容易混淆的部分。

### 结论先行

| 维度                 | 是否变化 | 原因      |
| ------------------ | ---- | ------- |
| Mapping behavior   | ❌ 不变 | 防火墙不改地址 |
| Filtering behavior | ✅ 改变 | 防火墙控制入栈 |

---

## 四、防火墙如何“覆盖” filtering behavior

### 1. Stateless 防火墙（ACL）

如果配置规则：

```
只允许来自 2001:db8::/32 的 UDP
```

那么 filtering 行为变为：

> **Address-Dependent Filtering**

---

### 2. Stateful 防火墙（最常见）

典型规则：

```
允许 ESTABLISHED / RELATED
拒绝 unsolicited inbound
```

其行为在 UDP 上等价于：

> **Address-and-Port-Dependent Filtering**

即：

* 必须先向 (IP_o, Port_o) 发包
* 才允许对方回包

这与 **IPv4 Port-Restricted Cone NAT 的 filtering 行为完全一致**。

---

### 3. 关键区别：这不是 NAT

尽管 **filtering 行为相同**：

| IPv4 NAT          | IPv6 防火墙          |
| ----------------- | ----------------- |
| filtering 依附于映射状态 | filtering 依附于连接状态 |
| 映射和过滤强耦合          | 映射不存在             |
| 地址被隐藏             | 地址仍是全局可见          |

---

## 五、从 RFC 5780 视角看：IPv6 + 防火墙的“行为组合”

可以用同一套术语描述：

### 1. 无防火墙 IPv6

```
Mapping:   Direct
Filtering: Endpoint-Independent
```

---

### 2. IPv6 + Stateful Firewall（默认家庭路由）

```
Mapping:   Direct
Filtering: Address-and-Port-Dependent
```

---

### 3. IPv6 + 严格 ACL

```
Mapping:   Direct
Filtering: Address-Dependent 或更严格
```

---

## 六、为什么 RFC 5780 体系在 IPv6 下依然成立

这正好反证了你前一个问题的结论：

> **RFC 5780 描述的是“行为语义”，不是“NAT 技术”**

* IPv6 可以被自然纳入该模型
* 防火墙行为可以无缝映射
* ICE / WebRTC 不需要“IPv6 特判”

---

## 七、一句话总结

> **IPv6 的 Direct + Endpoint-Independent 并不是“放开”，而是“不再滥用 NAT 作为安全机制”。**

一旦你显式启用防火墙：

* mapping 依旧是 Direct
* filtering 则完全由你定义
* 甚至可以精确复刻任何 IPv4 NAT 的 filtering 行为

如果你愿意，我可以进一步从 **ICE candidate 优先级、Happy Eyeballs、或 IPv6-only + NAT64** 的角度，把这些行为如何影响实际连通性讲透。
