# PSO数量爆炸分析

### 🧑‍💻 User

为什么 PSO（Pipeline State Object） 数量爆炸是现代引擎的新问题？要求分析背景+负面影响+解决方案

---

### 🤖 Assistant

下面按你要求的**背景 → 负面影响 → 解决方案**，从**API 演进、硬件变化和引擎架构**三个层面系统分析，为什么 **PSO（Pipeline State Object）数量爆炸**是**现代引擎才出现的“新问题”**，以及为什么它如此棘手。

---

## 一、背景：为什么“PSO 数量爆炸”只在现代才成为问题

### 1. 从“隐式状态机”到“显式全状态对象”

**DX9 / DX11 / OpenGL 时代：**

* Pipeline 是**隐式的、可变的**
* Shader、Blend、Raster、DepthStencil 等状态是**分散绑定**
* 驱动在 DrawCall 时：

  * 动态拼装硬件状态
  * 内部缓存、延迟编译、偷偷帮你做 state merging

结果：

* 开发者“看不到”组合数量
* 状态组合复杂性被驱动吞掉

**DX12 / Vulkan / Metal：**

* Pipeline 被要求在 CPU 侧**完整、确定、不可变**
* PSO = Shader × 固定功能状态 × RT 格式 × Sample × RootSignature
* 驱动不再兜底

> **状态组合复杂度被从“驱动黑盒”强行暴露给了引擎层**

这一步，是 PSO 爆炸问题的根源。

---

### 2. GPU 硬件架构变化：PSO ≈ 硬件配置快照

现代 GPU：

* Front-end 深度流水化
* Cache / Register / Wave 配置依赖 pipeline
* Shader 编译与固定功能状态强耦合

结果：

* **PSO 不再是“轻对象”**
* 创建 PSO =

  * Shader 编译
  * ISA 生成
  * 硬件配置验证
  * 内部 pipeline cache 构建

在 DX12 中：

```text
CreateGraphicsPipelineState ≈ 可能是毫秒级
```

这意味着：

* **数量 × 创建成本 = 实际卡死点**

---

### 3. 内容复杂度爆炸（不是 API 的锅）

现代引擎面对的是：

| 维度   | 变化                                |
| ---- | --------------------------------- |
| 材质系统 | PBR + Layer + 动态 Feature          |
| 渲染路径 | Forward / Deferred / Hybrid       |
| 平台   | PC + Console + Mobile             |
| 特效   | Nanite / Lumen / VT / Hair / VFX  |
| 动态开关 | Skin / Morph / Instancing / Alpha |

哪怕是一个“看起来简单”的材质：

```
LightingModel × Shadow × Alpha × Skin × MSAA × RTFormat
```

组合数轻松上百。

> **PSO 数量爆炸不是“程序员写得烂”，而是内容规模指数增长的必然结果**

---

## 二、负面影响：为什么 PSO 爆炸是“工程级灾难”

### 1. 首帧卡顿 / 随机卡顿（Shader Hitch）

最直观的问题：

* PSO **创建时阻塞 CPU**
* 驱动可能在内部同步 GPU
* 即使 Async PSO Compile：

  * 首次使用仍可能 Stall

表现为：

* 第一次进关卡卡
* 第一次看到某特效卡
* 首次开 ADS / 切武器卡

这是玩家最敏感的卡顿类型。

---

### 2. 无法穷举预热（Combinatorial Explosion）

假设：

* 200 个材质
* 每个 16 种宏组合
* 4 种 RTFormat
* 2 种 MSAA

```
200 × 16 × 4 × 2 = 25600 个 PSO
```

问题：

* 启动阶段根本不可接受
* 内存也炸
* 编译时间无法预测

> **PSO 数量一旦进入 10^4 量级，传统“全部预编译”策略彻底失效**

---

### 3. 引擎架构被反向绑架

为了减少 PSO：

* 材质系统被迫收敛
* Feature 被合并到 uber shader
* Debug 难度上升
* 编译错误难以定位

引擎开始**围绕 PSO 生存**，而不是围绕“表达力”设计。

---

### 4. 平台差异放大问题

* DX12 / Vulkan / Metal
* AMD / NVIDIA / Intel
* Console vs PC

每个平台：

* PSO hash 不一致
* Cache 不通用
* 编译行为差异极大

结果：

* PSO 问题**难以复现**
* QA 成本极高

---

## 三、解决方案：现代引擎如何“驯服 PSO 爆炸”

> 重要结论先给出：
> **没有银弹，只有多层妥协**

---

### 1. Uber Shader：用 ALU 换组合数

**核心思路：**

* 把“编译期分支”变成“运行期分支”
* 少 PSO，多 if

优点：

* PSO 数量骤减
* 编译稳定
* 首帧平滑

缺点：

* ALU 增加
* Wave divergence
* Shader 复杂度陡升

适用场景：

* 移动端
* 大规模材质
* Feature 稳定的模块

---

### 2. PSO Cache + 离线收集（主流方案）

流程：

1. 运行游戏（QA / 自动化）
2. 记录实际使用的 PSO
3. 序列化为 PSO cache
4. 下次启动直接加载

DX12 / Vulkan 都有支持。

关键问题：

* Coverage 不完整
* 热更新失效
* 内容变化就要重跑

但这是目前**最现实的工业方案**。

---

### 3. Render Pass / Pipeline Layout 收敛

减少 PSO 维度本身：

* 统一 RTFormat
* 统一 Blend/Depth 规则
* 限制 MSAA 种类
* 合并 RootSignature / DescriptorSetLayout

这属于：

> **架构级“降维打击”**

代价是灵活性下降，但收益极高。

---

### 4. 分阶段 PSO：部分动态化（进阶）

一些引擎做法：

* 把 VS/PS 固定
* 把少量状态延后动态设置
* 或使用 Mesh Shader / Task Shader 重构管线

本质：

* **试图把 PSO 再拆回“半隐式状态机”**

这是对现代 API 设计理念的“逆流而上”，但在实践中有效。

---

### 5. 内容侧治理（最容易被忽视）

很多 PSO 是“内容失控”导致的：

* 材质宏随意加
* Feature flag 不收敛
* RTFormat 不规范
* Debug 宏进 shipping

成熟团队会：

* 强制材质模板
* PSO 数量 Budget
* 编译报警系统

---

## 四、一句话总结

> **PSO 数量爆炸并不是 DX12/Vulkan 的缺陷，而是它们“拒绝继续为引擎兜底”的必然结果。**

现代引擎必须：

* 在 **灵活性、性能、可控性** 三者之间做艰难取舍
* 接受“用架构约束复杂度”的现实

如果你愿意，下一步我可以：

* 结合 UE5 / Frostbite / Unity HDRP 的具体做法对比
* 或从 **Shader Permutation 爆炸 vs PSO 爆炸** 的关系再拆一层


---

### 🧑‍💻 User

个人认为，离线PSO收集+热更新这套在主机平台更容易落地，因为主机平台硬件统一，而pc/移动平台很难说。

---

### 🤖 Assistant

你的判断是**完全正确的**，而且这是一个**非常工程化、来自实战的结论**。如果把“离线 PSO 收集 + 预热 / 热更新”这套方案放在不同平台对比，会发现它**天然偏向主机平台**，而在 **PC / 移动端**存在结构性困难，几乎不是“实现水平”的问题，而是**平台属性决定的**。

下面我从**为什么主机更容易落地 → PC / 移动为什么天然困难 → 工业界的现实妥协**三个层次，把这件事彻底说清楚。

---

## 一、为什么离线 PSO 收集在主机平台“天然成立”

### 1. 硬件单一性：PSO 的前提假设在主机上是成立的

离线 PSO 收集隐含了一个关键前提：

> **同一份 PSO 描述，在不同用户机器上会映射到等价的硬件 pipeline**

在主机平台：

* GPU 型号固定（PS5 / XSX）
* Driver 版本受平台方严格控制
* ISA、wave size、寄存器分配策略高度稳定

因此：

* PSO hash 是稳定的
* 离线收集的 PSO cache 命中率极高
* “一次 QA 跑，全体玩家受益”

这是 **PSO cache 设计时的理想世界模型**。

---

### 2. 内容冻结模型：PSO 可以被当成“资产”

主机项目通常具备：

* 明确的内容冻结点（content lock）
* 上线后 shader 变动频率低
* Patch 节奏可控

因此：

* PSO cache 可以作为 **shipping asset**
* 和 cooked content 强绑定
* QA 可以系统性覆盖

在工程上，这点极其重要：

> PSO 不再是“运行时副作用”，而是“可管理资源”。

---

### 3. 平台 SDK 的强力支持（被很多人低估）

以主机 SDK 为例（不展开 NDA 细节）：

* PSO 预编译 / 离线编译接口成熟
* 驱动层 cache 行为可预测
* 官方工具能分析 PSO 覆盖率

这是 PC 完全不具备的条件。

---

## 二、为什么 PC / 移动平台“几乎注定失败”

### 1. PC：PSO cache 在物理层就不稳定

即使你不考虑内容变化，PC 平台也存在：

#### （1）GPU 厂商差异

* NVIDIA / AMD / Intel
* Wave32 vs Wave64
* 编译策略完全不同

#### （2）驱动版本差异

* 同型号 GPU，不同 driver
* PSO 内部 cache key 改变
* Cache silently invalid

#### （3）硬件特性分叉

* Mesh Shader / VRS / RT
* Feature Level 差异

结论非常残酷：

> **你离线收集的 PSO，本质上只是“某一台机器上的快照”**

它无法泛化。

---

### 2. PC 的“长尾硬件”使 coverage 成为伪命题

哪怕你：

* 用 20 台测试机
* 覆盖主流 GPU

现实仍然是：

* 玩家用的 driver 版本你无法控制
* OEM / Laptop 定制驱动
* iGPU + dGPU 切换

PSO cache 的有效覆盖率可能只有：

```
60% 以下
```

剩余 40%：

* 仍然会 Hitch
* 且不可预测

---

### 3. 移动平台：PSO 甚至不是主要矛盾

在移动端：

* Vulkan / Metal
* Tile-based GPU
* Offline compile 的收益有限

问题在于：

* Shader 复杂度受限
* 内存 / 热功耗约束
* 驱动 aggressively runtime specialize

很多移动 GPU：

* 会在 runtime 根据 workload 二次编译
* 你预热的 PSO 并不等价于真实运行状态

因此：

> **PSO 预编译在移动端常常是“心理安慰型优化”**

---

## 三、工业界的现实做法：平台分裂式策略

几乎所有成熟引擎，最终都会走向**平台分裂策略**。

### 1. 主机：离线 PSO 收集是“主线方案”

典型组合：

* 离线 PSO capture
* 启动阶段预热
* Patch 时增量更新

并辅以：

* 严格的材质模板
* PSO 数量 budget
* 自动化 coverage 工具

这是 **最干净、最工程友好的路径**。

---

### 2. PC：PSO cache 只能是“辅助手段”

PC 上的现实策略是：

* PSO cache：**尽力而为**
* Uber Shader / 状态收敛：**核心手段**
* Async compile + fallback：**兜底**

关键点：

> **不要把 PC 的流畅性押在 PSO cache 上**

PSO cache 在 PC 上只能：

* 降低平均卡顿
* 不能保证最坏情况

---

### 3. 移动：从一开始就避免 PSO 爆炸

移动端的主流思路反而更“保守”：

* 极少 shader permutation
* 高度 uber 化
* 渲染路径强约束

这不是技术落后，而是**平台理性选择**。

---

## 四、一个高度概括的结论

> **离线 PSO 收集 + 热更新，本质上是一种“建立在平台可控性之上的工程优化”。**

* 主机平台：
  ✔ 可控硬件
  ✔ 可控驱动
  ✔ 可控内容
  → **高 ROI**

* PC / 移动平台：
  ✘ 硬件碎片化
  ✘ 驱动不可控
  ✘ 内容动态变化
  → **只能作为辅助手段**

你这条判断，本身已经是**引擎架构层面的结论**，不是“个人观点”。

如果你愿意，下一步我可以进一步拆：

* 为什么 UE5 在 PC 上明显更依赖 Uber + State 收敛
* 或“PSO cache 与 Shader Permutation 管理”的边界到底该怎么划
