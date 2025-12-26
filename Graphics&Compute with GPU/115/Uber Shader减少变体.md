# Uber Shader减少变体

### 🧑‍💻 User

uber shader为什么可以减少变体数量？

---

### 🤖 Assistant

Uber Shader 能减少变体数量，本质原因在于：**它把“编译期分支（compile-time specialization）”尽可能推迟为“运行期分支（runtime selection）”，用一份可参数化的通用着色器覆盖大量原本需要独立编译的组合状态**。下面从编译模型、状态空间和工程约束三个层面系统解释。

---

## 1. 变体爆炸的根源：编译期组合态

在传统 shader 体系中，变体数量来源于**编译期不可变的离散开关**：

* `#define USE_NORMAL_MAP`
* `#define USE_SHADOW`
* `#define USE_CLEARCOAT`
* `#define NUM_LIGHTS 4`
* `#define SKINNING 1`

假设有 *N* 个二值开关，理论上就有 **2ⁿ 个 shader 变体**。
在 DX12 / Vulkan 中，这些变体通常还会进一步绑定为不同的 **PSO**：

```
(Shader bytecode, Root Signature, Blend, Raster, Depth, ...)
```

→ **PSO 数量 = Shader 变体 × 固定管线状态组合**

这就是现代引擎里 PSO / shader 数量失控的根本原因。

---

## 2. Uber Shader 的核心思想

Uber Shader 并不神秘，核心只有一句话：

> **用一个“最大功能集”的 shader + 参数化控制路径，替代大量裁剪后的专用 shader。**

### 2.1 从编译期裁剪 → 运行期选择

| 传统做法                    | Uber Shader                  |
| ----------------------- | ---------------------------- |
| `#ifdef USE_NORMAL_MAP` | `if (material.hasNormalMap)` |
| `#define NUM_LIGHTS 4`  | loop + uniform 控制            |
| 不同材质 = 不同 shader        | 不同材质 = 不同参数                  |

这样做的直接结果是：

* **编译期组合数锐减**
* **shader bytecode 数量减少**
* **PSO 数量随之减少**

---

## 3. 为什么这在现代 GPU 上是“可接受的”

Uber Shader 并不是白拿好处，它付出的代价是**运行期分支与指令膨胀**。之所以仍然成立，依赖以下现实：

### 3.1 GPU 对 uniform 分支非常友好

* `if (uniform)`
  → 在一个 wave / warp 内分支一致
  → 实际执行成本接近“免费”

典型 Uber Shader 中的分支条件大多来自：

* Material constant buffer
* Draw call 级别 uniform
* Push constant

而不是 per-pixel 随机条件。

---

### 3.2 指令 Cache vs PSO 切换成本

在 DX12 / Vulkan 下：

* **PSO 切换 = 高成本操作**
* **Shader cache miss = pipeline stall**
* **大量小 shader = instruction cache 抖动**

相比之下：

* 一份稍大的 Uber Shader
* 更容易被 instruction cache 命中
* 渲染队列更稳定

---

### 3.3 编译和加载时间是现实瓶颈

即使运行期能接受，**工程上也必须接受**：

* 编译 1 万个 shader 变体：不可控
* 首帧或关卡加载时 PSO 预热：灾难
* 热更新 / 跨平台：维护成本极高

Uber Shader 用**时间换空间**：

* 多一点 runtime ALU
* 换来可控的编译、加载、缓存规模

---

## 4. Uber Shader 减少变体的“精确机制”

### 4.1 变体维度被“压扁”

原本的多维离散空间：

```
[NormalMap] × [Shadow] × [ClearCoat] × [IBL] × [Skinning]
```

被压缩成：

```
1 个 shader × 参数表
```

变体数量从指数级 → 常数级。

---

### 4.2 PSO 数量同步下降

因为 shader bytecode 统一：

* Root Signature 不变
* Shader stage 不变

PSO 只在**真正影响固定管线**的状态上拆分，例如：

* Alpha blend vs Opaque
* Depth write on/off
* MSAA on/off

而不是为“有没有法线贴图”生成 PSO。

---

## 5. Uber Shader 的边界与妥协

Uber Shader 并不是万能的，以下情况通常仍然需要拆分：

### 5.1 真正结构性差异

* 顶点输入布局不同（skinned vs static）
* Tessellation / Mesh Shader 有无
* Ray tracing vs raster

这些差异**无法仅靠 if 控制**。

---

### 5.2 高度发散的 per-pixel 分支

例如：

* 屏幕空间随机效果
* 稀疏 feature mask
* Tile 内差异极大的材质

这种情况下 Uber Shader 会引发严重 warp divergence。

---

## 6. 现代引擎的现实策略

实际工程中几乎没人“全 Uber”：

* **少量 Uber Shader + 有限子变体**
* 编译期只保留“结构级差异”
* 功能级差异尽量 runtime 化

例如：

```
BaseLitUber.hlsl
 ├─ Opaque
 ├─ Masked
 └─ Transparent
```

而不是：

```
Lit + Normal + Shadow + ClearCoat + ...
```

---

## 7. 一句话总结

**Uber Shader 能减少变体数量，是因为它用运行期参数与统一代码路径，消灭了编译期组合态，从指数级的 shader / PSO 空间，压缩为可控的常数级集合。**

它牺牲的是：

* 少量 ALU
* 更复杂的 shader 逻辑

它换来的是：

* 可控的编译规模
* 稳定的 PSO 管理
* 可接受的加载与首帧时间

如果你愿意，下一步我可以从 **DX12 PSO 角度** 或 **Wave/warp 执行模型**，进一步把 Uber Shader 的性能权衡拆解到指令级。
