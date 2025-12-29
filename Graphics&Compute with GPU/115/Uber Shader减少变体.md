# Uber Shader减少变体

### 🧑‍💻 User

uber shader为什么可以减少变体数量？

---

### 🤖 Assistant

从工程与编译体系的角度看，**Uber Shader 能减少变体数量的根本原因在于：它把“编译期的组合爆炸”转移为“运行期的条件选择”，用一次编译覆盖原本需要多次编译的功能组合**。下面从机制、数学本质、以及代价三个层面系统说明。

---

## 1. 传统 Shader Variant 爆炸是如何产生的

在传统做法中，功能开关通常通过 **编译期宏（`#define` / keyword）** 控制：

```c
#define USE_NORMAL_MAP
#define USE_SHADOW
#define USE_CLEARCOAT
```

每一个布尔特性都会引入一个 **乘法维度**：

* N 个二值开关
* 变体数 = 2^N

例如：

| 功能        | 状态       |
| --------- | -------- |
| 法线贴图      | on / off |
| 阴影        | on / off |
| 透明        | on / off |
| IBL       | on / off |
| ClearCoat | on / off |

→ **2⁵ = 32 个 shader 变体**

再叠加：

* 光源类型（dir / point / spot）
* 渲染路径（forward / deferred）
* 平台（pc / mobile / console）

最终在 DX12 / Vulkan 下直接表现为：

* PSO 数量爆炸
* 编译时间、磁盘占用、运行期缓存压力暴涨

---

## 2. Uber Shader 的核心思想

**Uber Shader 的定义：**

> 一个覆盖“所有功能子集”的单一（或极少数）shader，通过运行期条件、mask、动态分支来选择实际执行路径。

### 对比一句话总结

| 传统 Variant    | Uber Shader      |
| ------------- | ---------------- |
| 编译期选择路径       | 运行期选择路径          |
| 每个组合一个 shader | 一个 shader 覆盖所有组合 |
| PSO 数量随功能指数增长 | PSO 数量近似线性增长     |

---

## 3. Uber Shader 为什么能减少变体数量（关键点）

### 3.1 把「编译期布尔空间」压缩为「运行期数据」

传统方式：

```c
#ifdef USE_NORMAL_MAP
    normal = SampleNormalMap();
#else
    normal = vertexNormal;
#endif
```

Uber Shader：

```c
if (material.useNormalMap) {
    normal = SampleNormalMap();
} else {
    normal = vertexNormal;
}
```

本质变化：

| 层面  | 变化                |
| --- | ----------------- |
| 编译  | 不再为每个开关生成新 shader |
| PSO | 一个 PSO 可服务多种材质    |
| 材质  | 状态从“宏”变成“参数”      |

→ **N 个宏不再产生 2^N 个编译结果**

---

### 3.2 变体数从「乘法」退化为「加法」

传统：

```
变体数 = ∏ 每个 feature 的状态数
```

Uber Shader：

```
变体数 ≈ 少量结构性差异
```

通常只保留真正**无法运行期切换**的维度：

* Forward vs Deferred
* Skinned vs Static
* Shadow Pass vs Base Pass

例如：

```
3 渲染路径 × 2 几何类型 × 2 Pass = 12 个 Uber Shader
```

而不是成百上千个细粒度组合。

---

### 3.3 DX12 / Vulkan 下 PSO 维度被显著压缩

在现代 API 中：

* PSO = Shader + RootSignature + RenderState
* **PSO 是运行期高成本对象**

Uber Shader 的直接收益是：

* 同一个 PSO

  * 绑定不同材质参数
  * 走不同代码路径
* **PSO 缓存命中率大幅提升**
* 离线 PSO 收集复杂度下降

这也是为什么 **PSO 数量爆炸是 DX12 时代才成为“新问题”**，而 Uber Shader 变成“工程必选项”。

---

## 4. GPU 执行层面：为什么运行期分支不再那么可怕

很多人直觉上会担心：

> “if 多了，GPU 会不会慢？”

在现代 GPU 上，情况已经明显不同：

### 4.1 Uniform / Coherent Branch

Uber Shader 中的分支大多是：

* **per-draw / per-material 常量**
* 同一 wave / warp 内条件一致

例如：

```c
if (material.flags & USE_CLEARCOAT)
```

→ GPU 能整波执行同一路径
→ **几乎等价于静态分支**

---

### 4.2 少量“死代码”比变体爆炸更便宜

现实工程中：

* 多几个 ALU
* 多一次 texture fetch

通常远比：

* 多一个 PSO
* 多一次 shader bind
* 多一次 cache miss

要便宜得多。

因此 Uber Shader 是一种 **以算力换系统复杂度** 的典型工程取舍。

---

## 5. Uber Shader 并不是“零代价”

为了完整，必须明确它的边界。

### 5.1 潜在代价

| 代价          | 说明                      |
| ----------- | ----------------------- |
| Shader 体积变大 | 指令缓存压力                  |
| 寄存器占用上升     | 可能降低 occupancy          |
| 分支不一致       | 在极端情况下有 warp divergence |

### 5.2 工程上的常见折中

实际引擎通常采用 **Hybrid 策略**：

* **大粒度：Uber Shader**
* **小粒度：关键路径保留少量变体**

例如：

* 移动端：更激进的 Uber
* 桌面端：热点路径拆分专用 shader

---

## 6. 总结一句话版

> **Uber Shader 之所以能减少变体数量，是因为它把原本在“编译期做笛卡尔积枚举”的问题，转化为“运行期用参数选择路径”的问题，从指数级复杂度退化为线性复杂度。**

如果你愿意，我可以进一步结合：

* Unity URP / HDRP 的 keyword → Uber 演进
* UE5 的 Material Permutation 设计
* DX12 PSO 收集与 Uber Shader 的直接关系

从具体引擎实现角度继续拆。
