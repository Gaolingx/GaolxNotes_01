# 实时GI中的Final Gather

### 🧑‍💻 User

实时全局光照中的finalgather是什么？

---

### 🤖 Assistant

在**实时全局光照（Real-time Global Illumination, RTGI）**中，**Final Gather（最终聚集）**并不是一个单一、固定实现的算法名，而是一个**阶段性概念**：

> 它指的是在已经获得“间接光的粗略表示”之后，在**着色点（shading point）处对入射间接辐射进行一次局部积分/重建**，从而得到可用于最终像素着色的间接光照结果。

下面从**物理含义、在实时管线中的位置、与离线渲染的区别、以及现代实时GI中的典型实现**四个层面系统说明。

---

## 1. 物理与渲染意义（它在算什么）

从渲染方程角度看，Final Gather 本质是在近似计算：

[
L_o(x,\omega_o) = \int_{\Omega} L_i(x,\omega_i), f_r(x,\omega_i,\omega_o), (\omega_i \cdot n), d\omega_i
]

在实时 GI 中：

* **(L_i)**（入射间接光）
  并不是通过无限反弹精确算出来的
  → 而是来自某种**预计算 / 屏幕空间 / 体素 / 探针 / 光照缓存**的近似表示

* **Final Gather 的任务**就是：
  **在 shading point 处，对这些近似的入射辐射进行采样、过滤、方向加权、BRDF 调制**，输出稳定、连续、可用的间接光结果。

一句话总结：

> Final Gather = “把已经算好的（但很粗糙的）间接光信息，正确地‘拿来用’。”

---

## 2. 在实时 GI 管线中的位置

一个典型的实时 GI 流程可以抽象为三层：

```
① 间接光的生成（粗）
   - 屏幕空间追踪
   - 探针更新
   - 体素注入
   - SDF / BVH trace
   - 历史缓存

② 间接光的存储（中间表示）
   - Irradiance Volume / Probe Grid
   - Radiance Cache
   - Surfels
   - Voxel mipmap
   - DDGI probes

③ Final Gather（你问的这个）
   - 在像素/表面点上
   - 采样 / 插值 / 方向投影
   - 结合法线、粗糙度、BRDF
   - 输出最终间接光
```

Final Gather **永远在第③步**。

---

## 3. 和离线渲染中的 Final Gather 的关系

### 离线渲染（Mental Ray / V-Ray 时代）

* Final Gather =
  在每个 shading point 发射大量半球射线
* 每条射线打到表面
* 查询 irradiance cache
* 本质是 **低频间接光的蒙特卡洛积分**

### 实时渲染

* 不可能每像素发几十/上百条射线
* 于是：

  * 射线数极少（甚至 0）
  * 更多依赖缓存、插值、历史

但**思想完全一致**：

| 离线                  | 实时                  |
| ------------------- | ------------------- |
| 半球射线                | 探针/体素/屏幕空间采样        |
| 精确遮挡                | 近似可见性               |
| 高质量、慢               | 稳定、快                |
| Final Gather = 减少噪声 | Final Gather = 重建信号 |

---

## 4. 现代实时 GI 中 Final Gather 的典型形式

### 4.1 探针式 GI（DDGI / UE Lumen Surface Cache）

Final Gather =

* 根据 shading point 的位置
* 查找周围 8 / 27 个 irradiance probes
* 对 probe 中的 SH / Octahedral radiance：

  * 做三线性插值
  * 按法线方向投影
  * 用 visibility / bent normal 修正
* 再乘 BRDF

**这是最经典、最“正统”的 Final Gather**

---

### 4.2 体素 GI（VXGI / SVOGI）

Final Gather =

* 沿法线方向对 voxel radiance mipmap 采样
* 多方向 cone tracing（锥形追踪）
* 各 cone 的能量加权累积

你会发现：

> cone tracing 本身就相当于把 “trace + final gather” 合并了
> 但从语义上，它仍然是 Final Gather 阶段。

---

### 4.3 屏幕空间 GI（SSGI）

Final Gather =

* 在 shading point 周围：

  * 屏幕空间射线 / taps
* 获取邻域的 radiance
* 用：

  * 法线相似度
  * 深度差
  * 粗糙度
* 做滤波和加权

这里 Final Gather 更像是：

> “空间 + 法线 + 材质域上的非线性滤波重建”

---

### 4.4 硬件光追 GI（RTGI）

即使有 RTX：

* 一般也只做：

  * 1–2 条 ray / pixel
* 得到的是：

  * 噪声极大的 radiance 样本

Final Gather =

* 对 ray hit 的 radiance：

  * 空间滤波
  * 时间累积（TAA）
  * 法线/深度权重
* 重建一个稳定的间接光

**这里 Final Gather ≈ 去噪 + 重建**

---

## 5. 为什么 Final Gather 非常关键

如果没有 Final Gather：

* GI 会：

  * 噪声大
  * 闪烁
  * 漏光
  * 与材质不匹配
* 间接光只是“存在”，但**不可用**

Final Gather 决定了：

* GI 的稳定性
* 能否与 BRDF 正确耦合
* 低频是否干净
* 高粗糙度表面是否自然

在现代引擎里：

> **GI 的“好不好看”，70% 取决于 Final Gather，而不是 trace 本身。**

---

## 6. 一句话精确定义

> 在实时全局光照中，**Final Gather 是在像素/表面点处，对来自缓存、探针、体素或射线的间接辐射进行采样、插值、可见性修正与 BRDF 加权，从而生成最终可用间接光照的阶段**。

如果你愿意，我可以：

* 结合 **UE5 Lumen** 的具体 pass 拆解 Final Gather
* 或对比 **DDGI vs SSGI vs RTGI** 的 Final Gather 数学形式
* 或从 **频域 / 信号重建** 的角度解释为什么 Final Gather 本质是低频重建
