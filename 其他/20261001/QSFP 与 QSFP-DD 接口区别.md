# QSFP 与 QSFP-DD 接口区别

## 🧑‍💻 User

qsfp qsfp-dd接口区别

---
## 😀 默认助手 (7)

# QSFP 与 QSFP-DD 接口区别

一句话概括：**QSFP-DD 是把 QSFP 从「4 通道」翻倍成「8 通道」的 400G/800G 版本**（DD = **D**ouble **D**ensity，双密度），外形宽度和深度与 QSFP 保持一致，靠增加第二排触点把端口密度和带宽翻倍，同时**向下兼容 QSFP**。

## 核心对比

| 对比项 | QSFP 家族（QSFP+/QSFP28/QSFP56） | **QSFP-DD** |
|---|---|---|
| 全称 | Quad Small Form-factor Pluggable | QSFP Double Density |
| MSA 发布 | 2006 起，陆续演进 | 2016 |
| 电气通道数 | **4 lane** | **8 lane** |
| 连接器触点 | **38 pin，单排** | **76 pin，双排**（下排与 QSFP 对齐） |
| 每通道速率 | 10G / 25G NRZ，50G PAM4 | 25G NRZ，50G PAM4；**DD800 支持 100G PAM4** |
| 典型总速率 | 40G / 100G / 200G | **200G / 400G / 800G** |
| 调制方式 | 以 NRZ 为主（QSFP56 起 PAM4） | NRZ + **PAM4** |
| 模块尺寸 | 18.35 mm（宽）× 72 mm（深） | **同宽同深，高度略增**（容第二排触点） |
| 功耗上限 | 约 3.5–5 W | 约 7 W → 14 W+ |
| 管理接口 | SFF-8636（I2C） | **CMIS**（兼容并扩展 SFF-8636） |
| 端口互插 | 插不进 QSFP-DD 端口 | 端口**可插 QSFP** 模块（向下兼容） |

## 关键区别详解

### 1. 通道数：4 → 8（这是本质差异）
- **QSFP**：4 个收发通道，$4 \times 25\text{G} = 100\text{G}$（QSFP28）、$4 \times 50\text{G} = 200\text{G}$（QSFP56）。
- **QSFP-DD**：8 个收发通道，$8 \times 25\text{G} = 200\text{G}$、$8 \times 50\text{G} = 400\text{G}$、$8 \times 100\text{G} = 800\text{G}$。

### 2. 连接器：单排 38 pin → 双排 76 pin
QSFP-DD 在同样 18.35 mm 的宽度内把触点排成**上下两排**。**下排的 38 个触点与 QSFP 完全对齐**，这就是它能插 QSFP 模块的物理基础。

<details>
<summary>为什么叫「双密度」（点开看原理）</summary>

- 传统 QSFP28 端口只用到 4 条差分对，面板上要做到 400G 就得用 4 个端口。
- QSFP-DD 用同一个 18.35 mm × 72 mm 的笼子放进 8 条差分对，相当于**单位面积端口密度翻倍**，一个端口就能顶 400G。
- 代价是模块更高、发热更大（8 通道同时工作），对散热和整机风道要求更高。

</details>

### 3. 向下兼容是「单向」的
- ✅ **QSFP-DD 端口 ← 插 QSFP 模块**：物理可插，速率需主机端口支持（绝大多数 400G 端口都支持 100G/40G）。
- ❌ **QSFP 端口 → 插 QSFP-DD 模块**：插不进去（模块更高、触点多），也无法协商 400G。

### 4. 速率与调制演进

| 标准 | 通道 | 速率/通道 | 总速率 | 调制 |
|---|---|---|---|---|
| QSFP+ | 4 | 10G | 40G | NRZ |
| QSFP28 | 4 | 25G | 100G | NRZ |
| QSFP56 | 4 | 50G | 200G | PAM4 |
| QSFP112 | 4 | 100G | 400G | PAM4 |
| **QSFP-DD** | 8 | 25G / 50G | 200G / 400G | NRZ / PAM4 |
| **QSFP-DD800** | 8 | 100G | 800G | PAM4 |

> 注意别混淆：`QSFP112` 是**单排连接器**、靠每通道 100G 提速率；`QSFP-DD` 是**双排连接器**、靠通道数翻倍提速率。两者都叫「QSFP」，但路线不同。

### 5. 分线（Breakout）能力
- QSFP28 100G → 4×25G 或 2×50G
- QSFP-DD 400G → **2×200G / 4×100G / 8×50G**

### 6. 管理接口
- QSFP 用 **SFF-8636** 定义的 I2C 寄存器地图。
- QSFP-DD 采用 **CMIS（Common Management Interface Specification）**，是 SFF-8636 的扩展超集，支持 8 通道、更多告警和更复杂的能力上报。

## 与该领域其他「双密度」封装的关系

| 封装 | 通道 | 外形 | 定位 |
|---|---|---|---|
| QSFP-DD | 8 | 与 QSFP 同宽同深 | 400G/800G，主流数据中心 |
| **OSFP** | 8 | 略大（Octal SFP） | 400G/800G，散热更好 |
| SFP-DD | 2 | 与 SFP 同宽 | 100G，替代 SFP28/56 |

---

<details>
<summary>选型建议（点开）</summary>

- **现有 100G 网络升级**：QSFP-DD 端口可直接复用旧 QSFP28 模块，平滑过渡。
- **新建 400G 数据中心**：QSFP-DD 与 OSFP 二选一；QSFP-DD 面板密度高、生态广，OSFP 单模块散热和功耗余量更大。
- **QSFP112 vs QSFP-DD**：若只需 400G 且想兼容现有 QSFP 单排布线，可看 QSFP112；若要上 800G，选 QSFP-DD800 或 OSFP。
- **别只看接口名**：兼容性还取决于**交换机端口速率、调制方式、线缆类型（DAC/AOC/光模块）**是否匹配。

</details>
