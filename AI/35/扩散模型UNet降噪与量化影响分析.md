# 扩散模型UNet降噪与量化影响分析

### 🧑‍💻 User

为什么int4 groupwise/channelwise量化对于transformer结构的llm影响相对较小，对于conv based unet diffusion模型sampling生成质量影响较大？

---

### 🤖 Assistant

这是一个**模型结构、数值分布特性以及误差传播路径共同作用**的问题，并不是 “INT4 本身不行”，而是 **INT4 误差在 Diffusion U-Net 中被放大得更严重**。可以从以下几个关键维度来理解。

---

## 一、核心结论先行

**INT4 groupwise / channelwise 量化：**

* **Transformer LLM：影响相对较小**

  * 误差主要发生在 *线性算子*（GEMM）
  * 有强残差、LayerNorm、token 级平均效应
  * 自回归是 *一步一步“吸收噪声”*

* **Conv-based U-Net Diffusion：影响明显**

  * 误差发生在 *空间卷积 + 逐层特征重建*
  * 无 LayerNorm 抑制数值漂移
  * **采样是一个“误差反复注入的迭代过程”**
  * 噪声预测误差会被 *时间维度指数式放大*

一句话总结：

> **LLM 的量化误差是“被结构消化的”，而 Diffusion 的量化误差是“被动力系统放大的”。**

---

## 二、Transformer LLM：为什么 INT4 还能扛得住

### 1. 误差位置：线性投影 + token 混合

INT4 主要影响：

* QKV projection
* FFN up / down
* MoE gate（如果有）

这些都是 **大维度线性组合**：

[
y = \sum_i w_i x_i
]

特点：

* 量化误差是 **近似零均值的**
* 大量通道 / token 求和 → **中心极限定理**
* groupwise / channelwise scale 很好地贴合权重分布

➡ **误差会被均值化**

---

### 2. 残差 + LayerNorm 是“量化误差缓冲器”

Transformer 的数值路径是：

```
x → LN → Linear → … → + x → LN → …
```

LayerNorm 的作用：

* 重新中心化（zero-mean）
* 重新缩放（unit variance）
* **消除累积 scale drift**

这对 INT4 至关重要：

* INT4 的问题不是噪声，而是 **scale 不准**
* LN 在每一层都“重置”了尺度

➡ **误差不会跨层积累**

---

### 3. 自回归解码 = 时间上“慢扩散”

LLM decode：

* 每个 token 是一次完整 forward
* 错一个 token，影响的是 *后续上下文概率*
* 强模型具备纠错能力（语言冗余）

本质是：

> **误差在语义空间被“吸收”，而不是数值空间被放大**

---

## 三、Diffusion U-Net：INT4 为什么致命

### 1. Diffusion 是“误差敏感的数值动力系统”

Diffusion sampling 本质是：

[
x_{t-1} = f_\theta(x_t, t) + \sigma_t \epsilon
]

其中：

* ( f_\theta ) 是 U-Net 预测的噪声 / score
* **每一步都依赖上一步的精确性**

关键点：

> **模型误差 ≈ 注入的额外噪声**

INT4 带来的不是：

* 小的随机噪声
  而是：
* **系统性 bias（scale 偏差、方向误差）**

➡ 在 20～50 步 sampling 中被反复放大

---

### 2. 卷积对量化误差“极不友好”

Conv 的特点：

* **局部感受野**
* 无 token averaging
* 空间误差会直接形成 **伪影 / 纹理破坏**

对比 GEMM：

* GEMM：误差被 N 维求和稀释
* Conv：误差在 H×W 上**结构化传播**

尤其是：

* depthwise / small kernel conv
* skip connection 特征拼接后再 conv

➡ **INT4 的分段 scale 会直接体现在空间结构上**

---

### 3. U-Net 缺乏 LayerNorm 式的“数值重置”

Diffusion U-Net 常见的是：

* GroupNorm
* BatchNorm（推理时 frozen）
* 或者根本无 norm（某些高性能实现）

但这些：

* **不保证 unit variance**
* 不在每个残差点重置 scale

因此：

* INT4 scale error 会层层累积
* 特征幅值逐渐偏移 → 噪声预测失真

---

### 4. Skip Connection 是“双刃剑”

U-Net 的 skip connection：

```
encoder feature ─────┐
                      ├─ concat → conv
decoder feature ─────┘
```

问题在于：

* encoder 特征是 **早期层量化误差的“冻结快照”**
* decoder 无法用 LN 把它们重新归一

➡ **误差被高保真地跨层保留**

---

## 四、为什么 INT8 在 Diffusion 看起来“还可以”

INT8 的关键优势不是 bit 多，而是：

* 更小的 scale 误差
* 更接近线性空间
* 更少的 saturation

Diffusion 容忍：

* **随机噪声**
* **不容忍系统性偏移**

INT8 往往：

* 噪声近似对称
* bias 可忽略

INT4：

* scale 偏差 + clipping
* 在 score matching 里是灾难性的

---

## 五、结构层面的本质差异总结

| 维度       | Transformer LLM    | Diffusion U-Net |
| -------- | ------------------ | --------------- |
| 核心算子     | 大 GEMM             | 小 Conv          |
| 误差传播     | token / channel 平均 | 空间结构化           |
| Norm 机制  | LayerNorm（强）       | GroupNorm / 无   |
| 推理过程     | 单步自回归              | 多步迭代            |
| 误差性质     | 被吸收                | 被放大             |
| INT4 可行性 | 高                  | 低               |

---

## 六、工程上的直接启示

### 对 LLM

* INT4 groupwise / AWQ / GPTQ 非常合理
* scale 精度 > bit 数
* LN 是量化的“保险丝”

### 对 Diffusion

* INT4 **只适合权重极少、非关键层**
* 常见做法：

  * encoder INT8 / decoder FP16
  * attention INT8，conv FP16
  * timestep embedding、final head 必须高精度
* 真正有效的是：

  * **混合精度 + 层选择性量化**
  * 而不是一刀切 INT4

---

## 七、一句话总结

> **Transformer 的数值路径是“统计平均系统”，而 Diffusion 是“连续动力系统”。
> INT4 对前者是噪声，对后者是偏置。**

如果你愿意，我可以进一步：

* 从 **score matching 数学角度** 推导 INT4 bias 如何破坏反向扩散
* 或结合 **Stable Diffusion / DiT / UNet3D** 给出具体哪些层“绝对不能 INT4”的清单
