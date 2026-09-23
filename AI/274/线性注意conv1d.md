# 线性注意conv1d

## 🧑‍💻 User

线性注意力中的conv1d kernel有什么用途？

---
## 🤖 Assistant

# 线性注意力中 Conv1d 的用途

在 Mamba、RWKV、GLA、DeltaNet、RetNet 等线性注意力/SSM 类架构中，你几乎总能看到一个小的 **depthwise 因果卷积**（`kernel size = 2~4`），通常位于 `in_proj` 之后、注意力/SSM 计算之前。它不是为了"加容量"这么简单，而是在补线性注意力的结构性短板。

## 核心用途

### 1. 提供局部 token 混合（最主要的作用）

线性注意力的状态是一个外积累加：

$$S_t = \gamma S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t$$

这本质上是把历史信息压缩成一个"模糊的全局均值"。它擅长长程、衰减式的信息聚合，但对**紧邻 token 之间的精确依赖**（局部 bigram/trigram、词法、复制、格式化模式）非常弱。Conv1d 相当于免费送了一个 n-gram 感受野，让每个 token 先"看到"邻居再进注意力。

### 2. 注入顺序与局部性先验

纯线性注意力对序列的**局部顺序不敏感**——query 对历史的检索是各向同性的加权平均，缺少 softmax attention 那种天然的 recency bias。Conv1d 是一个固定形状的、带顺序的局部算子，把"相邻 token 更相关"这一归纳偏置直接写入结构，模型不必浪费状态容量去学这个。

### 3. 增加非线性与特征变换

Conv 之后通常接 SiLU/Swish 激活，打破了 Q/K/V 投影的纯线性结构，让进入注意力的特征更接近局部模式检测器（类似 CNN 的边缘/纹理检测器），提升表达力。

### 4. 充当"廉价版局部注意力"的替代

有研究（如 *Based*、*Zoology* 系列）指出线性注意力在精确回忆（exact recall / lookup）任务上失败，常用解法是混合局部 attention 层。Conv1d 是比 sliding-window attention 更便宜的近似替代——无 attention 矩阵、无需 KV cache。

### 5. 平滑与稳定

Depthwise 卷积沿通道独立操作，相当于一个可学习的低通滤波器，能抑制 token 特征中的高频噪声，使后续递归状态更新更平稳。

## 实现细节

<details>
<summary>点击展开：具体形态与工程考虑</summary>

- **Depthwise（逐通道）**：每个通道独立卷积，参数量只有 $O(C \cdot k)$（$k$ 是 kernel size），计算开销可忽略。
- **因果 padding**：只在左侧补零，保证不泄露未来信息。
- **训练 vs 推理**：
 - 训练时用并行 `conv1d`（可用 chunked scan 加速）；
 - 推理时只需维护长度为 $k-1$ 的滑动窗口缓存，内存 $O(1)$，完全兼容 RNN 式逐 token 解码。
- **RWKV 的 token shift** 是特例：

$$y_t = w \odot x_{t-1} + (1-w) \odot x_t$$

这等价于 `kernel size = 2` 的 depthwise conv1d。
- **Mamba/Mamba-2**：`in_proj` 后先过 `conv1d(k=4)` + SiLU，再进 SSM 分支，另一个分支 $z$ 做门控。
- **消融结论**：去掉它通常在需要精确局部依赖的任务（复制、induction head、句法敏感任务）上明显掉点，整体困惑度也会变差。

</details>

## 一句话总结

Conv1d 用几乎为零的代价，给只会做"全局模糊加权平均"的线性注意力补上了**精确的局部模式捕捉能力**——这是它在这些架构里几乎成为标配的原因。
