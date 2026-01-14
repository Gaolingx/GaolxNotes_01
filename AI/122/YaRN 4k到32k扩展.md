# YaRN 4k到32k扩展

### 🧑‍💻 User

如何用yarn rope scaling将4k上下文扩展到32k？

---

### 🤖 Assistant

下面从**原理、公式、具体配置方式、训练/推理注意事项，以及常见误区**几个层面，系统说明如何用 **YaRN (Yet another RoPE extrapolatioN) rope scaling** 将 **4k 上下文扩展到 32k**。

---

## 1. 问题背景：为什么普通 RoPE 不能直接 4k → 32k

标准 RoPE（Rotary Positional Embedding）中，位置编码以**频率空间的旋转角度**形式注入：

[
\theta(p, i) = p \cdot \omega_i,\quad \omega_i = 10000^{-2i/d}
]

其中：

* (p)：token 位置
* (i)：head 内维度索引

问题在于：

* **低频维度负责长程顺序**
* **高频维度负责短程细节**

当你直接 extrapolate 到 32k：

* 高频维度在 4k 之后已经发生**严重 aliasing**
* attention score 退化，模型会“忘记顺序”

---

## 2. YaRN 的核心思想（非常关键）

YaRN 的本质不是“统一拉伸位置”，而是：

> **只对“负责长程建模的低频维度”进行 rope scaling，高频维度保持原始 RoPE 行为**

也就是说：

* **分频段处理 RoPE**
* 不是 naive 的 linear scaling

这点和 NTK scaling、linear scaling 本质不同。

---

## 3. YaRN 的数学形式（简化版）

定义：

* 原始最大上下文：(L_{\text{orig}} = 4096)
* 目标上下文：(L_{\text{new}} = 32768)
* scaling factor：
  [
  s = \frac{L_{\text{new}}}{L_{\text{orig}}} = 8
  ]

YaRN 对 RoPE 频率进行 **piecewise scaling**：

[
\omega_i' =
\begin{cases}
\omega_i / s & \text{低频区} \
\omega_i & \text{高频区}
\end{cases}
]

但**真正工程实现**不是硬切，而是：

### 平滑过渡（关键）

引入一个平滑函数 (f(i))：

[
\omega_i' = \omega_i / \bigl(s^{f(i)}\bigr)
]

其中：

* (f(i) \in [0, 1])
* 低频 (f(i)\approx 1)
* 高频 (f(i)\approx 0)

这样避免 attention 结构突然断裂。

---

## 4. 实际工程参数（4k → 32k 的推荐配置）

以 HuggingFace / LLaMA 系为例。

### 4.1 scaling factor

```text
rope_scaling.factor = 8
```

这是硬性比例，来自 32k / 4k。

---

### 4.2 rope_scaling 类型

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 8
}
```

如果框架支持 `original_max_position_embeddings`，务必显式指定：

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 8,
  "original_max_position_embeddings": 4096
}
```

---

### 4.3 YaRN 特有参数（极其重要）

不同实现名字略有差异，但核心一致：

| 参数                     | 含义                  | 推荐   |
| ---------------------- | ------------------- | ---- |
| `beta_fast`            | 高频过渡起点              | 32   |
| `beta_slow`            | 低频完全缩放点             | 1    |
| `extrapolation_factor` | 有时等同 factor         | 8    |
| `attention_factor`     | attention re-weight | ≈1.0 |

典型设置（来自社区实践）：

```python
rope_scaling = {
    "type": "yarn",
    "factor": 8,
    "beta_fast": 32,
    "beta_slow": 1
}
```

直觉解释：

* **最底层频率完全拉伸**
* **最高频率几乎不动**
* 中间频率平滑过渡

---

## 5. 训练 vs 推理：是否需要再训练？

### 5.1 仅推理（zero-shot extrapolation）

可以直接用 YaRN：

* **结构正确**
* **顺序感知基本保持**
* 但：

  * 长上下文 recall 会下降
  * reasoning 深度受限

适合：

* 信息检索
* 长文摘要
* RAG

---

### 5.2 轻量继续训练（强烈推荐）

最佳实践：

* **保持 YaRN rope**
* 在 **8k–32k 混合长度**数据上继续训练
* 1–5B tokens 通常就能显著修复长上下文能力

原因：

* 模型需要重新适配新的 attention phase space

---

## 6. 为什么 YaRN 比 Linear / NTK scaling 好？

从 attention score 角度看：

[
QK^\top \sim \cos(\theta_q - \theta_k)
]

* Linear scaling：所有频率同时压缩 → 短程 token 顺序被破坏
* NTK scaling：过度偏向低频 → 近邻 token attention 模糊
* **YaRN：频率分工不被破坏**

一句话总结：

> YaRN 是唯一同时保住 **local coherence** 和 **long-range ordering** 的 RoPE 外推方案

---

## 7. 常见坑（非常常见）

### ❌ 只改 factor，不改 type

→ 变成 linear scaling，效果很差

### ❌ 原始 max_position_embeddings 没设

→ 频率分段错误，attention 退化

### ❌ 直接训练 32k，从头跑

→ 成本高、收敛慢、无必要

### ❌ 结合 FlashAttention 时忘记同步 RoPE 实现

→ kernel 和 Python RoPE 不一致，直接数值错误

---

## 8. 一句话操作清单（TL;DR）

**4k → 32k，用 YaRN：**

1. `factor = 8`
2. `type = "yarn"`
3. 设置 `beta_fast / beta_slow`
4. 显式声明原始 4k
5. 最好做少量继续训练

---

如果你愿意，我可以进一步：

* 对比 **YaRN vs NTK scaling 在 attention phase space 的几何差异**
* 给你一段 **HuggingFace / vLLM / llama.cpp 的具体 patch**
* 分析 **32k 下 KV cache、FlashAttention tile size 的实际影响**

这些都和你现在关心的系统层问题是强相关的。


---

### 🧑‍💻 User

但是如果模型本身没有进行过后训练长上下文微调 直接在config.json里面添加yarn参数 外推还是会掉点，针对original model的ft要怎么做？

---

### 🤖 Assistant

这是一个**非常关键且经常被低估的问题**。结论先行：

> **YaRN 只能保证“相位空间是对的”，但并不能让模型“知道如何在 32k 相位空间里工作”**。
> 如果原始模型只在 4k 以内训练过，那么**不做针对性的 long-context FT，掉点是必然的**。

下面我从**为什么会掉点 → FT 的目标是什么 → 具体可操作的 FT 方案 → 训练配方与工程细节 → 常见失败模式**几个层面，给你一个**可直接落地的方案**。

---

## 1. 为什么只改 YaRN config 仍然会掉点（本质原因）

即使 YaRN 修复了 RoPE 的 extrapolation 几何结构，模型仍然存在三个“未适配”的问题：

### 1.1 Attention 的统计分布完全变了

在 4k 训练时：

* attention score 的有效支持区间：[0, 4096]
* softmax temperature、head specialization 都隐含地适配了这个范围

扩到 32k 后：

* token pair 距离分布发生 **8× 扩张**
* 大量 attention score 落在模型从未见过的相位区间

**这不是 RoPE 能解决的问题，是统计适配问题。**

---

### 1.2 FFN / MLP 里存在“隐式长度假设”

很多模型在：

* summary
* delimiter
* reasoning chain

上学到的是：

> “关键信息不会离我太远”

FT 前：

* 中后段 token 的表示范数、激活分布明显不同
* FFN gate 对远距离信息的利用效率极低

---

### 1.3 Head specialization 被破坏但未重建

大量 head 在 pretrain 中形成：

* local head
* mid-range head
* global head

但这个“global”的上限仍然是 **4k 级别**。
YaRN 只是把“世界地图扩展了”，但**head 并不知道哪些 head 该负责 16k–32k**。

---

## 2. Long-context FT 的真正目标（不是“训长文本”）

**错误理解**：

> “喂 32k 长文本给模型就行”

**正确目标是三件事**：

### 2.1 重标定 attention 的距离感知

让模型学会：

* 8k / 16k / 32k 的距离是“正常存在的”
* attention 不应该在 4k 以后指数衰减

---

### 2.2 重新分配 head 的职责

让部分 head：

* 明确负责 very-long-range dependency
* 形成“稀疏但稳定”的远程 attention

---

### 2.3 校准激活与数值稳定性

尤其是：

* RMSNorm / LayerNorm 后的分布
* KV cache 在极远距离下的数值范围

---

## 3. 针对 original model 的 **最小可行 FT 方案（强烈推荐）**

这是社区和工业界实践下来 **性价比最高** 的做法。

---

### 3.1 核心原则（一定要记住）

1. **从不改模型结构**
2. **保持 YaRN RoPE 全程启用**
3. **只训练极少参数 / 或低 LR 全模型**
4. **混合上下文长度训练**

---

## 4. 具体 FT 配方（4k → 32k）

### 4.1 RoPE 配置（FT 全程固定）

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 8,
  "original_max_position_embeddings": 4096,
  "beta_fast": 32,
  "beta_slow": 1
}
```

**FT 时不要切换回原 RoPE。**

---

### 4.2 上下文长度 curriculum（非常重要）

不要直接全 32k。

推荐比例：

| 序列长度  | 占比  |
| ----- | --- |
| 2k–4k | 40% |
| 8k    | 25% |
| 16k   | 20% |
| 32k   | 15% |

目的：

* 保住短上下文能力
* 防止 attention 完全“长化”

---

### 4.3 数据类型（比“长文本”更重要）

**有效数据 ≫ 随便拼接**

强烈建议包含：

1. **远距信息检索任务**

   * 问题在前，证据在 10k+ token 之后
2. **多段 reasoning**

   * 中间插入大量无关段落
3. **Key-value recall**

   * “在第 18342 个 token 出现的变量 X 是什么？”
4. **结构化长文**

   * 代码仓库、论文、日志

> 单纯把短文本拼成 32k，收益非常有限。

---

## 5. 训练策略：两条现实可行路径

---

### 路径 A：LoRA / QLoRA（推荐起步）

**训练对象**：

* QKV projection
* O projection
* 可选：MLP gate

**参数规模**：

* <1% 参数

**学习率**：

```text
lr = 1e-4 ～ 5e-5
```

**优势**：

* 快
* 不易破坏原模型
* 掉点可控

---

### 路径 B：全模型低 LR FT（效果最好）

**学习率**：

```text
lr = 1e-5 ～ 5e-6
```

**训练量**：

* 1B～5B tokens 通常足够
* 不需要 full pretrain 级别

**注意**：

* 必须有 gradient clipping
* attention 数值稳定性很关键

---

## 6. 一些“反直觉但很重要”的工程细节

### 6.1 不要冻结 attention

很多人 freeze attention 只训 FFN —— **长上下文会几乎没改善**。

---

### 6.2 FlashAttention / SDPA 必须一致

FT 时用的 RoPE 实现：

* kernel
* Python reference

必须完全一致，否则学到的是错相位。

---

### 6.3 KV cache dtype 会影响 FT 成败

* fp16 KV 在 32k 下数值误差明显
* bf16 / fp32 KV cache 在 FT 阶段更稳

---

## 7. 如何判断 FT 是否“训对了”

几个强信号：

1. **attention entropy**

   * 32k 位置不再接近 uniform
2. **远距 token 的 logit sensitivity**

   * 修改 20k 处 token，输出明显变化
3. **4k 内任务几乎不掉点**

如果：

* 长上下文提升了
* 4k 掉点 <1–2%

说明 FT 是成功的。

---

## 8. 总结一句话

> **YaRN 解决的是“几何合法性”，
> long-context FT 解决的是“统计与功能适配”。**

**正确做法不是“能跑 32k 就算成功”，而是：**

* attention 结构稳定
* head 职责重新分化
* 短上下文能力不被破坏

---

如果你愿意，下一步我可以帮你：

* 设计一套 **最小 50M–100M token 的 long-context FT 数据方案**
* 给出 **QKV-only LoRA 的具体 target_modules**
* 或从 **attention phase / head entropy 的角度，给你一个“是否训对”的定量指标**

这些都可以直接接到你现在的实验流程里。
