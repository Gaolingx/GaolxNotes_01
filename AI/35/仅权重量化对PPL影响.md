# 仅权重量化对PPL影响

### 🧑‍💻 User

从sampling的角度来说，为什么常见的仅权重量化对transformer llm的ppl影响相当小？

---

### 🤖 Assistant

从 **sampling / 推断统计** 的角度看，“仅权重量化（OWQ，如 w4a16 / w8a16）对 Transformer LLM 的 PPL 影响很小”，并不是一个偶然经验结论，而是多种结构性因素共同作用的结果。下面我按 **“误差注入位置 → 概率分布 → 采样统计”** 的逻辑来系统分析。

---

## 1. OWQ 的核心事实：噪声只注入在“静态参数侧”

仅权重量化意味着：

* **权重 (W)**：被量化，引入一个固定的、输入无关的扰动
  [
  \hat W = W + \Delta W
  ]
* **激活 / hidden state (x)**：仍然是 FP16 / BF16，高精度、动态变化
* **softmax、layernorm、残差**：全部保持高精度

因此在推断时，误差形式是：
[
y = \hat W x = Wx + \underbrace{\Delta W x}_{\text{结构化但可控的噪声}}
]

这是一个**线性、输入相关但幅度受限**的噪声项，而不是量化激活那种“动态截断 + 饱和 + 非线性噪声”。

---

## 2. 从 sampling 角度：PPL 只关心“分布排序是否被破坏”

### 2.1 PPL 的本质

PPL 衡量的是：
[
\mathbb{E}[-\log p_\theta(x_t \mid x_{<t})]
]

等价地，它只关心：

* **目标 token 的 logit 是否仍然在高概率区**
* **Top-k / Top-p 排序是否稳定**

而不是 logit 的绝对数值是否有微小偏差。

---

## 3. Transformer 输出分布的一个关键事实：**logit margin 很大**

在一个训练好的 LLM 中，通常有：

* 正确 token 的 logit
* 与次优 token 之间，存在明显的 **logit margin**

即：
[
z_{\text{gt}} \gg z_{2}, z_{3}, \dots
]

只要量化噪声满足：
[
|\Delta z| \ll \text{logit margin}
]

那么：

* argmax 不变
* softmax 后概率变化极小
* PPL 基本不变

而 **OWQ 的噪声规模通常恰好处在这个“安全区间”内**。

---

## 4. 为什么 (\Delta W x) 在统计上“很温和”

### 4.1 权重量化误差是近似零均值的

对常见对称量化（int8 / int4）：

* (\mathbb{E}[\Delta W] \approx 0)
* 误差在通道 / group 内近似均匀

因此：
[
\mathbb{E}[\Delta W x] \approx 0
]

这意味着它更像 **additive noise**，而不是 systematic bias。

---

### 4.2 多头注意力 + FFN = 强“噪声平均器”

一个 token 的最终 logit 来自：

* 多层 Transformer block
* 多个 attention head
* 大维度 FFN 累加

这导致：
[
\Delta z = \sum_{l,h} \Delta z_{l,h}
]

在统计上近似满足中心极限定理：

* 单个 head 的误差不小
* **总和后的相对误差非常小**

这点和你之前分析过的 **fused matmul / MoE 累加路径** 在数值稳定性上的现象是高度一致的。

---

## 5. Residual + LayerNorm 在 sampling 视角下的“保险作用”

Transformer 的结构使得：

1. 每一层是
   [
   x_{l+1} = x_l + f_l(x_l)
   ]
2. 后面跟 LayerNorm

这意味着：

* 量化误差被“稀释”在 residual 中
* LayerNorm 强制恢复均值 / 方差

从 sampling 看，这等价于：

> 每一层都会把 logit distribution 拉回一个“熟悉的尺度空间”

因此量化噪声**难以指数级放大**，只能线性积累。

---

## 6. 为什么“无需校准数据”在 OWQ 下也成立

校准的本质是为了：

* 估计激活分布
* 确定 scale / zero-point

但 OWQ：

* 不量化激活
* 不涉及动态 range mismatch

而权重本身：

* 在训练后已经是一个“统计稳态分布”
* 不依赖具体推断输入

因此：

* **权重量化是“input-agnostic”的**
* sampling 分布的稳定性不依赖校准集覆盖

---

## 7. 与激活量化（wXaY）在 sampling 视角下的对比

| 量化方式        | sampling 层面的问题                  |
| ----------- | ------------------------------- |
| wXa16（OWQ）  | 只加小噪声，不改分布形态                    |
| w8a8        | 激活截断 → softmax 前分布形态改变          |
| w4a8 / a4   | attention score 分布被压扁，top-p 不稳定 |
| KV cache 量化 | 时间相关误差，影响后续 token 采样            |

这也是为什么：

> **PPL 对 OWQ 极不敏感，但对激活 / KV cache 量化非常敏感**

---

## 8. 一个 sampling 角度的“总结性结论”

从 sampling 的角度可以这样理解：

> **LLM 的 token 采样，本质是一个对 logit 排序极其鲁棒、但对分布形态变化极其敏感的过程。**

而仅权重量化：

* 不改变分布形态
* 只引入小幅、零均值、可被残差和 LN 吸收的噪声
* 不破坏 top-k / top-p 排序

因此 PPL 几乎不变。

---

如果你愿意，下一步我们可以进一步深入到：

* 为什么 **attention projection 的权重量化比 FFN 更“敏感但仍然安全”**
* 或者用 **SNR / Fisher 信息** 的角度重新推一遍 OWQ 的鲁棒性
* 或结合你现在关注的 **groupwise / mixed-quant (Q4_K / IQ4_NL)** 讨论哪些子矩阵才是 PPL 的“真正风险点”

这些都可以继续往下拆。


---

### 🧑‍💻 User

经过研究发现，原因其实是因为logits他是取softmax那个，而quantization对softmax的结果其实影响不会很大，因为通常第一大的logits比第二大的大很多，可能90％的probability mass都在前几个token上面，对于一个vocabulary来说，他是很sparse的，所以quantization可行。这种说法是否存在一定的道理？

---

### 🤖 Assistant

这套说法**有相当扎实的道理**，而且在 *sampling / 信息论 / 排序稳定性* 这三个层面都是可以被形式化解释的。不过，它也**不是充分条件**，需要稍作精炼，避免被过度泛化。下面我按“成立之处 → 严格化表述 → 不成立或不足之处”来评估。

---

## 一、这套说法为什么“本质上是对的”

### 1. Softmax 本质上是一个 **rank-dominated** 的映射

Softmax：
[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

当存在明显 logit gap 时：
[
z_{(1)} \gg z_{(2)} \gg z_{(3)} \dots
]

则有两个关键事实：

1. **概率质量高度集中（peaky distribution）**
2. **softmax 对小幅 logit 扰动不敏感**

形式化地，如果对 logit 加一个扰动 (\delta z)：
[
\delta p_i \approx p_i(\delta z_i - \mathbb{E}_{p}[\delta z])
]

当：

* (p_{(1)} \approx 0.8 \sim 0.95)
* 其余 token 的 (p_i \ll 1)

则绝大多数 token 的 (\delta p_i) 被指数级抑制。

> 换句话说：**softmax 天然是一个“误差吸收器”，前提是分布足够尖锐。**

---

### 2. “vocab 是 sparse 的”——从 sampling 角度这是准确表述

这里的 “sparse” 不是权重稀疏，而是：

* **在当前上下文下的条件分布是低熵的**
* 有效支持集（effective support）很小

常见现象是：

* top-1 ~ top-5 就覆盖了 80–95% 的概率质量
* tail token 的 logit 之间差距很小，但整体概率质量极低

在这种情况下：

* quantization 改变 tail 的 logit 几乎不影响 PPL
* 只要不打乱 top-k 的排序，采样行为不变

---

### 3. 这正好解释了「为什么看 PPL 几乎没变化」

PPL 本质上是 cross-entropy：
[
\mathcal{L} = -\log p_{\text{gt}}
]

当 ground-truth token 本来就在 top-1 / top-2：

* quantization 前后：
  [
  p_{\text{gt}} = 0.92 \rightarrow 0.90
  ]
* (-\log p) 的变化是 **二阶小量**

但如果排序翻转：
[
0.92 \rightarrow 0.40
]
那 PPL 会直接爆炸。

OWQ 的经验事实恰好是：**几乎不发生这种翻转**。

---

## 二、把这套说法“严格化”后，它等价于什么结论？

可以把你的说法改写成一个更精确的版本：

> **仅权重量化可行，是因为 LLM 推断时的条件 token 分布在绝大多数 time step 上是低熵、强排序的，而 softmax 对保持排序的小扰动是 Lipschitz-stable 的。**

或者更工程一点：

> **量化噪声只要小于 logit margin，就不会改变有效采样支持集（top-k / nucleus），因此 PPL 与生成质量几乎不变。**

这比“因为 softmax”本身更准确：
**softmax 只是放大了 logit margin 的鲁棒性，而不是根因。**

---

## 三、这套说法的三个“隐含前提”（很重要）

如果忽略这三点，就容易得出错误结论。

---

### 1. 前提一：logit gap 必须“真的大”

在以下场景中，你的说法会明显减弱甚至失效：

* 语义歧义高的上下文
* 多种 token 同概率（例如 list、JSON key、代码补全）
* 高 temperature sampling

此时：
[
z_{(1)} \approx z_{(2)} \approx z_{(3)}
]

quantization 很容易改变排序。

这也是为什么：

* 代码模型
* structured output
* low-bit activation / KV quant

比自然语言更脆弱。

---

### 2. 前提二：quantization 不能引入 systematic bias

你的说法默认的是：

* 量化误差是近似零均值、独立的

但如果是：

* per-tensor scale
* asymmetric clipping
* 某些 channel 被压扁

那么：
[
\delta z \text{ 不再是噪声，而是偏移}
]

这时 softmax 不但不“吸收误差”，反而会**指数放大 bias**。

---

### 3. 前提三：误差不能破坏 attention score 的分布形态

你的说法**主要成立于 final logits 层**。

但如果量化影响的是：

* QKᵀ / √d 的尺度
* attention score 的相对差异

那么 softmax 前的分布形态已经被改变，后面的 logit gap 可能根本不会形成。

这就是：

* w4a16 OK
* w4a8 可能不 OK
* KV cache 量化经常灾难性的根本原因

---

## 四、一个更完整的总结判断

**结论是：**

* 你的说法在 *sampling 视角* 下是**高度正确的**
* 它抓住了 **“排序稳定性 + 低熵分布”** 这个真正的关键
* 但需要补充：

  * 不是 softmax 本身救了量化
  * 而是 *logit margin + 低 entropy* 让 softmax 成为鲁棒映射

一句更“精确但不削弱直觉”的版本是：

> **LLM 的 token 分布在推断时极度集中，只要量化噪声不打破 logit 排序，softmax 输出和采样统计几乎不变，因此仅权重量化在 PPL 上表现得异常安全。**

如果你愿意，下一步我们可以直接把这个观点和：

* Fisher 信息 / KL 对扰动的一阶展开
* 为什么 diffusion / UNet sampling 完全不满足这个前提
* 为什么 KV cache 量化是“时间相关破坏排序”

这些角度对齐，会非常漂亮。

---

### 🧑‍💻 User

既然softmax分布特性决定了llm inference的compute forward本身就有一定的容错率在里边，为什么在temperature低的情况下，量化的影响应该更不明显？

---

### 🤖 Assistant

这个问题问得非常到位，而且**结论直觉是对的**：

> **temperature 越低，量化对 LLM 推断结果（尤其是 PPL / argmax 行为）的影响，理论上应该越不明显。**

但这里面有一个容易混淆的点：**softmax 的“容错性”并不是无条件的**。它在 *低 temperature* 下之所以更“安全”，本质原因和你前面提到的 **logit margin、排序稳定性、采样退化为 argmax** 强相关。下面我从严格的计算角度拆解。

---

## 1. 先把问题写成数学形式

带 temperature 的 softmax：

[
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
]

量化后的 logit：
[
\hat z_i = z_i + \delta z_i
]

我们关心的是：
在 (T \downarrow) 的情况下，(\delta z) 对 (p_i) 和 sampling 行为的影响是否减小？

---

## 2. 一个反直觉但关键的事实

> **softmax 对 logit 的“数值敏感度”在低 T 时是变大的，而不是变小的。**

因为：
[
\frac{\partial p_i}{\partial z_j} = \frac{1}{T} p_i(\delta_{ij} - p_j)
]

所以从**纯连续函数的 Lipschitz 常数**看：

* (T \downarrow \Rightarrow \frac{1}{T} \uparrow)
* softmax *更* 敏感

**如果只看这个，会得出完全相反的结论。**

那为什么在 LLM inference 中，我们观察到：
低 temperature 下量化更“安全”？

---

## 3. 关键区分：**概率扰动** vs **采样决策扰动**

### 3.1 在低 temperature 下，采样退化为“排序问题”

当 (T \to 0)：

[
\text{softmax}(z/T) \to \text{one-hot}(\arg\max z)
]

此时 inference 的行为本质是：

> **比较 logit 的大小关系，而不是它们的精确数值。**

因此：

* 只要 quantization 不改变
  [
  \arg\max_i z_i
  ]
* 采样结果完全不变
* PPL 几乎不变

这是一种 **离散鲁棒性**，而不是连续鲁棒性。

---

### 3.2 用 logit margin 表述这个条件

设：
[
\Delta = z_{(1)} - z_{(2)}
]

量化误差上界：
[
|\delta z_i| \le \epsilon
]

则排序保持的充分条件是：
[
2\epsilon < \Delta
]

这个条件 **与 temperature 无关**。

而当 (T) 很低时：

* 采样只关心是否满足这个条件
* 一旦满足，softmax 的指数放大反而**把所有次优 token 压到 0**

---

## 4. 为什么“低 temperature + 量化”在经验上特别稳定

### 4.1 LLM 的真实 logit gap 在低 T 下被指数放大

注意：temperature 是在 **推断时**施加的：

[
z_i \mapsto z_i / T
]

这意味着：
[
\Delta_T = \frac{z_{(1)} - z_{(2)}}{T}
]

当 (T \downarrow)：

* 有效 logit gap 被 **人为拉大**
* 排序翻转的概率急剧下降

从 sampling 视角看：

> **低 temperature 等价于“人为增加模型的决策 margin”。**

这对量化噪声是极其友好的。

---

### 4.2 概率质量塌缩，tail token 完全失去影响力

在低 T 时：

* (p_{(1)} \to 1)
* (p_{(i>1)} \to 0)

此时：

* 即便 quantization 让某些 tail token 的 logit 有相对大扰动
* softmax 后仍然是 0
* 对 cross-entropy 的贡献几乎为 0

---

## 5. 从 PPL 的角度再看一遍

PPL 用的是：
[
-\log p_{\text{gt}}
]

在低 T + 排序正确的前提下：

* (p_{\text{gt}} \approx 1)
* (-\log p \approx 0)

量化前后：
[
0.999 \rightarrow 0.995
]

这是一个 **数值上极小、二阶量级的变化**。

---

## 6. 为什么这套逻辑在高 temperature 下会失效

当 (T \uparrow)：

* logit gap 被压缩
* 多个 token 进入 nucleus
* sampling 对概率微小变化极其敏感

这时：

* quantization 不再只是“噪声”
* 而是直接改变采样路径
* PPL / generation quality 都会明显恶化

---

## 7. 一个非常重要的边界条件（必须强调）

> **低 temperature 只能“掩盖”量化误差，不能“修复”错误。**

如果量化已经导致：

* attention score 分布变形
* logit 排序系统性偏移
* 某些 channel 被 scale 压死

那么：

* (T \to 0) 只会**更坚定地输出错误 token**
* 表现为“错误但非常自信”

这也是为什么：

* w4a16 + T=0.7 很安全
* w4a8 + T=0.1 可能更糟

---

## 8. 一句话总结

可以这样精确地回答你的问题：

> **低 temperature 并不是让 softmax 对量化噪声“更平滑”，恰恰相反，它放大了 logit 差异，使采样行为退化为一个只关心排序的离散决策过程。只要量化误差不足以破坏 logit 排序，inference 结果就完全不变，因此量化影响在经验上更不明显。**

如果你愿意，下一步我们可以把这个结论直接和：

* KV cache 量化的时间相关误差
* diffusion / score-based model 为何完全不具备这种“排序鲁棒性”
* 甚至和 beam search / greedy decode 的极端情况

做一个对照，会非常清晰。
