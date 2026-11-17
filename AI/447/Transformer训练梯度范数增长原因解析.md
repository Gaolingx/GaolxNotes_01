# Transformer训练梯度范数增长原因解析

## 🧑‍💻 User

一直不明白为什么transformer训着训着gradnorm一定会慢慢涨 这种无界increase是harmful的吗

---
## 😀 默认助手 (7)

## 先纠正一个直觉

我们本能地觉得：`loss` 在下降、曲线在变平，那 $\|\nabla_\theta L\|$ 就该趋于 0。但这两件事根本不是一回事。

- `loss` 的下降速度是 $-\dot L = \eta\|\nabla_\theta L\|^2$（对 SGD），而**梯度范数本身只是“当前这点有多陡”**，不是“离最优有多近”。
- `Adam` 的每坐标更新是 $\eta\, g/\sqrt{v}$，如果 $g$ 和 $\sqrt{v}$ 同步变大，**实际 update 根本不变**。

所以“`grad norm` 涨了”这件事，一半是真实信号（模型在变陡），一半是测量偏差（参数空间范数并不衡量函数空间里的移动量）。分开看才有意义。

---

## 为什么会涨：四层原因，从表象到根本

### 1. 根本原因：交叉熵没有下界，而 LN 挡住了便宜的路径

这是我认为最能解释“**必然性**”的视角。

对 CE 损失，要让 $\ell<\epsilon$，必须让 logit margin $m = z_y-\log\sum_{j\neq y}e^{z_j}$ 满足
$$m > \log\frac{1}{\epsilon}$$
而 $z = W_U h$。也就是说，**只要 `loss` 还想继续降，logits 的量级就必须继续涨**：CE 没有有限的最优解，它是一个可以无限下降的目标。

关键在于：`LayerNorm` 让“直接把权重乘个 $c$”这条便宜的放大路径失效了（$LN(cW x) = LN(Wx)$）。于是模型只能绕远路去放大那些 **LN 管不到的量**：

- 残差流的范数 $\|h\|$（它随深度累加，且没有被归一化）
- embedding 的范数、`unembedding` 的范数
- LN 的 affine 参数 $\gamma$、以及后来出现的 [massive activations](https://arxiv.org/abs/2402.17762)（少数通道的激活值涨到 $\sim10^3$ 倍，充当近似的常量 bias，专门用来把 attention sink 位置压死）

这些量是**缓慢、单调、近似对数或线性增长**的，因为便宜的路径被堵住了，只能慢慢爬。梯度范数就在跟踪这些量级的增长。这正好解释了为什么观察到的是一条**平滑缓慢的斜坡**，而非阶跃。

### 2. 梯度结构变化：从“许多样本取平均”变成“少数难样本主导”

训练早期，所有 token 的贡献差不多，梯度是大量近似同向项的均值，**平均会压低范数**。

训练后期，绝大多数 token 已被预测得很准，$p-y \to 0$（指数级地趋近 0），它们的梯度基本消失。剩下的是那批**难 token**：对它们来说 $\|p-y\|=O(1)$，而单项梯度幅值 $\propto \|h\|$，**这个 $\|h\|$ 正在涨**。

于是总梯度变成了一个**重尾的、稀疏的**量：均值在降，但幸存的项在变大。这同时解释了 `grad norm` 缓升和 `loss spike` 这两件事。

### 3. 渐进锐化（progressive sharpening）

`loss` 下降时，网络往往不是走进平坦的谷底，而是走进**更尖**的地方。Cohen 等人（Edge of Stability, 2021）在 MSE 上观察到：sharpness $\lambda_{\max}(H)$ 随训练上升，直到 $\approx 2/\eta$ 才被“顶住”。对二次型 $L=\tfrac12\theta^\top H\theta$，$\|g\|=\|H\theta\|$，沿主特征方向走时 sharpness 和 `grad norm` 同步增长。语言模型的 CE 不是二次型，但“把 confidence 拉满”正是在同一方向上加深曲率。

更贴合现代实践的框架是 **(L0, L1)-smoothness**（Zhang et al., ICLR 2020, *Why Gradient Clipping Accelerates Training*）：他们给出
$$\|\nabla f(x)-\nabla f(y)\| \le (L_0+L_1\|\nabla f(x)\|)\,\|x-y\|$$
并且经验上观察到 `grad norm` 随 step **近似线性增长**。这个不等式正好说明为什么 clipping 有效、为什么 `grad norm` 涨本身不是“错”，而是**“函数变陡”的信号**。

### 4. 度量本身不可靠（这一部分是假象）

- **尺度对称性**：LN 之前的权重有一整条 loss 恒定的方向，$\langle \nabla_W L, W\rangle = 0$。沿这条方向的 `grad norm` 大小没有函数意义。
- **各层量纲不一致**：总范数被某一个漂移得最凶的层主导，而这个层级往往是 `unembedding`、最后一个 LN 的 $\gamma$、attention 的 output projection。至于具体是哪一层，取决于配置，**必须测**。
- **框架报的是 clip 前、跨 micro-batch 平均的值**，和真正的优化步长之间隔了好几层换算。

---

## 为什么“Transformer 特别明显”

- **残差累加**：信号在深度上做加法，范数天然膨胀，而 `pre-LN` 又让每个 block 的*相对*贡献变小（Xiong et al. 2020 讨论过 `pre-LN` 初始化时深层梯度结构的问题）。
- **softmax 无界**：logits 可以无限涨，指数敏感，没有 tanh 那种压缩。
- **LN 的尺度不变性**：挡住了便宜的放大路线，把增长挤到慢变量上，于是变成“必然的缓坡”。
- **CE 没有 label smoothing / temperature / z-loss**：目标本身对尺度无下界。
- **宽度与 token 数**：梯度是 $B\times T$ 项的和，重尾效应被放大。

CNN 的现代训练里同样能看到 `grad norm` 上涨（Zhang et al. 在 ViT 上也观察到了），只是 BN/ReLU/较浅的深度让它更温和。

---

## 无界增长是否有害

**直接结论：温和的缓坡基本无害；真正的害处不是“涨”，而是“涨到什么后果”。**

“`grad norm` 无界增长”要先经过一次拆分：

- **缓慢漂移**（全程涨 2–10 倍，平滑单调）：这是**系统性副作用**，Adam 的 per-coordinate 归一化基本把它抵消了。
- **指数爆炸**：那不是“慢慢涨”，那是发散，是另一类问题。

判断依据：

| 观察到的现象 | 判断 |
|---|---|
| 平滑单调、涨几个数量级、`loss` 稳定下降、clip 触发率 ≪ 1、$\|\Delta\theta\|/\|\theta\|$ 稳定 | **无害**。GPT-2/3 级别的训练用普通 `AdamW + clip 1.0` 就能跑完，本身就是证据 |
| `grad norm` 长期低于 clip 阈值、最后越过阈值，**clip 长期饱和** | **有害**。clipping 不满足尺度不变性，一旦阈值长期被触发，你就退化成归一化梯度下降，**丢掉了 Adam 的自适应预条件**，还引入了一个全局耦合的缩放 |
| 伴随 `loss spike`、`attention entropy collapse`（max attention weight → 1）、max logit 爆炸、bf16 大激活溢出 | **有害**。可用 LR 上限被压低、训练效率下降、可能发散 |
| `grad norm` 涨但 `update/weight` 比稳定 | 多半是测量偏差（尺度方向/C 常数），**不用处理** |
| 只有 `grad norm` 涨导致尾部实际停滞 | **真实代价**：它把固定的 LR 隐式变成了“越来越小的有效步长”，所以你不得不做 cosine decay 来补偿 |

一个实用的判断标准：**别把“压低 `grad norm`”当目标**。你真正在乎的是函数空间里的位移量，而 `grad norm` 只是个粗糙的 proxy。该盯的是：`loss` 是否平滑、clip fraction、`update-to-weight ratio`、max logit / attention entropy。

---

<details>
<summary><b>如果真想动手验证（可展开）</b></summary>

最小实验集，跑在 `nanoGPT` 规模就够：

1. **分层记录** `grad norm`：embedding / 每个 block / final LN / unembedding。看是谁在涨。
2. **记录 clip fraction** 和 clip 前后的比值。如果 >5%，说明已经到了“有害”那一区。
3. **记录** `max|logit|`、per-head **attention entropy**、per-layer **残差流范数**、**max activation**。这些是“变尖”的直接证据。
4. **记录** $\|\Delta\theta_l\|/\|\theta_l\|$ per layer（update-to-weight ratio）。它稳而 `grad norm` 涨 ⇒ 测量偏差；它也在涨 ⇒ 是真的在变陡。
5. 做一个**消融**：加 `z-loss` 或 logit soft-capping，看 `grad norm` 曲线是否被压平而 `loss` 不变差。如果压平了 loss 也没变差，说明那部分增长确实没有用处（属于尺度冗余）。

对应的缓解手段（按性价比）：`z-loss` / logit soft-cap / `QK-norm` → 直接砍掉 softmax 锐化这条增长通道；σReparam；μP（或至少 output layer 按 $1/\sqrt{d}$ 缩放 + 按宽度缩 LR）；per-layer / AGC 式自适应裁剪（比全局固定阈值稳得多）；decoupled weight decay（注意：它会通过尺度方向**抬高**报出来的 `grad norm`，别把它误读成不稳定）。

</details>

---

## 一句话总结

`grad norm` 的缓慢上涨，是 `“CE 无下界”` × `“LN 堵死了便宜的放大路径”` × `“难样本重尾化”` 三者的必然产物；它是**训练仍在推进的标志**，不是训练的障碍。它的**无界性**来自目标函数的无界性，而不是数值不稳定。真正要警惕的是它带来的**次级效应**：clip 长期饱和、spike、attention 退化，以及有效学习率被暗中吃掉。

---
## 🧑‍💻 User

我感觉我们不能只探究现象 虽然这可能和训练的梯度动力学有关 但是具体归因还需要一套理论支撑
有previous表明
grad norm炸开的原因绝大多数都是因为free scale太多 hidden state飞升
所以对应的reader的gradient爆炸

---
## 😀 默认助手 (7)

## 你的直觉一半是对的，但“free scale”这个词必须拆开——数学上它们的梯度符号是**反**的

你说“不能只探现象”，方向我完全同意。但要把它变成理论，第一步是承认：**transformer 里的“自由尺度”不是一个东西，而是三类，它们的梯度行为完全不同。**

| | 方向 | 损失对它 | 参数梯度对它 |
|---|---|---|---|
| **S1 精确 gauge** | LN 前的 $(W,b)\to(cW,cb)$ | 严格不变 | $\propto 1/c$（**会变小**） |
| **S2 半 gauge** | attention logit 尺度、残差流范数 $\|h\|$、embedding 的径向 | 一阶几乎不可见 | 可升可降，取决于是否饱和 |
| **S3 真正方向** | 特征方向 | 敏感 | 反映曲率 |

“free scale 太多 ⇒ hidden state 飞升 ⇒ 对应 reader 的梯度爆炸”这个命题，**只在 S2 且网络尚未饱和的区间成立**。在完全饱和区间，梯度反而趋 0。这一步区分，决定了后面所有归因。

---

## 先证明一件事：精确 gauge 下，$\|\nabla\| \propto 1/c$，不是 $c$

设 block 为 $y = Wx+b$，$u = \mathrm{LN}(y)$，$\mathcal{L}=\mathcal{L}(u)$。

做 gauge 变换 $(W,b)\to(cW,cb)$，则 $y \to cy$，而 $\mathrm{LN}(cy)=\mathrm{LN}(y)$，所以 $u$ 不变、$g:=\partial\mathcal{L}/\partial u$ 不变。LN 的 Jacobian 满足
$$J(cy) = \frac{1}{c}J(y)$$
于是
$$\nabla_{cW}\mathcal{L} = J^\top g\,x^\top = \frac{1}{c}\,\nabla_W\mathcal{L}$$

**结论：沿着真正的 gauge 方向放大 $c$，参数梯度范数是 $1/c$ 衰减的。** 同时自然有 $\langle\nabla_W\mathcal{L}, W\rangle + \langle\nabla_b\mathcal{L}, b\rangle = 0$（梯度恒垂直于尺度向量）。

所以“free scale 多 ⇒ 梯度爆炸”**不能**靠 S1 解释。S1 的真实代价是另一件事——见下面第 4 节，那是一个真定理。

---

## 能涨的是什么：一个可以拿去归因的分解式

$$\nabla_\theta\mathcal{L} = J_\theta^\top\nabla_z\mathcal{L},\qquad \|\nabla_\theta\mathcal{L}\| \le \underbrace{\|J_\theta\|_{\mathrm{op}}}_{G(t)\ \text{网络增益}}\cdot\underbrace{\|p-y\|_2}_{\epsilon(t)\ \text{损失敏感度}}$$

观测到的“缓慢单调上涨”= **$G(t)$ 涨得比 $\epsilon(t)$ 落得快**。所以归因问题被压缩成命题：**$G$ 从哪里来？** 三个候选，且它们的修法完全不同：

- **G1｜残差流范数 $\|h\|$ 增长** —— 但它被每一层的 LN 挡住（$J_{\mathrm{LN}}\propto 1/\|h\|$），所以它对往后传的梯度是**衰减**的。净效应符号不明，**必须测**，不能假设。
- **G2｜attention logit 尺度** —— 这是**唯一真正没有被任何东西归一化的尺度**（$QK^\top/\sqrt d$ 的量级随 $Q,K$ 增长而增长）。softmax 对尺度敏感，logit 涨 ⇒ Jacobian 条件数变差（$\mathrm{diag}(p)-pp^\top$ 的特征值 $p_i(1-p_i)$ 在饱和时退化到 0，同时 $\partial p/\partial Q \propto K$ 反向变大）。这条线上有干净的实证：Zhai et al. 的 σReparam 论文明确把 **attention logit 幅值增长 ⇔ entropy collapse ⇔ 训练不稳** 三者绑在一起，QK-norm / logit soft-cap 是直接针对它的。**这是你直觉最接近成立的地方。**
- **G3｜曲率/特征锐化** —— (L0,L1)-smoothness（Zhang et al., ICLR'20）：$\|\nabla f(x)-\nabla f(y)\|\le(L_0+L_1\|\nabla f(x)\|)\|x-y\|$，其中 $L_1>0$ 意味着**梯度越大、局部光滑性越差**；他们直接观察到跨任务的 $\|\nabla\|\sim O(t)$。加上 progressive sharpening（Cohen et al. 2021：$\lambda_{\max}(H)\to 2/\eta$ 然后被顶住）。这条是纯 function-level 的，和 scale 无关。

**顺带修正一个证据层面的细节**：LLM 里被观测到的“hidden 飞升”主要是 **sparse massive activations**——不到 0.1% 的通道涨到 $\sim10^3$ 倍，充当近似常量的 bias（Sun et al. 2024）。这**不是**整体范数各向同性膨胀，而是少数固定方向。所以“hidden state 飞升”这个机制要精确成“少数 outlier 通道变成准常量 bias”，那它影响梯度的方式就变成“改变了残差流的有效秩”，而不是“各向同性放大”。

---

## Reader 的梯度：这里我可以给你一个精确结论，而且它直接支持/修正你的直觉

两种 reader，结论完全不同：

**版本 A（有 final LayerNorm）**：$z = W_U\,\mathrm{LN}_f(h_L)$，$\|u\|_2=\sqrt d$（$\gamma=1,\beta=0$，**被 LN 钉死**）
$$\left\|\nabla_{W_U}\mathcal{L}\right\|_F = \|p-y\|_2\cdot\|u\|_2 = \|p-y\|_2\sqrt d$$
**与 $\|h_L\|$ 完全无关。** hidden state 飞升不会让 reader 梯度爆炸。

**版本 B（无 final LN，或 $\gamma$ 自己在涨）**：$z = W_U h_L$
$$\left\|\nabla_{W_U}\mathcal{L}\right\|_F = \|p-y\|_2\cdot\|h_L\|_2$$
**线性于 $\|h_L\|$。你的直觉精确成立。**

同时把反向信号送进残差流时：$\partial\mathcal{L}/\partial h_L = J_{\mathrm{LN}}^\top W_U^\top(p-y)$，$\|J_{\mathrm{LN}}\|\sim\sqrt d/\|h_L\|$。

于是有一个很干净的定性图景：

> **$\|h_L\|$ 上涨对 reader 是放大器，对 writer（往回流）是衰减器。** 谁的梯度涨，取决于哪一边被归一化保护住了。

**这就直接给了一个探针**：如果你怀疑“hidden 飞升导致 reader 爆炸”，那首先要测的是 $\|u\|=\|\mathrm{LN}_f(h_L)\|$ 是否恒定。它恒定 ⇒ 你看到的 reader 梯度增长**不是** hidden 范数造成的，去别处找（大概率是 $\|p-y\|$ 的结构变化，或者 reader 之外的层）。

---

## “无界”这个性质，理论上有两个来源，速率不同

**(a) 目标无下界 ⇒ runaway 尺度。** CE 没有有限极小点。这类问题有定理：**Lyu & Li (2020)** 对可分数据上的齐次网络证明 $\|W\|\to\infty$、方向收敛到最大间隔解、损失 $\to0$。注意——**同一个定理里，梯度是 $\to 0$ 的**。所以“无界地涨”和“梯度爆炸”在渐近意义上**是互斥的**。这是对你命题最硬的一次反驳：纯 runaway scale 给的是 $L\to0,\|\nabla\|\to0,\|W\|\to\infty$。

**(b) 有效学习率定理 ⇒ 自由尺度的真实代价。** 对尺度不变损失，van Laarhoven / Arora-Li-Lyu / Hoffer 这条线给出：$\nabla_W\mathcal{L}$ 关于 $W$ 是 $-1$ 次齐次的，GD 更新恒 $\perp W$，所以函数空间里的步长
$$\text{effective step}\ \propto\ \frac{\eta}{\|W\|}$$
**自由尺度增长 1 倍，有效学习率就掉一半。** 再叠加平坦方向上的梯度噪声扩散（$\|W\|^2 \sim \eta t/\lambda$ 或 WD 平衡点），你得到 $\|W\|\sim\sqrt t$ 或 $\log t$。

**这两个加起来恰好给出“无界但缓慢”的 log/sqrt 型缓坡**——而且注意它们给出的**都不是梯度爆炸**，是“梯度随 $1/c$ 衰减 + 有效 LR 衰减”。

---

## 所以缺的是什么理论——诚实说

**没有定理说 transformer 的 grad norm 必须单调上涨。** 现在能撑起“free scale ⇒ 爆”的最好材料是三块，但都只说了一半：

1. **μP coordinate check（Yang et al.）**：标准 parametrization 下，宽度变大时 logits/特征更新会偏离初值量级，需要 $\mu$P 才能钉住。这是**关于宽度**的定理，且它讲的是**函数量级漂移**，不是参数梯度的单调增长。
2. **Wortsman et al. (2024)**：明确**假设**不稳定源于 attention logits 和 output logits 的增长，用 QK-norm + z-loss 验证。这是 hypothesis，不是 theorem。
3. **σReparam（Zhai et al. 2023）**：attention logit 幅值 ⇔ entropy collapse 的实证绑定。

也就是说：**“free scale ⇒ logit 漂移 ⇒ 不稳定”这条链是被支持的；“free scale ⇒ hidden 范数升 ⇒ 参数梯度单调爆炸”这条链在数学上是错的（符号反了），在实证上是未经区分的。**

---

## 如果你真要做归因，用这个判别准则（从上面的理论直接推出来）

对每一层 $l$ 记录三个量：

| 量 | 含义 |
|---|---|
| $\|\nabla_{W_l}\mathcal{L}\|\cdot\|W_l\|$ | **函数空间梯度范数**（gauge 不变的量） |
| $\|W_l\|$ 的走向 | 是否在 drift |
| $\|p-y\|_2$ 的走向 | $\epsilon(t)$ |

判据：

- $\|W_l\|$ 在**降**（被 WD 拉）而 $\|\nabla_{W_l}\|$ 在**升**，且 $\|\nabla\|\cdot\|W\| \approx \text{const}$ ⇒ **纯 S1 gauge drift，函数上什么都没发生，无害**。你不用管。
- $\|\nabla_{W_l}\|\cdot\|W_l\|$ 在**升** ⇒ 真实锐化（G2/G3）。这才是要处理的，处理对象是 logit 尺度（QK-norm / soft-cap / z-loss）或优化（LR、clip、SAM 类），**不是“把 grad norm 压下去”**。
- 分 token 再切一刀：把梯度按 loss 分位数分组（top-1% / top-10% / 其余）。如果增长**全部**集中在高 loss 子群 ⇒ 支持“难样本重尾 + 特征锐化”（G3）；如果**均匀**分布在全体 ⇒ 支持“尺度漂移/增益”（G1/G2）。
- 测 $\|u\|=\|\mathrm{LN}_f(h_L)\|$ 是否恒定：恒定 ⇒ reader 被保护，你观察到的 reader 梯度增长不能归因于 $\|h\|$；不恒定 ⇒ 你找到了真正的漏洞（缺 final LN / $\gamma$ 失控 / 参数化选择不当）。

---

## 我的结论，直说

你的直觉抓住了正确的东西（**尺度自由度是核心变量**），但把因果链的方向接反了：

> **自由尺度增长的直接后果，是参数梯度按 $1/c$ 衰减、有效学习率按 $1/\|W\|$ 衰减——不是爆炸。**
> **能真正让 $\|\nabla\|$ 涨的，是 function-level 的增益/曲率增长：attention logit 尺度漂移（未被归一化的那部分）和特征锐化。**
> **而 reader 的梯度是否随 hidden 范数爆炸，取决于 final LayerNorm 是否在钉住 $\|u\|$——这一条可以用两行计算判定，是整条链里唯一可以证死的环节。**

也就是说：真正可证的是 **“free scale ⇒ effective LR 衰减 ⇒ 你被迫用 cosine decay 去补偿”**；而 **“free scale ⇒ grad norm 爆炸”** 目前是个被过度概括的实证关联，它需要附加“在 LN 保护之外 + 未饱和 + attention logit 尺度漂移”这几个条件才成立。归因实验应该做的事情，正是把“gauge 的 $\|∇\|\|W\|$ 不变”和“真实增益的 $\|∇\|\|W\|$ 上升”分开——这两件事在日志上看起来一模一样，但在理论上是完全不同的对象。
