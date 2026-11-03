# 深度解析大模型训练异常：RMSNorm盲区、下投影奇异值失配与瞬时尖峰捕获全链路诊断方案

## 🧑‍💻 User

为什么训练transformer时候，ffn的down proj出现异常值会反映在grad norm spike上？

---
## 😀 默认助手 (7)

# 简短回答

因为 FFN down proj 的梯度**恰好是两个可能爆掉的东西的外积**，而全局 grad norm 又是**对所有参数求平方和**——数学上对单点异常值极其敏感。所以 down proj 里出现异常值时，grad norm 尖峰是它必然的代理信号，并非巧合。

---

## 1. down proj 的梯度公式：一个纯外积

FFN（SwiGLU 或 GeLU）写成：

$$
h = W_{down}\,a,\qquad a = \mathrm{SiLU}(W_{gate}x)\odot W_{up}x,\qquad x' = x + h
$$

记上游梯度 $\delta = \partial L/\partial h \in \mathbb{R}^{d}$，则

$$
\frac{\partial L}{\partial W_{down}} = \delta\, a^{\top}
\;\Longrightarrow\;
\left\|\frac{\partial L}{\partial W_{down}}\right\|_F = \|\delta\|_2 \cdot \|a\|_2
$$

（单 token 时**严格等式**，rank-1；一个 batch 是若干 rank-1 之和，量级由最大那一项主导。）

这个式子的关键含义是：**down proj 的梯度大小等于 $\|\delta\|$ 和 $\|a\|$ 的乘积，中间没有任何衰减环节。**

对比一下同一层的 up proj：

$$
\frac{\partial L}{\partial W_{up}} = \Big(W_{down}^{\top}\delta \odot \sigma'(z)\Big)x^{\top}
$$

它至少被激活函数导数 $\sigma'$ 压了一道，还可能被小的 $W_{down}$ 压一道。down proj 没有这道缓冲。

所以：**FFN hidden 空间里任何一个维度爆掉($\|a\|$ 变大)，或残差流任何一路梯度变大($\|\delta\|$ 变大)，都会以乘性方式原封不动地出现在 down proj 的梯度里。**

---

## 2. 单个异常值能让全局 grad norm 尖峰的原因

全局 grad norm 是所有参数梯度范数的平方和：

$$
\|G\|_2^2=\sum_{\theta} g_\theta^2,\qquad \frac{\partial \|G\|_2}{\partial |g_\theta|}=\frac{|g_\theta|}{\|G\|_2}
$$

这是**平方和聚合**，对尾部极端敏感：

- 一个坐标从 $1$ 变成 $10^3$，平方后是 $10^6$；哪怕模型有 $10^{10}$ 个参数、其余都是 ${\sim}1$，单点也能把全局 norm 抬高 3 个数量级。
- 一旦它主导，$\|G\|\approx |g_{\max}|$，于是日志上的 grad norm 呈现出**突兀的尖峰**（对数坐标下是一条竖线）。
- 一个 batch 中一个序列里的一个 token，$\|a\|=10^3$ 而典型值是 $1$，这一项对 $\|G_{down}\|_F$ 的贡献是其他所有 token 之和的 $10^3/\sqrt{T}$ 倍——$T$ 再大也压不住。

反过来，如果用 $\ell_\infty$ 或 top-$k$ 坐标监控，你会看到“某一行的值很大”；用 $\ell_2$ 求全局，你看到的就是“尖峰”。**两者其实是同一件事的两种读数。** 这也是为什么很多人先看到 grad norm spike，回头查 per-tensor / per-layer 才发现是某一层 down proj 的某一行或某一列。

---

## 3. down proj 的梯度异常更常见的原因

有三个结构性原因：

**(a) 它是块内唯一“未被压住”的张量。** Attention 里有 softmax（天然有界），进入每个子层前有 RMSNorm（归一化），唯独 FFN hidden 维 $a$ 只经过 SiLU/GeLU——**上无界**。而 $W_{down}$ 正是从这个无界空间读出到残差流的读出矩阵。任何 FFN 内部数值膨胀，都会在这里被“折算”成梯度异常和权重异常。它是整个 block 的**瓶颈汇合点**。

**(b) 梯度坐标高度集中，不做平均。** $G[i,j]=\delta_i a_j$。若残差流存在 outlier 通道（几乎必然存在，attention sink 那几路），则对应那一行 $\delta_i$ 很大；若某个 hidden 神经元有 massive activation，对应那一列 $a_j$ 很大。两者交叉的那个元素就是 outlier，整行/整列被它主导。**这解释了为什么观测到的往往不是“随机噪声”，而是 $W_{down}$ 里特定几行/几列的异常。**

**(c) 它直接写进残差流，且后面紧跟 RMSNorm。** $h$ 是直接加到残差上的。一旦 $h$ 某一路变大，下一个 RMSNorm 的 RMS 就被这一路撑大，**其余所有维度被同比压小**——动态范围被一个 outlier 独占，信息量骤降。这既是 loss spike 的来源，也让下一轮的 $\delta$ 更离谱。

---

## 4. 权重异常与梯度尖峰的自激循环

题设里“异常值”可能是权重、激活、或梯度本身，三者在上面公式里是同一个闭环上的一环：

$$
\underbrace{W_{down}\text{ 某行/列变大}}_{\text{权重异常}}
\;\Rightarrow\;
\underbrace{h\text{ 输出 outlier}}_{\text{前向}}
\;\Rightarrow\;
\underbrace{\text{loss/RMSNorm 被污染}}_{\text{下游 }\delta\text{ 变大}}
\;\Rightarrow\;
\underbrace{\|G_{down}\|=\|\delta\|\|a\|\text{ 尖峰}}_{\text{梯度}}
$$

$$
\text{同时}\quad \|G_{up}\|\propto \|W_{down}^{\top}\delta\| \;\Rightarrow\; \text{up proj 也跟着尖峰}
$$

也就是说：**权重异常是过去梯度尖峰留下的结果，也是下一次梯度尖峰的原因。** 这解释了为什么在实际训练日志里，权重 outlier、activation max、grad norm spike、loss spike 往往是**同一层、相隔几步**一起出现；grad norm 只是最早、最便宜的那个检测器。

<details>
<summary>为什么 up proj 的梯度也被带飞（展开）</summary>

$\|G_{up}\|$ 的上界里含 $\|W_{down}^{\top}\delta\|$。所以 $W_{down}$ 的谱范数一旦被 outlier 抬高，up proj 的梯度同步骤升，而 up proj 的权重增长又会让 $a$ 变大，回到循环起点。这就是为什么看 per-layer grad norm 时常看到"整层同时尖峰"，而不只是单个张量——真正的**源头**要靠 $|a|_{\max}$ 或 $|W_{down}|_{\max}$ 定位。
</details>

---

## 5. Adam 把瞬时尖峰写成长期权重异常

$$
w \leftarrow w - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}
$$

对**二阶矩还很小**的坐标，突然来一个大梯度时，Adam 的更新幅度约等于 $\eta$（与梯度大小无关）。这意味着：

- 一个 outlier token 的单步大梯度，可以**一步之内**把一个还没"热"起来的坐标推 $\sim\eta$ 的距离；
- 而它要花很多步才被 weight decay 或后续小梯度拉回去；
- 于是**一次数据异常被永久地写入权重**，之后持续污染前向，反复激发新的尖峰。

这也说明为什么用 AdamW 的大模型里，grad norm spike 的**后果**比 SGD 严重：梯度尖峰不是被平均掉，而是被**记忆**下来。

<details>
<summary>数值精度的放大器作用</summary>

- bf16 只有约 3 位十进制有效位。当 $W_{down}$ 某一行范数涨到 $10^2$ 量级，该行其余元素的 bf16 量化台阶也变成 $O(1)$ 量级——**其余参数的更新直接被舍入掉**，表现为"梯度算出来了但没学进去"，同时 outlier 坐标的相对误差仍然很大。
- 反向的 bf16 matmul 里，异常值造成的累加误差量级与最大值成正比。
- 实践上：master weights 用 fp32，矩阵乘用 bf16，并单独监控 $|a|_{\max}$。
</details>

---

## 6. 如何定位和应对

<details>
<summary>诊断清单</summary>

1. **分层、分张量记 grad norm**，不只记全局。定位到 `layers.i.mlp.down_proj` 后，再看是**哪一行/哪一列**。
2. **记 $|a|_{\max}$ 与 $|h|_{\max}$**（FFN hidden 与 FFN 输出）。如果是 $|a|$ 先涨 → 源头在 up/gate；如果是 $|h|$ 先涨 → 源头就是 $W_{down}$ 自己。
3. **记 $|W|_{\max}$ 与权重谱范数**，看 outlier 是"新出现"还是"长期存在、只是偶尔被激活"。
4. 对齐时间轴：依次看 $|a|_{\max}$、$|W_{down}|_{\max}$、grad norm、loss，通常 grad norm spike 早于 loss spike 1–若干步。
5. 注意 **DDP/FSDP 的全局 norm 会把某一个 rank 的坏 batch 摊到全队**，以及 gradient accumulation 窗口内只要一个 micro-batch 坏就尖峰。
</details>

<details>
<summary>缓解手段（按性价比）</summary>

- **down proj（及所有残差输出投影）用缩放初始化**（如 GPT-2 的 $1/\sqrt{2N}$、或 Megatron 的 scaled init）。直接压住 $\|W_{down}\|$，同时压住前向 $h$ 和反向的 $W_{down}^{\top}\delta$——性价比最高。
- **每层/每张量裁剪**（per-tensor clipping）而非只做全局裁剪：全局 norm 被 outlier 主导时，全局裁剪会把所有正常层的学习率一起打死。
- **对 embedding / 输出层加 z-loss 或 logit soft-cap**，降低 $\delta$ 的尾部。
- **更长 warmup、更低峰值 lr**；spike 发生时 warmup 重启或回滚到 checkpoint + 跳过数据分片（data skipping）。
- **spike 后重置 optimizer state**（丢掉被污染的 $v$），或对该层单独降 lr。
- 结构性手段：**QK-norm / sandwich-norm**、限制 massive activation（如对 $a$ 做 clamp 或 gating 归一化）。
- 数据侧：去重、过滤异常长/异常重复样本——很多 spike 追根究底是单个坏 batch。
</details>

---

**一句话总结：** down proj 的梯度是 $\|\delta\|\cdot\|a\|$ 的外积，没有衰减环节，而全局 grad norm 是平方和、对尾部单点极端敏感；再叠加 Adam 的"尖峰固化"和 RMSNorm 的"outlier 独占动态范围"，异常值→权重 outlier→前向污染→梯度尖峰就成了一个自激闭环。grad norm spike 只是这个闭环最便宜、最早的观测窗口。

---
## 🧑‍💻 User

那去仔细观察逐层的rms norm前后极值，观察残差流通道极值，topk通道能量，分析downproj的矩阵奇异值、能量谱，能发现问题吗

---
## 🧑‍💻 User

那去仔细观察rms norm通道极值，分析downproj的矩阵奇异值、能量谱，能发现问题吗

---
## 🧑‍💻 User

那去仔细观察rms norm极值，观察残差流通道极值，topk通道能量，分析downproj的矩阵奇异值、能量谱，能发现问题吗

---
## 😀 默认助手 (7)

# 结论先行

**能发现，但很可能只发现“慢性”的那一半，漏掉“急性”的那一半；而且你列的四个量里有一个（post-RMSNorm 极值）在数学上几乎测不到东西，另一个（down proj 的 SVD）测的东西和“稀疏异常值”是失配的。**

核心原因是：你列的观测大多是**状态量**，grad norm spike 是**速率事件**。状态量能反映“已经被写进权重的东西”，看不到在单个 step 中产生的瞬时异常（一次坏 batch、一个 token）。而这类 spike 恰恰常常就是瞬时的。

下面逐项说明。

---

## 1. post-RMSNorm 极值：数学上的系统性盲区

写成 $y = g \odot x/\mathrm{rms}(x)$，其中 $\mathrm{rms}(x)^2 = \frac1d\sum_i x_i^2$。取 $g\equiv 1$：

$$
\|y\|_2^2 = \frac{\|x\|_2^2}{d\cdot \mathrm{rms}(x)^2} = d \quad(\epsilon \to 0)
$$

**整个向量的能量是常数 $d$。** 由此产生两个后果：

- $\max_i|y_i| \in [1,\sqrt d]$，且会**饱和**：一旦一个通道吸收了所有能量，这个量就到头了。它对“继续恶化”不敏感，所以作为升级或恶化检测器很差。
- 更严重的是**尺度不变性**：$x \to cx$ 完全不改变 $y$。残差流整体放大 100 倍，在 post-norm 上**完全不可见**。

所以 post-RMSNorm 的 `max` 基本只是在测 $g$ 和能量的份额，不测尺度。真正有信息的统计量是**有效维数**：

$$
\mathrm{PR}(y)=\frac{\left(\sum_i y_i^2\right)^2}{\sum_i y_i^4}\in[1,d]
$$

一个通道独占时，$\mathrm{PR}\to1$；这是有界但**单调**的集中度度量。这也就是你想要的“top-k 通道能量”的正确归一化版本。

而且，注意分解式：

$$
y_j^2 = g_j^2\cdot \underbrace{\frac{x_j^2}{\mathrm{rms}(x)^2}}_{\text{份额 } s_j,\ \sum_j s_j = d}
$$

post-norm 的通道能量 = **慢变量 $g_j$** × **快变量份额**。也就是说，post-norm 上的通道极值其实是“这个通道被 RMSNorm 增益**指定**成了 outlier”的缓慢记录，并非尖峰探测器。它的价值是：$g$ 是 outlier 通道的**长期记忆**，通道索引的稳定性可以用来区分良性/恶性异常（见下）。

**→ 建议：post-norm 只看 $\mathrm{PR}(y)$、$\min_i|y_i|$（死通道数）和 $g$ 的谱，不看 max。**

---

## 2. pre-RMSNorm 极值 / 残差流通道极值：有效，但有一个关键混淆

有效，而且是主要观察对象。但**不能直接用 $\|x\|_\infty$ 判断恶性**，因为残差流里确实有大量**良性** outlier 通道（attention sink 类：索引稳定、训练早期就出现、充当 no-op/bias、消融后 loss 变化很小）。

区分良性/恶性要靠三条，单张快照一条都做不到：

| 判据 | 良性 | 恶性 |
|---|---|---|
| 通道索引随时间的稳定性 | 固定 | 新生、游走 |
| 与 loss/grad norm 的时间相关性 | 无 | 有，且滞后 $<0$ |
| **干预**（clamp 该通道或 zero 该行）后 loss | 不变 | 变差/尖峰消失 |

第三条是**唯一决定性**的。所以这套分析必须包含“拿保存的 checkpoint 做 replay + clamp”这一步，而不只是看曲线。

顺便修正一个常用的错误指标：只看 $\|x\|$ 不够，因为有 RMSNorm 在中间重新归一化。**真正该测的是“相对写入”**：

$$
r_{l,j}=\frac{|h_j^{(l)}|}{\mathrm{rms}(x^{(l)})},\qquad r_l=\frac{\|h^{(l)}\|}{\|x^{(l)}\|}
$$

因为 $\|x'\|^2=\|x\|^2+2\langle x,h\rangle+\|h\|^2$，网络对 $x$ 的绝对尺度不敏感、只对这个比值敏感。$r_{l,j} \gg O(1)$ 就是“这一层往通道 $j$ 里写了一发远超 RMS 的东西”——**这个量既尺度不变、又直接对应前向危害，还能定位到通道**。它应该作为你的主要指标。

---

## 3. down proj 的奇异值/能量谱：方法上失配

三个问题，从根本到具体：

**(a) 稀疏现象不能用密集聚合量去测。** $\sigma_{\max}$ 是“最坏方向”的增益，对所有方向一视同仁。而你有的是**稀疏异常**（几行/几列）。数学上可以证：

$$
\sigma_{\max} \ge \max\Big(\max_i\|W_{i,:}\|_2,\ \max_j\|W_{:,j}\|_2\Big)
$$

所以 outlier 行**确实会出现**在 $\sigma_{\max}$ 里——但同时它会被 $d_{ff}$ 个正常坐标的密集贡献淹没（未归一化的随机矩阵谱范数天然有 $O(\sqrt d+\sqrt{d_{ff}})$ 的**背景底噪**），反过来，一个良性但密集的方向也会把 $\sigma_{\max}$ 抬得很高却完全无害。**稀疏成因要直接看 per-row / per-col / per-coordinate 范数**，那是 $O(1)$ 开销且带定位信息的统计量，而 SVD 是 $O(d\,d_{ff}\min)$ 且不带定位。

**(b) 只看权重无法回答“是否正在被激发”。** 前向输出能量是

$$
\mathbb{E}\|W_{down}a\|^2=\mathrm{tr}\!\left(W_{down}KW_{down}^{\top}\right),\qquad K=\mathbb{E}[aa^{\top}]
$$

若 $W=U\Sigma V^\top$，则危险量是 $\sigma_i^2\lambda_i$，即**大奇异值必须对齐到高方差激活方向**才有害。$\sigma_1$ 很大但 $v_1$ 上 $\lambda_1\approx0$，完全无害。所以你要的谱属于 $W_{down}KW_{down}^\top$，而非 $W_{down}W_{down}^\top$。同理反向的增益是 $\|W_{down}^\top\delta\|$，取决于 $\delta$ 的对齐。

$K$ 是 $d\times d$，累积成本 $\sim d^2$，每层每 $N$ 步算一次完全可负担。

**(c) 静态谱测不到瞬时事件。** 这是最关键的。如果 spike 是单个 token 触发，那么尖峰前后的 checkpoint 里所有静态指标**一模一样、完全正常**。你会得出“没发现问题”的结论，而问题真实存在。

<details>
<summary>“能量谱”两种理解，用途完全不同，别混</summary>

- **$W_{down}$ 的 $\sigma_i^2$ 衰减谱**：测**容量/有效秩**。秩塌缩是“模型表达能力下降”的问题，与稳定性只有弱相关。可以看，但别指望它解释 spike。
- **残差流通道协方差的能量谱**：测表示有效维数，与 $\mathrm{PR}$ 同类。对“模型是否在退化”有用，对定位 spike 无用。

两者都不是尖峰探测器。
</details>

---

## 4. 你列的四个量都缺的那个分解

既然 $\|G_{down}\|_F=\|\delta\|_2\|a\|_2$，就直接按层、按 step 记录 $\|\delta^{(l)}\|$ 和 $\|a^{(l)}\|$：

- $\|a\|$ 先涨 → 源头在 gate/up proj、激活无界膨胀；
- $\|\delta\|$ 先涨 → 源头在下游（loss 尺度、RMSNorm 把 outlier 放大、数值精度）。

这样一个 spike 立刻被二分成前向还是反向，而且不用额外算任何矩阵。**这是信噪比最高的一步，成本几乎为零。**

还有两个必补的量：

**per-token 而非 per-batch 的极值。** 用 `max` / `p99.99` over (batch × seq)，绝不用均值——一个 $10^3$ 的 token 在 $10^5$ 个 token 里，均值完全看不见。同时记 token 位置和样本 id，才能 replay。

**单步增量 $\Delta W$。** Adam 下 $|\Delta w_i| = \eta|\hat m/(\sqrt{\hat v}+\epsilon)| \lesssim \eta$，所以：

$$
\text{“Adam 饱和坐标数”} \; N_\eta=\#\{i: |\Delta w_i| > 0.9\eta\}
$$

正常训练时它很平稳；在 outlier **产生的那一步**会突跳——因为梯度尖峰会把一个原本 $\approx 0$ 的坐标一步推到 $\sim\eta$。这个量把“瞬时事件”变成了一个**永久留在日志里的痕迹**，正好补上静态分析的盲区。

<details>
<summary>分布式下的一个陷阱</summary>

Tensor parallel 下 down proj 是 row-parallel，$d_{ff}$ 维切分到各 rank：**massive activation 只活在其中一个 rank 上**，各 rank 的 per-tensor 统计都可能“看起来正常”，而 all-reduce 后的输出爆掉。所以 per-channel 统计要在 all-gather 之后做，或者明确记“哪个 rank 出现了最值”。同理 FSDP/DDP 的全局 norm 会把某一个 rank 的坏 batch 摊到全队，必须记 per-rank。
</details>

---

## 5. 可行方案

<details>
<summary>最小可行监控集（按性价比排序）</summary>

| # | 指标 | 粒度 | 回答什么 |
|---|---|---|---|
| 1 | $\|a\|_2,\ |a|_{\max}$ + argmax 神经元 | 每层每 step（rollmax） | 前向源头 |
| 2 | $\|\delta\|_2$ | 每层每 step | 反向源头 |
| 3 | $r_{l,j}=|h_j|/\mathrm{rms}(x)$ | 每层每 step 的 max 与 top-5 通道 | **相对写入**，尺度不变，带定位 |
| 4 | per-token $\ell_\infty$（$x$, $a$, $\delta$） | 触发时 dump | 定位到 token |
| 5 | $N_\eta$ 与 $\max|\Delta W|/\|W\|$ | 每 step | 捕捉尖峰留下的“化石” |
| 6 | $\mathrm{PR}(y_{\text{post-norm}})$, $\|g\|_\infty$ | 每层每 step | 有效维数、outlier 通道长期记忆 |
| 7 | $\mathrm{tr}(WKW^\top)$ 与 $WKW^\top$ top-1 特征对 | 每层每 $N$ 步 | 真实的放大算子 |
| 8 | $\mathrm{PR}(\hat v)$ | 每层每 $N$ 步 | 二阶矩是否被单点污染 |

**工程上最重要的一条**：全量诊断不可能每步做，所以无论是否已出现 spike，都保留一个**滚动 buffer**（最近 K 步的上述 per-channel/per-token 统计 + 原始 input_ids）；当 grad norm 超过阈值时，把**事件前的那个窗口**转储下来。这样才看得见“瞬时”事件及其时间顺序（谁先谁后才是因果线索，而不是相关性）。

然后用这个 dump 做两件事：
1. **滞后互相关**：$|a|_{\max}$、$r_{l,j}$、$\mathrm{PR}$、grad norm、loss 两两之间哪个稳定领先；
2. **replay + 干预**：把那个 batch 重跑，`clamp` 那个通道 / zero 掉 $W_{down}$ 那一行 / 只保留该 token，看尖峰是否消失。**这是唯一能把“相关”升级成“因果”的手段。**
</details>

---

## 一句话

你的方案能测到**慢变量**（$g$ 的指定、通道索引、矩阵容量、谱增益），因此能诊断“慢性 outlier 体质”；但 post-RMSNorm 极值被归一化吃掉了尺度信息，SVD 的密集最坏方向视角和稀疏异常值本质失配，而整套静态分析都看不见“单 token 瞬时事件”——**而 grad norm spike 往往正是后者**。补上 per-token $\ell_\infty$、$\|a\|$ 与 $\|\delta\|$ 的分解、尺度不变的相对写入 $|h_j|/\mathrm{rms}(x)$、以及 $\Delta W$ 饱和坐标数，再加上“触发式转储 + replay 干预”，这套观测才算完整。
