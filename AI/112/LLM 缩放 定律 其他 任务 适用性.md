# LLM 缩放 定律 其他 任务 适用性

## 🧑‍💻 User

有个问题，缩放定律在LLM之外的任务也成立吗
ref：A Universal Theorem on Learning Curves, 1992
Statistical Mechanics of Learning from Examples, 1993
Deep Learning Scaling Is Predictable, Empirically, 2017

---
## 😀 默认助手 (7)

## 先给结论

要分三层看，三层的答案不一样：

| 层次 | 是否成立 | 说明 |
|---|---|---|
| **幂律形状**（loss 对 n/N/D 呈 $L = L_\infty + a\,n^{-\alpha}$） | **相当普遍** | 视觉、语音、翻译、强化学习、核方法、蛋白质、气象，甚至人类技能习得都能看到 |
| **指数普适**（$\alpha$ 是同一个数） | **不成立** | 指数是任务和数据分布特有的，属于“普适类”级别的量 |
| **可解析分解 + 跨 4~5 个数量级外推**（$E + aN^{-\alpha} + bD^{-\beta}$） | **只在 LLM 上比较靠谱** | 其他任务上更常见的表现是断幂律、平台期、双下降 |

所以严格说：**“学习曲线是幂律”是个老结论，不依赖 Transformer；“缩放定律”作为可用来做资源规划的定量工具，才是 LLM 时代的产物。**

---

## 你给的这三篇文献实际上分别说明什么

先小勘误：Amari 那篇是 **1993 年** *Neural Networks*（1992 年 IJCNN 有一篇前身 *Universal property of learning curves under entropy loss*）；Seung–Sompolinsky–Tishby 是 **1992 年** *Phys. Rev. A* 45, 6056。年份对调了，内容没问题。

<details>
<summary><b>1. Amari 1993：定理说的是“指数 = 1”，而不是“指数随便”</b></summary>

对**正则（可识别）**统计模型，用极大似然类估计，泛化误差有渐近展开：

$$E_n = E_\infty + \frac{c}{n} + o\!\left(\frac{1}{n}\right),\qquad c=\tfrac12\operatorname{tr}\!\left(I^{-1}G\right)$$

（$I$ 是 Fisher 信息矩阵）。**指数硬编码为 1，与模型结构无关**——这才是“universal”的含义。

关键反面：深度网络是**奇异模型**（参数到函数的映射不可识别），Watanabe 的奇异学习理论给出 $E_n - E_\infty \sim n^{-\lambda}$，$\lambda$ 是 real log canonical threshold，通常 $<1$。**这正是为什么 LLM 的指数是 0.05~0.3 而不是 1。** 换句话说，这条定理反而预言了大模型会偏离它自己。
</details>

<details>
<summary><b>2. Seung–Sompolinsky–Tishby 1992：形式普适、指数不普适</b></summary>

teacher–student 设定下的统计力学，得到 $\epsilon_g \sim n^{-\alpha}$，其中 $\alpha$ 由 **teacher rule 的平滑性 / 模型类 / 噪声**决定。不同规则属于不同 universality class：学一个感知机规则、学一个 committee、学一个带噪声的规则，指数各不相同。

这是最早把“**幂律是普遍的，但指数是任务特有的**”讲清楚的工作——也就是你现在这个问题的标准答案。同时它给出了样本复杂度上的相变结构，而不只是一条平滑曲线。
</details>

<details>
<summary><b>3. Hestness et al. 2017：跨四个领域验证，并给出方法论</b></summary>

机器翻译、语言建模、图像、语音**四个领域都出现幂律**。两个观察尤其重要：

- **架构改进只平移误差，不改斜率**（改变截距 offset，不改 exponent）
- 模型尺寸随数据量**次线性**增长

它就是 Kaplan 2020 / Chinchilla 2022 的直接前身。注意它自己写了 “power-law exponents yet to be explained by theoretical work”——2017 年时理论还是空白的。
</details>

---

## LLM 之外的实证清单

| 领域 | 代表工作 | 备注 |
|---|---|---|
| 图像分类 | Sun et al. 2017《Revisiting Unreasonable Effectiveness of Data》；Zhai et al. 2022（ViT） | 数据量幂律非常干净 |
| 语音识别 | Droppo & Elibol 2021 | 声学模型上的幂律 |
| 机器翻译 | Ghorbani et al. 2021 | 指数比 LM 陡 |
| 强化学习 | Hilton et al. 2023 | 样本效率随模型/数据幂律改善 |
| 核方法 / KRR | Canatar, Bordelon, Pehlevan 2021 | **理论可预测指数**，见下 |
| 蛋白质、气象、材料 | 近两年一批 | 多为 PEFT / 大模型迁移场景 |
| 人类技能学习 | Newell & Rosenbloom 1981 的 power law of practice | 形状同构，指数约 0.3~0.5，挺耐人寻味 |

---

## 为什么会成立：真正的机制不在 Transformer

误差分解是所有解释的共同骨架：

$$L(n, N) = L_\infty + \underbrace{A\,N^{-\alpha_N}}_{\text{近似误差}} + \underbrace{B\,n^{-\alpha_n}}_{\text{估计误差}}$$

而**指数从哪来**有几种等价刻画：

- **非参数 minimax 率**：目标函数平滑度 $\beta$、数据内在维度 $d$，则
$$n^{-\frac{2\beta}{2\beta+d}}$$
内在维度越高、目标越不平滑 ⇒ 指数越小、越难 scale。这是最干净的“指数为什么因任务而异”的答案。
- **核谱衰减 + 任务对齐**：核特征值 $\lambda_i \sim i^{-(1+\gamma)}$，目标在各特征方向上的系数衰减速度决定指数，且**多段幂律**天然出现（不同尺度由不同特征段支配）——Canatar et al. 2021。
- **四个 regime**（Bahri et al. 2021）：variance-limited、resolution-limited 等，解释了为什么 log-log 图上往往不是一条直线而是折线。
- **内在维度视角**（Sharma & Kaplan 2020）：指数与数据流形维度挂钩。

一句话：**幂律来自“自然数据有低维可压缩结构”这一事实，任何学习系统撞上它都会产生幂律。** LLM 只是这个现象最干净、最贵的实例。

---

## 什么时候会不成立

<details>
<summary><b>点击展开：已知的失效模式（这部分比“成立”更重要）</b></summary>

1. **饱和**：撞上 Bayes error / 数据上限后曲线变平。强行用单幂律拟合，两端都会拟合得不准。
2. **双下降**：随模型尺寸误差先降后升再降（Belkin et al. 2019），非单调，幂律叙事不适用。
3. **断幂律 / S 形**：Alabdulmohsin et al. 2022 明确论证单一幂律在宽阔区间上不成立，需要 broken power law 或 sigmoidal 形式。
4. **数据质量而非数量**：数据去重、data pruning 会**直接改变指数**（Sorscher et al. 2022）；数据受限时重复数据带来收益递减（Muennighoff et al. 2023）。
5. **指标假象**：连续 loss 平滑下降，但阈值化的 accuracy 会呈现“涌现”跳变——很多“涌现能力”是度量方式的产物（Schaeffer et al. 2023）。
6. **任务类型依赖**：组合性、算法性、长程精确推理，或内在维度高、缺乏结构的任务，指数很差甚至直接平台化；表格数据上 GBDT 的 scaling 也明显偏弱。
7. **超参与小 regime**：小模型/小数据阶段常出现伪平台，不遵循大 regime 的幂律，容易被误读为“定律失效”。
8. **外推崩溃**：1~2 个数量级内拟合可信，跨 4~5 个数量级经常系统性偏离。

</details>

---

## 自己动手验证的清单（Hestness 方法论的真传）

1. 先估 $L_\infty$（误差下界），然后画 $\log(L - L_\infty)$ vs $\log n$，**不要**直接画 $\log L$。
2. 至少覆盖 3 个数量级、8 个以上点；少于这个量级谈“定律”没意义。
3. 看残差：有系统性弯曲就换断幂律，不要拉直线。
4. **留出一个数量级的点做真正的外推验证**——这是区分“拟合得好”和“定律成立”的唯一办法。
5. 报告指数 + 置信区间，而不是“缩放定律成立/不成立”这种二元判断。

---

## 一句话总结

**幂律学习曲线是统计学习理论里早就被推导出来的普遍现象（1990s 就证明了它源自模型平滑性与数据结构），LLM 不是例外，只是最显眼的实例；但“指数普适”和“跨任务可外推”都不成立——每个任务有它自己的普适类，指数是那个类的标签，而不是自然常数。**
