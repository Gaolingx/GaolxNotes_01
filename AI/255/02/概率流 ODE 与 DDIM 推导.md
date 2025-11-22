# 概率流 ODE 与 DDIM 推导

### 🧑‍💻 User

说说 Probability Flow ODE 与 DDIM 的严格推导？

---

### 🤖 Assistant

好 — 我把 **Probability Flow ODE（概率流 ODE）** 与 **DDIM（Denoising Diffusion Implicit Models）** 的严格推导和它们之间的联系做一个紧凑但尽量完整的数学推导说明。为方便阅读，我先把符号和常见参数说明清楚，然后按步骤推导：从前向 SDE → 反向 SDE → 概率流 ODE，再从离散 DDPM 的后验出发推导 DDIM 的确定性更新，并说明为什么 DDIM 可以看作概率流 ODE 的一种离散化 / 等价采样器。

---

## 1. 符号与背景

* (x\in\mathbb{R}^d) 为数据向量。
* 时间 (t\in[0,T])，我们用小写 (t) 表示连续时间，离散化下用索引（例如 (t_k)）。
* 前向（正向）扩散 SDE（score-based literature 常用形式）：
  [
  \mathrm{d}x = f(x,t),\mathrm{d}t + g(t),\mathrm{d}w,
  ]
  其中 (w) 为标准维纳过程（Brownian motion）。这个 SDE 在 (t=0) 生成数据分布 (p_0)，到 (t=T) 变成简单的噪声分布 (p_T)（通常接近高斯）。
* 对任意时刻的边缘密度记为 (p_t(x))，score 定义为 (\nabla_x\log p_t(x))。
* 反向时间的布朗运动标记为 (\bar w)。

在离散 DDPM 参数化中，常用：
[
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t},\epsilon,\qquad \epsilon\sim\mathcal N(0,I),
]
并有一系列 ({\alpha_t}) 或 ({\beta_t}) 定义噪声比例（离散时间设定）。DDIM 与 DDPM 在训练上共享同样的去噪网络 (\epsilon_\theta)（或等价的 (x_0) 预测器）。

---

## 2. 反向 SDE（Reverse-time SDE）的公式（经典结果）

由 Anderson（1982）和 later score-based papers（Song et al.）可得：给定前向 SDE
[
\mathrm{d}x = f(x,t),\mathrm{d}t + g(t),\mathrm{d}w,
]
其**反向时间**过程（从 (t=T) → (0)）的 SDE 为
[
\mathrm{d}x = \big( f(x,t) - g(t)^2 \nabla_x \log p_t(x)\big),\mathrm{d}t + g(t),\mathrm{d}\bar w,
]
这里的 (\mathrm{d}t) 是指沿着反向时间走（符号上不影响常数项，只要注意方向）。这个公式可以由 Fokker–Planck 方程配合 Bayes 后验推导出来；关键点：score（(\nabla\log p_t)）出现在漂移项上，负责把简单噪声推回数据分布。

**注意**：上式中噪声项系数仍是 (g(t))，而漂移中出现了 (-g^2 \nabla\log p_t)。

---

## 3. 概率流 ODE（Probability Flow ODE）的推导与形式

**目标**：找到一个确定性 ODE，使其在每个时刻的边缘分布 (p_t) 与原 SDE 的边缘分布一致（即 ODE 的解轨迹的分布随时间演化满足相同的 Fokker–Planck 方程）。这是通过“把反向 SDE 中的随机扰动平均掉”而得到的。

考虑同一前向 SDE 的伴随的概率流 ODE：
[
\frac{\mathrm{d}x}{\mathrm{d}t} = f(x,t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x).
]
**证明要点（直观）**：

* SDE 的 Fokker–Planck 方程给出密度演化（含扩散项 (\tfrac{1}{2}g^2 \Delta p)）。
* 若我们构造一个确定性流场 (v(x,t)) 满足连续性方程（continuity / Liouville 方程）
  [
  \partial_t p_t(x) + \nabla_x\cdot\big(p_t(x) v(x,t)\big) = 0,
  ]
  并且让这个 (p_t) 与 SDE 的 (p_t) 相同，则把 SDE 的 Fokker–Planck 的扩散项 (\tfrac{1}{2}\nabla\cdot(g^2 \nabla p_t)) 用漂移项产生的散度来抵消，得到
  [
  v(x,t) = f(x,t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x).
  ]
* 由此得出上面的概率流 ODE。关键是：虽然 ODE 路径是确定性的，但若初始 (x_T) 按 (p_T) 采样，沿 ODE 回溯得到的 (x_t) 的边缘分布会等于 SDE 在该时刻的 (p_t)。因此概率流 ODE 与原 SDE 在边缘分布上等价（但样本路径不同：SDE 随机，ODE 确定）。

**与反向 SDE 的关系**：反向 SDE 在漂移项中带 (-g^2 \nabla\log p_t) 并加噪声；概率流 ODE 则把噪声的影响“平均”到漂移中，变成 (-\tfrac{1}{2}g^2\nabla\log p_t)。这两个过程在边缘分布 (p_t) 上是一致的，但路径层面不同。

**score 表示**：在实际训练中我们用去噪网络估计 score (\nabla_x\log p_t(x))，或等价地预测 (\epsilon) 或 (x_0)。因此概率流 ODE 可写为（用 score 表示）：
[
\frac{\mathrm{d}x}{\mathrm{d}t} = f(x,t) - \tfrac12 g(t)^2 s_\theta(x,t),\qquad s_\theta\approx \nabla_x\log p_t(x).
]

---

## 4. 从离散 DDPM 到 DDIM：后验形式与 DDIM 更新

现在回到离散时间 DDPM 风格的设定（Ho et al.）：定义一系列时间点（下标为整数）(t = T, T-1, \dots, 0)，前向采样为
[
q(x_t|x_0) = \mathcal N\big(x_t;,\sqrt{\alpha_t}x_0,,(1-\alpha_t)I\big),
]
常见的参数关系：(\alpha_t = \prod_{i=1}^t (1-\beta_i))。

在 DDPM 的最大后验/重建形式下，训练模型去预测噪声 (\epsilon_\theta(x_t,t))，满足
[
\epsilon \approx \epsilon_\theta(x_t,t),
]
并可通过它得到对 (x_0) 的预测
[
\hat x_0(x_t) := \frac{x_t - \sqrt{1-\alpha_t},\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}.
]
（这就是常见的 `pred_x0` 公式。）

**DDPM 的马尔可夫后验**：真实后验
[
q(x_{t-1}\mid x_t,x_0)=\mathcal N\Big(x_{t-1};,\tilde\mu(x_t,x_0),\tilde\beta_t I\Big)
]
其中 (\tilde\mu) 与 (\tilde\beta_t) 有明确定义（由条件高斯推得）。在标准 DDPM 采样中用网络去估计 (\tilde\mu) 并且在采样时加入噪声（按 (\tilde\beta_t) 采样），这给出随机的逆扩散采样。

**DDIM 的关键想法**（Deterministic / non-Markovian）：
Ho 等人在后续工作中（DDIM）注意到：如果我们用训练好的 (\epsilon_\theta) 来预测 (\hat x_0)，可以直接构造一个 **确定性的** 映射从 (x_t) → (x_{t-1})，如下：

令 ( \hat x_0 = \hat x_0(x_t) )（上式），则定义
[
x_{t-1} = \sqrt{\alpha_{t-1}},\hat x_0 ;+; \sqrt{1-\alpha_{t-1}},\epsilon_\theta(x_t,t).
]
展开写出代数形式（把 (\hat x_0) 代入）得到：
[
\boxed{,x_{t-1} = \sqrt{\alpha_{t-1}}\frac{x_t - \sqrt{1-\alpha_t},\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}} ;+; \sqrt{1-\alpha_{t-1}},\epsilon_\theta(x_t,t),}
]
这个更新 **不加入随机噪声**（即是确定性的），并且在训练网络 (\epsilon_\theta) 的条件下能产生高质量样本（这就是 DDIM deterministic sampler）。上式是 DDIM 常见的实现形式。

**另一种直观表达（用 “预测 x0 再重合成” 解释）**：

1. 用 (\epsilon_\theta) 从 (x_t) 预测 (\hat x_0)。
2. 把 (\hat x_0) 投射到时间 (t-1) 的边缘（用相应的系数 (\sqrt{\alpha_{t-1}}) 与 (\sqrt{1-\alpha_{t-1}}) 重构），但用同一个 (\epsilon_\theta) 作为“伴随噪声”分量，从而得到 (x_{t-1})。这是非马尔可夫性的：因为 (x_{t-1}) 依赖于在 (t) 时对 (x_0) 的预测（与标准 DDPM 的仅依赖 (x_t) 的马尔可夫后验不同）。

---

## 5. DDIM 与 Probability Flow ODE 的联系（为什么 DDIM 可被看作 ODE 的离散化）

* 概率流 ODE 是一个确定性流，其边缘随时间变换与 SDE 相同。对概率流 ODE 进行数值积分（例如用欧拉或改进的数值方法）会得到确定性的时间步进公式，把 (x_t) 映射到 (x_{t-1})。

* Song et al.（score-based diffusion 文献）与 DDIM 的分析表明：**DDIM 的确定性更新可以看作对概率流 ODE 的一种显式离散化（或等价数值求解方案）**，尤其当我们用 (\epsilon_\theta) 精确估计 score（或 (\hat x_0) 精确）时，DDIM 的轨迹趋向于概率流 ODE 的数值解轨迹。换句话说：

  * DDPM 随机采样 ↔ 反向 SDE 的随机解（加入噪声）
  * DDIM 确定性采样 ↔ 概率流 ODE 的确定性解（或其数值近似）

* 在离散参数化（(\alpha_t)）下，DDIM 的更新式可以从对概率流 ODE 应用一个特定的时间步长和重构/预测策略得到；而引入一个控制参数 (\eta)（DDIM 原文使用）可以把 DDIM 在确定性与随机性之间平滑插值：(\eta=0) 得到纯确定性（概率流 ODE 对应）；(\eta=1) 的某些设置会得到与 DDPM 类似的随机采样行为。这个 (\eta)-扩展是通过在更新中加入额外的高斯噪声项并按比例缩放实现的（因此 DDIM 是一个包含确定性与随机两端的采样家族）。

---

## 6. 小结（把重要结果收束）

* **反向 SDE**：
  [
  \mathrm{d}x = \big(f - g^2 \nabla_x\log p_t\big),\mathrm{d}t + g,\mathrm{d}\bar w.
  ]
* **概率流 ODE**（确定性，与 SDE 在边缘分布上等价）：
  [
  \frac{\mathrm{d}x}{\mathrm{d}t} = f - \tfrac{1}{2} g^2 \nabla_x\log p_t.
  ]
  在实操中，(\nabla\log p_t) 用 score 网络 (s_\theta) 近似；对这个 ODE 做数值积分就能得到一种确定性的采样方法。
* **DDIM（确定性更新）** 在离散 DDPM 参数化下的简洁形式：
  [
  x_{t-1} ;=; \sqrt{\alpha_{t-1}},\frac{x_t - \sqrt{1-\alpha_t},\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}} ;+; \sqrt{1-\alpha_{t-1}},\epsilon_\theta(x_t,t).
  ]
  该更新可以看作是利用网络预测的 (\hat x_0) 将当前 (x_t) 映射到 (x_{t-1})，并且（在 (\epsilon_\theta) 精确时）与概率流 ODE 的数值解相吻合。

---

## 7. 实用说明与直觉

* **直觉**：概率流 ODE 给出“确定性的反向路径”——样本沿着一个没有噪声的流回到数据分布。DDIM 用离散步逼近这种流；若网络完美，DDIM 路径理论上会与 ODE 的轨迹一致。
* **为什么用 DDIM？** 它通常需要更少步数就能得到高质量样本（因为是确定性且沿着更“正确”的流）；同时可以通过 (\eta) 控制随机性来权衡多样性与速度。
* **注意**：严格数学证明需要保证 (p_t) 的光滑性、score 的良好估计以及数值离散误差项的界定；以上呈现的是常见且实用的推导与直观链接（与文献中更严谨的 Fokker–Planck 推导和数值分析一致）。

---

如果你想，我可以：

* 把上面每一步用更形式化的 Fokker–Planck 推导写出来（逐步展开偏微分方程），或者
* 给出带 (\eta) 的 **DDIM 带噪声** 一般化公式并详细推导那个噪声项的系数（把确定性与随机性连接起来），或者
* 用一个小的示例（具体 (\alpha_t) 序列）把 DDIM 的逐步数值计算写成伪代码 / 算子，方便实现。

你想从哪一部分深入？
