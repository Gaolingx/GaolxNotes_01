# CFG是条件残差的线性放大器

## 🧑‍💻 User

扩散模型（Diffusion / Flow Matching）中，cfg和条件残差的关系是什么

---
## 😀 默认助手 (7)

## 一句话结论

CFG 不是一个独立机制；它是**条件残差（conditional residual）的线性放大**：

$$
\tilde v_\theta(x_t,t,c)\;=\;\underbrace{v_\theta(x_t,t,\varnothing)}_{\text{共享/无条件基底}}\;+\;w\,\underbrace{\big[v_\theta(x_t,t,c)-v_\theta(x_t,t,\varnothing)\big]}_{\text{条件残差 }\Delta}
$$

因此：$w=0$ 时退化为无条件采样；$w=1$ 时恰好回到条件模型；$w>1$ 时只是在“条件方向”上做线性外推。CFG 的全部效果都来自对 $\Delta$ 的缩放。

---

## 1. 三条等价写法

设条件预测 $v_c := v_\theta(\cdot,c)$，无条件预测 $v_\varnothing := v_\theta(\cdot,\varnothing)$，条件残差 $\Delta := v_c - v_\varnothing$。

$$
\tilde v = v_\varnothing + w\Delta
\;=\; v_c + (w-1)\Delta
\;=\; (1-w)\,v_\varnothing + w\,v_c
$$

第三式说明 CFG 本质上是**两个预测的加权外插**；因为 $w>1$，权重中有一项为负，所以它不是凸组合，而是外推。这一点决定了它为什么“有效”，也决定了它为什么“会崩”（见第 5 节）。

> 注意：$\Delta$ 是对**同一个 $x_t$** 分别跑两次前向得到的差。它依赖 $x_t$ 的方式是非线性的，并不是“把条件的贡献单独拆出来”。

---

## 2. 残差在概率上是什么：Likelihood 修正项

这是理解 CFG 最关键的一步。由 $p(x|c)=p(x)p(c|x)/p(c)$：

$$
\nabla_x \log p(x|c) \;=\; \underbrace{\nabla_x \log p(x)}_{s_\varnothing} \;+\; \underbrace{\nabla_x \log p(c|x)}_{s_c - s_\varnothing \;=\; \Delta s}
$$

也就是说：**条件残差对应贝叶斯公式里的似然梯度项 $\nabla_x\log p(c|x)$**，它衡量“把 $x$ 往哪个方向移动，能让它更像 $c$”。

代入 CFG：

$$
s_w = s_\varnothing + w\,\Delta s = \nabla_x\log p(x) + w\nabla_x\log p(c|x)
= \nabla_x \log\!\big[p(x)\,p(c|x)^{w}\big]
$$

$$
\boxed{\;\text{CFG}\;\Longleftrightarrow\;\text{从 } p_w(x)\;\propto\; p(x)\,p(c|x)^{w} \text{ 采样}\;}
$$

即：**先验不变，把条件似然“退火/锐化” $w$ 次幂**。这解释了三个现象：

| 现象 | 该框架下的解释 |
|---|---|
| $w$ 越大越“听话”、越贴合 prompt | $p(c|x)^w$ 把质量推向高似然区域 |
| $w$ 越大多样性越差、越容易过饱和 | $w\to\infty$ 时 $p_w$ 集中到 $\arg\max_x \log p(c|x)$ 附近的少数模态 |
| $w<1$ 或 $w<0$（negative prompt） | 反向外推，主动远离 $c$ 的高似然区 |

一个直观类比：$p(c|x)^w$ 类似“把条件 $c$ 重复采样 $w$ 次得到的后验”，$w$ 相当于条件的“证据强度”。

---

## 3. 几何：残差 = 条件方向上的一阶响应

把网络拆成“条件无关主干 + 条件调制”，记条件嵌入 $e(c)$，则

$$
\Delta(x_t,t,c) \;\approx\; J_\theta(x_t,t)\cdot\big(e(c)-e(\varnothing)\big),
\qquad J_\theta=\frac{\partial v_\theta}{\partial e}
$$

**CFG 就是在条件嵌入方向上做步长为 $(w-1)$ 的线性外推**。这直接说明其失效边界：$w$ 太大时，$\tilde v$ 跑到训练分布（条件嵌入空间）之外，网络输出不可信，于是出现伪影、过曝、肢体错乱等“外推失控”症状。

---

## 4. 架构层面：残差就是“调制量之差”

这也是实践中可以直接量化的形式：

- **DiT / PixArt 类（AdaLN）**：调制参数 $\gamma,\beta,\alpha = \text{MLP}(t) + \text{MLP}(c)$（+ 共享 base）。此时
  $$
  \Delta_{\text{mod}} = \text{MLP}(c)-\text{MLP}(\varnothing)
  $$
  CFG 就是在缩放这个 **shift/scale 偏移量**。
- **UNet 交叉注意力**：残差主要来自 cross-attn 输出项，CFG 等价于放大 attention 写入的 value 增量。
- **可加性假设**：只有当条件分支的输出是线性可加的，$\Delta$ 才等于“纯条件的贡献”；实际上有 $\text{softmax}$ 与 LayerNorm 的非线性，$\Delta$ 只是近似的贡献度量。

---

## 5. 数量关系与几个必知的坑

<details>
<summary><b>A. 范数膨胀：$\tilde v$ 的模长被放大，不只是一个方向问题</b></summary>

$$
\|\tilde v\|^2 = \|v_\varnothing\|^2 + 2w\langle v_\varnothing,\Delta\rangle + w^2\|\Delta\|^2
$$

$w>1$ 时通常 $\|\tilde v\|>\|v_c\|$。在 diffusion ODE / flow matching 里，这一步等价于**局部步长被放大**，是 $w$ 过高导致崩坏、以及需要 guidance interval / rescaling 的直接原因。

把残差分解为 $\Delta=\Delta_\parallel+\Delta_\perp$（相对 $v_c$ 分解），$\Delta_\parallel$ 只改变模长、几乎不改变采样方向，却是过饱和的主要来源。据此有 **APG（Adaptive Projected Guidance）**：

$$
\tilde v = v_c + (w-1)\,\Delta_\perp
$$

**只放大残差中垂直于条件预测的分量**——这也反过来证明：CFG 真正有用的部分是残差的“方向”，而不是“大小”。
</details>

<details>
<summary><b>B. Flow Matching 里 CFG 与 score-CFG 严格等价</b></summary>

对高斯概率路径，velocity 与 score 是仿射关系（如 VP-SDE 下 $v = f(x,t) - \tfrac12 g^2\nabla\log p_t$）。关键点是**该仿射偏移与 $c$ 无关**，所以在取差时被抵消：

$$
\Delta v = -\tfrac12 g^2\,\Delta s \quad\Longrightarrow\quad \tilde v = v_\varnothing + w\,\Delta v \;\Longleftrightarrow\; \tilde s = s_\varnothing + w\,\Delta s
$$

即 CFG 在这个变量替换下是**等变的**。因此第 2 节的 $p(x)p(c|x)^w$ 解释对 flow matching / rectified flow 完全成立，不需要单独理论。在“预测位移”视图下 $\Delta v$ 就是**条件引起的期望漂移之差**。
</details>

<details>
<summary><b>C. 什么时候这个关系不再成立</b></summary>

- **$w=w(t)$ 随时间变化**：各时刻的残差尺度不同，合成场不再是任何单一密度的 score（梯度场），只能是近似 → 这是 guidance schedule 的理论代价。
- **残差是偏差估计**：$\Delta$ 只是 $\nabla\log p(c|x)$ 的有偏近似，两处偏差相减还会放大概率质量泄漏；$w$ 在实践中只能调参，不能由 $\log p(c|x)$ 解析给出。
- **连续条件（文本嵌入）的 $p(c|x)$ 本身不合法**：文本不是连续随机变量，“$p(c|x)^w$”只是类比；CFG 在文本条件下的理论解释本质是启发式的（这也是它有效但难以完全解释的根源）。
- **$\Delta\approx 0$ = 条件被忽略**：这是诊断“模型没学会/提示词无效”的直接指标；同时它也说明训练时必须做 condition dropout，否则 $\varnothing$ 分支无定义，残差无意义。
</details>

---

## 6. 由“残差视角”导出的 CFG 变体（都是同一恒等式的改写）

| 方法 | 对残差的改动 |
|---|---|
| **CFG** | $\Delta = v_\text{cond} - v_\text{uncond}$ |
| **Negative Prompt** | $w<0$，即减去残差 |
| **AutoGuidance** | 用“更弱模型”替代 $\varnothing$：$\Delta = v_\text{strong}-v_\text{weak}$（残差 = 能力差而非条件差） |
| **APG / CFG++** | 取 $\Delta_\perp$，或把残差从“改速度”改为“改 $x_0$ / 改采样轨迹” |
| **CFG Distillation** | 直接回归 $\tilde v = v_\varnothing+w\Delta$ 这个**已放大的残差**，把两步前向压成一步 |
| **Perturbed-Attention Guidance** | 残差 = 原图与“注意力扰动图”的差，用内部结构差异代替条件差异 |

**同一条公式 $\tilde v = v_0 + w\Delta$，换掉 $\Delta$ 的定义就得到上面所有这些方法**——这正是“CFG 与条件残差关系”的实用价值。

---

## 7. 如果你说的“条件残差”是另一个意思

上面按“$v_c-v_\varnothing$”理解。若你指的是：

- **去噪残差 / 预测误差** $\hat x_0 - x_0$：它与 CFG 无直接等式关系，但 CFG 会放大该误差（$w$ 乘在外推项上），属于“引导放大了模型偏差”的来源；
- **残差网络意义上的 residual**（$h_{l+1}=h_l+F(h_l)$）：CFG 与它无关，只是词面重合；
- **损失里的 condition residual**（如 controlnet 的 zero-conv 残差分支）：那是架构上的条件注入通道，CFG 会线性放大该通道的净输出。

如果是后两种，请告诉我具体语境，我可以就那条线展开。

---

### 记住这一句

> **CFG $=$ 无条件场 $+$ $w$ 倍条件残差；条件残差 $=$ score 里的 $\nabla\log p(c|x)$ $=$ 条件方向上的线性响应 $=$ 调制量之差。CFG 就是把后验“锐化”成 $p(x)p(c|x)^w$ 的采样器。**

---
## 🧑‍💻 User

1. 数学本质：CFG 增量就是「隐式分类器梯度残差」
2. 几何与画质：高 CFG 为什么会「过饱和/烧焦」？ （APG / CFG-Rescale）
3. 网络结构与加速：DiT/U-Net 残差流与 CFG 残差缓存
分点概述

---
## 😀 默认助手 (7)

## 1. 数学本质：CFG 增量 = 「隐式分类器梯度残差」

<details>
<summary><b>核心推导（展开）</b></summary>

由贝叶斯公式 $p(x_t|c)=p(x_t)p(c|x_t)/p(c)$：

$$
\nabla_{x_t}\log p(x_t|c)=\underbrace{\nabla_{x_t}\log p(x_t)}_{s_\varnothing}+\underbrace{\nabla_{x_t}\log p(c|x_t)}_{\text{隐式分类器梯度}}
$$

用 Tweedie 关系 $x_t=\alpha_t x_0+\sigma_t\epsilon$，$s=-\epsilon/\sigma_t$ 换算：

$$
\nabla_{x_t}\log p(c|x_t)\;\approx\;-\frac{1}{\sigma_t}\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)\;=\;-\frac{\Delta\epsilon}{\sigma_t}
$$

因此 CFG 增量 $(w-1)\Delta$ 就是「隐式分类器梯度」的一个有偏但**同尺度**的估计。

</details>

**关键要点：**

- **它是“比值”而非“绝对值”**：$\Delta s=\nabla\log\frac{p(c|x_t)}{p(\varnothing|x_t)}$。残差估计的是 log-odds 的梯度，不是 $\log p(c|x)$ 本身；$\varnothing$ 分支近似 $\nabla\log p(x)$，所以差正好补出似然项。
- **这是“渐进分类器”族**：残差是 $t$-索引的一族分类器。$t$ 大时 $x_t$ 几乎不含 $c$ 信息，$\Delta\approx0$；$t$ 小时才锐利。这解释了为什么 CFG 必须按 $t$ 调度。
- **关键是“同训练、同噪声、同流形”**：Classifier Guidance 需要额外训练“带噪输入分类器”；CFG 的分类器与生成模型联合训练、共用同一 $x_t$ 表示——**这是 CFG 优于传统 CG 的数学根源**。
- **$w-1$ 反映“对贝叶斯规则的超额施加”**：若模型完美，$w=1$ 可证最优；实践中必须 $w>1$，因为 MSE 回归给的是**条件均值** $\mathbb{E}[\epsilon|x_t,c]$，天然“被平均掉”、条件利用不足。CFG 补偿的是这个均值化偏差，把质量推向条件**模态**而非均值。
- **它是合法的梯度场（有前提）**：固定 $w$ 时 $\tilde s=\nabla\log[p(x)p(c|x)^w]$，积分性良好；一旦 $w=w(t)$ 或在不同 $t$ 上残差尺度不一致，合成场**不再可积**，只是近似 score——这是各种 guidance schedule 的理论代价。
- **flow matching 下等价**：$v$ 与 $s$ 的仿射偏移与 $c$ 无关，取差被抵消，故 $\Delta v\propto\Delta s$，同一套解释原样成立。

---

## 2. 几何与画质：高 CFG 为什么「过饱和 / 烧焦」

**四条叠加机制（按贡献排序）：**

1. **范数膨胀 → 等效步长失控**
   $$
   \|\tilde v\|^2=\|v_\varnothing\|^2+2w\langle v_\varnothing,\Delta\rangle+w^2\|\Delta\|^2
   $$
   $w>1$ 时 $\|\tilde v\|>\|v_c\|$，Euler/Heun 每步实际位移被放大 → 超调（overshoot）。
2. **残差的“平行分量”只有害无益**：分解 $\Delta=\Delta_\parallel+\Delta_\perp$（相对 $v_c$），$\Delta_\parallel$ **只改模长、不改方向**，是纯噪声式的步长注入。
3. **朝“最可分”而非“最像”移动**：$p_w\propto p(x)p(c|x)^w$ 在 $w\uparrow$ 时把质量从先验的高密度区推向**类别支持集的边界**——那里 $\log p(c|x)$ 最高但 $p(x)$ 最低。产物就是极端对比度、塑料皮肤、死黑死白。
4. **高质量细节先被牺牲**：纹理/中高频细节与隐式分类器梯度几乎正交，不贡献可判别性；放大残差后，细节相对权重被压制 → 平滑、块状。最后再被 clip 到 $[-1,1]$ 截断 → 色彩偏移 + 高光断层 = 「烧焦」。

**主流修法（公式）：**

| 方法 | 公式 | 修的是什么 |
|---|---|---|
| **APG**（Adaptive Projected Guidance） | $\tilde v=v_c+(w-1)\Big(\Delta-\frac{\langle\Delta,v_c\rangle}{\|v_c\|^2}v_c\Big)$ | 只放大 $\Delta_\perp$，去掉纯步长注入 |
| **CFG-Rescale**（Lin et al. 2024） | $\hat x_0\leftarrow\hat x_0^{w}\cdot\dfrac{\mathrm{std}(\hat x_0^{c})}{\mathrm{std}(\hat x_0^{w})}$ | 每步恢复无引导预测的统计量 → 直接治过曝 |
| **范数重标定** | $\tilde v\leftarrow\dfrac{\|v_c\|}{\|\tilde v\|}\tilde v$ | 保模长，最小侵入 |
| **Dynamic Thresholding**（Imagen） | $\hat x_0\leftarrow\mathrm{clip}(\hat x_0,\,-s p,\,sp)$ | 分位数截断，硬性防饱和 |
| **Interval Guidance** | 仅在 $t\in[t_\text{lo},t_\text{hi}]$ 施加 CFG | 避免低噪段累积误差（顺带省算力） |
| **CFG++**（软引导） | 引导改 $\hat x_0$ 的流形投影而非速度 | 减少离流形漂移 |
| **CFG-Zero\*** | 前若干步用未归一化的 $\varnothing$ | 修 $t\to1$ 时 $\Delta$ 无意义导致的构图崩坏 |

> **一句话**：过饱和 = 残差中的“径向放大”+ 引导目标偏离数据流形 + 末端截断。APG 治径向，Rescale 治统计量，Interval/Zero* 治调度。

---

## 3. 网络结构与加速：DiT/U-Net 残差流与 CFG 残差缓存

**A. 残差“在哪儿”——结构视角**

- DiT/AdaLN 下条件只改调制参数：$\gamma,\beta,\alpha=\text{MLP}(t)+\text{MLP}(c)$，因此
  $$
  \Delta_\text{mod}=\text{MLP}(c)-\text{MLP}(\varnothing)\quad(\text{每层一个 }\mathbb{R}^d\text{ 小向量})
  $$
  **这是全网最便宜的残差表示**——存它比存激活便宜几个数量级。
- UNet/PixArt 下残差主体来自 cross-attn 的 value 写入量；ControlNet/IP-Adapter 的 zero-conv 残差分支使其在初始化时**恰好等于** $\Delta$，天然可解释。
- 残差是**跨全部 block 的分布式差值**，无法在单层读出；但它沿深度方向的“形状”高度一致 → 可低秩/低精度近似。

**B. 加速手段——从“省一半”到“消掉一半”**

| 层级 | 手段 | 收益 |
|---|---|---|
| 工程 | cond/uncond **batch 合并**为一次前向（$2B$），text encoder 只跑一次，null 分支 KV 复用 | 无质量损失，纯吞吐 |
| **残差缓存** | $\Delta_t$ 在 $t$ 上**远平滑于** $v_t$（慢语义 vs 快细节）→ 每 $k$ 步算一次 $\Delta$，中间步复用/线性插值/EMA | 逼近 2×，显存极低 |
| 调制缓存 | 只缓存 $\Delta_\text{mod}$，重建代价≈0 | 近乎免费 |
| **引导蒸馏** | 训 $v_\theta(x,c)\to\tilde v$（Guidance Distillation / SDXL-Turbo / DMD2 的 guidance-free） | **彻底消除 2×** |
| 区间跳过 | $t\notin[t_\text{lo},t_\text{hi}]$ 时不跑 uncond 前向 | 真实 FLOPs 节省 |
| 非对称精度 | 只在差里出现的 uncond 分支用 int8/低精度、或更少步数 | 残差对量化误差不敏感 |
| 步数蒸馏 | 4–8 步学生 + 引导蒸馏 | 10× 量级 |

**C. 三条实操要点**

- **缓存 $\Delta$ 比缓存 $v$ 安全**：缓存 $v$ 会丢掉高频细节（DeepCache 类方法的通病）；缓存 $\Delta$ 只动“语义漂移量”，基底保真度不降。误差 $\propto\|\partial_t\Delta\|$，中段小、两端大 → **缓存粒度需按 $t$ 调度**（两端 $k$ 小）。
- **AutoGuidance 是“换残差”而非“省算力”**：用弱模型当 $\varnothing$，$\Delta=v_\text{strong}-v_\text{weak}$，在同等 $w$ 下画质更好，但不省前向。
- **蒸馏与缓存可叠加**：先做引导蒸馏消除 2×，再在剩余步内做残差/块缓存，二者正交。

---

### 三者串起来

> **数学上**：$\Delta$ 是隐式分类器梯度（log-odds 的梯度），$w-1$ 是对贝叶斯规则的超额施加。
> **几何上**：放大 $\Delta$ 会把样本推向“最可分”的低密度边界 + 径向超调 → 饱和/烧焦，故需 APG（去径向）、Rescale（修统计）、调度（限区间）。
> **结构上**：$\Delta$ 是跨层分布式的差值，但**沿 $t$ 平滑**——这一条既解释了它为何可被低精度/低秩近似，也直接给出“缓存 $\Delta$、蒸馏 $\Delta$”这两条加速主线。
