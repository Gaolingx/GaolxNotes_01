# ddpm vpred fm 差异

## 🧑‍💻 User

ddpm下的vpred和fm有什么区别

---
## 😀 默认助手 (7)

# 一句话答案

**v-pred 和 FM(flow matching)在 DDPM 框架下本质上是同一类对象——“预测条件期望的速度场”——只是路径(path)与时间权重不同。** v-pred 是在 **VP 路径（单位圆）上、用角度做时间重参数化**的速度预测；FM 把这件事推广到**任意 $(\alpha_t,\sigma_t)$ 路径**，实践中最常用的是**直线路径(rectified flow)**。而且可以严格证明：在 VP 归一化下，**v-pred ≡ ε-pred 加一个 $1/\alpha_t^2$ 的 loss 权重**，两者只差重加权。

---

## 1. 两套记号

**DDPM(VP 前向)**

$$x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I),\qquad \alpha_t^2+\sigma_t^2=1,\quad t\in[0,T]$$

loss 就是回归某个目标 $y$:$\ \|y-\hat y_\theta(x_t,t)\|^2$。

**FM(条件流匹配)**

先指定一条从数据到噪声的**概率路径**

$$x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad t\in[0,1]$$

再回归该路径上的**真实时间导数**：

$$u_t(x)=\mathbb E[\dot x_t\mid x_t=x]=\dot\alpha_t\,\mathbb E[x_0|x_t]+\dot\sigma_t\,\mathbb E[\epsilon|x_t]$$

采样时直接积分 ODE $\dfrac{dx}{dt}=u_\theta(x,t)$。

> 注意：FM 的“插值路径”写成 $\alpha_t x_0+\sigma_t\epsilon$ 时，和 DDPM 的**前向过程形式完全一样**，这也是两者能对上的根本原因。CFM 的定理是：只要噪声路径相同，**marginal velocity = 扩散模型的 probability-flow ODE 速度**。

---

## 2. v-pred 到底是什么

Salimans & Ho (2022) 的定义：

$$v=\alpha_t\epsilon-\sigma_t x_0$$

它其实是 $(x_0,\epsilon)$ 到 $(x_t,v)$ 的**正交旋转**：

$$
\begin{pmatrix}x_t\\ v\end{pmatrix}
=
\begin{pmatrix}\alpha_t & \sigma_t\\ -\sigma_t & \alpha_t\end{pmatrix}
\begin{pmatrix}x_0\\ \epsilon\end{pmatrix},
\qquad \det=1
$$

所以逆变换是免费的：

$$x_0=\alpha_t x_t-\sigma_t v,\qquad \epsilon=\sigma_t x_t+\alpha_t v$$

**关键理解**：令 $\alpha_t=\cos\phi,\ \sigma_t=\sin\phi$，则

$$\frac{dx_t}{d\phi}=-\sin\phi\,x_0+\cos\phi\,\epsilon=\sigma_t\epsilon-\alpha_t x_0=-v$$

也就是说，**v 就是“角度时间”下的速度，且被归一化成单位速度**（因为 $(x_0,\epsilon)$ 到 $(x_t,v)$ 是纯旋转，$\|v\|$ 的量级与 $\|x_0\|,\|\epsilon\|$ 一致，与 $t$ 无关）。这就是 v-pred 的全部数学内容。

---

## 3. 严格关系:v-pred ≡ 重加权 ε-pred

若模型输出 $\epsilon_\theta$，用 $x_0=(\alpha x_t-\sigma\cdot\cdot\cdot)$ 的标准换算，可以证明（VP 且 $\alpha^2+\sigma^2=1$）：

$$v-v_\theta=\frac{1}{\alpha_t}\bigl(\epsilon-\epsilon_\theta\bigr)
\quad\Longrightarrow\quad
\|v-v_\theta\|^2=\frac{1}{\alpha_t^2}\|\epsilon-\epsilon_\theta\|^2$$

即 **v-pred = ε-pred 配上权重 $w(t)=\dfrac{1}{\alpha_t^2}=1+\dfrac{1}{\mathrm{SNR}(t)}$**。

这解释了 v-pred 的一切“优点”，也说明它不是新的生成模型，只是**换了一个 loss 权重 + 换了一个数值条件更好的参数化**。Min-SNR 那一类工作的分析正是在这个式子上做的。

---

## 4. 区别对照表

| 维度 | v-pred (DDPM) | Flow Matching (linear/RF) |
|---|---|---|
| 路径 $(\alpha_t,\sigma_t)$ | 单位圆：$\alpha^2+\sigma^2=1$，VP/cosine 为主 | 直线：$\alpha=1-t,\ \sigma=t$（OT 直线插值） |
| 时间域 | $t\in[0,T]$ 离散步 + 非线性 schedule | $t\in[0,1]$ 连续，$\log$-SNR 随 $t$ 显式可控 |
| 回归目标 | $v=\alpha\epsilon-\sigma x_0$，**随 $t$ 旋转** | $u=\epsilon-x_0$，**沿轨迹恒定**（直线路径，与 schedule 无关） |
| 目标方差 | $\approx$ 常数（旋转不改变范数） | $\approx 2$（数据与噪声方差之和），常数 |
| loss 权重 | 等价于 ε-pred 的 $1/\alpha^2$ 加权 | 均匀 $t$ 采样 → 对应 $\lambda(t)=2\log\frac{1-t}{t}$ 的隐式加权 |
| 终端 SNR | $\alpha_T\neq0$，需要 **zero terminal SNR** 修正 | $t=1$ 时 $\alpha=0$ 严格为纯噪声，天然满足 |
| 采样 | DDPM/DDIM 后验公式；需 $\beta$、方差项 | 纯 ODE 积分（Euler/RK），无方差项 |
| 轨迹曲率 | VP 路径弯曲更大，少步数误差大 | 直线/OT 更直，few-step 友好；reflow 可进一步拉直 |
| 典型使用者 | SD 2.x、Imagen、多数蒸馏工作 | SD3/MMDiT、Flux、Sora 类视频模型 |

---

## 5. 结论层面要抓住的三点

<details>
<summary><b>(1) 数学上：v-pred 是 FM 在单位圆路径上的特例（差一个时间重参数化）</b></summary>

把 VP 路径写成角度 $\phi$ 为时间，FM 速度 $u=\frac{dx}{d\phi}=-v$。所以**“v-pred 是不是就是 velocity prediction？”——在“预测速度场”的意义上是；在“flow matching 的 $u=\dot x_t$”意义上不是**，因为 FM 的 $u$ 幅度依赖 schedule，而 $v$ 被归一化了。两者可通过逆旋转 + 链式法则互相换算：

$$u_t=\dot\alpha_t(\alpha_t x_t-\sigma_t v)+\dot\sigma_t(\sigma_t x_t+\alpha_t v)$$

所以给定同一个模型，v 和 $u$ 携带的信息量完全相同，**差别只在“哪个量更容易回归”**。

</details>

<details>
<summary><b>(2) 真正的差别是“路径 + 权重 + 采样器”，不是“模型类别”</b></summary>

- **路径**决定信号-噪声如何混合：VP 圆 vs 线性直线，导致相同 $t$ 上 SNR 不同、轨迹曲率不同、few-step 误差不同。
- **权重**决定训练时各噪声段的注意力：v-pred 隐式放大高噪声端（$1/\alpha^2$ 在 $t\to T$ 爆炸），FM 用 $\log$-SNR/$\logit$-normal 采样来显式控制这件事。
- **采样器**：DDPM 是 SDE/祖先采样，FM 是 ODE；但 DDIM 的 ODE 与 VP 路径的 FM ODE 是**同一条**积分轨迹。

</details>

<details>
<summary><b>(3) 实践差异（为什么大家现在往 FM 走）</b></summary>

1. **训练目标更“平”**：直线路径的 $u=\epsilon-x_0$ 与 $t$、与 schedule 无关，模型不需要同时学“目标如何随 $t$ 旋转”，条件数好。
2. **没有终端 SNR 病**：$t=1$ 严格是纯高斯，不用打 zero-terminal-SNR 补丁；而原始 DDPM 的 $\alpha_T\approx0.008$ 会让 v-pred 在 $t=T$ 处发散（这也是 SD2 v-pred 需要修 schedule 的原因）。
3. **few-step / 蒸馏更自然**：直线轨迹让 Euler 少步误差小，这也是 rectified flow 与后续一致性模型/蒸馏路线更亲近的原因。
4. **但**：v-pred 在“预测 $\epsilon$ 与 $x_0$ 之间取得平衡”这件事上非常有效，且纯实现层面只是换 target + 换 loss weight，改造成本极低——所以它常被当作“DDPM 生态里最接近 FM 的便宜升级”。

</details>

---

## 6. 一个常见误区

> ❌ “v-pred 和 FM 是两种不同的训练目标，一个是预测噪声，一个是预测速度。”

准确说法：**ε-pred / x0-pred / v-pred 是同一个 $(x_t,t)\to$ 高斯条件期望的三种线性重参数化**，loss 之间只差确定性的 $t$ 权重（v-pred ↔ ε-pred 是 $1/\alpha^2$）。而 **FM 是把“回归哪个量”升级成“回归你指定的任意路径的切向量”**——v-pred 恰好是它在 VP 路径上的一个特例。所以三者关系是：

$$
\underbrace{\epsilon\text{-pred}}_{\text{VP 路径},\ w=1}
\;\xrightarrow{\ \times 1/\alpha^2\ }\;
\underbrace{v\text{-pred}}_{\text{VP 路径},\ w=1/\alpha^2}
\;\subset\;
\underbrace{\text{FM}(u=\dot x_t)}_{\text{任意路径，通常直线}}
$$

---

如果你是在做具体实现（比如想把一个 v-pred 的 DDPM checkpoint 迁到 FM 训练、或者对比两者的 FID/少步采样曲线），告诉我你用的 schedule、数据归一化方式和采样器，我可以把对应公式和权重换算写具体。
