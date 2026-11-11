# PPO重要性采样原理

## 🧑‍💻 User

为什么做ppo需要重要性采样？

---
## 😀 默认助手 (7)

## 一句话回答

因为 PPO 要在**同一批数据上做多次梯度更新**，而这批数据是用旧策略 $\pi_{\theta_{\text{old}}}$ 采的，更新后策略已经变成 $\pi_\theta$。重要性采样就是用来**把"用旧策略采的样本"换算成"新策略下的期望"**，让梯度估计仍然指向新策略的正确方向。

---

## 1. 矛盾的来源：数据复用

策略梯度的本质是：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta(\cdot\mid s)}\Big[\nabla_\theta\log\pi_\theta(a\mid s)\,A^{\pi_\theta}(s,a)\Big]
$$

注意期望里的动作是**从当前策略 $\pi_\theta$ 采的**。这是 on-policy 方法必须"采完就更新、更新完就扔"的原因——一旦 $\theta$ 变了，手上的数据就不再服从新策略。

而 PPO 的全部卖点恰恰是**反着来**：

```
用 π_old 采一批数据 (rollout)
  → 在这批数据上做 K 个 epoch 的 minibatch 梯度下降
  → 扔掉，重新采
```

第 2 步之后，策略每更新一次就变一次，数据相对于新策略已经是 **off-policy** 的了。如果不做修正，你算出来的就不是 $\pi_\theta$ 的梯度。

---

## 2. 重要性采样如何补上这个缺口

对同一个状态 $s$，把期望从 $\pi_\theta$ 换到 $\pi_{\text{old}}$ 上：

$$
\mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)A\big]
=\sum_a \pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)A
$$

$$
=\sum_a \pi_{\text{old}}(a\mid s)\underbrace{\frac{\pi_\theta(a\mid s)}{\pi_{\text{old}}(a\mid s)}}_{r_t(\theta)}\nabla_\theta\log\pi_\theta(a\mid s)A
=\mathbb{E}_{a\sim\pi_{\text{old}}}\Big[r_t(\theta)\nabla_\theta\log\pi_\theta(a\mid s)A\Big]
$$

于是得到可用的估计量：

$$
\hat g=\frac{1}{N}\sum_{t} r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t
$$

这里 $r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$ 就是**重要性采样比（IS ratio）**。这个估计对 $\nabla_\theta J(\theta)$ 是**无偏**的，只要 $r_t$ 有限。

---

## 3. 它恰好就是 PPO 的目标函数

由于 $\nabla_\theta\log\pi_\theta=\dfrac{\nabla_\theta\pi_\theta}{\pi_\theta}$，有

$$
r_t\,\nabla_\theta\log\pi_\theta=\frac{\pi_\theta}{\pi_{\text{old}}}\cdot\frac{\nabla_\theta\pi_\theta}{\pi_\theta}=\frac{\nabla_\theta\pi_\theta}{\pi_{\text{old}}}
=\nabla_\theta\!\left(\frac{\pi_\theta}{\pi_{\text{old}}}\right)
$$

所以上面那个估计量，正是下面这个 **surrogate 目标** 的梯度：

$$
L^{\text{IS}}(\theta)=\mathbb{E}_{t}\big[r_t(\theta)\,\hat A_t\big]
$$

**PPO 的 clipped 目标就是在这个 surrogate 上包了一层 clip：**

$$
L^{\text{CLIP}}(\theta)=\mathbb{E}_t\Big[\min\big(r_t\hat A_t,\ \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]
$$

也就是说——**没有重要性采样比，PPO 的目标函数根本写不出来**，clip 也就无从谈起。$r_t$ 是 PPO 的骨架，不是附加项。

---

## 4. 直观理解：为什么非这样不可

| 做法 | 后果 |
|---|---|
| 不做 IS，直接 $\mathbb{E}_{a\sim\pi_{\text{old}}}[\nabla\log\pi_\theta A]$ | 估计的是**旧策略**的梯度方向。θ 一动，方向就偏了，多步更新等于在错误方向上累积，训练容易崩 |
| 做 IS，但不管 $r_t$ | 无偏、但**方差爆炸**。一旦 $\pi_\theta$ 偏离 $\pi_{\text{old}}$，某个动作的 $r_t$ 可能变成 10 或 0.01，梯度被少数样本主导 |
| 做 IS + clip | 有偏、但方差可控。**故意用一点偏差换稳定性**，这是 PPO 的核心权衡 |

> 换句话说：重要性采样给了你"复用数据"的**权利**，clip 保证你**不会滥用**这个权利——它把 $r_t$ 摁在 $[1-\epsilon,1+\epsilon]$ 内，等价于约束 $\pi_\theta$ 不能离 $\pi_{\text{old}}$ 太远，也就是一个软性的 trust region（TRPO 的廉价近似）。

---

## 5. 几个容易被忽略的点

<details>
<summary><b>① 第一次更新其实不需要 IS（ratio 恒等于 1）</b></summary>

rollout 刚结束时 $\theta=\theta_{\text{old}}$，所以 $r_t\equiv 1$，修正项是恒等变换。重要性采样**只在第 2 次及以后的梯度步才真正起作用**。这也解释了为什么 PPO 的 `n_epochs` 越大，IS 的负担越重：越靠后的 epoch，$\pi_\theta$ 偏离 $\pi_{\text{old}}$ 越远，$r_t$ 越离谱。

</details>

<details>
<summary><b>② PPO 只修正了动作分布，没修正状态分布（这是个近似）</b></summary>

严谨地说，策略梯度里的期望还包括**状态分布** $d^{\pi_\theta}(s)$，而 PPO 的 $r_t$ 只对 $a$ 做了重要性采样，状态还是从 $d^{\pi_{\text{old}}}$ 里来的。所以 $L^{\text{CLIP}}$ 严格讲是**有偏**的。实践上能work，是因为 clip/KL 把两个策略约束得很近，$d^{\pi_\theta}\approx d^{\pi_{\text{old}}}$。这是"工程上够用"而非"理论上严格"的地方。

</details>

<details>
<summary><b>③ 所以 PPO 到底算 on-policy 还是 off-policy？</b></summary>

教科书说 PPO 是 on-policy（因为数据用完就扔，不做 replay buffer）。但**严格意义上，多 epoch 更新期间它是"轻度 off-policy"**。重要性采样就是这两者之间的桥。区分标准其实是：*数据能否跨策略版本重复使用* —— PPO 允许在**紧邻的**几个版本内复用，不允许放进 buffer 里长期复用。

</details>

<details>
<summary><b>④ 在 LLM 的 RLHF 里，ratio 是 token 级的</b></summary>

GRPO/PPO 用于大模型时，$r_t$ 通常按 **token** 计算 $\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{old}}(y_t\mid x,y_{<t})}$，再配合 mask；序列级的联合 ratio 理论上是各 token 之积，但实现上一般不做，因为乘积方差更不可控。这也是"PPO 的 IS 只是一个受控近似"的另一个体现。

</details>

---

## 总结

$$
\boxed{\text{数据复用}\ \Longrightarrow\ \text{off-policy}\ \Longrightarrow\ \text{分布不匹配}\ \Longrightarrow\ \text{重要性采样修正}}
$$

PPO 需要重要性采样，不是因为它想做 off-policy，而是因为它想**在一步 rollout 上榨出更多梯度更新（样本效率）**，这个诉求必然导致数据与新策略错配；而 IS ratio 既是修正这种错配的数学工具，也恰好是 clip 机制得以存在的载体。二者共同构成了 PPO "稳定地多步复用数据"这一设计的正反面。
