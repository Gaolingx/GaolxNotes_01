# DQN损失函数计算

### 🧑‍💻 User

强化学习中的dqn loss是什么，如何计算？

---

### 🤖 Assistant

# DQN 中的 Loss 是什么、如何计算

## 一、先理解 DQN 的核心目标

DQN（Deep Q-Network）的核心是用神经网络 $Q(s, a; \theta)$ 去近似**最优动作价值函数** $Q^*(s, a)$，其满足贝尔曼最优方程：

$$Q^*(s, a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

其中：
- $s$：当前状态
- $a$：当前动作
- $r$：即时奖励
- $s'$：下一状态
- $\gamma$：折扣因子（discount factor）

DQN 的训练本质上是一个**回归问题**：让网络预测的 $Q$ 值去逼近上面这个"目标值"。

---

## 二、DQN Loss 的定义

DQN 使用**均方误差（MSE）**作为损失函数（原始论文用 MSE，实践中也常用 Huber Loss 来稳定训练）：

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{目标值 } y} - \underbrace{Q(s, a; \theta)}_{\text{当前预测值}} \right)^2 \right]$$

写成更直观的形式：

$$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( y_i - Q(s_i, a_i; \theta) \right)^2 \right]$$

其中目标值 $y_i$（也就是标签）为：

$$y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-)$$

---

## 三、关键符号说明

| 符号 | 含义 |
|------|------|
| $\theta$ | 当前（在线）网络的参数，需要被更新 |
| $\theta^-$ | 目标网络（Target Network）的参数，**周期性复制自** $\theta$，不参与梯度回传 |
| $Q(s, a; \theta)$ | 当前网络对 $(s,a)$ 的预测值 |
| $\max_{a'} Q(s', a'; \theta^-)$ | 目标网络对下一状态所有动作取最大 Q 值 |
| $y_i$ | TD 目标（Temporal Difference target） |

---

## 四、为什么要有两个网络（$\theta$ 和 $\theta^-$）

如果用同一个网络同时计算"预测值"和"目标值"，目标值会随着参数更新而不断移动（相当于"追着自己的尾巴跑"），导致训练不稳定、容易发散。

**解决方法**：引入目标网络 $\theta^-$，它在一段时间内保持固定，使目标值相对稳定。通常每隔 $C$ 步执行一次硬拷贝：

$$\theta^- \leftarrow \theta$$

---

## 五、如何计算（完整步骤）

假设从经验回放缓冲区 $\mathcal{D}$ 中采样一个 mini-batch，大小为 $N$，其中每个样本为 $(s_i, a_i, r_i, s'_i, d_i)$（$d_i$ 表示是否终止）：

### 步骤 1：计算目标值 $y_i$

```python
# 用目标网络计算下一状态各动作的 Q 值
next_q = target_net(s_prime)  # shape: [batch, num_actions]

# 取最大 Q 值
max_next_q = next_q.max(dim=1).values  # shape: [batch]

# 计算 TD 目标（终止状态无下一项）
y = reward + gamma * max_next_q * (1 - done)
```

### 步骤 2：计算当前网络的预测值

```python
q_values = current_net(s)              # shape: [batch, num_actions]
pred_q = q_values.gather(1, action)    # 只取实际执行动作对应的 Q 值
```

### 步骤 3：计算 Loss 并反向传播

```python
loss = (pred_q - y.detach()).pow(2).mean()  # MSE
# 或者用 Huber Loss:
# loss = F.smooth_l1_loss(pred_q, y.detach())

optimizer.zero_grad()
loss.backward()   # 注意：梯度只流向 current_net（θ），不流向 target_net（θ⁻）
optimizer.step()
```

**关键点**：`y.detach()` 非常关键——目标值不参与梯度计算，Loss 只对当前网络参数 $\theta$ 求导。

---

## 六、一个具体数值例子

假设：
- $\gamma = 0.99$
- 当前状态 $s$，动作 $a = 1$
- 奖励 $r = 1.0$
- 下一状态 $s'$ 上，目标网络给出的各动作 Q 值为 $[2.0, 5.0, 3.0]$，则 $\max = 5.0$
- 当前网络对 $(s, a=1)$ 的预测值为 $Q(s, 1) = 3.0$

计算过程：

$$y = 1.0 + 0.99 \times 5.0 = 5.95$$

$$\mathcal{L} = (5.95 - 3.0)^2 = 8.7025$$

这表示当前网络"低估"了这个状态动作对，梯度会推动预测值向 $5.95$ 靠近。

---

## 七、MSE vs Huber Loss

| 损失函数 | 公式 | 特点 |
|----------|------|------|
| MSE | $(y - Q)^2$ | 对误差敏感，误差大时梯度大，可能不稳定 |
| Huber Loss | $\begin{cases} \frac{1}{2}(y-Q)^2 & \text{if } \lvert y-Q \rvert \le \delta \\ \delta(\lvert y-Q \rvert - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$ | 对大误差线性惩罚，训练更稳定，是 DQN 实际常用选择 |

---

## 八、总结

1. **DQN Loss 本质是 TD 误差的平方**：$\left( r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \right)^2$
2. **计算三要素**：当前网络预测值、目标网络计算的 TD 目标、均方误差（或 Huber）聚合
3. **两个关键技巧**：经验回放（打破样本相关性）+ 目标网络（稳定目标）
4. **梯度只更新当前网络** $\theta$，目标网络定期拷贝

这就是 DQN 的损失函数及其完整计算流程。如果你想看更完整的 PyTorch 实现代码，我可以继续展开。