# RL尖锐值采样影响

## 🧑‍💻 User

rl之后的模型output logits distribution会更尖锐是正常现象吗，对sampling会产生什么影响

---
## 😀 默认助手 (7)

结论先说：**是的，非常正常，甚至是 RL(RLHF / PPO / GRPO / DPO / RLVR 等)优化的一个典型副产物**——模型输出的 logits 分布会变得更"尖锐"(低熵、高置信)。但"变得更尖"本身不是问题，**尖到发生 entropy collapse / 多样性坍缩才是问题**。下面分机制、异常边界、对 sampling 的影响三部分讲清楚。

---

## 一、为什么会变尖（正常的原因）

RL 的目标函数和 SFT 有本质区别：

- **SFT** 是在人类数据上做 cross-entropy，本质是"模仿数据分布"，因此输出的熵大致**贴合数据的多样性**，不会刻意压到极点。
- **RL**(policy gradient / PPO / GRPO)最大化 $\mathbb{E}_{y\sim\pi_\theta}[r(y)] - \beta\, \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$。对 log-prob 求梯度时，advantage $A>0$ 的 token 被**推高** logit，$A<0$ 的被**压低**：

$$\nabla_\theta \mathcal{J} \propto \mathbb{E}\Big[\sum_t A_t \,\nabla_\theta \log \pi_\theta(y_t)\Big]$$

这个更新是**模式寻找(mode-seeking)**的：反复把高回报模式的概率往上顶，于是分布自然变尖。

- 直觉等价关系：logits $z$ 变尖 $\approx$ 温度 $T$ 变小。因为
$$p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$
RL 把 $z$ 的尺度放大，**和降低 $T$ 的效果是等价的**。

<details>
<summary><b>补充：为什么 RLVR / 推理模型上尤其明显</b></summary>

- 可验证奖励(数学/代码)会把"某条正确推理路径"的概率显著抬高，伴随**策略熵骤降**，这在 GRPO 类训练里被称为 **entropy collapse**，是很常见的观测。
- 很多工作(如关于 reasoning LLM 熵机制的论文)都报告：训练中 policy entropy 先缓降后快速坍缩，rollout 多样性随之下降，最终限制继续提升。

</details>

---

## 二、什么时候"尖"开始不正常

| 现象 | 是否正常 | 说明 |
|---|---|---|
| 训练后期 entropy 缓降 | ✅ 正常 | RL 的自然结果 |
| top-1 / max prob 明显升高 | ✅ 正常 | 置信度上升 |
| **熵在早期骤降为接近 0** | ⚠️ 异常 | entropy collapse，探索枯竭 |
| **同一 prompt 多次采样几乎完全相同** | ⚠️ 异常 | 多样性坍缩 / mode collapse |
| **高置信但答错(校准变差)** | ⚠️ 需注意 | RL 常使模型 overconfident |

关键判断：**看熵下降的速度和下限，而不是下降本身**。缓坡下降 = 学习;断崖式坍缩 = 需要加 entropy bonus / 调 KL 系数 / 重调采样。

---

## 三、对 sampling 的具体影响

设采样概率为 $\tilde p = \mathrm{softmax}(z/T)$，RL 让 $z$ 变尖等价于**有效温度被调低**：

1. **多样性下降 / 更确定**：同 prompt 的输出方差变小，pass@1 可能上升，但 pass@k、best-of-n 的收益变小(候选间差异被压缩)。
2. **temperature 需要重调**：想要维持和 RL 前相同的多样性，通常要**把 $T$ 调高**。用旧 $T$ 采样会觉得"变呆、变重复"。
3. **top-k / top-p 的有效集合缩小**：
   - 分布更尖 → 累积概率更快达到 $p$ →`top-p` 实际选中的 token 数变少;
   - `top-k` 名义不变但尾部 token 概率趋近 0，等价于更贪心。
   - 结果可能"想贪心又不完全是贪心"，反而产生偶尔的怪异 token。
4. **greedy 基本不变(但有例外)**：argmax 对 logit 单调缩放不变，所以 greedy 输出通常不因"变尖"而变;但**近似并列的 token 可能翻转**，行为并非完全稳定。
5. **重复 / 退化输出风险上升**：熵过低时，一旦进入重复循环，采样也难跳出，需要靠 repetition penalty、更高 $T$ 或 min-p 来救。
6. **校准与不确定性估计变差**：logits 尖 ≠ 更可信。做置信度 / 拒绝采样 / 不确定性量化时，RL 后的概率**被高估**，需要重新校准(temperature scaling 可部分修复)。
7. **训练中 rollout 探索变差(自反馈陷阱)**：如果 RL 过程中采样温度不随之调整，越训练采样越确定性 → 探索不足 → 早熟收敛、reward 增长停滞。

---

## 四、实践建议

```text
1. 监控指标：policy entropy、top-1 prob、distinct-n、pass@k（不只 pass@1）
2. 训练侧控熵：entropy bonus、增大 KL 系数 β、clip、动态温度
3. 采样侧重调：RL 后提高 temperature，重扫 top-p / top-k / min-p
4. 评估策略：报告多个 (T, top-p) 组合，区分"能力提升"与"只是变确定"
5. 若要保多样性：考虑 KL 约束到 ref、或做 diversity-aware RL / 事后温度校准
```

<details>
<summary><b>一句话总结</b></summary>

RL 后 logits 变尖是**预期行为**(模式寻找 + 等效降温)，不必担心;真正要防的是**熵断崖式塌陷导致多样性丧失**。对 sampling 的直接后果是"有效温度变低"——**你需要重调 temperature / top-p / top-k，否则会明显感到输出更单调、更重复，且概率置信度被系统性高估**。

</details>

如果你能告诉我具体场景(RLHF 还是 GRPO/RLVR?base 模型规模?采样参数?),我可以帮你更具体地判断当前熵水平是否已进入异常区间,并给出建议的温度/采样参数区间。
