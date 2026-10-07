# GSPO稳MoE模型

## 🧑‍💻 User

强化学习中，dense更适合grpo，moe更适合gspo吗？

---
## 😀 默认助手 (7)

# 结论：这个说法**方向大致对，但表述过于绝对、容易误导**

更准确的说法是：

> **GSPO 不是“为 MoE 专门设计的替代品”，而是 GRPO 的一个通用升级版。它的收益在 MoE、长序列、大规模 RL 场景下被放得最大，因此 MoE “最需要” GSPO；但 dense 模型上用 GSPO 通常也不会更差，往往还更好。**

所以“dense 适合 GRPO / MoE 适合 GSPO”这个二分法，只对了一半。下面拆开讲。

---

## 一、两者的本质差异（这才是关键）

| | GRPO | GSPO |
|---|---|---|
| 重要性权重粒度 | **token 级** $w_{i,t}(\theta)=\dfrac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}$ | **序列级** $s_i(\theta)=\left(\dfrac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\text{old}}}(y_i\mid x)}\right)^{1/\lvert y_i\rvert}$ |
| 裁剪 (clip) 粒度 | 每个 token 单独 clip | 整条 response clip |
| 奖励/优势粒度 | 序列级（一条序列一个 $A_i$） | 序列级 |
| clip range 量级 | ~0.2 | ~3e-4（差 2~3 个数量级） |

GSPO 论文的核心论点是：**GRPO 的目标函数本身就是“病态的”（ill-posed）**。

- 重要性采样要成立，需要对同一个分布采样 **多次**（$N\gg1$）才能有效做分布校正。
- 但 GRPO 在每个 token 位置只采样了**一个** token，却硬套一个 token 级的 importance weight——这个权重根本起不到校正作用，只是在往梯度里注入**高方差噪声**。
- 这种噪声**随着序列变长而累积**，再被 clip 机制放大，最终导致**不可逆的模型崩塌**。
- 而奖励是发给整条序列的，所以 GSPO 主张：**优化单位必须和奖励单位对齐 → 用序列级 ratio**。

这里有个很反直觉的实验发现：GSPO clip 掉的 token 比 GRPO **多两个数量级**，反而训练效率更高——说明 GRPO 的 token 级梯度信号本身就又噪又低效。

---

## 二、为什么 MoE 特别受益于 GSPO

<details>
<summary><b>展开：MoE 的“专家激活漂移”问题（这是核心机制）</b></summary>

MoE 相比 dense 有一个致命的不稳定来源：**expert routing volatility（专家激活漂移）**。

论文实测（Qwen3-30B-A3B，48 层）：

> 每做一次 RL 梯度更新后，**同一条 rollout 样本**在新策略 $\pi_\theta$ 下激活的专家，约有 **10%** 和旧策略 $\pi_{\theta_{\text{old}}}$ 不同；模型越深越严重。

这会带来连锁反应：

1. 同一 token 的前后向计算“走的不是同一套参数”。
2. token 级 importance ratio $w_{i,t}$ 会因此**剧烈抖动**。
3. 结合上文的高方差噪声 → RL 训练**无法正常收敛**，甚至崩塌。

**以前的做法**：Routing Replay —— 缓存旧策略激活的专家，在新策略里“重放”这些路由，强行让 $w_{i,t}$ 稳定。
- 代价：额外的显存/通信开销，还**人为限制了 MoE 的容量**（相当于捆住手脚）。

**GSPO 为什么能根治**：
- GSPO 只关心**序列似然** $\pi_\theta(y_i\mid x)$，对**单个 token 的似然不敏感**。
- MoE 的语言建模能力（整句概率）是稳定的，不会因为路由抖动而剧烈变化。
- 于是序列级 ratio $s_i$ 天然稳定 → **完全不需要 Routing Replay**，解放了 MoE 的全部容量。

</details>

所以论文标题里那句 “notably stabilizes **Mixture-of-Experts (MoE)** RL training” 是实打实的机制性结论。

---

## 三、那“dense 更适合 GRPO”对不对？

**要打问号。** 理由：

1. **GRPO 在 dense 上确实被大规模验证过**（DeepSeekMath、DeepSeek-R1、QwQ 等），生态成熟、实现简单、超参经验丰富。这是事实。
2. **但这是“够用”，不是“更适合”**。GSPO 指出的 GRPO 病态问题是**算法层面的**，和 dense/MoE 无关。dense 模型只是没有“路由漂移”这个**额外放大器**，所以崩塌没那么容易触发而已。
3. **GSPO 在 dense 上通常也更好**：更高的训练效率、更好的 benchmark、对精度差异更宽容。
4. **论文的主要实验恰恰是在 MoE（Qwen3-30B-A3B）上做的**——所以“GSPO 只在 MoE 上验证过、dense 没证据”这种反向解读也不成立；它更多是**收益幅度**在 MoE 上更夸张。

---

## 四、GSPO 的额外红利（不只是稳定性）

<details>
<summary><b>展开：对 RL 基础设施的简化</b></summary>

训练引擎（Megatron）和推理引擎（SGLang/vLLM）之间存在**数值精度差异**，所以通常要用训练引擎**重算** old policy 的 likelihood。

而 GSPO **只需要序列级 likelihood**，对精度差异远比 token 级宽容，因此：

- 可以直接用**推理引擎返回的 likelihood** 做优化，省掉重算；
- 对 **partial rollout、多轮 RL、训推分离（disaggregated）** 框架特别友好。

此外论文还给了 **GSPO-token** 变体：允许 token 级自定义优势（比如多轮场景），数值上与 GSPO 等价，但更灵活。

</details>

---

## 五、实践建议（怎么选）

| 场景 | 推荐 | 说明 |
|---|---|---|
| 小模型 / 短序列 / 快速实验 | GRPO 够用 | 生态最成熟，接入成本低 |
| **MoE 模型** | **GSPO（强烈建议）** | 否则要么不收敛，要么被迫上 Routing Replay |
| 长序列 / reasoning 大 batch RL | GSPO | 长序列正是 GRPO 噪声累积的重灾区 |
| 多轮 RL / partial rollout / 训推分离 | GSPO | infra 层面的天然优势 |
| 已有成熟 GRPO 管线、暂不想改 | GRPO + 调好 clip | 但要警惕规模上去后的崩塌风险 |

---

## 一句话总结

**不是“dense↔GRPO、MoE↔GSPO”的绑定关系，而是：**

- **GRPO** 是作用在**错的粒度**上的早期方案（token 级 IS + 序列级奖励不匹配），dense 上“能跑”，MoE 上“容易崩”。
- **GSPO** 把优化单位对齐到序列级，**通用更稳更好**，而在 **MoE + 长序列**上收益最大，因此 MoE 是它最“值回票价”的战场。

> 参考：Zheng et al., *Group Sequence Policy Optimization*, arXiv:2507.18071（Qwen Team, 2025）。
