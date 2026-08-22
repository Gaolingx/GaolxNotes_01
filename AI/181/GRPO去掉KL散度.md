# GRPO去掉KL散度

### 🧑‍💻 User

GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了 KL 散度？

---

### 🤖 Assistant

# GRPO 的 KL 散度：为什么加、怎么算、为什么又被去掉

下面按"为什么加 → 怎么算 → 为什么 DAPO/GSPO 去掉"的顺序讲清楚。

---

## 一、GRPO 为什么加 KL 散度

GRPO（DeepSeekMath，arXiv:2402.03300）是对 PPO 的改造：**去掉了 critic（价值模型）**，用"同一题采样一组 $G$ 条回答、按组内 reward 归一化"来估计优势 $\hat A$。但它在目标函数里**保留了 PPO/RLHF 的 KL 正则项**：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\underbrace{\min\big(r_{i,t}\hat A_{i,t},\,\mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big)}_{\text{裁剪后的策略梯度项}}-\beta\,D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\Big)\Bigg]
$$

加 KL 的动机是**从 RLHF 直接继承来的"信任域"思想**：

1. **防止 reward hacking**：纯 reward 优化会钻 reward model / 规则的空子，产出高分但乱码、重复、跑偏的输出。KL 把策略"拴"在参考模型（通常是 SFT 模型）附近，防止它过度偏离。
2. **防止语言能力退化 / 灾难性遗忘**：KL 让输出分布不要离人类可读、语法正常的分布太远。
3. **稳定训练**：RL 的梯度方差大，KL 相当于一个软约束，让每步更新不至于把策略改得面目全非。

本质上是：**奖励信号负责"往哪走"，KL 负责"别走太远"。**

---

## 二、KL 散度在 GRPO 里具体怎么算

真正的 KL 是期望形式：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})=\mathbb{E}_{o\sim\pi_\theta}\left[\log\frac{\pi_\theta(o|q)}{\pi_{\mathrm{ref}}(o|q)}\right]
$$

但在 RL 里 $o$ 是从**旧策略 $\pi_{\theta_{old}}$** 采样的，无法直接算这个期望，所以 GRPO 用的是 **Schulman 博客里的 k3 估计器**（无偏但方差大、且单样本值可能为负）：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\;\approx\;\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)}-\log\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_\theta(o_i|q)}-1
$$

其中 $\pi_\theta(o_i|q)=\prod_t \pi_\theta(o_{i,t}|q,o_{i,<t})$ 是**整条序列的概率**（token 概率连乘）。

几个要点：

| 估计器 | 形式 | 性质 |
|--------|------|------|
| k1 | $\log\frac{\pi_{ref}}{\pi_\theta}$ | 有偏、低方差、非负 |
| k3（GRPO 用） | $\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$ | **无偏**、高方差、**可能为负** |

k3 无偏：$\mathbb{E}_{\pi_\theta}[\,\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1\,]=D_{\mathrm{KL}}(\pi_\theta\|\pi_{ref})$，这是它被选中的原因。但它也有代价：单条样本的估计值可以是负的、且方差大——这也是后续被诟病的点之一。

---

## 三、为什么 DAPO 去掉了 KL

DAPO（字节跳动，arXiv:2503.14476）专门有一节 **"2.3 Removing KL Divergence"**，理由很直接：

> KL 惩罚项的目的是限制在线策略与冻结参考策略的偏离。在 RLHF 场景中，RL 的目标是"对齐而不偏离初始模型太远"。**但在训练 long-CoT 推理模型时，模型分布本来就应当显著偏离初始模型**（要涌现出自我反思、回溯、长思考等新行为），所以这个限制没有必要。

因此 DAPO 的目标函数里**完全删掉了 KL 项**：

$$
\mathcal{J}_{\mathrm{DAPO}}(\theta)=\mathbb{E}\Bigg[\frac{1}{\sum_i|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\hat A_{i,t},\,\mathrm{clip}\big(r_{i,t},\,1-\epsilon_{\mathrm{low}},\,1+\epsilon_{\mathrm{high}}\big)\hat A_{i,t}\Big)\Bigg]
$$

它靠另外四招来"替代" KL 提供的稳定作用：

1. **Clip-Higher（非对称裁剪）**：把上界 $\epsilon_{\mathrm{high}}$ 放大（如 0.2→0.28），解决 KL 存在时常见的**熵坍缩**问题，保留探索空间。
2. **Dynamic Sampling**：过滤掉组内 reward 全 0 或全 1 的无梯度样本。
3. **Token-level loss**：从"按样本平均"改成"按 token 平均"。
4. **Overlong Reward Shaping**：对超长截断样本做软惩罚。

**一句话总结 DAPO 的逻辑：在"可验证奖励 + 推理 RL"的设定下，KL 不是保护而是枷锁；信任域交给裁剪（clip）就够用了。**

---

## 四、为什么 GSPO 也去掉了 KL

GSPO（阿里 Qwen 团队，arXiv:2507.18071，用于 Qwen3）的目标函数里**同样没有 KL 项**。不过要说明一点：GSPO 的**核心贡献并不是"删 KL"**，论文里明说"本文省略 KL 正则项，因为它不是重点"。它真正的主张是：

> GRPO 的不稳定源于**在 token 级别误用/失效了重要性采样权重**。token 级比率 $w_{i,t}=\frac{\pi_\theta(y_{i,t})}{\pi_{\theta_{old}}(y_{i,t})}$ 会给每个 token 一个差异极大的权重，随着序列变长噪声不断累积，再被 clip 放大，最终导致不可逆的模型崩溃。

所以 GSPO 改用**序列级重要性比率**（长度归一化）：

$$
s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{1/|y_i|}
=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)
$$

目标函数为：

$$
\mathcal{J}_{\mathrm{GSPO}}(\theta)=\mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i\hat A_i,\ \mathrm{clip}\big(s_i,\,1-\epsilon,\,1+\epsilon\big)\hat A_i\Big)\Bigg]
$$

它去 KL 的逻辑可以概括为：**"信任域"职责完全由序列级裁剪承担，而 KL 这种额外的软正则（尤其还是 token 级、方差大的估计）变得多余且有害。** 对 MoE 模型来说，序列级似然对专家路由抖动不敏感，因此连 GRPO 必需的 Routing Replay 策略都不再需要了。

---

## 五、共性的洞察

把三者串起来，其实是一条清晰的技术演进线：

| 算法 | KL 项 | 信任域靠什么 | 核心场景 |
|------|-------|-------------|---------|
| PPO / GRPO | ✅ 有 | KL + clip | RLHF（对齐，要贴住 SFT） |
| DAPO | ❌ 去掉 | 非对称 clip + 动态采样 | RLVR（可验证奖励、long-CoT） |
| GSPO | ❌ 去掉 | **序列级** clip | RLVR + 大规模 MoE |

- **为什么以前要 KL**：RLHF 时代，奖励模型不完美、目标是对齐而非"进化"，必须防止策略跑远。
- **为什么现在能去掉**：进入 R1 式的**可验证奖励推理 RL（RLVR）**后，目标是让模型学会全新的推理行为，它**本来就该远离初始化模型**；此时 KL 变成负作用（限制探索、引起熵坍缩、估计有偏/高方差）。
- **谁来补位**：信任域的职责被更纯粹的**重要性比率裁剪（clip）**接管，DAPO 用非对称 clip，GSPO 用序列级 clip，都证明在可验证奖励下**"clip 已足以稳定训练，KL 可以不要"**。

如果你还想深入，我可以进一步展开：k3 估计器的方差问题、DAPO 的 Clip-Higher 为什么能缓解熵坍缩、或 GSPO 的梯度公式推导。