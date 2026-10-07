# GRPO长程局限

## 🧑‍💻 User

@联网搜索 我发现agentic rl loss用grpo对long horizon太不友好了，除非是训一些math task，不然现在agentic训练建议别上grpo，long horizon要不就ppo，或者opd，要不完全offline都好些，直接sft agentic traces。grpo在math的reasoning rl好用主要是math的horizon不长，长程agentic task，比如swe那些，rollout一条trajectory都够费劲了更别提grpo。之前看glm5.2paper agentic是把critic请回来了，你怎么看

---
## 😀 默认助手 (7)

下面把联网查到的证据 + 我的判断一起给你。核心结论先说：**你的直觉大方向对，但"GRPO 不行 / 必须把 critic 请回来"这个对立面其实被简化了**——真正的分界线不是 critic vs critic-free，而是**"advantage 是 trajectory-constant 还是 token/state-level"**，以及**"同 prompt 的 rollout 还是否可比"**。GLM-5.2 换 PPO 恰好发生在后者被破坏的那一档任务里。

---

## 1. 为什么 GRPO 在 math 行、在 long-horizon agentic 不行

GRPO 的 advantage 是**把整条轨迹的组内相对收益，广播给这条轨迹里每一个 token**：

$$
\hat{A}_{i,t} \;=\; \frac{R_i - \operatorname{mean}(R_{1:G})}{\operatorname{std}(R_{1:G})}\quad \forall\, t
$$

这里面藏了三个在 math 里无所谓、在长程 agentic 里致命的假设：

<details>
<summary><strong>① 广播天花板（broadcast ceiling）：所有 token 拿同一个 credit</strong></summary>

长程任务里，一条几十步轨迹中真正决定成败的可能只有 2–3 个 action（比如某个错误的编辑、某次没先跑测试）。GRPO 给整条轨迹每个 token 同一个标量，**无法做时序 credit assignment**。步数越长，这个"平均主义"误差越大。Limes Labs 的 *The Broadcast Ceiling* 工作汇就是一个专门针对 trajectory-constant estimator 的 diagnostic——结论是这类估计器在长程、异质 credit 场景下有系统性上限。

PPO 用 critic 给出 $V_\phi(s_t)$，配 GAE 得到逐步 advantage：

$$
\hat{A}_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t = r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
$$

这才是"token/state-level credit"。
</details>

<details>
<summary><strong>② 长程下组内方差爆炸 / 奖励全 0 梯度消失</strong></summary>

长程 SWE 类任务，单 rollout 成功率可能只有几个百分点。一个 group 里绝大多数 rollout 都是 0 分 → $\operatorname{std}(R)\to 0$ → advantage 全趋近 0 → **几乎没有梯度**。而 group size 又不能随便加大，因为**每加一条 rollout 就是线性加一份最贵的开销**（见③）。

math 恰好相反：hint 短、终局奖励密集、成功率有意义，group size 16–64 就能把方差压住，GRPO 的组内 baseline 近乎免费又低偏。**这就是 math reasoning RL 好用、long-horizon agentic 不好用的本质——不是算法错了，是任务形状变了。**
</details>

<details>
<summary><strong>③ rollout 是瓶颈，而 GRPO 的"免费 baseline"其实最贵</strong></summary>

GRPO 相比 PPO 省掉 critic，但代价是**每个 prompt 要采 K 条完整轨迹**。long-horizon 下最贵、尾延迟最长（tail latency）的恰恰就是 rollout。GLM-5 报告里专门有一整节在讲怎么让 group-wise 的 async rollout 活下来：TITO、double-sided importance sampling、丢掉过期 off-policy 样本、DP-aware routing——**这些全是给 GRPO-family 打的补丁**。换句话说，你把 GRPO 用在长程上，工程成本最后都会还给 rollout 侧。
</details>

---

## 2. GLM-5 vs GLM-5.2：到底改了什么

正本清源，这里有两份不同文档，别混：

| | **GLM-5**（arXiv 2602.15763，2026-02） | **GLM-5.2**（z.ai / HF blog，2026-06） |
|---|---|---|
| Reasoning RL | **GRPO + IcePop**（去 KL，group size 32） | — |
| Agentic RL | **仍是 group-wise（critic-free）**：采 K 条 trace，advantage $r(x,y_i)-\bar r(x)$；async 解耦 | **critic-based PPO**，critic 估计 **token-level advantage** |
| 触发原因 | 长程 → GPU idle，靠 async + 各种 off-policy 修正扛 | **compaction 把超长轨迹切成数量/长度都可变的多条 sub-trace** |
| 蒸馏 | on-policy cross-stage distillation | **并行 OPD**，把 10+ 个专家模型 merge 进最终模型（约 2 天） |
| 额外 | — | anti-hack 模块（rule filter + LLM judge，在线拦截、只废那一步不废整条） |

GLM-5.2 原文（关键句，值得背下来）：

> "…once a super-long trajectory is split by compaction into multiple sub-traces, different rollouts under the same prompt yield different numbers of trainable traces with highly variable lengths. **We therefore move from group-wise optimization to a critic-based PPO formulation that learns from individual rollouts**, relying on a critic to estimate token-level advantages rather than group-relative comparisons. This single-rollout formulation fits compaction naturally, as it **places no constraint on how many traces a prompt produces or on their relative lengths**."

翻成一句话：**compaction 炸掉了 GRPO "同组可比" 的前提**。当一条 prompt 被切成数量不一、长短不一的 sub-trace，"组内归一化" 这个 baseline 就不再是一个合法的对照，PPO 的单轨迹 + critic 反而更自然。

---

## 3. 我怎么看"把 critic 请回来"

<details>
<summary><strong>观点：方向对，但这不是 GRPO 的死刑判决，是"任务形状选择工具"</strong></summary>

**① 把 critic 回来解释成"GRPO 全面被否"是过度解读。** GLM-5.2 的迁移条件是**很具体的**：compaction-heavy、sub-trace 数量可变、长度极不均衡。这个 regime 下 group 被拆得没法比，critic 几乎是唯一自然选择。但反例同样硬：

- **HGPO**（ICLR 2026, arXiv 2602.22817）在 ALFWorld/WebShop 上**全程 critic-free**，还专门指出 PPO-with-critic 更差（Qwen2.5-1.5B ALFWorld：PPO 54.4 vs GRPO 72.8 vs HGPO 92.8）。它修的是 GRPO 的"历史上下文不一致"偏差，而不是换成 critic。→ 说明**critic-free 路线在被正当地修，且能赢**。
- Limes Labs 那篇也把 VIMPO-style policy-implied values、BRPO-style prefix baselines、structural critics、process rewards 列为有意义的"中间地带"。

**② 真正的坐标轴不是二元的，是三维的：**
1. **credit 粒度**：trajectory-constant（GRPO 原始） vs token/state-level（PPO/critic、process reward、stepwise group）。
2. **组是否可比**：同 prompt 干净组 → GRPO 划算；被 compaction / 变长 / 异步打散 → 必须换 baseline。
3. **算力/延迟预算**：rollout 贵 → 倾向 critic、倾向 offline、倾向 OPD；rollout 便宜且奖励可靠 → GRPO-family 更省。
</details>

**③ 一个容易被忽略的事实：critic 回来不是为了"更准"，而是为了"能装下变长"。** 单轨迹 + critic 让"一条 sub-trace 也能训"成为可能，这对 compaction 训练是结构性刚需，而不是精度提升。所以 GLM-5.2 是**系统约束驱动**的算法选择。

---

## 4. 对你实践路线的建议

你说的 PPO / OPD / offline SFT 三条路，我按**任务形状**给你排一下，而不是按"哪个更先进"：

| 场景 | 首选 | 理由 |
|---|---|---|
| 短程、终局可验证奖励（math/代码题） | **GRPO 家族**（GRPO/DAPO/GSPO） | 组内 baseline 近乎免费且低偏，critic 纯属额外成本 |
| 长程、奖励稀疏、组仍可比（如 ALFWorld 类） | **critic-free 改造**：GiGPO / **HGPO**、prefix baseline、process reward | 先把 GRPO 的偏差修好，保留下"无 critic"的工程收益 |
| 长程 + compaction/变长 sub-trace（SWE、terminal、多文件） | **critic-based PPO**（GLM-5.2 路线） | 组不可比，单轨迹 + token-level advantage 才能容纳变长 |
| rollout 是唯一瓶颈、有高质量专家轨迹 | **offline SFT / rejection-sampling + OPD** | 不在线探索，避免最贵的 rollout；OPD 还能 merge 多专家、防遗忘 |
| 想省 critic 训练又要有时序信号 | **OPD / 策略蒸馏（teacher→student，on-policy）** | 用现成强教师提供 dense 监督，绕开 critic 不稳定问题 |

几点补充判断：

- **OPD 不是 GRPO 的替代品，是"奖励/信用信号来源"的替代品。** 它本质是有监督的（教师给信号），能解决"没有好 critic、奖励又稀疏"，但**不会自己探索出新策略**。GLM-5.2 拿它来 merge 10+ 专家、GLM-5 拿它做 cross-stage 防遗忘——都是"合并/巩固"，不是主训练循环。你想纯 offline 训 agentic trace，OPD 才是那套的正确落点，纯 SFT 会更弱。
- **offline SFT agentic traces 能上，但天花板在 exploration 和 self-correction。** 配合 rejection sampling / masking 错步（GLM 就是这个做法：错步保留但 mask loss）能显著抬上限；但一旦你依赖在线环境反馈（跑测试、改代码闭环），纯 offline 会很快到顶。
- **别忽略 anti-hacking。** 长程 coding RL 的奖励是 pass/fail verifier，天然易 hack（读 eval 文件、curl 拉参考解）。GLM-5.2 花了一整节做在线拦截。**算法再好，reward 信号被 hack 就是白训**——这点在 agentic 里比选 GRPO 还是 PPO 更关键。
- **算法的战场其实在 infra。** GLM-5 的 agentic RL 一大半篇幅在 async + off-policy 修正（TITO、双面 IS、丢过期样本、DP-aware routing）。你如果决定上 GRPO-family 跑长程，真正决定成败的是这些,而不是 loss 公式。

---

### 一句话总结

> **GLM-5.2 把 critic 请回来，不是因为"长程就该用 PPO"，而是因为 compaction 让"同组可比"这个 GRPO 赖以成立的前提崩了。** 正确的表述是：**credit 应该是 trajectory-constant 还是 token-level、组还香不香，决定了你选 GRPO-family、critic-free 的 stepwise 变体（HGPO/GiGPO），还是回归 critic-PPO；math 只是恰好落在这个光谱最省事的一端。** 你的 PPO / OPD / offline 三选一，本质是这道题在不同预算下的三个解，而不是谁替代谁。

**主要来源**
- GLM-5 技术报告：`arXiv:2602.15763`（Reasoning RL = GRPO+IcePop；Agentic RL = group-wise + async + TITO/双面 IS；on-policy cross-stage distillation）
- GLM-5.2 官方 blog：`z.ai/blog/glm-5.2` 与 HF `zai-org/glm-52-blog`（group-wise → critic-based PPO；OPD merge 专家；anti-hack）
- HGPO（critic-free 长程，含 PPO 对比）：`arXiv:2602.22817`（ICLR 2026）
- Burny's Substack：*GLM-5.2 moves from GRPO to PPO*（对"两种锤子锤两种钉子"的解读）
- Limes Labs：*The Broadcast Ceiling*（trajectory-constant estimator 的信用分配上限诊断）

---
## 🧑‍💻 User

是的，实际上 agentic 强化学习的确有很多理由返回 PPO：
1. infra 上，PPO 不需要 decode，只需要 prefill，只增加了 1/4 的计算
2. 计算能力上，相当多的 bad pattern（错误 toolcall 等）是应该可以被 critic model 一次前向传播计算的
3. agentic RL 越来越长程、越来越近似经典的 MDP，没有理由继续使用 AI4MATH 中 contextual bandit 式的 GRPO 型算法

确实很有理由怀疑agentic 强化学习会逐渐返回 PPO

---
## 😀 默认助手 (7)

基本同意——但我想把这句"返回 PPO"拆得更准一点，因为你三条理由里第三条才是根，前两条是它的推论。

**更准确的表述是：返回 *temporal credit assignment*（逐步信用分配），PPO 只是它最常见的一种实现。** 真正被否掉的，是 AI4MATH 里那个隐含假设——**"prompt → response 是一个 bandit arm"**。

形式化地说，GRPO 是 MDP 在 $T=1$ 处的特例：$T=1$ 时，最优 baseline 就是组均值；当 $T$ 拉长，最优 baseline 变成 $V(s)$ 或 prefix baseline，组均值只是它在 horizon=1 下的投影。所以"agentic 返回 PPO"不是时尚回摆，是把一个 $T=1$ 的近似换回正确的 $T$。这个视角下你的三条理由其实是：**#3 是原因，#1/#2 是它可行且划算的证据。**

---

## 逐条看你这三点

### 1. Infra：critic 只 prefill —— 方向对，而且我认为它比你说的还更硬

先把成本结构摆出来：

| 组件 | 计算形态 | 在关键路径上? | 相对成本 |
|---|---|---|---|
| Rollout | **decode**，$L$ 次串行前向 | ✅ 是 | 高，且 GRPO 每 prompt 要 $K$ 条 |
| Actor update | prefill fwd + bwd | 否 | $\sim 3\times$ fwd |
| Critic value | **prefill** fwd（可与 actor logprob 前向融合） | 否 / 可 async | $\sim 1\times$ fwd；共享 backbone 时 ≈ 只有 value head |
| Critic update | fwd + bwd | 否 / 可 async | $\sim 3\times$ fwd，或复用 actor 前向 |

你说的"1/4"我理解为**墙钟/带宽口径**而非 FLOP 口径：decode 是 memory-bandwidth bound（每步都要把权重整体读一遍），prefill 是 compute-bound 且高度可并行，单 token 成本比常常就是 3–10×。所以 critic 只走 prefill，边际成本约等于一次 decode 等价物的 $\frac14$，这个量级站得住。

但补两点，一个让它更划算、一个会把它打回原形：

- **更划算**：如果 actor 和 critic 共享 backbone（只加一个 value head），那"额外一次前向"基本不存在——你本来就要为 policy ratio 算一遍 logprob，value head 挂在这条前向上几乎是免费。注意 GRPO **不是**零前向：它同样要 prefill 算 ratio 的 logprob，所以 PPO 相对它的真实增量只是 *value head + 一次反向*（或一个独立小型 critic）。
- **打回原形**：① 用**独立 critic model**（不共享权重，为稳定性常见做法）→ 多一整个模型 + optimizer state，而长 context agentic 里**内存(KV cache + 长 prefill)往往才是真瓶颈**，不是 FLOP；② 需要覆盖 compaction 后的全 context 做 per-token value → prefill 长度本身巨大，"prefill 便宜"的前提就弱了；③ 异步 critic 有 staleness bias，得靠 clipping / importance 修。

**但我认为比"1/4"更本质的一条你没说：critic 可以 off critical path，GRPO 的成本在 critical path 上。** GRPO 把 $K\times$ 的 rollout 乘在整条流水线最贵、尾延迟最长的环节上；PPO 的 critic 可以异步、滞后、甚至在训练循环之外对历史轨迹离线打分。当单条 rollout 成本 → ∞，这个差别就是决定性的——Amdahl 意义上的。

### 2. 一次前向算 bad pattern —— 机制对，但要分清 $V$ 和 $Q$

你的直觉有严格对应。只有终局奖励时，GAE 里 $r_t = 0$（末端除外），于是

$$
\hat{A}_t=\sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l},\qquad \delta_t=\gamma V(s_{t+1})-V(s_t)
$$

它退化成 $V$ 的差分——**critic"看一眼 prefix 就判断要凉"**，这正是你说的"一次前向算 bad pattern"，而且是 state/token-level 的。但有三点要澄清，否则会高估 critic：

<details>
<summary><strong>$V(s)$ 是 state value，不是 action value</strong></summary>

它能说"这个 prefix 期望回报低"，**不能**直接说"同一 state 下哪个 action 更差"（那要 $Q(s,a)$）。而且它无法把"action 烂"与"环境随机"分开——确定性环境里没问题，随机环境里这正是 critic 偏差/方差的来源。所以"critic 一次前向就能定位错误 toolcall"在确定性 sandbox 里成立，在嘈杂环境里会退化。
</details>

<details>
<summary><strong>$V$ 是 policy-specific，判别力上限 = 策略自身可恢复性</strong></summary>

它学的是"**当前策略**从这个 prefix 通常能拿到什么"，不是"这步客观上对不对"。一个错误的 toolcall，如果当前策略后面总能把局面救回来，$V$ 就不低。所以 critic 能捕捉的是**"会导致不可恢复状态的坏动作"**，而不是"所有坏动作"。
</details>

<details>
<summary><strong>critic 抓不到 reward hacking</strong></summary>

它学的就是 reward，自然会喜欢 hack 出来的轨迹。所以 GLM-5.2 的 anti-hack 用的是 rule filter + LLM judge，**不是** critic。别指望 critic 兼任 verifier——这两个信号是互补的，不是替代的。
</details>

### 3. MDP 化 —— 最强的一条，但"没有理由"要降级为"没有**算法上**的理由"

同意，而且它顺带解释了 GRPO 为什么曾经"对"：**RLVR 任务本身就是 bandit**（prompt → 单个 response → 终局可验证奖励），bandit 算法配 bandit 问题，所以它赢，且赢得干净。#3 说的正是问题形状从 bandit 滑向 MDP，算法就该跟着回 MDP。

但"没有理由继续用"我建议降级成"**没有算法上的理由**"：保留 group 方法还有真实的**工程**理由——无 bootstrap 偏差、无 value 超参数、无 value head 内存、对 reward scale 不敏感、infra 简单。这些理由是真实存在的，只是在长程下**边际收益在快速衰减**。

还要注意一个中间地带没被杀掉：**HGPO / GiGPO** 把 group-relative 从"整条轨迹"下移到 **step level**。它们和 PPO 的差异只是"**用组做 baseline 还是用 critic 做 baseline**"，而不是"有没有 temporal credit"。所以更严谨的说法是：**credit 粒度在细化，baseline 从"组"换回 $V$ 是其中一条（重要的）路，不是唯一路径。** GLM-5（arXiv 2602.15763）自己的 agentic RL 至今仍是 group-wise，就是个活证据——它和 GLM-5.2 的差别只在后者遇到了 compaction 把组打散。

---

## 反方向：PPO 也会输的地方

别把它讲成单向收敛，否则下一波又会有人往回搬。PPO/actor-critic 的固有弱点在 agentic 里一样会被放大：

- **POMDP + 随机环境 + 稀疏奖励** → $V$ 难学、有偏且有高方差；而在组可比时 GRPO 的组 baseline 至少是**无偏**的。这是 classic bias-variance 交换，不是免费午餐。
- **超长 context 的 value 泛化差**，bootstrap 误差沿轨迹累积。
- **reward 可 hack 时，value 会放大 hacking**（它学 reward）。
- **独立 critic 的训练稳定性 + 内存**——这恰好是当年 GRPO 流行的原始理由，现在只是被"rollout 太贵"盖过去了。

换句话说：**PPO 不是因为"更准"赢，是因为"把成本挪出关键路径 + 能做 state-level credit"赢。** 一旦任务回落成短程 bandit，它立刻又不划算。

---

## 落地判断

- **主趋势**：actor-critic 回归。compaction-heavy、多轮工具、长 context 的 SWE / terminal / computer-use 会最先换。
- **但不是回到 2017 的 vanilla PPO**，更可能长这样：**异步 + 共享 backbone 的 critic（或 value head）+ process reward / $Q$ 混合 + verifier 做硬约束 + critic 可离线打分**。critic 能给"历史/离线轨迹"打分这件事很关键——它把"必须在线 rollout 才能拿到的信号"变成"可离线 amortize 的监督"，正好和你上一轮说的"完全 offline 也行"在架构上合流了。
- **GRPO 不会死**，会退回到它最擅长的一格：短程、终局可验证、rollout 便宜（数学、单轮代码、浅 toolcall）。这一格它至今是优解。

一句话：**你说的"agentic 会逐渐返回 PPO"，我改一个字——会返回"信用分配"，而 PPO 是当 $T$ 变大、rollout 变贵之后，在这件事上**边际成本最低**的那把扳手。**

---
## 😀 默认助手 (7)

AbortError
Request was aborted
