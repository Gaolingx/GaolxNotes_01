# LLM教师方法提升学生模型

## 🧑‍💻 User

llm让teacher去提高student我想到的玩法有很多
1. 让强模型当rubic
2. 直接拿输出结果sft
3. 用于训练grm
4. 还可以llm as judge
继续补充

---
## 😀 默认助手 (7)

你列的四条其实都在“teacher 产出什么”这根轴上，即 rubric、answer、reward、judgement 这四类产出。按“产出物”来补齐最不容易漏。下面分六组。

## A. teacher 生产训练数据（SFT 侧）

| 玩法 | teacher 做什么 | 备注 |
|---|---|---|
| **Rejection sampling / Best-of-N (RFT)** | 对每条 prompt 采 N 个回答，用答案或验证器过滤，保留通过的 | STaR、RFT 的最常用基线；比单次采样直接 SFT 质量高很多 |
| **Edit 蒸馏** | 不强求 teacher 从头写，让 teacher 只**改写 student 的错答案** | 数据更贴近 student 的分布，比“纯 teacher 输出”更省 token，也缓解分布漂移 |
| **Self-Refine 三元组** | 生成 `(prompt, student 错答, teacher 修正)` | 天然带 critique 信号，可同时训 SFT 和 critic |
| **Evol-Instruct 难度爬坡** | 把简单题改写成更深、更宽、更多约束的版本 | WizardLM 路线，用来做 curriculum |
| **Instruction Backtranslation** | 给无标注语料**反推** instruction | Humpback 路线，把预训练语料变成指令数据 |
| **合成教科书 / 长文语料** | teacher 写“教材体”长文档，而不是 QA 对 | phi 系列路线；对 small model 特别有效，因为知识密度高 |
| **Hard negative 制造** | 故意生成**似是而非、看起来对但错**的答案 | 给对比学习、reward model、verifier 当负例，比随机负例强 |
| **主动学习 / 难度定位** | teacher 先诊断 student 在哪些题上挂，再**定向出题** | 形成“评测→出题→训练”闭环，性价比最高的一类 |
| **上下文蒸馏** | 把长 system prompt、few-shot、CoT 的效果压进权重 | context distillation，推理时就不用带那坨 prompt |
| **Hint 蒸馏** | teacher 给的是**分级 hint**而不是完整解法 | “Learning to Reason with Hints”：让学生自己走，卡住才给提示 |
| **风格与人设增广** | 同一条指令用多种语气、格式、长度改写 | 提升鲁棒性、减少“AI 味”，也方便做 style control |
| **多语言与跨语言迁移** | 用高资源语言 teacher 生成，翻译到低资源语言 | 反向也行：低资源语料 → teacher 抬成高质量平行数据 |

## B. 偏好与 RL 侧（你第 3、4 条的延伸）

| 玩法 | teacher 做什么 | 备注 |
|---|---|---|
| **On-policy 蒸馏** | **student 自己采样**，teacher 给每个 token 或整段打分、纠正 | GKD / MiniLLM 路线；解决“SFT 数据不在 student 分布上”这个根本问题 |
| **SPIN 式自博弈** | teacher 判断“这段像人写的还是 student 写的” | 把弱模型输出逐步拉到人写分布 |
| **Iterative / Online DPO** | 每轮用当前 student 采 pair，teacher 排序，再 DPO | 比离线偏好数据好，因为 pair 始终在 student 能力边界上 |
| **只给“谁更好”的弱信号** | teacher 不写理由，只给 A>B | 便宜、噪声小，适合大规模 |
| **PRM / 过程奖励** | teacher 标注**每一步**对不对，而不只标最终答案 | Let's Verify Step by Step；贵，但对数学、代码收益大 |
| **PRM 的廉价替代：rollout 打标** | teacher 只标最终答案，靠 student 的多次 rollout 反推步级标签 | 自动过程标注，比人工标便宜一个量级 |
| **Generative Verifier (GenRM)** | 生成式验证器：输出“对不对 + 理由”，而不是一个分数 | 可 test-time scaling（多次验证投票） |
| **Reward-as-code** | teacher **写 reward function 的代码**（含单测），不写自然语言偏好 | 代码可执行、无 reward hacking 空间，适合 RLVR |
| **可验证任务生成** | teacher 造**自带答案或单测**的题（数独、代码、约束满足） | 唯一能绕开“reward model 被 hack”的路子 |
| **环境、用户模拟器** | teacher 扮演 user、工具、对手方，造多轮、agentic 交互 | 把静态 SFT 变成可交互 RL |
| **Red team 闭环** | teacher 专门找 student 的失败模式，产出攻击集 | 再回到 A 组定向补数据 |

## C. 白盒信号（teacher 可拿 logits 才成立）

| 玩法 | 说明 |
|---|---|
| **Top-k logit KD / 正反向 KL** | 反向 KL 更“mode-seeking”，小模型上常更稳；forward KL 覆盖更全 |
| **稀疏 logit 蒸馏** | 只传 top-20 logits，工程上可行，适合闭源 teacher 只暴露 top-k 时 |
| **中间层、attention 对齐** | 要求同 tokenizer、近似架构，跨家族基本不可用 |
| **用 teacher 的熵或置信度做加权** | 低熵 token 当硬标签，高熵 token 加权减弱；也可反用来做 curriculum |
| **对比 logits：teacher 与 student 的差** | 差分信号定位“student 在哪一步开始错”，比只看 loss 有用 |

## D. teacher 当“数据策展人”（最被低估的一组）

| 玩法 | 说明 |
|---|---|
| **质量打分器蒸馏** | 先用 teacher 打一批分，训一个**廉价小分类器**当过滤器，然后去筛海量网页、语料 | FineWeb-Edu 那套思路；teacher 只出现一次，成本极低 |
| **难度分档** | teacher 给数据打难度，用来配 curriculum 顺序和采样比例 |
| **数据 mixture 优化** | teacher 或小代理模型搜索各数据源的配比 |
| **去重与多样性控制** | 用 teacher 判断“语义重复”，而不是靠 n-gram |
| **数据溯源、归因** | teacher 标每条数据命中哪个能力维度，做能力配额 |

## E. 评测与选型（很多团队只在这儿用 teacher，浪费了）

| 玩法 | 说明 |
|---|---|
| **生成 eval 集** | teacher 造评测题、rubric、打分脚本；不要用 benchmark 训练，用 teacher 造私有集防污染 |
| **Checkpoint 选择** | teacher 当 judge 做 model selection，比固定 benchmark 更贴业务 |
| **错误归因 taxonomy** | teacher 把失败分成“不会、不听话、格式错、幻觉”，再对症下药 |
| **模拟 A/B** | teacher 扮演不同用户群体做偏好评测，先筛掉明显差的版本 |

## F. 反向与混合（容易漏的方向）

| 玩法 | 说明 |
|---|---|
| **Weak-to-Strong** | 反过来：用弱模型或 teacher 的**噪声标签 + 置信度损失**去提升强模型，考察可扩展监督 |
| **多 teacher 集成与路由** | 按领域选 teacher、投票、或训一个 router；单 teacher 的偏置会被学生整体继承 |
| **Teacher 当 assistant** | 训一个中间尺寸的“助教”代替超大 teacher 来教小模型，缓解 capacity gap |
| **Scaffold → Weights** | teacher 给学生设计 prompt 框架或工具流，先在推理时用，跑通后再蒸馏进权重 |
| **Self-distillation** | teacher = 自己的历史 checkpoint、EMA 或 model soup，零额外成本 |
| **Cascade 与 Speculative** | 不是训练，但常一起上：student 做 draft、teacher 验；或按难度路由 |

## 几个必须提前想的坑

1. **Capacity gap**：teacher 的 CoT 又长又绕，小模型学的是**形式**不是**推理**。对策：截断、压缩 CoT、只学 hint、用助教模型。
2. **Reward hacking**：teacher 当 judge 会被长度、格式、谄媚 hack。对策：可验证奖励优先、成对比较优先、rubric 与 judge 分离、定期红队。
3. **多样性塌缩**：纯 distillation 会把 student 压进 teacher 的 mode，pass@1 涨、pass@k 跌。对策：保留一部分真数据、mix on-policy 采样、温度或采样多样性约束。
4. **分布漂移的恶性循环**：off-policy 数据训完，student 分布变了，旧数据失效，因此需要**迭代**，且越迭代越要 on-policy。
5. **验证成本**：如果只有 teacher 能判对错，那每轮迭代都要付一次全量 judge 账，因此要想清楚“哪些环节能换成小 verifier 或可执行验证”。
6. **污染与合规**：teacher 可能背出 benchmark 原题；另外商用 API 的 ToS 常限制“用输出去训竞品模型”，得先看条款。

如果要挑投入产出比最高的三条：**主动学习闭环（A 组最后一条）**、**on-policy 蒸馏（B 组第一条）**、**可验证任务 + reward-as-code（B 组后两条）**。这三条都直接针对“数据不在 student 分布上”和“reward 被 hack”这两个真问题。

想深挖哪一条？我可以给具体的数据管线或 loss 设计。

---
## 🧑‍💻 User

说起来，RFT+黑盒distill（针对闭源frontier）现在好像很流行啊，不管 RFT 是指经典的【拒绝采样微调（Rejection Sampling Fine-Tuning）】还是现在火热的【强化微调（Reinforcement Fine-Tuning）】，Teacher + Student 混着玩都是最出效果的黄金组合：

1. 拒绝采样 RFT（ReST / Iterative RFT）里的混合双打：
   如果光拿 Teacher 采样过滤（Off-policy），Student 学多了容易水土不服；如果光拿 Student 自己采样过滤，遇到难题通过率为 0 就断粮了。
   现在最香的玩法是「Student 主采 + Teacher 筛查与兜底改写」：
   - 简单/中等题：让 Student 自己 Rollout N 条，由 Teacher 当严苛的 Judge/PRM（不仅看最终答案对不对，还筛掉「过程凑答案/跳步幻觉」的伪正确轨迹）；
   - 0 通过率难题：一旦 Student 采样全军覆没，让 Teacher 介入——但不是让 Teacher 重头写一篇风格迥异的答案，而是让 Teacher「顺着 Student 写错的草稿做最小化修改（Minimal Edit / Rewrite）」救活成正样本再训，最大程度保留 Student 自己的分布习惯。

2. 强化微调 RFT（GRPO / RLOO）里的 Teacher 强行带飞（比如 LUFFY 框架）：
   在 GRPO 这种组内算相对优势（Advantage）的算法里，可以直接搞「混血 Group」：
   - 比如每道题采样 G = 8 条，其中 7 条由 Student 自己跑，遇到 Student 不会的题，强行塞 1 条 Teacher 生成的黄金轨迹进同一个 Group 一起算 Advantage（配合重要性采样权重修正）。这样直接解决了难题组内全 0、梯度消失的死局；
   - 还能把「Teacher 的分步 Rubric 打分」或「Token 级 Teacher Logits 逆向 KL」跟客观规则 Verifier（0/1 奖励）加权拼在一起当复合 Reward，专治 Student 蒙对答案但过程胡说八道的 Reward Hacking

---
## 😀 默认助手 (7)

方向是对的，但你这套东西里有三个地方最容易翻车：**混血 Group 的梯度其实没走通**、**minimal edit 常常救出“错过程+对答案”**、**黑盒拿不到 token 级逆向 KL**。逐条说。

## 1. 混血 Group：teacher 样本大概率不产生梯度，只是在“修 baseline”

GRPO 里：
$$A_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}$$
全 0 组时 std=0、A=0，梯度确实死了，塞一条 teacher 黄金轨迹进来能救活。但要注意两件事：

**(a) teacher 那条轨迹自己几乎不贡献梯度。** PPO 式 clip 用的是 $\rho_t = \pi_\theta(a_t|s_t)/\pi_{\text{old}}(a_t|s_t)$，而 teacher 轨迹是在 teacher 分布下采的，它相对 student 的似然极低，$\rho_t$ 远偏离 1，基本全被 clip 到 0 梯度。**所以混血 Group 的真实机制是“teacher 样本改变了组内均值/方差，从而给 7 条 student 轨迹提供了正确的负优势”**，而不是“student 在学 teacher 的 token”。这两件事经常被混为一谈。

**(b) 想真的学 teacher，必须额外挂一个模仿项。** LUFFY 那套的核心不是“塞进 group”，而是 off-policy 数据上单独的 SFT/KL 正则项 + 重要性过滤。所以正确写法是：
$$\mathcal{L} = \mathcal{L}_{\text{GRPO}}(\text{student rollouts}) + \lambda_t \cdot \mathcal{L}_{\text{SFT}}(\text{teacher/trajectory})$$
而不是指望 clip 后的 PG 项把 teacher 轨迹学进去。

由此推出三个更干净的替代方案，按推荐度排序：

| 方案 | 做法 | 为什么更好 |
|---|---|---|
| **把 teacher 注入 prompt，而不是注入 group** | 难题时让 teacher 给 hint/分步提示，**轨迹仍由 student 自己 rollout**（hint-conditioned RFT） | 轨迹 100% on-policy，${\rho_t\equiv1}$，无 IS 偏差，还顺便解决了 0 通过率；后期再把 hint 退火掉（hint→no-hint 蒸馏） |
| **teacher 当 value baseline** | 用 teacher 对 rollout 的评分训一个 $V_\phi$，$A=r-V_\phi(s)$ 代替组内均值 | 直接绕开“全 0 组 advantage 退化”，不需要注入任何 off-policy 样本；代价是丢掉 GRPO 免 critic 的显存优势 |
| **混血 group + 退火** | 冷启动阶段 teacher 占比高，随训练把 $\lambda_t$ 从 0.5 退到 0 | 防止 student 长期泡在 teacher 分布里，pass@k 塌缩 |

另外两个细节：**难题上一条 teacher 正样本方差太大**（1 正 7 负），梯度噪声极高，要么每道难题塞 3–4 条 teacher 轨迹，要么干脆别用 PG 学它、只借它修 baseline。**简单题千万别注**，teacher 会把均值抬高、稀释掉本来正确的 student 的优势。

## 2. Minimal Edit：救活的是样本，也可能是“伪正确”

你的动机（保留 student 分布习惯）完全正确，但 minimal edit 有个隐蔽失败模式：**teacher 往往只补最后一步或改个数，前面那段错推理原封不动留着**。这正是你说要筛掉的“过程凑答案”，结果被你亲手造出来又喂回去了。

三个补丁：

1. **强制 edit locality 指标**：edit 必须发生在“首个错误步之后”。用 token 级前缀一致性检查，如果 teacher 改了第 1 步，那它就是 rewrite 不是 edit，直接丢弃或降权。可度量的量：`修改起点位置 / 总长度`。
2. **错误点分叉续写，而不是整段重写**：定位 first-error step $t^*$，保留 student 的 $x_{<t^*}$（前缀完全不动），让 teacher 只写 $x_{t^*}$ 那一步（或给一句 hint），**剩下的尾巴让 student 自己接着 rollout**。这样既有 teacher 的纠偏，又保住了后半段的 on-policy 性，而且天然带“错→对的转折点”信号，做 step-level PRM 标注特别顺。
3. **teacher 判决要用 listwise + 定位**：一次调用把 8 条 rollout 一起给 teacher，要求输出「排序 + 每条的首个错误步 + 错误类型」。比 8 次 pairwise 便宜一个量级，而且顺手拿到 first-error 标注，正好喂给第 2 条的分叉续写。

顺带一个高性价比变体：**hint ladder（最近发展区）**。对 0 通过率的题，给 k 级递增强度的 hint，找到“最小 hint 级别下 student 能自己做出来”的那一档。这种轨迹信息量最大（既非抄答案也非完全不会），比 teacher 直接给黄金轨迹更适合小模型。

还有一条硬门槛：**0% 的题可能是 student 能力根本够不着**。判据很简单——拿救活的数据做几步 SFT，pass rate 还是 0，就说明是噪声源，扔掉。别把“不可学”的题当宝贝。

## 3. 黑盒到底能拿到什么：token 级逆向 KL 基本不可行

这是你第 2 条里技术上最站不住的一点，得澄清方向：

$$\underbrace{D_{KL}(p_T \| p_S)}_{\text{teacher 采样，student 打分}} \quad\textbf{可行}\qquad \underbrace{D_{KL}(p_S \| p_T)}_{\text{student 采样，teacher 打分}} \quad\textbf{基本不可行}$$

- **正向 KL / 序列级 KD**：teacher 出样，student 算 logprob，纯黑盒完全可行。这也是绝大多数“黑盒蒸馏”的实质。
- **逆向 KL（GKD 的 on-policy 版本）**：需要对 **student 采出的 token** 求 $p_T(x_t\mid x_{<t})$，等价于要求 teacher 提供“任意续写的 scoring / logprobs”接口。部分 API 有 `logprobs`（通常只到 top-20），所以只能拿到**截断支撑集上的稀疏近似**——student 概率质量落在 teacher top-20 之外的部分全部丢失，逆向 KL 有系统性偏差，而且按 token 计费，成本爆炸。
- 所以工程上更现实的是：**teacher 给序列级/步级标量分数**（rubric、judge、PRM），把 off-policy 的分布对齐交给 SFT/KL 正则，别去碰 token 级逆向 KL。

顺便说黑盒 API 清单里真正好用的是：多次采样（拿多样性）、listwise judge（拿排序+定位）、rubric 分解打分、以及“按指令改写”（拿 minimal edit）。这四样拼起来已经能覆盖 90% 的收益。

## 4. 复合 Reward 的拼法：硬门控 + 非对称 + 二元分解

“规则 verifier + teacher rubric 加权”最容易死在 scale 不匹配和 reward hacking 上。建议这样拼：

$$r = \mathbb{1}[\text{correct}]\cdot\big(1 - \lambda\cdot \text{proc\_penalty}\big) - \mu\cdot \mathbb{1}[\text{format\_violation}]$$

三条约束：
- **正确性是硬门控**：过程分**只能扣分，不能把错的捞上来**。反过来（过程分能加分救错误答案）就是 reward hacking 的直通车。
- **rubric 必须分解成二元判据**，别用 1–10 Likert。每条判据 yes/no，再平均，方差小十倍，也方便做 per-criterion 的 hack 监控。
- **长度要单独处理**：长度惩罚显式写进 reward，别指望 judge 自己公平；同时用 per-token 归一化的 KL 项（$\beta \cdot \frac{1}{|o|}\sum_t D_{KL}$）避免长回答天然占优。

## 5. 训练 schedule：别一条路走到黑

```
Stage 0  冷启动：minimal-edit / 分叉续写数据做 SFT（或 teacher 数据 + student 数据 1:1 混合）
Stage 1  混合 GRPO：难题注入 teacher（占比 λ 从高退到低），简单题纯 student
Stage 2  纯 on-policy GRPO + 自蒸馏 PRM（用 Stage 1 的 teacher 标注训出的小 PRM 替代 teacher 调用）
Stage 3  hint 退火 + 短 CoT 压缩（把 teacher 的冗长风格挤掉）
```

<details>
<summary>工程账：teacher 调用成本怎么砍（点击展开）</summary>

- **judge 蒸馏**：先用 teacher 标 2–5 万条，训一个小 PRM/verifier，之后 90% 的过滤走小模型，teacher 只处理小模型置信度低的样本（主动学习）。
- **listwise 一次判 8 条**，比 pairwise 省 8 倍调用。
- 按 `(prompt, trajectory)` 哈希缓存；同一 prompt 的多轮迭代复用。
- 分层：先跑便宜 verifier，只有 verifier=1 的“疑似伪正确”才送 teacher 做过程审查——这正是性价比最高的一刀，因为**伪正确样本才是 RFT 里最毒的东西**。
</details>

<details>
<summary>评测陷阱：几个会让你误判“涨了”的指标（点击展开）</summary>

- **pass@1 涨、pass@k 跌**：纯 teacher 数据/长退火不足的典型症状，务必同时报 pass@k 和输出熵。
- **风格漂移**：平均长度、连接词频率、“Let me...” 这类 teacher 指纹，量化一下就知道是不是在背 teacher。
- **难度带监控**：只保留 0.1–0.9 通过率的题，训练集通过率分布要画出来，别让 curriculum 悄悄退化成只会做中等题。
- **污染**：teacher 可能背出 benchmark 原题，私有 eval 集必须有。
</details>

一句话总结：你这套的骨架（student 主采、teacher 兜底、混血 group 破零、复合 reward 防伪正确）是对的，但**把 teacher 从「group 里的一条样本」改成「prompt 里的一个 hint + 首错点的一次分叉续写」**，能在不引入任何 off-policy 偏差的前提下拿到你想要的 90% 效果。剩下 10% 靠那条单独的 SFT/KL 项补。

要不要我把 Stage 1 的混合 group 具体实现（advantage 归一化怎么排 teacher、$\lambda_t$ 怎么退火、clip 会不会把 teacher 项吃掉）写成伪代码？
