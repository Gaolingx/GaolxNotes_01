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
