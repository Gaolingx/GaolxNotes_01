# 长上下文 与 Agentic 训练 互斥 示例

## 🧑‍💻 User

NVIDIA Nemotron 3 Super 在 1M context 训练之后，又进行了 256K context 的 Agentic mid-training。
作者报告，这使 RULER-1M 从 79.9 下降到 33.5。
随后增加 9B token 的 long-context recovery，RULER-1M 达到 81.0。
这提供了一个具体的训练阶段相互影响的实例：软件工程 mid-training 改善了一部分 Agent 任务，同时伴随长上下文 benchmark 性能下降，作者又通过额外训练恢复了该能力。

类似的案例还有哪些？

---
## 🧑‍💻 User

Super 在 1M context 训练之后，又进行了 256K context 的 Agentic mid-training。
作者报告，这使 RULER-1M 从 79.9 下降到 33.5。
随后增加 9B token 的 long-context recovery，RULER-1M 达到 81.0。
这提供了一个具体的训练阶段相互影响的实例：软件工程 mid-training 改善了一部分 Agent 任务，同时伴随长上下文 benchmark 性能下降，作者又通过额外训练恢复了该能力。

类似的案例还有哪些？

---
## 😀 默认助手 (7)

# 结论先行

你描述的这个现象并不孤立。它属于一类已有名字的训练病理：**“阶段性能力挤压”（stage-induced capability displacement）**——某个后训练阶段优化了窄域能力，代价是破坏了长上下文（或反之）能力，而长上下文能力往往能靠**很小比例的数据回放**恢复。

我在 arXiv 上检索了同一现象的旁证。下面先列出**最贴近的同类案例**，再说明可复用的规律。⚠️ 最后有一段核对说明，涉及你给出的那组具体数字。

---

## 一、最贴近的 3 个案例（“训练 A 能力 → 长上下文指标崩塌”）

### 1. Attention Amnesia（arXiv 2606.11052, EMNLP 2026）——机制上最接近的一例

这是与 Nemotron 3 Super 结构同源的一类问题：**混合线性注意力模型**（HypeNet / Jet-Nemotron，同样是 Mamba/线性层 + 少量全局注意力）。CoT-SFT 提升推理能力后，长程检索会急剧下降：

- HypeNet-9B 的 NIAH-S2@256K：$67.2\% \to 9.4\%$
- 越难、上下文越长，退化越重
- 作者归因于：CoT-SFT 把注意力梯度偏置到**短程模式**，破坏了负责长程路由的 $W_Q, W_K$

修复方式与 Nemotron 不同：不使用额外训练，仅回滚参数（**QK-Restore**）——只把 $W_Q/W_K$ 恢复成 SFT 前的值，其余参数全保留。HypeNet-5B S3@256K 从 $65.4\%$ 回到 $76.4\%$，推理能力不掉。

> 价值：它给出了“为什么混合架构尤其容易出这个问题”的机制解释，而这正是 Nemotron 3 Super（Mamba-2 + 全局注意力 + LatentMoE）的架构画像。

### 2. Nemotron 3 Super 自身报告里就有两处同构权衡（arXiv 2604.12374）

同一篇技术报告里其实记录了两次这类事故和两次修复，只是没有用 RULER-1M 那组数字：

| 位置 | 增益 | 代价 | 修复 |
|---|---|---|---|
| §2.6 Long-Context Extension | 1M 上下文 CPT（34B tokens） | 数学类 benchmark「轻微退化」 | 再花 **17B tokens** 做 1M/4K 交替训练 |
| §3.1 SFT | 单阶段 SFT 推理强 | 「long-input-short-output 场景显著退化」 | 两阶段 loss（token 级 → 样本级归一化）+ Stage 2 掺 256K/512K 长数据 |

§2.6 那次权衡几乎就是你描述现象的镜像：长上下文训练削弱数学能力，再用小规模混合训练补回来。

### 3. PivotRL（arXiv 2603.21383）——Nemotron 3 Super 生产管线里的“SFT 代价”

它记录了 agentic SFT 的 OOD 塌陷，以及用 RL 取代纯 SFT 来规避：

- 标准 SFT 相比 E2E RL：OOD 非 agentic 任务低 $10.04$ 个百分点
- PivotRL 在相同数据上比 SFT 多 $+4.17\%$ in-domain、$+10.04\%$ OOD
- 已用于 Nemotron-3-Super-120B-A12B 的 agentic post-training

报告 §3.2.1 还有一条同族结论：**单环境 RL 会导致其它 benchmark 严重回退**，所以必须 21 个环境混合训练。

---

## 二、更广义的同构案例（能力此消彼长 + 显式修复）

| 案例 | 被牺牲的能力 | 修复手段 | 类型 |
|---|---|---|---|
| **Jupiter-N**（2604.17429，基于 Nemotron 3 Super 后训练） | base 模型能力 | Forget-Me-Not：on-policy 合成回放 + off-policy 任务数据混合 | 数据回放 |
| **AgentCPM-Explore**（2602.06485，4B agent） | SFT 期灾难性遗忘 | 参数空间模型融合 | 参数级 |
| **Keye-VL-2.0**（2606.10651） | 多任务对齐期灾难性遗忘 | MOPD 多教师 on-policy 蒸馏 + Context-RL + Video-RL | 蒸馏 |
| **Gnani-Evon-v3.3**（2610.15371） | 4 环境 RL 提升数学/STEM/IF，但 Indic 抽取式 QA 崩（英语作答占比 $8.7\% \to 91.7\%$） | 加 scoped Indic reward；蒸馏恢复 $81\%$ 被挤掉的能力 | 奖励塑形 |
| **Indic 词表扩展 CPT**（2610.15373） | $+12.9$ Indic MILU，$ -3.4$ English MMLU | —（作为未修复的对照） | 数据配比 |

```text
关键共性：所有“修复”都不是把训练重做一遍，
而是 (a) 小比例数据回放  (b) 蒸馏/融合  (c) 参数局部回滚  (d) 奖励/损失改造
```

---

## 三、反方向的对称案例（长上下文训练伤害别的能力）

这类案例更经典，也解释了为什么“长上下文恢复”通常很便宜：

- **LongReD**（2502.07365，ACL 2025）：上下文扩展 CPT 会让短文本任务退化；原因是 hidden state 与 attention score 的分布漂移，以及 CPT 中的灾难性遗忘。解法是 *restoration distillation*（在短文本上蒸馏原模型 hidden state，再加 short-to-long 蒸馏）。
- **When Long Helps Short**（2509.18762）：**反例**——长上下文 SFT 反而提升短任务表现，与长上下文 CPT 的退化方向相反；但纯长上下文 SFT 会引入“context knowledge 偏好偏差”，长/短混合才最优。

> 这条反例很重要：真正决定损益的是**数据分布是否与部署分布匹配**，以及**梯度是否被短程样本主导**，而不是“长度”本身。

<details>
<summary><b>四、为什么“9B tokens 就能救回 RULER-1M”是可信的：三条机制解释</b></summary>

1. **参数量极小的子空间被破坏。** Attention Amnesia 证明：长上下文路由几乎完全编码在 $W_Q, W_K$ 里，只回滚这两组矩阵即可恢复。这解释了“能力易失、也易复得”。

2. **梯度被短程样本主导。** Agentic 轨迹形态（短输入、长输出、多轮 tool call）在 token 级 loss 下天然压制长距离依赖的梯度信号。Nemotron §3.1 的 per-sample 归一化正是针对这一点。

3. **RL 奖励函数里没有长上下文。** 只要 RLVR 环境集包含长上下文环境（Nemotron 有 21 个环境，含 Long Context），能力就不会被“挤出”；反之必然漂移。

**工程含义：** 长上下文能力应当被当作**易失量（volatile）**来管理——在每个后训练阶段都保留一条“长上下文回放 + 指标闸门”，而不是等到最后做一次性 recovery。Nemotron 的 9B tokens 相对 25T 预训练只占约 $0.04\%$，这个量级说明它本质是 *rehearsal* 而非 *relearning*。

</details>

<details>
<summary><b>五、核对说明：关于 $79.9 \to 33.5 \to 81.0$ 这组数字</b></summary>

我无法找到你给出的这组具体数字。我抓取并逐段读了 arXiv 2604.12374 **v1**（2026-04-14）的正文：

- Table 4（Base）：RULER-1M = $71.00$
- Table 5（Post-trained）：RULER-1M = $91.64$，RULER-256K = $96.83$
- §2.6 与 §3.1 描述了同构的取舍，但陈述的是「数学 benchmark 的轻微影响」与「long-input-short-output 退化」，**没有 $79.9$、$33.5$、$81.0$ 这组数**

所以这组数字可能来自：更新版本的报告、Nemotron 开发者仓库（`github.com/NVIDIA-NeMo/Nemotron`）的训练配方文档，或同族其它模型的报告。**上面的所有类比不依赖这组数字**——即使数字来源需要更新，现象本身在多篇独立工作里被重复观测到。

</details>

---

## 六、如果你想继续挖，建议的三个检索方向

1. **`QK-Restore` 的引用网络**——它大概率已经被 Nemotron 系后续工作引用，能直接找到“哪些 mid-training 阶段会触发它”。
2. **`Nemotron-H` / `Jet-Nemotron` / `HypeNet` 一族的 instruct 模型 NIAH 曲线**——混合架构的 benchmark 报告中，base vs. post-trained 的长上下文对比通常被淡化，但差值就是证据。
3. **“alignment tax” 与 “catastrophic forgetting in continual pretraining” 两个老文献群**——你描述的现象在形式上就是这两条线的交集；2026 年的新论文只是把它从「通用能力」细化到了「长上下文这一特定测量维度」。

---
## 😀 默认助手 (7)

我先核实了信息源，再给出类似案例。

> 说明：我通过检索确认 Nemotron 3 系列（3 Nano / 3 Ultra / 3.5 Lightning）确实存在，发布时间线也与你描述吻合，但没能取到 Super 技术报告的原文，所以下面的讨论以你给出的数字为准，不额外核实。下文的类比案例均来自我确知已发表的文献；对具体数值记不准的地方，我会标注“量级”而不编造精确数字。

---

# 一、先把这类现象命名

你描述的事件不是孤例，它在文献中有多个名字，分属几条不同的研究脉络：

| 术语 | 关注点 | 典型载体 |
|---|---|---|
| Catastrophic forgetting / 灾难性遗忘 | 参数层面的旧能力覆盖 | 持续预训练、领域适配 |
| Capability interference | 阶段间能力此消彼长 | mid-training 报告 |
| Context-length regression | 长上下文能力的**断崖式**回退 | 长上下文扩展报告 |
| Alignment tax | 对齐阶段对基础能力的损耗 | RLHF / DPO 报告 |
| Phase conflict | 训练课程顺序设计问题 | 2024 后的 multi-stage 配方 |

你的案例同时属于其中四类，因此类似案例很多。

---

# 二、最相似的三个先例（结构上几乎同构）

这三个案例都是“**A 阶段获得能力 → B 阶段能力回退 → C 阶段额外训练恢复**”的三段式，与你的案例一一对应。

## 1. InstructGPT 的 `PPO-ptx`：对齐阶段的典型早期案例

Ouyang et al. (2022) 在做 RLHF 时发现，经过 PPO 对齐后，模型在 SQuAD、DROP、HellaSwag 等**标准 NLP benchmark 上系统性回退**。作者直接将其命名为 **alignment tax**。

解决办法是在 PPO 的奖励里混入预训练的对数似然梯度，即 `PPO-ptx`；这相当于**在 RL 阶段同时并行一段“恢复性预训练”**。

> 结构同构点：你案例里的“9B token long-context recovery”就是一次针对特定能力的、定向的 `ptx`。

## 2. Code Llama：软件工程数据导致自然语言能力回退

Rozière et al. (2023) 明确写道：只在代码上训练会导致**自然语言能力的灾难性遗忘**。他们并未简单调小学习率，而是**在训练批次里插入自然语言 batch**（约 8%），让代码和自然语言交替出现以对抗漂移。

> 这是与“软件工程 mid-training”最直接对应的先例。你的案例中“改善了一部分 Agent 任务”正是代码 mid-training 的预期收益，而“RULER-1M 掉到 33.5”是同一枚硬币的另一面。

## 3. 长上下文扩展后，短上下文 SFT 破坏长上下文能力（双向干扰）

这是**方向相反**的同构案例，也是目前文献里记录最多的：

- **ProLong (Gao et al., 2024)**《How to Train Long-Context Language Models (Effectively)》：核心发现就是长上下文能力**高度依赖 SFT 阶段的数据混合方式**，纯短数据 SFT 会显著破坏已获得的长度能力；且长/短数据配比是决定性的。
- **Llama 3.1 (Meta, 2024)**：分阶段从 8K 扩展到 128K，报告里明确提到必须混入大量短序列，否则通用 benchmark 会退化。
- **Qwen2.5-1M 技术报告**：报告了同类现象，SFT 阶段若只用短数据，1M 检索能力会明显回落，因此在 SFT 中显式保留长数据。
- **Fu et al. (2024)**《Data Engineering for Scaling LMs to 128K Context》：纯长数据训练会让短任务性能显著下降，需要 data mixture 工程。

> **注意方向**：这些案例是“长扩展 → 短能力受损”，而你的案例是“短（256K）mid-training → 超长（1M）能力受损”。两者合起来说明这是**双向**的，本质是训练长度分布的支撑集（support）一旦收窄，分布外的那一端就会被覆写。

---

# 三、按领域归类的其余案例

<details>
<summary><b>3.1 领域 mid-training / 继续预训练 → 通用能力回退</b>（展开）</summary>

| 案例 | 干扰方向 | 缓解方式 |
|---|---|---|
| **BloombergGPT** (2023) | 金融语料继续预训练 → 通用 benchmark 回退 | 金融:通用 = 1:1 混合；作者承认即便如此仍有 trade-off |
| **Meditron / PMC-LLaMA / BioMistral** | 医疗继续训练 → 通用推理、数学明显下降 | 数据混合 + 通用数据回放 |
| **DeepSeek-Math / 早期数学专精模型** | 纯数学语料 → 指令跟随灵活性、安全性下降 | 混通用 SFT |
| **StarCoder2** | 代码语料 → 自然语言生成流畅度 | 多语言 + NL 混合，沿袭 Code Llama |
| **OLMo 3 (AI2, 2025)** | 显式引入“mid-training”阶段（长上下文 + 思考能力）；报告里记录了阶段性能力起伏 | 阶段化配方 + 回归测试 |

共同规律：**回流比例（replay ratio）是唯一能同时控制两个方向的参数**，也是所有这类报告的结论落点。

</details>

<details>
<summary><b>3.2 Agentic / 工具调用 / RL mid-training 的双刃剑</b>（展开）</summary>

你案例的另一半是“改善了 Agent 任务”，而 Agentic mid-training 本身有若干已知副作用：

- **过度工具调用（over-tooling）**：大量 agent trajectory（JSON / XML 工具 schema）SFT 后，模型对不需要工具的普通问答也发起调用，甚至凭空生成工具名（hallucinated tool call）。
- **格式漂移与啰嗦化**：轨迹数据格式单调，会使自由文本生成出现“schema 化”倾向，回复长度上升、散文质量下降。
- **Search-R1 / ToolRL 类工作**：报告 RL 训练后通用 QA benchmark 回落，需要在 reward 里加通用能力正则。
- **推理 RL 导致的 overthinking**：`Do NOT Think That Much` 等工作展示了 reasoning RL 让模型在非推理任务上过度思考、成本上升且答案变差。这正是一种“一个能力上升、另一个能力下降”的同类现象。

这类副作用通常不会像 RULER 那样出现 79.9 → 33.5 的断崖，而是**缓慢漂移**，因此更常被漏检。

</details>

<details>
<summary><b>3.3 多模态指令微调 → 纯文本能力受损</b>（展开）</summary>

- **LLaVA 系列 / 早期 VLM**：视觉指令微调后，纯文本 benchmark 下降，文献里称为 *language forgetting*。
- **MM1 / Qwen-VL / Idefics**：明确报告需要混入纯文本数据来维持语言能力，否则文本推理明显回退。

机制上与你的案例一致：**新模态/新格式的指令数据重新塑造了输出分布**，旧分布的尾部被削平。

</details>

<details>
<summary><b>3.4 安全对齐的“表面性”：最极端的覆盖案例</b>（展开）</summary>

- **Qi et al. (2023)**《Fine-tuning Aligned LMs Compromises Safety》：仅 **10 个恶意样本**、成本不到 $0.2，就能打掉 100+ 个已对齐模型的安全行为。
- **Shadow Alignment**、**LoRA 微调破坏安全对齐**：同类结论。
- **Anthropic "Sleeper Agents"**：安全训练只覆盖了行为表面，后训练可以被“重新唤醒”。

这些案例证明：**后训练阶段的覆盖往往是“策略层”的，而非“表示层”的**，因此能被很小规模的定向训练逆转。这既解释了 33.5 的断崖（策略被覆写），也解释了 9B token 就能回到 81.0（策略被重新装回）。

</details>

---

# 四、一个值得单独说的观察：为什么是“断崖”而不是“斜坡”

你的数字很有信息量：$79.9 \to 33.5 \to 81.0$。

三点推断（属于我的解释，非原文结论）：

1. **跌幅量级排除容量因素**。若只是容量被占用，应是渐进下降。掉到 33.5 说明 1M 位置上的**检索策略（attention 分配模式）被整体替换**，而不是被"挤占"。
2. **$81.0 > 79.9$ 说明 recovery 是“重学”而非“复原”**。恢复后的能力**超过了被破坏前的水平**，这在 continual learning 里很常见：recovery 数据顺带覆盖了原训练没覆盖的位置外推情形。这也意味着**“回退”未必是纯损失**，有时是一次被迫的、更彻底的重训。
3. **RULER-1M 是合成 benchmark**，对位置编码外推和 attention 锐度极度敏感，因此它对这类干扰的分辨率远高于真实长文档任务。**33.5 未必等于“1M 上下文真的废了”**，更可能是“在这类合成检索模式上失效”。这一点在解读时应保留。

---

# 五、工程上如何系统性地防这类问题

<details>
<summary><b>配方层面 / 架构层面 / 评估层面（展开）</b></summary>

**配方层面**
- **Replay / Rehearsal**：每个 mid-training 阶段强制保留上一阶段能力的“能力保持集”（capability-preserving set），比例是一阶超参。
- **课程顺序**：长短交错优于“先短后长”或“先长后短”；Llama 3.1 与 ProLong 都指向这一点。
- **能力定向 recovery 段**：把“恢复段”当作常规组件而非补救措施（你的 9B token 就是这个）。

**架构层面**
- **Hybrid Mamba-Transformer**（Nemotron 3 本身就是）：状态空间分支对超长依赖的保持性优于纯 attention，架构选择本身就是一种遗忘缓解。
- **MoE / 专家路由**：理论上可做能力隔离，实践中路由共享仍会互相干扰。
- **参数隔离与合并**：LoRA 分支、task arithmetic、model merging——先分阶段训再合并，避免串行覆盖。

**评估层面（最容易被忽略）**
- 把长上下文 benchmark 放进**每次 mid-training 后的回归测试套件**，而不是只在最终模型上跑一次。
- 同时监控“被改善的能力”和“被牺牲的能力”两条曲线，避免只报好消息。
- 用**多条不同性质的长上下文 benchmark**（合成检索、真实长文档 QA、多跳聚合）交叉验证，区分“benchmark 分布回归”与“真实能力恢复”。

</details>

---

# 六、想自己继续找同类案例，检索建议

关键词组合（英文命中率远高于中文）：

```
"catastrophic forgetting" + "mid-training"
"long-context degradation" + "SFT"
"context length regression" + "post-training"
"capability regression" + "training stage"
"replay ratio" + "continual pretraining"
"alignment tax" + "PPO-ptx"
```

**最高产的位置不是论文正文，而是技术报告的 "Ablations" / "Findings" / "Lessons" 小节**——厂商通常把这类负面结果放在那里，摘要里看不到。Llama 3.1、Qwen2.5-1M、OLMo 3、ProLong 的关键结论都在这些角落。

---

# 七、需要保留的谨慎

- 我给的三段式同构案例（InstructGPT / Code Llama / ProLong-Llama3.1-Qwen）在**机制**上高度可比，但**幅度**不能直接类比：RULER 的 $79.9 \to 33.5$ 是合成 benchmark 上的极端敏感反应，通用能力回退通常只有几个点。
- “9B token 恢复”这个数字，其有效性高度依赖**这 9B token 的构成**（多少条 1M 序列 vs 256K 序列）。如果主要是 256K 序列，那 81.0 更可能是“回到了 benchmark 的评测路径”而非“1M 能力真的重建”。
- 我的信息截止较早，2026 年的 mid-training 文献我没系统覆盖；如果你要写综述，建议直接以 Nemotron 3 Super 报告里的相关工作章节为起点做**前向引用/后向引用**追踪，这比关键词检索更完整。

需要我把某一类（比如“长上下文 ↔ 短上下文双向干扰”）展开成带具体数字和方法对比的详表吗？
