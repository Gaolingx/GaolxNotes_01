# 补全CoT用Prefix重推替代Backfill，做RL冷启动更高效

## 🧑‍💻 User

我们最近在做agenticrl，关于“缺reasoning trace数据”的问题，需要设计或者扩展claudecode/codex等harness
可不可以给harness写一个数据蒸馏+导出trajectory plugin
让一个agent生成gt和solution，然后harness记录tool和reasoningcontent（from session file），再开个agent审核是否可以作为可训练的trajectory
还能开一堆sub并行roll做rft

---
## 😀 默认助手 (7)

## 结论先行

这个设计可行，但有三处需要替换，否则产出的数据基本不可用：

| 你的原方案 | 问题 | 建议替换为 |
|---|---|---|
| Agent A 生成 GT | 自蒸馏的“GT”不可信，审核 agent 又会和生成方犯同样的错，形成循环论证 | **任务合成与 solution 蒸馏解耦**；GT 来自可执行 verifier（隐藏测试或状态 diff），而不是来自模型 |
| Agent B 审核“能否训练” | LLM judge 单独使用时，接受的率和偏见都不稳定；它决定的是**过滤**，而不是**奖励** | verifier 给 reward，judge 只做**质量门槛**，并单独对 judge 做校准集（注入错误看能否抓出） |
| 开一堆 sub 并行 roll | 同一任务上 p̂=0 或 p̂=1 的 rollout 对 RFT **零信息**，纯烧钱 | **难度自适应**的 rejection sampling：先小样本估计 p̂，只在大约 [0.1, 0.7] 区间批量 roll |

另外关键一点：**如果要拿到 reasoning trace 做 on-policy RFT，必须有办法让自己的模型运行在 harness 里**。Claude Code 支持 `ANTHROPIC_BASE_URL` 指向自建 endpoint，因此 recorder 这一层对“蒸馏”和“RFT”来说通用，但 rollout 循环是否能接上自己的 policy，决定了你是在做 distillation 还是 RFT。如果这两件事混在一起做，最后会得到一批 teacher 的 off-policy 数据，却拿它当自己的 RFT 数据使用。

---

## 架构：分层 sidecar，不要 fork harness

fork Claude Code 或 Codex 意味着需要永远维护 rebase。正确的做法是**独立进程的 recorder**，通过 hook 和结构化 stream 接入。

```
┌─ rollout layer ──────────────────────────────────┐
│ task synthesizer → sandbox(container/worktree)   │
│   → claude -p / codex exec  (N 并行)             │
└──────────────┬───────────────────────────────────┘
               │ hooks(stdin JSON) + stream-json
┌──────────────▼───────────┐
│ L0 recorder  (per-harness adapter)               │
│ L1 canonical IR  ← 唯一真源                       │
└──────────────┬───────────┘
               │
┌──────────────▼───────────┐
│ L2 verifier  (executable) + auditor (gate)       │
└──────────────┬───────────┘
               │
┌──────────────▼───────────┐
│ L3 exporter: SFT / RFT / DPO-pairs / stats       │
└──────────────────────────┘
```

### L0：接入方式（比解析 session 文件干净得多）

**Claude Code** — 两条路，建议主用第二条：

1. Hooks（`settings.json`）：`PreToolUse`/`PostToolUse`/`Stop`/`SubagentStop`/`SessionEnd`，stdin 拿到 `session_id`、`transcript_path`、`cwd`、`hook_event_name`、`tool_name`、`tool_input`、`tool_response`。`Stop` hook 返回 `{"decision":"block","reason":...}` 能让 agent 继续，这是接 verifier 反馈回 agent 的钩子。
2. **`claude -p --output-format stream-json --verbose --include-partial-messages`** —— 直接吐 `thinking_delta`、`tool_use` 增量。做 CoT 数据这条路径保真度最高，不用事后重建。

session 文件在 `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`，每行带 `parentUuid`/`uuid`/`type`/`message`/`isSidechain`/`requestId`。**回溯顺序要用 uuid 链而不是 `timestamp`**（并发时时间戳会撞）。`isSidechain: true` 是 Task 子 agent 的独立轨迹，必须单独处理，不要混进主轨迹。

**Codex** — `codex exec --json` 流事件比 `~/.codex/sessions/rollout-*.jsonl` 稳。⚠️ 一个必须提前知道的坑：Responses API 的 reasoning item 默认返回 `encrypted_content`（不透明），只有摘要时才给 plaintext `summary`。**意味着从 Codex 拿不到原始 CoT，只有 reasoning summary**。要真 CoT 就得上 Claude Extended Thinking，或开源自建（vLLM/SGLang + `--reasoning-parser` 出裸 reasoning text）。

> ⚠️ 这类格式每周都在变。字段名落地前务必用当前版本 dump 一份样本对一遍，不要照抄博客。

### L1：canonical IR

```python
Event = {
  "kind": "system|user|assistant|reasoning|tool_call|tool_result|compact_boundary|interrupt",
  "text": str | None,
  "tool": {"name": str, "args": dict, "id": str, "exit_code": int|None, "duration_ms": int} | None,
  "tokens": [int] | None,          # 关键：存 token_ids，别只存 text
  "loss_mask": 0|1,                # 是否对这段算 loss
  "meta": {"request_id":..., "model":..., "ts":...}
}
Trajectory = {"task": {...}, "events": [Event...], "label": {...}, "provenance": {...}}
```

`loss_mask` 在 recoder 阶段就写好，别留到 exporter——**tool_result 和 user turn 永远 mask 掉**；reasoning 段是否参与 loss 要作为显式配置项（训练 CoT 就开，只训“行为克隆”就关）。

### L2：verifier 优于 auditor（排序不要搞反）

可信度排序，越靠前越硬：

1. **隐藏单测 / 差分执行 / 状态断言**（唯一可作为 reward 的）
2. 环境最终状态 diff（文件树、DB、进程）
3. rubric LLM judge + logprobs
4. 纯 LLM judge（只能当 gate）

Auditor 的正确职责是**拒掉“作弊型成功”**，不是评好坏。必须显式检测：

- 改了测试文件 / `git checkout` 测试 fixture → 跑测前 **reset 测试文件到 pristine 副本**
- 硬编码输出、`print` 期望值、跳过/`xfail`、monkey-patch assert
- 网络访问、`git log`/`git show`/changelog/PR 里翻答案 → 沙箱断网 + 剥 `.git`（或 shallow 化）
- 同名文件重复写、长时间空转后突然成功

并且**给 auditor 建校准集**：人工标 200 条 + 故意注入 20 条作弊轨迹，测它的 precision/recall，别直接信它 accept rate。

### L3：导出与“RFT 经济学”

RFT ≈ expert iteration / STaR，所以真正决定效率的是**每美元采到的有效样本数**：

- 难度带调度：round 0 每任务 k=8 采样估 $\hat p$，只对 $0.1 \le \hat p \le 0.7$ 的任务加批量；$\hat p=0$ 存档等模型变强再回访，$\hat p=1$ 直接丢
- **缓存 + 快照**：setup 通常占大部分成本，用镜像 digest 固定的容器快照或 `git worktree`（不能只用同一个 cwd），并对确定性工具结果做缓存
- 每个任务最多留 1–3 条 accepted，按 `(task_id, normalized_patch_hash)` 去重；reasoning 段做 n-gram MinHash 去重
- 多样性 > 体量：温度别降，长度做分层（防 length hacking——模型会学会写长 CoT 讨 judge 欢心）
- DPO 数据是免费副产品：同一任务上 accepted vs rejected 直接成对

---

<details>
<summary><b>数据质量陷阱清单（这些不处理，训出来的模型会带上难以根除的问题）</b></summary>

- **Compaction**：Claude Code 会压缩上下文，压缩后的 session 已经不是真实 token 序列。检测到 `compact_boundary` 就整条弃用，或只训最后一段
- **重试重复**：API 报错会产生完全相同的 assistant message，按 `request_id` 去重
- **训练/推理模板漂移**：记录 `template_hash` + tokenizer 版本，否则总有一天发现线上模板和训练时不一样
- **中断轨迹**：用户打断、工具超时、`aborted` 必须显式过滤
- **token 级对齐**：有 `token_ids` 才能免重 tokenize；没有就得保证 chat template 完全一致
- **长度离群**：> p99 轨迹单独看，通常是 agent 卡死循环
- **可复现性**：存 container image digest + git sha + seed + sampling 参数，否则事后无法 replay 定位

</details>

<details>
<summary><b>合规红线（建议先确认再投入）</b></summary>

用闭源模型的 CoT 蒸馏成可训练数据集，涉及 provider 的 ToS：Anthropic / OpenAI 对“用输出训练竞品模型”都有明确限制。自己模型 + 自己的 harness + 自己的 API key 在企业内通常是可做的，但**把 Claude 的 thinking 文本当作训练语料**这一步值得让法务先看一眼。另外 extended thinking 的 `signature` 是校验用的，离线重放会失败——也意味着那段 thinking 只能当 text 用，不能当可回灌的对话历史。

</details>

---

## 落地顺序（建议 3 步，别一次做完）

1. **Week 1**：recorder + IR + `claude -p --output-format stream-json` 适配器。目标是能稳定 dump 一条带 `loss_mask` 的轨迹。同时用 `ANTHROPIC_BASE_URL` 指向自建模型，验证“换 policy 不改代码”。
2. **Week 2**：verifier（隐藏测试 + 测试文件 reset + 断网沙箱）+ 难度带调度器。此时指标是 **accepted / \$**，不是 accepted 总量。
3. **Week 3**：auditor + 校准集 + DPO 配对导出。跑一个小 held-out eval，确认蒸馏后的 checkpoint 真的涨了——不然你只是在给 auditor 的偏见做 SFT。

需要我把 L1 的 IR schema 和 exporter 的 `loss_mask` 构造写成可跑代码，或者先给 `PostToolUse` 那个 hook 的 recorder 骨架？

---
## 🧑‍💻 User

有没有成熟的recipe可以参考，设计思路是怎么样的

---
## 😀 默认助手 (7)

没有“一个”成熟 recipe——但 **4 条独立演化的路线已经收敛到同一个结构**，把它们的公共部分抽出来，就是你需要的 recipe。下面按“哪些能直接照搬”来组织。

> ⚠️ 下文的论文/仓库名和数字（尤其规模）请落地前自行核对一遍，这个领域 2025 下半年变动极快，我给的是结构与结论，不是可引用的精确值。

## 一、四条正在收敛的路线

| 路线 | 代表工作 | 解决的是 |
|---|---|---|
| **A. 环境/任务合成** | `SWE-smith`(Nebius)、`R2E-Gym`、`SWE-Gym`、`Kimi-Dev`(72B)、`neom-gym`/`NeMo-Skills` | 怎么造出**自带可执行 GT** 的任务 |
| **B. RL 训练系统** | `verl-agent`(+`GiGPO`)、`SkyRL-agent`、`Agent Lightning`(MSR)、`AWorld-RL`、`prime-rl` + `verifiers` | 怎么把异构 harness 的 rollout 喂进 trainer |
| **C. 轨迹蒸馏** | `FireAct`、`AgentTuning`、`Agent-FLAN`、`AgentOhana`、`AgentGym`/`AgentEvol`、`LIMO`/`s1` | 轨迹 schema、loss mask、数量 vs 质量 |
| **D. 轨迹审计/评测** | `AgentRewardBench`、`Agent-as-a-Judge`、`τ-bench`(pass^k)、`SWE-bench Verified` | judge 该信到什么程度 |

四条线的交集，正好就是你要的那个 plugin。

## 二、收敛出的 5 阶段 recipe

```
① 环境可执行化 → ② 反向任务合成 → ③ 并行 rollout + 可执行打分
   → ④ 难度课程/批量筛选 → ⑤ 去污 + 去重 + 导出
```

<details>
<summary><b>① 环境可执行化：任务 = 一个 package，不是一段 prompt</b></summary>

这是 `verifiers`(Prime Intellect)、`neom-gym`、`AgentGym`、`AWorld` 都独立收敛到的同一种形状：

```
task_0042/
  ├─ task.md            # 给 agent 的 prompt
  ├─ Dockerfile         # 或 image_digest: sha256:...
  ├─ setup.sh           # 起环境、剥 .git、断网
  ├─ tests/             # 隐藏测试（agent 不可见）
  ├─ rubric.py          # 多分量 reward：tests 0.7 + lint 0.1 + 无作弊 0.2
  └─ meta.json          # difficulty 先验、来源 repo、license、decontam key
```

**设计思路**：把“任务”变成**数据而非代码**。这样并行 rollout 只是 `for task in tasks: spawn(container, task)`，你的 recorder 完全不需要知道任务内容。你原来的“agent A 生成 GT”在这个形状下会自动被拆成两个可独立验证的产物——`task.md` 和 `tests/`。

</details>

<details>
<summary><b>② 反向任务合成：把“判断题”降级成“选择题”</b></summary>

这是所有产线里**最重要的一条设计思路**，也是对你原方案最直接的修正。

- **`SWE-smith` 的做法**：拿 128 个真实 repo，用一套 bug-injection 程序把**已知能跑的代码改坏**。于是“修复 patch”就是原始代码，**GT 由构造保证**。生成 5 万级实例，成本接近零。
- **`R2E-Gym` 的做法**：从 repo 挖 commit 造环境，用 **“gold patch 通过 + base commit 失败”** 双向筛选，不满足就丢。
- **`Kimi-Dev` / `DeepSWE`**：同样是 bug-injection + 可执行测试，规模的量级靠合成而非人工。

**关键洞察**：forward 方向（agent 先解、再判定对不对）你需要判定“答案是否正确”，这是个**开放判断题**，只能靠 judge。backward 方向（先有正确答案、再构造任务）你只需要判定“任务是否 pass 了真实 verifier”，变成了**封闭选择题**。

> 一句话：**不要问“这个 GT 对吗”，要构造出“错不了”的 GT。**

真实任务（你的用户真实 bug）不是不能用，而是只能当**评估集**（像 `SWE-bench Verified` 那样人工过一遍 500 条），不能当训练集的主力。

</details>

<details>
<summary><b>③ rollout + 打分：解耦生产与训练</b></summary>

`Agent Lightning` 的核心论点值得直接抄：**agent 执行层与 trainer 之间只允许有一个 IR 边界**，双方都不知道对方存在。它把这个边界叫 “LLM traced” 接口；`SkyRL-agent` 叫 trajectory buffer；`verl-agent` 叫 async rollout worker。名字不同，形状一样：

```
harness 进程池 ──IR事件流──> 轨迹缓冲(队列/对象存储) ──> trainer 拉批
```

对你直接的好处：Claude Code 和 Codex 只是两个 adapter，多接一个 harness 不碰训练代码。而且**失败的 rollout 也是资产**——`AgentRewardBench` 明确指出只用 outcome 判分漏掉大量失败模式，中间证据（工具调用、报错堆栈）对 judge 质量影响很大。

</details>

<details>
<summary><b>④ 难度课程：筛选是“按 p̂ 分层”，不是“按通过与否”</b></summary>

- `WebRL`：用成功率做动态课程。
- `OpenR1` / `Open-Reasoner-Zero`：只保留“模型有时能做对”的 prompt。
- `Tulu 3`：完整公开的后训练 recipe，含 RLVR 数据筛选与去污流程，是少数把“脏活”写清楚的一手材料。

产业界共识的可用性边界：$0.1 \lesssim \hat p \lesssim 0.7$。两端都是纯烧钱。区别只在于 $\hat p = 0$ 的那批**别删**——标 `too_hard` 存档，模型换代后回访，这是免费课程。

</details>

<details>
<summary><b>⑤ 去污 + 去重 + 导出</b></summary>

- **去污**：`meta.json` 里存 decontam key（repo+commit+函数签名），和你的 eval set 做交集检查；再扫工具调用里有没有 `git log`/`git show`/网络访问。
- **去重**：`LIMO` 和 `s1` 的共同结论——**千条精选 ≫ 十万条垃圾**。轨迹按 `(task_id, patch_hash)` 去重 + reasoning 段 MinHash。
- **格式**：`AgentOhana` 专门做多源轨迹 schema 统一，`Agent-FLAN` 发现**混格式训练必需**、且“工具调用”和“推理”应拆成不同 objective——这直接对应你 IR 里的 `loss_mask` 设计。

</details>

## 三、三条反直觉但被反复验证的设计原则

### 原则 1：teacher 给分布，verifier 给标签

蒸馏时 teacher 只负责提供 **reasoning 的风格/长度/工具使用习惯**的分布；**label 一律来自可执行 verifier**。一旦让 teacher 或 LLM judge 出 label，你训练的就是那个 judge 的偏好。

`AgentRewardBench` 的实测结论支持这个分层：judge 在有完整中间证据时明显更好，但**仍不足以当 reward 用**——当 gate 可以，当 reward 会 reward-hack。

### 原则 2：advantage 的粒度是必调超参，不是副产品

- `RAGEN`(StarPO) 记录了一个具体失败模式 **“Echo Trap”**：纯 trajectory-level RL 在稀疏奖励下熵塌缩，模型开始输出与训练样本无关的冗长 reasoning——看着像在思考，其实是在模仿格式。
- `GiGPO` 的解法是 **两级 advantage**：episode-level + 用 anchor state 分组算 step-level。

你的场景（Claude Code，动辄几十轮）几乎必然是极端长 horizon，**必须**预留 sub-trajectory 拆分能力。而且注意跟 `τ-bench` 的 `pass^k` 对齐——agent 的可靠性衰减主要发生在 k=1→k=8 这一段的塌陷上，单次成功率高估了数据价值。

### 原则 3：session 文件是日志，不是数据源

`OpenHands`、`SWE-agent`、`mini-swe-agent` 的实践都表明：**结构化 stream > 事后解析 session**。`mini-swe-agent` 故意做成 100 行，就是为了让轨迹干净可读。

对应到你的场景：优先 `claude -p --output-format stream-json`，**jsonl 只当 cross-check 或 fallback**。因为 compaction、重试、sidechain 这些在 jsonl 里是“事后猜测”，在 stream 里是“当场知道”。

## 四、如果只抄一个最小组合

| 你的模块 | 直接抄 |
|---|---|
| 任务/环境容器 | `verifiers` 或 `neom-gym` 的 package 规范 |
| 任务合成 | `SWE-smith` 的 bug-injection（有现成实现） |
| 环境筛选 | `R2E-Gym` 的 “gold pass + base fail” 双向门 |
| rollout 系统 | `SkyRL-agent` 或 `Agent Lightning` 的解耦架构 |
| advantage | `verl-agent` + `GiGPO` 的两级设计 |
| judge 校准 | `AgentRewardBench` 的评测协议 |
| 数据质量 | `Tulu 3` 的去污 + `LIMO` 的“少量精选” |
| IR schema | `AgentOhana` + `Agent-FLAN` 的格式/objective 拆分 |

---

**最后一个判断**：你原方案里最有价值、也最少人做对的是那一步——**“从 session file 抽 reasoning content 并保留 loss mask 语义”**。因为所有开源 recipe 用的都是 ReAct 风格纯文本轨迹，没有 Claude Code 这种带 thinking block + compaction + sidechain 的真实工业 harness 数据。这块没有 recipe，得自己趟，也是你相对公开工作的真实优势位。

要我把“environment package 规范 + recorder IR + exporter loss_mask”写成一份可直接 diff 的目录骨架吗？

---
## 🧑‍💻 User

还有一种成本更低的方案
拿着 ground truth 补全 cot 比拿 ground truth cot 练效果更好
对于闭源模型拿不到raw cot
可以根据现有的gt让teacher（一般是开源sota）补全每轮的cot
清洗后再作为训练用的trajectory
是否可行？

---
## 😀 默认助手 (7)

## 直接回答

**可行，而且方向是对的——但你说的“补全 CoT”里藏着两种完全不同的做法，一种是成熟 recipe，另一种有系统性陷阱。** 你的直觉里那个“更好的结论”是对的，但归因需要换一下。

## 一、真结论 vs 危险混淆

<details>
<summary><b>先验证你的直觉：它是对的，但原因不是“因为条件在 GT 上”</b></summary>

「补全的 CoT 比 GT 自带的 CoT 更好练」——这句话在文献里基本被支持，但成立的原因有三个，都不是“conditioning on answer”本身：

1. **闭源 GT 的 CoT 你根本拿不到。** Claude 的 thinking 有 `signature` 校验、Codex 的 reasoning 是 `encrypted_content`。所谓“GT CoT”在多数情况下是个不存在的对象，你比较的其实是“摘要版/伪造版”。
2. **GT 自带的 CoT 里全是不可回收的死路。** 真实解题轨迹包含大量试错、自我否定、走错再回头。这些 token 教给学生的不是能力，是“可以乱试”——这正是 RAGEN 那个 Echo Trap 的温床。
3. **清洗/压缩 CoT 本身就是涨点。** `LIMO`、`s1`、`Bespoke-Stratos` 的共同结论：**千条干净短 CoT 打十万条原始蒸馏**。

所以准确的表述是：**「事后重建的、干净的、能力匹配的 CoT」 > 「teacher 原始解题痕迹」**。这跟“要不要看答案”是两件事，别混。

</details>

### 关键的混淆：Variant A 和 Variant B

```
Variant A（成熟 recipe，已被反复验证）
  teacher 只看 prompt → 自己解 → 用 GT 过滤答案
  代表：Bespoke-Stratos-17k、LIMO、s1、OpenThoughts、R1 的 cold-start 语料

Variant B（你的提案，风险高）
  teacher 看 prompt + GT 答案 → 事后写 rationale（rationalization）
  代表：没有成熟产线；文献上有明确的 unfaithfulness 证据
```

两者成本几乎一样，**但 Variant A 的 teacher 看不见答案，所以它写出的 CoT 天然是“因果”的；Variant B 的 teacher 已经知道终点，写出的 CoT 天然是“叙事”的。**

单轮 QA 上这个差异还能靠过滤掉一半，因为答案只有对错。**在你这个场景（agentic，多轮工具）上，Variant B 会精确地破坏你最想训的那项能力。**

## 二、为什么 agentic 场景下 Variant B 的伤害被放大

单轮 QA 里，CoT 的作用是“把 prompt 推演到答案”。Agent 里，CoT 的作用是**决定下一步去获取什么信息**。这是本质区别。

具体失效模式：

| GT 轨迹里的动作 | 仅看 prefix 是否可推 | backfill 的后果 |
|---|---|---|
| `pytest tests/test_auth.py`，而上一轮 traceback 里恰好有 `auth.py:42` | 可以推 | ✅ 高保真，学到的就是“读报错定位” |
| `vim src/core/handler.py:88`，prefix 里没有任何线索指向它 | 不可推 | ❌ teacher 会**编造**一个理由（“我怀疑问题在 handler”），学生学到“凭直觉跳文件” |
| 一次性写出正确 patch | 不可推 | ❌ teacher 编造“分析过程”，学到的推理**与动作无因果关联** |

**信息获取（localization / 探索）本身就是 agentic RL 的核心能力**，而 Variant B 恰恰在这一环上给了你**基于全知视角的伪推理**。你用 GT 省下的探索成本，等于把“怎么找到答案”这段最重要的轨迹替换成了“因为我已知答案，所以我直接去了那里”。

这就是文献里 **unfaithful CoT** 问题的 agentic 版本，而且更严重：单轮的不忠实只影响可解释性，agentic 的不忠实**直接污染动作选择策略**。

> ⚠️ 顺带一个真实成本项：GT 轨迹未必是你这套 harness/tool 产生的。prefix 里可能根本没有支持该动作的 observation，teacher 被迫 confabulate 的比例会比你预想的高。

## 三、推荐 formulation：要 prefix，不要 GT

把「拿 GT 补全 CoT」改成 **「只给 prefix 的逐步重推」**，成本几乎不变，faithfulness 变成可验证的。

对第 $t$ 轮：

$$\text{CoT}_t \sim \pi_{\text{teacher}}(\cdot \mid \underbrace{x_0, a_1, o_1, \dots, a_{t-1}, o_{t-1}}_{\text{student 当时看到的全部}},\ a_t)$$

即在 **t 时刻 prefix + 该轮实际动作** 的条件下，让 teacher 补这一段思考。**故意不给 $a_{t+1}, o_t^{\text{future}}$，也不给最终答案。**

这同时解决三件事：

1. **faithfulness 可检验**了——见下面第 5 节的“实体前瞻泄漏”检查，能自动化。
2. **prefix 是 student 分布**（因为 GT 轨迹的 observation 就是真实环境返回的），所以 CoT 也是 student 能力档位的。这是你“比 GT CoT 好练”直觉的真正来源。
3. **天然是 turn-level 密度信号**。上面聊到的 GiGPO 两级 advantage 里那个 step-level 那一级，这里的 per-turn CoT 正好提供 dense 监督——episode-level 稀疏奖励的问题被绕开了。

实际上这个形状对应的是 **DAgger / expert-takeover 的蒸馏版**：不让学生自己走，用 GT 的 observation 序列当 prefix，只蒸每一步的“决策理由”。这条路比 STaR 稳，因为你不需要 rejected samples。

<details>
<summary><b>成本账：为什么它确实便宜，便宜在哪</b></summary>

设 rejection sampling 单任务通过率 $p$，想收 $N$ 条正样本需 rollout 约 $N/p$ 条完整轨迹（agentic 场景 $p$ 常在 $0.1\sim0.3$），每条 $T$ 轮、context 线性增长，成本 $\Theta(T^2 \cdot \text{tok})$。

Prefix 重推的成本：
- **收率 = 1**，不需要 $1/p$ 倍浪费；
- 只需 teacher 的**共享 prefix prefill**，多个 turn 的 prefix 可以按 batch 组织复用 KV（同任务同 prefix 只算一次）；
- **不需要**执行工具（工具结果直接复用 GT 的）；
- 但你需要 $T$ 次 per-turn 生成，且每次 prefix 都长——所以成本是 $\Theta(T^2)$ 的 prefill 但**无 1/p 放大**。

量级上通常是 rejection sampling 的 $1/3 \sim 1/10$。**便宜的来源是“零浪费”+“零环境执行”，不是“省了思考”。** 这条账成立的前提是：GT 轨迹的 observation 你必须能拿到（这也意味着你必须先有一条 GT 轨迹，那就回到上面的“反向任务合成”了）。

</details>

## 四、清洗环节该做什么（这是成败所在）

按“能自动化”排序：

1. **实体前瞻泄漏检查（最硬的过滤，务必做）**
   抽 turn $t$ 的 reasoning 里的标识符（文件路径、函数名、报错字符串、行号），检查它首次出现在 observation 序列的位置 $j$。若 $j > t$ → **该轮 CoT 是从答案视角写的，丢弃**。这条能把 Variant B 的污染自动筛掉 60%+。

2. **证据消融测试（faithfulness 探针）**
   把 turn $t$ 的 CoT 声称用到的那个 observation **删掉或扰动**，重新让 teacher 生成。如果 CoT 几乎不变 → 它是从答案写的，不是在读证据。抽样 100 条估个比例，这是你唯一能真正量化 faithfulness 的手段。

3. **前缀可推性门（prefix-only re-derivation）**
   另起一个看不见 $a_t$ 的 model，只给 prefix，看它能否推出同一个动作。能 → 这轮的 CoT 是合法的推导；不能 → 这一轮的动作本来就是“灵感跳跃”，backfill 只会产生 confabulation，**丢或降权**。

4. **常规清洗**：去首段复述题目、去 “the user wants me to…”、去 meta 腔（“as an expert I will…”）、强制 Markdown/工具名合法、单语言、长度离群截断。

> 建议直接**丢掉 GT 轨迹里所有“不可推”轮次的 CoT**，只保留高可推性轮次。数据变少，但这批数据是干净的——参考 LIMO：少而精选是正收益。

## 五、一天内验证可行性

别先搭 pipeline，先做这个 A/B（同一批 GT 轨迹、同一个 student base、相同 token 预算）：

| 组 | 数据 |
|---|---|
| A | teacher 只看 prompt 自己解，过滤答案 = **Variant A** |
| B | 你的方案 = **Variant B**（看答案写 rationale） |
| C | **prefix-only 重推 + 三道过滤**（我的建议） |
| D | 闭源模型原始轨迹（能拿到多少算多少，作上界参考） |

评测必须包含两项 agentic 专属的：**held-out 任务的 localization 子任务成功率**（专测“找到该改哪”），和 **`pass^k`**（k=1→8 的衰减曲线，`τ-bench` 那个协议）。只看 pass@1 会把 B 的伤害完全掩盖——B 在“照着已知答案改代码”上很强，在“找到该改哪”上会显著掉。

我预测的结果是：**B 在 pass@1 上和 C 接近甚至略高，但在 localization 与 $pass^{k}$ 上明显塌。** 如果实验反而是 B 全面胜出，那说明你的 GT 轨迹本身 exploration 成分很低（大部分动作由报错直接指向），这种情况下 B 可用——但务必确认这不是评测集泄漏。

## 六、判定的红线

<details>
<summary><b>这些情况不要用这个方案</b></summary>

- **任务的成功依赖“探索成本”**（长 repo 定位、模糊需求澄清）→ Variant B 教的是作弊直觉
- **GT 轨迹由别的 harness/tool 产生**，动作在你的环境里不可执行或不可解释 → confabulation 比例失控
- **你要的是 on-policy RFT**（上一轮聊的 `ANTHROPIC_BASE_URL` 那条路）→ backfill 必然是 off-policy，只能当 cold start / SFT 底料，不能直接替代 rejection sampling
- **教师和学生的工具集不一致** → prefix 里的 action 语义对不上

反过来，**强烈适用**的情况：GT 轨迹的每一步都有明确证据支撑（SWE 里报错→定位、测试失败→改断言），且你只需要 cold-start 数据 + dense turn-level 监督。这恰好是 R1 那条“cold start 语料 + 后续 RL”的标准姿势。

</details>

---

**一句话总结**：你的方案把成本压下来的机制是对的（零浪费 + 复用 observation），但要**把“看 GT 补 CoT”换成“只给 prefix 重推 CoT”**，否则你省下的是探索成本，同时把“探索能力”从数据里删掉了——而这恰好是 agentic RL 最值钱的部分。加上“实体前瞻泄漏”这个自动过滤器，Variant B 也能用，但会砍掉一大半，不如直接按 C 来做。

要我把 **prefix-only 重推的采样脚本 + 实体泄漏过滤器**写成可跑的骨架吗？这两个是这套方案里唯一需要动脑子的部分，其余都是工程。
