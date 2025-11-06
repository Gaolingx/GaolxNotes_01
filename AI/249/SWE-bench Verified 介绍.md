# SWE-bench Verified 介绍

### 🧑‍💻 User

SWE-bench Verified这个benchmark能否介绍下？

---

### 🤖 Assistant

下面是我整理、总结的 **SWE-bench Verified**（有时写作 *SWE-bench Verified*）这个 benchmark 的详细介绍、优点、局限，以及当前研究／对比情况。如果你希望我重点讲某一部分（如指标、对比别的 benchmark、如何用模型跑这个），可以继续问我。

---

## 一、背景：什么是 SWE-bench / SWE-bench Verified

### SWE-bench 的起源与目标

* “SWE” 是 **Software Engineering** 的缩写。SWE-bench 是一个用于评估大型语言模型（LLM）在软件工程 / 代码修复 / issue 解决能力上的 benchmark。([swebench.com][1])
* 基本思路：选取开源项目中真实的 GitHub issue + 对应的 pull request（PR）作为 ground truth。给定 issue 描述 + 出发时的代码状态（issue 被提出的代码状态），让模型（或 agent）给出一个补丁（patch）去解决该 issue。然后将模型生成的补丁应用到代码，并运行相应的单元测试去判断是否“真的解决了问题”而且没有破坏其他功能。([GitHub][2])
* 在这个过程中，有两组测试（unit tests）用来验证：

  1. **FAIL_TO_PASS**：那些在原始状态下会失败、在正确 PR 合并后会通过的测试，用来验证新补丁是否解决了问题。([OpenAI][3])
  2. **PASS_TO_PASS**：那些不受此次变更影响的测试，它们在补丁前后都应该能通过，用来验证代码补丁没有引入回归（破坏已有功能）。([OpenAI][3])
* SWE-bench 数据集总体规模比较大，包括多个代码库、涵盖不同类型的修复 / issue。([GitHub][2])
* 在实际评测中，不仅模型本身重要，其“agent 架构”（即 prompt 构造、工具接口、编辑流程等）也极其关键——换一个 agent 架构，同样的模型可能表现差很多。([Epoch AI][4])

---

### 为什么出 “Verified” 版本

在原始 SWE-bench 被广泛使用之后，人们（特别是 OpenAI 团队）注意到这个 benchmark 存在若干“问题样本”（edge cases / 难以用常规方式可靠解决的问题），这些问题可能导致模型即使“合理”地写出补丁也会被评为失败。OpenAI 团同 SWE-bench 作者一起设计了一个子集——**SWE-bench Verified**，来尽量剔除这些不良 / 不可靠样本，以得到一个更干净、能更可靠衡量模型核心能力的版本。([OpenAI][3])

具体动机包括：

* 有些单元测试写得过于“精细”或“耦合”，可能 enforce 了与 issue 本身无关的细节，使得合理的补丁反而因为测试不匹配被判为错。([OpenAI][3])
* 有些 issue 描述本身过于模糊或不完整，导致任务本身“无法确定正确意图”——模型可能写出一种合理补丁，但与评测标准不一致。([OpenAI][3])
* 环境搭建、依赖、代码状态设置等问题，有时会导致评测环境不可复现、测试失败。([OpenAI][3])

因此，SWE-bench Verified 选出了 **500 个样本**（从原始 SWE-bench 的测试集中筛选）作为“人工验证 / 质量较高”的子集。([OpenAI][3])
这些样本每一个都经过软件工程师审核，以保证 issue 描述明确、测试合理、补丁可行。([OpenAI][3])

OpenAI 在其博客中指出，这个版本旨在“更加可靠地评估 AI 模型解决真实软件问题的能力”——即减少 benchmark 本身对模型性能的低估或偶然误判。([OpenAI][3])

---

## 二、SWE-bench Verified 的设计细节和评估流程

下面是这个子集 / 评测流程的一些关键特性与注意点。

| 项目                  | 说明                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| 样本数量                | 500 个样本（问题-补丁对）([OpenAI][3])                                                                                |
| 问题来源                | 原始 SWE-bench 的问题集合（多个 Python 开源项目）([GitHub][2])                                                             |
| 是否手工验证              | 是，对每个样本进行人工审核 / 过滤，以剔除有问题或不可行样本 ([OpenAI][3])                                                               |
| 评测目标                | 给模型 / agent 一个 issue 描述 + 初始代码状态，让其输出补丁；然后自动执行 FAIL_TO_PASS 和 PASS_TO_PASS tests，若都通过则认为成功 ([OpenAI][3])    |
| 测试公开性 / 污染风险        | 因为这些都是开源项目 + 在公开 benchmark 中，模型可能在训练时“见过”这些代码 / issue（数据泄露或污染风险）([Epoch AI][4])                             |
| Scaffold / agent 依赖 | 实验中通常基于一个 agent 架构（如 SWE-agent）提供对代码的导航 / 编辑 /测试工具接口等。模型能否正确使用工具、prompt 设计、调试、错误恢复策略等对结果影响极大。([vals.ai][5]) |
| 难度划分 / 注释           | OpenAI 还为所有 SWE-bench 样本（包括 Verified 子集）提供了人工注释，允许按难度、题型等切片分析。([OpenAI][3])                                 |

特别要强调的是：**在 SWE-bench Verified 中，agent 的设计 / scaffold（如何调用工具、分步操作、prompt 结构、重试机制等）对最终表现的影响往往与基础模型本身同样重要**。有些模型如果在某个 agent 架构下表现不好，换一个更合适的脚手架 (scaffold) 就可能大幅提升。([Epoch AI][4])

还有一点：因为样本数量缩小（500 个），在做统计或细分对比时要注意样本偏差和随机性。

---

## 三、优点 / 为什么人们愿意用 Verified 版本

使用 SWE-bench Verified 有若干优点：

1. **降低噪声 / 错误样本影响**
   通过人工审核过滤掉不合理或偏难的样本，可以减少 benchmark 本身对模型的误判（例如合理补丁被误标为错误）。([OpenAI][3])

2. **更公平 / 更聚焦于模型能力**
   由于剔除了部分“难以评估”的极端样本，Verified 子集更能集中考察模型确实能处理的那部分“真实、可解的问题”，避免把偏差过大或异常样本拖低总体成绩。([OpenAI][3])

3. **便于对比与进步追踪**
   使用固定、质量更高的样本作为竞赛 / leader board 的评测基准，有助于不同模型之间的可靠对比。([hal.cs.princeton.edu][6])

4. **提供切片 / 注释分析能力**
   OpenAI 公开了对原始 SWE-bench 所有样本（含 Verified 子集）的注释（包括难度、潜在问题标记等），使得研究者可以按难度 / 题型做深入分析。([OpenAI][3])

5. **避免极端不可解 / 模糊样本拉低整体表现**
   在原始 SWE-bench 中，有些问题被认为“几乎无法在给定信息下可靠解决”；这些样本可能对大多数模型构成障碍。Verified 在设计上就是剔除这种样本，从而让 benchmark 更具判别性和实用性。([OpenAI][3])

因此，很多最新模型、agent 系统在报告自己在 SWE-bench 上的性能时，常常以 **Verified** 版本成绩作为主要对比点。([hal.cs.princeton.edu][6])

---

## 四、局限 / 风险 / 注意事项

虽然 SWE-bench Verified 有较多优点，但也并不是完美无缺的 benchmark。以下是一些需要注意的局限和风险：

1. **覆盖范围偏窄 / 任务单一**

   * 这些样本主要是 Python 语言的 bug 修复 / issue 解决任务，对于全功能的软件工程（设计、架构、性能优化、新功能开发等）能力测评覆盖不足。([Epoch AI][4])
   * 虽然样本来自多个开源库，但总体上几种热门库可能占比过高，这造成数据分布偏差。([Epoch AI][4])

2. **数据污染 / 训练暴露风险**
   因为这些代码 + issue 是公开的，模型训练过程中可能就“见过”其中的代码片段或相似 issue，从而使得评测结果存在泄露 / 作弊风险。([Epoch AI][4])

3. **Scaffold 依赖强 / “系统设计”过度干预**
   在实际评测中，agent 的 scaffold（工具设计、 prompt 设计、错误恢复、探索策略等）对成绩影响极大。若某个模型表现不好，可能未必是模型能力差，而是 scaffold 没给好。导致 benchmark 分析时很难分清“模型能力 vs 工具设计” 的贡献比例。([Epoch AI][4])

4. **低样本下的统计不稳定性**
   虽然 500 样本比起极小子集要好，但做细分（例如某个项目、某个难度等级、某个题型）时，样本数仍可能不够稳定。

5. **“通过单元测试” ≠ 真正语义正确 / 无 bug**
   即便模型生成的 patch 通过了 FAIL_TO_PASS + PASS_TO_PASS 测试，也不保证它在代码库的所有角落、边缘情况下无问题。测试覆盖本身就可能不完善。换言之，benchmark 的判定是基于有限的单元测试，而不是完全语义对齐。([Epoch AI][4])

6. **与真实工程差距 / 通用性问题**
   在现实工程中，很多问题比这些 issue 更复杂、跨模块更多、依赖更多，或者需要业务背景知识、性能约束、安全考量等。SWE-bench Verified 虽然在“可解补丁任务”上是有意义的，但不能代表模型在所有软件工程场景下的通用能力。

---

## 五、目前模型 / Agent 在 SWE-bench Verified 上的表现 / 最新进展

下面列出一些有代表性的结果和趋势。

* OpenAI 在其 “Introducing SWE-bench Verified” 博文中就展示了一些模型（和 agent）在 Verified 子集上的基准表现。([OpenAI][3])
* 在 Anthropic 的博客（“Raising the bar on SWE-bench Verified”）中，其模型 **Claude 3.5 Sonnet** 在 Verified 子集上达到约 49% 的解决率。([Anthropic][7])
* 在 Warp 的博客中，他们宣称其 agent 在 SWE-bench Verified 上能够解决 **71%** 的样例（自动补丁 + 通过测试）——这在当前是较为突出的一项成绩。([warp.dev][8])
* Hal 的 SWE-bench Verified leaderboard 上，可以看到不同 agent / 模型在 Verified 子集上的排名、解决率、成本等对比数据。([hal.cs.princeton.edu][6])
* 在一些社区讨论或开源模型中，也看到有用户报告其在 Verified 上取得 ~40% 甚至更高的 pass@1 成绩。([Reddit][9])

这些成绩说明：当前已经有模型 / agent 在这个 benchmark 上取得了相当不错的成绩，但仍未“饱和”——还有提升空间。

也有一些研究 /应用基于这个 benchmark 的总结性观察：

* 架构设计、prompt /工具策略优化，是拉开模型性能差距的关键。
* 多次尝试 / 生成多个候选补丁 (best-of-k) + 选优机制，是提升成功率的常见策略。
* 成本 /计算资源消耗成为实用性考量：即便模型能在 benchmark 上得高分，但若代价太大（计算、时间、内存），在工程化应用中不一定落地。
* 污染 / “训练中见过数据”问题被频繁提及：要注意 benchmark 结果是否被模型“提前看到过”。

---

## 六、小结 & 对比建议

* **SWE-bench Verified** 是 SWE-bench 的一个“精炼 / 质量更高”的子集，目的是提高 benchmark 本身的可靠性，剔除难以评估 / 模糊 /问题样本。
* 它适合作为模型 / agent 在 “自动修复 / 代码补丁能力” 上的对比基准，但不能全面覆盖软件工程的所有维度。
* 在实际使用时，要特别注意 scaffold 设计、工具接口、补丁候选机制、模型-工具交互能力、以及是否存在数据污染等因素。
* 若要对一个新的模型 / agent 在这方面评估，建议：

  1. 用 **Verified** 子集作为主评估标准（但也可以对比 Original / Lite 子集）；
  2. 做切片分析（按难度、项目、题型）；
  3. 报出 “单次尝试成功率 (single-trace)” vs “best-of-k” 成绩，以更贴近用户体验；
  4. 若可能，测试模型在未见过代码库 /真实项目中的泛化能力，而不仅局限于公开代码库。

如果你愿意的话，我可以帮你把 SWE-bench Verified 和其他常见的 code / software engineering benchmarks（如 HumanEval, CodeXBench, MBPP 等）做一个对比，看看它们各自评测的侧重点／优劣。你要吗？

[1]: https://www.swebench.com/SWE-bench/?utm_source=chatgpt.com "Overview - SWE-bench documentation"
[2]: https://github.com/SWE-bench/SWE-bench?utm_source=chatgpt.com "SWE-bench [Multimodal]: Can Language Models Resolve ... - GitHub"
[3]: https://openai.com/index/introducing-swe-bench-verified/?utm_source=chatgpt.com "Introducing SWE-bench Verified - OpenAI"
[4]: https://epoch.ai/blog/what-skills-does-swe-bench-verified-evaluate?utm_source=chatgpt.com "What skills does SWE-bench Verified evaluate? | Epoch AI"
[5]: https://www.vals.ai/benchmarks/swebench-2025-07-17?utm_source=chatgpt.com "SWE-bench Benchmark - Vals AI"
[6]: https://hal.cs.princeton.edu/swebench?utm_source=chatgpt.com "SWE-bench Verified - Holistic Agent Leaderboard"
[7]: https://www.anthropic.com/research/swe-bench-sonnet?utm_source=chatgpt.com "Raising the bar on SWE-bench Verified with Claude 3.5 Sonnet"
[8]: https://www.warp.dev/blog/swe-bench-verified?utm_source=chatgpt.com "Warp scores 71% on SWE-bench Verified"
[9]: https://www.reddit.com/r/LocalLLaMA/comments/1kh0hcd/cracking_40_on_swebench_verified_with_open_source/?utm_source=chatgpt.com "Cracking 40% on SWE-bench verified with open source models ..."


---

### 🧑‍💻 User

为什么说SWE-bench对于现在的agent llm来说意义重大，很多sota模型都focus on这个领域？

---

### 🤖 Assistant

好——下面直接说明 **为什么 SWE-bench（尤其是 Verified 子集）对现在的 agent 化 LLM 意义重大**，并用要点把原因、它推动的研究方向、以及需要注意的风险都讲清楚。要点后面我放了支持该断言的来源链接（可点开查看原文）。

# 核心结论（一句话）

SWE-bench 把“**真实工程中的端到端修 bug / 提交补丁**”变成了可量化、可复现的评测任务——这正命中了 agent 化 LLM 最需要被检验和优化的能力集合（跨文件理解、工具调用、迭代调试、运行时验证与回退），所以所有想把模型用于真实软件工程的团队都会把它当成重中之重去攻克与优化。 ([openai.com][1])

---

# 详细理由（为什么 SOTA 模型都把它当目标）

1. **任务是真实且端到端的**：SWE-bench 的任务不是单句代码补全，而是从一个真实 GitHub issue 出发——理解上下文、在代码库里定位、生成补丁、把补丁应用并运行单元测试确认“真的有效且无回归”。这种 end-to-end 测评更贴近工程落地需求。 ([openai.com][1])

2. **它同时考验模型 + agent 能力**：成功通过 SWE-bench 不只是语言建模能力，还需要工具使用（bash、测试运行、文件编辑）、长上下文管理、迭代调试和失败恢复策略——这些刚好是“agent 化系统”要解决的问题，所以研究者把改进集中在 agent 设计和模型交互能力上。 ([swebench.com][2])

3. **高业务价值，容易量化 ROI**：自动修复真实 bug 的能力直接对应工程效率（减少工时、提升交付质量）。因此公司/产品团队愿意投入大量工程资源把模型调到在 SWE-bench 上表现好，以便说服内部产品化落地。 ([warp.dev][3])

4. **Leaderboard & 可比较性驱动竞争**：SWE-bench 有公开排行榜和多家厂商/研究团队的提交（包括公开成绩和技术博文），这些排名成为宣传/对比的关键指标，促使 SOTA 团队争相优化到更高分（从而把 attention 聚拢到这个 benchmark）。 ([swebench.com][2])

5. **可复现的自动验证（单元测试）减少噪声**：通过 FAIL_TO_PASS 与 PASS_TO_PASS 的测试来判定修复是否生效，提供了比纯人工评审更客观的量化标准（虽然不是完美，但比只看生成质量更可靠）。这对工程化评估非常重要。 ([openai.com][1])

6. **能揭示 agent 的真实短板**：SWE-bench 不仅显示“哪些模型能成功”，还会暴露出诸如依赖解析失败、工具调用不当、测试-环境重现问题、以及对安全 / 漏洞修复能力的不足，这些是把模型推向生产必须攻克的点。许多分析/论文也开始用它来揭示 benchmark 的局限与模型“表面强”但实际脆弱的地方。 ([runloop.ai][4])

---

# 为什么社区同时既重视又警惕它

* **重视**：因为它是最接近“能否协助写代码并修真实 bug”的实测指标，企业价值明显，且排行榜能直接展示进步。 ([warp.dev][3])
* **警惕**：因为公开代码/issue 导致**数据泄露/污染**（模型训练时可能已经见过很多样本），以及研究者发现存在“bench overfitting / benchmark gaming”的风险；因此社区同时在做更严格的子集（如 Verified）和更难的延伸版本（如 SWE-bench Pro/SWE-bench+）来缓解这些问题。也有研究提示 SOTA 在更严格、更真实的变体上会显著降级。 ([openai.com][1])

---

# 对研究者 /工程师的实用启示

* **把 agent 能力当第一阶目标**（工具接口、重试策略、长上下文摘要、生成-验证循环），不仅只靠提升裸模型参数。 ([swebench.com][2])
* **报告时同时给出：single-try 与 best-of-k、compute cost、以及是否存在数据污染**，这样结果更可信、可比。 ([runloop.ai][4])
* **不要把高分当作“万无一失”的证据**，要用更难、更长时景的测试集（或内部未公开样本）做泛化验证。 ([arXiv][5])

---

# 简短结语

SWE-bench（尤其是 Verified）之所以成为众多 SOTA 团队的聚焦点，是因为它把“能不能在真实代码库里自动修 bug”这个高度工程化、可量化的问题，变成了一个**可以复现实验、产出 leaderboard、直接衡量商业价值**的 benchmark。攻克它，意味着你的 agent 在工程落地上迈出了一大步；但同时也要警惕 benchmark 的局限与数据泄露风险。 ([openai.com][1])

如果你想，我可以：

* 把当前 Verified leaderboard 的前 10 名和它们的技术亮点列出来（我可以抓最新排名并汇总）。
* 或者把 SWE-bench 的典型失败案例列出来，帮你看到哪些场景仍然很难自动化。

你想看哪一个？

[1]: https://openai.com/index/introducing-swe-bench-verified/?utm_source=chatgpt.com "Introducing SWE-bench Verified"
[2]: https://www.swebench.com/?utm_source=chatgpt.com "SWE-bench Leaderboards"
[3]: https://www.warp.dev/blog/swe-bench-verified?utm_source=chatgpt.com "Warp scores 71% on SWE-bench Verified"
[4]: https://www.runloop.ai/blog/swe-bench-deep-dive-unmasking-the-limitations-of-a-popular-benchmark?utm_source=chatgpt.com "SWE-bench Deep Dive: Benchmarking AI Coding Agents"
[5]: https://arxiv.org/html/2506.12286v3?utm_source=chatgpt.com "The SWE-Bench Illusion: When State-of-the-Art LLMs ..."
