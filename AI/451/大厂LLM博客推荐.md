# 大厂LLM博客推荐

## 🧑‍💻 User

你能推荐一些类似 A\ 和 Deepmind 这样的 LLM 大厂 tech report/tech blog 看嘛，这几天看 A\ 以前发的，看爽了，有点想多找一些看看

---
## 😀 默认助手 (7)

你喜欢的“A\ 那种”内容，大致有三种特质：**工程细节够具体、敢写踩坑和失败案例、语气偏“研究者的第一人称叙事”**。DeepMind 那边更偏“成果发布+方法综述”。按这个口味，我分了几档，你可以先扫一遍表格，再按需展开。

## 核心推荐（最接近你现在的阅读口味）

| 来源 | 为什么值得看 | 代表作切入点 |
|---|---|---|
| **Transformer Circuits**（`transformer-circuits.pub`） | Anthropic 的另一面：可解释性长文，动辄几万字，是“看爽”的顶点 | *Towards Monosemanticity*、*Scaling Monosemanticity*、*On the Biology of a Large Language Model* |
| **Meta AI / AI at Meta** | Llama 系列的技术报告附件、FAIR 的 scaling 与数据工作 | Llama 3 报告里的“数据配比+infra”章节 |
| **OpenAI**（`openai.com/index`） | system card、reasoning 系列的方法叙事 | o1/o3 的 *Learning to Reason*、GPT-4o system card |
| **Google Research Blog** | 比 DeepMind 博客更“技术”、更少公关式表述 | scaling laws、Gemini 的训练稳定性 |
| **Ai2（Allen Institute）** | **OLMo**：数据、代码、训练日志全公开，还能看到中途失败和重启 | OLMo 2 tech report + 训练日志 |
| **Mistral AI News** | 短小但干货密集，MoE 和稀疏激活讲得直白 | Mixtral、Large 2 的模型卡 |
| **Databricks Mosaic AI Blog** | 高效训练、小模型、合成数据的工程实践 | DBRX、MPT、Megablocks |
| **NVIDIA Developer Blog** | 训练和推理的底层细节（Megatron、NeMo、FP8） | FP8 训练、长上下文并行 |

## 国产/中文实验室的技术报告（近两年信息密度最高的一档）

这批技术报告有一个很实际的优点：**篇幅长、公式推导和消融实验写得细，而且敢写“我们试了但没work”**。DeepSeek 和 Qwen 的报告是其中最有代表性的。

- **DeepSeek**：V2（MLA + DeepSeekMoE 的动机推导）、V3（FP8 训练、无辅助损失的负载均衡）、R1（纯 RL 的 “aha moment”）。这三篇连着读的体验，几乎和看 Anthropic 博客一样令人满足。
- **Qwen**：`qwenlm.github.io`，Qwen2.5、Qwen3 的报告，以及 Qwen3 的 “thinking budget” 设计。
- **Moonshot**：Kimi k1.5（long-CoT RL）、Kimi K2（MuonClip、万亿 MoE 的稳定性）。
- **智谱 GLM-4.5**、**MiniMax-01**（lightning attention 的工程化）、**阶跃 Step-3**（多模态解码）。
- **字节 Seed**（`seed.bytedance.com`）、**腾讯混元**、**上海 AI Lab**（InternLM、InternVL）、**OpenBMB MiniCPM**（小模型“密度定律”）。

<details>
<summary><b>展开：Infra / 系统 / 训练框架类（想理解“为什么这么设计”必看）</b></summary>

- **PyTorch Blog** — Horace He 的 *How to Scale Your Model* 系列，把并行策略、通信量、roofline 讲得极清楚，强烈推荐作为“读懂所有 tech report 的前置课”。
- **vLLM Blog** 和 **LMSYS / SGLang Blog**（`lmsys.org/blog`）— PagedAttention、前缀缓存、PD 分离这些推理侧设计，论文和博客两版都有。
- **Hazy Research**（Chris Ré 组）— FlashAttention 系列的思路源头。
- **Together AI Blog** — 推理优化、投机解码的实测数据。
- **Anyscale Blog** — Ray 上的分布式训练。
- **EleutherAI Blog** — 开源复现的“我们真的跑了”记录。
- **Apple ML Research**（`machinelearning.apple.com`）— 端侧模型、量化、隐私。
- **Microsoft Research Blog** — Phi 系列，主打“小模型+高质量数据”。
- **Amazon Science Blog** — 偏工业落地。

</details>

<details>
<summary><b>展开：个人博客/Newsletter（“像 A\ 那样讲人话”的替代品）</b></summary>

如果 A\ 那种“一个研究者认真跟你聊他在想什么”的感觉最吸引你，个人博客往往比公司博客更贴近：

- **Lilian Weng**（`lilianweng.github.io`）— 必读，*Why We Think*、*LLM Powered Autonomous Agents*、*Extrinsic Hallucinations*。综述能力和叙事节奏都是顶级的。
- **Jason Wei**（`jasonwei.net/blog`）— 短、狠、洞察密度高。
- **Nathan Lambert / Interconnects** — post-training 与 RLHF 的生态观察，是最靠谱的“行业内视角”。
- **Sebastian Raschka / Ahead of AI** — 从零实现视角拆架构，配代码。
- **Jay Alammar** — 图解派，适合回炉基础。
- **Simon Willison**（`simonwillison.net`）— 亲手跑、亲手测，落地的第一手体验。
- **Eugene Yan** — 评测与应用侧。
- **Hamel Husain**（`hamel.dev`）— evals，很多“为什么你的评测是错的”的实战文。
- **Colin Raffel**、**Tim Dettmers** — 训练细节与量化的“反常识”观点。
- **Andrej Karpathy** — 不用多说，*Deep Dive into LLMs* 那类长视频/长文的文本版。

</details>

<details>
<summary><b>展开：聚合与“跟进度”渠道（不想自己筛 arXiv 时用）</b></summary>

- **Hugging Face Blog + Daily Papers** — 每天的人工精选，覆盖最全。
- **The Batch**（Andrew Ng）、**Import AI**（Jack Clark）、**TLDR AI**、**AlphaSignal** — 每周摘要。
- **Alignment Forum / LessWrong** — safety 与可解释性的一手讨论，和 Anthropic 的价值观一脉相承。
- **Epoch AI**（`epoch.ai/blog`）— 算力、数据、scaling 趋势的硬数据，写报告引用它很稳。
- **Distill**（`distill.pub`，已停更但仍值得读）— 互动式解释的巅峰。

</details>

## 如果只挑三个先看

1. **DeepSeek-V3 技术报告** — 目前“工程细节 + 设计动机”平衡得最好的一篇，读完你会发现很多后续模型都在抄它的结构。
2. **PyTorch 的 How to Scale Your Model** — 补上这块，之后再看任何 tech report 的 infra 章节都不会跳过。
3. **Lilian Weng 的任意一篇** — 用来“校准”自己的阅读口味，看她怎么把散落的论文串成一条线。

## 一个读法上的建议

公司 tech report 里最容易被跳过、但信息量最大的其实是这三块：**消融实验（ablation）**、**失败案例（what didn't work）**、**limitations / system card 的风险段**。尤其是失败案例这部分，官方正文常常略过，反而在附录或博客的 FAQ 里；Anthropic 和 DeepSeek 是少数会大方写出来的，这也是你看得“爽”的真正原因。

需要的话，我可以帮你把上面这些整理成一份带链接、按“入门→进阶→硬核”排序的阅读清单；或者按你感兴趣的某个方向（推理 RL、MoE 训练、可解释性、推理加速）再横向拉一遍，只看那条线上的代表作。

---
## 🧑‍💻 User

我记得最近一个是微软的maibase2预训练 还有一个是marin那个500b的moe好像也pretrain跑完了 不知道发blog没 不过它的代码和log都是public的

---
## 😀 默认助手 (7)

让我先确认一下这两个消息的实际情况——尤其 Marin 那边到底发没发 blog。

都记得没错，两件事我都确认了：微软那篇已经发了，而且正是你想要的那种“技术报告密度”；Marin 的**预训练确实跑完了（终评已出）**，但**还没有一篇“完工”的 blog**。下面分别说明。

---

## 1. 微软：`MAI-Base-2` 真的发了，而且是长文

**《Frontier Pre-training at Microsoft AI》**（Lab Notes，2026-10-15，约 15 分钟读完，另有 PDF 版）
- 博客：`microsoft.ai/news/frontier-pre-training-at-microsoft-ai/`
- PDF：`microsoft.ai/pdf/Frontier-Pre-training-at-Microsoft-AI.pdf`

**规格**：MoE，**80B 激活 / 2.8T 总参**；**50T tokens**，token 全部自建管线处理、只用 organic data、不做任何模型蒸馏（这也是他们敢做 weight licensing 的底气）；**原生多模态**（encoder-free VLM，VLM 语料 4.5T unique tokens，占 50T 混合的 18%）。

**run 的硬指标**：16k 张 GB200，2026 年 8–9 月跑了 **38 天**，平均 **MFU 21%**（按 GB200 BF16 2.5 PFLOPS 归一化，且不含 remat 和 optimizer 的 FLOPs）、**goodput 89%**。结论是 held-out BPB 全面低于所有开源权重模型，并声称比 `DeepSeek-V4-Pro-Base` **FLOPs 效率高 4.2–12.3×**、比自家 MAI-Base-1 高 6.3–12.7×。

**最值得读的四段**（这篇的“爽点”全在这里）：

| 章节 | 干货 |
|---|---|
| **Source scaling** | 在 scaling ladder 的小档位上**人为限制可用 unique 样本量**，让每个数据集被重复的次数和 50T 大跑时一致。理由是：固定 TPP 下这等价于固定“unique data / params”比例，否则小规模实验会偏向过拟合、欠正则的配置。他们说这是能稳到 2.8T 的关键。 |
| **超参 scaling law** | 发现最优点落在 $\log LR$ 与 $\log WD$ 的对角窄带上，所以不在矩形网格上扫，而在 $(\log LR,\ \log(LR\times WD))$ 上扫，再拟合 2D 二次曲面取解析极小值。最后给出可用的经验式：<br>$$LR^* \times WD^* = \frac{17.04}{\text{steps}}$$<br>$$\log LR^* = -1.51 + 0.175\log \text{TPP} - 0.573\log \text{steps}$$<br>$$\text{Dropout} = \mathrm{sigmoid}\!\left(0.526(\log \text{ActiveParams} - 27.2)\right)$$<br>其中 LR·WD 对 steps 的指数≈$-1$，正好对应“解耦 weight decay 的时间尺度应占训练固定比例”；LR 对 steps 指数≈$-0.5$，就是“batch 翻倍、LR 乘 $\sqrt2$”那条经验法则。 |
| **多轮 dropless MoE** | 用固定 token capacity + 多轮 dispatch→compute→collect 处理溢出，还允许过载专家借用同组专家的空闲容量，backward 按专家组做细粒度重算——**任意路由不均衡下显存占用恒定**，既不丢 token 也不炸。配合分组 GEMM/NVLink、device 端 token metadata、symmetric memory 直连 all-to-all、跨层 batch 化 LB loss，把 MoE 层的 MFU 从 **24% 提升到 35%**。这正是他们敢“每层都上 MoE”（而不是隔层）的原因。 |
| **Activation gradient clipping** | 不裁权重梯度的全局 norm，而是裁**若干位置的 activation 梯度**——因为 outlier 只来自 batch 里极少数样本，全局裁会连好样本一起压。效果是和 Muon 一起把 grad norm 压得更紧，而且**收益随规模变大**（他们说绝大多数改动都是收益恒定的）。更妙的是把它当**数据诊断工具**：图中定位到一个被截断的 license header，顺藤摸出一个预处理 pipeline 的 bug。 |

另外 Muon 那段也很实：为配合 ZeRO，在 optimizer step 里把动量 **reshard → BF16 all-to-all → 本地对称性感知 Newton–Schulz → all-to-all 换回**，再叠加 prefetch、metadata 缓存、router 权重免 shard，最终 Muon 只占单步 **2.1%（165 ms）**，理论下界 55 ms，差的 110 ms 老实承认是 straggler、kernel dispatch、负载不均和 reshard 拷贝。

> 小建议：这篇可以配合同一批的两篇短读——`Optimizing the frontier performance curve` 和 `Hill-climbing MAI models for GitHub Copilot and Excel`，基本是同一套“EG（efficiency gain）驱动决策”方法论的展开；他们还在文末明说 **MAI-Base-3 已经在准备**。

---

## 2. Marin 535B-A23B：预训练已完成，但**完工 blog 还没写**

先说结论：**跑了，而且达标了，但还没有“final report”**。

- **规格**：535B 总参 / **23B 激活**，48 层，rack-local expert parallelism，两阶段 “Harrier” 数据混合，**18T tokens**；2026-08-19/20 起跑，算力来自 **Huang 基金会（Jensen & Lori）的 GPU 捐赠**——9 月 2 日那篇就是为此发的。
- **结果**（实时追踪页 `mtracker.oa.dev/hero-run-535b` 的当前横幅）：**“✓ Mochi has finished pre-training. Post-training begins soon.”** 终值 **Paloma macro eval loss 2.035**（step ~393,490，对应 18T 附近），final train loss **0.878**。而首页 preregistered 的目标是 **2.04**——**略微打赢了自己的预注册预测**。
- **但 blog 里没有“完工”文**。最新一篇是 10-15 的 *There is no posttraining data commons*，再往前是 9-29 *Expert parallelism at hero scale*、9-11 *Curating 25 Trillion Tokens*、9-03 *535B-A23B launch note*。权重也还没上 HF（`marin-community` 的 Hero Runs collection 目前只有 8B / 32B / 8B-instruct）。

所以现在想看这个 run，**最好的“报告”其实是那个 tracker**。它每几分钟自动更新，把事故日志也留着，这段很能说明问题：

- `Host memory exhaustion recurs about every three hours; tuning the XLA memory fraction and slop factor did not help.` → 后来靠 jemalloc + 限制 TensorStore restore 并行度 + checkpoint offload cache + 每次 save 后 `malloc_trim` 才压住；
- `An uncorrectable NVLink error on one node took down three tasks in the same rack.`
- 还有 “每次 resume 到 checkpoint load 阶段就被容器 runtime 杀掉” 那轮，PR #8480 才修好。

这种“故障-定位-修复”的连更，本质上就是 Anthropic 那种 postmortem 叙事的实时版。

**建议的读法**：先看 9-03 的 launch note 建立背景 → 9-29 的 expert parallelism（为什么 EP 让 535B-A23B 变得可行，含 FSDP / 固定容量 EP / scaling ladder 对比）→ 然后挂着 tracker 等完工文和权重。按他们 8B/32B 的惯例（retro 文 + base/instruct 权重），完工文大概率会叫 *535B retrospective*，值得蹲 `marin.community/blog/`。

---

## 3. 顺带：这条“开放式预训练”线上还有谁

既然你吃这套，这两篇确认完正好可以顺着扩：

- **Marin / Open Athena 全家桶**：`marin.community/blog/`（Delphi 那篇“外推 300× 只差 0.2%”的预注册预测、Quantile Balancing、Iris 调度都值得看），以及 `openathena.ai/blog/`、`tecunningham.github.io/open-labs/` 这类第三方综述。
- **Ai2 OLMo**：唯一和 Marin 同级别的“数据+代码+日志全开”，而且有 mid-run 失败的记录。
- **EleutherAI**：复现向的“我们真的跑了”日志。
- **MS 的 Lab Notes 系列**：这是微软 AI 新开的技术长文栏目，MAI-Base-2 这篇是开山级的一篇，后续 MAI-Base-3 值得订阅。

要不要我把 MS 这篇的四个技术点（source scaling、LR/WD 联合律、多轮 dropless MoE、activation grad clipping）各自再往下挖一层，找对应的论文/前作读一遍？尤其 **source scaling** 和 **activation grad clipping** 这两个我印象里没有直接对应的公开论文，可能是他们原创的点，值得单独追。
