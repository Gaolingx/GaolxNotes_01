# KohakuGBC标注优势：结构化接地工程化复现优于原版GBC不敌Tagger速度词表

## 🧑‍💻 User

https://github.com/KohakuBlueleaf/KohakuGBC
横向比较relativework：kohakugbc在caption标注优越性在哪（对比pixai-tagger、WD14、原版GBC paper等）

---
## 😀 默认助手 (7)

# KohakuGBC 在 caption 标注上的横向定位

先说结论，然后逐条拆。

> **一句话**：KohakuGBC 并不试图在“打标签”这件事上胜过 tagger，它做的是把**标注的内容形态**从“扁平标签集合”改成“带坐标、带层级、带关系的引用图”，并且在**工程上**让这张图能大规模生成；相对原版 GBC 论文，它的优越性主要体现在**可复现的忠实度、可承受的算力成本、以及把 PoC 的结构性缺陷（父节点等于调用树、单名单框、检测器在插画上失效）逐个修掉**。

但有一句必须提前讲清楚，否则容易过度解读：**README 里没有任何“caption 质量”评测**（没有人评、没有 CLIP 下游训练、没有对 tagger 的图级别 F1）。它给出的证据只有吞吐、忠实度（digest 级别的等价）和模型替换后的统计量。所以“优越性”里，**结构性与工程性是有据可查的，语义质量上是推断而非实测**。

---

## 0. 先把对齐的基准放平：三者不在同一层

| 维度 | WD14 (`wd-v3-*`) | PixAI Tagger v0.9 / v1.0 | 原版 GBC（paper + `apple/ml-gbc`） | KohakuGBC |
|---|---|---|---|---|
| 输出空间 | 扁平多标签与逐标签置信度 | 同上（general / character / copyright 分组） | typed graph：`image` / `entity` / `composition` / `relation`，**每个节点都有自由文本** | 同 GBC，JSON 与 `ml-gbc` 兼容 |
| 词表 | 封闭约 $10.8$k | 封闭约 $13.5$k（Danbooru 2025-01 快照） | 开放自然语言（无词表约束） | 开放自然语言 |
| 空间定位 | 无（要自己切 crop 重跑） | 无 | 每节点 bbox，边三重接地 | 同 + SAM 3 实例分割 + 整图 token 池化 |
| 层级（part-of） | 无 | 无 | 有（父 = 命名它的 crop） | 有，且父由**包含关系 + 逐条核验**决定（GBC-Nest） |
| 关系与组合 | 无 | 无 | 有 relation / composition 节点 | 有 |
| 一致性/可校验 | 阈值可调，无证据结构 | 同 | 三重接地可程序化检查 | 同，且可 record & replay 到 digest |
| 单图代价（同卡） | 单次前向 | 移植版 $65.8$ pics/s（5090, MXFP8） | 数十次串行 VLM 调用 | $2.06$ img/s（5090，约 $14$ 次 VLM 调用、共约 $33$ 次引擎调用/图，mid+sam3） |
| 许可 | 各自 | Apache-2.0 权重 | 代码 Apple Sample Code；GBC10M **CC BY-NC 4.0** | **Apache-2.0** |

这个表本身就已经说明了“优越性在哪”：**它不在同一个评测口径上，所以不存在“谁打标签更准”的问题；真正的问题是“你需要什么样的标注对象”**。

---

## 1. 对比 WD14、PixAI tagger：优越性在“结构”、“接地”与“能就地使用 tagger”

<details>
<summary><b>1.1 结构信息：tagger 在原理上就无法表达的三类东西</b></summary>

tagger 的输出是 $\{(t_i, p_i)\}$，即“词”与“概率”，**没有论证结构**。GBF 图里每个节点带文本、每条边带一个接地短语，并且边满足三重接地：

- 它是一条 graph edge；
- 边上的 label 是**父节点 caption 里的一个短语**；
- 子节点的 box **落在**父节点的 box 之内。

这三条都是可以写成断言来检查的。也就是说 GBC 标注**自带 QA 判据**：你可以统计 containment 违规率、phrase 缺失率，作为整批数据的质量指标。tagger 的输出连“这条标签是哪个视觉证据支撑的”都无从谈起，更不用说自动核查。在一个图里还有 `composition`（把同一物体的多个实例聚到一起）与 `relation`（“A 抱着 B”）两类节点——这两类恰恰是 CLIP 类对比学习最缺的信号（原论文的核心卖点就是 composition / relation 节点的下游增益）。
</details>

<details>
<summary><b>1.2 绑定问题（binding）：标签落在“谁”身上</b></summary>

这是我认为 KohakuGBC 最实质、也最被低估的一点，见 `src/kohakugbc/nodes/tags.py`。它把 tagger 的整图标签**分配回图的顶点**，规则是显式写的：

| 标签类别 | 落点规则 |
|---|---|
| `rating` / `meta` / `style` / `copyright`、计数类（`1girl`、`multiple_girls`）、景别/背景类 | 只挂到 image 顶点 |
| `character`（身份） | 挂到得分通过的**最上层**顶点（部件也会得分，但不该再挂一份） |
| 普通属性 | 挂到**最深**的顶点（`pink hair` 挂到头发，不重复挂到“女孩”上） |
| 复数标签（`boots`） | 若命中多个顶点，上移到它们的最低公共祖先（`2 boots` 这个 composition，否则穿戴者） |

tagger 天生只知道“图里有 `pink_hair`”，不知道“pink hair 属于第二个女孩”。KohakuGBC 用一个现成的 tagger 加一张图就补上了这块——**不需要重训 tagger**。而且每个顶点还会生成一条 `bagofwords` caption，等于让“可检索的一致性词表”和“自由描述”长在同一个节点上。
</details>

<details>
<summary><b>1.3 它甚至能让扁平 tagger 变成区域标注器（零训练）</b></summary>

`tag_regions` 有两种：`crop`（把顶点 box 切出来当独立图片跑）和 `pool`（**从整图那一次前向里，把落在 box 内的 token 池化**）。后者的依据是：ViT 的全局注意力使每个 token 都见过整图，所以区域也能得分，只是比真正含该物体的区域低——代码里用 `tag_region_scale` 与 `tag_region_margin`（logit 空间）来卡这个落差。这意味着“整图分类器 → 区域标注器”的转换是**推理侧的技巧**，不花训练成本。
</details>

<details>
<summary><b>1.4 tagger 更强的地方（必须承认）</b></summary>

- **速度**：移植版 PixAI tagger v1.0 在 5090 上 $65.8$ pics/s，即约 $15$ ms/图；而整条 GBC 流水线是 $0.49$ s/图，**约 $32$ 倍**。按每图约 $14\times(1000+100)\approx 1.5\times10^4$ tokens 计，tagger 的 token 账是 $0$。
- **词表一致性与可检索**：13k 封闭词表天生适合倒排索引、去重、训练时的固定条件向量。自由文本的 GBC 描述需要额外解析。
- **标注来源更“人”**：WD/PixAI 的监督信号来自 Danbooru 人工标签，因此**概念层面的精度有人类兜底**；GBC 的文本由 VLM 生成，存在幻觉（README 也没否认）。
- **无“结构性幻觉”**：tagger 不会编造一条不存在的 `relation` 边；GBC 的 JSON 语法约束只保证**格式合法**，不保证事实正确。
- **生态**：tagger 直接喂 T2I 条件、直接做检索；GBC 的输出要再转一层才能用（原论文用 `graph→text` 的 transform 做这件事）。
</details>

<details>
<summary><b>1.5 于是关系不是替代，而是“升级”或“互为校验”</b></summary>

README/`engines.md` 里 `tag` 是**可选的第 5 个 slot**，默认模型就是 `pixai-labs/pixai-tagger-v1.0`（GBC 自己的移植，ViTDet backbone 跑在 SAM 3 的 ViT 层与 kernel 上，fp8，batch 级 CUDA graph）。两条附加能力：

1. **`root = "tags"`**：把现成 Danbooru tag 当作**种子根**（`tag_threshold` 0.35、可选 `tag_hint` 把 tag 塞进 element prompt），让 GBC 从你已有的 tag 出发去补结构。这是与 tagger **互补**而不是对立的路径。
2. **互为校验（我的建议，非 README 内容）**：每个顶点同时有 `tags`（带 category 与 score）与 `descs`（自由文本）。两者不一致（例如文本说 “two girls” 但 tag 集只有 `1girl`）就是最廉价的**幻觉筛选信号**，可用来批量挑出需要人工复核的图。
</details>

---

## 2. 对比原版 GBC paper / `ml-gbc`：优越性在“可复现”、“成本”、“修掉了 PoC 的结构缺陷”

### 2.1 先看原版是什么形态

`ml-gbc` 的 captioning 默认是 **Pixtral-12B（或 LLaVA 1.6）+ GroundingDINO / YOLO-World**，Hydra 配置、逐图串行的 round 引擎；GBC10M 实际是用 **LLaVA-1.6-Yi-34B + YOLO-World + Jina embedding** 产出的，图统计为：$12.24$ 顶点 / $21.81$ 边 / $17.67$ 条 caption / $534$ 词 / 图，平均直径 $4.41$。这是一个**研究型 PoC + 数据集发布**，不是一条为吞吐设计的产线。

### 2.2 忠实度：这是它最硬的差异化

| 检查 | 结果 |
|---|---|
| Python / Rust 两套图解析 vs `ml-gbc` 的转换 | 462 条 PoC 记录全等；Rust $0.21$ ms、Python $1.27$ ms 每图 |
| KIR 参考解释器重放六条 PoC arm | $180/180$ 图与 PoC **digest 全等**，且每个请求都是 PoC 发过的请求 |
| 机器在 replay 引擎上 | $1320$ 个 job 全等、全部符合 contract |

也就是说，“改写”没有引入语义漂移——你可以把它当成**原程序的等价可执行规格**。对一个需要反复重生成数据集的项目，这是很值钱的性质（`record` 任意比例的 job 都能逐位重放）。

### 2.3 它修掉了 PoC 的几处**真实标注缺陷**

<details>
<summary><b>(a) 编码 bug：非 ASCII prompt 被 locale 弄坏</b></summary>

PoC 用平台默认 locale 读 prompt 文件（录制机是 cp950），**image-query 的 system prompt 里的非 ASCII 字符被弄成乱码**。KohakuGBC 改成 UTF-8。这不是性能问题，而是**直接损害 caption 质量**的 bug——而且它被修掉了，还保持了 digest 等价（说明乱码只影响被替换的字符）。
</details>

<details>
<summary><b>(b) 未播种的随机 → 可复现</b></summary>

关系条数、命名一组子节点的短语、composition 的根与措辞，PoC 用的是**无种子 generator**；重复句用 `set` 去重（顺序受进程 hash seed 影响）。KohakuGBC 改成按“被问的东西”播种，并按首次出现顺序去重。对数据集生产来说，“同一张图两次跑出两条不同标注”是不可接受的。
</details>

<details>
<summary><b>(c) 检测器换代：OWLv2 → SAM 3，直接在插画域见效</b></summary>

README 给的例子很具体：一张动漫风肖像上，**OWLv2 最好的 “girl” 框（0.415）只框住了腿**，而 GBC 的“单物体”规则会把腿当成女孩 → 后面所有“part 探测”都基于错的 crop，尾巴被识别成 “hair”。**SAM 3 给出全身框 0.89、头发 0.94、尾巴 0.91，并把尾巴和头发分开。** Grounding DINO 族在这类插画/二次元输入上的同类失效是众所周知的。

对一个以 anime/illustration 为主的生态（Kohaku 的 `TIPO` 系列就训在 GBC10M 上），这一条比任何 abliteration 都更实际：**检测框错 → 层级错 → 整张图的 caption 全部错位**，而 tagger 根本不产生这种错误（它没有层级）。同时 SAM 3 的移植版相对 transformers 是 $2.3\times$（5090 bf16），EmbeddingGemma 移植是 $1.9$–$8.3\times$，说明“换更好的模型”没有付出吞吐代价。
</details>

<details>
<summary><b>(d) GBC-Nest：针对 PoC “父节点 = 调用树”的结构性修正</b></summary>

原版 GBC 的层级来自**调用树**：一个区域的父节点，就是“哪个 crop 的提问命名了它”。这带来三个已知病灶，GBC-Nest（第二个程序）逐个处理：

| 病灶 | GBC-Nest 的做法 |
|---|---|
| 根只保留“简洁 caption 点到的元素”，漏标的物体永远进不了图 | 根保留 image query **列出的每个元素**，另外用 `probes`（文字、logo、漫画分格、人）去扫“没被命名的地方” |
| 单名单框：一个词只留一个 box，`two boots` 会塌成一个 | 共享 head noun 的单数名**一对一配框**；单名取 `single_tie` 内最大的框 |
| 合并会撞上祖先（部件被并进整体） | `nest_register` **永不合到祖先**；先按 head noun + 双向重叠，再退回 GBC 的 embedding 规则 |
| 父子关系不可质疑 | 每个候选嵌套问一次 yes/no（holder 框红、child 蓝），并允许**一个顶点有多个 holder** |
| composition 与“一片区域的两个碎片”混淆 | 每个 group 问一次：是分开的东西，还是同一区域的两块 |
| 关系问得太泛（每个多子节点都问） | 只在 image 顶点问，问“交互与物理连接”，且**允许回答“无”** |

<summary>这些修改的**评测**仍是缺失的</summary>（见第 3 节），但方向上有两个明确的收益：**层级不再由“谁来提问”偶然决定**，且**父子关系有了一次显式核验机会**。
</details>

### 2.4 规模化：从“跑得动”到“跑得起”

| 引擎/卡数 | 图数 | img/s | job p50 / p90 / p99 | GPU 占用 |
|---|---|---|---|---|
| 机器进程内 | 1 | 1200 | $1.886$ | $44/58/71$ s | 97.2% |
| 每卡一个 engine host（默认） | 1 | 1200 | $\mathbf{2.061}$ | $41/71/87$ s | 98.1% |
| 同上 | 4 | 4800 | $\mathbf{8.165}$ | $42/67/84$ s | 96.2–97.8% |

换算成实际成本（按 README 数字外推，非官方声明）：

| 规模 | 1×5090 | 4×5090 |
|---|---|---|
| $10^5$ 图 | 约 $13.5$ h | 约 $3.4$ h |
| $10^6$ 图 | 约 $5.6$ 天 | 约 $34$ h |
| $10^7$ 图（GBC10M 规模） | 约 $56$ 天 | 约 $14$ 天 |

同期性上限由 admission 限定（`max_jobs` 与 `vlm_waiting`），所以**吞吐与单图时延不随源长度漂移**——这是冲“百万到千万图”时最常被忽略的性质，PoC 的串行 round 引擎没有它。并且**同样的图统计用 4B 模型做到了**：PoC 图均 $15.2$ 顶点 / $26.3$ caption，KohakuGBC 的 `mid+sam3` 是 $14.7$ / $26.8$，而 GBC10M 用的是 34B LLaVA。$8\times$ 的模型规模差，密度不掉。

一个关键工程点：**顺序程序 → 机器执行**。GBC 一张图的流程是分叉的（$3$ 到 $30+$ 次 VLM 请求，视回复而定），无法手工跨图分阶段批处理。Togeari 让每个 job 乱序发射、以程序序退休，跨 job 把同类调用并成一次引擎批——这就是它能把 VLM lane 顶到 384 并发、单图同时有最多 10 个自己的请求在飞的原因。

### 2.5 其他实用性上的领先

- **许可证**：代码 Apache-2.0、你自己产出的标注归你；GBC10M 是 **CC BY-NC 4.0（非商用）**，`ml-gbc` 代码是 Apple Sample Code 授权。要商用数据集，这条几乎是决定性的。
- **引擎可换**：任意 OpenAI 兼容 / Anthropic API、in-process vLLM、自研 `qwen35`、跨节点 `gbc host`；`replay` 引擎让你**不加载任何模型**就能跑通全流程与回归测试——做 captioning 基础设施时这是极高的调试效率。
- **产物即是训练数据**：输出 GBC dataset（KohakuClip 分片 + KohakuVault 元数据）；`kohakugbc-data` 单独发包，训练侧不需要 GPU 栈。
- **控制面变成数据**：`root` / `min_nodes` / `max_nodes` / `max_rounds` / fan-out / `relational` / prompt set 都是**逐图 `controls`**，还给了 `small`/`mid`/`large` 预设——可以按数据集统计量反向定预算，PoC 没有这套。

---

## 3. 反过来：它**并不**在这些方面更优（诚实清单）

<details>
<summary><b>展开</b></summary>

1. **没有 caption 质量证据。** 没有人评/A-B、没有 GBC10M 那样的 CLIP 训练下游实验、没有对 tagger 的图级 F1。仓库自己也写明：“GBC is a compute project: the tests check data and formats only”。**“KohakuGBC 的标注比 GBC10M 好”目前是未经测量的命题。**
2. **不能直接继承原论文的结论。** GBC10M 用 34B LLaVA 生成；下游 CLIP 增益是**那批标注**的属性。换成 Qwen3.5-4B 生成的图，增益会平移多少，未知。
3. **成本仍高一个量级以上。** $0.49$ s/图 vs tagger 约 $0.015$ s/图、$1.5\times10^4$ tokens/图 vs $0$。对“只要 tag 做 T2I 条件”的场景，这是纯亏。
4. **幻觉不可免。** 结构化与接地能**发现**一部分不一致，但不能阻止 VLM 编造；`ask` 失败还会以空回复继续（“as GBC reads one”），批量跑时会产生空节点。
5. **检索与词表一致性弱于 tagger。** 自由文本无法直接倒排；`bagofwords` 顶点级 caption 是补丁，不是替代。
6. **域校准偏科。** 检测选择参数（`box_score` 0.1、NMS IoU 0.2/0.05、`drop_containers`）是**按 OWLv2 调的 PoC 原值**；`sam3` 预设改了 0.3/0.7，但整体调参经验集中在插画/二次元，并非通吃所有域。
7. **生态耦合。** KIR/Togeari/KohakuvLLM 是作者自有栈，学习与运维成本不为零；42 页论文的读者未必想读一门 IR 语言。
8. **并非“原版替代品”。** 它复制的是 **PoC + ml-gbc 的解析/后处理/合并/关系查询**，不含 t2gbc 的 200M 提示生成模型、不含 GBC→SDXL 的采样路径。做生成侧实验仍要用 `ml-gbc`。

</details>

---

## 4. 选型建议

| 你的目标 | 建议 |
|---|---|
| T2I 训练条件、检索索引、成本敏感 | **WD14 / PixAI tagger** 直接用；不要上 GBC |
| 需要部件/层级/关系/组合的结构信号，做 CLIP 类预训练 | **KohakuGBC**（`mid+sam3` 起步），或直接用 GBC10M 对齐论文结论 |
| 已有 Danbooru tag，想补结构 | KohakuGBC `root = "tags"` + `tagging = "regions"` + `tag_placement = "place"` |
| 两者都要 | `tag` slot 挂 PixAI v1.0（$65.8$ pics/s），图顶点拿文本+坐标、顶点拿 `bagofwords`；再用 tag 与文本的不一致做幻觉筛选 |
| 复现论文的实验数字 | 原版 `ml-gbc` + GBC10M（注意 CC BY-NC） |
| 要在千万图上**重生成可控数据集** | KohakuGBC：$4\times5090$ 约 $14$ 天跑完 GBC10M 规模，且 job 可 record & replay 到 digest |

---

### 参考来源

- `KohakuBlueleaf/KohakuGBC` README、`docs/guides/engines.md`、`docs/internals/program.md`、`src/kohakugbc/nodes/tags.py`
- `apple/ml-gbc` README（默认 Pixtral-12B / LLaVA / GroundingDINO / YOLO-World；许可）
- HF `graph-based-captions/GBC10M` dataset card（图统计、用 LLaVA-1.6-Yi-34B 生成、CC BY-NC 4.0）
- HF `pixai-labs/pixai-tagger-v0.9` model card（13.5k 词表、character F1 0.865、EVA02 冻结头、数据到 2025-01）
- arXiv:2407.06723 摘要（GBC 的定义与 composition/relation 节点的下游增益）
