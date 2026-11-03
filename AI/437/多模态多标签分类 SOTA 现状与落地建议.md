# 多模态多标签分类 SOTA 现状与落地建议

## 🧑‍💻 User

@联网搜索 我有个问题，imaging classification，这个事情，被论证的实是 input image output label上卷积是最好的，还是General？
意思就是，在一个文本图片双模态输入的分类任务下，是否存在一个Global Optimal Recipe？

---
## 🧑‍💻 User

我有个问题，imaging classification，这个事情，被论证的实是 input image output label上卷积是最好的，还是General？
意思就是，在一个文本图片双模态输入的分类任务下，是否存在一个Global Optimal Recipe？

---
## 😀 默认助手 (7)

先说结论，再拆论证。

## 一句话回答

**没有。** “卷积在 image→label 上最优”从来不是一个被证明的定理，它是 ImageNet-1k 那个数据规模下的经验事实被误当成了普遍规律。而一旦进入图文双模态，问题本身就从“找最优点”变成了“找随数据量/算力/标签空间移动的最优点轨迹”，所以不存在那个 Global Optimal Recipe——但存在一个**条件最优的、只有少数几个自由度的 recipe 族**，这个是可以论证的。

---

## 1. 把三个不同的命题分开

你的问题里其实混了三个强度完全不同的说法，它们的真值不一样：

| 命题 | 状态 |
|---|---|
| A. 卷积是 image→label 的**全局最优**算子 | **未被证明，且原则上不可证**（NFL） |
| B. 卷积在**小样本/小算力**下样本效率最高 | **可论证**，且已被反复验证 |
| C. 在大规模数据下，架构差异会**收敛** | **经验上成立**（下面有具体数字） |

### A 为什么不可能被证明

Wolpert & Macready 的 No Free Lunch 直接封死了这条路：对任意两个学习算法，在所有可能的目标函数上取平均，性能完全相同。所以“卷积最优”必须限定在一个分布族上才有意义——而一旦限定，它就退化成 B。

退一步做形式化，能写出来的只有风险分解：

$$R(\mathcal{H}) - R^{\star} \;\le\; \underbrace{\inf_{h\in\mathcal{H}} R(h) - R^{\star}}_{\text{近似误差}} \;+\; \underbrace{O\!\left(\sqrt{\tfrac{\mathrm{Cap}(\mathcal{H})}{N}}\right)}_{\text{估计误差}}$$

卷积通过**权重共享 + 局部性**把假设类的容量 $\mathrm{Cap}(\mathcal{H})$ 压下来，用更大的近似误差换更小的估计误差。它在小 $N$ 下赢，是因为第二项主导；注意力在小 $N$ 下输，是因为容量大、第二项爆炸。**这个交叉点 $N^\star$ 是数据分布的性质，不是架构的性质**——所以“哪个最好”这个问题本身缺少定义域。

真正接近定理级别的东西是**对称性论证**（Geometric Deep Learning 那一套）：若数据分布在群 $G$ 下不变，则 $G$-等变假设类有严格更好的样本复杂度。自然的图像大致具备平移等变性，所以卷积“占便宜”。注意这里的措辞——**卷积不是最优，而是“匹配了一个真实的对称性”**。这句话也正好预告了多模态为什么没有统一答案：文本带来的语义不变性（词序、组合性、开放词表）不属于平移群，没有任何单一算子的对称性假设能同时覆盖两者。

---

## 2. 那些“conv 输给 transformer”的转折，全是 regime 问题

我核对了 arXiv 原文，数字如下（都是论文自己报的）：

| 论文 | 关键结果 | 说明什么 |
|---|---|---|
| ConvNeXt (2201.03545) | 纯卷积 **87.8%** IN-1k top-1，COCO/ADE20K 超 Swin | “conv 不行了”是训练 recipe 问题，不是算子问题 |
| CoAtNet (2106.04803) | 无额外数据 **86.0%** → IN-21k **88.56%** → JFT-3B **90.88%** | 原文明确写：Transformer 缺“right inductive bias”所以泛化更差；**最优解随数据规模上移** |
| Scaling ViT (2106.04560) | 2B 参数 **90.45%** | 纯 attention 在大数据下上限更高 |
| MetaFormer / PoolFormer (2111.11418) | token mixer 换成 identity，仍有 **74.3%** | 主导因素是 **macro-architecture + recipe**，token mixer 是次要项 |
| Early Convolutions (2106.14881) | ViT 的 patchify stem 换成 4 层 3×3 conv，**+1~2%** 且优化稳定性大增 | “卷积”以 stem 形式至今仍是最优选择之一 |

把这五条放在一起看，结论很干净：**不是一个算子在所有尺度上赢，而是一条“归纳偏置强度”的连续轴在随 $N$ 移动**。ConvNeXt 是把卷积推到“现代 recipe 下的上限”，CoAtNet 是把这条轴中间的混合点做扎实，Scaling ViT 是轴的右端。它们互相不否定，只是各自在曲线的不同位置取样。

---

## 3. 图文双模态：问题变形了，不是加了个输入

这是你问题的真正核心。加了文本分支之后，“input image → output label”这个形式化已经不成立了，具体变了四件事：

| 文本/双模态带来的变化 | 对“最优架构”的影响 |
|---|---|
| 输出侧从 `softmax over K` 变成**联合嵌入空间里的对齐/检索** | 评价指标从 accuracy 变成 zero-shot transfer、retrieval R@1——**排序会变** |
| 类别名/描述文本提供**额外的语义先验** | 等效于把有效样本量 $N$ 放大，B 命题里卷积的边际收益被稀释 → 更偏向大容量 backbone |
| **融合模块**（early/late/cross-attention/冻结 backbone）进入搜索空间 | 融合方式造成的方差往往**大于** backbone 造成的方差，backbone 之争被降权为二阶问题 |
| 目标变成开放词表 / 分布外泛化 | 需要更强的形状偏置与语义组合性，而不是局部纹理统计 → ViT 类相对占优 |

经验证据也很一致：

- **CLIP** 同时用 ResNet 和 ViT 做 image encoder，两者都能work；后续 open_clip / LAION 也用 ConvNeXt backbone 训出可比的 CLIP 模型——说明在对比对齐目标下，backbone 的可替换性相当高。
- **SigLIP** (2303.15343)：只把 softmax 对比损失换成 pairwise sigmoid，就做到 **84.5% zero-shot，4 块 TPUv4、两天**，且 batch 32k 就够（不需要百万级）。**loss 设计是一个和架构同量级的独立杠杆。**
- **DataComp** (2304.14108)：固定 CLIP 训练代码与算力，**只改数据筛选**，ViT-L/14 从零训练到 **79.2% zero-shot，比 OpenAI CLIP 高 3.7pt**。这是对“架构存在全局最优”最直接的反驳——在固定算力预算下，**数据 curation 的 effect size 大于架构选择**。

> 所以对双模态分类，“conv 还是 general”这个二选一是个**被错误维度化的提问**。正确的维度是：在你能拿到的数据规模与算力下，该分配多少“先验”给架构、多少留给数据与对齐目标。

---

## 4. 那“什么才是真的可以论证的”

有一个可操作的替代命题，我认为它才是你想找的东西：

**不存在全局最优 recipe；存在一个由 4 个坐标张成的条件最优曲面：**

$$
\text{recipe}^{\star} = f\big(\,N_{\text{eff}},\; C_{\text{compute}},\; \mathcal{S}_{\text{label}},\; \mathcal{M}_{\text{metric}}\,\big)
$$

- $N_{\text{eff}}$：**有效**样本量（含预训练、含文本先验带来的等效放大）
- $C_{\text{compute}}$：训练+推理预算（注意 *The Efficiency Misnomer*, 2110.12894 指出的：FLOPs 和 latency 会互相矛盾，单指标会给出错误结论）
- $\mathcal{S}_{\text{label}}$：闭集 vs 开放词表
- $\mathcal{M}_{\text{metric}}$：in-distribution acc / OOD / zero-shot / 校准

这四项一固定，最优架构基本就被钉住了；这四项一变，答案就翻转。**Global optimal recipe 之所以不存在，是因为这四个坐标里的任何一个都不是架构的内禀属性。**

<details>
<summary><b>如果你想为自己的任务真的找到那个点：一个 5 步协议（可展开）</b></summary>

1. **先冻结评价协议**：zero-shot / linear-probe / full-FT 必须分开报，混报会让结论完全反过来。
2. **先扫数据与 loss，后扫 backbone**：按 DataComp 与 SigLIP 的 effect size 排序，数据筛选和对齐损失的收益高于换 backbone。顺序反了会浪费大量算力。
3. **把架构当成一维连续量扫，而不是离散名字**：`纯 conv (ConvNeXt) → conv stem + attention (CoAtNet 式) → 纯 ViT → pooling/MLP mixer`，这条轴上同一位置的模型差异远小于跨位置的差异，能直接定位你的 $N^\star$。
4. **报告多维效率指标**：latency、throughput、峰值显存、FLOPs 至少报三个，否则比较无意义。
5. **承认不可识别性**：如果你的规模偏小，很可能多个 recipe 在置信区间内不可区分——这时候正确答案是**选最便宜、最稳的那个**，而不是继续调。这不是妥协，这是统计上正确的决策。

</details>

---

## 5. 一个诚实的补充

有人会说“可 NAS 已经被证明找不到全局最优了”——确实，架构搜索文献反复观察到**同一批架构在不同数据集上的排名相关性很弱**，这就是“全局最优”在经验层面也不存在的证据，比 NFL 那种纯理论的反驳更有说服力。再加一条：任何声称“A 优于 B”的实验，只要没固定 $(\text{参数预算}, \text{训练算力}, \text{数据}, \text{增广})$ 这四项，结论都不可迁移。

**所以，回答你的原话：**

> imaging classification 上“卷积最好”**没有被论证过**，它只是一个在 $N$ 较小时成立、且随 $N$ 增大而失效的经验陈述；而图文双模态下，因为你换了标签空间、换了评价指标、又多了一个大 effect size 的融合/对齐设计维度，**Global Optimal Recipe 在原理上不存在**——但一个只有 4 个自由度的条件最优族是存在的，而且是可以被你实验定位的。

要不要我按你的具体场景（数据集规模、是否允许预训练、推理延迟约束、闭集还是开放词表）把这四个坐标代入，给一个更具体的候选 recipe 排序？

---
## 🧑‍💻 User

多模态多标签分类或者多模态多标签分类
现在general在各大视觉数据集上SOTA的Recipe/Method是什么？
再帮我调研下

---
## 😀 默认助手 (7)

# 多模态 / 多标签视觉分类：2025–2026 的 SOTA 现状调研

> 说明：我检索了 arXiv 上 2024 年末至 **2026 年 10 月**的最新文献（你的“现在”我按最新投稿时间对齐），下面所有**加粗数字都来自论文摘要或正文自述**，未核实的我会明确标注。

---

## TL;DR — 先给结论

**没有任何一篇论文被公认为“通用 SOTA recipe”。但已经形成了一个极其稳定的 de facto 事实标准**，而且它在 VOC / COCO / NUS-WIDE / CXR 系数据集上几乎通用：

$$
\text{SOTA} \approx \underbrace{\text{预训练 VLM/ViT backbone}}_{\text{可替换，}\Delta \text{ 小}} + \underbrace{\text{query-based label decoder}}_{\text{ML-Decoder 系}} + \underbrace{\text{ASL/DB-Focal}}_{\text{必选}} + \underbrace{\text{label-correlation 模块}}_{\text{GCN / cross-attn}} + \underbrace{\text{TTA + EMA + 阈值后处理}}_{\text{涨点最猛}}
$$

**最关键的一条实证**：最近的标准 benchmark 第一名（VOC **97.4**、COCO **92.8**）来自 *MAT-Agent*（NeurIPS 2025），它**没有改任何架构**——它用多智能体 bandit 去自动调**数据增广、优化器、学习率、loss 组合**。这基本坐实了上一轮我们讨论的结论：**当前多标签任务的 SOTA 增量，主要来自 recipe 调优与后处理，而非 backbone 选择。**

---

## 1. 标准基准的现状（含核对过的数字）

| 基准 | 规模 | 当前报告 SOTA | 出处 | 状态 |
|---|---|---|---|---|
| **PASCAL VOC 2007** | 20 类 / 9.9k | **97.4 mAP**（前 SOTA PAT-T 96.2） | MAT-Agent, [2510.17845](https://arxiv.org/abs/2510.17845) | ⚠️ **已饱和**，不再有区分度 |
| **MS-COCO** | 80 类 / 122k | **92.8 mAP**（前 SOTA HSQ-CvN 92.0） | MAT-Agent, 同上 | 主流主战场 |
| **MS-COCO（head 对比基线）** | 80 类 | **91.4 mAP** | ML-Decoder, [2111.12933](https://arxiv.org/abs/2111.12933) | 头部设计的事实基线 |
| **NUS-WIDE（零样本）** | 81 概念 | **31.1 ZSL mAP** | ML-Decoder, 同上 | 零样本难度仍很高 |
| **Visual Genome 256** | 256 类 | **60.9 mAP** | MAT-Agent | 长尾/多标签更真实 |
| **VOC-LT / COCO-LT** | 长尾版 | SOTA（具体值未核） | CAPNET, [2511.20641](https://arxiv.org/abs/2511.20641) | 长尾是当前热点 |
| **Open Images V6** | 9.6k 类 | ❌ 无统一 leaderboard | — | **这是空白**，别信任何“SOTA”宣称 |
| **MIMIC-CXR / CheXpert** | 14 类 | mean AUROC **0.840**, mAP 0.467（内部） | [2609.09185](https://arxiv.org/abs/2609.09185) | 医学侧自建口径 |

> **必读警告**：上表数字**不可直接横向比较**。ML-Decoder 的 91.4 与 MAT-Agent 的 92.8 用了不同的 backbone 规模、是否外部数据、是否 TTA/ensemble。MLIC 领域至今**没有强制报告协议**（这点在 2026 年的综述 [2607.00839](https://arxiv.org/abs/2607.00839) 里被明确点出）。看到“SOTA”先问四件事：backbone? 外部数据? TTA? ensemble?

---

## 2. 把 recipe 拆成六层，看每层的真实边际收益

这是我调研后认为最有用的视角——**按 effect size 从大到小排**：

| 层 | 当前默认选择 | 边际收益 | 依据 |
|---|---|---|---|
| **① 训练配置自动化** | bandit/agent 调 aug+optimizer+LR+loss | **最大**（VOC +1.2, COCO +0.8） | MAT-Agent |
| **② 后处理** | TTA（多尺度+flip）+ EMA + 逐类阈值搜索 | **极大**，但常被隐藏 | CAPNET、MAT-Agent 均含 TTA |
| **③ Loss** | **ASL**（非对称 loss）或 **DB-Focal**（分布平衡） | 大，尤其长尾 | CAPNET、CPRFL 均采 ASL |
| **④ 标签相关性建模** | label query + cross-attention，或 co-occurrence **GCN** | 中–大 | CAPNET（GCN）、HyperLabel（超图）、ML-Decoder（group decoding） |
| **⑤ 头部设计** | **query-based decoder**（ML-Decoder 式），group decoding 支持千类 | 中（大类别数时变大） | ML-Decoder：同算力下优于放大 backbone |
| **⑥ Backbone** | ConvNeXt / Swin / ViT，常以 CLIP 或 DINOv2 初始化 | **最小**（在①–⑤都做对之后） | 多个 2026 工作直接写 “using a common ML-Decoder head”，backbone 可换 |

关键旁证：Med-AR（[2609.29156](https://arxiv.org/abs/2609.29156)）在比较 6 种预训练 encoder 时，明确写**“using a common ML-Decoder classification head”**——说明 ML-Decoder 已经成为跨领域比较时的**中立标准件**。这跟当年 ResNet 的角色一样：一旦某个组件变成“公共分母”，说明它已经收敛了。

---

## 3. 多模态（图文）多标签：三条技术路线

这里要先区分你说的“双模态输入”到底是哪种，因为**文献里是三条完全不同的线**：

### 路线 A：文本只提供**标签语义**（label-as-text，最主流）

不把文本当输入样本，而是把类名/描述当 prompt 或 query 初始化。

- **ML-Decoder 的 word query**：直接用词向量当 query，可泛化到未见类（NUS-WIDE ZSL **31.1 mAP**）
- **CAPNET**（[2511.20641](https://arxiv.org/abs/2511.20641)）：CLIP text encoder → GCN 传播标签相关性 → 可学习 soft prompt → DB-Focal → PEFT 微调 → TTA。面向 VOC-LT / COCO-LT
- **LDSA**（[2608.22313](https://arxiv.org/abs/2608.22313)）：CLIP + 密集对比 adaptor + 类 prompt tuning，处理**部分标注**，自称 SOTA
- **HP-DVAL**（[2509.17747](https://arxiv.org/abs/2509.17747), TIP）：双视角对齐 + 层次 prompt，COCO/VOC 长尾 **+10.0 / +5.2 mAP**

> 这条线的共识：**文本的价值是“语义先验”，不是“额外模态”**。用 text encoder 给 label 一个良好初始化，比多一路视觉输入更划算。

### 路线 B：**训练-free / 少样本**的零样本多标签（VLM 当 backbone）

- **SPARC**（[2502.16911](https://arxiv.org/abs/2502.16911)）：**完全不训练**，把 CLIP 当黑盒，用 LLM 生成 object co-occurrence 的**复合 prompt**（"a photo of A and B"），再做去偏 + 分数融合。发现一个反直觉现象：**复合 prompt 的最高分反而不如第二高分**——"AND/OR" 语义在对比模型里是混淆的。这是很实用的 insight
- **SugaFormer**（[2501.05728](https://arxiv.org/abs/2501.05728), AAAI'25）：super-class 初始化 query 降低 query 数 → 千级属性仍可扩展；已见/未见属性知识迁移

### 路线 C：**MLLM 生成式 / 当标注器**（2026 的新主线）

- **TagLLM**（[2602.20972](https://arxiv.org/abs/2602.20972)）：用 MLLM 直接产多标签。结论很有信息量——**标注成本降到人工的 1/1000，质量只有人工的 50–80%，但下游训练任务能保住 90%+ 性能**。也就是说 MLLM 当标注器**可行**，当分类器**还不行**
- **MultiEmo-Bench**（[2605.14635](https://arxiv.org/abs/2605.14635)）：多标签视觉情感，评了 Qwen3-VL / GPT / Gemini / Claude，结论是"有进步但差距仍大"，且 **LLM-as-a-judge 对主观多标签任务不靠谱**
- **ATPO**（[2610.02019](https://arxiv.org/abs/2610.02019)）：把多标签视频安全检测做成 RL，用 **Adaptive Tversky Reward** 控制 precision-recall 工作点，Jaccard **40.66 → 75.44**
- **FLARE 2026 / 医学 VM 线**（[2610.19901](https://arxiv.org/abs/2610.19901)、[2609.23139](https://arxiv.org/abs/2609.23139)）：Qwen3.5-9B / MedGemma + **任务特定 LoRA + 结构化输出**，把多标签压成 next-token 生成

> **路线 C 的现状判断**：MLLM 目前在多标签上是"**标注器 / 辅助推理器**"角色，还不是"**分类器**"。因为多标签的本质是 **$K$ 个独立 sigmoid 的边缘概率**，而 MLLM 的自回归生成天然是序列联合分布，两者不匹配。想用生成式做多标签，必须显式处理这个 mismatch（ATPO 的 RL 奖励、CARVE 的 cardinality 估计都是在补这个洞）。

---

<details>
<summary><b>展开：2025–2026 还有哪些独立子问题被单独立项了（说明主战场已迁移）</b></summary>

主 benchmark 上 VOC 已饱和、COCO 逼近上限，所以**研究热度已转移到下面这些"脏数据"场景**——这本身就是一个强烈信号：方法论的天花板来自数据，不来自架构。

| 子问题 | 2025–26 代表工作 |
|---|---|
| 部分标注 / 单正例 | PML-CD（[2610.17635](https://arxiv.org/abs/2610.17635), AAAI'25）、双曲球分类（[2510.15296](https://arxiv.org/abs/2510.15296)） |
| 多标签类增量 (MLCIL) | KBK（[2608.30316](https://arxiv.org/abs/2608.30316)，MS-COCO B0-C10 **+2.7%**）、P2L-CA（[2601.12714](https://arxiv.org/abs/2601.12714)） |
| 长尾 | CAPNET、HP-DVAL、CPRFL |
| 噪声标签 / webly | DBMLCL（[2607.20874](https://arxiv.org/abs/2607.20874)）、MHSS（Sewer-ML 线） |
| 标注质量本身 | **CVPR'26：把 ImageNet 训练集自动重标成多标签**（[2603.05729](https://arxiv.org/abs/2603.05729)），COCO **+4.2 mAP**、VOC **+2.3 mAP**、ReaL **+2.0 top-1** |
| 大类别数 / 极端多标签 | ML-Decoder（千类）、HyperLabel 超图（[2609.32276](https://arxiv.org/abs/2609.32276)） |
| 校准 | 多标签 label-wise ECE 的新分箱方案（[2609.26468](https://arxiv.org/abs/2609.26468)） |
| 层次几何 | Hyper³-CLIP 双曲空间 entailment（[2608.29313](https://arxiv.org/abs/2608.29313), ECCV'26 W） |

**其中最具说服力的一条是 CVPR'26 那个 ImageNet 多标签重标**：不动模型，只把标注从单标签改成多标签，跨架构一致涨点，迁移到 COCO **+4.2 mAP**。这比任何架构论文都更直接地证明——**当前瓶颈在监督信号，不在归纳偏置**。

</details>

---

## 4. 所以“general recipe”到底怎么定？给你的落地判断

| 你的场景 | 推荐 recipe |
|---|---|
| 类数 ≤ 100，数据充足，闭集 | ConvNeXt/ViT + ML-Decoder 头 + ASL + 多尺度 TTA。**别再比 backbone 了**，把时间花在调 aug/LR/阈值上 |
| 类数 ≥ 1000（Open Images 式） | 必须用 group decoding 的 query 头（ML-Decoder 系），否则 FC 层参数量和过拟合都失控 |
| 长尾 | CLIP 初始化 + GCN 标签相关性 + DB-Focal + PEFT + TTA（CAPNET 组合） |
| 标注不全 / 只有单正例 | 别硬训。走 CLIP + prompt/adaptor（LDSA 路线），或先用 MLLM 补标（TagLLM 证明下游能保住 90%+） |
| 要开放词表 / 零样本 | 训练-free 的 SPARC 式复合 prompt 去偏融合，是当前性价比最高的 baseline |
| 有真实文本输入（标题/描述/报告） | **目前没有公认 benchmark**，见下方第 5 条 |

---

## 5. 必须说清的三件事（避免你踩坑）

1. **“多模态多标签分类”在公开文献里没有统一 benchmark。** 真实的多模态输入（商品图 + 标题、病历图 + 报告）在中英文论文里都散落在各自业务数据集上，没有 VOC/COCO 那样的公共分母。**如果你要做这件事，第一件事是自建一个能对比的评测协议**，否则你永远无法回答"我的方法是否 SOTA"。

2. **VOC 2007 的 mAP 已经失去区分度。** 97.4 距离上限太近，1 个点的差异往往来自 TTA 或随机种子。要发论文/做决策，用 COCO + VG-256 + 一个长尾版。

3. **mAP 和 F1 是两个不同的决策问题。** mAP 不需要阈值，F1 需要**逐类阈值**。很多"方法不行"其实只是阈值没调（这也是为什么 MAT-Agent 这类调参框架能刷出 SOTA）。生产部署一定要单独做 per-class threshold calibration。

---

## 一句话收尾

> 2026 年的答案不是"哪个方法 SOTA"，而是：**架构层已经收敛到“预训练 ViT/ConvNeXt + query label decoder + ASL”，真正的 SOTA 差异全部产生在 recipe 层（自动化调参、TTA、标签相关性、标注质量）**。而多模态这一侧，文本目前的最佳角色是**给标签提供语义先验和当标注器**，而不是当第二路输入——最后这一点，恰好又回到了上一轮的结论：**决定上限的是数据与监督信号，不是归纳偏置。**

要不要我针对你的具体场景（类数、是否有文本输入、是否长尾、标注完整度）直接把 recipe 定到可执行的配置级别（含 loss 权重、TTA 策略、阈值方案）？给我这四个参数就行。

---
## 🧑‍💻 User

VLM的话（Qwen3.8 27B 或者Gemma系列），那我感觉得训。但得很大力了。但从先验的角度来说，只做分类任务，真的要上VLM吗
图像标注labeling都有了 只要分类

---
## 😀 默认助手 (7)

## 直接回答：**不要。你这个场景不需要 VLM。**

你的直觉是对的。我们逐项核对一下成本收益。

---

## 1. 先确认你付费换来了什么

一个 27B VLM 表征的是：

$$p(\text{text} \mid \text{image}, \text{prompt})$$

你要的是：

$$p(y \mid x), \quad y \in \{0,1\}^K,\ K \text{ 固定，标注齐备}$$

**当标签集合封闭、标注完整时，文本空间对你来说是冗余自由度。** 更关键的是，如果你推理时 prompt 是固定模板，那么

$$f(x, t_{\text{const}}) \equiv g(x)$$

即，你用一个 27B 的联合分解去参数化一个只需要 $g(x)$ 的函数，相当于把 99% 的参数花在了永远不变化的维度上。

**“从先验角度”这句话本身成立，但你把两类先验混淆了：**

| 先验类型 | 你是否需要 | 获取成本 |
|---|---|---|
| **视觉先验**（objectness、部件、材质、纹理） | ✅ 极需要 | DINOv2 / SigLIP / ConvNeXt 权重，**90M–1B 参数，免费下载** |
| **标签语义先验**（“猫”和“狗”在语义空间近） | ⚠️ 有用 | **用冻结 text encoder 初始化分类头权重即可，一行代码** |
| **语言推理先验**（理解一段 policy 文本再判断） | ❌ 你没有这个需求 | 这才是 27B LLM 唯一不可替代的东西 |

第三行才是 27B 的定价理由，而你不需要它。

---

## 2. 目标函数不匹配：这是比成本更致命的问题

| | 多标签分类 | 生成式 VLM |
|---|---|---|
| 输出分布 | $K$ 个**独立 Bernoulli** | 文本序列的**自回归联合分布** |
| Loss | sigmoid + ASL/BCE，**直接对齐 mAP/F1** | next-token CE，对齐的是“像不像人话” |
| 推理 | **一次前向**，O(1) | 自回归解码 $K$ 个 token，**延迟 O(K)** |

你已经有了 label，不需要“用自然语言把标签说出来”这一层中介。中介的代价是实在的：

- 对标签**顺序和格式敏感**（同一组标签，不同书写顺序 → 不同 loss）
- 长尾类上，概率质量被浪费在**词形**上而不是**判别**上
- 多标签的阈值校准（per-class threshold）在生成式模型上几乎无法做，因为你没有干净的 $K$ 维 logit 向量

---

## 3. 成本差 2–3 个数量级（具体数字）

| | 27B VLM 全参微调 | ConvNeXt-L / ViT-L 全参 |
|---|---|---|
| 优化器状态（Adam fp32 m,v） | **≈ 216 GB**（仅此一项） | ≈ 1.6 GB |
| 最低硬件 | 8×H100-80G 起 + 梯度检查点 | **单机 4–8×A6000/A100** |
| 迭代速度 | 天–周 | **小时** |
| 推理吞吐（每图） | 生成 $K$ token，慢 | 单次前向，快 **100×+** |

而且这不是“贵一点”，而是**贵 100 倍还不一定更准**，见下一节。

---

## 4. 实证：闭集分类上，专项 encoder 更强

我核对了 2025–26 的几篇横向对比，结论高度一致：

- **[2608.29802]** 系统性比了 **16 个 open-weight MLLM + 30 个 vision encoder**，五种范式（MLLM zero-shot prompting / 输出 logit 上训浅层模型 / PEFT 微调 MLLM / 冻结 encoder + linear probe / 微调 encoder）。结论：**微调 encoder 拿到跨数据集 SOTA**；MLLM 的独有价值是附带 textual reasoning，不是 accuracy。
- **[2610.37510]** 横评 LP / FFT / LoRA / Visual Prompting，跨 Attention / Conv / Mamba 三族：**LP 是极强 baseline，FFT 的增量不稳定**。
- **[2610.21809, WACV'27]** 医学跨域：**“加深 adaptation depth 并不能改善跨人群泛化”**——训得更狠 ≠ 更好。
- **EMG-FM-Bench**：预训练收益 “**not universal**”，取决于模型与任务匹配。

另外补一条上一轮提到的：**CVPR'26 把 ImageNet 重标成多标签，不动模型，COCO +4.2 mAP**。结论很明确——**当前瓶颈在监督信号与决策头，不在模型容量。**

> 顺带说一个反直觉点：**27B 全参微调在 $K$ 类小数据上是负收益风险**。容量暴涨 + 归纳偏置缺失 → 泛化变差（CoAtNet 的原话），而多标签最怕的是**校准崩坏**，mAP 可能还凑合，但 per-class threshold 完全失效。你花大力气换来的可能是一个更差的决策器。

---

## 5. 你要的“文本先验”，有几乎零成本的实现

这是最关键的一条。你想要的文本先验，**不需要 27B，一个冻结的 text encoder 就够了**：

```text
分类头权重 W ∈ R^{K×d}
初始化:  W[k] ← FrozenTextEncoder( "a photo of a {class_k}" )
之后:    正常训 sigmoid / ASL
```

这就是 **ML-Decoder 的 word query** 方案（COCO 91.4 mAP，NUS-WIDE 零样本 31.1 mAP）。它同时给你两个好处：

1. **语义先验**：类间相似度天然注入（“哈士奇”与“狼”的权重接近）
2. **零样本能力**：新类来了不用重训，写下类名就有初值

**用 300M 的 CLIP text encoder 三行代码实现的东西，没必要用 27B 去做。**

---

## 6. 决策判据：什么时候 VLM 才真的对

只有命中下面**任意一条**才值得：

- ☐ 标签集合会变，或需要 zero-shot（新类没有标注）
- ☐ 文本是**逐样本的自由文本**，且需要推理（policy 段落、病历、用户 query）
- ☐ 每类样本极少（≲50）且必须靠 prior transfer
- ☐ 需要输出 **rationale / 解释**，而不仅是标签
- ☐ 要一个模型统一覆盖多任务接口（分类+检测+VQA+报告）

**你描述的情况（标注齐备、只要分类、闭集）不命中任何一条 → 不上。**

---

## 7. 该走的阶梯（成本递增，别跳级）

| 级 | 做法 | 成本 | 预期 |
|---|---|---|---|
| **L0** | 冻结 SigLIP/DINOv2 image tower + **ML-Decoder 头 + ASL** + 文本初始化权重 | **半天** | 先把这个打了，很多项目在这里就够 |
| **L1** | + 末层 block 微调 / LoRA | 1–2 天 | +1~3 mAP |
| **L2** | + 标签相关性（GCN / label cross-attn）+ TTA + **per-class 阈值校准** | 2–3 天 | +1~3 mAP，且 F1 大幅改善 |
| **L3** | **蒸馏**：用 MLLM 产软标签/描述扩充训练，训小模型 | 1 周 | TagLLM 证据：MLLM 标注质量只到人工 50–80%，但下游保住 **90%+** |
| **L4** | 才轮到 VLM 微调 | 大力 | 只有 L0–L3 全撞墙且**诊断确认是表征瓶颈**时才做 |

### 先做这个诊断（10 分钟决定要不要花钱）

比较同一 encoder 下 **frozen linear probe** 与 **full FT** 的 mAP 差：

$$
\Delta = \text{mAP}_{\text{full-FT}} - \text{mAP}_{\text{frozen}}
$$

| $\Delta$ | 瓶颈定位 | 该干什么 |
|---|---|---|
| **< 2** | 决策/损失/后处理 | 调 head、loss、阈值。**别碰模型** |
| **2–5** | 表征略欠 | 换更强 encoder（SigLIP2 / DINOv2-g / 更大 ViT），仍然不是 27B |
| **> 5** | 表征明显不足 | 先查分辨率与增广，再考虑更大视觉 backbone |
| 两者都低，但训练 loss 能压到底 | **标注噪声 / 标签定义** | 去修数据（这正是 CVPR'26 那条路的收益） |

---

## 8. 如果文本确实是逐样本输入（这点要跟你确认）

你说“图像标注都有了，只要分类”，听起来文本不是逐样本输入。但如果是（比如商品标题、报告），**正确形态是双塔 + 融合头，不是生成式 VLM**：

```text
img_enc(x) ─┐
            ├─→ fusion (concat / cross-attn / 门控) ─→ K 维 sigmoid
txt_enc(t) ─┘
```

原因很简单：**你要的是对齐与融合，不是生成。** 双塔方案总参数 ~300–600M，单卡可训，且 loss 与任务同构；27B 生成式模型在“图文 → 多标签”上没有任何结构优势，只是把交叉熵换了个外壳。

---

## 一句话总结

> **你没有标注缺失的问题，就没有 VLM 的用武之地。** 27B 的价格买的是“语言侧推理 + 开放词表 + 可解释输出”，而这三样你一个都不需要。你需要的是**视觉先验**（300M encoder 免费给）**+ 标签语义先验**（text encoder 初始化权重，三行代码）**+ 对齐的损失与阈值**。把省下的算力投到 L2/L3（标签相关性、TTA、蒸馏、数据质量）上，回报率高一个数量级。

**先做那个 $\Delta$ 诊断**——如果 frozen linear probe 和 full FT 差不到 2 个点，你连 L1 都不用做，问题根本不在模型上。

---
## 🧑‍💻 User

| 候选 | 本项目采用的具体配置 | 主要回答的问题 |
| --- | --- | --- |
| ResNeXt | ResNeXt-50 32×4d，MoA 监督训练 | 常规卷积监督基线能够达到什么表现？ |
| ConvNeXt V2 | Tiny，同架构分别进行监督训练和 FCMAE 预训练 | 掩码预训练对卷积编码器贡献多少？ |
| MAE / CA-MAE | ViT-S/16，比较固定通道与通道无关编码 | 像素重建目标及通道处理方式贡献多少？ |
| DINOv2 | ViT-S/14，进行多通道适配和领域内自监督训练 | 图像级与 patch 级自蒸馏是否改善形态表征？ |
| DINOv3 | ViT-S/16，使用发布权重进行迁移 | 大规模通用视觉预训练能够提供多少迁移收益？ |
| Cell-DINO | 获得适用许可后加入领域预训练参考 | 已有细胞图像预训练模型的表现如何？ |

暂时先做这些，再把EmbedGemma2加上个，sam3/sam3.1这些的vision part还有SigLIP/CLIP系列是不是也可以放

---
## 😀 默认助手 (7)

## 先给出可判定的标准

你这张表会越来越长，但**信息密度不一定增加**。建议每条候选必须满足两个条件才保留：

1. **它提供一个正交轴**（不是另一个“模型点”，而是一个未被覆盖的因子）
2. **它能被同一条 probe 协议归因**（否则结果无法解释）

按这两条检验，现有 6 行里其实只有 **1 行是明确对照实验**：

| 行 | 变的东西 | 是否明确 |
|---|---|---|
| ResNeXt-50 vs ConvNeXt V2-T(sup) | 架构 **+** 训练配方（跨 8 年） | ❌ 只得“现代 vs 老”，非“卷积好坏” |
| ConvNeXt V2 sup vs FCMAE | **仅** 预训练目标 | ✅ **明确** |
| MAE vs CA-MAE | 重建目标 **+** 通道处理 | ❌ 两变量同时变 |
| DINOv2 | 多通道适配 **+** 域内自监督 | ❌ 两变量同时变 |
| DINOv3 (repo) | 仅权重来源 | ✅ 明确 |
| Cell-DINO | 仅数据域 | ✅ 明确 |

**建议**：把实验组织成“**固定 probe 加逐层扫描**”，架构族 × 预训练目标作因子，适配深度和通道/分辨率作横向因子。现在这种“一行一个模型”的逻辑，会让 DINOv2 那行即使赢了，也答不出是“通道适配”还是“SSL”起作用。

---

## 三个提议逐个看

### (a) EmbeddingGemma：**放错表了，但位置很重要**

先更正一个细节：`EmbeddingGemma` 不是 Gemma 2 系，它是基于 **Gemma 3** 的 **纯文本** embedding 模型（~300M，Matryoshka 768→128）。**没有 vision tower，不能作为 backbone 行。**

但它在你项目里有正当位置——就是上一轮说的“标签语义先验”。应该**单开第二张表：标签嵌入源**：

| 候选 | 回答什么 |
|---|---|
| EmbeddingGemma | 通用文本语义能否表达形态学术语 |
| SigLIP2 text tower | 与 image tower **同空间**（唯一支持零样本的选项） |
| 生物医学文本编码器（BioLORD / PubMedBERT 类） | 领域文本先验的增益 |
| LLM 生成的类名描述再嵌入 | 描述扩充（“vacuolization”→“cytoplasm with multiple clear vacuoles”）的增益 |

**分界线要划清**：

| 用途 | 任意文本空间可用？ |
|---|---|
| **初始化分类头权重** $W_k \leftarrow \phi(\text{class}_k)$ | ✅ 可以。$W$ 后续会被训练，只需保住类名间的**相似度结构** |
| **零样本推理**（图文直接比余弦） | ❌ 不行。必须与 image tower 共享空间。DINOv3 和 EmbeddingGemma 不共享空间 |

> 所以：**EmbeddingGemma 能当“初始化器”，不能当“对齐器”。** 别把它和 DINOv3 放在一起做零样本实验，那个组合从定义上就是坏的。

---

### (b) SAM3 / SAM3.1 的 vision part：**可以加，但降级为探索行**

事实核对（我查了原文）：

- **SAM 3**（[2511.16719](https://arxiv.org/abs/2511.16719)）是 *detector + tracker* **共享一个 backbone**，用 concept prompt（名词短语/图像 exemplar）；配套 4M 概念标签 + SA-Co benchmark
- 它的骨干来自 **Perception Encoder (PE)** 家族（[2504.13181](https://arxiv.org/abs/2504.13181)）：**纯对比式图文学习**，zero-shot ImageNet robustness 86.6、COCO 66.0 box mAP

它回答的独特问题是：**“大规模检测/分割 + 概念级语言对齐” 这一训练目标能否迁移到形态学分类**。这是你现有表里**唯一缺的 dense-prediction 监督轴**（MAE=重建，DINO=自蒸馏，CLIP/SigLIP=图文对齐，SAM=mask+检测）。从正交性讲，它确实**值一个槽位**。

但有三个坑必须知道：

1. **PE 论文标题就是警告**：*"**The best visual embeddings are not at the output of the network**"* —— 它的可用表征藏在**中间层**。你若照常规取最后一层 pooled feature，会系统性低估它。这一条对所有 encoder 都成立，见后面。
2. **域匹配存疑**：SAM 系是对“**物体实例**”的分割先验。细胞/亚细胞结构是否构成“object”本身就是问题——荧光细胞粘连、无清晰边界，split 出来的是伪实例。我预期它在你的形态学任务上**不占优**，但这个负结果本身有价值。
3. **成本与许可**：PE/SAM3 骨干是大 ViT + 高输入分辨率（>1000px）+ 检测头，推理很重；且 SAM3 用**自定义 SAM License**（非 Apache），和 Cell-DINO 一样要走一遍许可。

**建议**：如果只是想验证“mask 监督有没有用”，用 **SAM2 的 ViT-B**（Apache 2.0、轻、MAE 初始化）作廉价代理，成本低一个数量级；SAM3 放探索清单最后，等主表收敛后再动。

---

### (c) SigLIP / CLIP 系列：**应该加，而且优先级很高**

这条我支持，但要选得精确，别写成“CLIP 系列”这种模糊表述。

**理由**：它对应一个明确且高价值的轴——**语言对齐预训练**。你的标签天然是形态学术语（语言），所以“语义→视觉”的桥上有理论优势。而且它能**顺带解决 (a)**：用同族 text tower 做标签嵌入，初始化 + 零样本一次拿到，不用在两套空间之间折腾。

**选型建议**：

| 选 | 理由 |
|---|---|
| **SigLIP 2**（优先，尤其 NaFlex 变体） | 迁移性普遍优于 CLIP；**NaFlex 支持原生分辨率**——对形态学这是决定性的 |
| CLIP (OpenAI ViT-B/L) | 只在需要“最经典对照”时留一个 |
| ~~两者都上~~ | ❌ 会挤掉更宝贵的槽位，且信息高度重叠 |

**关键点**：对细胞形态，**输入分辨率的影响往往大于 backbone 选择**。CLIP 系固定 224/patch14 是硬伤；SigLIP2-NaFlex 能吃任意分辨率，这个差别可能比你换 5 个 backbone 都大。

---

## 重排后的建议表

### T1 表征来源（主表，≤8 行）

| # | 候选 | 轴 | 优先级 |
|---|---|---|---|
| 1 | ResNeXt-50 32×4d（监督） | 老 conv 配方锚点 | P2 |
| 2 | ConvNeXt V2-T（监督） | 现代 conv 监督锚点 | P1 |
| 3 | ConvNeXt V2-T（FCMAE） | **掩码预训练**（干净对照） | **P0** |
| 4 | ViT-S/16（MAE） | 与 3 构成 conv×ViT 的 2×2 一角 | **P0** |
| 5 | DINOv2 ViT-S/14（**仅用发布权重**） | 通用自蒸馏 | P1 |
| 6 | **DINOv3 ViT-S/16** + **DINOv3 ConvNeXt-T** | 通用大规模 SSL；后者给“同 SSL 配方 conv vs ViT”的干净对照 | **P0** |
| 7 | **SigLIP 2**（NaFlex 或 ViT-S/B） | **语言对齐** | **P0** |
| 8 | Cell-DINO | 域内预训练天花板 | **P0** |
| 9 | (探索) SAM2 ViT-B / SAM3-PE | dense 监督 | P3 |
| — | EmbeddingGemma | → **不进主表**，进 T2' | — |

> 注意 #4：要补齐 2×2 的另外两角（ViT 监督、ConvNeXt 自蒸馏），否则“conv vs ViT”这个结论立不住。现在只有一角。

### T2 适配协议（横向因子，不是每行单列）

`frozen linear probe` → `末层 block FT / LoRA` → `full FT`，**所有 backbone 用同一套超参搜索预算**。

### T3 输入处理（横向因子）

通道（RGB 三通道复制 / 通道无关编码 / 原生多通道）、分辨率（224 / 512 / 原生）、切块（全局 / 多 crop + MIL）。

---

## 比 SAM3 更该补的三行

1. **分辨率与切块策略**：形态学任务上，这条的效应量常常**大于所有 backbone 差异之和**
2. **通道处理升为横向因子**：现在它被绑在 CA-MAE 一行里，导致该行结果不可归因。改成“每个 backbone 都跑 固定通道 vs 通道无关 两个版本”
3. **领域专用权重**：这条取决于你的模态，但缺了会被审稿人直接问——
   - 若 **H&E/病理** → UNI、UNI2、Virchow、Prov-GigaPath、Phikon、Hibou 必比
   - 若 **荧光细胞 / Cell Painting** → Cell-DINO + OpenPhenom 类

---

## 两个执行层面的提醒

**① 一定做逐层特征扫描，不要只取最后一层。**

PE 那篇的结论（好特征不在输出层）具有普适性。SSL 和语言对齐模型的最后一层是为**预训练目标**优化的（自蒸馏/图文匹配），不是为**分类线性可分**优化的。多标签 mAP 在中层常出现峰值。建议协议：

```text
for layer in {blocks[-1], [-2], [-3], [-6], [-9], [embed]}:
    feat = pool(CLS) ⊕ pool(mean patch)        # 两种都试
    head = ML-Decoder / linear  (固定同一搜索预算)
    metric = mAP, per-class AP, macro-F1
```

**② DINOv2 那行要拆成两步，否则不可归因。**

- 先跑 **frozen + 发布权重**（成本 1 小时）→ 得到“通用 DINOv2 迁移基线”
- 再单独加 **域内自监督** 这一行 → 得到“域内 SSL 的增量”

一步做完，你无法回答“这个收益值不值得那 10 张卡跑两周”。

---

## 需要你确认的一件事

**你的图像模态是什么？** 染色病理（H&E）、荧光多通道、还是明场细胞？

这个答案会直接决定第 3 条（领域专用权重）该补哪些，也决定 SAM3 那条是真有可能赢（有清晰实例边界，如类器官、细胞核）还是注定陪跑（荧光粘连细胞）。给我这个信息，我可以把 T1 表按你的模态直接定下来。
