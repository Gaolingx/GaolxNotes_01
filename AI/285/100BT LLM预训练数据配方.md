# 100BT LLM预训练数据配方

## 🧑‍💻 User

https://huggingface.co/datasets/HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT/tree/main
https://huggingface.co/blog/codelion/optimal-dataset-mixing
参考这个recipe帮我设计一套300/500/1000bt的llm预训练数据配方
可以参考的dataset：ultrafineweb、fineweb/finewebedu、dclm、finepdfs、c4、thepile

---
## 🧑‍💻 User

https://huggingface.co/datasets/HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT/tree/main
https://huggingface.co/blog/codelion/optimal-dataset-mixing
参考这个recipe帮我编写一套100bt的llm预训练数据配方
可以参考的dataset：ultrafineweb、fineweb/finewebedu、dclm、finepdf、c4、thepile

---
## 😀 默认助手 (7)

I'll start by studying the two references you shared, then design the recipes.

Let me pull the dataset card and key source dataset metadata to ground token counts and availability.

Let me extract the FinePDFs stats and check UltraFineWeb.

抓到了关键事实，可以开始设计了。下面先给结论，再展开。

## 0. 先看清参考 recipe 到底给了什么

参考的 [codelion 博客](https://huggingface.co/blog/codelion/optimal-dataset-mixing) 与它落地的 [HF Smol-Data 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT)，核心结论只有四条，全部可复用：

1. **50% FinePDFs + 30% DCLM + 20% FineWeb-Edu** 是"质量锚点 / 真实多样性 / 教育桥接"的黄金配比。
2. **静态混合 > 课程学习**：最差的课程方案在验证集上比静态差 $1.8\times$、泛化差 $2.7\times$，还慢 $2\times$。
3. **硬切换 = 灾难**：质量→多样会遗忘，多样→质量会过拟合（泛化比冲到 $366\times$）。若要做衰减，每阶段改动 $\le 20\%$。
4. 有效性的判据是**验证困惑度 vs 域外困惑度的权衡**，不是单看验证集。参考里最优点的比值约 $346/27.38 \approx 12.6\times$；比值爆炸就是过拟合。

另外注意一个容易被忽略的细节：博客里的 "finePDFs" 是作者自己采样的一小份、被描述成"教科书式"内容；而生产环境的 `HuggingFaceFW/finepdfs` 是**全量 PDF 语料（书籍/报告/论文/文档结构文本）**。HF 自己落地的 Smol-Data 用的是全量版，同时也提供了 `finepdfs_edu_*` 教育过滤版。设计时要把这两个当**不同池子**处理。

---

## 1. 可用数据源盘点（含硬约束）

这是决定配方能否成立的关键。我核实了各池英文可用量：

| 数据源 | 英文可用量（≈） | 性质 | 关键约束 |
|---|---|---|---|
| **FinePDFs** (`eng_Latn`) | **~726B** | 结构化长文档（PDF 提取） | 池子小，是配方的**真正瓶颈** |
| **FinePDFs-Edu** (`eng_Latn`) | **~136B** | 教育过滤 PDF | 极稀缺，只能当"调味" |
| FineWeb-Edu | 1.3T (阈值3) / 5.4T (阈值2) | 教育网页 | 够用，但别超 1 epoch |
| FineWeb | 15T | 广域去重网页 | 近乎无限，做"余量填充" |
| DCLM-baseline | 4T / 3B docs | 模型过滤高质量网页 | 充足 |
| UltraFineWeb (`openbmb`) | ~1T 英 + 120B 中 | fastText 过滤高质量网页 | 充足，替代/补充 DCLM |
| C4 (en) | ~156B | 老一代网页 | 池小、质量低，只能微量 |
| The Pile | ~0.3T (825 GiB) | 22 个子集 | 池小，价值是"领域多样性" |

> ℹ️ 张量口径提醒：这些卡上的 token 数用 **GPT-2/tiktoken** 计。换你自己的 tokenizer（尤其 128k BPE、中文、代码），总量会漂 **±10–20%**，非英文语料每字符 token 数还会再高 $1.5\text{–}2\times$。**最终定比时务必用目标 tokenizer 重算。**

**Smol-Data 的现成积木**（强烈建议先用它验证，再放大）：提供了每个源的 100BT 子集 `HuggingFaceFW/{finepdfs,dclm,fineweb,fineweb_edu,finepdfs_edu}_100BT`（含 `-shuffled` 版本），以及两个拼好的 ~100B 混合包：`finepdfs_50BT-dclm_30BT-fineweb_edu_20BT` 和 `finepdfs_edu_50BT-...`。

---

## 2. 设计原则（三套配方共同遵循）

1. **保留三支柱语义**：结构化文档锚点 + 教育网页 + 高质量多样网页。这是参考 recipe 的可迁移内核。
2. **小池用"占比"，大池用"余量"**。把 FinePDFs / FinePDFs-Edu / C4 / Pile 这类小池当作**绝对 token 预算**（按 epoch 上限锁死），剩余预算全部由 DCLM / UltraFineWeb / FineWeb 吸收。这样预算从 300B 涨到 1T 时，混合比例会**自动从"质量偏重"漂向"多样性偏重"**，而不是让 FinePDFs 被重复 2–3 遍。
3. **单池曝光上限**：premium 池 $\le 0.5$ epoch，教育网页 $\le 0.3$ epoch，其余 $\le 1$ epoch。定义 $\text{epochs} = \dfrac{\text{抽取 token 数}}{\text{池子大小}}$。
4. **多样性地板 $\ge 30\%$**：真实网页（去重后）占比不得低于 ~30%，否则重演"纯 finePDFs 泛化崩溃"。
5. **静态打散**，不做硬切换；若用衰减尾段，只做平滑上调（$\le 20\%$/阶段，尾段 premium 占比 $\le 20\text{–}25\%$）。

---

## 3. 三套配方

### 3.1 推荐版（英文，仅用你列出的池子）

这是可直接落地的主配方。占比随规模小幅漂移，但三支柱结构不变：

| 数据源 | 池子 | **300B** | **500B** | **1000B** | 1000B 时 epochs |
|---|---|---|---|---|---|
| FinePDFs (eng) | 726B | 30% = 90B | 30% = 150B | **28% = 280B** | $0.39$ |
| FinePDFs-Edu (eng) | 136B | 3% = 9B | 3% = 15B | 3% = 30B | $0.22$ |
| FineWeb-Edu | 1.3T | 15% = 45B | 15% = 75B | 15% = 150B | $0.12$ |
| DCLM-baseline | 4T | 20% = 60B | 20% = 100B | 20% = 200B | $0.05$ |
| UltraFineWeb | 1T | 20% = 60B | 20% = 100B | 20% = 200B | $0.20$ |
| FineWeb | 15T | 9% = 27B | 9% = 45B | **11% = 110B** | $0.01$ |
| C4 (en) | 156B | 1% = 3B | 1% = 5B | 1% = 10B | $0.06$ |
| The Pile | ~0.3T | 2% = 6B | 2% = 10B | 2% = 20B | $0.07$ |
| **合计** | | **300B** | **500B** | **1000B** | |

按"支柱"聚合，可见配方骨架在多尺度下非常稳定：

| 支柱 | 300B | 500B | 1000B |
|---|---|---|---|
| 结构化文档（FinePDFs + Edu） | 33% | 33% | 31% |
| 教育网页（FineWeb-Edu） | 15% | 15% | 15% |
| 高质量多样网页（DCLM + UltraFineWeb） | 40% | 40% | 40% |
| 广域/长尾（FineWeb + C4 + Pile） | 12% | 12% | 14% |

### 3.2 忠实版（严格 50-30-20，只换池子不换比例）

如果你要的就是对参考 recipe 的**直接放大**：

| 数据源 | 300B | 500B | 1000B |
|---|---|---|---|
| FinePDFs | 50% = 150B ($0.21$ ep) | 50% = 250B ($0.34$ ep) | 50% = 500B (**$0.69$ ep**) |
| DCLM-baseline | 30% = 90B | 30% = 150B | 30% = 300B |
| FineWeb-Edu | 20% = 60B | 20% = 100B | 20% = 200B |

**结论：它确实能从 1B 一路放大到 1000B 而不过 1 epoch。** 但 1T 时你会用掉**整个英文学术/文档型 PDF 池的 69%**，OCR 噪声与体裁单一的风险被放大。所以我的建议是：**把小规模（≤500B）直接照搬 50-30-20；到 1000B 时切到 3.1 的推荐版**，用 UltraFineWeb/FineWeb 稀释 PDF 占比、补足多样性。

---

## 4. 为什么是这些数（逐组件解释）

- **FinePDFs 30%→28%**：保留参考里最强的质量信号（长文档、句法规范、结构清晰），但用"绝对预算"锁住曝光——1T 时 280B 只消耗 $0.39$ epoch，留出替换与去污染余量。
- **FinePDFs-Edu 3%**：136B 的池子只能当高浓"味精"，1T 时 $0.22$ ep 已接近合理上限，再多就是重复记忆。
- **FineWeb-Edu 15%**：教育/知识密度信号，但比参考的 20% 略降——因为 FinePDFs 本身已提供大量结构化知识，二者功能有重叠，降一点可减少"过度教育化"导致的 HellaSwag/PIQA 类常识任务退化（这正是 FineWeb-Edu 论文里阈值抬高时的现象）。
- **DCLM 20% + UltraFineWeb 20% = 40% 多样网页**：这是**泛化保命项**，比参考的 30% 还更保守。两池并用是因为单靠 DCLM 在 1T 预算下多样性不够，且 UltraFineWeb 的 fastText 过滤覆盖面不同，能互补。
- **FineWeb 9%→11%**：15T 池子近乎无限，专门当"余量填充 + 冷门话题覆盖"，防止模型只见过被过滤器偏好的窄分布。
- **C4 1% + Pile 2%**：C4 是老一代锚点，成本极低；Pile 的真正价值是**它在纯网页/PDF 里缺失的领域**（代码、数学、书、法律/医学）注入少量多样性，是便宜保险。两者都别超过 2–3%。

---

## 5. 可选增强模块（按需启用）

<details>
<summary><b>通用型变体：加入代码 / 数学</b></summary>

参考 recipe 是纯英文网页/PDF，**没有代码、没有数学、多语言也很弱**。要训通用基座，必须补。

做法：从 `DCLM + UltraFineWeb + FineWeb` 三池按比例扣减约 **8 个百分点**，重新分配：

| 模块 | 占比 | 1000B 例 | 推荐数据 |
|---|---|---|---|
| 代码 | 6% | 60B | `bigcode/the-stack-v2`、`bigcode/starcoderdata` |
| 数学 | 2% | 20B | `open-web-math`、`FineMath`、`EleutherAI/proof-pile-2` |

其余配方不变（PDF 支柱保持 31%、FineWeb-Edu 15%）。
</details>

<details>
<summary><b>多语言扩展</b></summary>

若需中/多语：用 `HuggingFaceFW/fineweb-2`、FinePDFs 的多语言分片、`openbmb/UltraFineWeb` 的 120B 中文分片，替换 FineWeb 的填充位；建议总非英占比先设 10–20%，并按语言分层保证小语种不被大语种淹没。
</details>

<details>
<summary><b>尾段衰减（可选，谨慎）</b></summary>

参考明确说硬切换有害，但**平滑衰减尾段**在工业界常见且兼容：在最后 10–20% token 里把 premium 占比平滑上调（例如 FinePDFs 28%→40%），每阶段改动 $\le 20\%$。**绝不要**在尾段用"100% 某单一源"。
</details>

---

## 6. 落地实现

<details>
<summary><b>采样率直接可用的配方表 + 代码</b></summary>

一个关键等式让实现变得极简：若对某源按概率 $r$ 做**逐文档均匀采样**，则

$$ \text{epochs} = \frac{r \cdot N_{\text{pool}}}{N_{\text{pool}}} = r $$

**即"采样率 = epoch 数"。** 所以 1000B 配方可直接写成逐源采样率：

```python
RECIPE_1000B = {
    # dataset path                              rate(≈epochs)   tokens
    "HuggingFaceFW/finepdfs":                   0.386,  #  280B
    "HuggingFaceFW/finepdfs-edu":               0.221,  #   30B
    "HuggingFaceFW/fineweb-edu":                0.115,  #  150B
    "mlfoundations/dclm-baseline-1.0-parquet":  0.050,  #  200B
    "openbmb/UltraFineWeb":                     0.200,  #  200B
    "HuggingFaceFW/fineweb":                    0.0073, #  110B
    "allenai/c4":                               0.064,  #   10B
    "EleutherAI/pile":                          0.067,  #   20B
}
# 300B / 500B 的 rate 按预算线性缩放（0.3x / 0.5x）即可
```

生产路径（复刻 HF `smol_data.py`）：

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.tokens import DocumentTokenizer
import hashlib

def rate_for(seed: int, name: str, rate: float):
    # 用稳定哈希保证可复现的均匀下采样
    ...

pipeline = [
    ParquetReader("hf://datasets/HuggingFaceFW/finepdfs", glob_pattern="data/*/*.parquet"),
    SamplerFilter(rate=0.386, seed=42),
    JsonlWriter("s3://mix/finepdfs"),
]
LocalPipelineExecutor(pipeline, tasks=64).run()
# 对每个源跑一遍 -> 得到分片 -> 全局 shuffle -> DocumentTokenizer
```

要点：
- **务必用 `-shuffled` 版本或自己做全局 shuffle**，否则各源内部按 crawl 时间排序会引入分布漂移（等价于软课程）。
- FinePDFs 类数据集带 `token_count` 字段，可**先算预算再采样**，避免超采。
- 去重：源内 MinHash 已做（FinePDFs 有 `minhash_cluster_size` / `duplicate_count`），你只需做**跨源去重 + 与评测集去污染**。
</details>

<details>
<summary><b>验证与消融协议（强烈建议先做）</b></summary>

沿用参考的方法论，分三级筛选，**不要直接上全量**：

| 阶段 | 模型 | token | 目的 |
|---|---|---|---|
| P1 筛选 | 70M | 2–5B | 在 2–4 个候选配比里筛出 1–2 个 |
| P2 确认 | 300M–1B | 20–50B | 复核排名、验证质量/多样性权衡 |
| P3 全量 | 目标模型 | 300/500/1000B | 正式训练 |

**评测集设计**（关键）：
- 域内：每源各留 held-out split 测 PPL；
- 域外/泛化：FineWiki、Wikipedia、Pile-val（OOD PPL 是核心指标）；
- 下游：ARC-E/C、HellaSwag、PIQA、WinoGrande、MMLU（若加了 code/math 再加对应集）。

**决策规则**：选 Pareto 前沿点——OOD PPL 不再下降或下游均值见顶即停；若 $\text{OOD PPL}/\text{域内 PPL}$ 比值远超 ~$13\text{–}20\times$，说明朝着"过拟合到 curate 池"的方向走了，需上调多样性池。
</details>

<details>
<summary><b>风险与常见坑</b></summary>

1. **tokenizer 漂移**：卡上的 GPT-2 token 数≠你的 token 数，定比前重算。
2. **FinePDFs 体裁单一 + OCR 噪声**：1T 时别超过 ~30%；若发现下游常识任务退化，把它换成 `finepdfs-edu` 或下调 5pp。
3. **UltraFineWeb 口径歧义**：我按 `openbmb/UltraFineWeb`（~1T 英）设计；若你指的是 DatologyAI 的版本，池子更大，可给 FinePDFs 更多稀释。
4. **别把 Pile/C4 当主力**：它们是"保险"，超 3% 会拉低整体质量。
5. **纯网页/PDF 无益于推理与代码**：不加 code/math，模型在代码数学任务上会近乎随机。
</details>

---

## 7. 一句话总结

- **≤500B**：直接放大 50-30-20（FinePDFs:DCLM:FineWeb-Edu），简洁且被验证。
- **1000B**：把比例改成 **28-3-15-20-20-11-1-2**（PDF/PDF-Edu/FW-Edu/DCLM/UltraFineWeb/FineWeb/C4/Pile）——本质是把"小池按 epoch 封顶、大池吸收余量"，从而在 1T 预算下既保留参考 recipe 的质量锚点，又把 FinePDFs 曝光压在 $0.39$ epoch、多样性网页保持在 40%。
- **无论哪档**：静态打散、不做硬切换、下限 30% 真实网页、单池 $\le 0.5$ epoch，并且先做 70M/2–5B 的 proxy 消融再放大。

如果你告诉我**目标模型规模、是否要代码/数学能力、以及是否只需英文**，我可以把这套表精确到每源的采样率和 shuffle 脚本，并给出可直接跑的 datatrove 配置。
