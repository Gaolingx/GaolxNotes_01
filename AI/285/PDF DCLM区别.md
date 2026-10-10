# PDF DCLM区别

## 🧑‍💻 User

finepdfs、dclm和finewebedu数据集有什么区别，如何搭配使用？

---
## 😀 默认助手 (7)

I'll help you understand these datasets. Let me first discover what tools are available to research this.

Let me inspect the relevant fetch tools and gather information about these datasets.

Now let me gather information about all three datasets in parallel.

The page fetch returned JSON-LD metadata, not the card content. Let me fetch the raw README files directly.

The YAML front-matter is huge. Let me fetch the full READMEs and extract just the body content.

The regex failed due to nested content. Let me slice from the first markdown header instead.

The frontmatter is extremely long. Let me inspect the structure by searching for header positions instead.

Now I found the body sections. Let me extract them.

I have rich detail on all three. Let me quickly check official guidance on mixing recipes (e.g., SmolLM3 web mixture).

Let me try the SmolLM3 model card for the official data mixture.

需要说明的是，我是通过抓取 HuggingFace 官方数据集卡片（DCLM / FineWeb-Edu / FinePDFs 的 README）整理出以下内容的。三者都是 2024–2025 年主流的 **开源大模型预训练语料**，但定位和来源截然不同。

---

## 一、一句话总结

| 数据集 | 一句话定位 |
|---|---|
| **DCLM-baseline** | 用「模型打分 + 分类器」精挑出来的 **英文高质量网页语料**（通用主力） |
| **FineWeb-Edu** | 从 FineWeb 里用 **教育质量分类器** 筛出的「教科书味」英文网页（偏知识/教育） |
| **FinePDFs** | 史上最大的 **PDF 专属** 语料，多语言、长文档、偏法律/科研/教育（领域知识 + 长上下文） |

它们的共同根源都是 **Common Crawl（CC）**，所以**彼此高度重叠，必须做跨源去重**。

---

## 二、核心区别总览

| 维度 | DCLM-baseline | FineWeb-Edu | FinePDFs |
|---|---|---|---|
| 发布方 | Apple / mlfoundations 等（DataComp-LM） | HuggingFaceFW | HuggingFaceFW |
| 规模 | 4T tokens / 3B 文档 | 1.3T（另有 5.4T 的 score-2 版） | ~2.92T tokens / 4.75 亿文档 |
| 载体类型 | HTML 网页（纯文本） | HTML 网页 | **PDF 提取文本** |
| 语言 | 英文 | 英文 | **1733 个语言-文字对**（66 个 >1B tokens） |
| 许可 | CC-BY-4.0 | ODC-By 1.0 | ODC-By 1.0 |
| 核心过滤 | 启发式清洗(RefinedWeb) + Bloom 去重 + **fastText 指令式分类器** | **教育质量分类器**（Llama3-70B 标注训练的 BERT，阈值 3，F1=82%） | 仅模型过滤（去广告/垃圾），**无启发式过滤、无 NSFW 过滤** |
| 文档长度 | 普通网页长度 | 普通网页长度 | 平均约为网页 **2 倍**，大量 >10 万字符 |
| 定位 | 通用研究基线 | 教育/知识数据 | 领域知识 + 长上下文 |

---

## 三、逐个深挖

<details>
<summary><b>① DCLM-baseline —— 通用英文主力语料</b></summary>

- **规模**：4T tokens / 3B documents，衍生自 240T tokens 的 DCLM-Pool（CC 全量）。
- **加工链**：
  1. 启发式清洗（复现 RefinedWeb）
  2. Bloom filter 去重
  3. **模型过滤**：用 OpenHermes 2.5 + r/ExplainLikeImFive 指令数据训练的 fastText 分类器打分筛选
- **实力**（7B 模型对比，同级 token）：

  | 数据 | tokens | CORE | MMLU |
  |---|---|---|---|
  | FineWeb-Edu | 0.14T | 38.7 | 26.3 |
  | DCLM-baseline | 0.14T | **44.1** | **38.3** |
  | DCLM-baseline | 2.6T | **57.1** | **63.7** |

- **短板**：**代码和数学能力弱**（官方明确说 no code/math），定位是研究基线，不适合直接训练生产模型。
- **字段**：保留大量 WARC 元数据、`fasttext_...prob` 质量分、语言 id 等，**方便二次筛选**。

</details>

<details>
<summary><b>② FineWeb-Edu —— 「教育/知识」专门化子集</b></summary>

- **本质**：FineWeb（HTML 网页）→ 用分类器再筛一遍，只保留"教育性强"的页面，得到 1.3T tokens。
- **关键创新**：用 Llama3-70B-Instruct 给 50 万条样本打 0–5 教育分，再训一个 Snowflake-arctic-embed 的回归分类器，**阈值 3** 保留。证明了"合成数据训分类器"的威力。
- **效果**：显著优于原始 FineWeb。
- **短板**：**几乎不含代码**；官方建议搭配 The Stack v2 等代码集，以及 Wikipedia 等专门来源补知识。
- 提供 `sample-10BT / 100BT / 350BT` 小样本，便于消融实验。

</details>

<details>
<summary><b>③ FinePDFs —— PDF 专属、多语言、长文档</b></summary>

- **规模**：~2.92T tokens、4.75 亿文档、1733 个语言-文字对、3.65 TB（105/106 个 CC 快照，2013 夏–2025.2，含部分重新抓取）。
- **为什么要做**：PDF 抽取成本极高但内容质量高，**法律/科研/教育** 领域占比远超 HTML；文档更长 → 对**长上下文**训练极有价值。
- **加工链**（很硬核）：PDF/截断识别 → 截断文件回源重下 → **XGBoost 路由**（数字原生 PDF 走 CPU 的 Docling，扫描件走 GPU 的 RolmOCR）→ 文本后处理 → GlotLID 逐页语言识别 → 精确 + MinHash 去重 → 英文模型过滤 → PII 匿名化。
- **表现**：单独用几乎追平 SOTA 的 SmolLM3-Web；**与 HTML 语料混合后提升显著**。
- **官方关键建议**：**PDF 占比建议低于 $25\%$**。
- **v1.5.0 新增**：内置 **DCLM / EDU / EDU-V2 / OCR-QUALITY** 四个分类器标签 —— 这一点对"搭配使用"极其关键（见下文）。

</details>

---

## 四、它们的互补关系

```
                Common Crawl (原始)
                ├── HTML  →  FineWeb ──(教育分类器)──► FineWeb-Edu   （英文·教育知识）
                │          └──  DCLM 精筛            ──► DCLM-baseline （英文·通用主力）
                └── PDF   →  Docling/RolmOCR 抽取    ──► FinePDFs      （多语言·长文档·领域）
```

- **DCLM vs FineWeb-Edu**：同源（CC 网页）、同语言（英文），**高度重叠**。区别在过滤哲学——DCLM 偏"通用高质量"，FineWeb-Edu 偏"教育性"。二者**不叠加也能用，但要跨源去重**。
- **FinePDFs vs 前两者**：来源不同（PDF vs HTML），**内容互补性最强**——补上网页语料缺失的**多语言、长文档、专业领域**知识。

---

## 五、如何搭配使用（推荐配方）

### 方案 A：通用英文预训练（三者混用）

| 数据 | 建议占比 | 作用 |
|---|---|---|
| DCLM-baseline | ~50–60% | 通用能力主力 |
| FineWeb-Edu | ~20–30% | 知识/教育/推理 |
| FinePDFs（可用 DCLM 标签筛出的英文部分） | **< 25%** | 长文档 + 领域深知识 |

### 方案 B：分阶段课程（更接近 SmolLM3 的做法）

1. **主预训练阶段**：DCLM + FineWeb-Edu（大量、通用）
2. **退火/中训练阶段**：提高 **FinePDFs** 与 FineWeb-Edu 权重，强化长上下文与知识
3. 注意 SmolLM3 整体是 **11.2T tokens 的分阶段课程（web / code / math / reasoning）**，这两个数据集只是其中的 "web" 部分

### 方案 C：用分类器做交叉筛选（最实用的技巧）

FinePDFs v1.5.0 **已经内置 DCLM 和 EDU 分类器标签**，因此你可以：
- 用 **DCLM 分类器分**从 FinePDFs 中挑出"通用高质量"页；
- 用 **EDU / EDU-V2 分类器分**从 FinePDFs 中挑出"教育性"页。

这样就实现了三个数据集在**打分维度上的统一**，便于做统一阈值混合。

### 关键注意事项

1. **必须跨源去重**：三者都源自 CC，重复率不低（MinHash / 精确去重）。
2. **补代码和数学**：DCLM 和 FineWeb-Edu 都缺代码，建议另配 The Stack v2 等；数学需专门数据。
3. **PDF 别超量**：官方明确 PDF 占比 **< $25\%$** 效果最佳。
4. **版权/合规**：注意 FinePDFs/FineWeb-Edu 的 ODC-By 与 CC Terms，以及 DCLM 的 CC-BY-4.0。

---

## 六、加载代码示例

```python
from datasets import load_dataset

# DCLM-baseline（英文通用）
dclm = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)

# FineWeb-Edu（教育，先从小样本试）
fwe = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)

# FinePDFs（按语言取，例如英文；也可取中文 cmn_Hani）
fpdf = load_dataset("HuggingFaceFW/finepdfs", name="eng_Latn", split="train", streaming=True)
```

```python
# 用 FinePDFs 内置分类器标签做筛选取样（示意）
def keep(doc, dclm_th=3, edu_th=3):
    return (doc.get("dclm_score", 0) >= dclm_th) or (doc.get("edu_score", 0) >= edu_th)

fpdf_high = (d for d in fpdf if keep(d))
```

---

## 七、选型速查

- 只想训**英文通用模型** → **DCLM** 为主。
- 想强化**知识/教育/推理** → 加 **FineWeb-Edu**。
- 想要**长上下文 / 多语言 / 法律科研领域** → 加 **FinePDFs（< $25\%$）**。
- 想要**完整开源配方参考** → 对标 **SmolLM3** 的分阶段数据混合。

**主要来源**：HuggingFace 上 `mlfoundations/dclm-baseline-1.0`、`HuggingFaceFW/fineweb-edu`、`HuggingFaceFW/finepdfs` 官方数据集卡片，以及各自论文（DCLM: arXiv 2406.11794；FineWeb-Edu: arXiv 2406.17557）。

需要我帮你**算一份具体的 token 级混合比例表**，或者给某个目标（如中文/代码/长上下文）定制搭配方案吗？
