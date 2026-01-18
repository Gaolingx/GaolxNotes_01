# LLM预训练数据清洗方法详解

### 🧑‍💻 User

llm预训练阶段的数据集获取之后如何清洗？

---

### 🤖 Assistant

LLM（大语言模型）预训练阶段的数据质量直接决定了模型的最终效果（即所谓的 "Garbage In, Garbage Out"）。获取原始数据（如 Common Crawl、GitHub 代码、书籍、论文）后，通常需要经历一个复杂且严谨的清洗流水线。

以下是工业界（如 GPT-3, Llama, Bloom 等）通用的数据清洗全流程指南：

### 1. 文本提取与格式规范化 (Text Extraction & Formatting)

原始数据通常包含大量的非文本内容（如 HTML 标签、CSS、JS 代码、WARC headers）。

*   **HTML 解析:** 使用工具（如 `trafilatura`, `newspaper3k`, `BeautifulSoup`）从 HTML 中提取纯文本。
*   **编码修复:** 统一将编码转换为 UTF-8，修复 `ftfy` (fixes text for you) 等乱码问题。
*   **格式统一:** 将所有换行符统一为 `\n`，处理非打印字符。

```python
import ftfy
from bs4 import BeautifulSoup

def clean_text(raw_html):
    # 1. 提取文本
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n")
    # 2. 修复编码错误
    text = ftfy.fix_text(text)
    return text
```

### 2. 启发式过滤 / 基于规则的过滤 (Heuristic / Rule-based Filtering)

这是为了快速剔除低质量文本（如导航栏、SEO 垃圾词、随机字符）。通常参考 **Gopher** 或 **C4** 数据集的清洗规则。

常见的过滤指标包括：
*   **语言识别 (Language ID):** 使用 `fastText` 或 `langid` 剔除目标语言以外的数据（例如训练中文模型时剔除纯阿拉伯语页面）。
*   **文本长度:** 剔除过短的文档（例如少于 50 个 token 或 100 个字符）。
*   **平均词长:** $L_{word} = \frac{Total\ Characters}{Total\ Words}$。如果过长或过短，可能是代码或乱码。
*   **符号比例:** 如果 `#`, `...`, `{}` 等符号占比过高，予以剔除。
    *   例如：如果 $Ratio_{symbol} > 0.4$，则过滤。
*   **停用词密度:** 有效文本通常包含一定比例的常用停用词（如 "的", "是", "the", "and"）。如果完全没有停用词，可能是单词列表而非连贯文本。

**数学表达示例（符号密度过滤）：**
设 $N_{total}$ 为文档总字符数，$N_{symbol}$ 为符号字符数，则保留文档当且仅当：
$$ \frac{N_{symbol}}{N_{total}} < \theta $$
其中 $\theta$ 是设定的阈值（如 0.3）。

### 3. 去重 (Deduplication)

互联网数据重复率极高，重复数据会导致模型死记硬背（Overfitting）且浪费算力。去重通常分为三个层级：

#### A. 精确去重 (Exact Deduplication)
*   计算文档的 SHA-256 哈希值，删除哈希碰撞的文档。
*   适用于完全一样的复制粘贴。

#### B. 模糊去重 (Near-Deduplication)
这是最关键的一步，旨在去除相似但不完全相同的文章（如只有页脚不同的新闻转载）。
*   **算法:** **MinHash + LSH (Locality Sensitive Hashing)**。
*   **原理:**
    1.  将文档转换为 N-grams 集合。
    2.  计算 MinHash 签名。
    3.  利用 Jaccard 相似度判断重复。
  
    Jaccard 相似度定义为：
    $$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} $$
    如果 $J(A, B) > 0.8$（通常阈值），则视为重复。

#### C. 语义去重 (Semantic Deduplication)
*   利用 Embedding 模型计算向量余弦相似度（Cosine Similarity），计算量巨大，通常只用于特定高质量小数据集。

### 4. 质量过滤 / 基于模型的过滤 (Quality Filtering)

通过规则过滤后的数据仍可能包含通顺但无意义的文本（如生成的广告文案）。需要用模型来评分。

#### A. 困惑度过滤 (Perplexity / PPL Filtering)
使用一个高质量的小型语言模型（如 KenLM 训练在 Wikipedia 上的模型）计算文档的困惑度。
*   **逻辑:** 如果一段文本的 PPL 过高，说明它不像正常的人类语言（乱码）；如果 PPL 过低，可能是重复的短语（如 "1, 2, 3, 4..."）。
*   **公式:** 对于序列 $W = (w_1, w_2, ..., w_N)$，困惑度计算为：
    $$ PPL(W) = P(w_1, \dots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1 \dots w_N)}} $$
*   **操作:** 设置 PPL 的上下阈值，或者根据 PPL 分布截断两端的离群值（Outliers）。

#### B. 分类器过滤 (Classifier-based)
*   训练一个二分类器（如轻量级的 BERT 或 fastText）。
*   **正样本:** Wikipedia, 书籍, ArXiv 论文。
*   **负样本:** Common Crawl 中随机抽取的未经清洗的数据。
*   用该分类器预测新数据，保留预测为“高质量”的数据。

### 5. 个人隐私与安全清洗 (PII & Safety)

为了防止模型泄露隐私或生成有害内容。

*   **PII (Personally Identifiable Information) 去除:**
    *   使用正则表达式（Regex）或命名实体识别（NER）工具（如 Microsoft Presidio）。
    *   **对象:** 电子邮件地址、电话号码、IP 地址、身份证号/SSN。
    *   **处理:** 将其替换为特殊 Token，如 `<EMAIL>`, `<PHONE>`。
*   **毒性过滤 (De-toxification):**
    *   使用词表匹配（Bad words list）过滤色情、暴力、仇恨言论。
    *   使用 Perspective API 或类似的 Toxicity Detection 模型打分。

### 6. 评测集去污染 (Decontamination)

这是一个常被忽视但极重要的步骤。确保训练集中不包含测试集（Benchmark）的数据，否则评测结果将虚高。

*   **N-gram 匹配:** 将所有 Benchmark（如 MMLU, GSM8K, C-Eval）的问题部分构建为 N-gram 索引。
*   **移除:** 在预训练数据中搜索这些 N-gram，如果发现重叠长度超过阈值（如连续 13 个 token 相同），则从训练集中删除该文档。

### 总结：数据清洗流水线概览

| 步骤 | 目标 | 常用工具/算法 |
| :--- | :--- | :--- |
| **1. 提取** | HTML -> 纯文本 | Trafilatura, JusText |
| **2. 格式化** | 统一编码, 换行符 | ftfy, Unicode Normalization (NFC) |
| **3. 规则过滤** | 去除过短、过长、代码比例高的数据 | FastText (LangID), Gopher Rules |
| **4. 模糊去重** | 去除转载、相似文档 | MinHash + LSH, Datasketch |
| **5. 质量过滤** | 确保文本通顺、高质量 | KenLM (Perplexity), BERT Classifier |
| **6. PII/安全** | 去除隐私和有害内容 | Presidio, Regex, Bad-words list |
| **7. 去污染** | 移除测试集数据 | N-gram Overlap matching |

### 推荐开源工具库

如果不想从头造轮子，可以使用以下成熟的开源数据处理库：

1.  **HuggingFace `datatrove`:** 专为大规模 LLM 数据处理设计，包含上述大部分 Pipeline。
2.  **`text-dedup`:** 专门用于 MinHash LSH 去重的库。
3.  **Meta `CCNet`:** 处理 Common Crawl 数据的经典流水线代码。