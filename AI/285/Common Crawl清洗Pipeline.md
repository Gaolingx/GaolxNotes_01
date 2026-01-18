# Common Crawl清洗Pipeline

### 🧑‍💻 User

用于预训练和sft的数tb的Common Crawl（包含原始网页数据（WARC）、元数据（WAT）和文本提取（WET））清洗和去重pipeline要如何设计？

---

### 🤖 Assistant

设计处理数TB（甚至PB级）Common Crawl (CC) 数据的清洗和去重 Pipeline 是训练高质量 LLM 的核心工程挑战。对于数TB的数据量，单机处理已不可行，必须设计基于分布式的流水线。

以下是一个工业级的 Pipeline 设计方案，分为**架构选型**、**详细处理流程**、**关键算法**和**SFT特定处理**四个部分。

---

### 1. 架构选型与基础设施

对于 TB 级别的数据，建议使用 **Apache Spark** 或 **Ray** 作为分布式计算框架。数据格式应全程采用 **Apache Parquet** 或 **Arrow**，以获得列式存储的高压缩比和读取速度。

*   **输入源**: Common Crawl S3 Bucket (`s3://commoncrawl/`)
*   **计算框架**: Ray Data (推荐，更适合异构计算和现代 LLM 栈) 或 Apache Spark。
*   **存储**: 对象存储 (S3/MinIO) + 高速 NVMe SSD (作为缓存)。
*   **开源参考工具**: Hugging Face **Datatrove**, NVIDIA **NeMo Curator**, 或 Facebook **CC-Net**。

---

### 2. 详细 Pipeline 流程设计

建议优先处理 **WARC** 文件而不是 WET，因为 WET 丢失了 HTML 结构，而结构信息对于提取高质量正文（去除导航栏、广告）至关重要。

#### Phase 1: Ingestion & Extraction (提取与格式化)
目标：从 WARC 中提取纯文本，同时保留少量 Markdown 格式（如标题、列表）。

1.  **URL 过滤 (基于 Index)**:
    *   在使用 WARC 之前，先查询 Common Crawl Index (CDX)。
    *   **Blocklist**: 过滤掉成人网站、恶意域名、低质量 SEO 农场。
    *   **Allowlist**: 如果资源有限，优先下载 Wikipedia, Reddit, StackOverflow, Arxiv 等高质量域名的 WARC segment。
2.  **HTML 解析与提取**:
    *   不要直接使用 WET。使用 `trafilatura` 或 `resiliparse` (FastWARC) 解析 HTML。
    *   **Boilerplate Removal**: 去除页眉、页脚、侧边栏、Cookie 弹窗。
    *   **Markdown Conversion**: 将 HTML 转为 Markdown 格式（保留 `<h1>` 至 `#`, `<table>` 等），这有助于模型理解文档结构。

#### Phase 2: Filtering & Cleaning (清洗与过滤)
这是提升数据质量最关键的一步，通常能过滤掉 70%-90% 的原始数据。

1.  **语言识别 (Language Identification)**:
    *   使用 fastText (`lid.176.bin`) 识别语言。
    *   过滤掉非目标语言（如只保留中英文），或按语言分桶。
    *   设置置信度阈值（如 Score > 0.65）。
2.  **启发式过滤 (Heuristic Rules)** - 参考 Gopher/C4 规则:
    *   **长度过滤**: 删除少于 50 个 token 或多于 100k token 的文档。
    *   **符号比率**: 删除包含过多 `#`, `...`, `{}` 等非自然语言符号的文本。
    *   **停用词密度**: 如果一段文本中常用词（the, is, of）占比过低，通常是机器生成的垃圾或代码片段。
    *   **脏词过滤**: 基于黑名单过滤包含过多色情、暴力词汇的文档。
3.  **质量模型过滤 (Model-based Filtering)**:
    *   训练一个轻量级的 5-gram KenLM 模型（在 Wikipedia 等高质量数据上训练）。
    *   计算 CC 文档的 **Perplexity (PPL)**。
    *   PPL 过高（乱码）或过低（重复语句）的文档被丢弃。

#### Phase 3: Deduplication (去重)
去重分为三个层级。数TB数据必须使用 **MinHash + LSH** 进行模糊去重。

1.  **URL 去重**: 标准化 URL（去除 `?query=` 参数），对相同 URL 只保留最新的 snapshot。
2.  **精确去重 (Exact Dedup)**:
    *   计算文档的 SHA-256 哈希值。
    *   全局比对，删除哈希碰撞的文档。
3.  **模糊去重 (Fuzzy Dedup / Near-dedup)**:
    *   解决“只有一两个词不同”或“增加了广告banner”的重复问题。
    *   **算法**: MinHash + LSH (Locality Sensitive Hashing)。
    *   **步骤**:
        1.  **Shingling**: 将文本切分为 N-grams (如 5-grams)。
        2.  **MinHash**: 生成 MinHash 签名（Signature），通常使用 128 个置换函数。
        3.  **LSH**: 将签名分段（Bands），映射到 Hash Buckets。
        4.  **Cluster**: 在同一 Bucket 内的文档计算 Jaccard 相似度，若 $J(A, B) > 0.7$ (阈值)，则视为重复。
        5.  **Remove**: 在重复簇中保留最长或质量最高的一个。

#### Phase 4: PII Removal (隐私去除)
*   **正则匹配**: 识别并掩盖 Email、IP 地址、电话号码、Social Security Numbers。
*   **替换策略**: 将 `john@example.com` 替换为 `<EMAIL>` 或生成虚假数据替换（保持分布一致）。

---

### 3. 关键算法与数学原理

#### MinHash 与 Jaccard 相似度
为了在数亿文档中找到相似文档，我们利用 Jaccard 相似度：

$$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} $$

其中 $A$ 和 $B$ 是文档的 N-gram 集合。
MinHash 的核心属性是：两个集合 MinHash 值相等的概率等于它们的 Jaccard 相似度。

$$ P(h_{\min}(A) = h_{\min}(B)) = J(A, B) $$

#### LSH (Locality Sensitive Hashing)
为了避免 $O(N^2)$ 的两两比较，使用 LSH。将 MinHash 签名分为 $b$ 个 bands，每个 band 有 $r$ 行。
两个文档成为候选对（Candidate Pair）的概率为：

$$ P(\text{candidate}) = 1 - (1 - s^r)^b $$

其中 $s$ 是 Jaccard 相似度。我们需要调整 $b$ 和 $r$ 以使得当 $s$ 超过阈值时，概率迅速趋近于 1。

---

### 4. 针对 SFT (Supervised Fine-Tuning) 的特殊处理

Common Crawl 主要是无监督数据（Pre-training），若要用于 SFT，需要进行 **Data Uplift (数据升级)**。

1.  **自指令构造 (Self-Instruct / Back-translation)**:
    *   从清洗后的 CC 中抽取高质量片段（如维基百科条目、新闻）。
    *   使用强模型（如 GPT-4 或 Qwen-72B）作为 Teacher。
    *   **Prompt**: "请根据以下文本，构建 3 个问答对，用于考核读者的理解能力。"
    *   这能将非结构化文本转化为 `{Instruction, Input, Output}` 格式。

2.  **格式化数据提取**:
    *   **FAQ 提取**: 专门针对 HTML 中的 `FAQ` Schema 或 "Q: ... A: ..." 结构编写提取器。
    *   **论坛对话**: 针对 Reddit 或 StackExchange 类型的页面，通过 HTML 树状结构提取多轮对话历史，转化为 ChatML 格式。

---

### 5. 代码实现示例 (基于 Hugging Face Datatrove)

Datatrove 是目前处理这种规模数据最高效的库之一。

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
)
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.writers import ParquetWriter

# 定义 Pipeline
pipeline = [
    # 1. 读取 WARC
    WarcReader(
        data_folder="path/to/warc/files",
        limit=1000  # 示例限制
    ),
  
    # 2. 提取文本 (HTML -> Text)
    Trafilatura(),
  
    # 3. 语言过滤 (保留英文)
    LanguageFilter(languages=["en"]),
  
    # 4. 质量过滤 (Gopher 规则)
    GopherRepetitionFilter(),
    GopherQualityFilter(
        min_doc_words=50,
        max_doc_words=100000
    ),
  
    # 5. 写入 Parquet (用于后续 MinHash 去重步骤)
    ParquetWriter(
        output_folder="s3://my-bucket/processed/",
        compression="snappy"
    )
]

# 执行器 (可替换为 SlurmPipelineExecutor 用于集群)
executor = LocalPipelineExecutor(pipeline=pipeline, tasks=10, workers=4)
executor.run()
```

### 总结建议

1.  **不要全部下载**: 先下 WET 做实验，确定 Pipeline 逻辑，再下 WARC 跑全量。或者利用 AWS 上的 Common Crawl 公共数据集直接在云端处理。
2.  **重视去重**: 重复数据是导致模型“复读机”现象和性能下降的主要原因。MinHash 是必须的。
3.  **保存元数据**: 在 Parquet 中保留 `url`, `timestamp`, `source_domain`，这对于后续的数据混合（Data Mixing）和根据来源调整权重非常重要。