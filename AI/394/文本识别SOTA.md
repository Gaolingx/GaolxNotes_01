# 文本识别SOTA

## 🧑‍💻 User

现在ocr sota是哪个，OCR 是不是已经被LLM干碎了？

---
## 🤖 Assistant

# 现在的 OCR SOTA 是哪个？

先说结论,再展开。

## TL;DR

- **"OCR 被 LLM 干碎了"这个说法,方向对了一半,但表述不准确。** 被干碎的其实是"**传统 pipeline 式 OCR(检测+识别两段式)**"这种架构,而不是 OCR 这个任务本身。任务还在,只是换了个"载体"——几乎所有 SOTA 都变成了**端到端的文档理解 VLM**。
- **当前文档解析(OmniDocBench v1.6)榜首是 `TeleOCR`(1.2B,Overall 96.91)**,紧随其后是 `OvisOCR2`(0.8B)、`PaddleOCR-VL-1.6`(0.9B)、`MinerU2.5-Pro`(1.2B)、`GLM-OCR`(0.9B)。
- 最有意思的一点:**通用大模型(GPT-5.2、Gemini 3 Pro、Qwen3-VL-235B)在这个榜上被一堆 1B 上下的小模型按在地上摩擦。** 所以准确说法不是"LLM 干碎了 OCR",而是"**专用小 VLM 干碎了通用大模型 + 传统 OCR**"。

---

## 当前 SOTA 快照(基于 OmniDocBench v1.6 端到端榜单)

OmniDocBench 是目前文档解析领域最主流的公开榜单(CVPR 2025),`Overall = ((1 - 文本编辑距离)×100 + 表格 TEDS + 公式 CDM) / 3`。

| 模型 | 类型 | 参数量 | Overall ↑ |
|---|---|---|---|
| **TeleOCR** | 专用 VLM | 1.2B | **96.91** |
| **OvisOCR2** | 专用 VLM | 0.8B | 96.47 |
| **PaddleOCR-VL-1.6** | 专用 VLM | 0.9B | 96.34 |
| MinerU2.5-Pro | 专用 VLM | 1.2B | 95.75 |
| GLM-OCR | 专用 VLM | 0.9B | 95.22 |
| PaddleOCR-VL-1.5 | 专用 VLM | 0.9B | 94.93 |
| Unlimited-OCR | 专用 VLM | 3B | 94.00 |
| Gemini 3 Pro | 通用 VLM | — | 92.91 |
| dots.ocr | 专用 VLM | 3B | 90.77 |
| DeepSeek-OCR 2 | 专用 VLM | 3B | 90.25 |
| Qwen3-VL-235B | 通用 VLM | 235B | 89.78 |
| GPT-5.2 | 通用 VLM | — | 86.59 |
| MinerU-Pipeline | 传统 pipeline | — | 86.47 |
| olmOCR | 专用 VLM | 7B | 85.74 |
| Marker | 传统 pipeline | — | 78.44 |

> 数据来源:OmniDocBench 官方仓库 `README`(v1.6_full)。榜单更新频繁,TeleOCR / OvisOCR2 / Unlimited-OCR 是最近才加入的。

**读这张表的三个要点:**

1. **断层**:传统 pipeline(Marker 78、MinerU-Pipeline 86)已经明显掉队,而前排全是 VLM。这就是"OCR 被干碎"的直观来源。
2. **专用 > 通用**:前 7 名清一色是 0.8B~3B 的专用小 VLM。GPT-5.2、Qwen3-VL-235B 反而垫底。说明做文档解析,**架构 + 数据 + 专项训练**,远比堆参数重要。
3. **小而强是趋势**:0.9B 的 PaddleOCR-VL 能干翻 235B 的 Qwen3-VL,意味着 OCR/文档解析的"性价比拐点"来了,端侧部署变得现实。

---

## "OCR 是不是已经被 LLM 干碎了?"

这里有三个容易混淆的概念,拆开看才清楚:

<details>
<summary><b>① 被干碎的:传统两段式 pipeline(检测→识别)</b></summary>

以前 OCR = 文本检测(DB 等)+ 文本识别(CRNN/SVTR 等)+ 版面分析 + 表格/公式识别,一条长流水线,每一环都要单独训练和调参。现在一个端到端 VLM 一张图直接吐 Markdown/HTML/LaTeX,误差不累积,还能顺便做版面、阅读顺序、表格、公式。

**这部分,确实被"干碎"了。** 典型代表就是 Marker / Tesseract / 早期 EasyOCR 在榜上的位置。

</details>

<details>
<summary><b>② 没被干碎的:OCR 这个任务本身</b></summary>

LLM 并没有让"把图上的字读出来"这件事消失,只是把它的**实现方式**换成了 VLM。而且现在最卷的恰恰是 OCR 专项模型(榜上前排全是 `xx-OCR`)。所以更准确的说法是:**OCR 从"一个独立赛道"变成了"多模态大模型的一个基础能力 / 子任务"。**

</details>

<details>
<summary><b>③ 仍然没解决的硬骨头</b></summary>

别被 96 分骗了,榜单高分有很多"迎合指标"的成分(比如编辑距离、TEDS 对局部错误的惩罚有限)。真实场景里 LLM-VLM 系 OCR 仍然翻车的地方:

- **幻觉(最致命)**:模型会"脑补"看不清的字,或者把不存在的表格/公式补全,而你无法察觉。传统 OCR 顶多识别错,不太会无中生有。
- **超长文档 / 密集小字**:200+ DPI 的报纸、密集脚注、手写草稿,错误率仍高。
- **精确定位(bounding box / 坐标)**:VLM 输出的是文本流,要回到像素级坐标仍需要额外的 grounding,做版面还原(如 PDF→可编辑文档)、文档问答定位时是刚需。
- **结构化保真**:复杂合并单元格、跨页表格、嵌套公式,离"可直接用"还有距离。
- **成本/延迟/吞吐**:传统 OCR 在端侧、大批量扫描件上依然便宜得多、快得多;VLM 推理成本是数量级差异。
- **确定性/可审计**:金融、政务、票据这类场景,传统 OCR 的确定性输出反而更友好。

</details>

---

## 结论

1. **"现在 OCR SOTA 是哪个"** → 在文档解析这个维度,是 **TeleOCR / OvisOCR2 / PaddleOCR-VL 系列 / MinerU2.5-Pro 这类专用小 VLM**;并且迭代极快,今天的榜一几周后可能就换了。
2. **"OCR 是不是被 LLM 干碎了"** → 更准确的说法是:**传统 pipeline OCR 被端到端 VLM 范式取代了**;而"通用大模型"并不占优,**专用小 VLM 才是真正的赢家**。
3. **给选型建议**:
 - 要**效果优先、文档解析/表格/公式** → 用 PaddleOCR-VL、MinerU2.5、dots.ocr 这类开源专用 VLM。
 - 要**低成本、高吞吐、端侧、确定性** → 传统方案(Tesseract / PaddleOCR 传统 pipeline / 各云 OCR)仍是理性选择,远没到"必须用 LLM"的地步。
 - 要**幻觉敏感场景** → 一定要加交叉验证 / 人工兜底,不要裸信 VLM 输出。

如果你告诉我具体场景(票据、论文、身份证、手写、多语言……),我可以帮你缩小到 1~2 个最合适的模型。
