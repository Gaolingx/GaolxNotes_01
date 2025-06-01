# KV Cache精度要求分析

### 🧑‍💻 User

关于llm推理引擎的"kv cache量化"，如果只是简单的知识问答，不涉及多轮对话、长文本生成等任务，对"Key"精度要求是否会相对更低？

---

### 🤖 Assistant

简短结论
——在只做一次性的知识问答、上下文很短（几十到一两百 token）的场景里，**Key 的精度可以比长文本生成时适当再压低一点，但仍然不能低到比 Value 更粗**。实测：

| 量化方案                                 | 典型任务（MMLU/CEVAL 等）精度损失 | 适用建议                                   |
| ------------------------------------ | ---------------------- | -------------------------------------- |
| FP16 → INT8 (Key=8 bit, Value=8 bit) | ≈0 – 0.1 pp            | 几乎无感知，首选通用方案 ([LMDeploy][1])           |
| INT4 (Key=4 bit, Value=4 bit)        | 0.3 – 1 pp             | 短问答基本可接受；长上下文开始明显劣化 ([LMDeploy][2])    |
| INT2 (Key=2 bit, Value=4 bit)        | >15 pp 下降              | Key 过低精度会放大误差，不建议，即便上下文很短 ([arXiv][3]) |

---

### 为什么短问答对 Key “宽容”一些？

1. **误差累积路径短**

   * Key 量化误差通过“Query·Key”的点积影响注意力分布。
   * 当缓存里只有十几个 Key 时，分布被拉偏的机会较少；生成 1000+ token 时，每层、每步都在累加误差，容错度就大幅降低。([Hugging Face][4])

2. **相对动态范围小**
   问答提示通常只含一个问题 + 少量系统指令，Key 向量之间的相似度差异没那么极端，低 bit-width 更易覆盖其数值范围。

---

### 仍需给 Key “比 Value 更高”或“至少相同”精度的原因

* 系统性分析（KV-AdaQuant, Feb 2025）发现 Key 矩阵的谱范数更大，量化噪声被放大得更厉害；同样 4 bit，只要把 Key 和 Value 对换，准确率可差 20 pp 以上。([arXiv][3])
* 许多部署框架（如 LMDeploy v0.4.0）默认也是 **Key≥Value**（支持 8/4 bit，可混合）。([LMDeploy][1])

---

### 实战建议

| 场景                   | 推荐 KV 方案                                     | 备注                                 |
| -------------------- | -------------------------------------------- | ---------------------------------- |
| 单轮问答、≤128 token      | **Key 8 bit / Value 8 bit**；存储压力大时可试 **4/4** | 几乎无可感知差异，压显存即用 4/4                 |
| 中长度摘要或代码解释（几百 token） | **Key 8 bit / Value 4 bit**                  | mixed-precision，节省一半 Value 显存、影响微弱 |
| >1 k token 生成 / Chat | 保守用 **8/8**；尝试 4 bit 时要回归测试                  | 长上下文对 Key 量化极敏感                    |

> ⚠️ 当模型要做数值计算或逻辑推理（如 GSM8K）时，即使只有一个问题，也建议保持 Key 8 bit，因为注意力误差会直接影响推理链条。

---

**结论**
在“简单知识问答”里，你可以把 Key 从 FP16 压到 INT8；如果显存吃紧，再试 INT4。再往下（ <4 bit）Key 精度的损失通常会立刻体现为答案错漏，不值得冒险。若需要进一步压缩，采用 **Key≥Value** 的混合精度或基于梯度/谱范数的自适应方案，会比单纯降低 Key 位宽稳健得多。

[1]: https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html?utm_source=chatgpt.com "INT4/INT8 KV Cache — lmdeploy - Read the Docs"
[2]: https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/kv_quant.html?utm_source=chatgpt.com "Key-Value(KV) Cache 量化 — lmdeploy"
[3]: https://arxiv.org/abs/2502.15075?utm_source=chatgpt.com "More for Keys, Less for Values: Adaptive KV Cache Quantization"
[4]: https://huggingface.co/blog/kv-cache-quantization?utm_source=chatgpt.com "Unlocking Longer Generation with Key-Value Cache Quantization"


---

### 🧑‍💻 User

具体到各项任务以及不同的序列长度里面，"Key"量化对generation quality影响有多大？具体是哪些方面的性能下降？

---

### 🤖 Assistant

### 1. 一眼结论

| 任务 →<br/>序列长度                              | **Key 8 bit** (Val 8 bit)<br/>损失 | **Key 4 bit** (Val 4 bit)<br/>损失 | **Key 2 bit** (Val ≥ 4 bit)<br/>损失 | 主要失真表现                     |
| ------------------------------------------ | -------------------------------- | -------------------------------- | ---------------------------------- | -------------------------- |
| 纯知识问答 (MMLU/TriviaQA, ≤ 128 tok)           | ≈ 0 pp                           | 0 – 0.5 pp                       | ≥ 5 pp                             | top-1 选项混淆                 |
| 代码/数学推理 (GSM8K, \~300 tok)                 | ≈ 0 pp                           | 0.4 pp (ZipCache)                | **>15 pp**，接近随机                    | Chain-of-Thought 被截断、算错中间量 |
| 新闻摘要 (CNN/DailyMail, \~512 tok)            | ≤ 0.3 ROUGE-L                    | 1 – 1.5 ROUGE-L                  | 8 + ROUGE-L                        | 开始漏掉细节、段落重复                |
| LongBench 检索/QA (4 k tok)                  | < 1 pp                           | 1 – 3 pp (SimLayerKV)            | 10 + pp、召回率骤降                      | 早期 token 遗忘、答案漂移           |
| Needle-in-Haystack & code-doc (≥ 32 k tok) | 2 pp 内                           | **4 – 6 pp**；需混合精度               | **崩溃**：随机 / 无法停止                   | 长距依赖彻底失效、重复段落激增            |

<sub>pp = percentage points；所有数字取自公开基准平均值，见引用</sub>

### 2. 关键实测依据

* **Key 8 bit 基本无损**：LMDeploy 在 50 个 OpenCompass 基准上给出的平均差异 <0.1 pp；推理吞吐量 +30 % – 40 % ([LMDeploy][1])
* **Key 4 bit 可接受但可见**：

  * LongBench 16 项任务平均 ↓1.2 pp（SimLayerKV） ([OpenReview][2])
  * 在 4 k-token 检索任务中，KVCrush 4× 压缩仅 ↓0.9 pp，但重复率上升 7 % ([arXiv][3])
* **Key 2 bit 风险极高**：

  * KV-AdaQuant 对 LLaMA-7B：Key-2 / Val-4 仅 54.7 % 准确，而 Key-4 / Val-2 达 75.2 % ——同样 3 bit/entry 总预算，却差 20 pp+ ([arXiv][4])
  * Emergent-Ability study：2-bit 时 GSM8K、MMLU 接近随机 (≈ 25 %) ([ar5iv][5])
* **推理类任务最敏感**：ZipCache 在 GSM8K 上 4.98× 压缩（Key≈4 bit 等价）只 ↓0.38 pp，但在 2 bit 设置下直接 >10 pp ([NeurIPS 会议录][6])
* **极端低 bit 仍有人做到**：AKVQ-VL & SKVQ 借助注意力显著性/WHT，可在 2 - 2.5 bit 下保持 <1 pp 损失，但需复杂动态分配，且目前只在多模态或滑窗场景报告成功 ([arXiv][7], [OpenReview][8])

### 3. 量化后到底坏在哪？

| 失真点              | 具体表征                      | 触发条件                      | 关联机制                                   |
| ---------------- | ------------------------- | ------------------------- | -------------------------------------- |
| **注意力分布抖动**      | log-prob 梯度噪声↑ → 选错 top-K | Key <4 bit；长序列            | Query·Key 内积误差 ∝ Key 谱范数² ([arXiv][4]) |
| **远距 token 掉权重** | 长文摘要漏信息、Needle-QA 答偏      | >2 k tok 时尤甚              | 量化噪声随距离累积，早期 Key 权重被稀释                 |
| **循环重复**         | “所以所以所以…”；句块复制            | Key 4 bit+极长推理 (>8 k tok) | Attention 随机游走造成 entropy 减小            |
| **逻辑链断裂**        | GSM8K 解题步骤丢一行             | 任意长度，Key <4 bit           | 误差在每步自回归中放大                            |
| **对齐退化**         | 有害/幻觉回答 ↑                 | 当 Key\<Val 精度或错配          | 自监督对齐目标敏感于信息检索失败                       |

### 4. 实操建议矩阵

| 场景                | 推荐 Key/Val 位宽                               | 备注               |
| ----------------- | ------------------------------------------- | ---------------- |
| ≤128 tok 单轮 QA    | **8/8** → 4/4 如显存紧张                         | 几乎无感知差异          |
| 256–512 tok 摘要、翻译 | **8/4** (混合)                                | 节省 ½ 显存，质量微损     |
| 代码/数学推理           | **8/8**；或 KV-AdaQuant 8/4                   | 推理链条对 Key 抖动极敏感  |
| ≥4 k tok 长文本生成    | **至少 8/8**，或 SimLayerKV 4/4 + Token-Pruning | 先压 Value，再考虑 Key |
| 超长窗 (≥32 k)       | KVQuant / RotateKV / SKVQ 动态 2–4 bit        | 必要时逐层自适应；离线校准    |

> **经验法则**：**Key 的平均 bit 数应 ≥ Value**；若必须低于 4 bit，务必采用谱范数/显著性驱动的 *mixed-precision* 或滑窗方案来“重点保护”高权重 Key，而不是一刀切降位。

---

#### 5. 进一步阅读

KV-AdaQuant (spectral 分析)、SimLayerKV、ZipCache、RotateKV、AKVQ-VL、KVCrush、SKVQ 等论文在不同任务和序列长度上给出了系统实验，建议按与你模型规模最接近的那篇进行回归测试再上线。

[1]: https://lmdeploy.readthedocs.io/zh-cn/v0.4.2/quantization/kv_quant.html?utm_source=chatgpt.com "Key-Value (KV) Cache 量化 — lmdeploy 0.4.2 文档"
[2]: https://openreview.net/forum?id=UjSmUlUU6y&utm_source=chatgpt.com "SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction"
[3]: https://arxiv.org/pdf/2503.00022?utm_source=chatgpt.com "KVCrush: Key Value Cache size-reduction using similarity in head-behaviour"
[4]: https://arxiv.org/abs/2502.15075?utm_source=chatgpt.com "More for Keys, Less for Values: Adaptive KV Cache Quantization"
[5]: https://ar5iv.labs.arxiv.org/html/2307.08072?utm_source=chatgpt.com "Do Emergent Abilities Exist in Quantized Large Language Models:"
[6]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html?utm_source=chatgpt.com "ZipCache: Accurate and Efficient KV Cache Quantization with Salient ..."
[7]: https://arxiv.org/abs/2501.15021?utm_source=chatgpt.com "AKVQ-VL: Attention-Aware KV Cache Adaptive 2-Bit Quantization for Vision-Language Models"
[8]: https://openreview.net/forum?id=nI6JyFSnyV&utm_source=chatgpt.com "SKVQ: Sliding-window Key and Value Cache Quantization for Large..."
