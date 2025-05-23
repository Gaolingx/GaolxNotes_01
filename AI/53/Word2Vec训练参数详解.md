# Word2Vec训练参数详解

### 🧑‍💻 User

训练word2vec的有哪些重要参数？

---

### 🤖 Assistant

在训练 Word2Vec 模型时，以下是一些关键的**超参数**和**调参建议**，它们会显著影响生成的词向量质量：

---

### **一、核心模型参数**
1. **`vector_size`（size）**
   - **作用**：词向量的维度大小（如 100、200 等）。
   - **调参**：通常选择 100~300，维度越高表达能力越强，但可能需要更多数据和时间训练。

2. **`window`**
   - **作用**：上下文窗口大小，即预测目标词时考虑的上下文范围。
   - **调参**：较小的窗口（如 2-5）适合捕捉局部队义（如词性），较大的窗口（5-10）适合全局语义。

3. **`min_count`**
   - **作用**：过滤低频词，仅保留词频大于等于该值的词。
   - **调参**：默认值为 5，需根据语料规模调整。若语料大且词汇丰富，可适当提高。

4. **`workers`**
   - **作用**：用于多线程训练的线程数（仅 `CBOW` 支持）。
   - **调参**：通常设置为 CPU 内核数，加速训练。

---

### **二、训练优化参数**
5. **`sg`（或 `model`）**
   - **作用**：选择模型结构（`sg=1` 为 Skip-gram，`sg=0` 为 CBOW）。
   - **调参**：Skip-gram 适合小数据和语义任务，CBOW 适合大规模数据和词法任务。

6. **`hs`（Hierarchical Softmax）**
   - **作用**：是否使用分层 Softmax 优化。
   - **调参**：默认关闭（`hs=0`），若开启（`hs=1`）可替代负采样（`negative_sampling`）。

7. **`negative`**
   - **作用**：负样本采样数（负采样方法）。
   - **调参**：一般设为 5~20。数值越高，负采样越充分，但计算量增加。

8. **`sample`**
   - **作用**：对高频词的下采样比例（用于缓解常见词主导训练）。
   - **调参**：通常设为 0.001~0.00001，或直接关闭（`sample=0`）。

---

### **三、训练过程参数**
9. **`alpha`**
   - **作用**：学习率（初始值）。
   - **调参**：通常设为 0.025（默认值），若词向量质量不佳可尝试调小。

10. **`min_alpha`**
    - **作用**：学习率的最小值（线性衰减）。
    - **调参**：一般设为 0.0001，避免过度调小学习率。

11. **`epochs`**
    - **作用**：训练的轮数（对数据的迭代次数）。
    - **调参**：通常设为 5~20，过高的 epoch 可能导致过拟合。

12. **`batch_size`**
    - **作用**：批处理的样本数。
    - **调参**：若内存允许，可增大 batch_size 以加速训练。

---

### **四、调参建议**
- **数据规模**：
  - **小数据**：用 Skip-gram（`sg=1`）且增大 `window`。
  - **大数据**：用 CBOW（`sg=0`）且减小 `window`。
- **词频分布**：
  - 若存在大量低频词，提高 `min_count`。
  - 若语料中噪声较多，提高 `sample` 降低高频词影响。
- **任务适配**：
  - **词性标注/句法任务**：较小的 `window`（2-5）。
  - **文档分类/聚类**：较大的 `window`（5-10）。
- **资源限制**：
  - 若需平衡效果和速度，默认开启负采样（`hs=0`，`negative>0`）。

---

### **五、代码示例（Gensim）**
```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=processed_sentences,  # 预处理后的句子列表
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,                # 1 for Skip-gram, 0 for CBOW
    hs=0,                # 0 for negative sampling, 1 for hierarchical
    negative=5,          # 负采样数量（若 hs=0 时必填）
    sample=1e-4,         # 高频词下采样
    alpha=0.03,          # 学习率
    min_alpha=0.0001,    # 最小学习率
    epochs=10
)
```

---

### **六、总结**
| 参数类型       | 推荐值/范围        | 说明                          |
|----------------|-------------------|-----------------------------|
| `vector_size`  | 100~300           | 通常越高越好，但需权衡资源     |
| `window`       | 2~10              | 小窗口局部队义，大窗口语义     |
| `min_count`    | 5~10（小数据）     | 过滤低频词                   |
| `sg`           | 0（CBOW），1（Skip-gram） | 根据数据规模选择              |
| `negative`     | 5~20              | 仅当 `hs=0` 时有效           |
| `sample`       | 0.001~0.00001     | 缓解高频词干扰               |
| `alpha`        | 0.01~0.1          | 学习率初始值                 |

---

### **七、其他技巧**
- **早停法（Early Stopping）**：监控验证集损失，提前终止训练防止过劳。
- **词干化（Stemming）**：减少词形变化对模型的干扰。
- **子词信息（如 fastText）**：适合处理 OOV（未登录词）问题。
- **词向量评估**：通过类比任务（如 `man -> woman`，`king -> ???`）验证质量。

通过调整这些参数，可以更高效地训练出适配具体任务的词向量。实际应用中，建议从默认参数开始，逐步调参并结合验证结果优化模型。