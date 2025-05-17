# LLM推理框架KV Cache大小计算公式

### 🧑‍💻 User

llm推理框架的kv cache大小和模型的哪些参数有关（例如注意力头个数、序列长度等），并给出计算公式？

---

### 🤖 Assistant

KV Cache的大小与模型的**层数（num_layers）**、**注意力头数量（num_heads）**、**每个头的维度（head_dim）**、以及**最大序列长度（max_seq_len）**直接相关。其计算公式如下：

---

### **1. KV Cache的组成**
每个注意力头的 Key 和 Value 矩阵需要分别存储为缓存。对于每个头，其 Key 和 Value 的维度均为 `[max_seq_len, head_dim]`。因此，**单个头的KV缓存大小为**：
$$
2 \times \text{max\_seq\_len} \times \text{head\_dim} \quad (\text{2表示Key和Value})
$$

---

### **2. 单层的KV Cache大小**
每个层包含 `num_heads` 个头，因此单层的KV缓存大小为：
$$
2 \times \text{num\_heads} \times \text{max\_seq\_len} \times \text{head\_dim}
$$

---

### **3. 全部层的KV Cache总大小**
总共有 `num_layers` 层，因此 **KV Cache的总大小** 为：
$$
\text{KV Cache大小} = 2 \times \text{num\_layers} \times \text{num\_heads} \times \text{head\_dim} \times \text{max\_seq\_len}
$$

---

### **4. 与d_model的关系**
若模型的参数中提供 **隐藏层维度**（d_model），而 `head_dim = d_model / num_heads`，则公式可简化为：
$$
\text{KV Cache大小} = 2 \times \text{num\_layers} \times \text{d\_model} \times \text{max\_seq\_len}
$$

---

### **5. 参数影响总结**
| 参数 | 对KV Cache的影响 | 说明 |
|------|------------------|------|
| **层数（num_layers）** | 成正比 | 层数越多，缓存越大 |
| **注意力头数（num_heads）** | 成正比 | 头数增加时，缓存线性增长（若head_dim不变） |
| **头维度（head_dim）** | 成正比 | 每个头的维度越大，缓存越大 |
| **序列长度（max_seq_len）** | 成正比 | 序列越长，缓存越大 |
| **d_model（若提供）** | 成正比 | d_model越大，缓存越大 |

---

### **6. 示例计算**
假设：
- num_layers = 12（如BERT-base）
- num_heads = 12
- head_dim = 64（d_model = 12 × 64 = 768）
- max_seq_len = 512

则：
$$
\text{KV Cache大小} = 2 \times 12 \times 12 \times 64 \times 512 = 9,437,184 \quad (\text{约9.4MB，若以FP16存储则需乘以2字节，总约18.8MB})
$$

---

### **7. 关键结论**
- KV Cache的存储需求与 **序列长度（max_seq_len）** 和 **模型复杂度**（层数、头数、维度）直接相关。
- 在相同模型规模下，长文本（大max_seq_len）会显著增加内存占用。
- 优化建议：减少序列长度、使用低秩注意力（LoRA）或分块缓存（Paged Attention）可降低KV Cache开销。