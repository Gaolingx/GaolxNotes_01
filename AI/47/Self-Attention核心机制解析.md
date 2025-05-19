# Self-Attention核心机制解析

### 🧑‍💻 User

Self-Attention 的核心是什么？

---

### 🤖 Assistant

Self-Attention 的核心在于其 **动态计算输入序列中不同位置之间依赖关系的能力**，通过以下机制实现：

---

### **1. 查询-键-值（QKV）框架**
- **输入到Q/K/V的线性投影**：通过可学习参数将输入序列分别映射为**Query（Q）**、**Key（K）**、**Value（V）**。
- **点积度量相似性**：通过 $ QK^T $ 计算不同位置的相关性，捕捉长距离依赖（如句子中 "猫" 和 "抓破" 的跨句依赖）。
- **Softmax 归一化**：将点积结果转换为概率分布（注意力权重），强调强相关项，抑制弱相关项。

---

### **2. 并行化与全局建模**
- **并行性**：直接通过矩阵运算一次性计算所有位置的注意力权重，突破RNN的时序依赖限制。
- **全局感受野**：每个位置都能直接感知整个序列的信息（如文本中的全局依存，图像中的长距离目标关系）。

---

### **3. 多头注意力（Multi-Head）**
- **子空间分解**：将Q/K/V线性投影到多个子空间（头），每个头独立计算注意力模式。
- **多子空间融合**：通过拼接与线性变换，融合不同头捕捉的多样化依赖（如语言中不同粒度的句法/语义关系）。

---

### **4. 可微分建模**
- **端到端训练**：注意力权重通过梯度反向传播优化，自适应地为不同任务分配关注点。
- **软注意力（Soft）与硬注意力（Hard）**：根据任务需求选择可微（Soft）或离散（Hard）注意力机制。

---

### **5. 归一化与位置信息**
- **位置编码（Positional Encoding）**：通过正弦/余弦函数或学习的位置嵌入，为Self-Attention注入序列位置信息。
- **Layer Normalization**：稳定模型训练，缓解梯度爆炸/消失问题。

---

### **核心优势**
- **长程依赖建模**：直接捕捉任意距离的依赖关系（如 $ \text{log}(N) $ 层Transformer可建模 $ O(N) $ 距离依赖）。
- **灵活性**：可适用于NLP、CV、语音等多种模态（如图像Transformer中将图像分块为序列块）。
- **可解释性**：注意力权重提供模型决策的部分可解释性（如定位文本中的关键支撑词）。

---

### **数学形式化（以点积注意力为例）**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)  # QK^T / sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)  # Apply mask by adding -infinity
    attn = tf.nn.softmax(scores)  # Softmax(QK^T / sqrt(d_k))
    context = tf.matmul(attn, V)  # Context vector = softmax * V
    return context
```

---

### **典型应用场景**
- **自然语言处理**：机器翻译中直接建模“the cat sat on the mat”中“cat”与“mat”的长距离依赖。
- **计算机视觉**：通过视觉Transformer（ViT）捕获图像中远距离目标的上下文关系。
- **语音识别**：对齐语音帧与文本中的音素（如Wav2Vec 2.0）。

---

### **与传统注意力的区别**
- **输入依赖**：传统注意力（如Seq2Seq）中注意力权重由编码器状态与当前解码器状态计算，而Self-Attention中Q/K/V均来自输入序列。
- **无显式位置信息**：需通过位置编码弥补Self-Attention平移不变性的缺陷。

---

### **数学视角的延伸**
Self-Attention 可形式化为：
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
其中 $ Q = XW^Q, K = XW^K, V = XW^V $，参数 $ W^{*} $ 在训练中优化，X为输入嵌入。

这一框架使模型能够**自适应地聚焦于输入中的关键区域**，成为Transformer架构在多种任务中取得SOTA性能的核心机制。