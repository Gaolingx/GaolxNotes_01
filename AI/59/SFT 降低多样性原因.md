# SFT 降低多样性原因

### 🧑‍💻 User

对llm进行fine tune，我总感觉基于sft的方式几乎一定会降低多样性，导致一个模型总是生成一种风格的答案，因为sft数据是预训练分布的一个子集。。。整个分布的支撑集都变小了，有什么理论依据可以支持吗？

---

### 🤖 Assistant

以下回答从概率分布、信息论、统计学习、以及当前对 LLM 微调的理论理解四个角度，说明为何**SFT（监督微调）确实会不可避免地降低生成多样性**，并给出可引用的理论依据与机制解释。

---

# 1. 从概率分布角度：SFT = 强制把模型的条件分布收缩到数据子分布上

预训练语言模型学习的是接近自然语言分布的 **最大似然估计**，使得模型近似：

[
p_{\text{pretrain}}(y|x) \approx p_{\text{natural}}(y|x)
]

而 SFT 的目标是让模型拟合 SFT 数据集的分布：

[
p_{\text{SFT}}(y|x)
]

但 SFT 数据天然是 **高度稀疏、风格统一、意图明确的子分布**。例如指令数据都是简短、规范化、风格统一的“任务式”对话。
那么优化目标变成了：

[
\theta^{*}=\arg\max_\theta \mathbb{E}*{(x,y) \sim p*{\text{SFT}}} \log p_\theta(y|x)
]

这一范式下会发生：

### **分布支撑集（support）收缩现象**

预训练时的原始分布的支持集为：

[
\text{Supp}(p_{\text{pretrain}}) = {\text{所有自然语言可能的输出}}
]

而 SFT 数据是一个严格子集：

[
\text{Supp}(p_{\text{SFT}}) \subset \text{Supp}(p_{\text{pretrain}})
]

MLE 优化会把概率质量集中在这个子集上，导致：

* 模型自动压低了不在 SFT 数据中的表达方式
* 模型更倾向于复制 SFT 的风格和模式
* **多样性减少、输出模式化**

这是概率论上最直接的解释：
**MLE + 子分布 = 支撑集收缩 = 多样性损失。**

---

# 2. 从信息论角度：SFT = 向子空间投影

预训练 LM 的表达可以视为学习到一个丰富的语言流形。
SFT 的目标是把模型映射到“指令式”语言流形：

[
\mathcal{M}*{\text{SFT}} \subset \mathcal{M}*{\text{pretrain}}
]

这相当于一种 **信息投影 (information projection, I-projection)**：
模型为了最小化 KL(p_SFT || p_model)，会朝 p_SFT 的高密度区域收缩，从不同方向来的概率质量被削弱。

信息论上，两种投影都会失去多样性：

* **I-projection**：牺牲覆盖率，偏向子分布
* **M-projection**：牺牲精度，扩大覆盖（RLHF 更偏向这一方向）

SFT 属于 I-projection，因此必然：

**减少覆盖范围 → 降低表达多样性。**

---

# 3. 从统计学习角度：SFT 是一种过拟合风险极高的 fine-tuning

SFT 通常数据量小（几十万到几百万），而模型具有数百亿参数。
典型后果：

1. **模型会过拟合语言风格（style overfitting）**
   产生“统一的答案结构”“统一的格式化表达”。

2. **梯度更新大多数来自同一类 prompt**
   导致：

   * 表达方式高度同质化
   * 少见行为被遗忘（catastrophic forgetting）

这一点已经在多篇论文中有明确观测，例如：

* SFT 会使大模型出现 **mode collapse**（模式坍缩）
* 微调对输出熵（entropy of output distribution）有显著下降

---

# 4. 从 LLM 微调理论视角：SFT = 降低熵 (entropy reduction)

如果你把模型输出视为一个分布，预训练模型输出的熵很高（多种可能的回答）。
而 SFT 则推动模型最大化某特定 y 的概率，使得：

[
H(p_{\text{SFT}}(y|x)) < H(p_{\text{pretrain}}(y|x))
]

实验结果通常显示 SFT 后：

* 输出熵显著下降
* 高概率质量集中到窄范围的输出模式
* 多样性 metrics（Self-BLEU, Distinct-n）下降

**这是目前微调理论中最明确的证据：
SFT 的优化目标本质上就是降低输出分布的熵。**

---

# 5. 为什么 RLHF/ORPO/DPO 通常比纯 SFT 保持更好多样性？

因为：

* RLHF/DPO 的本质不是预测唯一答案
* 而是优化排序或偏好，使模型在多个候选之间做选择
* 没有强制压缩分布的支持集

因此理论和经验都表明：

* SFT → 输出分布收缩，模式化
* RLHF/DPO → 保留多样性，更接近预训练分布

---

# 总结：SFT 造成多样性下降的理论依据

你直觉完全正确，而且有非常坚实的理论支撑：

1. **概率支撑集收缩理论**（SFT 数据是预训练分布的子集）
2. **信息论的 I-projection / KL 最小化机制**
3. **统计学习中的过拟合与 catastrophic forgetting**
4. **语言模型输出熵降低（entropy reduction）**
5. 实证研究表明 SFT 造成模式坍缩（mode collapse）

**因此，SFT 几乎必然降低模型的多样性。**

---

如果你感兴趣，我可以补充：

* 具体实验数据：SFT 前后模型输出的熵和 diversity metrics 变化
* 多样性损失的可视化（t-SNE embedding shift）
* 如何设计一个 SFT pipeline 来最大限度保持多样性（混合训练、freeze 技巧等）

你希望深入哪一部分？
