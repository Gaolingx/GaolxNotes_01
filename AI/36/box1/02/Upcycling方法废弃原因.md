# Upcycling方法废弃原因

### 🧑‍💻 User

Upcycling 是一个将dense model转化为MoE model的方法，具体做法是我们复制dense model中的FFN layer得到对应MoE layer中的Expert，然后我们再结合router训l练，这样可以提高整体的训l练效率。相关模型有MiniCPM，Qwen1.5和Mixtral MoE（疑似），目前这种方法为什么已经被废弃了？

---

### 🤖 Assistant

下面是对 **为什么 upcycling（从 dense 模型复制 FFN 参数转成 MoE，再加 router 训l练）在最新大规模 MoE 模型训练中逐渐被认为“不足/弃用”的主要原因** 的全面整理，基于近一年多学术和工程实践的观察和研究进展：([arXiv][1])

---

## 一、什么是 upcycling

简要回顾一下定义（与你描述一致）：

* Upcycling 指从已经**预训练好的 dense 模型开始**构造 MoE 模型；
* 具体做法是复制 dense 模型中的 FFN 层作为 MoE 的多个 expert 初始化，然后加上 router（稀疏门控）；
* 其余部分（注意力层、norm、嵌入等）直接从 dense checkpoint 拷贝；
* 最后对带 router 的新架构进行训l练。([OpenReview][2])

这种做法在早期 MoE 探索中被用来**降低从 scratch 训练 MoE 的计算成本**。

---

## 二、Upcycling 方法的优点（但不是决定性优势）

Upcycling 在早期实验中确实展现了一些积极表现，如：

1. **初期性能提升且比从零开始训练更快收敛**（因为复用已有的 dense 表征）；
2. 可以减少部分训l练 FLOPs 和时间成本；
3. 有助于在资源有限情况下“快速过渡”到 MoE 架构。([datahacker.rs][3])

这些优势在较小规模或中等规模任务上是明显的。

---

## 三、为什么这种方法逐渐被认为不够理想甚至“被废弃”的核心原因

### 1. Upcycling 会导致训练后期表现不佳，难以发挥 MoE 潜力

最新研究（例如 ICLR 2025 的 Drop-Upcycling 论文）明确指出：

* Upcycled MoE 模型 **起始表现虽然不错，但后期训练进度明显慢**；
* 与从 scratch 训练的 MoE 相比，**长期训l练的性能会比从 scratch 差**；
* 这表明 Upcycling 初始化带来的 **模型容量利用与专家分化（expert specialization）不足**，无法充分发挥稀疏架构在大规模数据上的优势。([arXiv][1])

总结来说，**Upcycling 的优势主要是初始阶段**，而**在大规模长期训l练中存在渐进的性能劣化趋势**。

---

### 2. Dense 复制 FFN 作为专家，会导致专家 specialization 不够

经典的 upcycling 是直接复制 FFN 层权重作为多个 expert：

* 这种方式会导致每个 expert 在初始化时的能力非常相似；
* 相邻专家间缺乏显著差异，使得路由器难以引导不同 expert 各自“精专化”不同语义空间；
* 结果可能是 “多数 token 被分配到少数专家”，专家利用不均衡。([Medium][4])

后续研究发现，为了让 MoE 专家之间真正分工（task-driven specialization），需要更复杂的初始化或正则化，而不是简单复制 dense FFN。

---

### 3. Training 动力学与路由优化更困难

Upcycling only **fixes expert initialization**，但真正重要的是路由器（router）如何学会有效 dispatch：

* router 参数是 **随机初始化**，但对于 MoE 来说，**routing 训l练是整个系统中最关键的部分**；
* 初始专家差异不明显会导致 **routing 训l练停滞或变慢**；
* 结果是 upcycled MoE 对 gating 训l练相对敏感，训l练更慢，甚至在大规模数据上鲁棒性不足。([OpenReview][5])

这同样解释了为什么研究者提出 Drop-Upcycling 等改进措施，通过部分 re-init 或正则化来缓解。([arXiv][1])

---

### 4. 在大规模训练（Hundreds Billions Tokens）上，upcycled 模型往往被更好方法替代

实际大模型训练经验显示：

* 直接从 scratch 或改进初始化（如虚拟分组、权重缩放等）比简单 upcycling 更稳定；
* 许多最新 MoE 模型（行业级别，例如 Mistral 8x7B、Qwen 系列 MoE）**在正式产品/学术论文中并不强调 pure upcycling**，反而强调更先进的初始化和 routing策略；
* 像 Qwen1.5-MoE、Mixtral 等，即便采用 upcycling 的思想，其实现都包含额外机制来改善专家差异化，而不是简单复制。([Qwen][6])

因此在大规模训练中，**“从密集模型复制 FFN 直接转 MoE”越来越少被单独使用**。

---

## 四、总结：Upcycling 被认为不理想的几个技术层面因素

| 问题                   | 原因                           |
| -------------------- | ---------------------------- |
| 后期性能欠佳               | Upcycling 较慢收敛、难以充分利用 MoE 容量 |
| 专家 specialization 不足 | FFN 复制导致专家特征太过相似             |
| 路由训l练问题              | 对随机初始化 router 敏感，训l练更难       |
| 随着规模增长收益递减           | 在大规模长期训l练中效果不如更先进方案          |

总体来说，**Upcycling 作为一个过渡性策略在早期 MoE 探索中很有价值，但对于需要大规模、大训练量和高性能的工业级模型，它已被更优化的初始化、专家差异化方法和更强的 routing 训练策略所取代**。([arXiv][1])

---

## 五、当前研究趋势

最新工作不是完全放弃 upcycling，而是在其基础上引入改进，例如：

* **Drop-Upcycling**：部分 re-init 以促进专家 specialization；
* **Dirichlet-Prior Shaping**：引导 routing 概率分布提高专家之间差异；
* **虚拟 group 初始化 & 权重缩放策略**：更细粒度的 upcycling；([arXiv][7])

这些都表明，**社区对 upcycling 的核心思想仍有兴趣，但简单复制并训l练的方案已经不足以满足大规模 MoE 的训练需求**。

---

如有需要，我可以进一步整理相关论文链接和具体超参数设置对比表。

[1]: https://arxiv.org/abs/2502.19261?utm_source=chatgpt.com "Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization"
[2]: https://openreview.net/pdf/26daacd90342fd132fdecb046ebd41f634bedd17.pdf?utm_source=chatgpt.com "Upcycling Instruction Tuning from Dense to Mixture-of-Experts via"
[3]: https://datahacker.rs/llms-scratch-004-mixture-of-experts-moe-models-the-architecture-powering-2025s-best-ai-systems/?utm_source=chatgpt.com "LLMs Scratch #004: Mixture of Experts (MoE) Models: The Architecture Powering 2025's Best AI Systems - Master Data Science 25.11.2025"
[4]: https://bhakta-works.medium.com/efficient-model-expansion-upcycling-dense-language-models-into-sparse-mixture-of-experts-fd1a75a62443?utm_source=chatgpt.com "Efficient Model Expansion: Upcycling Dense Language Models into Sparse Mixture-of-Experts Architectures | by Bhakta Vaschal Samal | Medium"
[5]: https://openreview.net/pdf/161bbf8b9a237f60ba0aee05309543ab2c3f0b1b.pdf?utm_source=chatgpt.com "Published as a conference paper at ICLR 2025"
[6]: https://qwenlm.github.io/blog/qwen-moe/?utm_source=chatgpt.com "Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters | Qwen"
[7]: https://arxiv.org/abs/2510.01185?utm_source=chatgpt.com "Dirichlet-Prior Shaping: Guiding Expert Specialization in Upcycled MoEs"
