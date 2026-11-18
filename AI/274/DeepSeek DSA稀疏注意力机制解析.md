# DeepSeek DSA稀疏注意力机制解析

## 🧑‍💻 User

解释下deepseek的dsa
他这个可以降低成本，用低成本的全局摘要维持信息覆盖，再用动态选择的原始内容补回关键细节。
能这么理解吗

---
## 😀 默认助手 (7)

## 先说结论

**“两段式”的骨架是对的；“摘要”这个说法是错的。**

DSA（DeepSeek Sparse Attention，随 DeepSeek-V3.2-Exp 在 2025-09 发布）的结构确实是：**先用极低成本对全部历史做一次打分扫描，再只对被选中的原始 token 做全精度注意力计算**——这在计算结构上就是“粗排 + 精排”，你的直觉没错。

但它**没有摘要**：不存在任何压缩表示。第二步不是“补回细节”，而是“细节从来就没被压缩过，没被选中的则被彻底丢掉”。

---

## 一、DSA 的真实机制

每个 MLA 层旁边挂一个轻量 **lightning indexer**，两层动作：

**第 1 步：低成本全局打分（粗排）**

对每个 query 位置 $t$，给所有历史 token $s \le t$ 算一个相关性分数：

$$I_{t,s}=\sum_{j=1}^{H^{I}} w_{t,j}\cdot \mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot \mathbf{k}^{I}_{s}\right)$$

- $H^{I}=64$ 头、$d^{I}=128$ 维，**FP8** 计算，K 侧只做单头投影 + LayerNorm + Hadamard 旋转（为了 FP8 量化友好），query 侧**复用 MLA 的 query LoRA latent**，几乎不额外花钱。
- 这一步确实是 $O(L^2)$ 的：每个 query 都扫全部历史。但单对的计算量只有主注意力的几分之一且是 FP8，所以常数极小。

**第 2 步：Top-k 选择 + 原始精度精算**

$$\mathcal{S}_t=\mathrm{TopK}_{s\le t}\left(I_{t,s}\right),\quad k=2048$$

$$\mathbf{o}_t=\mathrm{softmax}_{s\in\mathcal{S}_t}\!\left(\frac{\mathbf{q}_t\cdot\mathbf{k}_s}{\sqrt{d}}\right)\mathbf{v}_s$$

关键点：**这里用的是真的 MLA 分数、真的原始 KV、真的 softmax**，不是近似值。未被选中的 $s$ 被 mask 成 $-\infty$，权重恰好为 0。

工程上还有两个细节：
- 选择粒度是**「每层、每个 token 位置一次」**，128 个 query head **共享**同一个 top-k 集合（indexer 用可学习的 per-head 权重 $w_{t,j}$ 把多头压成一个标量分数）。
- decode 时 MLA 走 **MQA 模式**（KV latent 共享，$512+64$ 维，fp8 KV cache），所以带宽也省。

> 所以复杂度是 $O(L\cdot k)$ 的主注意力 + $O(L^2)$ 的廉价索引。**L 越大收益越大，短文本下收益有限**——这就是为什么官方主打的是长文本降价。

---

## 二、逐句校准你的理解

| 你的说法 | 判定 | 说明 |
|---|---|---|
| 可以降低成本 | ✅ 完全正确 | 长上下文 prefill/decode 成本大幅下降，官方同步把 API 价格砍了一半以上（输出降幅更大），对长 CoT 的 RL rollout 也很关键 |
| 低成本地维持信息覆盖 | ⚠️ 半对 | “低成本”对，“全局”对（确实扫了全历史），“维持信息覆盖”是**统计意义上的**（训练让 indexer 大概率选中重要的），**不是表示意义上的**（没有任何压缩表示兜底） |
| 用全局摘要 | ❌ 错 | 全局那一步的产物是**索引分数 / 位置下标**，一个路由信号，不是“内容”，不参与 attention，不可复用；系统里不存在任何 summary 向量 |
| 动态选择原始内容 | ✅ 完全正确 | 这是 DSA 的核心，也是“fine-grained sparse attention”这个说法的由来 |
| 补回关键细节 | ❌ 措辞不成立 | 细节不是“补回”，而是**从未经过有损压缩**。被选中的 token 直接精确计算；没被选中的 token 是**硬性丢弃**，没有“摘要”替它兜底 |

**风险点也在这里**：DSA 的失效模式是**召回失败**，也就是 indexer 漏掉关键 token 就会直接损失信息，影响无法挽回。而“摘要 + 细节”的架构里，漏掉的细节至少还有压缩表示可以保底。DeepSeek 的应对办法是把 indexer 训练好：先 dense warm-up（用 KL 散度把 indexer 的分数分布对齐到真实注意力分布），再切稀疏训练。所以它不是推理期的 plug-in，而是**要重新训练的架构改动**——这也是它挂 `-Exp` 的原因。

---

## 三、一句话的准确版本

> DSA 用**极低成本的索引打分对全部历史做一次粗排**，再**只对 top-k 个原始 token 做精确注意力**；信息覆盖是**概率性的软保证**，不是靠摘要兜底的硬保证。

或者用检索的类比：**它不是“摘要 + 回表”，而是“向量粗排 + 原文精排”；没有摘要层，只有 ANN 式的召回。**

---

## 四、你这段话其实更像在说 NSA

如果你印象中确实是“有一个全局摘要分支”，那描述的更接近 DeepSeek 早半年的 **NSA（Native Sparse Attention，2025-02）**——它是**三分支并行**的：

| | NSA（三分支，并行） | DSA（V3.2-Exp，两段式，串行） |
|---|---|---|
| 全局覆盖 | ✔ **压缩分支**：把 token block 过 MLP 压成表示，这才是真正的“低成本全局摘要”，并**参与** attention | ✘ 没有。只有 indexer 的标量分数，不参与 attention |
| 关键细节 | ✔ **选择分支**：top-k 个原始 KV block（block 粒度） | ✔ top-k=2048 个原始 token（token 粒度，更细） |
| 局部 | ✔ 滑窗分支 | ✘ 靠 top-k 自己覆盖（近期 token 通常分高） |
| 漏选后果 | 有压缩表示兜底 | 直接丢，权重为 0 |

可以理解为：DSA 把 NSA 的“压缩分支”换成更省钱的“索引打分”，只保留“选择”这条线，从而换得更极致的工程效率。**“低成本全局摘要 + 动态选择原始内容”这句话，作为 NSA 的描述基本准确，作为 DSA 的描述不成立。**

<details>
<summary>附：可核对的硬数字（来自官方 config 与 inference 代码）</summary>

```
index_n_heads   = 64        # 索引器头数
index_head_dim  = 128       # 索引器 head 维度
index_topk      = 2048      # 每个位置保留的 token 数
q_lora_rank     = 1536      # 索引器 query 复用这个 latent
kv_lora_rank    = 512       # MLA KV latent（MQA 共享）
qk_rope_head_dim= 64
n_heads         = 128       # 主注意力头数
n_layers        = 61
dtype           = fp8 (ue8m0 scale)
```

对照代码里的 `Indexer.forward()`：分数由 `fp8_index(q_fp8, weights, k_cache, k_scale)` 得到，形状 `[b, s, t]`（头维已加权求和 → 全头共享一个 top-k），再 `index_score.topk(min(index_topk, end_pos))`，最后 `scatter_(-1, topk_indices, 0)` 把选中位置置 0、其余 $-\infty$ 加到真实 attention score 上。**清晰可控，但没有任何“摘要”张量**。
</details>

<details>
<summary>附：DSA 省的是什么、不省什么（常见误解）</summary>

- **省**：attention 的 FLOPs（$O(L^2)\to O(Lk)$）、HBM 上 KV 的读取带宽（只读 top-k 的 KV）→ 长文本下 prefill 和 decode 都变快。
- **不省**：KV cache 的**显存占用**并没有减少，因为全部 token 的 KV latent 仍然都要存下来（否则后期无法被重新选中）；索引器自己还额外带来一份 128 维 fp8 的 k_cache（相对 640B/token 的 KV 大约多 20%）。
- 因此“稀疏注意力 = KV cache 变小”是错的；**DSA 省的是计算和带宽，不是显存。**
</details>

如果你看的是某篇讲 DSA 的中文文章，写了“全局摘要”，大概率是把 NSA 的分支结构套到 DSA 上了——这两个概念在传播里经常被混在一起。
