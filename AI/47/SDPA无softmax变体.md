# SDPA无softmax变体

## 🧑‍💻 User

不用softmax的sdpa还有哪些

---
## 🤖 Assistant

“不用 softmax 的 SDPA”其实是个不小的家族。标准 SDPA 是

$$
\text{Attn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right)V
$$

去掉 softmax 后，核心改动无非是**换个归一化/激活、把 softmax 核线性化、或干脆不用 $QK^\top$**。按路线分大致如下。

## 1. 线性注意力（核方法，最主流）

把 softmax 核 $\kappa(q,k)=\exp(q^\top k/\sqrt d)$ 替换成可分解的 $\phi(q)^\top\phi(k)$，从而把 $O(N^2)$ 降到 $O(N)$：

$$
\text{Attn}=\frac{\phi(Q)\big(\phi(K)^\top V\big)}{\phi(Q)\big(\phi(K)^\top \mathbf 1\big)}
$$

- **Linear Transformer**（Katharopoulos 2020，《Transformers are RNNs》）：$\phi=\text{elu}+1$
- **Performer / FAVOR+**（Choromanski 2021）：用随机特征近似 softmax 核
- **Random Feature Attention**（Peng 2021）
- **cosFormer**（Qin 2022）：用 $\cos$ 重加权去逼近 softmax 的集中分布
- **Taylor 展开式线性注意力**：对 $\exp$ 做级数截断
- **DeltaNet / GLA / Lightning Attention** 等现代线性注意力

## 2. 用逐元素激活替代 softmax

保留 $QK^\top$，但把 softmax 换成别的激活：

- **ReLU attention**：$\mathrm{relu}(QK^\top/\sqrt d)$，“ReLU Strikes Back”（2024）
- **ReLU² / squared-ReLU**：**Primer**（So 2021）
- **Polynomial / 平方注意力**：如 $(\cdot)^2$ 或多项式核
- **Sigmoid / tanh / ELU attention**
- **Cosine / L2-normalized attention**（用余弦相似度代替缩放点积）

## 3. 门控 / SSM 风格

本质是带遗忘门的线性注意力，没有 softmax：

- **Gated Linear Attention (GLA)**（Yang 2024）
- **Mamba-2 / SSD**（Dao & Gu 2024，《Transformers are SSMs》）
- **RWKV、HGRN/HGRN2、RetNet** 等

## 4. 显式替代归一化函数（仍非 softmax）

- **sparsemax**（Martins & Astudillo 2016）：软化的 argmax，可产生稀疏分布
- **$\alpha$-entmax**（Correia 2019）
- **Sinkhorn attention**（Tay 2020）：用 Sinkhorn 迭代做双随机归一化
- **Hard attention / argmax**

## 5. 完全不用 $QK^\top$

- **Synthesizer**（Tay 2020）：用随机/可学习矩阵直接生成注意力权重
- **SOFT**（Lu 2021）：高斯核 + Nyström，无 softmax 的视觉 Transformer
- **SimA**（Kwak 2023）：简单 softmax-free 注意力，用于 ViT

## 6. 容易被误认的“近似/加速”方法（仍用 softmax）

这些**不是** softmax-free，注意区分：

| 方法 | 是否用 softmax |
|---|---|
| FlashAttention | ✅ 用（只是 IO 优化） |
| Linformer | ✅ 用（低秩投影 $K,V$） |
| Nyströmformer | ✅ 用（Nyström 近似，内部仍是 softmax） |
| Performer | ❌ 不用（核线性化） |

## 一句话总结

主流分三条线：**① 核线性化**（Performer / Linear Transformer / cosFormer）、**② 换激活**（ReLU / ReLU² / 余弦）、**③ 换归一化或不用 $QK^\top$**（sparsemax / entmax / Sinkhorn / Synthesizer / SOFT）。近期工程上最活跃的其实是第 1、3 类（GLA、Mamba-2、DeltaNet 等线性/门控注意力）。

需要的话，我可以针对某一类给出 PyTorch 实现或按复杂度/精度做对比表。
