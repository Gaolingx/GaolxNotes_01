# Transformer MLP通常仅有一个Norm

## 🧑‍💻 User

transformer的mlp是前后都有norm吗

---
## 😀 默认助手 (7)

## 简短回答

**默认情况下不是。** 主流 Transformer 中的 MLP 子层（FFN）**只有一个 Norm**，位置要么在**前面**（Pre-LN），要么在后面（也就是残差相加之后，Post-LN），不会同时放在前后。所谓“前后都有”是一种少数变体，称为 **Sandwich-LN**，一般被称为早期稳定性技巧或某些新模型的特殊设计。

---

## 两种标准范式

以 MLP 子层为例（Attention 子层的写法完全相同）：

<div align="center">

**Post-LN**（原始 Transformer, Vaswani 2017）

$$x_{l+1} = \mathrm{LN}\Big(x_l + \mathrm{MLP}(x_l)\Big)$$

**Pre-LN**（GPT-2 之后几乎所有 LLM：GPT-3、LLaMA、Qwen、Mistral…）

$$x_{l+1} = x_l + \mathrm{MLP}\big(\mathrm{LN}(x_l)\big)$$

</div>

注意 Post-LN 里的 LN 是在**残差相加之后**，作用是归一化整个子层的输出，而不是“在 MLP 后面再补一个 norm 来稳定 MLP 内部”。因此在只有一条残差支路的情况下，两种写法的 Norm 数量一样，都是 1 个/子层。

> Pre-LN 成为主流的原因：梯度可以直接通过恒等残差支路回到输入，训练不需要 warmup 也能稳定；代价是深层网络的表示范数会随层数增长，所以通常要加一个**最终 Norm**（`final_norm` / `ln_f`）再接 `lm_head`。这个最终 Norm 是全局的，不属于 MLP。

---

## 确实存在“前后都有”的例外

<details>
<summary><b>1. Sandwich-LN（CogView, 2021）</b></summary>

为解决 Pre-LN 深层训练时的数值爆炸而提出，在 Pre-LN 基础上**额外**在子层输出（残差相加前）加一个 LN：

$$x_{l+1} = \mathrm{LN}\Big(x_l + \mathrm{MLP}\big(\mathrm{LN}(x_l)\big)\Big)$$

这是最字面意义上的“前后都有 norm”。论文报告它比 Pre-LN 更稳定，但多了一次归一化开销和一层额外的非线性，后续少有人沿用。

</details>

<details>
<summary><b>2. Gemma 2 的 “sandwich norm”</b></summary>

Gemma 2 在**每个**子层（attention 和 MLP）外层都加了一个 RMSNorm，形式上是：

$$x_{l+1} = x_l + \mathrm{RMSNorm}_{post}\Big(\mathrm{MLP}\big(\mathrm{RMSNorm}_{pre}(x_l)\big)\Big)$$

与 CogView 的区别是：post-norm 放在**残差支路内部**，残差本身仍是干净的恒等连接（keep the residual clean），所以既不破坏梯度直通，又能抑制激活值增长。这是目前少见的、在现代大模型里实际使用“前后都 norm”的案例。

</details>

<details>
<summary><b>3. 容易混淆的其他 norm（都不是 MLP 的前后 norm）</b></summary>

- **QK-Norm**：对 attention 的 $q,k$ 各做一次 RMSNorm（ViT-22B、Chameleon 等），作用点不在 MLP。
- **Final norm**：整个网络最后、`lm_head` 之前的那一次归一化，所有 LLaMA 系模型都有。
- **MLP 内部**（`up_proj` 与 `down_proj` 之间）加归一化：主流 LLM 基本**没有**，属于罕见的非标准做法。
- **归一化位置 ≠ 归一化对象**：Post-LN 的 LN 数学上“在 MLP 之后”，但它归一化的是残差和，不是 MLP 输出本身。

</details>

---

## 实践建议

| 目标 | 选择 |
|---|---|
| 从零训练、想省事且稳定 | **Pre-LN** + 最终 Norm（事实标准） |
| 极深 / 大 lr / 训练发散 | 考虑 **Sandwich norm**（Gemma 2 式，post-norm 在残差内）或 QK-Norm |
| 复现 BERT / 原论文 | Post-LN，注意需要 warmup 或 post-LN 初始化技巧 |
| 微调现成模型 | 跟随原架构，不要自行增删 Norm，它会影响预训练权重的尺度假设 |

一句话总结：**标准做法是每个子层 1 个 Norm（Pre 或 Post），“前后都有”只在 Sandwich-LN、Gemma 2 这类刻意加稳定性的设计里出现；MLP 两个线性层之间通常不加 Norm。**
