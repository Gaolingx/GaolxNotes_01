# FP64训LLM：成本巨大收益为零

## 🧑‍💻 User

用FP64训LLM会发生什么

---
## 😀 默认助手 (7)

# 用 FP64 训练 LLM：结论与细节

**一句话结论：** 几乎不会得到任何有价值的变化，训练照样收敛、loss 曲线几乎重合，但你会付出约 $2\times$ 显存和 $10\sim30\times$ 时延的代价。工程上这基本等于“把训练成本放大一个数量级，换取零可测量的质量收益”。

---

## 一、会发生什么：三个层面

### 1. 数值层面：和你想象的相反

FP64 提供的是 **52 位尾数（约 15–16 位十进制有效数字）**，而不是“更宽的数值范围”。但 LLM 训练里真正的数值敌人是：

| 问题 | 根因 | FP64 有用吗 |
|---|---|---|
| attention logits 溢出 / loss spike | 数值**范围**太大 | ❌ 无效。这是 `exp` 溢出，用 max-subtraction 和 BF16 的宽范围解决 |
| 梯度下溢 / 上溢 | 范围 | ❌ FP32/BF16 的指数位（8 bit）跟 FP64（11 bit）比完全够用 |
| 长 $K$ 维累加、长序列 reduction 的精度损失 | 累加误差 | ✅ 有用，但**FP32 累加 + 分块**就够（Tensor Core 本来就是 FP32 accumulate） |
| 优化器更新被“吃掉”（$\Delta w / w$ 太小） | 尾数不够 | ⚠️ 理论上有用，但混合精度里 master weight 已是 FP32，$\Delta w/w \sim 10^{-3}$，FP32 相对误差 $10^{-7}$，完全无损 |

所以 FP64 提升的是**“已经不被需要的那一项”**。实践中，FP32 与 BF16 混合精度的收敛曲线差异在噪声量级以内；社区大量消融（含 Intel 那篇经典的 BF16 训练评估）都表明 BF16 + FP32 accumulate 与纯 FP32 训练终态 loss 无实质差别。

### 2. 工程层面：很多基础设施会失效

<details>
<summary><b>展开：哪些东西会退化 / 不可用（实践中真正麻烦的部分）</b></summary>

- **FlashAttention 不支持 FP64**。你会退回朴素 attention，显存变 $O(N^2)$ —— 在长上下文下直接 OOM，这一条比算力损失更致命。
- **融合算子大面积缺失**：fused Adam、fused LayerNorm、RMSNorm 的 CUDA kernel、大部分 Triton kernel 都只写 FP16/BF16/FP32 路径，FP64 会走 fallback，甚至直接报错。
- **Tensor Core 基本用不上**。A100/H100 有 FP64 Tensor Core，但要让 cuBLAS 真正走到那条路径需要特定 shape 和配置；否则就是 CUDA Core 标量 FP64。
- **通信量翻倍**：梯度 all-reduce、ZeRO 的 parameter/gradient 分片全是 2× 字节，NCCL 压力翻倍，在万卡规模上这是实打实的墙钟时间。
- **算子库的精度假设被打破**：很多 kernel 假定 `accumulate in FP32`，传 FP64 进去要么类型报错，要么静默做 cast（那你就白付了成本）。
</details>

### 3. 成本层面：算力断崖

数据中心 GPU 的 FP64 吞吐是刻意做低的（消费卡更是直接砍到 1/64）：

| GPU | FP64（向量 / Tensor Core） | BF16 Tensor Core（dense） | 差距 |
|---|---|---|---|
| A100 SXM | 9.7 / 19.5 TFLOPS | 312 TFLOPS | $\sim16\text{–}32\times$ |
| H100 SXM | 34 / 67 TFLOPS | 989 TFLOPS | $\sim15\text{–}29\times$ |
| RTX 4090 | **1.3 TFLOPS**（1/64） | 165 TFLOPS | $\sim128\times$ |

注意：**BN 层/Adam/softmax 这类带宽敏感算子只慢 $2\times$**，但大矩阵乘主导了 FLOPs，所以整体墙钟大概 10–30× 慢，纯 matmul 密集的小模型会顶到 30×。原本训 1 个月的 run 变成 2–3 年——这才是真正让人放弃的原因。

<details>
<summary><b>展开：7B 模型的显存账本</b></summary>

$$
\text{参数}=7\times10^9
$$

**标准 BF16 混合精度 + Adam：**

$$
\underbrace{14}_{\text{BF16 }W}+\underbrace{14}_{g}+\underbrace{28}_{\text{FP32 master}}+\underbrace{56}_{m,v}\approx 112\ \text{GB}
$$

**全 FP64：**

$$
(56\ \text{GB})\times\{\,W,\ g,\ m,\ v\,\}=224\ \text{GB}
$$

即 **$2\times$** 模型状态显存，激活值同样 $2\times$。所以常见结局是：单卡直接 OOM，必须要两倍的卡，而每张卡又慢 30 倍 → 总成本 $\sim60\times$。
</details>

---

## 二、什么情况下 FP64 真的合理

它几乎只出现在这些地方，而且都**不是**模型主干训练：

- **参考实现 / 数值基准**：写 `numpy` float64 的 reference 来对拍你的 kernel，验证 FP32 路径的误差上界。
- **诊断工具**：用 FP64 重算某一步的 loss / logits / 梯度范数，判断 loss spike 是数值问题还是数据问题。
- **评估指标与数据统计**：算 perplexity 的累积、语料统计、embedding 相似度之类的长归约。
- **科学计算型模型**（PINN、天气预报、分子动力学）：这里 FP64 是刚需，但那是另一个问题域。

## 三、如果你真担心数值稳定性，正确的做法

按性价比从高到低，基本能覆盖 FP64 的所有收益：

```text
1. BF16 计算 + FP32 累加        # Tensor Core 默认，免费
2. FP32 master weights          # 混合精度标准做法
3. 分块/分段累加（block-wise）   # 解决长归约误差，FlashAttention 就是这么做的
4. softmax / LayerNorm / loss 走 FP32（含 max-subtraction）
5. Kahan / 补偿求和              # 只在少数极端 reduction 上需要
6. 低位优化器状态 + stochastic rounding  # Dettmers 的 8-bit Adam 路线
7. 梯度裁剪 + 数据清洗           # 解决 loss spike 的真正手段
```

如果还有疑虑，做一次“BF16 vs 纯 FP32”的双跑消融就够了——你会发现曲线在噪声里重合，而这就已经证明 FP64 那一档多出来的精度是**观测不到**的。

---

**收尾：** FP64 训练 LLM 不是“错误”，而是**把一个已被解决的问题（尾数精度）用 30 倍的算力重新解决一遍**，同时还顺手打残了 FlashAttention 和整套融合算子生态。它更像一个调试和验证工具，而不是训练配置。
