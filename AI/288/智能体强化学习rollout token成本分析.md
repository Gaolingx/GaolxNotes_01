# 智能体强化学习rollout token成本分析

## 🧑‍💻 User

在准备agenticrl的训练，现在是出题（query）+rollout，rollout花的token比出题多多了（用llm api key），这合理吗？
想了想 你出一道题目花个30M很正常把，然后这道题还有去测不同模型的pass@n，avg@n，再调出合适的轨迹去训吧
这块得花不少token
而且一般还要做RFT（拒绝采样）啥的

---
## 😀 默认助手 (7)

## 先给结论

**合理，而且这是 agentic RL 的结构性特征，不是 bug。** 但你把“出题 30M”和“rollout 远多于出题”放在一起时，这两个说法在成本上存在矛盾，值得你先算清楚，因为“rollout 贵”还是“出题贵”会决定你的优化重点。

---

## 一、先把成本模型写出来，判断就不再主观

设：

- $N$ = 任务池大小，$C_q$ = 单题合成与校验的成本（token）
- $S$ = 训练步数，$B$ = 每步 prompt 数，$G$ = 每题采样数（GRPO 的 group size）
- $L$ = 单条轨迹的**累计** token（多轮 agent，输入随轮次膨胀，远大于单轮）

$$T_{\text{出题}} \;=\; N \cdot C_q \quad(\text{一次性，可摊销})$$

$$T_{\text{rollout}} \;=\; S \cdot B \cdot G \cdot L \quad(\text{每步都付})$$

$$\rho \;=\; \frac{T_{\text{rollout}}}{T_{\text{出题}}} \;=\; \frac{S\,B\,G\,L}{N\,C_q}$$

关键区别在于：**出题是资本开支，rollout 是运营开支**。前者一次性，后者随 $S, G, L$ 线性增长。只要训练步数不太少，$\rho \gg 1$ 就是必然结果。

---

## 二、“30M token 出一道题”与“rollout 更多”的算术矛盾

这就是我要你重新核对的地方。代入一个常见配置：$G = 8$、$L \approx 60\text{k}$、每题在整个训练中被复用 $R = 4$ 轮：

$$\text{单题 rollout} \approx 4 \times 8 \times 60\text{k} = 1.92\text{M}$$

**如果 $C_q = 30\text{M}$，那么 $\rho \approx 0.06$，出题比 rollout 贵 15 倍。** 这跟你的观察正好相反。

所以 $C_q = 30\text{M}$ 只有三种可能：

1. **这 30M 里混进了 rollout**：你为了验证“这题可解、难度合适”跑了一堆 pass@n、avg@n 采样，那些本质上就是 rollout，只是被你记在“出题”账上。这是最常见的重复计算。
2. **复用太少**：你花 30M 造了一道题，只跑 4 轮、8 个样本就换掉了。造题成本没有摊销，这 30M 相当于沉没成本。
3. **30M 是整批题目的数字**，而非单题。如果是这样，取 $N=500$，rollout:query 做到 $100:1$ 甚至 $1000:1$ 都完全正常。

**建议**：把统计方式明确划成三栏：`合成` / `校验（含 calibration rollout）` / `训练 rollout`。你会发现“出题贵”的判断大多来自第二栏。第二栏应该一次性完成，产出“难度标签”，并可在训练全程复用（这一栏正是你判断 $\rho$ 的分母），不要每一步重跑。

---

## 三、token 比例不等于账单比例

这一点常被忽略，但对你（用 API key 付钱）很重要：

- **出题**的 token 大量是**输出**（写环境代码、推理解题、反复改），单价最高；
- **rollout** 的 token 大量是**输入**（system prompt、tool schema、累积上下文），而同一 prompt 的 $G$ 个样本前缀完全相同，**KV/prefix cache 命中率可以很高**，缓存输入通常便宜 5–10×。

所以**你看到的 token 差距（可能 100:1）在账单上会明显缩小（可能 10:1~20:1）**。反过来说：如果你不做 prefix cache，agentic rollout 就是纯烧钱，因为上下文随轮次增长，而前几轮又不复用。

> 结论：比较“出题 vs rollout”时看**实际账单金额**，不要看原始 token 数。把 cache hit rate 作为一项核心指标监测。

---

## 四、$\rho$ 多大算“正常”

没有标准值，但有**诊断信号**。看比例没用，应看**产出率（yield）**：

$$\text{yield} = \frac{\#\text{进入 batch 且有非零 advantage 的轨迹}}{\#\text{总 rollout 轨迹}}$$

| 现象 | 大概率病因 | 处理 |
|---|---|---|
| rollout 极高但 loss 不降、梯度稀疏 | 题目 pass@G 全 0 或全 1，**组内方差为 0**，advantage 全变成 0 | 训练前做难度分层，只留 pass rate $\in [0.1, 0.85]$ 的题 |
| 单题 token 极高 | 无轮次上限、工具输出未截断、陷入重试/报错循环 | 截断 observation、设 max turns、失败早停 |
| 每步都重跑 calibration / 跨模型评测 | 把一次性工作放进了循环 | 评测移出训练 loop |
| $\rho$ 高但 yield 也高 | **健康**，这是 RL 的本来成本 | 只优化单位成本（cache、batch、异步），别砍量 |

判断标准不是 $\rho$，而是**“每得到一个有效梯度样本的边际成本”**。

---

<details>
<summary><b>五、真正能降低 rollout 成本的杠杆（按性价比排序）</b></summary>

1. **Prefix/KV cache 复用**：同一 prompt 的 $G$ 条轨迹共享 system + tool schema + 前几轮。这是 agentic 场景中收益最大的一项，能直接减少 60–90% 的输入账单。注意把 `temperature` 变高不影响 cache（前缀相同），但会导致更多失败轨迹。
2. **难度校准前置（把成本移到“出题”栏）**：每题先跑少量采样，测 pass rate **和组内方差**，只把有梯度的题放进训练。这一步一次性完成、可摊销，并且能用“便宜模型或低 $n$”完成。
3. **失败早停 + 轮次上限**：检测到重复报错、编译死循环、“模型在自言自语”即终止。尾部的轮次上下文最大，取消它们最省。
4. **动态 group size**：pass rate 接近 0.5 的题多采，两端少采。比固定 $G=16$ 节省 30%+。
5. **轨迹过滤放在存储前**：截断、报错、超长的不写入存储，避免后面再处理一遍。
6. **同一批 rollout 多训几步**：$K$ 个 PPO epoch 直接让 $\rho \propto 1/K$，代价是 off-policy 陈旧度。这是最直接降低 $\rho$ 的旋钮，比省 token 更划算。
7. **异步 rollout + 连续批处理**：不省 token，但能把 API 吞吐和成本/墙钟压下来。

</details>

<details>
<summary><b>六、RFT（拒绝采样）的账要单独算</b></summary>

RFT 和 RL 的成本结构完全不同，别混在同一个预算里：

$$T_{\text{RFT}} \approx N \cdot k \cdot L \quad(\text{一次性}), \qquad \text{之后 SFT 训 } E \text{ 个 epoch 几乎免费}$$

- RFT 是**离线一次采样 + 验证器过滤 + SFT**，产物可跨 epoch 复用 → **单位有效 token 的成本远低于 RL**；
- 但过滤越狠（只留 top-$k$），每题需要采的 $k$ 越大，rollout 成本会放大；
- **RFT 是你校准难度的天然下游验证**：如果一道题 RFT 都采不出正样本，它在 RL 里也只会贡献零梯度。

**推荐配方**：用 RFT（便宜模型、大 $k$、一次性）先把任务池筛到“可解且非饱和”区间并打上难度标签 → 再用 RL 把 rollout 预算只花在有梯度的切片上。这样“出题 + 校准”承担了本来会浪费在 rollout 上的那部分 token。

</details>

---

## 七、给你的三个监控量（比盯 $\rho$ 有用）

1. **cache hit rate**（输入 token 命中率）：低于 50% 说明你的 agentic 采样在重复付钱。
2. **yield**（有效轨迹占比）：低于 20%，先修难度分布，不要加 $G$。
3. **cost per valid gradient token**：用账单金额算，分子是支付的钱，分母是进入 loss 且 advantage 非零的 token 数。

盯这三个量后，“rollout 比出题多”就不再是问题；它只是公式里 $S \cdot G$ 的必然结果。真正需要调整的是 yield 和 cache。
