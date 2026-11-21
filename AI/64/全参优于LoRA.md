# 全参优于LoRA

## 🧑‍💻 User

llm sft训练
10bt的数据集 用lora感觉好浪费
低秩的表达能力还是太受限了
而且就算玩peft你还是要搜参
如果infra允许感觉能上全参还是全参好点（？
而且不用担心奇奇怪怪的架构问题（比如moe的nn.parameter参数要单独去适配）

---
## 😀 默认助手 (7)

大方向上我同意你，但「浪费」这件事得拆开看：LoRA 省的其实不是算力。

## 先纠正一个常见的账算错误

前向和反向都是全量穿过的：

$$\text{FLOPs}\approx 6ND$$

LoRA 冻结 $W$ 后只省掉反传里那部分 $dW$；$dX$ 那部分照样要算。所以：

- **算子（FLOPs）**：只省下约 $2ND/6ND \approx 1/3$ 的反向计算，约等于整体 **15~20%**。
- **显存**：从 ~16 bytes/param（bf16 权重+梯度 + fp32 m/v + master）降到 ~2 bytes/param + 一点点 adapter。

8B 模型、10B tokens：$6\times8\times10^9\times10^{10}=4.8\times10^{20}$ FLOPs。8×H100 按 40% MFU 算约 3.2 PFLOPS，**大约 42 小时**。也就是说，用 LoRA 你最多把两天省成一天半，但换来的是一个只动了低秩子空间的模型。

所以「浪费」的准确说法是：**token 的算力成本一分没少，模型的更新能力却被砍了个零头**。如果在 infra 允许的情况下还上 LoRA，你确实相当于白花了绝大部分计算预算。

## 你提到的两个问题确实存在

**超参数搜索量。** 这一点被严重低估。LoRA 的超参数维度其实更高（rank、alpha/scaling、target_modules 子集、A/B 分开的 LR、dropout、init），而且 **LR 的最优值随 rank 漂移**——标准初始化下大致是 $\text{LR}\propto 1/r$（rsLoRA 改成 $1/\sqrt{r}$ 就是想把这个尺度掰回来）。改一次 rank 就是一次重新搜索。全参只有 LR、schedule、wd 几个，跨规模还相对稳定。所谓「LoRA 便宜」，只便宜在显存，不便宜在调参人力。

**架构适配。** MoE 上这个问题尤其明显：专家权重经常被实现成 `nn.ParameterList` 或者打平的 3D grouped-GEMM 张量，peft 那套按 module 名做 `target_modules=["q_proj","v_proj"]` 的匹配直接失效；router 那个小 Linear 到底要不要适配也没有标准答案（给了容易搞坏 load balancing，不给又可能学不动路由）。merge 回去还要按 expert parallel 重新 gather。全参就没有这一层「论文里没写、但能坑你三天」的成本。

> 额外一条：全参可以无差别套任何 trick（Muon、EMA、selective weight decay、差异化一层 LR），adapter 上这些基本都要重新推导一遍。

## 但要提醒的是：10B tokens 这个量本身可能不是 SFT

这一点我建议你先确认，因为它直接决定「全参 vs LoRA」的答案。

常规 SFT 是 **100M~1B tokens** 量级。10B tokens 已经接近 mid-training / continued pretraining 的尺度了。分两种情况：

1. **真的是高质量指令/agentic/reasoning trace**（多轮、工具调用、长 CoT）→ 这是在教**新行为、新知识**，正是低秩最吃力的一类。全参几乎无悬念，你的判断完全成立。
2. **其实是大量格式化、风格、模板data** → 这类任务本身的 $\Delta W$ 就接近低秩，r=64 可能真的够，多出来的容量只会拿去拟合噪声。这时候 LoRA 不那么「浪费」，但训练集本身值得重新审视。

Biderman 那篇 *LoRA Learns Less and Forgets Less* 的结论很符合直觉：**LoRA 在 code / math 这类需要「学新东西」的域上明显落后全参，但在对齐类任务上差距很小；同时它确实更少遗忘。** 另一篇 *An Illusion of Equivalence* 从谱性质上说 LoRA 会引入「intruder dimensions」，跟你说的表达能力受限是一回事。

<details>
<summary>如果确实要为 LoRA 辩护 / 或者不得不 LoRA，怎么做才不掉队</summary>

- rank 拉高到 256~512，**别只挂 attention**，MLP 的 `gate/up/down` 和 `lm_head`/`embed` 都要考虑；
- 用 rsLoRA 修 scaling，LoRA+ 给 B 更大的 LR，条件允许上 DoRA；
- 记住：当 rank 高到能覆盖真实更新的有效秩时，LoRA ≈ 全参——但那个秩在知识密集任务上可能是几百上千，此时显存优势已经被吃掉大半，不如直接全参；
- 中间地带其实更好：**partial FT**（只训后 25~50% 层，或冻 embedding + 前几层）在相近显存下通常稳压同预算的 LoRA。但你都 10B tokens 了，没必要绕。

</details>

<details>
<summary>全参在 10B tokens 上真正要小心的三件事</summary>

1. **遗忘 / alignment tax**。全参忘得比 LoRA 多。配方上：LR 压到 1e-5~2e-5（LoRA 那套 2e-4 别搬过来）、1~2 epoch、混 5~10% 预训练数据 replay、cosine 退到底。
2. **评估信号**。SFT 的 val loss 在 10B tokens 上几乎没有意义（必然降）。要挂 MMLU/通用能力 + 目标能力的分项 eval，看有没有「越训越窄」。
3. **数据质量被放大**。模型有容量去拟合的时候，脏数据就是被完整记住。10B tokens 的量级下，去重和过滤的收益可能比选 full FT 还是 LoRA 大得多。
4. 如果是真·10B 指令数据，考虑拆成 **mid-training（主体 + 少量指令混入，全参）+ 短 SFT（100~500M 精品，全参）** 两段，比一次性 10B 纯 SFT 稳得多。

</details>

## 我的决策规则

| 条件 | 选择 |
|---|---|
| GPU 够放下 16 bytes/param（8B≈128GB，70B≈1.1TB） | **全参**，没得聊 |
| 只是实验迭代 / 需要多租户 hot-swap adapter | LoRA，但别指望它是最终版 |
| 任务 = 格式/风格对齐，且数据量 < 500M tokens | LoRA 合理 |
| 任务 = code/math/新知识/长程 agent 行为 | 全参，LoRA 只用来快速验证数据管线 |
| MoE | 全参，除非你愿意自己写 expert-wise adapter |

一句话：**LoRA 的定位应该是「显存不够时的妥协」和「快速迭代的工具」，不是「省钱的 SFT 方案」**——它省的那 20% 算力，远不够补偿你在调参和适配上的额外开销。你这个判断在 10B tokens 这个尺度上是对的。
