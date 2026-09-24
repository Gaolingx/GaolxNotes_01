# RL中Rollout过程

## 🧑‍💻 User

回答以下问题：
1. 什么是 RL 中的 Rollout 过程？其核心目标是什么？
2. 为什么自回归依赖（Autoregressive dependency）会导致 Rollout 成为系统瓶颈？
3. 在量化加速中，为什么 W4/W8 量化即便不减少逻辑乘法数量也能加速计算？
4. 如何根据 Roofline 模型判断一个算子是受限于计算（Compute-bound）还是受限于访存（Memory-bound）？
5. 在超稀疏 MoE 训练中，“Silent Expert Death”现象是如何产生的？
6. 将 MoE 的 Top-k 路由类比为路径追踪（Path Tracing）时，为什么说它是一种“有偏估计”？
7. 请解释分布式并行中 TP（张量并行）切分 free index 和 reduction index 分别对应的通信操作。
8. 为什么 EP（专家并行）的通信模式天然是 All-to-All？
9. 在知识蒸馏中，Forward KL 为什么被称为 "Mode-Covering"?
10. Reverse KL 配合 Student on-policy 训练如何改善推理任务中的“曝光偏差（Exposure Bias）”？

---
## 🤖 Assistant

以下逐题作答。这 10 个问题属于模型系统与训练方法的理论问题，我直接根据领域知识给出解释（未涉及需要检索的外部数据，故未调用工具）。

---

## 1. RL 中的 Rollout 过程及其核心目标

**Rollout（采样/轨迹生成）** 是 RL 训练循环中"收集经验（data collection / experience generation）"的阶段：让当前策略与环境（或任务本身、奖励模型、verifier）交互，生成一条条轨迹。

对语言模型 RL（PPO、GRPO、RLVR 等），rollout 就是把 prompt $x$ 喂给策略 $\pi_{\theta_{old}}$，**自回归地**采样出完整回答

$$
y = (y_1, \dots, y_T) \sim \pi_{\theta_{old}}(\cdot \mid x),\qquad y_t \sim \pi_{\theta_{old}}(\cdot \mid x, y_{<t})
$$

再对这些样本打分，得到 $(x, y, r, \log\pi_{old}(y_t\mid x,y_{<t}), V(x,y_{<t}))$。

**核心目标：**

1. **产生 on-policy 样本**——样本必须来自"当前/待更新的策略"，这样重要性采样比率 $\rho = \pi_\theta/\pi_{old}$ 才成立；
2. **估计策略梯度**的期望，例如

$$
\nabla_\theta J \approx \mathbb{E}_{y\sim\pi_{old}}\!\left[\sum_t \nabla_\theta \log\pi_\theta(y_t\mid\cdot)\,\hat A_t\right]
$$

3. 提供奖励/回报信号用于 advantage 估计。

一句话：rollout 是"用旧策略探索、采数据"，是策略改进的前提。

---

## 2. 自回归依赖为何使 Rollout 成为系统瓶颈

关键在于**序列维度不可并行**。自回归生成满足

$$
P(y)=\prod_{t=1}^{T}P(y_t\mid y_{<t}, x)
$$

即第 $t$ 个 token 依赖前面所有 token，**必须逐 token 串行前向**，无法像训练（teacher forcing）那样一次并行计算所有位置。

由此产生的系统问题：

- **串行延迟主导**：$T$ 步串行，每步只做极小计算量（单 token 的 GEMM），却要反复加载 KV-cache 和权重 → 算术强度极低，是典型 **memory-bound / latency-bound**，GPU 算力严重闲置。
- **吞吐无法横向扩展**：加 GPU 不能把一条序列"切"得更快（除非用投机解码等近似手段），只能堆 batch，受限于 batch 内长尾序列（长短不一）导致负载不均。
- **与训练耦合**：RLHF 里 rollout 与训练常常分离部署，还要频繁同步权重；一条长序列生成完才能算 reward。
- **超长 CoT / 推理任务**放大问题：$T$ 可达数千甚至上万，且随训练变长。

结果：在 PPO/GRPO 等循环中，**wall-clock 时间被 rollout 主导**（常占 70%+），成为整个 RL 训练流程的系统瓶颈。

---

## 3. W4/W8 量化为何"不减少逻辑乘法"也能加速

量化加速**不是靠减少 MAC 数量**，而是靠**硬件单元吞吐**与**访存字节数**两条路径：

**（a）低精度整数/低比特算力更高**

现代 Tensor Core / NPU 对低位宽有更高峰值吞吐，例如典型比例：

| 精度 | 相对吞吐 |
|---|---|
| FP16/BF16 | $1\times$ |
| INT8 | $2\times$ |
| INT4 | $4\times$ |

同样多次"逻辑乘法"，映射到 INT8/INT4 数据通路上**每周期能算更多**，墙钟时间下降。

**（b）访存流量降低（对 memory-bound 尤其关键）**

权重和激活从 16bit → 8bit/4bit，**字节数减半/四分之一**。由 Roofline：

$$
P = \min(P_{peak},\; I\cdot BW),\qquad I=\frac{\text{FLOPs}}{\text{Bytes}}
$$

量化让分母（Bytes）变小 → 算术强度 $I$ 变大 → 更靠近 compute roof。纯访存受限的算子（decode 阶段）**直接按字节比加速**。

**（c）附带收益**：占用更小、更容易进驻 cache/SMEM，减少 spill；能耗更低。

所以哪怕乘法次数不变，**"每 op 更便宜 + 数据搬得更少"** 就足以加速。

---

## 4. 如何用 Roofline 判断 Compute-bound 还是 Memory-bound

Roofline 模型：

$$
P_{\text{att}} = \min\bigl(P_{\max},\; I \times BW\bigr),\qquad I=\frac{\text{FLOPs}}{\text{Bytes moved}}
$$

定义**机器平衡点（ridge point）**：

$$
I^* = \frac{P_{\max}}{BW}
$$

**判断规则：**

- 若 $I > I^*$：**Compute-bound**（算子每搬一字节做了足够多的计算，算力单元被打满，性能 $\approx P_{\max}$）。
- 若 $I < I^*$：**Memory-bound**（性能线性受限于带宽，$P \approx I\cdot BW$，算力用不满）。

**实践中的两种判断法：**

1. **算/量强度法**：统计该算子的 FLOPs 与移动字节数（尤其 DRAM 流量，含权重、激活、KV-cache 的 load/store），算出 $I$，与 $P_{\max}/BW$ 比较。
2. **profiling 计数法**：
 - DRAM 利用率接近 ~100% 而 SM/算力利用率低 → **memory-bound**；
 - 算力单元接近峰值而 DRAM 未饱和 → **compute-bound**。
 - 或做强度扫描：性能随 $I$ 线性上升到某点后平台化 → 拐点即 $I^*$。

典型：大 tile GEMM = compute-bound；elementwise / LayerNorm / Softmax = memory-bound；LLM **decode 阶段 attention**（KV 反复读、每 token 计算少）= memory-bound。

---

## 5. 超稀疏 MoE 中 "Silent Expert Death" 如何产生

MoE 每层用 router 为 token $x$ 选 Top-k 专家：

$$
y=\sum_{e\in \text{TopK}(g(x))} \tilde g_e(x)\, E_e(x)
$$

**只有被选中的专家才会获得梯度更新**。由此形成自我强化的负反馈：

1. 随机初始化 / 噪声导致初始路由不均，某些专家一开始就很少被选中；
2. 该专家几乎拿不到 token → **梯度信号近似为零** → 参数几乎不更新；
3. 参数不更新 → 表现不提升 → router 更不会选它；
4. 循环放大，最终该专家"名义存在、实质停止学习"。

称为 **"Silent"（静默）** 是因为：它不会报错、不会被删除，只是**悄无声息地不再被激活、不再学习**；对外表现为整体有效容量缩水、token 高度集中到少数专家（router collapse）。

**加剧因素：** 超稀疏（如 top-1、几百选 k）使被选概率本就很小；缺乏负载均衡约束时更易塌缩；门控饱和/梯度消失。

**常见缓解：** 负载均衡辅助损失（aux loss）、capacity factor、noisy/jitter router、专家 dropout、更好的初始化与课程学习。

---

## 6. 把 MoE Top-k 路由类比 Path Tracing，为何是"有偏估计"

理想的 MoE 输出是对**全体专家**的混合期望：

$$
y^\star = \sum_{e=1}^{N} g_e(x)\, E_e(x)
$$

**Path tracing 的类比**：渲染积分 $\int f(x)\,\mathrm dx$ 若用 Monte Carlo **按正确概率密度采样全部路径**再平均，得到的是**无偏估计**（期望等于真值，只是有方差）。

MoE 的 Top-k 路由却是**确定性硬截断**：只取分数最高的 k 个专家，丢掉尾部，再对权重重新归一化：

$$
\hat y = \sum_{e\in \text{TopK}(g)} \tilde g_e(x)\, E_e(x)
$$

由于：

- **截断**了低概率专家贡献（真值里它们有非零权重）；
- **重新归一化**又改变了剩余项的权重；

所以 $\mathbb{E}[\hat y]\neq y^\star$，存在**系统性偏差（bias）**，不随样本/重复而消失。

对照：如果改为**按 $g(x)$ 概率采样专家**（类似 path tracing 的随机采样），则可得到无偏但高方差的估计。Top-k 相当于用"只走最亮的 k 条光路"来换低方差，代价就是引入了 bias。因此二者是 **bias–variance 权衡**：Top-k = 低方差、有偏；随机采样 = 无偏、高方差。

---

## 7. TP 中切分 free index 与 reduction index 对应的通信

以线性层 $Y=XW$，$X\in\mathbb{R}^{b\times h}$，$W\in\mathbb{R}^{h\times h'}$，$Y\in\mathbb{R}^{b\times h'}$ 为例：

- **reduction index（收缩维）** = $h$（输入特征，被求和掉）；
- **free index（自由/输出维）** = $h'$（输出特征），以及 batch 维。

**（1）切 free index（按输出特征切分，即 Column Parallel）**

每个 rank 持有 $W_i\in\mathbb{R}^{h\times h'/N}$，独立计算 $Y_i = XW_i$。

- **前向：无需集合通信**——各 rank 输出天然是结果的一个分片（free，独立可并行）。
- **反向：需要 All-Reduce**——对输入求梯度 $\nabla X=\sum_i \nabla Y_i W_i^\top$ 是对各 rank 的**部分和**，必须跨 rank 累加。

**（2）切 reduction index（按输入特征切分，即 Row Parallel）**

每个 rank 持有 $W_i\in\mathbb{R}^{h/N\times h'}$ 与输入分片 $X_i$，计算部分积 $Y_i=X_iW_i$。

- **前向：需要 All-Reduce（求和）**——结果 $Y=\sum_i Y_i$ 是各 rank 部分积之和，必须 All-Reduce 累加。
- **反向：输入梯度无需通信**（各 rank 得到自己那份分片），但权重梯度需要 All-Gather 输入分片。

**一句话规律：**

> 切 **reduction（收缩）维** → 各 rank 产生**部分和**，必须 **All-Reduce** 累加；
> 切 **free（输出）维** → 各 rank 结果**互相独立**，前向无通信，只有配套的反向才需要对"被切的那一维"做 All-Reduce。

---

## 8. 为什么 EP 的通信模式天然是 All-to-All

**EP（专家并行）** 把不同专家放在不同设备上，而 token 按内容被路由到其选中的专家所在设备。考虑一个 batch 的 token **均匀分布在所有设备**上，专家也**分散在所有设备**上：

- 设备 $i$ 手中的 token 会分别被路由到**多个不同设备**的专家；
- 同时设备 $i$ 又会**从多个不同设备**收到 token。

这是一个**数据依赖的动态置换（permutation）**：每个设备都要向其他**所有**设备发送一份**各不相同**的数据，并从每个设备接收不同数据。这正好就是 **All-to-All** 的定义（每个 rank 向每个 rank 发送不同分块）。

因此 EP 分为两个 All-to-All 阶段：

1. **Dispatch（分发）**：All-to-All 把 token 送到对应专家设备；
2. **Combine（汇总）**：专家算完后 All-to-All 再把结果送回原设备。

由于 token→expert 的映射随输入动态变化、天然是全局散射/聚集，EP 只能且必须用 **All-to-All**（与 static 的 All-Reduce/All-Gather 不同，它是动态数据的 many-to-many 交换）。

---

## 9. Forward KL 为什么叫 "Mode-Covering"

设 $p$ 为目标/教师分布，$q$ 为学生分布。Forward KL：

$$
D_{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
$$

关键性质：**只要 $p(x)>0$ 而 $q(x)\to 0$，该项趋于 $+\infty$**（因为 $\log(p/q)\to\infty$ 且 $p(x)>0$）。

这意味着：凡是教师有概率质量的地方，学生**必须也分配非零概率**，否则损失无穷大。于是最优 $q$ **必须覆盖 $p$ 的整个支撑集**，不能忽略任何一个 mode。

- 它追求**全局覆盖/质量匹配**，最优解偏向 $p$ 的"均值"，倾向于**过散（over-dispersed）、高方差**；
- 被称为 **Mode-Covering（模式覆盖）/ Mass-Covering / Mean-Seeking**。

对照 **Reverse KL** $D_{KL}(q\|p)$：当 $q(x)>0$ 而 $p(x)=0$ 时也趋于 $\infty$，即惩罚学生把质量放到教师没有的地方，结果是"抓住一个模式、忽略其余"，故称 **Mode-Seeking / Zero-Forcing**。

---

## 10. Reverse KL + Student on-policy 如何改善"曝光偏差"

**曝光偏差（Exposure Bias）** 的来源：SFT/teacher forcing 时用**真实前缀**训练，推理时却用**自己生成的前缀**；训练分布与推理分布不一致，一旦早先 token 出错，误差在自回归中不断累积。

**Reverse KL + 学生 on-policy 采样**正好对准这个痛点：

**（1）On-policy 采样 = 在"自己会走到的状态"上训练**

$$
D_{KL}\bigl(\pi_\theta \,\|\, \pi_{teacher}\bigr)
$$

其中期望是在**学生自身分布 $\pi_\theta$ 下**取的（学生采样自己的轨迹/前缀）。也就是说：

$$
\mathbb{E}_{y\sim \pi_\theta}\Bigl[\log \tfrac{\pi_\theta(y)}{\pi_{teacher}(y)}\Bigr]
$$

训练时看到的前缀，正是推理时学生会遇到的状态 → **消除训练-推理分布错配**，这是治曝光偏差的根本。

**（2）Reverse KL 的 mode-seeking 特性抑制误差累积**

Reverse KL 惩罚"学生把概率放到教师概率低的地方"（$q>0, p\approx0$ 时代价 $\to\infty$）。所以学生倾向于**收敛到教师若干高置信、可靠的模式上**，而不是勉强去覆盖所有可能的教师路径。在推理任务里，这意味着：

- 不冒险生成自己"没把握、但教师分布里存在"的路径，从而**减少把错误 token 概率推高**导致的后续雪崩；
- 与学生自身可稳定复现的输出对齐，鲁棒性更好。

**综合效果**：on-policy 让"训练=测试"（对齐状态分布），reverse KL 让"只在自己可靠的模式上匹配教师"（抑制错误扩散）。二者配合，直接削弱了曝光偏差的核心机制——*训练时依赖真值前缀、推理时误差自累积*。

---

如需，我可以进一步对其中某几题（如 #3/#4 的 Roofline 推导、#7 的 Megatron 并行前后向通信表、#10 的目标函数推导）展开成更形式化的版本。
