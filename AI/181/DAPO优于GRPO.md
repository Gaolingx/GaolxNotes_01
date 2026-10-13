# DAPO优于GRPO

## 🧑‍💻 User

为什么agenticrl的critic free方案中，dapo比grpo更优？

---
## 😀 默认助手 (7)

## 一句话结论

DAPO 并不是一个和 GRPO 平行的"新算法"，而是 **GRPO 的 critic‑free 骨架上打了四个补丁 + 去掉 KL**：Clip‑Higher、Dynamic Sampling、Token‑Level Loss、Overlong Shaping。它在 agentic RL 里显得"更优"，不是因为 baseline 估计更聪明（两者用的都是组内相对优势，都是轨迹级 $\hat A$），而是因为它**逐个修掉了 agentic 场景会把 GRPO 的病放大到致命的那些失效模式**：长尾轨迹长度、稀疏/二值结局奖励、组合式动作空间、昂贵的环境交互。

---

## 前提澄清：两者的差异其实只有几行公式

<details>
<summary>点击展开：GRPO 与 DAPO 目标函数对比</summary>

GRPO（组内归一化优势 + 序列级损失 + 对称 clip + KL）：

$$J_{GRPO}=\mathbb{E}\Big[\tfrac{1}{G}\sum_{i=1}^{G}\tfrac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_i\big)-\beta D_{KL}\Big]$$

$$\hat A_i=\frac{r_i-\mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)},\qquad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{old}(o_{i,t}\mid q,o_{i,<t})}$$

DAPO：

$$J_{DAPO}=\mathbb{E}\Big[\tfrac{1}{\sum_i|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\varepsilon_{low},1+\varepsilon_{high})\hat A_i\big)\Big]$$

- 归一化：$\frac{1}{G}\sum_i\frac{1}{|o_i|}$ → $\frac{1}{\sum_i|o_i|}$（序列级 → token 级）
- $\varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$（解耦上下界）
- 无 KL；批次构造上加 Dynamic Sampling；奖励上加 Overlong Shaping

所以当 $\varepsilon_{high}=\varepsilon_{low}$、恢复序列级归一化、不做动态采样时，DAPO 就退化回 GRPO 的一个特例。**"更优"完全来自这四处修改，而不是骨架。**

</details>

---

## 为什么这四处修改在 agentic 场景价值被放大

| GRPO 的失效模式 | DAPO 的对策 | 为什么 agentic 里特别严重 |
|---|---|---|
| 熵崩溃、探索消失 | Clip‑Higher | 动作空间是组合式的（工具×参数×轮数×终止时机），且关键 token（工具名、终止标记）初始概率极低 |
| 组内奖励全同 → 梯度为 0 / std=0 除零 | Dynamic Sampling | 二值 outcome + 难任务 → all‑fail 组占比高；一次 rollout 要真实环境交互，丢弃=烧钱 |
| 序列级归一化 → 长度偏差 | Token‑Level Loss | 轨迹长度重尾（3 步 vs 30 步、工具输出上千 token），真实决策点被稀释 |
| 截断轨迹奖励无效 | Overlong Shaping | 多轮 + 长工具输出极易撞上下文上限，噪声奖励直接教坏模型 |

下面逐条说机制。

### 1. Clip‑Higher：对抗熵崩溃，保住探索

对称 clip 的问题在于：对**正优势**的 token，梯度被 $1+\varepsilon$ 截断；而概率越低的 token，更新后 $\rho_{i,t}$ 越容易冲高——**最需要被强化的那批低概率 token 恰好最先被掐断**。于是策略迅速变尖，entropy policy collapse。agentic RL 里这表现为最典型的 mode collapse：固定用同一种工具、输出固定模板、过早终止（early stop）、不再尝试新计划。

DAPO 把上界放宽到 $1.28$，让低概率但正优势的动作持续拿到梯度，熵维持得住。agentic 任务的正确轨迹往往在初始策略下概率极低（多个正确决策的联合概率 = 各步概率连乘），所以这条补丁的收益比单轮数学题更大。

### 2. Dynamic Sampling：把算力花在有信号的样本上

$\mathrm{std}(\mathbf r)=0$ 时 $\hat A_i\equiv 0$（不保护还会 NaN）。GRPO 会把这些 prompt 计算一遍再拿到零梯度；agentic 中 all‑fail 组尤其多，等价于有效 batch size 缩水、advantage 尺度漂移、梯度噪声上升。DAPO 直接**丢弃全对/全错的组并重采样**，保证批次中每条 prompt 都贡献非零梯度——在"每条轨迹都要跑真实环境"的设定下，这是样本效率问题，不只是稳定性问题。

### 3. Token‑Level Loss：修掉长度偏差与"少调用工具"的捷径

$\frac{1}{|o_i|}$ 让每条轨迹与长度无关地等权，于是：

- 长轨迹中**单个决策 token 的权重 $\propto 1/|o_i|$**，真正重要的"选哪个工具、传什么参数、何时收尾"被大量冗余 token（回显、长工具输出）淹没；
- 短轨迹整体被相对放大 → 模型学到"少交互、快出答案"的退化策略（跳过检索/验证、直接猜）。

改成按全批 token 总数归一化后，每个 token 等权，长轨迹里的关键决策获得足量梯度。**代价是它反过来会奖励变长**（token 多 → 总梯度大），所以必须和 Overlong Shaping 打包使用——这也解释了 DAPO 为什么把四个技巧一起提出，而不是零散使用。

### 4. Overlong Shaping / Masking：别用无效数据训练

被长度截断的轨迹奖励无意义（既非成功也非失败）。当成负样本会教出"缩短输出以规避截断"或把好行为误判为坏。DAPO 给超出阈值的部分线性软惩罚，并把截断样本的 loss mask 掉，减少噪声梯度污染。多轮 + 长 observation 的 agentic 轨迹撞上限的频率远高于单轮。

### 5. 去掉 KL

GRPO 的 per‑token KL 在长序列上逐 token 累加，对长轨迹施加更强约束，还对每个 token 引入偏置项（Dr. GRPO 亦论证 KL 项引入 bias 且非必要）；同时省一个参考模型的前向与显存。agentic 中策略需要在线学到参考模型分布里几乎不存在的工具使用行为，KL 约束的性价比更低。

---

## 反方与边界条件（"更优"不是无条件的）

- **证据边界**：DAPO 原文的消融主要在单轮数学推理（AIME 等）上做，四个组件各自有增益。迁移到 agentic 是社区广泛复现的经验，**环境、奖励设计、并行框架都会改变结论**，不宜当成定理。
- **不是唯一解**：把 GRPO 调好（熵正则、长度归一化、按难度/方差过滤 prompt）也能显著缩小差距；DAPO 的很多效果是可以"用别的手段近似"的。
- **不同维度的问题还没解决**：DAPO 仍是**轨迹级优势广播到所有 token**，无法回答"哪一步工具调用错了"这个 agentic 的核心 credit assignment 问题。真正针对它的是 GiGPO（anchor state 分组 + step‑level 优势）、turn‑level advantage 这类工作；DAPO 只是让这套粗粒度信号更干净。
- **GSPO 的角度**：长序列不稳定的根因可能在于 **token 级重要性比值 $\rho_{i,t}$ 的粒度**不对，应改用 sequence‑level ratio。这与 DAPO 的 token‑level **loss 权重**是不同的维度，二者可以叠加，不构成互斥。
- **有偏换效率**：Token‑Level Loss 与 Dynamic Sampling 都改变了目标，不再是序列级目标的蒙特卡洛估计——是典型的"牺牲无偏性换优化效率"。
- **工程代价**：Dynamic Sampling 需要能重采样/丢弃整条昂贵轨迹；$\varepsilon_{high}$ 调太大直接不稳定。

**一句话给面试/综述用**：critic‑free 把方差控制的担子全压在了策略梯度那一侧，而 GRPO 的组内归一化 + 序列级损失 + 对称 clip 这套设计是为**短、等长、稠密可验证**的单轮任务准备的；agentic 把它推到了长尾长度、稀疏二值奖励、组合动作空间的区间，于是四个失效模式同时发作，DAPO 正是逐条对症的补丁集合。
