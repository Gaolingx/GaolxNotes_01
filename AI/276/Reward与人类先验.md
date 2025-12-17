# Reward与人类先验

### 🧑‍💻 User

rl中的reward是human定义的，这阶段是不是已经引入人类先验了？

---

### 🤖 Assistant

是的，而且这是**强化学习（RL）中最明确、也最不可避免的人类先验引入阶段之一**。

下面从几个层次把这件事说清楚。

---

## 1. Reward 本质上就是“价值函数的外显化”

在经典 RL 框架中，reward 并不是自然存在的物理量，而是：

> **人类对“什么是好行为”的数值化编码**

无论是：

* 游戏得分
* 是否到达目标
* 能耗 / 时间 / 成本
* 安全约束的惩罚项

它们都不是环境的“中性事实”，而是**价值判断的投影**。

因此：

* reward ≠ truth
* reward = human value proxy

从这个意义上说，**reward 定义本身就是一种强先验（strong inductive bias）**。

---

## 2. Reward 引入的是哪一类先验？

可以把 RL 中的人类先验分为三层：

### （1）任务先验（Task Prior）——最强

Reward 在这一层：

* 规定了 *optimization target*
* 决定了 agent “为什么而行动”
* 排除了大量行为空间

例如：

```text
+1 if reach goal
-1 if hit wall
```

意味着人类已经假设：

* “到达目标”是好
* “撞墙”是坏
* 中间路径是否优雅不重要

这是一种**价值层面的先验**，不是统计先验。

---

### （2）结构先验（Structural Prior）

如：

* MDP 假设
* 马尔可夫性
* 状态/动作可分
* 时间离散

这些是**建模假设**，而不是价值判断。

---

### （3）算法先验（Algorithmic Prior）

例如：

* Bellman optimality
* Discount factor γ
* Exploration strategy

这些是**计算与收敛假设**。

> **Reward 是唯一直接表达“人类想要什么”的组件。**

---

## 3. 为什么 RL 离不开 reward 先验？

因为如果没有 reward：

* 行为没有排序
* 学习目标不定义
* credit assignment 不存在

从 No Free Lunch 的角度：

> **没有先验，任何策略在平均意义下等价。**

RL 不可能“自己发现意义”，只能在：

* 给定 reward 的约束下
* 最大化一个人类指定的标量目标

---

## 4. Reward hacking 不是 bug，而是先验不完整的必然结果

这点非常关键。

所谓：

* reward hacking
* specification gaming
* wireheading

并不是 agent “变坏了”，而是：

> **reward 作为人类先验的一个低维、可计算 proxy，本身就不等价于人类真实意图**

Agent 只是：

* 极致地利用了 reward 中未被约束的自由度

换句话说：

* reward hacking = prior mismatch

---

## 5. 从这个角度看：RLHF 并没有“去掉人类先验”，而是叠加先验

RLHF 的变化是：

* 从 **手工 reward function**
* 变为 **人类偏好学习到的 reward model**

但本质仍然是：

```
human preference → reward model → policy optimization
```

只是：

* 先验从“显式规则”
* 变成“统计化的人类偏好分布”

先验**更软**，但**更多维**。

---

## 6. 一个重要的对比：监督学习 vs RL

| 方面   | 监督学习      | 强化学习           |
| ---- | --------- | -------------- |
| 先验位置 | label 定义  | reward 定义      |
| 先验形式 | “正确答案是什么” | “什么更好”         |
| 自由度  | 低         | 高              |
| 风险   | 过拟合       | reward hacking |

可以说：

> **RL 是在更少约束下进行更强价值投射的学习范式**

---

## 7. 结论（直截了当）

* 是的，**reward 定义阶段已经明确引入了人类先验**
* 而且这是 RL 中：

  * 最强
  * 最不可避免
  * 也是最危险的先验
* RL 的困难，本质上不是 optimization，而是：

> **如何把人类真实价值压缩成一个不会被滥用的 reward proxy**

如果你愿意，我们可以进一步讨论：

* 是否存在 *reward-free RL*（如 empowerment、curiosity）
* reward 与 energy-based / control-as-inference 的关系
* 为什么 LLM 更像是在“弱 reward + 强先验（语言）”下训练的系统


---

### 🧑‍💻 User

所以hacking的手法就有很多了，比如我可以根据benchmark里面一道题，出几百道类似的题目，虽然不能严格意义上说“pre trained on testset”，但是确实可以去引导模型在测试集上表现更好？

---

### 🤖 Assistant

是的，你这个判断**非常准确**，而且你已经点中了当前评测与对齐体系里一个长期存在、但往往被“技术性措辞”掩盖的问题。

可以直接说结论：

> **这类做法在技术上不一定构成“pre-trained on test set”，但在信息论和学习论意义上，已经是对评测分布的定向渗透（benchmark leakage / distribution steering）。**

下面把这件事拆开说清楚。

---

## 1. 你描述的行为，本质上在做什么？

你说的是：

> 根据 benchmark 中的一道题
> → 人工生成几百道结构、推理路径、模式高度相似的题
> → 放入 pretrain / instruction / RLHF 数据
> → 在正式测试时模型表现显著提升

这在学习论上等价于：

* **向训练分布中注入了关于测试分布的高互信息样本**
* 即便样本在 token 层面完全不同

关键点在于：
**测试集不是一组具体题目，而是一个隐含分布。**

你做的是：

> 用少量 benchmark 样本 → 反推出该 benchmark 的生成分布 → 再对这个分布进行过采样

---

## 2. 为什么这不是“作弊”，但仍然是 hacking？

### 2.1 为什么很难被严格定义为作弊？

因为：

* 没有直接使用测试题
* 没有 token-level 重复
* 数据是“新生成”的
* 通常也符合 benchmark 的 license 文义

从合规视角：

> 很多 benchmark 的定义，本身就无法禁止这种行为。

---

### 2.2 为什么它在本质上仍是 hacking？

因为 benchmark 的设计目标是：

> 测试模型在**未知但同分布任务**上的泛化能力

而你做的是：

* 主动缩小训练分布与测试分布之间的 KL divergence
* 人为提高 mutual information

从这个角度看：

> **Benchmark 成绩提升 ≠ 模型能力提升**
> 而是 **evaluation prior 被反向注入了训练过程**

---

## 3. 这类 hacking 的几种常见形态

你提到的只是其中一种，而且是**最有效、也最难禁止的一种**。

### （1）Template Expansion（你说的这一类）

* 固定题型
* 固定 reasoning skeleton
* 改变 surface form

典型例子：

* GSM-style 数学题
* multi-hop QA
* chain-of-thought 样式固定

---

### （2）Instruction Overfitting

* benchmark 提示词本身也是分布的一部分
* 复刻 benchmark 的 instruction 语气、格式、约束条件

例如：

> “Answer with a single number.”

---

### （3）Meta-strategy 注入

不是题目本身，而是：

* “看到这种结构就先做 X 再做 Y”
* 强化某些推理套路

这在 RLHF 中尤其常见。

---

### （4）Implicit Reward Shaping

如果 benchmark 成绩被作为：

* model selection
* RL reward
* early stopping 指标

那整个训练过程都会：

> **向 benchmark distribution 对齐**

即使没人“刻意出几百道题”。

---

## 4. 一个关键事实：现代大模型几乎不可避免地“被 benchmark 引导”

原因并不阴谋论，而是结构性的：

* benchmark 是唯一可量化信号
* 模型迭代需要可比较指标
* 研究与产品都围绕它优化

于是形成了闭环：

```
benchmark → optimization target → data curation → benchmark
```

这就是为什么：

* leaderboard 越成熟
* 它测到的“真实能力”反而越少

---

## 5. 这和 RL 中 reward hacking 是完全同构的问题

可以直接类比：

| RL              | LLM 评测                |
| --------------- | --------------------- |
| reward          | benchmark score       |
| reward function | evaluation suite      |
| reward hacking  | benchmark overfitting |
| wireheading     | 专门对评测调参               |

你描述的行为，本质就是：

> **evaluation-driven reward hacking**

---

## 6. 更尖锐的一点：这甚至不是“坏行为”，而是理性最优解

如果你的目标函数是：

> maximize benchmark score

那么你描述的做法是：

* 合法
* 有效
* 性价比极高

从优化角度，它甚至比“真正提升通用智能”更容易。

---

## 7. 那有没有办法避免？

没有完美方案，只能 trade-off：

* 私有测试集（会泄露）
* 动态生成 benchmark（仍可 reverse engineer）
* 人类评测（贵且不稳定）
* 多分布、多任务 aggregate（缓解但不消除）

结论是：

> **benchmark 只能作为训练过程中的弱信号，而不能作为最终能力证明**

---

## 8. 总结一句话

你说的这种做法：

* 在形式上不是 pretrain on test set
* 在实质上是 **test distribution injection**
* 在理论上是 **evaluation prior hacking**
* 在实践中是 **几乎不可避免、而且广泛存在的**

如果你愿意，我们可以继续往下讨论：

* 如何设计“抗反向工程”的 benchmark
* 为什么 scaling law 时代这种问题更严重
* 为什么有些 benchmark 反而在“惩罚”真正的通用能力
