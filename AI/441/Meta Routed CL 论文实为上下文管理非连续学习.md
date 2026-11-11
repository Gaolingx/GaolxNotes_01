# Meta Routed CL 论文实为上下文管理非连续学习

## 🧑‍💻 User

https://arxiv.org/abs/2610.38796

Meta 这个感觉只是上下文管理或者记忆系统，也能叫 continuous learning 吗🤔

---
## 😀 默认助手 (7)

## 速答：你的质疑对了一大半，但“错”的部分不在 Meta，而在这个词的用法上

这篇是 FAIR at Meta 的 *Routed Continual Learning*（arXiv:2610.38796）。它**确实一行参数都没改**，论文自己写得很清楚：

> "Memory systems interact with the agent through context; they neither act in the environment nor modify its parameters."
> "The router is prompted rather than trained. Its prompt and memory-system library remain fixed during evaluation."

它唯一会“变”的是：cluster 里的状态文本，以及 adapter 的 schema 绑定配置 $\theta_c$，而且后者是“bounded configuration”，由 LLM 现场生成 prompt 级别的绑定，不是训练出来的。因此从传统 ML 的定义看，这确实**不是** continual learning，而是一个带路由的 episodic memory 系统外加 prompt 注入 hook。

但问题在于：**如果你用这个标准，你会把 2025–26 年整个 agent memory 文献全都排除掉**——Dynamic Cheatsheet 自称“test-time learning with adaptive memory”，ACE、ReasoningBank、Mem0、ExpeL 全都这么叫，CL-Bench 这个 benchmark 本身也是把 memory system 当作 CL 候选在评。所以 Meta 是**沿用子领域的命名惯例**，并没有自己发明话术。

---

## 真正值得批评的地方：它自己的数据撑不起“CL”这个说法

与其争论“能不能叫”，不如看它拆出来的贡献：

| 对比 | 归一化 reward 提升（4 个模型） |
|---|---|
| Global ICL → Routed ICL（**只做上下文分桶**） | $+0.241,\ +0.219,\ +0.117,\ +0.171$ |
| Routed ICL → Routed CL（**加上 5 种“学习机制”的选择**） | $+0.002,\ +0.077,\ +0.034,\ +0.061$ |

也就是说：**“给上下文分桶”的收益是“选学习机制”的 3–100 倍**，Gemini 上机制选择只值 $+0.002$（作者自己都承认 “mean improvement is only 0.002”）。

所以剥掉包装，它的核心发现接近一句话：*在异构交错的流里，别把所有历史塞进一个 100K 上下文，按任务类型分桶后各自维护历史更好，而且更省 token。*——这确实是上下文管理/memory 工程，你的直觉命中。

再加三个可以直击的弱点：

- **没有遗忘、没有 backward transfer 的测量**。398 个 instance 单遍流过，每个源任务实例不回头重访，只能画出 forward transfer 的 learning curve。经典 CL 的 stability–plasticity 权衡一个都没量化。
- **学习信号是被掐掉的**：系统拿不到 reward，只能看到 task 界面自然暴露的 feedback；router 也看不到 raw trajectory。
- **元层面是静态的**：负责“怎么分桶”的 router 是 prompted & frozen，论文的 Ground-truth 诊断显示换成真值路由还能多涨 $0.069$–$0.071$（Opus 4.7 / Gemini）——即“学会怎么路由”这一层基本没学，只有 cluster summary 喂给 router 算是一点点 in-context 的弱适应。

另外那个 65–72% 的成本下降，本质是“分桶后每段历史更短所以 token 更少”，是**服务成本**贡献，不是学习贡献。FAIR 推这类工作，工程/成本动机可能比科学动机更重。

---

## 但也别全否掉：CL 的定义本来就在分裂

给它辩护的三条，我认为是成立的：

<details>
<summary><b>1. 非参数 CL 是一个合法的类别</b></summary>

CL 的核心判据是“第 $t$ 个任务的表现是 $1..t-1$ 经验的函数，且不破坏旧能力”，并**没有**规定必须改权重。外部记忆、in-context 适应、prompt 演化都属于 non-parametric CL。这个分支在 LLM 时代不是退化，而是主干之一——因为闭源 API 模型根本不允许你改权重。
</details>

<details>
<summary><b>2. 它的问题形式化是真 CL，不是检索</b></summary>

它把 CL 定义成一个**联合决策**：经验用什么表示（机制），以及哪些 instance 共享状态（cluster 划分）。这两个决策是耦合的，而且有明确 trade-off（分太粗会混淆，分太细丢失迁移），并且决策在**线、因果、无标签**做出。这个“状态分配”问题在经典 CL 里有对应物（task-conditional 容量分配、modular CL），论文把它形式化成 routing 是新的表述，不是把检索包装成学习。
</details>

<details>
<summary><b>3. 它证伪了一个流行假设，这是实打实的贡献</b></summary>

CL-Bench 原论文的结论是“朴素 ICL 最强”，但那是**每个任务独立跑同质流**。这篇把七个任务的 instance 交错成一条异构流后，Global ICL 相对 stateless 只涨 $+0.015$，Opus 5 上甚至全程不如 stateless。这说明“把历史全留着”这个默认做法的有效性高度依赖同质性——这是一个有信息量的负面结果。
</details>

---

## 我的判断

- **作为 CL 工作**：合法但偏软。它属于 non-parametric / test-time CL，而且是其中“机制最薄”的一端——状态更新是字符串和统计量，没有元学习，没有参数适应。
- **作为它自己宣称的“routing 问题”**：反而**被自己的数据削弱了**。如果“选机制”只值 $+0.002\sim+0.077$，那这篇的标题和核心 claim 撑不住；真正被验证的是“分区上下文”。审稿人最该问的是：*把 memory library 砍到只剩 ICL，你的主要结论还剩多少？* 答案基本没变。
- **更诚实的标签**：`test-time memory routing` / `continual memory orchestration` / `experience partitioning`。叫 CL 不算学术不端（子领域惯例如此），但确实借了 CL 的话语红利——“continual”暗示了跨时间的适应与保留，而这里跨时间积累的只有数据，不是能力。

一句话：**它把“记忆管理”做成了“学习”的样子，但它测出来的东西证明，收益来自管理而非学习。** 你觉得不对劲，是因为你在用“改参数/会遗忘/会变强”这套旧 CL 标准；而作者在用“状态随流演化”这套新标准。两套标准都合理，但混用会让标题读起来比论文本身更野心。

<details>
<summary>顺带一个方法论上的疑点（若想继续追问作者）</summary>

成本对比用的是**冻结的公开价目表**估算，而不是实测；并且没有做**等成本**对比（Global ICL 成本更高，所以“更便宜且更准”在 Pareto 上成立，但没做“把 Global ICL 的钱花在别处”的对照）。另外 Sales Prediction 的任务容器被改成每个 instance 重置、Codebase Adaptation 只给 base commit（Appendix A.4），这两处让结果与历史 CL-Bench 数字不可直接比较——作者诚实披露了，但也意味着“Global ICL 崩掉”这个核心论据有一部分是协议改动带来的。
</details>
