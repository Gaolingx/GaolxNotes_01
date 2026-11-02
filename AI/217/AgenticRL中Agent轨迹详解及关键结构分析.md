# AgenticRL中Agent轨迹详解及关键结构分析

## 🧑‍💻 User

agenticrl中的agent的轨迹到底是什么东西呢

---
## 😀 默认助手 (7)

## 一句话

在 Agentic RL 里，**轨迹（trajectory）就是“一次完整任务执行的全过程记录”**，形式是

$$\tau = \big(o_0,\ a_0,\ o_1,\ a_1,\ \dots,\ o_T,\ a_T\big)$$

其中 $o_t$ 是 agent 在第 $t$ 步观察到的内容（用户指令、上一轮的工具返回、网页、编译器报错、环境状态），$a_t$ 是它产出的动作（一段思考加一次工具调用或一段最终答案）。在论文里它对应 POMDP 的一条 episode；在代码里它对应一条带着 token id、mask、reward 的 dict。

关键对照就在这里：**普通 LLM-RL（RLHF/GRPO 做数学题）是退化成单步的 MDP**——$T=1$，$o_0$ 就是 prompt，$a_0$ 就是整段回答，$\tau = (x, y)$；而 **Agentic RL 把 $T$ 从 1 拉回几十上百步**，于是“轨迹”从一个标量对变成了一棵带时间结构的记录。这正是那篇综述（*The Landscape of Agentic RL for LLMs*, arXiv 2509.02547）的核心论点：LLM-RL 的单步 MDP 与 Agentic RL 的 POMDP 之间的区别。

---

## 一、轨迹的三层结构

理解这三层，基本就不会混淆了：

| 层级 | 是什么 | 谁关心它 |
|---|---|---|
| **Episode（轨迹本身）** | 从初始状态到终止的一次完整交互 | 奖励、advantage、评估、日志 |
| **Turn、Step** | 一次“模型生成 + 环境返回” | 多轮 rollout 调度、turn-level reward、上下文拼接 |
| **Token** | 真正的梯度单位 | 策略梯度、loss mask、logprob 重算 |

$$ \text{轨迹是 advantage 的单位，token 是梯度的单位。} $$

---

## 二、工程里一条轨迹到底长什么样

抽象定义不重要，重要的是它落盘时有哪些字段。典型的轨迹记录（简化）：

```json
{
  "trajectory_id": "run7-task042-3",
  "task_id": "swe-bench/django__1234",
  "agent_id": "solver",
  "model": "qwen3-8b@step1200",
  "sampling": {"temperature": 0.9, "top_p": 0.95, "seed": 42},
  "env_version": "docker-image-sha256:ab12...",

  "messages": [
    {"role": "system",    "content": "你是修复 bug 的工程师…", "mask": 0},
    {"role": "user",      "content": "<issue 描述>",            "mask": 0},
    {"role": "assistant", "content": "先看文件结构",
     "tool_calls": [{"name": "bash", "args": {"cmd": "ls -R"}}],  "mask": 1},
    {"role": "tool",      "tool_call_id": "c1",
                          "content": "src/\n tests/\n ...",        "mask": 0},
    {"role": "assistant", "content": "打开 utils.py",
     "tool_calls": [{"name": "read_file", "args": {"p": "src/utils.py"}}], "mask": 1},
    {"role": "tool",      "tool_call_id": "c2", "content": "...",  "mask": 0},
    {"role": "assistant", "content": "问题在边界判断，改成 <= 即可", "mask": 1}
  ],

  "input_ids":  [101, 2054, ...],
  "loss_mask":  [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, ...],
  "logprobs":   [null, null, -1.23, -0.44, ...],
  "turn_ids":   [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4],

  "reward": 1.0,
  "reward_components": {"tests_pass": 1.0, "format_penalty": -0.0, "turn_cost": -0.1},
  "num_turns": 4,
  "num_tokens": 18342,
  "finish_reason": "submitted",
  "success": true
}
```

下面几点是“轨迹”这个概念里最容易被忽略、但决定了训练是否报错的地方：

1. **`loss_mask` 用来区分“模型说的”和“环境说的”**。`role=tool` 的内容是环境产生的，不能算进 loss；否则模型会学着去**伪造工具返回**，这是 agentic RL 里最经典的失败模式。
2. **`logprobs` 记录的是行为策略（rollout 时的旧策略）的概率**，用于计算重要性比。异步训练时，它和当前策略差得越远越不稳。
3. **`turn_ids` 让“第几步”成为一级索引**，这样 turn-level reward、per-turn advantage、上下文截断才能被定位。
4. **`finish_reason`** 必须记录。`max_turns`、`max_tokens`、`tool_error`、`env_crash` 这些终止原因混在一起，会导致 reward 分布完全不可解释。

---

## 三、容易混淆的几个词

- **Context（上下文）**：每一次 forward 前拼出来的那串 token，动态重算，随 turn 增长。
- **Trajectory（轨迹）**：整段交互的**全部历史**，包含所有 turn 的 observation，是持久化对象。
- **Rollout**：**产生**轨迹的动作或过程（采样）。
- **Sample（训练样本）**：从轨迹里按 mask 切出来的“token + advantage”，是真正进入 batch 的东西。一条轨迹可能产生 1 个样本，也可能因为分段训练产生多个。
- **Trace / eval log**：同一条轨迹在评测或调试视角下的展示形态，多一份可读性、少一份张量。

一句话：**context 是视图，trajectory 是事实，sample 是切片。**

---

## 四、轨迹在训练算法中的双重角色

这是回答“为什么要专门定义轨迹”的关键。

**角色 A：advantage 的作用域。** 以 GRPO 为例，同一个 prompt 采样 $G$ 条轨迹，组内归一化：

$$ A_i = \frac{R(\tau_i) - \operatorname{mean}_{j}\big(R(\tau_j)\big)}{\operatorname{std}_{j}\big(R(\tau_j)\big)} $$

注意 $R$ 是**整条轨迹一个标量**。然后这个 $A_i$ 会被**广播到这条轨迹中所有 `loss_mask=1` 的 token 上**：

$$ \nabla_\theta J \approx \frac{1}{\sum_i |\text{valid}(\tau_i)|} \sum_{i}\sum_{t \in \text{valid}(\tau_i)} A_i \, \nabla_\theta \log \pi_\theta(a_t \mid h_t) $$

这就是为什么轨迹长度差异会直接变成偏差：一条 200 token 的成功轨迹和一条 20000 token 的成功轨迹，拿到同样的 $A_i$，梯度权重却相差 100 倍。实践中要么按 token 数归一化，要么加长度惩罚，要么对超长轨迹做截断或丢弃。

**角色 B：mask 与对齐的载体。** 多轮 rollout 中，第 3 轮的 observation 在第 2 轮训练时还不存在；off-policy 修正时又需要重算整套 logprob。所以轨迹必须是一个**自洽的、可重放的 token 序列**，而不是一堆拼接字符串。

---

## 五、轨迹的几种“非链式”形态

<details>
<summary><b>树搜索下的轨迹：一条路径，不是一条链</b></summary>

在 tree search、best-of-N、MCTS 类方法里，同一个 prefix 会被多条轨迹共享。此时一条轨迹是**从根到叶的一条路径**，需要额外记录 `parent_id` 或 `path`，否则无法正确计算 advantage（sibling 之间是配对的），也无法复用 KV cache。
</details>

<details>
<summary><b>多智能体：联合轨迹与子轨迹</b></summary>

多 agent 场景下有两种记录方式：

- **交织式**：所有 agent 的消息按时间顺序排进一个 joint trajectory，用 `agent_id` 标记归属。
- **分段式**：每个 agent 一条 sub-trajectory，靠 `causal_parent` 连接。

训练时的区别是信用分配：前者可以共享 team reward（联合 advantage），后者必须按 agent 拆开算（各自的 $A$，或者用 counterfactual baseline），否则会出现“搭便车”和“互相甩锅”。
</details>

<details>
<summary><b>轨迹的分组与过滤</b></summary>

轨迹还承担**数据治理**职能。一条 rollout 生成后，通常要过几道关：格式校验 → 环境校验（测试是否真跑通）→ 去重 → 难度过滤（全对的组梯度为 0，直接扔）。所以轨迹不只是 `(states, actions)`，它还是**一个带验证结果的数据单元**。
</details>

---

## 六、Agentic RL 中轨迹特有的难点

1. **部分可观测和非平稳环境**：同一个 action 在不同时刻可能返回不同 observation（网页变了、工具超时了），轨迹不可完全复现，同一条轨迹的重复采样方差很大。
2. **长度长尾**：轨迹长度分布通常极度右偏，几十万 token 的轨迹会撑爆显存，也会在 GRPO 里主导梯度。截断策略本身就是超参。
3. **上下文累积导致的计算量约为 $O(T^2)$**：$T$ 轮里每轮都要重跑前缀，这是 agentic RL 比单步 RL 慢一个数量级的主因，也是 prefix caching 必不可少的原因。
4. **off-policy 与 staleness**：异步框架里 rollout 轨迹由旧权重生成，训练时策略已更新好几版，重要性比会漂移，需要 clip 或重算。
5. **mask 对齐 bug 极难发现**：多轮 packing 后 token 位置错一位，loss 依然会降，只是降的是错的东西。所以轨迹要能**逐 token 回放检查**。

---

## 七、实际操作：怎么“读”一条轨迹

调试 agentic RL 时，看轨迹基本按这个顺序：

```
1. finish_reason 分布   → 是模型不行，还是 max_turns 太紧？
2. 每个 turn 的 tool_call 参数   → 有没有调错工具、传错 schema？
3. observation 是否被正确解析     → 工具返回的是 JSON 还是错误页？
4. 有没有循环/重复调用           → 典型的 reward hacking 前兆
5. loss_mask 覆盖率             → 是否意外把 tool 输出算进 loss？
6. reward 与 success 的一致性     → 验证器有没有被绕过？
7. 长度 vs reward 相关性         → 是否存在“越啰嗦分越高”的退化
```

---

## 小结

**Agentic RL 的轨迹 = 一个 POMDP episode 的完整、可重放的 token 级记录**。它同时是三样东西：算法上的**奖励与 advantage 作用域**、工程上的**mask 与对齐容器**、数据上的**可验证性单元**。单步 RL 里轨迹退化成 $(x, y)$ 一对，所以你可以不关心它；而一旦 $T > 1$，轨迹就成了这个领域全部工程复杂度的落点。
