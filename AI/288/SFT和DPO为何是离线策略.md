# SFT和DPO为何是离线策略

### 🧑‍💻 User

为什么DPO和SFT是Off-Policy思路？

---

### 🤖 Assistant

这是一个非常深刻的问题，触及了现代大模型训练（LLM Alignment）与强化学习（RL）理论的交叉点。

简短的回答是：**因为在这两种方法中，模型（Agent）学习所使用的数据，不是由当前的策略模型（Current Policy）实时生成的，而是来自固定的、预先收集的外部数据分布。**

在强化学习（RL）术语中：
*   **On-Policy (同策略)**: “边做边学”。模型自己产出数据，自己从中学习，学完就丢掉旧数据（例如 PPO）。
*   **Off-Policy (异策略)**: “通过观察别人（或过去的自己）来学”。模型利用存储好的数据（Replay Buffer 或静态数据集）进行学习，产生数据的策略 ($\pi_{\text{behavior}}$) 与当前优化的策略 ($\pi_{\theta}$) 不同。

下面详细拆解为什么 SFT 和 DPO 都属于 Off-Policy 思路。

---

### 1. SFT (Supervised Fine-Tuning) 为什么是 Off-Policy？

在强化学习的视角下，SFT 等同于 **行为克隆 (Behavior Cloning, BC)**。

#### 核心逻辑
*   **数据来源**: SFT 的训练数据是 $(x, y)$ 对，其中 $y$ 是由人类（专家）或者更强的模型（如 GPT-4）撰写的。
*   **策略差异**:
    *   **行为策略 ($\pi_{\text{behavior}}$)**: 是人类或专家模型。数据是由他们生成的。
    *   **目标策略 ($\pi_{\theta}$)**: 是我们正在训练的模型。
*   **学习过程**: 模型并没有在环境中尝试生成句子并观察奖励，而是被强制要求去拟合专家的轨迹。模型在学习“别人的经验”，而不是“自己的探索”。

因此，SFT 是最典型的 Off-Policy 方法，它假设专家的动作永远是正确的，并试图最大化专家动作的概率。

---

### 2. DPO (Direct Preference Optimization) 为什么是 Off-Policy？

DPO 的提出正是为了绕过 PPO 这种复杂的 On-Policy 流程，将其转化为一个在固定数据集上的 Off-Policy 优化问题。

#### 核心逻辑
*   **数据来源**: DPO 需要一个偏好数据集 $\mathcal{D} = \{x, y_w, y_l\}$（提示词，胜者，败者）。这些数据通常是在训练开始前就已经收集好的（Offline Dataset）。
*   **数据的产生**: 这些 $y_w$ 和 $y_l$ 是由某个旧的策略（Reference Policy 或其他采样策略）生成的，而不是由当前正在进行第 $t$ 步梯度下降的 $\pi_{\theta_t}$ 生成的。
*   **与 PPO 的对比**:
    *   **PPO (On-Policy)**: 每一轮更新，模型必须先用当前的 $\pi_\theta$ 去生成一批新的回答，计算 Advantage，更新参数，然后**销毁**这些数据。旧数据不能复用，因为分布变了（Importance Sampling 只能处理微小的分布偏移）。
    *   **DPO (Off-Policy)**: DPO 直接在静态数据上反复迭代。它通过数学推导，将 RL 的奖励最大化目标转化为了一个类似分类损失的函数。虽然 DPO 内部隐含了对 $\pi_{\text{ref}}$ 的依赖，但它不需要模型在训练过程中实时与环境交互采样。

#### 数学视角的 Off-Policy
DPO 的推导基础是将最优策略 $\pi^*$ 表示为参考策略 $\pi_{\text{ref}}$ 和最优奖励 $r^*$ 的函数。它本质上是在优化一个静态的偏好分布，不需要实时采样估计期望回报。这种对“离线偏好数据”的利用方式，符合 Off-Policy 的定义。

---

<details>
<summary><strong>📐 点击展开：数学公式层面的深度解析</strong></summary>

#### SFT (Behavior Cloning)
SFT 的目标是最大化似然函数：
$$ \mathcal{L}_{SFT}(\pi_\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{expert}}} [ -\log \pi_\theta(y|x) ] $$
这里的期望 $\mathbb{E}$ 是在专家数据分布 $\mathcal{D}_{\text{expert}}$ 上计算的，而不是在模型当前分布 $\pi_\theta$ 上计算的。这就是 Off-Policy 的标志。

#### DPO vs PPO

**RL 的一般目标:**
$$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [r(x,y)] $$
On-Policy 方法（如 PPO）必须通过蒙特卡洛采样来近似这个期望，即 $y \sim \pi_\theta$。

**DPO 的目标:**
DPO 通过闭式解变换，导出了如下损失函数：
$$ \mathcal{L}_{DPO}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] $$

请注意期望 $\mathbb{E}$ 的下标：$(x, y_w, y_l) \sim \mathcal{D}$。
这里的数据集 $\mathcal{D}$ 是固定的离线数据。模型 $\pi_\theta$ 的更新不需要重新生成 $y_w$ 或 $y_l$。模型是在评估“如果我（$\pi_\theta$）生成这两个数据的概率比值，与参考模型（$\pi_{\text{ref}}$）相比，是否符合人类偏好”。

因为训练数据的分布固定为 $\mathcal{D}$，不随 $\pi_\theta$ 变化而变化，所以这是 Off-Policy。

</details>

---

### 总结对比

| 特性 | SFT (Supervised Fine-Tuning) | DPO (Direct Preference Opt) | PPO (Proximal Policy Opt) |
| :--- | :--- | :--- | :--- |
| **RL 类型** | Off-Policy (Behavior Cloning) | Off-Policy | **On-Policy** |
| **数据来源** | 固定的专家演示数据 | 固定的偏好对数据 (Winner/Loser) | 模型**实时生成**的数据 |
| **行为策略** ($\pi_{\text{behavior}}$) | 人类专家 / 教师模型 | 历史模型 / 数据采集策略 | **当前的** $\pi_\theta$ |
| **训练效率** | 极高 (由于数据利用率高) | 高 (无需实时采样) | 低 (采样昂贵，数据用完即弃) |
| **分布偏移 (Shift)** | 有 (Exposure Bias) | 有 (Data vs Model distribution) | 较小 (通过 Clip 限制偏移) |

**一句话总结：**
SFT 和 DPO 之所以是 Off-Policy，是因为它们**不需要**模型在训练过程中实时生成数据来探索环境，而是直接利用**已经收集好的静态数据**来更新策略。