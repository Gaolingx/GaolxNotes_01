# DAPO适合代理式RL GRPO难点解析

## 🧑‍💻 User

dapo为什么很适合agenticrl，grpo有什么问题

---
## 😀 默认助手 (7)

DAPO 本质上是给 GRPO 打了一组补丁，专门解决长序列、稀疏奖励下训练不稳、效率低的问题。Agentic RL 的轨迹更长、奖励更稀疏、截断更多，所以 GRPO 的这些毛病在 agent 场景里更严重，DAPO 的修补也就更有用。

## GRPO 的主要问题

GRPO 的目标函数（省略 KL 项）：

$$
J_{\text{GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(r_{i,t}\hat A_i,\ \text{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\Big)\right],\quad \hat A_i=\frac{R_i-\text{mean}(R)}{\text{std}(R)}
$$

1. **熵塌缩（entropy collapse）**：clip 是对称的，比如 $\epsilon=0.2$。一个概率只有 0.01 的 token 每步最多涨到 0.012，而高概率 token 几乎不受限制。结果是低概率的探索性 token 很难被拉上来，策略很快变得确定，同组采样越来越相似，探索就停了。
2. **零优势组浪费算力**：如果一组 $G$ 条回答全对或全错，$\hat A_i=0$，这一组没有梯度。训练越往后，这种组越多，有效 batch 越来越小，梯度方差变大。
3. **样本级 loss 平均带来长度偏差**：先在每条样本内部按 $1/|o_i|$ 平均，长回答里每个 token 的权重就被稀释了。后果有两个：
   - 长的错误回答（重复、乱码）惩罚不够，模型容易长出又长又烂的输出；
   - 长的正确推理里，关键 token 学得也不够。

   Dr. GRPO 也指出过，std 归一化会带来难度偏差（太难或太简单的题被放大）。
4. **截断样本的奖励噪声**：被最大长度截断的回答直接判 0 分，但它的推理可能是对的，只是没写完。这等于惩罚了"思考"本身，信号很噪。
5. **KL 惩罚多余**：做长 CoT 或 agent 任务时，策略本来就该大幅偏离初始模型，KL 项反而束缚了它。

## DAPO 的对应修复

| GRPO 问题 | DAPO 方案 |
|---|---|
| 熵塌缩 | **Clip-Higher**：上下界解耦，$\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$，给低概率 token 留出上涨空间 |
| 零优势组 | **Dynamic Sampling**：过滤准确率为 0 或 1 的组，持续采样直到 batch 填满有效样本 |
| 长度偏差 | **Token-level loss**：所有 token 一起平均，$\frac{1}{\sum_i |o_i|}\sum_i\sum_t(\cdot)$ |
| 截断噪声 | **Overlong filtering** 加 **soft overlong punishment**：截断样本要么 mask 掉，要么按超长程度线性扣分 |
| KL 束缚 | 去掉 KL 项 |

## 为什么特别适合 Agentic RL

Agent 场景把上面每个问题都放大了：

- **奖励更稀疏、更二值**：任务成功是 1，失败是 0。SWE、网页操作这类任务里，大量题目要么全组都失败（太难），要么全组都成功。零优势组的比例远高于数学题，所以 Dynamic Sampling 的收益很大。
- **轨迹极长**：多轮工具调用加上环境返回，轻松到几万 token。样本级平均造成的稀释更严重，token-level loss 几乎是必需的。而且环境 observation 的 token 要 mask 掉，token-level 聚合和 loss mask 也能很自然地配合。
- **截断非常普遍**：最大轮数、上下文窗口都会截断轨迹。被截断不代表做错了，不处理的话奖励噪声很大，overlong 的处理思路可以直接迁移过来。
- **需要持续探索**：agent 要尝试不同的工具和策略，熵一塌缩就会锁死在固定套路里。Clip-Higher 能把熵维持住。
- **策略偏离大**：异步 rollout 会让数据略微 off-policy，再加上去掉 KL 后允许策略大幅变化，解耦的 clip 能更好地控制更新幅度。

一个实际例子是 DeepSWE（Agentica）的 GRPO++。它基本就是 DAPO 的 Clip-Higher、去 KL、超长过滤，加上 Dr. GRPO 的去 std 归一化，再加 leave-one-out 优势估计组合起来的。

## 需要注意的局限

- **Dynamic Sampling 在 agent 场景很贵**：一次 rollout 要跑环境、调工具，耗时可能是分钟级，反复补采的代价很大。实践中常见的做法是只过滤、不补采，或者通过课程学习控制难度。
- **DAPO 没有解决多轮的信用分配**：整条轨迹共用一个优势值，不知道是哪一步工具调用起了作用。这方面要看 turn-level 或 step-level 的方法，比如 GiGPO、带过程奖励的方法等。
- **MoE 或超长序列下 token 级重要性比率依然可能不稳**：这也是后来 GSPO 改用序列级比率的动机。

总结一下：DAPO 适合 agentic RL，是因为它修的正好是"长、稀疏、易截断、需探索"这几类问题，而这些正是 agent 训练的常态。GRPO 原版是为较短的单轮推理设计的，直接搬到 agent 场景，很容易出现熵塌缩、有效样本少、长度失控的情况。

---
## 😀 默认助手 (7)

I'll verify the key technical details before answering.

Both references check out. Let me check whether there's specific work applying DAPO-style tricks to agentic training.

## 一句话结论

DAPO 的四个改动本质上都在修**"稀疏奖励 + 长且变长序列 + rollout 昂贵"**这三件事造成的优化器病。这恰好就是 agentic RL 的默认设定——所以 DAPO 不是"恰好也能用在 agent 上"，而是它的每一项修改都在 agent 场景里被放大器推到了最前面。反之，GRPO 的设计假设是**单轮、长度相对均匀、奖励可验证的数学题**，这些假设在 agent 里几乎全部失效。

---

## 一、先把 GRPO 的目标函数摆出来

GRPO 的核心（序列级平均 + 对称裁剪 + 组内标准化）：

$$\mathcal{L}_{\text{GRPO}}(\theta)=-\frac{1}{G}\sum_{i=1}^{G}\underbrace{\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}}_{\text{序列级平均}}\min\!\Big(r_{i,t}\hat{A}_i,\;\mathrm{clip}\big(r_{i,t},\,1-\epsilon,\,1+\epsilon\big)\hat{A}_i\Big)$$

$$r_{i,t}=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})},\qquad \hat{A}_i=\frac{R_i-\mathrm{mean}(\{R_j\}_{j=1}^{G})}{\mathrm{std}(\{R_j\}_{j=1}^{G})}$$

从这三处（`1/|o_i|`、对称 `clip`、除 `std`）就能推出几乎所有已知问题。

---

## 二、GRPO 的问题（按"在 agent 上疼不疼"排序）

| # | GRPO 的问题 | 机理 | 在 agentic 场景为什么被放大 | DAPO 对策 |
|---|---|---|---|---|
| 1 | **长度偏差**（最严重） | `1/|o_i|` 使每个 token 的梯度权重 $\propto 1/|o_i|$：长轨迹的 reward/penalty 被严重稀释 | agent 轨迹长度重尾（5k ~ 50k+ token，取决于调几次工具）；且"错误的长轨迹"惩罚不足 → 模型学会啰嗦、反复调工具、刷步数 | **Token-Level Loss**：$\frac{1}{\sum_i \lvert o_i\rvert}\sum_i\sum_t$ |
| 2 | **熵坍塌 / 探索窒息** | 对称 $\epsilon$ 下，概率空间的作用不对称：高概率 token 能以 $(1-\epsilon)$ 快速下坠（绝对幅度大），低概率 token 最多涨到 $(1+\epsilon)p$（绝对幅度小）→ 熵单调下降 | 长 horizon 下 $T$ 步成功率 $\approx\prod_t p_t$，必须探索到"罕见但正确"的动作序列（正确的工具+正确的参数+正确的顺序）；熵一塌，行为模式锁死（永远只 `search` 不 `python`） | **Clip-Higher**：$\mathrm{clip}(r,1-\epsilon_{low},1+\epsilon_{high})$，$\epsilon_{low}=0.2,\ \epsilon_{high}=0.28$ |
| 3 | **零方差组 = 零梯度** | 组内奖励全对或全错 → $\hat A_i\equiv 0$，整个 prompt 白采 | agent rollout 是全流程里最贵的资源（环境 reset、API、代码执行、浏览器）。难任务普遍全错、简单任务普遍全对，**大量环境预算换来零梯度** | **Dynamic Sampling**：只保留 $0<n_{\text{correct}}<G$ 的 prompt 并重采样，保证每个 batch 都有非零优势 |
| 4 | **除 std 的难度/方差偏差** | 组内方差极小时除 std 会把数值噪声放大成巨大优势；Dr. GRPO 还指出这带来难度偏置与长度偏置 | agent 奖励常是 0/1 outcome，甚至混入 format/cost 项，组内方差经常趋零 | 动态采样直接剔除退化组（另一条路线是 Dr. GRPO 的"干脆不除 std"） |
| 5 | **截断/超长处理粗糙** | 超长截断样本一律给 0 奖励，若还对 partial trajectory 算 loss，则注入大量噪声 | "死循环 / 反复重试 / 上下文塞满"是 agent 的**第一号失败模式**，不是边缘情况 | **Overlong Reward Shaping**（软长度惩罚，缓存区 $L_{cache}$ 内线性衰减到 $-1$）+ **对截断样本 mask loss** |
| 6 | **KL 约束** | 拉向 reference，限制偏离 SFT 行为的探索；还需常驻一份 ref 权重 | agent 的 SFT 策略只覆盖"能调工具"，真正要学的是**新的多步策略**，KL 主要在拖后腿；长序列下 ref 前向很贵 | DAPO **完全去掉 KL**（消融显示不加更好） |
| 7 | **序列级优势广播 → 多轮 credit assignment 极差** | 一个 $\hat A_i$ 广播到轨迹里所有 token，无法区分"第 1 步选错查询"和"第 5 步参数写错" | 多轮环境下奖励延迟、稀疏，不同 step 贡献差异巨大 | ❌ **DAPO 没解决这个**，需要 GiGPO 那类 step-level 结构（见下） |

<details>
<summary>补充：第 4、7 点的细节与证据（点击展开）</summary>

**关于 std 归一化。** GRPO 的组内标准化同时承担两个角色：(a) reward 尺度归一化（对多任务混合奖励是必要的），(b) 难度加权。当 $\mathrm{std}\to 0$ 时 (b) 会失控。DAPO 的做法是"把 std 趋零的组直接删掉"，保住了 (a) 但引入过滤偏置；另一条路线是保留组均值减中心但去掉除 std，并靠 prompt 级难度课程来补。两者在实践中常混用。

**关于多轮 credit assignment。** GiGPO（arXiv:2505.10978，NeurIPS 2025）明确把这一点作为 GRPO 在 agent 上的核心瓶颈：它引入 episode-level 的 macro 相对优势 + 基于 **anchor state grouping**（在跨轨迹中识别重复的环境状态，把同一状态下采取的动作聚成 step-level 组）的 micro 相对优势，在 ALFWorld 上比 GRPO 高 **>12%**、WebShop 高 **>9%**，且**显存、rollout 数量和耗时基本不变**。这是"GRPO 不够用"最直接的实证。

</details>

---

## 三、为什么 DAPO 的每一项在 agentic RL 里"刚好对症"

```text
agentic 的三个基本事实           →  DAPO 对应设计
─────────────────────────────────────────────────────────
长、变长、重尾的轨迹             →  Token-Level Loss
终局稀疏奖励 + 复合误差          →  Clip-Higher（保探索/熵）
rollout 含环境交互，极贵         →  Dynamic Sampling（不浪费组）
随时可能死循环/截断              →  Overlong Shaping + 截断 mask
SFT 只教会"能用工具"             →  去 KL（自由探索新策略）
```

再补几个实践层面的原因：

- **训练稳定性**：动态采样让有效 batch size 恒定 → 梯度量级可预测 → 可以放心用固定的高学习率（DAPO 用恒定 LR、无 KL、大 batch）。agent 训练里环境本身有随机性，优化器再抖就没法 debug 了。
- **可解释的失败归因**：去掉 KL 后，训练崩溃基本只剩"熵"和"截断"两个旋钮，agent 训练调参从玄学变成两三个可观测指标（entropy、clip fraction、truncation rate、group pass rate）。
- **工程侧默认值**：DAPO 的配方（clip-higher / token-level loss / 过滤零方差组 / 长度塑造）已经是 `verl` 系及各 agentic RL 框架里最常见的默认组合；很多所谓"agentic RL recipe"其实就是 GRPO + DAPO 技巧 + 环境 token loss mask。

---

## 四、诚实的边界：DAPO 不是 agentic RL 的完整答案

1. **它只修损失/优化器层，不修优势估计层。** 多轮 credit assignment 要靠 step/turn 级优势（GiGPO、turn-level GRPO 等），这一层是 DAPO 的空白。
2. **Token-level loss 有反向偏差。** 从"短轨迹权重过大"翻转为"长轨迹总权重过大"，而长轨迹往往是失败轨迹 → 可能出现过度压制长输出的现象，必须配合长度塑造、损失 mask 和对 env token 的严格 `loss_mask`。
3. **动态采样有代价与偏置。** 需要重采样（rollout 更贵，agent 尤其明显）；且系统性丢弃"全错"和"全对"的 prompt，意味着**最难的任务永远学不到**，需要课程学习或"过滤后跨 update 复用"来缓解。
4. **环境 token 必须 mask（最常被踩的坑）。** observation、工具返回、system prompt 不应进入 policy loss；在 token-level loss 下这一点更关键，否则优势会被环境 token 稀释。
5. **奖励/验证器才是上限。** agent 的 reward hacking（刷步数、假成功、绕过检查）不是任何 RL 算法能修的。

---

## 五、如果要做 agentic RL，最小可用的配置清单

1. 基座用 SFT 过的 tool-use 策略；**无 KL**（或 KL 系数极小）。
2. `clip_higher`：$\epsilon_{low}=0.2$，$\epsilon_{high}=0.28$。
3. **token-level loss**，且只对 $\mathcal{T}=\{$模型生成的、非环境返回的、非截断的$\}$ token 求和。
4. **动态采样**过滤零方差组（或去 std 归一化 + 难度课程），保证每 batch 有梯度。
5. **长度处理**：软长度惩罚 + 截断样本 loss mask；监控 `truncation rate` 作为一等指标。
6. **超出 DAPO 的部分**：加 step-level 优势（anchor-state 分组或 turn-level GAE）解决多轮 credit assignment；异步 rollout 时注意 off-policy 程度与重要性比裁剪的交互。

需要的话，我可以把 DAPO 的四个改动逐条对照 `verl` 里的实现位置，或者给一个 agentic 场景下的 loss mask + token-level loss 的伪代码。
