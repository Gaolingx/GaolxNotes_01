# EDM2 是删自由度的数学方法，你的 gain budget 是深度指数的通用化

## 🧑‍💻 User

我在做的是数值稳定性的东西 是model arch跟optimization
就是x = x+f(norm(x))是现在最常见的模型架构不是吗
x = x + attn(norm(x))
x = x + mlp(norm(x))

但是很多transformer模型（包括llm mmdit）、VAE这种都有各种奇怪的residual stream growing问题
hidden state growing通常会造成灾难性后果所以大家会乖乖去修正
所以有问题的模型九成九是residual

为什么weight decay没有把这个漂移拉回来
主要是 这个漂移的规模太大了
而且RMSNorm的漂移更恶心
RMSNorm他是一两个channel飘移几百倍
你weight decay拉不回来啊是整体拉低的 个别值是别人的几十倍
靠特制的outlier decay来修的

所以现在大家norm越加越多实际上是牺牲了hidden state/residual stream的数值稳定性换来了block内部的稳定性(当然 也没多稳定)
有没有什么能治本的办法：
1. 绝大多数人在：不管magnitude，跟我的norm还有bf16说去吧
2. EDM2在: 所有hidden state都必须magnitude一致，所有操作必须magnitude preserving
3. 折中的办法: 禁止任何free moving(magnitude飘移不影响loss/learning)的行为，但moving随意，反正你能move又符合loss要求那就是好的 我发现你不需要所有hidden state都magnitude preserving而是要避免任何"免费飘移" 然后增加norm是会增加free moving的区块的

---
## 🧑‍💻 User

所以现在大家norm越加越多实际上是牺牲了hidden state/residual stream的数值稳定性换来了block内部的稳定性(当然 也没多稳定)
有没有什么能治本的办法：
1. 绝大多数人在：不管magnitude，跟我的norm还有bf16说去吧
2. EDM2在: 所有hidden state都必须magnitude一致，所有操作必须magnitude preserving

---
## 😀 默认助手 (7)

## 先给判断

你的 #3 我认为方向对，但可以再收紧一格：你描述的“禁止 free moving”在数学上就是**局部规范固定 (local gauge fixing)**，它和 weight decay 的差别根本上是**作用在哪个空间** —— 而现在的整套 stack 恰好在两个层面都做了 scale-invariance，把漂移的可见性和负反馈全掐掉了。

一句话总结病根：**residual stream 是整个网络里唯一既不被归一化、也不被正则化、却唯一负责累加的张量。**

---

## 1. 为什么 weight decay 结构上治不了

不是“力度不够”，是四条同时错位：

**① 作用变量错（state vs. parameter）**
$$x_{l+1}=x_l+f_l(\mathrm{rms}(x_l))\quad\Rightarrow\quad x_L=x_0+\sum_l f_l$$
这是 $z=1$ 的**纯积分器**，没有漏项。WD 缩 $\|W_l\|$ 只让每项按 $(1-\delta)$ 缩，而漂移是**和**。要压住 $\|x_L\|$ 得让所有 block 缩 $1/L$ —— 用全网络功能退化换一个局部病症。这对任何 $\lambda$ 都成立。

**② 作用几何错**
$\lambda\|W\|^2$ 各向同性。你的漂移是**一两个 channel、接近 rank-1** 的。WD 对秩 1 方向和其余方向等比例收缩，所以 $x_i/\mathrm{rms}(x)$ 这个**比值**在 WD 下近似不变——这正是“整体拉低、个别仍是别人几十倍”。你想要的 ratio penalty，WD 给的是 absolute penalty。

**③ 双重 scale-invariance，负反馈被消掉两次**
$$\frac{\partial\,\mathrm{norm}(cx)}{\partial \ln c}\equiv 0$$
所以 block 自己的 read 对径向分量的梯度**恒等于零**，径向梯度只能从 identity path 来，而它是所有下游 $L-l$ 个 read 诉求的**带符号求和** → 近似零均值噪声。更糟的是 Adam 对梯度 scale 不变：就算 $\mathrm{rms}$ 变大把进入 block 的梯度压小 $1/c$，Adam 的更新步长不变 → “漂移会拖慢学习”这个唯一的负反馈也没了。

**loss 看不见（③）、优化器感觉不到（③）、WD 又不在同一个几何里（①②）** → **state space 里根本没有回复力**。WD 只在 weight space 里有，而 weight→state 的映射被 norm 削平了，且维持一个漂移所需 weight 量级是 $O(1)$ 的小量，$\lambda\cdot O(1)$ 的势垒对梯度噪声完全不构成约束。顺带一提：WD 在**参数**的规范自由度上其实是正确工具（这正是 AdamW 之于 normalization-heavy net 的地位），现代 stack 恰好只在参数那边做了规范固定，state 那边一片空白。

**④ 时间常数错**
WD 回复时间 $\sim 1/\lambda$ 步；漂移是每个 forward 重新注入、按**深度**累积的结构量。

<details>
<summary><b>【推导】为什么是“线性累积 → 几百倍”，以及为什么必然集中在少数 channel</b></summary>

norm 的输出范数恒为 $\sqrt d$，所以 $\|f_l\| \le \|W_l\|\,\sqrt d\,\|g\|$ 有界 ⟹ **norm 把指数漂移换成了线性漂移**（无 norm / post-norm 时 $f$ 正比于 $\|x\|$，指数爆掉）。这就是 Pre-LN 稳但漂、Post-LN 直接炸的机制区别。

而 linear 意味着：某 channel 每层有恒定正偏置（softmax sink、attention 输出的均值分量、SiLU 的非零均值…），则 $x_l[i]\approx c\,l$；其余 channel 走 $\sqrt l$ 随机游走 → 比值 $\sim\sqrt l \sim l$。100 层 → 几十到几百倍。**你观察到的“一两个 channel 飘几百倍”是这个线性累积的直接推论**，也解释了为什么它一定集中在少数 channel：谁先被选成 sink，谁拿正反馈。
</details>

## 2. 为什么“norm 越加越多”反而更糟 —— 你的 trade 可以更狠地说

不是“牺牲 residual 换 block”，而是：**每个 norm 把一份规范自由度从 block 内部搬到了 stream 上。** block 内部有 WD、有 loss 梯度、有 per-op 的 magnitude 设计；stream 上什么都没有。所以 norm 越密，被搬到“无人区”的自由度数越多。

## 3. 你的 #3 的严格版本

> 不要求所有 hidden state magnitude-preserving，只要求：**每个可累积张量的尺度都由 loss-sensitive 的量决定；不存在二阶导为 0 的非零测度方向。**

按成本三档（我按“能不能今天就上”排）：

**A 档｜标量 gauge 固定（一行代码）**
```python
y = x + block(rms(x))
y = y * (tau / rms(y))     # tau 常数 或 每层 learnable log-tau
```
额外好处：它同时消掉了“层间对 stream 尺度诉求不一致”这个**噪声源**（所有层锚到同一个 $\tau$），并让每层新增比例恒定 → 后层不被前层淹没。DeepNorm 是它的“只在 init 做”版本，这个是每步做。注意用 learnable $\tau$，因为“stream 变大”可能承载了“后期层影响力衰减”这个有效 curriculum，硬常数会掉点。

**B 档｜leaky integrator（唯一真正 bounded 的）**
$$x_{l+1}=e^{-\epsilon}x_l+f_l(\mathrm{rms}(x_l))$$
谱半径 $<1$，输入有界 ⟹ 状态有界，增益 $\le 1/\epsilon$。**关键：gate 必须作用在 state 上，不能作用在 update 上。** $x+\alpha f$（LayerScale / ReZero 那类）完全无法 bound 已有 state，只是慢一点注入——这是最常被搞混的一点。Mamba 的 $\bar A=e^{\Delta A}$ 就是这个，SSM 天生没有 residual growth 正是因为它的 $\bar A$ 是收缩的。代价：深度上有限记忆，用 $\epsilon\sim10^{-3}$ 换 $\sim1000$ 层记忆基本无损。

**C 档｜固定 profile，而不是固定范数（这条是对 EDM2 的批评，也是对你问题的直接回答）**
$\ell_2$ 保持**不够**。$\|x\|_2$ 固定时单通道仍可吃掉全部能量：$\|x\|_\infty/\|x\|_2$ 可以从 $1/\sqrt d$ 涨到 $1$。**真正伤 bf16 和梯度的是通道间动态范围（profile），不是标量。** EDM2 的 UNet 里 channel 同质，所以 $\ell_2$ 够用；transformer 的 per-token outlier 本质是各向异性 pathology，$\ell_2$ 守恒对它完全瞎。所以要 pin 的是 $\|x\|_\infty/\mathrm{rms}(x)$ 这类 ratio。做法：每次 residual add 后做 per-channel RMS 归一 + learnable per-channel gain —— 这是**把规范自由度参数化**，搬到参数上之后 WD 就真的有效了（这正好是你 #3 想要的“允许 move，只要不是免费 move”）。

⚠️ 还有一个方法论结论：**规范自由度应该 hard fix，不该 soft penalize。** 因为 free 方向上 loss 本就没有反向力，penalty 只是把一场你已经知道会输的拔河重新摆出来（diffuse 罚 vs. 集中收益）。

**D 档｜功能解耦（最便宜的根因消除）**
先诊断 outlier 是不是“有用”的：causal mask + softmax 必须有个地方 dump 概率质量 → attention sink → 少数 channel 变成常数列。这时它**不是 free，而是在用一个极贵的方式实现一个 bias 参数**：污染全局 RMS、把其他 channel 的梯度按 $1/c$ 缩放、让 fp32 的平方和丢掉小通道贡献。诊断出来的话直接给它一个参数（register token / 显式 sink logit / pre-softmax bias），drift 自己会掉下去。

<details>
<summary><b>【细节】bf16 上真正的伤害路径（定义你的目标函数用）</b></summary>

不是 overflow —— bf16 动态范围巨大。真正的三条：
1. $\mathrm{rms}^2$ 的 fp32 累加被大通道吃掉小通道贡献（$10^8$ vs $1$ 已经超出 24-bit 尾数）；
2. 梯度被 $1/\mathrm{rms}$ 缩放 → 少数通道 effective 学习率 1/100，变成双速网络（Adam 只救回一部分，噪声比还是坏的）；
3. 大通道的归一化输出饱和在 $\sqrt d$，继续长大只把别人压小 —— 白占容量 + 白占精度。

所以**你的目标应该定义成“per-channel ratio 有界”，不是“$\|x\|$ 有界”**。这个定义差异决定了 A 档够不够。
</details>

## 4. 风险与我不确定的地方

- **方向漂移没被任何方案覆盖**：stream 可以累积与所有有用子空间正交的分量，$\ell_2$ 守恒和 profile 守恒都允许。这块我没看到干净解，可能得做 per-layer stream 的低秩/谱审计。
- **硬 pin 会不会掉点**：EDM2 说不掉，但 diffusion/UNet ≠ LLM。所以我坚持 learnable $\tau$ 而不是常数。
- **$\mu$P 不覆盖这件事**：它管 init 时的宽度缩放，不管训练中的累积。训练中实际漂到的 $\tau$ 和 $\mu$P 理论值差几个量级，这件事本身就是“漂移不在 $\mu$P 的可控范围里”的证据。

<details>
<summary><b>【下一步】最小诊断三件套 + 消融</b></summary>

每层、每 token 记：
- $\mathrm{rms}(x_l)$（绝对值）
- $\max_i|x_l[i]|/\mathrm{rms}(x_l)$（profile，这才是会爆的）
- $\arg\max$ 的 channel 身份是否**逐层稳定**（稳定 ⟹ 功能性的 sink，不稳定 ⟹ 随机游走）
- 每 channel 的 $\|\partial L/\partial x_l\|$ ratio

然后四个消融：常数 $\tau$ pin / learnable $\tau$ / leaky $\epsilon$ / 显式 sink 参数。看 ratio 曲线和 loss 的联合形状，而不是只看 loss。
</details>

---

要不要我在沙箱里跑一个 toy（纯积分器 + RMSNorm block + Adam/AdamW，20 行 numpy）？我想验证两个可证伪的预测：**(a)** per-channel ratio 随深度线性涨；**(b)** 加 WD 只让 $\|x\|$ 整体平移，ratio 曲线几乎不动 —— 如果 (b) 成立，你的“WD 结构上无效”就不是经验观察而是可以打印出来的曲线。

---
## 🧑‍💻 User

我们的做法是
把权重的每一个vector单独看都保证他是unit vector的方法
然后这样子可以解决怎样怎样的问题这样
还有怎么update可以保证这个性质
其实就是EDM2的复杂rule给它简化成两个操作
单独做都有效 并且比EDM2更general
EDM2必须对复杂结构单独处理 但我们分析了之后确认了某些条件下才要考虑单独处理 很多时候放着不管让模型学就行
EDM2 -> 万物皆掌控于我手
我 -> 只要保证那些乱涨也不影响loss的东西被掌控就行了，其他模型自己学

其实最简单的概念就是
我们知道x  = x+f(norm(x))对吧
然后这个x out是下一个norm的输入
那很明显下一个输入都norm了 你前面的x out乘以几倍不影响的rms/layer norm都对scaling insensitive
那就很可能出现前面几层暴力增加residual stream的scale 因为output head可能init值太小了
然后momemtum又给他继续冲
然后residual stream越来越大 然后各层为了传递资讯 output projection也越来越大(不然你ratio太低)
然后output head就炸开了
那我的解决办法就是free moving区块对权重做unit vector的限制
所以你freemoving但是不让你free
普通moving的区块继续move 允许任意scale output
EDM2会不允许任意scale output的 因为万物皆preserve

---
## 😀 默认助手 (7)

## 你这一步做对的东西，比你说的更强

先给一个更精确的定位：你的两个操作是把**参数空间紧化 (compactification)** —— 从 $\mathbb{R}^{d}$ 换成 $S^{d-1}$ 的乘积。这解释了一件你 causal chain 里点到但没展开的事：

> “然后 momentum 又给他继续冲”

**这句话其实是你方法的机制级证明。** 漂移的来源是 momentum 的**径向分量**：$\mathbb{E}[g_\perp]$ 可以零均值，但 $m_t=\beta m_{t-1}+g_t$ 的时间常数是 $1/(1-\beta)$，所以径向梯度噪声被积分成一个 $O(\sigma/\sqrt{1-\beta^2})$ 的**持续推力**。而参数空间的径向方向恰好没有回复力（因为下一层 read 是 norm，$\partial/\partial\ln\|W\| \equiv 0$）。于是参数空间里的积分器（momentum）× state 空间里的积分器（residual） × 零梯度方向 = 漂移。

约束到球面后，$\mathrm{proj}_{T}(m)$ 把径向分量**精确消掉**：momentum 只能转，不能推。这不是"压住了"，是**该方向从参数空间里被删掉了**。

顺带一个漂亮的推论：**在你的约束下，weight decay 恒等于零。**
- decoupled: $w \leftarrow (1-\eta\lambda)w$ 再 renormalize → 完全复原；
- coupled: $\nabla\frac{\lambda}{2}\|w\|^2=\lambda w$ 是纯径向 → 投影后为 0。

所以 WD 在那些 vector 上**可以被删掉**。而这件事本身就是你那套 thesis 的**证明**：WD 在那条 orbit 上的唯一功能就是做规范固定，你把 gauge 直接固定了，它就没事可做了。这句话可以直接写进 paper 的 introduction。

---

## 但你的 claim 要劈成两层，否则会被打

这是我认为最需要修的一点。**“整个矩阵的标量尺度”和“每一行的尺度”在 gauge 论上完全不是一回事：**

$$f(W,x)=\mathrm{read}(Wn(x)),\quad \mathrm{read}=\text{norm}\ (\text{degree }0)$$
- $W\to cW$：$\mathrm{read}$ 完全看不见 → **exact gauge，约束它零 loss 代价，可证**。
- $W$ 的第 $i$ 行 $\to c_i w_i$：改变的是输出 channel 的 profile，而 RMSNorm 的分母是**全局**的 → **loss 看得见**。这一层不是 gauge fixing，是 **一个真实先验**（"所有输出 channel 的 gain 预算相等"）。

这个区分很重要，因为：
1. **Tier 1（标量）**：零代价，可以直接宣称"免费"。EDM2 的 $w/\sqrt{\text{fan\_in}}$ 就属于这一层（只是它把 gauge 值钉死成一个具体数）。你把它放开成"半径钉住、方向自由"，这是**严格更一般**的。
2. **Tier 2（per-row）**：有代价，但它是你真正想解决 profile pathology 的那一半。好处是它有大量既有证据说"良性"：WeightNorm 里的 $w=g\,v/\|v\|$（你等价于把 $g$ 冻成常数）、Weight Standardization、以及 per-row 归一在稳定性 recipe 里的常见用法。

所以你的"比 EDM2 更 general"应该这样表述，才站得住：

> EDM2 在整个丛上选了一个**具体的截面**（每个 op 一条 functional-form-specific 的规则，把 gauge 值钉死）；我们只在 **flat 子丛**上取截面，半径钉住、方向与其余尺度全部交给模型。因为只在 gauge orbit 上动手，所以代价可证为零；EDM2 的规则里有一部分（比如 $\sigma$-conditioning / preconditioning 那套）根本不是在处理 gauge，而是在服务 diffusion 的另一个目标 —— 所以"更 general"不是因为 EDM2 冗余，而是因为 SLM/LLM 不需要付那部分。

<details>
<summary><b>【update rule 的三个坑】</b></summary>

你说你已经有 update 保证了性质，这三个是最容易静默出错的：

1. **$v$ 必须在切空间里更新。** 标准 Adam 的 $v_t=\beta_2v_{t-1}+g_t^2$，如果 $g$ 用的是 ambient 梯度，$v$ 里掺了径向分量（且 $\mathrm{cov}(g_\text{rad}, g_\text{tan})\neq0$），会系统性压小沿 $\hat w$ 对齐的那些坐标的有效步长 → 你看到的是"某些方向莫名学得慢"。正确做法：先 $g\leftarrow g-\langle g,\hat w\rangle\hat w$，再进 $m,v$。
2. **normalize 必须在 momentum 之后。** `w = w - lr*m; w /= norm(w)` 是对的；`m = project(m)` 之后再 normalize 也行，但**不要**只 project 不 normalize —— 浮点下会缓慢长出径向分量。另外注意 normalize-after-step 的步角是 $\arcsin(\eta\|g_\perp\|)$ 量级，LR 的物理意义变了，得重扫。
3. **重参数化 $\neq$ 投影。** $w=u/\|u\|$ 然后对 $u$ 跑 Adam：Jacobian 是 $(I-\hat w\hat w^\top)/\|u\|$，Adam 的逐坐标 $v$ 活在 $u$ 坐标系里，跟球面度量不对齐 → 隐含了一个 $1/\|u\|$ 的 LR schedule + 坐标权重畸变。工程上更省事，但**理论性质只在 $w$-空间投影下干净**。如果你要写"等价于 Riemannian/Normalized GD"，就用投影版。

补一句：投影版 update 在数学上就是 **Normalized Gradient Descent（球面乘积上的）**，它有一个已知的隐含 bias —— 走向 max-margin 解。这是双刃：可能白拿鲁棒性，也可能让你丢掉"靠放大某一行 gain 来锐化某个特征"的能力。这是可测的（见下）。
</details>

---

## 我认为最关键的一个漏洞：闭包 (closure) 不成立

你的规则写作"对 free-moving block 的权重做 unit vector 限制"。但**只要路径上存在任何一个没被 pin 的乘子，自由度就从那里漏出去，unit norm 白做**。这是我认为它现在离"general"还差的一步。

设路径上从"normed read"到"normed write"之间所有可学习标量乘子为 $s_1,\dots,s_k$（含 fan-in 归一、$g$、$\alpha$ 等），则需要

$$\prod_j |s_j| \equiv 1 \quad (\text{或} \equiv \text{钉死值}),\ \text{对每条这样的路径}$$

具体的漏点，按我在 transformer 里见到的频率排：

| 漏点 | 为什么漏 |
|---|---|
| **norm 的 learnable gain $g$** | 你约束的是 $W$ 的行范数，但实际映射是 $W\mathrm{diag}(g)$，行范数是 $\|w_i\odot g\|$。$\|g\|$ 涨 10 倍 = 你的约束被抵消。**正确对象是 $W\mathrm{diag}(g)$ 的行**，或者让 $g$ 也在球面上。 |
| **LayerScale / ReZero 的 $\alpha$** | `x + alpha*f` 里的 $\alpha$ 就是被你禁掉的 free gain，它以标量形式复活了整套 gauge。 |
| **bias** | 无约束 bias 是最理想的漂移把手（也正是 attention sink 的实现方式）。free 路径上要么删 bias，要么把 bias 也纳入范数约束。 |
| **embedding / pos-emb 的 scale** | 训练中会长（或相对缩小）→ 改变 stream 的初始尺度，等价于给整条前缀换了一个 gauge 值。 |
| **attention temperature / per-head scale** | 见下。 |

这条我建议直接上升成方法的一条 axiom：

> **任何可学习尺度，必须挂在 loss-visible 的 consumer 上；否则它就是一个漂移把手，必须被 pin 或删除。**

这条比"对 free block 做 unit norm"更本质，而且它解释了**为什么 EDM2 看起来规则那么复杂** —— 它其实就是在手工地对每个 op 做这个闭包，只是因为 diffusion 的 op 种类多，所以规则多。你把闭包条件写出来，就可以在**不动 op 种类**的情况下自动判定哪些乘子要 pin。这才是"更 general"的真正论据。

---

## 一个你可能没注意到的红利，和一个代价

**红利**：unit rows 在 $W_Q,W_K$ 上自动给出 QK-norm 的效果。
$$\|q_i\|=\|W_{Q,i}\,n(x)\|_2 \le \|n(x)\|_2 = O(\sqrt d)\ \Rightarrow\ |q^\top k| = O(d)$$
logit scale 被结构性 bound 住 → **你的方案可能让 QK-norm / logit softcapping 变成多余**。这是个很好的 ablations 卖点（"一个约束替代了三个 patch"）。

**代价**：它同时也**封顶**了模型 sharper attention（温度退火）的能力。如果掉点，你大概会想加回 per-head 可学习温度 —— 但它按上表就是新的 free handle，必须一起进闭包（比如把温度绑到 $W_Q$ 的行范数上，而不是独立参数）。

---

## 你还差另一半：state 没有紧化

参数空间现在紧（乘积球面 → 范数爆不了，方向只能随机游走但在紧集里，安全），但 **residual stream 本身仍然是非紧的**。所以你的预测应该是：

- **权重侧爆掉** → 被你的约束根除；
- **stream rms 仍可能线性涨**（只要每层 $f$ 有恒定正均值 / sink，$x_l\approx cl$），只是**不再有放大器**：因为 $\|f\|$ 现在被 $\|W\|_{\text{row}}=1$ 与 normed input 双双钉住，$\|f\|/\|x\|\to0$，漂移变成"自限幅"的——但它**不会自己回来**。

所以补上我上次说的 A 档（每层 `x *= tau/rms(x)`，$\tau$ learnable 或很小的 leaky $\epsilon$），整个系统才是 closed 的：

$$\text{参数紧} + \text{state 有界} + \text{increment 有界} \Rightarrow \text{Lyapunov 稳定}$$

这也给你一个**可证伪的判别实验**，用来分开两种漂移源：

| 观察 | 结论 |
|---|---|
| 加了 unit-norm 后 `rms(x_l)` 曲线**变平** | 漂移主要来自权重尺度放大 → 你的理论对 |
| 加了 unit-norm 后 `rms(x_l)` 仍线性涨，但 `max_i|x_i|/rms` 变平 | 漂移来自 $f$ 的**均值/sink/bias**，需要显式 sink 参数（根因消除）而不是约束 |
| 两者都没变 | 漂移来自 norm 的 $g$ 或某个漏掉的乘子 → 回去查闭包表 |

这三条曲线的组合能直接告诉你"free moving 区块"到底在哪，比手写判定规则可靠。

<details>
<summary><b>【prior art 对照 / 避免审稿人打】</b></summary>

- **Weight Normalization**（Salimans & Kingma）：$w=g\,v/\|v\|$。你是它的 $g\equiv1$ 特例（per-row）。要解释"为什么去掉 $g$ 是 feature 不是 bug"—— 因为 $g$ 就是 free handle。
- **Riemannian / Stiefel SGD**（Wisdom et al., unitary RNN）、**product-of-spheres optimization**：你的 update 属于这一族，别当成新东西卖，卖"用在哪些 tensor 上 + 闭包判据"。
- **Weight Standardization**（Qiao et al.）、**WS+GN**：per-channel gain 拉平这件事的良性证据。
- **EDM2**（Karras et al. 2024）：映射到 Tier 1 + 具体 gauge 值 + 与 gauge 无关的 diffusion 规则。
- **μP**：管 init 的 width 缩放，不管训练中的累积 —— 所以你的方法和 μP 是正交的，可以叠加，这点值得明说（很多人会问）。
</details>

---

## 一个我需要你确认的点

你的 causal chain 里“**然后 output head 就炸开了**”—— 如果 head 前面有 final RMSNorm（现代 LLM 都有），那 stream 怎么涨都传不到 head，head 的输入被完全 scale-decouple 了。所以 head 爆掉要么是：

1. **没有 final norm**，或
2. head 的 logit 增长是**loss-visible 的、故意**的（模型在自学 temperature / margin 退火）—— 这属于你规则里"普通 moving，允许任意 scale"的那一半，**不该管**。

这两者在你框架里的地位完全不同。如果是 (2)，那你的框架其实在说一件更漂亮的事：**同一个现象（某权重范数在涨）在 gauge 方向上是有害的、在 loss-visible 方向上是有益的，而你的两个操作能自动区分它们** —— 这才是对 EDM2 的真正超越，而不是"更少规则"。建议你把这条做成核心 claim，比 stability 更好卖。

要不要我把上面那张"漂移源判别表"写成一段可跑的 toy 代码（纯积分器 + RMSNorm block + 球面投影 Adam），把三条曲线一次打出来？我可以在这边直接跑。

---
## 🧑‍💻 User

挺好的，数值稳定性这块感觉一直缺一个通用的理论
我发现我这个分析方法确实没人用
"免费漂移"这个，感觉和动力学分析里面的病态解很像
所以我觉得是现在模型整体动力学特性出了问题 甚至现在模型设计根本没考虑过动力学特性

我告诉你是怎么样吧
是大家的设计方案主打一个鸵鸟
他们考虑的不是解决问题是隐藏问题
residual grow了就加norm residual又grow了就用bf16
经典面多了加水，水多了加面
要是面多加水水多加面也还好
他们是面多加水结果面变更多了
他们选择加更多水
而不是去看看面怎么可以在加水之后变多了
然后我的做法是 我把那些面多加水会给你加面的模块给限制住不让加面了
这时候再去面多加水水多加面 就健康了

大家: 面多加面那我就用更大的箱子，水多加水那我就用更大的水桶
我: 面多加面禁止，水多加水禁止，面多加水水多加面挺好的
EDM2: 不准加
这个类比应该更值观 AI也有数值膨胀

---
## 😀 默认助手 (7)

## 把你这个类比翻译成控制论，它就变成了一句可证伪的定论

面 = 作动器（写入侧的 gain 预算）；水 = 归一化。

现在全场的做法是 **mask 传感器**：norm 把信号的尺度从 loss 的视野里抹掉。
你的做法是 **saturate 作动器**：unit-norm 把写入侧的 gain 预算钉死。

这一句就是分野，而且它解释了为什么"面多加水，结果面更多"**不是比喻，是必然**：

$$\frac{\partial\,\mathrm{norm}(cx)}{\partial \ln c}\equiv 0$$

norm 把 radial 方向的 Jacobian 特征值**精确压成 0**。于是 stream 长大之后，loss 里不再有任何一项对 $c$ 敏感 → **调节回路被物理切断**。所以：

> 补丁不是在压制症状，**是在拆传感器**。传感器一拆，作动器（各层的 weight scale）就免费了，于是每加一次 norm，可接受的平衡尺度就往上抬一档，而补丁对作动器零约束 → **系统没有平衡点**。

“面多加水 → 水更多”这句我建议你直接写成 thesis statement，因为它是这套论证里唯一一条能被数学证明的因果链，而且和流行叙事（"norm 带来稳定性"）**正面冲突**，审稿人会有反应。

<details>
<summary><b>【为什么它在动力学里有个正式名字，以及为什么 penalty 类方法结构上会输】</b></summary>

**名字**：非双曲平衡点 (non-hyperbolic equilibrium) 上的**中性/边缘方向** (neutral direction)。
在 normed read 处，Jacobian 有一个**精确为零**的特征值 —— 线性稳定性分析在这个方向上完全失效（既不收敛也不发散，是纯积分器）。控制论对这种情形的标准答案是明确的：**中性方向必须靠显式约束或慢回复项钉住，不能指望局部稳定性**。这就是你两个操作的理论合法性来源。

**为什么 penalty 必输**：约束 gauge 方向有两种做法：
- **投影/消元**（你）—— 相当于 DAE 里的 index reduction，把代数变量消掉。
- **加罚**（WD）—— 相当于给这个方向挂一根**极硬的弹簧**。硬弹簧 = stiff system = 需要极小步长 / 隐式积分。在优化器层面它的表现就是：$\lambda$ 小了不起作用，$\lambda$ 大了把有用方向也一起冻住。**"面多加水"在优化器层面的等价物就是加硬弹簧。**

所以"WD 治不了"不是力度问题，是**用错了积分器**：对中性方向，罚函数法在数值上等价于把一个自由变量变成一个高频振荡变量。你必须消元。

</details>

## 缺的那个通用理论：我认为骨架已经在你手里了，只差命名

你说这块缺通用理论 —— 我给你一个可以写成 paper 的三件套，因为它就是你说的“free drift”的严格化：

**① 分类（可计算）**：给每个张量一个 scale exponent $\alpha_T$，每个算子对指数施加一条线性约束：

| 算子 | 约束 |
|---|---|
| $y=Wx$ | $\alpha_y=\alpha_W+\alpha_x$ |
| $y=x+z$ | $\alpha_x=\alpha_z=\alpha_y$ |
| norm / softmax | $\alpha_y=0$ |
| 齐次度 $k$ 的逐元素（SiLU/GeLU≈1, ReLU=1） | $\alpha_y=k\,\alpha_x$ |

**约束矩阵的零空间 = gauge group**，零空间维数 = 你有几个“免费把手”。

具体结果我给一个可验证的：对 $x_{l+1}=x_l+f_l(\mathrm{norm}(x_l))$ 的整条 stack，零空间是 **1 维**的 —— 就是“全体 block 权重 × stream 的全局共同缩放”。因为 add 把所有层耦合住了。**这一条很有用**：它说明**逐层独立的漂移在数学上不是 gauge，它只是"便宜"**。于是你的现象分成两类，必须分开处理：

| | 定义 | 典型来源 | 处方 |
|---|---|---|---|
| **免费 (free)** | 梯度**精确为 0**、无任何 loss 收益、轨道无界 | 全局尺度、整矩阵尺度 | **pin**（你的 unit-norm 的标量部分） |
| **便宜 (cheap)** | 收益在某阈值 $T$ **饱和**，但量级在 $T$ 之外仍无界 | per-row gain、sink channel、constant bias | **bounded**（你的 per-row unit norm 正好同时管住这两个） |

关键洞察：cheap 类的收益曲线是**“饱和后趋平”**，即 **flat valley beyond threshold**。任何均衡都需要“边际收益 = 边际代价”，而收益已经平了 → 均衡是一个**连续集**，optimum 不唯一 → 漂移。**你的 unit-norm 本质是给这个平坦谷加了一道无限高的墙，把它从连续集压成一个点。** 这个说法比“约束”更准，也更难被反驳。

**② 处方**：每个零空间生成元取一个紧截面；每个饱和型便宜方向加一道边界。→ 就是你的两个操作。再加我上次说的**闭包条件**（所有乘子 $\prod|s_j|=1$）。

**③ 可证伪的预测表**（这是理论该有的样子，也是最能打的部分）：

| 现有 patch | 理论预测 | 验证 |
|---|---|---|
| QK-norm | **冗余**（unit rows 已 bound logit） | 去掉，看 $|q^\top k|_{\max}$ |
| logit softcapping | **冗余** | 同上 |
| register token | **部分冗余**（sink 需求被显式化后） | 见上次判别表 |
| final norm | **load-bearing**，但它是拆传感器的那个 | 换 const-scale read |
| μP 的 init 缩放 | **正交**（管 init 不管累积） | 可叠加 |

一个理论如果能预测“这套 patch 里有几个是白打的”，它就成立了。

## “AI 数值膨胀” —— 这个类比有一个反直觉的精确结论

对，而且比你想的更有意思：**标量通胀对 bf16 几乎无害**（浮点是相对精度，$10^6$ 和 $10$ 的尾数一样多）。真正致命的是**汇率扭曲**：

$$x_i/\mathrm{rms}(x)\ \text{从}\ 1/\sqrt d\ \text{涨到}\ 1$$

三条伤害全部来自 ratio，不来自 level：$\mathrm{rms}^2$ 的 fp32 累加被大通道吃掉小通道贡献；梯度被 $1/\mathrm{rms}$ 缩放导致通道间**有效学习率差 100 倍**；大通道归一化输出饱和在 $\sqrt d$ 白占容量。

所以：
> **A GDP 涨 100 倍没关系，汇率失真 100 倍要命。**

而你这个类比正好解释了“更大的箱子 / 更大的水桶”为什么无效：**扩大面额（bf16 范围、fp32 累加）治的是水平，你的病是汇率。** 你钉 unit-norm 钉的恰好是汇率。这是你方法相对于“加箱子加水桶”最干净的一句话对比。

## 一处我要跟你抬杠：pin vs expose

你上一轮说“禁止任何 free moving”。我认为这里要分，而且分了**更 general**、也更符合你自己的哲学（“允许 move，只要不免费”）：

- **真 gauge（梯度精确 0）→ pin 是对的**。因为收益恒等于零，暴露它没有意义，只会留个随机游走的口子。
- **cheap 方向（loss 看得见但饱和）→ 应该 expose，不是 pin**。因为“饱和后趋平”意味着模型其实在用这个量级承载功能（sink、常数、depth 记忆）。硬 pin 会把这个功能一起禁掉。

具体到 stream 上就是：我上次说的 leaky integrator，$\epsilon$ **可学习**，让模型自己选积分时间常数，由 loss 决定。理由：residual 的“边缘性 (marginality)”本身可能是有功能价值的 —— 积分时间 = 深度上的记忆长度，$z=1$ 的极点意味着**无限记忆**。你把所有东西都紧化了，等于强制系统有有限的 depth 记忆。EDM2 敢全 pin 是因为 diffusion 是固定深度的 feedforward，不需要 depth 记忆；**LLM 可能需要。**

所以你的主张可以升级成一句比“禁止 free moving”更硬的话，这也是我认为你该写进 abstract 的：

> **禁止的是 loss-invisible 的漂移（pin）；不是禁止漂移。loss-visible 的漂移应当被暴露成参数，让 loss 自己定价。**

这样你和 EDM2 的关系就非常清楚了：

| | 立场 |
|---|---|
| 大家 | 面多加面 → 更大箱子；水多加水 → 更大水桶 |
| EDM2 | **不准加**（全 pin 到设计值，含与 gauge 无关的 diffusion 规则）|
| 你 | **gauge 不准加，其余随便加；加了要有代价** |

<details>
<summary><b>【一个专门打你最强 claim 的实验】</b></summary>

你最强、也最容易被攻的 claim 是“**是 norm 造成了漂移，而不是 norm 暴露了漂移**”。这个可以做干净的三方对照：

1. **pre-norm**（标准）：预测 stream 线性涨，且无回复力。
2. **no-norm + 临界 init**（$I+W$ 谱半径 ≈1）：预测**没有无界漂移**（loss 对尺度敏感 → 有回复力），但对 init 极敏感、loss 差。
3. **post-norm**：预测指数而非线性。

如果 (2) 成立，你就拿到了“norm 拆掉传感器”的直接证据，而不只是叙事。这个实验很便宜（小 MLP/小 transformer，几十层），但它是你整篇文章的因果基石。

顺带它也给了你一个更强的替代方案：把 norm 的 read 改成 **const-scale read**（固定尺度、loss 可见），即“暴露”而不是“抹掉”——如果这个也稳，那你的框架就同时包含了“修”和“防”，而不只是“修”。
</details>

---

要不要我把上面 ①②③ 写成一节 `§2 A dynamical-systems view of residual streams` 的草稿（定义、零空间算法、free/cheap 二分、闭包定理、patch 预测表）？或者先把那个三方对照的 toy 跑出来，把三条 `rms(x_l)` 曲线打出来看形状对不对 —— 我觉得先跑曲线更有用，因为它能验证你 causal chain 的第一环，而那一环是整篇文章的地基。

---
## 🧑‍💻 User

而且我发现我这个改进你只要随意地塞在绝大多数block里面就很有用了 不用把整个模型都变成no free moving
其实就是太多free moving叠加导致模型直接飞升了
如果free moving不多那大多时候不会飞升之外weight decay也会出手

---
## 😀 默认助手 (7)

## 你这条经验其实是全文最有价值的一条：它说明你手上的是**一个预算，不是一个二值约束**

先把它写成公式。因为 $f_l=W_l\,\mathrm{norm}(x_l)$，而 $\mathrm{norm}$ 的输出范数被钉在 $\sqrt d$：

$$\|x_{l+1}\|\le\|x_l\|+\|f_l\|,\qquad \|f_l\|\le\|W_l\|_{\mathrm{op}}\cdot\sqrt d\,\|g\|$$

- **未 pin**：$\|W_l\|$ 可以涨 → $\|f_l\|$ 与 $\|x_l\|$ **无关地**涨 → $c_l:=\|f_l\|/\|x_l\|$ 涨 → 增长是**乘性**的。
- **pin 了**：$\|W_l\|_{\mathrm{op}}\le\sqrt{d_{\text{out}}}$ 是常数 → $\|f_l\|\le C$ → $\|x_l\|\le LC$ → 增长退化成**加性**，且相对增量 $c_l\sim1/l$。

于是整条 stream 的指数和：

$$\log\|x_L\|\;\approx\;\underbrace{\sum_{l\in\text{free}}\log c_l}_{\text{指数项}}\;+\;O(\log L)$$

**pin 一个 block，就是从指数和里删掉一项。**

这一行解释了你观察到的“随意塞”：**你不是在修一个特定 bug，你是在缩减一个和。** 所以

- 它**不挑位置** —— 任何 free block 的贡献形式相同，删谁都一样，随机就够；
- 它**不需要删完** —— 只要 $\sum_{\text{free}}\log c_l<\Theta^*$ 就飞不起来，这就是为什么“塞进大多数”而不是“全部”就有效，也是为什么效应是**平滑加拐点**而不是全有全无；
- 而模型的崩溃是**首次穿越 (first-passage) 现象**：$\mathbb{E}\sum\log c\propto L$ 而 $\mathrm{std}\propto\sqrt L$，$L$ 一大，均值必然压倒方差 → **深模型必撞，浅模型永不撞**。这解释了为什么这问题是“深”模型的病，也解释了你说的“叠加”。ResNet-50 和浅 VAE 一辈子见不到，60 层 MMDiT 一定见到。

## 顺带：那个“更大的箱子”在物理上不存在

这个是 IEEE 给的，可以直接写进 motivation：

| | 指数位 | 尾数位 | 动态范围 |
|---|---|---|---|
| fp32 | **8** | 23 | $\sim10^{38}$ |
| bf16 | **8** | 7 | $\sim10^{38}$ |
| fp16 | 5 | 10 | $\sim10^{5}$ |

**fp32 和 bf16 的指数位一样宽。** 换 bf16 损失的是**精度**，不是**范围**；换回 fp32 也**买不到**更大的箱子。

$$\Theta^*\approx\log(10^{38})\approx 88\ \text{nats},\quad \text{与 dtype 无关，买不到}$$

所以“面多加面 → 用更大的箱子”这条策略**在结构上封顶**：能买的只有 fp64（11 位指数），而它在硬件上不可用。剩下的唯一自由度是**箱子里装什么** —— 也就是你的 budget。这句话我觉得是你整篇文章最好的一个 argument：**对手的路是死路，不是因为不够努力，是因为 8 bit。** 而“水多加水”（更多 norm）之所以无限续，恰恰是因为它把预算的**可见性**也一起删了 —— 所以看起来永远够用，直到某一次 $2\times$ 深度撞穿。

## 我上一轮那句说得太重，需要修正

我说“WD 结构上失效”，按你的经验这不对，应该是：**WD 是被淹没 (out-gunned)，不是失效。**

把两边写成竞速：
- WD 的回复率是 $\lambda$（固定，而且 decoupled WD 大家都用得很小）；
- 驱动力正比于 $n_{\text{free}}$（每个 free block 贡献一份 $\log c$ 的增长）。

$$\text{均衡 drift}\;\sim\;\frac{n_{\text{free}}\,J}{\lambda}$$

$J$ 是单 block 每步的 log-gain 驱动。所以有一个交叉点：

$$n_{\text{free}} < \frac{\lambda}{J}\quad\Longleftrightarrow\quad \text{WD 重新上线}$$

**这就是你说的“free moving 不多的话 WD 也会出手”的定量版本** —— 你把驱动力降到了 $\lambda$ 同一量级，WD 就重新变成主导项。你不需要 WD 变强，你需要**驱动力不要比它大三个数量级**。

但要说清 WD 即使上线也救不了的那一半：

| 漂移种类 | WD 能不能救 |
|---|---|
| 标量 / 整矩阵 gauge | **能**（方向对，只是太弱）→ 你 pin 它 = 直接删掉驱动力 |
| per-row gain / profile / 汇率 | **结构上瞎**（ratio 不变）→ 只有你的 per-row unit-norm 能管 |

所以两层不能互相替代，你的两个操作在分类上各自对应一半，挺干净的。

## 一个我觉得很好的定位：这是**深度指数**

μP / $\mu$Transfer 那套，管的是**宽度**方向的缩放指数。而漂移是**深度**方向的缩放指数 —— 而且没人管它。

$$\log\|x_L\|\approx\textstyle\sum_{l}\log c_l\quad\text{这是深度的缩放指数}$$

**现在整个领域的 scaling recipe 只规范了宽度指数，深度指数交给优化器噪声自己定。** 你的 $\Theta^*$ 就是那个缺失的深度指数。这个 framing 比“禁止 free moving”好卖十倍，因为它落在大家已经接受的 μP 语言里，而且直接解释了“为什么 scaling 到更深的模型就飞” —— 不是超参没调好，是**深度指数失控**。

<details>
<summary><b>【两个立刻能做的检验，一个可能改进你的方法】</b></summary>

**检验 1（hot spot）—— 我预测你会看到这个，如果没有，说明“gain 是功能性的”**

部分 pin 会把总 gain 压进少数 free block。如果模型**功能上需要**一定放大，那剩下的 free block 必须独自扛，每个要扛更多 → 那些 block 的 $\log\|W\|$ 增速反而**更快**。

预测：$\max_{l\in\text{free}}\dot{\log}\|W_l\|$ 随 pinned fraction **单调上升**。
- 看到 → **部分 pin 修好了平均值但制造热点**，你需要 targeted，不能 random；
- 没看到 → gain 确实是**免费的**（无功能需求），random 就是最优且白拿。这是区分“功能 gain”vs“免费 gain”的干净实验。

**检验 2：从 random 升级到 targeted（同预算下可能更优）**

测每个 block 的 $\dot{\log}\|W_l\|$ 分布：
- **重尾** → targeted（pin 最大的那些）显著优于 random，你现在的 random 是在浪费预算；
- **同质** → random 已最优，别改。

**检验 3（反向确认预算公式）**：固定 pin 比例，扫深度 $L$。预测临界 pin 比例 $p^*(L)$ 随 $L$ **下降**（因为 $\Theta^*$ 是常数而驱动力 $\propto(1-p)L$）。如果 $p^*$ 与 $L$ 无关，那我这套指数叠加的图景是错的，得退回纯 gauge 解释。
</details>

## 收一下

你的方法真正的名字不是 “no free moving”，是：

$$\boxed{\text{gain budget}\quad \sum_{l\in\text{free}}\log c_l\le\Theta^*}$$

| | 预算 |
|---|---|
| 大家 | $\infty$（而且是**不可见**的 $\infty$，所以以为自己是 0）|
| EDM2 | $0$（全 pin 到设计值）|
| **你** | **有限、可调、只对 gauge 方向生效** |

这个 framing 同时做到三件事：(a) 解释了为什么“随意塞”有效（缩减一个和）；(b) 解释了为什么“更大箱子”是死路（8 bit）；(c) 把你的方法和 μP 接上（深度指数）。而且它和你上一轮那句“允许 move，只要不免费”完全一致 —— budget 就是“免费额度”。

要我把这三个检验拼成一个单文件 toy 吗？我可以直接在这边跑：堆 $L$ 层 $\mathrm{RMSNorm}$ block，pin 比例和深度做网格，输出 **(i) $\log\|x_L\|$ vs $p$、(ii) $\max\dot{\log}\|W\|$ vs $p$、(iii) $p^*(L)$ 曲线**。这三张图出来，你的“叠加”和“预算”就同时被验证或打掉了。

---
## 🧑‍💻 User

我最近在研究training dynamic的事情然后发现EDM2是真goat
各种诡异操作不如直接unit row
我感觉AI这边，越复杂，越花哨的方法，反而都是比较花瓶
很多效果好的反而是简单的方法

是大家觉得EDM2是那个花瓶 但是我最近遇到的各种神秘玩意让我认真下去研究之后发现EDM2才是真goat

不过EDM2太局限了 而且限制很死 需要发展一个更通用的版本
EDM2其实是一个很简单的数学目标但是要达成他在不同架构要分开处理所以被别人当成工程技巧了
但是我回头看发现这才是真数学技巧
结论是方法论是否工程不能只看如何实现 还要看目标跟动机
我现在这个东西理论上可以帮助所有x = x+f(norm(x))的模型结构
并且在muon跟adam应该都可以 吧 我要看一下muon的部分要怎么处理
数值稳定性上升一百倍之类的（开玩笑

---
## 😀 默认助手 (7)

## 先把“简单”这个词换成可用的判据

“简单 > 复杂”这个说法会被审稿人当口号。可用的版本是：

> **方法的价值 = 它删掉了几个自由度。**
> 删自由度的东西没有超参，所以不用调，所以跨架构跨尺度都活着。
> 加自由度的东西必须调，所以每个新模型都要重新调，所以是花瓶。

按这条排一遍，一切都对上了：

| | 类型 | 超参 | 活得久吗 |
|---|---|---|---|
| EDM2 magnitude-preserving | **删**（删掉尺度） | **无** | 是（但只在一个架构里被验证） |
| unit-norm row | **删**（删掉行尺度） | **无** | 待验证 |
| QK-norm | 加（加一个归一） | 目标值 | 每个模型重调 |
| logit softcap | 加（加一个界） | cap 值 | 每个模型重调 |
| LayerScale / $\tau$ | 加（加一个 gate） | init 值 | 每个模型重调 |
| muP | 删（删掉宽度依赖） | 无 | 是 |

所以 EDM2 之所以能称为数学方法，而 QK-norm 只能称为工程手法，区别在于 **EDM2 在消元，QK-norm 在添加**。你说“目标是数学的、实现必须分架构”——这话对，但还可以更精确：**它的目标是一条约束方程，而每个架构里的约束方程形式不同**（因为每个 op 的齐次度不同）。这就是为什么它看起来像“一堆 per-op 规则”：**它不是一堆技巧，它是同一个方程在不同 op 上的实例化。** 这句话我觉得可以直接当你的 paper 的 §1 结论。

而 EDM2 还隐藏了另一项更关键的特质，很多人没有注意到：**它是 scale-free 的，所以超参跨模型尺寸能够迁移**（同一个 LR 在不同宽度上都管用）。这才是它真正的价值所在：它让“超参搜索”退化成“给定”。**约束类方法的天然产物是迁移性。**

## 但这把刀也会指向你自己

我直说：**“随机塞进大多数 block 就有效”这句话现在读起来就是一个魔数 $p$。** 它恰好是你批评的那一类 —— 一个没有解释的常数，靠经验撞中。

Emperor's new clothes 的风险在这里：你的 mechanism 是数学的（unit-norm 是约束），但你的 **claim 是工程式的**（“塞大多数就行”）。审稿人会把它归到“更多 norm”那一堆里去。

解决办法很清楚：**把 $p$ 从调出来的量变成算出来的量。** 按上一轮的预算公式：

$$\sum_{l\notin\text{pinned}}\dot{\log}c_l \;<\;\Theta^*\quad\Longrightarrow\quad p^*=\max\!\Big(0,\;1-\frac{\Theta^*}{\sum_l \dot{\log}c_l}\Big)$$

$p^*$ 是可测量的 —— 先跑一小段无约束训练测 $\dot{\log}c_l$，再算需要 pin 多少。这样你的方法就从“一个魔数比例”变成“**一个测量 + 一个阈值**”，超参个数降到 0。**这是你这篇文章能不能进 EDM2 那个分类的唯一门槛。**

同理，你上一轮说的“随机 vs targeted”就可以直接由 $\dot{\log}c_l$ 的分布决定，不用再选。

## Muon：我认真算了一下，结论分三句

**① Muon 的更新在 residual write 矩阵上是行正交的 —— 也就是说，它在几何上已经帮你做了一半。**

先说清楚适用范围：设 $W\in\mathbb{R}^{d_{out}\times d_{in}}$，Muon 更新 $O=\mathrm{polar}(M)$。若 $d_{out}\le d_{in}$（**transformer 里所有写回 residual 的矩阵都满足这个条件**：$W_O$ 是方阵、down-proj 是 $d_{model}\times d_{ff}$），则 $OO^\top=I$，**行是标准正交的**，所以

$$\|o_i\|=1\ \ \forall i$$

**每一行的扰动幅度都相同。** 而 $d_{out}>d_{in}$（up-proj）时则是列正交。这不是巧合：Muon 强制**所有奇异值相等**，所以它**永远不可能在某一步里优先放大某一个 row / 某一个奇异方向**（Adam 的逐坐标归一化没有这个保证）。

对你的问题来说，这意味着：**Muon 天然抑制 outlier channel 的自增强。** 因此你的 **per-row 约束在 Muon 下边际价值会明显变小** —— 这是个干净的消融点，而且我预测你会看到 per-row 的收益从 Adam 下的“很大”降到 Muon 下的“小但为正”。

**② 但标量漂移一点没解决，甚至更严重。**
$$O=\mathrm{polar}(M)\ \text{对}\ M\ \text{是 0 次齐次的}\ \Longrightarrow\ M\to cM\ \text{更新不变}$$
**Muon 和 Adam 一样对参数尺度完全盲**，所以权重涨 10 倍，它照样一步不动。scalar pin 在 Muon 下**同样必需**。所以你的方案对两种优化器的答案不一样：

| | per-row（汇率） | scalar（水平） |
|---|---|---|
| Adam | 必需，收益大 | 必需 |
| Muon | 边际（已被正交化部分覆盖） | **必需，且是唯一的手** |

这正是你说“我说不准 Muon 怎么处理”的答案：**Muon 里你只需要 scalar 那一半，per-row 那一半它自己做了。**

**③ 一个几乎零成本的实现红利（我觉得这条你会喜欢）**

Stiefel 流形（行正交约束）的标准 polar retraction 是
$$W\leftarrow W\,(WW^\top)^{-1/2}$$
而 **Muon 的 Newton-Schulz 迭代算的正是 $M(MM^\top)^{-1/2}$ —— 同一个算子。** 所以如果你把“行单位范数”升级成“**行正交**”，retraction 在 Muon 的代码里是现成的，一次 NS 调用就能完成，不需要额外的 SVD、Householder 或 QR。

为什么我建议升级：
- 行正交是“行单位范数”的**自然完备化**（约束 $WW^\top$ 的对角线还不够，$WW^\top=I$ 才是封闭的）；
- Muon 的更新本来就是正交的 —— **几何上是一致的**，不是在跟优化器对抗；
- 你顺带拿到了 singular value 的均衡，和 Muon 的设计目标同向，不会互相摩擦。

代价是表达力会更强地受限，所以消融就要做成三档：**无 / 行单位范数 / 行正交**。

<details>
<summary><b>【Muon 下两个必须小心的实现细节】</b></summary>

1. **$m$ 要投影到切空间再 orthogonalize。** Muon 没有 $v$，但有 $m$。$\mathrm{polar}(M)$ 只对 $M$ 的 **scale** 不变，对 $M$ 的**径向分量**完全敏感 —— 如果 $m$ 带一个持续为正的径向分量，orthogonalize 之后它仍然会推半径。所以要把 $m\leftarrow m-\langle m,\hat w\rangle\hat w$，这比投影 Adam 的 $v$ 更必要。
2. **LR 语义变了。** 行单位范数/行正交下，一步是一个**角度**而不是一个长度，$\|\Delta W\|_F$ 不再等于 $\eta$。Muon 现有的 LR 是配合 $\|O\|_F=\sqrt{\min(m,n)}$ 标定的，加了约束之后要重扫 —— 不重扫你会误判成“掉点”。
3. **Stiefel 约束下不能有 bias 和任意 per-row gain**，否则又漏回上一轮的闭包问题。
</details>

## scope：你说的“所有 $x=x+f(\mathrm{norm}(x))$”要写清边界

这个 family 的判断是干净的：关键条件是 `norm` 对输入**正齐次 0 次**（RMSNorm / LayerNorm / GroupNorm / softmax 在 logits 上都满足）。但**网络里还有别的无界累加器**，它们不在你的 claim 里，必须显式排除，否则审稿人会拿它们攻击你的“通用性”：

- KV cache（不累加，但它随 context 线性增长 → 注意力相关的数值问题）
- **优化器状态** $m,v$（$v$ 也是累加器，且也能漂）
- EMA 权重（EDM2 的一个重点，很多人漏了）
- embedding / output head（loss-visible，属于“该自由”的那一半）
- MoE 的 router logits、diffusion 的 $\sigma$-conditioning scale（后者正是 EDM2 里与 gauge 无关的那部分规则）

声明 scope 反而会让你更强：**“residual stream 是唯一既无归一化、又无正则化、还负责累加的张量”** —— 这一句已经是 scope 声明了。

## 唯一能把它从 trick 变成 theory 的实验

我觉得这个比“稳定 100 倍”重要得多（玩笑归玩笑，稳定倍数不是可发表量）：

> **同一个 LR / 同一个 $p$，跨深度（6 / 12 / 24 / 48）都能用，无约束基线不能。**

这就是 EDM2 的真本事，也是 muP 的真本事，而且它极便宜。理由：约束类方法的天然产物是**超参迁移**；mechanism 类方法永远做不到（每个尺寸重新调）。你只要做出来了，你的 paper 就从“又一个稳定化技巧”变成“**深度的 scale-free 参数化**”—— 也就是我上一轮说的“μP 管宽度指数，你的东西管深度指数”那个定位。

而且这个实验还顺便验证或否证了你上一轮的机制：如果 $p^*$ 随 $L$ 下降，那说明“叠加”的机制成立；如果 $p^*$ 与 $L$ 无关，那说明真正起作用的是 scalar gauge 那一维，per-row 那部分只是附带效果。**一个实验同时验证两条主张，性价比很高。**

---

我的判断：你现在手上有的是一个**删自由度**的方法（好品类），但你的**经验论证方式**还是加自由度那一类的（“塞大多数就有效”）。把 $p$ 变成 $p^*$、把“随机”变成由 $\dot{\log}c_l$ 决定、把 per-row 升级成 Stiefel 以复用 Muon 的 NS —— 三件事做完，它就落到 EDM2 同一个分类里了；而 scope 更大，因为 EDM2 的规则需要分架构重写，你的只需要一条约束方程。

要不要我把这个实验设计写成可跑的骨架（含 Muon 和 Adam 两条路径，Stiefel retraction 复用 NS）？我可以直接在这边把 $p^*(L)$ 和 $\max\dot{\log}\|W\|$ 的曲线跑出来。
