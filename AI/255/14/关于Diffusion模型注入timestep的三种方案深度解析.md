# 关于Diffusion模型注入timestep的三种方案深度解析

## 🧑‍💻 User

说起来，conv based diffusion不是一般要注入timestep嘛，我可不可以把time变成一个input channel：t * torch.ones(H, W)
纯卷积就加在通道维度 或者拼上去
纯dit就直接作为一个token拼序列上

偷懒就不加 感觉区别不大
听说有些模型好像直接不给 t 条件，也不是不能训

其实本质就是
给定一个xt预测t
对于现在的模型来说是naive task
所以你不加真没差
模型可以通过输入推测出来当前时间步 问题不大
miniT2I就把这玩意砍掉了 uvit是最早砍的了（好像是2023y）

---
## 🧑‍💻 User

说起来，conv based diffusion不是一般要注入timestep嘛，我可不可以把time变成一个input channel：t * torch.ones(H, W)
纯卷积就加在通道维度 或者拼上去
纯dit就直接作为一个token拼序列上
偷懒就不加 感觉区别不大
听说有些模型好像直接不给 t 条件，也不是不能训

其实本质就是
给定一个xt预测t
对于现在的模型来说是naive task
所以你不加真没差
模型可以通过输入推测出来当前时间步 问题不大
miniT2I就把这玩意砍掉了 uvit是最早砍的了（2023）

---
## 😀 默认助手 (7)

可以，你这两个方案都对，而且第三点（不加 t）也确实已经有人这么干过。下面分几个层面说，顺便回应一下你几个可能有些偏差的记忆点。

## 1. `t * ones(H,W)` 拼到通道上

这在数学上退化成**输入层的一个 rank-1 FiLM**。第一个 conv 对这个常数通道的权重 $w_c$ 作用上去就是常数偏置：

$$h^{(1)} = W * [x_t;\, c\cdot\mathbf{1}] + b = W_x * x_t + b + w_c\,c$$

也就是说，如果只做到这一步，它`= 一个随 t 变的 per-channel bias`。而标准做法的 FiLM、adaLN 是每层 $\gamma(t)\odot h + \beta(t)$，是全秩调制。因此**通道拼接是这个做法的退化特例**，能用，但信息注入的“带宽”低很多。

两个实际问题：

- **归一化要处理**。直接喂原始 `t`（0~1000）会污染第一层 conv 的初始化量级，换成 $t/T$ 或者 log-SNR $\lambda_t$ 比较好。
- **有 GroupNorm/BatchNorm 的网络里，纯常数偏置会被 GN 削掉一部分**（GN 是按 (C/G, H, W) 减均值的，空间上恒定的量几乎正好落在被减掉的方向上）。真正让它起作用的，是后续非线性（比如 ReLU）把加性偏置转成乘性调制这一步。

<details>
<summary>展开：为什么“加了非线性就活下来了”</summary>

设某层特征 $h$ 在空间上有方差，注入 $h + b(t)$ 再经过 ReLU：

$$\mathrm{ReLU}(h + b(t)) = \mathrm{ReLU}(h)\cdot \mathbb{1}[h > -b(t)]$$

也就是 $b(t)$ 变成了一个**依赖于 t 的门控**，这才是真正有用的信号。所以通道拼接在 conv net 里能 work，但它是“绕了一圈”的 FiLM，而不是干净的 FiLM。

</details>

## 2. DiT 里当 token 拼

这条路是成立的，而且 **U-ViT 就是这么干的**（time token 和 ViT 的 class token 同样处理）。代价是多一个 token 的 attention 计算，收益是架构干净、不用 adaLN。DiT 本身选择了 adaLN-zero，本质上也是同一件事的不同实现，只是 adaLN 更便宜，也更难被初始化破坏。

顺带纠正一下：**U-ViT 我印象里是保留 t 的**（time token），不是最早砍掉 t 的那批。你记的那个“2023 年最早砍 t”可能是某篇的消融实验，或者是别的模型——这点我不敢打包票，如果你手上有出处可以发我。

## 3. 不加 t：为什么可行，代价在哪

你的直觉大体正确，但“naive task”这个说法要加个前提。

**可辨识性取决于你的 schedule 有没有保留尺度信息。** 前向过程

$$x_t = \alpha_t x_0 + \sigma_t \varepsilon,\qquad \mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2$$

如果按 VE/EDM 那种把 $\sigma$ 直接放在数据尺度上的写法，$\|x_t\|/\sqrt d$ 几乎就是 $t$ 的充分统计量，模型随手就能估出来，砍掉 t 基本无损。但如果你用 VP 并且把输入归一化成 $x_t/\sqrt{\alpha_t^2+\sigma_t^2}$（这是 SD 系列的做法），**尺度信息被你自己除掉了**，模型只能靠内容统计去反推 $t$。这时候 $\mathbb{E}[t \mid x_t]$ 的后验不再尖锐，砍 t 会掉点，而且掉得主要在中间噪声段。

不加 t 的真实代价，按重要性排序：

1. **采样器和训练耦合死了**。加 t 的模型对换 schedule、换 offset noise、换 time shift 都是鲁棒的；不加 t 的模型，推理时的整体缩放约定必须和训练严格一致，否则“隐式推断的 t”就错了。这是最容易被忽略的一条。
2. **只靠内容推断 t 会引入后验平均偏差**。有些 $x_t$ 在视觉上确实像多个 $t$ 共享，模型只能输出各 $t$ 目标的加权平均，轻微模糊。
3. **CFG 和训练无关的 guidance 全都默认依赖 t**。不加 t 你就没法在推理时用 $t$ 做 scaling 或换 guidance 强度。
4. 高噪声段反而没差：这时 $x_t \approx \varepsilon$，学 $x_0$ 本来就没戏，t 的作用也小。

## 4. 我的建议

探索性实验、小模型、就想要个能跑的东西——砍掉完全没问题，别在这个上面纠结。

但如果是正经训，**加上 t 的边际成本基本为零**（一个 2 层 MLP + 每层 FiLM/adaLN-zero，参数量可以忽略），换来的是解耦 schedule、可用 CFG、少一类调试地狱。从“偷懒收益比”来看，这笔交易不划算——省下的代码量大概是 10 行，放弃的是后面所有的灵活性。

至于「conv 用通道拼接 vs. DiT 用 token」这个选择本身：真正该做的是**每层的 FiLM/adaLN**，通道拼接和 token 都是它的廉价近似。你既然已经在纠结这个，不如直接上 FiLM。
