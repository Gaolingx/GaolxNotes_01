# 模型损失不成nan

## 🧑‍💻 User

话说llm这种东西到底要什么情况才会出现loss变成nan，窝炼了四五十炉（预训练/sft 400m/1.5b/4b）一回都没炸过（用的llama3/deepseek v2代码），是运气好还是本来就不可能炸？

---
## 🤖 Assistant

先说结论：**主要是你运气好 + 你踩的这套配置本身就极其抗炸，但绝不是"本来就不可能炸"**——存在一批系统性的操作会稳定把 loss 打成 NaN，只是那 40~50 炉你恰好一个都没触发而已。

下面把 NaN 的来源、你为什么没遇到、以及什么情况下一定遇到拆开讲。

---

## 一、loss 变 NaN 的几大来源

### 1. 数值溢出（头号原因，也最"冤枉"）
- **fp16 溢出**：fp16 指数位只有 5 bit，最大值约 $65504$。梯度或激活一旦超过这个值就变成 $+\infty$，然后 $\infty - \infty = \text{NaN}$ 开始级联传染。
- **fp16 下溢**：小于约 $6\times10^{-8}$ 的数直接变 0（这个不产生 NaN，但会让训练变慢/没梯度）。
- 这就是为什么 fp16 训练必须配 **loss scaling**：scale 太大 → 梯度溢出成 inf → NaN；scale 太小 → 梯度全下溢成 0。很多 NaN 事故本质是 dynamic loss scaling 没调好。

### 2. 上溢/下溢之外的真实计算 NaN
| 来源 | 公式 | 后果 |
|------|------|------|
| softmax 整行被 mask 掉 | $\frac{e^{x_i-\max}}{\sum}$ 里 $\max=-\infty$，$e^{-inf-(-inf)}=e^{\text{NaN}}$ | attention 直接 NaN |
| $\log(0)$ | $-\log(p)$，$p=0$ | loss $=+\infty$ |
| 除以 0 | LayerNorm $\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$ 里 $\sigma=0$ 且无 $\epsilon$ | NaN |
| 激活爆炸 | GeLU/SiLU 输入过大（其实这些激活本身不炸，主要是上游线性层炸） | inf |

### 3. 优化/超参问题
- **LR 太大 / warmup 太短**：训练前期就发散，梯度范数指数增长到溢出。
- **没有梯度裁剪**：一次离群 batch（脏数据、超长序列）就能产生巨大梯度，直接冲爆。
- **deep 网络 + 不好初始化**：残差分支尺度失控。

### 4. 数据问题
- 数据里混进 NaN/inf（清洗不干净、加载损坏的文件、fp16 存的 token embedding 溢出）。
- 空序列 / 全 padding 的样本导致 attention 全 mask（对应上面第 2 条）。
- 极端长度样本让 position embedding / rope 外推出问题。

### 5. 结构/算子特有
- **MoE**：router logits 异常、专家坍缩、load-balancing loss 尖峰。
- **RLHF/PPO**：ratio $=\exp(\log p - \log p_{old})$，log-prob 一旦为 $-\infty$ 或 nan 立刻炸。
- **自定义 CUDA kernel**：FlashAttention 早期版本、fused optimizer、fp8 kernel 的边界 bug 是重灾区。

---

## 二、为什么你 40~50 炉一次没炸？（拆解你的配置）

你的组合实际上把"最容易炸的那些雷"全绕开了：

<details>
<summary><b>① 大概率跑的是 bf16 —— 这一条就消掉了 80% 的 NaN</b></summary>

Llama3 / DeepSeek V2 这套代码默认用 **bf16**。bf16 有 **8 位指数**（和 fp32 一样），动态范围约 $\pm 3.4\times10^{38}$。

- 它**几乎不会上溢**成 inf，所以 fp16 那套"溢出→inf→NaN"的经典级联基本不存在。
- bf16 **不需要 loss scaling**，也就没有 scale 调错导致的 NaN。
- 代价只是尾数位少（7 bit），精度略糙，但换来的是数值稳定。

如果你把这些实验换成 fp16 且关掉 loss scaling，我保证能复现出一堆 NaN。
</details>

<details>
<summary><b>② 代码是"久经沙场"的生产级实现</b></summary>

Llama3、DeepSeek V2 的官方/社区实现早就把角落 case 处理干净了：
- attention mask 保证了 causal 对角永远没被 mask（不会出现整行全 mask），padding 也处理得当；
- LayerNorm 带 $\epsilon$；
- softmax 内部做了减 max 的数值稳定写法（并有 inf 防护）；
- 初始化、rope、weight tying 都是验证过的。
</details>

<details>
<summary><b>③ 训练脚本标配了防护手段</b></summary>

绝大多数预训练/SFT recipe 默认就带：
- **gradient clipping**（通常 clip 到 1.0，`total_norm` 裁剪）—— 这一条就挡掉了绝大多数"离群 batch 冲爆"的情况；
- **LR warmup**；
- 合理的 LR（预训练 1e-4~3e-4 量级）、Adam 的 $\epsilon=10^{-8}$ 防止除零。
</details>

<details>
<summary><b>④ 模型规模/长度还够不上最危险的区间</b></summary>

400M / 1.5B / 4B 属于中小规模，加上正常的序列长度，梯度爆炸的累积效应远没有 70B+、8k/32k 长上下文那么凶。MoE 用的也是 DeepSeek V2 那种带 bias 的、"auxiliary-loss-free" 的相对温和方案，不像早期 MoE 那样动不动 router 炸。
</details>

---

## 三、什么情况下你**一定会**炸（想复现就去踩这些）

- 切到 **fp16 且关掉/调错 loss scaling**。
- 把 **LR 乘 10**、或把 **warmup 设成 0**。
- **关掉梯度裁剪**。
- 数据集里**注入一个 NaN / 一个全 padding 样本**。
- 写一个 **MoE router 或 attention mask 的 bug**，制造整行全 mask。
- 换一个**没测过的 fused kernel / fp8 路径**。
- 跑 **PPO/RLHF**，且 log-prob 出现 $-\infty$。
- 手动实现自定义 loss 时算了 $\log(0)$ 或 $0/0$。

换句话说：**NaN 不是随机降临的，是特定配置下的必然结果**。你没炸，是因为你的"配方"里恰好没有触发条件。

---

## 四、实用建议（万一哪天真炸了）

1. **优先 bf16**：能躲掉绝大多数溢出类 NaN；如果必须 fp16，务必确认 loss scaling 正常工作。
2. **开梯度裁剪 + 监控 `grad_norm`**：clip 之前先打日志，炸之前通常能看到 norm 先起飞。
3. **查第一个 NaN 的位置**，而不是只看最终 loss：
 - 用 `torch.autograd.set_detect_anomaly(True)` 定位；
 - 打印每一步的 loss、grad_norm、以及各层激活的 absmax；
 - 看是前向就 NaN（多半是数据/attention mask），还是反向才 NaN（多半是梯度溢出）。
4. **复现最小样本**：能定位到某个 batch / 某个 token，基本就锁定了数据或 mask 问题。
5. **加防护**：
 - 梯度裁剪（clamp grad_norm）；
 - attention 里对全 mask 行加安全值；
 - 检查数据清洗 pipeline 的 NaN 过滤；
 - 用 `torch.nan_to_num` 做兜底（治标，但救命）。
6. **别用 `div` 之前忘了加 $\epsilon$**，softmax/`log_softmax` 尽量用框架的稳定实现。

---

**一句话总结**：你（大概率）在用 bf16 + 生产级代码 + 带裁剪和 warmup 的配方，这套组合让 NaN 变得**很罕见但并非不可能**。它没炸不是"数学上不会炸"，而是你还没触发那些系统性条件——想验证的话，切 fp16 关裁剪、LR 放大 10 倍，基本立刻就能见到 NaN。

---
## 🧑‍💻 User

可能是咱没玩过fp16 true training（bf16掩盖了数值爆炸），然后这种某一层出现的outlier一般都被rmsnorm压住了吗，什么情况下才会导致fwd激活>65504？

---
## 🤖 Assistant

你的直觉基本对，但中间有个很关键的细节容易想反：**RMSNorm 压住的是"尺度"，压不住"残差流"和"norm 之后的放大"**。而且 bf16 掩盖的只是"溢出成 inf/NaN 的后果"，outlier 本身在 bf16 里照样存在、照样在涨，只是它不会变成 inf 而已。下面拆开说。

---

## 一、先纠正一个点：bf16 没消灭 outlier，只消灭了"溢出"

- 你 bf16 训练时看到的 loss 平稳，**不代表没有数值爆炸的瞬间**。高 LR 下所谓 loss spike，本质就是某些激活/梯度范数先冲高、再回落——bf16 因为动态范围 $\pm 3.4\times10^{38}$，这个过程不产生 inf，所以只是"抖一下"就恢复了。
- 换到 fp16（上限 $65504$），**同一个 outlier 只要数值超过 $6.5\times10^4$ 就变 inf → 级联 NaN**。
- 所以更准确的说法是：**bf16 把"数值爆炸"从"必炸"降级成了"抖一下"**，而不是让它消失。

---

## 二、RMSNorm 到底压住了什么？

### 1. 它其实有一个硬上界（这点很关键）

RMSNorm：
$$
\hat x_i = \frac{x_i}{\mathrm{RMS}(x)},\quad \mathrm{RMS}(x)=\sqrt{\tfrac1n\textstyle\sum_j x_j^2+\epsilon}
$$

因为 $\mathrm{RMS}(x)\ge \dfrac{|x_i|}{\sqrt n}$，所以

$$
|\hat x_i| \le \sqrt n,\qquad |y_i| = |\gamma_i \hat x_i| \le \gamma_i\sqrt n
$$

也就是说，**RMSNorm 的输出每个元素被死死夹在 $\sqrt n\cdot\gamma$ 以内**（$n=4096$ 时约 $64\gamma$）。它是一个 **scale-invariant（尺度不变）** 的算子：$\mathrm{RMSNorm}(cx)=\mathrm{RMSNorm}(x)$。

后果有两面：
- 好处：**不管残差流涨到多大，每个 sublayer 拿到的输入永远是 ~unit RMS**。这就是 pre-norm + RMSNorm 结构稳的根本原因——上一层涨多少，下一层"看不见"。
- 坏处：**正因为看不见，残差流的失控增长会被悄悄藏起来**，直到某个环节（比如最终 logits、某个线性层）才暴露。

### 2. 它的三个盲区

<details>
<summary><b>盲区 ①：残差流本身完全不受约束</b></summary>

$$
x_{l+1} = x_l + \mathrm{Sublayer}(\mathrm{RMSNorm}(x_l))
$$

那个加法 $x_l + (\cdot)$（identity 分支）**没有任何归一化**。RMSNorm 只归一化了"喂给子层的输入"，不归一化残差累加的结果。

- 每一层的增量虽有界（输入有界 × 权重），但**随深度累积**（随机游走式增长，量级 ~ $\sqrt L$ × 单层贡献）。
- 单层贡献又取决于权重范数，权重在训练中会涨，所以残差范数 $\|x_l\|$ 随训练/深度单调爬升是常见现象。
- **残差流逼近/超过 $65504$ 完全可能**，尤其是在 massive activation 场景（见下）。
</details>

<details>
<summary><b>盲区 ②：通道级 outlier（压的是整体尺度，不是通道分布）</b></summary>

假设某一个通道 $x_k = M$ 特别大，其余通道 $O(1)$：

$$
\mathrm{RMS}(x)\approx \frac{M}{\sqrt n}\;\Rightarrow\;\hat x_k \approx \frac{M}{M/\sqrt n}=\sqrt n
$$

所以**单个通道被压到 $\sqrt n$ 量级**——RMSNorm 确实能把"尖峰"压平。但它压的是**整体幅度**，通道之间的**相对分布原样保留**。这就是为什么会有 "outlier feature / massive activation" 现象：少数通道比其他通道大几十上百倍，这个结构在 norm 之后依旧存在（表现为 $\hat x$ 里某个分量顶到 $\sqrt n$）。

> 顺带：LLM.int8、SmoothQuant 那类工作盯的就是这种 outlier 通道——它们不影响 fp16/bf16 训练，但会搞死 per-tensor 量化。
</details>

<details>
<summary><b>盲区 ③：norm 之后的线性层 / 门控会重新放大</b></summary>

norm 输出被压在 $\sqrt n\gamma$ 只是"第一站"。紧接着：

$$
z = W\hat x,\qquad |z_j|\le \sqrt n\sum_i |W_{ji}|\,\gamma_i
$$

只要权重行范数够大，**$z$ 可以轻松回到 $10^4\sim10^5$**。norm 给你的那点"安全余量"（几百）在乘以权重、再摊到 $n$ 维求和之后很容易被吃掉。MLP 更狠：

$$
\text{SwiGLU}(x)=\underbrace{\mathrm{SiLU}(W_{gate}x)}_{\approx W_{gate}x\ (z\gg0)}\odot (W_{up}x)
$$

是**两个可能都很大的量做逐元素乘积**，量级叠加，是整张网里最容易冲高的一层。
</details>

---

## 三、那 fwd 激活到底什么时候 > 65504？

按机制归几类，想复现就照着踩：

| 触发场景 | 机制 | 典型位置 |
|---|---|---|
| **残差流累积 + 深模型** | identity 分支无归一，$\|x_l\|$ 随深度/训练增长 | 深层残差流 |
| **Massive activations** | 特定 token（BOS、标点、分隔符）的少数通道涨到 $10^4\sim10^5$ | 残差流的个别通道 |
| **MLP 中间层** | SwiGLU 门控乘积 / up-proj 放大 | FFN 的 $W_{up}/W_{gate}$ 输出 |
| **Attention logits** | $q\cdot k/\sqrt d$ 在高 LR / 长上下文 / 大 head_dim 下冲高（softmax 前） | attention scores |
| **训练不稳定期** | loss spike 前夜，激活范数先爆炸再回落 | 全局，尤其是 spike 那几步 |
| **权重增长** | 无 weight decay / 无 QK-norm，权重范数漂大 | QKV、FFN 投影 |
| **MoE 路由** | 专家门控权重（router softmax × expert 输出）被大权重放大 | MoE 层 |
| **最终 logits / lm_head** | logit 本身就能到几十上百，异常时更大；且通常保持 fp32 | lm_head |

**最关键的一条**：fwd 激活要 > 65504，几乎总是发生在 **"RMSNorm 之后、被线性层重新放大"** 的位置，或者 **"根本没经过 RMSNorm 的残差流/logits"** 上。RMSNorm 自己那一站永远是安全的（$\le\sqrt n\gamma$）。

---

## 四、为什么你 400M/1.5B/4B 就没见过

综合起来，你不炸是因为同时满足了：

1. **bf16**：outlier 就算真出现，也远没到上溢——$10^5$ 在 bf16 里只是个小数字。
2. **RMSNorm 把 sublayer 输入钉死**：绝大多数层的输入幅度被夹住，不会因为上一层异常而雪崩。
3. **中等规模**：残差累积深度有限，massive activation 更常见于大模型/长训练。
4. **LR 温和 + grad clip + warmup**：没进入"激活失控增长"的区间。

换句话说，**outlier 可能确实发生过，只是被 bf16 + RMSNorm 两道闸拦在了肉眼可见的范围外**。

---

## 五、想故意复现的话

- 切 **fp16** 并**关掉 dynamic loss scaling**，同时把 LM head / softmax 前的 logits 保持在 fp16——大概率能在 attention 或 logits 处看到 inf。
- 把 **LR 乘 5~10**、**去掉 warmup / gradient clip**，盯着 `grad_norm` 和每层 `激活.abs().max()` 的曲线，通常在 spike 前能看到某层 absmax 冲到 $10^4$ 以上。
- 构造 **massive activation 型数据**（大量以特殊 token 开头），观察残差流个别通道是否有 $10^4$ 量级尖峰。
- 在 fp16 下把 **MLP 中间层**（SwiGLU 输出）打成 log：这是最可能先越过 $65504$ 的地方。

**一句话**：RMSNorm 管的是"喂给算子的输入永远规整"，但残差流、norm 后的线性放大、以及门控乘积都不归它管——fwd 激活超 $65504$ 基本都出在这三个地方，只是在 bf16 + 这套稳结构下，你得专门去踩才会看到。
