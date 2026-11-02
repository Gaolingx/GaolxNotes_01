# Kimi Delta Attention 深度解析与原始论文总结

## 🧑‍💻 User

关于kimi delta attention，找找原始paper，并且给我最详细的解释，我是完全不懂这里面所有数理名词的，你要告诉我它跟普通的attention之间的区别是什么，跟attention相比，trade off又是什么？是否适合scaling？

---
## 😀 默认助手 (7)

我先确认一下工具和检索方式，然后去找原始论文。

找到了：KDA 出自 **Kimi Linear** 技术报告（arXiv:2510.26692）。我现在去拿全文细节。

已拿到原文。以下是完整拆解。

---

# 0. 先说论文出处

| 项目 | 内容 |
|---|---|
| 标题 | **Kimi Linear: An Expressive, Efficient Attention Architecture** |
| arXiv | [`2510.26692`](https://arxiv.org/abs/2510.26692)（v1: 2025-10-30，v2: 2025-11-01） |
| 团队 | Kimi Team / Moonshot AI（作者列表 59 人，含 **Jianlin Su**，即 RoPE 提出者） |
| 论文类型 | 技术报告（tech report），非会议论文 |
| 全文 | [HTML](https://arxiv.org/html/2510.26692v2) · [PDF](https://arxiv.org/pdf/2510.26692) |
| 代码 | `github.com/MoonshotAI/Kimi-Linear`（KDA Triton/CUDA kernel + vLLM 集成） |
| 权重 | `moonshotai/Kimi-Linear-48B-A3B-Base` / `-Instruct`（MIT 协议） |

⚠️ **一个关键事实**：**KDA 没有独立论文**。它是在 Kimi Linear 报告的第 3 节里被提出的一个**模块**。你要的"原始 paper"就是这一篇。KDA 的完整数学定义在 §2.2 + §3，算法实现在 §3.1，效率论证在 §3.2 和 §6.2，位置编码解释在 §6.1，复杂度在 §6.3。

---

# 1. 一句话版本

> **KDA = 在 Gated DeltaNet 的基础上，把"每个头一个遗忘旋钮"升级成"每个特征通道一个遗忘旋钮"，同时重排运算顺序，让它在 GPU 上比等价的通用 DPLR 形式快约 2 倍。**

它是**线性注意力**（本质是一个 RNN），不是标准 attention 的近似或加速版。这一点是理解后面所有内容的前提。

---

# 2. 从零开始：五级台阶（所有名词都会翻译）

## 2.1 台阶一：标准 attention 到底在做什么

标准 attention（softmax attention，SA）干的事是**查字典**：

- 每个 token 生成三个向量：$\boldsymbol{q}$（query，我想查什么）、$\boldsymbol{k}$（key，我能被什么查到）、$\boldsymbol{v}$（value，我携带的内容）
- 第 $t$ 个 token 输出 = 拿自己的 $\boldsymbol{q}_t$ 去和**所有历史** $\boldsymbol{k}_j$ 比对相似度，相似度越大，对应的 $\boldsymbol{v}_j$ 权重越大，加权求和：

$$\boldsymbol{o}_t=\sum_{j=1}^{t}\underbrace{\exp(\boldsymbol{q}_t^\top\boldsymbol{k}_j)}_{\text{相似度→权重}}\boldsymbol{v}_j$$

"softmax" 就是把一堆相似度分数归一化成概率。

**它的两个致命代价**：

1. **计算量**：每个 token 都要和所有历史比一遍 → 总复杂度 $O(T^2)$，序列翻倍，算力翻 4 倍。
2. **KV cache**：推理时为了不重复计算，要把所有历史 $\boldsymbol{k},\boldsymbol{v}$ 全部存在显存里 → 显存随 $T$ **线性增长**。100 万 token 时，KV cache 能吃掉几十上百 GB。

这就是为什么长上下文这么贵。**KDA 要解决的就是这两条。**

## 2.2 台阶二：线性注意力 = 把 attention 改写成 RNN

关键技巧：$\exp(\boldsymbol{q}^\top\boldsymbol{k})$ 这个形式可以拆成"先映射再内积"。于是把求和重排，可以先把所有 $\boldsymbol{k},\boldsymbol{v}$ **攒成一个矩阵** $S$：

$$\boxed{S_t=S_{t-1}+\boldsymbol{k}_t\boldsymbol{v}_t^\top,\qquad \boldsymbol{o}_t=S_t^\top\boldsymbol{q}_t}$$

这里：

- $S_t\in\mathbb{R}^{d_k\times d_v}$ 是**状态矩阵**（state），论文里叫 "memory" 或 "fast weights"，直译"快速权重"。它是**一个字典**：$d_k$ 行 × $d_v$ 列，把"key 空间"映射到"value 空间"。
- $\boldsymbol{k}_t\boldsymbol{v}_t^\top$ 叫**外积**（outer product）：一个向量乘一个向量的转置，得到一个 $d_k\times d_v$ 的**秩 1**（rank-1）矩阵。你可以把它理解成"只在 $\boldsymbol{k}_t$ 这一个方向上、写入 $\boldsymbol{v}_t$ 这份内容"。
- $\boldsymbol{o}_t=S_t^\top\boldsymbol{q}_t$ 就是"用 $\boldsymbol{q}_t$ 这个键去字典里查值"。

**收益立竿见影**：

- 计算：每个 token 只做 $O(d_k d_v)$ 运算 → 总 $O(T)$，**线性**。
- 内存：$S$ 的大小**固定**，跟 $T$ **完全无关** → 没有 KV cache 增长。

**代价也立竿见影**：注意 $S_t = S_{t-1} + \boldsymbol{k}_t\boldsymbol{v}_t^\top$ 是**只加不减**的。所有历史都被无差别累加进同一个固定大小的矩阵里，越堆越满，互相污染（论文原话："leads to interference over long contexts"）。

打个比方：**这是一本只能往后写、不能擦的笔记本，写满了字就糊成一片。** 这就是"有限状态"（finite state）的瓶颈——第一代线性注意力（Linear Attention / 早期 GLA）效果明显不如 softmax attention，根因就在这。

## 2.3 台阶三：DeltaNet——把"只写不擦"改成"先擦后写"

论文用了一个非常漂亮的视角：**把学这个字典 $S$ 看成一个在线优化问题**。

第一代线性注意力的目标函数是（论文式）：

$$\mathcal{L}_t(S)=-\langle S^\top\boldsymbol{k}_t,\boldsymbol{v}_t\rangle$$

这是个**无界**的目标：它只说"请把 $\boldsymbol{k}_t\mapsto\boldsymbol{v}_t$ 记住"，从不说"该忘什么"，所以永远在加。

DeltaNet（$\Delta$ 是"差值"的意思）换了一个目标：**要求"读回来的值"要等于"应该存的值"**：

$$\mathcal{L}_t(S)=\tfrac12\left\|S^\top\boldsymbol{k}_t-\boldsymbol{v}_t\right\|^2$$

这是**重建损失**（reconstruction loss）：用 $\boldsymbol{k}_t$ 查一次，查出来的结果应该等于 $\boldsymbol{v}_t$。对它做一步梯度下降（学习率 $\beta_t$）：

$$\nabla_S\mathcal{L}_t=\boldsymbol{k}_t\left(\boldsymbol{k}_t^\top S-\boldsymbol{v}_t\right)=-\left(\boldsymbol{k}_t\boldsymbol{v}_t^\top-\boldsymbol{k}_t\boldsymbol{k}_t^\top S\right)$$

$$S_t=S_{t-1}-\beta_t\nabla_S\mathcal{L}_t=\underbrace{\left(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\right)}_{\text{擦除算子}}S_{t-1}+\underbrace{\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top}_{\text{写入}}$$

**这就是 delta rule（德尔塔规则）**。把它翻译成人话：

1. **读**：用 $\boldsymbol{k}_t$ 查一下现在字典里存的是什么 → $S^\top\boldsymbol{k}_t$
2. **算差**：$\boldsymbol{v}_t - S^\top\boldsymbol{k}_t$，即"应该存的"减"已经存的"
3. **只写差值**：$S \mathrel{+}= \beta_t\boldsymbol{k}_t(\boldsymbol{v}_t-S^\top\boldsymbol{k}_t)^\top$

这四步跟上面的公式完全等价（中学代数展开即可验证）。所以 **delta rule = "只写残差"**。旧的关联会被顺手抵掉，不再越堆越糊。$\beta_t\in[0,1]$ 是"写入强度/学习率"。

那个 $(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top)$ 矩阵，在数值线性代数里叫 **Householder 变换**（当 $\beta=2$ 时是标准的镜像反射矩阵）。它的作用是：**把 $S$ 在 $\boldsymbol{k}_t$ 这个方向上的分量"削掉一部分"**。$\mathbf{I}$ 是单位矩阵（对角全 1，什么都不变）。

## 2.4 台阶四：Gated DeltaNet（GDN）——加一个遗忘旋钮

DeltaNet 擦得**不彻底**：它只在当前 $\boldsymbol{k}_t$ 这个方向上擦，别的方向上的旧记忆仍然永远保留。于是 GDN 引入一个**标量遗忘门** $\alpha_t\in[0,1]$：

$$S_t=\alpha_t\left(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\right)S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top$$

$\alpha_t$ 乘在整个 $S$ 上，意思是"**把整本笔记本的所有内容按同一个比例 $\alpha_t$ 淡化一遍**"。$\alpha_t=0.9$ 表示每步整体忘掉 10%。

论文把它解释为**快速权重的权重衰减**（weight decay）——就是深度学习里 L2 正则那个"让参数往 0 缩"的机制。当 $\alpha_t<1$ 时，记忆的时间尺度变有限，长上下文干扰被抑制。

到这里，GDN 已经是当前线性注意力里质量最好的一档（RWKV-7、Qwen3-Next 等都基于它）。

## 2.5 台阶五：KDA——把 1 个旋钮变成 128 个旋钮

**GDN 的旋钮是"每个头一个标量"。** 一个 head 有 $d_h$ 维（通常是 128），$\alpha_t$ 只是一个数，等于强行要求这 128 个特征通道**用同一个速率遗忘**。

KDA 干的唯一一件事：**把这个标量换成对角矩阵** $\mathrm{Diag}(\boldsymbol{\alpha}_t)$，"对角矩阵"听起来唬人，其实就是一个数组：对角线上是 128 个不同的 $\alpha_t^{(1)},\dots,\alpha_t^{(128)}$，其他位置是 0。它作用于向量时，就是对每一维**分别乘以自己的系数**——也就是 **128 个独立的遗忘旋钮**。

$$\boxed{S_t=\left(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\right)\underbrace{\mathrm{Diag}(\boldsymbol{\alpha}_t)}_{\text{128 个独立旋钮}}S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top}$$

就这么个改动。论文的表述是"**channel-wise / fine-grained gating**"（逐通道 / 细粒度门控）。

> **注意一个容易被忽略的细节**：KDA 把 $\mathrm{Diag}(\boldsymbol{\alpha}_t)$ 放在 Householder 项的**左边**（先衰减、再擦写）。GDN 里 $\alpha_t$ 是标量，所以写左边写右边无所谓；KDA 里它是矩阵，**顺序就有意义了**。这个顺序选择不是随意的，它是后面所有工程加速的来源（见 §5.3）。这一点论文没大张旗鼓强调，但它其实是 KDA 设计的**核心机关**。

**这个"细粒度门控"的想法本身不新**——GLA（Gated Linear Attention）早就用了 $\mathrm{Diag}(\boldsymbol{\alpha})$。KDA 的新意在于**把细粒度门控和 delta rule 组合起来，还组合得能跑得快**。在此之前，"delta rule" 和 "逐通道衰减" 这两条线的组合体（如 RWKV-7）在工程上很慢。

---

# 3. 完整对比表：Attention / Mamba2 / GDN / KDA

论文 Table 6 给出了所有变体的统一形式。我摘核心的：

| 模型 | 状态更新规则 | 遗忘粒度 | 会不会"擦除"方向 | 位置信息 |
|---|---|---|---|---|
| Softmax Attn | 无显式状态（KV cache 全存） | — | — | 需外加 RoPE |
| Mamba2 | $S_t=\alpha_t S_{t-1}+\boldsymbol{k}_t\boldsymbol{v}_t^\top$ | 标量 | ❌ 只加不擦 | 隐式衰减 |
| DeltaNet | $S_t=(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top)S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top$ | 不遗忘 | ✅ 秩 1 擦除 | ❌ |
| **GDN** | $S_t=\alpha_t(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top)S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top$ | **标量** | ✅ | 隐式、单频率 |
| **KDA** | $S_t=(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top)\mathrm{Diag}(\boldsymbol{\alpha}_t)S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top$ | **逐通道** | ✅ | 隐式、多频率 |

另一个等价视角（论文 §6.1，我个人认为这是全文最深刻的观察）：

$$o_t=\sum_{j=1}^{t}\left(\boldsymbol{q}_t^\top\prod_{s=j+1}^{t}\mathrm{Diag}(\boldsymbol{\alpha}_s)\left(\mathbf{I}-\boldsymbol{k}_s\boldsymbol{k}_s^\top\right)\boldsymbol{k}_j\right)\boldsymbol{v}_j$$

把它和标准 attention + RoPE 的形式对比：

$$s_{t,i}=\boldsymbol{q}_t^\top\left(\prod_{j=i+1}^{t}\mathbf{R}_j\right)\boldsymbol{k}_i$$

**结构完全一样。** 区别在于：RoPE 乘的是**固定的旋转矩阵** $\mathbf{R}_j$（角度由位置 $j$ 决定，不可学、正交）；KDA 乘的是**数据驱动、可学习的转移矩阵** $\mathrm{Diag}(\boldsymbol{\alpha}_s)(\mathbf{I}-\beta_s\boldsymbol{k}_s\boldsymbol{k}_s^\top)$。

> **所以 KDA 是一个"可学习的位置编码"。** 而且 RoPE 的强项正是"给每一对维度分配不同频率"（等价于对特征维做非均匀傅里叶变换）；GDN 只有每个头一个标量，天生缺这个多样性——**这正是论文提出 KDA 的动机**。

一个直接推论，也是论文的实际工程决定：既然 KDA 层已经自带位置信息，那些**全局 attention 层就不需要再加 RoPE 了**，于是他们把 MLA 层改成 **NoPE**（No Positional Encoding，无位置编码）。消融显示：短上下文成绩相近，但长上下文显著更好（128k 平均分 54.5 vs RoPE 版 51.8）。

---

<details>
<summary><b>§4. 补充：为什么"细粒度"真的有用（三个不同层次的理由）——点开</b></summary>

**理由一：记忆管理（容量层面）**

$S$ 是固定大小的（每头 $d_k\times d_v=128\times128=16384$ 个数）。这是个容量极其有限的"工作记忆"。如果 128 个通道共用一个衰减率，你没法同时做到"这一维记住 10 万 token 前的信息、那一维只记最近 100 个 token"。逐通道衰减 = 允许**不同的记忆槽有不同的寿命**，这是把一个 $O(1)$ 容量的记忆用满的必要条件。

**理由二：位置编码（信息论层面）**

见上文的 RoPE 类比。128 个不同的 $\alpha$ 相当于 128 个不同半衰期的"时钟"，它们的组合能编码出更丰富的相对位置信息——和 RoPE 用一组不同频率的旋转是同一个道理。

**理由三：实验证据**

论文做了三个合成任务（§5.1），这三个任务专门用来压测"记忆能力"：

- **Palindrome（回文）**：把输入序列倒着复述出来。这对固定状态模型极难，因为要精确检索整段历史。
- **MQAR（多查询联想回忆）**：给一串 `A→1, C→3, B→0, M→8, G→5`，然后问 `B=? G=?`。这是"字典查询"的直接测试，和语言模型能力高度相关。
- **Stack（栈状态跟踪）**：模拟 64 个 LIFO 栈的 push/pop。

结果：KDA 在**所有**任务、**所有**长度（256→2048）上都最高；在 Palindrome 和 MQAR 上收敛速度明显快于 GDN；而**Mamba2 在三个任务上全部失败**（因为它没有 delta rule 的擦除能力）。

</details>

---

# 5. 工程魔法：为什么它不只是"数学上更漂亮"

这是 KDA 论文里最有价值、也最少被科普的部分。**一个线性注意力如果 kernel 写不快，就等于没用。**

## 5.1 问题：RNN 形式在 GPU 上是灾难

$S_t=S_{t-1}+\dots$ 是**严格串行**的，$T$ 个 token 就要 $T$ 次顺序迭代，GPU 的几万个核心全部闲置。所以所有现代线性注意力都用 **chunkwise（分块）算法**：

- 把序列切成 $C=64$ 的块
- **块内**：因为块很短，可以像 attention 一样用矩阵乘法**并行**算（$C\times C$ 的小矩阵）
- **块间**：只有块级别的串行递推（$L/C$ 次，比 $L$ 少 64 倍）

## 5.2 WY 表示 + UT 变换

块内那 64 个连续的秩 1 更新 $(\mathbf{I}-\beta\boldsymbol{k}\boldsymbol{k}^\top)$ 连乘起来，可以用 **WY 表示**（Wang-Yau，1980 年代的数值代数结果）压缩成一个紧凑形式，代价是需要求一个 $C\times C$ 下三角矩阵的逆。下三角求逆可以用**前向代入**（forward substitution）高效做完。

**UT 变换**（UT transform）是进一步把这一步从"BLAS-3 矩阵乘"降成"更少的非 matmul FLOPs"。论文说这一招对训练时的硬件利用率"至关重要"——因为 GPU 上**矩阵乘法（matmul）几乎免费，"非 matmul" 运算（除法、逐元素）极贵**。

## 5.3 关键技巧：DPLR 的约束（全文最硬核的地方）

先解释 **DPLR = Diagonal-Plus-Low-Rank（对角 + 低秩）**。这是一个比 KDA 更一般的状态转移矩阵形式：

$$S_t=\left(\mathbf{D}-\boldsymbol{a}_t\boldsymbol{b}_t^\top\right)S_{t-1}+\boldsymbol{k}_t\boldsymbol{v}_t^\top$$

$\mathbf{D}$ 是对角矩阵（细粒度衰减），$\boldsymbol{a}_t\boldsymbol{b}_t^\top$ 是秩 1 扰动（擦除）。**RWKV-7 就是这种形式**，$\boldsymbol{a}_t,\boldsymbol{b}_t$ 是两个独立学习的向量。

**DPLR 的问题**：在 chunkwise 形式里会出现**累积衰减的倒数** $1/\Gamma$。$\Gamma$ 是 64 个 $[0,1]$ 之间的数连乘，可能小到 fp16 直接下溢成 0 —— 除以 0 就炸了。实践中只能：

1. 用 **fp32 高精度**算，
2. 再加一层 **secondary chunking**（把 64 再切小，比如切成 16）。

这两条都**堵死了半精度矩阵乘法的路**，而半精度 matmul 正是 Tensor Core 的全部价值所在。结果：DPLR 的 kernel 慢得让人不想用。

**KDA 的做法**：把 $\boldsymbol{a}_t$ 和 $\boldsymbol{b}_t$ **都绑定到同一个 $\boldsymbol{k}_t$ 上**。

把 KDA 的式子展开（把 $\mathrm{Diag}(\boldsymbol{\alpha}_t)$ 乘进去）：

$$S_t=\underbrace{\left(\mathrm{Diag}(\boldsymbol{\alpha}_t)-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\mathrm{Diag}(\boldsymbol{\alpha}_t)\right)}_{\mathbf{D}-\boldsymbol{a}_t\boldsymbol{b}_t^\top}S_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top$$

对应关系是：

$$\mathbf{D}=\mathrm{Diag}(\boldsymbol{\alpha}_t),\qquad \boldsymbol{a}_t=\beta_t\boldsymbol{k}_t,\qquad \boldsymbol{b}_t=\boldsymbol{k}_t\odot\boldsymbol{\alpha}_t$$

（$\odot$ 是逐元素乘。）**$\boldsymbol{a}_t$ 和 $\boldsymbol{b}_t$ 都从同一个 $\boldsymbol{k}_t$ 派生出来**——这就是论文说的"constrained variant of DPLR"。因为共享了 $\boldsymbol{k}$ 的方向，那些危险的倒数项可以在代数上消掉，不再需要高精度 + 二次分块。

**量化收益**（论文 §3.2 / §6.2，我核对了原文数字）：

| 指标 | 改善 |
|---|---|
| 二次分块（secondary chunking）的矩阵计算次数 | 4 → 2 |
| 额外消除的矩阵乘法 | 3 个 |
| kernel 执行速度（序列长度到 64k） | **约 2×** |
| 相对 GDN 的 prefill 延迟 | **几乎零增加**（曲线重合） |

最后一条特别重要：**KDA 比 GDN 表达力更强，但没有付出任何速度代价**。这是它能在 Kimi Linear 里直接替换 GDN 的前提。

---

# 6. 与普通 attention 的逐项对比

| 维度 | Softmax Attention (MLA) | KDA |
|---|---|---|
| **训练计算量**（每头，长度 $T$） | $2T^2d_h$ | $6Td_h^2+3TCd_h+TC^2$ |
| **复杂度阶** | $O(T^2)$ | $O(T)$ |
| **推理显存** | KV cache $\propto T$ | 状态 $\propto 1$（每头 $128\times128$ 固定） |
| **每 token 解码代价** | 随 $T$ 增长 | 恒定 |
| **能否精确检索** | ✅ 任意远 | ❌ 受固定容量限制 |
| **能否"忘掉"** | 靠 softmax 权重天然衰减 | 显式门控，可学 |
| **位置信息** | 需外加 RoPE | 内生于门控（故 MLA 层可用 NoPE） |
| **kernel 生态** | 极度成熟 | 需自定义 kernel（已开源） |
| **算术强度** | 高（大 matmul） | **低**（解码时是 memory-bound） |

**具体算一下交叉点**（这个数论文没给，是我代入式 13/14 算的）：取 $d_h=128$、$C=64$：

$$\text{KDA}\approx 98304T+24576T+4096T=126{,}976\,T$$
$$\text{Attn}=256\,T^2$$

令两者相等 → $T\approx 496$。

> **也就是说：序列超过大约 500 个 token，KDA 在 FLOPs 上就已经赢了。** 这也是为什么论文的实测里，4k–16k 区间两者打平（此时 attention 的成熟 kernel 效率更高，且小矩阵时 KDA 的非 matmul 开销占比大），到 128k 之后 KDA 才开始拉开，1M 时是压倒性优势。

---

# 7. Trade-off：它到底付出了什么代价

这部分是论文自己承认的，加上我从它的数字里推出来的。我按"是否真的是代价"分级。

## 7.1 代价一：**它不能纯用，必须混着用**（这是最根本的）

固定大小的状态 $S$ 意味着**信息容量有个硬上限**。这不是工程问题，是**信息论下界**：有些任务（比如"精确复述一遍输入"）需要的状态量是 $\Omega(T)$，任何 $O(1)$ 状态的模型都做不了。论文自己引用了 [104, 4, 45] 承认这一点。

所以 Kimi Linear **不是纯线性模型**，它是 **3:1 混合**：每 3 层 KDA 插 1 层完整的 MLA attention。

消融（论文 Table 1，PPL 越低越好，比的是 KDA:MLA 比例）：

| 比例 | 训练 PPL | 验证 PPL | 论文评语 |
|---|---|---|---|
| **3:1** | **9.23** | **5.65** | ✅ 最优，最终采用 |
| 0:1（纯 attention） | 9.45 | 5.77 | "performed poorly" |
| 1:1 | 9.29 | 5.66 | 质量可以，但推理开销更大 |
| 7:1 | 9.23 | 5.70 | 训练持平，**验证明显变差** |
| 15:1 | 9.34 | 5.82 | 更差 |

**读这张表的正确方式**：

- 优点是真的：3:1 打赢了 0:1（纯 attention），**在质量上就赢了**，不是"用质量换速度"。
- 但**纯线性（15:1 / 1:0）是崩的**。所以"75% KV cache 削减"是精确的 3/4，不是 100%。**剩下 25% 的层仍然是全 attention，仍然要吃 KV cache，仍然有 $O(T^2)$ 项**（只是系数小了 4 倍）。
- 7:1 训练集和 3:1 打平但验证集掉 —— 这是个**泛化**信号，说明线性层的比例上限比训练 loss 显示的要低。

## 7.2 代价二：表达力 vs. 数值稳定性的张力

KDA 是 DPLR 的**受限版本**（constrained variant）。它比 RWKV-7 的通用 DPLR **表达能力更弱**——论文明确说"general DPLR introduces richer model interactions"。

它自愿放弃这部分表达力，换来 2× 的 kernel 速度。**这是一个纯粹的工程取舍，不是免费的午餐。** 如果哪天硬件/kernel 生态能让通用 DPLR 跑得一样快，KDA 的约束就没有必要了。

## 7.3 代价三：工程生态

- 需要**专门写的 kernel**（chunkwise + WY + UT transform），朴素 PyTorch 实现会慢到不可用。
- 作者开源了 Triton/CUDA kernel 和 vLLM 集成——这本身就是在承认"不开源你就用不了"。
- 这一点在生态上是有真实摩擦的：所有推理框架、量化方案、投机解码、prefix cache 逻辑都要为"没有 KV cache 的层"重新设计（论文说他们的实现"drop-in compatible, requiring no modification to caching or scheduling interfaces"，但这需要相当多的适配工作）。

## 7.4 代价四：解码时是 memory-bound，不是 compute-bound

线性注意力解码时每 token 只做 $O(d_h^2)$ 的运算，但要把整个状态（128×128，每层每头）从 HBM 读进 SM。**算术强度极低** → 瓶颈在显存带宽而非算力。

这有两面：

- 坏处：单条请求的延迟不会因为"省了算力"而大幅下降。论文 Fig 7(b) 的 batch=1 实测，1M 上下文下 TPOT 只快 **2.2×**，不是 6×。
- 好处：因为省掉了 KV cache 的显存，**可以把 batch 开大很多**，吞吐才能真正起飞。论文声称的 **6×/6.3×** 是"吞吐"，不是"单请求延迟"。

> ⚠️ **论文在这里有个小不一致，值得你知道**：Abstract 和 §5.6 说 1M 上下文 6× 加速；但 §6.3 又写 "achieves a 2.3× speedup at a 1M token context"，Fig 7(b) 标注的是 2.2×。**2.2–2.3× 是 batch=1 的单请求实测，6×+ 是理论/大 batch 吞吐。** 引用时别混。

## 7.5 代价五：训练端收益有限

这是很多人误解的地方。训练时 attention 的 FLOPs 在总 FLOPs 里**占比不大**（MLP/MoE 才是大头）。所以 $O(T^2)\to O(T)$ 在训练上省的算力有限。论文的 scaling law 也印证了这点（见下）。

---

# 8. 它适合 scaling 吗？

分三个独立的问题回答，别混在一起。

## 8.1 算力上：适合，而且随规模恒定占优

论文做了正经的 scaling law 实验（§5.3，Table 2）：5 个 MoE 模型，653M → 1.7B **激活**参数，训练 38.8B → 128B token，全部 4096 上下文，Muon 优化器，8/64 专家激活。拟合结果：

$$\text{MLA}: \quad L=2.3092\times C^{-0.0536}$$
$$\text{Kimi Linear}: \quad L=2.2879\times C^{-0.0527}$$

（$C$ 是 PFLOP/s-days）

**怎么读这两个数**：

- **指数几乎相同**（0.0527 vs 0.0536）。这说明两条曲线**几乎平行** —— 增益是一个**近似恒定的乘性因子**，不会随规模增大而消失，也不会放大。
- 常数项更小（2.2879 < 2.3092）+ 指数略优 → 论文结论是 **~1.16× 计算效率**。人话：**达到同样 loss，KDA 省约 14% 算力。**

**这个 1.16× 要诚实对待**：它不算大。它不是"颠覆性加速"，而是"在质量不亏的前提下白拿 16%"。真正的价值在**推理**（长上下文），不在训练。

论文还说了一句很实在的话："We expect that careful hyperparameter tuning will yield superior scaling curves for KDA"——即这个 1.16× 是**没调参**的结果，可能被低估了。

## 8.2 质量上：目前证据很强，但验证范围有限

**强的部分**（这是 KDA 最出圈的结论）：

- 3B 激活 / 48B 总参数，**1.4T token**，同一配方下：短上下文全面超过 MLA 和 GDN-H；长上下文 128k 平均分 **54.5 vs MLA 52.2**（RULER 84.3 vs 81.3，MRCR 29.6 vs 22.6）；**RLVR 训练曲线全程优于 MLA**（AIME 2025、MATH500）。
- **5.7T token** 版本（Appendix D）：RULER@1M 拿到 **94.8**，远超同规模 Moonlight（16B total）几乎所有基准。

论文的措辞是"for the first time, outperforms full attention under fair comparisons"——**在短上下文、长上下文、RL 三个 regime 上同时赢**。这确实是第一次有线性/混合架构做到。

**必须打的折扣**：

1. **规模上限只到 1.7B 激活 / 5.7T token**。真正的 frontier 规模（百 B 激活、数十 T token）没验证。scaling law 的"平行曲线"结论在这个区间内成立，外推是**假设**。
2. **状态尺寸没被 scaling**。所有实验固定 $d_h=128$、16 头。$128\times128$ 的状态容量是否够用，在更长上下文下会不会成为瓶颈 —— **论文没做这个维度的 scaling**。这其实是这套架构最该被追问的地方。
3. **"公平比较"是自证的**。同一配方、同一 token 数、自己写的 kernel、自己选的 3:1 比例。第三方独立复现还没有。
4. **长上下文优势建立在一个特定的选择上**：把 MLA 层改成 NoPE。这个选择本身有风险（NoPE 在短上下文上会略吃亏，消融里 LiveBench 45.2 vs MLA 45.7、MATH500 81.2 vs 83.0 就是体现）。

## 8.3 工程上：适合，但门槛在这

**适合的理由**：

- 状态大小与 $T$ 无关 → 显存占用可预测，**长上下文可以开大 batch**，吞吐随规模线性受益。
- chunkwise 全是 matmul → 吃满 Tensor Core，训练和 prefill 都硬件友好（这正是 KDA 相对 DPLR 的核心卖点）。
- 混合结构可以**逐层平滑扩展**——你不需要一次换掉整个模型。

**门槛**：

- 你要么用他们的 kernel（Moonshot 已开源 + vLLM 集成），要么自己重写一遍。TP/PP/EP 并行、context parallel、prefix caching 都要重新适配"无 KV cache 层"。
- 这不是研究问题，是**工程债**问题。

---

# 9. 我的整体判断

<details open>
<summary><b>结论（点开可折叠，但建议读）</b></summary>

**1. KDA 不是 attention 的"加速版"，而是一条不同的技术路线的当前最优解。** 它属于 RNN 阵营（state-based），不是近似 attention。理解它的正确框架是"**一个用梯度下降规则在线更新的联想记忆**"，而不是"稀疏/低秩 attention"。

**2. 它对 attention 的真实区别就一句话**：attention 把所有历史都存着、每次都重新加权（$O(T)$ 内存、$O(T^2)$ 算力）；KDA 把历史**压缩进一个固定大小的矩阵**，用"逐通道遗忘 + 残差写入"来管理压缩质量（$O(1)$ 内存、$O(T)$ 算力）。**前者是无损检索，后者是有损摘要。**

**3. trade-off 的本质是"容量换成本"**，具体是：

- 你必须保留 1/4 的全 attention 层，因为固定状态在信息论上做不了精确检索 → **省的是 75% KV cache，不是 100%**
- 你必须接受 DPLR 受限带来的表达力损失，换 2× kernel 速度
- 你必须接受解码是 memory-bound，单请求延迟只快 2× 左右，真正的 6× 要靠大 batch

**4. 关于 scaling**：**架构层面适合（$O(T)$ 计算 + $O(1)$ 内存 + matmul 友好，这是 scaling 的三个必要条件），但实证层面只证明了"到 1.7B 激活为止"**。1.16× 的计算效率增益在这段区间内是稳定的乘性因子，不会随规模消失——这是好信号。但别忘了这是**没有精调超参**的下界。

**5. 最该被追问的一个问题**：状态的**尺寸**没被 scaling。所有实验都固定每头 $128\times128$。如果任务需要的记忆量随规模增长，而 $d_h$ 不变，这个瓶颈迟早会撞上——那时候要么加宽 head、要么提高全 attention 层比例，两条路都会侵蚀掉现在的效率优势。论文没回答这个问题，这是它最大的空白。

</details>

---

## 10. 想继续深挖的话，按这个顺序读原文

| 想搞懂什么 | 读哪 |
|---|---|
| 所有名词的严格定义（$\Gamma$、$\mathrm{Tril}$、chunk 记号） | §2.1 Notation |
| 从线性注意力到 GDN 的完整推导链 | §2.2 |
| KDA 定义 + 完整 chunkwise 算法（式 1–9） | §3 + §3.1 |
| 为什么 alpha 在前、为什么 $a=b=\beta k$ 能提速 | §3.2 + §6.2（含 DPLR/KDA 逐行对照伪代码 Listing 8a/8b） |
| 合成任务证据（MQAR / Palindrome / Stack） | §5.1 + Fig 4 |
| 消融（3:1 为什么最优、sigmoid vs swish、卷积层、NoPE vs RoPE） | §5.2 + Table 1 |
| scaling law 细节 | §5.3 + Table 2 + Fig 5 |
| RoPE 类比（最漂亮的一节） | §6.1 + Table 6 |
| FLOPs / 复杂度 / 推理策略 | §6.3 + 式 13–14 |
| 推到 5.7T token 的成绩 | Appendix D |

**一句话收尾**：KDA 的价值不在于那个 $\mathrm{Diag}$，而在于**它证明了"把 delta rule 的擦除能力和逐通道衰减的细粒度结合起来，同时不牺牲 kernel 速度"这件事是可做的** —— 而这个"同时"才是真正的贡献。数学上它只改了一个符号，工程上它改了三个矩阵乘法，结果上它第一次让混合线性架构在同等配方下打赢了全 attention。
