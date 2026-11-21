# LoRA训练前反向与显存节省分析

## 🧑‍💻 User

lora训练前向传播和反向传播相比全参完全不省吗

---
## 😀 默认助手 (7)

## 一句话结论

**「完全不省」这个说法，对前向完全成立，对反向只成立一半，真正省下来的是显存，不是算力。**

- **前向**：不省，反而**略增**（多的那点几乎没有）
- **反向**：理论上省掉约 **1/3** 的 FLOPs，但实际 wall-clock 往往接近 0，甚至更慢
- **显存**：省掉的是**优化器状态 / 梯度 / 可训练参数**这一大块，**激活值一点不省**

---

## 1. 前向：不但不省，还多算一点

设一层线性层 $Y = XW$，$X \in \mathbb{R}^{bs \times d_{in}}$，$W \in \mathbb{R}^{d_{in} \times d_{out}}$。

全参前向 FLOPs：

$$
2\,bs\,d_{in}d_{out}
$$

LoRA 前向（$W$ 冻结，加旁路 $BA$，$B \in \mathbb{R}^{d_{in}\times r}$，$A \in \mathbb{R}^{r \times d_{out}}$）：

$$
\underbrace{2\,bs\,d_{in}d_{out}}_{\text{主分支，一分不少}} + \underbrace{2\,bs\,r\,(d_{in}+d_{out})}_{\text{旁路新增}}
$$

旁路那项相对占比约为 $r/d$。取 $d=4096,\ r=16$，只有 **约 0.8%**。所以前向基本原地踏步、略增。而且训练时不能把 $BA$ 合并进 $W$（每步都在变），所以这 0.8% 一定得付。QLoRA 更糟：4-bit 权重每步还要反量化，前向是**净变慢**的。

---

## 2. 反向：理论省 1/3，实际常被吃掉

反向对一层线性层有两块计算：

| 计算 | FLOPs | LoRA 是否省 |
|---|---|---|
| $\mathrm{d}X = \mathrm{d}Y\,W^\top$（传给下层） | $2\,bs\,d_{in}d_{out}$ | ❌ **不省**，必须算 |
| $\mathrm{d}W = X^\top \mathrm{d}Y$（权重梯度） | $2\,bs\,d_{in}d_{out}$ | ✅ **省掉**（$W$ 的 `requires_grad=False`） |
| $\mathrm{d}A,\mathrm{d}B$（旁路梯度） | $\sim 2\,bs\,r(d_{in}+d_{out})$ | 新增，极小 |

于是：

$$
\text{全参} \approx 3M,\qquad \text{LoRA} \approx 2M + \varepsilon,\qquad M = 2\,bs\,d_{in}d_{out}
$$

**理论上总 FLOPs 省 1/3**，这个省法是真的，`dW` 那个矩阵乘确实被跳过了。

但 wall-clock 上常常看不出来，原因是：

1. **$\mathrm{d}X$ 才是反向的大头**，它一分没省。而它本身是**访存受限**的，不是算力受限，砍掉 $\mathrm{d}W$ 不大幅改变墙钟时间。
2. **激活值一点不省**：要算 $\mathrm{d}X$、$\mathrm{d}A$ 就得把每层输入 $X$ 存下来。所以 gradient checkpointing 照开，重算成本完全一样——而开了 checkpoint 之后，重算的前向开销会进一步稀释那 1/3 的理论收益。
3. **注意力算不动**：长序列下 FLOPs 被 $O(s^2 d)$ 的 attention 支配，LoRA 一根手指都碰不到；序列越长，LoRA 的算力收益越趋近于 0。
4. **kernel 开销**：多两个小 matmul + 额外的 kernel launch，小模型上 LoRA 常常**比全参还慢一点**。

---

## 3. 真正省的是显存（而且省得很猛）

以 7B 模型、AdamW、混合精度为例：

| 项目 | 全参训练 | LoRA (r=16, ~0.1% 可训练) |
|---|---|---|
| 权重 (fp16) | 14 GB | 14 GB（冻结但仍在显存里） |
| 梯度 | 14 GB | ~0.04 GB |
| fp32 master weights | 28 GB | ~0.08 GB |
| Adam $m,v$ | 56 GB | ~0.16 GB |
| **小计（不含激活）** | **~112 GB** | **~14.3 GB** |
| 激活值 | $A$ | **$A$（完全相同）** |

**可训练状态从 ~98 GB 掉到 ~0.3 GB**，这是几千倍的差别。但激活值那一项 $A$ 原封不动——所以「LoRA 显存不省」的说法，说的是**激活值**那一半，不是全部。

> 顺带一个坑：用 FSDP / ZeRO-3 时，冻结参数在 forward 前仍要 all-gather，参数显存的分片收益会被部分吃掉；用 ZeRO-2 则梯度/优化器状态的收益是实打实的。

---

## 4. 那到底「省不省」——分场景

<details>
<summary><b>LoRA 确实更快 / 更省的时候</b></summary>

- **显存被参数/优化器状态卡死**的模型（>7B）：全参根本跑不起来，LoRA 能跑，还能把省下的显存换成**大 batch**，吞吐反而更高——但这个提速来自 batch size，不是来自 per-sample FLOPs。
- 单卡多任务、需要同时挂多个 adapter：adapter 显存可以复用同一份冻结底座。
- 优化器开销占比较大时（Adam 的 $m,v$ 更新、fp32 master 拷贝），LoRA 把这部分直接干掉了。
</details>

<details>
<summary><b>LoRA 几乎没收益 / 更慢的时候</b></summary>

- **长上下文**：attention 主导 FLOPs 和激活，LoRA 收益 → 0。
- **小模型 + 短序列**：kernel 开销占比高，LoRA 常比全参慢。
- **QLoRA**：前向多一次 dequant，反向多一次 dequant，实测吞吐通常低于 bf16 LoRA，更低于同 batch 的全参。
- 已经开了 activation checkpointing 的大模型：重算成本不变，1/3 的理论收益被摊薄。
</details>

---

## 5. 把话说准的两句话

1. **「LoRA 前向反向一点不省」**——前向对（还略增），反向不对（省约 1/3 的 FLOPs，主要是跳过了 $\mathrm{d}W$）。
2. **「LoRA 省显存不省算力」**——更准确的说法是：**省的是参数/梯度/优化器状态显存，不省激活值显存；省的是 $\mathrm{d}W$ 那份算力，不省 $\mathrm{d}X$ 那份算力。**

如果你在评估要不要上 LoRA，判断标准就一条：**瓶颈在显存还是吞吐？** 在显存（且是参数侧）→ LoRA 值得；在吞吐（尤其是长序列）→ LoRA 帮不上忙，得看序列并行 / FlashAttention / 更小的中间维度。

---
## 🧑‍💻 User

反正你无论是lora还是全参
完整的model.forward是一定会跑的
反向传播也是
甚至因为要更新适配器实际计算量更大了
但是多卡可以降低通信瓶颈稍微比全参快一点点（梯度同步那块）

---
## 😀 默认助手 (7)

## 三条对，一条要翻案

你的三条直觉里，**前向、反向、“通信省”基本都对**，但「计算量更大」这条要翻案：**FLOPs 其实是变小的，只是墙钟时间不一定变短**。这是两件事。

---

### ✅ 前向一定跑满 —— 对

`model.forward` 一行不落，跟全参一模一样，还多两个 skinny matmul。这条没有任何争议，省不了。

### ✅ 反向的「激活梯度链路」一定跑满 —— 对了一半

要往下层传 $\mathrm{d}X$，就必须一路 backprop 到底，这段图跟全参**完全一样**，省不了。

但有一刀是真的砍掉了：**$W$ 的权重梯度 $\mathrm{d}W = X^\top \mathrm{d}Y$ 那个矩阵乘**。`requires_grad=False` 之后 autograd 压根不建这个节点。

所以准确的记账（$N$ = 非 embedding 参数量，per token）：

$$
\text{全参} = \underbrace{2N}_{\text{fwd}} + \underbrace{2N}_{\mathrm{d}X} + \underbrace{2N}_{\mathrm{d}W} = 6N
$$
$$
\text{LoRA} = \underbrace{2N}_{\text{fwd}} + \underbrace{2N}_{\mathrm{d}X} + \underbrace{\varepsilon}_{\text{adapter},\ \varepsilon \approx 0.008N} \approx 4N
$$

### ❌ 「实际计算量更大了」 —— 这条不成立

按 FLOPs 算：**$6N \to 4N + \varepsilon$，少 1/3**，不是多。adapter 那点新增只占主分支的 $r/d \approx 0.8\%$，吃掉的是 $33\%$ 收益里的零头。

**但你的体感没错** —— LoRA 在小模型/短序列上确实常常持平甚至更慢。原因不是「算得更多」，而是**算得更烂**：

- $4096 \times 16$ 这种 skinny GEMM 算术强度极低，是纯 memory-bound，SM 占用率可能只有 5~10%
- 主分支虽然 FLOPs 是它的 125 倍，但跑在 60%+ 的峰值效率上
- 所以 adapter 在**时间**上可能占 5~15%，远高于它在 **FLOPs** 上的 0.8%
- 再叠上额外的 kernel launch、以及 QLoRA 每步的反量化

一句话：**LoRA 是 FLOPs 省了、GPU 效率掉了，两者相抵，小规模下经常打平。** 说成「计算量更大」是不准确的，说成「有效算力利用率更低」就对了。

### ✅ 多卡梯度同步省 —— 对，而且你低估了

这块收益比大多数人以为的大。DDP 每步 all-reduce 的通信量是 $2 \times$ 可训练参数（bf16）：

| | 全参 7B | LoRA (r=16) |
|---|---|---|
| 每步 all-reduce 量 | **14 GB** | **~40 MB** |
| 倍数 | — | **~350×** |

- 走 NVLink/InfiniBand：这段本来就被 backward 完全 overlap 掉了，省下来看不出什么
- 走 PCIe / 普通以太网：全参的 14 GB all-reduce 可能是每步几百毫秒到秒级 —— 这时候 LoRA 就是**实打实地把通信从瓶颈里删掉了**
- 附带红利：显存松了 → batch 变大 → **步数变少** → all-reduce 的总次数也变少，这是乘性收益

**但有个大坑**：这个收益只在 **DDP / ZeRO-1/2** 下成立。

> **FSDP / ZeRO-3 下通信省不掉**：冻结参数虽然不产生梯度，但 forward 前**仍然要 all-gather** 才能算。$W$ 的 gather 一分不减，只有 backward 的 reduce-scatter 省了。如果你的瓶颈是 FSDP 的 all-gather 带宽，LoRA 帮不上忙 —— 得靠「底座整份复制、只对 adapter 做 FSDP」这类技巧，代价是每卡要放全量底座。

---

## 最终对账

| 维度 | 相对全参 | 备注 |
|---|---|---|
| 前向激活值显存 | **1.0×** | 完全一样，checkpointing 照开 |
| 前向 FLOPs | 1.008× | 略增 |
| 反向 FLOPs | **0.5×** | 砍掉 $\mathrm{d}W$ |
| 总 FLOPs | **~0.67×** | $6N \to 4N$ |
| GPU 有效利用率 | **↓** | skinny GEMM，这才是「体感没快」的元凶 |
| 参数/梯度/优化器状态显存 | **~0.01×** | 几千倍 |
| DDP 梯度通信 | **~0.003×** | 但你得是 DDP/ZeRO-1/2 |
| FSDP all-gather 通信 | **1.0×** | 冻结参数照样 gather |

所以更精准的表述是：**LoRA 省的是 FLOPs、参数侧显存和集合通信量；不省的是激活值、前向计算、和 GPU 的时间效率。**

---
## 🧑‍💻 User

至于为什么能降低显存占用单纯是training params变少了吗

---
## 😀 默认助手 (7)

## 是，但「变少」要拆成**三件相互独立的事**，其中一件根本不是「变少」

说「单纯是 training params 变少吗」，答案接近是。但精准点讲，LoRA 的显存收益来自三个**可分离**的机制，只盯着参数量会漏掉两个。

---

## 账本：一个可训练参数要花多少字节

混合精度 Adam 训练，全参时**每个参数**的开销：

| 缓冲 | 精度 | 字节/参数 |
|---|---|---|
| 前反向用的参数副本 | bf16 | 2 |
| 梯度 | bf16 | 2 |
| fp32 master weight | fp32 | 4 |
| Adam 一阶矩 $m$ | fp32 | 4 |
| Adam 二阶矩 $v$ | fp32 | 4 |
| **合计** | | **16 B** |

$N=7\text{B}$ 时：$16 \times 7\text{e}9 \approx 112\ \text{GB}$。

LoRA 时：
- **冻结的底座**：2 B/参数（bf16），**梯度/优化器/master 全部不分配**
- **可训练 adapter**：仍是 16 B/参数，但只有 $0.001N$

$$
\underbrace{2 \times N}_{\text{底座}} + \underbrace{16 \times 0.001N}_{\text{adapter}} \approx 2.02N \approx 14\ \text{GB}
$$

**$112\text{GB} \to 14\text{GB}$，约 8 倍。**

---

## 三个机制，逐个拆

### 机制一：优化器状态只给可训练参数分配 ← 你想到的那个

Adam 的 $m, v$ 是 fp32，加起来 **8 B/参数**，是 16 B 里最大的一块。这个量严格正比于**可训练参数量**，所以 $N \to 0.001N$ 直接把它干到几乎为零。

$56\text{GB} \to 0.06\text{GB}$，这一项省了 ~900×。

> 细节：PyTorch 的 Adam 是**惰性**分配，`grad is None` 的参数不会建 state，所以就算你误传 `model.parameters()` 也不会爆炸。但 DeepSpeed / 某些 fused 优化器会预先按 param group 分配，还是显式 `filter(lambda p: p.requires_grad, ...)` 更稳。

### 机制二：梯度缓冲区 —— 也是「变少」的直接推论

冻结参数不产生 `.grad`，梯度从 14 GB → 40 MB。这跟机制一其实是同一个根因，但独立站得住。

### 机制三：底座可以掉到推理精度 ← **这个不是「变少」，是「降精度」**

这条容易被忽略，但**贡献了 8 倍里的整整一半**。

全参训练时，那 4 B 的 fp32 master weight 是**必须**的：因为优化器更新量是 $10^{-5}$ 量级，直接加在 bf16 参数上会被舍入吃掉（bf16 只有 8 位尾数）。

而 LoRA 里底座**只读不写**，前向用 bf16 完全够，不需要任何 master copy。**参数量一个没少，但每个参数从 4 B 变 2 B。**

$$
\text{全参}: 16N \qquad\longrightarrow\qquad \text{LoRA}: \underbrace{2N}_{\text{只读, 可降精度}} + \underbrace{16\times0.001N}_{\text{可训练}}
$$

如果只看「参数量」不看精度，你会以为省了 1000 倍；实际是 8 倍 —— 差的那个系数就是这条。

### 机制四（严格说是另一件事）：QLoRA 的 4-bit

NF4 量化后底座约 **0.5 B/参数** + 量化常数（double quant, blocksize 64）约 **0.13 B**，合计 ~0.63 B：

| 方案 | 字节/底座参数 | 7B 底座 |
|---|---|---|
| 全参 Adam (bf16 mixed) | 16 | 112 GB |
| LoRA (bf16) | 2.02 | 14 GB |
| **QLoRA (NF4)** | **0.63** | **4.4 GB** |

这跟「训练参数少」**毫无关系**，纯是量化技术。它是第三个独立机制。

---

## LoRA 完全省不到的三块（这就是「实测省得没那么多」的原因）

<details>
<summary><b>展开：真正的显存大户</b></summary>

1. **激活值** —— $O(bs \cdot s \cdot d \cdot L)$，attention 还有 $O(bs \cdot h \cdot s^2)$。开 checkpointing 后是 $O(bs\cdot s\cdot d\cdot\sqrt{L})$ 但要多付一次前向。**长上下文下这是最大头，且一分不省。**
2. **Logits + CrossEntropy** —— $bs \times V$ 的 bf16 张量再 upcast 成 fp32。$32\text{k}$ token × $128\text{k}$ 词表：bf16 就 $8.6\text{GB}$，fp32 版 $17\text{GB}$。**单张量就能顶掉整个模型**，LoRA 一点不碰。缓解：chunked CE / fused linear-cross-entropy。
3. **临时 workspace** —— cuBLAS workspace、attention workspace、all-reduce/NCCL buffer、memory fragmentation 造成的 peak 尖峰。

所以 7B + 8k 序列的场景，你可能看到「全参 OOM 在 80GB，LoRA 只用 45GB」—— 省下的 67GB 是参数状态，剩下 45GB 里绝大部分是激活和 logits。

</details>

---

## 一句话收口

**主因确实是可训练参数变少（优化器状态 + 梯度），但完整的解释还需要加上「冻结参数可降精度」，以及显式使用时的「底座量化」。而真正限制你 batch size 的激活值和 logits，LoRA 从来就没打算省过。**

顺带一个对照：如果**两边都开 FSDP**，底座参数本来就被分片到每张卡，LoRA 在「参数显存」上的优势就被抹平了 —— 那时剩下的纯收益就是 **优化器状态**（以及前面的梯度通信），这也是 FSDP + LoRA 组合下实测省得没 DDP 场景那么夸张的原因。

---
## 🧑‍💻 User

计算量的话 总计算量不会变大
forward会大点 反向是小于正常计算的
forward一层lora会多一次mm和allreduce

---
## 😀 默认助手 (7)

## 两条对，一条要改，一条要拆

你的判断里 **「总计算量不增大」✅、「forward 略大 / backward 显著小」✅**，但最后一句有两个点需要修正：**mm 是两次不是一次**，而 **forward 在 DDP 下根本没有 all-reduce**。

---

## 先把账算准

$$
\text{全参} = \underbrace{2N}_{\text{fwd}} + \underbrace{2N}_{\mathrm{d}X} + \underbrace{2N}_{\mathrm{d}W}_{\text{backward}=4N} = 6N
$$

$$
\text{LoRA} = \underbrace{2N + \varepsilon_f}_{\text{fwd}} + \underbrace{2N + \varepsilon_b}_{\text{backward}=2N+\varepsilon_b} = 4N + \varepsilon
$$

| | 全参 | LoRA | 比值 |
|---|---|---|---|
| forward | $2N$ | $2N + \varepsilon_f$ | **1.008×** |
| backward | $4N$ | $2N + \varepsilon_b$ | **~0.5×** |
| 总计 | $6N$ | $4N+\varepsilon$ | **~0.67×** |

反向不是「小一点」，是**直接砍半** —— 因为 $\mathrm{d}W$ 那一整半没了。总账少 1/3。这条你说对了，而且比你想的更激进。

---

## 修正一：forward 多的是**两次** mm，不是一次

$Y = XW + XBA$ 里的旁路是 $X \xrightarrow{\ \times B\ } (bs \times r) \xrightarrow{\ \times A\ } (bs \times d_{out})$ —— **两个串行 GEMM**。

- FLOPs 上等价于一次秩-$r$ 的线性映射：$2bs\,r(d_{in}+d_{out})$
- 但 kernel 层面是两次 launch，而且都是 **skinny GEMM**（$r=16$）

这就是为什么它 FLOPs 只占 0.8%、时间上却能吃掉 5~15%：**两次低占用率的 kernel，比一次高占用率的大 kernel 慢得多。**

> 有些框架会把 $BA$ 预合并成一个 $d_{in}\times d_{out}$ 矩阵再乘 —— 但训练时 $A,B$ 每步都在变，合并不划算（合并本身就是一个 $2 d_{in} r d_{out}$ 的 GEMM，跟直接算差不多），所以标准实现就是两次。

---

## 修正二：forward 里的 all-reduce —— **DDP 下是零**

这是关键区分。**DDP 的 forward 完全没有集合通信**，梯度 all-reduce 全部发生在 backward。所以「forward 多一次 all-reduce」在 DDP 下不成立 —— 加 LoRA 也不会凭空造出一个。

真正的位置是这样：

| 并行方式 | forward 通信 | backward 通信 | LoRA 的影响 |
|---|---|---|---|
| **DDP** | **无** | 梯度 all-reduce | fwd 不变；bwd 通信量 **↓~350×** |
| **FSDP / ZeRO-3** | 参数 all-gather | 梯度 reduce-scatter | all-gather **不变**（冻结参数照样 gather）；reduce-scatter ↓ |
| **TP（Megatron 式）** | 部分层后 all-reduce | 对应反向 | 见下 |

- **DDP**：LoRA 只让 backward 的 all-reduce 变小，forward 一字未动。
- **FSDP**：forward 的 all-gather 一分不省 —— 这是 FSDP + LoRA 实测收益不如 DDP 场景的主要原因。
- **TP**：如果 adapter 按 $W$ 的分片方式切（$B$ column-parallel、$A$ row-parallel），它的输出可以**并进主分支原有的那个 all-reduce**，不新增通信。

<details>
<summary><b>唯一会「多出通信」的情形（这是实现问题，不是 LoRA 的固有成本）</b></summary>

如果 TP 已经把 hidden 维切开了，而 adapter 却**在完整 hidden 维上算**（某些 PEFT 实现直接复制一份全量 $B,A$ 到每张卡），那 forward 前就得多一次 **all-gather $X$**。

这属于「adapter 没跟上底座的切分方式」，是工程实现的锅。把 adapter 按同样方式切开就能消掉。
</details>

---

## 收口

**总 FLOPs 少 1/3（fwd +0.8%、bwd 砍半），forward 多两次 skinny GEMM，通信只在 backward 且显著变少 —— forward 没有任何新增集合通信。**

至于「体感没快」：FLOPs 的 33% 收益被 skinny GEMM 的低效率和额外 kernel launch 吃掉了一部分，再被激活值不变、logits 不变、FSDP 的 all-gather 不变这几项一摊，剩下的净收益就取决于你的瓶颈到底在**算力、还是访存/通信/显存容量**。
