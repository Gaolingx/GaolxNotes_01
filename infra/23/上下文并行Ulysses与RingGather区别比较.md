# 上下文并行Ulysses与RingGather区别比较

## 🧑‍💻 User

llm训练中，上下文并行 ulyssess/all gather cp/ring attn 有什么区别

---
## 😀 默认助手 (7)

# 上下文并行（CP）中 Ulysses / Ring / All-Gather 的对比

## 0. 一句话核心结论

三种方案**在数学上完全等价**（都是精确的 full attention），per-rank 的 **FLOPs 也完全相同**（$O(L^2 H d_h / N)$），连**通信量都在同一量级**（$\sim 2Ld$ 每层）。真正的区别只有一个：

> **它们只是"把 KV 通过什么方式重分布给需要它的 query"的三种不同调度**，差别体现在**显存峰值、通信拓扑/消息数、kernel 复杂度、GQA 友好度、causal 负载均衡**。

- **All-Gather CP**：把 KV **广播给所有人**（replicate KV）
- **Ulysses**：把序列 **all-to-all 转成 head 并行**（replicate sequence，shard heads）
- **Ring Attention**：KV **沿环流水线传递**（shard everything，stream KV）

三者都可以看成"沿 (序列维, head维) 两个轴选择哪个 replicated、哪个 sharded"的不同取点。USP 那篇工作就是把这个统一视角写清楚的。

---

## 1. 记号与前提

设序列长 $L$、hidden $d$、注意力头数 $H$、head dim $d_h$（$H d_h = d$）、KV 头数 $H_{kv}$（GQA/MQA）、CP 并行度 $N$。

CP 的通用做法：**所有激活沿序列维切 $N$ 份**，每卡持 $L/N$ 个 token。MLP、LayerNorm、dropout 这些逐 token 算子天然无通信；**只有 attention 需要跨卡**，因为每个 query 需要看全部 KV。三种方案就是这三种不同的处理方式。

---

## 2. 三种方案逐个说

### 2.1 All-Gather CP（也叫 KV all-gather / Megatron `allgather`）

**做法**：每层 attention 前，把各卡的 K、V 沿序列维 **all-gather** 成全长 $L$；然后每卡只用**自己那 $L/N$ 个 query** 与全长 KV 算注意力（本地 FlashAttention，通常不需要 num-head 拆分）。

```
rank0: Q[0:L/N]  +  gather → K,V[0:L]   → attn(Q_local, K_full)
rank1: Q[L/N:2L/N] + gather → K,V[0:L]  → ...
```

- **通信**：每层 2 次 all-gather（或 QKV 融合的 1 次），每卡流量 $\approx 2L d_{kv}\cdot\frac{N-1}{N}$
- **显存**：**KV 是复制的**，每卡峰值 $O(L \cdot d_{kv})$，而不是 $O(L d_{kv}/N)$
- **优点**：最简单、**任何 attention 实现都能直接使用**（FlashAttention、任意 mask、ALiBi、sliding window、varlen 打包全兼容）；不要求 $H$ 能被 $N$ 整除；没有 kernel 改造；数值最稳
- **缺点**：KV 显存不随 CP 缩小；每层都要重新 gather（backward 要么存、要么再 gather 一次）；causal load imbalance 要靠 **zigzag 切分**解决
- **GQA 时的关键优势**：$d_{kv} = \frac{H_{kv}}{H}d$ 很小，gather 的量和显存都随 GQA 比例**自动降低 4~8 倍**，所以对 GQA 模型其实非常划算

### 2.2 Ulysses（DeepSpeed，`a2a`）

**做法**：attention 前做一次 **all-to-all**，把布局从「**序列切分 × 全 head**」转成「**全序列 × head 切分**」：

$$\underbrace{(L/N,\; H,\; d_h)}_{\text{seq-shard}} \xrightarrow{\text{all-to-all}} \underbrace{(L,\; H/N,\; d_h)}_{\text{head-shard}}$$

然后每卡在**完整 $L$** 上、只对**自己那 $H/N$ 个头**做标准 attention（直接用普通 FlashAttention），算完再 all-to-all 变回序列切分布局。

- **通信**：每层 2 次 all-to-all，需要搬 **Q、K、V 三个**（因为每卡要拿到全长序列的所有行），每卡流量 $\approx 3Ld\cdot\frac{N-1}{N}$ —— 比另外两种多 ~1.5×
- **显存**：attention 输入 $L \times (H/N) \times d_h = Ld/N$，**和其他方案一样**，没有 KV 复制
- **独特优点：causal 负载天然完美均衡**。每卡手里是全长序列，mask 是完整可见的，不存在"rank i 只看 i 份 KV"的倾斜，**不需要 zigzag**
- **硬约束**：$N \mid H$（甚至 $N \mid H_{kv}$），所以 CP 度受限于 head 数；**MQA（$H_{kv}=1$）直接做不了**
- **GQA 时很不友好**：Q 必须全量搬运（$H$ 个头），KV 头数不足 $N$ 时要复制 KV，流量和显存都会上升 —— 这是 Ulysses 在 Llama 类模型上最大的局限
- **拓扑要求**：all-to-all 需要全互联，实践中基本**必须在一个 NVLink 域内**（单机 8 卡，或 NVLink Switch 域），跨 IB 网做 all-to-all 会很差
- 另外 all-to-all 伴随 seq-major ↔ head-major 的 permute，需要额外 contiguous copy

### 2.3 Ring Attention（`p2p`）

**做法**：每卡保留自己的 KV 块（$(L/N, H, d_h)$），然后让 KV 块**沿 ring 依次传递**：第 $k$ 步用本地 query 与当前 KV 块算 attention（配合 FlashAttention 的 online softmax 累积），算完把 KV 发给下一卡、从上一卡收新的。走完 $N-1$ 步，每个 query 都看过全部 KV。

```
step 0: rank i  attn(Q_i, KV_i)
step 1: rank i  attn(Q_i, KV_{i-1})
...
step N-1: rank i attn(Q_i, KV_{i+1})
```

- **通信**：$N-1$ 轮 P2P（每轮发 K、V 两块），每卡流量 $\approx 2L d_{kv}\cdot\frac{N-1}{N}$ —— 和 all-gather 一样
- **显存：三种里唯一真正 $O(L/N)$ 的**，KV 流式进出、从不同时驻留，因此**序列长度可以随卡数线性扩展，CP 度不受 head 数限制**
- **可以用 compute-comm overlap 把通信几乎完全隐藏**（blockwise + double buffer），这是它设计的核心
- **缺点**：需要专门的 ring attention kernel（online softmax 重标定、跨块 mask 逻辑、backward 重算），实现复杂度最高；**消息条数多**（每层 $2(N-1)$ 次），小消息时受延迟制约，overlap 做不好反而比 all-gather 慢；**causal 负载严重不均**，必须用 **zigzag / striped**（把序列切成 $2N$ 块，rank $i$ 拿块 $i$ 和块 $2N-1-i$）来均衡
- **天然适合 GQA**：只搬 KV，量随 $H_{kv}/H$ 直接缩小

---

## 3. 对比表

| 维度 | **All-Gather CP** | **Ulysses (a2a)** | **Ring (p2p)** |
|---|---|---|---|
| 重分布方式 | KV 广播给所有卡 | all-to-all：seq→head | KV 沿环流式传递 |
| 每卡 attention 计算 | $(L/N)\times L$，全 head | $L\times L$，$H/N$ head | $(L/N)\times L$，全 head（分 $N$ 步） |
| 每卡 FLOPs | $O(L^2Hd_h/N)$ | 同左 | 同左 |
| 每卡通信量（每层） | $\sim 2Ld_{kv}$ | $\sim 3Ld$（含 Q） | $\sim 2Ld_{kv}$ |
| 通信原语 | all-gather（树/环） | **all-to-all** | **P2P ring** |
| 每层同步次数 | 2 | 2 | $2(N-1)$ |
| **KV 显存/卡** | $O(Ld_{kv})$ ❌ | $O(Ld/N)$ ✅ | $O(Ld_{kv}/N)$ ✅✅ |
| CP 度上限 | 任意 | $\le H$（常 $\le H_{kv}$） | 任意，可跨机 |
| 跨节点 | 可以但流量大 | ❌ 基本不行 | ✅ 最适合 |
| GQA 友好度 | ✅ 成本自动降 | ❌ 受 $H_{kv}$ 限制 | ✅✅ |
| causal 负载均衡 | 需 zigzag | **天然均衡** | 需 zigzag/striped |
| kernel 改造量 | **零** | 零（标准 FA） | **大**（自研 kernel） |
| mask 灵活性 | 最高（任意 mask/varlen） | 高 | 较低 |
| 数值 | 精确 | 精确 | 精确（softmax 归约顺序不同） |
| 主要瓶颈 | KV 显存 + 重复 gather | all-to-all 带宽、permute | 延迟、overlap 效率 |

---

<details>
<summary><b>4. 通信量推导（点开看）</b></summary>

每层、每卡 in/out 流量（按元素个数计）：

- **Ulysses**：本地持有 Q,K,V 各 $Ld/N$ 个元素，all-to-all 后仍是 $Ld/N$（变布局）。发出的比例是 $\frac{N-1}{N}$，三个张量：
$$ V_{\text{ulysses}} \approx 3Ld\cdot\frac{N-1}{N} $$
- **Ring**：$N-1$ 步，每步发 K、V 各 $(L/N)d_{kv}$：
$$ V_{\text{ring}} \approx 2Ld_{kv}\cdot\frac{N-1}{N} $$
- **All-Gather**：gather 出全长 KV，每卡需补齐 $\frac{N-1}{N}$：
$$ V_{\text{allgather}} \approx 2Ld_{kv}\cdot\frac{N-1}{N} $$

MHA 下 $d_{kv}=d$，所以量级都是 $\Theta(Ld)$，**同阶**。差别在于：

1. **Ulysses 多搬了 Q**（1.5×），GQA 下差距被放大到 $\frac{3d}{2d_{kv}}$ 倍（Llama-3 8B 的 $H_{kv}/H=1/4$ → 6×）
2. 拓扑：all-to-all 要全互联 vs ring 只要邻居 → 跨机时差距是数量级的
3. 显存：ring 是唯一真正省 KV 显存的

</details>

<details>
<summary><b>5. 混合方案与工程实现索引（点开看）</b></summary>

**混合（推荐用于 128K 以上）**：先把 CP 组拆成 $N = N_1 \times N_2$，机内用 Ulysses（$N_1 \le$ 机内卡数）、机间用 ring，即 **a2a + p2p**。Megatron-Core 的 `--cp-comm-type` 里就是 `a2a+p2p` 这一档（另有 `allgather` / `p2p` / `a2a`），对应 USP、LoongTrain 的 double-ring 等思路。

**参考实现**：
- Ulysses — *Sequence Parallelism: Long Sequence Training from System Perspective* (arXiv 2305.13245)，DeepSpeed 集成
- Ring Attention — *Ring Attention with Blockwise Transformers for Near-Infinite Context* (arXiv 2310.01889)；Striped Attention (arXiv 2311.09431) 解决 causal 均衡
- USP (arXiv 2405.07719) — 统一视角 + 混合
- Megatron-Core `context_parallel/`
- `ring-flash-attention` 等开源 kernel

**必踩的工程细节**：
1. **loss 归约**：CE loss 跨序列切分，需要跨 CP 组 all-reduce（分子分母要一起，否则梯度错）
2. **数据加载**：同一 batch 的 CP 组内必须拿到**同一句**的连续切分，不能各自随机取样（等价于"同一 DP rank"语义）
3. **RNG**：dropout 之类需要保证 CP 组内对应 token 的随机流正确（通常按 rank 各自不同种子 + 位置对齐）
4. **zigzag 索引**：all-gather / ring 的 causal mask 边界需要按重排后的全局 token index 计算，varlen 打包时尤其容易出错
5. **梯度/参数同步不受影响**：CP 只切激活，参数仍由 DP/TP 管；但 CP 会**挤占 DP 度**，进而放大梯度 all-reduce 的单轮大小
6. **别和 TP 的 "Sequence Parallel" 混淆**：Megatron 的 SP 只切 LayerNorm/dropout 等激活，attention 前要 all-gather 回全长 —— 它**不省 attention 显存**，不是 CP

</details>

---

## 6. 怎么选（决策树）

```
需要 CP 吗？ → 序列长 / 单卡激活爆了 / 想增大 micro-batch 的 token 数
│
├─ 只是想让激活变省、L 不算极端（≤32K）、模型是 GQA、mask 很花（varlen/文档 mask）
│   └─► All-Gather CP。最稳、零 kernel 改造，KV 成本被 GQA 摊薄
│
├─ 单机内、head 数够（H_kv ≥ N，如 H=32/64）、想要最低延迟和最简单负载均衡
│   └─► Ulysses。2 次 all-to-all，causal 天然均衡，NVLink 上效率最高
│
├─ CP 度 > 8、要跨机、或 MQA / H_kv 很小、或 L 到 64K~1M
│   └─► Ring Attention（配 zigzag/striped）
│
└─ 大规模长上下文训练（128K+，几十~几百卡）
    └─► 混合 a2a + p2p（机内 Ulysses + 机间 ring），Megatron `a2a+p2p`
```

**几条经验判据**：

- **看瓶颈在哪**：显存卡住 → 选 ring；通信延迟卡住（NVLink 内、CP 小）→ 选 Ulysses；实现成本卡住 → 选 all-gather
- **看模型结构**：GQA/MQA 模型优先 all-gather 或 ring，Ulysses 会因 $H_{kv}$ 小而失效甚至退化
- **看并行度**：$N$ 必须整除 $H$ 才能用 Ulysses，这条常常一票否决
- **别只看通信量**：三者量级相同，胜负手是 **显存、拓扑适配、overlap 能力**。很多实测里 all-gather 在单机 8 卡、中短序列、GQA 模型上并不比 ring 慢，而且省下的调试成本很可观

---

## 7. 一句话记忆

> **All-Gather = 把 KV 复制出去；Ulysses = 把序列换成 head 并行；Ring = 让 KV 排队流过。** FLOPs 相同，通信量同阶；分发方式不同，主要是显存和拓扑的权衡。

---
## 🧑‍💻 User

cp的话 一开始每张rk上都是完整序列吧，然后再切序列唯独分发。k，v cache也是同理的，拿到完整的 再切分这样
然后推理出一个token 就接在后面存 kv cache
具体是怎么做的
我记得cp好像每个rank并不持有完整序列
kv cache是不是分片存储
或者只存在一张卡上全量 然后通过通信传（？

---
## 😀 默认助手 (7)

# CP 里到底谁持有完整序列/KV

先直接回答你的三个疑问，再展开。

| 你的判断 | 实际情况 |
|---|---|
| "一开始每张卡都是完整序列，再切分发" | **数据在 host 上是冗余的同一份**（CP 组必须看到同一句），但**进 GPU 前就切成 $L/N$**，显存里从不出现全序列；而且常常是 **zigzag 两块不连续**，不是简单切片 |
| "拿到完整的再切" | 只有 **metadata**（position id、全局 token 偏移、mask 索引）是"全局"的，激活从来不是 |
| "KV cache 分片存储吗" | **训练没有 KV cache**；**推理有，且确实分片**——按序列切（CP）或按 head 切（TP），取决于并行方式 |
| "只存一张卡全量然后通信传" | 不是主流。个别 master-KV/broadcast 设计存在，但那张卡立刻成为容量+带宽双瓶颈 |

---

## 一、训练：数据从磁盘到显存的真实路径

关键点：**CP 组内所有 rank 是"同一个数据副本"**，它不是数据并行。

```
world = TP × PP × CP × DP
        └── CP 组内共享同一句数据 ──┘
```

数据流：

1. **host 侧**：dataloader 在 CP 组每个 rank 上都产出**完整 batch**（冗余读，浪费一点 IO/内存带宽，换来零通信；也有实现改成 rank0 读 + broadcast）
2. **进 GPU 前切分**：`get_batch_on_this_cp_rank()` 这类函数按 CP rank 取出自己那份，**只把 $L/N$ 个 token 的 tensor 搬上卡**
3. **更常见是 zigzag**：为了 causal 负载均衡，把序列切成 $2N$ 块，rank $i$ 取**第 $i$ 块 + 第 $2N-1-i$ 块**，并做一次 roll 让二者在本地看起来是连续相对顺序

$$\text{rank } i \;\leftarrow\; \text{chunk}_i \;\Vert\; \text{chunk}_{2N-1-i}$$

所以 rank 上拿到的既不是全长，**也不是一整段连续区间**，是两段。

4. **全局信息靠本地计算**：position id 用全局偏移 $i \cdot L/N$ 现算，attention mask 用全局 index 现算——这些是 $O(L)$ 级别的整数元信息，不是激活，代价可忽略。
5. **loss 要跨 CP 归约**：CE 的分子分母都在各自分片上算，最后 all-reduce，否则梯度错。

> 所以你记的"每个 rank 不持有完整序列"——**对，而且对得很彻底**：激活层面从头到尾只有 $L/N$。

---

## 二、训练：attention 里的"全量"是瞬态的

attention 是唯一的例外，因为每个 query 必须看见全部 KV。三种方案的区别，本质是**"全量"在谁身上、以什么形式、存在多久**：

| | 全序列出现在哪 | 形态 | 生命周期 |
|---|---|---|---|
| **All-Gather CP** | **每张卡**的 KV 显存里 | gather 出全长 K、V | 本层 attention 前 gather，算完即放（backward 要么存住、要么再 gather） |
| **Ulysses** | **每张卡**的 Q/K/V 显存里，但只有 $H/N$ 个头 | all-to-all 换布局 | 本层 attention 内，算完 all-to-all 换回 |
| **Ring** | **任何卡上都不出现** | KV 块沿环流过，只占一块 buffer | 每步用完即扔，peak 就是 $L/N$ 那一块 |

这直接对应上一轮的显存结论：

- all-gather / Ulysses：峰值含 $O(L d_{kv})$ 或 $O(Ld/N)$
- ring：峰值只有 $O(L d_{kv}/N)$ —— **唯一真正"没见过全量"的方案**

一句话：**CP 不是"把完整序列摊到多卡上算"，而是"每卡只拿一段，谁需要全量谁临时凑出来，凑完就扔"。**

---

## 三、训练里根本不存在 KV cache

这点值得单独强调，因为概念容易串：

- **KV cache 是推理专属**：因为 decode 每步只有 1 个新 token，但要用前面所有 token 的 K/V，所以必须存下来复用。
- **训练时**：一次 forward 就把 $L$ 个 token 全算完，K、V 就是本层 attention 的中间激活，算完即弃；反向靠**重算**（activation recomputation）而不是缓存。
- 所以训练里对应的问题不是"KV cache 怎么放"，而是"**activation memory 怎么放**"——CP 在这里的作用是把所有逐 token 激活（attention 输入输出、MLP 中间）统一切成 $1/N$。

---

## 四、推理：KV cache 确实是分片存的，但要先分清哪个轴

这是你问题的核心。**三种并行切三个不同的维度**：

| 并行 | 切什么 | 每卡 KV cache 形状 | 全序列？ |
|---|---|---|---|
| **TP** | 切 **head** | $(L,\; H/N,\; d_h)$ | 序列是**全的**，头是 $1/N$ |
| **CP** | 切 **序列** | $(L/N,\; H,\; d_h)$ | 头是**全的**，序列是 $1/N$ |
| **PP** | 切 **层** | 只存自己那几层 | — |
| **DP** | 复制 | 完整 | 各副本独立 |

所以"KV cache 分片吗"——**取决于你用哪个轴，两个轴都会分片，但分法完全不同**。CP 的 KV cache 是**按序列位置分片**：rank $r$ 只负责 token 区间 $[rL/N, (r+1)L/N)$ 的 K/V，头是全的。

### decode 一步（CP 版）具体怎么算

每个 rank 手里只有自己那段 KV，只能算出**部分注意力**，然后跨卡合并（这就是 FlashDecoding 的 split-KV 思路）：

对每个 head，rank $r$ 在本地分片 $S_r$ 上算：

$$m_r=\max_{j\in S_r}s_j,\qquad \ell_r=\sum_{j\in S_r}e^{s_j-m_r},\qquad o_r=\sum_{j\in S_r}e^{s_j-m_r}v_j$$

跨 CP 组归约（$o_r$ 未归一化）：

$$m=\max_r m_r,\qquad \ell=\sum_r \ell_r e^{m_r-m},\qquad o=\frac{1}{\ell}\sum_r o_r e^{m_r-m}$$

**代价**：每层每 token 需要 **2 次 all-reduce**（一次合 $(m,\ell)$ 的二元素向量，一次合 $o$ 的 $d$ 维向量）。$L_{layer}=32$、$N=8$ 时，单 token 就是上百次延迟受限的集合通信 —— 这就是为什么**CP 推理在 decode 阶段基本没人用**。

### 一个很关键的对称性

> **decode 时，Ulysses 退化成 TP。**

因为 Ulysses = "复制序列、切分 head"。decode 序列长度只有 1，**复制的代价 ≈ 0**，剩下的全是"切分 head" —— 那不就是 TP 吗？

所以工程上的分工非常清晰：

- **prefill 阶段**（compute bound，$L$ 很大）→ 值得上 CP：Ulysses（机内）/ ring（跨机）/ 混合，把 $O(L^2)$ 摊开，直接改善 TTFT
- **decode 阶段**（memory-bandwidth bound，每步 1 token）→ 用 TP 切 head 降 KV 带宽和容量；CP 的 softmax merge 通信是纯亏
- **PD 分离**的两条路径正好对应这两个阶段

长上下文推理实际的组合拳通常是：**TP（切 head）+ KV 量化/offload + 稀疏/线性注意力 + prefix cache**，而不是序列维 CP。CP 在推理里主要出现在 **prefill 加速**和少数"超长上下文 + 大 batch"的吞吐场景。

---

## 五、关于"只存一张卡全量，然后通信传"

这个思路的变体确实存在（master-KV / KV-broadcast）：

- 一张卡（或一组）持完整 KV，每步把新 KV 或必要的 KV 段广播给其他卡
- **问题 1：容量**。那张卡的 KV cache 容量 = 全序列，等于 CP 白做，序列长上限被单卡 HBM 锁死
- **问题 2：带宽**。每层每 token 要传的量是 $O(L d_{kv})$ 全量（如果是全广播），而分片方案每卡只需 $O(L d_{kv}/N)$ —— **差 $N$ 倍**，而这恰好是 decode 的瓶颈所在
- 唯一可能合理的场景：**KV 本身极小**（MQA / MLA 压缩后）且卡数不多 —— 但那时通常 TP 或干脆单卡就够，轮不到这个方案

反过来，**ring 的分片传法**才是正解：每卡只把自己那段 KV 往邻居递，全组拼起来覆盖全序列，**任何一张卡都不需要完整 KV，也没有任何一张卡成为瓶颈**。

---

## 六、总结：一张表看清"谁持有什么"

| 阶段 | 完整序列激活 | 完整 KV | 说明 |
|---|---|---|---|
| 训练 · host | ✅ 全量冗余（CP 组内） | — | 只为省通信，是 IO/内存开销 |
| 训练 · GPU 激活 | ❌ 只有 $L/N$（zigzag 两段） | — | 从进卡起就没有全量 |
| 训练 · attention 内 | 瞬态 | 瞬态 | all-gather/Ulysses 临时凑、ring 压根不凑 |
| 训练 · 反向 | ❌ | ❌ | KV 靠重算，不缓存 |
| 推理 · TP | ✅ 全 | ❌ 按 head 切 | KV 容量/带宽 ÷ N |
| 推理 · CP | ⚠️ prefill 有，decode 只有 1 token | ❌ 按序列切 | decode 要 softmax merge，很贵 |
| "master-KV" | — | ✅ 单卡全量 | 容量+带宽双瓶颈，非主流 |

**一句话记住**：CP 是"**激活层面从来没有全量，attention 层面谁需要谁临时凑**"；训练没有 KV cache，推理的 CP 是序列维分片 + softmax 合并，而 decode 阶段这个合并太贵 —— 所以 decode 走 TP 切 head，两者是**正交的两个轴**。

---
## 🧑‍💻 User

我其实一直没搞清SP和CP是什么区别
哪个是RingAttn那套分rank attn计算
然后attn不用特制算子切不了seq吧
sp就是切序列 cp感觉上是指计算注意力的时候 还在做序列并行（？

---
## 😀 默认助手 (7)

# SP vs CP：本质区别

你大概被**命名撞车**坑了。先把名字理清，后面就都通了。

## 0. 两个“SP”是完全不同的东西

| 叫法 | 出处 | 实质 | 和 attention 的关系 |
|---|---|---|---|
| **Sequence Parallel (SP)** | **Megatron-LM**（2022, `--sequence-parallel`） | TP 的配套优化：把 LayerNorm/dropout/残差的激活按序列切 | **attention 里序列是全长**，不切 |
| **Sequence Parallelism** | **DeepSpeed Ulysses 论文标题**（2023, arXiv 2305.13245） | 就是 Ulysses | 序列贯穿 attention 切分 |
| **Context Parallel (CP)** | Megatron-Core（`--context-parallel-size`）、Llama-3 长文训练 | 序列贯穿 attention 切分（含 all-gather / p2p / a2a 三种实现） | 序列贯穿 attention 切分 |

⚠️ **你说的“SP”如果是 Megatron 那个，那它和 CP 毫无关系、也不省 attention；如果是 Ulysses 论文自称的那个 SP，那它就是 CP 的一种。** 后来社区为了躲开撞车，把 Ulysses 改叫 “CP-Ulysses”，这才有 `--context-parallel-size`。

---

## 1. 一句话区别

> **SP：序列切分到 attention 门口就“断”了，attention 前 all-gather 回全长。**
> **CP：序列切分“贯穿”attention 内部，靠重分布/流式让全长 KV 逻辑上可达。**

你说的“**CP 是计算注意力的时候还在做序列并行**”——**完全正确，这就是 CP 的定义**。而 Megatron 的 SP 恰恰相反：它到了 attention 就把序列拼回去。

---

## 2. Megatron SP 到底在干什么

**它本来是为 TP 服务的，不是为长序列。**

TP 的结构性痛点：ColumnParallel / RowParallel 线性层之间，**LayerNorm、dropout、残差 add 这些逐 token 算子在 TP 下是复制的**（激活 `[s, b, h]` 在每张 TP 卡上完全一样），纯浪费显存和带宽。

经典解法：把 TP 的 **all-reduce 拆成 all-gather + reduce-scatter**（Megatron 原论文的 `f` / `g` 算子）。拆开后，all-gather 之后到 reduce-scatter 之前，中间那段激活是 `[s/TP, b, h]`——**序列切分的副产品就这么“免费”掉出来了**。

带 shape 走一遍（TP=8，`s` 序列长，`b` batch，`h` hidden）：

```
LN 输入:        [s/8, b, h]        ← 每卡持不同序列段，hidden 是全的（这就是 SP）
  f = all-gather(seq)              ← 序列拼回全长，hidden 复制
                [s,   b, h]
QKV ColParallel → [s, b, 3h/8]     ← TP 切列，各卡 h/8 个头
Attention:      全长 s，只算自己 H/8 个头    ← ★ 序列没切！
Out RowParallel + g = reduce-scatter(seq)
                [s/8, b, h]
残差 / LN 继续在 [s/8, b, h] 上
```

**关键结论：**

1. **attention 仍然在每张卡上跑全长 $s$**，$O(s^2)$ 的计算和 KV 激活**一点没省**
2. 省的只是 LN/dropout/残差区的激活：`O(sbh)` → `O(sbh/TP)`
3. **通信量不增加**：SP 的 f/g 就是原来那次 TP all-reduce 的另一种实现，总量相同——**它是 TP 的免费搭车，不是新增并行维**
4. **SP degree 被锁死等于 TP degree**，不能独立调，所以它**根本没法用来扩序列长度**

所以：**SP 不解决“序列太长”问题。** 想扩 $L$，必须 CP。

---

## 3. CP 在干什么

CP 的目标恰恰是 SP 碰不到的那两块：

- 摊开 attention 的 **$O(L^2)$ FLOPs**
- 摊开长上下文下 attention 的**激活/KV 显存**（这一块随 $L$ 线性增长，是 128K 训练的真正杀手）

手段就是**让序列切分一直切到 attention 内部**，代价是必须解决“query 看不到别的卡上的 KV”这个问题。

---

## 4. 二维正交表（这张表能一劳永逸）

| 方案 | 激活按 seq 切分？ | **attention 算子按 seq 切分？** | 省 $O(L^2)$ 计算？ | 省长文激活？ | degree 可独立调？ |
|---|---|---|---|---|---|
| 纯 TP | ❌ | ❌（切 head） | ❌ | ❌ | — |
| **Megatron SP** | ✅（仅 LN 区） | ❌ **各卡仍算全序列** | ❌ | ❌ | ❌ 锁 = TP |
| **CP（Ulysses/Ring/AG）** | ✅（全程） | ✅ | ✅ | ✅ | ✅ 自由 |

看第 3 列就够了：**SP 的切分不进入 attention，CP 的切分进入 attention。**

> 补充：**两者不冲突，可以同时开**。Megatron 里 `tensor_model_parallel_size=8, context_parallel_size=8, sequence_parallel=True` 是合法组合，因为作用在**不同 process group** 上：CP 先在序列上切 $N_{cp}$，SP 再在 TP 组内切 $N_{tp}$，叠加使用。

---

## 5. “切了 seq 的 attention 必须用特制算子吧？”

**这个判断只对 1/3。** 分三种情况：

### (a) 特制算子**不需要**：All-Gather CP

attention 前 all-gather 拿到**全长 KV**，然后每卡在自己 $L/N$ 个 query 上跑**标准 FlashAttention**。Kernel 零改造，任意 mask、varlen、ALiBi 全兼容。代价是 KV 显存在每张卡上复制成 $O(L)$。

### (b) 特制算子**也不需要**：Ulysses

all-to-all 换轴，换完变成“**全长序列 × $H/N$ 个头**”，还是标准 FA。也不需要新 kernel。

### (c) 特制算子**必须要**：Ring Attention

KV 沿环流式经过，必须在**单次 kernel launch 内跨块累积 online softmax**（running max / running sum / 重标定），还要处理跨块 mask、反向重算、zigzag 负载均衡。这是真正要写 kernel 的那一个（`ring-flash-attention` 这类）。

**但有一点三选一都逃不掉：通信。** 因为标准 attention 算子要求 query 能看到全部 KV，这不是 kernel 能力问题，是**数据不在同一张卡上**。

### 顺带纠正一个常见错误

**切了 seq 但啥通信都不做** → 每卡只在自己那段 KV 上算 attention → 结果变成**块对角**，等价于把长文档砍成 $N$ 段各自做 attention。**数学上是错的**，这不是 CP，这是“错误实现”（或者你要的本来就是文档内独立 attention + varlen）。

---

## 6. 各框架的叫法对照（防踩坑）

| 你在哪看到的 | 实际指什么 |
|---|---|
| Megatron `--sequence-parallel` | TP 配套的 LN 激活切分，**不是 CP** |
| Megatron `--context-parallel-size` + `--cp-comm-type={allgather,p2p,a2a,a2a+p2p}` | CP，四种实现分别是 all-gather / ring / Ulysses / 混合 |
| DeepSpeed “Sequence Parallelism” / Ulysses | CP |
| 论文里的 “Sequence Parallelism” (2023 后) | 大概率指 CP（Ulysses 那篇把名字占了） |
| 推理框架里的 “sequence parallel decode” | KV cache 按序列分片 + split-softmax 合并，**是 CP 的推理用法** |
| 某些博客的 “SP = 序列并行” | 大概率指 CP，别当成 Megatron 的 SP |

---

## 7. 三句话记住

1. **Megatron SP = TP 的副产品**：把 all-reduce 拆成 all-gather + reduce-scatter，顺手省了 LN 区激活；**attention 里序列是全长，$O(L^2)$ 不变，不能用来扩序列**。
2. **CP = 真正切 attention**：序列切分贯穿 attention，独立并行维，$O(L^2)$ 和长文激活一起摊。
3. **“切 seq 就要特制 kernel”不成立**：只有 Ring 要；all-gather 和 Ulysses 用**标准 FA + 一次集合通信**就能跑（代价分别是 KV 显存复制、head 数受限）。
