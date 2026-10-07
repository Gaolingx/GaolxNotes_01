# 纳米GPT记录速提升

## 🧑‍💻 User

https://github.com/KellerJordan/modded-nanogpt/pull/360
分析这个pr干了什么

---
## 😀 默认助手 (7)

I'll analyze this PR. Let me first discover what tools are available for fetching GitHub content.

Let me inspect the fetch tools and then retrieve the PR content from GitHub.

I have the PR description and discussion. Now let me get the actual changed files and code diff.

Let me get the detailed record README and the key `train_gpt.py` diff to fully understand the changes.

# PR #360 深度分析：ANVIL2 世界纪录

这是 `KellerJordan/modded-nanogpt` 的一次**刷榜记录提交（speedrun record submission）**，而非普通功能 PR。PR 由 `devenpzak` 提交，已被协作者 `ClassicLarry` 合并（2026-09-28），成为该仓库的**第 92 号世界纪录**。

## 一句话总结

> 相比上一记录 #89（1.23 分钟 / 73.9 s），本 PR 在**同一台 8×H100 机器**上把训练时间压到 **39.914 s（0.665 分钟）**，即 **-33.98 s / -46.0% / 1.85×**，同时保持验证集交叉熵 $\le 3.28$（均值 3.27731 ± 0.00099，单侧 $p = 9.4\times10^{-10}$，n=18，全部冷启动）。

---

## 1. 背景：这个仓库在比什么

`modded-nanogpt` 是一个"极限优化竞赛"：在 8×H100 单节点上把一个 GPT-2 124M 级别的模型训练到 FineWeb 验证集 CE $\le 3.28$，**谁用的时间短谁赢**。规则核心是：数据不变、目标不变、同硬件更快。

因此，这里"性能优化"的手段可以非常激进——包括架构改动、kernel 融合、图捕获、甚至超大的 embedding 表。这决定了本 PR 的性质。

---

## 2. 成绩与可信度

| 指标 | 本 PR | 记录 #89（同机、交织测量） | 差值 |
|---|---|---|---|
| 运行次数 | 18 | 9 | |
| 墙钟（训练段） | **39.914 ± 0.120 s** | 73.889 ± 0.137 s | **−33.98 s（−46.0%，1.85×）** |
| 最终验证 CE | **3.27731 ± 0.00099** | 3.27828 ± 0.00205 | |

- 单侧 $t$ 检验对 3.28 门槛：$t=11.5$，$p=9.4\times10^{-10}$（满足 rule 2 的 $p<0.01$）。
- 作者声称被**四个独立来源**复现/审查过。
- 训练步数：1122 计划步 + 52 增长步 + 20 扩展步 = **1194 步**。
- 峰值显存约 53.4 GB allocated / 64.7 GB reserved。

---

## 3. 代码层面改了什么

PR 表面上是 `+457,251 / −2,466，81 个文件`，但**其中约 189,260 行是记录文件夹里的运行日志（evidence）**，不是代码。真正的代码改动约为 **+5,068 / −2,465，9 个文件**（作者原话）。

### 新增文件

| 文件 | 作用 |
|---|---|
| `anvil_attn_kernels.py` | 打包 FP8 QKV 投影、QK-norm/RoPE/pad 融合、混合宽度注意力 |
| `bigram_kernels.py` | hashed n-gram（bigram + trigram）前/反向、稀疏梯度 sink、行压缩 Adam |
| `fuse_tiny_kernels.py` | 优化器 tail 的小 kernel 融合（replicated-Adam 及其动量） |
| `value_embed_op.py` | value-embedding 的 selected-load 反向（自定义 scatter-add） |

### 修改 / 删除

- **`train_gpt.py`**（+3,328 / −662）：主训练器，所有配置固化。
- **`triton_kernels.py`**（+714 / −483）：全 FP8 MLP + 融合 CE kernel，新增 sampled-softmax / prefix-CE 分支。
- **`Dockerfile` / `requirements.txt`**：锁定 `nvidia/cuda:13.1.1` 基镜像 + `torch==2.10.0+cu128`（非 nightly）+ `kernels==0.16.1`，驱动要求 ≥ 580。作者强调**必须用 pinned 版本，nightly 会导致 NaN**。
- **删除 `dc_triton_kernels.py`**（−1,316 行）：dual-chunk attention correction，因对应层被移除。

---

## 4. "34 秒省在哪"——消融表

作者把每个机制**单独移除并重跑**（26 条移除腿 + 7 条锚点腿），用统一汇率 164 ms/millinat（实测区间约 120–250）把验证损失折算成时间：

| 机制 | 净省时间 |
|---|---:|
| **Sampled-softmax 训练损失** | **+8.40 s** |
| **Embeddings（hashed n-gram 表）** | **+7.05 s** |
| **ANVIL 优化器** | **+5.08 s** |
| **Full-stack FP8** | **+4.96 s** |
| **混合宽度注意力** | **+3.82 s** |
| **训练步 CUDA 图捕获** | **+2.99 s** |
| **稀疏梯度通信 & 数据加载器** | **+1.20 s** |
| **MUDDformer 残差混合** | **+0.48 s** |
| **合计** | **+33.98 s** |

---

<details>
<summary><b>5. 各核心机制的技术细节（点击展开）</b></summary>

**① ANVIL 优化器（Averaged Normalized Velocity with Isotropic Lanes）**
是 #89 中 Muon/NorMuon 栈的重写：
- 双轨速度：快轨（计划式 $\beta$，平台 0.93）+ 慢轨（恒定长时域 $\beta$），存于一个 `[2, *chunk]` fp32 状态，混合成 Nesterov-lookahead 速度。
- 六个五次频谱映射构成的级联（Polar Express / Newton-Schulz 家族），系数用 **CEM + minimax** 从头重推（尾部增益 643.13，包络 $[0.9951, 1.00041]$）。
- 逐 lane 能量均衡、符号对齐的 cautious 解耦衰减、通过 uint16 尾数 sidecar 做精确 fp32 提交、以及 bank tail-blend 出货步。

**② Sampled-softmax 训练损失**（+8.40 s，最大单项）
训练阶段只对词表的一个子集算 softmax，而**验证始终是完整 50,304 路 softmax**（与 #89 一致）。配合 prefix-token CE 和融合 CE kernel。

**③ hashed n-gram embedding 表**（+7.05 s，也最具争议）
- 表从 **377,280 行扩到 84,602,880 行（224×）**，维度 768 → 约 **650 亿参数**。
- 关键工程：把表**分片到 8 张 GPU**，每步只 gather 当前 batch token 实际 hash 到的行；梯度以 segment-sum 而非全表张量回传；Adam **只更新被触及的行**。于是优化器开销随"每步用到的行数"缩放，**几乎与表大小无关**。
- bigram（$x_{t-1},x_t$）与 trigram（$x_{t-2},x_{t-1},x_t$）两条通道共享同一张表。
- 索引是 hash：42.3M 行对应 25.3 亿可能 bigram，**每行平均共享约 60 个 token 对**（#89 约 6,700），用大量碰撞阻止记忆化；另有 sign-trick 让碰撞项相互抵消。

**④ Full-stack FP8**（+4.96 s）
MLP 前向**和反向**（dx/dW1/dW2）全 FP8，配合静态无 clip 缩放；Q/K/V 打包量化成可复用的行主序 + 转置 FP8 布局；fp8 lm_head 缓存喂给融合 CE。

**⑤ 混合宽度注意力**（+3.82 s）
Q/K 头宽从 128 降到 **96**，V 保持 128（Q/K 决定路由，V 承载内容）。QK-norm、RoPE、key offset、padding、布局转换融合进**单个 Triton kernel**，QK 流不落地 HBM。

**⑥ 训练步 / 优化器更新 CUDA 图捕获**（+2.99 s）
整个训练步和优化器 tail 用 CUDA graph 捕获，消除 launch 开销。

**⑦ 稀疏通信与数据加载**（+1.20 s）
value-embedding 稀疏梯度交换（+1.13 s）、并行分片加载器 openpack（+0.07 s）。

**⑧ MUDDformer 残差混合**（+0.48 s）
动态稠密连接（10 系数混合）。

</details>

---

## 6. 最大争议：65B 参数的"查表"

PR 讨论区（13 条评论）最激烈的辩论围绕那张巨大的 n-gram 表：

- 有人引用推文称"**用 65B 参数查表训练 124M 模型，最可疑的改动**"；另一位评论者说他"**直接喷了饮料**"，认为若合并会严重损害该仓库的现实意义。
- 质疑点：(a) 是否等价于"提前在 CPU 上训练好 n-gram 模型"（未计时训练）；(b) 是否算硬编码；(c) 表是巨型 lookup table，还有多少是真正的 GPT？

**作者的辩护：**
1. 表**零初始化、100% 在计时内学习**，没有任何预计算，trainer 还断言分片为零并恢复。
2. 硬编码规则针对的是来自以往运行或验证集的值；这里全是 Adam 学习的参数。
3. hash 有 ~60:1 的碰撞，"**是压缩不是存储**"，无法像 infini-gram 那样精确检索。
4. 该机制**并非原创**——bigram hash 表自 27 个记录前就存在，被有意批准过 4 次；作者只是"加宽了行数并增加了 trigram 通道"。
5. 即使整张表被判非法，消融显示它**只值 7 秒**，本 PR 仍比 #89 快约 27 秒。

**最终裁决：** 仓库所有者 `KellerJordan` 表态"**我认为这里描述的所有改动 100% 合法**"；协作者 `ClassicLarry` 也认可，并最终合并。

---

## 7. 结论与评价

| 维度 | 判断 |
|---|---|
| **性能** | 显著且经充分校准（18 次运行、同机交织基线、$p=9.4\times10^{-10}$），是目前最激进的一次提速 |
| **工程** | 分片稀疏 n-gram 表、图捕获、融合 FP8 kernel、打包 QKV 等，都是高质量的 kernel/系统级优化 |
| **科学性** | 消融表完整、不丢腿，但"把 val 损失折算成时间"用了统一汇率，属于归一化约定而非严格可加 |
| **争议** | 超大 embedding 表在规则边缘，虽被维护者判定合法，但确实拉低了该仓库的"现实可迁移性" |
| **元层面** | 这是一份带日志、统计脚本、复现指南的"记录提交"，其中真正有价值、可迁移的可能是 **ANVIL 优化器**与 FP8/图捕获等系统优化 |

> 作者自己在讨论中的总结很中肯：这个挑战的价值不只在于"整份配置能直接用于真实 LLM 训练"，更在于像 **Muon、ANVIL** 这类会外溢到前沿训练的优化。**即使完全去掉那张表，本 PR 仍比 #89 快约 27 秒**，主要来自可复用的真实创新。
