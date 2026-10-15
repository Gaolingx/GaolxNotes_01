# MOPD显存爆炸解决方案与集群规模评估

## 🧑‍💻 User

如何解决mopd显存占用爆炸问题，假如1个模型1t参数，10个领域模型就相当于10t参数模型参与训练吗，要什么规模的万卡集群才能做这种后训练

---
## 😀 默认助手 (7)

# MOPD 显存爆炸：先把概念对齐

你问的 **MOPD = Multi-Teacher On-Policy Distillation**（多教师在线策略蒸馏）。范式是：先让一个基座模型在不同领域各自做 RL，得到 $K$ 个"领域专家"教师；然后让**一个学生**自己 rollout，由对应领域的冻结教师对**它生成的每个 token** 给密集 logprob 奖励，学生用 PPO/反向 KL 更新。

关键结构事实（决定了后面所有结论）：

> **教师不生成，只打分。学生只有一份，教师是奖励函数。**

所以我先把最关键的一个误解拆掉，再讲显存。

---

## 一、"10 个 1T 教师 = 10T 模型参与训练"吗？

**不是。这个直觉在三个轴上同时是错的，而且错的方向不一样。**

### 1）显存轴：训练态 vs 只读态，每参数成本差 8–16 倍

| 角色 | 需要保存的东西 | Bytes/param |
|---|---|---:|
| **学生**（bf16 混合精度 + AdamW） | bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + fp32 $m$ 4 + fp32 $v$ 4 | **16** |
| 学生（只算优化器分片后的净存储） | 同上，只是被切得更碎 | 16（总量不变） |
| **教师**（推理，bf16） | 只有权重 | **2** |
| **教师**（推理，FP8） | 只有权重 | **1** |
| **教师**（推理，NVFP4） | 只有权重 | **0.5** |

于是：

$$\text{朴素理解}:\; 11\text{T} \times 16\,\text{B} = 176\ \text{TB}$$
$$\text{正确算法}:\; \underbrace{1\text{T}\times16\text{B}}_{16\ \text{TB 学生}} + \underbrace{10\text{T}\times1\text{B}}_{10\ \text{TB 教师 FP8}} = 26\ \text{TB}$$

**已经小了 6.8 倍**。再叠加"同基座 + delta"（下面第二节）可到约 17 TB，**小 10 倍以上**。

原因很简单：教师**没有梯度、没有优化器动量、没有 master weight**。它们只被 forward 一次。

### 2）计算轴：按 token 算，routed MOPD 只比普通 RL 贵 25%

设学生/教师激活参数量都是 $N_{\text{act}}$（同基座同架构，这是 MOPD 的标准设定）。每个 rollout token 的 FLOPs：

| 方案 | 每 token FLOPs | 相对普通 RL |
|---|---|---|
| 普通 RL（GRPO，无教师） | $2N + 6N = 8N$ | $1.00\times$ |
| **硬路由 MOPD**（1 token 只给 1 个教师） | $2N + 6N + 2N = 10N$ | $\mathbf{1.25\times}$ |
| **共识 MOPD / C-MOPD**（10 个教师都打分） | $2N + 6N + 20N = 28N$ | $\mathbf{3.5\times}$ |
| 真的 10T dense 模型训一步 | $6 \times 10\text{T} = 6\times10^{13}$ | $\approx \mathbf{150\times}$ |

取 $N_{\text{act}} = 40\text{B}$（1T 总参 / MoE，参考 Kimi-K2 的 1T/32B、DeepSeek-V3 的 671B/37B）：

- 路由 MOPD：$4.0\times10^{11}$ FLOP/token
- 共识 MOPD：$1.12\times10^{12}$ FLOP/token
- "10T 模型"：$6.0\times10^{13}$ FLOP/token

**"10T 模型参与训练"这个直觉高估了约两个数量级。** 只有在"10 个教师对每个 token 都打分"（共识/ensemble）时，才有 3.5× 的真实成本，而且那也只是 3.5×，不是 10×。

### 3）优化轴：容量还是 1T，难点是信用分配

所有 MOPD 论文（Open-MOPD / DN-MOPD / UP-MOPD / PMOPD / $\Delta$-MOPD / SF-MOPD）解决的都**不是容量问题**，而是**同一个 1T 学生内部跨领域更新预算的分配与冲突**：

- Open-MOPD：prompt 占比 39.8/39.8/20.3%，但**梯度 token 占比 49.7/49.3/0.99%**——因为 Math/Code 平均 10,500 token，IF 只有 409 token。
- DN-MOPD：IF 的 reward 方差比 Math 大好几倍，导致它主导更新。
- 这些都是 $1\text{T}$ 维空间里的**加权/投影**问题。

**所以：不存在"10T 的 forward"，不存在"10T 的梯度"，不存在"10T 的优化器状态"。** 存在的是"1 份可训练的 1T + 10 份只读的 1T 打分服务"。

---

## 二、显存到底炸在哪：9 个真实来源

<details>
<summary><b>展开：按"最容易被踩"排序的失败模式清单</b></summary>

| # | 失败模式 | 爆炸倍率直觉 | 修法 |
|---|---|---|---|
| 1 | **教师按 DP rank 复制**（每个 trainer replica 各存一份教师做低延迟打分） | $\times$ DP 数，可到 $\times64$ = **640 TB** | 教师做成**共享推理服务**，请求队列，绝不 per-replica 复制 |
| 2 | **把 11 个模型塞进同一 ZeRO-3 进程组**（launch 脚本直接加 `--teacher` 10 次） | 所有 all-gather/通信桶全放大 11× | 训练池 / 打分池**分离部署**（disaggregation） |
| 3 | **存全词表教师 logits** | $10^6\ \text{token} \times 150\text{k} \times 4\text{B} = 600$ GB / 每 $10^6$ token | 只存 **top-$k$**（Open-MOPD 用 top-16，省 152×），或只存采样 token |
| 4 | 11 份 bf16/fp32 教师权重常驻 | bf16 20 TB / fp32 **40 TB** | FP8（10 TB）；Blackwell 上 NVFP4（5 TB） |
| 5 | 教师在 step 0 全部加载，但硬路由下一批只用得到 1 个 | 峰值 10× 浪费 | 按 domain batch **懒加载 + 驱逐** |
| 6 | **教师做自回归 rollout**（拿教师生成 SFT 数据，而不是打分） | KV cache 随 seq len 线性增长 | 教师只做**单次 prefill 打分**，KV 用完即弃 |
| 7 | 每步 checkpoint 11 个模型 | 每步 $\sim$22 TB 落盘 | 只存学生；教师冻结、存一次 |
| 8 | reward staleness 逼你开大 batch（Open-MOPD：为估 token share 需要大混合 batch，再切成 $K$ 个 inner update；$K=4$ 时 rollout→current KL 0.059，$K=32$ 时 0.216，clip 比例到 0.86） | 大 batch → KV + 激活爆 | reward refresh（$K{=}4$ 就够）；PowerOPD 有界奖励 |
| 9 | 10k+ CoT token 的激活 | 与 $\text{seq len}\times\text{batch}$ 成正比 | 梯度重计算、SP/CP、MLA、FP8 KV |

</details>

---

## 三、12 个解法，按杠杆从高到低

### 🥇 第 0 档：先问要不要做

这一档最容易被人忽略，但证据最硬：

- **权重合并**：Open-MOPD 自己的表里，**ParamMerge-TA 用几分钟 CPU 时间拿到 71.7% 的能力恢复**，而 Open-MOPD 是 83.4%（满分 100% 是 RouteRL）。差距 11.7 个点，换 15–23× 的 GPU 时间。
- **Rethinking Self-Distillation for Multi-Teacher Capability Merging** 直接测了：把 SFT 基线用 rejection-sampled 教师轨迹重新调，四种方法（SFT / soft-label KD / teacher-prefix / MOPD）**精度几乎相同**，但 **MOPD 用了 SFT 的 14.8–23.1× GPU-hours**。

> **如果你的瓶颈是显存和集群，最有效的"解法"是：先用 Task-Arithmetic 合并试一版，再用调好的 off-policy 基线试一版。** 只有在能证明 MOPD 的增益来自算法本身（而不是超参）时，才值得付这个代价。

另外还有一路是**自蒸馏**：OPSD（单模型同时扮演教师/学生，教师角色只需额外获得"已验证推理轨迹"的上下文）报告 **峰值显存降 40–60%**，因为根本不存在第二个模型。

### 🥈 第 1 档：共享基座 + delta 分解（显存降 10–20×）

MOPD 的标准前提就是"**多个专家从同一 base checkpoint 出发**"。那就不该存 11 份完整权重：

$$\underbrace{1\ \text{份 base}}_{\text{FP8} = 1\ \text{TB}} + \sum_{i=1}^{10}\delta_i$$

- $\delta_i$ 若是 LoRA rank-64 级别：每个 $\delta_i \approx 5\!-\!20$ B 参数 → bf16 下 10–40 GB，10 个合计 **50–400 GB**。
- 即使 $\delta_i$ 是全秩 dense，FP8 存 delta 也只是 10 × 1 TB。

**关键**：这不只是压缩技巧。$\Delta$-MOPD（arXiv 2610.10460）证明，直接蒸馏教师的**端点策略**会被"继承自 base 的偏好"污染（inherited base pull 可以超过 post-training 的 shift），正确的监督对象就是 **teacher-minus-base 的 logit shift**。也就是说 **delta 本身就是算法需要的对象**，共享基座是"算法正确性"和"显存"双赢。

### 🥉 第 2 档：时间复用（phase separation）—— 让峰值 = max 而非 sum

Open-MOPD 的算法流程天然支持这个：

```text
每个 step:
  ① 学生 rollout（学生权重常驻）
  ② 硬路由 → 每个 domain 的样本交给对应教师 → 教师对 top-16 token 打分
  ③ 【缓存教师 logprob，释放教师显存】      ← 关键
  ④ K=4 次 PPO inner update（只需要学生 + 缓存）
```

因为教师 logprob **每 rollout batch 只算一次**（reward refresh 只重算学生那一侧），所以：

$$\text{峰值显存} = \max(\text{学生训练态},\ \text{教师推理态}) \quad \text{而不是} \quad \text{二者之和}$$

再叠加**硬路由**：一批里只有 1 个教师被用到 → 可以把 10 个教师**按 domain 分时切片**，任何时候只有 1 个常驻。

### 第 3 档：量化教师

教师只需产出 logprob，对量化远比训练宽容：

| 精度 | 1T 教师 | 10 个教师 |
|---|---:|---:|
| bf16 | 2 TB | 20 TB |
| **FP8** | 1 TB | **10 TB** |
| **NVFP4**（Blackwell） | 0.5 TB | **5 TB** |

### 第 4 档：缩小监督载荷

- **top-$k$ logprob 缓存**（Open-MOPD: $k=16$）→ 中间缓冲降 152×（注意这只省 buffer，不省主干 FLOPs，词表投影本来就只占 $\sim2.4\%$）。
- **只保留方向而不是数值**：*Do We Really Need KL Divergence for OPD?*（arXiv 2609.33791）发现只需在"教师-学生分歧大"的**一小部分 token** 上把更新方向推向教师即可（$\pm1$），据此提出 C-MOPD。→ 教师只需覆盖**稀疏位置子集**。
- **验证器式**：SAKI（arXiv 2609.36601）用最大耦合做教师引导 rollout，教师只在"纠正位置"上给直接监督，配 **engine-resident speculative verifier**，matched-workload 吞吐 $\mathbf{4.22\times}$。
- **有界奖励**：PowerOPD（arXiv 2606.17199）用 Box-Cox 变换把 log-ratio 奖励变成有界、符号一致的奖励，报告**峰值显存 $-23.1\%$、wall-clock $-59.2\%$**。
- **半在线**：SODA 用一次性学生快照替代动态 rollout，**快 10×、峰值显存 $-27\%$**。

### 第 5 档：offload / streaming / 通信

- 教师权重是**只读**的 → 天然适合 layer-wise 流式加载（ZeRO-Inference / FlexGen 式）。10 TB 放不进单机 CPU DRAM，但"每个教师 1 TB 分片到一组机器"完全可行。
- **KV cache 只在学生 rollout 阶段存在**（教师是单次 prefill，不增长），这是 MOPD 相比"教师自回归生成"的巨大优势。学生侧再用 MLA/GQA + FP8 KV + paged attention + prefix caching。
- 尾部长样本治理：TailSieve（arXiv 2608.22788）用 partial rollout 做 tail routing，uniform routing 上 $\mathbf{1.67\times}$；配 MTP/DFlash 推测解码到 $\mathbf{2.79\text{--}2.59\times}$。
- **双模型同实例**：IPD（arXiv 2609.36546）专门做了"co-hosts both models in one serving instance with **separate KV caches**"的融合推理引擎，且只用 $\sim1/4$ 的样本和步数就超过 OPD 训一个 epoch。

---

## 四、集群规模：把"容量"和"吞吐"分开算

这是你第三个问题的核心。**很多人把两者混了，于是得出"必须万卡"的错误结论。**

### 4.1 容量：塞得下需要多少卡？

假设 1T 学生 + 10 × 1T 教师（FP8），80 GB 卡：

| 卡数 | 聚合 HBM | 学生态/卡（16 TB/N） | 10 教师 FP8/卡（10 TB/N） | 剩余给激活/KV | 结论 |
|---:|---:|---:|---:|---:|---|
| 128 | 10.2 TB | 125 GB | — | — | ❌ 塞不下 |
| 256 | 20.5 TB | 62 GB | 39 GB | 超了 | ❌ |
| 512 | 41 TB | 31 GB | 20 GB | 51 GB | ⚠️ 勉强可用 |
| **1024** | **82 TB** | **16 GB** | **10 GB** | **74 GB** | ✅ 舒适 |
| 2048 | 164 TB | 8 GB | 5 GB | 84 GB | ✅ 很宽裕 |

但注意：**上表假设把 10 个教师完美切分到全部 N 张卡上——这是一个糟糕的通信模式**（每次教师 forward 都变成跨千卡 all-reduce）。

现实做法是**分离池**：

```text
训练池（学生）:  512 – 1024 × 80GB
教师打分池:      每个 1T 教师 FP8 = 1 TB → 至少 14 张卡，实用 16–32 张
                 10 教师 × 16 = 160 张；× 32 = 320 张
--------------------------------------------------
合计 ≈ 700 – 1,400 GPU
```

**结论：把 1T 学生 + 10 × 1T 教师跑起来，1–2 千卡就够，不需要万卡。**

### 4.2 吞吐：什么情况下真的需要万卡？

| 场景 | 是否需要万卡 | 说明 |
|---|---|---|
| **训练 10 个领域教师本身** | ✅ **是** | 每个教师都是一次完整的 1T RL 运行。10 个并行 ≈ 5,120–10,240 卡。参考 RollArt：在阿里集群上用 **>3,000 GPU** 训几百 B 的 MoE |
| 蒸馏阶段（路由式） | ❌ 不需要 | 1–2k 卡，1–3 天 |
| 蒸馏阶段（共识式 C-MOPD） | ⚠️ 建议 | 教师算力 $\times2.8$，约 3–6 天 |
| 要在 1 周内完成全流程 | ✅ 是 | 这是排期压力，不是容量压力 |
| 同时还在做预训练/其他阶段 | ✅ 是 | 集群复用 |

**所以"万卡"的真实来源是：你要跑 10 次 1T 的 RL 后训练。** 把 10 个教师并行摊到 5–10k 卡上，1 周能出；串行在 1,024 卡上跑，就是约 2 个月。

### 4.3 一个可落地的参考方案

```text
Stage A  训 10 个教师（真·万卡阶段）
  每个: 1T MoE + DAPO/GRPO, ~1–3e10 tokens
  单卡组: 512–1024 GPU × 3–7 天（rollout 是带宽瓶颈，MFU 通常 <10%）
  10 教师并行 → 5–10k GPU × ~1 周

Stage B  MOPD 蒸馏（1–2k 卡阶段）
  训练池: 512–1024 GPU, Megatron/FSDP + ZeRO-3 + 梯度重计算
  打分池: 64–320 GPU, vLLM/SGLang, 每教师一个 endpoint
  机制:  硬路由 · top-16 logprob 缓存 · 教师每 batch 打分一次后驱逐
         K=4 inner update + reward refresh · token-share/gap 平衡
  时长:  1–3 天

Stage C  评测
```

工程栈参考：**Open-MOPD 是基于 patch 过的 verl 的**（`training/verl/`），verl 本身支持 hybrid engine / colocate / disaggregated rollout，和教师池分离的架构天然契合；1T 学生主干用 Megatron-LM 的 EP + SP/CP；教师和学生 rollout 用 vLLM/SGLang。

### 4.4 硬件选择上的两个实际点

1. **EP all-to-all 带宽决定一切**。1T MoE 需要 EP=64/128 跨节点。H100 HGX（NVLink 900 GB/s 节点内）vs **B200 NVL72（1.8 TB/s，72 卡一个 NVLink 域）**。后者可以直接把**一个完整的 1T FP8 教师塞进一个 NVLink 域**，把"共享服务"做成零跨域通信——这对 MOPD 是结构性优势。
2. **容量别按 nameplate 算**。80 GB 卡实际可用 76–80 GB，且要留通信桶、碎片、重计算 buffer。上表已经按 $N$ 理想切分，实际留 20–30% 余量。

---

## 五、一句话总结

| 你的问题 | 答案 |
|---|---|
| 10 个 1T 教师 = 10T 模型参与训练？ | **不是。** 显存是 $1\text{T}\times16\text{B} + 10\text{T}\times(0.5\!-\!1)\text{B} = 16\!-\!26$ TB（不是 176 TB）；FLOPs 路由式是普通 RL 的 **1.25×**，共识式是 **3.5×**；容量只有 1T，优化对象也只有 1T |
| 显存爆炸怎么解？ | ①先考虑不做（合并/SFT 便宜 15–23×，恢复 71.7%）②**共享基座 + delta**（$\Delta$-MOPD 让 delta 成为算法对象，$10\!-\!20\times$）③**训练池/打分池分离**，教师是只读无状态服务，绝不 per-DP-replica 复制 ④**打分一次 → 缓存 top-$k$ → 驱逐教师**，峰值 = max 而非 sum ⑤FP8/FP4 教师 ⑥方向而非数值（KL-free/SAKI）⑦KV 用 MLA+FP8 |
| 要什么规模的集群？ | **容量上 1–2 千卡就够**（训练池 512–1024 + 打分池 64–320）。**万卡是因为你要训 10 个 1T 教师**，或者要在 1 周内交付，或者你选了共识式蒸馏 |

---

### ⚠️ 别把这些数字当成硬结论

需要诚实说明几点：

1. **上述 FLOPs/显存是我基于你给的"1T × 10"假设做的估算**，不是从任何一篇论文里抄的。论文里的实验规模是 3B（Open-MOPD/SmolLM3）、7B–8B（PMOPD/UP-MOPD）、Qwen3.5 多尺寸（DN-MOPD），外推到 1T 是我自己的推断。
2. **显存爆炸的实证数字**来自：PowerOPD（$-23.1\%$ 峰值显存）、OPSD 综述（$-40\text{–}60\%$）、SODA（$-27\%$、快 10×）——这些是 0.6B–7B 量级的测量。
3. **MOPD 是否值得做**本身在 2026 年是有争议的（Rethinking Self-Distillation 那篇直接质疑），如果你正在做技术选型，我建议把权重合并和调好的 off-policy 基线作为第一梯队方案先跑掉。
4. 这一领域 2026 年下半年论文极其密集（$\Delta$-MOPD、UP-MOPD、PMOPD、SF-MOPD、Latent-MOPD、SAKI、IPD、TailSieve 都在近两个月），**具体数字几个月内就会过时**，最好按结构（角色 × bytes/param、角色 × FLOPs/token）来思考，而不是记数字。
