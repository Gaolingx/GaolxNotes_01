# LLM 后训练蒸馏 OPD OPSD MOPD 区别

## 🧑‍💻 User

llm后训练蒸馏对齐teacher中，opd mopd opsd有什么区别，哪个训练成本最高

---
## 🧑‍💻 User

llm后训练蒸馏对齐teacher中，opd mopd opsd有什么区别

---
## 😀 默认助手 (7)

# OPD / MOPD / OPSD 的区别与训练成本

先给结论：**三者其实是同一个"on-policy 蒸馏"框架的三个变体**——OPD 是基础范式，MOPD 是"多个 teacher 版本"，OPSD 是"teacher 换成自己"的版本。**训练成本通常是 `MOPD > OPD > OPSD`**，MOPD 最高（有论文实测是 SFT 的 $14.8\sim23.1\times$ GPU-hours）。

> 说明：以下内容基于我检索到的 arXiv 文献（该领域 2026 年前后论文密集，名词已基本固化）。这些是研究社区口径，不是单一权威标准，不同实现会有偏差。

---

## 一、三者定义与核心区别

| | 全称 | teacher 是谁 | 需要几个模型 | 核心信号 |
|---|---|---|---|---|
| **OPD** | On-Policy Distillation | 一个**外部**更强的模型 | student + 1 teacher | student 自己 rollout 上，teacher 逐 token 给 dense 监督（通常 reverse KL） |
| **MOPD** | Multi-Teacher On-Policy Distillation | **K 个领域专家**（各自 RL/SFT 训出来的） | student + K teachers | 按 domain 路由，把多个专家的能力合并进一个 student |
| **OPSD** | On-Policy Self-Distillation | **模型自己**，但带着 privileged 信息（参考答案/计划/环境反馈） | 只有 1 个模型（可能配 frozen/EMA 副本） | 同一网络：带答案时当 teacher，不带答案时当 student |

一句话记忆：

- **OPD** = "student 生成，teacher 打分"（原始范式）
- **MOPD** = OPD + **多 teacher 能力合并**（routing / 冲突消解是主要难点）
- **OPSD** = OPD + **teacher 换成更"知情"的自己**（不需要第二个模型）

<details>
<summary>展开：每个方法的机制要点与代表工作</summary>

### OPD（On-Policy Distillation）
- 动机：传统离线的静态模仿（SFT teacher 文本）有 exposure bias，长推理链误差按序列长度平方级累积；OPD 让 teacher 对 **student 实际产生的轨迹** 给反馈，把累积项压到近似线性。
- 目标：本质是 student-sampled 轨迹上的 $f$-divergence 最小化，一般用 reverse KL。
- 已观察到的问题：可能与 RLVR 互补（dense 但 outcome-agnostic）；会出现 reward hacking 式的"越训越长/重复"崩塌；对 teacher 的"评估学生 rollout 的能力"比"自己解题能力"更敏感。
- 代表：综述 `arXiv:2604.00626`、公式化综述 `arXiv:2606.22793`、`arXiv:2609.04172`（1 条 query 就能恢复大部分增益，"data-overfed but algorithm-starved"）。

### MOPD（Multi-Teacher On-Policy Distillation）
- 动机：RL 能把一个 base 训成多个专家（math / code / IF / 领域），但用户要一个模型全都会——MOPD 把专家蒸馏融合进一个 student。
- 关键设计轴：
  - **teacher 路由**：prompt 级 hard-routing → token 级 routing（`arXiv:2609.30837` MOPD-Router、`arXiv:2609.23697` TrustMOPD、`arXiv:2609.33791` C-MOPD 让所有 teacher 都监督）；
  - **信号强度归一化**：各 domain 反馈方差不同会导致某些域主导更新（`arXiv:2609.35347` DN-MOPD）；
  - **参数冲突**：capability seesaw / 跨域 interference，用子空间投影、update projection、slow-fast EMA 等手段（`arXiv:2609.34605` PMOPD、`arXiv:2610.08398` UP-MOPD、`arXiv:2610.02324` SF-MOPD）；
  - **teacher signal 构造**：传 endpoint 分布 vs 传 teacher−base 的 logit shift（`arXiv:2610.10460` Δ-MOPD）。
- 代表应用：Video-MOPD、ChemOPD、Qwen-Audio M²-OPD、ME-VLM、SWE agent 融合等。

### OPSD（On-Policy Self-Distillation）
- 动机：OPD 需要**第二个更大的模型**当 teacher，这是主要成本；OPSD 直接去掉它——teacher 就是模型自己，只是被喂了 privileged context（参考答案、hint、plan、verifier feedback、环境反馈）。
- 关键区别：teacher **不比 student 强，只是"更知情"**（`arXiv:2608.25936` 综述原话："The teacher is no stronger than the student, only better informed"）。
- 主要失效模式：**collapse / entropy overshoot**——student 熵冲过 teacher 且不回落，推理路径逐渐收窄（`arXiv:2610.05048` E²-OPSD、`arXiv:2610.07842` 特权上下文与 policy drift）。
- 代表：`arXiv:2610.10447` JOLT、`arXiv:2610.05070` OG-OPSD、`arXiv:2610.02117` Where-OPD 等。

</details>

---

## 二、训练成本拆解

蒸馏的训练成本主要由这几块构成：

1. **on-policy rollout 生成**（student 自采样，长 CoT 时最贵）
2. **teacher 打分前向**（逐 token 分布 / top-k KL）
3. **teacher 数量与来源**（要几个模型、是否要先训出来）
4. **模型驻留显存 / 服务开销**

| 成本项 | OPD | MOPD | OPSD |
|---|---|---|---|
| rollout 生成 | ✅ 需要 | ✅ 需要 | ✅ 需要 |
| teacher 前向次数 | 1（外部 teacher） | $1\sim K$（取决于 hard-route 还是 token-route） | 1（同模型，但 prompt 更长） |
| 额外 teacher 模型 | 1 个外部强模型 | **K 个专家模型** | **0**（或 frozen/EMA 副本） |
| 是否要先训 teacher | 一般复用现成强模型 | **要先 RL/SFT 训出 K 个专家** | 不需要 |
| 主要工程难点 | divergence 方向、稳定性 | 路由、域间冲突、能力 seesaw | collapse、entropy overshoot |

**排序结论：**

- 🥇 **MOPD 成本最高**。即使 hard-routing 下每步只前向 1 个 teacher，它仍额外背负：(a) K 个专家的训练成本；(b) K 个 teacher 的存储/服务/显存；(c) token 级 routing 变体要 K 个 teacher 都前向，成本再乘 $K$。实测数据：`arXiv:2610.04272` 报告 **MOPD 用掉 SFT 的 $14.8\sim23.1\times$ 训练 GPU-hours**，且四种方法（SFT、soft-label 蒸馏、hybrid teacher-prefix、MOPD）精度"几乎相同"，质疑了 MOPD 的性价比。
- 🥈 **OPD 居中**。相比 OPSD 多了"必须有一个外部大 teacher"的显存与前向成本；teacher 越大、student 越小，成本差越明显。
- 🥉 **OPSD 最低**。综述明确写 OPSD "removes that cost"（去掉第二个模型的开销），且"accuracy comparable to RL at a fraction of the generated tokens"。但要留两个 caveat：
  - 若 privileged 上下文（参考答案/修复指导）是**训练中用一个更大模型在线生成**的，成本会明显上升（`arXiv:2610.02700` AIR-OPD 就区分了 self-guidance 与 external guidance）；
  - OPSD 的 teacher 前向 prompt 更长（含解法），单次前向 FLOPs 高于普通 student 前向。

---

## 三、怎么选（实用建议）

- **有一个现成的强 teacher，且要提升特定能力** → 用 **OPD**，最直接。
- **已有 K 个领域专家，要把能力合并进一个模型** → **MOPD**，但先想清楚：如果只想省钱，weight merging 往往几分钟 CPU 就能拿到接近的效果，SFT/soft-label 蒸馏也可能追平（`arXiv:2610.04272`、`arXiv:2608.27409`）。
- **没有更强的 teacher（或不想付第二个模型的钱），但有参考答案/verifier/环境反馈** → **OPSD**，最省；但要重点防 collapse / entropy overshoot（token 选择、KL 方向、privileged 上下文的"内容 vs 来源"设计都很关键）。
- **在线环境/长 horizon agent** → OPD/OPSD 的 rollout 成本会被放大，可关注加速方法（如 `arXiv:2610.06105` Flash-OPD 报 $2.2\sim7.5\times$ 加速）。

---

### 一句话总结
三者的关系是"**一个母范式 + 两个正交扩展**"：MOPD 扩展 teacher **数量**（成本最高），OPSD 扩展 teacher **来源**（成本最低，因为不需要外部 teacher）。若题目问"哪个最贵"，答案是 **MOPD**。
