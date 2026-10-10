# 工业级LLM框架

## 🧑‍💻 User

llm 预训练/sft工业级框架一般用哪些？

---
## 😀 默认助手 (7)

# LLM 预训练 / SFT 工业级框架

工业界做大规模训练时，通常**不是用单一框架，而是"训练框架 + 并行后端 + 数据/调度系统"的组合**。下面按用途分类梳理。

## 一、预训练（Pretraining）框架

这是最"重"的部分，核心解决**超大模型 + 超长序列 + 数千卡集群**的并行问题。

| 框架 | 主导方 | 核心能力 | 典型使用者 |
|------|--------|----------|-----------|
| **Megatron-LM / Megatron-Core** | NVIDIA | TP + PP + DP + EP + 序列并行，事实上的工业标准 | NVIDIA、多数大厂自研底座 |
| **DeepSpeed** | Microsoft | ZeRO 1/2/3、Offload、混合精度、MoE | 大量开源/学术 + 工业 |
| **PyTorch FSDP / FSDP2** | Meta / PyTorch | 原生分片数据并行，易用性好 | Meta、HuggingFace 生态 |
| **torchtitan** | PyTorch 官方 | 原生 PyTorch 大模型训练参考实现 | 追求纯 PyTorch 栈的团队 |
| **Nanotron** | HuggingFace | 极简 3D 并行，速度快 | HF 训练栈 |
| **Composer** | MosaicML/Databricks | 训练效率、容错、弹性 | Databricks、Mosaic |
| **Colossal-AI** | 潞晨科技 | 多种并行 + 异构内存 | 国内外中小团队 |

**JAX/XLA 路线**（Google 系为主）：

- **MaxText**（Google 官方，TPU 优先）
- **Pax**（Google，原 PaxML）
- **Levanter**（Stanford CRFM）
- **EasyLM**（Koala 团队）

> 现实中最常见的是 **Megatron-LM + DeepSpeed 组合**（即 *Megatron-DeepSpeed*），许多头部模型（如早期 BLOOM、部分国产大模型）都基于此。

<details>
<summary><b>并行技术栈速查（点开）</b></summary>

- **DP / DDP**：数据并行
- **ZeRO-1/2/3**：优化器 / 梯度 / 参数分片，DeepSpeed 核心
- **TP**：张量并行（层内切分）
- **PP**：流水线并行（层间切分，含 1F1B、interleaved）
- **EP**：专家并行（MoE 必备）
- **SP / CP**：序列并行 / 上下文并行（长序列）
- **Offload**：把优化器/参数卸载到 CPU/NVMe

</details>

## 二、SFT / 微调（Fine-tuning）框架

SFT 规模比预训练小，更强调**易用性、数据管线、LoRA/QLoRA 等参数高效方法**。

| 框架 | 特点 | 适合场景 |
|------|------|----------|
| **LLaMA-Factory** | 国产、开箱即用，WebUI + 全流程 | 快速实验、中小团队 |
| **Axolotl** | 配置驱动，支持 FSDP/DeepSpeed | 生产化微调 |
| **torchtune** | PyTorch 官方微调库，纯原生 | 想要干净 PyTorch 栈 |
| **HuggingFace TRL + PEFT** | SFT/DPO/PPO 全家桶 | 生态最广 |
| **Unsloth** | 单卡极致提速 + 省显存 | 消费级 GPU / 单机微调 |
| **Swift (ms-swift)** | 魔搭出品，多模态友好 | 国内多模态微调 |
| **DeepSpeed / FSDP** | 作为分布式后端被上面框架调用 | 大模型全参微调 |

## 三、对齐 / RLHF / RL 框架

虽然你问的是预训练+SFT，但工业级流水线通常需要配套：

- **OpenRLHF** — 轻量、集成 vLLM 推理
- **veRL**（原 verl）— 字节开源，HybridFlow，工业级
- **TRL**（HuggingFace）— PPO/DPO/GRPO
- **DeepSpeed-Chat** — 端到端 RLHF
- **rerl / AReaL 等** — 较新

## 四、真实工业界的典型"组合"

大厂自研流水线一般长这样：

```
数据管线（自研 + Spark/Ray）
        │
训练框架：Megatron-LM / Megatron-Core
        │
并行后端：DeepSpeed 或 自研（部分直接用 FSDP2）
        │
推理采样：vLLM / TensorRT-LLM / SGLang
        │
调度与容错：K8s + Ray / Slurm + 自研 Checkpoint
        │
SFT/对齐：LLaMA-Factory / Axolotl / veRL / OpenRLHF
```

<details>
<summary><b>按团队规模给建议（点开）</b></summary>

- **个人 / 单机**：Unsloth、LLaMA-Factory + QLoRA
- **小团队（<64 卡）**：Axolotl / torchtune + FSDP 或 DeepSpeed
- **中型团队（百卡级）**：Megatron-LM + DeepSpeed，或 torchtitan
- **大厂（千卡+）**：Megatron-Core 自研底座 + 自研调度/容错 + 推理引擎闭环

</details>

---

**一句话总结**：预训练看 **Megatron-LM / DeepSpeed / FSDP（+ torchtitan）**，SFT 看 **LLaMA-Factory / Axolotl / torchtune / TRL**，对齐看 **veRL / OpenRLHF / TRL**；工业界主流是 **Megatron + DeepSpeed 组合**并配合 vLLM 推理与自研数据/调度系统。

需要我针对某个具体场景（比如"8 卡 A100 做 7B SFT"或"千卡预训练选型"）给更细的技术方案吗？

---
## 🧑‍💻 User

那常规后训练/持续预训练中，llamafactory/trl/nanotron/pytorchlightning这些方案怎么样

---
## 🧑‍💻 User

那llamafactory trl pytorchlightning自研这些方案怎么样

---
## 🧑‍💻 User

那常规后训练/持续预训练中，llamafactory/trl/pytorchlightning这些方案怎么样

---
## 😀 默认助手 (7)

# 后训练 vs 持续预训练：这四个方案的定位

先说结论：**这四个框架其实分属两个阵营，不能放在同一把尺子上比**。

| 框架 | 本质定位 | 后训练(SFT/DPO/RL) | 持续预训练(CPT) |
|------|----------|:---:|:---:|
| **LLaMA-Factory** | 后训练**一站式平台** | ✅ 强项 | ⚠️ 能用但不推荐上规模 |
| **TRL** | 对齐**算法库** | ✅ 强项（尤其 RL） | ❌ 不适合 |
| **Nanotron** | 预训练**引擎** | ❌ 基本不管 | ✅ 强项 |
| **PyTorch Lightning** | 通用**训练循环外壳** | 🔧 要自己搭 | 🔧 要自己搭 |

关键要先分清两个任务，它们的**瓶颈完全不同**：

- **后训练**：数据量小（万~百万条）、步数少、瓶颈在**数据管线 / chat template / PEFT / rollout / 评测**，而不是极致吞吐。
- **CPT**：本质就是"缩小版预训练"，瓶颈在**吞吐、长上下文稳定性、大规模并行、断点续训**，数据是流式 token 流。

<details>
<summary><b>为什么这个区分决定选型（点开）</b></summary>

后训练框架的设计目标是"灵活 + 好上手"，CPT 引擎的设计目标是"稳定 + 高 MFU"。两者优化方向几乎正交，所以很少有一个框架同时做好两端。硬用后训练框架跑 CPT，通常会在长稳、吞吐、checkpoint 上翻车。

</details>

---

## 逐个点评

### 1. LLaMA-Factory —— 后训练首选，CPT 仅限小规模

<details>
<summary><b>展开详情</b></summary>

**优点**
- 真正的一站式：SFT / DPO / KTO / ORPO / PPO / RM / **pretrain(CPT)** 全都在配置里一个 `stage` 搞定。
- 对中文/多模态友好，WebUI + CLI，几天就能上手。
- 后端能切 DeepSpeed / FSDP，LoRA/QLoRA 开箱即用。

**后训练**：非常合适。中小团队做 7B~70B 的 SFT/对齐，几乎是默认答案。

**CPT**：它**确实支持** `stage: pretrain`，但：
- 没有 **TP**（张量并行），超大模型切不动；
- 长上下文、流式数据、超大规模 checkpoint 容错不是它的优化重点；
- 吞吐（MFU）不如专门引擎。

**结论**：CPT 用在 **单机或几十卡、≤14B、非超长上下文** 的领域继续预训练是够的；再大就该换 Nanotron / Megatron / torchtitan。

</details>

### 2. TRL —— 对齐算法库，不是训练框架

<details>
<summary><b>展开详情</b></summary>

**优点**
- `SFTTrainer` / `DPOTrainer` / `PPOTrainer` / `GRPOTrainer` 算法最全、更新最快，紧跟 HF 生态。
- 和 PEFT / Accelerate / DeepSpeed 无缝，写几行就能跑 DPO/GRPO。

**本质**：它是个**库**，建在 `transformers.Trainer` 之上，不负责数据/调度/大规模工程。

**后训练**：强，尤其 RL/偏好对齐和想自己控制训练逻辑的场景。

**CPT**：**不适合**。没有 TP/PP 这类大模型并行，也不为流式预训练设计。硬做等于是拿 Trainer 扛预训练，得不偿失。

**结论**：后训练/对齐的"算法层"用它；CPT 别用。

</details>

### 3. Nanotron —— 专为（持续）预训练而生的引擎

<details>
<summary><b>展开详情</b></summary>

**优点**
- HuggingFace 出品，目标就是**从零预训练**：内置 **3D 并行（DP/TP/PP）**、序列并行、flash-attn、ZeRO。
- 极简、代码少、性能好，和 HF `datasets`/`tokenizers` 打通。

**CPT**：✅ **强项**。想做大规模/长上下文持续预训练、又熟悉 HF 生态，它是比 Megatron 轻得多的选择。

**后训练**：❌ 不提供 SFT/DPO/RL 的便利（chat template、PEFT、rollout 都没有），拿它做 SFT 等于自己重写 LLaMA-Factory。

**局限**：社区比 Megatron 小、生态薄，MoE/超大规模场景的支持不如 Megatron-Core 成熟；生产容错要自己补。

**结论**：CPT 好选择；后训练别用它。

</details>

### 4. PyTorch Lightning —— 通用外壳，两者都要自己搭

<details>
<summary><b>展开详情</b></summary>

**优点**
- 把训练循环标准化：`LightningModule` + `Trainer`，日志、checkpoint、AMP、梯度累积、分布式启动都很规整，工程可复现性强。
- 配 FSDP / DeepSpeed 插件后能做中等规模训练。
- 研究/实验迭代体验好。

**缺点（对 LLM 尤其明显）**
- **不含任何 LLM 专属优化**：没有 TP/PP、不默认集成 flash-attn / fused kernel / 序列并行。
- 有抽象层开销，且社区在 LLM 大规模训练上几乎不用它（前沿用 Megatron，HuggingFace 系用 Trainer/Accelerate）。
- 写 CPT 或 SFT 都得自己实现数据处理、packing、loss、并行切分。

**后训练 / CPT**：🔧 都能做，但**都是从零搭**，等于你自己写了个小框架。适合"团队已有 Lightning 技术栈、规模不大（单机~几十卡）、想统一工程规范"的场景。

**结论**：它是"外壳"而非"LLM 方案"。要么当研究底座，要么别在 LLM 场景选它。

</details>

---

## 选型矩阵

| 你的场景 | 推荐 | 理由 |
|----------|------|------|
| 7B~70B SFT / DPO / 对齐 | **LLaMA-Factory** 或 **TRL** | 一站式 / 算法最全 |
| 想要 RL（PPO/GRPO）闭环 | **TRL**（+ OpenRLHF/veRL 上规模） | 算法层最完整 |
| 小规模领域 CPT（≤14B、几十卡） | **LLaMA-Factory** 勉强 / **Nanotron** 更稳 | 看是否要 TP、长上下文 |
| 大规模 / 长上下文 CPT | **Nanotron** 或 **Megatron-LM / torchtitan** | 3D 并行 + 长稳 |
| 已有 Lightning 工程栈、中等规模、自定义多 | **PyTorch Lightning** + FSDP | 统一规范优先 |
| 千卡+ / MoE / 前沿 | **Megatron-Core** | 唯一工业级标准 |

## 一句话总结

> **后训练 → LLaMA-Factory（图省事）/ TRL（要算法）；持续预训练 → Nanotron（要轻量 3D 并行）/ Megatron·torchtitan（要上规模）；Lightning 只是外壳，两头都得自己造轮子。**

<details>
<summary><b>常见的"正确组合"举例（点开）</b></summary>

- **领域 CPT → SFT → DPO** 全流程：
  Nanotron 做 CPT → LLaMA-Factory 做 SFT → TRL 做 DPO
- **不想换框架的温和方案**：
  LLaMA-Factory 一个配置走完 CPT+SFT（规模别太大）
- **上规模对齐**：
  Megatron 预训练/CPT → OpenRLHF 或 veRL 做 RL

</details>

需要的话，我可以按你的**具体规模（卡数/模型大小/上下文长度）**给一条明确的框架搭配建议。
