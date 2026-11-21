# verl和slime类似开源RL框架盘点

## 🧑‍💻 User

verl slime 还有哪些类似的开源rl框架

---
## 😀 默认助手 (7)

## 先说结论

`verl` 和 `slime` 属于同一类框架：**rollout 引擎（vLLM/SGLang）与训练后端（FSDP/Megatron）解耦，支持 colocate 或分离部署，并面向大规模 RLVR/RLHF**。同类的还有：**OpenRLHF、ROLL、NeMo-RL、AReaL、Relax、Miles、AsyncFlow、PipelineRL、SkyRL、prime-rl** 等。差异主要在“异步程度、Agent 支持、工程成熟度”三个方面。

---

## 一、主流“全栈式”RL 训练框架

| 框架 | 出品方 | 核心特点 | 适合场景 |
|---|---|---|---|
| `OpenRLHF` | OpenLLMAI（社区） | 采用 Ray + vLLM + DeepSpeed，是最早用 vLLM 加速 rollout 的一批框架；支持 PPO/GRPO/REINFORCE++/RLOO/DPO | 学术界的基线方案，代码易读、改动成本低 |
| `ROLL` | 阿里巴巴 | 把多角色 Actor/Critic/RM/Gen 放在同一集群并做混合部署，面向超大规模 RLHF | 大规模、需要极致资源利用率的团队 |
| `NeMo-RL`（前身 NeMo-Aligner） | NVIDIA | 使用 Megatron-LM 训练后端和 vLLM 推理后端，支持 FP8、TP/PP 大并行 | 在千卡以上、有英伟达技术栈经验 |
| `AReaL` | 蚂蚁集团 | **完全异步**，将 rollout 与 train 彻底解耦，采用 stale-sample 容忍机制；提供 AReaL-boba 配方 | 追求吞吐，可接受一定 off-policy |
| `Relax` | 小红书 rednote-hilab | 面向大规模 post-training 的异步框架，强调训练/推理角色分离与弹性调度 | 生产级异步训练 |
| `Miles` | RadixArk | **基于 slime 扩展**的生产级版本，增加了异步 rollout、FP8、大规模稳定性 | 想用 slime 但需要生产级功能 |
| `AsyncFlow` | 字节 Seed | 异步流式流水线 RL，论文中给出系统设计 | 研究异步调度，或做吞吐对比 |
| `PipelineRL` | ServiceNow | 流水线式异步 on-policy RL，权重同步开销小 | 长序列、长尾 rollout 场景 |
| `ms-swift` | ModelScope/阿里 | 工程化封装程度高，支持 GRPO/DAPO/GSPO/PPO，模型覆盖最广 | 想直接跑通、少改代码 |
| `ChatLearn` | 阿里 PAI | 基于 Megatron/vLLM 的 RLHF 全流程，偏平台化 | 企业平台集成 |

---

## 二、Agent / 多轮 / 工具调用方向（近一年主要增长点）

- **`SkyRL`（NovaSky-AI）**：早期基于 verl，现在自研 `skyrl-train` + `skyrl-gym`，专注长程 Agent 训练
- **`rLLM`（rllm-org）**：用于 Agent RL，支持在 verl 后端运行，包含轨迹与环境抽象
- **`verl-agent` / GiGPO**：基于 verl 的 Agentic RL 分支，支持多轮和分组轨迹
- **`prime-rl` + `verifiers`（Prime Intellect）**：环境库与异步训练分离，用于 INTELECT 系列
- **`Atropos`（Nous Research）**：异步 RL 环境框架，强调“任意环境接入”
- **`ART`（OpenPipe）**：Agent Reinforcement Trainer，可为任意 Agent 框架（LangChain、自己实现的 loop）接入 GRPO
- **`RLinf`**：面向 LLM Agent 与具身智能的统一 RL 基础设施
- **`Agent-R1`、`RAGEN`、`Open-Teach`**：偏研究型的多轮 Agent RL 复现代码

---

## 三、入门 / 单机 / 教学类

- **`TRL`（HuggingFace）**：`GRPOTrainer` 是大多数人训练的起点；规模上不去，但 API 最友好
- **`LLaMA-Factory`**：配置式启动 PPO/DPO/GRPO，几乎零代码
- **`Unsloth`**：单卡 GRPO 显存优化，适合 7B 以下
- **`EasyR1`（hiyouga）**：基于 verl 的精简版，主打多模态 GRPO，配置少
- **`XTuner`（上海 AI Lab）**：支持 GRPO/DPO，与 InternLM 生态配合使用
- **`TinyZero`、`nano-aha-moment`、`MiniMind`**：用几十行到几百行复现 R1 式 RL，适合读代码
- **`open-instruct`（AI2）**：Tülu 3 的 RLVR 训练配方与代码

<details>
<summary>四、已逐渐被取代的老牌框架（了解即可）</summary>

- `DeepSpeed-Chat`：最早的 RLHF 全流程实现，现在基本只用于教学
- `trlX`（CarperAI）：分布式 RLHF 先驱项目，已停止维护
- `ColossalChat`：Colossal-AI 的 RLHF 实现，现在更常使用其推理和微调部分
- `RL4LMs`：传统 NLP 的 RL 微调库，与现在的 RLVR 范式不同
- `RLlib` / `OpenRL` / `CleanRL`：通用 RL 库，不是 LLM 专用，但在环境侧仍有价值

</details>

---

## 五、选型建议（按你的实际约束）

| 你的情况 | 建议 |
|---|---|
| 只有 1–8 卡，想快速验证 GRPO 有没有效果 | `TRL`、`LLaMA-Factory`、`Unsloth` |
| 有几十卡，想稳定跑 7B–32B 的 RLVR | `verl` 或 `ms-swift`，生态最全，issue 最多 |
| 想复刻 GLM 那种规模、同时看重 SGLang 生态 | `slime`，并关注 `Miles` 的生产化扩展 |
| 瓶颈明确在 rollout 长尾和 GPU 空转 | `AReaL`、`Relax`、`PipelineRL`（异步路线） |
| 必须使用 Megatron 训练千卡以上模型 | `NeMo-RL`、`ROLL` |
| 做多轮、工具调用、Agent 任务 | `SkyRL`、`rLLM`、`verl-agent`、`prime-rl`；先把环境标准定下来 |
| 想读代码理解原理 | `TinyZero`、`EasyR1`，然后阅读 `verl` 的 controller |

---

## 六、几点实践提醒

1. **这些框架之间大量互相 fork**：`EasyR1`、`verl-agent`、早期 `SkyRL` 都来自 verl；`Miles` 来自 slime。看 issue 和 PR 时，要注意你所在的是哪个仓库。
2. **共同技术难点只有三个**：rollout 引擎的权重热更新、训练/推理的 colocate 显存调度，以及长尾序列的负载均衡。选型时直接看它怎么解决这三点，比看 README 的功能列表有用。
3. **异步路线（AReaL、Relax、PipelineRL）要单独验证收敛性**，因为它的吞吐提升来自 off-policy 容忍，不同任务（数学、代码、Agent）的容忍度差别很大。
4. **框架迭代极快**：上面列出的项目，半年内 API 和定位都可能变化；选型前务必查看仓库最近 30 天的 commit 频率和 release 记录。

如果需要，我可以针对某个具体场景（卡数、模型规模、任务类型）给出一份更细的对比，或者直接查看某个仓库的当前实现，确认它是否支持异步和 Agent 多轮训练。
