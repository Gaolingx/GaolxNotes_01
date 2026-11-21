# 大模型训练与RL框架梳理

## 🧑‍💻 User

预训练+sft megatron torchtitan nanotron llamafactory trl
rl verl slime trl openrlhf
继续补充

---
## 😀 默认助手 (7)

下面按你说的两条主线继续往上加，并补充几层“周边但绕不开”的东西。名字基本就是仓库名，可以直接在 GitHub 搜。

## 一、预训练 + SFT

<details open>
<summary><b>大规模预训练 / 分布式训练引擎</b>（补充 Megatron、TorchTitan、Nanotron 一侧）</summary>

| 项目 | 出处 | 定位 / 特点 |
|---|---|---|
| `gpt-neox` | EleutherAI | 经典 3D 并行预训练代码库，很多项目的起点 |
| `NeMo` (NeMo Framework) | NVIDIA | 工业化封装 Megatron，含数据、并行、容错，企业落地多 |
| `MindSpeed` / `MindSpeed-LLM` | 华为 | 昇腾版 Megatron 加速库，国产卡必备 |
| `MaxText` | Google | JAX/TPU 优先，与 Megatron 对位 |
| `Levanter` | Stanford CRFM | JAX + Haliax，fp8/TPU 友好，代码干净 |
| `EasyLM` | CRFM（Koala） | JAX 轻量预训练+SFT |
| `OLMo-core` / `OLMo` | AI2 | 全开源（数据+代码+权重），可复现研究范本 |
| `lingua` | Meta | 极简、可复现的 Llama 预训练研究代码（较新） |
| `Picotron` | HuggingFace | Nanotron 的后续，4D 并行极简实现 |
| `Composer` / `LLM Foundry` | Databricks / MosaicML | 训练库 + LLM 配方，强调效率与可复现 |
| `Colossal-AI` | HPC-AI | 显存优化、异构并行，ColossalChat 同门 |
| `DeepSpeed` | Microsoft | ZeRO / Offload / 训练引擎，作为底层被大量框架依赖 |
| `InternEvo` | 上海 AI Lab | 大规模预训练系统（InternLM 底座） |
| `litgpt` | Lightning AI | 教学友好，从零预训练到微调全链路 |
| `MindFormers` | 华为 | MindSpore 栈 |
| `PaddleFormers` | 百度 | 飞桨栈 |
| `OneFlow` | 一流科技 | 国产框架，曾做全局视角并行 |

</details>

<details open>
<summary><b>SFT / 微调工具层</b>（补充 LLaMA-Factory、TRL 一侧）</summary>

| 项目 | 出处 | 定位 / 特点 |
|---|---|---|
| `torchtune` | Meta | PyTorch 原生 recipe 式，SFT/DPO/QLoRA 一把梭，和 TorchTitan 互补 |
| `Axolotl` | 社区 | 配置驱动（YAML）的 SFT/DPO/GRPO 全家桶，社区最活跃之一 |
| `Unsloth` | 社区 | 单卡 QLoRA/FFT 加速、省显存，现在也支持 GRPO |
| `ms-swift` | 阿里 ModelScope | 全模态 + 全流程（SFT/DPO/RL/评测/部署） |
| `XTuner` | 上海 AI Lab | 轻量、显存省，多模态支持好 |
| `Open-Instruct` | AI2 | SFT + DPO + RLVR 全流程脚本（Tülu 配方） |
| `FastChat` | LMSYS | 训练 + 服务 + 评测一体化，早期 Vicuna 生态 |
| `PEFT` | HuggingFace | LoRA/QLoRA/DoRA 等适配器标准库 |
| `mistral-finetune` | Mistral | 官方轻量微调，代码极简 |
| `MiniMind` | 社区 | 从零训练小模型的完整教学工程 |

</details>

## 二、RL（后训练 / 对齐）

<details open>
<summary><b>补充 verl、slime、TRL、OpenRLHF 一侧</b></summary>

| 项目 | 出处 | 定位 / 特点 |
|---|---|---|
| `NeMo-RL` / `NeMo-Aligner` | NVIDIA | 官方 RL 栈，DPO/GRPO/PPO，与 NeMo 生态打通 |
| `AReaL` | 蚂蚁 | 异步 RL，训练/推理解耦，主打吞吐 |
| `SkyRL` / `SkyRL-v0` | UC Berkeley NovaSky | 长时程 agent RL，含完整训练-环境闭环 |
| `ROLL` | 阿里 | 大规模 RL for LLM，支持 MoE、多场景 |
| `rLLM` | Berkeley/UCLA | agent RL 框架，底层可接 verl |
| `torchforge` | Meta | 基于 TorchTitan 的 RL 库（较新，注意跟进） |
| `PipelineRL` | ServiceNow | 异步流水线式 RL，重吞吐 |
| `EasyR1` | 社区 | 基于 verl 的多模态 RL，开箱即用 |
| `ART` | OpenPipe | Agent RL / GRPO 训练服务，偏工程化 |
| `Tinker` | Thinking Machines | 把 SFT/RL 做成 API，无需自建集群（闭源服务） |
| `ChatLearn` | 阿里 PAI | RLHF 训练框架，老牌工程实现 |
| `ColossalChat` | HPC-AI | Colossal-AI 的 RLHF 流水线 |
| `DeepSpeed-Chat` | Microsoft | 早期 RLHF 三阶段流水线，现在多作参考 |
| `Open-RLHF` / `trlX` | LAION / CarperAI | 历史项目，读代码仍有价值 |
| `MindRLHF` | 华为 | 昇腾栈 RLHF |

还需注意：**算法层仓库**往往比框架更新快，例如 `DAPO`、`VAPO`、`GSPO`、`Open-Reasoner-Zero`、`Logic-RL`、`TinyZero`，它们是“配方”而非框架，但常直接决定你选哪个框架。

</details>

## 三、周边层（选型时经常被忽略）

<details>
<summary><b>推理引擎（RL 的 rollout 后端，强绑定）</b></summary>

`vLLM`、`SGLang`、`TensorRT-LLM`、`LMDeploy`、`TGI`、`llama.cpp`、`MLC-LLM`。

选 RL 框架时第一件事就是看它接了谁：verl 主推 vLLM/SGLang，slime 本身长在 SGLang 上，OpenRLHF 也以 vLLM 为主。**没接上你惯用的引擎，等于要自己写适配层。**

</details>

<details>
<summary><b>数据</b></summary>

`Data-Juicer`（阿里）、`NeMo Curator`（NVIDIA）、`datatrove`（HF）、`dolma`（AI2）、`DataFlow`、`text-dedup`、`MinerU`（PDF 解析）、`RedPajama` / `FineWeb`（数据集本身）。

</details>

<details>
<summary><b>评测</b></summary>

`lm-evaluation-harness`（EleutherAI）、`OpenCompass`（上海 AI Lab）、`lighteval`（HF）、`EvalPlus`、`BigCode-Evaluation-Harness`、`AlpacaEval` / `Arena-Hard`、`simple-evals`。

</details>

<details>
<summary><b>Agent / 环境（RL 时代的新缺口）</b></summary>

`AgentGym`、`AWorld`、`RAGEN`、`Agent-R1`、`NeMo Gym`、Prime Intellect 的 environments hub。这块目前最碎，没有事实标准。

</details>

<details>
<summary><b>调度 / 集群</b></summary>

`Ray`（`Ray Train`）、`SkyPilot`、`Kubeflow Training Operator`、`Volcano`、`Determined AI`。

</details>

## 四、几组常见搭配

| 场景 | 推荐组合 |
|---|---|
| 从零预训练大模型 | `Megatron-LM` / `NeMo`（或昇腾 `MindSpeed`）+ 自研数据管线 |
| 中规模可复现研究 | `torchtitan` + `torchtune`，或 `OLMo-core` / `lingua` |
| 单机/少卡快速微调 | `LLaMA-Factory` 或 `Axolotl` / `Unsloth` |
| 推理模型 RL（GRPO 类） | `verl` 或 `slime` + `vLLM`/`SGLang` |
| Agent / 长时程 RL | `SkyRL` 或 `rLLM` |
| 多模态 RL | `EasyR1`（verl 系）或 `ms-swift` |
| 企业级一站式 | `NeMo` 全家桶（训练+RL+Curator+Eval） |

以上是知识性梳理，2025 年下半年新出的几个（`torchforge`、`Picotron`、`lingua`、`NeMo-RL`、`SkyRL-v0`）建议用前先确认仓库现状和文档完成度——这一波项目迭代快，命名和定位常有变动。如果你是要整理一份调研表，我可以帮你补成带 license、star、并行策略、支持的 RL 算法这几列的版本。
