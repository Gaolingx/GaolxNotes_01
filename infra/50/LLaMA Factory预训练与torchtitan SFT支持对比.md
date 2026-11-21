# LLaMA Factory预训练与torchtitan SFT支持对比

## 🧑‍💻 User

llamafactory支持预训练吗 torchtitan支持sft吗

---
## 😀 默认助手 (7)

我会先通过文档确认这两个框架各自支持哪些训练阶段，随后再回答。

torchtitan 的 README 内容有些异常（出现了 2026 年日期和未见过的模型名），我需要再核实仓库的实际结构；同时确认 LLaMA-Factory 的 `stage` 定义。

jsdelivr 的索引看起来是旧的快照（没有 `rl`、`sft` 目录），与 README 描述不一致，我需要更可靠的证据。

torchtitan 主分支的数据文档里已有明确的 SFT 章节（`ChatProcessor`、`FirstFitPackingConfig`、renderer、prompt label 掩码）。再确认 LLaMA-Factory 的 `stage` 枚举与预训练示例。

两个问题的答案都是“支持”，但成熟度和适用场景差别很大。我核对了 LLaMA-Factory 的源码和 docs、torchtitan 的 `main` 分支目录结构与数据管线文档。

## 结论速览

| | LLaMA-Factory | torchtitan |
|---|---|---|
| 预训练 (PT) | ✅ 原生支持，`stage: pt` | ✅ 本来就是核心场景 |
| SFT | ✅ 主打功能，模板/数据集生态最全 | ✅ `main` 分支已支持（较新，2025 年后加入） |
| 突出能力 | 100+ 模型、LoRA/QLoRA、WebUI/CLI、大量 chat template | FSDP2/TP/PP/CP/EP 组合并行、MXFP8/NVFP4、大规模吞吐 |
| 定位 | 单机~中等规模的一站式微调 | 面向千卡级的原生 PyTorch 训练平台 |

---

## LLaMA-Factory 支持预训练吗？支持

`stage` 参数就是训练阶段开关，`pt` 是其中的一等公民（与 `sft`/`rm`/`ppo`/`dpo`/`kto` 并列）。

要点：

- **数据格式**：Alpaca 格式的纯文本，`{"text": "document"}`；`dataset_info.json` 里映射 `"columns": {"prompt": "text"}`。官方样例数据集是 `c4_demo`。注意 **预训练数据集不支持 ShareGPT 格式**。
- **默认开启 packing**：源码中 `data_args.packing` 缺省值就等于 `finetuning_args.stage == "pt"`，所以 pt 阶段不显式设置也会走拼接打包。
- **可与 LoRA/full/freeze 组合**，官方示例就是 `stage: pt` + `finetuning_type: lora` + `lora_target: all`。
- **阶段互斥的坑**：`neat_packing`、`train_on_prompt`、`mask_history`、`predict_with_generate` 在非 `sft` 阶段会直接报错；Megatron Bridge 后端也只接受 `pt` / `sft` 两个阶段。

<details>
<summary>官方给的预训练配置示例（节选）</summary>

```yaml
### method
stage: pt
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: c4_demo
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
```
</details>

本质上这是**续训 / continual pretraining**（在已有权重上继续做因果语言建模），不是从零复现一个 base 模型；要真从零训，规模和并行效率该交给 torchtitan 那类框架。

---

## torchtitan 支持 SFT 吗？当前 `main` 支持

这一点容易踩坑：**早期 torchtitan 是纯预训练框架，网上很多文章仍写“只做 PT”**。现在官方 README 的自我定位已是 "a PyTorch-native platform for **pretraining and post-training**"，Features 里明确写了 "Pretraining, **SFT on chat datasets**, and LoRA"。

我在 `torchtitan/components/data/README.md` 里确认了完整 SFT 章节，实现方式是换 processor 和 packing 策略，loader 不变：

- `ChatProcessor`：把对话套 tokenizer 的 chat template，生成 next-token 的 `input_ids`/`labels` 对，**prompt 部分 labels 置 `IGNORE_INDEX`**。
- 边界判定：用 `add_generation_prompt=True` 渲染 prompt，要求它是完整渲染的**精确 token 前缀**，否则抛 `ValueError`（模板改写了前文轮次、或分隔符只在上下文中合并，就会失败）。
- 多轮对话：建议显式选 renderer（如 `Qwen3RendererConfig`），一次性拿到 token + loss mask，监督模型生成部分（含轮次终止符）；一条会话算一个 sample，packing **只在会话之间**重置位置，不在轮次之间。
- packing 用 `FirstFitPackingConfig`；超 `max_context_length` 的样本整条丢弃。
- 再往上还有 `torchtitan/rl`（TitanRL，GRPO/DAPO）与 vLLM 共享模型定义。

<details>
<summary>SFT 数据管线写法（文档示例，GSM8K 改造成 chat）</summary>

```python
from torchtitan.components.data import (
    FirstFitPackingConfig, GrainDataLoader,
    HuggingFaceRandomAccessSource, SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor

def gsm8k_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]

gsm8k_ds = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(path="openai/gsm8k", name="main", split="train"),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)
config.dataloader = GrainDataLoader.Config(dataset=FirstFitPackingConfig(dataset=gsm8k_ds))
```
</details>

---

## 怎么选

- **要做 SFT / DPO / LoRA，单机 8 卡以内，模型 100+ 任意架构** → LLaMA-Factory。chat template 兼容性、数据集注册、WebUI 这些工程细节它已经替你趟平了（torchtitan 的“精确前缀”约束在冷门模板上会直接报错）。
- **要从零/大规模续训，需要 TP+PP+CP+EP 组合、FP8/NVFP4、几百卡吞吐** → torchtitan。它的 SFT 目前是“够用”级别：chat 数据集 + packing + loss mask + LoRA，灵活性远不如 LLaMA-Factory，且依赖 nightly PyTorch。
- **既要大规模预训练又要 SFT** → torchtitan 现在一条栈能走完（PT → SFT → RL），不用中途换框架重接并行；如果只是小模型对齐，没必要为了这个迁移。

⚠️ 两点不确定性：我读的是 `main` 分支，**稳定版 release（如 v0.2.0）里 SFT 可能还没有**，用 pip 装的版本请以实际 `torchtitan/components/data` 是否存在为准；另外 torchtitan README 里的部分新特性我未实机验证，仅基于官方文档与仓库文件结构判断。
