# 为什么给llm外挂visionencoder

### 🧑‍💻 User

为什么给llm外挂visionencoder一般是训练一个Projector（e.g. PatchMerger MLP (pre_norm → linear_1 → GELU → linear_2), 1152→4608→6144 — trained），冻结Vision tower（ViT）和Text backbone（LLMs）？

---

### 🤖 Assistant

**核心原因：只训练小型 Projector（模态对齐器）就能高效地把视觉特征“翻译”到 LLM 的语义空间，同时最大化保留两边已经很强的预训练能力，避免高成本和灾难性遗忘。**

这是 LLaVA、MiniGPT-4、Qwen-VL 系列、InternVL 早期版本等主流“LLM 外挂 Vision Encoder”架构的经典做法（Vision Tower 冻结 + LLM 冻结/部分解冻 + 只训 Projector）。

### 1. Projector 的作用（为什么需要它）
Vision Encoder（如 CLIP/SigLIP/InternViT 的 ViT）输出的是视觉特征（例如 patch tokens，维度常为 1152 或 1024 等），而 LLM 的输入是文本 token embedding（维度通常是 4096/5120/6144 等）。两者：
- 维度不同
- 特征分布/语义空间不同（对比学习 vs. 自回归语言建模）
- 序列长度也常需要压缩（所以有 PatchMerger：把多个 patch 合并再投影，降低后续 LLM 的计算量）

Projector（常见是 MLP：`pre_norm → linear → GELU → linear`，如 1152 → 4608 → 6144）的职责就是学习一个**映射**，把视觉特征对齐到 LLM 能理解的 embedding 空间。训练目标通常是图像-文本对齐（captioning 或对比/生成损失），让 LLM “看懂”图像 token。

一旦对齐做好，冻结的 LLM 就能直接用其强大的语言能力做 VQA、描述、推理等。

### 2. 为什么冻结 Vision Tower（ViT）和 Text Backbone（LLM）
| 原因 | 具体说明 |
|------|----------|
| **计算与工程成本** | ViT（尤其大分辨率/高维）+ 7B~70B+ LLM 参数量巨大。全量训练需要海量 GPU、长时间和复杂优化（学习率、混合精度、ZeRO 等）。只训一个小 MLP（几百万到几千万参数）极快、便宜，一张/几张卡就能做 alignment pretrain。 |
| **保留预训练知识，避免灾难性遗忘** | - Vision Tower（CLIP/SigLIP 等）已经在海量图文数据上学到了鲁棒、通用的视觉表征。<br>- LLM 已经具备强大的语言理解、推理、指令跟随能力。<br>全量微调容易破坏这些能力（尤其 LLM 对分布偏移敏感），导致“会看图但语言变差”或视觉表征退化。冻结能最大程度守住两边的“硬实力”。 |
| **聚焦对齐任务 + 数据效率** | 多模态高质量数据（尤其带 instruction 的）相对稀缺且噪声大。只训小连接器，用图文对就能快速学会“视觉→语言”的接口；全量训容易过拟合或需要更多数据/正则。 |
| **训练稳定性** | 端到端同时更新两个异构大模型（视觉特征尺度、梯度特性与 LLM 差异大）容易不稳定。冻结后，优化目标更简单、收敛更稳。经验上 LLaVA 风格“只训 projector”已经能取得很强 zero-shot / few-shot 表现。 |
| **模块化与可复用性** | 可以快速换不同 Vision Encoder 或不同 LLM，只重新训/微调 Projector；也方便后续做 LoRA、Adapter 等轻量解冻。 |

### 3. 实际训练中的常见流程（多阶段）
1. **Alignment Pretraining（只训 Projector）**
   冻结 ViT + LLM，用大规模图文对（caption）训练 MLP/PatchMerger。目标是让视觉 token 进入 LLM embedding 空间后，LLM 能生成合理描述。这是“外挂”最核心的一步。

2. **Visual Instruction Tuning（可选解冻）**
   加入指令数据后，常继续冻 ViT，解冻 LLM（全量或 LoRA/QLoRA），有时也轻量解冻 ViT 的后几层或加 Adapter。目的是让模型学会“按指令用视觉信息回答”。很多强模型（Qwen2-VL、InternVL 后续、LLaVA-NeXT 等）会在此阶段做更多解冻或高分辨率适配，但**基础对齐仍高度依赖那个 Projector**。

3. **进阶变体**
   - 更复杂的连接器（Q-Former、Perceiver Resampler、Cross-Attention 等）仍然常冻 ViT/LLM，只训连接部分。
   - 高分辨率/动态分辨率时，PatchMerger 或类似压缩模块很有用（减少 token 数）。
   - 现在也有人解冻 Vision Encoder 做更好细粒度对齐，或用统一架构，但“冻结大骨干 + 训小投影”仍是最主流、最高性价比的起点。

### 4. 为什么这个简单做法有效？
- 预训练 Vision Encoder 和 LLM 已经足够强，瓶颈往往在**接口对齐**而非两边表征本身。
- 经验验证充分：LLaVA 证明“CLIP + Vicuna + 两层 MLP”就能大幅提升多模态能力；后续工作不断在此范式上迭代（更好 encoder、更高分辨率、更好数据、更优 projector 结构）。
- 风险可控：即使后续要提升，也可以在冻结对齐的基础上再做参数高效微调，而不是从零全量训。

### 简要总结
给 LLM 外挂 Vision Encoder 时，**只训练 Projector（MLP / PatchMerger 等）并冻结 ViT 和 LLM**，本质是：
- 用最小成本完成跨模态对齐；
- 最大程度保护已有强大的视觉与语言能力；
- 获得训练稳定、数据高效、可快速迭代的系统。

这是当前工程与研究里最务实、验证最多的范式。更强的模型会在此基础上做部分解冻、更高分辨率、更好数据配方或架构改进，但“Projector 对齐 + 大骨干冻结”仍然是地基。

如果你关心具体模型（LLaVA、Qwen-VL、InternVL 等）的 projector 细节、维度选择、训练 loss 或后续解冻策略，可以继续问。