# 首次大规模SFT LoRA参数设置

### 🧑‍💻 User

llm基模预训练阶段后的第一次大规模sft，lora adapter如何设置参数？

---

### 🤖 Assistant

在LLM基座模型（Base Model）预训练结束后的**第一次大规模SFT（Supervised Fine-Tuning，通常称为指令微调或对齐）**阶段，模型需要经历从“续写文本”到“理解指令并遵循对话格式”的巨大行为转变。

因为这种转变非常剧烈，行业内的最佳实践通常是进行**全参数微调（Full Fine-Tuning, FFT）**。但如果受限于算力必须使用 LoRA，那么 LoRA 的参数设置逻辑必须**尽可能逼近全参数微调的表达能力**。这与针对特定垂直领域的下游小规模微调（通常使用较小的参数）有本质区别。

以下是针对“首次大规模SFT”的 LoRA Adapter 参数设置建议：

### 1. 核心 LoRA 参数设置

首次 SFT 需要极高的参数容量和表达能力，因此不能使用默认的“轻量级”配置。

*   **秩 `r` (Rank): 建议设置在 $64$ 到 $256$ 之间**
    *   **原因：** 秩 $r$ 决定了低秩矩阵更新的维度 $W = W_0 + \Delta W = W_0 + BA$，其中 $B$ 和 $A$ 的内维度即为 $r$。在首次 SFT 中，模型需要学习复杂的指令模板、拒绝策略和安全边界，低秩（如 $r=8$）会导致严重的欠拟合。通常设置为 $r=64$ 或 $r=128$。
*   **缩放系数 `lora_alpha`: 建议设置为 `r` 的 1 倍或 2 倍**
    *   **原因：** LoRA 的实际更新步长会乘以一个缩放因子 $\frac{\alpha}{r}$。如果 $r=128$，设置 $\alpha=128$（缩放因子为 $1.0$）或 $\alpha=256$（缩放因子为 $2.0$）可以保证梯度稳定。**不要**在 $r$ 很大的情况下保持 $\alpha=16$，这会使更新权重微乎其微。
*   **目标模块 `target_modules`: 建议设置为 `all-linear`（所有线性层）**
    *   **原因：** 很多早期教程只建议微调 Attention 层（`q_proj`, `v_proj`）。但在首次大规模 SFT 中，必须同时微调 MLP 层（`gate_proj`, `up_proj`, `down_proj`）。MLP 层往往存储着模型的知识和特征表达，修改它们对于彻底改变模型的输出风格（从续写到对话）至关重要。

### 2. 代码配置示例 (HuggingFace PEFT)

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,                     # 高秩，保证表达能力
    lora_alpha=256,            # 通常设为 r 或 2*r
    target_modules=[           # 覆盖所有线性层
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.0,          # 大规模数据下不需要 dropout
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"] # 强烈建议！
)
```

**⚠️ 关键提示：`modules_to_save`**
在首次 SFT 中，你通常会引入新的特殊 Token（例如 `<|im_start|>`, `<|im_end|>` 等对话模板控制符）。因此，**必须将词表 embedding 层和输出层（`lm_head`）纳入全参数训练**，否则模型根本无法学习到这些新 Token 的意义。

---

### 3. 高阶配置与训练超参

<details>
<summary><b>点击展开：学习率与训练超参设置</b></summary>

因为使用了高秩 LoRA 和 `all-linear`，你的学习率需要介于传统 LoRA 和全参数微调之间。

*   **学习率 (Learning Rate):** 建议设置在 $5 \times 10^{-5}$ 到 $2 \times 10^{-4}$ 之间。
    *   使用余弦退火（Cosine Annealing）调度器。
    *   预热步数（Warmup steps）通常占总训练步数的 $3\% \sim 5\%$。
*   **批次大小 (Global Batch Size):** 首次大规模 SFT 数据量通常在十万到百万级别。Global Batch Size 应该设置得较大，通常在 $128$ 到 $512$ 之间，以保证梯度的稳定性。
*   **Epoch:** 通常为 $1$ 到 $3$ 个 Epoch。对于高质量指令数据，往往 $1$ 到 $2$ 个 Epoch 即可，训练过度会导致灾难性遗忘。

</details>

<details>
<summary><b>点击展开：推荐的高阶 LoRA 变体 (DoRA / rsLoRA)</b></summary>

在首次大规模 SFT 时，普通的 LoRA 可能仍然不够完美，推荐使用以下变体：

1.  **rsLoRA (Rank-Stabilized LoRA):**
    *   **原理：** 传统 LoRA 的缩放因子是 $\frac{\alpha}{r}$。rsLoRA 将缩放因子修改为 $\frac{\alpha}{\sqrt{r}}$。
    *   **优势：** 这使得你可以放心地使用极高的 $r$ 值（例如 $r=256$ 甚至 $r=512$）而不用担心梯度爆炸，非常适合首次 SFT 需要高容量的场景。
    *   **使用方式：** 在 `LoraConfig` 中设置 `use_rslora=True`。
2.  **DoRA (Weight-Decomposed Low-Rank Adaptation):**
    *   **原理：** 将预训练权重分解为“幅度（Magnitude）”和“方向（Direction）”，LoRA 只负责更新方向分量。
    *   **优势：** 论文表明 DoRA 的学习模式和最终性能极其接近全参数微调（FFT），是目前用来替代首次 SFT FFT 的最佳方案。
    *   **使用方式：** 在 `LoraConfig` 中设置 `use_dora=True`。

</details>

<details>
<summary><b>点击展开：如何判断参数设置是否合理？（避坑指南）</b></summary>

在首次 SFT 中，如果你发现以下情况，说明 LoRA 表达能力不足：

1.  **格式遵循失败（Format Failure）：** 模型无法稳定输出 `<|im_start|>` 等特殊 token，或者经常在回答一半时开始生成毫无意义的乱码/无限循环。
    *   *解决思路：* 检查 `embed_tokens` 和 `lm_head` 是否加入训练；增大 $r$；确保覆盖了 MLP 层。
2.  **Loss 降不下去：** 训练 Loss 停留在较高水平，或者 Eval Loss 很快反弹。
    *   *解决思路：* 调整学习率，检查 `lora_alpha` 的缩放比例是否过小。
3.  **“人味”不足：** 模型能够回答问题，但语气依旧像预训练模型一样生硬，喜欢续写而不是直接回答。
    *   *解决思路：* 这是典型的欠拟合，必须扩大 `target_modules`，或者直接改用全参数微调。

</details>

---

### 🧑‍💻 User

对于qwen/deepseek moe这种带有共享专家的moe模型要如何设置target？

---

### 🤖 Assistant

针对带有**共享专家 (Shared Experts)** 机制的 MoE 模型（如 Qwen1.5/2-MoE, DeepSeek-V2/V3/MoE），其 MLP 层结构与传统 Dense 模型有显著不同。

在进行预训练后的首次大规模 SFT 时，设置 `target_modules` 的核心原则是：**全量覆盖专家网络，坚决避开路由网络。**

以下是具体的设置方案和原理：

### 1. Target Modules 设置策略

HuggingFace 的 PEFT 库在匹配 `target_modules` 时使用的是**字符串子串匹配（Substring Matching）**。这意味着你只需要写最基础的线性层后缀，就能自动匹配到 MoE 结构中的相应部分。

*   **Attention 层：** 全部参与微调（`q_proj`, `k_proj`, `v_proj`, `o_proj`）。
*   **共享专家 (Shared Experts)：** 必须参与。它们相当于模型的基础认知底座，负责处理通用知识。
*   **路由专家 (Routed Experts)：** 必须参与。首次 SFT 需要最大的参数容量来完成行为对齐。
*   **路由器 (Router/Gate)：** **绝对不要加入！**（下文会解释原因）。

**注意区分：** 专家内部的激活门控层叫 `gate_proj`（需要微调），而负责分配 token 给专家的路由器叫 `gate` 或 `router`（禁止微调）。

### 2. 代码配置示例

你只需按照标准的 Dense 模型写法，PEFT 就会自动帮你穿透并覆盖到所有共享专家和路由专家内部的线性层：

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128,                     # 首次 SFT 依然保持高秩
    lora_alpha=256,
    target_modules=[
        # Attention 模块
        "q_proj", "k_proj", "v_proj", "o_proj", 
        # 匹配非 MoE 层的 MLP，以及 MoE 层中共享专家和路由专家的内部线性层
        "gate_proj", "up_proj", "down_proj" 
    ],
    # 再次强调：绝对不要写 "gate" 或 "router" 
    lora_dropout=0.0,
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"] # 对话模板 token 必须全参微调
)
```

**为什么上述代码能同时覆盖共享和路由专家？**
以 DeepSeek / Qwen 为例，其内部的权重命名格式通常如下：
*   共享专家层：`model.layers.10.mlp.shared_experts.gate_proj`
*   路由专家层：`model.layers.10.mlp.experts.0.gate_proj`
由于你的 `target_modules` 里写了 `"gate_proj"`，它作为子串，会同时命中上述两种专家。

---

### 3. MoE 专属排坑与高阶策略

<details>
<summary><b>🚨 避坑指南：为什么绝对不能微调路由器 (Router)?</b></summary>

在 MoE 模型中，路由器（通常命名为 `mlp.gate`，注意不是 `gate_proj`）输出的是用于 Softmax 或 Sigmoid 的 Logits，用来决定 Token 分配给哪个专家。

如果在首次 SFT 时用 LoRA 微调路由器：
1.  **破坏负载均衡：** 预训练阶段好不容易学到的专家负载均衡（Load Balancing）会被瞬间打破，导致大量 Token 涌向极少数专家（Token Dropping / Routing Collapse）。
2.  **遗忘专家特长：** 模型会将本该由“代码专家”处理的请求错误路由给“闲聊专家”，导致模型智商骤降。
3.  **Loss 震荡：** 路由的微小改变会导致网络拓扑结构的剧变，使得 SFT 阶段的 Loss 呈现极不稳定的锯齿状。

**结论：** 保持路由器参数冻结，让它沿用预训练时的路由策略。

</details>

<details>
<summary><b>💡 显存优化：如果全量专家 LoRA 导致 OOM 怎么办？</b></summary>

如果你使用的是 8x H800 等集群，上述配置没有问题。但如果算力受限（例如只有两三张卡微调 DeepSeek-V2-Lite），对所有路由专家做高秩 LoRA 可能会导致显存爆炸（OOM）。

**妥协方案（按效果降序排列）：**

1.  **降秩保全面：** 将秩 $r$ 降低到 $32$ 或 $64$，但依然保留 `"gate_proj", "up_proj", "down_proj"` 以覆盖所有专家。
2.  **仅微调共享专家 (Shared Experts Only)：**
    *   *原理：* 共享专家参与每个 Token 的计算，修改它们能最快地改变模型的整体输出格式。
    *   *修改代码：* 不要直接写 `"gate_proj"`，而是写完整的正则匹配，避开独立的 experts。
    ```python
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        # 仅匹配共享专家和浅层的常规 MLP
        "shared_experts.*_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" 
    ]
    ```
3.  **退回 Attention-Only：** 仅微调 `["q_proj", "v_proj"]`，但在首次大规模 SFT 中，强烈不建议这样做，模型往往无法学会复杂的指令边界。

</details>

---

### 🧑‍💻 User

lora不就是节约显存的吗，为什么moe相比dense反而容易显存爆炸？

---

### 🤖 Assistant

这是一个非常经典且直觉上容易产生误解的问题。

我们常说“LoRA 能节约显存”，这句话的完整前提是：**在同等参数规模的模型下，LoRA 相比全参数微调（FFT）能大幅节约显存。**

但在 MoE（混合专家）模型与 Dense（稠密）模型的对比中，很多人混淆了**“激活参数量（Active Parameters，决定计算速度）”**和**“总参数量（Total Parameters，决定显存占用）”**。

导致对 MoE 全量专家应用 LoRA 容易引发 OOM（显存溢出）的核心原因，主要有以下四个维度：

### 1. MoE 的“静态显存”基数极其庞大

MoE 模型的特点是“推理快但极占显存”。模型在处理一个 Token 时，可能只激活 $2$ 个专家（算力消耗等同于一个小模型），**但物理显存必须把所有 $64$ 个专家（甚至更多）全部加载进去。**

*   **Dense 模型举例：** 一个典型的 $14\text{B}$ Dense 模型，半精度加载只需要约 $28\text{GB}$ 显存。
*   **MoE 模型举例：** 以 DeepSeek-V2 为例，激活参数只有 $21\text{B}$（感觉是个中等模型），但它的**总参数量高达 $236\text{B}$**！光是把这个基座模型以半精度（BF16/FP16）加载进显存，就需要将近 $472\text{GB}$ 的显存。

LoRA 虽然只训练极少量的增量参数，但**基座模型依然需要驻留在显存中**参与前向传播。MoE 庞大的底座直接吃掉了绝大部分显存预算。

### 2. LoRA Adapter 数量的“乘数效应” (核心原因)

当我们在 Dense 模型中设置 `target_modules=["gate_proj", "up_proj", "down_proj"]` 时，每一层只需要挂载 $3$ 个 LoRA 矩阵。

但如果在 MoE 模型中设置相同的 target，情况就完全不同了。以一个拥有 $64$ 个路由专家的 MoE 模型为例：

*   **Dense 的一层：** $3$ 个权重矩阵。
*   **MoE 的一层：** 每个专家内部都有这 $3$ 个矩阵。因此一层总共有 $64 \times 3 = 192$ 个权重矩阵。

由于 PEFT 的匹配机制，**LoRA 会给这 $192$ 个矩阵每一个都外挂一个低秩矩阵 $A$ 和 $B$！**
这意味着，在相同的秩 $r$ 下，MoE 模型的 MLP 层产生的 LoRA 可训练参数量，是同规模 Dense 模型的 **$64$ 倍**。

### 3. 优化器状态 (Optimizer States) 的膨胀

LoRA 节约显存的核心在于减少了“可训练参数”，从而减少了 Adam 等优化器需要保存的状态（动量和方差，通常是 FP32，每个参数占 $8$ 字节）。

正如第二点所述，MoE 全量专家微调时，LoRA 参数量会激增几十倍。
假设你的 LoRA 参数在 Dense 模型上是 $1$ 亿（占用约 $800\text{MB}$ 优化器显存），在 MoE 模型上可能变成 $64$ 亿。那么仅仅是 **Adam 优化器的状态就会占用约 $50\text{GB}$ 显存**，这足以撑爆一张 $80\text{G}$ 的 A100/H800。

---

<details>
<summary><b>点击展开：4. 动态路由导致的反向传播激活值 (Activations) 碎片化</b></summary>

在微调训练时，反向传播（Backward Pass）需要依赖前向传播时保存的激活值（Activations）。

*   **Dense 模型：** 矩阵乘法是规则的稠密运算，激活值的显存占用是可预测且连续的。
*   **MoE 模型：** 由于采用了动态路由，同一 Batch 内的不同 Token 会被发送到不同的专家。这会在底层导致大量的 `gather` 和 `scatter` 操作，产生极其复杂的计算图和显存碎片。为了计算那些挂在冷门专家上的 LoRA 梯度的反向传播，PyTorch 需要缓存大量额外的上下文信息。

当 Batch Size 或者 Sequence Length 稍微增大时，这种由 MoE 路由机制带来的额外 Activation 显存开销会呈指数级增长。

</details>

### 总结与应对策略

综上所述，**MoE 只是算力上的“伪小模型”，在显存占用上是不折不扣的“巨无霸”。** 对它的所有专家挂载 LoRA，相当于同时在微调几十个小模型的集合，显存不爆炸才怪。

如果你在对 MoE 首次 SFT 时遇到 OOM，回顾上一次对话中提到的妥协方案即可解决：
1.  **首选方案：** 仅微调 Attention 层和 Shared Experts（共享专家），放弃对 Routed Experts（路由专家）内部添加 LoRA。这能直接砍掉 $90\%$ 以上的 LoRA 参数。
2.  **备选方案：** 如果必须微调全量专家，将 秩 $r$ 压低到 $16$ 甚至 $8$。
3.  **系统级优化：** 开启 Gradient Checkpointing（梯度检查点，必须开），使用 DeepSpeed Zero2/3，或者使用分页优化器（paged_adamw_8bit）。