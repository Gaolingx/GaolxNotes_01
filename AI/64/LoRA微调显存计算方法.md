# LoRA微调显存计算方法

### 🧑‍💻 User

llms lora微调需要的显存大小如何计算？

---

### 🤖 Assistant

计算大语言模型（LLM）进行 LoRA 微调所需的显存大小，主要取决于**模型参数量**、**量化精度**、**批次大小**、**序列长度**以及**优化器类型**。

由于 LoRA 仅更新一小部分旁路矩阵，显存占用远低于全量微调。我们可以将总显存需求拆解为以下几个核心部分。

### 核心显存计算公式

$$
VRAM_{total} \approx VRAM_{weights} + VRAM_{lora\_states} + VRAM_{activations} + VRAM_{overhead}
$$

<details>
<summary><b>📖 详细公式拆解（点击展开）</b></summary>

1. **模型权重显存 ($VRAM_{weights}$)**
   这部分是最大的显存开销。取决于加载模型的精度。
   $$VRAM_{weights} = P \times B_{model}$$
   - $P$: 模型参数量（例如 7B 代表 $7 \times 10^9$）
   - $B_{model}$: 每个参数占用的字节数。16-bit (fp16/bf16) 为 2 字节；8-bit 为 1 字节；4-bit (QLoRA) 为 0.5 字节。

2. **LoRA 训练状态显存 ($VRAM_{lora\_states}$)**
   包含 LoRA 权重本身、梯度和优化器状态。
   $$VRAM_{lora\_states} \approx P_{lora} \times (B_{weight} + B_{grad} + B_{opt})$$
   - $P_{lora}$: LoRA 参数量（通常为原模型的 $0.1\% \sim 1\%$）
   - $B_{weight}$: LoRA 权重精度，通常为 2 字节 (fp16/bf16)
   - $B_{grad}$: 梯度精度，通常为 2 字节
   - $B_{opt}$: 优化器状态。AdamW 8-bit 约需 2 字节；标准 AdamW 需 8 字节（保留动量和方差）。
   *结论：由于 $P_{lora}$ 很小，这部分通常仅占几百 MB 到 1-2 GB。*

3. **激活值显存 ($VRAM_{activations}$)**
   前向传播过程中保留的中间变量，用于反向传播计算梯度。与批次大小和序列长度成正比。
   $$VRAM_{activations} \propto B \times S \times H \times L$$
   - $B$: Batch Size (批次大小)
   - $S$: Sequence Length (序列长度)
   - $H$: Hidden Size (隐藏层维度)
   - $L$: Layers (Transformer层数)
   *注：开启梯度检查点会将该部分显存大幅降低，代价是约 20% 的计算时间。*

4. **固定开销 ($VRAM_{overhead}$)**
   CUDA 上下文、PyTorch 预分配缓存等，通常预留 $1 \sim 1.5$ GB。

</details>

---

### 实用显存估算表

为了方便快速参考，以下是基于**序列长度 2048**、**AdamW 8-bit 优化器**、**开启梯度检查点**的典型显存需求估算表：

| 模型规模 | 精度 (量化) | 模型权重显存 | Batch Size = 1 估算显存 | Batch Size = 4 估算显存 | 推荐显卡 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.5B** | 16-bit (fp16) | ~3.0 GB | ~5 GB | ~8 GB | RTX 3060 12G |
| **1.5B** | 4-bit (QLoRA) | ~1.0 GB | ~3 GB | ~5 GB | RTX 3060 8G |
| **7B** | 16-bit (fp16) | ~14.0 GB | ~18 GB | ~24 GB | RTX 3090 / 4090 |
| **7B** | 4-bit (QLoRA) | ~4.0 GB | ~7 GB | ~12 GB | RTX 3090 / 4080 |
| **13B** | 16-bit (fp16) | ~26.0 GB | ~32 GB | ~46 GB | A6000 / A100 |
| **13B** | 4-bit (QLoRA) | ~8.0 GB | ~12 GB | ~18 GB | RTX 3090 / 4090 |
| **70B** | 16-bit (fp16) | ~140.0 GB | ~160 GB+ | ~200 GB+ | 多卡 A100/H100 |
| **70B** | 4-bit (QLoRA) | ~35.0 GB | ~42 GB | ~55 GB | 双 RTX 3090 或 A6000 |

*注意：以上为估算值，实际显存会因框架实现（如 Flash Attention 2 的使用）略有波动。*

---

### 降低显存的关键技术手段

如果你发现显存不足，可以通过在 `TrainingArguments` 或框架配置中调整以下参数来大幅降低显存消耗：

<details>
<summary><b>⚙️ 核心降显存代码配置（点击展开）</b></summary>

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 1. 梯度检查点：用计算时间换显存，大幅降低 Activation 显存（必开）
    gradient_checkpointing=True,
  
    # 2. 梯度累积：在保持等效 Batch Size 的同时降低单步显存峰值
    per_device_train_batch_size=1,          # 物理批次设为 1
    gradient_accumulation_steps=8,          # 累积 8 步，等效 batch_size=8
  
    # 3. 优化器选择：使用 8-bit Adam 替代原生 AdamW
    # 需要安装 bitsandbytes
    optim="adamw_8bit", 
  
    # 4. 混合精度训练：bf16 通常比 fp16 更稳定且显存占用一致
    bf16=True, 
  
    # 5. 开启 Flash Attention 2 (需模型支持且安装 flash-attn)
    # 这会降低注意力机制部分的显存占用，特别是长序列时
    attn_implementation="flash_attention_2",
)
```

**关于 QLoRA 加载配置：**
```python
from transformers import BitsAndBytesConfig
import torch

# 使用 4-bit 量化加载基础模型，大幅减少 VRAM_weights
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, # 二次量化，再省一点显存
)
```
</details>

### 总结

计算 LoRA 微调显存最简单的心智模型是：
1. **查参数量**：确定模型有多少 B（十亿）参数。
2. **看量化**：决定加载策略。16-bit 乘 2，4-bit 乘 0.5。这决定了基础显存底座。
3. **留冗余**：在基础显存底座上，加上 $1\sim3$ GB 的 LoRA 优化器状态，加上 $2\sim5$ GB 的激活值显存（取决于序列长度和 Batch Size），再加 $1$ GB 的框架开销。

**例如：** 运行 7B 模型，底座是 4-bit（约 3.5GB），加上其他开销 $5$ GB，总共约需要 $8\sim9$ GB 显存即可启动。这也是为什么 RTX 3060 12G 可以勉强跑 7B 模型的 QLoRA 微调的原因。

---

### 🧑‍💻 User

看来小rank瓶颈主要在模型权重和激活值吗？

---

### 🤖 Assistant

是的，您的理解非常准确！在 LoRA（特别是小 rank，如 $r=8$ 或 $r=16$）微调中，LoRA 本身的训练状态显存几乎可以忽略不计，真正的显存瓶颈确实只剩下**基础模型权重**和**激活值**。

我们可以通过具体的数学拆解来看看为什么会这样。

### 1. 为什么 LoRA 状态显存可忽略？

假设我们微调一个 7B 模型，隐藏层维度 $d=4096$，使用较小的秩 $r=8$。

LoRA 主要加在 $q, k, v, o$ 投影矩阵上。对于单层 Transformer，单个矩阵（如 $W_q$）的 LoRA 参数量为 $2 \times d \times r = 2 \times 4096 \times 8 = 65,536$。
四层投影总计：$4 \times 65,536 = 262,144$。
假设有 32 层 Transformer，总 LoRA 参数量约为 $8.4 \times 10^6$（约 8.4M）。

相比于 7B（$7,000M$）的原始参数量，LoRA 参数仅占约 $0.12\%$。

我们计算一下这 8.4M 参数的训练状态显存：
$$
VRAM_{lora\_states} = P_{lora} \times (B_{weight} + B_{grad} + B_{opt})
$$
- 权重 ($B_{weight}$，bf16): 2 字节
- 梯度 ($B_{grad}$，bf16): 2 字节
- 优化器 ($B_{opt}$，AdamW 8-bit): 2 字节
- 总计：6 字节/参数

$$
VRAM_{lora\_states} = 8.4 \times 10^6 \times 6 \text{ Bytes} \approx 50.4 \text{ MB}
$$
**结论**：即使在 7B 模型上，小 rank 的 LoRA 状态也只占约 $50$ MB。这在动辄十几 GB 的显存总开销中，完全属于四舍五入的误差。

---

### 2. 两大真正的瓶颈

既然 LoRA 状态不占显存，显存去哪儿了？全部被**静态的模型权重**和**动态的激活值**吃掉了。

#### 瓶颈一：基础模型权重（静态开销）
这部分显存在加载后就固定了，决定了显存的“底座”。

- **不量化 (16-bit)**：7B 模型需要 $7 \times 2 = 14$ GB。
- **QLoRA (4-bit)**：7B 模型需要 $7 \times 0.5 = 3.5$ GB。

**对策**：如果这部分显存不够，唯一的办法就是降低量化精度（如使用 4-bit 的 `nf4` 量化）。这也是为什么 QLoRA 如此流行的原因——它直接把底座显存砍掉了 75%。

#### 瓶颈二：激活值（动态开销）
这部分显存在前向传播时产生，用于反向传播计算梯度。它随着 Batch Size ($B$) 和 Sequence Length ($S$) 线性增长，是导致训练过程中 `OOM (Out Of Memory)` 的最常见元凶。

在不开启梯度检查点时，激活值显存粗略估算为：
$$
VRAM_{activations} \approx L \times B \times S \times H \times \text{常数}
$$
- $L$ (层数): 32
- $B$ (批次): 1
- $S$ (序列长度): 2048
- $H$ (隐藏维度): 4096

在长序列（如 $S=4096$ 或 $8192$）时，激活值显存甚至可能超过未量化的模型权重显存。

<details>
<summary><b>🔬 激活值显存的优化策略（点击展开）</b></summary>

既然瓶颈在激活值，我们就必须想办法“压榨”它。目前业界有三种主要手段：

1. **梯度检查点**
   - **原理**：不保存所有前向传播的激活值，只保存部分节点的。在反向传播时，重新计算缺失的激活值。
   - **效果**：将激活值显存从 $\mathcal{O}(L)$ 降到 $\mathcal{O}(\sqrt{L})$，**大幅降低显存**。
   - **代价**：增加约 20%-30% 的训练时间（用算力换显存）。

2. **Flash Attention 2**
   - **原理**：优化注意力机制的计算过程，避免将巨大的注意力矩阵 $S \times S$ 显式写入显存（HBM）。
   - **效果**：在长序列下显著降低激活值显存，且**不增加计算时间，甚至加速训练**。

3. **极限参数配置**
   在 `TrainingArguments` 中调小物理显存占用：
   - 调小 `per_device_train_batch_size`（极限压榨可设为 1）。
   - 使用 `gradient_accumulation_steps` 补偿等效批次大小。
   - 限制 `max_seq_length`（如限制在 1024 或 2048）。
</details>

### 总结

您看得非常透彻。在小 rank LoRA 场景下：
1. **LoRA 训练状态** $\rightarrow$ 几乎为 0，可忽略。
2. **模型权重** $\rightarrow$ 静态瓶颈，决定了能不能“装下”模型（靠 4-bit 量化解决）。
3. **激活值** $\rightarrow$ 动态瓶颈，决定了能不能“跑起来”数据（靠梯度检查点和 Flash Attention 解决）。