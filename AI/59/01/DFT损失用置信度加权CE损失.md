# DFT损失用置信度加权CE损失

### 🧑‍💻 User

```
# CE Loss
def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> torch.Tensor:
    """Compute token-level next-token loss with optional label smoothing."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)

    with tensor_parallel_loss_context(loss_parallel_enabled):
        return F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )


def compute_standard_cross_entropy_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute standard CE loss and manually add MoE auxiliary loss."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_router_logits=True
    )

    ce_loss = compute_cross_entropy_loss(
        outputs.logits,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        loss_parallel_enabled=loss_parallel_enabled,
    )
    loss = ce_loss

    aux_loss_computer = MoEAuxLossComputer(model)
    aux_loss, aux_metrics = aux_loss_computer.compute(outputs, attention_mask)
    if aux_loss is not None:
        loss = loss + aux_loss.to(loss.device)

    metrics = collect_moe_metrics(outputs)
    metrics.update(aux_metrics)
    metrics["lm_loss"] = ce_loss.detach()
    metrics["_policy_outputs"] = outputs

    return loss, metrics


def selective_log_softmax(logits, index) -> torch.Tensor:
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    # for index with shape (...):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    # for index with shape (..., K):
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(..., K)` or `(...)`, specifying the positions to gather from the log-softmax
            output. When the last case is used, `K` log-probabilities are gathered per position (e.g. for top-K)

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:]
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


def compute_standard_sft_loss(
    loss_type: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    loss_parallel_enabled: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute SFT/DFT loss based on the specified loss type.
  
    Args:
        loss_type: Type of loss to compute. Options:
            - "sft": Standard cross-entropy loss
            - "dft": DFT loss from paper "On the Generalization of SFT"
        model: The policy model
        batch: Input batch containing at minimum 'input_ids'
        labels: Target labels for next-token prediction
        ignore_index: Token id to ignore in loss computation
        label_smoothing: Label smoothing factor for CE loss
        loss_parallel_enabled: Whether to use loss parallelism for tensor parallel
      
    Returns:
        Tuple of (loss, metrics_dict)
    """
    if loss_type == "nll":
        return compute_standard_cross_entropy_loss(
            model=model,
            batch=batch,
            labels=labels,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            loss_parallel_enabled=loss_parallel_enabled,
        )
    elif loss_type == "dft":
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
      
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=True
        )

        # Compute DFT loss
        loss = dft_loss(outputs, labels)

        # Compute MoE auxiliary loss if applicable
        aux_loss_computer = MoEAuxLossComputer(model)
        aux_loss, aux_metrics = aux_loss_computer.compute(outputs, attention_mask)
        if aux_loss is not None:
            loss = loss + aux_loss.to(loss.device)

        # Gather metrics
        metrics = collect_moe_metrics(outputs)
        metrics.update(aux_metrics)
        metrics["lm_loss"] = loss.detach()
        metrics["_policy_outputs"] = outputs

        return loss, metrics
    else:
        raise ValueError(f"Unknown SFT loss_type: {loss_type}. Expected 'nll' or 'dft'.")
```
dft loss和一般的celoss有什么区别？

---

### 🤖 Assistant

**DFT loss 与标准 CE（NLL）loss 的核心区别**

### 1. 标准 Cross-Entropy Loss（代码中的 `nll` / CE）

```python
# 本质就是：
loss = -log p(y | x)     # 对正确 token 的负对数似然
```

- 使用 `F.cross_entropy`（内部是 log_softmax + NLL）。
- 对每个有效 token 直接优化最大化正确 token 的概率。
- 梯度方向始终是“把正确 token 的 logit 拉高”，**权重固定为 1**（或 label smoothing 后的软标签）。

### 2. DFT Loss（Dynamic Frequency Temperature / 论文提出的版本）

```python
logprobs = selective_log_softmax(outputs.logits, shift_labels)  # log p(y|x)
per_token_loss = -logprobs.exp().detach() * logprobs            # - p(y|x).detach() * log p(y|x)
```

即：

$$
\mathcal{L}_{\text{DFT}} = -\, p(y|x)_{\text{detach}} \cdot \log p(y|x)
$$

或者写成：

$$
\mathcal{L}_{\text{DFT}} = p(y|x)_{\text{detach}} \cdot \mathcal{L}_{\text{CE}}
$$

**关键点**：
- 用模型**当前**对正确 token 的预测概率 \(p(y|x)\)（**detach** 掉）去重新加权标准 CE loss。
- \(p\) 越大 → 该 token 的 loss 权重越大；\(p\) 很小 → 权重被压低。
- 因为 detach 了，梯度只通过 \(\log p\) 回传，但被当前 \(p\) 缩放。

### 3. 直观理解与动机（来自论文视角）

论文标题：*On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*

从 RL 角度看 SFT：
- 标准 SFT 相当于把每个正确 token 的 reward 设为 1（或常数），然后做 policy gradient 的某种形式。
- 这会导致模型对**已经很有把握的 token**（高 \(p\)）和**还很不确定的 token**（低 \(p\)）给予相同强度的更新。
- DFT 通过乘上 \(p_{\text{detach}}\)，相当于做了一种 **reward rectification（奖励校正）**：
  - 对模型已经学得比较好的 token，给予更强的强化（高 \(p\) 权重高）。
  - 对模型还很不确定/难的 token，降低其更新强度，避免过度拟合噪声或困难样本。

这有点类似：
- Focal Loss 的思想（但方向相反，focal 是压低易分样本）。
- 或者 importance sampling / 自适应温度的某种形式。
- 也被解释为一种动态的、与模型置信度相关的 label weighting。

### 4. 代码层面的其他细节差异

| 方面              | 标准 CE (`nll`)                          | DFT                                      |
|-------------------|------------------------------------------|------------------------------------------|
| 实现方式          | `F.cross_entropy`                        | 手动 `selective_log_softmax` + 加权      |
| 是否支持 label smoothing | 支持                                     | 不支持（代码中无）                       |
| 对齐方式          | `logits[..., :-1]` vs `labels[..., 1:]` | 通过 pad + shift 实现等效对齐            |
| 归一化            | 默认 mean over non-ignored tokens        | 手动 `sum / num_items_in_batch`（等效）  |
| MoE 辅助损失      | 两者都加                                 | 两者都加                                 |
| 内存优化          | 标准实现                                 | `selective_log_softmax` 更省显存         |

### 5. 行为上的实际影响

- **高置信度 token**（模型已经预测得很准）：DFT 会给更大的梯度，进一步巩固。
- **低置信度 token**（模型还在犹豫或预测错）：DFT 会减小其贡献，训练更“保守”，有利于泛化（论文主张的点）。
- 训练初期 \(p\) 普遍较小 → DFT 整体 loss 尺度较小，后期随着 \(p\) 上升，有效 loss 权重逐渐增大。
- 对过拟合敏感的场景（尤其是 SFT 数据质量参差不齐时），DFT 通常比纯 CE 更稳、泛化更好。

### 总结一句话

**标准 CE**：均匀地最大化每个正确 token 的 log-prob。
**DFT**：用当前模型对该 token 的置信度 \(p\) 重新加权 CE，相当于做了置信度自适应的奖励校正，目标是提升 SFT 的泛化能力。

如果你需要，我可以进一步推导两者的梯度差异，或者对比它们在训练动态上的表现。

---

### 🧑‍💻 User

dft loss可以用于pretrain吗？

---

### 🤖 Assistant

**可以技术上使用，但强烈不推荐用于标准 Pretrain。**

### 简短结论
DFT Loss 是专门为 **SFT（Supervised Fine-Tuning）** 设计的，从 RL 视角对 reward 做校正，以提升 SFT 的泛化能力。它与 Pretraining 的目标和训练动态存在明显冲突，直接用于 Pretrain 大概率会**变差或训练变慢**。

### 为什么不适合 Pretrain？

| 维度              | 标准 CE Loss（Pretrain 常用）          | DFT Loss                                      | 影响 |
|-------------------|---------------------------------------|-----------------------------------------------|------|
| **核心机制**      | \(-\log p(y\|x)\)                      | \(-p(y\|x)_{\text{detach}} \cdot \log p(y\|x)\) | DFT = 当前置信度加权的 CE |
| **早期训练**      | 正常梯度，能有效学习                   | \(p \approx 1/V\) 极小 → 梯度几乎消失          | 训练极慢甚至卡住 |
| **对 hard token** | 给予正常压力                           | 低 \(p\) 的 token 权重被压低                   | 长尾知识、罕见模式学习变差 |
| **对 easy token** | 后期自然变小                           | 高 \(p\) 的 token 权重被放大                   | 过度巩固已学好的常见模式 |
| **数据特性**      | 大规模、噪声大、分布广                 | 更适合高质量、相对干净的 SFT 数据              | Pretrain 数据不匹配 |
| **目标**          | 学习完整数据分布、能力涌现             | 提升 SFT 泛化（reward rectification）          | 目标不一致 |

### 详细原因解释

1. **早期训练致命问题**
   - Pretrain 初期模型接近随机，正确 token 的 \(p(y|x)\) 大约是 \(1/\text{vocab_size}\)（32k~128k 量级）。
   - DFT 会把 loss 整体缩小几个数量级，有效学习信号极弱。
   - 标准 CE 在这个阶段是最稳健的。

2. **与 Pretrain 目标相反**
   - Pretrain 需要尽可能均匀地覆盖数据分布，尤其要学好**低频、困难、长尾** token 和模式。
   - DFT 会自动给模型已经比较有把握的 token 更高权重，相当于“马太效应”，更容易让模型沉溺于高频模式，损害泛化与知识覆盖。

3. **和 Focal Loss 等对比**
   - Focal Loss 是**压低 easy 样本**，聚焦 hard 样本（更接近 Pretrain 可能想要的方向）。
   - DFT 是**抬高 easy 样本**（高 \(p\)），方向完全相反。

4. **论文定位**
   - 论文标题明确是 *On the Generalization of SFT*，核心是把 SFT 看作带 reward 的 RL 过程，然后做 rectification。
   - 并没有针对 Pretrain 的实验或主张。

### 什么情况下可以考虑用 DFT？

- **Continued Pretraining / Mid-training**：已经过了随机初始化阶段，模型有一定能力，数据质量较高时。
- **Annealing 阶段**（最后的高质量数据微调阶段）：类似 SFT 的性质。
- **高资源实验探索**：想验证“置信度加权”在 Pretrain 后期是否有巩固作用。
- **Domain-adaptive Pretrain**：数据相对干净、领域明确时。

即使在这些场景，也建议：
- 先 warmup 用标准 CE，再切换到 DFT。
- 或者把 \(p\) 做温度缩放 / clip（例如 \(\max(p, \tau)\) 或 \(p^\alpha\)），降低早期影响。
- 严格对比验证（loss 曲线、下游能力、长尾知识等）。

### 代码层面

当前代码结构已经把 DFT 放在 `compute_standard_sft_loss` 里，和 `nll` 并列，说明作者也是把它当作 SFT 变体：

```python
if loss_type == "nll":
    ...
elif loss_type == "dft":
    ...
```

如果要强行用于 Pretrain，需要额外处理：
- 早期训练的数值稳定性
- 与 packing、document masking、MoE 等的兼容
- 学习率、warmup 策略的重新调整

### 建议

- **标准 Pretrain**：继续用 CE（或已知有效的变体，如带 Z-loss、label smoothing 等）。
- **SFT / Preference Tuning 阶段**：可以认真尝试 DFT，论文主打的就是这里的泛化收益。
- 如果想改进 Pretrain loss，更值得看的方向是：数据质量、curriculum、token-level reweighting based on difficulty/rarity、self-distillation 等，而不是直接搬 DFT。

需要我帮你改一版更适合 Pretrain 的加权 CE 变体（比如加入温度、clip、或者结合 rarity 的版本），或者分析 DFT 的梯度具体形式，可以继续说。

---

### 🧑‍💻 User

为什么完全随机初始化的模型用dft会出现train loss gradnorm几乎为0但val loss高的问题？

---

### 🤖 Assistant

**根本原因：随机初始化时 \(p(y|x) \approx 1/V\)，DFT 把 loss 和梯度同时缩小了约 \(V\) 倍。**

### 1. DFT Loss 的数学形式

```python
logprobs = selective_log_softmax(...)          # log p(y|x)
per_token_loss = -logprobs.exp().detach() * logprobs
# 即 L = - p_detach * log p
```

等价于：

$$
\mathcal{L}_{\text{DFT}} = p(y|x)_{\text{detach}} \cdot \mathcal{L}_{\text{CE}}
$$

其中 \(p = \mathrm{softmax}(\text{logits})\) 的正确 token 概率。

### 2. 完全随机初始化时的数值

假设 vocab size \(V = 32000 \sim 128000\)（常见范围）：

- 模型输出接近均匀分布 → 正确 token 的 \(p_* \approx \dfrac{1}{V} \approx 3\times10^{-5} \sim 8\times10^{-6}\)
- \(\log p_* \approx -\log V \approx -10.4 \sim -11.8\)
- **DFT loss** \(\approx p_* \cdot \log V \approx 3\times10^{-4} \sim 1\times10^{-4}\)
  → 日志里看到的 train loss 几乎为 0（相比标准 CE 的 ~10.5）

### 3. 为什么 GradNorm 也几乎为 0

因为 \(p_{\text{detach}}\) 被 detach 了，它对梯度是常数：

$$
\nabla \mathcal{L}_{\text{DFT}} = p_* \cdot \nabla \mathcal{L}_{\text{CE}}
$$

标准 CE 的梯度量级是 O(1)，乘上 \(p_* \approx 1/V\) 后：

- 梯度整体被缩小 **3万~12万倍**
- GradNorm 自然也接近 0

（Adam 等自适应优化器虽然有二阶动量，但在这么极端的小梯度下，有效更新步长仍然极小，模型几乎不移动。）

### 4. 为什么 Val Loss 却很高？

常见有两种情况（你遇到的大概率是第一种）：

1. **Validation 用的是标准 CE / NLL / Perplexity**（最常见）
   因为模型几乎没有更新（梯度太小），参数仍接近随机初始化 → 标准 CE 仍然 ≈ \(\log V\)（很高）。
   而 train loss 报的是 DFT 值（极小），所以出现“train loss ≈ 0，val loss 很高”的割裂现象。

2. **Validation 也用 DFT**
   此时 val loss 数值也会很小，但模型能力极差（准确率接近随机），如果你额外看 token accuracy 或真正的 CE，会发现很高。很多人把“模型很差”描述成 val loss 高。

### 5. 对比表格（随机初始化阶段）

| 指标              | 标准 CE Loss          | DFT Loss                     | 比例          |
|-------------------|-----------------------|------------------------------|---------------|
| Loss 数值         | \(\approx \log V \approx 10.5\) | \(\approx (\log V)/V \approx 3e-4\) | \(1/V\)      |
| 梯度幅度          | O(1)                  | O(1/V)                       | \(1/V\)      |
| GradNorm          | 正常                  | 几乎为 0                     | \(1/V\)      |
| 参数更新量        | 正常                  | 几乎为 0                     | 极小         |
| 模型是否学习      | 是                    | 基本不学习                   | -            |

### 6. 额外的实际问题

- **Loss scale 随训练剧烈变化**：早期 \(p\) 很小 → loss 极小；后期 \(p\) 变大 → loss 尺度上升。这会让学习率调度、梯度裁剪、混合精度 loss scaling 全部失效。
- **数值稳定性**：极小的 loss + 极小的梯度，在 bf16/fp16 下更容易出现 underflow 或无效更新。
- **与 MoE / 其他辅助损失的交互**：如果 aux loss 是正常量级，会被 DFT 的主 loss 淹没或反过来主导。

### 总结一句话

完全随机初始化时，正确 token 概率 \(p \approx 1/V\)，DFT 相当于给标准 CE 乘了一个 \(1/V\) 的系数，导致 **train loss 和梯度同时缩小 V 倍** → 日志上 loss/gradnorm 接近 0，模型几乎不更新，所以验证集上的真实预测能力（CE/Perplexity）依然很差（很高）。

这就是为什么 DFT **绝对不适合从随机初始化开始的 Pretrain**，只适合已经有一定能力的模型（SFT 阶段）。

---

**如果想强行在早期使用类似思想**，可以做这些修正（供参考）：
- 对 \(p\) 做 clip 或温度：`p_clipped = max(p, τ)` 或 `p ** α`
- 前期用 CE，warmup 后再切换 DFT
- 把权重改成 \(p / \mathbb{E}[p]\) 做归一化，保持 loss 尺度稳定

需要我帮你改一版更稳健的 DFT 变体代码吗？