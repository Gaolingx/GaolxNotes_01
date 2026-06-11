# LigerKernel导致token指标不可见

### 🧑‍💻 User

用了ligerceloss kernel之后所有tokenlevel metrics看不到吗？

---

### 🤖 Assistant

是的，简单来说，**使用了 Liger Kernel 的 Fused Cross Entropy Loss（特别是融合了 LM Head 的版本）之后，通常无法直接计算和查看 Token-level 的 metrics（例如 Token Accuracy、每 Token 的 Perplexity 等）。**

这并非 bug，而是 Liger Kernel 为了极致优化显存和计算效率而做出的**架构上的妥协**。

以下是具体原因以及可行的解决方案：

### 为什么 Token-level Metrics 会消失？

Liger Kernel 的核心优化思想是**避免在显存 (HBM) 中实例化庞大的 Logits 张量**。

在传统的 PyTorch 训练中，流程如下：
1. 模型输出 `hidden_states`，维度为 $B \times S \times H$。
2. 经过 LM Head（线性层），输出完整的 `logits`，维度为 $B \times S \times V$（$V$ 是词表大小，通常在 30k 到 130k 之间）。**这一步会占用巨大的显存。**
3. 计算 Token-level accuracy：通过 `argmax(logits, dim=-1)` 得到预测的 Token ID，再与 Labels 对比。
4. 计算 Loss：将 `logits` 传入 `CrossEntropyLoss`。

**Liger Kernel (例如 `LigerFusedLinearCrossEntropyLoss`) 的做法：**
Liger 使用 Triton 编写了底层 Kernel，将**步骤 2 和 步骤 4 融合 (Kernel Fusion)**。它在 SRAM 中分块 (chunk) 读取 `hidden_states` 和 `weights`，直接在片上计算出局部的 Logits 并立刻计算出 Loss，然后**直接丢弃局部的 Logits**，最后只向显存返回一个标量 Loss 值。

因为完整的 $B \times S \times V$ Logits 张量从未在显存中生成过，你自然也就无法执行 `argmax` 操作，从而**无法计算 Token Accuracy 等依赖预测概率的指标**。

---

### 如何解决这个问题？

如果你在训练过程中必须监控 Token-level metrics，可以参考以下几种替代方案：

<details>
<summary><b>方案一：仅在 Evaluation 阶段关闭 Liger Kernel（推荐）</b></summary>
训练阶段（Training）对显存要求极高（需要保存激活值用于反向传播），而评估阶段（Evaluation）只需要前向传播，显存压力通常小得多。

**做法：** 
在 Training 时使用 Liger Kernel，在 Validation/Evaluation 时切换回原生的 PyTorch `nn.CrossEntropyLoss` 和标准的 LM Head。这样你在验证集上依然可以看到完整的 Token-level metrics。

```python
if is_training:
    # 训练时使用 Liger 节省显存
    loss = liger_fused_linear_cross_entropy(hidden_states, weight, labels)
else:
    # 评估时使用原生 PyTorch 获取 Logits 计算 metrics
    logits = F.linear(hidden_states, weight)
    loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    preds = torch.argmax(logits, dim=-1)
    # 计算 token accuracy...
```
</details>

<details>
<summary><b>方案二：不使用 Fused 版本，仅使用 Chunked Cross Entropy</b></summary>
如果你不使用 `LigerFusedLinearCrossEntropyLoss`（融合了线性层），而是仅使用 `LigerCrossEntropyLoss`，这意味着你仍然需要在外部计算出完整的 `logits`。

只要你生成了 `logits`，你在将其传入 Liger 的 CE Loss 之前，就可以先计算你的 metrics。
*注意：这会丧失 Liger Kernel 很大一部分的显存优化效果（因为 logits 还是被实例化了），只保留了计算 CE 时的显存优化。*
</details>

<details>
<summary><b>方案三：修改 Liger Kernel 返回 Argmax (高阶玩法)</b></summary>
如果你有修改 Triton Kernel 的能力，可以修改 Liger 的前向传播 Kernel。
在 SRAM 中计算完局部 Logits 时，除了计算 Loss，顺便计算一下局部的 `argmax` 并将其写入显存返回。返回一个 $B \times S$ 的预测 ID 张量占用的显存极小。
*目前 Liger 官方版本默认注重极致性能，不返回这个值，但社区中已有相关的讨论和 Pull Request 尝试引入这个功能。*
</details>

### 总结

*   **要性能和显存：** 接受 Training 阶段没有 Token-level metrics 的现实，只看整体 Loss。
*   **要指标监控：** 在 Eval 阶段切回原生 PyTorch 代码，或者接受显存开销不使用 Fused Linear 版本。