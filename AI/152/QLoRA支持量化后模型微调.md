# QLoRA支持量化后模型微调

### 🧑‍💻 User

qlora支持用量化后模型进行微调吗？

---

### 🤖 Assistant

是的，**QLoRA 的核心设计理念就是支持并利用量化后的模型进行高效微调**。

简单来说，QLoRA 的工作流程如下：

1.  **加载预训练模型：** 首先加载一个标准的、通常是全精度的预训练大语言模型。
2.  **量化模型权重：** 在微调开始前，QLoRA 会**动态地**将这个预训练模型的所有权重**量化到 4-bit**（或其他指定的精度，如 8-bit）。这个量化过程使用了特殊的 **4-bit NormalFloat** 数据类型和**双重量化**技术，以最小化量化误差。
3.  **冻结量化权重：** 量化后的主模型权重**被冻结**，**不会**在微调过程中更新。这是节省显存的关键一步。
4.  **注入可训练适配器 (LoRA)：** 在模型的每一层（通常是注意力层的 Q, K, V, O 投影矩阵旁）**注入低秩适配器模块**。这些 LoRA 模块（包含 A 和 B 两个小矩阵）是用**全精度（通常是 BF16 或 FP16）** 表示的。
5.  **微调适配器：**
    *   在微调过程中，**只有这些低秩适配器模块的参数会被更新**。
    *   前向传播时，对于每个包含适配器的层：
        *   从冻结的 4-bit 主权重中**反量化**出该层权重的一个**临时全精度版本**。
        *   计算该层原本的输出：`output = (反量化权重) * input`。
        *   计算 LoRA 适配器的输出：`lora_output = (B * A) * input`（其中 A 和 B 是可训练的小矩阵）。
        *   将两者相加得到该层的最终输出：`final_output = output + (scaling * lora_output)`。
    *   反向传播时，**梯度只通过 LoRA 适配器的路径传播并更新 A 和 B 矩阵**。冻结的 4-bit 主权重不接收梯度。
6.  **保存与部署：**
    *   微调完成后，**只保存训练好的 LoRA 适配器权重**（文件通常很小）。
    *   在推理时，需要将保存的 LoRA 适配器权重加载到**原始的、未量化的预训练模型**上，或者加载到与原始模型结构相同但**运行时量化**的模型上（将原始权重量化 + 合并 LoRA 适配器权重）。

**为什么说 QLoRA 支持用量化后模型微调？**

*   **核心操作对象是量化权重：** 微调过程中，模型的前向计算是基于从冻结的 4-bit 量化权重反量化出来的临时全精度值进行的。模型的主要权重确实是“量化后”的状态。
*   **显存节省来源：** 主要的显存节省正是来自于将庞大的主模型权重存储为 4-bit 格式，而不是通常的 16-bit (BF16/FP16) 或 32-bit (FP32)。如果没有量化这步，显存占用会大得多。
*   **适配器是全精度：** 虽然主模型是量化的，但进行参数更新的部分（LoRA 适配器）是全精度的，确保了微调的质量。

**重要澄清：**

*   **不是微调“量化后模型文件”：** QLoRA 不是指你拿到一个别人已经量化好的模型文件（如 GPTQ 量化后的 `.safetensors` 或 GGUF 文件），然后直接在那个文件上做微调。QLoRA 的量化是在微调*开始前*，在你的训练环境中，对加载的原始模型进行的动态量化。
*   **起点是原始模型：** 你微调的起点是标准的预训练模型（如 Hugging Face 上的模型），QLoRA 框架会在训练开始时自动对它进行量化。
*   **适配器独立于量化：** 训练得到的 LoRA 适配器本身是独立于量化过程的。它们可以被应用到原始的全精度模型上，也可以被应用到运行时量化的模型副本上（在应用适配器后，通常需要将适配器权重合并回主权重再进行量化以获得最佳推理效率）。

**总结：**

QLoRA 通过**在微调过程中将主模型权重动态量化为 4-bit 并冻结，同时只训练注入的全精度低秩适配器 (LoRA)**，实现了在**效果上**对量化后模型进行高效微调，并极大地降低了微调所需的显存开销（通常可以在单张 24GB GPU 上微调 30B+ 参数的模型）。其核心就是利用量化来压缩存储主模型，利用 LoRA 来减少可训练参数并保持微调质量。