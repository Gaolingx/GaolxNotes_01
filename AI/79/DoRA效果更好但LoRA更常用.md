# DoRA效果更好但LoRA更常用

### 🧑‍💻 User

为什么dora被证明能更接近全参微调效果，为什么大多数情况下依旧使用lora？

---

### 🤖 Assistant

这是一个非常深刻且切中当前大模型微调工程实践痛点的问题。

你的前提完全正确：**DoRA（Weight-Decomposed Low-Rank Adaptation）在数学原理和实际评测中，确实被证明比 LoRA 更接近全参微调（FFT）的学习模式，且效果通常更好。** 

DoRA 通过将权重分解为**方向（Direction）**和**幅度（Magnitude）**，使用 LoRA 更新方向，并单独学习幅度参数（$m$），解决了普通 LoRA 难以兼顾权重幅度和方向按比例变化的问题。

然而，在工业界和开源社区的绝大多数情况下，大家依然首选 LoRA。这种“理论更优但落地未普及”的现象，主要由以下几个核心的**工程与生态原因**导致：

### 1. 生态与工具链的绝对统治力 (Ecosystem Maturity)
LoRA 已经不仅仅是一个算法，它已经成为了 PEFT（参数高效微调）的**代名词**和**基础设施**。
* **无缝集成**：Hugging Face 的 `peft` 库、vLLM、Ollama、text-generation-webui 等所有主流生态都对 LoRA 提供了最原生的支持。
* **一键可用**：各种微调框架（如 LLaMA-Factory, Axolotl, Unsloth）默认配置都是 LoRA。虽然很多框架现在也支持了 DoRA，但经常需要依赖特定的版本，或者在某些边缘功能（如量化微调 QLoRA 结合 DoRA 时）存在兼容性 Bug。

### 2. 训练成本与计算开销 (Training Overhead)
虽然 DoRA 只比 LoRA 多了一个幅度参数矩阵 $m$，参数量增加极少，但它的**计算图更复杂**。
* DoRA 在前向传播时，需要计算方向矩阵的范数（Norm）来进行归一化：
  $$W' = m \frac{W + BA}{\|W + BA\|_c}$$
* 这一步在反向传播时需要计算梯度，导致了额外的显存开销和计算延迟。在未经极致优化（如 Unsloth 级优化）的框架中，DoRA 的训练速度通常比普通的 LoRA **慢 15% 到 30%**，且显存占用略高。

### 3. “足够好”原则与 ROI (The "Good Enough" Principle)
在工程实践中，技术选择往往不是追求“绝对的完美”，而是追求“投入产出比（ROI）的最大化”。
* 对于 90% 的垂直领域微调任务（如客服问答、特定风格对话、信息抽取），标准 LoRA 带来的效果已经完全达到了业务及格线或优秀线。
* DoRA 带来的那 1% 到 3% 的性能提升（MMLU 或特定 Benchmark 上的涨点），通常不足以驱使工程师去承担更换算法带来的兼容性风险、更长的训练时间以及可能出现的未知 Bug。

---

<details>
<summary><b>点击展开：其他次要但关键的原因（多LoRA推理、合并难度与超参差异）</b></summary>

### 4. 动态多适配器推理（Multi-Tenant Serving）的困难
在生产环境中，像 vLLM 这样的推理框架支持 **Multi-LoRA serving**。即：加载一个底座模型，在推理时根据用户的 Request 动态挂载不同的 LoRA 权重。
因为普通 LoRA 只是简单的线性加法（$W' = W + BA$），动态切换的计算开销极小。而 DoRA 由于涉及权重的归一化，动态挂载多个 DoRA 适配器在计算上要复杂得多，目前主流推理引擎对动态 DoRA 的支持远不如 LoRA 成熟。

### 5. 权重合并（Weight Merging）的坑
微调结束后，我们通常会将 Adapter 合并回底座模型以提升推理速度。
* 普通 LoRA 合并非常简单：$$W_{merged} = W_{base} + BA$$
* DoRA 合并虽然在数学上是可行的，但在早期的 PEFT 库中，DoRA 的合并逻辑存在不少 Bug（例如与 FP16/BF16 精度截断有关的问题），导致合并后的模型性能暴跌。虽然这些问题正在被修复，但给早期尝试者留下了“不稳定”的印象。

### 6. 超参数经验的失效 (Hyperparameter Intuition)
炼丹师们经过长期的摸索，已经对 LoRA 的超参数（如 Rank $r$, Alpha $\alpha$, Learning Rate）建立了强烈的直觉。
* 例如，大家知道 LoRA 的学习率通常可以设得比 FFT 大一些（如 $2 \times 10^{-4}$）。
* 但 DoRA 引入了新的动态，其最佳学习率和 Alpha 的缩放比例往往与 LoRA 不同。很多初学者直接把 LoRA 的超参套用到 DoRA 上，发现效果反而变差，从而退回了 LoRA。

</details>

---

### 总结：你应该用哪个？

* **默认使用 LoRA / QLoRA**：如果你只是想快速验证想法，或者做常规的业务微调，并且依赖 vLLM 等框架做动态推理，LoRA 依然是首选。
* **尝试使用 DoRA**：如果你在使用 **LLaMA-Factory 或 Unsloth**（它们对 DoRA 的优化已经非常好），并且你的任务对逻辑推理、数学能力要求极高，或者是持续预训练（Continual Pre-training），且你打算最终把权重合并（Merge）成单模型部署，那么 **DoRA 是绝对值得尝试的高阶技巧**。

随着 Unsloth 等底层算子库对 DoRA 训练速度的优化，以及 Hugging Face 生态的完善，DoRA 的使用率正在稳步上升。但在短时间内，LoRA 凭借其极致的简单性和生态惯性，依然会是“默认的王者”。