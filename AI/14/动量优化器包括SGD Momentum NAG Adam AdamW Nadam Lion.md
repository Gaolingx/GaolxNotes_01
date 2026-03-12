# 动量优化器包括SGD Momentum NAG Adam AdamW Nadam Lion

### 🧑‍💻 User

基于Gradient Descent with Momentum的optimizer有哪些？

---

### 🤖 Assistant

基于**动量梯度下降（Gradient Descent with Momentum）**的优化器是深度学习中最核心、使用最广泛的优化算法家族。动量的核心思想是通过引入物理学中的“惯性”概念——即计算历史梯度的指数加权移动平均（Exponential Moving Average, EMA）——来加速收敛并减少在优化过程中的震荡。

基础的动量更新公式为：
$$v_t = \gamma v_{t-1} + \eta \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$
其中 $v_t$ 是动量项，$\gamma$ 是动量因子（通常设为 $0.9$），$\eta$ 是学习率。

以下是基于或融合了动量机制的主流优化器，按其演进和重要程度分类：

### 一、 基础动量变体 (Fundamental Variants)

这类优化器直接在传统梯度下降（SGD）上修改动量计算方式。

1. **SGD with Momentum (Classical Momentum / Polyak Momentum)**
   * **原理**：最经典的动量算法，由 Boris Polyak 提出。它直接累加历史梯度，使得在梯度方向一致的维度上加速，在梯度方向改变的维度上减速，从而减少震荡。
   * **应用**：在计算机视觉（如 ResNet 训练）中仍然是首选，通常能取得比自适应方法更好的泛化性能。
   * **代码**：`torch.optim.SGD(params, lr=0.1, momentum=0.9)`

2. **Nesterov Accelerated Gradient (NAG)**
   * **原理**：Nesterov 动量是经典动量的改进版。它具有“预见性（Lookahead）”。在计算当前梯度时，它不是在当前位置 $\theta_t$ 计算，而是在**施加动量后的未来位置** $\theta_t - \gamma v_{t-1}$ 计算梯度。这能提供更及时的修正，防止冲过极小值。
   * **公式**：
     $$v_t = \gamma v_{t-1} + \eta \nabla J(\theta_t - \gamma v_{t-1})$$
   * **代码**：`torch.optim.SGD(params, lr=0.1, momentum=0.9, nesterov=True)`

---

### 二、 动量 + 自适应学习率 (The Adam Family)

这类优化器将**动量（一阶矩估计）**与**RMSprop（二阶矩估计，用于自适应调整每个参数的学习率）**结合在一起，是目前 NLP 和多模态大模型最常用的优化器家族。

1. **Adam (Adaptive Moment Estimation)**
   * **原理**：同时计算梯度的指数加权移动平均（一阶矩 $m_t$，即动量）和梯度平方的指数加权移动平均（二阶矩 $v_t$），并进行了偏差校正。
   * **公式**：
     $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad \text{(动量项)}$$
     $$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad \text{(自适应项)}$$
     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
   * **代码**：`torch.optim.Adam()`

2. **AdamW (Adam with Weight Decay)**
   * **原理**：修复了标准 Adam 中 $L_2$ 正则化（权重衰减）与自适应学习率耦合导致正则化失效的问题。AdamW 将权重衰减直接应用在参数更新步骤上，而不是梯度计算中。
   * **地位**：**目前 Transformer 模型（如 BERT, GPT 系列）的绝对标配**。
   * **代码**：`torch.optim.AdamW()`

3. **Nadam (Nesterov-accelerated Adaptive Moment Estimation)**
   * **原理**：将 Adam 中的传统动量替换为 **Nesterov 动量**。它在计算一阶矩估计 $m_t$ 时融合了前瞻机制，通常比标准 Adam 收敛略快。
   * **代码**：`torch.optim.Nadam()`

---

### 三、 进阶与专用优化器 (Advanced & Specialized Variants)

随着模型规模和训练方式的改变，研究者基于动量和 Adam 提出了更多改进方案。为了版面整洁，这部分次要或专用的优化器已折叠。

<details>
<summary><b>点击展开查看：RAdam, AdaBelief, LAMB, Lion 等进阶优化器</b></summary>

4. **RAdam (Rectified Adam)**
   * **原理**：在训练初期，由于样本量少，Adam 的二阶矩（方差）估计极不稳定。RAdam 引入了一个“整流器（Rectifier）”来动态调整自适应学习率的方差。在训练早期自动关闭自适应学习率（退化为带缩放的动量 SGD），随着方差稳定再开启。这通常能消除对 `Warmup`（学习率预热）的需求。
   * **代码**：`torch.optim.RAdam()`

5. **AdaBelief**
   * **原理**：修改了 Adam 中二阶矩的计算方式。Adam 计算的是 $g_t^2$，而 AdaBelief 计算的是梯度与动量差值的平方 $(g_t - m_t)^2$。这代表了优化器对当前梯度的“信念（Belief）”：如果梯度与预期（动量）一致，步长变大；如果偏离预期，步长变小。
   * **优势**：号称兼具 Adam 的收敛速度和 SGD 的泛化能力。

6. **LAMB (Layer-wise Adaptive Moments optimizer for Batch training)**
   * **原理**：专为**超大 Batch Size** 训练设计。它在 AdamW 的基础上引入了逐层（Layer-wise）的学习率缩放机制（Trust Ratio），防止在极大学习率下梯度发散。
   * **应用**：被广泛用于缩短 BERT 等大模型的预训练时间（例如将 Batch Size 扩展到 32K 甚至更高）。

7. **Lion (EvoLved Sign Momentum)**
   * **原理**：由 Google Brain 通过符号搜索算法自动发现的优化器。它放弃了二阶矩（不计算自适应学习率），**仅保留动量**，并且只取梯度的符号（`sign()` 函数）来更新参数。
   * **公式**：
     $$c_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
     $$\theta_{t+1} = \theta_t - \eta \cdot \text{sign}(c_t)$$
     $$m_t = \beta_2 m_{t-1} + (1-\beta_2)g_t$$
   * **优势**：由于不存储二阶矩 $v_t$，相比 AdamW **节省了 1/3 的优化器显存**，非常适合训练超大语言模型。
</details>

---

### 总结：如何选择？

* **默认首选**：`AdamW`。适用于绝大多数深度学习任务，特别是 NLP、Transformer 模型、扩散模型等。
* **计算机视觉（CNNs）**：如果追求极致的测试集泛化性能（如图像分类的 ResNet），推荐使用带动量的 `SGD`（配合余弦退火学习率）。
* **超大 Batch Size 分布式训练**：考虑使用 `LAMB`。
* **显存极度受限的大模型训练**：尝试使用 Google 的 `Lion` 优化器。