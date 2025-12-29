# LLM量化技术对比

### 🧑‍💻 User

llm的后训练量化中，什么是仅权重量化（WOQ），例如w4a16 awq/w4a16 gptq，相较于权重+激活值量化（例如w4a8等）有什么区别和优劣？

---

### 🤖 Assistant

在 LLM 的**后训练量化（Post-Training Quantization, PTQ）**体系中，“仅权重量化（Weight-Only Quantization, WOQ）”与“权重 + 激活量化（Weight–Activation Quantization, WAQ）”是两条技术路线，二者在**误差来源、工程复杂度、性能收益以及适用场景**上都有本质差异。下面从原理到工程实践系统性对比。

---

## 一、什么是仅权重量化（WOQ）

**定义**
WOQ 指 **只对模型权重进行低比特量化**，而 **激活（activation）、中间累加（accumulator）和最终输出**仍保持较高精度（通常 FP16/BF16，有时 FP32 accumulation）。

典型形式：

* **w4a16**（权重 4bit，激活 16bit）
* 算法代表：**GPTQ、AWQ、NF4（QLoRA 中）**

**计算形式（以线性层为例）**：
[
y = (Q(w) \cdot x)_{\text{accum in fp16/fp32}}
]

量化只发生在：

* 权重存储（memory footprint）
* 权重加载（memory bandwidth）

---

## 二、什么是权重 + 激活量化（WAQ）

**定义**
WAQ 对 **权重和激活同时进行量化**，常见为：

* **w4a8**
* **w8a8**
* 甚至更激进的 w4a4

典型特征：

* 激活在运行时动态量化（per-tensor / per-channel / per-token）
* 计算完全落在整数域（int8 / int4 MAC）

---

## 三、核心差异对比（本质层面）

### 1. 误差来源完全不同

| 量化方式    | 主要误差来源                     |
| ------- | -------------------------- |
| **WOQ** | 仅来自权重量化误差                  |
| **WAQ** | 权重误差 + 激活量化误差 + scale 误差叠加 |

**关键点**

* 激活是 **输入相关的动态分布**
* 激活误差会被：

  * 残差连接（residual）
  * 多层传播
  * attention 的 softmax 放大

因此 WAQ 的误差 **更不可控、累积更快**。

---

### 2. 对 LLM 架构的“友好度”差异

#### WOQ 为什么特别适合 Transformer / LLM？

* LLM 的 inference **主要瓶颈是 memory bandwidth**
* 权重矩阵巨大（几十 GB）
* 激活相对小（seq_len × hidden）

→ **只压缩权重就能获得大部分收益**

同时：

* attention、softmax、layernorm 对数值精度敏感
* WOQ 完全避开这些模块的量化风险

---

### 3. 精度 / PPL 表现

经验结论（大量实测）：

| 方案               | PPL 退化       |
| ---------------- | ------------ |
| w4a16 GPTQ / AWQ | 极小（通常 <1–3%） |
| w8a8             | 可接受          |
| w4a8             | 明显不稳定，依赖校准   |
| w4a4             | 通常不可用        |

原因非常清晰：

> **激活分布是重尾（heavy-tailed）且随 token 波动，远比权重难量化**

---

## 四、性能与工程复杂度对比

### 1. 性能收益来源不同

| 项目   | WOQ  | WAQ   |
| ---- | ---- | ----- |
| 权重显存 | 大幅下降 | 大幅下降  |
| 内存带宽 | 大幅下降 | 大幅下降  |
| 算力   | 几乎不变 | 显著下降  |
| 推理加速 | 中等   | 理论上最高 |

> WOQ 的收益主要来自 **“少读数据”**
> WAQ 的收益来自 **“少算 + 少读”**

---

### 2. 工程复杂度

#### WOQ

* 不需要修改计算图
* 不依赖激活校准
* 可 fallback 到 FP16 accumulation
* 易于集成到现有推理框架（vLLM / llama.cpp / TensorRT-LLM）

#### WAQ

* 需要：

  * 激活统计 & 校准数据
  * runtime scale 计算
  * per-token / per-group 动态量化
* kernel 高度定制
* 不同 batch / seq_len 行为差异大

---

## 五、为什么后训练量化里 WOQ 是“主流选择”

你可以把它理解为一个**性价比最优点**：

* **80% 的收益来自权重量化**
* **80% 的风险来自激活量化**

WOQ：

* 避开高风险部分
* 保留主要收益
* 精度稳定、调试成本低

这也是为什么：

* GPTQ / AWQ 成为事实标准
* 开源模型默认给 w4a16 权重包
* 工业部署中极少使用 w4a8（除非强依赖 INT8 Tensor Core）

---

## 六、典型使用场景建议

### 选择 WOQ（w4a16）的情况

* 对生成质量敏感（chat / reasoning）
* 长上下文
* 小 batch / latency sensitive
* CPU / 非 TensorCore GPU
* 无法接受复杂校准流程

### 选择 WAQ（w8a8 / w4a8）的情况

* 批量吞吐型任务（offline serving）
* 硬件有成熟 int8 pipeline（A100 / H100）
* 可接受一定质量下降
* 推理框架高度定制

---

## 七、一句话总结

> **WOQ 是“结构友好、误差可控、工程稳定”的量化方案；
> WAQ 是“理论性能上限高，但风险和复杂度也最高”的方案。**

在当前 LLM inference 现实条件下，**w4a16 并不是妥协，而是最优工程解**。

如果你愿意，下一步可以进一步拆：

* 为什么 AWQ 的“activation-aware”本质仍然是 WOQ
* 为什么 attention 层几乎永远不量化激活
* 或从 roofline / bandwidth 模型解释 w4a16 的真实加速上限
