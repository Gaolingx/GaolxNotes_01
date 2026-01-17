# Negative prompt distribution shift

### 🧑‍💻 User

https://arxiv.org/abs/2406.02507v3
根据这篇论文说说，
negative prompt在guided diffusion model为什么是distribution shift？就是bad version of itself？

---

### 🤖 Assistant

根据这篇论文《Guiding a Diffusion Model with a Bad Version of Itself》，你提到的“Negative Prompt”（在论文语境下通常指代 **CFG 中的 Unconditional Model**，即无条件模型）之所以被视为一种 Distribution Shift（分布偏移），以及为什么它是“Bad version of itself”的一个特例，可以总结为以下几个核心观点：

### 1. 标准 CFG 的本质：任务不匹配 (Task Discrepancy)

在标准的 Classifier-Free Guidance (CFG) 中，采样公式是：
$$D_{guided} = D_{cond} + w \cdot (D_{cond} - D_{uncond})$$
这里，$D_{uncond}$（通常对应空 Prompt 或 Negative Prompt）充当了“负向引导”的角色。

论文指出，这种做法存在一个根本性的 **Distribution Shift（分布偏移）**，或者论文中称为 **Task Discrepancy（任务差异）**：

*   **$D_{cond}$ (主模型)**：试图拟合条件分布 $p(x|c)$（例如：“一只狗”的图像分布）。
*   **$D_{uncond}$ (负向模型)**：试图拟合边缘分布 $p(x)$（即“所有可能的图像”的分布）。

**为什么这是 Distribution Shift？**
因为这两个模型训练的目标完全不同。$D_{uncond}$ 必须覆盖整个数据集的所有可能性，任务比 $D_{cond}$ 难得多。因此，当我们用 $D_{cond} - D_{uncond}$ 作为引导向量时，我们不仅仅是在提高图像质量，我们还在强行让图像“不像无条件生成的通用图像”。这种**不同分布之间的相减**会导致引导方向并不总是指向更高质量的流形（Manifold），有时会导致采样轨迹偏离，产生过度饱和、构图单一或伪影。

### 2. 为什么 CFG 能提升质量？因为 $D_{uncond}$ 是一个“烂模型”

论文在第 3 节（*Why does CFG improve image quality?*）中提出了一个深刻的见解：CFG 之所以能提升画质，不仅仅是因为它增强了 Prompt 的依从性，更核心的原因是 **$D_{uncond}$ 本质上是一个“低配版/烂版”的 $D_{cond}$**。

*   **保守拟合 (Conservative Fit)**：根据最大似然估计的特性，模型倾向于覆盖整个数据分布。
*   **任务难度差异**：由于 $D_{uncond}$ 要拟合所有类别的图像，它的分布密度（Density）必然比 $D_{cond}$ 更“平铺散乱”（Spread out）。
*   **截断效应 (Truncation)**：当你执行 $D_{cond} - D_{uncond}$ 时，你实际上是在把采样点从“平铺散乱”的低概率区域，推向 $D_{cond}$ 所代表的“紧凑、高概率”区域（即数据流形的核心）。

**结论：** 标准 CFG 实际上已经在做“用一个烂模型来引导好模型”这件事了，只不过这个“烂模型”（无条件模型）烂得不够纯粹——它不仅质量差，而且分布还不对（Distribution Shift）。

### 3. Autoguidance：用“真正的 Bad Version”来修正

论文提出的 **Autoguidance** 方法，就是要消除上述的 Distribution Shift。作者认为，负向引导不应该来自一个“不同任务”的模型（无条件模型），而应该来自**同一个任务但能力较弱的模型**。

*   **做法**：让 $D_{negative}$ 也是一个条件模型 $D(x|c)$，但是它是**参数量更小**（Tiny）或者**训练时间更短**（Under-trained）的版本。
*   **原理**：
    *   好模型 ($D_{strong}$) 和 烂模型 ($D_{weak}$) 都在拟合同一个分布 $p(x|c)$。**（消除了 Distribution Shift）**
    *   但是，$D_{weak}$ 由于能力不足，会犯更多的错误，其分布会比 $D_{strong}$ 更松散、更模糊。
    *   通过 $D_{strong} + w \cdot (D_{strong} - D_{weak})$，引导向量精准地指出了“从模糊/错误指向清晰/正确”的方向，而不包含任何关于内容的改变。

### 总结：为什么 Negative Prompt 是 Bad Version？

根据这篇论文的理论：

1.  **Negative Prompt (Unconditional Model) 是一个“带有副作用”的 Bad Version。** 它之所以有效，是因为它比 Conditional Model 拟合得更差（更松散），从而产生了将图像推向流形核心的“截断力”。
2.  **它是 Distribution Shift。** 因为它拟合的是 $p(x)$ 而不是 $p(x|c)$，导致引导向量中混杂了“去除非条件特征”的信号，这会破坏图像的多样性或导致构图扭曲。
3.  **真正的 Autoguidance** 是用一个**在相同分布上训练、但能力受限**的模型来替代 Negative Prompt，从而剥离出纯粹的“质量提升”信号，实现既提高画质又不损失多样性（Record FIDs）。