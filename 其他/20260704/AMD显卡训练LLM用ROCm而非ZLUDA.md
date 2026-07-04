# AMD显卡训练LLM用ROCm而非ZLUDA

### 🧑‍💻 User

使用pytorch进行llms预训练，为什么amd显卡需要使用zluda，是 rocm 不好用吗？

---

### 🤖 Assistant

首先要澄清一个常见的误区：**使用 PyTorch 进行 LLM（大语言模型）预训练，并不是必须使用 ZLUDA，实际上 ROCm 才是官方推荐、企业界真正在用且性能最好的正统方案。**

如果您看到有人在 AMD 显卡上使用 ZLUDA 来跑深度学习，这通常是因为他们在使用**消费级显卡（如 Radeon RX 7000 系列）、Windows 操作系统**，或者依赖了**高度绑定 CUDA 的第三方库**。

以下是关于为什么会有“AMD 需要用 ZLUDA”这种印象，以及 ROCm 真实现状的详细解答。

### 1. ROCm 不好用吗？（ROCm 的真实现状）

客观地说，早期的 ROCm 确实存在生态落后、文档不全、Bug 较多的问题，但现在情况已经大为改观：

*   **PyTorch 官方原生支持：** PyTorch 官方早已原生支持 ROCm。在 Linux 环境下，您只需要一行命令（例如 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`）就可以安装支持 AMD 的 PyTorch，**完全不需要 ZLUDA**。
*   **企业级 LLM 训练的标配：** 在工业界，使用 AMD MI250、MI300X 等计算卡进行 LLM 预训练和微调时，使用的全是纯血的 ROCm 环境。Hugging Face、vLLM、DeepSpeed 等主流框架也都已经对 ROCm 提供了良好的官方支持。
*   **性能优异：** 在原生 ROCm 环境下，AMD 计算卡可以发挥出 100% 的硬件性能。

---

### 2. 既然 ROCm 能用，为什么还有人推荐 ZLUDA？

ZLUDA 是一个允许原本为 NVIDIA CUDA 编写的程序直接在 AMD GPU 上运行的“翻译层”。在 LLM 预训练或微调中，人们尝试使用 ZLUDA 主要有以下几个“痛点”原因：

#### A. 消费级硬件与操作系统的限制 (最常见原因)
ROCm 是为 Linux 系统和 AMD Instinct 数据中心计算卡设计的。如果您使用的是**Windows 系统**加上**普通的 AMD 游戏显卡（如 RX 7900 XTX）**：
*   Windows 下的 ROCm 支持一直处于实验性阶段（HIP SDK），配置极其繁琐，且很多深度学习框架的 Windows 版本没有编译 ROCm 后端。
*   在这种情况下，使用 ZLUDA 可以让用户直接下载现成的 CUDA 版本 PyTorch，实现在 Windows 上的“即插即用”。

#### B. CUDA 生态的极度依赖与“硬编码”
虽然 PyTorch 本身支持 ROCm，但 LLM 预训练依赖的底层算子库可能只写了 CUDA 版本。

<details>
<summary><b>点击展开：深入了解 CUDA/ROCm 生态算子差距</b></summary>
<br>
预训练 LLM 时，我们通常需要使用到一些加速技术：
<ul>
<li><b>FlashAttention:</b> 早期 FlashAttention 只有 CUDA 实现。虽然现在有了 ROCm 移植版（通过 Composable Kernel 实现），但每次更新都有滞后性。</li>
<li><b>量化与优化器算子:</b> 像 <code>bitsandbytes</code>（用于 8-bit/4-bit 优化器）、<code>xformers</code> 等库，很多底层算子是手写的 CUDA C++ 甚至 PTX 汇编代码。</li>
<li><b>编译成本:</b> 虽然 AMD 提供了 <code>HIPIFY</code> 工具可以将 CUDA 代码转化为 ROCm (HIP) 代码，但用户自己去编译这些底层库常常会遇到环境依赖报错，门槛极高。</li>
</ul>
在这种情况下，ZLUDA 可以在不修改、不重新编译这些底层 CUDA 库源码的情况下，直接让它们在 AMD 显卡上跑起来。
</details>

#### C. 闭源项目的限制
有些旧的或者特定的 AI 工具/脚本是闭源的，只提供了编译好的 CUDA 二进制文件。这时候 ROCm 无法发挥作用，只能依靠 ZLUDA 进行指令集层面的翻译。

---

### 3. ZLUDA vs ROCm：您应该怎么选？

如果您真的打算进行 **LLM 预训练**（Pre-training），这通常意味着庞大的计算量和长时间的运行：

*   **强烈建议使用原生 ROCm + Linux：** 预训练需要极致的性能和稳定性。ZLUDA 作为翻译层，存在性能损耗（通常损耗在 10% 到 30% 不等，甚至更高），并且在长时间高负载运行复杂的 LLM 算子时可能会遇到兼容性 Bug（例如内存泄漏或精度对齐问题）。您应该安装 Ubuntu系统，配置好 ROCm 环境，使用原生的 PyTorch 进行训练。
*   **什么时候用 ZLUDA：** 只有当您是在 Windows 系统下做一些简单的代码调试，或者某个依赖库死活无法在 ROCm 环境下编译通过，仅仅为了“能跑通”时，才考虑使用 ZLUDA。

**总结：**
ROCm 并非不好用，它是 AMD 显卡进行 LLM 训练的**唯一正统且最高效**的途径。ZLUDA 更多是消费级显卡用户、Windows 用户以及面对顽固 CUDA 专属库时的一种**妥协和过渡方案**。