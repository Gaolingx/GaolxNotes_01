# PCIe Redriver与Retimer区别

### 🧑‍💻 User

pcie redriver和retimer有什么区别？

---

### 🤖 Assistant

PCIe (Peripheral Component Interconnect Express) 随着速率的不断提升（从 Gen 3 的 $8 \text{ GT/s}$ 到 Gen 5 的 $32 \text{ GT/s}$ 甚至更高），信号在 PCB 传输过程中的衰减（Insertion Loss）和干扰变得非常严重。

为了解决长距离传输信号完整性问题，通常会使用 **Redriver（中继器/线性驱动器）** 或 **Retimer（重定时器）**。

简单来说：**Redriver 是“傻瓜式”的信号放大器，而 Retimer 是“智能”的信号重建者。**

以下是详细的对比分析。

---

### 1. 核心定义与工作原理

#### **Redriver (信号放大器)**
Redriver 是一种模拟器件（Analog device）。它的工作原理主要是通过 **均衡（Equalization, EQ）** 和 **预加重（Pre-emphasis）** 来补偿信号在传输线上的衰减。

*   **机制：** 它接收信号，放大信号的幅度，增强高频分量，然后输出。
*   **局限性：** 它**不会**重新生成时钟信号，也**不检查**数据的正确性。这意味着它在放大有用信号的同时，也会**放大噪声（Noise）和抖动（Jitter）**。
*   **协议透明：** 对于 PCIe 控制器和端点设备来说，Redriver 是透明的（就像一根更好的电线），它不参与链路训练（Link Training）。

#### **Retimer (重定时器)**
Retimer 是一种数模混合器件（Mixed-signal device）。它包含完整的 **CDR（时钟数据恢复，Clock and Data Recovery）** 电路。

*   **机制：** 它接收信号，利用 CDR 提取并锁定输入信号的时钟，解析出数据（0 和 1），去除噪声和抖动，然后利用内部生成的干净时钟，**重新生成（Re-generate）** 一个全新的信号发送出去。
*   **优势：** 它可以彻底消除累计的随机抖动和确定性抖动，重置时序预算。
*   **协议感知：** Retimer 是“协议感知”（Protocol-aware）的。它会参与 PCIe 的链路训练过程（LTSSM），可以说它是 PCIe 链路中的一个“隐形中继站”。

---

### 2. 详细对比表

| 特性 | Redriver | Retimer |
| :--- | :--- | :--- |
| **主要功能** | 信号放大、均衡 (CTLE) | 信号再生、时钟恢复 (CDR) |
| **抖动 (Jitter)** | **无法消除抖动**，甚至会放大抖动 | **重置抖动**，输出纯净信号 |
| **噪声 (Noise)** | 连同信号一起放大 | 滤除噪声 |
| **信号处理域** | 纯模拟域 (Analog) | 混合域 (Analog + Digital) |
| **协议感知** | 不感知 (透明传输) | 感知 (参与 Link Training) |
| **延迟 (Latency)** | 极低 ($< 100 \text{ ps}$ 级别) | 较高 (通常 $10 \sim 30 \text{ ns}$) |
| **功耗** | 低 | 高 (通常是 Redriver 的 2 倍以上) |
| **成本** | 低 | 高 |
| **PCIe 适用代系** | 常见于 Gen 1/2/3，Gen 4 短距 | 常见于 Gen 4/5/6 及长距传输 |

---

### 3. 形象类比

为了更好地理解，我们可以用“传话游戏”来打比方：

*   **信号传输：** 一个人（CPU）对着远处的人（GPU）喊话。
*   **距离太远：** 声音（信号）变小了，周围还有环境噪音。

**Redriver 就像一个“扩音喇叭（Megaphone）”：**
> 它把听到的声音直接放大传出去。
> *   **优点：** 声音确实变大了。
> *   **缺点：** 如果输入的声音里有杂音（比如风声），扩音器会把风声一起放大，远处的人可能还是听不清具体的字。

**Retimer 就像一个“中间传话人（Translator）”：**
> 他站在中间，仔细听这一句话，理解了这句话的内容（比如是“你好”），然后用自己洪亮、清晰的嗓音重新喊出“你好”。
> *   **优点：** 之前所有的环境噪音都被过滤掉了，传出去的是全新的、标准的声音。
> *   **缺点：** 传话人听懂并复述需要一点点思考时间（延迟），而且雇佣传话人比买个喇叭要贵。

---

<details>
<summary><strong>点击展开：深入技术细节 (EQ, CDR, Eye Diagram)</strong></summary>

#### 1. 眼图 (Eye Diagram) 的区别
*   **Redriver:** 试图打开闭合的眼图，增加眼高 (Eye Height)，但对眼宽 (Eye Width) 的改善有限，因为抖动依然存在。
*   **Retimer:** 因为有 CDR 重整时钟，它不仅大幅增加眼高，还能最大化眼宽，提供最佳的信号余量。

#### 2. 均衡技术 (Equalization)
*   **Redriver:** 主要依赖 CTLE (Continuous Time Linear Equalization) 和简单的驱动增强。
*   **Retimer:** 除了 CTLE，通常还包含 DFE (Decision Feedback Equalization)。DFE 是一种非线性均衡技术，能有效处理 ISI (码间干扰) 和反射，这在高速 (Gen 5 $32 \text{ GT/s}$) 下至关重要。

#### 3. 链路训练 (Link Training)
*   Retimer 必须实现 PCIe 规范中的 LTSSM (Link Training and Status State Machine) 的一部分。这意味着如果 Retimer 挂了或配置错误，BIOS 可能根本扫描不到后端的设备。而 Redriver 配置错误通常表现为链路降速（比如 Gen 4 降级为 Gen 3）。

</details>

---

### 4. 如何选择？

选择使用 Redriver 还是 Retimer，主要取决于**链路损耗预算 (Insertion Loss Budget)** 和**速率**。

根据 PCIe 规范，不同代系的通道损耗预算约为：
*   **PCIe Gen 3 ($8 \text{ GT/s}$):** $\approx 22 \text{ dB}$
*   **PCIe Gen 4 ($16 \text{ GT/s}$):** $\approx 28 \text{ dB}$
*   **PCIe Gen 5 ($32 \text{ GT/s}$):** $\approx 36 \text{ dB}$

**选择建议：**

1.  **PCIe Gen 3 及以下：**
    *   几乎总是使用 **Redriver**。成本低，技术成熟，足以应付 $8 \text{ GT/s}$ 的信号衰减。

2.  **PCIe Gen 4 ($16 \text{ GT/s}$):**
    *   **短/中距离 (比如主板内部)：** 可以使用 **Redriver**。
    *   **长距离 (比如通过 Riser 卡、背板连接器)：** 推荐使用 **Retimer**。因为 $16 \text{ GT/s}$ 对抖动非常敏感，Redriver 可能无法提供足够的信号余量。

3.  **PCIe Gen 5 ($32 \text{ GT/s}$) 及 Gen 6 ($64 \text{ GT/s}$):**
    *   **Retimer 是主流选择。**
    *   在如此高的频率下，PCB 板材的介质损耗非常大，且信号对串扰和反射极度敏感。单纯放大信号（Redriver）通常无法满足误码率（BER $< 10^{-12}$）的要求。AI 服务器（如连接 H100/H800 GPU）几乎全部使用 Retimer。

### 总结

*   **省钱、短距离、低速率 ($\le \text{Gen 3}$) $\rightarrow$ 选 Redriver。**
*   **追求性能、长距离、高速率 ($\ge \text{Gen 5}$) $\rightarrow$ 必须选 Retimer。**