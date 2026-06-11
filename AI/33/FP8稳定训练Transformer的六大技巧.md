# FP8稳定训练Transformer的六大技巧

### 🧑‍💻 User

mixed precision，Delayed Scaling，Random Hadamard Transform，Stochastic Rounding，2D Block Scaling，Selective Precision等trick是如何保证在fp8精度下稳定训练transformer的？

---

### 🤖 Assistant

在Transformer的训练中，将精度从16位（FP16/BF16）降至8位（FP8）是一个巨大的跨越。FP8具有极小的动态范围（容易溢出/下溢）和较低的精度（容易产生巨大的量化噪声）。

为了在FP8下实现Transformer的稳定训练，业界（如NVIDIA的Transformer Engine、微软的DeepSpeed等）结合了一系列算法层面的“Trick”。这些技术并不是孤立的，而是形成了一套完整的防御体系，从不同角度应对FP8带来的数值不稳定问题。

以下是您提到的六个核心技术是如何保证FP8稳定训练的详细解析：

<details open>
<summary><b>1. Mixed Precision (混合精度) —— 保护敏感计算与主权重</b></summary>

混合精度并不是将所有操作都降维到FP8，而是“好钢用在刀刃上”。

*   **工作原理：** 在训练过程中，模型的主权重（Master Weights）、梯度更新步骤以及敏感操作（如Softmax、LayerNorm、位置编码RoPE）仍然保持在较宽的精度（如FP32或BF16）。只有计算最密集、占用绝大部分时间的大型矩阵乘法（GEMM，如QKV投影、FFN层）被转换为FP8。
*   **如何保证稳定性：** FP8的尾数位极少（E4M3格式仅有3位尾数，E5M2仅有2位）。如果用FP8进行权重更新，微小的梯度乘以学习率后会因为精度不足被直接舍弃（变为0），导致模型停止学习（即**吞没误差**）。保留高精度的Master Weights使得累积的微小更新 $W_{t+1} = W_t - \eta \nabla L$ 能够精确进行。
</details>

<details open>
<summary><b>2. Delayed Scaling (延迟缩放) —— 兼顾性能与动态范围对齐</b></summary>

FP8的数据范围极窄，必须引入一个缩放因子（Scaling Factor）$S$，将张量的实际分布映射到FP8的最佳表示范围内。

*   **工作原理：** 在分布式训练中，如果要在当前步骤计算张量的绝对最大值（AbsMax）来确定缩放因子，需要进行全局同步，这会阻塞计算流水线。**Delayed Scaling** 的策略是：使用前 $N$ 个Step（通常 $N=1$ 到 $N=10$）的历史绝对最大值来作为当前Step的缩放因子。
*   **如何保证稳定性：** 在Transformer训练中，激活值和梯度的分布在相邻Step之间变化非常缓慢且具有时间连续性。使用历史尺度不仅消除了同步开销，保证了计算效率，而且能够提供足够精确的缩放界限，防止当前Step的数据在转换为FP8时发生大面积溢出（Overflow）或下溢（Underflow）。
</details>

<details open>
<summary><b>3. Random Hadamard Transform (随机哈达玛变换) —— 抹平异常值 (Outliers)</b></summary>

Transformer（尤其是LLM）在训练和推理时，其激活值存在极端的“异常值”（Outliers），某些特定通道的数值可能比其他通道大100倍。

*   **工作原理：** 在进行FP8量化前，将输入张量乘以一个随机的哈达玛矩阵（Hadamard Matrix）。这是一种正交变换，在数学上类似于对数据进行了一次“均匀搅拌”。
    $$ X_{transformed} = X \cdot H $$
*   **如何保证稳定性：** 如果直接对存在异常值的张量使用FP8缩放，为了不让异常值溢出，缩放因子会被拉得极大，导致99%的正常小数值在FP8下被压缩成0（严重的信息丢失）。**哈达玛变换将异常值的能量均匀分散到了所有维度中**，使得变换后的张量分布变得极其平滑且呈现正态分布。这使得FP8的量化网格能够完美匹配数据分布，极大地降低了量化误差，保证了前向传播的稳定。
</details>

<details open>
<summary><b>4. Stochastic Rounding (随机舍入) —— 打破舍入误差的系统性偏差</b></summary>

当高精度数字被转换为FP8时，传统的做法是“就近舍入”（Nearest Rounding）。

*   **工作原理：** 随机舍入是根据数值距离两端量化点的比例，以概率的形式进行舍入。例如，一个真实值 $x$ 落在FP8的两个可表示值 $a$ 和 $b$ 之间，且正好在 $a$ 和 $b$ 的 75% 处。随机舍入会以 25% 的概率将其舍入为 $a$，以 75% 的概率舍入为 $b$。
*   **如何保证稳定性：** 传统就近舍入会产生系统性偏差（Bias），在数千次迭代中，这种偏差会累积导致梯度方向偏移，造成训练崩溃。随机舍入的数学期望等于真实值：
    $$ \mathbb{E}[\text{Round}(x)] = x $$
    这保证了在宏观的统计层面上，量化带来的误差均值为0，从而使深层网络的梯度更新方向保持无偏，极大地提高了训练（尤其是在低精度梯度反传时）的收敛稳定性。

```python
# 随机舍入的概念性伪代码
def stochastic_round(x, fp8_grid):
    floor_val, ceil_val = find_adjacent_fp8_values(x, fp8_grid)
    distance_to_floor = x - floor_val
    gap = ceil_val - floor_val
    prob_ceil = distance_to_floor / gap
  
    if random.uniform(0, 1) < prob_ceil:
        return ceil_val
    else:
        return floor_val
```
</details>

<details open>
<summary><b>5. 2D Block Scaling (二维块缩放 / 细粒度缩放) —— 提升局部精度</b></summary>

传统的缩放因子通常是 Per-Tensor（整个矩阵共享一个）或 Per-Token/Per-Channel（一行或一列共享一个）。

*   **工作原理：** 2D Block Scaling 将一个大型权重矩阵或激活矩阵划分为多个较小的二维块（例如 $128 \times 128$），并为每一个微小的二维块分配一个独立的缩放因子 $S_{i,j}$。
*   **如何保证稳定性：** Transformer的权重和梯度矩阵在局部区域的方差差异很大。如果用全局缩放，方差小的区域会被方差大的区域“裹挟”，导致精度严重损失。2D Block Scaling 让每个局部区域根据自身的动态范围“量体裁衣”，最大化地利用了FP8仅有的256个离散状态，显著降低了矩阵乘法带来的误差累积，防止了训练后期的发散。
</details>

<details open>
<summary><b>6. Selective Precision (选择性精度) —— 避开雷区</b></summary>

并非Transformer中的所有模块对精度的容忍度都一样。

*   **工作原理：** 算法会动态或静态地评估不同层的敏感度。通常的做法是：
    1.  **首尾层保护**：Embedding层和最后的LM Head（分类输出层）保持在BF16/FP32。
    2.  **注意力分数保护**：Attention机制中的 $Q \times K^T$ 计算结果极为敏感（稍有误差就会经过Softmax被指数级放大），这部分通常不采用极低精度。
*   **如何保证稳定性：** 强行将敏感层量化为FP8是导致训练NaN（梯度爆炸）的直接原因。通过 Selective Precision，我们在占计算量 80% 的鲁棒层（如FFN层的线性变换）中榨取FP8的加速红利，同时在占计算量 20% 但决定生死存亡的敏感层中保留高精度，这是一种平衡风险与收益的兜底策略。
</details>

### 总结

Transformer在FP8下的稳定训练不是靠单一魔法，而是一个系统工程：
1. 用 **Selective Precision** 和 **Mixed Precision** 划定安全边界。
2. 用 **Random Hadamard Transform** 和 **2D Block Scaling** 从空间上消除异常值、细化量化粒度。
3. 用 **Delayed Scaling** 在时间上解决效率与动态范围的矛盾。
4. 用 **Stochastic Rounding** 在统计学上保证梯度无偏。

这些Trick的组合，成功地将量化误差从“破坏性突变”压制为“良性正则化噪声”，从而实现了与BF16几乎一致的收敛曲线。