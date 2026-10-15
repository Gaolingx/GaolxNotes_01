# 宽模型提升MFU原因

## 🧑‍💻 User

为什么模型越宽MFU越容易更高？

---
## 😀 默认助手 (7)

# 为什么模型越宽，MFU 越容易更高

**一句话**：变宽意味着线性层的 $K$、$N$（即 $d_{model}$、$d_{ff}$）变大——能切出的 tile/CTA 更多、算术强度更高、非 GEMM 开销被摊薄、通信/计算比更低。这些因素把训练从"访存受限、填不满机器"推向"算力受限"，实测 MFU 自然更容易高。窄模型则反过来被这些瓶颈压住。

下面拆成几条机制。核心的数学推导和边界条件我放进了折叠块，需要时展开。

---

## 1. GEMM 形状决定张量核能否跑满

一个线性层是 $[M,K]\times[K,N]$，其中 $M=\text{batch}\times\text{seq}$ 是 token 数，$K,N$ 是特征维。

- **Tensor Core 的 MMA 粒度是固定的**（如 `m16n8k16`、Hopper 的 `wgmma m64nNk16`）。$K$ 太小时主循环迭代次数少，prologue/epilogue、流水线填充（pipeline fill/drain）、同步的占比就高，张量核真正在跑的时间比例被压低——而 MFU 的上限正是"张量核忙的比例"。
- **$N$ 太小 → 切不出足够的 CTA**。GPU 靠大量 thread block 铺满 SM。$N=128$、$K=128$ 的 GEMM 只能切出个位数 tile，$132$ 个 SM 大多空转；$N=16384$ 的 GEMM 能均匀切出成百上千 tile，正好把每个 SM、每个 SM sub-partition 填满。
- **Wave quantization**：block 数不是 SM 数整数倍时，最后一波只有零星 tile 在跑。tile 越多，这波"尾巴"占总时间的比例越小。

宽模型的 MLP（$d\times 4d$ / $4d\times d$）和 attention 投影恰好都是 $N$ 很大的 GEMM，所以每个 kernel 的实测效率（% of peak）本身就更高。

---

## 2. 算术强度（Roofline）：变宽把你推过"屋脊点"

<details>
<summary><b>展开：单个 $d\times d$ 线性层的算术强度推导</b></summary>

<br>

以 bf16 为例（每元素 2 字节），$M$ 个 token 过一个 $d\times d$ 层：

$$
\text{FLOPs} = 2Md^2
$$

$$
\text{Bytes} \approx \underbrace{2d^2}_{\text{权重}} + \underbrace{4Md}_{\text{输入+输出激活}} = 2d^2 + 4Md
$$

$$
\text{AI} = \frac{2Md^2}{2d^2+4Md} = \frac{Md}{d+2M}
$$

注意这是一个**调和平均式的双向上限**：

$$
AI < \min\!\left(M,\; \frac{d}{2}\right)
$$

- 若 $d \ll M$：$AI \approx d/2$ —— 由宽度决定，跟 batch 关系不大。**窄模型就卡在这**。
- 若 $d \gg M$：$AI \approx M$ —— 权重被读一次只服务 $M$ 个 token，变成权重加载受限。

而 GPU 的"屋脊点"（compute bound 门檻）为：

$$
AI_{\text{ridge}} = \frac{\text{peak FLOPs}}{\text{HBM BW}}
\approx \frac{312\,\text{TFLOP/s}}{2.0\,\text{TB/s}} \approx 150 \ \text{(A100)}
\approx \frac{989}{3.35} \approx 300 \ \text{(H100)}
$$

代个数：$M=4096$（batch 8 × seq 512）

| $d$ | $AI = \frac{Md}{d+2M}$ | 状态 |
|---|---|---|
| 512 | 241 | 内存受限（A100 上偏低效） |
| 1024 | 455 | 计算受限 ✅ |
| 4096 | 1820 | 远在屋脊点右侧 |

结论：**固定 token 数时，$AI$ 随 $d$ 单调上升**（$d$ 越大越靠 $M$ 这个天花板），所以加宽会把你从 memory-bound 区间推进 compute-bound 区间。窄模型光靠加 batch 也能提升，但会被 $d/2$ 这个天花板限制住。

</details>

---

## 3. 非 GEMM 开销被"摊薄"

MFU 只把 matmul 的 FLOPs 计入分子，但墙钟时间还包括大量访存受限的算子：

- LayerNorm / residual / 激活函数：访存量 $\propto \text{tokens}\cdot d$
- Softmax：$\propto \text{tokens}\cdot s$（与 $d$ 无关）
- 它们在总时间里的占比 $\propto \dfrac{1}{d}$，而 GEMM 的 FLOPs $\propto d^2$

所以宽度一涨，**"低效算子"的份额按 $1/d$ 缩小，高效 GEMM 的份额上升**，整体 MFU 上升。这也解释了为什么 attention 里的 softmax、RoPE 等零碎 kernel 在大宽度模型上显得没那么致命。

---

## 4. 分布式训练：通信/计算比随宽度下降

这条在 TP（张量并行）里尤其明显：

$$
\text{通信量 (all-reduce/all-gather)} \propto \text{tokens}\cdot d, \qquad \text{计算量} \propto \text{tokens}\cdot d^2
$$

$$
\Rightarrow \frac{\text{通信}}{\text{计算}} \propto \frac{1}{d}
$$

维度越宽，每个 TP rank 分到的 GEMM 越大、单位计算需要交换的数据越少，通信更容易被计算掩盖 → 有效 MFU 更高。窄模型常出现"通信等待拖垮 MFU"，宽模型则缓释。

---

## 5. Attention 也随之变"重"

宽度增长通常体现为 `head_dim` 变大（head 数固定）或 head 数变多：

- `head_dim` 增大 → 每个 head 的 $QK^\top$、$PV$ GEMM 的 $K$ 更大，更接近算力上限；
- attention 的 FLOPs $\propto s^2 d$，而 softmax 等访存部分 $\propto s^2$ 与 $d$ 无关 → 同样是"分子涨得比分母快"。

---

<details>
<summary><b>什么时候"越宽 MFU 越高"不成立（反例与边界）</b></summary>

<br>

1. **token 数 $M$ 太小**。由 $AI < \min(M, d/2)$，若 batch 极小（如 $M\ll d$），$AI$ 被钉在 $M$ 附近，纯粹是权重加载受限，再宽也上不去。所以"宽"必须配足够的 batch×seq 才有效。
2. **推理 decoding 阶段**：$M$ 甚至 $=1$，每步要读全部权重，本质 memory-bound，宽反而更慢，MFU 极低——这与训练场景不同。
3. **被 optimizer state / KV cache / ZeRO 的带宽卡住**：此时瓶颈不在 GEMM 形状，加宽无效甚至更差。
4. **MoE 模型**：total params 很大但 per-token 激活的 FLOPs 小，且专家 GEMM 的 $M$ 被 token 路由切碎 → 形状反而变差，不能简单套用"越宽越好"。
5. **极端宽 + 太浅**：宽度带来的并行度已饱和后，收益递减；同时注意 attention 在超长序列下的 $s^2$ 项可能重新成为瓶颈。

一句话：宽度的增益本质来自 **GEMM 形状 + 算术强度 + 开销摊薄 + 通信比**，凡是这几条不成立（$M$ 太小、瓶颈转移到带宽/通信/访存），"越宽越高"就会失效。

</details>

---

## 小结

| 机制 | 窄模型 | 宽模型 |
|---|---|---|
| GEMM tile 数 / 填 SM | 少，wave 尾巴重 | 多，铺满 SM |
| 张量核流水效率 | K/N 小，主循环短 | K/N 大，主循环长 |
| 算术强度 $AI=\frac{Md}{d+2M}$ | $\approx d/2$，易内存受限 | 靠近 $M$，计算受限 |
| 非 GEMM 开销占比 | $\propto 1/d$，偏大 | 被摊薄 |
| 通信/计算比 | 高 | 低（$\propto 1/d$） |

所以"模型越宽 MFU 越高"不是一个玄学现象，而是 **GEMM 形状、Roofline 算术强度、开销摊薄、通信比** 四条机制叠加的结果；而它成立的前提是 token 数（batch×seq）要足够大，否则会被 $\min(M, d/2)$ 这个天花板挡回去。
