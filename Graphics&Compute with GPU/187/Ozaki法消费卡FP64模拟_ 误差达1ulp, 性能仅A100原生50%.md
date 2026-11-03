# Ozaki法消费卡FP64模拟: 误差达1ulp, 性能仅A100原生50%

## 🧑‍💻 User

估算ozaki法实现的消费卡FP64对比A100原生FP64之间的误差和性能

---
## 😀 默认助手 (7)

I'll verify a few key figures (Ozaki-scheme DGEMM throughputs and consumer-GPU tensor-core/FP64 ratios) before estimating, since the answer hinges on them.

Good — that gave me strong primary sources. Let me pull the abstracts with the actual measured numbers.

# 消费卡上 Ozaki 法模拟 FP64 vs A100 原生 FP64

## 一句话结论

| 维度 | 结论 |
|---|---|
| **误差** | 当切片数足够时，Ozaki 的舍入误差约为 **1 ulp（$\sim10^{-16}$，基本与内维 $k$ 无关）**，**不劣于、甚至在 $k$ 较大时优于** A100 原生 DGEMM；但它**并不能提升精度**——最终答案的误差仍由 $\kappa(A)$ 决定 |
| **性能** | 在 RTX 4090 上，Ozaki-II/INT8 实测为 **7.4–9.8 TFLOPS**（原生 FP64 仅 1.29，即 **约 6–7.6× 自加速**）；但**绝对值仍只有 A100 原生 FP64（19.5 TFLOPS）的约 40–50%** |
| **核心机理** | 收益 $\approx P_{\text{low}}/P_{\text{fp64}}$ 与 GEMM 次数之比：消费卡 INT8/FP64 $\approx 512\times$，A100 $\approx 32\times$；所以同一套算法在消费卡上收益明显，在 A100 上基本没有收益 |

---

## 一、误差：量级、模型与失效条件

<details>
<summary><b>1.1 两种误差的解析形式（点击展开）</b></summary>

设 $u_{64}=2^{-53}$，$u_{32}=2^{-24}$，内维为 $k$。

**原生 FP64（A100 / cuBLAS）**：累加在 FP64 中顺序进行（或分块进行），
$$\|\hat C_{\text{nat}}-C\| \le \gamma_k\,\|A\|\|B\|,\qquad \gamma_k=\frac{k\,u_{64}}{1-k\,u_{64}}$$
误差**随 $k$ 线性增长**：$k=10^4$ 时最坏可达 $\sim 10^{-12}$ 量级（统计上通常为 $10^{-15}\!\sim\!10^{-14}$）。

**Ozaki（FP16 切片，$s$ 片，FP32 累加 + 无误差变换）**：
$$C=\sum_{i=1}^{s}\sum_{j=1}^{s}A_iB_j,\qquad \text{仅 } \tfrac{s(s+1)}{2} \text{ 个块非平凡}$$
$$\|\hat C_{\text{oz}}-C\| \lesssim \big(c\,s\,u_{64}\big)\|\,|A||B|\,\| \;+\; k\,u_{32}\,2^{-p_{\text{lo}}(s-1)}\|\,|A||B|\,\|$$
- 每片 11 bit（FP16/TF32），$s_{\min}=\lceil 53/11\rceil=5$，实用取 $s=7$（77 bit 覆盖，含指数对齐的余量）。
- 第二项 $2^{-11\times6}=2^{-66}$ 已低于 FP64 的 ulp，**被丢弃的尾部远小于舍入误差**，故前导项主导 → 结果**与 $k$ 基本无关**。
- GEMM 次数 $N_g = s(s+1)/2 = 28$（$s=7$）或 $15$（$s=5$）。
</details>

**误差量级对照表**

| 方案 | 有效位 | 典型相对误差 | 随 $k$ 增长 | 正确舍入比例 |
|---|---|---|---|---|
| A100 原生 FP64 | 53 bit | $10^{-15}\!\sim\!10^{-13}$ | 线性 | 低（文献在同类原生 DGEMM 上测得仅 **2–13%**） |
| Ozaki I / FP16, $s=3$（6 GEMM） | ~33 bit | $\sim10^{-7}$（FP32 级） | 弱 | — |
| Ozaki I / FP16, $s=5$（15 GEMM） | ~55 bit | $10^{-15}$ 级 | 弱 | 中 |
| **Ozaki I / FP16, $s=7$（28 GEMM）** | ~77 bit | **$\le 1\!\sim\!3$ ulp $\approx 10^{-16}$** | 基本无关 | **高（常 >50%）** |
| **Ozaki II / INT8, $r\ge16$** | 正确舍入 | **$\le 0.5$ ulp** | 无关 | **接近 100%（well-scaled 输入）** |

> 关键事实：**在 $k$ 很大时，Ozaki 比 A100 原生 DGEMM 更准**。原生 DGEMM 只有极少数元素被正确舍入，而切片数足够的 Ozaki 可以对几乎所有元素正确舍入。

### 1.2 三个需要避免的误解

1. **Ozaki 不等于更高精度**。它只把"GEMM 这一步的舍入误差"压到 FP64 的 ulp 级。若 $\kappa(A)=10^{12}$，解仍然只有 4 位有效数字——这与 DD/MPFR 有本质区别。真的要超越 FP64，需要增大 $s$（多分量 / 任意精度），这正是 Ozaki-II 论文所做的事情。
2. **动态范围才是主要风险**。FP16 的指数范围仅约 $2^{-24}\!\sim\!2^{15}$（有效指数跨度约 40 bit），而 FP64 是 $2^{\pm1023}$。输入跨越多个数量级时，切片必须重新归一化，否则精度会快速下降。NVIDIA 的 ADP 工作用 **ESC（Exponent Span Capacity）** 估计所需切片数，并对病态输入**回退到原生 FP64**；这是让"保证精度"成立的必要工程手段。
3. **尾部切片不能无限制地累加**。在 FP32 累加器中累加 $k$ 项时，需要保证 $k\,u_{32}\,2^{-11(s-1)}$ 仍在预算内；当 $k$ 极大（$>10^6$）时，需要按 $k$ 分块（Mukunoki 的工作就专门做了内维分块）。

---

## 二、性能：先看理论收益，再对照实测

<details>
<summary><b>2.1 性能模型（点击展开）</b></summary>

$$P_{\text{eff}} \;=\; \underbrace{\frac{P_{\text{low}}}{N_g}}_{\text{GEMM 上限}}\cdot \eta_{\text{fuse}} ,
\qquad N_g=\frac{s(s+1)}{2}\;\text{（Ozaki I）或}\;\approx 3r+1\;\text{（Ozaki II）}$$

$$T \;=\; \frac{2n^3 N_g}{P_{\text{low}}}\,\eta^{-1} \;+\; \underbrace{T_{\text{split}}}_{\propto\, s\,n^2}\,\text{（带宽）} \;+\; \underbrace{T_{\text{recon}}}_{\propto\, N_g\,n^2}\,\text{（带宽）}$$

实测 $\eta_{\text{fuse}}\approx 0.3\!\sim\!0.6$：损耗主要来自 (a) FP64→低精度的拆分/取模（SIMT 流水），(b) 中间矩阵的反复写回显存。SC26 的 **EmuGEMM** 通过融合内核把冗余显存往返消除，达到 INT8 峰值的 81–83%——这说明**内核融合是实现高收益的关键**，不能只看朴素实现的实现效率。
</details>

**为什么只有消费卡能靠 Ozaki 获得收益**

| 设备 | FP16 TC | INT8 TC | FP64（原生/TC） | $\dfrac{P_{\text{INT8}}}{P_{\text{FP64}}}$ | 除以 $N_g\!\approx\!25$ 的上限 |
|---|---|---|---|---|---|
| A100 SXM 80G | 312 TF | 624 TOPS | 9.7 / **19.5** | $32\times$ | **$\approx1.3\times$** → 开销消耗掉后基本持平 |
| GH200 (H100) | 989 TF | 1979 TOPS | 34 / 67 | $30\times$ | $\approx1.2\times$ |
| RTX 4090 | 165 TF | 661 TOPS | 1.29 / 1.29 | **$512\times$** | **$\approx20\times$** → 实现后 6–8× |
| RTX 5090 | 419 TF | 838 TOPS | 1.64 / 1.64 | $511\times$ | $\approx20\times$ → 实现后 6–8× |

**这是因为生态位差异造成的**：A100 有专用 FP64 Tensor Core（其 FP64 与 INT8 的峰价比只有 32×），而消费卡 FP64 是 **1/64 速率**（与 INT8 的峰价比为 512×）。同一算法在两种硬件上，收益相差 16 倍。

### 2.2 关键性能数字

| 设备 | 原生 FP64 实测 | Ozaki I (FP16) 估算 | **Ozaki II (INT8) 实测/估算** | 相对自身原生 |
|---|---|---|---|---|
| A100 SXM | ~19.3 TFLOPS (TC 路径) | 6–11（**反而更慢**） | ~25–35（估） | 1.2–1.8× |
| GH200 | ~67 TFLOPS | — | **56.6–80.2（实测）** | 0.85–1.2× |
| RTX 4090 | ~1.2 TFLOPS | 2.5–4 | **7.4–9.8（实测）** | **5.7–7.6×** |
| RTX 5090 | ~1.6 TFLOPS | 6–9 | ~10–14（估） | 6–8.5× |
| RTX PRO 6000 Blackwell | ~1.9 TFLOPS | — | — | **13.2×（厂商实测）** |

> 注意一个反直觉的结果：**在 A100 上用 FP16 做 Ozaki 反而比原生 FP64 慢约 2–3 倍**（$312/28=11$，高于 19.5，仅当 $s\le5$ 时才勉强持平）。所以文献在 A100/Hopper 上采用的是 **INT8 路线**，FP16 切片路线只适合较老的 V100 或对精度要求较低的场景。

### 2.3 具体算例：$n=k=8192$ 的 DGEMM

| 实现 | 时间 | 相对 A100 |
|---|---|---|
| A100 原生 FP64（19.3 TFLOPS） | **0.057 s** | 1× |
| RTX 4090 原生 FP64 | 0.92 s | 16× |
| RTX 4090 + Ozaki I/FP16（28 GEMM） | 0.31–0.44 s | 5–8× |
| RTX 4090 + Ozaki II/INT8（9.8 TFLOPS） | **0.11 s** | **~1.9×** |
| RTX 5090 + Ozaki II/INT8（估 12 TFLOPS） | 0.09 s | ~1.6× |

（$2n^3=1.1\times10^{12}$ FLOP）

**结论**：在消费卡上使用 Ozaki 后，FP64 吞吐从约 1.3 TFLOPS 提升到约 10 TFLOPS（7 倍），**但仍然只有 A100 原生 FP64 的一半左右**；RTX 5090 大致能追到 60–70%。只有到了 Blackwell 工作站卡（RTX PRO 6000，13.2× 加速）这一级别，才真正超过 A100 单卡的绝对值。

---

## 三、工程实践要点

1. **优先 INT8（或 Blackwell 上的 FP8/FP4）实现，不要选 FP16 切片**。消费卡上 INT8 TC 速率是 FP16（FP32 累加）的 4 倍，且 Ozaki-II 的模数分解比 Ozaki-I 的 $s^2$ 组合更省 GEMM。
2. **必须做内核融合**。若不融合，重建阶段需要写回 $N_g\times n^2$ 个中间矩阵，在 4090 的 1 TB/s 带宽下会占 5–15% 时间；E 级方案（EmuGEMM、ADP）都实现了这一点。
3. **只在以计算为主的场景中使用**。TME 模型给出阈值：$\text{OI}^*=\dfrac{c_q\,r\,P_{\text{fp64}}}{8P_{\text{int}}}$（B300 上约 $0.56$ FLOP/B）。低于该强度时（GEMV、SpMV、低 batch），模拟结果只有原生的 **0.3–0.9×**，得不偿失。
4. **显存开销为 3–8×**。4090 的 24 GB 在 $n\approx 40000$ 的 FP64 矩阵上会先超出显存，而不是先达到算力上限。
5. **必须有回退路径**。对动态范围极大或可能是 NaN/Inf 的输入，ESC 判定需要 $s$ 过大时，应回退到原生 FP64（ADP 的做法是增加 <10% 的运行时开销来换取"保证精度"）。
6. **不要把它当成"高精度计算"**。需要超越 FP64 的场合（病态方程组、$\kappa>10^{14}$），要增加 $s$ 做成多分量（DD/TD/QD），成本随 $s^2$ 甚至 $s$ 增长，收益会快速消失。

---

## 四、估算的不确定性声明

- **有实测支撑**：RTX 4090 的 7.4–9.8 TFLOPS（Ozaki-II/INT8，IJHPCA 2026）；GH200 的 56.6–80.2 TFLOPS；RTX PRO 6000 Blackwell 的 13.2×；RTX 4090/5090 的原生 FP64 与张量核峰值（厂商规格）。
- **我的外推（可信度中等）**：A100 的 Ozaki-II 吞吐（由 GH200 按 INT8 峰值 $624/1979$ 缩放得到 25–35）；RTX 5090 的 Ozaki-II 吞吐（由 4090 按 $838/661$ 与时钟缩放得到 10–14）；消费卡 FP16 路线的 2.5–9 TFLOPS（纯解析模型，**无直接实测**）。
- **实际前需确认**：INT8 峰值是否受 FP32 累加惩罚影响；消费卡上 `INT32` 累加吞吐是否与标称一致；特定矩阵尺寸下的缓存/分块效率。

如果需要，我可以把这套模型写成脚本，输入具体 GPU 规格和 $n,k,s$，直接给出误差上界与耗时预算。
