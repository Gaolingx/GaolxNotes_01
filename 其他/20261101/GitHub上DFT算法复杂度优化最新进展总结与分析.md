# GitHub上DFT算法复杂度优化最新进展总结与分析

## 🧑‍💻 User

看看 github 上现在进行的根据 openai 论文作为出发点优化 DFT 的工作，查看最新进展，尝试进一步优化 DFT 的算法复杂度。

---
## 😀 默认助手 (7)

我先查看有哪些可用的浏览器工具和检索工具，然后开始检索。

有一个关键线索：OpenAI 在 2026 年 8 月 1 日发布了“Ten Advances in Mathematics and Theoretical Computer Science”，共含十章。我现在去确认其中哪一章与 DFT 相关。

关键线索：这里的 DFT 指的是**离散傅里叶变换**。OpenAI 有一篇预印本《An Explicit Power-Saving for the Exact Discrete Fourier Transform》，GitHub 上已出现若干跟进仓库。我继续深入调查。

找到了核心工作。现在我去确认最新进展，也就是确认是否还有更新的仓库或论文版本。

数据已经拿到。现在做定量复现与改进分析。

## 先说一句术语

这里的 **DFT = 离散傅里叶变换（Discrete Fourier Transform）**。OpenAI 2026 年 9 月放出的预印本 *An explicit power saving for the exact discrete Fourier transform* 证明：存在一个固定程序，对**任意长度** $n$ 都能在精确复数算术模型下以
$$T(n)=O\!\left(n(\log n)^{1-\kappa}\right),\qquad 1-\kappa\approx 2.1\times10^{-13}$$
算出精确 DFT，第一次打破了 $n\log n$ 这道“经典下界”外观屏障。GitHub 上的“竞赛”就是围绕这个 $\kappa$ 展开的。（如果你问的其实是密度泛函理论，我在最后一节单独给出说明。）

---

## 一、起点：OpenAI 到底放了什么

<details>
<summary><b>展开：论文、模型、形式化状态</b></summary>

- 仓库：`github.com/openai/math`（721 篇手稿 / 373 个 family，约 42% 有 Lean 形式化），该结果属于 **family 130**，在 `preprints/An-explicit-power-saving-for-the-exact-discrete-Fourier-transform-September-25-2026/`。
- 计算模型（关键，它决定了“能不能比”）：精确复算术、系数无限制、指定多项式次单位根、对数词长索引；根的选择与标量预处理也计入成本。**计的是算术工作量，不是比特复杂度**。
- 证明骨架三段：Local Fourier compiler（Newton 约化到三角 Toeplitz 因子）→ 张量网络给出 saving（Theorem 2.6 的 **moment 不等式**）→ §3–§5 的 exact-width Fourier word、扇区同步、小素数工作长度、chirp 卷积，把结论推到所有 $n$。
- 形式化：论文声称 $2.1\times10^{-13}$；Lean 库在 commit `3014888`（10 月 8 日）补上 uniform 版，**证明的是 $1-2\times10^{-11}$**（更保守）。此前 pin 的 commit `adc7f12` 里没有 uniform 语句的形式化。
</details>

---

## 二、GitHub 上正在进行的工作（最新到 10 月 27 日）

<details>
<summary><b>展开：仓库清单与技术路线</b></summary>

**真正的竞技场是排行榜** `bigospeedrun.com/exact-dft`（“Big O Speedrun”，作者 Nil Mamano，数据更新到 10 月 27 日）。它把每个结果分成 *Claimed / Human Verified / Lean Verified* 三档，Lean Verified 意味着它**从源码重建你的 Lean 证明并用 Lean kernel 检查它确实证明了排行榜自己的题目陈述**。当前 $\kappa$ 榜（节选）：

| $\kappa$ | 状态 | 作者 | 日期 | 关键做法 |
|---|---|---|---|---|
| $1-10^{-13}$ | Lean Verified（AI 突破，赛程起点） | OpenAI | 10-06 | 论文主定理 |
| $1-2\times10^{-11}$ | Lean Verified | OpenAI | 10-07 | 补的形式化 |
| $1-7.3\times10^{-5}$ | Claimed | Ryan Shea | 10-08 | **首次把整数乘法网络移植到 DFT 模型**（Swapnil Jain round-6 网络），先手权在它 |
| $1-4.856\times10^{-4}$ | Claimed | eumemic | 10-09 | **batched recursion** + paired-cube 网络（PR #144） |
| $1-8.76\times10^{-4}$ → $1-1.7887\times10^{-3}$ | Lean Verified | jacobalansussman（+Parker L） | 10-10 → 10-21 | “lever”递归 + helper storage tables |
| $1-1.26\times10^{-3}$ | Lean Verified | canghaimeng（Euler agent team） | 10-17 | C588AT 构造、4,824 helper banks |
| $1-1.2661\times10^{-3}$ | Lean Verified | Parker L | 10-20 | 把 4,824 banks 打包成 4,757 + 四阶 moment 证书 |
| $1-2.616\times10^{-3}$ | Lean Verified | Halit Ziya KARTAL | 10-22 | Sussman #29 的派生（386/429 文件 byte-identical） |
| **$1-3.0103\times10^{-3}$** | **Lean Verified（当前最佳）** | **Halit Ziya KARTAL** | **10-22** | 复用 Sussman 递归 + 换更大的整数乘法 building block（“logception”重复），348/430 文件与 Sussman 相同 |

其他非竞速但同等重要的仓库：

- `eumemic/exact-dft-bounds`：写出 **batched recursion** 的数学说明（Theorem 3.1），把 OpenAI 的“每个残维单位各付一层递归”改成“一个 residual dimension $\rho$ 的 frame transition 一次递归调用 $C^{\otimes\rho}$”，条件变成 moment 不等式 $\sum_\rho N_\rho(\rho/m)^\theta<W$；这是把 saving 放大 $2.3\times10^9$ 倍的理论杠杆。
- `CrocSwap/integer-mult-bounds`：复数网络的源头（paired-cube identity $K+H+B=I$、frame/carrier 编译器、三阶段 Cayley cover，$m=72$、29,937 roles/vertex）。DFT 竞赛本质是在“借用”整数乘法竞赛的有限网络再换目标函数。
- `cyberdid/fft-power-saving-verification`（10-14，Zenodo DOI 存档）：唯一一份**独立计算校验**。结论很有价值：local compiler 的正确性在 $\mathbb{Q}(\zeta_N)$ 里对 $r=4\ldots32$ 全部成立、$\mathbb{F}_p$ 上是必要检验；但**有限精度完全不可用**（$\kappa_2(F_r)=1$ 却仍丢 142 位十进制，float64/mpmath 皆然），单独执行时开销是 radix-2 FFT 的 $540\text{–}20{,}200$ 倍。它明确说不证实也不反驳主定理（张量网络、$O(\log^4 r)$ 深度只对上 10/12 个宽度）。
- `spele1100/exact-dft`：$\mathbb{Q}(\zeta_N)$ 上的分量级 C++ 实现。

**趋势一句话**：$\kappa$ 在 3 周内被抬高 $3\times10^{10}$ 倍（$10^{-13}\to3\times10^{-3}$），全部发生在“exact arithmetic 模型 + Lean kernel 验证”的框架内；**没有一篇是新的下界，全是同一个充分条件被喂进更好的有限网络**。
</details>

---

## 三、我尝试做的进一步优化：先把目标函数算清楚

我把 eumemic 的 paired-cube 证书（`certificates/network-children.json`：$m=72$、roles $W=29937$、rank $R=2{,}153{,}528$、deficit $=1936$、23 个宽度的直方图）当成可复算的对象，做了三件事。

### 3.1 精确复现 moment 判定

用 $\sum_\rho N_\rho(\rho/m)^{1-a}\le W$ 二分求解，得到
$$a_{\max}=4.856570\times10^{-4},$$
与社区认证值 $a_c=4856569/10^{10}$ **一致到 7 位**（说明这条不等式就是全部约束，而且是紧的）。

### 3.2 把高维直方图压成两个数（这一步才是重点）

由于 $\sum_\rho N_\rho\rho=R$ 且 $W$ 固定，令 $S=W-\sum N_\rho\frac{\rho}{m}$（**空余 role**）、$A=-\sum N_\rho\frac{\rho}{m}\ln\frac{\rho}{m}$（质量加权对数亏空），则
$$a_{\max}=\frac{S}{A}-\frac{B}{2A}\Big(\frac{S}{A}\Big)^2+\cdots,\qquad B=\sum_\rho N_\rho\frac{\rho}{m}\ln^2\frac{\rho}{m}.$$
数值：$S=26.8889$、$A=55332.97$、$B=135992.46$，一阶给 $4.8595\times10^{-4}$，二阶给 $4.85657\times10^{-4}$（精确值 $4.85657\times10^{-4}$）。进一步用 $R=mW-D$ 消去，得到**闭式**：
$$\boxed{\;a_{\max}\approx\frac{D}{\,R\,\langle\ln(m/\rho)\rangle\,},\qquad D=mW-R\;}$$
其中 $D=72\cdot29937-2153528=\mathbf{1936}$——正是 JSON 里的 `deficit_per_vertex`；$\langle\ln(m/\rho)\rangle=1.849975$（质量加权），对应“质量加权几何平均子宽度” $72/e^{1.85}=11.32$（而算术平均只有 6.22）。

**含义**：$\kappa$ 只由两个数决定——网络的 **rank 亏空 $D$**，和子节点宽度的 **质量加权几何平均**。这是个线性分式泛函。

### 3.3 灵敏度：哪个杠杆值钱？

| 改动 | $\Delta a$ |
|---|---|
| 多 1 个 spare role（$D\!+\!1$） | $2.510\times10^{-7}$ |
| 把 $\langle\ln(m/\rho)\rangle$ 降 0.1（子节点整体加宽） | $2.777\times10^{-5}$ |

**加宽度的边际收益是堆规模的约 110 倍**。也就是说，现在榜上“helper banks / 反复重复大 block / 换更大 building block”的路线是在用便宜杠杆，而贵杠杆（把质量从细子节点搬到宽子节点）还没被系统性开采。

### 3.4 真正的改造实验（保持 $m,W,R$ 全部不变，只重排宽度）

| 直方图 | $\langle\ln(m/\rho)\rangle$ | 几何平均宽 | $a$ | 倍数 |
|---|---|---|---|---|
| 原样（paired-cube） | 1.850 | 11.3 | $4.857\times10^{-4}$ | 1.0× |
| 全部子节点取算术均宽 6 | 2.485 | 6.0 | $3.62\times10^{-4}$ | 0.75×（更差） |
| $\rho\le 8$ 的尾部（占 35% 质量）折算到 $\rho=20$，rank 守恒 | 1.229 | 21.1 | $7.34\times10^{-4}$ | 1.5× |
| **同上折算到 $\rho=60$** | **0.848** | **30.9** | $\mathbf{1.064\times10^{-3}}$ | **2.2×** |
| 全部 $\rho=60$（不可实现，上界参考） | 0.182 | 60 | $4.95\times10^{-3}$ | 10× |

而在 $D,R$ 固定时，$a$ 与几何平均宽度几乎是反比：

$$\text{geow}=15\Rightarrow 5.7\times10^{-4},\; 20\Rightarrow7.0\times10^{-4},\; 30\Rightarrow1.03\times10^{-3},\; 36\Rightarrow1.30\times10^{-3},\; 45\Rightarrow1.91\times10^{-3},\; 60\Rightarrow4.93\times10^{-3}$$

**反推当前榜首**：$a=3.0103\times10^{-3}$ 在 $\langle\ln\rangle\approx1.85$ 时需要的 $D\approx 11{,}993$（$6.2\times$ paired-cube），或等价于把质量加权几何平均宽度推到 $\approx 42$。这两个数正好给出两条可检验的路线。

### 3.5 我建议的下一步（可直接落地）

1. **更换目标函数，而不是更换网络来源**。整数乘法竞赛优化的是它自己的 saving，与 DFT 的分式泛函不同——这就是为什么“移植”只能拿到 $4.86\times10^{-4}$，而专门重组（重复大 block / 换尾部）能到 $3.0\times10^{-3}$。建议在 `CrocSwap/integer-mult-bounds` 的搜索回路里把 fitness 换成
$$a(N)=\frac{\sum_\rho N_\rho(m-\rho)}{\sum_\rho N_\rho\,\rho\ln\frac{m}{\rho}},$$
在 $\sum N_\rho\rho=R$、子节点数与 role 容量约束下这是个**线性分式规划**，最优解在可行域顶点 ⇒ **只需枚举 $m$ 与至多 2–3 种宽度组合**，搜索空间极小（我上面枚举两宽度混合时，固定子节点数 $C=346499$ 的格点上已无可行整数解，说明最优解必然落在“少数宽节点 + 长细尾”之外的第三个维度上：$C$ 与 $W$ 必须一起放开）。
2. **优先做“尾部搬运”而不是“规模膨胀”**：按 3.4，把 $\rho\le8$ 的质量搬到 $\rho\in[20,60]$ 是 2.2× 级别的收益，且 $D$、$R$、$W$ 全不变——只动子节点分解。这也是唯一能把 $\kappa$ 推过 $10^{-3}$ 而不把常数再放大一档的路径。
3. **保持精确有理数校验 + 现有 Lean 流水线**：把 moment 用 `artanh` 级数在有理数上做上界（eumemic 的 `verify_moment.py` 就是样板，margin $\approx3.16\times10^{-3}/29937$），再走 `bigospeedrun/verifier` 的 kernel 重建 + OpenAI 的 `ComparatorChallenges` 接口。这样每一步“优化”都可被第三方 5 分钟复算。
4. **别指望实际加速**（这点必须说清）：当前构造有 $\sim2^{2570}$ 个 role、递归底段约 $10^9$ 比特，只有渐进意义；`cyberdid` 的实测是 540–20,200× radix-2 FFT，且 float64 完全不可用。真正可迁移的想法是 **batched recursion on frame transitions + exact-width Toeplitz word**，用在必须精确算术的场合（模小素数卷积、FHE 自举里的 CoeffToSlot 之类）。

---

## 四、几点风险提示

<details>
<summary><b>展开：读榜要注意的坑</b></summary>

- 榜首多为**派生提交**：第 1–3 名 348/430、386/429、411/477 个文件与 Sussman 的 #29 byte-identical，“只改 Fourier 文件、证书数据、导出的界”。这不是造假（排行榜逐条披露），但意味着**进展是若干人叠加的**，不是某个新构造。
- “Human Verified”档对这个题目是空的：**没有任何合资格的人类评审仓库**。`cyberdid` 那份独立校验也只覆盖 local compiler，且明确说“张量网络没实现，主定理既没证实也没反驳”。
- $4.86\times10^{-4}$ 那一条依赖 `CrocSwap/integer-mult-bounds` 的 paired-cube 网络正确性——该仓库只有 maintainer 复核（Codex 辅助），**无独立人类同行评审**；精确复相位部分是书面证明，验证器并没有构造出 cover。
- $(\log\log n)^{4-\theta}$ 这个因子在各家表述里时有时无（`eumemic` 保留、排行榜条目省略），比较 $\kappa$ 时要注意题目陈述是否一致——这正是 Lean Verified 档存在的理由。
</details>

---

## 五、如果你问的是密度泛函理论（Kohn–Sham DFT）

那是完全另一条线，而且 OpenAI 没有相关论文：主流是 DeepMind 的 **DM21**（Science 2021，`pyscf-dm21` 等复现）、**GradDFT**（JAX 可微 KS-DFT）、微软的 **Skala**（可微 KS 求解器），以及 2026 年的 “proxy XC functional 可微框架 + DeepChem”（arXiv:2602.15923）这类工作。那里的“复杂度”话题是 KS 求解从 $O(N^3)$ 到 $O(N)$ 的线性标度、以及神经网络 XC 泛函的振荡性导致 SCF 不收敛（arXiv:2501.12149 就是专门打 DM21 几何优化的）。你要哪一个方向，我可以按那条线重新做一遍同样的“复现 + 定量改进”。

---

**一句话总结**：起点是 OpenAI `math` 仓库 family 130 的精确 DFT 幂次节省（$\kappa\approx2\times10^{-13}$），三周内社区用「batched recursion + 借用整数乘法网络 + Lean kernel 竞技」把它推到 $\kappa=3.0\times10^{-3}$（$3\times10^{10}$ 倍）；我把判定条件压成 $a\approx D/(R\langle\ln(m/\rho)\rangle)$，并算出**把子节点宽度整形（尾部搬运）比扩大规模划算两个数量级**——同样 $m,W,R$ 下把 $\rho\le8$ 的质量搬到 $\rho=60$ 就能从 $4.86\times10^{-4}$ 到 $1.06\times10^{-3}$，这是目前没人系统开采的杠杆。
