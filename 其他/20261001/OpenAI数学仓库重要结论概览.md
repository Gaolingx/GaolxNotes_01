# OpenAI数学仓库重要结论概览

## 🧑‍💻 User

https://github.com/openai/math/tree/main 帮我看看这里面有什么计算机、拓扑、数论、几何等领域的重要结论。

---
## 😀 默认助手 (7)

I'll take a look at that repository for you.

The repo is real — OpenAI's `math` repository (722 manuscripts in 372 families). Let me pull the manuscript map and overview to find the important results.

CONTENTS.md came through as a single 577k-char line (newlines stripped). Let me get it in a form I can page through, plus the `overview.tex`.

Good — the Markdown version has 1098 real lines. Let me read it through in pages.

Halfway through the catalogue. Let me continue reading.

我已完整读取该仓库的 `README.md` 和全部 `CONTENTS.md`（1098 行，372 个结果族）——**覆盖率 100%**；各篇 PDF 正文与 `overview.tex` 未逐篇精读，因此下列判断基于每个结果族的官方摘要。

---

## 一、这是什么仓库

`openai/math` 是 OpenAI 公开的一个**数学手稿库**：由内部未发布模型生成的 **722 篇手稿、372 个结果族（family）**，按数学分支分类。

| 关键信息 | 内容 |
|---|---|
| 产出方式 | 平均每个结果约 **3 小时 ChatGPT Pro thinking 算力**；全程约喂了 **4000 个问题** |
| 例外 | Riemann zeta 零自由区、**CM 阿贝尔簇的 Hodge 猜想** 两项为特殊流程；zeta 那篇经人工编辑 |
| 验证状态 | 部分有 **Lean 形式化**（见 `lean/`、`lean/formalization.yaml`）；README 明确警告 **"未形式化的结果可能有错"** |
| 目录 | `preprints/`（PDF+源码）、`lean/`、`reasoning_traces/`（10 篇推理摘要） |
| 编号 | 结果族编号 001–377，但 **045、061、070 缺失**（缺号） |

> ⚠️ 重要提醒：这批结果**强度极高**（UGC、$\mathsf{L}=\mathsf{RL}$、$\omega\le 9/4$、Hilbert 第十问题、自由群因子同构……）。它们大多**未经同行评审、多数无 Lean 形式化**，应按"AI 生成的待验证声明"看待，而非已确立的定理。

---

<details>
<summary><b>二、计算机科学 / 计算复杂性（点击展开）</b></summary>

### 核心亮点

| 编号 | 结论 |
|---|---|
| **102** | 证明 **唯一博弈猜想（UGC）**；并给出 Max-Cut 达 Goemans–Williamson 比值、Vertex Cover 因子 2、Min-UnCut、有向 FVS 的**最优 NP-硬度** |
| **103** | $\mathsf{L}=\mathsf{RL}=\mathsf{BPL}$ —— 对数空间随机计算**完全去随机化** |
| **107** | 矩阵乘法指数 $\omega\le 9/4$（$\mathbb C$ 上）；进一步 $\omega<2.258$，任意域上 $\omega<2.371054886$ |
| **109** | 整数乘法突破 $n\log n$：$O(n(\log n)^{1-\kappa})$，$\kappa=2^{-182}$，反驳 Schönhage–Strassen 最优性猜想 |
| **130** | 精确 DFT 达 $O(n(\log n)^{1-10^{-13}})$，反驳 $\Omega(n\log n)$ 下界 |
| **108** | $n\times n$ 永久式的 border 行列式复杂度 $\Omega(n^3)$ |
| **112** | 深度 3 布尔电路下界越过平方根：$2^{\omega(\sqrt n)}$ 个门 |
| **133** | Weisfeiler–Leman 的 $n^{\Omega(k)}$ **无条件**时间下界；维数入输入时 EXPTIME-完全 |
| **126** | 完美匹配多面体**不存在多项式大小的半定提升**（PSD rank 指数级） |
| **138** | Subset Sum 的 $O(2^{0.49n})$ 随机算法 |
| **136** | PPAD 的**拟线性 PCP 定理** |
| **274** | Parity $\notin$ QAC⁰（Moore 猜想） |
| **275** | 连续库仑能量的 **QMA-硬度**（含仅单位电荷原子核） |
| **279** | 固定有限门集下的**精确量子因式分解** |
| **284** | 随机 vs 量子查询复杂度的**四次分离**是最优指数（否定三次猜想） |

### 其他值得一提

| 编号 | 结论 |
|---|---|
| 104 | mean-payoff / 随机 / 奇偶博弈的确定性拟多项式算法 |
| 105 / 106 | 2-to-1 博弈（完美完备性）；三色可着色图难染色 |
| 120 / 121 / 122 | 近线性匹配；近线性编辑距离近似；trace reconstruction 的拟多项式界与 $n^{\Omega(\log\log n)}$ 下界 |
| 113 / 114 / 115 | 完美匹配计数、整数多拟阵公共基、列联表的 **FPRAS** |
| 131 | Kannan–Tetali–Vempala 交换链 $O(n^8)$ 混合 |
| 132 / 127 | 敏感度 vs 块敏感度的**超二次分离**；PTF 平均敏感度 $\le 8d\sqrt n$ |
| 110 / 111 / 124 / 125 | $k$-server $O(\log^2 k)$；单样本拟阵 prophet inequality；三机单位作业调度；$k$-median 阈值 $1+2/e$ |
| 094 / 099 / 089 | $L_p$ 维数压缩 $n^{o(1)}$；编辑距离的 $\ell_1$ 失真是 $\exp(\Theta(\sqrt{\log d\log\log d}))$；平面图 $L_1$ 嵌入 |
| 135 / 141 / 142 | 深度 5 IMM 下界 $n^{\Theta(\sqrt n)}$；实存在理论落入计数层级；素域上确定性多项式分解 |
| 243 / 245 | 无选择多项式时间不捕获 P；纯类型系统中弱正规化⇒强正规化 |
| 235 | 随机 $k$-SAT 阈值存在性（注明 Carenini 优先权） |
| 376 | 强迫 Navier–Stokes 流中的**通用计算**（停机问题嵌入流体） |

</details>

<details>
<summary><b>三、拓扑学与几何拓扑（点击展开）</b></summary>

| 编号 | 结论 |
|---|---|
| **062** | **LeBrun–Salamon 猜想**：闭正四元数-Kähler 流形（实维 $\ge 8$）必同伦于紧对称 Wolf 空间；并分类复射影接触流形 |
| **304** | 任意维 **Hilbert–Smith 猜想**：局部紧群对流形的忠实作用是 Lie 群 |
| **305** | **4 维圆盘嵌入猜想失败**；$F_2$ 不是 Freedman–Quinn 意义下的 good；构造无球面流形模型的 PD4 群（否定 **Wall 猜想**） |
| **315** | **4 维 Singer 猜想**：非正曲率流形万有覆叠的 $L^2$-Betti 数仅在中间维非零 |
| **320** | 同伦等价但**不同胚**的闭非正曲率 4-流形 —— **4 维 Borel 猜想（同胚表述）失败** |
| **246** | **Cannon 猜想**：边界同胚于 $S^2$ 的 word-hyperbolic 群作用于 $\mathbb H^3$ |
| **254** | Artin 群的 **$K(\pi,1)$ 猜想**；抛物子群交猜想 |
| **249 / 250 / 257 / 252** | Eilenberg–Ganea 反例；**Boone–Higman 猜想**；无 CAT(0) 作用的双曲群；非剩余有限的无挠双曲群 |
| **311** | **Hovey–Strickland 猜想**：可对偶 $K(n)$-局部谱恰有 $n+2$ 个厚张量理想 |
| **313** | $K(n)$-局部球面的同伦群均为有限生成 $\mathbb Z_p$-模 |
| **308 / 309** | 每个 height 都存在有限 **Smith–Toda 复形**（$V(4)$ at $p=1009$）；素数 3 处 Kervaire 不变量问题 |
| **312 / 317** | **Grothendieck 同伦假设**；严格高阶范畴的 Thomason 模型结构 |
| **318 / 319** | 色分裂（chromatic splitting）与 **Hahn–Wilson 猜想**的反例 |
| **165** | 完全图 / 完全二部图交叉数（**Harary–Hill**、**Zarankiewicz/Turán 砖场问题**） |
| **168** | **Kazhdan–Lusztig 多项式的组合不变性猜想** |
| 310 / 316 / 314 | Quillen 猜想（有理同调）；Curtis 猜想；色不动点损失 = 循环长度 |
| 306 / 321 | 纯 cosmetic surgery 猜想；Wall 的 $D(2)$ 问题反例 |
| 177 / 180 | 每个维度的有界度 coboundary 扩张子；**Barnette 猜想**（3-连通立方二部平面图有 Hamilton 圈） |

</details>

<details>
<summary><b>四、数论（点击展开）</b></summary>

| 编号 | 结论 |
|---|---|
| **004** | **Hilbert 第十问题在 $\mathbb Q$ 上否定**：无算法判定整系数多项式是否有有理根 |
| **003** | **拟 Riemann 假设**：所有 Dirichlet $L$-函数在 $\Re s>7/8$ 无零点（另有 $11/12$ 的独立证明）；**一致排除 Landau–Siegel 零点** |
| **002 / 006** | 低 Selmer 余秩下的**完整 BSD 公式**（含 Tate–Shafarevich 有限）；**Goldfeld 猜想**（二次扭曲解析秩 0/1 各密度 $1/2$，平均解析秩 $\to 1/2$） |
| **030** | **虚二次域上椭圆曲线的模性猜想** |
| **010** | $p=2$ 的无限制 pro-modularity；**$p=2$ 处 Fontaine–Mazur**；Emerton 维数猜想 |
| **005 / 017** | **Catalan 常数无理**；**$\pi$ 的无理性指数恰为 2**（并推出 Flint–Hills 级数 $\sum 1/(n^3\sin^2 n)$ 收敛） |
| **007** | **普通二点 Chowla 猜想**；修正 Elliott 猜想（$\overline{\mathbb Q}$ 上带幂次对数节省） |
| **011** | $p-1$ 素因子统计服从 **Poisson–Dirichlet 律**（Ford–Konyagin–Luca 猜想）；Erdős 的 totient 大纤维 |
| **013** | **Ostmann 逆 Goldbach 猜想** |
| **016** | **阿贝尔簇的 Zilber–Pink 猜想**（含 $\mathcal A_2$ 中曲线的完整情形） |
| **019 / 018** | **$p$-adic section 猜想**（并得 $X_0(N)$、$X_1(N)$ 的 Grothendieck section 猜想）；**Margulis–Platonov 猜想** |
| **001 / 008 / 009** | Milne 理性猜想；**Deligne–Drinfeld 猜想**（GT Lie 代数自由生成）；Bogomolov–Pop 重构 |
| **029 / 028 / 031** | **Artin 原根猜想**（对所有 admissible 底）；**Gaussian moat 猜想**；Uchida 猜想 |
| **023 / 024** | 三次 Gauss 和的第一矩（Patterson）；totient 值个数 $V(x)$ 的渐近公式 |
| **020 / 021 / 022 / 025 / 026** | 四次式无平方值；**Jacobsthal 函数二次界** $h(k)\ll k^2/(\log\log 3k)^2$；非齐次 Duffin–Schaeffer；短埃及分数 $O(\log\log b)$；大素数间隙正下密度 |
| **193 / 194** | **Serre 交重数正性猜想**；**Lech 重数猜想** |
| **032** | **CM 阿贝尔簇的 Hodge 猜想** ⇒ Tate 猜想（有限域）、Hodge 标准猜想 |

> 附：极值组合/加性数论也大量出现——**159**（Erdős 倒数和猜想 + $r_k(N)\le C_kN e^{-c_k(\log N)^{\varepsilon_k}}$）、**160**（van der Waerden 数超指数增长）、**164**（Hindman 有限和积猜想）、**170/171/189**（Ramsey 数与 $r(C_m,K_n)$）。

</details>

<details>
<summary><b>五、几何（代数几何 / 凸几何 / 度量与黎曼几何，点击展开）</b></summary>

### 代数几何与复几何

| 编号 | 结论 |
|---|---|
| **038** | **Fujita 自由性猜想**：$K_X+mL$（$m\ge n+1$）全局生成 |
| **039** | **Nagata 猜想**与极大多点 Seshadri 常数 |
| **040 / 041** | **Bloch 猜想**（$p_g=0$ 曲面）；**强 hyperkähler SYZ 猜想** |
| **047 / 049** | $\mathbb A^4$ 的 **Zariski 消去反例**（$X\times\mathbb A^1\cong\mathbb A^5$ 但 $X\not\cong\mathbb A^4$）；四变量**稳定坐标反例** |
| **050 / 051** | **Griffiths 正性猜想反例**；**Kobayashi 典范丰沛猜想** |
| **058 / 063 / 067** | Kollár–Pardon 猜想（半代数万有覆叠）；**广义 Mukai 猜想**；6 维 **Campana–Peternell 猜想** |
| **032** | CM 阿贝尔簇的 Hodge 猜想；K3 曲面乘积的 Hodge 猜想；Kuga–Satake 对应的代数性 |
| 033–037 | Campana 轨道 Iitaka 子可加性、log abundance、奇点体积间隙 |
| 043 / 052 | $P=W$ 猜想（复合秩）；Beauville 分裂猜想 |
| 046 / 048 / 053 | Shafarevich 全纯凸性反例；Lipman–Zariski 反例；Pixton 完备性反例 |
| 054 | 有几何 K3 范畴的有理立方四维体仍**非有理**（否定 Kuznetsov 猜想） |

### 凸几何与几何组合

| 编号 | 结论 |
|---|---|
| **087** | **对称 & 一般 Mahler 猜想**（Hanner 多胞形与单形为极值，含所有等号情形）；Gromov 宽度 $=4$ |
| **091 / 092 / 101** | 对数 Brunn–Minkowski；覆盖密度最优阶 $\Theta(n\log n)$；强各向同性常数猜想（单形最大化） |
| **088** | Petty 投影体积猜想（$n\ge 4$） |
| **100 / 156 / 158** | Bang 半面积柱覆盖反例；**Borsuk 猜想在 9 维失败**；**平面不可 5-着色**（平面色数只能是 6 或 7） |
| **155** | $\mathbb Z^3$ 中存在无周期铺砌的平移铺砌（**周期铺砌猜想 3 维反例**） |
| **090 / 096 / 097** | 三角格子的普适最优性；Gaussian propeller；Euclidean Steinitz 常数 $\Theta(\sqrt d)$ |
| 095 / 098 / 094 | 无半定提升的双曲锥（否定 Projected Lax 猜想）；双 Lipschitz 维数压缩反例 |

### 度量 / 黎曼 / 分析几何

| 编号 | 结论 |
|---|---|
| **073 / 074** | **Falconer 距离猜想**（所有维数）；3 维 **Kakeya 极大猜想**、4 维 Hausdorff 维数 |
| **071 / 072** | **Koebe 圆域猜想**；**Brennan 猜想** |
| **143** | **Hilbert 第十六问题**：极限环数按次数一致有界 |
| **260** | 时空 **Penrose 不等式**（含带电、Kerr–Newman 版本） |
| **338 / 339 / 344** | **Yau 单值化猜想**（正双截面曲率 $\Rightarrow$ 双全纯于 $\mathbb C^n$）；**Katok 熵刚性**；**Blaschke 度量猜想** |
| **345 / 346 / 349 / 350** | 球面与闭 3-流形上无穷多闭测地线；稳定 integral varifold 奇异集维数 $\le m-1$；Solomon–Yau 最小体积；Yau 节点集猜想 |
| **348 / 347** | 非负曲率 Einstein 4-流形分类；**Arnold 猜想**反例（稳定 Morse 下界） |
| **335 / 337** | Gromov 积分标量曲率—单纯体积不等式、Gromov–Lawson；Cartan–Hadamard 等周与刚性 |
| **340 / 342 / 343** | nearby Lagrangian 反例；Donaldson tamed-to-compatible；**球填充 Siegel–Yao 猜想** |
| **369 / 370 / 375 / 377** | **hot spots**（单连通平面域）；Lane–Emden / Hénon–Lane–Emden；**De Giorgi 猜想（8 维）**；infinity-harmonic 内部 $C^{1,\alpha}$ |
| 333 / 334 | 闭曲面到 $\mathbb R^4$ 的等距浸入；无局部 $\mathbb R^3$ 等距浸入的光滑度量 |
| 358 / 361 / 360 | 无共轭点但无非正曲率的 3-流形；调和函数维数比较反例；弱 MTW ⇒ 凸性 |

</details>

<details>
<summary><b>六、其他领域（算子代数、概率、PDE、逻辑、表示论，点击展开）</b></summary>

- **算子代数/泛函**：**287** 自由群因子同构 $L(\mathbb F_2)\cong L(\mathbb F_3)$；**288** Kadison 相似猜想；**290** Connes 双中心化子猜想；**296** 生成元问题；**322** Tingley 问题；**325** 完备 Crouzeix 猜想；**328** 自反 Banach 空间的非扩张不动点。
- **概率/统计物理**：**221** Mézard–Parisi 公式；**217/227** SK 模型的低温涨落律与临界混合时间 $n^{2/3+o(1)}$；**211** 随机平面地图的 LQG 极限；**213/214** Benjamini–Schramm 猜想；**261** Anderson 模型（2 维纯点谱、$\ge3$ 维绝对连续谱）。
- **遍历论/动力系统**：**145** Rokhlin 多重混合问题；**150** 无理角三角台球弱混合；**146** 标准映射正度量熵；**155** 周期铺砌。
- **PDE/分析**：**362** 相对论 Vlasov–Maxwell 大初值整体光滑解；**367** 一相 Bernoulli 问题的临界维 $7$；**371** 散焦 NLS 稳定爆破；**075** $L\log L$ Fourier 收敛；**080** Carleson 收敛问题端点。
- **逻辑/集合论**：**240** Shelah 最终范畴性猜想（ZFC）；**241** Turing 度的刚性；**244** 划分原理不蕴含选择公理；**242** 单值 Diophantine 表示。
- **群论/表示论**：**248** Thompson 群 $F$ 非 amenable；**253** 无限有限表示单 amenable 群；**202/203** Alperin 权猜想与 Donovan 猜想；**205** Saxl 猜想；**210** Foulkes 猜想。

</details>

---

## 七、给你的阅读建议

1. **先看 `overview.pdf`**：372 个族的官方描述（我读的是 `CONTENTS.md` 的族摘要）。
2. **优先看有 Lean 的**：目录中标了 `([Lean](lean/docs/NNN.md))` 的结果族可信度相对更高（如 003、005、073、102、159、287、362 等）。
3. **看 10 篇推理摘要** `reasoning_traces/`：包含 007、017、087、102、159、197、221、271、287、362 的模型推理过程，是判断"这些结论如何来的"最直接的窗口。
4. **保持怀疑**：README 自己写明"未形式化的结果可能有问题"。像 UGC、$\mathsf{L}=\mathsf{RL}$、$\omega\le9/4$、Hilbert 第十问题、自由群因子同构这类**级联式重大突破集中在同一批未评审手稿**，与正常数学进展的节奏差异极大——建议把它们当作待检验的强声明，而不是既成事实。

如果你想，我可以针对**某一个学科**（例如"把所有数论族逐条列出并解释"）或**某几个编号**做更深入的展开。
