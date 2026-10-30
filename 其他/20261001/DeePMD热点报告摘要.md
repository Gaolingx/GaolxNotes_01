# DeePMD热点报告摘要

## 🧑‍💻 User

帮我写一篇关于DeePMD的热点报告

---
## 😀 默认助手 (7)

我先查一下可用的检索工具，然后为你收集 DeePMD 的最新进展。

# DeePMD 热点报告（数据截至 2026-10-20）

> **一句话结论**：DeePMD 已经从“一套深度学习势函数软件”演变为**以大原子模型（LAM）为核心的原子模拟基础设施**——2026 年的两条主线是 **DPA4/SeZM 架构对“精度—成本前沿”的正面冲击**，以及 **训练、部署、智能体化工作流的工程化闭环**（多后端 + 预训练模型仓库 + Agent Skills）。

---

## 一、为什么此刻值得关注：三条同时发生的变化

| 变化 | 关键事实 | 意义 |
|---|---|---|
| 架构换代 | `v3.1.0`（2025-06）引入 **DPA-3**；`v3.2.0b0`（2026-06-07）引入 **DPA4/SeZM**；`v3.2.0`（2026-08-22）正式发布 **DPA4 家族** | 从“描述符+MLP”走向**等变消息传递 + 大原子模型** |
| 论文兑现 | DPA4 预印本（arXiv:2606.02419，2026-06-01）宣称在 Matbench Discovery 各项指标领先 | 首次在公开榜单上正面挑战 MACE、eSEN、ORB 系模型 |
| 交付形态改变 | 内置预训练模型下载（`dp pretrained download`）、`pt_expt` 可导出后端、官方 **Agent Skills** | 使用者从“调参训练者”变成“模型调用者 + 微调者” |

---

## 二、头条热点：DPA4 / SeZM —— 精度—成本前沿

<details open>
<summary><b>技术要点（arXiv:2606.02419，Tiancheng Li 等，2026）</b></summary>

- **定位**：SE(3) 等变架构，明确针对“最强等变模型推理太贵”的痛点。
- **核心构件**：边-节点特征在**边局部坐标系**中跨所有角动量通道耦合（EMFA 型 **SO(2) 卷积**）+ Wigner 双线性非线性；在精确求积网格上**等变到机器精度**。
- **规模谱系**：从 $0.48\text{M}$ 到 $25\text{M}$ 参数共六档。
- **训练路径**：**只用保守能量—梯度路径**即可达到榜单级精度（无需去噪预训练或直接力预训练），依赖约 3 倍加速的编译实现。
- **宣称收益**：在若干档位上达到已发表最强模型的精度，同时推理吞吐高数倍到一个数量级。
- **工程侧面**：`v3.2.0` 中伴有 **DPA4C**——严格局域、可压缩的紧凑版本，主打高通量/大规模 MD（图原生执行、融合 CUDA 推理、原生自旋、电荷/多重度条件化）。
</details>

<details>
<summary><b>榜单真实位置：需要降温解读的“领先”</b></summary>

论文投出时声称“DPA4 leads every ranked metric”。但截至 2026-10-20，Matbench Discovery 榜单（按 CPS 综合分排序，43/53 个合规模型）显示：

| 排名 | 模型 | CPS | F1 | 参数量 |
|---|---|---|---|---|
| 1 | Prophet-OAME-MBD | 0.912 | 0.928 | 62.3M |
| 2 | TECE-OAM-RRA-1.0 | 0.908 | 0.929 | 222M |
| 3 | EquFlashV2* | 0.907 | 0.929 | 44.9M |
| 8 | eSEN-30M-OAM | 0.888 | 0.925 | 30.2M |
| **16** | **DPA-4.0.1-Pro-MPtrj** | **0.840** | **0.857** | **22.8M** |
| **20** | **DPA-3.1-3M-FT** | **0.802** | **0.884** | **3.27M** |
| 22 | MACE-MPA-0 | 0.795 | 0.852 | 9.06M |

*CPS = 发现 F1（50%）+ 几何 RMSD（10%）+ 声子 $\kappa_{\text{SRME}}$（40%）。*

两点判断：
1. **“领先”是时间切片而非稳态**——2026 年上半年 OAM 系（大规模开放原子数据集预训练）模型密集上榜，把前排整体推高；
2. **数据预算是隐藏变量**：DPA4-Pro 这一条目基于 MPtrj 训练，与 OAM 系模型的预训练数据量不可直接横向比较。**建议把“精度/参数量/训练数据/推理吞吐”四元组一起看**，这正是 DPA4 论文强调“Pareto 前沿”的方法论价值。
</details>

<details>
<summary><b>第三方独立验证（更有说服力的信号）</b></summary>

arXiv:2610.11878（2026-10-08）用一致 AIMD 参考数据，对 **9 个通用 MLIP** 在 5 种二元 + 2 种高熵难熔碳化物（300–1200 K）做弹性常数基准：

- 总体最优：**MACE-MH-1**，MAPE $5.7\%$；
- **热软化趋势**最准：MACE-MH-1（$2.5$ 个百分点）与 **DPA4-Mini（$2.3$ 个百分点）**；
- 所有模型普遍低估刚度；TaC 类群-V 碳化物最难。

→ DPA4 的最小档位在“温度相关物性”这一硬指标上已被独立工作认可，这是比自报榜单更可信的采用信号。
</details>

---

## 三、热点二：从势函数到“大原子模型”生态

演进链条清晰可查：

$$\text{DP (PRL 2018)} \rightarrow \text{DeePMD-kit v2 (JCP 2023)} \rightarrow \text{DPA-2 (npj Comput. Mater. 2024)} \rightarrow \text{DPA-3 (2025)} \rightarrow \text{DPA4 (2026)}$$

- **DPA-2**（多任务 LAM 原型）：确立“模型中心化”范式——预训练 + 微调 + 蒸馏，而非一次性拟合。
- **DPA-3**（`v3.1.0`，2025-06-11）：消息传递架构，可跨学科数据集**联合训练**，并支持电荷/自旋条件化。
- **预训练模型仓库**（`v3.1.3`，2026-03-21）：`dp pretrained download` 一行命令取模型，已收录 **DPA-3.1-3M**、**DPA-2.4-7M**、**DPA3-Omol-Large**（面向有机分子/OMol 类数据）。
- **DPA-ADAPT**（`v3.2.0`）：把预训练 DPA **从势函数扩展到下游原子性质预测**——描述符回归、冻结骨干、全量/多任务微调，附数据转换、预测、评估、交叉验证工具链。

```bash
# v3.1.3 起的三步上手（官方 release note 原文）
curl -fsSL https://dp1s.deepmodeling.com | bash
dp pretrained download DPA3-Omol-Large
dp test -m ~/.cache/deepmd/pretrained/models/DPA3-Omol-Large.pt -s path_to_your_system
```

---

## 四、热点三：工程化 —— 多后端、图原生、智能体

<details>
<summary><b>后端与部署（v3.x 主要工程量都在这里）</b></summary>

- **多后端框架**（v3 立身之本）：TensorFlow / PyTorch / **JAX** / PaddlePaddle，配套后端无关的 `dpmodel` 共享层。
- **`pt_expt`（可导出 PyTorch 后端）** 成为主线：`torch.export` + `torch.compile` + **`.pt2`（AOTInductor）** 推理，落地 C/C++、LAMMPS、DeepEval；`v3.2.0` 明确“从训练到图原生部署的完整路径”。动机之一是 `torch.jit` 退役。
- **图原生化**：`NeighborGraph` 抽象 + **vesin** 邻居表，Python/ASE 推理达到 $O(N)$；`dpa1`（`attn_layer=0`）全链路图原生。
- **JAX 侧**：训练、微调、多任务、Hessian、导出，以及新增 **JAX-MD 接口**。
- **宿主生态**：LAMMPS（含多 rank 推理）、**GROMACS**（NNPot + DeePMD C++/CUDA 后端）、i-PI、ASE、DeepEval、dpdata、DP-GEN（主动学习）。
</details>

<details>
<summary><b>“Agent-ready”：2026 年最值得注意的产品信号</b></summary>

`v3.2.0` 发布**官方 Agent Skills**：把安装、训练、微调、推理、MD 部署封装为可复用的领域指令，供 AI 智能体稳定执行，同时保留对科学工作流的显式控制。

这条与 **DPA-ADAPT**、**内置模型注册表**构成同一条逻辑：*降低“从想法到一次可信模拟”的操作成本*。对科研用户的实际影响是：DeePMD 正在从“需要学习的软件”变成“可被编排的服务”。
</details>

---

## 五、热点四：应用前沿（2025–2026 年公开文献抽样）

| 方向 | 代表工作 | 看点 |
|---|---|---|
| 聚变/核材料 | V-4Ti-4Cr 合金（arXiv:2610.08291） | MatterSim 采样的“两阶段”流程：基础模型采样 → 少量 DFT 精调；$9.2\,\text{meV/atom}$ 能量误差，算出弹性模量与熔点（纯 V 1950 K） |
| 固态电解质 | 3000 万候选高通量筛选（arXiv:2608.25592） | MatterSim 预筛 + 体系专用 DeePMD 做动力学验证；命中 97 个高导率候选；提出“Li 跳跃网络连通性”才是室温电导率主因 |
| 生物分子 MD | GROMACS 集成（arXiv:2602.02234 / 2604.07276） | DPA2 吞吐比 DPA3 高至 $4.23\times$（A100）/ $3.18\times$（GH200）；多 GPU 强扩展 16 卡 66%、32 卡 40%；**>90% 时间花在 DeePMD 推理** |
| 二维量子材料 | 扭转 TMD 莫尔超晶格（arXiv:2604.22343） | DeePMD + DeepH + DMRG 串联，3000+ 原子超胞，动态莫尔势与 Wigner 结晶 |
| 磁性与自旋 | Fe-Cr-C 磁性势（arXiv:2507.18935） | 非磁训练对输运性质更好、自旋极化训练对体积/晶格更好；预训练+微调把成本降一个量级以上 |
| 行星/星际化学 | CO 冰光解吸（arXiv:2506.10882） | 振动激发态 $v\lesssim 40$ 仍准确，揭示平动—转动能量耦合 |
| 相变存储 | Ge-rich GeTe 结晶动力学（arXiv:2404.15128） | ns 尺度相分离与成核机制 |
| 极端条件地球物理 | $\delta$-AlOOH 弹性（PRM 2024） | SCAN 泛函 + DeePMD，140 GPa / 2700 K |

**方法学趋势**：*“通用基础模型粗筛 + DeePMD 精调/验证”的两段式流程正在成为默认范式*——DeePMD 的角色从“唯一模型”变为“可问责的高精度终点”。

---

## 六、规模与性能里程碑（这条线一直是 DeePMD 的护城河）

- 2020：$10^8$ 原子、Summit 上 91 PFLOPS（双精度），>1 ns/天。
- 2024（SC24）：Fugaku 上 **149 ns/天**，相对原 DeePMD-kit 加速 $31.7\times$，首次让“毫秒级 ab initio 精度”进入一周可达范围。
- 2026：GROMACS 域分解多 GPU 推理打通，瓶颈被清晰定位为**幽灵原子成本（由截断半径决定）+ rank 间负载不均**。

---

## 七、竞争格局（2026 年 10 月视角）

<details>
<summary><b>展开：谁在压着谁</b></summary>

- **第一梯队（榜单）**：OAM 系预训练模型——Prophet-OAME、TECE-OAM、EquiformerV3+DeNS、GRACE-3L-OAM、PET-OAM-XL、eSEN-30M-OAM。共同点是**超大规模开放数据预训练**。
- **架构阵营**：MACE 系（含 MACE-MH-1，且在难熔碳化物上总体最优）、eSEN、ORB、SevenNet、GRACE、Nequix、Allegro、MatRIS。
- **DeePMD 的位置**：`DPA-4.0.1-Pro-MPtrj` CPS 0.840 / 第 16，`DPA-3.1-3M-FT` 0.802 / 第 20。**精度上不再“断层领先”，但工程完备度（后端、LAMMPS/GROMACS/i-PI、压缩、自旋、长程、多任务、主动学习闭环）仍属最完整的一档**。
- **差异化护城河**：DPLR（长程静电）、DeepSpin（自旋）、DeepPot-SE 系列工业级稳定性、以及中国本土 HPC（Fugaku/超算）适配与中文社区。

**判断**：2026 年 DeePMD 的竞争压力**不在“能不能算准”，而在“预训练数据规模与模型即服务化”**。DPA4 的“效率优先”路线本质上是对这一压力的回应。
</details>

## 八、社区与采用度看板

<details>
<summary><b>展开：量化指标</b></summary>

| 指标 | 数值（2026-10-20） |
|---|---|
| GitHub Stars / Forks | 2,062 / 661 |
| 仓库创建 | 2017-12-12（2025-12 满 8 年，2026-12 将满 9 年） |
| 未结 Issue / PR | 241 / 71 |
| PyPI 下载 | 近 30 天 **17,144**；近 7 天 8,069；昨日 1,772 |
| 引用（Semantic Scholar） | DeePMD PRL 2018：**1,691**；DeePMD-kit v2：**613**；DeePMD-kit v3（JCTC 2025）：**119** |
| 最新版本 | `v3.2.0`（2026-08-22），当日仍在推送提交 |

</details>

---

## 九、风险与未解问题

1. **基准饱和与“榜单通胀”**：Matbench Discovery 前排 CPS 已进入 0.9 区间，边际收益递减；CPS 权重（F1 50% / RMSD 10% / 声子 40%）本身是价值判断，单一分数易被过度解读。
2. **数据许可与可复现性**：OAM/OMat/OMol 系预训练数据与 MPtrj 的许可差异，使跨模型对比存在结构性不公平。
3. **物理边界**：磁性体系（自旋涨落）、激发态/非绝热耦合（DeePMD 已有扩展，如 arXiv:2407.10881）、长程静电仍是活跃前沿而非成熟功能。
4. **部署开销成为新瓶颈**：GROMACS 实测中 kernel launch 开销与域分解推理是首要优化目标；$O(N)$ 邻居表、融合 CUDA、图原生执行是对策，但尚未完全解决多节点弱扩展。
5. **生态碎片化**：`pt_expt` / `pt` / `tf2` / `jax` / `dpmodel` 多路径并存，短期提高了学习与迁移成本（`v3.2.0` 亦含破坏性变更，如移除 in-tree GROMACS 补丁）。

---

## 十、接下来 3–6 个月值得盯的节点

- **2026-12**：仓库 9 周年，通常伴随版本与生态总结（历史上 `v3.1.2` 即发布于 8 周年当天）。
- **DPA4 正式期刊发表**：arXiv 已至 v4（2026-08-20），转正后引用与采用会加速。
- **DPA4C 实测数据**：目前仅有“紧凑/高通量/可压缩”的定性描述，缺独立 benchmark。
- **DPA-ADAPT 生态**：若第三方在其上发布性质预测模型库，DeePMD 将从“势函数”真正跨到“原子性质基础模型”。
- **Agent × MD**：Agent Skills + MCP 类接口的组合，可能改变“谁在跑模拟”（人 vs 智能体）的分工。

---

<details>
<summary><b>信息来源（可核查）</b></summary>

- GitHub `deepmodeling/deepmd-kit` releases atom feed 与 `v3.2.0` / `v3.1.3` release notes（2026-08-22、2026-03-21）
- GitHub REST API 仓库元数据（stars/forks/pushed_at）与 pypistats.org 下载量
- arXiv:2606.02419《DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution》
- arXiv:2610.11878（2026-10-08，9 个 uMLIP 难熔碳化物弹性基准）
- Matbench Discovery 排行榜（matbench-discovery.materialsproject.org，2026-10-20 抓取）
- arXiv:2312.15492（DPA-2，npj Comput. Mater. 10, 293, 2024）
- arXiv:2502.19161（DeePMD-kit v3，JCTC 2025）；arXiv:2304.09409（v2，JCP 2023）；arXiv:1707.09571（PRL 120, 143001）
- arXiv:2602.02234、2604.07276（GROMACS 集成）；arXiv:2610.08291、2608.25592、2604.22343、2507.18935、2506.10882（应用案例）
- Semantic Scholar Graph API 引用计数
</details>

---

**需要我把它改成别形态吗？** 例如：① 面向管理层的 1 页版；② 面向组会的 10 页 PPT 大纲；③ 聚焦“DeePMD vs MACE”的深度技术对比；④ 附带你所在体系的落地路线（选模型 → 数据 → 训练 → MD 部署）。告诉我用途和读者，我直接改。
