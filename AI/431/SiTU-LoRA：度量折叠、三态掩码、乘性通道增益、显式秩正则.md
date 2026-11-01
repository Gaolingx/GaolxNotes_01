# SiTU-LoRA：度量折叠、三态掩码、乘性通道增益、显式秩正则

## 🧑‍💻 User

@联网搜索 peft库对llm进行训练，我们发现部分模型的ffn层用了SiTU门控（例如kimi k3），这个门控会导致大部分更新在一个通道上，adamw训练的时候也会倾向于让lora更新这个适配器参数，导致你虽然设置了rank，但是实际上训练只用了很少的rank，如何解决？

---
## 😀 默认助手 (7)

已联网核对（含 Kimi K3 的官方 config 与建模代码）。先说一个**对你前提的修正**，它直接决定了解法是否对症。

---

## 1. 事实核对：K3 的 SiTU-GLU 到底是什么

`moonshotai/Kimi-K3` 的 `config.json` 确认：`"hidden_act": "situ"`，`activation_situ_beta: 4.0`，`activation_situ_linear_beta: 25.0`。`modeling_kimi_linear.py` 里的定义是：

$$
\text{SiTU-GLU}(x)=\underbrace{\Big(\beta\tanh\!\frac{g}{\beta}\cdot\sigma(g)\Big)}_{\text{gate},\ \beta=4}\odot\underbrace{\Big(\beta_L\tanh\!\frac{u}{\beta_L}\Big)}_{\text{up},\ \beta_L=25},\qquad g=xW_g^\top,\ u=xW_u^\top
$$

关键点：**它是双向 soft-cap 的**，$|h_j|\le\beta\beta_L=100$。所以问题**不是** SwiGLU 那种"巨大激活值/离群点"（那种情况对应 massive activations，arXiv:2402.17762），而是：

> **饱和导致的门控二值化 + 导数稀疏**：$|g|\gtrsim 8$ 时 $\sigma(g)$ 与 $\tanh(g/4)$ 双双饱和，$\partial h_j/\partial g_j\!\to\!0$。这些通道"高能量但梯度被冻住"（只能通过 $up$ 支路 $\partial h/\partial u\approx 4$ 传梯度），只有落在过渡带的少数通道能通过 gate 被改写。

这就是"大部分更新落在一两个通道上"的物理来源：**假设你只对 `down_proj` 加 LoRA，那么可达更新集合被 $h$ 的有效激活通道数限定**。因为 $\Delta W_{down}$ 的行空间 $\subseteq \mathrm{span}(A^\top)$，而 $A$ 的梯度只由 $h$ 的能量谱驱动，于是：

$$
\mathrm{erank}(\Delta W_{down})\ \lesssim\ \mathrm{PR}\!\left(\mathbb{E}[hh^\top]\right)\ \ll\ r
$$

其余 $r-\mathrm{PR}$ 个方向拿不到可观梯度，AdamW 的逐元素归一化又把它放大成噪声随机游走（位移 $\sim lr\sqrt{T}$），而胜出方向是确定性的（$\sim lr\,T$），**winner-take-all，且训练越久越严重**。这个"AdamW 产生集中奇异谱、有效秩低，而 Muon 方向更丰富"的现象在 ISO-LoRA（arXiv:2609.12123）里有直接实证。

---

## 2. 先量化，再动药（否则容易治错）

```python
import torch

@torch.no_grad()
def rank_stats(M: torch.Tensor):
    s = torch.linalg.svdvals(M.float())                # 奇异值
    s2 = s**2
    stable = (s2.sum() / s2.max()).item()              # ||M||_F^2 / ||M||_2^2
    p = s / s.sum();  p = p[p > 0]
    erank  = torch.exp(-(p * p.log()).sum()).item()    # Roy-Vetterli 有效秩
    return stable, erank, (s[0]**2 / s2.sum()).item()  # 末项：top-1 能量占比

@torch.no_grad()
def chan_stats(h):                                     # hook 抓 down_proj 的输入 [N, 3072]
    e = (h.float()**2).mean(0)                          # 逐通道能量
    p = e / e.sum()
    pr = 1.0 / (p**2).sum()                             # participation ratio = 可用秩上界
    return pr.item(), (e.max() / e.median()).item()     # PR、通道离群比
```

判据（经验值）：

| 指标 | 健康 | 有问题 | 含义 |
|---|---|---|---|
| `PR(E[hhᵀ])` | 接近 $r$ | $\le 4$ | **硬上界**，再加 rank 也没用 |
| `erank(ΔW_down)` / r | > 0.6 | < 0.2 | 秩没被用起来 |
| 逐行 $\|A\|$ max/median | < 3 | > 20 | A 塌缩到单行 |
| 通道 $|h_j|$ max/median | < 50 | > 300 | 门控饱和簇 |

---

## 3. 五组解法，按性价比排序

### S1【最高性价比】换优化器：让每步更新的奇异谱天然变平

AdamW 的逐元素归一化给出的是"符号型"更新，谱极度集中；**Muon 用 Newton–Schulz 把动量矩阵正交化成 $G\to UV^\top$，所有奇异值被拉到 1，等价于对更新谱做了白化**，这是对上述失效模式的结构性解药——而且 K3 本身用 MuonClip 预训练，优化器谱系一致，不存在不匹配问题。

- PEFT 侧：对 $A,B$ 两个 2D 参数用 Muon/MuonClip；或直接用 ISO-LoRA（arXiv:2609.12123，做"切空间谱下降"）；Shampoo/SOAP 亦可（本质是对输入二阶矩做白化，与下一节的解析白化同源）。
- 小改动大收益：`peft` 里 `use_rslora=True`（arXiv:2312.03732）+ `lora_A`/`lora_B` 分学习率（LoRA+，arXiv:2402.12354）。前者修掉 $\alpha/r$ 在大 rank 下压低每方向步长的问题，后者修掉 A、B 学习速度不匹配。

### S2 参数化：正交约束 + gauge 重置

$\Delta W=BA$ 对 $A\to RA,\;B\to BR^{-1}$ 不变，但**AdamW 不变**。放任 gauge 漂移会让 $\kappa(A)$ 爆炸（SDS-LoRA arXiv:2606.16454 证明收敛率随条件数退化）。两个零成本的修法：

```python
# 每 k 步做一次 gauge 平衡（QR retraction，O(r^2 d)，可忽略）
Qa, Ra = torch.linalg.qr(A.T)      # A: [r, d] -> 行正交
A.copy_(Qa.T); B.copy_(B @ Ra.T)   # ΔW 完全不变，但每方向步长被拉平
```
- 或直接在 Stiefel 流形上优化 $B$（arXiv:2508.17901，EMNLP'25 Findings）：强制 $B^\top B=I$，有效秩几乎打满；FoRA（arXiv:2605.29317）把它用在 down_proj 上也 report 了稳定增益。
- `lora_dropout` 调到 0.05–0.1：打破"第一步锁定 rank-1 子空间"的确定性，注入各向同性噪声填谱。这条极便宜，先试。

### S3 输入条件数：逐通道白化，并把缩放折算进适配器

这是最"对症"的一条，本质是把 SmoothQuant（arXiv:2211.10438）的平移变换用在**优化**而非量化上。设 $s_j=\big(\mathbb{E}[h_j^2]\big)^{1/2}$，适配器分支吃 $h\oslash s$：

$$
\Delta y=\Delta W\,h=\underbrace{(\Delta W\,\mathrm{diag}(s))}_{新适配器}\;(h\oslash s)
$$

冻结权重完全不动（合并时把 $\mathrm{diag}(1/s)$ 右乘进适配器即可），但适配器看到的输入谱被拉平，$\mathrm{PR}$ 立刻从个位数升到几十。$s$ 只用少量 calibration forward 统计，成本可忽略。

### S4 换掉"低秩"这个错误参数化 —— 这才是根因解

要点：**门控 FFN 里最关键的自由度是"逐神经元的选通"，它是 $d_{ff}$ 维的对角自由度，低秩矩阵天生表达不了**（要表达一个 $d_{ff}$ 维对角 mask，rank 就等于不同取值的个数）。所以别硬在 down_proj 上堆 rank：

- **对角适配器** $\Delta W=\underbrace{BA}_{\text{低秩}}+\underbrace{\mathrm{diag}(s)}_{d_{ff}\ \text{维}}$。带宽很宽、每通道一个自由度，不受输入协方差谱上界约束；对 SwiGLU/SiTU 类 FFN 的性价比远高于同参数量的低秩项。
- **逐通道门控温度/增益** $\beta_j$ 或 $g_j\to g_j(1+\delta_j)$：直接把饱和通道"解饱和"，恢复 $\partial h/\partial g$ 的信息量。这是 SiTU 特有的、最贴合机理的旋钮，每 expert 仅 3072 个参数。
- **门控选择感知的 LoRA**：NA-LoRA（arXiv:2606.31717）把 gated FFN 的"selection misalignment"形式化，用导数时序重要性 mask 给 gate 侧 LoRA 分配容量，无推理开销，直接对标你的场景。
- 若必须保持纯低秩：用高有效秩构造（MoRA、KRAdapter arXiv:2508.00230），而不是把 $r$ 从 16 堆到 128。

### S5 把 rank 放对位置（对 K3 尤其重要）

- K3 是 **896 experts / 每 token 激活 16 个**，每个 expert 只见极少量 token → 输入协方差的样本方差巨大，AdamW 的 $v$ 被"哪个通道先开火"主导，rank-1 锁定更快更狠。
- 优先打**全 token 共享**的通路：`shared_experts`（2 个 dense expert）、`routed_expert_down_proj` / `routed_expert_up_proj`（LatentMoE 的 3584 瓶颈，统计量健康、且是所有 expert 的公共瓶颈，杠杆率最高）、以及 KDA/MLA 的投影层。**在 K3 上打这些模块的 rank，实际有效秩远高于打 routed expert 的 `w2`。**
- 初始化用 PiSSA（2404.02948）/ OLoRA（2406.01775）/ LoRA-GA（2407.05000）；per-module rank 用 Fisher 或梯度敏感性分配（AdaLoRA 2303.10512），而不是全局统一 $r$。

---

## 4. 针对 K3 的落地配置

<details>
<summary>展开：具体模块名、必须先修的坑、以及一个推荐配置</summary>

**必须先修的坑**：HF 的 `KimiSparseMoeBlock.moe_infer` 带 `@torch.no_grad()`，且 `forward` 在 `self.training` 时直接 `raise NotImplementedError`。也就是说**官方 transformers 实现下，routed expert 上的 LoRA 根本拿不到梯度**——你观察到的"只用了很少 rank"如果发生在 routed expert 上，先排查这一条（需换 Megatron/DeepSpeed/ms-swift 一类可训练的 MoE 实现）。

模块名：routed expert 是 `KimiBlockSparseMLP` 的 `w1`(gate)/`w3`(up)/`w2`(down)，dense/shared 是 `KimiMLP` 的 `gate_proj/up_proj/down_proj`；K3 的 `situ` 分支走的是 **`torch.cat([w1(x), w3(x)])` 融合路径**，若某些 checkpoint 融合成 `gate_up_proj`，LoRA 会同时覆盖两条支路，需注意权重初始化与 $s$ 统计要按前后半段分开做。

推荐组合（按优先级）：

```python
LoraConfig(
    r=32, lora_alpha=32,            # alpha=sqrt(r)*const，配合 use_rslora
    use_rslora=True,
    use_dora=True,                  # 逐列幅度（对 down_proj 的输入通道维度尤其有意义）
    lora_dropout=0.05,
    init_lora_weights="pissa_niter_16",
    target_modules=[                # 先只打共享通路
        "shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj",
        "routed_expert_down_proj", "routed_expert_up_proj",
        "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
    ],
)
# 优化器：Muon(MuonClip) 用于 A/B，AdamW 用于其余；每 500 步做一次 QR gauge 平衡
# 附加：diag(s) 逐通道适配器（仅 shared_experts + top-64 高频 expert，控制显存）
```
注意成本：per-expert 的 $s$ 向量是 $3072\times896\times92$，bf16 约 1 GB——所以对角适配器要**限定在 shared experts 与高频 expert**，或存 fp8。
</details>

---

## 5. 怎么证明你真的修好了

跑一个 $3\times3$ 消融，只看两个量（有效秩 + 下游指标），不要只看 loss：

1. **Rank-utilization 曲线**：横轴 $r\in\{8,16,32,64,128\}$，纵轴 `erank(ΔW)/r`。修好之前曲线是平的（加到 128 也只用 3）；修好后应单调上升到 0.5–0.8。这是唯一能证明"rank 被用上了"的证据。
2. **对照组**：同参数预算下比 diag-only、full-rank 参照（full FT 的 $\Delta W$ 有效秩在 700–1000 量级，见 arXiv:2607.21612 的 SVD 分析）、以及"只打 attention"的基线。
3. **机制验证**：把 $h$ 的 top-1 通道置零后重测 loss 与梯度范数——若 loss 断崖式变化，说明确实单通道主导；若加了 S3/S4 后影响变小，说明病治好了。

---

## 6. 文献速查

<details>
<summary>展开引用清单</summary>

- K3 架构与 SiTU-GLU：`moonshotai/Kimi-K3` config + `modeling_kimi_linear.py::SituAndMul`
- AdamW 在 LoRA 上产生低有效秩 / Muon 更好：**ISO-LoRA** arXiv:2609.12123
- LoRA 基向量冗余、Stiefel 正交强制满秩：arXiv:2508.17901；**FoRA** arXiv:2605.29317
- 低秩反向传播的各向异性缩放与条件数：**SDS-LoRA** arXiv:2606.16454
- 门控 FFN 的 LoRA 失效机理（selection misalignment）：**NA-LoRA** arXiv:2606.31717
- 名义 rank ≠ 校准有效秩（26 个公开 adapter 审计）：arXiv:2608.15351
- 高有效秩参数化：**KRAdapter** arXiv:2508.00230；MoRA
- LoRA 奇异结构 / intruder dimensions：arXiv:2410.21228
- 初始化与缩放：rsLoRA arXiv:2312.03732、LoRA+ arXiv:2402.12354、PiSSA arXiv:2404.02948、OLoRA arXiv:2406.01775、LoRA-GA arXiv:2407.05000、AdaLoRA arXiv:2303.10512、DoRA arXiv:2402.09353
- 激活离群与逐通道平滑：Massive Activations arXiv:2402.17762；SmoothQuant arXiv:2211.10438
- 曲率感知预条件：GRIT arXiv:2601.00231
</details>

**一句话总结**：在 SiTU-GLU 上，瓶颈不在 rank 而在"可达方向数"——先换 Muon/SOAP 类做更新谱白化（S1）+ QR gauge 平衡（S2）作为零成本第一步，再用逐通道白化（S3）把 $\mathrm{PR}$ 抬起来，最后用对角/门控温度（S4）替代"在 down_proj 上堆 rank"这条注定低效的路，并把 rank 集中到 shared expert 与 LatentMoE 共享瓶颈上（S5）。

---
## 🧑‍💻 User

所谓rank collapse 
观察到的退化现象分为以下几种：
1. FFN 的 gate/up 中间激活或梯度集中在少数神经元通道
2. LoRA 更新矩阵 \(BA\) 的奇异值集中，effective rank 很低
3. A/B 因子本身的梯度或参数更新集中在少数 rank 维度
4. 增大 r 后，参数量增加，但 loss 和验证集指标几乎没有改善


主要怀疑的模块：
1. gate_proj（SiLU 门控分支）
2. up_proj（门控的值分支）
3. down_proj（FFN 输出投影）
4. 整个 FFN，尚未定位到具体模块

如何设计消融实验判断所说的“更新集中在一个通道”，具体是指哪一种？
这一区分会直接决定下一步应该修改门控结构、优化器、LoRA 初始化，还是对有效 rank 施加显式约束。

---
## 😀 默认助手 (7)

## 0. 先澄清一个提法：这不是四个互斥假设，而是一条因果链

你列的 4 条其实是同一件事的四个**不同空间**上的投影，正常情况下会**同时出现**：

$$
\underbrace{\mathrm{PR}(\mathbb{E}[hh^\top])\ \text{小}}_{\text{L0 输入可达性}}\;\Rightarrow\;
\underbrace{\sigma(\Delta W)\ \text{集中}}_{\text{L1 权重空间}}\;\Leftrightarrow\;
\underbrace{\text{A/B 某些行占主导}}_{\text{L2 因子空间(gauge)}}\;\Rightarrow\;
\underbrace{\partial \text{loss}/\partial r\approx 0}_{\text{L3 容量饱和}}
$$

所以**纯观测无法定位根因**，因为观测到这 4 条是必然的。真正的判定只能靠**干预式消融**：一次只切断链上的一条边，看 $\mathrm{erank}$ 和 $\mathrm{loss}$ 哪一个同时恢复。下面协议的全部设计原则是这个。

另外两个必须先钉死的定义问题：

<details>
<summary><b>坑 1：现象 3（"A/B 更新集中在少数 rank 维度"）在规范未固定时不是一个良定义的命题</b></summary>

$\Delta W=BA$ 对 $A\to RA,\;B\to BR^{-1}$ 不变。你把 $R$ 取成"把主导行分散开"的矩阵，第 3 条的结论就反过来了。所以：

- ❌ 不要用"$\|A_{i,:}\|$ 分布 / 第 i 个 rank 维的更新占比"下结论；
- ✅ 用 **gauge-invariant 量**：$\Delta W$ 的奇异值谱 $\{\sigma_i\}$、$\mathrm{erank}(\Delta W)$、$\sigma_1^2/\|\Delta W\|_F^2$；
- ✅ 若一定要看因子空间，先做 **QR 规范固定**（`Qa,Ra=qr(A.T); A←Qa.T; B←B@Ra.T`），再看 $\|A_{i,:}\|\cdot\|B_{:,i}\|$ 的 Gini。

结论：**现象 2 与现象 3 是同一对象的协变/不变两种写法，不要当两个证据用。**
</details>

<details>
<summary><b>坑 2："一个通道" 有三个不同的索引空间，必须分别测量</b></summary>

| 说法 | 索引空间 | 物理含义 | 对应的处方 |
|---|---|---|---|
| 激活集中在少数神经元 | $d_{ff}$（$h_j$，$j=1..3072$） | down_proj 的**可达子空间**被限死 | 逐通道白化 / 结构改造 |
| **梯度**集中在少数神经元 | $d_{ff}$（$\nabla_h\mathcal{L}$） | 往往反映 $\|W_d\|$ 行范数不均，**不是门控问题** | 权重侧缩放，别怪 SiTU |
| gate 导数稀疏 | $d_{ff}$（$\partial h_j/\partial g_j$） | SiTU 饱和 → gate_proj 学到的东西传不出去 | **门控温度/解饱和** |
| $\Delta W$ 奇异值集中 | 权重空间 $d\times d_{ff}$ | 最终症状 | 优化器 / 显式秩约束 |
| A/B 因子集中 | rank 空间 $r$ | 依赖 gauge，**不是证据** | — |

你说的"更新集中在一个通道"，90% 的场合指的是第 1、3 行；但真正决定处方的是第 1 行还是第 3 行。
</details>

---

## 1. 先把测量基建建好（一次写，全程复用）

```python
import torch

@torch.no_grad()
def erank(M):                      # Roy–Vetterli 有效秩(尺度不变)
    s = torch.linalg.svdvals(M.float()); p = s / s.sum(); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))

@torch.no_grad()
def top_share(M, k=1):             # top-k 能量占比
    s2 = torch.linalg.svdvals(M.float())**2
    return float(s2[:k].sum() / s2.sum())

@torch.no_grad()
def pr(E):                         # participation ratio: 可达秩的硬上界
    p = E / E.sum(); return float(1.0 / (p ** 2).sum())

@torch.no_grad()
def gini(x):
    x = x.flatten().abs().float().sort().values
    n = x.numel(); i = torch.arange(1, n + 1, device=x.device)
    return float((2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n)

@torch.no_grad()
def subspace_capture(dW, h, k=8):  # ΔW 有多少能量落在 h 的前 k 个主方向上
    h = h.float(); h = h - h.mean(0)
    _, V = torch.linalg.eigh(h.T @ h)                 # 升序
    P = V[:, -k:]                                     # [d_ff, k]
    return float(((dW.float() @ P).pow(2).sum() / dW.float().pow(2).sum()))

# 采集 down_proj 的输入激活 h 与回传梯度
class Stat:
    def __init__(self): self.Eh = self.Eg = None; self.N = 0; self.trans = []
    def __call__(self, mod, inputs, output):
        h = inputs[0].detach().float()
        self.Eh = h.pow(2).sum(0) if self.Eh is None else self.Eh + h.pow(2).sum(0)
        self.N += h.shape[0]
        if inputs[0].requires_grad:
            inputs[0].register_hook(lambda g: self._g(g))
        return None
    def _g(self, g):
        g = g.detach().float()
        self.Eg = g.pow(2).sum(0) if self.Eg is None else self.Eg + g.pow(2).sum(0)
```

再加一个**可微的显式秩正则**（r×r 谱，O(r³)，r≤128 时开销可忽略），因为你的第 4 个选项需要它：

```python
def erank_loss(A, B, eps=1e-8):
    """ΔW=BA 的非零奇异值² = eig(AᵀBᵀBA)，只需 r×r 矩阵"""
    M = A.T @ (B.T @ B) @ A
    w = torch.linalg.eigvalsh(M.float()).clamp_min(eps)
    p = w / w.sum()
    return -(p * p.log()).sum()        # 训练时加 -λ·erank_loss 即最大化有效秩
```

---

## 2. 六个实验：按"切断哪条边"来编号

### E0（最重要，先做）：**目标更新的内在秩上界**——判断这是不是伪问题

> 如果任务本身只需要一个 rank-3 的更新，那"rank collapse"是**正确答案**，加 r 当然没用。这一步能省掉你几周的无效调参。

**做法**：取一个小规模参照 run，用"最强配置"（Muon + 正交 gauge + 输入白化，见 E3/E4）在**单个模块**上把 r 拉到 256，训到平台期。然后：

1. $\mathrm{erank}(\Delta W^\star)$ 与 $\|\Delta W^\star\|$ 的 rank-$k$ 截断能量曲线；
2. 同模块**直接学全秩 $\Delta W$**（绕过 LoRA 因子，Adam 直接优化 $d\times d_{ff}$ 矩阵）200–500 步，记录 loss 降幅 $\Delta\mathcal{L}^\star$——这是该模块的**可适应性上界**。

| E0 结果 | 判定 |
|---|---|
| $\mathrm{erank}(\Delta W^\star)\lesssim 4$，rank-8 截断已捕获 >95% 能量 | **伪问题（H4a）**：别再堆 rank，缩 $r$，把预算移到层/模块 |
| $\mathrm{erank}(\Delta W^\star)\gtrsim 20$ 而你 r=32 的 run 只有 3 | **真 collapse**，继续 E1–E5 |
| 任一模块的 $\Delta\mathcal{L}^\star\approx 0$ | **这个模块不该改**——问题不在 FFN，去看 attention / LatentMoE / embedding |

> 这一步同时给了你后面所有阈值的标定点：健康度 = $\mathrm{erank}_{\text{你的}}/\mathrm{erank}(\Delta W^\star)$。

### E1：**模块定位**——三选一还是"整个 FFN"

两个互补的测量，**必须都做**，因为它们回答不同问题（"能学到多少" vs "实际用上了多少"）：

- **E1a 等预算隔离训练**：4 个 arm，参数量严格对齐 —— 只 `down_proj` / 只 `gate_proj`+`up_proj` / 三个都上 / 只 `up_proj`。各自记录 $\Delta\mathcal{L}$ 与 $\mathrm{erank}$。
- **E1b 事后敲除归因（knockout）**：训练好一个全模块 adapter，逐个把某个模块的 $\Delta W$ 置零，测 loss 增量 $\delta_m$，再除以该模块参数量，得到**每参数的 loss 贡献**。

然后画"**能量 vs 归因**"二维图：

| | $\delta_m$（归因）大 | $\delta_m$ 小 |
|---|---|---|
| $\|\Delta W_m\|_F^2$（能量）大 | 该模块在干活，保留 | **在偷 rank**（典型病灶，把 rank 移走） |
| 能量小 | 欠参数化，加 rank | 无关，删掉 |

这一步就能回答你的"4. 整个 FFN，尚未定位"——如果 `down_proj` 落在右下格，那就是它偷了 rank。

### E2：**输入侧 vs 权重侧**——区分 L0 与 L1

逐层（不是全局）测量：

1. $x$ 轴：$\mathrm{PR}(\mathbb{E}[hh^\top])$（$\mathbb{E}[h_j^2]$ 的 PR）—— **L0 的硬上界**；
2. $y$ 轴：$\mathrm{erank}(\Delta W_{down})$；
3. 第三个数：$\mathrm{subspace\_capture}(\Delta W_{down}, h, k{=}8)$ —— **"是不是就一个通道"最直接的单一数字**。

```
判定：
  y ≈ x 且逐层 Spearman ρ > 0.8，且 capture@8 > 0.9
        → L0 是瓶颈：ΔW 完全被 h 的少数通道锁死
  x 大（PR>20）但 y 仍 < 0.2r
        → L0 清白，病灶在优化器/因子侧 → 直接去 E4
  PR(h) 小但 capture@8 也小（<0.5）
        → 更新跑到了"低能量通道"上，这是 W_d 行范数/噪声放大，不是门控
```

再补一个 **E2b：$\partial h/\partial g$ 的饱和占比**（SiTU 特有）

```python
g = gate_preact.float(); b, bl = 4.0, 25.0
dh_dg = torch.sigmoid(g) * (1/b) * (1 - torch.tanh(g/b)**2) \
      + b * torch.tanh(g/b) * torch.sigmoid(g) * (1 - torch.sigmoid(g))
trans_frac = (dh_dg > 0.05 * dh_dg.max()).float().mean()   # 过渡带占比
```
`trans_frac` 若 < 0.05，说明 gate 分支几乎全饱和 → **gate_proj 上的 LoRA 天然学不动**，这是处方"门控温度"的唯一依据。

### E3：**因果切断 ①：只白化适配器输入**（切 L0→L1）

冻结一切、只改一行代码：适配器分支吃 $h\oslash s$，$s_j=\sqrt{\mathbb{E}[h_j^2]}$（冻结权重不动，合并时把 $\mathrm{diag}(1/s)$ 折进适配器）。**其余（优化器、init、r、lr、seed、数据）全部不动。**

| 观测 | 判定 |
|---|---|
| $\mathrm{erank}(\Delta W)$ 从 3 跳到 ≥0.5r，且 loss 明显改善 | **L0 是束缚条件** → 处方：逐通道白化 / 门控温度（结构侧） |
| erank 上去但 loss 没改善 | erank 不是真瓶颈 → 回去看 E0 |
| erank 几乎不动 | L0 清白 → **优化器/初始化侧**，去 E4 |

### E4：**因果切断 ②：只换优化器 / 只加 gauge 固定**（切 L1→L2）

三个 arm，其他全固定：

1. `AdamW`（基线）；
2. `Muon/MuonClip` 用于 $A,B$（Newton–Schulz 把更新谱拉平，$G\to UV^\top$，所有 $\sigma_i\to1$）；
3. AdamW + **每 200 步 QR gauge 平衡**（$\Delta W$ 当步不变，但改变后续动力学）。

若 erank 在 2/3 上显著上升而 $h$ 统计量完全不变 → 这是**优化器侧的 winner-take-all**。理由可量化：确定性方向位移 $\sim lr\!\cdot\!T$，噪声方向 $\sim lr\sqrt{T}$，SNR 按 $\sqrt{T}$ 增长，所以**训练越久越集中**。

### E5：**因果切断 ③：换初始化 + 一步对齐度**（切 H3）

只换 `init_lora_weights`：`True`(默认) / `"pissa_niter_16"` / `"olora"` / `"eva"`。同时加一个**一步诊断**，成本几乎为零但信息量极大：

$$
\cos\!\big(\Delta W_{t=1},\ \Delta W^\star\big),\qquad \Delta W_{t=1}=B_1A_1
$$
（$\Delta W^\star$ 来自 E0）。若 < 0.1（近乎正交），说明初始子空间完全错配，那么**增大 r 也不会改善**，因为 r 个方向全落错地方——这本身就是现象 4 的一种解释。

### E6：**时间分辨**——区分"早期动力学"与"数据驱动的共适应"

把 $\mathrm{erank}(\Delta W_t)$ 每 50 步打一个点，画曲线：

- 前 200 步内塌到 <0.15r 然后冻结 → **初始化 / 早期动力学**（H3 + AdamW 锁定）→ 处方在 E5/E4；
- 缓慢单调下滑到平台 → **数据共适应**（L0 驱动）→ 处方在 E3/结构侧；
- 换 3 个 seed，曲线若大幅摆动 → 你的结论本来就在噪声里，先扩大测量再判。

---

## 3. 判定表：签名 → 根因 → 处方

| 观测签名 | 根因 | 下一步改什么 | 预期效应 |
|---|---|---|---|
| E0: $\mathrm{erank}(\Delta W^\star)\le4$，rank-8 截断 >95% | **H4a 真·低秩，不是 bug** | 缩 $r$，把预算移到其它层/模块 | loss 不变、显存↓，停止内耗 |
| E2: $y\approx x$，capture@8>0.9，逐层 ρ>0.8 | **L0 输入可达性** | 逐通道白化（E3 那样）、门控温度 | erank↑3–10×，loss↓ |
| E2b: `trans_frac`<0.05 且只有 gate_proj 差 | **SiTU 饱和 / gate 导数稀疏** | 逐通道门控增益 $\delta_j$ 解饱和；或把 rank 移出 gate | gate 侧 LoRA 开始有梯度 |
| E2: PR(h)>20 但 erank<0.2r | **优化器 winner-take-all** | Muon/MuonClip、SOAP、QR gauge 平衡 | erank↑，且随 r 单调 |
| E4 arm2/3 才恢复 | 同上，**已确诊** | 同上 + `use_rslora=True` | — |
| E5: $\cos(\Delta W_1,\Delta W^\star)$<0.1；E6 前 200 步塌陷 | **初始化/早期动力学** | PiSSA/OLoRA/LoRA-GA、`lora_dropout=0.05~0.1`、更长 warmup、A/B 分 lr | 塌陷点后移 |
| E1b: down_proj 能量大但 $\delta$ 小 | **rank 被偷** | 把 down_proj 的 rank 移到共享通路 | 同预算 loss↓ |
| E1a: 只有 down_proj arm 差，且 diag-only 同预算更好 | **参数化族错误** | $\Delta W=BA+\mathrm{diag}(s)$、门控温度、MoRA | 打破 rank 天花板 |
| 三个模块 $\Delta\mathcal{L}^\star$ 都小 | **FFN 不是位置** | 换 target（K3: `shared_experts`、`routed_expert_{down,up}_proj`、KDA/MLA 投影） | — |

对"显式秩约束"这条路的定位：**它只在 L1 已经被确认是独立束缚条件时才有意义**（即 E2 显示 PR(h) 大、E3 不改、E4 能改）。否则它只是把 $\sigma$ 谱强行拉平，去拟合一个本来不需要的方向，通常以 loss 变差收场。真要用，用上面的 `erank_loss` 加 $-0.01\sim0.1$ 的权重，并**同时监控 E0 的 $\Delta W^\star$ 谱**作为上限。

---

## 4. 工程坑（会让你整份结论作废的那种）

<details>
<summary>展开</summary>

1. **MoE 路由加权**：K3 是 896 experts / top-16。大量 expert 在一个 batch 里只有几十个 token，其 $\mathbb{E}[hh^\top]$ 与 $\mathrm{erank}$ 全是噪声。**按路由频次加权聚合，过滤 token 数 < N 的 expert**，并单独看 shared_experts（全 token，统计量最干净）。
2. **`moe_infer` 带 `@torch.no_grad()`**（HF 官方实现，且 `forward` 在 training 时直接 raise）。routed expert 上的 LoRA 拿不到梯度——如果你在 routed expert 上观测到"完全不动"，先排查这个，别急着归因于门控。
3. **等参数量比较**：比 $r=32$ 与 diag-only 时，要么对齐参数量，要么同时报告两条曲线；否则"diag 更好"可能只是参数更多。
4. **$\mathrm{erank}$ 必须在同一 calibration batch 上算**，且用 **held-out** batch，不要用训练 batch（训练 batch 上 $\Delta W$ 已经把数据背下来了，谱会虚高）。
5. **别在 step 1 就测**：AdamW 第一步是纯 sign，谱必然集中。至少跑到 lr×200 步，或报告整条曲线。
6. **DDP/FSDP 分片**：$\Delta W$ 的 SVD 需要先把 $A,B$ 全秩 gather，否则算出来是分片的秩。MoE + EP 时尤其容易错。
7. **MXFP4 冻结权重**：白化/折算 $s$ 时不要直接写回 4-bit 权重（精度损失会污染归因），把 $s$ 放在适配器侧。

</details>

---

## 5. 建议的执行顺序（一周可跑完）

```
Day 1   E0（oracle 内在秩 + 每模块可适应性上界）        ← 决定要不要继续
Day 2   E1a + E1b（模块定位，4 个 arm + knockout）      ← 回答"是不是整个 FFN"
Day 3   E2 + E2b（逐层 PR/erank 相关 + 饱和占比）        ← 回答"是不是一个通道"
Day 4   E3（白化，一行改动）                             ← 切 L0
Day 5   E4（Muon / QR gauge，两个 arm）                  ← 切 L1
Day 6   E5 + E6（init / 一步对齐 / erank-t 曲线）        ← 切 H3
Day 7   只在确诊的那一条上做全量 run
```

**一句话回答你的问题**：不要试图通过"观测更像哪一种"来区分——四条会同时为真。用 **E0 定上界**（排除伪问题）→ **E1 定模块**（是不是 FFN）→ **E2 定空间**（$h$ 侧还是 $\Delta W$ 侧）→ **E3/E4/E5 逐个切断单链**（输入可达性 / 优化器 / 初始化）。哪一个被切断后 $\mathrm{erank}$ 与 $\mathrm{loss}$ **同时**恢复，它就是当前的约束；对应地去改结构、改优化器、改初始化，或加显式秩约束。

---
## 😀 默认助手 (7)

你的四个观察其实处于**同一条因果链的不同位置**，并且其中只有一个是真正的“病”：

$$
\underbrace{\text{数据几何}}_{\text{观察1}}\ \longrightarrow\ \underbrace{\text{优化动力学}}_{\text{观察3}}\ \longrightarrow\ \underbrace{\text{更新谱}}_{\text{观察2}}\ \longrightarrow\ \underbrace{\text{性能}}_{\text{观察4}}
$$

观察 1、2、3 都**可能是良性的**：一个 erank=1 的 $\Delta W$ 完全可以是对的。因此，本回答的组织方式如下：**先把“一个通道”拆成三个互斥命题，再用能证伪它们的实验去区分**。

---

## 0. 三个核心原则（所有实验的设计依据）

1. **每个诊断都必须配一个“必要性测试”（counterfactual）**，否则你只能得到相关性，无法判断病因。
2. **只有 $\sigma(\Delta W)$ 是 gauge 不变的**。$\Delta W = BA$ 在 $A\to RA,\ B\to BR^{-1}$ 下不变，但“A 的某一行特别大”不是不变量——**观察 3 的朴素度量大半是 gauge 假象**。这一点被广泛忽略，也是最容易误判的地方。
3. **集中不等于有害**。必须把“集中度”和“零化/截断后的性能损失”配对测量。

---

## 1. “更新集中在一个通道”的三种互斥含义

这三种互斥含义分别对应不同的层级，而**修复方式完全不相交**：

| | 命题 | 精确定义 | 度量 | 若成立说明 | 对应修法 |
|---|---|---|---|---|---|
| **P-A** | 固定“超级通道” | 存在与输入无关的固定下标 $j^\star$，长期主导能量或梯度 | $\arg\max_j$ 的**熵**低、$p_{j^\star}$ 高 | 模型内在偏置通道（类似 attention sink / super weight） | 保护或隔离该通道；逐通道缩放；不做低秩 |
| **P-B** | token 相关的稀疏选通 | 每个 token 主导通道不同，但**每 token 只有少数几个**通道非零 | $\arg\max$ 熵**高**，但 $h$ 的 PR 低、Gini 高 | 门控饱和造成的准二值化，几何问题 | 逐通道白化 / 门控温度 / 对角适配器 |
| **P-C** | 权重的单一谱方向 | $\sigma_1(\Delta W)\gg\sigma_2\gg\dots$，与通道无关 | erank($\Delta W$)、top-1 能量占比 | 优化动力学停滞（winner-take-all）或几何上限 | Muon/SOAP、QR gauge 平衡、正交约束 |

> **判断口诀**：P-A/P-B 是**前向**属性（frozen 模型 + 数据，与 LoRA 无关，未训练即可测）；P-C 是**更新**属性。如果你不训练就能测到 P-A/P-B，那它们顶多是“背景条件”，不是 LoRA 的病根——真正的病根必须用“换了适配器就变好”来证明。

---

## 2. 七个判决性实验

### E1（10 分钟，不需训练）：通道身份 —— 区分 P-A 与 P-B

```python
@torch.no_grad()
def channel_identity(H):        # H: [N, d_ff] 某层 down_proj 的输入
    H = H.float()
    idx = H.abs().argmax(-1)                        # 每个 token 的主导通道
    cnt = torch.bincount(idx, minlength=H.shape[-1]).float()
    p = cnt / cnt.sum(); p = p[p > 0]
    H_ent = -(p * p.log()).sum().item()             # 越低 => P-A
    top1_share = p.max().item()
    e = (H**2).mean(0); q = e / e.sum()
    PR = 1.0 / (q**2).sum()                          # 能量有效秩
    return dict(ent=H_ent, top1_share=top1_share, PR=PR.item(),
                outlier_ratio=(e.max()/e.median()).item())
```
判读（$d_{ff}=3072$，$\ln 3072\approx 8.03$）：
- `ent < 2` 且 `top1_share > 0.3` → **P-A**（固定通道）
- `ent > 7` 但 `PR < 5` 且 `outlier_ratio > 100` → **P-B**（token 稀疏）

**配套的负对照：通道置换不变性。** 将 $W_g,W_u$ 的列与 $W_d$ 的行用同一个随机置换 $\pi$ 重排（这是冻结函数的精确对称性）。若 `ent/PR/outlier_ratio` 在置换后显著变化 → 你测的是索引伪影，别当病；若 `loss` 与 `erank` 不变而“主导通道下标”变了 → 确认是**统计几何**而非某个具体神经元，修法应指向白化/优化器，而不是“保护第 $j^\star$ 个通道”。

---

### E2（30 分钟）：门控饱和假说 —— 判断是否真的要动门控

对 SiTU（$\beta=4,\beta_L=25$）直接测饱和与局部导数：

```python
@torch.no_grad()
def saturation_stats(module, x, thr=8.0):
    g = x @ module.w1.weight.T                     # 或 gate_proj
    u = x @ module.w3.weight.T
    sat_g = (g.abs() > thr).float().mean().item()   # σ(g), tanh(g/4) 均已饱和
    sat_u = (u.abs() > 20.0).float().mean().item()  # up 支路 soft-cap 生效区
    alive = ((g.abs() <= thr)).any(0).float().mean().item()  # 至少偶尔处于过渡带的通道占比
    return sat_g, sat_u, alive
```
判读：
- `sat_g > 0.6` 且 `alive < 0.1` → 门控假说成立，**但注意你不能改 K3 的门控**（权重冻结），可做的只有：中间维度对角适配、门控输入增益/温度 $\beta_j$、或干脆别在 gate 上加 rank。所谓“修改门控结构”只在预训练或从头训小模型时才有意义。
- `sat_g < 0.05` → **门控假说被证伪**，别在结构上费力气，直接转到 E3/E6。

再做一次**温度扫描**（前向即可）：$g\to \lambda g$，测 \(\mathrm{PR}(h)\) 随 $\lambda$ 的变化。若 $\lambda=0.5$ 时 PR 从 4 跳到 40 → 饱和是主因，白化/降增益有效。

---

### E3（关键）：几何上限 vs 动力学塌缩

这两种病因需要**完全不同**的修复方式，用两条曲线的形状区分：

```python
# 每 k 步记录
curve = {
  "erank_dW":     rank_stats(BA.detach())[1],          # 累计更新的有效秩
  "erank_dWdot":  rank_stats(dW_step)[1],               # 单步增量 E[ΔW] 的谱
  "PR_input":     chan_stats(h)[0],                     # 输入几何上界
}
```
| 曲线形状 | 诊断 | 修法 |
|---|---|---|
| `erank_dW` **从第 1 步就 ≈ $\min(r,\mathrm{PR})$ 且全程平** | **几何上限**：$\mathrm{PR}(\Sigma_h)$ 限制了可用的秩，增加 rank 没有用 | 白化（S3）/ 对角适配（S4）/ 换模块（S5）。**优化器类措施无效** |
| `erank_dW` 前期高、**随 $t$ 单调下降** | **动力学塌缩**（winner-take-all：确定性方向 $\sim lr\,T$ vs 随机游走 $\sim lr\sqrt T$） | Muon/SOAP、QR gauge 平衡、正交约束、dropout、更长 warmup |
| `erank_dW ≈ r` 但性能仍差 | **根本不是 rank 问题** | 停止折腾，查数据/目标/层选择 |

**必测的辅助量**：$\mathrm{PR}(\Sigma_h)$ 与 $r$ 的相对大小。$\mathrm{PR}<r$ 时“加到 r=128 只用 3”是**数学必然**，不是 bug。

---

### E4：把塌缩归因到“输入各向异性”还是“优化器”

这是我最推荐的一个实验，因为它直接输出“该改哪个代码”：

$$
\text{align}=\frac{1}{r}\big\|V_{\Delta W}^{\top}E_k^{\Sigma_x}\big\|_F^2,\qquad \text{随机基线}=\frac{k}{d}
$$

其中 $V_{\Delta W}$ 是 $\Delta W$ 的右奇异向量（输入侧），$E_k^{\Sigma_x}$ 是输入二阶矩的 top-$k$ 特征向量。

```python
@torch.no_grad()
def alignment(dW, Sigma_x, k=8):
    _, _, Vh = torch.linalg.svd(dW.float(), full_matrices=False)   # Vh: [r, d]
    lam, E = torch.linalg.eigh(Sigma_x.float())
    Ek = E[:, -k:]
    return (Vh @ Ek).pow(2).sum().div(Vh.shape[0]).item(), k / dW.shape[1]
```
| align ≫ baseline | align ≈ baseline |
|---|---|
| 更新被**输入几何**锁死 → 白化/对角/换模块 | 更新塌缩**与输入无关** → gauge/优化器/初始化 |

---

### E5：必要性测试 —— 那个“浪费的秩”到底要不要紧

训练完后把 $\Delta W$ 截断再合并，看验证集掉多少：

```python
@torch.no_grad()
def truncated(dW, k):
    U, S, Vh = torch.linalg.svd(dW.float(), full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vh[:k]
```
- 截断到 $k=2$ 指标几乎不掉 → **观察4 与 rank collapse 无关**，你的瓶颈不是秩。这一条能省掉你几周时间，务必做。
- 截断到 $k=r/2$ 就明显掉 → 秩是有效瓶颈，继续往下走。

同理，对 baseline 做**反向测试**：既然你怀疑“实际只用了很少 rank”，那就主动把 rank 从 $r$ 降到 2、4、8 再训一遍。若 loss 完全相同，说明 rank 供给过剩，问题在别处。

---

### E6：参数化测试（最省预算的一步，且**几乎免费**）

**关键事实**：对 $d_{ff}=3072$，一个**满宽对角适配器**只有 3072 个参数，比 LoRA $r=1$（$3072+3584=6656$）**还便宜**。所以“逐通道自由度是否才是缺失的那一维”几乎零成本可测：

| 条件 | 参数量 | 若结果 |
|---|---|---|
| LoRA $r=1$ | 6.6K | 基线 |
| **diag only**（中间维逐通道缩放） | 3.1K | diag ≫ LoRA@r=1 → **低秩是错误参数化**（P-B 成立），直接上 S4 |
| LoRA $r=8$ | 53K | — |
| **LoRA $r=1$ + diag** | 9.7K | 增益几乎全来自 diag → 同结论 |

这组对比是“门控 FFN 中选通自由度是对角维、低秩天生表达不了”这一假设的**直接判决**，也是四条修法里唯一能靠 1 小时实验排掉一条的。

---

### E7：外部参照 —— 用 proxy 模型消除“LoRA 特有”与“任务内在”的混淆

**不要指望在 2.8T 的 K3 上跑几十组消融。** 一定要把机制发现下放到 0.5–3B 稠密模型：

- **A 组：full-FT 参照**。同任务 full-FT 的 $\mathrm{erank}(\Delta W)$ 是目标量级（文献里 700–1000，见 arXiv:2607.21612）。若 full-FT 的 PR 也只有 4 → 集中是任务/架构的内在属性，**不要治**。
- **B 组：门控结构对照**。把 proxy 模型的 SwiGLU 换成 SiTU（同 $\beta$）再换回来，其余不变。这是唯一能干净隔离“门控结构”这个因子的实验，在 K3 上做不了。
- **C 组：gauge 平衡后重测**。对已训好的 adapter 施加

  ```python
  @torch.no_grad()
  def balance_gauge(lora):                      # ΔW 严格不变，只改动力学
      Q, R = torch.linalg.qr(lora.lora_A.weight.T)   # A.T = Q R
      lora.lora_A.weight.copy_(Q.T)                  # A 行正交
      lora.lora_B.weight.copy_(lora.lora_B.weight @ R.T)   # B←B R^T，BA 不变
  ```
  由于 AdamW 不是 gauge 不变的，把 A 的行正交化会改变后续每方向的步长。若“平衡后继续训练 erank 上升” → 观察 3 确认是 gauge 病，S2 生效。同时这也验证了**观察 3 的朴素度量不可信**：只看 A 的行范数，在正交化后会立刻“消失”，但 $\Delta W$ 一点没变。

---

## 3. 模块定位消融：gate / up / down / whole

### 3.1 设计矩阵（proxy 模型，参数匹配）

固定可训练预算 $P$（因为 gate/up 是 $[d_{ff},d]$，down 是 $[d,d_{ff}]$，尺寸相近，用同一 $r$ 已近似匹配）：

|  | LoRA-$r$ | LoRA-$r$+QR平衡 | LoRA-$r$+diag | diag-only | **full-rank（oracle）** |
|---|---|---|---|---|---|
| gate only | ✓ | ✓ | ✓ | ✓ | ✓ |
| up only | ✓ | ✓ | ✓ | ✓ | ✓ |
| down only | ✓ | ✓ | ✓ | ✓ | ✓ |
| gate+up | ✓ | | | | ✓ |
| **all** | ✓ | ✓ | ✓ | | ✓ |

**必跑的行**（约 10 组）：`down-only` 的 5 个变体 + `gate-only` 的 LoRA/full-rank + `all` 的 LoRA/full-rank。

解读逻辑：
- `down-only full-rank` ≈ `all` → down 的瓶颈是**秩**，位置没错 → 走 S3/S4。
- `down-only full-rank` ≪ `all` → down 不是主战场 → 把容量挪到 gate/up 或共享通路（对 K3 是 `shared_experts` 与 LatentMoE 的 `routed_expert_down/up_proj`）。
- `gate-only full-rank` 明显好于 `gate-only LoRA` 且好于 `down-only full-rank` → **选通自由度**是主要瓶颈 → S4。
- 只有加 `diag` 才有增益、`+QR平衡` 无增益 → P-B，不是 P-C。

### 3.2 归因指标：必须测“重要性”，不能只测“集中度”

对每个模块 $m$，零化其 $\Delta W_m$ 后测验证集下降 $\Delta_m$：

```python
# 对已训练 adapter，逐个模块把 lora_B 置零（等价 ΔW=0），前向评估
```
然后画一张二维散点：**横轴 erank($\Delta W_m$)/r，纵轴 $\Delta_m$**。
- 右下角（erank 低但重要性高）= 真正的病因模块，优先投预算。
- 右上/左上 = 更新很“满”但不重要，别浪费 rank。

### 3.3 必须固定的协变量

否则归因无效：
- **MoE 路由**：K3 有 896 experts，每 token 16 个。固定 routing（用同一份 routing 记录做对比，或在 proxy 模型固定 router），且**按 expert 频率分层统计**——低频 expert 的统计量本身就没有意义（样本太少，$\mathrm{PR}$ 估计有巨大方差）。**最少要检查：被 adapted 的 expert 是否接收 ≥ 数百 token**。这是 K3 上最可能的伪信号源。
- **注意**：HF 官方 `KimiSparseMoeBlock.moe_infer` 带 `@torch.no_grad()` 且训练模式直接 `raise NotImplementedError`。若你在官方实现上跑 routed expert 的 LoRA，梯度恒为 0，你会测到“完美 rank collapse”，但那只是 bug。**先确认梯度非零再谈任何结论**。

---

## 4. 决策表：症状组合 → 唯一解

| E1 | E3（erank 轨迹） | E4（对齐度） | E5（截断） | E6（diag vs LoRA） | 诊断 | 唯一解 |
|---|---|---|---|---|---|---|
| P-A 固定通道 | 全程低 | 高 | 掉 | diag 无增益 | 内在偏置通道 | 隔离/缩放该通道，别加 rank；**不动优化器** |
| P-B 稀疏 | 平且 ≈ PR | 高 | 掉 | **diag ≫ LoRA** | 几何上限 + 参数化错误 | 逐通道白化 + 对角/门控温度适配（S3+S4），**不要加 rank** |
| — | **随时间下降** | ≈ 随机 | 掉 | — | 动力学塌缩 | Muon/SOAP + QR gauge + 正交约束（S1+S2），**不要改几何** |
| — | ≈ r 但性能差 | — | **不掉** | — | **不是 rank 问题** | 停手，查数据/目标/layer 选择 |
| — | 全程低 | 高 | 掉 | diag 也救不回 | 模块位置错 | 把预算挪到 gate/up / 共享通路（S5） |

---

## 5. 执行顺序与预算

<details>
<summary>展开：三阶段执行计划（含 K3 上能做/不能做的边界）</summary>

**Stage 0（几小时，纯前向，零训练成本）** — 在 K3 上做
- E1 通道身份、E2 饱和率与温度扫描、$\mathrm{PR}(\Sigma_h)$ 逐层逐 expert。
- 产出：门控假说成立/证伪；P-A 还是 P-B；几何上界 $\mathrm{PR}$ 的数值。
- **这一步就能砍掉一半的候选修法**，而且是唯一能在 2.8T 模型上廉价完成的诊断。

**Stage 1（1–3 天，proxy 0.5–3B）** — 机制发现
- full-FT 参照（A 组）、SwiGLU↔SiTU 对照（B 组）、E3 的 erank 轨迹、E5 截断、E6 的 diag 对照、第 3 节的模块矩阵。
- 产出：唯一诊断 + 唯一修法。

**Stage 2（K3，2–3 组）**
- 只跑 `LoRA-r baseline` / `baseline+选定修法` / `对照`。
- 指标：erank-utilization 曲线 + 零化重要性 + 验证集。

**K3 上做不到、别浪费时间的事**：
1. 改门控结构（权重冻结，SiTU 的 $\beta$ 已在 checkpoint 里）；只能通过适配器侧重参数化等效实现（逐通道增益/温度）。
2. 全模型 full-FT 参照（除极小学习率 LoRA 化替身）。
3. 逐 expert 的 SVD 白化（896×92 个 $3072^2$ 矩阵，≈3.5 TB）；只能做对角，或只做高频 expert。
4. 在官方 `transformers` 实现上训 routed expert（`moe_infer` 被 `no_grad` 包裹）。

</details>

---

## 一句话回答

“更新集中在一个通道”最可能是 **P-B（token 相关选通稀疏）+ 几何上限**，而不是“某个神经元变成超级通道”。**判据顺序是：先做 E1/E2（不训练就能定 P-A vs P-B、噪声门控假说），再做 E3 区分“几何上限”与“动力学塌缩”，最后用 E5 的截断测试确认那个被浪费的秩到底要不要紧。** 因为 E6 的 diag 对照几乎免费（比 $r=1$ 还便宜），建议和 E3 一起跑——它是对“低秩是不是错误参数化”这一根因假设最直接的判决。

---
## 🧑‍💻 User

这个问题本质上不是 PEFT 的 `rank` 设置失效，而是 SiTU 门控改变了 FFN 的局部梯度几何，使 LoRA 的有效更新集中在少数方向，导致名义 rank 与实际有效 rank 严重不匹配。AdamW 可能进一步放大这种现象，但需要先区分：究竟是梯度本身低秩、门控造成通道不均衡，还是 AdamW 的逐元素预条件改变了更新方向。

我认为针对 SiTU：考虑门控感知的 LoRA，这是我认为最值得针对我们模型研究的部分。假设 SiTU 使 FFN 的某些通道长期获得更大的梯度，普通 LoRA 会通过共享的低秩因子学习这些通道的更新。它并不知道你们希望各个通道或子空间都能获得有效的适配容量。如何重新设计更适合SiTU激活性质的lora算法？（需要修改peft库）

---
## 😀 默认助手 (7)

你的判断我基本同意，但我想把它再推进一步，因为这一步直接决定了算法该怎么设计：

> **纯加性的低秩 $\Delta W$ 无法改变梯度几何**。它对 loss 的作用是线性的，$\nabla_{\Delta W}\mathcal{L}=\delta h^\top$ 由冻结模型决定，无论 $r$ 取多少，学到的都是同一个 outer-product 张成的空间。

所以"gate-aware LoRA"必须**引入非线性或乘性自由度**，否则它只是在同一个低秩族里换基。这是你这个方向的立论基础，下面所有设计都从这个约束倒推。

---

## 0. 先把你说的三分法固定成三个可测标量（一次 30 分钟）

| 你的假设 | 一次性判据 | 若成立 |
|---|---|---|
| H-grad：梯度本身就低秩 | $\mathrm{erank}\big(\mathbb{E}[\delta h^\top]\big)$ 与 $\mathrm{erank}(\Delta W^\star)$（oracle 参照） | **加性 LoRA 无解**，必须走乘性/非线性适配器 ← 正是你的方向 |
| H-chan：门控造成通道不均衡 | $\mathrm{PR}(\mathbb{E}[h_j^2])$、`trans_frac`$=\frac{1}{d_{ff}}\sum_j\mathbb{1}[\phi'(g_j)>\epsilon]$、$\hat\sigma_j=\sqrt{\mathbb{E}[h_j^2]}$ 的 Gini | 度量折叠 + 三态掩码 |
| H-opt：AdamW 预条件放大 | 同 seed、同 init，只把 AdamW 换成 Muon，看 $\mathrm{erank}(\Delta W)$ | 优化器/正交约束 |

这三个**不互斥**，设计上应该做成三个正交的组件，各自有开关（见 §4 消融矩阵）。这样无论诊断结论是哪一条，你都能单独验证。

---

## 1. 从 SiTU 推导"正确的 LoRA 应该长什么样"

### 1.1 SiTU 的真实效应：不是"梯度少"，而是"梯度的度量极度不均"

$$
\phi(g)=\beta\tanh(g/\beta)\sigma(g),\qquad \phi'(g)=\sigma(g)\Big[\mathrm{sech}^2(g/\beta)+\beta\tanh(g/\beta)\big(1-\sigma(g)\big)\Big]
$$
$$
h_j=\phi(g_j)\,\psi(u_j),\quad \psi(u)=\beta_L\tanh(u/\beta_L),\ \psi'\approx1\ (\beta_L=25)
$$
$$
\frac{\partial h_j}{\partial g_j}=\phi'(g_j)\psi(u_j),\qquad \frac{\partial h_j}{\partial u_j}=\phi(g_j)\psi'(u_j)
$$

$\phi'$ 的形状是 **宽平台 + 陡峭肩部 + 双侧归零**（$\beta=4$ 时过渡带约 $|g|\lesssim7$，$|g|=12$ 处 $\phi'\approx0.01$，$g=-8$ 处 $\phi'\approx2\times10^{-5}$）。同时 $\phi$ 本身被钳在 $[0,4]$，$h_j\in[-100,100]$。

于是每个通道落在三个状态之一：

| 状态 | 判据 | $\phi'$ | $h_j$ 量级 | 适配它需要什么 |
|---|---|---|---|---|
| **off** | $g\ll0$ | ≈0 | ~0 | 几乎不需要容量 |
| **transition（可塑）** | $|g|\lesssim7$ | 大 | 中 | **需要门控/选择自由度** |
| **sat-on（饱和粘滞）** | $g\gg 7$ | ≈0 | 大（可达 $4\psi$） | 只需一个**标量增益**，写不进 gate |

关键结论：**"可塑性"（$\phi'$）与"能量"（$h_j^2$）在通道上是解耦的甚至反相关的**。而你若用 $\mathbb{E}[hh^\top]$ 做 LoRA 的输入度量（AdamW 事实上就是这么做的），你的秩预算会全花在**能量大但不可塑**的 sat-on 通道上——这正是"更新集中在一两个通道、$r$ 用不满"的精确来源。

### 1.2 异质性落在瓶颈的两侧（这决定了预处理该加在哪里）

对 K-FAC/Kronecker 结构求局部 GGN：

- **down_proj**（$\Delta W_d\in\mathbb{R}^{d\times d_{ff}}$）：梯度 $G_d=\delta h^\top$，其 GGN $\approx \mathbb{E}[\delta\delta^\top]\otimes\mathbb{E}[hh^\top]$。**异质性在输入侧**（$d_{ff}$ = A 的列）。
- **gate_proj / up_proj**（$\Delta W_{g,u}\in\mathbb{R}^{d_{ff}\times d}$）：链式上来一个 $\mathrm{diag}(\phi'\psi)$ 或 $\mathrm{diag}(\phi\psi')$，于是有效输出度量是
$$
D_{\text{out}}=\mathrm{diag}\big(\mathbb{E}[\phi'(g_j)^2\psi(u_j)^2]\big)\ \text{或}\ \mathrm{diag}\big(\mathbb{E}[\phi(g_j)^2\psi'(u_j)^2]\big)
$$
  **异质性在输出侧**（$d_{ff}$ = B 的行）。

所以三个模块需要**同一族但不同侧的**对角白化。这是本设计的第一个组件。

### 1.3 为什么低秩在门控 FFN 上本质不够（维数论证）

$\Delta g=(BA)x=B(Ax)$：$x$ 相关的变化只活在 $r$ 维系数里，而**所有 token 共享同一个固定的 $r$ 维列空间 $\mathrm{col}(B)$**。要让神经元 $j$ 和 $j'$ **独立**选通，必须 $e_j,e_{j'}\in\mathrm{col}(B)$——**每个可独立选通的神经元吃掉 1 个 rank**。

| 方案（$m{=}3584,n{=}3072,r{=}32$，K3 每 expert） | 参数量 | 可独立选通通道数 |
|---|---|---|
| vanilla LoRA(down) | 213k | **32** |
| LoRA(all 3) | 0.6M | 32 |
| **+ 通道增益（gate+up）** | **+6k**（+1%） | **3072 ×2** |
| + 对角修正 top-64 | +64 | 64（全宽、无 rank 限制） |

$r=32$ 只覆盖 $d_{ff}$ 的 **1%**。这不是调参问题，是参数化族选错了：**"神经元选通"是 $d_{ff}$ 维的对角自由度，低秩天生表达不了**。所以正确做法是**把两种自由度分开付钱**：

$$
\boxed{\Delta W=\underbrace{B\,\Lambda\,A}_{\text{子空间混合：}r\text{ 维}}+\underbrace{\mathrm{diag}(\rho)\,W\ \text{或}\ W\,\mathrm{diag}(\sigma)}_{\text{通道选通：}d_{ff}\text{ 维}}}
$$

好消息：**两项都严格可合并、零推理开销**。左乘 $\mathrm{diag}(\rho)$ 就是按行缩放 $W$（gate 增益 $g_j(1+\rho_j)$），右乘就是按列缩放（down 的输入通道）。

---

## 2. 算法：SiTU-LoRA（三个正交组件）

### 主公式

对 gate/up/down 统一写成

$$
\Delta W=B\,\mathrm{diag}(\lambda_r)\,A\,\mathrm{diag}(s_{\text{in}})\;+\;\mathrm{diag}(s_{\text{out}})\,W\,\mathrm{diag}(\sigma_{\text{in}})
$$

- $s_{\text{in}}\in\mathbb{R}^{d_{ff}}$：**输入侧度量折叠**（down_proj 用）+ 三态掩码
- $s_{\text{out}}\in\mathbb{R}^{d_{ff}}$：**输出侧度量折叠**（gate/up 用）
- $\lambda_r\in\mathbb{R}^r$：显式奇异值（DoRA/AdaLoRA 式），让秩维度可解释
- $\sigma_{\text{in}}$：逐通道增益（只在 $d_{ff}$ 索引侧有意义）

### 组件 A：Metric-Folded LoRA（度量折叠）

由 §1.2：
$$
s^{(d)}_{\text{in},j}=\mathrm{clip}\Big(\hat\sigma_j^{-\gamma},\ 1,\ \kappa\Big),\quad \hat\sigma_j=\sqrt{\mathbb{E}[h_j^2]}
$$
$$
s^{(g)}_{\text{out},j}=\mathrm{clip}\Big(\big(\mathbb{E}[\phi'^2\psi^2]\big)^{-\gamma/2},\ 1,\ \kappa\Big)\ ,\qquad s^{(u)}_{\text{out},j}\ \text{同理用}\ \mathbb{E}[\phi^2\psi'^2]
$$

三点说明：
- **只做单侧阻尼（clip 下界 1）**，不要全白化。放大低能量通道的噪声收益为负，而"压低 outlier"正是你诊断到的病灶。$\gamma=0$ 退化为 vanilla，$\gamma=1$ 全白化，$\kappa\approx4$ 控制动态范围。
- 因为 $B$ 初始化为 0，**加这个不改动 base model 的任何一个输出**，零风险。
- 用**可塑性能量** $\mathbb{E}[\phi'^2\psi^2]$ 而非 $\mathbb{E}[h^2]$ 作为 gate 侧度量——这是"gate-aware"的实质：你按可塑性而非幅值分配容量。

### 组件 B：Regime-Masked Capacity（三态掩码，秩只在可塑子空间取）

在每个模块上维护可塑掩码 $M\in\{0,1\}^{d_{ff}}$：

$$
M_j=\mathbb{1}\big[\phi'(g_j)>\epsilon\big]\ \vee\ \mathbb{1}\big[j\in\mathrm{top}\text{-}k\ \text{by}\ |h_j|\big]
$$
然后把 $s_{\text{in}}\leftarrow s_{\text{in}}\odot M$（down）/ $s_{\text{out}}\leftarrow s_{\text{out}}\odot M$（gate/up）。

效果：**$r$ 个方向再也不浪费在 sat-on 的"死通道"上**。sat-on 通道交给对角增益（组件 C）用 1 个参数处理。

预测（可证伪）：同质量下 $r$ 可以从 32 降到 16，参数量减半。

### 组件 C：Channel-Gain Bypass（通道级全宽旁路）

$$
h_j\to\phi\big(g_j(1+\rho_j)\big)\,\psi\big(u_j(1+\upsilon_j)\big),\qquad g_j\to g_j(1+\rho_j)\ \equiv\ W_{g}[j,:]\leftarrow(1+\rho_j)W_g[j,:]
$$
对 down 再加稀疏对角：$W_d[:,j]\leftarrow W_d[:,j](1+\sigma_j)$，$j$ 取 top-$k$ 主导通道（$k\approx64$）。

- $\rho$ 是**解饱和旋钮**：把 sat-on 通道拉回过渡带，恢复 $\partial h/\partial g$ 的信息通路——这是纯加性 LoRA 在原理上做不到的事。
- 也允许 $\rho<0$ 抑制病态通道。
- 全部可合并（行/列缩放）。

### 组件 D：白化坐标下的显式秩正则（可选）

在折叠后的坐标里做才有意义（否则你在惩罚"正确的集中"）：

```python
def erank_reg(A, B, s_in, s_out, eps=1e-8):
    A_eff = A * (s_in * 1.0)[None, :]          # [r, n]
    B_eff = B * s_out[:, None]                # [m, r]
    M = A_eff @ A_eff.T @ (B_eff.T @ B_eff)   # r×r，O(r^3)，r≤128 可忽略
    w = torch.linalg.eigvalsh(M.float()).clamp_min(eps)
    p = w / w.sum()
    return -(p * p.log()).sum()               # loss += -λ * 该项
```

配合 **Muon/MuonClip 用于 A/B**（把更新谱拉平）与 `use_rslora=True`。

### 参数预算（每 expert，$r{=}32\to16$）

| 项 | 参数量 | 可合并 |
|---|---|---|
| $B,\Lambda,A$（down+gate+up, r=16） | ~0.3M | ✅ |
| $s_{\text{in}},s_{\text{out}}$（固定 buffer） | 0 | ✅ |
| $\rho,\upsilon$（gate/up 增益） | 6k | ✅ |
| $\sigma$（top-64 对角） | 64 | ✅ |
| 统计 buffer（每 expert 3×3072 fp16） | 18KB | — |

增益项只占 1%，却提供了 3072 倍的选通分辨率。

---

## 3. 在 peft 里怎么改（可直接落地的骨架）

<details>
<summary><b>目录结构与关键 API 决策（先看这段，省一天）</b></summary>

```
peft_situ/
├── config.py     # SiTULoraConfig(LoraConfig)
├── stats.py      # SiTUStatsCollector：前向 hook 采集 φ/ψ/φ'
├── layer.py      # SiTULoraLinear（forward / merge / unmerge）
├── model.py      # SiTULoraModel(LoraModel)：覆写 _create_new_module
└── calibrate.py  # 一次性标定，产出 per-module buffer
```

- `LoraConfig` 走 `**kwargs` 存额外字段，`__post_init__` 里**必须先 `super().__post_init__()`**（它会校验 `target_modules`/`r`）。
- **不要用 `get_peft_model`**：`PeftType` 仍是 `LORA`，会分发到原生 `LoraModel`。直接 `SiTULoraModel(model, config, "default")`，它继承全部 `save_pretrained / merge_and_unload / get_peft_model_state_dict`。
- 覆写点只有一个：`LoraModel._create_new_module(self, lora_config, adapter_name, target, **kwargs)`，其中 `kwargs["current_key"]` 是模块全名，**用它查 per-module 的统计 buffer**（这是最近几个版本才有的，务必断言存在）。
- 合并：所有组件都是对角/因子级的，`merge()` 里有闭式解，**无需 `unmerge` 近似**。
</details>

```python
# peft_situ/config.py
from dataclasses import dataclass
from peft import LoraConfig

@dataclass
class SiTULoraConfig(LoraConfig):
    situ_stats_path: str | None = None
    metric_gamma: float = 0.5      # 0 = vanilla, 1 = full whitening
    metric_clip: float = 4.0       # 单侧阻尼动态范围
    regime_mask: bool = True       # 秩只在可塑通道
    gate_gain: bool = True
    up_gain: bool = False
    diag_channels: int = 64
    erank_reg: float = 0.0
    reg_every: int = 50

    def __post_init__(self):
        super().__post_init__()            # 必须
        self._custom_fields = ["situ_stats_path", "metric_gamma", ...]
```

```python
# peft_situ/layer.py（核心，节选）
import torch, torch.nn.functional as F
from torch import nn
from peft.tuners.lora.layer import LoraLayer

class SiTULoraLinear(nn.Module, LoraLayer):
    def __init__(self, base_layer: nn.Linear, config, adapter_name, stats: dict):
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config.r, config.lora_alpha,
                          config.lora_dropout, config.use_rslora, config.use_dora)

        ff_dim = stats["ff_dim"]                       # 神经元索引维 = d_ff
        side = stats["side"]                           # "in"(down) / "out"(gate,up)
        s = stats["metric"]                            # [d_ff]，已含 gamma/clip
        m = stats["plastic_mask"].float() if config.regime_mask else torch.ones_like(s)
        s = s * m
        # 放在神经元索引所在的那一侧
        self.register_buffer("s_neuron", s, persistent=True)
        self.side = side

        # 通道增益（乘性、可合并）
        if config.gate_gain or config.up_gain:
            self.rho = nn.Parameter(torch.zeros(base_layer.out_features))
        # 稀疏对角修正
        self.register_buffer("diag_idx", stats["top_idx"][:config.diag_channels], persistent=True)
        self.diag = nn.Parameter(torch.zeros(self.diag_idx.numel()))

    def _lora_out(self, x, A, B, lam, scaling):
        # ΔW = B diag(λ) A diag(s)，全部保持因子形态，O(r) 内存
        xs = x * self.s_neuron if self.side == "in" else x
        y = F.linear(xs, A.weight)                     # [.., r]
        y = y * lam                                   # 显式奇异值
        y = F.linear(y, B.weight)                      # [.., out]
        if self.side == "out":
            y = y * self.s_neuron
        return y * scaling

    def forward(self, x, *args, **kwargs):
        out = self.base_layer(x, *args, **kwargs)
        for name in self.active_adapters:
            if name not in self.lora_A:
                continue
            A, B = self.lora_A[name], self.lora_B[name]
            xd = self.lora_dropout[name](x)
            out = out + self._lora_out(xd, A.weight, B.weight,
                                       self.lora_embedding[name], self.scaling[name])
            if hasattr(self, "rho"):                   # 乘性增益
                out = out * (1.0 + self.rho)
            if self.diag.numel():
                out = out.index_add(-1, self.diag_idx,
                                    out.index_select(-1, self.diag_idx) * self.diag)
        return out

    def get_delta_weight(self, adapter):
        A, B = self.lora_A[adapter].weight, self.lora_B[adapter].weight
        lam = self.lora_embedding[adapter]
        if self.side == "in":
            A = A * (self.s_neuron * lam)[None, :]
            dW = B @ A
        else:
            B = B * (self.s_neuron * lam)[:, None]
            dW = B @ A
        return dW * self.scaling[adapter]

    def merge(self, safe_merge=True, adapter_names=None):
        # dW 加进 base；rho 折成行缩放；diag 折成列缩放
        ...
```

<details>
<summary><b>标定脚本（前向一次，产出所有 buffer）</b></summary>

```python
# peft_situ/stats.py
import torch

class SiTUStatsCollector:
    """挂在 MLP 模块（KimiMLP / KimiBlockSparseMLP）的 pre-activation 上"""
    BETA, LIN_BETA = 4.0, 25.0

    def __init__(self):
        self.acc = {}

    @torch.no_grad()
    def hook(self, name, g, u, h):        # 三段 pre/post activation，[N, d_ff]
        g, u, h = g.float(), u.float(), h.float()
        phi = self.BETA * torch.tanh(g / self.BETA) * torch.sigmoid(g)
        dphi = torch.sigmoid(g) * ((1 - torch.tanh(g / self.BETA) ** 2) / self.BETA) \
             + self.BETA * torch.tanh(g / self.BETA) * torch.sigmoid(g) * (1 - torch.sigmoid(g))
        a = self.acc.setdefault(name, dict(h2=0, a2=0, b2=0, n=0, gabs=[]))
        a["h2"] += h.pow(2).sum(0)
        a["a2"] += (dphi.pow(2) * u.pow(2)).sum(0)     # (∂h/∂g)^2
        a["b2"] += (phi.pow(2)).sum(0)                 # (∂h/∂u)^2 ≈ φ²·1
        a["n"]  += h.shape[0]
        a["gabs"].append(g.abs().amax(0))

    def finalize(self, gamma=0.5, clip=4.0, eps=0.02):
        out = {}
        for name, a in self.acc.items():
            n = max(a["n"], 1)
            h2, a2, b2 = a["h2"] / n, a["a2"] / n, a["b2"] / n
            sd = h2.sqrt()
            s_in  = (sd.clamp_min(1e-6) ** -gamma).clamp(1.0, clip)   # down 输入侧
            s_out = (a2.clamp_min(1e-12) ** (-gamma / 2)).clamp(1.0, clip)  # gate 输出侧
            trans = (a2 / a2.amax() > eps)
            top = torch.topk(h2, k=min(256, h2.numel())).indices
            plastic = (trans | torch.isin(torch.arange(h2.numel()),
                                          top.to(h2.device))).float()
            out[name] = dict(metric=s_in, metric_gate=s_out,
                             plastic_mask=plastic, top_idx=top,
                             trans_frac=float(trans.float().mean()),
                             pr_h=float(1 / ((h2 / h2.sum()) ** 2).sum()))
        return out
```
输出直接喂 `SiTULoraConfig`。
</details>

---

## 4. 消融矩阵：证明每一件都必要

固定参数量（对齐 $r$），只加一个组件：

| # | 配置 | 主要验证 | 目标签名 |
|---|---|---|---|
| A0 | vanilla LoRA | 基线 | $\mathrm{erank}/r<0.2$ |
| A1 | +$s_{\text{in}}$ 折叠(γ=0.5) | H-chan（度量） | erank↑，PR(h) 不变 |
| A2 | +三态掩码 | H-chan（分配） | erank↑ 且 $r{=}16$ 追平 A0 的 $r{=}32$ |
| A3 | +gate 增益 | **H-grad（乘性破局）** | 梯度 $\mathrm{erank}$ 本身↑，$s$ 的 PR 下降 |
| A4 | +Muon(λ_r) | H-opt | 前期不塌陷 |
| A5 | +erank 正则 | 显式约束是否还有增量 | — |
| A6 | A2+A3（无 A1） | 检查是否 A1 才是关键 | — |

必须报告的三个数（每层、按路由频次加权聚合）：

1. $\mathrm{erank}(\Delta W_\text{down})/\mathrm{erank}(\Delta W^\star)$ —— **秩利用率**
2. `trans_frac` 与 $\hat\sigma$ 的 Gini 变化 —— **几何有没有被改变**
3. rank-utilization 曲线（$r\in\{8,16,32,64\}$ vs erank/r）—— **A3 应使曲线变陡**（这是"乘性破局"的唯一硬证据）

判据：若 A3 只降低 loss 但不改变 $\mathrm{erank}$，说明它是**新增容量**而非**治好了 collapse**——那也对，但结论不同，别混。若 A3 让梯度 $\mathrm{erank}$ 上升，才证明 §1.3 的维数论证在你的模型上成立。

---

## 5. 风险与失效模式

<details>
<summary>展开</summary>

1. **静态标定 → 分布漂移**：$s$/$M$ 在标定集上算出，训练 5k 步后 gate 分布会变。**每 N 步用当前模型重标定一次**（纯前向，~1000 条序列，成本可忽略）。这是必须的，否则掩码会锁死本该被解锁的通道。
2. **掩码过刚**：用软掩码 + 熵正则（或每 epoch 重算并按 EMA 平滑），避免"因初始不可塑而永久弃用"。
3. **增益爆炸**：$\rho$ 加 clamp $[-0.5, 2]$ + weight decay；$\rho$ 是把 $g$ 放大，放大过头会加速其它通道饱和，形成正反馈。
4. **MXFP4 base**：不要往量化权重里 merge；merge 到 bf16 副本。
5. **MoE 统计噪声**：K3 是 896 experts/top-16。按路由频次加权，过滤 token 数 $<N$ 的 expert；**shared_experts（全 token）先做，routed experts 只在 top-64 高频上启用**。
6. **`moe_infer` 的 `@torch.no_grad()`**：HF 官方实现在此，routed expert 上任何 LoRA 都拿不到梯度——先确认你的训练框架patch 了这里。
7. **融合 `gate_up_proj`**：K3 走 `cat([w1(x), w3(x)])`，增益/度量必须**按前后半段切片**，两半的 $\beta$/统计完全不同。

</details>

---

## 6. 与已有工作的定位

<details>
<summary>展开</summary>

- **NA-LoRA**（arXiv:2606.31717）是最接近的：它把 gated FFN 的 "selection misalignment" 形式化，用导数时序重要性 mask 给 gate 侧 LoRA 分配容量。你的差分在于：① 显式的**度量折叠**（Kronecker 输入/输出侧白化，它们没有）；② **乘性通道增益**（它仍是加性低秩，受 §1.3 的维数上界约束）；③ 三态掩码 + 可合并性。
- **DoRA**（2402.09353）是逐列幅度分解的特例；组件 C 可看作"gate 侧的 DoRA，且不做归一化、作用在 neuron 索引上"。
- **ISO-LoRA**（2609.12123）/ **Stiefel-LoRA**（2508.17901）解决 H-opt；与本设计正交，应叠加使用。
- **K-FAC/SOAP** 是"自适应版的度量折叠"，代价是估 $d_{ff}\times d_{ff}$ 协方差。本设计用**标定得到的固定对角 + 三态掩码**替代，代价接近 0，且更可解释（你能直接画出"哪个通道被降权了"）。如果资源允许，**组件 A 用真 SOAP 预条件器是它的上界对照**，值得作为 A1 的 oracle。
- 理论上最接近的思想是 **Kronecker-factored natural gradient + 稀疏性（三态）先验**；把 §2 主公式读成"在 SiTU 诱导的对角度量下的自然梯度 + 通道分组的 $\ell_0$ 分块分配"是准确的。

</details>

---

**一句话总结**：SiTU 把 FFN 的神经元索引空间变成了一个**能量与可塑性解耦、且度量极度不均**的空间；加性低秩 $\Delta W$ 在这个空间里只有 $r$ 个独立选通自由度（$r/d_{ff}\approx1\%$），所以它必然塌缩。解法是**分侧度量折叠 + 三态掩码 + 乘性通道增益**三件套：前两个让 rank 花在可塑通道上，第三个把"神经元选通"这个 $d_{ff}$ 维自由度用 1% 的参数买回来，并且全部可合并、零推理开销。**先跑 §4 的 A3 vs A0，看梯度 $\mathrm{erank}$ 是否被乘性项改变——这是判断这条路是否成立的唯一硬指标。**
