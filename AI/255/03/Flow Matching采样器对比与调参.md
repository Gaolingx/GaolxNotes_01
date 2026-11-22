# Flow Matching采样器对比与调参

## 🧑‍💻 User

flow matching训练的diffusion模型采样器用ER-SDE就是最好的吗 Euler我基本都能看到瑕疵 思考 EulerA倒是没怎么试

---
## 😀 默认助手 (7)

<details><summary>结论先说（点开看决策表）</summary>

**「最好」不存在，而且对 flow matching 来说，ER-SDE 的收益往往被误判了。**

ER-SDE 不是靠“随机性”赢的，它是**高阶（指数积分器 / 多步）求解器**，而且只花 1 NFE/step。它的论文原话就是把 ODE 的速度和 SDE 的质量合起来（arXiv:2309.06169，*Elucidating the solution space of extended reverse-time SDE for diffusion models*，20 NFE 拿到 ImageNet 128 的 8.33 FID）。

但对 rectified flow / flow matching 模型，**你先该修的是“误差阶 + shift”，而不是换采样器**。Euler 的瑕疵是**确定性偏差**（换个 seed 还在同一位置），换 SDE 只是把它“抖动”成纹理，而不是消掉它。

| 你的诉求 | 该用 |
|---|---|
| 20–30 步 + 可复现 + 保结构 | `heun` / `dpm_2`（2 阶），或 `deis` / `res_multistep`（多步，1 NFE/step） |
| 20–30 步 + 要细节和多样性 | `er_sde`（`max_stage=3`，`eta≈0.5–1`），或更可控的 `sa_solver`（只在中间噪声段开 SDE） |
| ≤10 步 / 蒸馏模型（Turbo、LCM、Lightning） | 老实 `euler` + 正确 shift，**别上 SDE** |
| 视频 / 续写 / img2img 一致性 | 确定性 ODE（SDE 会闪、会漂） |
| CFG > 4 | 确定性 ODE；SDE 会加剧“烤焦”感 |

</details>

## 1. Euler 的瑕疵到底来自哪里

Euler 是 1 阶：局部误差 $O(h^2)$，全局误差 $O(h)$，而且**方向固定、不随机**。

$$x_{k+1}=x_k+h_k\,v_\theta(x_k,t_k),\qquad \text{err}\sim \frac{1}{2}h^2\,\dot v_\theta$$

关键推论：

- 误差 $\propto \left\| \dot v_\theta \right\|$，即**轨迹曲率**。flow matching 的**条件**速度场是直线的，但**边缘**速度场 $v_\theta(x,t)=\mathbb{E}[\dot x_t\mid x_t]$ 一般不是——条件期望把直线“平均弯了”。所以只在 $t\to 0$（细节精修）和 $t\to 1$（纯噪声区）曲率最大。
- 全局误差是 $C/N$：步数翻倍只降一半。这就是为什么 Euler 从 20 → 30 步像“没救”。
- **误差是 system bias**：表现为过平滑 / 塑料感 / 色偏 / 细节丢失 / 高对比边缘的振铃，且**跨 seed 稳定复现**。

> 快速判别：瑕疵换 seed 不变 → 确定性截断误差（加步数、升阶、调 shift）。换 seed 就变、呈颗粒或斑块 → 噪声/参数问题。

顺便，很多“Euler 瑕疵”根本不怪采样器：

- **sigma shift 不对**。1024 分辨率下 Flux 系要 shift≈3，SD3.x 要用 shift 3；shift 错 → 固定步长把大量 NFE 浪费在无用的噪声区。
- **CFG 太高**。CFG 后的场 $v=v_u+s(v_c-v_u)$ 已经不是任何 transport 的速度场了（不是某个 $\nabla\log p$ 的梯度），ODE 都只能算“勉强有意义”，SDE 的解释更是形式上的。
- **VAE decode 的 bf16 / tiling** 被误当成采样瑕疵。

## 2. ER-SDE 为什么“看起来最好”

看它的实现就明白了（ComfyUI `sample_er_sde`，论文参考实现 VP ER-SDE-Solver-3）：

- **Stage 1** 用 $\lambda_t=\sigma_t/\alpha_t$ 做**指数积分器的精确线性解**：$x=r_\alpha r\,x+\alpha_t(1-r)\,\hat x_0$。
- **Stage 2/3** 用 $\dfrac{d\hat x_0}{d\lambda}$ 的二阶差分做 Taylor 修正 → **有效 2 阶以上**，但**不额外花 NFE**（复用上一帧的 `denoised`）。
- 然后才叠加 churn：$\alpha_t\sqrt{\lambda_t^2-\lambda_s^2r^2}$ 量级的噪声。

所以 **ER-SDE > Euler 的主体收益是“升阶”，不是“随机”**。这在 ComfyUI 里可以直接验证——新版 `SamplerER_SDE` 节点有 `solver_type`：

```text
solver_type = "ODE",  eta = 0,  max_stage = 3   ← 纯确定性高阶积分器
solver_type = "ER-SDE", eta = 1, max_stage = 3  ← 论文完整版
```

**先跑 eta=0 这一档**。如果它就把 Euler 的瑕疵治好了 → 你真正要的是阶数，压根不需要 SDE（还能保住可复现性）。如果 eta=0 一眼糊、eta=1 才“活” → 那说明你的模型在低步数下确实靠噪声补纹理，那就继续用 SDE。

代价与坑：

- **多步法无误差控制**，`denoised` 差分在 $\le 10$ 步时极噪 → 建议 $\le15$ 步用 `max_stage=2`，$\ge20$ 步再上 3。
- 前两步被 clamp 成 stage 1（`min(max_stage, i+1)`），等于自带 warmup，但也意味着**开头两步质量就是 Euler**。
- `eta` 太大官方就警告会出 invalid output（我见过糊成色块）。
- 差分会把**高频结构错误放大成振铃/halo**，高 CFG、密集文字场景尤其明显。

## 3. EulerA 是另一回事，别拿它当“Euler 的修复版”

k-diffusion 的 ancestral 逻辑是：

$$\sigma_{\text{up}}=\sqrt{\sigma_i^2-\sigma_{\text{down}}^2},\qquad x\leftarrow \hat x_0+\sigma_{\text{down}}\hat n+\sigma_{\text{up}}n'$$

也就是**每步把噪声分量按 $\sigma_{\text{down}}$ 全量重采样一次**——这是 churn 的**上限**（eta=1），比 ER-SDE 的小扰动大一到两个数量级。

后果：

- 它**不修正**截断偏差，只是把偏差随机化 → 颗粒感更强、纹理更“生猛”，结构一致性更差；少步数下容易五官/文字崩。
- 收敛慢，同步数下通常**比 Euler 还飘**。你“没怎么试”是对的，它不是免费午餐。
- 对 flow 模型，$\sigma$ 空间两端 churn 特别大，等于每步都在重新注入几乎全量噪声。

例外：它是**蒸馏/LCM 类模型**的标配（要的就是每步重采样），以及你想要“同 seed 多张不同细节”的多样性时。

<details><summary>为什么说 flow matching 上「SDE 好坏」是个连续谱（理论，点开）</summary>

stochastic interpolants / flow matching 的一个核心结论：给定 $v_\theta$，**marginally 完全相同**的采样器是一整个族，由一个自由的扩散系数 $\epsilon_t\ge0$ 参数化：

$$dx_t=\Big[v_\theta(x_t,t)-\tfrac{\epsilon_t}{2}\nabla\log p_t(x_t)\Big]dt+\sqrt{\epsilon_t}\,dW_t$$

（符号取决于时间方向约定；$\epsilon\equiv0$ 就是那条 ODE。）

也就是说：

- **没有“正确的 SDE”**。ER-SDE 的 churn 量、EulerA 的 churn 量、SA-Solver 的窗口，都是设计选择，不是从训练目标推出来的。
- 所以“ER-SDE 是不是最好”这个问题本身没有良定义——真正影响结果的是四件事：**（a）确定性部分的阶数，（b）$\epsilon_t$ 的形状，（c）步数与 shift，（d）CFG**。
- 为什么加噪声反而“更好看”：FID/感知指标系统性偏好高一点的 $\epsilon_t$（把样本推向高概率区 + 颗粒被人类当成细节）。所以**你眼睛看到的“最好”可能有偏差**——建议同 seed 做 A/B，再缩到 50% 看细节还剩多少，别只看 100% 裁切。

</details>

## 4. 给你的实验顺序（成本递增，别跳第一步）

1. **`euler` 40 步 + 正确 shift**。瑕疵还在吗？如果 40 步就干净了，那你的问题只是步数，收工。
2. **`heun` 20 步**（2 NFE/step，约等于 Euler 40–60 步的精度）。确定性、可复现、1 行替换。这是性价比最高的一次实验。
3. **`SamplerER_SDE` 里 `solver_type=ODE, eta=0, max_stage=3`，20 步**。看纯升阶能到哪。
4. **同节点切 `ER-SDE, eta=0.5→1`**，和上一步对比。这一对比较能干净地分离“阶数收益”和“噪声收益”。
5. **`sa_solver`**：`eta=1, sde_start_percent=0.2, sde_end_percent=0.8`。这是我觉得对 flow 模型**最讲道理**的随机方案——两端走 ODE 保住结构和细节，只在中段（曲率大、误差被放大的地方）注入噪声。
6. 再考虑 `deis` / `res_multistep` / `dpmpp_2m_sde`，以及降 CFG 换 SDE。

---

想更精准的话，告诉我这几个：**模型/架构（Flux? SD3.5? 自训 rectified flow? 有没有 reflow/蒸馏）**、**分辨率**、**步数与 shift**、**CFG**、以及瑕疵具体长什么样（塑料感 / 色偏 / 结构崩 / 颗粒 / 局部糊）。我能直接给你收敛得最快的那一组参数。

---
## 🧑‍💻 User

嗯，其实flow matching基本上都是euler
eulera可以把ODE变成SDE所以会更dynamic一点

---
## 😀 默认助手 (7)

两句都只说对了一半，我逐个把话说准确。

## 1.「flow matching 都用 Euler」——这是**约定**，理由不是“它最优”

对的地方确实存在：rectified flow 的路径接近直线，$\|\dot v_\theta\|$ 小；Euler 是 1 NFE/step；Flux / SD3 的官方参考实现都用它，整个生态因此固定下来。所以它就是默认值。

但不要把这个默认当上限：

- rectified flow 原论文自己写得很清楚：**未蒸馏**的 Euler 要 ~100 步才接近“无损”，而“1 步 / 2 步”是 **reflow 蒸馏**出来的能力。你在 20–30 步看到的，就是那部分尚未被消化的截断误差。
- **反证很明确**：Wan2.1 官方 `generate.py` 的 `--sample_solver` 默认是 `unipc`，可选值只有 `['unipc', 'dpm++']`——**Euler 根本不在选项里**。可见“flow matching 就该用 Euler”只是图像小模型圈的经验，不是规律。

而你“基本都能看到瑕疵”，我猜主因不是 Euler 本身，而是 **CFG 把场弄弯了**：

$$v=v_u+s\,(v_c-v_u)\quad\Longrightarrow\quad \|\dot v\|\ \text{随}\ s\ \text{增大}$$

CFG 后的场已经不是任何 coupling 的边缘速度场了，曲率随 $s$ 增大，而 Euler 误差 $\propto\|\dot v\|$。所以 CFG=1–2 时 20 步基本干净，CFG=4–7 就露馅。你可以先验证这一点：**把 CFG 降到 3，用 euler 25 步**，如果瑕疵大幅减少，那就不是采样器的问题。

## 2.「EulerA 把 ODE 变成 SDE」——对，但它是**上限**，不是任意一档

这个说法本身准确：ancestral 就是后向 SDE 的一阶离散步，不是玄学。但有三层需要补：

**(a) churn 固定在最大值。** k-diffusion 里：

$$\sigma_{\text{up}}=\eta\,\sigma_{to}\sqrt{1-\sigma_{to}^2/\sigma_{from}^2},\qquad \sigma_{\text{down}}=\sqrt{\sigma_{to}^2-\sigma_{\text{up}}^2}$$

而 ComfyUI 的 `euler_ancestral` **$\eta$ 固定在 1**，没有中间档。$\eta=1$ 意味着每步把噪声分量**全量重采样**一次——churn 的极大值。

**(b) 它只有一阶。** 所以它是“一阶 + 最大 churn”，是 SDE 族里**最不准**的那个成员。它给出的“dynamic”是真的，但那来自“每步重新粗化”，不来自“更好地解 SDE”。

**(c) 最关键：$\eta=1$ 对应的扩散系数是前向过程的 $g(t)^2$。** 而 flow matching 训练时**只定义了 ODE/速度场**——那族等边缘分布的 SDE 的扩散系数 $\epsilon_t$ 完全是**自由参数**，训练从没指定过。所以 EulerA 挑的这一档，对 flow 模型来说是“最激进”的一档，不是“匹配”的一档。

## 3. 但你的直觉背后有个更充分的理由

SDE 真正的优势不是“dynamic”，是**纠偏**。这才是值得你记住的部分：

**ODE：误差沿轨迹累积**

$$e_{k+1}\approx \big(I+h\,\partial_x v\big)e_k+\text{local},\qquad \text{放大率}\propto\|\partial_x v\|$$

一步预测误差被后续步顺着轨迹带下去、在 $\|\partial_x v\|>1$ 的方向上被放大。这正好解释了你说的“**基本都能看到**”——同 seed 重现、位置稳定，是确定性的偏差，不是偶然。

**SDE：每步噪声 + score 项把样本投影回 $p_t$ 的支撑**

$$\text{drift}=v_\theta-\tfrac{\epsilon_t}{2}\nabla\log p_t\ \Longrightarrow\ \text{误差被收缩而非累积}$$

这就是 ER-SDE 那篇（arXiv:2309.06169）的 one-step prediction error 分析的核心结论：**ODE 有“误差传播天花板”，SDE 没有**。

所以“更 dynamic”是个**现象**，机制是“**抗误差累积**”。推论也就清楚了：**低 NFE 时 SDE 更稳，高 NFE 时 ODE 更准**——不是“谁更好”，而是“谁更适合你的 NFE 预算”。

## 4. 所以正确的做法是：保留 SDE，但换到「可调 + 高阶」那档

<details><summary>churn × 阶数 的完整谱（点开看表）</summary>

| 采样器 | 阶 | churn | ComfyUI 可调性 |
|---|---|---|---|
| `euler` | 1 | 0 | — |
| `dpm_2` /**heun** | 2 | 0 | — |
| **`euler_ancestral`** | **1** | **max（η=1 固定）** | ❌ 无 eta |
| `dpmpp_2m_sde` | 2（多步） | η 可调 | ✅ `eta` + `solver_type` |
| `er_sde` | 2–3 | 中，启发式固定 | ✅ `eta` + `max_stage` + `solver_type` |
| **`sa_solver`** | pred 3 / corr 4 | **η + 噪声窗口** | ✅ 最全 |

</details>

我实际推荐跑的是 **`sa_solver`**：

```text
eta                = 0.7 ~ 1.0
sde_start_percent  = 0.15 ~ 0.25
sde_end_percent    = 0.70 ~ 0.85
predictor_order    = 3
```

理由正是上面那套机制：**两端走 ODE，中段走 SDE**。

- 高噪声端（$t\to1$）churn 最大，但那里最该走 ODE——一注入噪声，结构就飘了，后面的步根本救不回来。
- 低噪声端（$t\to0$）是精修细节，走 ODE 才能保住锐度，SDE 只会用颗粒盖住。
- **中间段**才是曲率大、$\|\partial_x v\|$ 大、误差被放大的地方——也正是最需要“投影纠偏”的地方。

这就是“要 dynamic 但不想付全部代价”的正确形态。`er_sde` 是它的近似（churn 固定、没有窗口），所以它在 15 步以下会输给纯 ODE。

## 5. 两个必须注意的问题

**① CFG 下 SDE 在理论上就不成立。** 纠偏需要一个真正的 $\nabla\log p_t$，而 CFG 的场不是任何 $p$ 的梯度 → “纠偏”没有目标可纠，噪声只会把你推离 guided distribution。表现就是 CFG>4 时 SDE 容易烤焦/蜡像/过饱和。

这大概率也是你觉得 SDE 好看的**另一半原因**：它把 high-CFG 的僵硬感糊开了。**代价是你已经不在你要求的那个分布上了。** 要么降 CFG，要么用 ODE，要么上 cfg++/rescale。

**② 视频/续写/img2img，SDE 直接抹掉一致性**（帧间闪、init image 漂）。Wan 官方只给 `unipc`/`dpm++` 这种确定性多步、不给 ancestral，不是没理由的。

## 6. 提醒：“更 dynamic 更好看”很可能是错觉

FID 和人类主观偏好都**系统性地偏好一点 churn**（颗粒被当作细节）。给个不花钱的盲测：

1. 同 seed 跑 `euler` / `euler_ancestral` / `er_sde(eta=0)` / `sa_solver`；
2. 全部缩到 50% 再比（颗粒的优势会消失）；
3. 再加一个**不崩溃率**：同 prompt 换 8 个 seed，统计结构崩的个数。`euler_ancestral` 在这个指标上通常是最差的——因为它每步都在赌。

---

一句话概括：**Euler 是 flow matching 的默认，不是上限；EulerA 是 SDE 的下限，不是 SDE 的代表。** 你要的那点“dynamic”，用 `sa_solver` 的噪声窗口 + 高阶预测器能拿到，还不必牺牲结构。

哪家的 flow 模型、多少步、CFG 多少？给我这三个数，我可以直接把 `sa_solver` 的窗口给你算出对应的步号区间。

---
## 🧑‍💻 User

flow match的话，一阶方法里，euler又简单又好，还有别的吗

---
## 😀 默认助手 (7)

## 结论先给

**一阶这一类里，基本只有 Euler 一个真正的选项。因为 flow matching 的 Euler 已经就是“指数积分器 / DPM-Solver-1 / DDIM(η=0)”的精确形式了。**

所以正确的做法不是在“一阶”里继续找，而是**同样的 NFE 预算，直接跳到二阶**，多步法能免费拿到二阶。下面先给证明，再给真正值得动的几个旋钮。

## 1. 为什么 flow 的 Euler 已经占满了一阶空间

只取线性路径 $x_t=(1-t)\epsilon+t\,x_0$，$v=x_0-\epsilon$ 这一情形。

**v-Euler**：

$$x_{k+1}=x_k+h(\hat x_0-\hat\epsilon)$$

**x0-形式（DDIM / 指数 Euler / DPM-Solver-1）**：

$$x_{k+1}=(1-t_{k+1})\hat\epsilon+t_{k+1}\hat x_0,\qquad \hat\epsilon=\frac{x_k-t_k\hat x_0}{1-t_k}$$

代进去展开：

$$x_{k+1}=\underbrace{(1-t_k)\hat\epsilon+t_k\hat x_0}_{=\,x_k}+h(\hat x_0-\hat\epsilon)=x_k+h(\hat x_0-\hat\epsilon)$$

**代数恒等。** 原因是 $\alpha_t=1-t,\ \sigma_t=t$ 对 $t$ 是**线性**的，指数积分器里那个 $e^{\int}$ 因子退化成 1。

对比扩散模型：VP 的 $\alpha_t=\exp\!\big(-\tfrac12\int_0^t\beta_s ds\big)$ 对 $t$ 是**指数**的 → exp-Euler ≠ Euler，DPM-Solver-1 真的比 Euler 好。这就是为什么 SD1.5 圈没人裸用 Euler，而 flow 圈全员使用 Euler。**原因在于一阶空间里的选择确实已经重合，并不是偷懒。**

<details><summary>顺带：这也解释了你最初的困惑</summary>

你在扩散模型上建立的直觉（“Euler 太粗糙，得换 DPM++/ER-SDE”）来自一个 **stiff ODE**：Jacobian $\|\partial_x v\|$ 大，$h\|\partial v\|\gtrsim 1$，一阶根本不够。

flow matching 的 ODE **并不 stiff**：条件路径是直线，$h\|\partial v\|\ll 1$。所以一阶的误差常数本来就小一个量级。

⇒ **你看到的瑕疵，大概率是误差常数被“别的因素”抬高了**（CFG 让场变弯、shift 设错、VAE），而不是“一阶阶数不够”。这个判断很重要，因为下面所有“换方法”的努力都可能花在错误方向上。

</details>

## 2. 一阶类里真实存在的四个变体（收益从大到小）

| 变体 | 针对的误差来源 | 收益 | 代价 |
|---|---|---|---|
| **换时间网格**（shift / log-SNR 均匀 / Karras） | 让误差常数在 $t$ 轴上分配更合理 | **最大**，实际是唯一量级上的改进 | 免费 |
| DEIS-1 / iPNDM-1（一阶多步指数） | 一阶项系统性偏差 | 很小——因为线性部分本来就精确 | warmup + 有记忆 |
| Euler–Maruyama（Euler + churn，即 ER-SDE 的 stage 1） | 把偏差**随机化**而非减小 | 观感变好，真值不变 | 破坏确定性 |
| 末步直接返回 $\hat x_0$ + x0 clipping | 输出域/饱和 | 小但稳定 | 免费 |

**只有第一条值得认真调。** 因为 $|\partial_t x_t|$ 在噪声端大、在数据端小，**均匀 $t$ 网格是浪费**：

$$\sigma_i=\Big(\sigma_{\max}^{1/\rho}-\tfrac{i}{N}\big(\sigma_{\max}^{1/\rho}-\sigma_{\min}^{1/\rho}\big)\Big)^{\rho},\qquad \rho\approx 7$$

在 $\sigma=\frac{t}{1-t}$ 空间取网格再映回 $t$（或用 ComfyUI 的 `ModelSamplingAuraFlow` 直接调 shift）。**方法一个字没改，误差常数能降一个量级**，这是“一阶”里唯一真实、显著的提升。

## 3. 真正的建议：同 NFE 上二阶，这是完全免费的

多步法（AB2 / `deis` / `ipndm` / `res_multistep` / `dpmpp_2m`）是 **2 阶、1 NFE/step**。所以“在一阶里挑选”这个动作本身就是浪费——**同样的 NFE，二阶是白拿的**。

但代价必须说清楚，而这也正是 Euler 依然赢的原因：

- **有记忆 → 误差累积**。Euler 每步只看当前状态，**一步坏了下一步自愈**；多步法一个坏步会污染后面所有步。
- **高 CFG 下不稳**。CFG 的场在语义上是**非光滑**的（$v_u+s(v_c-v_u)$ 在不同样本间跳变），多步法的差分估计会振铃 → halo/环状伪影。**你是高 CFG 用户，这就是你最后总会回到 Euler 的物理原因。**
- **前 1–2 步退化成 Euler**（warmup），≤10 步时多步法基本废掉。

经验法则：

$$\text{CFG}\le 3\ \&\ \text{steps}\ge 20\ \Rightarrow\ \text{多步二阶赚翻};\qquad \text{CFG}\ge 5\ \Rightarrow\ \text{老老实实 Euler}$$

## 4. 一个你可能没注意的、专为 flow 写的采样器

ComfyUI 新加的 **`exp_heun_2_x0`**，本质是 `sample_seeds_2`（SEEDS-2, arXiv:2305.14267）的确定性版：

> *Deterministic exponential Heun second order method in data prediction (x0) and logSNR time.*

也就是在 **x0 预测 + logSNR 时间**上做二阶指数 Heun，**为 x0 参数化（也就是 flow）专门设计**的。代价 2 NFE/step → 20 步 = 40 NFE，但误差是二阶的，而且比 `heun` 更贴合 flow 的参数化。想升级就试它，别试 `heun`。

（同一族里 `exp_heun_2_x0_sde` 是随机版，`seeds_3` 是三阶。）

## 5. 一个半小时就能定论的客观实验（比眼睛可靠一个量级）

用 `euler` **200 步**当 pseudo-GT，同 seed 同 prompt，算

$$\text{relRMSE}(N)=\frac{\big\|x^{(N)}-x^{(200)}\big\|_2}{\big\|x^{(200)}\big\|_2}$$

画 relRMSE vs NFE。曲线会直接告诉你三件事：

1. **斜率**：Euler 是 $-1$（一阶），二阶法是 $-2$ —— 一眼验证阶数；
2. **交叉点**：在哪个 NFE 二者互换 —— 这就是“该切二阶”的临界点，因模型/CFG 而异，别听任何人（包括我）报数字；
3. **是否收敛到 0**：如果高 NFE 端还悬着不为 0，说明 **shift/网格设错了**，跟采样器无关。

先跑这个，再决定要不要换采样器。我赌你在 fix 了 shift 之后，会发现 `euler` 25 步和你现在用的 ER-SDE 的 relRMSE 差不到你眼睛能分辨的量级。

---

**一句话**：一阶已经被 Euler 占满了（线性参数化下它就是指数积分器）；能改进的只有**时间网格**。如果把同样的 NFE 花在二阶或网格上更好，就别在一阶里找了。

告诉我是哪家的模型（Flux / SD3.5 / 自训 rectified flow，有没有 reflow 或蒸馏），我可以直接说你的 1 NFE 该花在网格还是阶数上。
