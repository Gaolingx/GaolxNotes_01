# DiT加速方案概述

## 🧑‍💻 User

dit推理加速有什么办法吗（分别从训练侧（例如直接meanflow变种做1~4step fm）/推理侧（例如latent cache）角度解答）

---
## 🤖 Assistant

# DiT 推理加速方法全览

DiT（Diffusion Transformer）的推理瓶颈在于**多步迭代采样**（通常 20–50 NFE）以及**每步大模型前向**。加速思路可归为两大类：**训练侧**改变模型/训练目标（让模型天生就会少步生成），**推理侧**不改权重、只改计算复用与求解策略。下面分别展开。

---

## 一、训练侧加速（少步/一步生成）

核心目标：把「多步 ODE 求解」压缩成 **1–4 NFE**。主要有三条技术路线。

<details>
<summary><b>关键结论（点击展开）</b></summary>

- **蒸馏类**（Consistency / LCM / DMD / ADD）：效果好但依赖 teacher，训练昂贵。
- **直接回归类**（Rectified Flow / Shortcut / **MeanFlow**）：无需预训练 teacher，直接从数据学「平均速度场」，是当前最热的 1-step 方向。
- **MeanFlow** 的精髓是把「瞬时速度 $v$」积分成「平均速度 $u$」，一次前向即可跨整段区间。

</details>

### 1. 蒸馏类（Distillation）

| 方法 | 思想 | 步数 |
|------|------|------|
| Progressive Distillation | 逐步减半步数，$\frac{dz}{dt}$ 蒸馏 | 4–8 |
| Consistency Models / CD | 学 $f_\theta(z_t,t)\to z_0$ 自一致性映射 | 1–4 |
| **LCM** (Latent Consistency) | 在 latent 空间做 consistency 蒸馏，CFG 内嵌 | 1–4 |
| **DMD / DMD2** | 用分布匹配（KL 反向 + 对抗）蒸馏 | 1–4 |
| **ADD / SDXL-Turbo** | 对抗式扩散蒸馏 | 1–2 |

特点：需要训练好的 teacher、训练成本高，但 1-step 质量目前最强。

### 2. 直接回归类（无 teacher）

- **Rectified Flow / Reflow**：学习直线轨迹，让 ODE 路径尽量笔直，从而可用大步长；Reflow 迭代拉直。
- **Shortcut Models**：网络额外输入步长 $d=0/1$，学「从当前步直接跳到未来某步」的 shortcut，支持 self-consistency。
- **MeanFlow**：见下。

### 3. MeanFlow 及其变种（重点）

**核心定义**：普通 Flow Matching 学的是**瞬时速度**
$$v(z_t,t)=\frac{dz_t}{dt}$$

MeanFlow 直接学**区间 $[r,t]$ 上的平均速度**：
$$u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau=\frac{z_t-z_r}{t-r}$$

关键恒等式（MeanFlow identity，把 $u$ 与 $v$ 联系起来，训练时用 JVP 实现）：
$$u(z_t,r,t)=v(z_t,t)-(t-r)\frac{d}{dt}u(z_t,r,t)$$

训练目标用「stop-gradient 的 target」回归：
$$\mathcal{L}=\big\|u_\theta(z_t,r,t)-\mathrm{sg}\big(v-(t-r)\tfrac{d}{dt}u_\theta\big)\big\|^2$$

**推理**：一步即可跨整段
$$z_r=z_t-(t-r)\,u_\theta(z_t,r,t)$$

即取 $t=1, r=0$ 得 **1-NFE** 生成；也可 2–4 步微调质量。

<details>
<summary><b>MeanFlow 常见变种/改进方向</b></summary>

- **iMeanFlow**：改进训练稳定性与 CFG 融合。
- **MeanFlow + CFG**：把 guidance 融入平均速度，保持 1-step；或做 CFG distillation 避免每步两次前向。
- **Two-stage / 混合**：先用 MeanFlow 粗生成，再用 few-step 精修。
- **Latent MeanFlow**：在 VAE latent 上做，直接对接 DiT（如与 SD3/Flux 结构结合）。
- **加对抗损失**：弥补 1-step 的细节缺失（类似 DMD + MeanFlow 结合）。
- **Flow-map learning 统一视角**：Shortcut、Consistency 都可视为 $u$ 的不同特例。

</details>

### 4. 其他训练侧技巧

- **CFG Distillation / Guidance embedding**：把 CFG 的两次前向合并为一次（如 guidance scale 作为输入 embed），直接省 2× 计算。
- **Guidance-free / CFG-aware 训练**。
- **对步长条件化**（step-size conditioning），支持任意步数自适应。

---

## 二、推理侧加速（不改权重）

### 1. Latent / Feature Cache 系列（你最关心的）

核心思想：DiT 相邻时间步的**中间特征高度相似**，因此可以**复用**部分 block / latent 的计算。

| 方法 | 复用对象 | 特点 |
|------|----------|------|
| **DeepCache** | U-Net 高层 block 输出 | 经典，按固定 interval 复用 |
| **FORA** | 部分 attention/block 缓存 | 按重要性选择 |
| **Δ-DiT** | 缓存变化量（delta） | 用变化量近似，误差更小 |
| **TeaCache** | 基于输入时间步差的复用判据 | 自适应、无需训练，DiT 友好 |
| **ToCa** | token 级 cache | 细粒度 |
| **Learning-to-Cache** | 学习「哪些层可跳过」 | 数据驱动 |
| **Block / Layer Caching** | 整个 transformer block | 实现简单 |

典型收益：**1.5×–2.5×** 加速，质量损失较小；缺点是与少步方法叠加时收益递减（因为步数本来就少）。

<details>
<summary><b>Cache 的实现要点与坑</b></summary>

- **复用判据**：固定间隔 vs. 自适应（TeaCache 用累计误差阈值 $\Delta$）。
- **计算-缓存权衡**：缓存太激进会模糊细节（尤其高频纹理）。
- **与 CFG 交互**：cond/uncond 两支可分别缓存。
- **与 solvers 叠加**：高阶 solver 相邻步差更小，缓存更有效。

</details>

### 2. 采样器 / 求解器优化

- **高阶求解器**：DPM-Solver / DPM-Solver++、UniPC、DEIS —— 用更少步数达到同等质量（20 步 → 8~12 步）。
- **自适应步长**：根据误差自动分配步数。
- **高阶 + cache 组合**：是推理侧最直接的「即插即用」方案。

### 3. 模型压缩与系统优化（正交叠加）

- **量化**：W8A8 / FP8 / INT4（DiT 权重量化 + 激活量化），TensorRT / torch.compile 融合。
- **剪枝 / 稀疏**：结构化剪枝、稀疏 attention。
- **Attention 优化**：FlashAttention、稀疏/线性 attention（如 DiT 的 attention 近似）。
- **并行**：序列并行、context 并行、张量并行（长视频/高分辨率 DiT）。
- **编译**：`torch.compile`、CUDA Graph、算子融合。

---

## 三、方法选择建议

| 场景 | 推荐组合 |
|------|----------|
| 追求 **1–4 step** 极致速度 | 训练侧：**MeanFlow / DMD2 / LCM** |
| 只能改推理、有现成模型 | **高阶 solver + TeaCache/Δ-DiT** + 量化 |
| 视频/长序列 DiT | cache + 序列并行 + FP8 |
| 想要 1-step 且不想训 teacher | **MeanFlow / Shortcut** |
| 质量优先、容忍 8 步 | DPM-Solver++ + 轻量 cache |

**组合策略**：训练侧（少步模型）+ 推理侧（cache/量化）通常可以叠加，但要注意当步数降到 4 步以下时，latent cache 的收益会明显变小，此时应优先投入训练侧的 1-step 方法。

---

需要的话，我可以进一步：
1. 给出 **MeanFlow 的训练伪代码**（含 JVP 实现）；
2. 对比 **TeaCache / Δ-DiT / ToCa** 的具体复用判据公式；
3. 或帮你查最新的 arXiv 论文（若需要我调用检索工具）。
