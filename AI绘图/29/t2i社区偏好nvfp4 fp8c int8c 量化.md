# t2i社区偏好nvfp4 fp8c int8c 量化

## 🧑‍💻 User

为什么t2i社区更倾向于做nvfp4 fp8c和int8c的量化而不是mxfp8（包括微调和推理）

---
## 😀 默认助手 (7)

# 一句话结论

在 T2I 的 Pareto 前沿上，**MXFP8 是被"夹死"的中间格式**：8 bit 相比 `fp8c`/`int8c` 拿不到任何显存/带宽收益，却要付出新 kernel、新硬件、新 checkpoint 格式的成本；4 bit 那边又有 `nvfp4`（per‑16 的 E4M3 scale + 二级 FP32 scale）在数值上明确优于 MX 家族（per‑32 的 E8M0 scale）。所以社区要么选"最省事且几乎无损"的 8‑bit per‑channel，要么选"省得最多且被 NVIDIA 背书"的 NVFP4，中间没有 MXFP8 的位置。

---

## 1. 先把格式对齐（避免概念错位）

| 格式 | 元素 dtype | scale 粒度 | scale dtype | 硬件门槛 | T2I 生态现状 |
|---|---|---|---|---|---|
| `fp8c` (W8A16/ W8A8) | E4M3 | per‑channel（沿输出通道）/per‑tensor | fp32/bf16 | Ada SM89+ 起，通用 | ComfyUI / diffusers 原生单文件 safetensors |
| `int8c` | int8 | per‑channel | fp32 | 全平台 | GGUF `Q8_0`、quanto、bnb |
| `nvfp4` | E2M1 | **per‑16** + 二级 per‑tensor fp32 | **E4M3** | Blackwell（SM100/120） | NVIDIA modelopt diffusion recipe、Nunchaku、ComfyUI 节点 |
| `mxfp8` | E4M3/E5M2 | per‑32（沿 K） | **E8M0** | 基本只有 Blackwell；且主要为 LLM serving（vLLM/SGLang） | T2I 基本空白 |

注意这里其实混了两组比较：**NVFP4 vs MXFP8 是"4bit vs 8bit"**，**FP8C/INT8C vs MXFP8 才是"同 8bit 不同 scaling 方案"**。社区的选择要在两条轴上分别解释。

---

## 2. 核心原因

### 2.1 8‑bit 轴：MXFP8 相对 `fp8c`/`int8c` 是"严格劣"的

这是最关键的一条，也是很多人忽略的：

- **显存/带宽上零收益。** 都是 1 byte/element。扩散 Transformer 推理是 **memory‑bandwidth bound**（batch=1、M 维很小），瓶颈在权重读取。MXFP8 相对 FP8C 一个字节都省不下来，还要额外存 block scale（每 32 个元素 1 byte，约 +3%）和做 scale 的 swizzle/layout 重排。
- **数值上收益接近零，因为 T2I 推理几乎都是 weight‑only。** 社区主流做法是 **W8A16**：激活保持 bf16/fp16、权重 8bit 存、GEMM 里 on‑the‑fly dequant。MX 格式的全部卖点（动态 per‑32 激活 block scale）在这个模式下**完全用不上**。
- **对权重来说，per‑channel 比 per‑32 block 更"对"。** 扩散模型的权重/激活离群是**通道级（channel‑wise outlier）**结构，per‑channel scale 正好对齐这个结构；而且 per‑channel 没有 K 必须对齐 32 的约束，可以无痛融合进 fused QKV、patch‑embed 的 3D conv 等层，dequant 甚至可以做成纯权重变换 + `torch._scaled_mm` 的 per‑col scale，**不需要写新 kernel**。per‑32 则必须让 GEMM 本体去吃 block scale，实现成本陡增。
- **`torch._scaled_mm` 的现实约束**：per‑tensor / per‑row / per‑col 长期是第一等公民；block‑wise（MX）路径加入晚、限制多、编译器/后端支持面窄。对 ComfyUI 这类"用户自己拼节点"的栈,基本等于不可用。

> **一句话**：8 bit 这个价位上，`fp8c`/`int8c` 是"零成本、零风险、免校准、全硬件可用"的解；MXFP8 是"高成本、同收益"。

### 2.2 4‑bit 轴：NVFP4 家族确实赢，而 MX 家族在这里本来就弱

4 bit 是**真有收益**的（显存腰斩、Blackwell FP4 张量核吞吐是 FP8 的 2×），所以 4bit 值得折腾。而一旦决定折腾 4bit，选项是 NVFP4 vs MXFP4，NVFP4 明显更好：

- **粒度**：per‑16 vs per‑32。4bit 只有 6 个可表示幅值（0.5,1,1.5,2,3,4,6），block 内动态范围一小，粗粒度 scale 立刻掉点。
- **scale 精度**：E4M3（3 位尾数）vs **E8M0（0 位尾数，纯 2 的幂）**。E8M0 只能取 $2^k$，为了避免溢出必须把 scale 向上取整，浪费最多 2× 的动态范围——在 FP8 上这无所谓（浮点相对精度不变），但在 FP4 上会直接把小块里的小元素压到 0（下溢）。**这是 MX 家族最实质的数值弱点**。
- **二级 scale**：NVFP4 有 per‑tensor FP32 的二级缩放，能吃掉扩散模型（尤其 adaLN/modulation 后）那种**全局重尾分布**；MX 只有单级。

所以"4bit 选 NVFP4"和"8bit 不选 MXFP8"其实是同一件事的两面：**MX 的 scale 方案在所有比特宽度上都不是最优解**。

### 2.3 生态与证据：这才是决定性因素，不是理论

T2I 社区的选型逻辑高度路径依赖，本质是"**有没有一条能跑通、有人验证过的路**"：

1. **用户栈是 ComfyUI**（外加 diffusers/SD.Next 系脚本）。ComfyUI 的量化能力就是：原生 `fp8_e4m3fn` 单文件、GGUF（k‑quant + `Q8_0`=int8）、以及近一年加入的 NVFP4 节点 / Nunchaku（SVDQuant，4bit+低秩修正）。**MXFP8 在 ComfyUI 里没有入口。**
2. **单文件、可直接下载的 checkpoint** 是 T2I 的分发范式。`*-fp8_e4m3fn.safetensors` 是社区事实标准，谁都能拖进 `models/diffusion_models` 就跑；MXFP8 需要非标准 block‑scale 张量布局，**没人发布**。
3. **硬件基数**：绝大多数 T2I 用户在 RTX 30/40 系（Ampere/Ada）。MXFP8 和 NVFP4 都基本要 Blackwell，但 NVFP4 有"4× 省显存"这个足够强的动机让人上 50 系；MXFP8 在 Ada 上等于**什么都没有**，动机为零。
4. **主观质量零容忍**：现在 T2I 要做文字渲染（Qwen‑Image、Seedream 一类），**字形/排版对量化极其敏感**，text encoder 和 cross‑attention 稍微掉点肉眼可见。所以社区只要"**有公开 eval 的格式**"——NVIDIA 给 NVFP4 出过 FLUX 的 FID/CLIP/人评对照，FP8 有海量社区实测；MXFP8 在扩散上**没有可信的公开评测**，于是没人敢用。

### 2.4 微调（QAT / LoRA）侧的额外摩擦

<details>
<summary>展开：为什么 fine‑tuning 更不倾向 MX</summary>

- **已有工具链完全被占位**：`bitsandbytes` 的 INT8 与 NF4、`optimum-quanto` 的 fp8/int8、TorchAO 的 `int8_weight_only`/`float8`、diffusers 的 `BitsAndBytesConfig(load_in_8bit / 4bit)`，都是"一行配置就能 QLoRA"的成熟路径。MXFP8 的 QAT 只在 TorchAO 里有 block‑wise 模拟，**扩散端的端到端 recipe 基本不存在**。
- **E8M0 的 STE 很别扭**：对 $2^k$ 这种离散 scale 求梯度只能硬 STE，而 scale 的逐步取整会让 loss 面出现平台/抖动；相比之下 per‑channel 的 fp32 scale 是连续可学的，直接走标准 STE 就行。
- **block scale 与 LoRA 的交互差**：LoRA 在量化基座上加低秩旁路，基座权重是"冻结的、静态 scale"最省事；MX 的动态 block scale 在每次前向都要重算，训练时显存和 kernel 调用都更贵。
- **反向 GEMM 支持**：Blackwell 上 FP4/FP8 的 training GEMM 有 TransformerEngine / modelopt 的 NVFP4 支持；MXFP8 的训练路径主要在 LLM 侧，扩散的结构（adaLN、3D conv patchify、joint attention）没有对应优化。
</details>

---

## 3. 什么情况下 MXFP8 反而是对的

<details>
<summary>展开：MX 的真正主场</summary>

- **大 batch 的 LLM serving W8A8**：激活的 per‑32 动态 block scale 才有用武之地，vLLM/SGLang 在 B200 上确实吃这套。
- **跨厂商可移植性**：MX 是 OCP 标准，NVIDIA/AMD/其他都在推，如果你的目标是"同一份量化权重在多家加速器上跑"，MX 是唯一有标准保障的选择。**但 T2I 社区是单卡消费级 NVIDIA 主导，这个诉求不存在。**
- **超长序列 + 激活离群极强**的层，per‑tensor 会崩、per‑channel 不够、per‑block 才稳——这在扩散里是少数层，靠 layer‑wise 混合精度（把这几层留在 bf16）比整体换 MX 更划算。

</details>

---

## 4. 归因小结

| 维度 | `fp8c` / `int8c` | `nvfp4` | `mxfp8` |
|---|---|---|---|
| 显存收益 vs bf16 | 2× | **4×** | 2×（无差异化） |
| 数值风险 | 极低（per‑channel 对齐离群结构） | 中（但有 per‑16 + 二级 scale 兜底） | 低（但 scale 方案无优势） |
| 是否需新 kernel | 否（`_scaled_mm` 现成） | 是，但 NVIDIA 已给 | 是，且无人给 T2I 做 |
| checkpoint 分发 | 事实标准 | 已有官方/社区产物 | **无** |
| 硬件覆盖 | Ampere/Ada/Hopper/Blackwell | Blackwell | 基本 Blackwell |
| 公开扩散 eval | 海量 | 有（NVIDIA） | **无** |
| 微调工具链 | bnb/quanto/TorchAO 一线可用 | modelopt + TorchAO | TorchAO 实验性 |

**最终逻辑**：8 bit 选 per‑channel 是因为"免费且够用"，4 bit 选 NVFP4 是因为"只有它能同时给出 4× 显存 + 2× 吞吐 + 可接受的质量"，而 MXFP8 落在两者之间——**省不了显存、快不了、也没人验证过**。

> ⚠️ 说明：以上关于"MXFP8 在扩散/ComfyUI 侧几乎无落地路径"的判断基于我训练数据截止时的生态状况；2025 年之后 block‑scale GEMM 与各家 recipe 的推进速度较快，如果你的场景是新硬件 + 自研栈，建议直接查 TorchAO / modelopt / vLLM 当前的 `mxfp8` 支持矩阵再下结论。

---
## 🧑‍💻 User

@联网搜索 为什么llm预训练很早就已经用上了（包括deepseek v4、kimi k3、nemotron3、MAI-Base-2）

---
## 🧑‍💻 User

为什么llm预训练很早就已经用上了（包括deepseek v4、kimi k3、nemotron3）

---
## 🧑‍💻 User

为什么llm预训练很早就已经用上了（包括deepseek v4、kimi k3、nemotron3、MAI-Base-2）

---
## 😀 默认助手 (7)

# 先核对一下事实（各家用的其实不是同一个东西）

| 模型 | 训练/部署数值格式 | 时间 | 证据 |
|---|---|---|---|
| DeepSeek V3 | **FP8** 混合精度，weight/act 用 $1\times128$ 分块 scale，grad 用 $128\times128$ | 2024.12 | V3 技术报告 |
| **DeepSeek V4** | 延续 FP8 路线（论文里 MoE dispatch 就是 **FP8 Dispatch + BF16 Combine**）；HF 权重标 `fp8` / 8-bit | 2026.04 | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348)、[HF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) |
| **Kimi K3** | **MXFP4 权重 / MXFP8 激活**，从 SFT 阶段开始做 QAT | 2026.07 | [模型卡原文](https://huggingface.co/moonshotai/Kimi-K3)："MXFP4 weights with MXFP8 activations for broad hardware compatibility" |
| **Nemotron 3 Super / Ultra** | **NVFP4 预训练** | 2026 | [NVIDIA 原文](https://research.nvidia.com/labs/nemotron/Nemotron-3/)："NVFP4: Super and Ultra are trained with NVFP4." |
| **MAI-Base-2** | 公开博客**未披露**格式，只提到"forward 阶段权重量化与 shared-expert 计算重叠" | 2026.10 | [Frontier Pre-training at Microsoft AI](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/) |

所以修正一下时间线：**LLM 是"FP8 起家（2024）→ 2025–26 才走到 MXFP8/NVFP4 的 QAT"**，不是一开始就 MXFP8；而且 NVIDIA 那篇 [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)（2025.5）正是这条路线的"可复现说明书"——8B 模型、15T token，loss 曲线对齐 BF16。

---

# 核心：MX 类格式的收益函数，"覆盖面"决定一切

这是理解整个不对称的钥匙。请把两种格式的价值分开看：

- **8-bit 域里，MXFP8 相对 FP8/per-channel 不省字节**（都是 1 byte/element），它换来的是**"同样的字节、更高的精度"**。所以它的收益**只有在"你需要量化更多张量"时才兑现**。
- **4-bit 域里，NVFP4 是"一半字节、同样的精度档次"**，所以它的收益在**任何带宽受限场景**都兑现——包括 T2I 推理。

于是统一成一句话：

> **MXFP8 卖的是量化覆盖率，NVFP4 卖的是比特密度。** 前者只有"训练（含反向）+ 超大 batch 服务"这种必须全覆盖的负载才付得起；后者只要是 memory-bound 就要。

上一轮我说 T2I 选 `fp8c`/`nvfp4` 而不选 `mxfp8`——理由正是"它只需要量化权重，覆盖率收益兑现不了"。而 LLM 预训练恰恰是**覆盖率拉满**的负载。

---

# 为什么 LLM 预训练这个收益函数被拉满

<details>
<summary><b>1. 它是"一个 GEMM 重复一百次"，且完全计算受限</b></summary>

一个 LLM 预训练 run 里 99% 的 FLOPs 都走同几个形状的 GEMM，$M$ = batch × seq 是**百万到上亿 token** 量级。这是教科书级的 compute-bound 场景，FP8 张量核利用率能吃到 40–60%。

在这种负载下：
- per-32 block scale 的搬运（+3.1% 字节）、swizzle、epilogue 开销，被巨大的 $M$ 维**完全摊薄**；
- 而 per-**tensor** FP8 不够用——attention 输入、梯度、MoE 各专家的激活分布差异太大，一个全局 scale 会崩。**per-32 恰好是"能塞进现有 GEMM 的最小代价精度单位"**。

这就是 NVIDIA 那篇论文的原话：MX 格式"enables both quantization of **more tensors** than previous approaches and more efficient execution"。
</details>

<details>
<summary><b>2. 收益的量级：一次改动等于几千万到上亿美元</b></summary>

32T token、数万张卡、跑几个月。端到端 1.3–2× 的提速，换算成美元是**八位数到九位数**。为此投几百人年去写 block-scale GEMM、FP8 all-to-all、fused MoE mega-kernel（DeepSeek 甚至把 MegaMoE 开源了），ROI 是几十倍。

T2I 的 frontier 训练 run 通常是**几千卡-周**级别，同样的工程投入折算下来是负收益。**不是技术做不到，是账算不过来。**
</details>

<details>
<summary><b>3. 激活显存：长上下文把这条压成了刚需</b></summary>

$M$ 大 + 序列长（V4 支持 1M token）→ **激活显存而非权重显存成为瓶颈**，直接决定你能开多大 batch、要不要重算。8-bit 激活把这块砍半，等于免费换 batch size。

DiT 的 T2I 训练没有这个压力：单样本就是一个固定分辨率图，序列长度短且规整。
</details>

<details>
<summary><b>4. 格式必须"训练时就定"，因为没有第二次机会</b></summary>

LLM 的商业模型是**训练一次、服务十亿次**。PTQ 掉一点分，就是永久的、乘以全部流量的损失。所以最优解是把量化前移到预训练/QAT——Kimi K3 的模型卡写得很直白："broad hardware compatibility"。

反过来，**如果不在这里定格式，你后面要付的钱是持续的**；而训练侧的 QAT 成本是一次性的。这笔账 LLM 一定算得赢。
</details>

<details>
<summary><b>5. 数值安全网是"模型架构"一起设计的</b></summary>

低精度不是硬套上去的，是**共设计**出来的。2026 年这几篇里全是这类 knob：

- DeepSeek V4：**Anticipatory Routing** + **SwiGLU clamping 到 $[-10,10]$**，明确说是为了压 MoE 层的 outlier；
- MAI-Base-2：**activation gradient clipping**（逐激活裁剪而非全局范数）+ Muon；
- 更早的 QK-Norm / QK-Clip 同理。

这些手段让"极端离群值"在进入 GEMM 前就被驯服，block scale 才有意义。**LLM 有一整套明确的、正交的架构旋钮去吸收低精度误差；扩散模型这边对应手段还没成熟，而且采样是 20–50 步迭代的，误差会累积放大**——量化误差在迭代采样里不是"加性"的，是"乘性/反馈"的。
</details>

<details>
<summary><b>6. 验证标准低得多</b></summary>

LLM 只要证明"loss 曲线与 BF16 重合"就能发。这是**单一标量、可复现、一周内出结论**的证据。

T2I 的 loss 不预测感知质量，你必须跑 FID/CLIP/Arena/人评，而且掉点是**局部**的（字形、手指、纹理），要大量人工看。验证成本高一个数量级，反馈慢一个数量级——**这就是"没人敢第一个上"的真实原因**，不是理论不可行。
</details>

<details>
<summary><b>7. 硬件是"自有、同构、上一代"的</b></summary>

LLM 预训练跑在自己买的 B200/GB200 集群上，**100% Blackwell**，用 SM100-only 的格式零代价。

T2I 的产物要发到用户的 3080/4090/5090 上。选一个 Ada 上不存在的格式，等于**自断用户基数**。这条在推理侧是致命的，在训练侧不存在。
</details>

---

# 对照：为什么 T2I 这边账算不过来

| 维度 | LLM 预训练 | T2I（训练 + 本地推理） |
|---|---|---|
| GEMM 形态 | $M$ 上百万，compute-bound | 推理 $M\!\approx\!1$，**bandwidth-bound** |
| MX 的收益来源 | 量化覆盖率（fwd+bwd 全覆盖） | 不需要覆盖率（权重-only 就够） |
| 工程投入回收 | 数万卡 × 数月 | 几千卡-周 / 个人开发者 |
| 格式决策时机 | 训练时定，终身受益 | 必须兼顾 Ampere/Ada/50 系 |
| 误差容忍 | 单一 loss 对齐即可 | 迭代采样累积 + 人评主观 |
| 架构旋钮 | QK-Norm / clamp / 梯度裁剪已成熟 | 对应手段尚缺 |
| 谁写 kernel | NVIDIA + 各 lab 自建 | 消费者吃 ComfyUI 现成路 |

还有一个容易被忽略的**分发格式约束**：T2I 的交付物是"一个能拖进 ComfyUI 的 `.safetensors`"。MXFP8 需要非标准的 block-scale 张量布局，**没有社区约定**；`fp8_e4m3fn` 单文件是全世界的既成事实。这个摩擦比任何数值分析都更能解释现实。

---

# 但真正的分界线不是 "LLM vs T2I"

<details>
<summary><b>是"集中式服务 vs 本地分发"</b></summary>

沿着上面的逻辑推下去，你会得到一个可验证的预测：

- **T2I 的 API 服务端**（MAI-Image、Seedream、GPT-Image 这类）——经济学和 LLM serving **完全同构**：自有同构集群、按 token/张计费、带宽受限。**这里是 NVFP4/MXFP4 QAT 的天然主场，而且会比开源社区早 1–2 年落地。** 事实上 Kimi/MAI 这类"模型厂 = 服务商"的角色，天然就会在 SFT 阶段就上 QAT。
- **T2I 的开源本地分发**——会长期停在 `fp8c` / `int8c` / GGUF / Nunchaku 的 SVDQuant（用的是 NVFP4 on Blackwell、INT4 on 老卡）。
- **跨界点**：一旦某个 DiT 的 QAT checkpoint 同时发布"NVFP4 版 + FP8 版"（Kimi K3 就是这么干的），开源社区就会跟上——因为那时候"覆盖率收益"通过**别人替你付的工程费**兑现了。

所以答案是：**LLM 早，不是因为它更"需要"低精度，而是因为它同时满足了三个条件——算力账单够大、负载够同构、硬件够同构。T2I 缺的是后两个，而随着 T2I 走向集中式服务，第二个条件正在被补上。**
</details>

---

<details>
<summary><b>几个不该忽略的反面</b></summary>

1. **MAI-Base-2 我没找到公开的格式声明**。博客只写了 "weight quantization ... overlap with the exposed communication"，这暗示 forward 里在动态量化权重（典型 FP8 训练特征），但**不足以断言它用了 MXFP8 或 NVFP4**。如果你有更具体的来源，我建议以那份为准。
2. **Kimi K3 的 MXFP4/MXFP8 是 "from the SFT stage onward"**，不等于 30T+ token 的预训练全程用 MXFP4。措辞很讲究，值得注意。
3. **Nemotron 3 用 NVFP4 而非 MXFP8**——这其实进一步印证我的论点：NVIDIA 在自己能控制全栈的时候，直接跳到了"比特密度"更好的那一档，而不是"精度换覆盖率"的 MXFP8。MXFP8 在 LLM 里更像是"FP8 → NVFP4 之间的过渡态/兼容层"。
4. **MXFP8 的长期地位可能被高估**。它的核心价值是"在 FP8 张量核上多吃张量"，一旦 Blackwell 的 FP4 训练路径成熟（Nemotron 3 已经证明了），这块地就被 NVFP4 吃掉。也就是说 **MXFP8 可能是一个"两代半"的格式，而 NVFP4 才是终点**——这也解释了为什么 T2I 社区"跳过" MXFP8 直接上 NVFP4 反而是对的。
</details>

> 检索说明：以上模型事实来自各模型的 HF 模型卡 / 技术报告 / NVIDIA 研究页（2026-10 时点）；搜索引擎在本次会话中对部分深度内容（DeepSeek-V4 训练精度细节段、Nemotron 3 Ultra 技术报告 PDF）未能取得，相应之处我标注了证据强度的差异。
