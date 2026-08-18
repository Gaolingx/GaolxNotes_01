# DiffSinger中LynxNet解析

### 🧑‍💻 User

翻阅github，阅读paper，说说diffsinger中的lynxnet 是什么模型，结构长什么样，输入输出是什么？

---

### 🤖 Assistant

我来帮你调研 DiffSinger 中的 LynxNet。先并行搜索 GitHub 和相关论文。

GitHub 主仓库已拿到（注意 arxiv 2305.02447 是无关的数学论文，真正的是 2105.02446）。现在我需要直接去代码里找 LynxNet。让我拉取仓库完整文件树，并同时读取原始论文。

找到了关键代码文件：`modules/backbones/lynxnet.py` 和 `lynxnet2.py`。现在读取这些源码，同时搜索 LynxNet 对应的论文。

拿到了 LynxNet 和 LynxNet2 的完整源码。现在继续：查看声学编码器里 backbone 如何被使用、配置文档里如何配置，以及搜索 arxiv 上是否有 LynxNet 的论文。

arXiv API 返回的是 XML（不是 JSON），我改用文本获取；GitHub 代码搜索 API 需要认证，改用其他渠道。同时查看声学任务配置模板和训练任务，弄清 backbone 在整体声

arXiv 上搜 "LynxNet" 返回 0 条结果——说明它还没有独立论文，是 OpenVPI 社区在 DiffSinger 里实现的骨干网络。继续确认它在整体架构中的位置：查看顶层模型 `modules/toplevel.py`、基础声学配置、以及发布记录。



信息非常完整了——发布记录确认 **v2.5.0 引入 LYNXNet 作为新一代声学模型解码器骨干网络**。再补几个细节：扩散模块如何调用 backbone、aux decoder 的作用、常见层（GLU 变体）定义，以及 PR #200 的设计

PR #200 给出了设计出处：LYNXNet 由 KakaruHayate 贡献，参考 Diffusion-SVC 的实现改写。最后确认扩散模块内部对 backbone 的调用方式（x、t、condition 的形状与语义），再看 `compat.py`。



最后补一个关键组件的定义（`common_layers.py` 中的 SwiGLU / ATanGLU / SoftSignGLU / SinusoidalPosEmb 等），确保对 LYNXNet2 结构的描述准确

调研完成。我通读了 openvpi/DiffSinger 仓库中 `modules/backbones/lynxnet.py`、`lynxnet2.py`、`modules/toplevel.py`、`modules/core/ddpm.py`/`reflow.py`、配置文件、发布记录和 PR #200，也检索了 arXiv。下面是完整结论。

---

## 结论速览（TL;DR）

- **LYNXNet = Linear Gated Depthwise Separable Convolution Network（线性门控深度可分离卷积网络）**，是 DiffSinger 声学模型中**扩散解码器的骨干网络（backbone / denoise_fn / velocity_fn）**，用于替代原来的 WaveNet。
- 它由社区开发者 **KakaruHayate** 在 **PR #200** 引入，参考 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的实现改写，随 **v2.5.0（2025-04-15）** 正式发布：*"The new backbone shows better performance on acoustic models."*
- **没有独立论文**。我查了 arXiv（`all:"LynxNet"` 返回 0 条）——原版 DiffSinger 论文（arXiv:2105.02446, AAAI 2022）只讲 shallow diffusion 机制，那时还没有 LynxNet。LynxNet 纯粹是 GitHub 仓库里的社区实现，说明文档在 PR、release notes 和 `docs/` 里。
- 目前仓库里有两代：`lynxnet`（v1，卷积+残差风格）和 `lynxnet2`（v2，全 Linear+深度可分离卷积的 MLP 风格），**新版默认配置用的是 `lynxnet2`**。

---

## 1. 它在 DiffSinger 里是什么角色

DiffSinger (OpenVPI) 的合成管线分三段：

$$ \text{音素/乐谱} \xrightarrow{\text{variance模型}} \text{时长/音高/气息等} \xrightarrow{\text{acoustic模型}} \text{Mel谱} \xrightarrow{\text{vocoder}} \text{波形} $$

LynxNet 位于**声学模型**中。`DiffSingerAcoustic`（`modules/toplevel.py`）由三部分组成：

| 组件 | 作用 |
|---|---|
| `fs2`（`FastSpeech2Acoustic`） | 音素/音高等条件 → 逐帧条件向量 `condition` |
| `aux_decoder`（ConvNeXt） | 浅扩散辅助解码器，先快速粗预测一个 Mel（shallow diffusion 的起点） |
| `diffusion`（`GaussianDiffusion`/`RectifiedFlow`） | 扩散/整流流采样器，**内部持有 backbone = LynxNet** |

`diffusion` 里通过 `build_backbone(out_dims, num_feats, backbone_type, backbone_args)` 创建网络（`modules/backbones/__init__.py` 注册表：`wavenet` / `lynxnet` / `lynxnet2`）。在 DDPM 中它叫 `denoise_fn`（预测噪声 $\varepsilon$），在 Rectified Flow 中叫 `velocity_fn`（预测速度场 $v$），调用约定统一为：

```python
out = backbone(x, t, cond)        # 每一步采样/训练都调用它
```

---

## 2. LYNXNet v1 结构（`modules/backbones/lynxnet.py`）

完整数据流（以声学模型 `n_feats=1, in_dims=M=128` 为例）：

```
输入: x=[B,1,M,T]  t=[B,1]  cond=[B,H=384,T]
  │ x = spec[:,0]                     # [B,M,T]
  │ x = Conv1d(M→C=1024, k=1)         # input_projection [B,C,T]
  │ x = GELU(x)   （strong_cond=False 时）
  │ t_emb = SinusoidalPosEmb→Linear(C→4C)→GELU→Linear(4C→C)   # [B,C,1]
  ▼
  重复 num_layers=6 次【LYNXNetResidualLayer】:
     x = x + CondProj(cond) + DiffProj(t_emb)          # 都是 k=1 Conv1d
     h = LYNXConvModule(x)
         = LayerNorm → Conv1d(C→2·(C·exp)) → SwiGLU
           → 深度可分离 Conv1d(inner, k=31, groups=inner) → PReLU/SiLU/ReLU
           → Conv1d(inner→C) → Dropout
     x = x + h（残差）
  │ post-norm: LayerNorm(C)
  │ x = AdamWConv1d(C→M, k=1) 且权重初始化为 0   # output_projection
  ▼
输出: [B,1,M,T]（DDPM 下是预测噪声 ε，RF 下是速度 v）
```

关键组件：
- **门控**：`Conv1d(dim→2·inner) → SwiGLU`（Split 成 out/gate 两半，`gate = silu(gate)` 后与 out 逐元素相乘），即"Linear Gated"。
- **深度可分离卷积**：`Conv1d(inner→inner, kernel_size=31, groups=inner)`，即通道维度不混合，只做时间维局部建模（感受野 31 帧）。
- **条件注入**：每一步残差层把条件 `cond`（FastSpeech2 编码器输出）和时间步嵌入相加注入；`strong_cond=True` 时条件在前端直接加（`front_cond_inject`）。
- **激活**可配：`PReLU`（默认，风格接近 WaveNet）/ `SiLU`（声音更"明显"，作者不建议用于 DDPM）/ `ReLU`（反之声音更弱）。

---

## 3. LYNXNet2 结构（`modules/backbones/lynxnet2.py`，默认骨干）

v2 是全 Linear 的改进版，输入输出与 v1 完全相同，但每个 block 变成"深度可分离卷积 + 双重 GLU 的 MLP"：

```
输入: x=[B,1,M,T]  t=[B,1]  cond=[B,H,T]
  │ x = Linear(M→C, transpose)                       # [B,T,C]
  │ x = x + CondProj(cond) + t_emb                  # 条件随意者缓存 mode 下 CondProj 是 Conv1d
  ▼
  重复 num_layers=6 次【LYNXNet2Block】（残差）:
     LayerNorm → 深度可分离 Conv1d(C→C, k=31, groups=C) → transpose
     → Linear(C→2·inner) → GLU → Linear(inner→2·inner) → GLU → Linear(inner→C)
     → Dropout，再 x = x + block(x)
  │ post-norm LayerNorm(C)
  │ Linear(C→M)，权重初始化为 0
  ▼
输出: [B,1,M,T]
```

GLU 门控三种可选（`glu_type`）：
- `swiglu`：`gate = silu(gate)`（另有 FP16 下的溢出保护缩放）；
- `atanglu`：`gate = atan(gate)`，**当前模板默认**，并带自定义 autograd（省显存）；
- `softsign_glu`：`gate = softsign(gate)`，数值更稳，支持关闭 Triton 融合 kernel（配置 `use_fused_kernels: true`，`modules/kernels/fused_linear_softsign_glu.py` 会把 Linear+SoftSignGLU 融合加速）。

---

## 4. 输入 / 输出（具体形状）

以声学模型 `out_dims=128 mel bins`、`hidden_size=384`、`num_channels=1024` 为例：

| 张量 | 形状 | 含义 |
|---|---|---|
| `spec` (x) | `[B, 1, 128, T]` | 归一化到 $[-1,1]$ 的 Mel 谱（噪声化后的，$x_t$）；`t` 为时间帧数 |
| `diffusion_step` (t) | `[B, 1]` | DDPM：整数扩散步 $\in[0,K\_step)$；RF：$t\cdot 1000\in[0,1000]$ |
| `cond` | `[B, 384, T]` | FastSpeech2 编码器输出的逐帧条件向量 |
| **输出** | `[B, 1, 128, T]` | DDPM：预测噪声 $\hat\varepsilon$；RF：预测速度 $\hat v$ |

在完整前向里，backbone 的输入 `spec` 来自 `GaussianDiffusion.forward`：

```python
spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B,T,128] -> [B,128,T]
spec = spec[:, None, :, :]                          # [B,1,128,T]
x_recon, noise = self.p_losses(spec, t, cond=cond)  # 训练
```

最终 `diffusion` 外层再做 `denorm_spec`（$[-1,1]\to[spec\_min, spec\_max]$，如 $[-12,0]$），把 `[B,128,T]` 转回 `[B,T,128]` 的 Mel 谱交给 vocoder（NSF-HiFiGAN / pc-ddsp）合成 44.1 kHz 波形。

pitch/variance 模型里，同一 backbone 也作为 `PitchDiffusion`/`MultiVarianceDiffusion` 的骨干（`RepetitiveDiffusion`，把一维曲线复制成 `repeat_bins` 维再回归，输出取均值）。

---

## 5. 配置与推荐超参（v2.5+ 现行默认）

```yaml
# configs/acoustic.yaml / config template 默认
hidden_size: 384
backbone_type: 'lynxnet2'
backbone_args:
  num_channels: 1024
  num_layers: 6
  kernel_size: 31
  dropout_rate: 0.0
  use_conditioner_cache: true   # ONNX 条件缓存兼容
  glu_type: 'atanglu'
```

PR #200 作者按容量给了 v1 的推荐档位（`num_channels × num_layers`）：**base = 512×6（推荐）、medium-new = 1024×6（推荐）**、small/large 不推荐；`dim > 512` 时建议保留 LayerNorm 保证训练稳定。

---

## 6. 与 WaveNet 的对比 & 参考资料

- 旧默认 `wavenet`：20 层、512 通道、膨胀因果卷积（经典 WaveNet/扩散声码器风格）。LynxNet 用**大核深度可分离卷积（k=31）+ 门控线性单元（GLU）+ 每层条件/时间注入 + 后置 LayerNorm + 零初始化输出投影**，参数更省、速度更快，官方在 v2.5.0 中报告声学模型效果更好，此后成为默认。

**一手来源**：
- 代码：`openvpi/DiffSinger` → `modules/backbones/lynxnet.py`、`lynxnet2.py`、`modules/backbones/__init__.py`、`modules/toplevel.py`、`modules/core/ddpm.py`、`modules/core/reflow.py`、`configs/acoustic.yaml`
- PR：https://github.com/openvpi/DiffSinger/pull/200（"New AUX_Decoder/Backbone Network : LYNXNet"，作者 KakaruHayate）及 #218/#225/#228
- Release：v2.5.0 "New decoder backbone: LYNXNet"（2025-04-15）
- 论文：DiffSinger 原始论文 arXiv:2105.02446（不含 LynxNet）；arXiv 上暂无 LynxNet 独立论文
- 灵感来源：CNChTu/Diffusion-SVC `model_conformer_naive.py`

一句话总结：**DiffSinger 里的 LynxNet 是社区贡献的"线性门控深度可分离卷积"扩散解码器骨干，没有正式论文，只以代码发布；它接收 $([B,1,128,T],\ [B,1],\ [B,384,T])$（噪声 Mel、扩散时间步、条件特征），输出同形 Mel 域的噪声/速度预测，取代 WaveNet 成为 v2.5 起声学模型的默认生成网络。**