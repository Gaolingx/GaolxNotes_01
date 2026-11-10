# Qwen-Image-2512与Z-Image-Turbo架构代码差异深度解析

## 🧑‍💻 User

Qwen-Image-2512 zimageturbo架构代码差异

---
## 😀 默认助手 (7)

# Qwen-Image-2512 vs Z-Image-Turbo：架构与代码差异拆解

> 结论先说：两者**同为 flow-matching DiT + 16 通道 VAE + 8× 下采样**，但**主干范式完全不同**——Qwen-Image 是 **dual-stream MMDiT（文本/图像双流）**，Z-Image 是 **single-stream S³-DiT（全部 token 拼成一个序列）**；再叠加“50 步 true-CFG”与“8 NFE 蒸馏”的区别，代码形态几乎处处不同。

---

<details>
<summary><b>0. 两个模型的定位与版本关系（点击展开）</b></summary>

| 项目 | Qwen-Image-2512 | Z-Image-Turbo |
|---|---|---|
| 发布 | 2025-12-31（Qwen 团队） | 2025-11-26（通义 MAI） |
| 规模 | DiT **20.4B** + 文本编码器 8.3B ≈ 28.7B | DiT **6.15B** + 文本编码器 4.0B ≈ 10.2B |
| 采样 | 50 步 + `true_cfg_scale=4.0`（≈100 次前向） | 9 步调度 = **8 次 DiT 前向**，`guidance_scale=0` |
| 后训练 | SFT/数据迭代（2512 = 0820 的**同架构重训版**） | Decoupled-DMD 蒸馏 + DMDR（DMD×RL） |
| pipeline 类 | `QwenImagePipeline` | `ZImagePipeline` |
| 论文 | arXiv:2508.02324 | arXiv:2511.22699 / 2511.22677 / 2511.13649 |

⚠️ 关键事实：**Qwen-Image-2512 的 transformer config 与 8 月的 Qwen-Image 完全一致**（只是少了一个没用的 `pooled_projection_dim` 键），diffusers 里加载的是同一个 `QwenImageTransformer2DModel`。**2512 的“变化”发生在权重与数据，不在架构代码**。

</details>

---

<details open>
<summary><b>1. 配置级对照（来自两边 `config.json` 实测）</b></summary>

| 维度 | Qwen-Image-2512 | Z-Image-Turbo |
|---|---|---|
| 主干类 | `QwenImageTransformer2DModel` | `ZImageTransformer2DModel` |
| 流结构 | dual-stream MMDiT | single-stream S³-DiT |
| 层数 | `num_layers=60` | `n_layers=30` + `n_refiner_layers=2`（noise/context 各 2 层） |
| 隐藏维 | `inner_dim = 24×128 = 3072` | `dim=3840` |
| 头数 / head_dim | 24 / 128 | 30 / 128（`n_kv_heads=30`，无 GQA） |
| FFN | GELU-tanh approx，4× → 12288 | **SwiGLU（w1/w2/w3）**，$8/3 \times dim = 10240$ |
| 归一化 | LayerNorm(no affine) + AdaLN 6 参数 | RMSNorm + adaLN（scale/gate，gate 过 `tanh`） |
| 时间嵌入维 | `inner_dim = 3072`（`AdaLayerNormContinuous`） | `min(dim, 256) = 256`（瓶颈，再在每层升维） |
| 位置编码 | 3 轴 RoPE `(16,56,56)`，θ=10000，`scale_rope=True` | 3 轴 RoPE `(32,48,48)`，**θ=256**，`scale_rope=False` |
| 通道 | `in_channels=64` → `out_channels=16`（2×2 pack） | `in_channels=16`（patch 2×2 在 Linear 内做） |
| 文本编码器 | `Qwen2_5_VLForConditionalGeneration`（3584-d，**`hidden_states[-1]`**） | `Qwen3Model`（2560-d，**`hidden_states[-2]`**） |
| 文本投影 | `txt_norm(RMSNorm) → txt_in(3584→3072)` | `cap_embedder = RMSNorm(2560) → Linear(2560→3840)` |
| VAE | `AutoencoderKLQwenImage`（3D causal，`z_dim=16`，per-channel `latents_mean/std`） | `AutoencoderKL`（**Flux VAE**，`scaling_factor=0.3611`，`shift_factor=0.1159`） |
| 调度器 | FlowMatchEuler，dynamic shift $0.5\to0.9$，`shift_terminal=0.02` | FlowMatchEuler，`shift=3.0`，`use_dynamic_shifting=false`（脚本另算 $\mu$） |
| 引导 | `guidance_embeds=false` + 真 CFG | `guidance_embeds` 无，纯蒸馏（CFG=0） |

</details>

---

<details open>
<summary><b>2. 数据流：序列是怎么拼的（第一处根本差异）</b></summary>

### Qwen-Image：latent 先在 pipeline 里 pack，transformer 只吃「已打包 token」
```python
# pipeline_qwenimage.py
num_channels_latents = self.transformer.config.in_channels // 4   # 64//4 = 16
latents = self._pack_latents(...)      # (B, (H/16)*(W/16), 16*4=64)
img_shapes = [[(1, height//16, width//16)]] * batch_size           # 给 RoPE 用
```
transformer 内部**不做 patchify**，直接 `img_in: Linear(64 → 3072)`；文本用 `txt_in: Linear(3584 → 3072)`。

### Z-Image：transformer 内部自己 patchify + 补 pad token + 拼序列
```python
# transformer_z_image.py::patchify_and_embed
image = image.view(C, F//pF, pF, H//pH, pH, W//pW, pW)
image = image.permute(1,3,5,2,4,6,0).reshape(N, pF*pH*pW*C)   # (N, 64)
# 文本位置从 axis0=1 开始，图像位置从 cap_len+pad+1 开始
x = self.all_x_embedder["2-1"](x)          # 64 → 3840
x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
unified = torch.cat([x[i][:x_len], cap_feats[i][:cap_len]], dim=0)   # 注意：图像在前
```

三个直接后果：
1. **输入/输出形状约定不同**：Z-Image 的 `forward` 收/发的是 `List[Tensor]`（逐样本、形状 `(16, F, H/16, W/16)`）；Qwen 收/发单个 `(B, S, 64)` 张量。
2. **padding 机制不同**：Z-Image 有可学习的 `x_pad_token` / `cap_pad_token`，序列长度补齐到 `SEQ_MULTI_OF=32`（为 `torch.compile` 静态形状服务）；Qwen 靠 `encode_prompt` 做 zero-padding + `encoder_hidden_states_mask`，在 attention 里拼成 mask。
3. **Z-Image 保留 F 轴**（`f_patch_size=1`，`unpatchify` 输出 `(C,F,H,W)`），同一套代码天然支持 Omni/编辑多图；Qwen 用 `img_shapes` 的 `frame` 维度表达“多张图/多帧”，编辑版靠 `modulate_index` 区分。

</details>

---

<details>
<summary><b>3. 文本条件：取哪一层、怎么喂（第二处差异）</b></summary>

```python
# Qwen-Image：Qwen2.5-VL-7B，取最后一层，去掉 system prompt 的 34 个 token
self.prompt_template_encode = "<|im_start|>system\nDescribe the image by ...:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
self.prompt_template_encode_start_idx = 34
hidden_states = self.text_encoder(..., output_hidden_states=True).hidden_states[-1]
split = [e[34:] for e in self._extract_masked_hidden(hidden_states, mask)]
# max_sequence_length 默认 512（transformer 端），tokenizer_max_length=1024
```

```python
# Z-Image：Qwen3-4B（纯 LLM，无视觉塔），取倒数第二层，按 mask gather 成变长列表
messages = [{"role": "user", "content": p}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
prompt_embeds = text_encoder(...).hidden_states[-2]       # ← -2，不是 -1
prompt_embeds_list = [prompt_embeds[i][prompt_masks[i]] for i in ...]   # (L_i, 2560)
```
要点：Z-Image **不保留 pad 位置**（直接 gather 变长），Qwen **保留并依赖 mask**；前者靠 `cap_pad_token` 补齐，后者靠 attention mask。这决定了两者在 batch 推理时对不等长 prompt 的处理代码完全不同。

</details>

---

<details>
<summary><b>4. 位置编码：都叫 3 轴 RoPE，实现差别不小</b></summary>

| | Qwen-Image | Z-Image |
|---|---|---|
| 轴划分 | $(16,56,56)$ | $(32,48,48)$ |
| $\theta$ | 10000 | **256** |
| 频率构造 | 预计算 ±4096 索引，`scale_rope=True`（h/w 的后半段用**负频率**，形成中心对齐） | 预计算 `axes_lens=(1536,512,512)` 的查表，直接按整数坐标索引 |
| 文本 token 位置 | 整体偏移 `max(H,W)`：`txt_freqs = pos_freqs[max_vid_index : ...]` | 占 axis0 = `1...cap_len`；图像从 `cap_len + pad + 1` 起 |
| 实现方式 | `apply_rotary_emb_qwen(..., use_real=False)`（复数乘） | `view_as_complex(x.float() ...)`（复数乘，走 float32） |
| 约束 | `head_dim = 128 = 16+56+56` | `assert head_dim == sum(axes_dims)` → 128 |

Z-Image 的“文本当第 0 轴 + 图像从 offset 开始”其实是一种**把文本/图像拍平成 1D 序列坐标**的做法；Qwen 则是把文本频率搬到大于所有图像坐标的高位区间，避免与图像空间坐标冲突。

</details>

---

<details open>
<summary><b>5. Block 内部：最核心的代码差异</b></summary>

### Qwen-Image（`QwenImageTransformerBlock`，双流，60 层全程保持两条残差流）
```python
img_mod_params = self.img_mod(temb)          # Linear(3072, 6*3072)
txt_mod_params = self.txt_mod(temb)          # 另一套，独立调制
img_mod1, img_mod2 = img_mod_params.chunk(2, -1)
...
attn_output = self.attn(hidden_states=img_modulated,
                        encoder_hidden_states=txt_modulated,
                        encoder_hidden_states_mask=...,
                        image_rotary_emb=...)
img_attn_output, txt_attn_output = attn_output   # 一次 joint attention，再拆回两条流
hidden_states        = hidden_states + img_gate1 * img_attn_output
encoder_hidden_states= encoder_hidden_states + txt_gate1 * txt_attn_output
hidden_states        = hidden_states + img_gate2 * self.img_mlp(...)
encoder_hidden_states= encoder_hidden_states + txt_gate2 * self.txt_mlp(...)
```
attention 内部：`[txt, img]` 拼接 → 联合 SDPA → 切回（`QwenDoubleStreamAttnProcessor2_0`），Q/K 归一化用 `qk_norm="rms_norm"`，文本侧走 `add_q_proj/add_k_proj/add_v_proj/to_add_out`。

### Z-Image（`ZImageTransformerBlock`，单流，所有 token 一条残差流）
```python
mod = self.adaLN_modulation(adaln_input)              # Linear(256, 4*3840)
scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh() # ← tanh 门控，Qwen 没有
scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
attn_out = self.attention(self.attention_norm1(x) * scale_msa, ...)
x = x + gate_msa * self.attention_norm2(attn_out)                      # post-norm
x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
```

| 细节 | Qwen | Z-Image |
|---|---|---|
| 条件注入 | 每层两套 `6d` 调制（img/txt），`shift/scale/gate` 无 tanh | 每层一套 `4d` 调制，**gate 过 tanh**，`1+scale` |
| 调制参数位置 | `temb` 维 = 3072，每层 `3072→18432` ×2 ⇒ **单层 113M 参数** | `temb` 维 = 256，每层 `256→15360` ⇒ 单层 3.9M 参数 |
| 归一化 | 每层 4 个 `LayerNorm(3072, affine=False)` | 每层 4 个 `RMSNorm(3840)`（**有可学习 weight**） |
| QK-norm | `Attention(qk_norm="rms_norm")` | `RMSNorm(head_dim)` 手动施加 |
| 额外结构 | — | `noise_refiner`(2, 带调制) / `context_refiner`(2, 无调制) **先各自 refine，再拼** |

> 参数量差异巨大：Qwen 有 **27% 的参数花在调制层**（$60 \times 113\text{M} \approx 6.8\text{B}$），Z-Image 的 256 维瓶颈调制只占 0.13B —— 这是“单流省参数”的真正来源之一。

</details>

---

<details>
<summary><b>6. 输出层与 forward 签名</b></summary>

```python
# Qwen：packed 空间直接回归，不 unpatchify
self.norm_out = AdaLayerNormContinuous(3072, 3072, elementwise_affine=False)
self.proj_out = nn.Linear(3072, patch_size*patch_size*out_channels)   # 3072 → 64
def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask=None,
            timestep=None, img_shapes=None, guidance=None, attention_kwargs=None,
            controlnet_block_samples=None, additional_t_cond=None, return_dict=True)
```

```python
# Z-Image：FinalLayer + 逐样本 unpatchify 回 (C,F,H,W)
self.all_final_layer = ModuleDict({"2-1": FinalLayer(3840, 2*2*1*16)})   # → 64
def forward(self, x: List[Tensor], t, cap_feats: List[Tensor],
            return_dict=True, controlnet_block_samples=None,
            siglip_feats=None, image_noise_mask=None,
            patch_size=2, f_patch_size=1, joint_attention_kwargs=None)
# Omni/编辑路径：image_noise_mask + 双 adaLN（t_noisy / t_clean）逐 token 选择
```
Z-Image 还内置 `siglip_embedder / siglip_refiner / siglip_pad_token`（为 Omni 版预留，文本用 `[cap, x, siglip]` 顺序）；Qwen 的编辑能力放在 `zero_cond_t` + `modulate_index` + `use_layer3d_rope`（`QwenEmbedLayer3DRope`，条件图用负索引 −1）。

</details>

---

<details>
<summary><b>7. 采样循环（pipeline）差异</b></summary>

| | Qwen-Image-2512 | Z-Image-Turbo |
|---|---|---|
| 步数/NFE | `num_inference_steps=50` + true-CFG ⇒ **≈100 次前向** | `num_inference_steps=9`，`t==0` 最后一步跳过 ⇒ **8 次前向** |
| 时间步传参 | `timestep / 1000`，`t_emb` 内 `scale=1000`、`flip_sin_to_cos=True` | `timestep = (1000 - t)/1000`，模型内部 `t * self.t_scale(1000)`，且**输出取负**：`noise_pred = -model_out` |
| CFG 实现 | `comb = neg + s*(pos-neg)`，再按 `‖pos‖/‖comb‖` **逐 token 范数重标定** | 蒸馏模型 CFG=0；非 turbo 版才有 CFG，并额外提供 `cfg_normalization`（范数上限）与 `cfg_truncation`（高噪声段关闭 CFG） |
| 调度 | `FlowMatchEulerDiscreteScheduler`，$\mu$ 由 `calculate_shift(seq_len, base_shift=0.5, max_shift=0.9)` 动态给出，`shift_terminal=0.02` | 同族调度器，`shift=3.0` + 脚本自算 $\mu$；`sigma_min=0.0`，latent 全程 **float32** |
| latent 反归一化 | `latents/std + mean`（per-channel） | `latents/scaling_factor + shift_factor`（Flux 式标量） |

```python
# Qwen 的 CFG 归一化（细节但很关键）
comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
noise_pred = comb_pred * (cond_norm / noise_norm)
```

</details>

---

<details>
<summary><b>8. 参数量与算力（按 config 估算，与 safetensors 元数据吻合）</b></summary>

DiT 单层参数：

- Qwen：attention $8 \times 3072^2 \approx 75.5\text{M}$ + FFN（双流）$2 \times 2 \times 3072 \times 12288 \approx 151\text{M}$ + 调制 $2 \times 3072 \times 18432 \approx 113\text{M}$ ≈ **340M/层** × 60 ⇒ **≈20.4B**（HF 元数据 20.43B ✅）
- Z-Image：attention $4 \times 3840^2 \approx 59\text{M}$ + SwiGLU $3 \times 3840 \times 10240 \approx 118\text{M}$ + adaLN $256 \times 15360 \approx 3.9\text{M}$ ≈ **181M/块** × (30 + 2 + 2) ⇒ **≈6.15B**（HF 元数据 6.155B ✅，存储为 F32）

单 token 前向开销（MAC，粗略）：

$$\text{Qwen}: 60 \times (75.5 + 151)\,\text{M} \approx 13.6\ \text{GMAC/token},\qquad \text{Z-Image}: 30 \times (59+118)\,\text{M} \approx 5.3\ \text{GMAC/token}$$

1024×1024（$64\times64=4096$ token）下一次前向：Qwen ≈ $5.6\times10^{13}$ MAC，Z-Image ≈ $2.2\times10^{13}$ MAC。再乘 NFE（100 vs 8）⇒ **整图总计算量差约 1–2 个数量级**，这正是“sub-second / 16G 显存”与“50 步大模型”的工程分野。

</details>

---

<details>
<summary><b>9. 二次开发实务提示</b></summary>

1. **迁移 LoRA 不可能跨模型**：Z-Image 的注意力量是 `to_q/to_k/to_v/to_out.0` + SwiGLU `w1/w2/w3`；Qwen 是 `attn.to_q/.../attn.add_q_proj` + `img_mlp/txt_mlp.net.{0,2}`，且 Qwen 在 diffusers 里已有显式 `_tp_plan`（TP 列/行并行）与 `_cp_plan`（context parallel），Z-Image 只有 `controlnet_block_samples` 钩子。
2. **Z-Image 的 “step” 直觉要改**：`num_inference_steps=9` 实际 8 次前向；调大步数收益很小甚至变差（蒸馏模型），而 Qwen 用到 50 步 + 负向提示才是“满血”。
3. **VAE 不通用**：两者都是 16 通道/8×，但 Z-Image 用 Flux VAE（标量 shift/scale），Qwen 用带时间维的 `AutoencoderKLQwenImage`（per-channel mean/std、5D latent），latent 数值不能互换。
4. **显存**：Z-Image Turbo 整条 pipeline bf16 约 20GB 权重，配 `enable_model_cpu_offload()` 可进 16G；Qwen-Image-2512 bf16 约 57GB，通常需要量化/offload/多卡。
5. **文本模板别写错**：Qwen 必须带那段 `Describe the image by detailing...` prompt template（起始 34 token 会被丢弃）；Z-Image 必须走 `apply_chat_template(..., enable_thinking=True)`，否则画质/遵从度会明显掉。

</details>

---

### 一句话总览

> **Qwen-Image-2512 = “双流 MMDiT + 8.3B 多模态文本编码器 + 真 CFG 50 步”的重量级质量路线；Z-Image-Turbo = “单流 S³-DiT + 4B 纯 LLM + 256 维 adaLN 瓶颈 + Decoupled-DMD/DMDR 蒸馏”的效率路线。** 代码上的每一个差异（缩到 30 层、把调制压到 256 维、把文本/图像拼进一条序列、给 pad token 并补齐到 32 的倍数、输出取负、8 NFE）几乎都能追溯到“省参数 / 保编译静态形状 / 支持少步蒸馏”这三个目标。
