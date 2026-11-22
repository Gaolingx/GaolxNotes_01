# 人物LoRA训练仅训UNet与文本编码器过拟合

## 🧑‍💻 User

训练人物 lora 的时候要开仅训练 unet 吗，听说同时微调te容易过拟合

---
## 😀 默认助手 (7)

## 简短回答

**不必“必须只训 UNet”；但“只训 UNet”是最稳妥的默认选择，尤其在数据少的时候。**

“微调 TE 容易过拟合”这说法**有部分道理，但被过度简化了**。真正的问题往往不在于 TE 本身，而在于**TE 也用了和 UNet 一样的学习率**。TE 的参数量小、可塑性高、收敛快，几倍于 UNet 的步数就足以让它把数据集“背下来”。此时你看到的过拟合，其实来自 **TE 先过拟合，并非 UNet 过拟合**。

---

## 一、原理：TE 过拟合的机理

<details>
<summary><b>展开看具体机理（点开）</b></summary>

CLIP 文本编码器（SD1.5 的 ViT-L 约 1.23 亿参数，SDXL 双 TE 合计更大）负责把 token 转成 embedding。LoRA 挂在它的 `q/k/v/out_proj` 和 `fc1/fc2` 上后，它会**迅速把触发词和数据集里的全部视觉特征绑定**，包括背景、构图、光照、姿势习惯和画风。于是：

- 词向量的“语义空间”被挤占，其他概念的响应被削弱；
- 提示词灵活性下降，必须用“原句”才出效果；
- 风格和背景被固化进触发词，也就是常说的 concept bleeding。

而 UNet 的过拟合表现不同：**姿势重复、构图锁死、局部纹理崩坏、画面“糊”或出现网格感**。

两者的症状，你调整参数时应分别对照。

</details>

---

## 二、什么时候该开 / 不开 `network_train_unet_only`

| 情况 | 建议 |
|---|---|
| 数据集 **< 20 张**、单角色、只求“触发词出脸” | ✅ **只训 UNet**（或 TE 训前 N 步就停） |
| 数据集 **30–100+ 张**、多角度多服装 | ⚖️ **UNet + TE 都训**，TE 学习率设为 UNet 的 $\frac{1}{5}\sim\frac{1}{10}$ |
| 想要“触发词在低权重 0.4–0.6 也稳定生效” | ✅ 训 TE（绑定更强，泛化反而更好） |
| 想保留提示词自由度、做多角色混搭、风格可控 | ❌ 别训 TE，或只训前 1000 步 |
| **SDXL** | 多数人 **只训 UNet**；要训就只训 TE1、且学习率极低 |
| **Flux / SD3** | T5 那段通常冻结；按具体训练器默认走 |

> 补充：**SDXL 只训 UNet 的性价比明显更高**。SDXL 的 TE2（OpenCLIP-G）非常容易把数据集“背穿”，收益却不如把它加到 UNet 的 dim 上。

---

## 三、最优折中方案（推荐收藏）

很多人不知道 sd-scripts 提供这个参数：

```bash
# 前 800 步训 TE，之后自动冻结 TE，只继续训 UNet
--stop_text_encoder_training=800
```

这相当于**用 TE 快速建立触发词绑定，再用 UNet 慢慢学细节**，同时避开 TE 的长期过拟合。它比二选一更接近理想解，尤其适合 20–60 张数据集的角色 LoRA。

其他等价思路：
- 用 OneTrainer 时，可以给 TE 单独设置极低学习率，甚至只解冻部分层（例如只训 attention 层）；
- 使用正则化图（regularization images）配合 TE 训练，可以显著压制 concept bleeding。

---

## 四、参数模板

<details>
<summary><b>SD1.5 角色 LoRA（推荐起点）</b></summary>

```bash
# 方案 A：稳健（数据少）
--network_train_unet_only
--network_dim=64 --network_alpha=32
--learning_rate=1e-4
--text_encoder_lr=5e-5        # 此项应被忽略
--optimizer_type=AdamW8bit
--lr_scheduler=cosine_with_restarts
--unet_lr=1e-4

# 方案 B：UNet + TE（数据足）
--network_dim=64 --network_alpha=32
--unet_lr=1e-4
--text_encoder_lr=1e-5        # 关键：比 UNet 低 5~10 倍
--stop_text_encoder_training=800
```

</details>

<details>
<summary><b>SDXL 角色 LoRA（推荐起点）</b></summary>

```bash
--network_train_unet_only
--network_dim=16 --network_alpha=8     # 16~32 足够，别贪大
--unet_lr=1e-4
--optimizer_type=AdamW8bit
--lr_scheduler=constant
```

若要训 TE：`--text_encoder_lr=1e-5`，且只开 TE1，不要开 TE2。

</details>

---

## 五、怎么判断“过拟合是 TE 造成的还是 UNet 造成的”

1. **换提示词测试**：把触发词放进完全不同的场景句（`sks person, in a snowy forest, wide shot`）。
   - 场景不跟随、背景永远是训练图那几张 → **TE 过拟合**
   - 场景跟随但脸崩、姿势僵 → **UNet 过拟合**
2. **看 LoRA 权重曲线**：0.3 就能出脸且不掉画质是理想；0.5 就已经“油”了说明训过头。
3. **对比开关 TE**：同一个 LoRA，在 ComfyUI 里只加载 UNet 部分（或用 `lora` 的 block 权重过滤）看提示词灵活性是否恢复；恢复即证明 TE 是元凶。

---

## 一句话结论

> **不是“TE 一训就过拟合”，而是“TE 和 UNet 用同一个学习率才过拟合”。**
> 数据少 → `--network_train_unet_only` 最省心；
> 数据够 → 双训，但 TE 学习率压到 UNet 的 $\frac{1}{5}\sim\frac{1}{10}$，或用 `--stop_text_encoder_training` 训一段就冻结。

需要的话，把你的**模型型号、数据集张数、目标（纯角色还是含服装或画风）**告诉我，我可以直接给一套完整参数。
