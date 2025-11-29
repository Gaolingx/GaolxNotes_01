# 编写SD1.5与SDXL Prompt

### 🧑‍💻 User

对于sd1.5和sdxl，在stable diffusion webui中，如何编写prompt和negative prompt？

---

### 🤖 Assistant

下面我把 **Stable Diffusion WebUI（AUTOMATIC1111）** 中编写 **SD1.5** 与 **SDXL** 的 Prompt / Negative Prompt 的方式做一个 *实战级*、*经验型* 说明。
两者 **不是同一套 prompt 习惯**，尤其是 SDXL 的“短、高语义权重”风格与 SD1.5 的“堆标签”风格差异巨大。

---

# ✅ 总体规则：SD1.5 和 SDXL Prompt 的核心区别

| 项目              | SD1.5               | SDXL                      |
| --------------- | ------------------- | ------------------------- |
| 模型结构            | 传统 latent diffusion | 双编码器（base+refiner），语义理解更强 |
| Prompt 风格       | 关键词罗列、堆标签、类似“标签语言”  | 类自然语言、短句即可写出复杂意图          |
| Prompt 长度       | 通常 **长，堆很多关键词**     | **短、精炼、类句子结构**            |
| Negative Prompt | 很重要，需要堆长列表          | 也重要，但比 1.5 要“轻一些”         |
| 适合              | ACG、插画、LoRA 驱动强     | 写实、摄影、真实人像更强              |

---

# 🚀 一、SD1.5 Prompt 书写方式（关键词堆叠型）

SD1.5 的 prompt 更像写 **标签列表/tag list**，越具体越好。

### ✨ 正面 Prompt 模板

```
(masterpiece, best quality, 8k), 
extremely detailed, 
1girl, solo, 
long hair, blue eyes,
detailed face, soft lighting,
cinematic composition
```

### SD1.5 Prompt 技巧

* `masterpiece, best quality` 基本固定提升画质。
* 主体标签（1girl、solo）
* 外观（hair, eyes, clothes…）
* 场景（forest, room, street…）
* 光照（soft light, rim light, hdr）
* 风格（anime style, digital illustration）

“标签式”写法越多越好，SD1.5 不会觉得啰嗦。

---

### ✋ Negative Prompt（很关键）

SD1.5 的负面提示几乎必须使用 **全面的缺陷描述表**：

```
low quality, worst quality, blurry,
extra fingers, extra limbs, bad hands, bad anatomy,
text, watermark, signature,
nsfw, (if needed)
```

SD1.5 本身容易出现手部错误、文字、水印，所以这些 negative prompt 非常必要。

---

# 🚀 二、SDXL Prompt 书写方式（自然语言型）

SDXL 采用双文本编码器（CLIP-ViT-L + OpenCLIP bigG），**语义理解明显强于 SD1.5**。
因此它不需要大量标签，更倾向于自然语言描述。

### ✨ 正面 Prompt 模板

```
A beautiful woman standing near a window, soft morning light.
Ultra-detailed photography, natural skin texture, 85mm lens.
```

或更短：

```
A cinematic portrait of a woman in warm lighting, shallow depth of field.
```

### SDXL Prompt 技巧

* 类似 **摄影师会说的句子**
* 不需要 masterpiece/best quality（已经是默认高质量）
* 避免堆太多 “标签式关键词” → 反而会 **降低质量**

### SDXL 适合用摄影语言：

* 相机：`35mm/85mm lens, f1.8`
* 光圈：`shallow depth of field`
* 光照：`soft ambient light`
* 风格：`cinematic realism`

---

### ✋ SDXL Negative Prompt（比 1.5 简洁）

SDXL 的 negative prompt 不需要列到非常长，一般只处理解剖和噪声即可：

```
blurry, low resolution, bad anatomy, extra fingers, distorted hands, watermark, text
```

不建议堆太多 “风格型 negative prompt”，否则会破坏 SDXL 的语义理解。

---

# 🥊 SD1.5 vs SDXL Prompt 示例对比

### 🎨 目标：生成一个“阳光下的女孩肖像”

---

## 👉 SD1.5 写法（标签堆叠）

```
(masterpiece, best quality), 
1girl, solo, sunny day, outdoor,
long brown hair, smile, detailed eyes,
photorealistic, hdr, 8k, soft lighting,
upper body, looking at viewer,
```

#### Negative:

```
low quality, blurry, bad hands, extra fingers, bad anatomy,
text, watermark, logo
```

---

## 👉 SDXL 写法（自然语言）

```
A young woman smiling under warm sunlight, soft golden hour lighting.
Medium shot portrait, natural skin texture, cinematic look.
```

#### Negative:

```
bad anatomy, extra fingers, watermark, text
```

---

# 📌 三、Stable Diffusion WebUI 的 Prompt 格式建议

## 1. 使用权重控制

```
(masterpiece:1.2), (best quality:1.1)
(golden hour light:1.3)
```

## 2. 使用 () 提升权重，[] 降低权重

```
(golden light)
[harsh shadows]
```

---

# 🧪 四、如果要用 LoRA（特别是 SD1.5）

一般要把 LoRA 放在 prompt 最前面：

```
<lora:animeFace_v2:0.8>,
(masterpiece),
```

SDXL LoRA 比较少见，但写法类似。

---

# 📘 五、总结表

## SD1.5 Prompt 样式：

* 标签式：`1girl, blue hair, anime, ornate details`
* 长 prompt
* 强 negative prompt
* 对 LoRA 响应良好

## SDXL Prompt 样式：

* 自然语言：`A dramatic portrait of a woman in warm light`
* 简短 prompt
* 不需要 masterpiece/best quality
* 语义强，摄影语言效果好

---

# 如果你想，我可以进一步提供：

✔ 你给目标 → 我写一套 SD1.5 + SDXL Prompt
✔ 插画 / 写实 / 二次元 / 写真不同风格模板
✔ Negative prompt 专业表（按场景：人像/风景/插画）
✔ SDXL Base + Refiner 的完美 prompt 示例（含 2-stage workflow）

你想要哪一种？


---

### 🧑‍💻 User

Negative Prompt有没有特别的注意事项？

---

### 🤖 Assistant

下面给你一个 **Stable Diffusion（尤其是 SD1.5 & SDXL）Negative Prompt 的深度指南**，涵盖：

* ❗ 常见误区（大部分人都踩）
* 🚫 哪些词会产生反效果
* ⚙ 权重和长度原则
* 🧠 SDXL 与 SD1.5 的差异
* 📌 实战模板

---

# 🟥 1. Negative Prompt 的本质是什么？

**Negative prompt 不是禁止词表，而是 “优化方向的反面梯度”**。

➡ 不是黑名单
➡ 不是排除动作
➡ 是“模型不会朝这个方向强化特征”

**所以写错负面提示，会直接拉低画质。**

---

# 🟦 2. Negative Prompt 的几个致命误区（千万避免）

---

## ❌ 误区1：堆太多 negative prompt

（特别影响 SDXL）

一些人会抄一大串：

```
bad quality, worst quality, lowres, out of frame, jpeg artifacts,
bad anatomy, deformed, mutation, ugly, disfigured, watermark, ...
```

但实际上，**过多 negative prompt 会导致：**

* 画面灰暗 / 去饱和
* 细节变糊
* 表情僵硬
* 风格丢失
* SDXL 模型语义混乱

特别 SDXL，对 negative 非常敏感。

### ✔ 原则：

* **SD1.5：中等长度（20～40词）合理**
* **SDXL：尽量短（10词以内）更优**

---

## ❌ 误区2：使用贬义词（ugly、disgusting…）

很多博客推荐写：

```
ugly, disgusting, deformed face
```

但这是 **最反效果** 的东西，因为模型学习到：

> ugly = “人形、脸部、强对比、粗糙线条”

➡ 写 ugly，反而生成粗糙、线条硬、噪声重的画面。
（因为它会把“不是 ugly”理解成“远离人形特征”，不是“变漂亮”）

✔ 正确的写法应该是描述“结构错误或物理错误”，而不是美学评价：

```
bad anatomy, distorted face, asymmetrical eyes
```

---

## ❌ 误区3：在 SDXL 使用太多质量型负面提示

在 SDXL 写：

```
low quality, worst quality, lowres, jpeg artifacts
```

会破坏 SDXL 的高语义 embedding，让画面看起来：

* 色彩低饱和
* 高光丢失
* 面部自然度下降

**SDXL 不需要这些**，因为默认输出就很高质量。

✔ SDXL 推荐：

```
bad anatomy, extra fingers, watermark, text
```

---

## ❌ 误区4：权重写得太强

例如：

```
(bad hands:1.5)
```

会导致：

* 手部细节被“抹掉”
* 画质变得像涂抹式降噪
* SDXL 面部细节也会跟着消失

✔ 建议权重范围：

* SD1.5：负面权重一般 0.8～1.2
* SDXL：负面权重 ≤ 1.0，建议不写显式权重

---

## ❌ 误区5：使用“风格类 negative”（反向影响画风）

例如：

```
anime, 3d, illustration, painting, oversaturated
```

SD 会认为你想远离这些风格 → 结果画面变灰、噪声增加。

**绝不要在 negative prompt 里写风格。**

---

# 🟩 3. SD1.5 & SDXL Negative Prompt の差异

---

## SD1.5 → 负面提示非常重要

因为 SD1.5 常见问题：

* 多手指、错误解剖
* 水印、logo
* 图片有杂点、噪声
* 场景理解弱

**所以 SD1.5 使用“全面型负面提示”是合理的。**

示例：

```
bad anatomy, extra hands, extra fingers, bad proportions,
lowres, blurry, watermark, text, jpeg artifacts
```

---

## SDXL → 负面提示越少越好

SDXL 本身质量就非常高，过度 negative 会让模型“拘束性增长”，限制可表达空间。

推荐最小集：

```
bad anatomy, extra fingers, watermark, text
```

（尤其拍照风格 SDXL，负面越少越真实）

---

# 🟧 4. 如何正确写 Negative Prompt

## ☑ 原则 1：只描述“结构性错误”

例如：

**对人：**

```
bad anatomy, distorted hands, extra fingers, asymmetrical eyes
```

**对画质：**

```
blurry, noise, low contrast
```

**对干扰元素：**

```
watermark, signature, text
```

---

## ☑ 原则 2：不要写感性词、贬义词

* ugly
* disgusting
* weird
* creepy
* fat / thin（容易引发身体结构崩溃）

---

## ☑ 原则 3：不要写风格词

* anime
* 3d
* painting
* lineart
  （会导致模型退化到低细节区域）

---

## ☑ 原则 4：权重不要过强

推荐直接不用权重。

---

# 🟪 5. 实战用 Negative Prompt 模板

---

## ✔ **SD1.5 标准 Negative Prompt（通用）**

```
low quality, blurry, bad anatomy, bad proportions, 
extra fingers, extra limbs, duplicated limbs, 
distorted hands, broken fingers, missing limbs,
text, watermark, signature, logo
```

---

## ✔ **SDXL 标准 Negative（极简版本）**

```
bad anatomy, extra fingers, watermark, text
```

如果你想画得更清晰：

```
blurry, low contrast
```

---

## ✔ 插画风 Negative

```
bad anatomy, extra fingers, extra limbs, navel, nsfw,
distorted hands, wrong perspective, watermark, text
```

---

## ✔ 写实摄影 Negative

```
bad anatomy, asymmetrical eyes, extra fingers,
watermark, text, blurry, distorted face
```

---

# 🎯 6. 什么时候需要“强 Negative Prompt”？

* 画 **多人**（容易多手）
* **动作复杂**（容易骨骼扭曲）
* **LoRA 强度高**（容易改变解剖）
* **面部 close-up**（容易眼睛畸形）

---

# 如果你愿意，我可以帮你：

✔ 根据你的画风输出一套最佳 Prompt/Negative Prompt
✔ 给 SD1.5 / SDXL / AnimeXL / RealVisXL 的完整负面词表
✔ 自动生成 prompt 模板（可复制到 webui）

你想生成什么题材？我可以直接帮你写 Prompt。
