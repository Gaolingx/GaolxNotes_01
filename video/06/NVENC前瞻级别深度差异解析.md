# NVENC前瞻级别深度差异解析

## 🧑‍💻 User

nvenc look ahead level depth区别

---
## 😀 默认助手 (7)

## 结论速览

NVENC 里这三个概念属于**三个不同维度**，而且它们有依赖关系，并不是可以互相替代的同类参数：

| 名称 | SDK 字段 | 是什么 | 取值范围 | 主要影响 |
|---|---|---|---|---|
| look-ahead（开关） | `NV_ENC_RC_PARAMS::enableLookahead` | 是否开启前瞻分析 | 0/1 | 是否启用该功能本身 |
| look-ahead **depth**（深度） | `lookaheadDepth` | 向前分析**多少帧**（分析窗口的长度） | $0 \le \text{depth} \le 31 - N_B$（B 帧越多，上限越低） | 延迟、显存（surface 数量）、码率分配的平滑程度 |
| look-ahead **level**（等级） | `lookaheadLevel`（`NV_ENC_LOOKAHEAD_LEVEL`） | 前瞻算法在画质与性能之间的**档位** | `0/1/2/3`，`15 = AUTOSELECT` | GPU 开销与画质（在同样深度下，档位越高，分析越复杂） |

一句话概括：**depth 决定“向前看多远”，level 决定“看的时候算得多精细”，而 enableLookahead 只是总开关。**

```text
enableLookahead = 1          ← 总开关（开）
  ├── lookaheadDepth = 20    ← 窗口大小：向前看 20 帧（影响延迟/内存/RC 平滑度）
  └── lookaheadLevel = 2     ← 分析强度：同一窗口内用更贵的算法（影响算力/画质）
```

---

## 逐个解释

### 1. look-ahead（开关）

`enableLookahead` 的官方注释是：*“Set this to 1 to enable lookahead with depth `<lookaheadDepth>` (if lookahead is enabled, input frames must remain available to the encoder until encode completion)”*。

要点：开启前瞻分析后，输入帧必须一直保留到编码完成，因此需要更多 surface，并且会引入延迟。在 FFmpeg 中，这个开关由 `-rc-lookahead` 的值间接控制（`>0` 时会置 1）。

### 2. look-ahead depth（深度，单位：帧）

`lookaheadDepth` 的官方注释是：*“Maximum depth of lookahead with range 0-(31 - number of B frames). lookaheadDepth is only used if enableLookahead=1.”*

- 它直接对应“向前看 N 帧”，是**单个维度上可量化的参数**，理论上最直观。
- 上限取决于 B 帧数：`max = 31 - N_B`，所以 `-bf 3` 时最多约 28。
- 在 FFmpeg 里**不直接暴露成独立选项**，而是由通用选项 `-rc-lookahead` 映射过来：

```c
/* libavcodec/nvenc.c */
ctx->encode_config.rcParams.enableLookahead = 1;
ctx->encode_config.rcParams.lookaheadDepth  = av_clip(ctx->rc_lookahead, 0, lkd_bound);
```

  其中 `lkd_bound = min(nb_surfaces, async_depth) - frameIntervalP - 4`，如果 `delay` 或 `surfaces` 太小，这个值会被**静默截短**，FFmpeg 会打出警告：

```text
Clipping lookahead depth to 12 (from 20) due to lack of surfaces/delay
Lookahead not enabled. Increase buffer delay (-delay).
```

这就是很多人“设置了 `-rc-lookahead 20` 却没生效”的真正原因：需要同时给 `-delay`/`-async_depth` 留出足够空间。

### 3. look-ahead level（等级，算法档位）

`lookaheadLevel` 的官方注释是：*“Specifies the lookahead level. Higher level may improve quality at the expense of performance.”*

- 它**不改变窗口长度**，而是让驱动在同样的 depth 下做更重或更轻的分析（更长片段级的码率分配、场景切换与 B 帧决策、与 AQ 的联合优化等），属于画质与算力的取舍档位。
- 枚举值：`LEVEL_0 = 0`、`LEVEL_1/2/3`、`AUTOSELECT = 15`（让驱动按 preset 自行决定）。
- 它是 **Video Codec SDK 12.2 才加的**（对应头文件宏 `NVENC_HAVE_LOOKAHEAD_LEVEL`），可用性通过 `NV_ENC_CAPS_SUPPORT_LOOKAHEAD_LEVEL` 查询，老驱动或老显卡可能不支持高等级。
- FFmpeg 选项是 `-lookahead_level`（`-1 ~ 15`，默认 `-1` 表示不覆盖驱动默认值）；只有 `-rc-lookahead > 0` 时它才会被写入。

---

## 实战建议

<details>
<summary><b>常用参数组合（点击展开）</b></summary>

**离线转码，追求画质（延迟不重要）**
```bash
ffmpeg -i in.mp4 -c:v hevc_nvenc -preset p7 -tune hq \
  -rc vbr -cq 24 -b:v 0 \
  -rc-lookahead 20 -lookahead_level 3 \
  -delay 32 -async_depth 8 \
  -bf 3 -multipass fullres out.mp4
```

**直播或录屏，要求低延迟**
```bash
ffmpeg -i in.mp4 -c:v h264_nvenc -preset p4 -tune ll \
  -rc cbr -b:v 6000k -maxrate 6000k \
  -rc-lookahead 0 -zerolatency 1 out.mp4
```

**折中（默认推荐）**
```bash
-rc-lookahead 10 -lookahead_level -1   # 让驱动自己选档
```
</details>

<details>
<summary><b>几条容易踩的坑</b></summary>

- **depth 和 level 不能互相替代**：`depth` 大、`level=0` 通常是“看起来远、但算得粗糙”；`depth` 小、`level=3` 则可能存在算法上限。这两个参数需要配合调整。
- **AGC（自动）行为**：`AUTOSELECT` 下驱动会按 preset（p6/p7 常自带前瞻）决定；FFmpeg 里若没显式给 `-rc-lookahead`，会用 preset 自带的 `lookaheadDepth`（源码注释：*“P6 & P7 presets may enable lookahead for better quality”*）。
- **与 `-zerolatency`、`-b_ref_mode` 冲突**：零延迟不允许有重排、有前瞻缓冲；`b_ref_mode middle/hierarchical` 也要求 `enableLookahead` 关闭或 `lookaheadLevel = 0`。
- **`multipass` 与 lookahead 不同**：multipass 是整段或多 pass 的码控增强，lookahead 是有限的向前窗口；两者可以叠加。
- **`enableExtLookahead` + `NvEncLookaheadPicture()`** 是另一套由应用（如自研编码器）驱动外置前瞻的 API，和上面三个参数不是一回事。
- 不要把 NVENC 的这三个概念与 x264 的 `--rc-lookahead` 混为一谈：x264 只有一个深度参数，没有 level 概念。
</details>

<details>
<summary><b>参数生效链路（FFmpeg → SDK，源码级）</b></summary>

| FFmpeg 选项 | FFmpeg 字段 | SDK 字段 |
|---|---|---|
| `-rc-lookahead N`（通用 AVCodecContext 选项） | `avctx->rc_lookahead` | `enableLookahead=1`，`lookaheadDepth=N`（受 surf/delay 裁剪） |
| `-lookahead_level L` | `ctx->lookahead_level`（默认 -1） | `lookaheadLevel = L`（仅 `rc_lookahead>0` 时写入） |

`-lookahead_level` 若超出能力上限，FFmpeg 会直接报 `Lookahead level not supported. Maximum level: N` 并返回 `ENOSYS`；非法值则打印 `Invalid lookahead level.`。
</details>

> 说明：`level` 各档内部具体用多少深度、开哪些工具，NVIDIA 未在头文件中公开细项，只给了“更高档位可以提升画质但更耗性能”的定义；因此不要轻信网上“level 1 = 深度 10、level 2 = 深度 20”这类说法（多为推测）。实际选档最好用自己片源实测 $\text{SSIM/BD-rate}$ 与 GPU 占用。
