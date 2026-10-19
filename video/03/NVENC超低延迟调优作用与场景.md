# NVENC超低延迟调优作用与场景

## 🧑‍💻 User

nvenc NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY有什么用

---
## 😀 默认助手 (7)

I'll verify the details against NVIDIA's own NVENC documentation before answering.

# `NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY`（ULL）是什么

它是 NVENC（`nvEncodeAPI.h`）里的**调优档位（tuning info）**枚举值，用来告诉编码器"按延迟最敏感的场景来配置预设"：

```c
typedef enum NV_ENC_TUNING_INFO
{
    NV_ENC_TUNING_INFO_UNDEFINED          = 0,  /**< 无效值 */
    NV_ENC_TUNING_INFO_HIGH_QUALITY       = 1,  /**< 延迟容忍型编码 */
    NV_ENC_TUNING_INFO_LOW_LATENCY        = 2,  /**< 低延迟串流 */
    NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY  = 3,  /**< 超低延迟串流 */   // ← 你问的这个
    NV_ENC_TUNING_INFO_LOSSLESS           = 4,  /**< 无损 */
    NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY = 5,  /**< Turing+ 的 HEVC/AV1 高画质 */
} NV_ENC_TUNING_INFO;
```

（注释原文：*"Tune presets for ultra low latency streaming"*。注意该字段在 H264/HEVC 的 MEOnly 模式下不适用。）

**一句话**：它不是一个独立功能，而是"**一套为最小化解码/编码排队延迟而裁剪过的编码参数组合**"。SDK 10.0 起 NVENC 把配置拆成两级——`tuningInfo`（场景档位）× `presetGUID`（P1 最快 → P7 画质最好）。同一个 `P4`，在 ULL 和 HQ 下含义完全不同。

## 它实际改变了什么

核心思路：**凡是需要"缓存未来帧"或"重排序帧"的机制，一律不用**，这样输入一帧就能尽快吐出一帧码流。

| 机制 | HQ / UHQ | LL | ULL |
|---|---|---|---|
| 前瞻 lookahead（缓存 N 帧做码率分配） | 开 | 关 | 关 |
| 时域滤波 temporal filter | 开（UHQ） | 关 | 关 |
| B 帧（需重排序，输出延迟） | P3–P7 默认开（H.264/HEVC） | 默认关 | 默认关 |
| VBV 缓冲 | 大（1–4 秒） | 小 | 极小（官方推荐 ≈ 单帧 = 码率/帧率） |
| 每帧内部多趟分析 | 可开 | 视档位 | 视档位（迁移表里 ULL 对应旧版 `CBR_HQ`，即带多趟的 CBR；这不增加帧间延迟，只增加算力） |
| 帧级输出排队 | 几帧 | 少 | 最少（≈ 1 帧流水线延迟） |

结论：**延迟最低、显存占用最省，但同码率下压缩效率/画质最差**，因为去掉了所有"预知未来"的工具。

## 什么时候用

NVIDIA 官方在《NVENC Video Encoder API Programming Guide》里给出的推荐很明确：

| 场景 | 推荐档位 |
|---|---|
| 云游戏 / 串流 / 视频会议，**信道带宽充裕、能容忍偶发大帧** | **LL** + CBR |
| 云游戏 / 串流 / 视频会议，**带宽严格受限** | **ULL** + CBR |
| 高画质点播转码、归档 | UHQ / HQ |
| 无损存档 | LOSSLESS |

典型落地场景：GeForce NOW 类云游戏、Moonlight/Sunshine、Parsec、远程桌面/VDI、VR 串流、WebRTC 会议、机器人/遥操作回传、屏幕共享。

**不要用**：录屏归档、离线转码、VOD、本地存储（这些应该用 HQ/UHQ，ULL 是纯粹的画质换延迟）。

<details>
<summary><b>怎么设置（FFmpeg / OBS / SDK 代码）</b></summary>

**FFmpeg**（`-tune` 取值：`hq` / `ll` / `ull` / `lossless` / `uhq`）：

```bash
ffmpeg -f dshow -i video="..." \
  -c:v h264_nvenc -preset p4 -tune ull \
  -rc cbr -b:v 8M -maxrate 8M -bufsize 133k \
  -g 999999 -f flv rtmp://...
```

其中 `-bufsize 133k` ≈ 单帧 VBV（$8\,\text{Mbps} \div 60\,\text{fps} \approx 133\,\text{kbit}$），`-g` 拉长以配合 intra-refresh 等抗丢包手段。

**OBS**：设置 → 输出 → 输出模式「高级」→ 编码器 NVENC (H.264/HEVC/AV1) → 预设 `P1`–`P7`，调优(Tuning) 选 **Ultra Low Latency**，码率控制 CBR。

**SDK**（先按 tuningInfo+preset 取配置，再按需微调）：

```c
NV_ENC_PRESET_CONFIG presetCfg = {0};
presetCfg.version = NV_ENC_PRESET_CONFIG_VER;
presetCfg.presetCfg.version = NV_ENC_CONFIG_VER;

NvEncGetEncodePresetConfigEx(encoder, encodeGUID,
        NV_ENC_PRESET_P4_GUID,
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        &presetCfg);

NV_ENC_INITIALIZE_PARAMS init = {0};
init.version     = NV_ENC_INITIALIZE_PARAMS_VER;
init.tuningInfo  = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
init.presetGUID  = NV_ENC_PRESET_P4_GUID;
init.encodeConfig = &presetCfg.presetCfg;   // 需要再改 rcParams / frameIntervalP 就改这里
```

</details>

<details>
<summary><b>坑与常见误区</b></summary>

- **它不是"端到端低延迟"的开关**。它只管编码器内部的排队量，采集、色彩转换、网络传输、解码器缓冲的延迟它一概不管。
- **不可能真正 0 延迟**。NVENC 流水线本身仍有约 1 帧的处理延迟（1080p60 下 ≈ 16 ms 量级）。
- **tuningInfo 是初始化参数**，会话中途改不了（`NvEncReconfigureEncoder` 只支持码率等部分参数），要改必须重建编码会话。
- **preset 与 tuningInfo 联合生效**：显式写在 `encodeConfig` 里的微调值通常会被保留，没有被显式覆盖的参数由"preset × tuningInfo"决定，因此**不要只凭 P 编号跨档位比较**。
- **码率要给足**。去掉 lookahead/B 帧后同码率画质明显下降，CBR 下还可能出现质量"呼吸"、暗部条带。通常需要比 LL 高 20%–50% 的码率才能接近观感。
- 官方注释确认了一个细节：B 帧**只在 HQ/LOSSLESS 下由 P3–P7（HEVC 为 P2–P7）默认开启**；LL/ULL 下不会默认开 B 帧。如果确实想要 B 帧的压缩收益且能接受零重排序延迟，可以考虑**单向 B 帧（unidirectional B，HEVC）**。
- 可用档位随 GPU 世代和编解码器不同，正式产品里用 `NvEncGetEncodePresetConfigEx` / `NvEncGetEncodeCaps` 探测，不要硬编码。

</details>
