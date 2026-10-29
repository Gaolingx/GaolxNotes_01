# NVENC无限GOP长度使用场景与条件

## 🧑‍💻 User

nvenc什么情况下设置gopLength = NVENC_INFINITE_GOPLENGTH

---
## 😀 默认助手 (7)

## 一句话结论

`NVENC_INFINITE_GOPLENGTH`（`0xFFFFFFFF`）的含义是：不要由编码器按周期自动插入 IDR 关键帧。因此，只有在你能够自己控制关键帧出现的位置时，才应该使用它。

触发点有两个，满足任意一个即可：

1. 开启了 **intra refresh**（`NV_ENC_RC_PARAMS::enableIntraRefresh = 1`）；
2. 你通过 `NV_ENC_PIC_FLAG_FORCEIDR` / `FORCEINTRA` 手动强制刷新。

---

## 适用场景

| 场景 | 原因 |
|---|---|
| **低延迟 / 云游戏 / 云桌面 / CloudXR / GameStream** | 周期性出现 IDR 会造成码率尖峰和卡顿；改用 intra refresh 逐帧替换参考帧，即可平滑地完成错误恢复 |
| **WebRTC / RTSP 推流，关键帧按需请求** | 由接收端发送 PLI/FIR 或由业务层决定关键帧位置，编码器不应自行插入 IDR |
| **使用 intra refresh 做错误韧性（error resilience）** | SDK 要求此模式下必须将 `gopLength` 设为无限，否则自动 IDR 会破坏刷新节奏 |
| **多路合成 / 转封装 / 分段录制，GOP 边界由上层决定** | 分段点由 muxer 或录制器强制 IDR，编码器不能擅自插帧 |
| **长时间静止画面的监控、桌面串流** | 避免无意义的周期性全帧刷新，省码率 |
| **场景切换检测在上层完成** | 检测到切换时自行调用 `FORCEIDR`，比固定 GOP 更省码率 |

---

## 必须同时配置的两件事

<details>
<summary><b>1. 开启 intra refresh（推荐的低延迟方案）</b></summary>

```c
NV_ENC_CONFIG encCfg = {0};
encCfg.version = NV_ENC_CONFIG_VER;
// 先取 preset 基础配置
nvEncGetEncodePresetConfigEx(session, NV_ENC_CODEC_H264_GUID,
                             NV_ENC_PRESET_P3_GUID,
                             NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetCfg);
encCfg = presetCfg.presetCfg;

encCfg.gopLength                          = NVENC_INFINITE_GOPLENGTH;
encCfg.frameIntervalP                     = 1;   // 低延迟：不用 B 帧
encCfg.rcParams.enableIntraRefresh        = 1;
encCfg.rcParams.intraRefreshPeriod        = 60;  // 60 帧一个刷新周期
encCfg.rcParams.intraRefreshCnt           = 60;  // 周期内铺满 = 完整刷新一遍
```

`intraRefreshCnt` 小于 `intraRefreshPeriod` 时是**部分刷新**（更省码率，恢复更慢）；两者相等则是“一圈把所有 MB/CTB 都换成 intra”。

</details>

<details>
<summary><b>2. 手动强制 IDR</b></summary>

```c
NV_ENC_PIC_PARAMS picParams = {0};
picParams.version        = NV_ENC_PIC_PARAMS_VER;
picParams.encodePicFlags = needKeyframe ? NV_ENC_PIC_FLAG_FORCEIDR : 0;
```

- `NV_ENC_PIC_FLAG_FORCEIDR`：插入真正的 IDR（重置 DPB，新客户端能立刻解码）。
- `NV_ENC_PIC_FLAG_FORCEINTRA`：插入 I 帧但不是 IDR（仍可参考前帧，不重置 DPB），用于场景切换更省码率。
- 有 B 帧时，强制 IDR 会连带影响其后的重排帧序，需要在应用层保证参考链完整。

</details>

---

## 不要用的场景

- **HLS / DASH / 普通点播** —— 需要固定 GOP 让切片对齐关键帧，用无限 GOP 会导致切片无法独立解码。
- **需要随机 seek 的录制文件** —— 文件里可能长时间甚至完全没有关键帧。
- **没开 intra refresh、也不手动强制 IDR** —— 这是最典型的误用：码流里永远没有 IDR，任何中途加入的播放器会一直黑屏，解码器 DPB 会持续累积。
- **全 I 帧编码** —— 应该用 `gopLength = 1`，不是无限。

---

## 容易出错的地方

<details>
<summary><b>约束与兼容性细节</b></summary>

1. **B 帧对齐**：`frameIntervalP > 1` 时，`gopLength` 应为 `frameIntervalP` 的整数倍（SDK 里对某些 codec 会返回 `NV_ENC_ERR_INVALID_PARAM`），手动 IDR 的时间点也要落在重排边界上。
2. **preset 会覆盖**：`nvEncGetEncodePresetConfigEx` 返回的 `presetCfg` 里已经带了 `gopLength`（低延迟 preset 可能是很大的值也可能是 default），**拿完 preset 后务必显式改写**。
3. **旧版 API 的 `idrPeriod`**：SDK 2.x 的 `NV_ENC_CONFIG` 里同时有 `gopLength` 和 `idrPeriod`；3.x 之后 `idrPeriod` 被移除，IDR 只由 `gopLength` 和强制标志控制。从旧代码迁移时不要以为“不设 `idrPeriod` 就没有 IDR 了”。
4. **HEVC / AV1** 语义基本一致，但 AV1 的 S-frame / 参考帧选择能力更强，无限 GOP 的适用范围更宽。
5. **调试手段**：用 `ffprobe -show_frames` 看 `key_frame` 字段，确认关键帧真的只出现在你指定的位置；如果是“刚初始化后第一帧之外再也没有 I 帧”，就说明你漏配了 intra refresh 或强制 IDR。
6. **会话中途改 GOP**：`gopLength` 属于 `NV_ENC_CONFIG`，只能在 `nvEncInitializeEncoder` 时设定；如果运行中要改变，需要重新初始化会话，或者依靠强制 IDR 标志。

</details>
