# OBS NV_ENC_PARAMS_RC_CBR使用条件

## 🧑‍💻 User

obs什么情况下用NV_ENC_PARAMS_RC_CBR rateControlMode

---
## 😀 默认助手 (7)

## 结论

OBS 只在一种情况下会把 `rateControlMode` 设为 `NV_ENC_PARAMS_RC_CBR`：**`rate_control` 设置既不是 `VBR`、`CQVBR`、`CQP`，也不是 `lossless`**。也就是说，CBR 本身是兜底分支，源码注释里直接写着 `/* CBR by default */`。

<details>
<summary>代码依据：`plugins/obs-nvenc/obs-nvenc.c` 中的 `init_encoder_base()`（master 分支）</summary>

```c
bool cqvbr = astrcmpi(enc->props.rate_control, "CQVBR") == 0;
bool vbr   = cqvbr || astrcmpi(enc->props.rate_control, "VBR") == 0;
bool lossless = strcmp(enc->props.rate_control, "lossless") == 0;
bool cqp   = strcmp(enc->props.rate_control, "CQP") == 0;
...
config->rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;   /* 默认先给 VBR */
config->rcParams.averageBitRate  = bitrate * 1000;
config->rcParams.maxBitRate      = vbr ? max_bitrate * 1000 : bitrate * 1000;
config->rcParams.vbvBufferSize   = bitrate * 1000;

if (cqp || lossless) {
	config->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
	...
} else if (!vbr) {
	/* CBR by default */
	config->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
} else if (cqvbr) {
	...
}
```

细节：`VBR` 和 `CQVBR` 用 `astrcmpi`（大小写不敏感）比较，`CQP` 和 `lossless` 用 `strcmp`（大小写敏感）。默认值是 `obs_data_set_default_string(settings, "rate_control", "cbr")`（小写），走 `!vbr` 分支，落到 `NV_ENC_PARAMS_RC_CBR`。可见 H.264、HEVC、AV1 三个编码器共用这段逻辑（`init_encoder_base`），texture，soft 两套入口也一样。
</details>

## 进入 CBR 后会连带打开的设置（这才是需要关注它的原因）

| 位置 | 行为 |
|---|---|
| `h264_config->enableFillerDataInsertion = 1` / `hevc_config->enableFillerDataInsertion = 1` | 码率不够时插入 filler data，码率才是真正恒定 |
| `av1_config->enableBitstreamPadding = 1` | AV1 的等价项 |
| `outputBufferingPeriodSEI = 1`、`outputPictureTimingSEI = 1` | 输出 HRD 和 CPB 相关信息，供严格 CBR 解码端或平台使用 |
| `maxBitRate = bitrate * 1000` | 不使用 `max_bitrate`（那是 VBR 才读的） |
| `vbvBufferSize = bitrate * 1000` | 固定为 1 秒缓冲 |
| `can_change_bitrate` | 仍为 `NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE` 的结果（只有 CQP 和 lossless 会被强制 false），所以 CBR 下支持动态改码率（例如 Simple Output 中的 “Dynamically change bitrate”） |

以上各处都做了一次 `== NV_ENC_PARAMS_RC_CBR` 判断，因此“只有 CBR 才插入填充数据”是最需要知道的语义差异。

## 实际会出现 CBR 的场景

1. **推流（直播）**：Simple 输出模式推流、Advanced 输出模式默认值，都是 CBR；RTMP/HLS 平台（Twitch、YouTube、B 站等）普遍要求 CBR。
2. **Advanced 输出里手动把 Rate Control 选成 `CBR`**：常见于录像要求“码率绝对值可控”的场合（固定带宽回传、长时间录制预估体积、喂给只接受 CBR 的第三方工具、采集和转播链路）。
3. **任何未识别的 `rate_control` 字符串**：因为 CBR 是兜底分支，配置文件损坏或第三方写入非法值时也会落到 CBR。
4. **不适用的情况**：本地录像追求画质和体积比时，应选 `CQP`（恒定质量）或 `CQVBR`（质量目标加峰值码率约束）；`Lossless` 也是 `CONSTQP`。

<details>
<summary>历史情况和不使用的变体（`_HQ` 系列）</summary>

- 现在的 `nvenc.c` **完全不再使用** `NV_ENC_PARAMS_RC_CBR_HQ`、`NV_ENC_PARAMS_RC_CBR_LOWDELAY_HQ`、`NV_ENC_PARAMS_RC_VBR_HQ` 或 `NV_ENC_PARAMS_RC_2_PASS_*`。质量与延迟的调节改由两个独立字段表达：
  - `params->tuningInfo` = `HIGH_QUALITY` / `LOW_LATENCY` / `ULTRA_LOW_LATENCY` / `ULTRA_HIGH_QUALITY`（对应界面 `tune`）
  - `config->rcParams.multiPass` = `DISABLED` / `TWO_PASS_QUARTER_RESOLUTION` / `TWO_PASS_FULL_RESOLUTION`（对应界面 `multipass`，默认 `qres`）
- 旧版本（OBS 25–29 的 `jim-nvenc`，仅 Windows）里确实出现过 HQ 变体：
  ```c
  bool twopass = (astrcmpi(preset, "mq") == 0);   /* Max Quality 预设 */
  ...
  config->rcParams.rateControlMode =
      twopass ? NV_ENC_PARAMS_RC_2_PASS_QUALITY : NV_ENC_PARAMS_RC_CBR;
  ```
  `NV_ENC_PARAMS_RC_2_PASS_QUALITY` 就是 `NV_ENC_PARAMS_RC_CBR_HQ`（同为 `0x10`，`CBR2` 也等同 `CBR`，同为 `0x2`）。所以“旧版 OBS 选 Max Quality 预设并向 CBR 推流时”曾经真的传过 `CBR_HQ`；用 `ffmpeg_nvenc`（旧编码器）时，实际参数由 FFmpeg 的 `nvenc.c` 决定，不是 OBS 这段代码。
- 新版 NVENC SDK 文档把 `_HQ` 那几个枚举标为 deprecated，推荐的就是 tuning + multipass 组合，这也是 OBS 30 之后整体重写该插件的原因之一。
</details>

**一句话总结**：用户把 Rate Control 选为 CBR（或默认值 `cbr`，或推流场景）时使用 `NV_ENC_PARAMS_RC_CBR`；源码中它是 `!vbr && !cqp && !lossless` 的兜底路径，并同时开启 filler data 与 buffering-period SEI 来保证严格恒码率。
