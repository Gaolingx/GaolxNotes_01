# nvenc什么情况下会生成idr帧

## 🧑‍💻 User

nvenc什么情况下会生成idr帧

---
## 😀 默认助手 (7)

# NVENC 生成 IDR 帧的完整触发条件

以下内容基于 `nvEncodeAPI.h`（FFmpeg `nv-codec-headers`）中的实际注释逐条核对，并补充实践中的注意事项。

---

## 一、结论速查表

| 触发来源 | 具体条件 | 是否一定是 IDR |
|---|---|---|
| 初始化 | `nvEncInitializeEncoder()` 成功后的**第一帧** | ✅ 是 |
| GOP 周期 | `NV_ENC_CONFIG::gopLength` 到期 | ✅ 默认是（H.264/HEVC 每个 GOP 起点为 IDR） |
| IDR 周期 | `idrPeriod`（H.264/HEVC/AV1）到期 | ✅ 是 |
| 全 I 编码 | `gopLength == 1` | ✅ 每帧都是 |
| 重配置 | `NV_ENC_RECONFIGURE_PARAMS::forceIDR = 1` | ✅ 是 |
| 重配置 | `NV_ENC_RECONFIGURE_PARAMS::resetEncoder = 1` | ✅ 是（`enablePTD=1` 时强制下一帧为 IDR） |
| 显式标志 | `NV_ENC_PIC_FLAG_FORCEIDR` | ✅ 是 |
| 显式标志 | `NV_ENC_PIC_FLAG_FORCEINTRA` | ❌ 只保证 intra，**不保证 IDR** |
| 参考帧失效 | `nvEncInvalidateRefFrames()` 把参考帧全部作废 | ⚠️ 通常退化为 I/IDR |
| 分辨率/参数变更 | 会话内动态改分辨率等需内部 reset 的变更 | ⚠️ 会插入 IDR |

---

## 二、编码器自己决定 IDR（`enablePTD = 1`）

`NV_ENC_INITIALIZE_PARAMS::enablePTD = 1` 表示“图像类型由 NVENC 决定”，此时下面这些配置才生效。

### 1. 首帧
`nvEncInitializeEncoder()` 之后的第一帧输出固定为 IDR（这就是为什么你解码时总能立刻出图）。

### 2. GOP / IDR 周期

```c
// NV_ENC_CONFIG
cfg.gopLength = 60;   // 每 60 帧一个关键帧
// NVENC_INFINITE_GOPLENGTH (0xFFFFFFFF) → 不自动插关键帧

// NV_ENC_CONFIG_H264 / HEVC / AV1 里的 idrPeriod
// "Specifies the IDR interval. If not set, this is made equal to gopLength"
// 也就是 idrPeriod 默认为 gopLength
```

- `idrPeriod` 未设置时自动等于 `gopLength`，所以**每个 GOP 的起点都是 IDR**。
- `idrPeriod > gopLength` 时，中间的周期关键帧就可能退化为 non-IDR I 帧（H.264/HEVC 允许的“非 IDR I”）。
- `gopLength = NVENC_INFINITE_GOPLENGTH` 且 `idrPeriod` 也设为无穷 → **只有首帧是 IDR**，之后永远不会自动插 IDR（低延迟直播的常用配置）。

### 3. 全 I 帧
`gopLength = 1`（通常配合 `frameIntervalP = 0`）即 all-intra，每帧都是 IDR。

### 4. `nvEncReconfigureEncoder()` —— 运行时强制

```c
NV_ENC_RECONFIGURE_PARAMS rc = {0};
rc.version            = NV_ENC_RECONFIGURE_PARAMS_VER;
rc.reInitEncodeParams = initParams;   // 原初始化参数（可只改个别字段）
rc.forceIDR           = 1;            // 当前帧编成 IDR，需 enablePTD == 1
// rc.resetEncoder    = 1;            // 重置 RC 等内部状态；enablePTD=1 时下一帧也是 IDR
nvEncReconfigureEncoder(encoder, &rc);
```

> 注释原文：`resetEncoder`：*"This resets the rate control states and other internal encoder states. This should be used only with an IDR frame. If enablePTD is set to 1, encoder will force the frame type to IDR."*

这是**码率/分辨率切换、seek、加入会议**等场景最常用的 IDR 手段。

### 5. 参考帧失效
`nvEncInvalidateRefFrames()` 把某个参考帧标记为失效（比如该帧已丢失/被丢弃），后续帧不能参考它；极端情况下可用参考帧全部失效，下一帧只能 intra 编码。语义上属“隐含触发”，不如 `FORCEIDR` 可控。

---

## 三、客户端显式强制 IDR

### 1. `NV_ENC_PIC_FLAG_FORCEIDR`（推荐）

```c
pic.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
```

注释原文（很关键）：

> *"Encode the current picture as an IDR picture. **This flag is only valid when Picture type decision is taken by the Encoder [NV_ENC_INITIALIZE_PARAMS::enablePTD == 1]**."*

即：**`enablePTD = 0` 时这个标志是被忽略的**——这是非常常见的“我设置了却没生效”的原因。

### 2. `NV_ENC_PIC_FLAG_FORCEINTRA` ≠ IDR

```c
NV_ENC_PIC_FLAG_FORCEINTRA = 0x1,  // Encode the current picture as an Intra picture
NV_ENC_PIC_FLAG_FORCEIDR   = 0x2,  // Encode the current picture as an IDR picture
```

`FORCEINTRA` 只保证“intra 片”，**不保证 IDR**（不重置 POC、不带 SPS/PPS、不刷新 DPB）。这正是 FFmpeg 的 `nvenc` 封装必须提供 `forced-idr` 选项的原因：

```bash
ffmpeg -c:v h264_nvenc -g 60 -forced-idr 1 \
       -force_key_frames "expr:gte(t,n_forced*2)" ...
```

- 不加 `-forced-idr 1`：`-force_key_frames` 只会得到 **non-IDR I 帧**（切片/拼接、随机接入解码器可能报错或花屏）。
- 因此**要真 IDR，一律用 `FORCEIDR` + `forced-idr 1`**。

### 3. `enablePTD = 0` 时由客户端指定 `pictureType`
此时由客户端在 `NV_ENC_PIC_PARAMS::pictureType` 里给出 `NV_ENC_PIC_TYPE_I` 等；但这种“客户端 PTD”模式下 `FORCEIDR` 无效，实际是否 IDR 由内部逻辑判断，**不推荐**用于强 IDR 需求。

### 4. 注意：`forceIntraRefreshWithFrameCnt` 不是 IDR
`NV_ENC_PIC_PARAMS_H264/HEVC::forceIntraRefreshWithFrameCnt` 是把刷新摊到若干帧，属 intra refresh 机制，故意**避免** IDR 码率尖峰。

---

## 四、不会（或不该）生成 IDR 的情况

<details>
<summary><b>展开：intra refresh / AV1 / 无限 GOP 的细节</b></summary>

**1. 开启 intra refresh 后，周期性 IDR 被禁用**

```c
// NV_ENC_CONFIG_H264/HEVC/AV1
enableIntraRefresh  = 1;
intraRefreshPeriod  = 60;   // 刷新周期
intraRefreshCnt     = 10;   // 每次连续刷新 10 帧
// 注释: "Will be disabled if NV_ENC_CONFIG::gopLength is not set to NVENC_INFINITE_GOPLENGTH"
```

也就是说，intra refresh **要求** `gopLength = NVENC_INFINITE_GOPLENGTH` 才会生效——此时体系里基本没有 IDR，靠“逐行/逐条带刷新 + recovery point SEI”（`outputRecoveryPointSEI`）来找回随机接入点。WebRTC / 超低延迟直播常用这套。

**2. AV1 的差异**
`NV_ENC_PIC_FLAG_FORCEIDR` 在 AV1 上不受支持（AV1 叫 Key Frame），对应能力是：
- `NV_ENC_CONFIG_AV1::idrPeriod`（注释写的是 “IDR/Key frame interval”）
- `repeatSeqHdr`（每个关键帧输出 Sequence Header）
- `NV_ENC_PIC_FLAG_OUTPUT_SPSPPS` 手动要求输出头

**3. 无限 GOP**

`gopLength = NVENC_INFINITE_GOPLENGTH` 且 `frameIntervalP = 1`（注释要求无限 GOP 必须配 `frameIntervalP=1`），不显式 force 就再也不会有 IDR。

</details>

---

## 五、实践坑位清单

1. **`enablePTD = 0` → `FORCEIDR` / `forceIDR` 全部无效**，必须 `enablePTD = 1`。
2. **IDR 会造成码率尖峰**：CBR/VBV 下建议给缓冲留足余量（约 1~2 个 GOP 的 I 帧开销），或改用 intra refresh；否则弱网会连环丢包。
3. **每个 IDR 要带参数集**：H.264 用 `NV_ENC_CONFIG_H264::repeatSPSPPS = 1`，HEVC 用 `repeatSPSPPS`（VPS/SPS/PPS 随每个 IDR 输出），AV1 用 `repeatSeqHdr = 1`。否则接收端中途加入无法解码。
4. **怎么确认输出是不是 IDR**：
   - `NV_ENC_LOCK_BITSTREAM::pictureType`：`NV_ENC_PIC_TYPE_I`（非 IDR I）vs `NV_ENC_PIC_TYPE_IDR`；
   - 或直接解析码流：H.264 找 NAL type **5**（`IDR_W_RADL`），HEVC 找 NAL type **19/20**（`IDR_W_RADL` / `IDR_N_LP`），AV1 看 frame header 的 `KEY_FRAME` 标志。
5. **动态改分辨率/帧率**必须走 `nvEncReconfigureEncoder`（`maxEncodeWidth/Height` 要事先设为 0 以外的值并预分配缓冲），这类 re-init 通常自带一次 IDR。
6. **NVENC 没有 x264 那种可调的 scenecut 阈值**：公开 API 里没有暴露“场景切换自动插 I”的比例参数，关键帧来源只有上表那几条，不要指望它自己判断场景切换。
