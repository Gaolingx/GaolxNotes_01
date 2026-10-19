# Rustdesk 和 Sunshine 延迟 带宽差异

## 🧑‍💻 User

分开解释rustdesk为什么不如sunshine流畅和低延迟，但rustdesk带宽占用明显更低，明明两者都有ffmpeg integration？

---
## 😀 默认助手 (7)

I'll verify a couple of load-bearing technical claims first (whether Sunshine actually encodes via FFmpeg, and RustDesk's current capture/transport path), then give a full breakdown.

我查了 Sunshine 的官方配置文档（全 1030 行，逐条读完）和 RustDesk 的 Advanced Settings 全量条目（128 个设置，程序化检索 + 逐条核对），下面把两件事拆开讲。

## 先纠正那个前提：两者并没有"都靠 FFmpeg 做同一件事"

这正是你困惑的根源——`ffmpeg` 只是 `libavcodec`，它既不决定延迟，也不决定码率。

| | Sunshine | RustDesk |
|---|---|---|
| 编码实现 | **NVENC 走 NVIDIA Video Codec SDK 直连**（所以才有 `nvenc_split_encode`、`nvenc_latency_over_power`、`nvenc_realtime_hags` 这类 SDK 级参数）；AMF/QSV/VAAPI/VideoToolbox/软件才走 FFmpeg 封装 | 软件编码走 **libvpx (VP8/VP9)、libaom (AV1)** 的原始 API；硬编 H.264/H.265 走自封的 hwcodec 层 |
| 默认软件编码器 | x264（`sw_preset=superfast`, `sw_tune=zerolatency`），仅作回退 | **VP9 / AV1 的 realtime 模式**（构建依赖就是 `libvpx` + `aom`，没有 x264） |
| 默认编码目标 | 硬编、恒定目标码率、P1 最快档 | 软编 realtime VP9/AV1，或硬编 H.264/H.265 |

用不用 FFmpeg、用哪个 FFmpeg 封装，和端到端延迟几乎无关。真正决定延迟的是**流水线的每一段**。下面分开说。

---

## A. 为什么 RustDesk 不如 Sunshine 流畅、延迟高

把一帧画面从"游戏渲染出来"到"你的客户端屏幕亮起"拆成 7 段，逐段对比：

### 1. 目标帧率：默认 30 vs 60/120（最直观的一条）

RustDesk 的 `custom-fps` **默认 30**（范围 5–120）。也就是出厂状态下它根本不打算给你 60fps。

- $30\,\text{fps}$ → 每帧间隔 $33.3\,\text{ms}$
- $60\,\text{fps}$ → 每帧间隔 $16.7\,\text{ms}$

Sunshine 则按 Moonlight 客户端的请求跑 60/120fps，并用 `minimum_fps_target` 保证"即使画面静止也维持帧节奏"。**帧间隔本身就差了一倍**，这是"不流畅"的第一来源，跟编码器毫无关系。

### 2. 采集路径：GPU 显存直采 vs 回读到 CPU

- Sunshine：Windows 上是 **DXGI Desktop Duplication**，NVIDIA 还能用 **NvFBC**（文档原话："capture direct to GPU memory. This is usually the fastest method for NVIDIA cards"）。帧全程留在显存里，直接喂给编码器。
- RustDesk：`enable-directx-capture` 默认 `Y`，确实也用 DXGI/GDI 抓屏——**但抓下来的 BGRA 要落到 CPU 可访问缓冲**，再做色彩转换。多一次 GPU→CPU 回读，就是几毫秒起步。

### 3. 色彩转换：libyuv 在 CPU 上转

RustDesk 需要把 BGRA 转成编码器要的 NV12/YUV420（libyuv）。1080p 一次 swscale 级转换几毫秒；Sunshine 硬编路径基本不需要这一步。另外 RustDesk 的 `i444`（真色彩 4:4:4）默认关闭。

### 4. 编码器本身：realtime VP9/AV1 软编是重头

即使 `enable-hwcodec` 现在默认 `Y`，RustDesk 的 `codec-preference` 默认 `auto`，文档还特意警告 **"Options other than vp8 and vp9 may not work"**——即默认落点经常是 VP8/VP9 软件编码。realtime VP9 1080p 单帧在 CPU 上就可能 10–30ms，而且这个延迟是**串在每一帧的路径上**的。

对比 Sunshine 的 NVENC：默认 `nvenc_preset=1`（P1，最快档）、`nvenc_latency_over_power=enabled`（强制高功耗模式，"禁用会导致编码延迟显著增加"）、4K 还能 `nvenc_split_encode` 多编码单元切帧。硬件编码 1080p 是 2–5ms 级别。

### 5. 帧节奏（pacing）：主动补帧 vs 按需出帧

Sunshine 的 `minimum_fps_target` 说明写得很直白：客户端期望恒定帧流，所以静止时它**主动发重复帧**来维持节奏。

RustDesk 是"有变化/有带宽才出帧"，帧间隔不规则——这就是肉眼看到的 **judder（顿挫）**，比"整体慢"更难受。

### 6. 传输层：TCP 队头阻塞 vs UDP + 每帧 FEC（延迟差距最大的一段）

- Sunshine：**UDP** + **每帧块级 FEC**（`fec_percentage` 默认 20，意思就是每个视频帧里塞 20% 的纠错包）。丢一个包，用同一帧的 FEC 包就地补回，**不重传、不阻塞后续帧**。
- RustDesk：默认走 **TCP**（直连 21118 / 中继 21117），`disable-udp` 默认 `N`，但 UDP 21116 主要只用于打洞/打洞发现。TCP 一旦丢包，**重传 + 队头阻塞**会让后面所有帧一起等——这是"一卡就卡一大下"的直接原因。`enable-udp-punch` / `enable-webrtc` 是较新的可选路径（且自建服务器默认关闭），但 WebRTC 数据通道默认是可靠有序的，HOL 问题未必消失。

### 7. 客户端渲染：默认关掉快路径

RustDesk 的 `allow-d3d-render` **默认 `N`**（文档："D3D render can get high FPS and reduce the cpu usage, but…"），渲染走 Flutter 纹理上传 + 多层合成。Moonlight 是原生硬解 + 直接呈现 + vsync 对齐，缓冲深度压到 1 帧。

### 端到端延迟预算（局域网，1080p60 量级，示意）

| 环节 | Sunshine/Moonlight | RustDesk（默认） |
|---|---|---|
| 采集 | $1\text{–}3$ ms | $5\text{–}15$ ms |
| 色彩/内存往返 | ~0 | $3\text{–}10$ ms |
| 编码 | $2\text{–}5$ ms（NVENC P1） | $10\text{–}50$ ms（软编 VP9/AV1） |
| 传输/抗丢包 | $1\text{–}5$ ms（UDP+FEC） | $10\text{–}100+$ ms（TCP 重传，波动大） |
| 客户端解码+呈现 | $2\text{–}8$ ms | $10\text{–}25$ ms |
| **合计** | **约 $10\text{–}25$ ms** | **约 $40\text{–}150$ ms** |

---

## B. 为什么 RustDesk 带宽明显更低

这条路同样是"设计选择"，而且是**主动**的：

| 机制 | 具体体现 | 对带宽的影响 |
|---|---|---|
| **帧率低** | `custom-fps=30` 默认 | 运动画面码率需求**直接砍半** |
| **画质档位低** | `image-quality` 默认 `balanced`；`custom-image-quality` 默认 `50`（量程上限 2000） | 贴着"能看"走 |
| **自适应码率默认开** | `enable-abr=Y`（**默认开启**） | 网络一紧就主动降码率/质量 |
| **静止画面几乎不发数据** | 按变化驱动编码 | 办公场景平均带宽极低 |
| **没有 FEC 冗余** | Sunshine 默认 +20% 纠错包 | 省掉 20% 以上的"白花"带宽 |
| **不做码流平滑** | Sunshine `nvenc_vbv_increase=0`：单帧 VBV，每帧大小 $\le \text{bitrate}/\text{fps}$，主动用填充把突发磨平 | 省掉填充开销 |
| **码率是"够用即止"** | CBR + 固定目标码率 + 用户可设到 20–150 Mbps 且**不会自动降** | 恒定高位 |
| **TCP 拥塞控制** | 慢启动、丢包回退 | 天然压低吞吐，UDP CBR 能瞬间跑满 |
| **其他省流开关** | `i444` 默认关、Android `enable-android-software-encoding-half-scale=Y`（软编直接降半分辨率） | 又省一截 |

**注意最后两条的因果：RustDesk 之所以带宽低，恰恰因为它会降帧率、降画质、降分辨率、跟着网络抖——这些正是"不流畅"的来源。**

---

## C. 统一解释：这不是"两个独立现象"，而是同一个取舍的两面

存在一个三难选择：**画质 / 带宽 / 延迟**，三者不可能同时最优。两个软件站的位置相反：

<details>
<summary><b>为什么"低带宽"往往反而伴随"高延迟"（点开看机理）</b></summary>

关键在于**突发性**，而不只是平均速率。

1. **CBR + 填充 + FEC 是"花钱买确定性"**。Sunshine 每帧不超过 $\text{bitrate}/\text{fps}$，必要时发重复帧、发纠错包、发填充。这在你看任务管理器时是"浪费带宽"，但它保证码流平滑、无突发。
2. **突发 VBR 会在路由/AP 排队**。低平均带宽的脉冲流量，瞬时峰值照样把缓存填满，产生 **bufferbloat**，延迟抖动反而更大。省下的带宽换来了排队延迟。
3. **降低帧率/画质换带宽，会同时抬高延迟**。`enable-abr` 降码率 → 编码器/发送队列被"憋住" → 下一帧更晚出来 → 反馈环振荡。
4. **TCP 的可靠性语义与实时性冲突**。它保证"都送到"，代价是"都可能晚到"；Sunshine 选择"宁缺毋滥"——丢帧、用 FEC 补，绝不为了等一帧而卡住后面。

一句话：**Sunshine 花带宽买时间，RustDesk 省带宽赔时间。**

</details>

---

## D. 怎么验证 / 怎么调

<details>
<summary><b>想让 RustDesk 更接近 Sunshine（点开）</b></summary>

- `custom-fps=60`（或 120）——单项收益最大
- `image-quality=best`，或把 `custom-image-quality` 往上调（比如 200+）
- `enable-abr=N`——关掉自动降码率，止住画质/帧率波动
- `codec-preference=h264`（或 `h265`）——强制走硬编
- 确保直连：`direct-server=Y`、`enable-udp-punch=Y`、`enable-webrtc=Y`
- `allow-d3d-render=Y`、`use-texture-render=Y`
- 用 RustDesk 的 `show-quality-monitor`（`show-quality-monitor=Y`）看实际 fps/延迟/码率

</details>

<details>
<summary><b>反过来，想让 Sunshine 省带宽（点开）</b></summary>

- 降 Moonlight 请求码率，或设 `max_bitrate`（单位 Kbps）
- 编码器选 H.264、关 HDR / 关 4:4:4、降分辨率
- 降 `fec_percentage`（默认 20 → 更低），代价是抗丢包能力下降
- 注意 `minimum_fps_target` 会主动发重复帧，这是它"费带宽"的一部分，但也正是它流畅的原因

</details>

**测量提醒**：任务管理器看的是平均带宽，容易误导。Sunshine 是**恒定流**（你设定的码率），RustDesk 是**脉冲流**（静止时接近 0，动起来才涨）。要公平比较，请在**同一段动态游戏画面**下同时看平均码率、帧率和端到端延迟。

---

**结论**：两者"都有 FFmpeg"并不构成可比性。RustDesk 窄的带宽，是用「默认 30fps + 低画质档 + 默认开 ABR + 无 FEC + 无码流平滑 + TCP 传输」换来的；而同一套选择正好导致了更高延迟和顿挫。Sunshine 则「默认硬编 P1 + 恒定目标码率 + 每帧 FEC + 主动补重复帧 + UDP + 极浅缓冲」，用带宽把延迟买了下来。这不是实现水平的偶然差距，而是**远程桌面**和**游戏串流**两个目标不同的产品，在"画质/带宽/延迟"三角上选了不同的角。

---
## 🧑‍💻 User

sunshine的 NVIDIA Video Codec SDK是如何集成的，对比用ffmpeg编码有什么好处？

---
## 😀 默认助手 (7)

我把 Sunshine 源码里 `src/nvenc/` 的全部 15 个文件、`src/video.cpp`（133 KB，逐行相关段落）、`cmake/compile_definitions/common.cmake`，以及 FFmpeg master 的 `libavcodec/nvenc.c` / `nvenc_h264.c` / `nvenc_hevc.c` / `nvenc_av1.c` 都读了一遍。结论比想象的更有意思：**Sunshine 在 Windows 上完全不用 FFmpeg 编码 NVENC，但在 Linux 上又退回去用 FFmpeg** —— 等于项目自带了一组 A/B 对照。

## 0. 先看这个关键事实

`src/video.cpp` 里 `nvenc` 这个 encoder 被定义了**两次**：

```cpp
#ifdef _WIN32
  encoder_t nvenc {
    "nvenc"sv,
    std::make_unique<encoder_platform_formats_nvenc>(   // ← 自研 NVENC SDK 路径
      platf::mem_type_e::dxgi, platf::pix_fmt_e::nv12, ...),
    { {}, {}, {}, {}, {}, "av1_nvenc"s, {} },            // ← 所有选项字典都是空的
    { {}, {}, {}, {}, {}, "hevc_nvenc"s, {} },
    { {}, {}, {}, {}, {}, "h264_nvenc"s, {} },
    PARALLEL_ENCODING | REF_FRAMES_INVALIDATION | YUV444_SUPPORT | ASYNC_TEARDOWN
  };
#elif !defined(__APPLE__)
  encoder_t nvenc {
    "nvenc"sv,
    std::make_unique<encoder_platform_formats_avcodec>(  // ← FFmpeg 的 libavcodec 封装
      AV_HWDEVICE_TYPE_CUDA, AV_HWDEVICE_TYPE_NONE, AV_PIX_FMT_CUDA, ...),
    { { {"delay",0}, {"forced-idr",1}, {"zerolatency",1}, {"surfaces",1},
        {"cbr_padding",false}, {"preset",...}, {"tune", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY},
        {"rc", NV_ENC_PARAMS_RC_CBR}, {"multipass",...}, {"spatial-aq",...} }, ... },
    PARALLEL_ENCODING | YUV444_SUPPORT                   // ← 注意：缺了 RFI 和 ASYNC_TEARDOWN
  };
#endif
```

Windows 走 `src/nvenc/*`（NVENC SDK 直连），Linux/FreeBSD 走 FFmpeg 的 `h264_nvenc`/`hevc_nvenc`/`av1_nvenc`。下面主要讲 Windows 那条路。

---

## 1. SDK 是怎么集成的（五步倒推）

### ① 头文件来自 FFmpeg 的 `nv-codec-headers`

```cpp
// src/video.cpp:23        (Linux 侧)
#include <ffnvcodec/nvEncodeAPI.h>
// src/nvenc/nvenc_sdk.h
#include <ffnvcodec/dynlink_cuda.h>
#include <ffnvcodec/nvEncodeAPI.h>
#if NVENCAPI_MAJOR_VERSION * 100 + NVENCAPI_MINOR_VERSION != NVENC_SDK_VERSION
  #error NVENC_SDK_VERSION does not match the selected nv-codec-headers package
#endif
```

它用的不是 NVIDIA 官方 Video Codec SDK 压缩包，而是 FFmpeg 项目维护的 **`nv-codec-headers`（即 `ffnvcodec`）**——只含头文件、纯 MIT/开源、无 NDA 内容。

### ② 一次性编译 4 份实现，运行时挑最新

```cmake
# cmake/compile_definitions/common.cmake
function(add_nvenc_sdk_implementation target_name sdk_version sdk_include_dir)
    add_library(${target_name} OBJECT ${NVENC_IMPLEMENTATION_SOURCES})
    target_include_directories(${target_name} BEFORE PRIVATE "${sdk_include_dir}")
    target_compile_definitions(${target_name} PRIVATE
            NVENC_FACTORY_SUFFIX=${sdk_version}
            NVENC_NAMESPACE=nvenc_${sdk_version}
            NVENC_SDK_VERSION=${sdk_version})
endfunction()

add_nvenc_sdk_implementation(nvenc_sdk_1100 1100 "${NV_CODEC_HEADERS_11_INCLUDE_DIR}")
add_nvenc_sdk_implementation(nvenc_sdk_1200 1200 "${NV_CODEC_HEADERS_12_INCLUDE_DIR}")
add_nvenc_sdk_implementation(nvenc_sdk_1300 1300 "${NV_CODEC_HEADERS_13_INCLUDE_DIR}")
add_nvenc_sdk_implementation(nvenc_sdk_1301 1301 "${NV_CODEC_HEADERS_13_1_INCLUDE_DIR}")
```

同样的 6 个 `.cpp` 被编译 **4 次**，每次塞进不同的头文件目录和命名空间（`nvenc_1100` … `nvenc_1301`）。一个二进制里装了 4 套 NVENC API 绑定。

### ③ 运行时动态加载 + 版本协商

```cpp
constexpr auto nvenc_dll_name = "nvEncodeAPI64.dll";   // a64 / 64 / 32 三种
constexpr auto minimum_driver_version = "456.71";

auto dll = make_shared_dll(LoadLibraryEx(nvenc_dll_name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
auto get_max_version = /* GetProcAddress */ "NvEncodeAPIGetMaxSupportedVersion";
get_max_version(&packed_max_version);
const auto sdk_version = select_nvenc_sdk_version(decode_nvenc_driver_version(packed_max_version));
// >=13.1→sdk_13_1; >=13.0→sdk_13_0; >=12.0→sdk_12_0; >=11.0→sdk_11_0; else unsupported
```

`decode_nvenc_driver_version` 就是 $(version \gg 4) \times 100 + (version \mathbin{\&} 15)$。注意：**只按名字取 2 个符号**（`NvEncodeAPIGetMaxSupportedVersion`、`NvEncodeAPICreateInstance`），其余上百个入口点全部来自 `NvEncodeAPICreateInstance` 回填的 `NV_ENCODE_API_FUNCTION_LIST` 函数指针表 —— 这正是能在一个进程里同时支持 4 个 API 版本的原因，也意味着**构建时不需要链接 `nvEncodeAPI.lib`，也不需要 NVIDIA SDK 或 EULA**。

### ④ 三层抽象 + 两种输入表面后端

```cpp
nvenc_encoder            // SDK 无关接口 (nvenc/nvenc_encoder.h)
   └─ nvenc_base         // 平台无关：开 session、配 rate control、encode 循环
        └─ nvenc_d3d11   // D3D11 共通：init_library() / wait_for_async_event()
             ├─ nvenc_d3d11_native    // NV_ENC_DEVICE_TYPE_DIRECTX，直接注册 D3D11 纹理
             └─ nvenc_d3d11_on_cuda   // 从 D3D11 设备派生 CUDA 设备，cuMemcpy2D 拷到 CUDA 面
```

native 路径直接 `CreateTexture2D(D3D11_BIND_RENDER_TARGET)` 再 `nvEncRegisterResource(resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX)` —— **零拷贝**。CUDA interop 只在需要 10-bit 4:4:4 时才走（源码里写得很明确：`YUV444_10BIT` 无法用 D3D11 表面格式表达，必须 CUDA）。

### ⑤ 编码循环（含异步）

```cpp
nvEncMapInputResource(...)                    // 映射已注册纹理
nvEncEncodePicture(pic_params)                // pic_params.completionEvent = async_event_handle
                                              //   encodePicFlags = force_idr ? NV_ENC_PIC_FLAG_FORCEIDR : 0
if (async_event_handle && !wait_for_async_event(100)) return {};   // WaitForSingleObject
nvEncLockBitstream(lock_bitstream)            // doNotWait = 1（因为前面已经等过事件）
encoded_frame = { {ptr, ptr+size}, outputTimeStamp, pictureType == NV_ENC_PIC_TYPE_IDR,
                  rfi_needs_confirmation }
nvEncUnlockBitstream(...)
```

---

## 2. 它到底把哪些参数设成了什么

这是"低延迟"真正的来源，全部来自 `nvenc_base.cpp` 的 `configure_rate_control` / `create_encoder`：

```cpp
init_params.tuningInfo   = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;  // 关键
init_params.presetGUID   = quality_preset_guid_from_number(config.quality_preset); // P1..P7
init_params.enablePTD    = 1;
init_params.enableEncodeAsync = async_event_handle ? 1 : 0;
configure_split_frame(...);   // SDK>=13: NV_ENC_SPLIT_AUTO_MODE / FORCE / DISABLE

enc_config.gopLength                    = NVENC_INFINITE_GOPLENGTH;  // 不做周期性 IDR
enc_config.frameIntervalP               = 1;                         // 没有 B 帧
enc_config.rcParams.rateControlMode     = NV_ENC_PARAMS_RC_CBR;
enc_config.rcParams.zeroReorderDelay    = 1;                         // 不重排序
enc_config.rcParams.enableLookahead     = 0;                         // 没有前视
enc_config.rcParams.lowDelayKeyFrameScale = 1;
enc_config.rcParams.averageBitRate      = client_config.bitrate * 1000;
enc_config.rcParams.vbvBufferSize       = client_config.bitrate * 1000 / client_config.framerate;
//                        ↑ 单帧 VBV：每帧预算 = 目标码率 ÷ 帧率
```

H.264/HEVC/AV1 还各自有：`repeatSPSPPS = 1`、`idrPeriod = NVENC_INFINITE_GOPLENGTH`、`sliceMode = 3; sliceModeData = slicesPerFrame`（一帧切多 slice，便于分片打包且利于并行解码）、`chromaFormatIDC = 3`（4:4:4）、VUI 色彩直写、`enableIntraRefresh`（周期 300 / 计数 299 / `singleSliceIntraRefresh`）。参考帧默认 H.264/HEVC 5 个、AV1 8 个，但 `numRefL0 = 1`。

---

## 3. 对比 FFmpeg 封装：真实的收益清单

我把 FFmpeg 的选项表和它的初始化代码都做了核对，**先说结论：FFmpeg 的 NVENC 封装比多数人想的强得多**，`preset(p1-p7)`、`tune(hq/ll/ull)`、`rc`、`multipass`、`cbr_padding`、`delay`、`zerolatency`、`surfaces`、`rc-lookahead`、`bf`、`ldkfs`、`intra-refresh`、`single-slice-intra-refresh`、`split_encode_mode`、`spatial-aq`、`weighted_pred`、`dpb_size`、`max_slice_size` 都有。所以差距不在"能不能调"，而在**下面这几件 FFmpeg 结构上做不到或做得更绕的事**。

| 能力 | NVENC SDK 直连（Windows） | FFmpeg 封装 | 差异性质 |
|---|---|---|---|
| **参考帧失效 (RFI)** | `nvEncInvalidateRefFrames()` 逐帧失效；`REF_FRAMES_INVALIDATION` 标志 | **源码里 `nvEncInvalidateRefFrames` 出现 0 次**，只能 `forced-idr` 发整帧 I 帧 | **硬缺口** |
| **异步编码** | `nvEncRegisterAsyncEvent` + `pic_params.completionEvent` + `WaitForSingleObject` | `enableEncodeAsync` **写死为 0**，用 `surfaces=N` 表面池 + FIFO 代替 | **机制不同** |
| **VBV 窗口** | 显式 $=\text{bitrate}/\text{fps}$（60fps 下约 $16.7\,\text{ms}$） | 未设 `rc_buffer_size` 时默认 $= 2 \times \text{averageBitRate}$（即**约 2 秒**） | **默认差 ~120 倍** |
| **能力探测** | 10 项逐项 `nvEncGetEncodeCaps`，开流前就能拒绝非法组合 | 全文只调用 1 次，错误要等 `avcodec_open2` 才暴露 | 工程性 |
| **SDK 版本** | 一个二进制内置 4 套，按驱动挑最新（≥456.71 起） | 编译期绑死一个封装版本，新 API 要等 FFmpeg 跟进 + 用户升级 | 部署性 |
| **参数集 (SPS/VPS/VUI)** | 直接在 NVENC 配置里写 VUI，无二次处理 | 需 Sunshine 自己用 `cbs::validate_sps` + `inject`/`replacements` **替换 NAL 单元** | 复杂度 |
| **输入表面** | `nvEncRegisterResource` 直接挂 D3D11 纹理 | `AVHWFramesContext` + `AV_PIX_FMT_D3D11` 表面池 + refcount | 拷贝/层级 |
| **产物元数据** | `outputTimeStamp` / `pictureType` / `after_ref_frame_invalidation` 原样可读 | 需从 `AVPacket` 侧补齐 | 打包器需求 |
| **CUDA 依赖** | native 路径完全不碰 `nvcuda` | Linux 路径走 `AV_HWDEVICE_TYPE_CUDA` | 环境 |

### 其中最重要的是前两条

**RFI（参考帧失效）** 的价值在上一个问题里已经体现：Sunshine 的传输层是"每帧块级 FEC"，丢包恢复靠 FEC；当 FEC 不够、必须让解码端重建参考关系时，FFmpeg 路径只能发一个 IDR —— 一帧关键帧可能几百 KB，瞬间在网络里形成尖峰，直接推高排队延迟和抖动。SDK 路径可以只让受影响的帧区间失效，代价小一个数量级。这也是为什么 Windows 的 `nvenc` flags 有 `REF_FRAMES_INVALIDATION`，而 Linux 的没有。

**单帧 VBV** 则是"码流平滑"的来源。$\text{vbvBufferSize} = \text{bitrate}/\text{fps}$ 意味着每帧大小被强行约束在一帧的预算内，输出码流近乎恒定 —— 交换条件是编码器在难帧上必须牺牲质量。FFmpeg 的默认是 2 秒缓冲，等于允许长时间的平均化，瞬时突发不受约束。

> 有趣的细节：Sunshine 在自己的 FFmpeg 路径上 **手动补上了这一课**：
> ```cpp
> // src/video.cpp:2232
> if (!(encoder.flags & NO_RC_BUF_LIMIT)) {
>     ctx->rc_buffer_size = bitrate / config.framerate;   // 复制单帧 VBV 语义
> }
> ```
> 也就是说 FFmpeg 路径"可以"追上，但需要使用者自己知道要这么做。

---

## 4. 代价与局限

<details>
<summary><b>点开看不那么好看的一面</b></summary>

1. **只有 Windows**。`cmake` 里 `NVENC_IMPLEMENTATION_SOURCES` 被 `if(WIN32)` 包着 —— Linux/FreeBSD 完全没有这套代码，只能吃 FFmpeg 封装。所以"Sunshine 低延迟"这个结论在 Linux 上要打折扣（没有 RFI、没有 async teardown、没有 split-encode 的 SDK 直控）。
2. **维护成本高**：4 套 SDK 实现 × 2 种表面后端 = 8 条组合，还要跟驱动版本矩阵（`minimum_driver_version = 456.71`）纠缠。
3. **10-bit 4:4:4 必须走 CUDA interop**，多一次 `cuMemcpy2D` 和 CUDA context push/pop。
4. **社区命名也承认这是两条路**：FFmpeg 路径的配置项被放在 `config::video.nv_legacy.*`（`nv_legacy.preset`、`multipass`、`spatial_aq`、`h264_coder`、`vbv_percentage_increase`）—— "legacy" 这个词很说明设计意图。
5. 反过来，FFmpeg 的优势是**跨厂商统一**：AMF / QSV / VAAPI / VideoToolbox / Vulkan / 软件编码在 `src/video.cpp` 里全是 `encoder_platform_formats_avcodec` 一套代码，还顺手拿到了 FFmpeg 的选项解析、动态重配（`reInitEncodeParams`）、以及别人替你做好的驱动兼容矩阵。

</details>

---

## 5. 一句话总结

Sunshine 集成 NVENC SDK 的方式是：**用 FFmpeg 的 `nv-codec-headers` 做头文件、把同一份实现按 11.0/12.0/13.0/13.1 编译 4 遍、运行时从 `nvEncodeAPI64.dll` 动态协商版本、然后自己手写 rate control 把每一个延迟来源关掉（`tune=ULL`、单帧 VBV、无 B 帧、无前视、无重排序、无限 GOP）**。

相对 FFmpeg 封装的收益不在于"能设置超低延迟"（FFmpeg 也有 `-tune ull`），而在于三件 FFmpeg 结构性缺的东西：**① 参考帧失效（RFI）——FFmpeg 根本没有这个调用；② NVENC 原生异步事件模型——FFmpeg 把 `enableEncodeAsync` 写死为 0；③ 单帧 VBV 作为默认值——FFmpeg 默认给 2 秒缓冲。** 再加上运行时多 SDK 选择、逐能力探测、以及省掉 SPS/VPS 二次重写。

这也正好回答了上一个问题的一半：**Sunshine 的"低延迟"不是"因为用了 NVENC"，而是因为绕开通用转码框架之后，把单帧 VBV 和 RFI 这两件对网络时延影响最大的事做到了 FFmpeg 封装做不到的粒度。**

---
## 🧑‍💻 User

我最不明白的是为什么在编码质量差不多的情况下，同样的windows+nvenc环境全屏播放视频，sunshine的带宽是远大于rustdesk的（即使sunshine用p3 present也需要高得多的带宽才能达到相同的编码质量），是编码器/编码参数配置差异导致的吗

---
## 😀 默认助手 (7)

Good question — and it's the one that actually pins down the "encoder" red herring. Let me check RustDesk's actual NVENC parameters so I'm not hand-waving.

读完 RustDesk 的 `hwcodec`（`cpp/common/util.cpp`、`cpp/ffmpeg_ram/ffmpeg_ram_encode.cpp`、`libs/scrap/src/common/hwcodec.rs`、`src/server/video_qos.rs`）之后，答案很明确了 —— **不是编码器，也不主要是 preset。主因是两边喂给同一个 NVENC 的"目标码率"本身差了 10–20 倍；次因是 Sunshine 的配置在"每比特能买到多少画质"上明显更差，大约 2–4 倍。两者相乘就是你看到的现象。**

而且这里有一个绝妙的条件：**两边都用 CBR**。所以"CBR vs VBR"的解释在这题上不成立——是配置细节的差异，可以逐条对齐。

## 1. 把两边的 NVENC 配置完全摊开

| | Sunshine（Windows，NVENC SDK 直连） | RustDesk（`hwcodec`，FFmpeg `*_nvenc`） |
|---|---|---|
| rate control | `NV_ENC_PARAMS_RC_CBR` | `av_opt_set("rc","cbr")` |
| **目标码率** | `averageBitRate = client_config.bitrate × 1000`，即 **Moonlight 滑杆**（`max_bitrate` 默认 0 = 完全听客户端的） | `base_bitrate(W,H) × ratio × factor`，**1080p 默认约 2.8 Mbps** |
| **VBV 缓冲** | $vbvBufferSize = \dfrac{bitrate}{framerate}$ → **1 帧** | **从未设置** → FFmpeg 默认 $\dfrac{bitrate}{framerate}$ 不用，而是 $vbvBufferSize = 2 \times bitrate$ → **2 秒** |
| **tuning** | `NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY` | **从未设置** → FFmpeg 默认 `tune = hq`（`NV_ENC_TUNING_INFO_HIGH_QUALITY`） |
| B 帧 | `frameIntervalP = 1`，无 | `has_b_frames = 0; max_b_frames = 0`，无 |
| GOP | `NVENC_INFINITE_GOPLENGTH` | `gop_size = INT_MAX`，`keyint_min = INT_MAX` |
| preset | `quality_preset` 默认 **P1** | `p1`(Low) / `p4`(Medium) / FFmpeg 默认(High) |
| 帧率 | 60（客户端请求）+ 重复帧补足 | **`FPS = 30`**，且变化驱动 |
| FEC | `fec_percentage = 20` | 无 |
| 前视 | `enableLookahead = 0` | FFmpeg nvenc 硬编码 `enableLookahead = 0` |

RustDesk 那段关键代码：

```c
// rustdesk/hwcodec  cpp/common/util.cpp
void set_av_codec_ctx(AVCodecContext *c, const std::string &name, int kbs, int gop, int fps) {
  c->has_b_frames = 0;
  c->max_b_frames = 0;
  c->gop_size   = std::numeric_limits<int>::max();
  c->keyint_min = std::numeric_limits<int>::max();
  if (kbs > 0) c->bit_rate = kbs * 1000;
  c->flags |= AV_CODEC_FLAG2_LOCAL_HEADER;
  c->flags |= AV_CODEC_FLAG_LOW_DELAY;
  c->slices = 1;
  // ← rc_buffer_size 从未设置
}
// set_lantency_free(): nvenc 只设了 delay=0；tune 从未设置
// set_rate_control(): nvenc → rc=cbr
```

对比 FFmpeg 自己的默认值（`libavcodec/nvenc.c`）：

```c
ctx->init_encode_params.enableEncodeAsync = 0;          // 硬编码
ctx->encode_config.rcParams.zeroReorderDelay = 1;       // 硬编码
ctx->encode_config.rcParams.enableLookahead  = 0;
...
if (avctx->rc_buffer_size > 0)
    ctx->encode_config.rcParams.vbvBufferSize = avctx->rc_buffer_size;
else if (ctx->encode_config.rcParams.averageBitRate > 0)
    avctx->rc_buffer_size = ctx->encode_config.rcParams.vbvBufferSize
                          = 2 * ctx->encode_config.rcParams.averageBitRate;   // ← 2 秒
```

**RustDesk 不是"选了"2 秒 VBV，它只是没覆盖 FFmpeg 的默认值。** 而 Sunshine 是显式把它设成 1 帧。

---

## 2. 主因：目标码率不是一个量级

RustDesk 的码率公式（`libs/scrap/src/common/codec.rs` + `hwcodec.rs`）：

```
base   = base_bitrate(W, H) × ratio
factor = H.264 : base > 2000 ? 1 + 1/(1 + (base-2000)/1000) : 2.0
                 HEVC/AV1/VP9 : base > 2000 ? 1 + 0.5/(1 + (base-2000)/1000) : 1.5
bitrate = base × factor                                // kbps
```

而 `ratio` 的来源：

| 设置 | ratio |
|---|---|
| `image-quality = low` | `BR_SPEED` = 0.5 |
| `image-quality = balanced`（**默认**） | `BR_BALANCED` = **0.67** |
| `image-quality = best` | `BR_BEST` = 1.5 |
| `image-quality = custom` | $\dfrac{\text{custom-image-quality} \times 2}{100}$，默认值 50 → **1.0** |

`base_bitrate(1920,1080) = 2073` kbps（表内值，其余分辨率按像素数线性缩放）。

代入 1920×1080、H.264：

| 档位 | base | factor | **码率** |
|---|---|---|---|
| balanced（默认） | $2073 \times 0.67 = 1389$ | 2.0 | **2.78 Mbps** |
| custom 50 | $2073$ | 1.93 | **4.0 Mbps** |
| best | $3110$ | 1.47 | **4.6 Mbps** |
| custom 1180 | $\approx 48900$ | 1.02 | **≈ 50 Mbps** |

**RustDesk 在 1080p 默认只打算用 2.8 Mbps。** 你 Moonlight 滑杆大概在 20–150 Mbps。同一颗 NVENC、同一段视频，输入 2.8 和 50，差 $18\times$ —— 这已经解释了绝大部分。

Sunshine 侧的 `max_bitrate` 默认 0，文档原话：*"If set to 0, it will always use the bitrate requested by Moonlight."*

> 所以准确的说法不是"Sunshine 为了同画质需要更高码率"，而是"**Sunshine 被要求产出 50 Mbps，RustDesk 被要求产出 2.8 Mbps**"。

---

## 3. 次因：Sunshine 的配置让"每比特画质"差 2–4 倍

这部分才是你直觉上想问的"编码参数差异"。

### ① VBV 差 120 倍（最重要）

- Sunshine：$\text{vbvBufferSize} = \dfrac{\text{bitrate}}{\text{fps}}$。1080p60 @ 50 Mbps → 每帧上限 $\dfrac{50\,\text{Mbit}}{60} = 833\,\text{kbit} \approx 104\,\text{KB}$
- RustDesk：$2 \times \text{bitrate} = 100\,\text{Mbit}$，即 1080p60 下每帧上限 $1.67\,\text{Mbit} \approx 208\,\text{KB}$，但可以在 120 帧的时间窗内自由挪借

单帧 VBV 的代价是：**难帧（场景切换、快速运动、暗部细节）不能向时间轴上的邻居借比特，只能在这一帧内把 QP 拉上去**，于是出现块效应/糊。要在这些难画面上追平 2 秒 VBV 的观感，只能整体加码率。对于有场景切换的影视内容，这一条的代价通常是 **2–4 倍平均码率**。

Sunshine 自己也知道这是损失 —— `nvenc_vbv_increase` 的文档写着 "Relaxing this restriction can be beneficial and act as **low-latency variable bitrate**"，默认 0，最大可到 400（即允许 5 倍单帧上限）。

### ② tuning：`ull` vs `hq`

Sunshine 显式 `init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY`。RustDesk 从没设过 `tune`，于是拿到 FFmpeg 的默认 `hq`。NVENC 的 ULL 调优会关掉一批提高压缩率但引入延迟的东西（时域滤波/多帧统计等）。这是一个 **BD-rate 量级 15–30%** 的差距，方向和延迟相反。

### ③ 60fps vs 30fps

- RustDesk：`pub const FPS: u32 = 30`
- Sunshine：按客户端请求 60/120，并用 `minimum_fps_target` 补重复帧（文档：*"a certain amount of duplicate frames are sent when this happens"*）

24fps 的电影内容在 60fps 流里，一半以上是重复/近似重复帧。重复帧在 H.264/HEVC 里是零残差 P 帧，很便宜但不免费（slice 头 + 残差 + 打包头）。更关键的是**码率预算被摊薄**：同样是 50 Mbps，60fps 每帧预算只有 30fps 的一半。Sunshine 还要为这多出来的一倍帧数付账。

### ④ FEC +20%

`fec_percentage = 20`：每帧额外 20% 纠错包。50 Mbps 流里约 $50 \times \frac{20}{120} \approx 8.3$ Mbps **对画质零贡献**。RustDesk 没有这一项。

### ⑤ preset 排第五（这就是为什么你试 P3 没用）

Sunshine 文档明确：*"Higher numbers improve compression (quality at given bitrate) **at the cost of increased encoding latency**."* P1→P3 的 BD-rate 收益通常在 10–25% 量级，而 `ull` tuning 会压制一部分预设收益。**你用 preset 去追一个 10–20 倍的目标码率差距，方向对但杠杆太短** —— 这正好印证了主因不在熵编码/运动估计，而在码率分配策略。

---

## 4. 为什么"同画质"这个观察是对的，但结论要反过来

把上面的账算一下 Sunshine 的"有效码率利用率"：

$$\underbrace{2\text{–}4\times}_{\text{单帧 VBV}} \times \underbrace{1.15\text{–}1.3\times}_{\text{ull vs hq}} \times \underbrace{\text{约}2\times}_{\text{60 vs 30fps 的帧数}} \times \underbrace{1.2\times}_{\text{FEC}} \approx 6\text{–}12\times$$

也就是说：**Sunshine 里 50 Mbps 的"CBR 预算"，真正转化成画质的可能只相当于 RustDesk 的 5–8 Mbps 的 VBR 效率。** 再叠加 18 倍的目标码率差 ——

- **带宽差距**：$\sim 18\times$（你观察到的）
- **画质差距**：远小于 $18\times$（你观察到的"差不多"）

这两个观察同时成立，不矛盾。**Sunshine 高出的带宽，很大一部分没有被转换成画质，而是变成了 VBV 约束下的 QP 波动、重复帧和 FEC 冗余包。** 这是它买"恒定延迟"付的钱。

---

## 5. 一个能直接证伪或证实上面分析的实验

<details>
<summary><b>点开看具体做法</b></summary>

**做法 A：只改 RustDesk 一个参数，看码率是否跳到 Sunshine 量级。**

在 1080p + H.264 硬编路径下，把 `custom-image-quality` 从默认 50 提到 **约 1180**。按公式 $\text{ratio} = \frac{1180 \times 2}{100} = 23.6$：

$$\text{base} = 2073 \times 23.6 = 48900\,\text{kbps},\quad \text{factor} = 1 + \frac{1}{1 + 47.9} = 1.021$$
$$\text{bitrate} \approx 49900\,\text{kbps} \approx 50\,\text{Mbps}$$

如果带宽**没有**跳到 ~50 Mbps，说明我的公式理解有误（请告诉我，我再查）。如果它**跳到了**，那么"带宽差距的主因是目标码率"就被证实了 —— 而画质应该也同步提升到接近 Sunshine。

**做法 B：把 Sunshine 的码率降到 RustDesk 的水平。**

把 Moonlight 码率设成 3–5 Mbps，两端都跑 1080p30，然后对比观感。预期：Sunshine 在这个码率下**明显不如** RustDesk，因为它的 VBV 只有 1 帧。

**做法 C：直接验证单帧 VBV 假说。**

把 Sunshine 的 `nvenc_vbv_increase` 从 0 提到 300（允许 4 倍单帧上限），其他不变，码率也固定不变，看画质是否明显改善。这是最干净的单变量实验。

**做法 D：剥离 FEC 和帧率。**

- `fec_percentage = 5`（局域网不需要 20%）
- 客户端请求 30fps（对齐 24fps 片源）
- 再对比同码率同帧率下的画质

</details>

---

## 6. 回到设计意图

这三个软件的配置不是"调得好不好"，而是目标函数不同：

| 目标 | 代价 |
|---|---|
| Sunshine：**恒定延迟 + 恒定码率** | 单帧 VBV、ULL 调优、60fps 补帧、20% FEC、不补就不用完 CBR 预算 |
| RustDesk：**最小带宽 + 自适应** | 默认 30fps、低目标码率、ABR 随时往下调、2 秒 VBV（无意中） |

RustDesk 那个 2 秒 VBV 尤其值得注意 —— **它不是设计选择，只是 FFmpeg 的默认值没人去覆盖**。如果 RustDesk 哪天把 `rc_buffer_size` 也设成单帧，它的码率需求会立刻上升、延迟会更好，但带宽优势就消失了。反过来说，**Sunshine 那 50 Mbps 里有相当一部分，是被一个"为了低延迟而牺牲压缩效率"的参数故意燒掉的**。

---
## 🧑‍💻 User

sunshine有哪些参数是webui没有暴露但是可以改config文件实现的？（比如nvenc_vbv_increase、tuningInfo等）

---
## 😀 默认助手 (7)

我把三份清单做了脚本化的三向 diff：`src/config.cpp` 里所有配置键（99 个）、Web UI 的 `config_tabs.json`（99 个）、官方 `docs/configuration.md`（99 个）。**结果是差集为空。**

所以先纠正你的两个例子——它们恰好都不成立。

## 0. 两个例子的实际情况

| 你举的例子 | 实际情况 |
|---|---|
| `nvenc_vbv_increase` | ✅ **Web UI 里就有**。在 "NVIDIA NVENC Encoder" 页签，是 `<input type="number" min="0" max="400">`，默认 0。它是在 **v0.22.0 进入 config.cpp** 的；如果你的界面里看不到，要么版本早于 v0.22.0，要么它在页面下方那个折叠的 "Miscellaneous" 区域里 |
| `tuningInfo` | ❌ **它根本不是配置项**，config 文件写它无效。源码里写死： |

```cpp
// src/nvenc/nvenc_base.cpp  create_encoder()
init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;   // ← 硬编码

// src/video.cpp  （Linux 走 FFmpeg 时同样写死）
{"tune"s, NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY},
```

**结论：Sunshine 的"文档 / config 文件 / Web UI"三者现在是严格同步的 99 项，没有藏起来的配置项。** 你在 Web UI 搜索框里搜不到的键，在 config 文件里写了也不会生效（搜索框的候选列表就是 `config_tabs.json`，也就是唯一合法的键集合）。

真正值得关心的是**反方向的问题**：哪些参数连 config 文件都改不了，只能打补丁。下面这份清单才是你想要的。

---

## 1. NVENC：`nvenc_config` 里有 4 个字段没有任何配置键

`src/nvenc/nvenc_config.h` 的结构体共 11 个字段，config.cpp 只映射了 6 个：

| `nvenc_config` 字段 | 映射的配置键 |
|---|---|
| `quality_preset` | `nvenc_preset` |
| `two_pass` | `nvenc_twopass` |
| `vbv_percentage_increase` | `nvenc_vbv_increase` |
| `adaptive_quantization` | `nvenc_spatial_aq` |
| `h264_cavlc` | `nvenc_h264_cavlc` |
| `split_frame_encoding` | `nvenc_split_encode` |
| `weighted_prediction` | ❌ **无** |
| `enable_min_qp` | ❌ **无** |
| `min_qp_h264` / `min_qp_hevc` / `min_qp_av1` | ❌ **无** |
| `insert_filler_data` | ❌ **无** |

我 grep 了 `config.cpp`：`weighted`、`min_qp`、`filler` 的出现次数**都是 0**。这四个字段只能用源码里的默认值（`false` / `false` / `19/23/23` / `false`）。

> 顺带一个和上一轮直接相关的点：`insert_filler_data` 的注释是 *"Add filler data to encoded frames to stay at target bitrate, mainly for testing"* —— 它默认关闭，而且**没有任何配置键**，所以如果你想让 Sunshine 严格贴满 CBR 目标码率，得改源码。

## 2. `nvenc_base.cpp` 里彻底硬编码的一批（改 config 完全无效）

```cpp
// configure_rate_control()
enc_config.gopLength                      = NVENC_INFINITE_GOPLENGTH;  // 无限 GOP
enc_config.frameIntervalP                 = 1;      // 无 B 帧
enc_config.rcParams.rateControlMode       = NV_ENC_PARAMS_RC_CBR;      // 恒定 CBR
enc_config.rcParams.zeroReorderDelay      = 1;
enc_config.rcParams.enableLookahead       = 0;
enc_config.rcParams.lowDelayKeyFrameScale = 1;
enc_config.rcParams.vbvBufferSize         = client_config.bitrate * 1000 / client_config.framerate;
                       // ↑ 单帧 VBV，只能用 nvenc_vbv_increase 按百分比放宽，不能改公式

// configure_h264() / configure_hevc() / configure_av1()
format_config.repeatSPSPPS  = 1;
format_config.idrPeriod     = NVENC_INFINITE_GOPLENGTH;
format_config.sliceMode     = 3;
format_config.sliceModeData = client_config.slicesPerFrame;   // ← 来自 Moonlight 协商
configure_reference_frames(..., default_count = 5 /*H.264/HEVC*/ 或 8 /*AV1*/ , client_config.numRefFrames, ...);
                       // ↑ numRefL0 硬编码为 NV_ENC_NUM_REF_FRAMES_1
format_config.intraRefreshPeriod = 300;   // 硬编码
format_config.intraRefreshCnt    = 299;   // 硬编码

// create_encoder()
init_params.enablePTD = 1;
configure_split_frame(...);   // SDK>=13 时的 NV_ENC_SPLIT_AUTO_MODE
```

再补一条容易被忽略的：`client_config.slicesPerFrame`、`client_config.numRefFrames`、`client_config.enableIntraRefresh` 这三个是**由 Moonlight 客户端在 RTSP 握手时协商过来的**，Sunshine 侧的 config 和 Web UI 都碰不到，只能改客户端或改 `src/video.cpp` 里填 `config_t` 的地方。

---

## 3. `nvenc_vbv_increase` 的能力边界（它不是什么都能干）

它只做一件事：

```cpp
enc_config.rcParams.vbvBufferSize = client_config.bitrate * 1000 / client_config.framerate;
if (config.vbv_percentage_increase > 0) {
    enc_config.rcParams.vbvBufferSize += enc_config.rcParams.vbvBufferSize * pct / 100;
}
```

| `nvenc_vbv_increase` | 实际 VBV 窗口（1080p60 为例） |
|---|---|
| 0（默认） | $\frac{\text{bitrate}}{60}$ → **1 帧 ≈ 16.7 ms** |
| 100 | 2 帧 |
| 400（上限） | 5 帧 ≈ 83 ms |

想复刻 FFmpeg 那套 2 秒缓冲，需要 $pct \approx 11900$，**超出上限**。所以你只能"部分放宽"，不能改成另一种 rate control 模型——要那样只能改源码。

---

## 4. 那"config 能改但 UI 看不到"的窗口期存在过吗？

我按 tag 扫了一遍 `config.cpp`：

| 版本 | 新出现的 NVENC 相关键 |
|---|---|
| v0.19.1 / v0.20.0 | （都没有） |
| v0.21.0 | `nvenc_realtime_hags` |
| **v0.22.0** | `nvenc_vbv_increase`、`nvenc_latency_over_power` |
| 更晚（≈ v2025.122 之后） | `nvenc_split_encode`、`vk_quality`、`fp16_sdr_gamma_encoded`、`external_audio`、`amd_max_au_size`、`vaapi_strict_rc_buffer` |

注意方向：**新参数是先进 `config.cpp`，然后同一版通常也会进 Web UI**。所以"config 文件能设、UI 故意不给你"的窗口在 Sunshine 上基本不存在。老版本的情况反而是相反的——你从新版文档抄一个键写进老版本的 `sunshine.conf`，**它会被解析器直接忽略**。

---

## 5. 确实只有手写才能做的三件事（不是隐藏键，是能力差异）

<details>
<summary><b>点开看</b></summary>

1. **复合类型的自由度**。UI 是表单，手写是 JSON：
   - `global_prep_cmd = [{"do":"nircmd.exe setdisplay 1280 720 32 144","elevated":true,"undo":"..."}]`
   - `dd_mode_remapping = { "mixed": [...], "resolution_only": [...], "refresh_rate_only": [...] }`
   - `keybindings = [0x10, 0xA0, 0x11, 0xA2]`
   
   这些键 UI 里都有输入控件，但能力和手写不总是一一对应（比如 UI 的 `nvenc_preset` 是 1–7 下拉，config 里写 `nvenc_preset = 7` 等价，但 UI 的搜索框不会告诉你默认值是 1）。

2. **`apps.json` 是另一套 schema，不在这 99 项里**。它由 `src/process.cpp` / 应用解析代码消费。UI（`Apps.vue`）覆盖了 `prep-cmd`、`detached`、`auto-detach`、`wait-all`、`exit-timeout`、`working-dir`、`image-path`、`exclude-global-prep-cmd`、`elevated`、`output` 等，但像 `virtual-sink`、`playnite-*`、`scale-factor`、`use-app-identity`、`per-client-app-identity` 这类在表单里找不到对应的直接输入框，得手写。

3. **配置文件路径本身**：`sunshine ~/my_own.conf` 可以指定任意路径，Web UI 只认默认位置。

</details>

---

## 6. 如果你真想调 `tuningInfo` 这类硬编码项

最小补丁路径（项目是 GPL-3.0，得自己编译）：

| 想改的东西 | 改哪里 |
|---|---|
| `tuningInfo` / `tune` | `src/nvenc/nvenc_base.cpp` 搜 `init_params.tuningInfo`；Linux 路径在 `src/video.cpp` 的 nvenc 选项字典里搜 `"tune"` |
| VBV 公式、`frameIntervalP`、`zeroReorderDelay`、`enableLookahead` | `src/nvenc/nvenc_base.cpp` → `configure_rate_control()` |
| `weighted_prediction` / `enable_min_qp` / `min_qp_*` / `insert_filler_data` | 已在 `src/nvenc/nvenc_config.h` 里，只缺 config 映射。要暴露成配置项改两处：`src/config.cpp` 的 `nv_*` 映射块（大约在 1630 行附近，`video.nv_legacy.* = video.nv.*` 那一段）+ `config_tabs.json` 的 `nv` 页签 |
| `intraRefreshPeriod/Cnt`、参考帧默认值、`sliceMode` | `src/nvenc/nvenc_base.cpp` → `configure_h264_hevc_metadata()` / `configure_reference_frames()` |

---

## 一句话总结

在 master 上，`sunshine.conf` 与 Web UI 是**同一套 99 个键**，没有"藏在配置文件里的后门"——你举的 `nvenc_vbv_increase` UI 里有（≥ v0.22.0），`tuningInfo` 则连 config 都改不了。真正只能靠打补丁的是另一批东西：`nvenc_config` 里 4 个无映射字段（`weighted_prediction`、`enable_min_qp`、`min_qp_*`、`insert_filler_data`），以及 `nvenc_base.cpp` 里硬编码的 `tuningInfo`、单帧 VBV 公式、无 B 帧、无限 GOP、`intraRefresh` 300/299 等——**而这几项恰好都是上一轮我们讨论的那几个"用压缩效率换延迟"的开关**。这不是巧合：它们被硬编码，正是因为它们是产品定位的一部分，而不是留给用户调的旋钮。

---
## 🧑‍💻 User

 sunshine为什么在固定码率情况下，降低FEC Percentage会适当改善画质，但也会换来略高的静置带宽消耗（表现为静止状态带宽更高，运动画面更容易吃满预设码率）？

---
## 😀 默认助手 (7)

这个问题需要看 packetization 的实际实现，我去读一下 Sunshine 的 FEC 代码。

**找到了。** 机制在 `src/rtsp.cpp` 里——FEC 百分比会直接改写编码器的目标码率。

找到了。答案不在 packetizer，而在 `src/rtsp.cpp` 里的一段码率修正——**`fec_percentage` 会直接改写编码器的目标码率**。所以你调的不是"冗余开销"，而是"分给画面的预算"。

## 1. 关键代码

```cpp
// src/rtsp.cpp —— 处理 RTSP SETUP 时的码率协商
// If the client sent a configured bitrate, we will choose the actual bitrate ourselves
// by using FEC percentage and audio quality settings.
if (configuredBitrateKbps) {
  // If the FEC percentage isn't too high, adjust the configured bitrate to ensure video
  // traffic doesn't exceed the user's selected bitrate when the FEC shards are included.
  if (config::stream.fec_percentage <= 80) {
    configuredBitrateKbps /= 100.f / (100 - config::stream.fec_percentage);
  }
  auto audioBitrateAdjustment = (HIGH_QUALITY ? 256 : 96) * config.audio.channels;
  configuredBitrateKbps -= std::min((std::int64_t) audioBitrateAdjustment, configuredBitrateKbps / 5);
  configuredBitrateKbps -= std::min((std::int64_t) 500, configuredBitrateKbps / 10);

  config.monitor.bitrate = (int) configuredBitrateKbps;   // ← 这个值最终喂给编码器
}
```

也就是：

$$\text{编码器目标码率} \approx \text{configured} \times \frac{100 - \text{fec}\%}{100} \;-\; \text{音频} \;-\; 500\,\text{kbps}$$

代入 50 Mbps + 2 声道 HQ 音频（$256 \times 2 = 512$ kbps）：

| `fec_percentage` | 系数 | × 之后 | 减音频 | 减 500 | **编码器目标** |
|---|---|---|---|---|---|
| 20（默认） | 0.80 | 40 000 | 39 488 | 38 988 | **≈ 39.0 Mbps** |
| 10 | 0.90 | 45 000 | 44 488 | 43 988 | ≈ 44.0 Mbps |
| 5 | 0.95 | 47 500 | 46 988 | 46 488 | **≈ 46.5 Mbps** |
| 1 | 0.99 | 49 500 | 48 988 | 48 488 | ≈ 48.5 Mbps |

**20% → 5% 等于把编码码率抬高约 19%**，而不是"少发 15% 的纠错包"。这就是"适当改善画质"的全部来源。

> 两个细节：① 用 $\times(100-\text{fec})/100$ 是 $\div(1+\text{fec}/100)$ 的线性近似，所以 Sunshine 其实略微**多预留**了一点（fec=20 时预留 20% 而实际只需 16.7%）。② `fec_percentage <= 80` 是门限，超过 80 就不做这个折算了，所以 81–255 是"只加开销、不还码率"。

---

## 2. 为什么"降 FEC"在静止画面上什么都省不到

`src/stream.cpp` 的 `fec::encode()`：

```cpp
auto data_shards   = aligned_data_shards + (pad ? 1 : 0);
auto parity_shards = (data_shards * fecpercentage + 99) / 100;   // ← 向上取整
if (parity_shards < minparityshards && fecpercentage != 0) { ... }
```

是 **ceil**，于是节省量是非线性的：

| `data_shards` | fec=20% 的奇偶包 | fec=5% 的奇偶包 | 节省 |
|---|---|---|---|
| 1 | $\lceil 0.20 \rceil = 1$ | $\lceil 0.05 \rceil = 1$ | **0** |
| 4 | $\lceil 0.80 \rceil = 1$ | $\lceil 0.20 \rceil = 1$ | **0** |
| 5 | $\lceil 1.00 \rceil = 1$ | $\lceil 0.25 \rceil = 1$ | **0** |
| 10 | 2 | 1 | 1 |
| 100 | 20 | 5 | 15 |

**阈值是 $data\_shards \ge \dfrac{100}{\text{fec}\%}$**：fec=20 时是 5 个数据分片，fec=5 时是 20 个。

而一个静止画面的重复帧 payload 通常只有几百字节到 1 KB，`data_shards = 1` —— **永远在阈值以下，一个 FEC 包都省不掉。**

---

## 3. 两个现象的解释

### ① 静止状态带宽更高

静止画面的网络字节 ≈ $2 \times \text{payload}$（1 个数据分片 + 1 个奇偶分片 ≈ 100% 实际开销，远高于名义的 20%），**而且这个倍数对 fec=20 和 fec=5 完全一样**。

而 payload 由 CBR 目标决定：39.0 → 46.5 Mbps（+19%）。FEC 侧零抵补。

**净结果：静置带宽上升约 19%。**

叠加效应：`minimum_fps_target` 会让 Sunshine 在静止时继续按帧节奏发重复帧（文档原话："a certain amount of duplicate frames are sent"）。CBR 会把这些帧预算烧掉——通过压低 QP 保留更多噪声/抖动/渐变细节。所以编码码率一抬，静置的"地板"就整体抬高。

### ② 运动画面更容易吃满预设码率

运动帧的 `data_shards` 有几百个，远超阈值，FEC 开销接近名义值 `fec%`。所以 20% → 5% 确实省下了 ~12.5 个百分点。

但同时编码器目标从 39.0 → 46.5 Mbps。两者大致相抵，**总网络 ≈ 你设的码率**；变化的是**成分**：payload 更多、纠错包更少。于是你观察到"运动画面更容易吃满预设码率"——因为编码器现在被允许（并被要求）去占满更高的 payload 预算。

---

## 4. 一句话总结

$$\underbrace{\text{省下的 FEC 开销}}_{\text{只在 } data\_shards \ge 100/\text{fec}\% \text{ 时出现}} \quad\text{vs}\quad \underbrace{\text{码率上浮 } \frac{\text{fec}_{old}-\text{fec}_{new}}{100}}_{\text{无条件生效}}$$

**降 FEC 的本质是把预算从"纠错冗余"挪到"画面数据"。在静止画面（低 `data_shards`）上这是一笔纯亏的交换：开销一分没省，码率白抬 19%；只有在运动画面（高 `data_shards`）上才真的赚到。** 这正好解释了为什么你看到的是"静置带宽变高、运动更容易跑满"这种不对称的表现。

---

## 5. 三个需要注意的点

<details>
<summary><b>点开看</b></summary>

1. **`fec_percentage` 实际上是一个隐蔽的画质旋钮。** 它同时改变冗余开销和编码码率，以维持"总网络 ≈ 预设码率"。所以"降 FEC 提升画质"不是错觉，是源码里明晃晃的算术。代价是抗丢包能力下降——而且降到的码率是给编码器的，丢包时没有多余冗余去补。

2. **只在客户端发 `x-ml-video.configuredBitrateKbps` 时生效。** 如果走的是传统 GameStream 通道（`x-nv-vqos[0].bw.maximumBitrateKbps`），rtsp.cpp 走另一条分支，只减 500 kbps（上限 10%），**不做 FEC 折算**。那种情况下改 `fec_percentage` 就只有纯开销变化，不会有码率上浮——表现会和你现在看到的完全不同。

3. `x-nv-vqos[0].fec.minRequiredFecPackets` 默认是 `"0"`（客户端不发时），所以上面的 ceil 分析成立。如果客户端要求了最小值（比如 2），静止画面的开销会更夸张（1 数据 + 2 奇偶 = 200%），**而且依然与 `fec_percentage` 无关**。

</details>

**验证方法**：设 `min_log_level = debug`，日志里会依次打印：

```
Client configured bitrate is 50000 Kbps
Final adjusted video encoding bitrate is 46488 Kbps
```

把 `fec_percentage` 在 20 和 5 之间切换，看第二行的比值——应该精确等于 $\frac{100-5}{100-20} = 1.1875$。这就是最直接的证据。
