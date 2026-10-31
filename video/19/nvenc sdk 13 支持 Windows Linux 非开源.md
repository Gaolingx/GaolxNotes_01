# nvenc sdk 13 支持 Windows Linux 非开源

## 🧑‍💻 User

nvenc sdk 13只支持windows吗 开源吗

---
## 😀 默认助手 (7)

## 结论先说

| 问题 | 答案 |
|---|---|
| 只支持 Windows？ | ❌ 不是。SDK 13 支持 Windows 和 Linux（含 Jetson） |
| 开源吗？ | ❌ 不开源（专有许可），但存在开源替代路径 |

---

## 一、平台支持情况（已核对官方 13.x 文档）

NVENC / NVDEC 的 NVENCODE API 在以下平台可用：

| 应用架构 | 对应库 | 备注 |
|---|---|---|
| Windows x64 | `nvEncodeAPI64.dll` | Win10 及以上 |
| Windows on Arm（原生 Arm64） | `nvEncodeAPIa64.dll` | 13.x 文档已列出 |
| Linux x86_64 / aarch64 | `libnvidia-encode.so` | 随驱动提供 |
| NVIDIA Jetson Linux | 同上 | 2026 年 1 月新增 Jetson Thor 编解码支持，Jetson 仅支持 CUDA device type |

设备类型差异也体现了这一点：

- `NV_ENC_DEVICE_TYPE_DIRECTX`（DX9/10/11/12）→ **仅 Windows**
- `NV_ENC_DEVICE_TYPE_OPENGL` → **仅 Linux**
- `NV_ENC_DEVICE_TYPE_CUDA` → Windows + Linux + Jetson（Jetson 只支持这个）

**不支持的平台**：macOS（SDK 9.0 起已移除 Mac 支持）、Android/iOS 无官方 SDK、FreeBSD 等无官方支持。所以准确说法是 **“Windows + Linux 双平台”**，不是只有 Windows；如果你的场景是 macOS，那确实没有任何官方途径。

---

## 二、是否开源

**不开源。** 具体情况：

- 整体是 **NVIDIA 专有许可**（下载完整 SDK 必须登录 NVIDIA 开发者账号并勾选同意 EULA），许可证禁止逆向、再分发等。
- 完整 SDK 包（13.1.x）现在给的是 **预编译 samples + 文档 + lib**；**sample 源码需要单独“申请源码”**（Apply for Source Code，也要登录授权），不随包公开。
- 唯一可公开下载的是 **头文件包** `video_codec_interface_13.1.24.zip`（不需要登录），但这也是 NVIDIA 版权代码，不是开源协议。
- 更关键的是：**编码器真正的实现根本不在 SDK 里**，而在 GPU 驱动 + 硬件固件中，永远不可能有源码。SDK 只是一层 C-API 头文件 + 函数指针表（`NvEncodeAPICreateInstance` 填充）。

<details>
<summary><b>三、如果你需要“开源可用”的替代方案（展开）</b></summary>

1. **FFmpeg 的 nvenc 封装**（最常用）
   - 配置：`./configure --enable-nvenc --enable-cuda-llvm ...`
   - 需要 **`nv-codec-headers`**（videolan/ffmpeg 维护，MIT 协议）来提供 `nvEncodeAPI.h`，这样可以不装 NVIDIA SDK 就编译出带 nvenc 的 FFmpeg。
   - 注意：这是“开源外壳 + 闭源驱动”，硬件编码本身依然是闭源驱动实现的。
2. **Vulkan Video**（Khronos 开放规范，含 encode 扩展）
   - 规范、头文件完全开放，NVIDIA 在 Windows/Linux 上提供驱动支持，SDK 13 也开始附带 Vulkan SDK 相关示例。
3. **PyNvVideoCodec**（NVIDIA 官方 Python 封装，pip 可装，免费但闭源）
4. **rigaya/NVEnc（NVEncC）** — GitHub 上源码可得的命令行编码器，适合做研究/批处理参考。
5. 纯开源方案（x264 / x265 / SVT-AV1 / libaom）：CPU 编码，无 NVIDIA 依赖。

</details>

---

## 四、实践建议

- **Linux 服务器、容器、Jetson 场景完全没问题**，用 `libnvidia-encode.so` + CUDA device type 即可，这也是绝大多数转码服务的选择。
- **只需要基础转码、不追求极致参数控制**：直接用 FFmpeg 的 `h264_nvenc / hevc_nvenc / av1_nvenc`，配合 `nv-codec-headers` 编译，可以完全绕开 SDK 下载授权流程。
- **需要 SDK 13 的新特性**（UHQ 模式的 AV1、422 编解码、MV-HEVC、CUDA Array 输入、迭代编码等）：必须接受 EULA 下载官方包，且注意这些多为 **Blackwell（RTX 50 系 / B200 等）** 硬件专属能力，老卡升级 SDK 也拿不到。

> 说明：平台与许可信息来自 NVIDIA 官方 Video Codec SDK 页面和 13.x NVENC 编程指南；`nv-codec-headers` 的 MIT 授权性质属于社区事实，NVIDIA 并未官方背书，商用前建议自行确认合规性。
