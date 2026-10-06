# VVC软解FFmpeg

## 🧑‍💻 User

高性能vvc软解方案有哪些，ffmpeg计划支持吗

---
## 😀 默认助手 (7)

I'll research the latest information on VVC software decoders and FFmpeg support.

Good leads. Let me fetch the key pages for details.

Let me gather more details on additional decoders and confirm FFmpeg's current status.

I now have solid information. Let me verify FFmpeg's latest encoder/decoder details.

# 高性能 VVC（H.266）软解方案 & FFmpeg 支持现状

> 结论先说：**高性能开源软解目前以 `VVdeC` 为标杆**，**FFmpeg 已原生支持 VVC 解码**（自 7.1 起正式稳定，8.0 继续纳入），编码侧主要通过外挂 `libvvenc` 实现。

---

## 一、主流高性能 VVC 软件解码方案

| 方案 | 来源 | 定位 | 许可证/开放度 | 特点 |
|------|------|------|--------------|------|
| **VVdeC** (`vvdec`) | Fraunhofer HHI | 快速软解（事实标准） | 开放源码（Fraunhofer 自有许可） | 完整支持 **VVC Main10**，主打实时，跨平台最广 |
| **FFmpeg 原生 `vvc`** | FFmpeg / Nuo Mi 等 | 通用解码器 | LGPL | 集成进 FFmpeg 生态，7.1+ 稳定 |
| **OpenVVC** | IETR / INSA Rennes | 实时软解 | **LGPLv2.1** | 仍在开发，8-bit、Frame/Tile 并行、RPR |
| **O266dec** | 腾讯媒体实验室 | 工程化软解 | **仅二进制**（600 帧试用） | CPU 高效，已集成到改造版 VLC |
| **VTM** | JVET 参考软件 | 正确性基准 | 参考软件 | 精度最高但**极慢**，不可实用 |
| 商业方案 | MainConcept / Bitmovin / Sharp 等 | 商用软解 | 闭源授权 | 面向广播/OTT 工程化 |

### 重点方案详解

<details>
<summary><b>1. VVdeC —— 开源快速软解首选</b></summary>

- 仓库：`github.com/fraunhoferhhi/vvdec`
- 由 Fraunhofer HHI（VVC 标准主要推动者之一）开发，论文 *"Towards A Live Software Decoder Implementation For VVC"* (ICIP 2020) 提出，初衷就是**实时解码**。
- 支持 **VVC Main10 profile 全部特性**，覆盖 8/10-bit。
- 平台覆盖极广：Windows（x86/x64）、Linux（x86_64/armv7/aarch64）、macOS、Android、iOS，甚至 **WASM（浏览器）**。
- 构建：CMake，`make install-release`；可用 `enable-bitstream-download=1` 下载一致性码流测试。
- 常被作为 FFmpeg、VLC 等的后端解码器。
</details>

<details>
<summary><b>2. FFmpeg 原生 VVC 解码器</b></summary>

- 作者主要为 **Nuo Mi（nuomi2021）**、Frank Plowman、Christian Bartnik 等。
- 基于 libavcodec 的原生实现，无需外部库即可解码 VVC。
- 支持 IBC（Intra Block Copy）等关键工具，已通过大量一致性测试。
</details>

<details>
<summary><b>3. OpenVVC</b></summary>

- 仓库：`github.com/OpenVVC/OpenVVC`，官网 `openvvc.github.io`。
- 定位：**ITU-T H.266 合规的实时开源软件解码器**，LGPLv2.1，便于商用集成。
- 现状：仍在开发；v1.0.0 支持 8-bit、Frame + Tile 并行、RPR；测试程序 `dectest`。
- 文档明确提供 **"Embedding OpenVVC into FFmpeg"**，可作为 FFmpeg 外挂解码器。
</details>

<details>
<summary><b>4. 腾讯 O266dec / 其他</b></summary>

- **O266dec**：腾讯媒体实验室的高性能、CPU 高效解码库，面向播放器/转码；**仅提供二进制**，评估版限 600 帧，参考 JVET-T0095。
- **VTM**：JVET 官方参考软件，用于一致性验证与算法研究，速度不可用于生产。
- 编码侧可关注 **VVenC**（Fraunhofer）、**uvg266**、**Ali266**（阿里）等。
</details>

---

## 二、FFmpeg 对 VVC 的支持：**已支持**

FFmpeg 不仅"计划"支持，**早已落地**。关键时间线：

| 时间 | 事件 |
|------|------|
| 2024-01-04 | FFmpeg 原生 VVC 解码器合入（实验性，缺 IBC） |
| 2024-02-24 | IBC（Intra Block Copy）合入，解码功能补全 |
| 2024-03-14 | 因少数码流 MD5 不一致，被 James Almer 重新标为 experimental |
| 2024-06-15 | **外部 VVC 编码器**支持（`libvvenc`）合入 |
| 2024-09-07 | Frank Plowman / Nuo Mi 移除 experimental 标记，判定稳定 |
| **2024-09-30** | **FFmpeg 7.1 正式发布，包含完整原生 VVC 解码器** |
| 2025-08-22 | **FFmpeg 8.0 发布**，继续纳入原生 VVC 解码（并提及需更多 fuzzing） |

**要点：**

- **解码**：`vvc` 解码器为 **原生（native）实现**，无需外部库。命令行直接 `ffmpeg -i input.vvc ...` 或 `ffplay` 即可。
- **编码（VVC）**：正式版主要通过**外部库 `libvvenc`** 实现，编译时加 `--enable-libvvenc`：
  ```bash
  ./configure --enable-libvvenc ... 
  ffmpeg -i input.y4m -c:v libvvenc output.mp4
  ```
- **已知限制（编码侧）**：默认 10-bit（可用 `-vvenc-params InternalBitDepth=8` 出 8-bit）；SAR/DAR 变形处理、Matroska 封装、多 pass 等仍有坑，生产编码建议直接用 `vvencapp`。
- 若需与 VVdeC 互操作，可用外部解码器路径或社区补丁；官方主流路线是**原生 vvc 解码 + libvvenc 编码**。

---

## 三、选型建议

- **要快、要跨平台、要开源** → 选 **VVdeC**（尤其实时/多平台/WASM 场景）。
- **已用 FFmpeg 工具链** → 直接升级到 **7.1+ / 8.0**，用原生 `vvc` 解码。
- **要 LGPL 商用集成** → 关注 **OpenVVC**。
- **播放器工程化 / 低 CPU** → 评估 **O266dec**（注意二进制与授权）。
- **一致性验证/科研** → 用 **VTM**。

<details>
<summary><b>说明</b></summary>

以上信息结合了我的既有知识与本次对官方仓库/发布页的检索（`fraunhoferhhi/vvdec`、`OpenVVC/OpenVVC`、`TencentCloud/O266player`、FFmpeg 7.1/8.0 发布说明、`VVCEasy` 的 VVC 时间线记录）。版本迭代较快，落地前建议再核对目标 FFmpeg 版本与各库最新 Release。
</details>

需要我帮你写一份**具体平台的编译/调用命令**（例如 FFmpeg+libvvenc、或用 VVdeC 解码 VVC 码流）吗？
