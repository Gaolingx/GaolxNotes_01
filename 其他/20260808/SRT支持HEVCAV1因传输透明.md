# SRT支持HEVCAV1因传输透明

### 🧑‍💻 User

srt地址和rtmp有什么区别，为什么前者可以支持hevc甚至av1推流？

---

### 🤖 Assistant

**SRT 是现代、编解码器无关的可靠传输协议（基于 UDP），而 RTMP 是较老的基于 TCP 的协议（常绑定 FLV 容器），因此 SRT 天然更容易支持 HEVC 甚至 AV1。**

### 核心区别对比

| 方面 | RTMP | SRT |
|------|------|-----|
| **全称/来源** | Real-Time Messaging Protocol（Adobe/Flash 时代） | Secure Reliable Transport（Haivision 发起，开源，SRT Alliance） |
| **底层传输** | TCP | UDP + 自带可靠性机制（ARQ 重传、拥塞控制、可调延迟缓冲） |
| **网络表现** | 可靠但易受丢包影响（TCP 重传导致延迟抖动/卡顿，Head-of-Line Blocking） | 专为不可靠网络（公网）设计，低延迟下更好处理丢包、抖动，可调 latency |
| **延迟** | 通常较低，但差网时会飙升 | 可精确控制延迟（常见 20–500ms+），差网下更稳定 |
| **安全性** | 有 RTMPS（TLS），但很多场景未启用 | 内置 AES 加密，配置简单 |
| **容器/封装** | 主要配合 FLV（或类似） | 通常封装 MPEG-TS（也可其他），几乎不限制内容 |
| **编解码器支持** | 生态上以 H.264 + AAC 为主；HEVC 支持不标准/不普及，AV1 几乎没有 | **编解码器无关（codec-agnostic）**，轻松支持 H.264、HEVC、AV1、VP9 等 |
| **典型用途** | 直播推流到 CDN/平台（YouTube、Twitch 等历史主流推流方式，正在被替代） | 专业贡献链路（编码器→服务器/云）、点对点传输、公网可靠推流 |
| **地址形式** | `rtmp://server/app/streamkey` 或 `rtmps://...` | `srt://ip:port?mode=caller&latency=120&passphrase=xxx&streamid=...`（参数丰富） |
| **现状** | 逐渐边缘化（Flash 已死，很多平台仍兼容但优先推新协议） | 专业直播、远程制作、云贡献中快速普及 |

**简单总结**：
- RTMP 更像“老牌直播推流协议”，绑定历史生态（Flash/FLV），实现简单但扩展性差。
- SRT 更像“可靠媒体传输层”，只负责把数据包稳定、安全、低延迟地送到对端，**不关心里面是什么视频/音频**。

### 为什么 SRT 能轻松支持 HEVC 甚至 AV1，而 RTMP 不行/很难？

1. **协议本身是否“管”编解码器**
   - **SRT 完全编解码器无关**：它只是传输管道。常见做法是把编码后的视频/音频打成 **MPEG-TS** 包，再通过 SRT 发送。MPEG-TS 对 HEVC（H.265）有标准支持，AV1 也可以通过适当的 stream type/descriptor 承载（虽然 AV1-in-TS 不如 HEVC 普及，但技术上可行，FFmpeg 等工具支持）。
   - **RTMP + FLV 有历史包袱**：RTMP 常配合 FLV 容器。FLV 最初为 Flash 设计，对现代编码支持很弱。虽然有人扩展过 HEVC over RTMP/FLV，但属于非标准或厂商私有实现，很多服务器（nginx-rtmp 等）、CDN、播放器不认或支持不完整。AV1 在 RTMP 生态里基本没人认真支持。

2. **生态与实现现实**
   - 主流 RTMP 推流链路（OBS 默认、很多硬件编码器、平台 ingest）默认/强制 H.264 + AAC，是为了最大兼容性。
   - SRT 从设计之初就是为专业视频传输，Haivision 等厂商和开源社区（libsrt）直接面向 HEVC 等高效编码，现代编码器（FFmpeg、OBS + SRT 插件、硬件编码器）推 HEVC/AV1 over SRT 很常见。
   - 接收端（媒体服务器、转码节点）用 SRT 时通常直接解 TS 流，再按需转封装/转码，不依赖 RTMP 那套老解析逻辑。

3. **技术细节对比**
   - RTMP 有自己的消息分片、chunk、AMF 元数据等，和旧 Flash 播放器深度绑定，扩展新编码成本高。
   - SRT 只负责“可靠 UDP 传输 + 加密 + 延迟控制”，媒体格式完全由上层决定（TS 最常见，也有人用其他）。因此 HEVC、AV1、甚至未来编码都能直接用。

### 实际使用注意点

- **推流场景**：很多平台仍主要收 RTMP（H.264），但专业/自建/云贡献链路越来越多用 SRT，尤其想省带宽推 HEVC 时。
- **地址与参数**：SRT 地址通常带 `mode`（caller/listener/rendezvous）、`latency`、`passphrase`、`streamid` 等，比 RTMP 灵活得多，也更需要正确配置。
- **兼容性**：SRT 需要双方都支持（发送端和接收端）。接收端常见有 mediasoup、SRS、Wowza、自建 FFmpeg/GStreamer、云厂商 SRT ingest 等。
- **HEVC/AV1 推流**：用 SRT + MPEG-TS 是目前较干净的方式之一。FFmpeg 示例思路大致是 `ffmpeg ... -c:v libx265 / libaom-av1 -f mpegts srt://...`（具体参数需调）。
- **其他现代选择**：WebRTC（超低延迟互动）、RIST（类似 SRT 的可靠传输）、直接 QUIC/HTTP/3 等也在发展，但 SRT 在“可靠贡献 + 现代编码”场景已经很成熟。

**一句话**：RTMP 受历史容器和生态限制，主要卡在 H.264；SRT 是纯传输层 + 常用 TS 封装，所以 HEVC、AV1 都能直接推，而且在差网络下更稳。

如果你有具体场景（比如 OBS 推流、FFmpeg 命令、服务器选型、延迟要求等），可以补充，我可以给更针对性的配置建议。