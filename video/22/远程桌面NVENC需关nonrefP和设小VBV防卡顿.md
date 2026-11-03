# 远程桌面NVENC需关nonrefP和设小VBV防卡顿

## 🧑‍💻 User

nvenc enableNonRefP对于远程桌面一般要关吗（大vbv会明显卡顿）

---
## 😀 默认助手 (7)

## 先给结论

**是的，远程桌面场景一般建议关掉（保持默认 `0`）。** FFmpeg 里就是 `-nonref_p 0`；NVIDIA 这个开关的定位是“给**高运动量**画面做率失真优化”，而远程桌面的内容是“大面积静止 + 小区域更新 + 文字/UI”，它能带来的码率效率收益接近于零，却会引入副作用。

而且它和你说的问题其实是**同一类毛病**：都会让**逐帧码率波动变大**。

---

## 它到底做了什么

`-nonref_p` 对应 NVENC 的 `enableNonRefP`，FFmpeg 的帮助文本是：

> "Set this to 1 to enable automatic insertion of non-reference P-frames"（默认 `0`）

机制上：编码器**自己判断**某一帧不值得进入参考链，就把这一帧编码成一个 `nal_ref_idc = 0` 的 P 帧——后续帧不会拿它当参考（语义上接近“一次性的 P 帧/时间层顶层帧”）。好处是：

- 这一帧的量化误差不会向后传播，允许用更高的 QP 编得更省；
- 丢包时少一环误差扩散；
- 不需要重排序，**不增加解码延迟**。

但代价是：编码器在“哪些帧进参考链、哪些不进”上做了自主决策，**参考链不再完全确定**，逐帧码率分布也更抖。

<details>
<summary>为什么桌面内容基本吃不到这个收益（点开）</summary>

- 桌面内容 80%+ 的区域是静止的（桌面壁纸、窗口边框、大面积白底）。这些区域**必须**持续作为参考才能“零成本”保持清晰；把它们踢出参考链 = 下一帧要重新编码，纯亏。
- 高运动量场景（游戏、视频播放）里 non-ref P 的收益来自“快速运动区域本来就要花很多比特，误差扩散代价高”。桌面里这类区域是滚动网页、视频窗口——占比小。
- 文字/UI 边缘对**瞬时质量抖动**极其敏感。non-ref 帧用更高 QP 编码后，会出现“某块更新区域先糊一下、下一帧又变清楚”的闪烁感，人眼在静态桌面上特别容易察觉。
- 部分驱动版本上，它和 `-intra-refresh 1` 同时使用时，恢复点（recovery point）的推进节奏会和“每帧都更新参考”的假设不一致，恢复行为变得更难预测。
- 兼容性：码流本身是合法的，绝大多数解码器没问题；但个别老硬解或自行实现“丢帧后保留上一帧好帧”逻辑的客户端，对 non-ref 语义处理不严谨，会放大卡顿观感。
</details>

**什么时候可以留 `1`**：远程桌面主要用来跑游戏 / 看视频，客户端解码器确认正常，且网络**有丢包**（non-ref 能加快恢复）。其余情况一律 `0`。

---

## 关于“大 VBV 明显卡顿”——这个才是主因

VBV buffer 是编码器的“弹性预算池”。**buffer 越大，编码器越允许“攒一波比特、然后一次性爆发”**。

对实时桌面链路（无大缓冲、走 UDP/QUIC 或裸 TCP）来说：

$$
\text{单帧传输时间} \approx \frac{\text{帧大小}}{\text{可用带宽}}
$$

大 VBV 下，一帧可以瞬间吃到 buffer 允许的几十帧预算 → 这一帧传输时间暴涨 → 接收端 jitter buffer 被顶穿 → 表现为**周期性/突发性卡顿**，而且延迟会“呼吸”。静态桌面时它又长期低于目标码率，把额度攒起来，等你拖动窗口那一刻集中释放——正好是最不该卡的时候卡。

**VBV 该给多大：** 按“帧预算”算，1~2 帧即可。

$$
\text{bufsize} \approx \frac{\text{bitrate}}{\text{fps}} \times (1 \sim 2)
$$

例：1080p60 @ 10 Mbps → $\frac{10\times10^6}{60} \approx 167$ kbit/帧 → `-bufsize 200k`（≈1.2 帧）。给了 `-bufsize 10M` 就等于放开 60 帧的爆发空间，必卡。

另一个配套坑：VBV 压到 1 帧后，**I/IDR 帧会被 VBV 掐死**导致周期性花屏。这时用 `-ldkfs`（low delay keyframe scale，允许关键帧超出的倍数）或干脆用 `-intra-refresh 1` 消灭 IDR 尖峰。

---

## 一套可用的基线参数

```bash
ffmpeg -f gdigrab -framerate 60 -i desktop \
  -c:v h264_nvenc -preset p4 -tune ull \
  -rc cbr -b:v 10M -maxrate 10M -bufsize 200k \
  -bf 0 -zerolatency 1 -rc-lookahead 0 -multipass disabled \
  -nonref_p 0 -spatial-aq 0 -g 600 -intra-refresh 1 \
  -pix_fmt yuv420p -f mpegts udp://... 
```

| 参数 | 建议 | 说明 |
|---|---|---|
| `-nonref_p` | `0` | 桌面场景无收益、有抖动风险 |
| `-tune` | `ull` | 关掉 lookahead/B/多 pass 的延迟路径 |
| `-bufsize` | 1~2 帧 | 控制卡顿的关键，别用“大 VBV 求画质” |
| `-bf` | `0` | 需要 B 帧时用 `-b_ref_mode disabled` 并接受延迟 |
| `-multipass` | `disabled` | `qres` 可小幅提质，代价是 GPU 负载 |
| `-spatial-aq` | 桌面关、游戏开 | 桌面开 AQ 容易把文字“涂抹” |
| `-g` / `-intra-refresh` | 大 GOP + 周期刷新 | 消除 IDR 尖峰 |
| `-delay` | 默认或 1 | 即 `async_depth`，别积压多帧 |

> 4:4:4（`-profile:v rext -pix_fmt yuv444p`，HEVC 同理）对文字锐度是巨大提升，但绝大多数**硬解**客户端不支持 4:4:4，只在纯软解客户端上用。

<details>
<summary>怎么验证 nonref_p 到底有没有在起作用（点开）</summary>

H.264 上可以直接看 slice header：

```bash
ffmpeg -i in.h264 -c copy -bsf:v trace_headers -f null - 2>&1 | grep -i nal_ref_idc
```

出现 `nal_ref_idc = 0` 的 P slice 就说明编码器真的插了 non-ref 帧；一直是 1 说明驱动没触发（NVENC 内部触发条件由驱动决定，不同驱动版本行为会有差异）。HEVC 没有 `nal_ref_idc` 字段，这个方法不适用。
</details>

---

## 一句话总结

- `nonref_p`：**远程桌面关掉**，它是为高运动量视频准备的优化，桌面赚不到还容易闪。
- 卡顿的真凶更可能是 **VBV 给太大**：把 `-bufsize` 压到 1~2 帧、配 `-tune ull` + `-bf 0` + 大 GOP/`intra-refresh`，逐帧大小平稳了，卡顿基本就没了。

注：NVENC 内部对 `enableNonRefP` 的触发条件、以及与 `ull` tuning 的优先级属于驱动实现细节，不同驱动版本可能有出入（我是基于 FFmpeg 的选项语义和 NVENC 公开行为推断的，没有在你这套驱动上验证过）——建议以实测的 `nal_ref_idc`/帧大小分布为准。
