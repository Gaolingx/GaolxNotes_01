# 单向B帧零延迟替代P帧提升画质

## 🧑‍💻 User

我记得nvenc有一种不增加延迟的B帧（只看过去），这个加了有收益吗？具体是哪些场景

---
## 😀 默认助手 (7)

## 先确认：你记的是 **Unidirectional B Frames**（单向 B 帧）

这不是“没有 lookahead 的普通 B 帧”，而是 NVENC 在 Video Codec SDK 12.2 里专门加的一个特性，官方编程指南第 8.19 节写得很直白：

> Unidirectional B frames use past frames only for **both L0 and L1** reference list and **avoid the latency issues** observed with conventional B frames. Therefore they can be used **in place of P frames** especially in low latency encoding. Unidirectional B frames **improve video encoding quality and has little performance impact**.

也就是说：普通 B 帧的 L0 取自过去、L1 取自未来，所以必须等未来帧；单向 B 帧的 L0、L1 都取自**过去两帧**，因此编码器不需要向后缓冲。

### 为什么它真的“零延迟”（这条是关键）

| | 常规 B 帧 | 单向 B 帧 |
|---|---|---|
| L0 / L1 | 过去 / **未来** | 过去 / 过去 |
| 编码顺序 vs 显示顺序 | 不同（需重新排序） | **相同** |
| DPB `max_num_reorder_frames` | 2~4 | **0** |
| 额外延迟 | 编码端缓冲 + 解码端重排，两端都付出延迟 | **两端都是 0** |

因为两路参考都在显示顺序的“过去”，解码顺序天然等于显示顺序，播放器/浏览器不需要多缓冲几帧，WebRTC 的 jitter buffer、LL-HLS 的延迟也都不会被抬高。这是它和“把 `bf` 调小”本质上的区别：后者无论多小，都要付出重排延迟。

---

## 收益有多大

NVIDIA 官方博客给出的实测：

- **相对 P 帧平均 3~4% BD-BR 节省**
- **特定序列最高可到 15%**
- 官方描述：*little performance impact*（编码速度基本不掉）

也就是说，这是一项收益中等偏小、但在当前配置下基本没有额外成本的编码增益。3~4% 大致相当于“白送”的码率：在 20 Mbps 下，相当于画质不变、码率降到约 19.3 Mbps，或者码率不变、画面更干净一点。

> 注意收益方向：它是 **vs P 帧** 的节省。如果你现在跑的是 IPPP（`-bf 0`，低延迟场景的默认），那么这 3~4% 可以拿到；如果你本来就在用 `b_ref_mode=middle` 的层级 B，再叠加它就没什么意义了，因为传统 B 帧本身的增益大得多。

---

## 具体哪些场景值得开

<details>
<summary><b>✅ 强烈建议开启（这就是它被设计出来的场景）</b></summary>

1. **云游戏 / 远程游戏串流**（Moonlight+Sunshine、类 GeForce NOW 自建）
   延迟预算 < 50 ms，绝对不能承受 B 帧重排；同时画面是高频大范围运动，双向预测收益比平均值高。
2. **会议 / 屏幕共享 / 摄像头采集**（WebRTC、自建 RTC）
   CBR + 极小 VBV，单向 B 帧正好是官方推荐表里与 ULL/LL tuning 并列的那一项。
3. **低延迟直播**（Twitch/YouTube RTMP `-tune ll`、电竞赛事、连麦）
   有 ~1 s 缓冲，但你不想为了 B 帧再多背 100 ms。
4. **遥控/图传/无人机/机器人/车载/工业相机回传**
   端到端延迟敏感，解码端资源也有限；这两路参考都取自过去，重排延迟为 0，正好满足要求。
5. **安防/执法记录仪类低延迟回传**（自然画面、有传感器噪声，收益接近官方测算上限）

</details>

<details>
<summary><b>⚠️ 可开可不开 / 收益接近 0</b></summary>

- **超高码率本地录制**（如 4K60 50 Mbps 以上、画面接近视觉无损）：P 帧本来就够好，3~4% 几乎看不见。
- **极静态内容**（纯桌面、PPT、代码屏）：几乎没有双向预测可利用的残差相关性。
- **本身已经开了 `-tune ull` 且没开 AQ/multipass**：先把 `-spatial-aq 1 -temporal-aq 1`、`-multipass full` 打开，这些增益更大，再考虑单向 B。

</details>

<details>
<summary><b>❌ 不要用 / 用不了</b></summary>

- **H.264 管线**（Discord、老浏览器、只收 H.264 的 RTMP 目标）：**不支持**，官方明确 HEVC only。
- **AV1**：截至 SDK 13.0 文档仍写 “currently supported for **HEVC**”，H.264/AV1 都没有这个开关。
- **归档 / VOD / 离线转码**：这里没有延迟约束，直接用传统 B 帧（`-bf 3~5 -b_ref_mode middle`）或 UHQ tuning，增益是 10~20% 量级，远大于 3~4%。
- **MV-HEVC 等特殊模式**：文档明确列出 UnidirectionalB 与其不兼容。
- **硬件/驱动不支持**：必须查询 `NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B`；FFmpeg 在不支持时直接报错退出：`Unidirectional B-Frames not supported by the device`。

</details>

---

## 怎么开（FFmpeg）

单向 B 帧在 FFmpeg 里由 `nvenc_hevc` 的 `unidir_b` 选项开启（2024-03 加入，`libavcodec` 61.5.100，**FFmpeg 7.1 起才有**，7.0 的构建里通常没有）：

```bash
# 先确认你的构建有这个选项
ffmpeg -hide_banner -h encoder=hevc_nvenc | grep unidir_b

# 典型低延迟 CBR 配置：保持 IPPP 结构，让 P 帧位置用单向 B 代替
ffmpeg -i in.mp4 \
  -c:v hevc_nvenc -preset p4 -tune ll \
  -rc cbr -b:v 20M -maxrate 20M -bufsize 333k \
  -bf 0 -unidir_b 1 \
  -spatial-aq 1 -temporal-aq 1 \
  -g 600 -rc-lookahead 0 \
  -f mpegts udp://...
```

几个要点：

- **`-bf 0` 建议保留**：单向 B 的设计用途就是“顶替 P 帧”，GOP 结构保持 IPPP 最干净；如果同时开传统 B，两者不会叠加收益，先关掉再对比。
- `-bufsize` 按 1 帧给最激进：例如 $20\,\text{Mbps} \div 60\,\text{fps} \approx 333\,\text{kbit}$。
- 它**不依赖** lookahead，`-rc-lookahead 0` 完全兼容。
- 想再加码可以叠加 8 bit→10 bit 输出（官方另一项 3~4% 增益，需解码端支持 Main10）。

---

## 决策：和“真 B 帧”怎么取舍

按延迟预算选，不要凭感觉：

$$
\text{延迟预算} < 1 \text{ 帧} \rightarrow \text{单向 B（唯一选项，收益 3~4\%）}
$$

$$
\text{预算} \ge 2\text{--}4 \text{ 帧} \rightarrow \text{传统层级 B + } b\_ref\_mode\text{（收益 10~20\%）}
$$

以 60 fps 为例，一帧是 $\frac{1}{60}\text{s} \approx 16.7\,\text{ms}$；层级 B 在编码端要缓冲约 4 帧、解码端还要重排 2~3 帧，合计 **100 ms 量级**——这就是为什么云游戏/会议只有单向 B 可选。

---

<details>
<summary><b>如果你要走“自证收益”这条路线（3~4% 太小，很可能被主观感受吃掉）</b></summary>

建议固定码率、固定 preset，只切 `-unidir_b 0/1`，在你自己的素材上跑：

```bash
# 两遍只差一个开关
ffmpeg -i in.yuv -c:v hevc_nvenc -preset p4 -tune ll -rc cbr -b:v 20M \
       -bf 0 -unidir_b 0 -y out_off.mkv
ffmpeg -i in.yuv -c:v hevc_nvenc -preset p4 -tune ll -rc cbr -b:v 20M \
       -bf 0 -unidir_b 1 -y out_on.mkv
```

然后对比 VMAF/SSIM 与文件大小。注意：**内容决定收益上限**——快速转视角的 FPS、手持拍摄、无人机航拍这类双向预测有优势的素材容易接近 15%；静态桌面基本为 0。

</details>

---

**一句话结论**：你记得没错，这叫 **Unidirectional B Frames**（HEVC only，SDK 12.2+）。它在**编码端和解码端都不增加任何重排延迟**，代价几乎为 0，官方实测相对 P 帧 **平均 3~4%、最高 15%** 的码率节省——凡是“延迟预算只有 0~1 帧、又跑 HEVC”的场景（云游戏、会议、低延迟直播、图传、安全回传）都值得无条件打开；而在可以接受延迟的归档/转码场景里，它反而不如传统 B 帧，不用考虑。

---
## 🧑‍💻 User

这个和ref_frames_in_dpb有关系吗？

---
## 😀 默认助手 (7)

## 有关系，但它是**三个不同参数**中的一个，不能混为一谈

先把名字对上：你看到的 `ref_frames_in_dpb` 在 NVENC 里的真名是

- HEVC / AV1 → `NV_ENC_CONFIG_HEVC::maxNumRefFramesInDPB`（头文件注释：*“Specifies the maximum number of reference frames in the DPB”*）
- H.264 → `NV_ENC_CONFIG_H264::maxNumRefFrames`
- FFmpeg 暴露为 **`-dpb_size`**（`0` = 让驱动决定，也是默认值）

和它容易混淆的还有两个：

| 参数 | NVENC 字段 | FFmpeg | 管什么 |
|---|---|---|---|
| DPB 容量 | `maxNumRefFramesInDPB` | `-dpb_size` | DPB 里最多放几帧参考 |
| 列表候选数 | `numRefL0` / `numRefL1` | `-refs` | L0/L1 每个列表最多用几个参考 |
| GOP 结构 | `frameIntervalP` | `-bf` | 两个 P 之间插几个 B |

头文件里 `numRefL0`/`numRefL1` 的原文是 *“max number of reference frames in reference picture list L0/L1, that can be used by hardware for prediction”*，注意这是**每个列表**的上限，和 DPB 容量不是一回事。

---

## 关系一：单向 B 让 DPB 的“重排项”归零（这是核心）

DPB 要装两类帧：

$$\text{DPB} \;\ge\; \underbrace{N_{\text{ref}}}_{\text{参考帧}} \;+\; \underbrace{N_{\text{reorder}}}_{\text{解码顺序超前于显示序的帧}}$$

- **常规 B 帧**：`N_reorder > 0`。NVENC 文档讲 temporal SVC 时给过这个下界：`maxTemporalLayers > 2` 时，“为帧重排所需的最小 DPB” $= (\text{maxTemporalLayers}-2)\times 2$。也就是说你必须在 SPS 里把 `maxNumRefFramesInDPB` 抬高，否则 B 帧根本插不进去。
- **单向 B 帧**：两个参考全在过去 → `N_reorder = 0`，DPB 里不需要为“未来帧”预留槽位。

FFmpeg 侧可以直接验证这个差异（`libavcodec/nvenc.c`）：

```c
if (ctx->encode_config.frameIntervalP > 1)
    avctx->has_b_frames = 2;        // 只有真 B 帧才声明 2 帧重排延迟
```

单向 B 配合 `-bf 0` 时 `frameIntervalP` 仍然是 1，所以 `has_b_frames` 保持 **0**——和纯 P 帧一模一样。**这就是“不增加延迟”在代码层面的具体出处。**

> 这条也解释了一个常见误区：你**不需要**为了开 unidir_b 去加大 `dpb_size`。它是“在原有 P 帧 DPB 结构里做双向预测”，不是“加一类新帧”。

---

## 关系二：它对 `refs` 有隐性下限要求（这条最实用）

官方的参考列表图（博客 Figure 5）写得很清楚：单向 B 的 **L0 指最近一帧、L1 指更早的一帧**，两路都取自过去。

所以：

- `-refs 1`（或 `dpb_size 1`）→ DPB 里只有一个过去帧，L1 拿不到“更早的另一帧”，双向加权退化成近似单向预测，**3~4% 的收益基本归零**（不会报错，只是白白启用、没有收益）。
- 建议 `-refs 2`：IPPP 结构下每个 P 都是参考帧，DPB 里天然有 2 个不同的过去帧可用，够了；给到 3~4 收益边际很小，还会增加 ME 的显存带宽开销。

注意 `numRefL0/L1` 才是那个“多参考帧要查 `NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES` cap”的参数（见 `nvenc.c` 第 571 行），也就是说**多参考帧是有硬件支持门槛的，而 unidir_b 本身是另一个独立的 cap**（`NV_ENC_CAPS_SUPPORT_UNIDIRECTIONAL_B`）。

---

## 关系三：显存差异其实不在 DPB，而在 `frameIntervalP`

FFmpeg 的 surface 分配公式：

```c
// ×2 是同一 GPU 上两个 NVENC 引擎，×2 是避免阻塞下一个 PBB 组
int nb_surfaces = FFMAX(4, ctx->encode_config.frameIntervalP * 2 * 2);
```

| 配置 | frameIntervalP | surface 数 |
|---|---|---|
| `-bf 0` + `-unidir_b 1` | 1 | **4**（与纯 P 帧相同）|
| `-bf 3`（常规层级 B）| 4 | **16（4 倍）** |

这和 NVIDIA 文档“降低显存占用”那节的建议完全对得上：

> **Avoid using B-frames.** B-frames requires additional buffers for **reordering**, hence avoiding B-frames would result in savings in video memory usage.

单向 B 不属于这个范畴——它没有 reorder buffer，surface 数和 P 帧一致。**这是它在低延迟/低显存设备（如 Jetson、图传、车载）上比常规 B 帧更有优势的地方。**

---

## 落到参数上

```bash
ffmpeg -c:v hevc_nvenc \
  -bf 0            # frameIntervalP=1，不引入 reorder，surface 保持 4
  -unidir_b 1      # P 帧位置换成"两路过去参考"的 B（HEVC only）
  -refs 2          # 保证 L1 有"更早的那一帧"可指；给 1 等于白开
  -dpb_size 0      # 默认，让驱动决定，通常不用动
```

**验证是否真的生效（推荐）**：看输出流的 `has_b_frames`，单向 B 应该是 0，常规 B 是 2。

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=has_b_frames -of csv=p=0 out.mkv
```

如果这里输出 `2`，说明你没有真正走 unidir 的路径（多半是 `-bf` 没归零，或者机器不支持该 cap 被回退了）。

---

**一句话总结**：`ref_frames_in_dpb`（= `maxNumRefFramesInDPB` / `-dpb_size`）是“DPB 能装几帧”，单向 B 改变的是“DPB 里**需不需要**为未来帧留位置”——答案是**不需要**，所以 `N_reorder = 0`、`has_b_frames = 0`、surface 不涨；但它自己反过来要求 DPB 里**至少有两个不同的过去帧**，所以 `-refs` 别压到 1，给 2 就够。

---
## 🧑‍💻 User

ref_frames_in_dpb+uni-b frame和真b帧有什么区别，为什么延迟低这么多？

---
## 😀 默认助手 (7)

## 一句话：真正的 B 帧之所以延迟高，是因为**因果性**，并非性能问题

单向 B 帧和真正的 B 帧的差别，并不在于“B 帧够不够快”，关键在于 **L1 参考列表里的那个“未来帧”到底存不存在**。

$$
\text{L1} \in \{\text{未来}\} \;\Rightarrow\; \text{必须等未来帧到达} \;\Rightarrow\; \text{延迟无法避免}
$$

$$
\text{L1} \in \{\text{过去}\} \;\Rightarrow\; \text{编码/解码顺序} = \text{显示顺序} \;\Rightarrow\; \text{延迟项根本不存在}
$$

所以，两者的差距是 **0 与 80 ms** 的差距，而非 3% 和 5% 的差距。这种差距直接决定“能不能用”。

---

## 编解码顺序：差异的实际位置

### 真正的 B 帧（`-bf 3`，`frameIntervalP = 4`）

```
显示顺序:  I0  B1  B2  B3  P4  B5  B6  B7  P8
编码顺序:  I0  ────P4────  B1  B2  B3  ──P8──  B5  B6  B7
                     ↑
           要编码 B1，必须先有 P4 的重建帧
```

- **编码端**：收到第 4 帧之前，第 1~3 帧只能留在队列里，导致输出落后 **3 帧**。（FFmpeg 里就是这个公式：`delay = frameIntervalP - 1`）
- **解码端**：要显示 B1，必须先解出 P4；因此 B1~B3 必须在缓冲里等，导致重排深度 **2 帧**，编码器会把这个值写进码流（HEVC 写在 SPS 的 `sps_max_num_reorder_pics` 中，FFmpeg 侧对应 `has_b_frames = 2`）。

### 单向 B（`-bf 0 -unidir_b 1`）

```
显示顺序:  I0  B1  B2  B3  P4  B5  B6  B7  P8
编码顺序:  I0  B1  B2  B3  P4  B5  B6  B7  P8   ← 完全一致
                ↘  ↙     ↘  ↙
              L0/L1 都指向过去的帧
```

- 编码端一旦收到第 $n$ 帧就能立刻编完，流水线深度 **0**；
- 解码端解出即显示，`N_reorder = 0`，DPB 里没有“等未来帧”的槽位。

---

## 延迟到底差多少：四项拆解

| 延迟项 | 真正的 B 帧（`-bf 3`）| 单向 B |
|---|---|---|
| ① 编码端等待未来帧（流水线深度）| `frameIntervalP − 1` = **3 帧** | **0** |
| ② 编码器 lookahead | 通常还要额外 +2~4 帧 | 不需要 |
| ③ 解码端重排（`has_b_frames`）| **2 帧** | **0** |
| ④ 码率突发 → 接收端 jitter buffer | 需要加大：P4 先出，B1~B3 成簇补发，VBV 和抖动缓冲都得放宽 | 无突发，码率平滑 |
| **合计** | **5 帧起** | **0** |

换算成时间（一帧 = $\frac{1}{\text{fps}}$）：

| 帧率 | 真正的 B 帧 | 单向 B |
|---|---|---|
| 60 fps | $5 \times 16.7\text{ ms} \approx$ **83 ms** | **0 ms** |
| 30 fps | $5 \times 33.3\text{ ms} \approx$ **167 ms** | **0 ms** |

这就是“低这么多”的原因：**它不是延迟小一点，而是同样多的延迟在两端各付一次**（$N_{enc}$ + $N_{dec}$），而云游戏/会议的总延迟预算本来就只有 40~60 ms；额外多出 83 ms 的纯重排延迟，会让总延迟直接翻倍以上。

> 注意 ④：重排会让输出码率不平滑（P 先出、B 后补一组），因此 VBV 和接收端抖动缓冲都必须放大；网络层再放大一点，延迟还会被再抬高一截。这一项常被忽略，但真实管线里往往比理论帧数更致命。

---

<details>
<summary><b>那为什么单向 B 的收益只有真正的 B 的 1/4~1/5？（3~4% vs 10~20%）</b></summary>

真正的 B 帧比单向 B 多出两块收益，而单向 B 一块都拿不到：

1. **未来帧带来的新信息**
   L1 取未来帧时，能看到当前帧“还看不见”的内容，尤其是**遮挡解除区（unocclusion）**和匀速/加减速运动；这些地方只能靠后向参考来补。单向 B 的两个参考都在同一方向，信息高度冗余，只吃到了“双预测平均降残差”这一项。

2. **层级参考结构（`b_ref_mode` 的 pyramid）**
   真正的 B 允许 B 帧本身作参考，形成时间分层：底层小步长、高层大步长，比特按时间层分配。这部分往往是层级 B 相对 IPPP 增益的**主要来源**（10% 量级），单向 B 完全没有它——它的 B 帧只是“在 P 的位置上换了个预测方式”，GOP 结构还是平的。

所以本质是：**单向 B ≈ 升级版 P 帧，只买到 bi-prediction；真正的 B 额外买到未来信息和时间分层。**

</details>

---

## 顺手澄清一个常见误解

`ref_frames_in_dpb`（`maxNumRefFramesInDPB` / `-dpb_size`）**本身不产生任何延迟**。

它只是“DPB 里允许放几帧”的**容量上限**，不是“要等几帧”。把 `dpb_size` 从 2 调到 6，延迟一帧都不会增加，代价只是显存和 ME 带宽。

延迟的唯一来源是**重排方向**：

$$
\text{延迟} \propto N_{\text{reorder}},\qquad N_{\text{reorder}} = f(\text{参考方向})
$$

所以：

- 真正的 B 帧：`N_reorder > 0` → 必须加大 `dpb_size`（否则 B 帧插不进去）→ 延迟与容量**同时**上升，两者被绑死；
- 单向 B：`N_reorder = 0` → `dpb_size` 给 2 就够，容量和延迟**解耦**。

这也解释了上一轮那个结论：为什么“`dpb_size` + uni-B”这套组合的延迟是 0——**不是因为它 ref 少，而是因为 ref 全在过去**。

---

## 折中档位（如果真想要 B 的增益）

| 方案 | 编码延迟 | 解码重排 | @60fps 合计 | 收益 |
|---|---|---|---|---|
| uni-B (`-bf 0 -unidir_b 1`) | 0 | 0 | **0 ms** | 3~4% |
| `-bf 1` + 禁用 b_ref | 1 帧 | 2 帧（NVENC 声明值）| ≈ 50 ms | ~8% |
| `-bf 3` + `b_ref_mode middle` | 3 帧 | 2 帧 | ≈ 83 ms+ | 10~20% |
| `-bf 5` + UHQ | 5 帧+ | 2 帧+ | 120 ms+ | 最高（但对标 x265 Slow）|

按延迟预算选档位，而不是按“哪个参数更好”选。**预算 < 1 帧时，uni-B 是唯一选项**；预算 ≥ 2 帧时，`-bf 1` 通常是性价比拐点。
