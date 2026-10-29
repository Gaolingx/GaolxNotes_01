# NVENC qpDeltaMap 用法与延迟介绍

## 🧑‍💻 User

nvenc的NV_ENC_PIC_PARAMS::qpDeltaMap + NV_ENC_QP_MAP_DELTA有什么用，会增加延迟吗

---
## 😀 默认助手 (7)

## 先说结论

`qpDeltaMap` 本身**不会增加编码延迟**。它只是每帧随 `nvEncEncodePicture()` 一起提交的一张小查找表，不需要额外的 surface、额外的同步点或额外的提交次数。真正的延迟风险来自**你如何生成这张表**，以及它与码率控制之间的相互影响。

---

## 这个功能在做什么

<details>
<summary><b>API 语义与几个容易出错的地方</b></summary>

它提供了两个字段，分别控制两件事：

| 位置 | 字段 | 作用 |
|---|---|---|
| `NV_ENC_RC_PARAMS` | `qpMapMode` | **解释方式**，在 `nvEncInitializeEncoder` 时设置，一次会话内保持不变 |
| `NV_ENC_PIC_PARAMS` | `qpDeltaMap` + `qpDeltaMapSize` | **每帧的数据**，`int8_t*` 数组，按光栅扫描顺序 |

`qpMapMode` 的取值按 `nvEncodeAPI.h` 原样：

```c
NV_ENC_QP_MAP_DISABLED = 0x0  /* qpDeltaMap 被忽略 */
NV_ENC_QP_MAP_EMPHASIS = 0x1  /* 值当“强调等级”用，仅 H.264 支持 */
NV_ENC_QP_MAP_DELTA    = 0x2  /* 值当 QP 偏移用 —— 你要的这个 */
NV_ENC_QP_MAP          = 0x3  /* 文档明说不支持 */
```

在 `DELTA` 模式下要特别注意两点：

1. **粒度按编码块计算，并不按像素计算。** 原文是“per MB for H264, per CTB for HEVC and per SB for AV1”，数组大小必须是 `picWidthInMbs * picHeightInMbs`（H.264）、`picWidthInCtbs * picHeightInCtbs`（HEVC）或 `picWidthInSbs * picHeightInSbs`（AV1）。
   - 1080p H.264：`120 × 68 = 8160` 字节
   - 1080p HEVC（CTB64）：`30 × 17 = 510` 字节
   - 1080p AV1（SB128）：`15 × 9 = 135` 字节
2. **`qpDeltaMap` 直接叠加在码率控制算出的 QP 上**，不做第二套估值。文档的原话是“This QP modifier will be applied on top of the QP chosen by rate control”。所以它和 AQ、lookahead、multi-pass 叠加使用，最终 QP 仍会受 `minQP`、`maxQP` 约束。
3. 数组需要在 `nvEncEncodePicture` 返回后继续有效，直到该帧编码完成，因为提交是异步的；不要过早复用该缓冲区，否则可能把 A 帧的 ROI 用到 B 帧上。这是使用这个 API 最常见的 bug。

</details>

## 它主要用于什么

它的用途是**在不改变码率控制目标的前提下，对区域做质量再分配**，也就是 ROI。这是硬件编码器上唯一可行的“按区域控质”手段，因为 NVENC 不开放 CU、MB 级参数：

- 人脸、字幕、屏幕文字等区域：给负 delta（如 -6 ~ -10），把码率集中到这些区域
- 明显不重要的背景、平坦天空：给正 delta（如 +4 ~ +8），节省码率
- 屏幕内容、文档共享、云游戏 UI 小字
- 反过来做 privacy blur、马赛克时统一降低某区域的码率

对 CBR 低延迟流，它常用于提高主观质量：总码率不变，只调整码率的分配方式。

因此典型效果是**在相同码率下获得更好观感，或者让码率更节省**；编码速度不会明显提升，也可能略有下降。

## 关于延迟：逐项拆解

### 编码器侧：约等于零

- 不需要额外提交，也不需要阻塞等待（不引入 submit 与 sync 的往返）
- 不需要额外的 `NV_ENC_INPUT_PTR` surface，也没有显式的 GPU 内存分配
- 数据量小：1080p H.264 每帧 8KB，4K 也只有 ~32KB，PCIE 开销可以忽略
- 该表在编码器本来就要执行的 QP 决策阶段被读取，不增加流水线级数

### 真正的风险在客户端

| 风险点 | 说明 | 缓解 |
|---|---|---|
| **map 必须在提交前就绪** | 如果 ROI 由**下游**分析产生（如后一帧的人脸检测、场景分析），等待这个结果就会造成帧级串行化，延迟增加一个推理周期 | 让 ROI 来源使用当前帧或更早的帧；或只在 I 帧、固定间隔更新 ROI |
| **生成 map 本身** | 若 mask 在 GPU 上，则需要 D2H 读回并同步，这才是常见实际增加的延迟 | 在 GPU 侧把 mask 直接降采样成 MB 网格；或将 map 缓冲使用 pinned memory 与 ring buffer |
| **缓冲生命周期** | 异步编码期间反复用 map 会出错，但为“保险”而每帧同步完成则会真的增加延迟 | 使用深度为 `maxEncodeDelay` 的 map 缓冲环 |
| **与 RC 相冲突** | delta 太大（±15 以上）会让帧级 RC 频繁回调：需要 O 尺寸时码率超，之后又压低 QP，表现为质量来回震荡；在 CBR 低延迟下更明显，还可能造成 VBV 欠载并触发跳帧（跳帧才是真正增加延迟/卡顿的因素） | delta 控制在小范围（经验值 ±10 内），并在 AQ 关闭或调低时使用 |
| **`EMPHASIS` 模式** | 文档明确写了它会 **override 码率控制**，可能违反 bitrate、VBV 约束，且**仅 H.264** | 低延迟 CBR 场景不要用 EMPHASIS，用 DELTA |

### 一句话总结

> NVENC 侧不增加延迟；如果你的 ROI 生成路径和帧提交在同一个关键路径上，那增加的延迟全是你的管线造成的，和这个 API 无关。

## 实现参考

```c
/* init 时：确定解释方式 */
enc_cfg.rcParams.qpMapMode = NV_ENC_QP_MAP_DELTA;

/* 每帧：填表 + 传指针 */
memset(&pic, 0, sizeof(pic));
pic.version       = NV_ENC_PIC_PARAMS_VER;
pic.qpDeltaMap    = mb_map;                                   /* int8_t*, 必须活到编码完成 */
pic.qpDeltaMapSize = mbs_w * mbs_h;                           /* 1080p H.264 = 8160 */
nvEncEncodePicture(session, &pic);
```

在 FFmpeg 里要动手的话，改的点是 `libavcodec/nvenc.c` 中填 `NV_ENC_PIC_PARAMS` 的位置：在 `nvEncEncodePicture` 调用前给 `pic_params.qpDeltaMap` 和 `qpDeltaMapSize` 赋值，并保证这块内存在 `NV_ENC_OUTPUT_PTR` 完成事件到来前不被改写。

## 补充：几个容易忽略的限制

- **`qpMapMode` 在 init 时设置，不能逐帧切换模式**，只能在模式内逐帧更换指针；如果要更换模式，需要通过 `nvEncReconfigureEncoder`。
- **HEVC、AV1 的粒度很粗**：CTB 64×64、SB 128×128，ROI 边缘会呈格子状。如果需要细粒度 ROI，应使用 H.264（16×16）或调整 HEVC 的 CTB 配置。
- **`EMPHASIS` 与 AQ（spatial、temporal）互斥**，文档明确写了不支持两者同时开启。
- 任何情况下都要确认 map 的字节数与 `picWidthInMbs × picHeightInMbs` 精确匹配；如果尺寸偏小，可能读取越界，而且不会报错，只会静默产生错误的 QP。

---
## 🧑‍💻 User

这个和temporal_aq、spatial_aq有什么区别？

---
## 😀 默认助手 (7)

## 一句话区分

- **spatial AQ / temporal AQ**：由编码器**自己查看画面**后计算出的块级 QP 调整。你只提供一个开关和强度，剩下由硬件自动完成。
- **`qpDeltaMap` + `NV_ENC_QP_MAP_DELTA`**：由**你告诉编码器**的块级 QP 偏移。编码器不检查内容，完全相信这张表。

前者是**自动的内容自适应**；后者是**外部先验或语义信息注入**，例如人脸、字幕、UI、隐私区等编码器永远无法推断的东西。

---

## 三者各自根据什么决定 QP

| | 判据 | 想解决的问题 |
|---|---|---|
| **spatial AQ**<br>`rcParams.enableAQ` | 帧内空间复杂度（块内方差、纹理、边缘能量） | 平坦区容易出 banding/blocking，人眼对平坦区的劣化敏感；把码率从“看不出成本的平坦区”转移到“有细节的区域” |
| **temporal AQ**<br>`rcParams.enableTemporalAQ` | 时域预测残差、与参考帧的差异 | 静态且预测良好的块，多给比特是浪费，因此提高 QP；时域活动大或新出现的区域降低 QP。此外还能抑制静态区因 QP 波动而闪烁 |
| **qpDeltaMap (DELTA)** | 你提供的 `int8` 表，一元素一块 | 表达“重要性”，而不是“复杂度”，例如 ROI、文字清晰度、隐私区模糊 |

一个具体的差异：一段**纹理很复杂的背景**中，三者的反应是

- spatial AQ：复杂度高，**会降低** QP，投入更多比特
- temporal AQ：如果它在动，**会降低** QP；如果是静止的树林，则**会提高** QP，因为预测残差小
- qpDeltaMap：**完全取决于你的意图**。你认为它是背景，就给正 delta；你认为是焦点，就给负 delta。AQ 无法做出这种判断

反过来，**一张人脸**在 static 且低复杂度时，spatial AQ 可能觉得“平坦区域可以多压”，恰好把肤色区压出 banding；而 `qpDeltaMap` 可以直接把整块人脸强制降低 8 个 QP 来保住肤色。这就是它存在的核心价值：**AQ 永远不可能知道那是脸**。

## 在 QP 决策流程中的位置

```
帧级 RC 目标 QP（CBR/VBR 按 VBV 和目标比特率算出）
   ├─ + spatial AQ 块级调制（强度由 aqStrength 1..15 控制）
   ├─ + temporal AQ 块级调制
   └─ + qpDeltaMap 块级 delta        ← 你的 mask，文档明确写 "applied on top of
                                          the QP chosen by rate control"
   └─ clamp 到 [minQP, maxQP]
```

$$QP_{最终} = \mathrm{clamp}\left(QP_{RC} + \Delta_{AQ} + \Delta_{map},\ minQP,\ maxQP\right)$$

**关键点：AQ 是 RC 的内部流程，map 是叠加在 RC 结果之上的外部修正。** 因此两者会叠加；这既是带来便利的原因，也是最容易出问题的地方。

## 对比表

| 维度 | spatial / temporal AQ | qpDeltaMap (DELTA) |
|---|---|---|
| 决策者 | 硬件分析 | 你的算法 |
| 粒度 | 编码器内部子块级 | MB（H.264）、CTB（HEVC）、SB（AV1） |
| 客户端成本 | **零** | 每帧都要生成表，见下一条 |
| 表大小 | 无 | 1080p H.264 8160 B，HEVC 510 B，AV1 135 B |
| 延迟 | 零 | 零（编码器侧），但生成 mask 会占用时间 |
| 语义感知 | 无 | 有 |
| 需硬件能力 | `NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ`（temporal 需查询） | 无单独 cap，一般均有 |
| 可调强度 | spatial 有 `aqStrength`（1 低，15 激进，4 bit）；**temporal 没有强度参数** | 逐块任意 delta |
| 随场景自适应 | 自动 | 需要你自己跟踪；镜头切换后 mask 不更新会向错误区域分配比特 |
| 文档互斥约束 | — | `EMPHASIS` 模式：**不支持 AQ，且仅 H.264**；`DELTA` 无此限制 |
| 码率守恒 | 大致守恒 | 大幅 delta 会违反 VBV/目标比特率 |

## 组合使用时的三个主要问题

<details>
<summary><b>展开：AQ 与 map 叠加时的失败模式</b></summary>

**1. 重复计算，导致码率过度倾斜**

两者对同一个判断采用相同的思路：复杂或重要的区域分配更多比特，平坦区域分配更少比特。如果你给一个**本来就复杂**的区域再标一个 `-10` delta，就等于叠了两次加权：该块 QP 被压到下限，码率被吸过去，帧内其他区域反而变差。测码率分布时便会看到“ROI 区极清晰、周边糊成一片”。

**缓解方式**：把 mask 当作**只表达 AQ 之外的信息**的载体。用 mask 之前，先问一句“这块的复杂度 AQ 是不是已经处理了”。是，就别再标 delta。

**2. 在平坦区叠加正向 delta**

`+8` 叠上 spatial AQ 本来就给平坦区的高 QP，很容易顶到 `maxQP`，banding 与块效应被放大到无法接受，尤其在天空、墙面上。

**3. 参数互相掩盖**

`aqStrength` 默认由 driver 自动选，你无法判断当前强度；它和 mask 的叠加效果就变得不可预测。要做稳定、可回归的 ROI 曲线，**最干净的办法是关掉 AQ，只用 map**；这样 $QP$ 的每一项你都完全掌控，调参时也容易单变量处理。

</details>

## 如何选择

| 场景 | 建议 |
|---|---|
| 通用转码或录制，没有语义信息 | 只开 **spatial AQ**（`aqStrength` 从 6~8 开始试），temporal AQ 视硬件支持再开 |
| 视频会议、屏幕共享，需要保证文字和 UI 可读 | **关掉或调低 AQ，再加 DELTA mask** 标在文字与 UI 区。屏幕内容大多是高对比度平坦色块，spatial AQ 对它的判断经常是反的 |
| 有 face、ROI 检测，想提观感 | AQ 照开（处理复杂度），mask **只标“人脸”，不标“复杂”**，delta 幅度控制在 ±10 以内 |
| 需要可复现、可严格约束的 ROI 曲线（比如直播质量 SLA） | **关掉 AQ + 只用 DELTA map**，控制范围可以预测 |
| 想做隐私马赛克或降质区 | 只能用 mask，AQ 无法表达这种意图 |
| 想获得“绝对指定 QP” | 不要用 `EMPHASIS`（只支持 H.264，会违反 VBV，而且与 AQ 互斥）；`NV_ENC_QP_MAP`(=3) 文档明确写了**不支持**。正确做法是 `constQP` RC + `DELTA` 表 |

## 关于延迟（接上一轮的问题）

- **AQ 完全零延迟**：它在硬件编码管线内部，不增加 surface、内存或提交次数。
- **temporal AQ 本身也不增加延迟**，但——如果你为了它去开 `enableLookahead`，**真正增加的延迟来自 lookahead 的 `lookaheadDepth` 帧**，而不是 AQ。低延迟场景不要误开 lookahead。
- **qpDeltaMap 编码器侧零延迟**，代价全在“你得在提交前把 mask 备好”。这一点和 AQ 的差别是根本性的：AQ 不需要任何客户端工作，mask 需要你实时算出每一帧的每一块。

**所以延迟排序是：AQ ≈ 0 < qpDeltaMap（编码器侧 0，但管线端取决于你的 mask 生成）< lookahead（真实增加 N 帧）**。

## 一个容易忽略的细节

`qpMapMode` 和 `enableAQ` 同在 `NV_ENC_RC_PARAMS` 里，都是**初始化时定、会话内不变**的。所以“先试 AQ 再试 map”这类对比实验需要重建 encoder（或用 `nvEncReconfigureEncoder`），不能每帧切换模式——上次说的“只能换指针不能换模式”，在这里同样适用。
