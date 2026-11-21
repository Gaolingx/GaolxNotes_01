# NVENC非参考P帧原理与生效条件

## 🧑‍💻 User

nonref-p是nvenc内部行为吗 nonref-p生效的前提条件是什么

---
## 😀 默认助手 (7)

我会先核对 NVENC 头文件里 `enableNonRefP` 的原文，避免凭记忆答错。

头文件是一整行紧凑文本，直接读取超限，因此我用浏览器把 `enableNonRefP` 附近的注释抓取出来。

核对完 NVENC 头文件和 FFmpeg 的实现后，答案很明确。

## 结论速览

| 问题 | 答案 |
|---|---|
| 是不是 NVENC 内部行为？ | **是**。应用只设一个 1-bit 开关，选哪几帧、怎么做 DPB 维护、怎么打参考标记，全在固件里 |
| 最硬的前提 | `NV_ENC_INITIALIZE_PARAMS::enablePTD = 1`（让编码器自己决定图像类型） |
| 其他前提 | Turing+ 硬件 / P-only 或极低延迟 GOP / 初始化时设置 |
| 是否要求 lookahead | 头文件**未**要求（对比 `disableIadapt`/`disableBadapt` 明确写了 “only with lookahead”） |

---

## 一、`enableNonRefP` 是纯内部行为

`nvEncodeAPI.h` 里这一位是 `NV_ENC_RC_PARAMS` 中的一个 bitfield，原文只有一句：

```c
uint32_t enableNonRefP : 1;
/**< [in]: Set this to 1 to enable automatic insertion of non-reference P-frames
             (no effect if enablePTD=0) */
```

关键词是 **automatic insertion**：你只是“允许”编码器插入非参考 P 帧，**插入位置、插入比例、什么时候不插，全由 NVENC 的 GOP/Picture-Type-Decision 模块自己决定**。API 层面你没有任何手段指定“第 N 帧要/不要做非参考帧”。

真正的实现在 ASIC 固件里，包括三件事：

1. 判断当前 P 帧是否值得做成“后续帧不会参考它”；
2. 在码流里正确打标 — H.264 是 `nal_ref_idc = 0` 的 P slice，HEVC/AV1 是该帧不进任何后续帧的 RPS（参考帧集）；
3. 维护 DPB / 滑动窗口 / `frame_num` 连续性（非参考帧不推进 `frame_num`，因此连续插入数量天然受限）。

所以它不是“新增一种帧类型”的语法扩展，H.264/HEVC 本来就有非参考帧的合法表达，NVENC 只是把“该不该插”这个决策自动化了。好处也很实际：解码端（尤其是 DPB 紧张的硬解）不必把这一帧留在参考缓存里，省参考帧内存和带宽。

<details>
<summary>为什么 NVIDIA 把它归到“低延迟 + 抗丢包”而不是“省码率”</summary>

官方 *NVENC Programming Guide* 的 tuning 特性表里，`Non-reference P frames` 和 `Long term reference pictures`、`Intra refresh`、`Force IDR` 一起，列在 **Ultra-low latency / Low latency** 那一行，并统一脚注为：

> “These features are useful for error recovery during transmission across noisy mediums.”

也就是说它的主要价值是**弱网/丢包后的恢复**（非参考帧丢了不会污染后续帧），省码率是副产品。
</details>

---

## 二、生效的前提条件（按“卡得有多死”排序）

### 1. 必须 `enablePTD = 1` —— 唯一在文档里明写的硬条件

```c
NV_ENC_INITIALIZE_PARAMS init = {0};
init.enablePTD = 1;   // 图像类型决策交给编码器；为 0 时 enableNonRefP 完全无效
```

如果你自己做 PTD（`enablePTD = 0`，自己填 `pictureType` / `refPicFlag` / `displayPOCSyntax`），这一位**直接失效**，没有任何报错，`NvEncInitializeEncoder` 照样返回成功。这是最常见的“设了没效果”原因。FFmpeg 的 nvenc wrapper 就是无条件 `enablePTD = 1` 的：

```c
ctx->init_encode_params.enablePTD = 1;
...
if (ctx->nonref_p)
    ctx->encode_config.rcParams.enableNonRefP = 1;
```

### 2. 硬件与驱动代际

社区口径和 Turing 时代的发布说明是 **Turing（TU10x/TU11x/TU116）及以后**（Ampere / Ada / Hopper / Blackwell）。但要注意两点：

- 当前头文件注释里**已经不再写**任何代际限制（我核对了 `n9.1.23.1` 和 master 两个版本，措辞一致，都没有 “Turing and above”）；
- `NV_ENC_CAPS` 枚举里**没有**对应的能力查询项（搜不到 `NON_REF` 相关 caps），所以你**无法在初始化前探测**支持性。

结论：不支持的卡上它就是静默无效。判断是否支持只能靠实测码流（见第三节）。

### 3. 码流结构要“配得上”

非参考 P 帧只在 **P-only / 极低延迟 GOP** 里有意义：

- `frameIntervalP = 1`（纯 P，无 B）或最多 uni-directional B；
- 配合 `NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY` / `LOW_LATENCY`；
- 典型组合：CBR + 极小 VBV（单帧级 buffer）+ `zeroReorderDelay` + `enableNonRefP`。

在 B 帧层级 GOP（`frameIntervalP > 1`、hierarchical B）里，层级结构本身已经在做“部分帧不被参考”的事，再叠加非参考 P 帧通常看不到明显收益。

### 4. 会话级参数，初始化时设

它是 `NV_ENC_RC_PARAMS` 的字段，属于 `NvEncInitializeEncoder` 的配置。运行中改变只能走 `NV_ENC_RECONFIGURE_PARAMS`（且会触发 GOP/RC 状态变化）或重建会话——不是逐帧参数。

### 5. 头文件版本够新

该字段随 Turing / Video Codec SDK 9.0 引入。用旧 header 编译会没有这个成员；用新 header 编译但在旧驱动上跑，结构体版本协商可能让它被忽略。

---

## 三、怎么验证真的生效了

NVENC 不会通过任何 API 告诉你“我刚插了一个非参考帧”，只能看码流：

```bash
# 1. 先让 NVENC 编出来
ffmpeg -hwaccel cuda -i in.mp4 \
       -c:v h264_nvenc -preset p1 -tune ull -rc cbr -b:v 4M \
       -nonrefp 1 -zerolatency 1 out.h264

# 2. H.264：找 nal_ref_idc == 0 的 P slice（非参考 P）
ffmpeg -i out.h264 -c copy -bsf:v trace_headers -f null - 2>&1 \
  | grep -E "nal_ref_idc|slice_type"
```

判定标准：

- **H.264**：出现 `nal_ref_idc = 0` 且 `slice_type` 为 P → 生效。
- **HEVC**：该 P 帧不出现在后续任何帧的 short-term RPS（`num_negative_pics` / `num_positive_pics`）里，编完即出 DPB。
- 未生效的典型现象：全程 `nal_ref_idc = 1`，帧类型序列和关掉时逐帧一致。

---

## 四、两个常见混淆

| 容易混的东西 | 区别 |
|---|---|
| `zeroReorderDelay` | 控制**重排序延迟**（`num_reorder_frames=0`），和 `enableNonRefP` 是两个独立 bit。FFmpeg 的 `-zerolatency` 只映射前者，`-nonrefp` 才映射后者，两者要一起用 |
| `-tune ull` / `NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY` | 这是 preset 层面的“调优倾向”，可能内部已经开了 nonref-P；此时再单独 `enableNonRefP=1` 属于重复设置，不会“再强一点” |

如果你要的是“保证某一帧绝对不被参考”，NVENC 不给你这个控制权；能拿到的确定性手段是 `forceIDR` / intra-refresh，而不是 nonref-P。
