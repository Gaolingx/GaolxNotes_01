# nvenc numRefL0设置作用与延迟影响分析

## 🧑‍💻 User

nvenc设置numRefL0>0有什么用？相比固定的全1，这会增加延迟吗，会导致一些编码延迟的波动吗

---
## 🧑‍💻 User

nvenc设置numRefL0>0有什么用？会增加延迟吗

---
## 😀 默认助手 (7)

## 结论先说

`numRefL0 > 0` **本身不会引入结构性（重排序）延迟**，因为它只是把“可用参考帧数的上限”从 1 放宽，用到的仍然全是**已经编码并重建完成的过去帧**。真正的代价是**编码器每帧的搜索/RDO 工作量略微增加**，以及 DPB 和码流语法上的变化。在 GPU 有余量的情况下，它带来的延迟与抖动变化基本可以忽略；只有当你本来就把 NVENC 吞吐压到极限时，才会出现尾延迟和抖动变差。

---

<details>
<summary><b>一、这个字段到底是什么（避免一个常见误解）</b></summary>

在 `nvEncodeAPI.h` 里，H.264 的 `NV_ENC_CONFIG_H264` 和 HEVC 的 `NV_ENC_CONFIG_HEVC` 各有一对字段：

```c
NV_ENC_NUM_REF_FRAMES numRefL0;  /**< [in]: Specifies max number of reference frames in
                                      reference picture list L0, that can be used by hardware
                                      for prediction of a frame. */
NV_ENC_NUM_REF_FRAMES numRefL1;
```

枚举定义：

| 值 | 含义 |
|---|---|
| `NV_ENC_NUM_REF_FRAMES_AUTOSELECT = 0` | 由驱动/固件自己决定参考帧数 |
| `NV_ENC_NUM_REF_FRAMES_1..7 = 1..7` | 显式指定上限 |

三个关键点：

1. 它是**“max number that can be used”，是上限，不是强制**。编码器在逐帧 RDO 中自行决定实际引用几帧、引用哪几帧。所以调大并不等于每帧都真的做 7 帧运动估计。
2. 它和 `maxNumRefFrames`（= **DPB 大小**）是两个不同的概念。`maxNumRefFrames` 决定 DPB 里保留多少帧，`numRefL0` 决定 L0 列表里**最多允许多少帧参与预测**。要使 `numRefL0 > 1` 有意义，DPB 必须足够大。
3. 需要硬件能力位 `NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES` 支持（老 GPU/老驱动上可能直接初始化失败或被钳制），要先用 `NvEncGetEncodeCaps` 查一次。它同时会写进码流语法（`num_ref_idx_l0_active_minus1`），因此对端的硬解也必须能接受更大的 DPB。

另外注意：`numRefL1` 只在 B 帧（`frameIntervalP > 1`）下才有实际意义；纯 P 帧低延迟配置里它是 0/1。
</details>

<details>
<summary><b>二、numRefL0 > 0（比如 2~4）有什么用</b></summary>

**1. 码率/质量：主要是 BD-rate 收益**

多参考帧让编码器能在**更早的帧**里找匹配块。收益分布很不均匀：

| 内容类型 | 收益 |
|---|---|
| 静态背景 + 局部运动、屏幕共享/PPT、动画、慢平移镜头 | **明显**（大面积区域可以在 N 帧前的参考里找到几乎完美的匹配） |
| 周期性/往复运动（风扇、挥手、遮挡后重现、闪烁） | 明显 |
| 高运动、手持抖动、噪声重的实拍 | 很小，甚至被噪声淹没 |
| 快速场景切换 | 几乎为 0 |

典型量级是几个百分点 BD-rate（同质量省码率，或同码率提 PSNR），收益随参考数递减（1→2 的收益最大，2→3、3→4 迅速变小）。

**2. 错误恢复：这个用途常被忽略**

配 `NvEncInvalidateRefFrames()` 做丢包恢复时，官方文档明确建议把 `maxNumRefFrames`（DPB）设大，让编码器保留一批**旧参考帧**：

> *The client is recommended to set `maxNumRefFrames` to a large value so that encoder can keep a backup of older reference frames in the DPB and can use them for motion estimation when the newer reference frames have been invalidated.*

但“DPB 里存着”只是能存，**能不能真的拿来预测，取决于 `numRefL0`**。如果 `numRefL0 = 1`，最近一帧作废后编码器往往只能被迫打一个 IDR（瞬时大帧 → 码率尖峰 → 延迟尖峰）；有 `numRefL0 = 2~4` 才能平滑降级到稍旧的参考帧，不打断 GOP。所以对“弱网 + 低延迟”场景，`numRefL0 > 1` 的价值更多在**抗抖动的鲁棒性**，而非压缩率。

**3. 与 B 帧/时域分层配合**

开了 B 帧（`frameIntervalP ≥ 2`）或用时域分层（`maxTemporalLayers`）时，L0/L1 的长度必须够用，否则 B 帧的“双向”形同虚设，分层 GOP 的压缩效率也会被截断。注意头文件还有一条约束：`maxTemporalLayers ≥ 2` 时要求 `maxNumRefFrames ≥ (maxTemporalLayers - 2) * 2`。
</details>

---

## 三、会增加延迟吗？——要区分两种“延迟”

$$T_{\text{e2e}} = T_{\text{capture}} + T_{\text{queue}} + T_{\text{encode}} + T_{\text{network}} + T_{\text{decode}} + T_{\text{display}}$$

`numRefL0` **只可能影响 $T_{\text{encode}}$，而且只影响其中的可变部分**。

<details>
<summary><b>A. 结构性延迟：0，完全可以排除</b></summary>

结构性延迟来自**输出顺序 ≠ 显示顺序**，也就是需要攒帧重排。产生它的是：

- `frameIntervalP > 1`（引入 B/P 重排，延迟 $\approx$ 重排深度 $\times$ 帧间隔）
- `enableZeroReorderDelay` 没开
- 编码器侧的 lookahead / 多 pass / 时域滤波（`tfLevel`，文档要求 `frameIntervalP >= 5`）

而 P 帧的 L0 列表里**只可能是显示顺序更早的帧**，编码一帧就能立即输出，输出顺序 = 显示顺序，**不存在任何“等未来帧”的等待**。DPB 里多留几帧“过去的帧”只影响内存占用和 level 约束，不产生输出延迟。

所以：`numRefL0` 从 1 调到 4，**重排序延迟增量 = 0**。它和“开 B 帧”完全不是一个性质的东西。
</details>

<details>
<summary><b>B. 计算延迟：小幅增加，且因为是“上限”而不会均匀增加</b></summary>

- 由于它是 **cap 而不是 mandate**，编码器只在“确实有收益”时才去翻更早的参考帧。静止画面里第一帧就匹配上了，几乎不额外花钱；只有复杂运动帧会真的把 2~4 帧都用上。
- 开销是硬件 ME（运动估计）+ RDO 的搜索空间变宽，量级通常是**每帧编码时间的个位数到十几个百分点**，且随分辨率和内容变化。NVENC 的 `NvEncEncodePicture` 是异步提交的，只要编码器还有余量，这点增量会被流水线吸收，**端到端延迟看不出变化**。
- 真正的风险在**容量**而非时延：如果你在 4K/高帧率/多路并发下已经把 NVENC 用满（消费级卡尤其明显），任何额外开销都会让队列积压，表现为延迟整体抬升 —— 这才是需要担心的部分。
</details>

### 横向对比表

| 手段 | 结构性延迟 | 计算开销 | 抖动风险 | 压缩收益 |
|---|---|---|---|---|
| `numRefL0 = 1 → 2..4` | **无** | 小~中（内容相关） | 低（余量足时）/ 中（吞吐打满时） | 几个 % BD-rate |
| `frameIntervalP > 1`（B 帧） | **有，随重排深度线性增长** | 中 | 中 | 大 |
| lookahead / 多 pass / `tfLevel` | **有，且是最大的一项** | 大 | 高 | 大 |
| `maxNumRefFrames`（DPB）增大 | 无 | 内存/带宽 | 低 | 间接（配合 LTR/抗丢包） |

---

## 四、会导致编码延迟的“波动”吗？

先要明确：**流媒体链路里的抖动主要来自码控和内容，不是来自参考帧数。** 抖动的主因排序通常是：

1. **码控（RC）**：场景切换强制 IDR / 大 P 帧 → VBV 抽干 → 后续帧被压 QP → 一帧特别大、一帧特别小，这才是延迟抖动的头号来源；
2. **内容复杂度**（每帧编码时间本身就不恒定）；
3. **GPU 资源竞争**（同一卡上的 CUDA / 图形 / 其他编码会话、显存带宽争用）；
4. **提交端背压**（采集→编码队列没做丢帧策略）。

`numRefL0` 在其中的作用有**两个相反的方向**：

- **变差的一面**：搜索空间变大 → 单帧编码时间的内容相关性更强（复杂运动帧耗时相对更多），**尾延迟（P99）分布略微变宽**。若编码器已经接近吞吐上限，就会出现“某几帧超时 → 队列积压 → 延迟爬升 → 追帧 → 回落”的典型锯齿，这才是用户能感知到的“延迟波动”。
- **变好的一面**：预测更准 → P 帧码率更平稳 → **VBV 更不容易抽干 → 码控引起的延迟尖峰反而减少**。在 CBR + 严格 VBV 的低延迟链路上，这一项有时比开销增加更显著。

净效果取决于你在哪一侧：**有余量 → 抖动变化基本淹没在噪声里；打满吞吐 → 尾延迟变差。** 这也是为什么“平均值”看不出来，必须看 P99。
</details>

---

## 五、给你的实操建议

<details>
<summary><b>按场景选配置</b></summary>

**实时互动（视频会议 / 云游戏 / 远程桌面 / WebRTC / 连麦）**

```text
tuningInfo        = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY
frameIntervalP    = 1            // 纯 P，无重排
enableZeroReorderDelay = 1
numRefL0          = NV_ENC_NUM_REF_FRAMES_1
numRefL1          = NV_ENC_NUM_REF_FRAMES_1
maxNumRefFrames   = 1..4（配 NvEncInvalidateRefFrames 抗丢包时可放大）
rcMode            = CBR + 小 VBV / enableMinQP
tfLevel           = OFF（文档要求 frameIntervalP>=5，本来也用不了）
multiPass         = OFF
```

这里 `numRefL0 = 1` 是合理的默认值：额外的几个 % 压缩率不值那点尾部延迟风险和更长的错误传播链。**只有在“弱网 + 已启用参考帧作废恢复”这一条上，值得把 `numRefL0` 提到 2~3**，因为它能把 IDR 兜底换成“退到旧参考帧”。

**直播 / 录屏 / 转码存文件（可接受一点延迟）**

```text
numRefL0 = 2..3, numRefL1 = 1..2（配 B 帧时）
maxNumRefFrames >= numRefL0，且满足 level 的 DPB 限制
```

在屏幕共享、PPT、动画类内容上这是近乎免费的画质提升。

</details>

<details>
<summary><b>必须检查的约束</b></summary>

- `NvEncGetEncodeCaps` 查 `NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES`，不支持就别设 >1，否则可能初始化报 `NV_ENC_ERR_INVALID_PARAM` 或被静默钳制；
- `numRefL0 ≤ maxNumRefFrames`（DPB）；
- 注意 **level 的 MaxDpbMbs 限制**：4K 高分下 DPB 帧数会被 level 卡得很紧，`numRefL0` 想设 4 而 level 不升是做不到的；
- 多参考帧会拉长错误传播链，必须配合 intra refresh / LTR，且对端硬解 DPB 要够；
- 已启用 `intraRefresh` 时，多参考帧下 refresh 的收敛行为会略有差异，需实测确认无残留花屏。

</details>

<details>
<summary><b>怎么验证“到底生效了没有 + 抖动有没有变差”</b></summary>

1. **确认生效**：它是 cap，有时根本不会被用满。可靠做法是对比 `numRefL0=1` vs `=3` 两路输出的**同帧大小 / 平均 QP / 解码后 PSNR-VMAF**；若码率与质量曲线几乎重合，说明编码器压根没用额外参考，调整它没有意义。
2. **量化延迟**：在 `NvEncEncodePicture` 提交处打时间戳，在 `NvEncLockBitstream` 拿到对应输出处打时间戳，取 `δ = t_out - t_in`，统计 **P50 / P95 / P99 和 σ**。只看均值会掩盖问题。
3. **加压力测试**：把分辨率/帧率/并发路数推到目标峰值再测，否则吞吐余量会把开销完全吸收，你会得出“零影响”的错误结论。
4. **A/B 时保持其他变量固定**（尤其 `maxNumRefFrames`、VBV、QP 范围），否则收益/代价会被码控噪声盖住。

</details>

---

**一句话总结**：`numRefL0` 是“L0 参考帧数上限”（0 = 自动，1~7 = 显式），调大只是**允许**用更多过去的帧做预测，收益在压缩率和丢包恢复鲁棒性上，**不增加任何结构性延迟**；延迟的固定部分不变，只可能有少量计算开销，抖动风险来自 GPU 吞吐余量而非这个字段本身 —— 互动场景保持 1 是稳妥选择，直播/录屏场景用 2~3 性价比很高。
