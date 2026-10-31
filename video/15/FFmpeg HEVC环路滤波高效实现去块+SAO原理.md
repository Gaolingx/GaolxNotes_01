# FFmpeg HEVC环路滤波高效实现去块+SAO原理

## 🧑‍💻 User

ffmpeg环路滤波是如何高效实现HEVC 去块 + SAO的？

---
## 😀 默认助手 (7)

# FFmpeg 的 HEVC 环路滤波：设计与实现

FFmpeg 的 HEVC 解码器（`libavcodec/hevc/`）并没有"先解完整个帧，再扫一遍去块，再扫一遍 SAO"。它把这三级滤波**合并到 CTB 解码流程中，并延后一级 CTB 来触发**，这使滤波几乎不产生额外的帧遍历、缓存也没有额外开销。下面按"先调度、再去块、再 SAO"的顺序说明。

---

## 一、核心思想：一次遍历、延后一级触发

解码完一个 CTB 就调用 `ff_hevc_hls_filters()`，但它**不给当前 CTB 做滤波**，而是给"左上、上、左"这三个 CTB 做：

```c
void ff_hevc_hls_filters(HEVCLocalContext *lc, const HEVCLayerContext *l,
                         const HEVCPPS *pps, int x_ctb, int y_ctb, int ctb_size)
{
    int x_end = x_ctb >= pps->sps->width  - ctb_size;
    int y_end = y_ctb >= pps->sps->height - ctb_size;

    if (y_ctb && x_ctb) ff_hevc_hls_filter(lc, l, pps, x_ctb - ctb_size, y_ctb - ctb_size, ctb_size);
    if (y_ctb && x_end) ff_hevc_hls_filter(lc, l, pps, x_ctb,            y_ctb - ctb_size, ctb_size);
    if (x_ctb && y_end) ff_hevc_hls_filter(lc, l, pps, x_ctb - ctb_size, y_ctb,            ctb_size);
}
```

**为什么必须延后一个 CTB？**

1. 强滤波会横跨边界写 3–4 个像素，因此去块 CTB$(x,y)$ 的左/上边界时会**写进邻居 CTB 的像素**；
2. SAO 在语义上必须在"整帧去块完成之后"才能作用，而 SAO 的 EO 模式要读上下左右各 1 个像素的邻域，也就是说它读到的邻居像素必须是**被邻居自己的去块改完之后**的值。

差一拍（对角 CTB 已完成）恰好保证：CTB$(x-1,y-1)$ 的所有邻域 CTB 都已解码，且已对其施加的去块不会再改动它。再加上 `ff_hevc_hls_filter()` 内部的 `x_delay`/`x_sao` 偏移，SAO 又会进一步延后（色度子采样时为 2 个 CTB），这样在读取邻居样本时**不需要任何"是否已滤波"的判断分支**。

> 这一点是 FFmpeg 与"整帧两趟扫"实现（多数硬件、部分参考软件风格实现）最大的性能差异：去块和 SAO 的读写都落在同一个 CTB 行带的工作集里，基本命中 L1/L2。

---

## 二、去块滤波（DBF）

去块分成两个完全解耦的阶段：**推导 bS（含大量比较、候选列表访问）** 和 **像素滤波（纯 SIMD 运算）**。

### 1. bS 在解析期就算好，滤波期零依赖

`ff_hevc_deblocking_boundary_strengths()` 在 `hls_transform_unit()` 里被调用（TU 边界一确定就算），此时 `tab_mvf`（PU 级 MV/参考索引）、`cbf_luma` 都刚写进缓存，因此**访问模式最友好**。结果写进两个紧凑数组：

```c
l->horizontal_bs[((x) + (y) * l->bs_width) >> 2] = bs;   // 每 4 像素一条边，1 字节
l->vertical_bs  [((x) + (y) * l->bs_width) >> 2] = bs;
```

关键优化点：

| 手段 | 效果 |
|---|---|
| bS 只有 $\{0,1,2\}$ 三个值，每 4 像素占 1 字节 | 一个 64×64 CTB 的边信息只有 16×16 字节，整行常驻 L1 |
| `curr->pred_flag == PF_INTRA \|\| top->pred_flag == PF_INTRA → bs = 2`，`cbf` 非零 → 1 | 绝大多数边在 2 次布尔运算内直接定值，不进 MV 比较 |
| `mvf_equal()` / `mvf_run_equal()` 用 `memcmp` 一次比 4/7/12/15 个 `MvField` | 静态背景、skip 大片区域一次判完，省掉逐像素的 `boundary_strength()` 调用 |
| 只有 `log2_trafo_size > log2_min_pu_size && !is_intra` 才扫 TU 内部 PU 边界 | 帧内块直接跳过 |
| slice/tile 边界先判 `boundary_flags`，`loop_filter_across_*_enabled_flag == 0` 时直接置 0 | 省掉整条边的滤波 |
| `ff_hevc_tile_boundary_bs()` 单独处理延迟的 tile 边界（`lc->tile_bs_defer`） | 让 tile 并行/WPP 下 bS 也能算对而不用同步 |

bS = 0 的边在后面会被**完全跳过**（连像素都不读），这是跳过区域最主要的加速来源。

### 2. 每条边只过滤一次：只做本 CTB 的"左"和"上"

`deblocking_filter_CTB()` 里两个循环的起点体现了这一点：

```c
for (y = y0; y < y_end; y += 8) {
    // 垂直边（luma）：从 x0 开始，x0 == 0 时从 8 开始（跳过图像左边界）
    for (x = x0 ? x0 : 8; x < x_end; x += 8) { ... }
    if (!y) continue;
    // 水平边（luma）：从 x0 - 8 开始，跨 CTB 的那 8 列块由本 CTB 负责
    for (x = x0 ? x0 - 8 : 0; x < x_end2; x += 8) { ... }
}
```

注意 `x_end2` 的写法：`x_end2 = x_end; if (x_end2 != width) x_end2 -= 8;`。它让相邻两个 CTB 分工"每一段 8 像素的边恰好被一个 CTB 访问一次"（本 CTB 处理左邻最右边的 8 列块，最右边的 8 列块留给右邻处理），既不会重复滤波，也不需要在循环里做边界判断。

### 3. 一次函数调用处理两条 4 像素子边

luma 滤波每次调用传 `bs0`/`bs1` 和 `tc[2]`，也就是把同一个 8 像段的上下两条 4 像素边合并处理（`y` 与 `y+4` 两组）：

```c
const int bs0 = l->vertical_bs[(x + y           * l->bs_width) >> 2];
const int bs1 = l->vertical_bs[(x + (y + 4)     * l->bs_width) >> 2];
if (bs0 || bs1) {            // 两条都 0 → 这一列/行段完全不碰像素
    qp = (get_qPy(.., x-1, y) + get_qPy(.., x, y) + 1) >> 1;
    beta  = betatable[av_clip(qp + beta_offset, 0, MAX_QP)];
    tc[0] = bs0 ? TC_CALC(qp, bs0) : 0;
    tc[1] = bs1 ? TC_CALC(qp, bs1) : 0;
    s->hevcdsp.hevc_v_loop_filter_luma(src, linesize, beta, tc, no_p, no_q);
}
```

- $\beta$ 与 $t_C$ 查表：`betatable[qp+offset]`、`tctable[qp + 2(bs-1) + tc\_offset]`，零除法零浮点；
- QP 用 `qp_y_tab` 的**两侧平均值**：$QP = \left(QP_p + QP_q + 1\right) \gg 1$；
- `beta_offset/tc_offset` 在解析期已按 CTB 存进 `l->deblock[ctb]`（按 CTB 地址索引，支持 slice 级覆盖），滤波期直接用；
- 色度只处理 `bs == 2` 的边（符合规范），并且用固定步长 `8 << hshift` / `8 << vshift` 一次跨过整块；
- 色度 QP 映射用 14 项小表 `qp_c[]`，不做公式。

### 4. PCM / transquant_bypass 的"双入口"技巧

规范要求 PCM 块和 TQB 块不能被滤波。代码没有在每个段里加分支，而是准备了两套函数指针：

```c
if (pcmf)  s->hevcdsp.hevc_v_loop_filter_luma_c(src, ..., beta, tc, no_p, no_q);
else       s->hevcdsp.hevc_v_loop_filter_luma  (src, ..., beta, tc, no_p, no_q);
```

`pcmf`（`pcm_loop_filter_disabled || transquant_bypass_enable_flag`）为假时，`no_p/no_q` 是哑参，走纯 SIMD 版本；只有存在 PCM/TQB 的流才走会查 `is_pcm` 的 C 版本。**常见路径上没有一个 per-segment 的 `if (is_pcm)`**。

### 5. SIMD 覆盖

`dsp.c` 里每个 kernel 都有 8/9/10/12 bit 四份模板实例（`#define BIT_DEPTH ... #include "dsp_template.c"`），再被各架构覆盖：

- x86：`hevc_v/h_loop_filter_luma_{8,10,12}` 有 `sse2 / ssse3 / avx`，chroma 有 `sse2 / avx`（luma 宽 8 像素、要 4+4 抽头，AVX 已够；chroma 只有 4 抽头）；
- aarch64/arm：NEON 版本同等覆盖；
- 每个 kernel 一次处理 8 像素（luma 两条 4 像素边 + 强弱滤波判决），判决与滤波在同一次向量寄存器生命周期内完成，不落地到内存。

结论：单个 8 像素段大约几十个周期，而 bS=0 时是几条指令的纯数组读。

---

## 三、SAO

### 1. 两种模式，都用"5 项 offset 表 + 索引"实现

**BO（band offset）**：先把 5 个 offset 散到一个 32 项表里，然后逐像素查表加饱和：

```c
int offset_table[32] = { 0 };
for (k = 0; k < 4; k++) offset_table[(k + sao_left_class) & 31] = sao_offset_val[k + 1];
d = offset_table[src[x] >> 5];          // 8bit：取高 5 bit 定 band
dst[x] = av_clip_pixel(src[x] + d);
```

正是这个"32 字节查表"让 SIMD 实现极其高效：8bit 上 `sao_band_filter` 用 `pshufb` 做 16 路并行查表 + `paddusb/psubusb` 饱和加减，一段 16 像素只要几条指令。

**EO（edge offset）**：按 `eo_class` 选 0°/90°/45°/135° 方向，比较当前像素与两个邻居并把结果累加成 0–4 的索引，再 `src[x] + sao_offset_val[edge_idx]`。同样只有 5 项 LUT、无分支（用比较结果做算术累加）。

### 2. 按宽度特化的函数表（消除尾部处理）

```c
int tab = sao_tab[(FFALIGN(width, 8) >> 3) - 1];   // width → 8/16/32/48/64
s->hevcdsp.sao_band_filter[tab](src, src, stride, stride, offset, band_pos, width, height);
s->hevcdsp.sao_edge_filter[tab](src, dst, stride, offset, eo_class, width, height);
```

`dsp.c` 里 `sao_band_filter[0..4]` / `sao_edge_filter[0..4]` 在 C 版是同一函数，但各架构提供了**5 个宽度专用 kernel**（`ff_hevc_sao_band_filter_{8,16,32,48,64}_8_avx2` 等）。宽度作为立即数进入 kernel ⇒ 完全展开、无循环、无尾部标量处理；正好覆盖 CTB=64 的各种边缘裁剪宽度。

### 3. 难点：EO 要读"未滤波"的邻居 → 行列边缘缓存

EO 的语义是：读取的邻居值必须是**去块后、SAO 前**的原始重建值。但邻居 CTB 可能已经被 SAO 处理过了。FFmpeg 的方案不是重跑或整帧备份，而是**增量备份两行/两列**：

```c
/* 保存本 CTB 的上/下 1 行、左/右 1 列 到 sao_pixel_buffer_h / _v */
copy_CTB_to_hv(l, sps, src, stride_src, x0, y0, width, height, c_idx, x_ctb, y_ctb);
```

- `sao_pixel_buffer_h[c_idx]` 每 CTB 存 2 行（上传给下面、下传给上面），`sao_pixel_buffer_v` 存 2 列；
- 邻域 CTB 用完后立刻把 `sao->type_idx[c_idx] = SAO_APPLIED` 打上标记。邻居取样本时据此二选一：

```c
src_idx = (CTB(l->sao, x_ctb, y_ctb-1).type_idx[c_idx] == SAO_APPLIED);
memcpy(dst1 + pos, src1[src_idx] + pos, width << sh);
```

即"邻居已 SAO → 从缓存行读；未 SAO → 直接从帧缓冲读"。于是每个 CTB 只多付出 $2 \times (W+C) \times \text{bytes}$ 的拷贝（对 64×64 约 256B），代价是 O(周长) 而不是 O(面积)。

### 4. 边界与"还原"：`sao_edge_restore`

当 slice/tile 边界不允许跨边界滤波（`lfase`/`no_tile_filter`），或者图像边界时，EO 读到的邻域样本本来就是不可用的。FFmpeg 的做法是：

1. 先在 `dst = lc->edge_emu_buffer` 里拼出一个带 1 像素 halo 的完整 CTB 副本（halo 值从邻居或缓存行取，边界处补自身值）；
2. `sao_edge_filter` 在副本上做滤波；
3. `sao_edge_restore[restore]` 把**不应被跨边界修改的那一/几行（列）**回写/修正；
4. 最后 `restore_tqb_pixels()` 把 PCM/TQB 块的原值还原（`memcpy` 回原像素）。

```c
s->hevcdsp.sao_edge_filter[tab](src, dst, stride_src, offset, eo_class, width, height);
s->hevcdsp.sao_edge_restore[restore](src, dst, stride_src, stride_dst, sao, edges,
                                     width, height, c_idx, vert_edge, horiz_edge, diag_edge);
```

`restore = no_tile_filter || !lfase` 时选 `restore_1`，否则选 `restore_0`（近乎空操作）——又是一种"用函数指针代替分支"的写法。同时只有 `pps->transquant_bypass_enable_flag || pcm_loop_filter_disabled` 时才走"先拷到 `edge_emu_buffer`、过滤、再还原"的重路径，否则 BO 可以直接原地 `src == dst` 滤波。

### 5. 跳过开关

`ff_hevc_hls_filter()` 开头把 `skip_loop_filter` 映射到 `AVDISCARD_{ALL,NONKEY,NONINTRA,BIDIR,NONREF}`，命中就整帧/整片段跳过 DBF+SAO；`sps->sao_enabled == false` 时连 SAO 的调度都不进。对 `-skip_loop_filter all` 这类场景是零成本。

---

## 四、职责与代码位置对照

| 层次 | master（7.x） | 5.x / 4.x 旧布局 |
|---|---|---|
| 调度入口 | `libavcodec/hevc/filter.c`：`ff_hevc_hls_filter`、`ff_hevc_hls_filters` | `libavcodec/hevc_filter.c` |
| bS 推导 | `ff_hevc_deblocking_boundary_strengths`、`ff_hevc_tile_boundary_bs` | 同函数（`hevc_filter.c`） |
| 去块执行 | `deblocking_filter_CTB` | 同函数 |
| SAO 组织 | `sao_filter_CTB`、`copy_CTB_to_hv`、`restore_tqb_pixels` | 同函数（在 `hevc_filter.c`） |
| 像素 kernel 模板 | `libavcodec/hevc/dsp_template.c`（由 `hevc/dsp.c` 按 bit depth 实例化） | `libavcodec/hevcdsp_template.c` / `hevcdsp.c` |
| SIMD 分派 | `libavcodec/x86/hevc/dsp_init.c` + `*.asm`；`aarch64/hevc/dsp_init.c`（NEON） | `libavcodec/x86/hevcdsp_init.c` 等 |
| 每 CTB 状态 | `l->vertical_bs` / `l->horizontal_bs` / `l->bs_width`、`l->deblock[ctb]`、`l->sao_pixel_buffer_h/_v`、`l->is_pcm`、`l->cbf_luma` | 结构体名不同，语义一致 |

---

## 五、把"高效"归结为 6 条

1. **零额外遍历**：滤波在 CTB 解码循环里被"顺路"触发，重建样本尚在 L1/L2；只延后一个 CTB 换取无分支的样本可用性。
2. **计算与访存分离**：bS 在解析期用最热的元数据算出并压成 1 byte/4px；滤波期只剩查表 + SIMD 像素运算。
3. **每条边恰好一次**：只做本 CTB 的左/上边界，并用 `x_end2 -= 8` / `x0 - 8` 把 8 像素段在相邻 CTB 间无重叠划分。
4. **每段先看 bS 再碰像素**：`if (bs0 || bs1)` 让大片 skip/静态区域零像素访问；MV 相同用 `memcmp` 批量判定。
5. **SAO 用 O(周长) 的边缘缓存**解决"邻居样本已被修改"的语义问题，而不是备份 CTB 或二次遍历；宽度特化 kernel 消除尾部处理。
6. **用函数指针/表驱动替代内部分支**：PCM/TQB 走 `_c` 版、`sao_edge_restore[0/1]`、`sao_*_filter[0..4]`、`skip_loop_filter` 全帧跳过——常见路径上没有"特殊情形"判断。

<details>
<summary>与 VTM / 硬件实现的思路差别（点开）</summary>

- VTM 追求清晰与可验证，按 CTB 逐级调用 `xDeblockCU`/`xSao`，大量成员访问与虚调用，且以"整帧两次扫描（先 DBF 后 SAO）"组织；
- 硬件/ASIC 通常把 DBF 与 SAO 做成**流水线级**（DBF 输出经行缓存直接进 SAO），需要额外的行缓冲 SRAM，但吞吐最高；
- FFmpeg 的选择是"软件里最省访存、并且对 WPP/tile 线程友好"，用一点调度延迟和少量 `sao_pixel_buffer` 换掉整帧遍历与同步。

</details>

如果你要针对某一块深挖（例如 `sao_edge_filter` 的 SSSE3/AVX2 汇编怎么用 `pshufb` 做 5 项 offset 选择，或 deblocking 强弱滤波的 SSE2 实现），我可以继续展开。
