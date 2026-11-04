# NVENC CTU 是 HEVC 编码树单元 硬件流水线固定实现

## 🧑‍💻 User

nvenc ctu是什么，是硬件电路实现的吗

---
## 😀 默认助手 (7)

## 结论速览

| 问题 | 答案 |
|---|---|
| CTU 是什么 | HEVC/H.265 的**编码树单元**（Coding Tree Unit），是编码器的基本处理粒度，取代了 H.264 的 16×16 宏块 |
| NVENC 里的 CTU | 硬件流水线的**工作单元**。NVENC 的 HEVC 实际使用 **32×32 CTU**（AV1 用 64×64 superblock），且**不向用户暴露** |
| 是硬件电路实现的吗 | **是**。CTU 的划分决策、预测、变换、量化、环路滤波和熵编码全部由硬连线 RTL 完成，属于固定功能 ASIC，和 CUDA core/SM 是并列的独立引擎 |

---

## 1. CTU 到底是什么

CTU 是 HEVC 引入的概念，本质是“**一块正方形区域 + 一棵四叉树**”：

$$
\text{CTU} \xrightarrow{\text{四叉树递归}} \text{CU} \to \text{PU} + \text{TU}
$$

- **CTU**：最大 $64\times64$（SPS 里 `log2_min_luma_coding_block_size_minus3` 与 `log2_diff_max_min_luma_coding_block_size` 共同决定，也可以配成 32 或 16）
- **CU**：CTU 按四叉树递归切成 $64\to32\to16\to8$，深度 0~3
- **PU**：预测单元（$2N\times2N$、$2N\times N$、AMP 非对称等）
- **TU**：残差变换单元，自己还有一棵残差四叉树（4×4 ~ 32×32 整数 DCT）

和 H.264 的关键差别：H.264 宏块固定 16×16，HEVC 的 CTU 是**自适应可变大小**，这既是 HEVC 压缩率提升的主要来源之一，也是**硬件成本的主要来源之一**（要大块 SRAM 线缓冲 + 复杂的候选遍历逻辑）。

> 1080p 一帧在 32×32 CTU 下是 $\frac{1920}{32}\times\frac{1088}{32}=60\times34=2040$ 个 CTU（高度按 32 对齐补到 1088）。

---

## 2. NVENC 里 CTU 是“硬件电路”吗

**是，而且是纯硬连线的。** 理解这点要区分 GPU 上的两类单元：

- **SM / CUDA core**：可编程，跑着色器、CUDA kernel
- **NVENC / NVDEC**：固定功能引擎，独立前端 + 流水线 + 片上 SRAM + 显存接口，只会“编码”这一件事

在 NVENC 内部，一个 CTU 被光栅扫描顺序送进一条固定流水线，每一级都是专用电路：

| 流水线级 | 实现方式 |
|---|---|
| 帧内预测 | 硬连线的角度预测/DC/Planar 计算单元 |
| 帧间预测 | 硬连线运动估计，**固定或半固定的搜索模式和有限搜索范围**，不是全搜索 |
| 变换/量化 | 专用整数 DCT + 量化器阵列，支持 4/8/16/32 |
| 反量化/重建 | 硬件重建回路（编码器需要重建帧做参考） |
| 去块滤波 + SAO | 硬连线滤波器 |
| CABAC | 硬件熵编码状态机（HEVC 用 CABAC） |
| **CTU 四叉树划分决策** | **硬连线的比较器/状态机 + 早期终止启发式** |

**最关键的“CTU 是不是硬件”的点在最后一行**：软件编码器（x265）会真正做 RDO——对每个 CTU 遍历各种划分深度、每种 PU、每种 TU，算失真 + $\lambda \cdot$ 码率，选代价最小的。这个组合爆炸在硬件里既放不下也来不及（还要满足实时/固定延迟）。NVENC 的做法是：

1. 用 **SAD/SATD + 近似代价** 而不是完整 SSD + 真实码率
2. **自上而下或自下而上的早停启发式**，只试少量候选划分
3. 搜索范围、参考帧数、候选模式数都**写死在 RTL 里**，或者只由少数几个预设档位开关

所以 NVENC 的 `-preset p1..p7` **不是换算法**，而是拨动硬件的几个开关（子像素精度、搜索范围、是否多 pass、是否开 lookahead/B 帧自适应等），可调粒度远小于 x265 的 `--ctu 64 --rd 6 --subme 7`。

### 唯一“不完全硬连线”的部分

- **固件**：NVENC 片上有小处理器跑驱动加载的固件，负责调度、码率控制（CBR/VBR/HRD）、slice 组织。所以是“**硬连线数据通路 + 固件控制**”的混合体。码控策略、GOP 结构这类东西改固件就能改，但 CTU 的电路本身改不了。
- **SM 侧配合**：lookahead、temporal AQ、空间 AQ map、缩放/色彩转换这些**可以在 SM 上用 CUDA/NPP 算**，把结果喂给 NVENC 作为提示（hint）。这是 NVENC 唯一“软件可介入决策”的通道——但它给的是 **帧级/宏块级的 AQ 强度提示**，而不是逐 CTU 的划分树。

---

## 3. 为什么硬件要把 CTU 做成 32×32（而不是 64×64）

<details>
<summary>展开：面积/功耗/带宽的权衡（点开）</summary>

1. **线缓冲与 SRAM**：CTU 越大，帧内预测、去块滤波跨 CTU 边界时需要的行缓冲越大；$64\times64$ 的 CTU 需要的片上 SRAM 面积和功耗显著高于 $32\times32$。
2. **决策复杂度**：$64\times64$ 的四叉树有 4 层深度、85 个可能 CU，硬件要在有限周期内出决策，只能砍候选。
3. **延迟确定性**：显卡编码要保证固定吞吐（例如 8K 实时），流水线级数和每 CTU 周期数必须是常数上界。
4. **显存带宽**：参考帧读取是大头，CTU 越大、搜索越自由，带宽需求越不可控。

代价也很明确：**同码率下压缩效率不如 x265/SVT-HEVC**。CTU 偏小 + 划分决策启发式 + 搜索受限，这三点是 NVENC “快但不够省” 的根本原因。粗略经验值：1080p 同画质，NVENC HEVC 比 x265 `medium` 大约多 15~30% 码率（内容相关，动画/平坦画面差距更大，因为大平坦区域最吃大 CTU）。

横向对比（供参考，各家实现细节不完全公开）：

| 编码器 | HEVC CTU | AV1 superblock |
|---|---|---|
| NVENC | 32×32 | 64×64 |
| Intel Quick Sync | 64×64 | 64×64 |
| AMD VCN | 64×64 | 64×64 |
| x265 | 默认 64×64（可配） | — |
| SVT-HEVC/AV1 | 64×64 | 64/128 |

</details>

---

## 4. 怎么自己验证（很实用）

CTU 大小不通过 API 暴露，但**写在码流的 SPS 里**，可以直接从码流读出来：

```bash
# HEVC：看 log2_min_luma_coding_block_size_minus3 和 log2_diff_max_min_luma_coding_block_size
ffmpeg -i input.hevc -bsf:v trace_headers -f null - 2>&1 | grep -i "coding_block"

# AV1：看 use_128x128_superblock
ffprobe -show_frames input.ivf 2>&1 | grep -i superblock
```

换算公式：

$$
\text{CTU size} = 2^{\,3 + \text{log2\_min\_luma\_coding\_block\_size\_minus3} + \text{log2\_diff\_max\_min\_luma\_coding\_block\_size}}
$$

NVENC 输出的 HEVC 通常是 `min=0, diff=2` → $2^{3+0+2} = 32$；x265 默认 `min=0, diff=3` → $64$。

编码时你**没有任何办法**改 NVENC 的 CTU：

```bash
# 能调的只有档位，不是块大小
ffmpeg -i in.mp4 -c:v hevc_nvenc -preset p7 -tune hq \
       -rc vbr -cq 23 -b:v 0 -spatial_aq 1 -temporal_aq 1 \
       -rc-lookahead 32 out.mp4
```

需要真正的 $64\times64$ CTU 和完整 RDO，只能用软件编码器（或 Intel/AMD 的硬编）。

---

## 一句话总结

**CTU 是 HEVC 的编码树单元；在 NVENC 里它是硬连线流水线的工作粒度，划分决策由固定功能的启发式电路完成，不可由用户或 CUDA 程序改写——你能调的只有预设档位和 AQ/lookahead 这类“提示”，而这正是 NVENC 速度极快、但压缩效率不如 x265 的结构性原因。**
