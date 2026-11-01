# llama.cpp q8激活精度分析够不够用取决于权重量化位宽

## 🧑‍💻 User

llamacpp的qX_q8K fused dequant gemm kernel是不是用int8 tensorcore绕开fp16mmfp32acc限制（int8xint8+int32）
然后decode压根没用到张量核心（是cuda core跑的 vec_dot比gemv快）

---
## 😀 默认助手 (7)

## 先说结论

你两个判断**都对了一半**；准确说：第一条**方向对，但动机和命名要修正**；第二条**基本成立，但有两个例外**。

| 你的说法 | 判定 | 修正 |
|---|---|---|
| MMQ（qX×q8 融合 dequant GEMM）用 int8 tensor core | ✅ 在 Turing+ (sm_75) 上确实如此 | 但 CUDA 侧是 `q8_1`，不是 `q8_K`；且这是**后来的新增路径**，老版本只有 dp4a |
| 目的为“绕开 fp16 mm fp32 acc 限制” | ⚠️ 半对 | tensor core 从 Volta 起就是 fp32 累加了，fp32 acc 从来不是限制；真正动机是 **fp16 输入舍入** + 吞吐 ×2 + smem 减半 |
| decode 完全不用 tensor core，靠 cuda core 的 vec_dot | ✅ 大体成立 | batch ≤ 8 时成立；但 batch ≥ 9 的 decode、以及开启了 FA 的 attention 都还是用 tensor core |

以下均基于我刚才实际拉取的 `llama.cpp` master 源码（行号为当前 master）。

---

## 1. MMQ 现在是“双路”内核：dp4a（CUDA core）和 int8 MMA（tensor core）

关键证据：`mmq-vec-dot.cuh` 里每种类型都有两个实现，例如

- `ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(...)` → `__dp4a`，**CUDA core 的 INT8 点积指令**
- `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(...)`、`q8_0_q8_1_mma`、`q8_1_q8_1_mma`、`q8_0_16_q8_1_mma` → tensor core

MMA 分支内部（`mmq-vec-dot.cuh` 的 `#else` 即 Turing 分支，约 202–277 行）用的是：

```cpp
typedef tile<16, 8, int> tile_A;   // int！
typedef tile< 8, 8, int> tile_B;
typedef tile<16, 8, int> tile_C;
...
mma(C, A[n][k01/QI8_0], B);
```

而 `mma.cuh` 里 `int` 版本的 `mma()` 就落到：

```
mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {...}   // mma.cuh:1074
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {...}   // mma.cuh:1096
```

所以：**确实是 int8×int8→int32 的 MMA，accumulator 是精确整数**，然后在寄存器里做

```cpp
sum[...] += C.x[l] * dA[n][l/2][k01/QI8_0] * dB[l%2];   // 每 32 元素的 block scale 在 fp32 里补
```

即“整数 matmul + fp32 逐 block 缩放”，这也是为什么 k-quant 走这条路**没有** fp16 反量化精度损失：4/5/6-bit 反量化后的值本来就能被 int8 精确表示，先 round 到 fp16 才是那个多余误差（Q6_K、q3_K 在 fp16 下尤其脏）。

选路逻辑：`mmq.cuh:183–203` 用 `amd_mfma_available || amd_wmma_available || turing_mma_available` 决定用 mma 的 tile 尺寸表还是 dp4a 的 `tile_x_sizes`。也就是说 **sm_75 及以上（NVIDIA）默认走 int8 MMA，dp4a 只留给 Pascal 及更老 / pre-WMMA 的 AMD**。

### 更正“绕开 fp16 mm fp32 acc 限制”这个动机

- `mma.sync ... .f32.f16.f16.f32`（fp16 输入、**fp32 累加**）从 Volta 就有，`mma.cuh:1313` 里就在用。fp32 累加不是瓶颈，也从来没被限制过。
- int8 这条路真正的收益是三条：
  1. **精确性**：整数累加无舍入，反量化值在 int8 里可精确表示；
  2. **吞吐**：Turing/Ampere/Ada 上 INT8 TC 是 FP16 TC 的 **2×**；
  3. **smem / dequant 成本**：权重 tile 在 smem 里是 8-bit 而不是 16-bit，反量化只需要移位/查表拼 int8，不需要做 fp16 的乘加，smem 占用和带宽都减半（这对 MMQ 的 `MMQ_TILE_NE_K=32` tile 大小和 occupancy 很关键）。

注意 `mmq.cuh:9` 那句注释 `MMQ_DP4A_MAX_BATCH_SIZE 64 // ... when FP16 tensor cores are available` 已经有点“史前遗迹”味道了，别按字面理解。

---

## 2. Decode：确实不进 tensor core，但要说清是哪一层

`ggml-cuda.cu` 的派发顺序（约 2051–2080 行）：

```cpp
if (ggml_cuda_should_use_mmvf(...))  → ggml_cuda_mul_mat_vec_f   // 非量化权重 F16/BF16/F32
if (ggml_cuda_should_use_mmvq(...))  → ggml_cuda_mul_mat_vec_q   // 量化权重
if (ggml_cuda_should_use_mmq(...))   → ggml_cuda_mul_mat_q       // 量化权重，大 batch
// 否则 dequant→fp16/bf16 + cuBLAS
```

阈值：

- `mmvq.cuh:2` → `MMVQ_MAX_BATCH_SIZE 8`
- `mmvf.cuh:2` → `MMVF_MAX_BATCH_SIZE 8`

所以 **`ne11 ≤ 8` 的 decode 走 `mmvq.cu` + `vecdotq.cuh`**：一个 warp 负责一行输出，用 `vec_dot_q_cuda`（`__dp4a` / `__fmaf` 组合，f32 累加）边反量化边点积，**纯 CUDA core，没有任何 `mma.sync`**。这一步你说的没错。

`ggml_cuda_should_use_mmq`（`mmq.cu:439`）里对 Turing+ 是直接：

```cpp
if (turing_mma_available(cc)) return true;
```

→ 所以 **NVIDIA 现代卡上，量化类型的 GEMM 只要 batch ≥ 9 就一律是 int8 tensor core 的 MMQ**，cuBLAS/fp16 路线在量化权重上其实只是 fallback（`GGML_CUDA_FORCE_CUBLAS`、`smpbo < 48 KiB`、类型不在支持列表里）。

### 但“decode 完全不用 tensor core”有两个例外

1. **batch 是 8 还是 9 之差**：投机解码 / 多并发 serving 时 decode step 的 batch 经常 > 8，那一步的 FFN 就走 int8 TC 了。
2. **Attention 本身**：`fattn-mma-f16.cuh`（head dim 合适且开 `-fa` 时）是用 **fp16 tensor core** 做 $QK^\top$ 和 $PV$ 的，这部分跟 decode/prefill 无关，一直在用 TC。所以严格说：**decode 的 linear/FFN 不用 TC，attention 可能用。**

### “vec_dot 比 gemv 快”这句话的准确版本

不存在一个和它打擂台的 `gemv`。`mmvq` 本身就是“融合反量化的 GEMV”：它比“先把整块权重反量化成 fp16 落显存，再调 cuBLAS GEMV”快，是因为后者要**额外写+读一遍 fp16 权重**，显存流量直接翻 2–4 倍（4.5 bpw → 16 bit）。

算一下就明白为什么 TC 在 decode 毫无意义：每权重 2 FLOP、约 0.56 byte，$I \approx 3.5\ \text{FLOP/byte}$，远低于 roofline 拐点，kernel 完全被 HBM 带宽支配；MMA 要求 $n \ge 8$ 才有利用率，$n=1$ 时理论上直接浪费 7/8；而且还要多付一次 `quantize_row_q8_1_cuda` 的 kernel 和数据往返。

顺带一个有趣的细节：`ggml-cuda.cu:2053` 的注释说 MMVF 的设计目标就是“在**没有** tensor core 的卡上、或者 src0 很瘦（典型就是 attention 的 KQV）时比 cuBLAS 的 GEMV 快”——所以“vec_dot 比 gemv 快”在代码里是**明写的设计取舍**，你没记错，只是它比较的对象是 “dequant + cuBLAS”。

---

## 3. 术语修正：CUDA 上没有 `q8_K`

- `q8_K`（带 super-block scale + bsums）是 **CPU 侧** `ggml_vec_dot_q4_K_q8_K` 那套，主要为了 VNNI/AVX 上处理 q4_K/q5_K 的 min-offset 项。
- CUDA 侧只有 `q8_0` / `q8_1`：`quantize.cuh` 里是 `quantize_row_q8_1_cuda` 和 MMQ 专用的 `quantize_mmq_q8_1_cuda`（多了一个 `ds` 交织布局，就是 `MMQ_Q8_1_DS_LAYOUT_D4 / _DS4 / _D2S6`，用来在 fp32 里补 offset 项，取代 bsums 的角色）。

所以“qX_q8K fused dequant gemm”更准确的叫法是 **MMQ：qX × q8_1（在 smem 里再落到 int8 tile）**。

<details>
<summary><b>一张 dispatch 速查表（NVIDIA，当前 master）</b></summary>

| 条件 | 内核 | 计算单元 | 数据类型 |
|---|---|---|---|
| `ne11 ≤ 8`，量化权重 | `mmvq.cu` / `vecdotq.cuh` | CUDA core (`__dp4a`, fma) | 融合反量化 + f32 acc |
| `ne11 ≤ 8`，F16/BF16/F32 权重 | `mmvf.cu` | CUDA core | f32 acc |
| `ne11 ≥ 9`，量化权重，sm_75+ | `mmq.cu` + `mmq-vec-dot.cuh` (`_mma`) | **INT8 tensor core** (`s8.s8.s32`) | 整数 acc + f32 block scale |
| 同上，Pascal / 无 TC / `FORCE_MMQ` | `mmq.cu` (`_dp4a`) | CUDA core | 整数 acc + f32 block scale |
| 量化类型但 MMQ 不可用（smem<48K、`FORCE_CUBLAS`） | `ggml_cuda_mul_mat_cublas` | TC (fp16/bf16 MMA) | 先反量化到 fp16 |
| Blackwell + MXFP4/NVFP4 | `mmq` `fp4_fp4_mma` | TC (`mxf4.block_scale`) | e2m1 + ue8m0 缩放 |
| attention（`-fa`，head dim 合法） | `fattn-mma-f16.cuh` | fp16 TC | f16 输入 f32 acc |

</details>

<details>
<summary><b>版本提醒</b></summary>

我核对的是当前 master（文件路径已按新目录 `ggml/src/ggml-cuda/` 结构分裂成 `mmq.cuh` / `mmq-vec-dot.cuh` / `mmq-load-tiles.cuh` / `mmq-config-*.cuh`）。2024 年的版本里 MMQ 是**纯 dp4a**、没有 `_mma` 变体，那时你第一条就是错的；2025 年之后才补上 int8 MMA（并且加了 `MMQ_DP4A_MAX_BATCH_SIZE`、按 SM 分档的 `mmq-config-{pascal-ampere-blackwell-rdna*}.cuh`）。如果你在看旧 fork（比如 `ik_llama.cpp` 或某个 vendor 分支），结论可能不同，最好按 `mmq-vec-dot.cuh` 里有没有 `_mma` 后缀来判断。

</details>

---
## 🧑‍💻 User

问题是q8激活精度真的够用吗

---
## 😀 默认助手 (7)

## 结论先说

**够用，但“够用”有明确边界**：在 4–5 bit 权重的模型上，q8 激活误差基本**落在权重量化噪声以下**，可以视为免费；到 q6_K / q8_0 这种高保真权重、再叠加长上下文和重尾激活（少数通道远大于其余）时，它会成为最大的一项新增误差：量级 0.1%–0.5% PPL，**可以被测出**，但难得造成灾难。

而且，量化激活在 llama.cpp 里并非额外选择（opt-in）：GPU 上量化权重**只有** int8 激活这一条通用路径（MMQ 和 MMVQ 都把 src1 量化成 q8_1），CPU 上所有 k-quant 模型多年来也一直在跑 int8 激活（而且比 GPU 更粗）。如果想获得更高精度的激活，只能整体放弃 MMQ。

---

## 1. 先把误差量级算清楚

对 32 元素一块、对称量化、scale $d = m/127$（$m$ 为块内最大绝对值），舍入误差在 $[-d/2, d/2]$ 上均匀分布：

$$\sigma_e=\frac{d}{\sqrt{12}}\approx 0.289\,d\approx 2.27\times10^{-3}\,m$$

点积 $y=w\cdot x$ 的误差是 $\epsilon=\sum_k w_k e_k$，所以 $\sigma_\epsilon=\sigma_e\|w\|_2$；而典型输出幅度 $|y|\approx\|w\|_2\,x_{\text{rms}}$，于是

$$\frac{\sigma_\epsilon}{|y|}\approx 2.27\times10^{-3}\cdot\frac{m}{x_{\text{rms}}}$$

高斯型激活、32 个样本的块最大值 $m\approx 2.3\,x_{\text{rms}}$，得到**每次 matmul 输出约 0.5% 的相对噪声**。

把它和同一张表里的其他项横向对比（都折算成“相对块最大值 $m$ 的 RMS 误差”）：

| 表示 | 块/粒度 | 相对 $m$ 的误差 | 特征 |
|---|---|---|---|
| fp16 输入 | 逐元素 | $\approx 1.2\times10^{-4}$ | 相对精度 0.028%，均匀，与分布无关 |
| bf16 输入 | 逐元素 | $\approx 1.0\times10^{-3}$ | 相对精度 ~0.2%，**只比 q8 细 2×** |
| **q8_1（GPU MMQ 激活）** | 32 元素共享 1 个 scale | $2.3\times10^{-3}\cdot m$ | 元素级 ~0.5%$\sigma$，受 outlier 影响 |
| q8_K（CPU 激活） | **256** 元素共享 1 个 scale | 同上但更粗 | 用了很多年，默认路径 |
| q6_K 权重 | 16×（8-bit 子 scale） | $\approx 9\times10^{-3}$ | |
| q4_K / q4_0 权重 | 32× | $\approx 3.6\times10^{-2}$ | **比 q8 激活大一个数量级** |

两个立刻能得出的结论：

1. **q8 激活比 fp16 激活粗 ~15 倍，但和 bf16 激活只差 2 倍。** 而 bf16 是很多模型 tensor-core 路径上的默认计算精度，大家从来都觉得“无损”。所以“q8 激活”在精度直觉上并不离谱，它落在“半精度附近”的同一个区间。
2. **4-bit 权重的误差比 q8 激活大约 10–15 倍。** 当权重本身已经有 3–4% 的元素级噪声时，再注入 0.5%（且是不同位置、不同符号、无偏的）几乎不改变总的信噪比。这就是它“免费”的数学原因。

<details>
<summary><b>一处需要说清的“不对称”：MMQ 在权重侧是精确的</b></summary>

这条路径不是“权重和激活都变糙”，而是**权重侧几乎无损、激活侧换成了 int8**：

- 4/5/6-bit 的 k-quant 反量化后本来就是 $\text{int8}\times d$ 的形式，落到 int8 tile 里**零额外误差**（Q6_K 的值域正好是 int8），block scale 在 fp32 里补乘。
- 而 fp16/cuBLAS 路径要把反量化值 round 到 fp16，多一份 $2.8\times10^{-4}$ 的相对误差。

所以 MMQ 实际上是把 fp16 路径的“两处 0.03% 误差”换成了“权重侧 0 + 激活侧 0.5%”。净效果在 4-bit 权重上完全可以忽略，在 **q8_0 权重**上就很微妙了——那时权重侧误差被压到 0.4% 左右，激活侧反而成了主导项。

唯一例外是 i-quant（IQ2/IQ3/IQ4）：反量化值是查表得到的非整齐值，落 int8 时会再过一次舍入，权重侧也吃 0.23% 的误差——但 i-quant 自身的误差本来就更粗，无所谓。

</details>

---

## 2. q8 激活会被哪些情况影响

核心机制只有一条：**块内 max 归一化 + 32 元素共享 scale**。块的 $\max/\text{median}$ 越大，块内其余元素的有效位数就越少。

如果某块里有一个异常大值（LLM 里确实存在 "massive activations"，个别通道能到 RMS 的 50–100 倍），那么

$$\frac{m}{x_{\text{rms}}}\to 20\text{–}100 \quad\Longrightarrow\quad \text{块内其余元素有效位数掉 4–6 bits}$$

这时 0.5% 会变成 4–20%（按块算）。这**不是**理论担忧，正是为什么现实中：

- **q8_0 KV cache 基本无损，q4 KV 就明显掉点**——KV 里有大量这种 outlier 通道；
- 反过来说，`q8_1` 的 32 元素粒度已经是这类方案里的“细粒度”了，CPU 的 q8_K 是 256 元素粒度也活得很好。

具体到场景，我的排序（从“无所谓”到“要小心”）：

1. **q4_K / q5_K / i-quant 模型、普通对话** → 完全无所谓，权重噪声远大于它。
2. **q6_K / q8_0 模型** → 激活侧成为最大新增误差项，PPL 差异进入可测区间（0.1%–0.5% 量级，取决于模型）。
3. **需要高保真采样的任务**（长上下文、数学、代码、极低温采样） → 输出层 logits 上 0.5% 的噪声在“接近平局”时能翻转 argmax，主观上表现为“偶发性的答案不稳定”，而不是“系统性变笨”。
4. **MoE 路由** → 一般不受影响，因为 router（`ffn_gate_inp`）通常是 F32 小矩阵，走的是 MMVF/cuBLAS，不进 MMQ。

另外提一句：这个误差是**无偏的**（round-to-nearest，正负对称），不是系统性的幅度压缩；它跨层按大约 $\sqrt{\text{depth}}$ 累积，被 RMSNorm 反复重标定，不会指数放大。

---

## 3. llama.cpp 自己的态度（代码即证据）

这部分最能回答“官方觉得够不够”：

`mmq.cu` 里 src1 精度**只有一个旋钮**，取值域被 `GGML_ASSERT` 限定死：

```cpp
GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
```

- 优先级：环境变量 `GGML_CUDA_MMQ_PREC`（`q4` / `q8` / `auto`）→ 图上 `op_params[3]` → 默认 **`GGML_PREC_Q8`**；
- `GGML_PREC_Q4` 只对 Blackwell 上的 MXFP4/NVFP4 生效（原生 W4A4 指令 `mma.sync ... kind::mxf4.block_scale`），否则强制回落 Q8；
- **没有 F16/F32 激活的分支**。要更高精度只能整条路不走 MMQ：`GGML_CUDA_FORCE_CUBLAS=1`、或 `smpbo < 48 KiB`、或 `ggml_cuda_mmq_needs_fallback(ne01)` 触发时回落到 dequant+cuBLAS；
- gate/up/GLU 融合分支干脆硬编码 `prec_src1 = GGML_PREC_Q8`（`mmq.cu` 里 `const ggml_prec prec_src1 = gate ? GGML_PREC_Q8 : ...`）。

解读：**q8 被当作“安全默认”，q4 才需要显式打开**；而且“高精度激活”在这个后端里根本没被当成一个需要支持的选项。考虑 MMQ 从 2023 年底跑到现在、默认开启、覆盖所有 NVIDIA Turing+，这本身就是最大规模的实证。

顺带修正一个容易忽略的事实：**decode 也在用 int8 激活**。`mmvq` 路径同样先 `quantize_row_q8_1_cuda` 再算 `vec_dot`，所以“decode 精度更高”是错觉；只有 MMVF（f16/bf16/f32 权重）才是 fp32 激活。

---

## 4. 想自己验证的话

```bash
# baseline: 默认 MMQ (int8 激活, int8 TC)
llama-perplexity -m model.gguf -f wiki.test.raw -ngl 99

# 对照: fp16/bf16 权重+激活的 cuBLAS 路径
GGML_CUDA_FORCE_CUBLAS=1 llama-perplexity -m model.gguf -f wiki.test.raw -ngl 99

# 对照: CPU int8 激活 (q8_K, 256 元素 scale, 更粗)
llama-perplexity -m model.gguf -f wiki.test.raw -ngl 0
```

几个读结果时的注意点：

- PPL 差异通常在 **0.1%–1%**，必须固定 `-c`、同一份 `wiki.test.raw`、同 seed 才有意义；
- 想看更敏感的指标用 KL divergence 或 logit MSE（对 q6_K/q8_0 模型会立刻显出差别，对 q4_K 几乎为零——这正是上面噪声底论证的验证）；
- 如果真要“更准的激活”，比关掉 MMQ 更划算的做法通常是**换更细粒度的激活量化**——但 llama.cpp 目前没有这条路（fp8 e4m3 也没接 GEMM）。所以现实中你能选的只有“q8 激活 + int8 TC”和“fp16 路径”两者。

---

**一句话总结**：q8 激活的精度不是“绝对够”，而是“**相对于你已经接受的 4-bit 权重，它恰好落在噪声底以下**”。它的失效点是可预测的：高保真权重 + 重尾激活 + 长上下文。真要卡这条线，就得放弃 MMQ 的 2× INT8 吞吐，而 llama.cpp 的取舍是：默认认为不值得。

（上面所有误差数字都是按均匀舍入模型推的**量级估计**，不是实测值；实测请按第 4 节跑，差异也可能因模型/量化类型差一个量级。）
