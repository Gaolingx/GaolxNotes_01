# DeepGEMM SM100 MQA logits稀疏注意力打分内核

## 🧑‍💻 User

https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh这段代码有什么用

---
## 😀 默认助手 (7)

我已经读完该文件（34 KB，两个 kernel 入口及共享 core），下面给出结论。

## 一句话总结

这是 **DeepGEMM 面向 NVIDIA Blackwell（SM100）的 MQA logits 打分 kernel**，也就是 DeepSeek-V3.2 中 **Lightning Indexer（DSA 稀疏注意力索引器）的打分阶段**。它把「$q\cdot k$ 点积 → ReLU → 逐 head 乘权重 → 沿 head 维求和」这四步融合进一个 kernel，为每个 (query token, kv token) 对直接输出一个 BF16 标量 logits。

计算上等价于（源码 README 的公式）：

$$ \text{out}_{ij} = \sum_{h=0}^{H-1} \text{ReLU}\big(q_{i,h} \cdot k_j\big)\cdot w_{i,h}, \qquad j \in [\text{cu\_seq\_len\_k\_start}[i],\ \text{cu\_seq\_len\_k\_end}[i]) $$

其中 $q\in\mathbb{R}^{S_q\times H\times D}$（**MQA**：多个 head 共享同一份 KV），$k_j\in\mathbb{R}^{D}$，权重 $w\in\mathbb{R}^{S_q\times H}$ 为 BF16。

---

## 文件里到底有几个 kernel

它只有 **2 个对外的 `__global__` 入口**，但共用一份实现：

| 入口 | 适用阶段 | 特点 |
|---|---|---|
| `sm100_mqa_logits` | prefill（连续 KV） | KV 是连续的 `[seq_kv, D]`，用 `cu_seq_len_k_start/end` 界定每行的有效区间 |
| `sm100_paged_mqa_logits` | decode（分页 KV） | 走 `block_table` + `context_lens` + `indices`，KV 页在 HBM 中不连续 |

两者都只是包一层 `make_scheduler`，真正的逻辑全在 `sm100_mqa_logits_core_impl` 里，靠编译期常量 `Scheduler::kIsPaged` 做分支消除（`if constexpr`）。这样连续/分页、FP4/FP8 的全部组合只维护一份代码。

<details>
<summary><b>数据类型的支持范围（点开）</b></summary>

- 模板参数 `qk_dtype_t` 决定量化格式，`constexpr bool kIsFP4 = cute::is_same_v<...>`。
- 支持 **MXFP4** 与 **MXFP8**，scaling factor 都是 packed **UE8M0**（4 个 UE8M0 打包进一个 `uint32`），这是 SM100 的要求（SM90 用 FP32 SF）。
- `UMMA_K = kIsFP4 ? 64 : 32`，`head_dim` 只允许 32（仅非 FP4）/ 64 / 128（有 `DG_STATIC_ASSERT` 断言）。
- **输出固定是 `nv_bfloat16`**，且只写低 16 bit。

</details>

---

## 核心设计：4 个专用 warp + N 个数学 warpgroup

这是整个文件最值得看的部分。`kNumSpecializedThreads == 128`，也就是 4 个 warp 专门做搬运/发射，其余 warpgroup 做归约：

| warp | 角色 | 关键动作 |
|---|---|---|
| `kSpecWarpStart + 0` | **KV data producer** | TMA 搬 `kv`（以及 paged 模式下它负责的那部分 KV 页） |
| `kSpecWarpStart + 1` | **SF producer + Q 驱动** | 搬 `q_sf` / `kv_sf`，同时负责发起 `q` 与 `weights` 的 TMA |
| `kSpecWarpStart + 2` | **UTCCP warp** | 用 `SM100_UTCCP_4x32dp128bit_1cta` 把 SMEM 里的 scale factor 转置后 `tcgen05.cp` 进 **TMEM** |
| `kSpecWarpStart + 3` | **MMA warp（单线程发射）** | `cute::elect_one_sync()` 后循环发 `tcgen05.mma`（UMMA，block-scaled） |
| `warp_idx < kSpecWarpStart` | **数学 warpgroup** | `tcgen05.ld` 从 TMEM 读 FP32 累加器 → ReLU×weight → 写出 |

几个配套细节：

- **寄存器重分配**：spec warp 用 `warpgroup_reg_dealloc()` 把寄存器让出来，数学 warp 用 `warpgroup_reg_alloc()` 收回去。文件中还静态断言了「寄存器预算不能超过 CTA 入口池」（`kNumEntryRegisters` / `kNumMathRegisters`）。
- **三条 RingPipeline**：`q_pipeline` / `kv_pipeline` / `tmem_pipeline`，配 mbarrier 做生产者-消费者（`full_q_barriers` / `empty_kv_barriers` / `full_tmem_barriers` ...），初始化时按各角色线程数 `init(count)`。
- **KV 复用优化**：`reuses_kv_stages(task)` 检测相邻 task 是否 `kv_shared_with_prev && num_kv_splits == kNumKVStages`，命中就直接 `arrive()` 跳过搬运，SF 也留在 TMEM 里不重复转置。
- **TMEM 列预算**：累加器占 `UMMA_N * kNumTmemStages` 列，SF 接在后面，`DG_STATIC_ASSERT(kNumTmemCols <= 512)` —— 这正是 Blackwell TMEM 的硬限制。
- **重叠运算**：数学线程在读完某个 TMEM chunk 的最后一个 head 后立刻 `arrive()` 释放 TMEM stage，让下一轮 MMA 与当前 reduce 并行。

---

## 输出布局（容易踩坑的地方）

logits 是**压缩存储**的：形状 `[num_q_tokens, logits_stride]`，第 `i` 行的有效 KV 段**从第 0 列开始**，而不是放到全局列位置。所以代码里用了个技巧：

```cpp
output_bases[i] = logits + (task.q_token_base + i) * logits_stride - seq_k_start[i];
```

减掉 `seq_k_start[i]` 后，就能用统一的 `kv_offset` 直接索引，写入时再判断 `seq_k_start[i] <= kv_offset < seq_k_end[i]`（越界的静默丢弃）。

另外 `ptx::st_global_low_bf16` 只写 16 bit —— 因为 `kv_offset` 沿线程均匀分布、一次只落一个半字。

<details>
<summary><b>分页分支（paged）特有的处理</b></summary>

- `scheduler.get_kv_page_coords(task, kv_split_idx, page_coords)` 先把 page 坐标算出来（注释特意说明「在 stage 仍忙时就开始解析」，提前隐藏延迟）。
- SF 页用 **`tma_gather4`** 一次 gather 4 个 page（不足 4 个的尾巴退回普通 `tma::copy`），这是 TMA 的 gather 模式。
- KV 页在**两个 producer 之间均摊**：`kNumKVPagesFromSFProducer` 计算 SF producer 顺手多搬几页，剩下的交给 KV warp，减少两个 warp 的负载不均衡。
- `dispatch_num_block_tokens` 会把运行期才知道的有效 token 数（`num_q_tokens`）递归展开成 `cute::Int<N>`，让 token 循环保持编译期常量、便于完全展开。

</details>

---

## 名字里的 `sm100` 意味着什么

它依赖一整套 Blackwell 专属指令：`tcgen05.mma`（UMMA）、**TMEM**（Tensor Memory，`TMEM::Allocator1Sm().allocate` / `free`）、`tcgen05.cp`（UTCCP）、`tcgen05.ld`、`tma_gather4`、`cudaGridDependencySynchronize()`（PDL）。所以**只能在 B200 / GB200 这类 SM100 卡上跑**，SM90（H800）走的是另两个文件 `sm90_fp8_mqa_logits.cuh` / `sm90_fp8_paged_mqa_logits.cuh`。

相关兄弟文件还有 `sm100_sparse_mqa_logits.cuh`（稀疏版索引器，43 KB）。

---

## 怎么用

不直接调用这个 `.cuh`，它是 DeepJIT 运行时编译的模板体。Python 侧接口是：

```python
import deep_gemm

# prefill
logits = deep_gemm.fp8_fp4_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, ...)
# decode
logits = deep_gemm.fp8_fp4_paged_mqa_logits(q, kv, weights, block_table, context_lens, ...)
```

输入 `q` / `kv` 都是 `(data, sf)` 二元组，`weights` 在 SM100 上是 BF16、形状 `[seq_len, num_heads]`。完整示例见仓库的 `tests/test_attention.py`。

---

**一句话定位**：它是 DeepSeek 为自家 DSA 稀疏注意力索引器手写的、把 FA 式打分 + 加权归约融合成「一次输出一个标量」的极致专用 kernel —— 不用做 softmax、不用存中间分数矩阵，只把 $D$ 维点积的最大吞吐压出来，因此可以在 head 维上用很小的 TMEM chunk 分块归约，并用 4 warp 专用化 + 三环流水线把 Tensor Core 喂满。

> 说明：抓取时部分 `#include` 名与模板参数列表被去掉了（源码正常存在），上面引用的常量名/函数名均逐字来自文件正文，逻辑判断未受影响。
