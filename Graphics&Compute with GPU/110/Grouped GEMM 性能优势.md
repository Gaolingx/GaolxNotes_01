# Grouped GEMM 性能优势

### 🧑‍💻 User

GroupedGEMM for MoE为什么相较于启动topk专家个独立的batch gemm，性能更高？

---

### 🤖 Assistant

下面给你一个**从架构层面、kernel 级访存、调度、并行度**到 **混合量化** 的完整解释，为什么 **Grouped GEMM 在 MoE 中比依次执行 Top-k 专家 GEMM 要快得多**。

---

# ✅ TL;DR（核心原因）

Grouped GEMM 把 “多个专家的小矩阵” **合并成一个大批次（batched）的大型 GEMM** 来执行，使得：

* **更高的 GPU 利用率（SM occupancy）**
* **更好的 Tensor Core tile 利用率（更大 M/N/K → 更接近峰值）**
* **更有效的内存访问（coalesced, L2 reuse）**
* **减少 kernel launch 开销**
* **更少的 DRAM 往返**
* **减少小 GEMM 的调度碎片**

简单说：

> **GPU 不擅长跑很多“小矩阵”，擅长跑少量“大矩阵”。Grouped GEMM 把所有小矩阵自动打包成一个“大矩阵”去跑。**

---

# 1. 🟥 根本原因：小 GEMM 性能极差（Tensor Core tile 利用率低）

MoE 每个专家 FFN 通常是：

* hidden_dim = 4096
* ffn_dim = 11008（或类似）
* 每个 token 分配到极少数（top-1 / top-2）专家 → **分摊后的 M 很小**

如果每个专家只处理少量 token，比如 2～32 个，那么**独立 GEMM 的 M×K、N×K 都很小**。

Tensor Core 的固定 tile 是：

* Ampere/Hopper：16×16×16
* Blackwell：依然 16×16，但更密集

如果矩阵维度不大于 16/32，就会有大量碎片 → **性能掉到 10–30% 理论峰值**

GroupedGEMM 把来自多个专家的 **M 堆一起** 组成一个 large-batch，让矩阵更接近大 tile，因此性能立即暴涨到 70–95% 峰值。

---

# 2. 🟧 SM occupancy：独立 tiny GEMM 无法喂饱 GPU

假设有 160 专家，top-2 路由 → 每次要启动 320 个小 GEMM。
其中大部分专家每次只有极少 token（负载不均）。

这导致：

| 方案           | 每次 GEMM 大小            | 并行度                | SM 利用率         |
| ------------ | --------------------- | ------------------ | -------------- |
| 独立 GEMM      | 很多小 GEMM              | 每个 GEMM 启动很少 block | **低（20–40%）**  |
| Grouped GEMM | 合并成一个巨大的 batched GEMM | block 数巨大          | **高（80–100%）** |

Grouped GEMM 可以将整个 MoE 层当成一个巨型 GEMM 来跑，SM 完全吃满。

---

# 3. 🟨 Kernel launch 数量减少（非常大影响）

独立 GEMM:

* top-2 experts
* 每层 ~ 64 个 MoE 层
* 每层 ~ 20–40 个 token per expert（但稀疏）

你可能需要：

> **几百到几千个 GEMM launch**

GPU kernel-launch overhead 对于小 GEMM 占比极高（甚至 >30% 时间）。

Grouped GEMM：

> 只 launch 一次 → kernel 内部自动调度所有专家的 tile。

launch overhead 被完全摊薄。

---

# 4. 🟩 L2 / registers / shared memory reuse 更高

独立小 GEMM：

* 每个专家都要独立加载权重到 L2
* 重复带宽浪费巨大
* 由于 token 很少，权重缓存命中率低

Grouped GEMM：

* 权重连续布局，kernel 内部按 expert id 拉取
* 相邻专家矩阵常常能被 L2 命中
* 同一 kernel 内 reuse 更高

实际测量：Grouped GEMM 可以使 L2 hit rate 提升 20–50%。

---

# 5. 🟦 dispatch（token → expert）的结果天然适合 batched layout

MoE 必须把 token 分配到 expert：

* 做 scatter → buffer[M_expert][dim]
* 然后把 buffer feed 到专家 FFN GEMM

Grouped GEMM 直接读 “scatter 后的 buffer” 的指针数组，形成 **pointer-based batched GEMM**。

独立 GEMM 则需要：

* 分配很多小 buffer
* 启动大量小 GEMM
* 每个 GEMM 的带宽/调度浪费更明显

---

# 6. 🟪 混合量化（Q4_K、IQ4_NL 等）下 Grouped GEMM 更容易 fuse

你在调 `iqk_moe_fused_up_gate`、`mul_mat_up_gate_NxM` 时应该已经看到：
Grouped GEMM 能把：

* dequant
* matmul
* bias
* gate activation

融合成一个 kernel。

**多个专家分别做则无法融合。**

融合越深 → 带宽越省（节省 2–3× DRAM 流量） → 性能更高。

---

# 7. 🟫 GPU 调度减少碎片化（避免 context switching + pipeline stall）

独立 GEMM 情况：

* 每个小 GEMM 运行时间极短（几微秒）
* GPU pipeline 经常 IDLE
* 调度器无法 overlap

Grouped GEMM：

* 一个 kernel 跑几十到几百微秒
* 调度器可以充分 overlap MEM / compute
* 指令流连续无中断

GPU 最怕小 kernel：**Grouped 就是让 GPU 跑“大活儿”。**

---

# 8. 例子：为什么 tiny GEMM 易达不到 tensor core 峰值？

比如专家 M=16、K=4096、N=11008
独立 GEMM：

* M=16 → 非常小 → Tensor Core 只能用少数 tile
* warp-level 串行化严重
* occupancy 只有 30%

Grouped GEMM：

* 来自 40 个专家 token 堆叠 → M=16×40=640
* 足够大 → 完全吃满 Tensor Core

通常性能差距：**3×–10×**。

---

# 总结：Grouped GEMM 为什么更快？

| 原因                              | 效果                |
| ------------------------------- | ----------------- |
| 将多个专家的 tiny GEMM 合并 → 大 GEMM    | tensor core 利用率暴涨 |
| SM occupancy 从 20–40% → 80–100% | 更高并行度             |
| kernel launch 减少数百倍             | 显著减少 overhead     |
| L2 / cache reuse 更好             | 权重带宽降低            |
| 融合 dequant + gemm + gate        | 内存流量减少            |
| 减少 GPU 调度碎片                     | pipeline 更平滑      |

一句话总结：

> **Grouped GEMM 让 GPU 从“跑一堆小活”变成“跑一份大活”，从而无限接近理论峰值。**

---

如果你愿意，我可以基于你正在调的：

* `iqk_moe_fused_up_gate`
* `mul_mat_up_gate_NxM`

给你画一个 **Grouped GEMM vs 独立专家 GEMM 的 kernel-level 对比图**（包括 tile mapping、访存、指令流）。


---

### 🧑‍💻 User

在deepseek-ai的deepgemm（https://github.com/deepseek-ai/DeepGEMM）中，这具体是怎么实现top-k个专家的fused gemm的？

---

### 🤖 Assistant

好的 — 这是我根据公开资料 + 源码／文档，对 DeepGEMM（来自 DeepSeek AI）里，**top-k 专家 + fused / grouped GEMM** 是**如何实现**的比较清晰说明／推断。总体来说，DeepGEMM 提供了专门为 MoE 场景设计的 “Grouped GEMM” kernel，通过把多个专家 (experts) 的 token 批 (batch) 合并 (concatenate) 或 用 mask，统一做一次大 GEMM，从而实现前面我们讨论过的高性能。以下是关键机制 & 实现细节。

---

## ✅ DeepGEMM 支持 MoE：Grouped GEMM 的接口与设计

* DeepGEMM 明确在 README 中指出，它除了普通 dense GEMM，也支持 MoE 的 “grouped” GEMM。([GitHub][1])
* 对于 MoE forward (token → expert → FFN) 的场景，它提供 **M-grouped GEMM**：即 N, K 维对所有专家共享、只有 M 维 (代表 token 数 / batch size) 随专家变化。具体函数包括： `m_grouped_fp8_gemm_nt_contiguous`, 对于 SM100 还有 `m_grouped_fp8_gemm_nn_contiguous`。([DeepWiki][2])
* 对于 MoE 推理 (inference) 中 decode 阶段 (token 数 / expert assignment 可能动态、难预测)，还支持 **masked grouped GEMM**：`m_grouped_fp8_gemm_nt_masked`，用 mask tensor 来跳过那些没有被路由 (no token assigned) 的专家 / slots，从而避免无意义计算。([GitHub][1])
* 另外，对于 MoE 的权重梯度 (weight-gradient) 计算 (通常 backward / training)，它还提供 **K-grouped GEMM**：`k_grouped_fp8_gemm_tn_contiguous` (或 nt depending on layout)，适合 K 维 (特征维) 随专家或 shard 变化的情况。([DeepWiki][3])

所以：**接口层面**，DeepGEMM 就支持 “将多个专家的小矩阵 (小 batch, small M) 合并 (contiguous) 或按 mask 跳过 (masked)，一次性做大批量 GEMM**。

---

## 🔧 背后的实现机制 (kernel-level + memory/compute-layout) — 如何 “fuse + grouped + high 性能”

DeepGEMM 并不是简单循环调用多个小 GEMM，而是通过 GPU kernel + layout design + JIT优化 + FP8 + hardware 特性组合，实现高性能 fused grouped GEMM：

* **Contiguous layout**: 对于前向 / prefill 阶段 (训练或推理前 fill)，多个专家对 token 的输出 /中间 activation 被 **拼接 (concatenate)** 在一起 (沿 M 轴) 成为一个大矩阵 A (shape: total_tokens × K)，然后与专家权重 B (K × N) 做一次 GEMM。这样 N, K 对所有专家一致，M 只是 “所有被激活 token 总和”。 DeepGEMM 要求 “每个专家的 segment (group) 在 M 方向上对齐到 GEMM block size” (通过 `get_mk_alignment_for_contiguous_layout()`)。([artivio.eu - Art Through Technology][4])
* **Masked layout**: 对于 decode 时专家 assignment 动态 + batch 差异大 + 想用 CUDA Graph 的情况 (希望 kernel launch 固定)，DeepGEMM 提供 masked grouped GEMM。即传入一个 mask tensor，kernel 内部只对 “valid (被路由 / 有 token)” expert-slot 做计算，其它跳过。这样避免了为每个专家分别 launch kernel，也避免了对无 token 的专家做无用乘法。([GitHub][1])
* **FP8 + fine-grained scaling (quantization + dequantization)**: DeepGEMM 使用 FP8 对输入 (activation + weight) 做低精度存储／计算，以减小内存占用和带宽压力。为了兼顾精度，它实现了 “细粒度 scaling + 两级累加 (promotion / accumulate in higher precision)” — 也就是即使用 FP8 做乘法，也用更高精度 (e.g. FP32 或 BF16) 做累加/输出，以保障数值稳定性。([搜狐][5])
* **JIT 内核生成**: DeepGEMM 不是一个写死所有可能 shape 的模板库。它通过轻量级 JIT (Just-In-Time) 系统，在 runtime 根据实际输入 (M, N, K, expert group counts, layout type) 生成 kernel。这允许 kernel compile-time 常量 (block size, thread config, TMA 参数等) 与实际 shape 对齐，从而最大化性能 (减少寄存器浪费、静态优化、硬件特性利用)。([DeepWiki][3])
* **利用硬件 / GPU 特性 (如 TMA / Tensor Core / Hopper-specific optimizations)**: DeepGEMM 针对 NVIDIA Hopper (SM90) / Blackwell (SM100) 架构做优化 — 利用 TMA (Tensor Memory Accelerator) 来加速数据加载／存储、prefetch、broadcast 等；利用 tensor cores 做高吞吐量 FP8 GEMM；利用 layout + alignment + swizzling + L2 / cache reuse，减少内存带宽压力。([DeepEP][6])

综合以上，DeepGEMM 的 “grouped + fused MoE GEMM” 并非拼凑多个普通 GEMM，而是从数据 layout (contiguous / masked)、量化 (FP8 + scaling / accumulate)、kernel generation (JIT)、硬件特性 (TMA / Tensor Core) 再到 runtime 调度，多层优化 — 从根本上把 “many small expert GEMMs” 转化成 “one (or few) big, efficient GEMM(s) with minimal overhead + maximal throughput”。

---

## 🎯 在 top-k + MoE 场景中是怎样 “用起来”的

结合常见 MoE 推理 /训练流程 (即 top-k 路由 + 多专家 forward):

1. 对每个 token，通过 router 得到它对应的 top-k experts (topk_ids) + 权重 (topk_weights)
2. 将各 token 的 embedding / hidden_state 按 expert grouping + token assignment 做 scatter / bucket → 得到 per-expert “mini-batch” of tokens.
3. **Contiguous layout path** (prefill /训练 / 非 decode): 把所有专家 token 合并 (concatenate) → large tensor A; 同时将对应专家权重 (可能是 per-expert weight matrices) 对应 B 矩阵整理 (对于 experts share same shape, B shape 相同) → 调用 `m_grouped_fp8_gemm_nt_contiguous(A, B, ...)`。
4. **Masked layout path** (decode / dynamic assignment): 不做 concat，而是保留一个 buffer + mask，调用 `m_grouped_fp8_gemm_nt_masked(...)`，kernel 内部根据 mask 只做有 token 的专家 slot.
5. 输出拼回 (gather / reshape) 成 per-token per-expert outputs，再加权 / sum / gate / activation → 完成 MoE 层 forward。

对于 training backward / weight update 如果 K 维 (特征维) 有分片 / varying，使用 K-grouped GEMM，通过 `k_grouped_fp8_gemm_tn_contiguous` 等接口对多个专家的梯度做 batched update。

文档 /集成 (例如在 vLLM) 就是通过 DeepGEMM 的这些 grouped-GEMM 接口 (contiguous/masked) 来实现 MoE layer 的 fused GEMM + activation + routing +量化。([vLLM][7])

---

## ✅ 优势 & 设计权衡 (为什么这种实现方式对 top-k MoE 特别合适)

* 减少 kernel-launch overhead：不用为每个专家 / 每个 token 批分别 launch GEMM。
* 高硬件利用率：因为是大矩阵 → tensor core / TMA / cache / L2 / shared memory 都能充分利用。
* 动态适应 top-k 路由变动 (masked) → decode 阶段也高效。
* 支持量化 (FP8) + scaling + accumulate → memory / bandwidth 节省 + 数值稳定。
* JIT + layout + alignment → 对不同 shape / expert count / token count 都自动适配 (可插拔)。

---

## ⚠️ 限制 / 要注意的地方

* 所有专家必须“共享 shape (N, K)” —— 这是 grouped GEMM 的前提 (否则 N/K 不一致就不能合并 B 矩阵)。DeepGEMM README 明确这样设计。([artivio.eu - Art Through Technology][4])
* 对于 contiguous 模式，每个专家分配到的 token 数 (group size) 必须对齐到 GEMM block 大小 (alignment constraint)，需要使用 `get_mk_alignment_for_contiguous_layout()`。([artivio.eu - Art Through Technology][4])
* FP8 + fine-grained scaling + accumulate 虽然兼顾速度与精度，但量化 + dequantization + scaling overhead +可能数值误差，需要谨慎验证 (尤其训练场景) — DeepGEMM 本身通过 “两级累加 (promotion)” 来减小误差。([MarkTechPost][8])

---

## 🧪 总结

DeepGEMM 用 **M-grouped (contiguous / masked) + FP8 + JIT + hardware-native GEMM kernel** 的组合，实现了对 top-k expert MoE 的 **fused GEMM** — 把多专家 / 多 token 的分散计算，汇聚为一次 (或少量) 高效矩阵乘法。这样既保留了 MoE 的 sparsity + routing + per-expert specialization，也极大提升了 GPU utilization, memory / bandwidth efficiency, 并降低 overhead。

如果你愿意，我可以帮你 **读一下 DeepGEMM 源码 (fp8_gemm.cuh + grouped相关部分)**，做一个 **伪代码 + kernel flow（含 memory layout / thread block / warp mapping / quantize-dequantize + accumulate）** 的详细草图。这样你可以非常直观看到它 “怎么 fuse top-k experts + GEMM” 的。你要吗？

[1]: https://github.com/deepseek-ai/DeepGEMM?utm_source=chatgpt.com "GitHub - deepseek-ai/DeepGEMM: DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling"
[2]: https://deepwiki.com/deepseek-ai/DeepGEMM/1.1-getting-started?utm_source=chatgpt.com "Getting Started | deepseek-ai/DeepGEMM | DeepWiki"
[3]: https://deepwiki.com/deepseek-ai/DeepGEMM?utm_source=chatgpt.com "deepseek-ai/DeepGEMM | DeepWiki"
[4]: https://artivio.eu/2025/02/26/deepseek-open-sources-deepgemm-clean-and-efficient-fp8-gemm-kernels/?utm_source=chatgpt.com "DeepSeek Open Sources DeepGEMM: Clean and efficient FP8 GEMM kernels - artivio.eu"
[5]: https://www.sohu.com/a/863998204_122014422?utm_source=chatgpt.com "DeepSeek推出高效开源DeepGEMM库，支持MoE分组运算_代码_核心_设计"
[6]: https://www.deepep.org/en/deepgemm?utm_source=chatgpt.com "DeepGEMM - Efficient FP8 Matrix Multiplication Library"
[7]: https://docs.vllm.ai/api/vllm/model_executor/layers/fused_moe/deep_gemm_moe.html?utm_source=chatgpt.com "deep_gemm_moe - vLLM"
[8]: https://www.marktechpost.com/2025/02/25/deepseek-ai-releases-deepgemm-an-fp8-gemm-library-that-supports-both-dense-and-moe-gemms-powering-v3-r1-training-and-inference/?utm_source=chatgpt.com "DeepSeek AI Releases DeepGEMM: An FP8 GEMM Library that Supports both Dense and MoE GEMMs Powering V3/R1 Training and Inference - MarkTechPost"
