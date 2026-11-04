# DeepSeek MegaKernel家族解析：MegaMoE融合通信计算拿1.5-2倍收益

## 🧑‍💻 User

去github翻翻deepseek-ai有哪些典型的megakernel，我记得moe是一个，通信占大头因此融到一块能拿不少收益
（其实有些感觉就是fuse的太多了就叫它mega了）

---
## 😀 默认助手 (7)

去 GitHub 看了一圈（看的是 `deepseek-ai` 组织当前 main 的状态，45 个仓库），你这记忆是准的，但 **“mega”其实已经是一个内核家族**了，而且它都在 `DeepGEMM` 这个仓库里，不在 `DeepEP`。下面按“真去通信”和“其实只是 fuse 多了”两档来分。

## 一、`DeepGEMM` 里的 Mega 家族

| Kernel | 实现文件 | 融了什么 | 官方给的收益 |
|---|---|---|---|
| **Mega MoE** | `impls/sm100_fp8_fp4_mega_moe.cuh`（88 KB）/ `sm100_bf16_mega_moe.cuh` | EP dispatch + L1 GEMM + SwiGLU + L2 GEMM + EP combine（**外加 shared expert 两段），并让 NVLink 通信与 tensor core 计算重叠** | 相对 legacy 路径 $\mathbf{1.5\times\sim2.0\times}$ |
| **Mega Gate** | `impls/sm100_bf16_mega_gate.cuh` | routing GEMM + 多个 MoE routing 操作（top-k 选择/打分等）合成一个 kernel | 30–75% |
| **Mega mHC** | `impls/sm100_mega_mhc.cuh` | 多个 Hyper-Connection 操作 + RMSNorm 合成一个 kernel | 45–85% |
| Sparse Indexer（没叫 mega，但同性质） | `impls/sm100_sparse_mqa_logits.cuh` | V4.1 的层级稀疏索引，吞吐逼近 Lightning Indexer | — |
| HC prenorm GEMM | `impls/sm100_tf32_hc_prenorm_gemm.cuh` | 把 prenorm 融进 GEMM epilogue | — |

两个入口 PR 值得直接点开看：`#304`（26/04 首次公开 Mega MoE + FP4 Indexer）、`#432`（26/09 加 Mega Gate / Mega mHC / Sparse Indexer / DeepJIT）。

## 二、Mega MoE 才是你说的那个“通信占大头”的

<details>
<summary><b>它具体包了什么（点开看）</b></summary>

公开 API 长这样，从签名就能看出融合粒度：

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank,
                                                num_topk, hidden, intermediate_hidden,
                                                num_shared_experts=n, mma_type='fp8xfp4')
l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1_w, l2_w)   # gate/up 交错 + SF 转 UTCCP layout
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer)                   # ← 一次 launch 干掉整个 MoE 层
```

几个关键细节：

- **对称内存**：走 `torch.distributed._symmetric_memory`（`SymmBuffer`，要求 PyTorch ≥ 2.9），buffer 里预先切好 `x, x_sf, topk_idx, topk_weights, shared_l1_acts, l1_acts, l2_acts…`。也就是说 **shared expert 也被塞进同一个 kernel**，不是单独一个 dense GEMM。
- **权重预变换**：gate/up 按 8 列交错（`_interleave_weights`），SwiGLU 一次读就能拿到配对；SF 按 `_transpose_sf_for_utccp` 转成 tcgen05 block-scale 布局。layout 层面的融合也是收益来源之一。
- **多进程 launch**：必须一个 GPU 一个进程（`tests/test_mega_moe.py` 是现成例子），不能用单进程多卡。
- **只能 SwiGLU**，`mma_type` 支持 `fp8xfp4` / `fp8xfp8`，另外有 `bf16_mega_moe`；kernel 末尾硬断言 `"This kernel only support sm_100f"` —— **Hopper 跑不了**（社区一直在 issue 里问 H100）。
- 模板参数一眼能看出它做了 token 环缓冲 staging：`kNumMaxTokensPerRank, kNumRingTokens, STORE_BLOCK_M_L1/L2, SF_BLOCK_M/N…`，代价是每个 shape 都要 JIT 特化（所以后来搞了 DeepJIT）。

</details>

<details>
<summary><b>Benchmark（PR #316，EP8，8 卡平均）</b></summary>

DeepSeek-V4-Flash（256 experts, top-6, hidden 4096, inter 2048）：

| batch | 时间 (us) | Compute (TFLOPS) | GMEM (GB/s) | Interconnect (GB/s) | vs legacy |
|---|---|---|---|---|---|
| 1 | 56.5 | 5 | 1311 | 1 | **1.96×** |
| 512 | 146.5 | 1056 | 3192 | 266 | 1.73× |
| 8192 | 1283 | 1928 | 998 | 499 | 1.56× |
| 32768 | 4855 | 2038 | 794 | 529 | 1.62× |

DeepSeek-V4-Pro（384 experts, top-6, hidden 7168, inter 3072）：batch=1 时 108 us / **1.61×**，batch=32768 时 10.6 ms / **1.54×**。

</details>

### 为什么能拿到 1.5–2×：你的直觉对，但只对了一半

- ✅ **通信与计算的粒度被改成了流式**：legacy 路径是 `dispatch kernel → grouped GEMM L1 → SwiGLU/cast → grouped GEMM L2 → combine kernel`，dispatch 和 GEMM 之间是**全局 barrier**，你要等 $\max_{\text{rank}}$ 的到达时间。融成一个 kernel 后，某个 expert 的 token tile 一到（对称 buffer 上打 flag），对应 SM 组就能开跑 L1，别的 rank 还在发——把“批量同步 all-to-all”变成了 producer–consumer 流水，付出去的是最大延迟而不是平均延迟。
- ✅ **不再抢走计算用的 SM**：DeepEP 文档自己写着 dispatch/combine 是要占 GPU SM 的。融合后通信退化成少数 warpgroup/CTA 的角色，其余 SM 全程压着 tensor core；顺带省掉 kernel 边界和 CUDA graph 的 bubble。
- ⚠️ **但 batch=1 时 interconnect 只有 1 GB/s** —— 这一档的 1.96× 跟带宽基本无关，主要是**权重流式加载（1311 GB/s GMEM）与通信延迟被并行掉了**，以及消掉了十几个 tiny kernel 的串行开销。大 batch（interconnect 400–530 GB/s）才是你记忆里那个“通信占大头”的场景。所以准确说法是：**小 batch 赚延迟与 kernel 边界，大 batch 赚带宽-计算重叠**。
- 📉 边际收益递减很明显：26/09 那次针对 V4.1 的优化只有“低延迟 +15% / 高吞吐 +4%”。

## 三、你说的“其实就是 fuse 多了”的那几个 —— 确实存在

- **Mega Gate**：routing GEMM + top-k 等（本质是 launch 开销 + 中间张量往返 HBM，没有任何通信），30–75%。
- **Mega mHC**：多个 Hyper-Connection 算子 + RMSNorm，45–85%，纯 memory-bound 融合。
- 同类还有 `TileKernels`（TileLang 写的）：fused SwiGLU+quant、Engram gate 融 RMSNorm、mHC 的 Sinkhorn 多算子拼接，官方口径就是“大部分接近算力/带宽上限”，也是“fuse 很多但没叫 mega”。
- 甚至有个更激进的趋势：`deep_gemm/locality_domain.py` + `localize()` 把权重从 `(…, N, K)` 切成 `(num_domains, …, N/num_domains, K)`，按 SM 的 locality domain（后端是 MLOPart）分发权重——这是在往“**让一个 kernel 独占整台机器**”走，mega 名字没加，方向是一样的。

## 四、容易被误当成 megakernel、但其实是别的层

- **`DeepEP`**：通信库，不是 megakernel。但它做了两件专门为 megakernel 铺路的事：dispatch 融 FP8 cast / expanded layout / zero padding，combine 融 bias 和 reduce，还有 `defer_epilogue=True` + `EventOverlap`，让上层能把通信塞进同一个流里。V2 把 `ElasticBuffer` 拆成 `EPBuffer`（Mega MoE 现在对接的就是它），并且把 NVSHMEM 依赖干掉了，改用 NCCL Gin。分支里的 `LL-SBO`（把 Down GEMM 和 combine send 重叠）本质就是同一个思路的局部版。
- **`FlashMLA`**：单个 MLA kernel 内部 QK^T + softmax + PV + 输出全融，但没挂 mega 名字。
- **`DeepSelect`**：DSA/采样器的 TopK 专用 kernel（V4 用 `topk=512`），2–20× vs `torch.topk`，属于“把一件事做到极致”，不是融合型。
- **`DualPipe` / `profile-data`**：在 pipeline / 调度层做计算-通信重叠，与 kernel 内融合是两条正交的路。

## 五、想动手的话最短路径

1. 读 `tests/test_mega_moe.py`（多进程 + benchmark 的完整例子）；
2. 读 `deep_gemm/mega/__init__.py`（API + 权重变换，8.7 KB，比 88 KB 的 CUDA 好啃）；
3. 再看 `sm100_fp8_fp4_mega_moe.cuh` 的模板参数列表，就能反推出它的流水结构；
4. 环境：SM100 + CUDA 12.9/13.1 + PyTorch ≥ 2.9 + NVLink 单 scale-up domain；PCIe-only 和 Hopper 目前都不行（issue #356、flashinfer #3196）。

一句话总结：**真 megakernel 只有一个「Mega MoE」**（因为它跨了 kernel 边界、跨了设备和设备之间的边界，并且重叠了通信与计算）；`Mega Gate`/`Mega mHC` 就是你说的“fuse 太多了顺手叫 mega”，但它们也确实卡在 launch 和 HBM 往返这两项上，收益 30–85% 并不虚。
