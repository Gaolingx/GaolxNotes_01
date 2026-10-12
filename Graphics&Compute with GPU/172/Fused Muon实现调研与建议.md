# Fused Muon实现调研与建议

## 🧑‍💻 User

astrai-muon-ns-fused-kernel-plan-2026-10-03.md 帮我调研下fused muon的相关实现

---
## 😀 默认助手 (7)

我先并行拉取你列出的几个外部实现参考，再补充搜索其他 fused Muon 实现。

已完成调研。下面把公开可用的 fused Muon / Newton–Schulz 实现逐个拆开，并对着你那份计划（SM 120a / 5090）逐条给结论。**我没有跑任何 benchmark，所有性能数字都是原仓库自报值，标注了硬件。**

---

## 一、结论速览

| 实现 | 语言/栈 | 融合层级 | 对称利用 | 硬件定位 | 自报收益 | 与你计划的关系 |
|---|---|---|---|---|---|---|
| [PyTorch `torch/optim/_muon.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py) | PyTorch | 无（1 `mm`+2 `addmm`/轮） | 无 | 通用 | — | **语义基线**，必须逐轮对齐的对象 |
| [StarrickLiu/fused-muon](https://github.com/StarrickLiu/fused-muon) | CUTLASS 2.x/CuTe + cuBLAS | GEMM1/2 用 SYRK + 双写 epilogue，GEMM3 留 cuBLAS | 下三角 tile，镜像写出完整矩阵 | SM80+（A800 实测） | ~1.5x vs `torch.compile`（A800，19/19 shape ≥1.03x） | **最贴近你第 1、2 步**的参考实现 |
| [NVIDIA-NeMo/Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) | Triton（TMA descriptor, Triton ≥3.4） | `tsyrk_ex` 支持 $\alpha A A^{\mathsf T}+\beta C$ + 镜像写 | 下三角 + 三角 grid 映射 | Hopper/Blackwell 取向 | 未给独立 kernel 数（Megatron 端 Muon≈AdamW 吞吐） | **最可能直接迁移到 SM120** 的路线 |
| [microsoft/dion](https://github.com/microsoft/dion)（MIT） | Triton | 三个 line 各一个 kernel（line3 也有） + 分布式编排 + CUDA Graph | `ns_line_1/2` 跳上三角 + 镜像写 | 通用 Triton | 未给 kernel 级数字 | 第 2、3 步模板 + **DDP 分摊**思路 + CUDA Graph 参考 |
| [thib-s/flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)（Turbo-Muon） | Triton | 同上 + AOL 预条件 | 同上 | 通用 | 2.8x（省一次迭代） | **改算法**路线，只能 opt-in |
| [Mantissagithub/muon_exps](https://github.com/Mantissagithub/muon_exps) | 手写 CUDA | 无融合，纯 kernel | 部分（gram-NS） | RTX 4060 Laptop (8.9) | **反面案例** | 警示：手写 kernel 可能比 PyTorch 慢 4–20x |
| ThunderKittens symmul | — | — | — | — | 论文 | **无公开 kernel 可复用**（见下） |

三个最重要的判断：

1. **没有任何公开实现是面向 SM 120/Blackwell 的 Newton–Schulz SYRK kernel。** fused-muon 用的是 CUTLASS 2.x 的 `SM80_16x8x16_F32BF16BF16F32_TN` + `cp.async.cg`（SM80 风格，无 TMA/tcgen05）；NVIDIA 的 `tsyrk_ex` 走 Triton TMA descriptor、面向 Hopper/Blackwell，但它要求 Triton ≥ 3.4；NVIDIA 博客明确把「CuTe DSL 的 SYRK+all-reduce 融合」列为*未来版本*。所以「5090 专用」这件事没有现成答案，这是你计划里风险最高的一项。
2. **第 2 步可以比 fused-muon 简单得多。** $B = bA + cA^2$ 可以直接用同一个 SYRK kernel 的 `alpha=c, beta=b` 一次算完（`tsyrk_ex(A, A, alpha=c, beta=b)`），**不必**写自定义 epilogue，也**不必**学 fused-muon 把 $aI$ 折到 $B$ 的对角线上。
3. **最省力的、改动最小的杠杆可能不是 kernel 融合**：Dion 自带 CUDA Graph 支持（`dion/cuda_graph.py`），而 `torch.compile` 是所有仓库用的基线——你 840 次 `mm` + 1680 次 `addmm` 的 launch/拷贝开销（profile 里的 1680 次拷贝 ≈5.2 ms）很大概率能被 CUDA Graph / 编译吃掉一部分，且**数值零影响**。

---

## 二、逐仓要点

<details>
<summary><b>A. PyTorch `torch.optim.Muon`（语义基线，必须逐字对齐）</b></summary>

关键代码路径（我已读全文）：

```python
ortho_grad = grad.to(dtype=torch.bfloat16, copy=True)
if grad.size(0) > grad.size(1): ortho_grad = ortho_grad.T   # 视图，非连续
ortho_grad.div_(ortho_grad.norm().clamp(min=eps))           # bf16 上归一化
for _ in range(ns_steps):
    gram_matrix = ortho_grad @ ortho_grad.T
    gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
    ortho_grad  = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
```

- 常量：`EPS=1e-7`、$(a,b,c)=(3.4445,-4.7750,2.0315)$、`NS_STEPS=5`，与你的计划一致。168 个矩阵 × 5 轮 = 840 `mm` + 1680 `addmm`，对得上。
- **注意：$B$ 里没有 $+aI$。** `a` 是第三个 `addmm` 的 `beta`。这一点是后面判断 fused-muon 是否「算法等价」的关键。
- 顺序是 **先 cast 到 bf16，再归一化**（在 bf16 上做除法）。NVIDIA 的实现是**先 fp32 归一化再 cast**——两者第一轮 GEMM 的输入就不是同一个 bf16 张量。这是你在第 0 步「算子契约」里必须明确的一条。
- 高矩阵用 `.T` 视图 → 送进 cuBLAS 的是非连续张量；如果这跟你 profile 里的 1680 次 device 拷贝有关，第 0 步应该能确认（不要凭调用数断定因果，你的计划这点写对了）。
- `foreach=True` 尚不支持；`_disable_dynamo_if_unsupported` 说明 `torch.compile` 可以捕获这条路径。

</details>

<details>
<summary><b>B. StarrickLiu/fused-muon（CuTe SYRK，最完整的一份工程记录）</b></summary>

结构：`csrc/ns_gemm.cu`（dispatcher）+ `csrc/syrk_common.cuh`（统一 SYRK 模板，`EpilogueMode` 0/1）+ `csrc/syrk_64.cu` / `syrk_128.cu`（**必须分文件编译**）+ `csrc/syrk_splitk.cuh` + `csrc/torch_binding.cu`；Python 侧 `muon_fused/ns_step.py`、`muon_fused/optimizer.py`；`docs/optimization_report.md` 是完整的 NCU 分析。

设计要点（我读了 `syrk_common.cuh` 与 `ns_gemm.cu` 源码）：

- **三角 block 映射**：`linear_to_lower_tri(blockIdx.x, i_tile, j_tile)`，grid 只发下三角块（m=2048、64-tile → 528 blocks，= 32·33/2）。
- **双写 epilogue**：正写 `copy(out, tCgC)`；上三角镜像：(a) 对角 tile 在寄存器里直接写 $gr>gc$ 的元素；(b) 非对角 tile 经 smem 转置（padding `TILE_M+8`）+ 合并写出。代价自报 **+2 µs @ m=4096**。
- **GEMM2 的 epilogue（`EpilogueMode==1`）**：
  $$D = \mathrm{bf16}\big(c\cdot acc + b\cdot \text{src}[i][j] + a\cdot[i{=}j]\big)$$
  `src` 从 global 读回 $A$。注意这里把 $a$ 放进了 $B$ 的对角线，即 $B' = cA^2+bA+aI$，于是 $B'X = aX + (cA^2+bA)X$——**实数域上与 PyTorch 等价**，但 bf16 舍入点不同（多一次对角加法的舍入）。
- **GEMM3 用 `cublasGemmEx`**，还利用 $B$ 对称做 col-major 技巧（`C^T = X^T B^T`，`B^T=B`）。也就是说这份实现自己的结论也支持你「第三个 GEMM 先不动」。
- **Dispatcher 启发式（A800 硬编码）**：`m ≤ 1280` → cuBLAS；`n > 8m` → cuBLAS；split-K 仅在 $5 \le n/m \le 8$ 且 `m%128==0`、`m ≥ 1280` 时启用，目标块数 = `108 SMs × 2 × 2 waves`，上限 8，且要求每 split ≥64 个 K-tile；128-tile 条件 `m>2048 || (m%128==0 && m%64!=0)`。
- **性能细节**：`cp.async.cg`（BYPASS L1）带来 **+56%**（m=4096，491→314 µs）；GEMM2 epilogue 总开销 ~9 µs（2.8%）；m=2560 时 TC 利用率只有 55.8%（210 blocks / 216 槽位，warp 并发不足）；vs CUTLASS GEMM 的残余差距来自 `mio_throttle` 25–29% vs 17%。
- **测量方法论**（与你的 ABBA 计划一致）：A800 空闲会降到 210 MHz，需 ≥5 s 连续 warmup；NCU `gpu__time_duration` 有 60–70 µs 固定开销，绝对时间以 cudaEvent 为准。

⚠️ **文档与代码不一致**：`muon_fused/ns_step.py` 的 docstring 说 "performs the entire iteration in a single launch"，但 `ns_gemm.cu` 每轮是 3 次启动，`docs/optimization_report.md` 也把「多步 NS pipeline / persistent kernel」列为*未来方向*。**别按 README 估算能省多少 launch。**

</details>

<details>
<summary><b>C. NVIDIA Emerging-Optimizers / Megatron Core（最可能上 5090 的一条路）</b></summary>

`emerging_optimizers/triton_kernels/syrk.py` 里的 `tsyrk_ex`：

- API：`tsyrk_ex(a, c=None, alpha=1.0, beta=0.0, skip_upper_triangle=False) -> (N,N)`，即 $D = \alpha A A^{\mathsf T} + \beta C$。
- **bf16 only**；要求 `a` 或 `a.T` 连续；走 **TMA TensorDescriptor**（`HAS_TRITON_340` 门控）；grid 覆盖全矩阵但 `IS_ABOVE_DIAG: return` 直接退出；写 `d` 后（非 `skip_upper_triangle` 时）`d_t_desc.store([offs_col, offs_row], d.T)` 做镜像写。
- 两个变体：通用版（autotune，含 `num_ctas`、`matmul_tma_set_block_size_hook`，`N≥5000` 时钉死 128×256×64）；**小矩阵版**用二次方程解 $m=\lfloor(\sqrt{8p+1}-1)/2\rfloor$ 把 pid 直接映射到下三角坐标，grid 只要 $B(B+1)/2$ 个块，且要求方形 tile。
- 调用点 `muon_utils.py::newton_schulz_step_tsyrk`：
  ```python
  A = triton_kernels.tsyrk_ex(X)                       # X @ X.T
  B = triton_kernels.tsyrk_ex(A, A, alpha=c, beta=b)   # c*A@A + b*A  ← 一次搞定 GEMM2
  X = torch.addmm(X, B, X, alpha=1.0, beta=a)          # GEMM3 保留 torch.addmm
  ```
  → **这正是「第 3 步先不动 GEMM3」的官方做法**；也证明第 2 步不需要自定义 epilogue。
- 门控条件：`use_syrk` 只在 `torch.get_float32_matmul_precision()=="medium"` 且 **M、N 都是 8 的倍数**（TMA 16-byte 对齐）时生效，否则回退 GEMM。你的四个形状都满足 8 对齐。
- 系数集很全（`simple / quintic / polar_express / cans / aol / deepseekv4 / cubic5`），并且**支持 batched (B,M,N)**。
- 归一化顺序：**fp32 归一化 → cast bf16**（与 PyTorch 相反，见上文）。

Megatron 博客里的工程结论（对你的第 4 步直接相关）：

- 复用型优化器要支持 Muon，必须走 **layer-wise distributed optimizer**（整层分配到某个 DP rank，避免为算 preconditioner 额外通信）。
- TP 下 NS 有三种模式：**duplicated**（先 all-gather，各 rank 重复算，延迟友好）/ **distributed**（每轮第一个 matmul 后 all-reduce，算力友好）/ **blockwise**（零通信，但等价于分块正交化，**数学上不等价**）。
- **SYRK 输出仍是完整矩阵**，所以 distributed 模式下 all-reduce 通信量没有减半；唯一真正的解法是把通信融进 SYRK kernel 的 tile 粒度——**规划中，未发布**。
- GB300 上 Muon ≈ AdamW 吞吐（Kimi K2：1051 vs 1080 TFLOPs/s/GPU，其中 51 是 Muon 开销）。

</details>

<details>
<summary><b>D. microsoft/dion（MIT；Triton NS + 分布式编排 + CUDA Graph）</b></summary>

`dion/newton_schulz_triton.py`：

- `ns_line_1`：$C = AA^{\mathsf T}$，`LOWER_UPPER` 为编译期常量决定跳过哪一半，镜像写 `tl.store(c_ptrs_t, output.T)`。
- `ns_line_2`：$C = \alpha A A^{\mathsf T} + \beta A$（从 $A$ 读 `a_add` 做 epilogue），带镜像写。
- `ns_line_3`：$C = aX + BX$，**把 $aX$ 作为累加器初值**（`acc = load(X)*alpha`），再跑标准 GEMM 主循环——这是忠于 `addmm(beta=a)` 的写法。
- ⚠️ **系数调度不同**：`newton_schulz_triton` 用的是 nanogpt 的 5 组 quintic 系数递进表 `(4.0848,-6.8946,2.9270) … (2.8366,-3.0525,1.2012)`，**不是** PyTorch 默认那单一组 `(3.4445,-4.7750,2.0315)`。拿它当 baseline 会同时比了算法和 kernel。
- `newton_schulz_triton_fast` = AOL rescaling + 4 组系数，省一轮（改算法）。
- 镜像写是「全 grid + early exit」风格（与 fused-muon 的三角 grid 不同），小矩阵变体才用三角映射。

分布式部分（对你的 DDP 阶段可能是**比 kernel 融合更大的收益来源**）：

- `dion/muon.py` 的注释说明：对 sharded 矩阵用 all-to-all 同时 unshard + 分发一批矩阵；**对 replicated 矩阵，把工作分摊到所有 device，最后 all-gather 结果**。也就是说 2 卡 DDP 下 NS 计算量可以按 rank 数摊分——这比「同一份冗余计算优化 10%」便宜得多，且不需要写任何新 kernel。
- `distributed_mesh` 可直接接 `ddp_model.process_group`；也支持 `num_heads`（逐 head NS）和 `split_sizes`（fused QKV 分块 NS），README 里对「per-head 会改变更新尺度、必须重新调 lr」有很详细的警告。
- **`dion/cuda_graph.py`（18 KB）+ `tests/test_cuda_graph.py`（39 KB）**：优化器步 CUDA Graph 化的现成参考。

</details>

<details>
<summary><b>E. flash-newton-schulz / Turbo-Muon（改算法路线，只能 opt-in）</b></summary>

- 用 **AOL rescaling** 取代 Frobenius 归一化：先算 $WW^{\mathsf T}$，取 $s=\mathrm{rsqrt}(\mathrm{clamp\_min}(\sum_j|A_{ij}|,\epsilon))$，把 $s_is_j$ 折进 $A$，从而**省掉 5 轮里的一轮**；另外补了一个 Triton `ns_line_3`（作者自己说只在「小矩阵、访存瓶颈」时有边际收益）。
- 自报 nanogpt（144M）：4 轮的 AOL ≈ 5 轮的 Muon；CIFAR speedrun 精度完全一致（0.9401）。
- 文中把「Muon+」（Triton kernel + 动态多项式系数，1.7x）作为中间基线，Turbo-Muon 叠加到 2.8x。**「Muon+」我没能定位到具体仓库**，引用时请自行核实。
- 已发布到 HF `kernels`：`get_kernel("tboissin/newton_schulz_triton")`。
- 你的计划明确写了「不改算法、固定 5 轮」，所以这条只能作为「如果 10% 门槛过不去」的备选，且必须走完整收敛验证。

</details>

<details>
<summary><b>F. 反例：Mantissagithub/muon_exps（手写 CUDA 可能更慢）</b></summary>

RTX 4060 Laptop（cc 8.9）上的数字：

| 形状 | 作者手写 CUDA NS | PyTorch Muon | flash-muon |
|---|---|---|---|
| 1024² | 4.62 ms | 1.05 ms | ~1.0 ms |
| 2048² | 30.59 ms | 1.99 ms | ~1.4 ms |
| 4096×1024 | 12.71 → 8.59（最优变体） | — | 1.48x |
| 8192×1024 | 23.98 → 11.41 | — | 2.10x |

价值在于模式：**方阵上自写 kernel 输给 cuBLAS 4–20 倍**，只有在「m 相对于 n 很大、下三角块足够多」的矩形场景才有优势。这正是 fused-muon 的 `m ≤ 1280 → cuBLAS` 兜底逻辑存在的理由，也说明你「按 shape 独立选择、禁止为单个大矩阵拖慢其余 96 个参数」这条纪律是必要的。

</details>

<details>
<summary><b>G. ThunderKittens 对称乘法：只有论文，没有可复用代码</b></summary>

我查了 `HazyResearch/ThunderKittens` 的 main 分支 `kernels/` 目录树（`gemm/` 下只有 bf16/fp8/nvfp4/int8 的 B200/H100 变体）以及 issues/PR 搜索（`symmul` 命中 0 条）——**没有公开的 symmul kernel 实现**。你的计划里「文中转置写回有已知 bug，不能直接用于训练」这个判断是对的，建议把它降级为**只读的性能思路来源**，不要作为代码参考。

</details>

---

## 三、横向对比：融合层级决定你该抄谁

| 维度 | PyTorch | fused-muon | NVIDIA `tsyrk_ex` | Dion |
|---|---|---|---|---|
| GEMM1 对称 kernel | ✗ | CuTe SYRK（三角 grid） | Triton TMA SYRK（全 grid+early exit / 小型号三角 grid） | Triton（全 grid + `LOWER_UPPER`） |
| 镜像写完整矩阵 | — | ✓（smem 转置，+2 µs） | ✓（`d_t_desc`） | ✓（`output.T`） |
| GEMM2 $bA+cA^2$ | ✗ | 自定义 epilogue（读回 $A$） | `tsyrk_ex(A,A,alpha=c,beta=b)` | `ns_line_2`（Triton epilogue） |
| $aX$ 的处理 | GEMM3 `beta=a` | 折进 $B$ 对角线 | GEMM3 `torch.addmm(beta=a)` | GEMM3 累加器初值 |
| GEMM3 | cuBLAS | cuBLAS | **`torch.addmm`（不动）** | 有 Triton `ns_line_3`（收益边际） |
| 精度策略 | bf16 存储、fp32 累加 | 同 | bf16 存储、fp32 累加 | 同 |
| 归一化顺序 | cast→norm | cast→norm | **norm→cast** | cast→norm |
| 系数表 | 单组 quintic | 单组 quintic | 7 种可切 | nanogpt 5 组递进 |
| 硬件取向 | — | SM80（A800 实测） | Hopper/Blackwell（TMA） | 通用 Triton |
| 许可 | BSD | Apache-2.0 | Apache-2.0 | MIT |

**给 5090 的路线取舍**：`tsyrk_ex` 是唯一「Blackwell 友好 + 许可宽松 + 自带 $\alpha/\beta$ 与镜像写 + 8 对齐兜底」的实现，直接拿来改比对着 fused-muon 重写 CuTe 更符合你的「首轮先用四种真实 shape 做算子级基准」策略。但它需要 Triton ≥ 3.4，且它的小矩阵三角映射要求方形 tile。

---

## 四、对着你的计划逐条给建议

**Step 0（固定基线与算子契约）— 建议加三件事**

1. 钉死**归一化顺序**（cast→norm vs norm→cast）与**$B$ 里是否含 $aI$**。这两条决定了你的 kernel 与 PyTorch 参考能不能逐元素对齐。
2. 在基线里额外测一条：**当前复用路径 + `torch.compile` / CUDA Graph**。所有外部仓库都是拿 `torch.compile` 当基线的，fused-muon 的「1.5x」是相对编译后的对手。如果你现在的基线没编译过，编译本身可能就吃掉相当一部分预算，且数值零风险。
3. 用 `.T` 视图复现一遍输入（`(6912,1536)` 转置后是非连续张量），确认 1680 次拷贝是否与之相关；现在 fused-muon 与 NVIDIA 的 kernel 对输入连续性的要求不同（NVIDIA 要求 `a` 或 `a.T` 连续）。

**Step 1（只换第一个 GEMM）— 你的 shape 落点**

用 fused-muon 的启发式推演你的四种形状（$\text{work } m=\min$）：

| 参数形状 | work (m,n) | $n/m$ | fused-muon 会怎么选 |
|---|---|---|---|
| (1536,1536)×48 | (1536,1536) | 1.0 | SYRK **64×64**（m<2048），无 split-K |
| (256,1536)×48 | (256,1536) | 6.0 | **cuBLAS 兜底**（m ≤ 1280） |
| (6912,1536)×48 | (1536,6912) | 4.5 | SYRK 64×64（$n/m<5$ 不触发 split-K） |
| (1536,6912)×24 | (1536,6912) | 4.5 | 同上 |

也就是说：**168 个矩阵里有 48 个（m=256）按现有启发式根本不会用 SYRK**，而 m=1536、64-tile 时下三角只有 $24\cdot25/2=300$ 个块。如果你按 170 个 SM 重推阈值，`m ≤ 1280` 这条线应该上移到约 1900–2000（$108\to170$ 等比），1536 正好卡在边界上；128-tile 时只剩 78 个块，几乎必然喂不饱。**结论：你在 5090 上大概率要用 64×64 tile（必要时 split-K），而不是 fused-muon 报告里大矩阵偏好的 128×128。**

**Step 2（融合第二个 GEMM）— 用 $\alpha/\beta$，别写自定义 epilogue**

$$B = bA + cA^2 \iff \text{tsyrk}(A, C{=}A, \alpha{=}c,\ \beta{=}b)$$

一次 kernel 调用即可，不需要像 fused-muon 那样在 epilogue 里从 global 读回 $A$。fused-muon 实测自定义 epilogue 只值 ~9 µs（2.8%），所以「省一个 kernel」的收益本来就不大，复杂度不值得。**同时建议保留 $a$ 在 GEMM3 上（`beta=a`），不要折进 $B$ 的对角线**——这样你与 PyTorch 的舍入点完全一致，第 2 步的数值对比会更干净。

**Step 3（第三个 GEMM）— 你的保守立场有官方背书**

NVIDIA 的 `use_syrk` 路径和 fused-muon 都保留 cuBLAS/`torch.addmm` 做 $BX$；Dion 与 flash-newton-schulz 有 Triton 版本但作者自己承认只在访存瓶颈的小矩阵上「边际」收益。**先不做是对的。**

**Step 4（接入）— 建议增加一条与 kernel 无关的收益项**

你双卡 DDP 的验收门槛是端到端 3%。在 DDP 下每个 rank 冗余地算同一份 NS，而 Dion 的 Muon 做法是**把一批矩阵按 rank 分摊**、算完 all-gather 更新量。2 卡就能把 NS 计算量砍掉一半（通信量 ≈ 一个 bf16 参数副本，可与反向/其他计算重叠），收益上界很可能**高于** 1.5x kernel 带来的 10%。它不改数值、不需要新 kernel（只用 `all_gather`/`all_to_all`），值得作为独立于 kernel 工作的并行分支，也能降低你「10% 算子门槛过不去」的整体风险。

---

## 五、必须钉死的 5 个数值契约（否则后续所有对比都会互相打架）

1. **归一化顺序**：先 cast 后 norm（PyTorch / Dion / fused-muon）vs 先 norm 后 cast（NVIDIA）。
2. **$a$ 的位置**：GEMM3 的 `beta`（PyTorch / NVIDIA）vs 折入 $B$ 对角线（fused-muon）vs 累加器初值（Dion）。三者实数域等价，bf16 下舍入点不同。
3. **系数表**：单组 quintic（PyTorch 默认、你的计划）vs nanogpt 5 组递进（Dion）vs AOL 4 组（Turbo-Muon）vs `cubic5`（$c=0$）。
4. **镜像写导致的非严格对称**：$A_{ij}$ 由下三角 tile 算出、$A_{ji}$ 是它的镜像副本，两者位模式相同但是「另一个归约次序」的结果，与参考的 $A_{ij}$、$A_{ji}$ 各自独立算出并不逐位相同。而且 $\text{addmm}(A,A,A)$ 会同时用两个三角，误差会传播。这一条必须按你计划里写的——逐轮记录每个 shape 的最大绝对误差、相对 L2、非有限值数量，不要声称 bitwise。
5. **累加精度**：bf16 存储 + FP32 累加（fused-muon 显式传 `CUBLAS_COMPUTE_32F`；`tsyrk_ex` 用 `tl.zeros(..., tl.float32)`），epilogue 里 $\beta C+\alpha\cdot acc$ 的顺序与 `addmm` 一致。

---

## 六、风险清单（按严重度排序）

1. **`mm.sync`/`cp.async` 路线在 Blackwell 上可能打不过 cuBLAS 的 tcgen05 GEMM。** 「SYRK 省 50% FLOPs」成立的前提是 FLOPs 是瓶颈；如果 SM120 上单位 FLOP 便宜、而你的 SYRK 只能用老式 mma 路径，净收益可能被吃掉。**缓解：Step 0 里先做一次「cuBLAS bf16 GEMM（全矩阵）vs `tsyrk_ex`」的裸对比，再决定要不要投入。**
2. **dispatcher 阈值全是 A800 硬编码**（`108` SM × 2 blocks/SM、`m ≤ 1280`、`5 ≤ n/m ≤ 8`），搬到 170 SM 的 5090 必须重新推导。
3. **不同 tile 必须分文件编译**：fused-muon 报告里 64-tile 与 128-tile 同文件编译会把寄存器从 90 顶到 254。这条要写进你的 CMake 组织方式。
4. **镜像写的正确性边界**：对角 tile 只能写一次（fused-muon 只在 $gr>gc$ 时写），非对角 tile 只有下三角那个 block 负责写镜像，否则出现 race。
5. **README 与代码不一致**（fused-muon 的「single launch」措辞），以及**无公开 SM120 目标实现**——不要按 README 的性能叙事做预算。
6. **`tsyrk_ex` 的 8 对齐门槛**：你的形状都满足，但回退路径必须实现（你的计划已写「形状未覆盖时自动回退」）。
7. **许可**：fused-muon Apache-2.0、Dion MIT、NVIDIA Apache-2.0，都可 vendor（保留署名）。

---

## 七、建议的最小实验集（在写任何 CUTLASS 代码之前）

1. **裸对比**：四个真实 shape 上，`torch.mm` / `torch.addmm` 全矩阵 vs `tsyrk_ex`（$\alpha/\beta$ 全默认）vs `tsyrk_ex` 的 small-matrix 变体，同一 stream、ABBA 交错、≥5 s warmup。这能在一天内回答「SYRK 在 5090 上是否值得」。
2. **CUDA Graph / `torch.compile` 基线升级测试**：对现有复用路径做图捕获与编译，看 1680 次拷贝和 launch 开销能掉多少（Dion 的 `dion/cuda_graph.py` 可参考）。
3. **`B = bA + cA$^2$` 是否真能用一次 `tsyrk_ex(A, A, alpha=c, beta=b)` 复现 PyTorch 的 `addmm(A,A,A,beta=b,alpha=c)`**（含对角项与舍入）。
4. **DDP 分摊 pilot**：双卡把 168 个矩阵按 rank 分半，各 rank 对本地一半做 NS，`all_gather` 后各自更新——纯 PyTorch 就能验证，可与 kernel 工作完全解耦。
5. 如果上述 1–4 都达不到 10%/3%，再考虑**改算法**分支（`cubic5`：5 步、$c=0$、10 次 matmul vs 15 次，即**整条 GEMM2 直接消失**，比 SYRK 省一半 FLOPs 更彻底；arXiv 2606.00371 报告在 1B–4B MoE/Mamba 上与 quintic 的 val loss 差距 ~$10^{-3}$；NVIDIA 的系数表里已内置 `"cubic5"`，但要求 `steps==5`）。这条路严格违反你「不改算法」的约束，只能作为 opt-in，但它决定了「如果最终要省一个数量级的 matmul，值不值得为 SYRK 投工程资源」。

---

## 八、链接汇总

- PyTorch `Muon`：`torch/optim/_muon.py`（[main](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py)）
- fused-muon：[StarrickLiu/fused-muon](https://github.com/StarrickLiu/fused-muon)（读 `docs/optimization_report.md` 和 `csrc/syrk_common.cuh`，README 仅供参考）
- NVIDIA：[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) 的 `emerging_optimizers/triton_kernels/syrk.py` 与 `emerging_optimizers/orthogonalized_optimizers/muon_utils.py`；[Megatron 博客](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)；`megatron/core/optimizer/layer_wise_optimizer.py`
- Dion：`dion/newton_schulz_triton.py`、`dion/muon.py`、`dion/cuda_graph.py`
- Turbo-Muon：[thib-s/flash-newton-schulz](https://github.com/thib-s/flash-newton-schulz)
- 反例：[Mantissagithub/muon_exps](https://github.com/Mantissagithub/muon_exps)
- cubic5：[How Much Orthogonalization Does Muon Need?](https://arxiv.org/abs/2606.00371)

**未能核实的部分**：ThunderKittens 的 symmul kernel（main 分支无代码，疑为论文/分支产物）；flash-newton-schulz 提到的「Muon+」具体仓库；A800→5090 的任何性能外推（我未做任何实测）。
