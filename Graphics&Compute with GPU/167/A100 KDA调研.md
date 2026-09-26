# A100 KDA调研

## 🧑‍💻 User

调研a100上kda kernel加速方案（chunk/recurrent

---
## 🤖 Assistant

我先查一下有哪些可用的搜索/资料工具。

下面是针对 **KDA（Kimi Delta Attention，Kimi Linear 的核心算子）在 A100 上 chunk/recurrent 两种 kernel 加速方案**的调研总结。

## 1. 背景：KDA 是什么

KDA 是 Gated DeltaNet 的逐通道（per-channel gate）推广版。递推形式为：

$$S_t = S_{t-1}\,\big(\mathrm{Diag}(\gamma_t) - \beta_t \gamma_t k_t k_t^\top\big) + \beta_t v_t k_t^\top,\qquad o_t = \gamma_t \odot (S_t^\top q_t)$$

其中 $\gamma_t \in \mathbb{R}^{K}$ 是逐通道 log-domain gate（Kimi Linear 中由 $-\exp(A\_\log)\cdot\mathrm{softplus}(g+b)$ 产生），转移矩阵是 **DPLR（Diagonal-Plus-Low-Rank）** 结构。论文用的特殊形式使 chunk 内累计的门 $\Gamma$ 满足 $a_{ij} = \exp(\Gamma_i - \Gamma_j)\cdot \mathbf{1}[i>j]$，可把通用 DPLR 的 $O(K)$ WY-representation 压缩成一个 rank-1 修正，显著减少 chunk 化的额外计算。

代码开源在：
- `fla-org/flash-linear-attention`：`fla/ops/kda/`（`chunk.py`、`chunk_fwd.py`、`chunk_bwd.py`、`fused_recurrent.py` 等）
- vLLM 已合入 KDA 支持（Triton 后端，chunk prefill + recurrent decode）

## 2. 两种计算模式的对比（A100 视角）

| 维度 | Recurrent（逐 token） | Chunk-wise（分块并行） |
|---|---|---|
| 复杂度 | $O(T)$ 顺序步，每步 rank-1 更新 $S[K,V]$ | chunk 内并行 GEMM，chunk 间顺序传递状态 |
| 并行度 | 只有 $B\times H$ 维并行，decode 时严重 SM 不足利用 | chunk 内 $64\times64$ 块级并行 + Tensor Core |
| 适用场景 | **decode**（$T_q=1$，无 prefill 可摊销） | **training / prefill / 长上下文** |
| A100 上的瓶颈 | HBM 带宽（每步换入/换出 $KV$ 状态） | Tensor Core 吞吐；KDA 支持 K≤256，state 较大导致 shared memory / register 压力，难用大 tile |
| 数值精度 | state 必须 fp32 保存 | state 累加用 fp32，GEMM 用 bf16 |

Kimi Linear 论文报告的 KDA kernel 相比 MLA 在 64K/1M 长度下训练吞吐有约 2 倍提升，但这主要针对 H800/Ampere-class 集群；A100（sm80）的特殊限制见下文。

## 3. A100（sm80）上的关键优化点

### 3.1 Chunk kernel（fla `fla/ops/kda/chunk_fwd.py` + `chunk_bwd.py`）

fla 已将 KDA chunk 化分解为几组独立的 Triton kernel：

1. **`chunk_fwd_kernel` 系列 / `fla.ops.common` 子 kernel**
 - **intra-chunk**：计算 chunk 内的 $A_{qk} = \gamma_i \odot q_i k_j^\top \gamma_j \odot \beta_j$（严格下三角）、求解 UT transform 的逆（`solve_tril` / 用 $I - A_{kk}$ 递推求逆，把 $C\times C$ 三角求逆做成 GEMM 序列）。
 - **inter-chunk**：顺序传递状态 $h$，$h \gets h\cdot\mathrm{Diag}(\exp(\Delta\Gamma)) + (w^\top) \cdot v\_new$。
 - **output**：$o = (A_{qk} \cdot v\_new) + (q\odot \gamma) \cdot h$。

2. **fused gate / fused l2norm / fused beta-sigmoid**
 - `use_gate_in_kernel=True`：在 kernel 内直接算 $-\exp(A_{\log})\cdot\text{softplus}(g+b)$ 及其 chunk 内 cumsum，省掉独立的 gate logits 物化 + cumsum kernel 及其反向（省显存和省 RTT 各一次）。
 - `use_qk_l2norm_in_kernel=True`：L2 norm 融合进 kernel。
 - `use_beta_sigmoid_in_kernel=True`：sigmoid 融合。

3. **`safe_gate` 与 Tensor Core 形状**（对 A100 很关键）
 - KDA 状态转移 $S_t$ 的递推需要每步一个 rank-1 修正，fla 的 chunk kernel 采用 **UT transform** 把 chunk 内的所有 rank-1 更新折叠成低秩矩阵乘法。Triton 的 `tl.dot` 在 Ampere 上要求矩阵维度 $\ge 16$；`safe_gate=True` + `lower_bound` 会把门 clamp 到 $[-5, 0)$，使 kernel 能走 **M=16 N=16 K=16 的 mma.16816** 形状，不然要被 pad 到更差的布局。
 - A100 的 bf16 Tensor Core 峰值 312 TFLOPS（稠密），TL.dot bf16 GEMM 是性能主来源；**K** 应取 64/128/256 的对齐值。

4. **chunk_size 选择**
 - KDA 仅支持 32/64（因为状态维度大），主流是 **64**。A100 上 shared memory 只有 164 KB / SM，$[C,K,V]$ 的中间 buffer 很容易爆，fla 的实现里配套了 `--maxnreg`、分阶段 matmul、以及 `BK/BV` 可调 block size。

5. **recompute vs cache**
 - 训练时默认 `disable_recompute=False`，即在 backward 里重算 $A_{qk}, A_{kk}, w, u$ 等中间量，换取显存大幅下降；A100 的 40/80 GB 显存对 7B~48B Kimi Linear 模型是紧的，**建议保留 recompute**，除非训练小模型且追求吞吐。

### 3.2 Recurrent kernel（decode 路径）

KDA 的 decode 是逐 token 的 $S \leftarrow S\circ(\gamma 1^\top - \beta\,\gamma k k^\top)$ 的另写法；fla 提供两个实现：

- **`fused_recurrent`**：单 kernel 循环整个 $T$，每个 step 实时更新 state。仅适合**短序列 / 1 token decode**。
- **`fused_sigmoid_gating_mlp_update`** 风格的单步 kernel：T=1 的 fused kernel，把 `sigmoid(beta)`、gate、L2 norm、状态更新、输出投影全部塞进 1 个 kernel，主要耗时是 state 的 global memory read/write。

A100 上 decode 优化点：
1. **state layout**：用 `state_v_first=True`（把 state 存成 $[V,K]$）对输出 $o = q^\top S$ 这一侧是 coalesced 的，deeper KV head 时收益明显；反之 $[K,V]$ 对写回 $S k$ 友好。A100 的 L2 40 MB，layout 选择影响显著。
2. **把 state 常驻 SMEM / register**：$K=V=128$、fp32 时 state 是 64 KB，已经超 164 KB 一半，因此 head 维度大时难以整块常驻；fla 默认走 HBM，靠 multi-stream + fused kernel 减少 launch overhead。
3. **CUDA Graph**：vLLM / fla 提供 `use_graph=True`，要求 `cu_seqlens` padded、固定 max_num_seqs，A100 上 decode 时 kernel launch 开销占比高（KDA 每步要 launch 5~6 个小 kernel 或 1 个 fused kernel），CUDA Graph 收益通常 >10-20% wall time。
4. **批量并行（GVA / Grouped Value Attention）**：K=HV 头数分离，解码时 batch 增大显著摊薄 state IO，A100 内存带宽 1.6/2.0 TB/s 是唯一硬上限。

## 4. 可调参数速查

```python
from fla.ops.kda import chunk_kda
o, ht = chunk_kda(
    q, k, v, g, beta,
    A_log=A_log, dt_bias=dt_bias,
    use_qk_l2norm_in_kernel=True,   # 省一个 kernel launch + 显存
    use_gate_in_kernel=True,        # 省 gate cumsum 显存（T×H×K 的 fp32 buffer）
    use_beta_sigmoid_in_kernel=True,
    safe_gate=True, lower_bound=-5, # mma 16×16 对齐 + 门可控
    chunk_size=64,                  # A100 唯一推荐值
    cu_seqlens=...,                 # varlen 训练/推理
)
# decode 用 fused_recurrent + use_graph=True, state_v_first=<按布局选>
```

## 5. A100 上的典型瓶颈与建议

| 问题 | 原因 | 对策 |
|---|---|---|
| chunk kernel occupancy 低 | $K,V$ 大 → shared mem / register 爆 | 用 chunk_size=64 + `chunk_size=32` 对比；调小 `BV` |
| backward 慢于 forward | recompute 路径多算一次 intra-chunk | 小模型可 `disable_recompute=True`；大模型保持开启并加大 micro-batch |
| decode 带宽打满 | state fp32 每 token 读写 $2KV$ bytes | `state_v_first` 选 layout；CUDA Graph；fp8/bf16 state 实验（fla 未默认支持） |
| varlen 训练 padding 浪费 | cu_seqlens 未展平 | 展平为 B=1 + `cu_seqlens`，配合 `use_graph` |
| Triton 在 sm80 上 autotune 抖动 | `TRITON_CACHE` key 变化 | 预热后冻结 config；`use_graph` 前先热身 |

## 6. 主要参考

- Kimi Linear 技术报告（KDA 的 chunkwise DPLR 推导）：`https://arxiv.org/abs/2510.26692`
- 官方实现：`https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda`
- vLLM 集成（`vllm/model_executor/layers/kda.py` 及 fla backend）：github.com/MoonshotAI/vllm 分支
- Gated DeltaNet（前身）：`arXiv:2412.06464`，chunk 化的 WY representation 同样适用

如果你告诉我具体场景（训练 vs 推理、模型尺寸、序列长度、是否 varlen），我可以进一步给出在 A100 上的 profile 建议（例如 nsight 里应该看哪些指标、推荐 Triton config）。

---
## 🧑‍💻 User

flashkda在fla（triton baseline）上还针对hopper/blackwell的硬件特性进行了哪些极致优化？

---
## 🤖 Assistant

让我先查一下有哪些可用的搜索/资料工具，再核实 FlashKDA 的实现细节。

找到了 **MoonshotAI/FlashKDA**。现在抓取官方 README 和几篇深度解析。

官方仓库的 default branch 是 `master`。现在抓取官方 deep-dive 文档和 README。

官方 deep-dive 显示它走的是 **SM80-only MMA 路径**（刻意不用 Hopper 专属指令）。我需要核实源码里是否另有 Hopper/Blackwell 专属分支。

已核实到全部一手资料（官方 README、官方 v1 deep-dive、`BENCHMARK_H20/GB200`、以及基于源码生成的架构文档）。先做一个重要的前提校正，再逐条拆解它在 NVIDIA 硬件特性上的"极致"用法。

## 0. 前提校正（很关键）

有三点需要先对齐，否则后面的分析会跑偏：

1. **FlashKDA 不是 FLA 里的 kernel**，而是 MoonshotAI 独立开源的 CUTLASS/CUDA 实现（`MoonshotAI/FlashKDA`，MIT，~1.3k star）。它通过 FLA 的 backend 机制在 `fla.ops.kda.chunk_kda` 里**自动接管**（FLA PR #852），设 `FLA_FLASH_KDA=0` 即回落 Triton 路径。所以准确说法是：**FlashKDA 是 fla_chunk_kda（Triton）的替代后端**。
2. **它刻意没有走 Hopper 的 wgmma / Blackwell 的 tcgen05**。官方 deep-dive 原话是"SM80-only MMA path"——所有 $\text{CHUNK}=16$ 的算子在 SM80 `mma.sync`（`SM80_16x8x16_F32BF16BF16F32_TN` / `_F16F16F16F16_TN`）上干净映射，卖点是**跨代可移植 + 简化 kernel**。DeepWiki 生成的文档里有 "GMMA" 的措辞，但与同页"SM80 16x8x16 atom"和官方文档冲突，我判断是派生文档的用词不严谨。
3. 因此**真正的 Hopper/Blackwell 专属优化集中在"数据搬运 + 流水 + 占用率 + 精度路径"，而不是张量核心指令本身**。下面按这条主线展开。

## 1. 硬件特性 → 用法 → 相对 Triton 的增量

| 硬件特性 | FlashKDA 用法 | 为什么 Triton baseline 拿不到 |
|---|---|---|
| **TMA** (`cp.async.bulk.tensor`) | K1/K2 全部用 TMA descriptor 收发；workspace 每个子 buffer 强制 128B 对齐（`static_assert`） | Triton 的 tile load 无法做到"单线程发射整块 + 地址算术与寄存器解耦" |
| **`cutlass::PipelineTmaAsync` + mbarrier** | K2 多级 pipeline 把 workspace 的 TMA load 与 MMA 重叠 | Triton 的 software pipelining 由编译器决定，stage 数与 barrier 不可控 |
| **Warp specialization** | `WarpRole::LOAD_QKG` / `MMA` / `STORE`，128 线程（4 warps）分工，warp0 在 load↔mma 间切换 | Triton 当前无法表达 producer/consumer warp 分工 |
| **`elect_one_sync`** | TMA 事务的 leader 选举，单线程 issue | 无对应原语 |
| **`movmatrix`**（文中称 `MOVM_T`） | K2 在寄存器文件内直接转置操作数，消除 stage 间 smem round trip | Triton 的 `tl.trans` 会走 smem |
| **高 smem / 高 occupancy** | `__launch_bounds__(256, 8)` + `SharedStorageK1` union 复用，省 ~14KB/CTA | Triton 无等价的 smem lifetime 复用与 launch bounds 控制 |
| **`ex2.approx` / `tanh.approx` SFU** | 门控激活走 PTX 近似指令 | Triton 一般生成 `exp`/`sigmoid` 的通用实现 |

## 2. 结构性前提：CHUNK=16 + 双 kernel 切分

这两步是"让上面的硬件特性能用上"的前置条件。

- **$\text{CHUNK}=16$（FLA 用 64）**，三个理由：
 1. `lower_bound=-5` 时 $\exp(\text{cumsum}(g))$ 落在 bf16 可表示范围内，**彻底不需要 chunk 内 rescaling**；
 2. $16\times16$ 求逆极便宜，且可用 **Neumann 级数**直接展开 $(I+L)^{-1}\approx I-L+L^2-L^3+\dots$，无需任何分解（`neumann_inv_fused_1warp`，fp16，单 warp）；
 3. 全部算子映射到 SM80 `mma.sync`，不依赖架构专属特性 → 一份代码编到 90a/100a/103a/120a。
- **K1 / K2 切分**：K1 token-parallel（grid $N\times H\times\text{num\_chunks}$），K2 head-parallel（grid $N\times H$）。早期单 kernel 原型里，K1 的高并行度被 K2 低并行度的递推拖住，**大量 SM 空转**；拆开后端到端 **≥15%** 提速，且两段可独立调优。
 - 对 Hopper 的意义：K2 只有 $N\times H$ 个 CTA，但靠 deep TMA pipeline + warp specialization，少量 CTA 也能把 tensor core 打满——这正是 Triton 单 kernel 顺序 recurrence 做不到的。

## 3. 精度路径上的非常规取舍

| 决策 | 内容 | 收益 |
|---|---|---|
| **state 存 bf16** | on-chip 递推状态 `state_acc` 用 bf16，更新用 **fp32 FMA** | smem 占用**减半**，且把 `fp32→bf16` cast 从每个喂 state 的 GEMM 关键路径上**移除**；实测无可见精度损失 |
| **逆矩阵用 fp16** | $16\times16$ 逆元素被证明有界于 $[-1,1]$ | fp16 动态范围足够，且比 bf16 给 Neumann 级数更多 headroom；同时避开 bf16 MMA 所需的 cast |
| **`ex2.approx.ftz.f32`** | `g_act` 阶段换底到 2 | 省掉换底 FMA，SFU 吞吐高于 `exp` |
| **`tanh.approx.f32` 实现 sigmoid** | $\text{sigmoid}(x)\approx 0.5\tanh(0.5x)+0.5$ | 比通用 sigmoid 快且门控路径精度够 |
| **fp32 state 用 smem 中转** | `state_fp32_buf` 与 pipeline buffer 做 union alias，`smem_cvt_fp32_to_bf16` 块级转换 | 只在用户真要 fp32 state 时才付代价 |
| **编译期全特化** | `DISPATCH_STATE` / `LAUNCH` 宏对 `HasStateIn × HasStateOut × StateFP32 × IsVarlen` 做模板分派 | 关键路径零 runtime branch |

## 4. 寄存器 / SMEM 级微观优化

- **`movmatrix` 寄存器内转置**：K2 把 chunk 递推的各 stage 紧融合，用寄存器转置替代 smem 中转。这同时做两件事——**去掉中间 smem 往返**（Hopper 上 smem/LSU 带宽是真实瓶颈），并**缩小 K2 的 smem 需求**。
- **K1 的 smem union + launch bounds**：`SharedStorageK1` 用 union 让 Phase A（`q,k,g`）与 Phase B（`k_decayed/q_decayed/k_restored/L/INV/Mqk`）**共享同一块内存**，省 ~14KB/CTA；配合 `__launch_bounds__(256, 8)`，用少量 register spill 换 8 个 CTA/SM。$256\times 8=2048$，正好是 Hopper 的 2048 threads/SM 上限 = **满 occupancy**。
- **state 常驻 smem**：K2 的 `state_acc` 只在 chunk 循环之间进出 smem，chunk 内的 delta-rule 与 output projection 都在 register 里完成。
- **workspace 布局编译期确定**：`WorkspaceSizes` 把每个 per-tile buffer（`k_decayed`/`q_decayed`/`k_restored`/`g_total`/`INV`/`Mqk`，各 512B~4KB）算成 `constexpr`，并 `static_assert` 全部 128B 对齐以满足 TMA。
- **`beta` 预处理转置**：host 侧把 `[T,H]` 转成 `[H,T]`，为的是在 kernel 里用 **1D TMA** 加载——减少 descriptor 维度与地址计算。

## 5. 编译与架构开关

- 支持 `FLASH_KDA_CUDA_ARCHS=auto|all|90a,100a,103a,120a`——注意 **`a` 后缀**（compute_90a/100a）才解锁 TMA、mbarrier、warp-specialized 等加速特性，普通 `sm_90` 编不出这些。
- 硬性依赖 **CUDA ≥ 12.9** + 锁定的 CUTLASS submodule（`cutlass @ 5c149f5`）。
- NVCC：`-O3 --use_fast_math --register-usage-level=10`（显式调 register 使用等级，配合 K1 的 spill-for-occupancy 策略）。

## 6. 为什么 Blackwell 上相对加速比 Hopper 更大

| 平台 | 配置 | flash_kda (ms) | fla_chunk_kda (ms) | 加速 |
|---|---|---|---|---|
| H20 | H=96 fixed | 2.62 | 4.84 | **1.85×** |
| H20 | H=96 varlen 1024×8 | 2.04 | 4.67 | **2.29×** |
| H20 | H=64 varlen 1024×8 | 1.40 | 3.22 | **2.31×** |
| GB200 | H=96 fixed | 1.01 | 2.33 | **2.31×** |
| GB200 | H=96 varlen 1024×8 | 0.71 | 2.31 | **3.27×** |
| GB200 | H=64 varlen 1024×8 | 0.48 | 1.54 | **3.21×** |

两个规律：

1. **varlen 收益最大**（最高 3.27×）——K1 的 token 级并行 + `cu_seqlens` 线性扫描在变长负载下负载均衡更好，而 K1 与 K2 是分离的，不会互相拖累。
2. **Blackwell 相对增益高于 Hopper**——按官方解读，这是 Blackwell 更大的 L1/smem 与更高的 warp scheduler 吞吐**直接放大了它的融合与 state 复用优势**。注意：这是**架构红利**，不是 kernel 里有 `tcgen05` 分支。同一份 SM80-MMA 代码被编到 `100a/103a/120a` 上跑。

## 7. 它没有做什么（边界）

- **只有前向**，没有 backward → 训练不支持。FLA 的 Triton `chunk_kda` 仍然是训练路径。
- 硬约束 $K=V=128$、主张量必须 bf16、SM90+。
- **K2 的 grid 只有 $N\times H$**：小 batch / 少 head 时并行度不足，全靠 deep pipeline 掩盖——这是设计上的软肋，也是它换来 K1 高 occupancy 的代价。
- 未使用 Hopper `wgmma` / Blackwell `tcgen05`+TMEM：上限上确实放弃了新一代张量核心的峰值（wgmma 能在更少寄存器下做更大 tile），换来的是跨代一份代码。**如果你只关心 H100/B200 的极限，这是一个可以质疑的设计选择。**

## 8. 一句话总结

FlashKDA 的"极致"不在张量核心指令，而在**把 Hopper 的 TMA + mbarrier + warp specialization 用满、把 smem/register 压到极限、并把数值精度设计成让这些优化成立的前提**：CHUNK=16 让 bf16 数值安全、让 $16\times16$ Neumann 求逆可用 `mma.sync` 完成；bf16 state + fp16 逆 + `movmatrix` + smem union + `__launch_bounds__(256,8)` 共同把 smem 与寄存器压力压到能维持满 occupancy；K1/K2 拆分裂把 token 级并行与 head 级递推解耦。相对 FLA Triton baseline 得到 1.85–3.27×，varlen 与 Blackwell 上收益最大。

**参考**：官方 deep-dive `docs/20260420-flashkda-v1-deep-dive.md`；`BENCHMARK_H20.md` / `BENCHMARK_GB200.md`；FLA PR #852。

---

需要的话，我可以继续做两件事之一：
- 把上面每一条优化映射到 `csrc/smxx/` 下的具体函数（`make_load_pipeline`、`neumann_inv_fused_1warp`、`smem_cvt_fp32_to_bf16` 等），给一份"读源码顺序"；
- 或者用 `benchmarks/ncu.sh` 的 `--set full` 配置，列出在 H100 上验证哪些指标能证明这些优化确实生效（而非被其他瓶颈掩盖）。
