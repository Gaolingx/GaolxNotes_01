# 调研KDA加速方案

### 🧑‍💻 User

调研a100上kda kernel加速方案（chunk/recurrent

---

### 🤖 Assistant

针对 A100，KDA（Kimi Delta Attention）应采用“训练走 chunk，单 token 解码走 recurrent”的双路径设计。两者数学等价，但硬件并行方式完全不同；FLA 代码也通常把 `chunk`、`parallel` 和 `fused_recurrent` 分开实现，且训练时一般不会使用 recurrent 路径。[FLA KDA issue](https://github.com/fla-org/flash-linear-attention/issues/1106)

KDA 的核心状态可以写成：

$$
S_t = D_t\odot S_{t-1}
      + \beta_t\left(v_t-k_t^\top S_{t-1}\right)k_t
$$

其中 \(S_t\) 是固定大小的矩阵状态，\(D_t\) 是逐通道衰减门控。输出通常为：

$$
y_t=q_t S_t
$$

因此它没有标准注意力那样随上下文增长的 KV cache，但每个 token 都要读写 recurrent state。

| 场景             | 推荐路径                    |                 并行度 | 主要瓶颈                             |
| -------------- | ----------------------- | ------------------: | -------------------------------- |
| 训练、长序列 prefill | chunk / chunk-WY        | token、chunk、head 并行 | 中间状态读写、矩阵乘                       |
| 批量 prefill     | chunk 或 parallel        |                  很高 | HBM 带宽与 GEMM 利用率                 |
| 单 token decode | fused recurrent         |       batch、head 并行 | recurrent state 读写、kernel launch |
| 小 batch decode | recurrent + state cache |                  很低 | 内存延迟和 launch overhead            |

## 1. Chunk 路径：A100 上最重要

朴素 recurrent 扫描是串行的：

```text
S0 -> S1 -> S2 -> ... -> ST
```

训练时应把序列切成长度 \(C\) 的块。每个块先计算块内的局部转移，再计算块之间的状态传播：

```text
chunk 0: local computation -> S1
chunk 1: local computation -> S2
chunk 2: local computation -> S3
```

块内计算可以通过 delta-rule 的 WY/低秩分解转化为并行矩阵运算，从而使用 Tensor Core。FLA 的 KDA chunk 实现就是这一方向；其训练路径会调用 `chunk_kda`，而 `fused_recurrent_kda` 主要用于单 token decode。[相关实现说明](https://github.com/fla-org/flash-linear-attention/issues/1106)

A100 的建议配置：

* CUDA 11.8/12.x
* PyTorch 2.5 以上
* Triton 3.x，或直接写 CUDA/CUTLASS kernel
* BF16 优先；FP16 也可
* chunk size 从 64、128、256 做 autotune
* `num_warps` 通常测试 4、8
* 对 \(d_k,d_v\) 为 128 或 256 的模型重点优化
* 累加使用 FP32，状态保存 BF16/FP16

A100 的 Tensor Core 对 BF16/FP16 很友好，但它没有 H100/Blackwell 的 FP8、TMA 等硬件特性。因此不能直接照搬 H100 上依赖 FP8 或 Hopper TMA 的实现。

## 2. Chunk kernel 的计算拆分

一个较实用的 kernel 拆分是：

### Intra-chunk

计算当前 chunk 内 token 两两之间的隐式线性注意力：

$$
Y_{\text{local}} = Q_{\text{chunk}} K_{\text{chunk}}^\top V_{\text{chunk}}
$$

实际实现不会显式形成完整的 \(C\times C\) 注意力矩阵，而是利用衰减和 delta 更新结构进行分块计算。

### Inter-chunk

每个 chunk 只输出一个边界状态：

$$
S_{i+1}=A_iS_i+B_i
$$

其中 \(A_i,B_i\) 是 chunk 的压缩转移量。这样只需在 chunk 粒度上传播状态，显著减少串行步数。

### Output stage

最后将块入口状态与块内局部结果合并，生成每个 token 的输出。

性能关键不在理论 FLOPs，而在于：

1. 不要把每个 token 的 \(S_t\) 全部 materialize 到 HBM。
2. 尽量让状态停留在寄存器或 shared memory。
3. 将 \(QK^\top\)、状态更新、\(SV\) 融合。
4. 避免 Python 循环逐 chunk 调 kernel。
5. 使用 persistent kernel 或 CUDA Graph 降低 launch 开销。

长序列训练中，chunk size 越大，kernel launch 和中间状态开销越低，但寄存器/shared memory 压力也越大。A100 上一般从 128 开始测试；若 head dimension 为 128、显存充足，可以测试 256。

## 3. Recurrent 路径：用于 decode

单 token decode 时，chunk 没有意义，因为每次只有一个新 token。此时直接执行：

$$
S_t = D_t\odot S_{t-1}
+\beta_t(v_t-k_t^\top S_{t-1})k_t
$$

推荐做成 fused recurrent kernel，把以下步骤合并：

```text
q/k/v projection
-> gate/decay
-> read q @ S
-> compute delta
-> update S
-> output projection
```

至少应融合：

```text
state read
state update
output read
```

否则每个 token 会产生多次 HBM 往返，A100 上很容易变成 memory-bound。

对于 state 形状：

$$
[B,H,D_k,D_v]
$$

如果 \(D_k=D_v=128\)，单个 head 的状态有 16384 个元素。BF16 下约 32 KB；如果每层有 32 个 head，则每层约 1 MB 每 batch item。decode 时状态读写量很快成为主要成本。

建议：

* batch 较大时，让一个 CTA 负责一个或多个 head。
* batch 很小时，使用 persistent CTA，避免每 token 重新调度。
* 状态更新采用 FP32 累加，写回 BF16。
* 对衰减门 \(D_t\) 使用向量化加载。
* 对固定 \(D_k,D_v\) 做编译期 specialization。
* 推理服务中使用 CUDA Graph 固定 batch/shape。
* 多请求连续 batching 时，优先按序列长度和 head layout 排布，减少 ragged 状态访问。

## 4. A100 上 Triton 与 CUDA 的选择

### Triton 方案

优点：

* 开发快；
* 容易支持 chunk size、head dimension autotune；
* 适合先验证数学正确性；
* A100 可以生成较好的 BF16/FP16 Tensor Core 代码。

缺点：

* recurrent state 的复杂访问模式可能生成较多寄存器和边界指令；
* 编译时间和版本兼容性需要管理；
* 某些 KDA 路径在 Triton 新版本存在编译问题，例如 `chunk_kda` 曾因在 kernel 内调用 `triton.next_power_of_2` 而失败。[问题记录](https://github.com/fla-org/flash-linear-attention/issues/1106)

### CUDA/CUTLASS 方案

适合最终生产版本：

* 用 CUTLASS/CuTe 实现 chunk 内矩阵乘；
* 用手写 CUDA 实现 recurrent state update；
* 使用 `cp.async` 做 Ampere shared-memory pipeline；
* 使用 WMMA 或 CUTLASS BF16 Tensor Core；
* 对状态矩阵采用 tile 化布局，保证 coalesced load/store。

A100 没有 Hopper 的 TMA，因此应使用 Ampere 的 `cp.async`，而不是依赖 TMA 代码。CUDA Tile/Triton CUDA Tile 后端主要面向更新架构，不能假设在 A100 上获得同等收益。[NVIDIA CUDA Tile 介绍](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)

## 5. FLA 现成方案的现实情况

最短路径是直接使用 `fla-core` 的 KDA 实现：

```python
from fla.ops.kda import chunk_kda, fused_recurrent_kda
```

典型 dispatch：

```python
if training or seqlen > 1:
    y, state = chunk_kda(q, k, v, beta, g)
else:
    y, state = fused_recurrent_kda(
        q, k, v, beta, g, initial_state=state
    )
```

需要注意：

* FLA 的 backend dispatch 可能根据 `torch.is_grad_enabled()`、dtype、设备和 shape 选择路径。
* 安装 `fla-core` 不等于一定调用了 FlashKDA；训练时通常仍会进入 Triton chunk 路径。
* A100 上应检查实际 kernel，而不是只看包是否安装：

```bash
nsys profile -t cuda,nvtx,osrt python train.py
```

或：

```bash
torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
)
```

重点搜索：

```text
chunk_kda
fused_recurrent_kda
flash_kda
triton
aten::einsum
aten::bmm
aten::matmul
```

如果看到大量：

```text
aten::einsum
aten::index
aten::copy_
aten::where
```

而没有大型 Triton/CUDA kernel，说明实现发生了 fallback，性能很可能只有理论值的一小部分。

## 6. A100 上最可行的优化优先级

我建议按以下顺序实施：

1. **先确认 dispatch**

   * 确认训练走 `chunk_kda`；
   * 确认 decode 走 `fused_recurrent_kda`；
   * 禁止隐式回退到 eager einsum。

2. **固定 shape specialization**

   * \(d_k,d_v\in\{64,128,256\}\) 分别编译；
   * 对常用 chunk size 单独 autotune；
   * 避免所有维度都用动态 shape。

3. **融合 KDA 更新**

   * 将 decay、delta、state update 放进一个 kernel；
   * 不要把中间的 \(k^\top S\) 写回显存。

4. **减少状态 materialization**

   * 只保存 chunk boundary state；
   * 反向时使用 recompute 或 selective checkpoint；
   * 不保存每 token 的完整 \(S_t\)。

5. **训练使用 BF16 Tensor Core**

   * Q/K/V 保持 BF16；
   * 矩阵累加 FP32；
   * 避免无意中的 FP32 输入导致 Tensor Core 失效。

6. **decode 使用 persistent recurrent kernel**

   * batch 小于 8 时尤其重要；
   * 配合 CUDA Graph 和 continuous batching。

7. **再考虑 CUDA 重写**

   * Triton 版本达到可接受性能后，再把热点路径移植到 CUTLASS/CUDA；
   * 不建议一开始就同时重写 chunk 和 recurrent 两条路径。

## 7. 可参考的新方向

TFLA 提出了在 chunk 内再做一层序列分块，以支持更大的 chunk size，目标是减少中间状态和 IO；这个思想对 KDA 的长上下文训练同样有参考价值。[TFLA 论文](https://arxiv.org/abs/2503.14376)

另一个方向是 DeltaLog：它不在每个 decode token 后都完整写回 dense recurrent state，而是累积紧凑更新，周期性合并。论文报告在 KDA 等模型上可显著减少状态写流量，并获得最高约 1.86 倍 recurrent kernel 加速。[DeltaLog](https://arxiv.org/abs/2608.15533)

不过 DeltaLog 更适合推理服务；训练反向传播仍应优先采用 chunk-WY 或其他 chunk-parallel 算法。

对你的 A100 目标，最现实的组合是：

```text
训练：BF16 Triton/CUDA chunk-KDA
长 prefill：chunk-KDA，chunk=128/256 autotune
单 token decode：fused recurrent-KDA + persistent CTA
服务端：CUDA Graph + continuous batching
后续优化：chunk 内序列 tiling，减少 state materialization
```

如果当前模型训练速度只有 full GQA 的十分之一，最应先排查的是 fallback、未融合的状态更新，以及反向阶段 materialize 全部 recurrent state；KDA 的理论线性复杂度本身并不能保证 A100 上取得高吞吐。


---

### 🧑‍💻 User

flashkda在fla（triton baseline）上还针对hopper/blackwell的硬件特性进行了哪些极致优化？

---

### 🤖 Assistant

是的。FlashKDA 相对 FLA 的 Triton baseline，核心提升不是简单地“把 Triton 改成 CUDA”，而是围绕 Hopper/Blackwell 的异步执行、Tensor Core 和内存层级重新设计了 kernel。

目前公开资料显示，Moonshot 的 FlashKDA 是基于 CUTLASS 的专用实现，而 FLA 的 KDA 主路径仍是 Triton chunk/recurrent 实现。[FlashKDA 发布信息](https://www.marktechpost.com/)

主要极致优化包括：

| 优化点     | Hopper H100/H800          | Blackwell B100/B200        |
| ------- | ------------------------- | -------------------------- |
| 矩阵乘指令   | WGMMA 异步 warp-group MMA   | `tcgen05.mma` / UMMA       |
| 全局到共享内存 | TMA、`cp.async.bulk`       | TMA、TMA multicast          |
| 累加位置    | 寄存器/共享内存                  | Tensor Memory（TMEM）        |
| 并行组织    | warp-group specialization | CTA pair、cluster           |
| 低精度     | FP16/BF16/FP8             | FP16/BF16/FP8、FP4/MXFP4    |
| 调度方式    | persistent kernel、异步流水    | persistent cluster、异步 UMMA |

## 1. Hopper：TMA + WGMMA 异步流水

FLA Triton baseline 通常还是：

```text
tl.load
-> tl.dot
-> state update
-> tl.store
```

FlashKDA 会把它拆成 producer/consumer 流水：

```text
TMA load tile i+1
      || 
WGMMA compute tile i
      ||
state/epilogue tile i-1
```

TMA 由单个线程发起多维 tensor 搬运，负责 global memory 到 shared memory 的地址生成和边界处理；WGMMA 则允许 warp-group 异步发起矩阵乘，同时与加载、门控和状态更新重叠。[Hopper TMA/WGMMA 机制](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction)

对 KDA 来说，收益主要来自：

* Q/K/V tile 与 recurrent state tile 的双缓冲；
* 在 WGMMA 执行期间计算 decay、beta、delta；
* 减少线程参与地址计算；
* 将局部 chunk 乘法转为 Tensor Core MMA；
* 让状态边界传播与下一块计算重叠。

这对 chunk-KDA 的 intra-chunk 部分尤其重要，因为它有大量形如：

$$
QK^\top,\quad KV,\quad KS,\quad QS
$$

的矩阵乘。

## 2. Warp specialization

FlashKDA 通常会把一个 CTA 中的 warp-group 分工：

```text
warp-group 0: TMA load
warp-group 1: WGMMA
warp-group 2: gate/decay
warp-group 3: epilogue/state store
```

这样做的目的，是避免所有 warp 在 load、MMA、状态更新之间同步切换。Hopper 的 WGMMA 和 TMA 都是异步指令，因此这种 producer/consumer 结构能显著提高流水重叠度。

FLA Triton kernel 如果使用普通 `tl.load`、统一 warp 执行流，通常会出现：

* load 阶段 Tensor Core 空转；
* state update 阶段 MMA 空转；
* `tl.where` 和 mask 指令占用执行资源；
* barrier 过多。

## 3. 更激进的 state tiling

KDA 的 recurrent state 是矩阵，而不是普通向量：

$$
S\in \mathbb{R}^{d_k\times d_v}
$$

FlashKDA 会对 \(S\) 做二维 tile 化，将其分片放入 shared memory 或寄存器，并让多个 warp-group 协作完成：

$$
qS,\quad k^\top S,\quad
S\leftarrow DS+\beta(v-k^\top S)k
$$

关键点是避免：

```text
每个 token：
  读完整 S
  更新完整 S
  写完整 S
```

而是：

```text
加载一个 state tile
-> 多个 token/多个 MMA 累积
-> 一次写回
```

这会降低 HBM traffic，也降低 recurrent path 的状态写回次数。

## 4. Hopper 上的 persistent kernel

FlashKDA 还可以让 CTA 常驻 SM，连续处理多个 chunk 或多个 decode token，减少：

* kernel launch；
* CTA 重新调度；
* state pointer 重新计算；
* 中间 tensor 写回。

对于 decode，persistent recurrent kernel 特别有价值，因为每个 token 的计算量不大，launch overhead 很容易占主要比例。

如果 batch 很小，FLA 的普通 recurrent Triton kernel 往往会表现为：

```text
launch latency + state load/store latency > arithmetic
```

persistent kernel 能显著改善这一点。

## 5. Blackwell：UMMA、TMEM 和 CTA pair

Blackwell 不只是 Hopper 的更快版本。其数据中心版本使用 `tcgen05.mma`/UMMA，替代 Hopper 的 WGMMA；同时增加 Tensor Memory（TMEM）作为 Tensor Core 专用累加存储。[Blackwell UMMA/TMEM 机制](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)

FlashKDA 如果针对 Blackwell 单独优化，重点会是：

* 用 UMMA 异步执行 chunk 内矩阵乘；
* 将累加器放入 TMEM，减少寄存器压力；
* 用 CTA pair 跨两个 SM 协同计算；
* 利用 cluster 内的分布式 shared memory；
* 使用 TMA multicast，让多个 CTA 共享 Q/K/V tile；
* 用硬件 block-scaled MMA 支持 FP4/MXFP4。

TMEM 对 KDA 很关键，因为 state tile 和 MMA accumulator 都很大。放在寄存器里会导致：

* register spilling；
* occupancy 降低；
* warp 数减少；
* shared memory 与寄存器之间频繁搬运。

Blackwell 的 UMMA 允许 MMA 累加直接落入 TMEM，从而释放寄存器。`tcgen05.mma` 还支持硬件 block scaling，可把反量化和 MMA 合并。[Triton block-scaled MMA 文档](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html)

## 6. Blackwell 的 FP4/MXFP4 路径

如果 FlashKDA 将 Q/K/V 或投影层使用 FP4/MXFP4，Blackwell 可以使用：

```text
quantized operand load
-> scale load
-> dequantization
-> MMA
```

融合成硬件 block-scaled MMA。这样避免显式：

```text
load FP4
-> convert FP16
-> multiply scale
-> MMA
```

Triton 文档报告，使用 TMA 加载量化数据和 scale，可比普通布局转换获得约 35% 加速；硬件 `tcgen05_mma_scaled` 则进一步融合缩放与矩阵乘。[Triton block-scaled MMA](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05-mma-scaled.html)

但这部分对 A100 不能直接复用。A100 没有 Blackwell 的 native FP4/MXFP4 MMA，只能软件反量化。

## 7. KDA 特有的门控融合

相对普通 linear attention，KDA 多了：

* channel-wise decay；
* beta gate；
* delta correction；
* state erase/update。

FlashKDA 会尽量把这些操作融合进主 kernel：

```text
load q/k/v
load beta/decay
compute kT_S
compute delta
apply decay
update state
compute qS
store output
```

FLA baseline 如果拆成多个 Triton kernel，则会产生大量中间张量：

```text
gate kernel
-> delta kernel
-> state kernel
-> output kernel
```

这会让理论上的 \(O(T)\) 复杂度在实际中被 HBM IO 主导。

## 8. 对 A100 的实际含义

FlashKDA 的 Hopper/Blackwell 专用部分，A100 不能原样获得：

* 没有 TMA；
* 没有 WGMMA；
* 没有 TMEM；
* 没有 UMMA；
* 没有 CTA pair/cluster DSMEM；
* 没有 native FP4 block-scaled MMA。

A100 对应的最佳实现应改成：

```text
cp.async 双缓冲
+ shared-memory tiled MMA
+ BF16/FP16 WMMA/CUTLASS
+ persistent CTA
+ fused gate/state update
+ chunk boundary state only
```

也就是说，A100 版本最值得移植的是 FlashKDA 的**算法级融合、state tiling、persistent 调度和减少 materialization**；不能直接移植其 Hopper/Blackwell 的指令级实现。

最现实的优先级是：

1. 先将 FLA 的 chunk-KDA 改成 fused CUDA/CUTLASS；
2. 用 `cp.async` 做 Q/K/V 和 state 双缓冲；
3. 用 shared memory 保存 state tile；
4. 用 BF16 Tensor Core 计算 intra-chunk GEMM；
5. 把 decay、beta、delta 和 state update 融合；
6. decode 单独实现 persistent recurrent kernel；
7. 最后再做 A100 专用 autotune。

另外，Blackwell 上的新 Triton 编译器可以自动利用部分新 Tensor Core 能力；官方称现有 Triton flash-attention kernel 在 Blackwell 上无需改代码即可获得最高约 1.5 倍 FP16 attention 加速，但这不等于复杂的 KDA recurrent/state kernel 会自动获得同样收益。[NVIDIA Triton Blackwell 优化](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
