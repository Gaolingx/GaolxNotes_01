# CUDA图与融合

## 🧑‍💻 User

torch里边用cuda graph和我手动写triton.jit fused kernel and patch eager有什么区别？

---
## 🤖 Assistant

# 一句话结论

这俩解决的**根本不是同一个瓶颈**，属于两个正交的优化层：

- **CUDA Graph** 是**调度层（launch / CPU overhead）优化**：kernel 本身一个都没变，还是那 N 个 kernel，只是把 "CPU 逐个发车" 变成 "一次回放整张图"。它**不改变计算量、不改变显存搬运**，只管"发车的成本"。
- **手写 Triton fused kernel + patch eager** 是**算子层（kernel / 访存）重写**：你把原来 5 个 eager op 合并成 1 个 kernel，直接减少 kernel 数量 **并且**减少中间张量在显存里的来回读写。它管的是"跑一趟要搬多少字节"。

可以记成：**Graph 省的是“打电话的功夫”，Triton fusion 省的是“路上的里程”。**

---

## 1. CUDA Graph 在做什么

把一个刚性的 kernel 序列**录制**成 DAG，之后一次 `replay()` 全部重放。

```python
import torch

g = torch.cuda.CUDAGraph()
static_in  = torch.randn(16, 512, 4096, device="cuda")
static_out = torch.empty_like(static_in)

# 1) warmup：必须在捕获前把 autotune / lazy init / cudnn benchmark 全部跑掉
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_out = model(static_in)
torch.cuda.current_stream().wait_stream(s)

# 2) 捕获
with torch.cuda.graph(g):
    static_out = model(static_in)

# 3) 回放：换数据要拷贝进 static buffer
static_in.copy_(new_input)
g.replay()
```

**收益来源**：每个 kernel launch 的 CPU 侧开销约 $3\sim10\,\mu s$，外加 driver / 调度 gap。当你有几百个小 kernel、GPU 又很快时，GPU 会大量空转等 CPU 发车，这就是 **launch-bound**。Graph 把整张图的提交压成一次，把这段 gap 抹平。

**代价 / 约束**：

- **形状必须静态**（地址、grid、参数都在捕获时固化）。
- 输入/输出必须用**固定的 static buffer**，配合 graph memory pool（或 `cudaMallocAsync`）。
- 捕获期间**不能有 CPU 同步、动态控制流、`.item()`、host-side branch**。
- RNG、cudnn workspace、多流同步都要特殊处理。
- **对显存流量、算力强度毫无帮助**。如果你本来就是 memory-bound 或 compute-bound，Graph 收益≈0。

---

## 2. 手写 Triton fused kernel + patch eager 在做什么

把一串 eager op 改写成**一个 kernel**，中间结果留在 register/SMEM，不落显存。

```python
import triton
import triton.language as tl

@triton.jit
def linear_gelu_bias(X, W, B, Out, M, N, K,
                     stride_xm, stride_xk, stride_wn, stride_wk,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(X + rm[:, None] * stride_xm + (k + rk)[None, :] * stride_xk,
                    mask=(rm[:, None] < M), other=0.0)
        w = tl.load(W + rn[None, :] * stride_wn + (k + rk)[:, None] * stride_wk,
                    mask=(rn[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
    b = tl.load(B + rn, mask=rn < N)
    acc += b[None, :]
    # GELU 直接算掉，不写回显存
    out = acc * 0.5 * (1.0 + tl.math.erf(acc * 0.70710678))
    tl.store(Out + rm[:, None] * N + rn[None, :], out,
             mask=(rm[:, None] < M) & (rn[None, :] < N))
```

patch eager 的方式（从糙到好）：

```python
# 粗暴：monkeypatch（注意 eval 复现性和 export 会挂）
import torch.nn.functional as F
F.gelu = lambda x: ...   # 危险，全局污染

# 推荐：注册 custom op，再让 torch.compile / 图捕获看得见
torch.library.define("mylib::linear_gelu", "(Tensor x, Tensor w, Tensor b) -> Tensor")
@torch.library.impl("mylib::linear_gelu", "CUDA")
def _impl(x, w, b):
    out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(x.shape[0], 64), triton.cdiv(w.shape[0], 64))
    linear_gelu_bias[grid](x, w, b, out, x.shape[0], w.shape[0], x.shape[1],
                           *x.stride(), *w.stride(), BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)
    return out

class FusedGELU(torch.nn.Module):
    used = ... 
    def forward(self, x, w, b):
        return torch.ops.mylib.linear_gelu(x, w, b)
```

**收益来源**：

- **减少 kernel 数量** → 附带的 launch 开销下降（和 Graph 收益部分重叠）。
- **减少显存流量**：原本中间张量要 write→read 全局显存，融合后留在片上。算术强度提升：

$$ I = \frac{\text{FLOPs}}{\text{Bytes moved}} $$

- 对 **memory-bound / elementwise 链 / norm+激活+dropout** 这类，收益巨大，因为瓶颈就是搬数据。
- 显存峰值下降 → 能开更大 batch。

**代价**：

- 自己保证**正确性**（边界、mask、dtype、layout、stride）。
- 需要 **autotune**（注意：autotune 必须在 graph capture 前完成）。
- Triton 覆盖度不如 cutlass/cuBLAS，某些形状未必打得过厂商库。
- 动态 shape 需要 masking 或多组 specialization，`torch.compile` 动态 shape 也是靠重编译解决的。

---

## 3. 核心对比

| 维度 | CUDA Graph | Triton fused + patch eager |
|---|---|---|
| 优化层 | 调度 / 提交（launch overhead） | 算子 / 访存（kernel rewrite） |
| kernel 数量 | **不变** | **减少** |
| 显存流量 | **不变** | **显著减少** |
| 峰值显存 | 略升（static pool） | 通常下降（合并中间张量） |
| 主要收益场景 | launch-bound、大量小 kernel、固定 shape 推理 | memory-bound、算子链、elementwise/reduction |
| 形状要求 | **必须静态** | 可 static，可 mask 支持动态 |
| 正确性风险 | 低（不改数值） | **高**（要自己写对） |
| 与 autograd / dynamo | 需注意捕获，`.item()` 等会炸 | 需配 custom op 才有 autograd/compile 支持 |
| 上手成本 | 低（几行包一层） | 高（写 kernel + autotune + 注册） |
| 能否叠加 | ✅ **可以，且推荐** | ✅ |

---

## 4. 什么时候用哪个

**优先 CUDA Graph**（改造成本低、几乎零数值风险）：

- 推理服务、decode 阶段，kernel 又小又多。
- 模型固定、shape 固定（batch/seq 静态）。
- `nsys` 里看到大量 **launch gap / GPU idle 等 CPU**。
- 典型：`torch.compile(mode="reduce-overhead")` → 底层就是 cudagraphs。

**优先 Triton fusion**：

- `ncu` / `nsys` 显示 **DRAM throughput 打满但算力没打满**（memory-bound）。
- eager 里有明显的**中间张量往返**（LayerNorm→Linear→GELU→… 每个都落一次显存）。
- 想要**降低显存峰值**塞更大 batch。
- 想利用 **TMA / warp specialization / 新精度**等 cuBLAS/eager 给不了的东西。
- 注意：`torch.compile` 的 inductor 已经在做这件事，**先看 compile 生成的代码够不够**，不够再手写。

**两个都别急着上的情况**：大 matmul、compute-bound、shape 频繁变化——收益有限。

---

## 5. 正确姿势：两者是叠加的，不是二选一

真实的高性能推理栈通常是：

```
eager op 链
  → 手写/inductor 生成 Triton fused kernel（减少 kernel 数 + 显存流量）
  → 把整个 forward 用 CUDA Graph 包起来（抹平残余 launch 开销）
```

```python
# 融合 kernel 负责搬得少，Graph 负责发得快
with torch.cuda.graph(g):
    static_out = model(static_in)   # 里面已经是融合后的少数几个 kernel
```

**组合时的坑**：

1. **autotune / 编译必须在 capture 之前完成**，否则捕获期会触发 host 同步。Triton 的 `@triton.autotune` 关键字要提前 warmup。
2. Triton kernel 的**首次 launch 有 JIT 编译**，绝不能落在 capture 窗口里。
3. patch eager 若用 `monkeypatch`，同时配 Graph 时容易出"地址漂移 / 参数变化"，建议走 **custom op + static buffer**。
4. 多流、通信（NCCL）进 graph 要小心拓扑固定。
5. 用 `torch.cuda.make_graphed_callables` 或 `torch.compile(mode="reduce-overhead")` 可以自动把两者合起来，但**它不会替你写融合 kernel**，只是把现有 kernel 序列图化。

---

## 6. 决策流程（实操版）

1. `nsys profile` 看时间线：
 - 大量 **短 kernel + 空隙** → launch-bound → **先上 CUDA Graph**。
2. `ncu` 看 memory-bound 的 kernel：
 - `dram__throughput` 高、`sm__throughput` 低 → **考虑 Triton 融合**。
3. 先试 `torch.compile`：
 - `mode="default"` 看它 fuse 得如何；
 - `mode="reduce-overhead"` 自动上 cudagraph。
4. 只有在 compile 结果仍不达标（或需要高级特性）时，才**手写 Triton 并注册 custom op**，最后再用 Graph 收尾。

> 一句话记忆：**CUDA Graph 不碰算子，Triton fusion 重写算子；前者治"发车慢"，后者治"跑得远"。性能栈里它俩是队友，不是对手。**

---
## 🧑‍💻 User

说实话，你去考虑什么cuda graph，真不如先看看单一kernel你推到极限了没有
什么cuda graph啥的不是影响最大的
launch开销影响真不大
cuda stream啥的在那里摆着
请先盯着你到底怎么做某个操作的
比如 flash attention最大的invention是online softmax
因为这个操作减少了很多冗余的memory rw
但这跟cuda graph不graph没有屁点关系
你用pytorch就是写不出online softmax
你graph搞成一个他还是offline

所以事情是这样的，你应该先考虑你实现一个算法（一个kernel）的时候
他的读写与运算性质
cache locality怎么样，thread之间有没有影响（atomic write很慢），读写跟运算的比例 交替
这些比cuda graph/launch overhead重要得多的多
我的观点是 你单个kernel都没推到极限的时候没有资格谈launch overhead
而最后做graph merge 说实在话你全都是optimized kernel的时候，一个torch.compile就结束战斗了
torch.compile在一些没有必要重排运算顺序和rw顺序的kernel上，有绝佳的速度
以及在把整个模型变成一个整体减少overhead这点上 也有绝佳的成效，例如RoPE，torch.compile跟你triton折腾到死的速度差不多
因为是本来就没有重排的必要，就是vector operation堆满
又或者我最后的模型上个torch.compile()，launch overhead又能啃出几个百分点的速度
但是我靠kernel optimization啃出了几倍的速度

今天这个问题的重点是
先把个别的kernel搞明白了 真的有需要再考虑launch overhead 最后你会发现除非你跟meta/x/google他们一样要一次调动几千张 不然没差
另外他们需要mega kernel更多是communication，因为超多卡的communication是很痛苦的，需要很多神奇的workaround

---
## 🤖 Assistant

同意，而且你点出的东西比"先讲 CUDA Graph"这个顺序正确得多。我把你的观点补几个精确的地方，顺便把边界条件讲清楚，免得走向另一个极端。

## 优化顺序本来就该是：算法 → 访存/布局 → 指令级 → 编排

你说的这套其实就是 roofline 驱动的标准顺序。launch overhead 属于**最后一步的编排**，它前面还压着两个数量级更大的东西：

1. **算法的访存复杂度**（online softmax 就是这一类）
2. **单 kernel 的 cache locality / 读写比 / 依赖结构**

前两个不动，编排怎么调都是给一个 memory-bound 的烂算法换了个更快的发车方式而已。

## online softmax 的本质是复杂度重写，不是"融合"

这个值得量化一下，因为这才是 flash attention 真正的发明。

朴素 attention（S=seq len, d=head dim）：

- 写 $S\times S$ 的 score 矩阵 → 再读它做 max → 再读它做 exp/sum → 再读它做归一化 → 再读它做 $PV$。

HBM 流量大致是 $O(S^2)$ 级别（好几趟 $S^2$ 的读写）。

Flash attention 用 online rescaling（running max $m$、running sum $l$），把归一化推迟，永远不需要完整一行：

$$
O_i = \frac{\sum_j e^{s_{ij}-m_i} V_j}{\sum_j e^{s_{ij}-m_i}}
$$

这个和是在 tile 内递推维护的，所以 score 矩阵**从不落 HBM**，流量降到 $O(S\cdot d)$——对 $S$ 是线性的，不是平方。

这才是关键：**它把问题从 $O(S^2)$ 访存变成了 $O(Sd)$**，还顺便把 softmax 的三趟 pass 合成一趟。这跟 launch 一个还是十个 kernel 没有任何关系。**图捕获不会让它从 offline 变 online**——你说得对，offline 就是 offline，你把 DAG 回放一百遍它还是先算完整 S 矩阵。算法层面的信息（可以递推、可以延迟归一化）只有写 kernel 的人知道。

backward 那边还叠了一层"重算换显存"（不存 attention matrix，重算），同样是算法决策，graph 无法表达。

## torch.compile 的边界恰好证明了你的观点

compile 干得好的是：**点态/归约链，运算可以结合律交换律重排、读写顺序无所谓**的情况——RoPE、norm+激活、residual、dropout 这种"vector op 堆满"。这种你手写 triton 折腾半天，也就和 inductor 生成的差不多，因为它本质就是 bandwidth-bound 的向量流,谁写都撞同一个 roofline。

compile 干不了的是：**算法重排**。它不会替你发明 online softmax，也不会替你做一个非标准的 scan、一个自定义 layout 的 tiling。inductor 能 pattern-match 出一些 attention，但那是它认得这个 pattern，不是它推导出来的。

所以结论应该是：compile = 单 kernel 优化完之后的**收尾和编排**，吃的是你已经优化好的 kernel 上那几个百分点的 launch/dispatch 开销（`mode="reduce-overhead"` 里就是 cudagraphs），加上把 Python dispatch 开销干掉。它是**最后一公里**，不是主菜。

## launch overhead 什么时候才真的重要

它是**相对 kernel 时长**的。单次 launch ~$3\sim10\,\mu s$：

- kernel 跑 $200\,\mu s$ → 摊掉 $<5\%$，忽略。
- kernel 跑 $5\,\mu s$ → 开销 $100\%$，直接主宰。

所以这里有个**自我强化**的结论，正好支持你：**你把 kernel 融合好、每个 kernel 变长之后，launch overhead 自己就不重要了。**不是"launch 不重要所以不用管"，而是"好的 kernel 工作本身让 launch 变得不重要"。

真正让 graph 有意义的窗口很窄：**你正确优化完，但仍然剩一大串不可避免的短小 kernel**——RL rollout 循环、小静态 shape 的 serving、model-parallel 里一堆碎 collective。这时候 graph 是低风险的小便宜。仅此而已。在单卡上它不是主战场。

## mega-kernel 的真实动机：communication，你说到点子上了

单卡上把整个模型塞进一个 persistent kernel，收益是尾部效应和 grid sync 那点东西，复杂度爆炸，ROI 很低。

几千卡上不一样，那里根本不是 launch overhead 的问题，而是：

- **通信延迟/带宽主宰**，你需要把 comm 藏进 compute 里（overlap、in-kernel allreduce、NVSHMEM、grid-constant + grid sync 的 persistent kernel）。
- **kernel 边界 = sync 点 = 通信必须收敛的地方**，mega-kernel 的意义是减少这些必须全局同步的边界，把 comm 和 compute 融进同一段流水。
- 卡多了以后还有 one-shot / two-shot allreduce、topology-aware 的算法选择。

所以 mega-kernel 是**分布式 scaled 之后的产物**，不是单卡性能手段。你说的"workaround"很准——它们很多时候是在对抗通信的物理限制，不是在省 launch。

## 单 kernel 到底该盯什么（这才是重点）

<details>
<summary>展开：一个 kernel 推到极限时的判断清单</summary>

- **Roofline 定位**：先算你这 kernel 的算术强度 $I=\text{FLOPs}/\text{Bytes}$，跟机器 $\text{Peak FLOPs}/\text{BW}$ 比，确定它是 compute-bound 还是 memory-bound。不知道 roofline 就没法判断"到极限了没有"，容易早停或瞎调。
- **访存**：合并访存、向量化（128-bit load/store）、shared memory bank conflict、swizzle/padding。
- **数据复用**：tile 尺寸、L2 驻留、blocking。复用不够就是白搬。
- **同步/写冲突**：atomic write 慢，能用 warp shuffle / 分层归约 / 两趟替代就替代；不要横向跨线程写。
- **读写-计算交替结构**：依赖链是 latency-bound 还是能流水化？能不能 double-buffer / cp.async / TMA + software pipeline 把 load 藏到 compute 后面。
- **寄存器 vs occupancy**：寄存器压力换 occupancy，找 knee point，不是越少越好。
- **精度/类型**：bf16 累加、tf32、accumulate in fp32 的取舍。
- **网格**：grid 大小、persistent vs 一般、tail effect。

</details>

一个务实的判据：**先把它干到 roofline 的百分之七八十**。到这一步之前，谈 launch、谈 graph、谈 stream 编排，都是本末倒置——因为你连"这趟车本身能不能跑满"都还没确定，去优化"多久发一趟车"没有意义。stream 就在那里摆着，它不会因为你不看它就消失。

## 所以你的结论我基本全盘接受，只加两个限定

1. **不是"launch overhead 永远不重要"，而是"kernel 优化完之前它不重要，优化完之后它自动变小"**。你如果确实落在一大堆不可避免的短 kernel 上，graph 还是能啃几个百分点，且几乎零风险——但那是收尾，不是主线。
2. **什么时候值得上升到 mega-kernel**：看**卡数**。单卡就 torch.compile 收个尾；上了通信密集的规模，才轮到和 comm 纠缠的融合/persistent kernel。

一句话：**先把一个 kernel 的读写性质和算法结构吃透（能重排就重排，能延迟就延迟，能留片上就留片上），真跑不动了再看 launch，最后你会发现 compile + graph 只是把已经很快的东西再抛光几个百分点。** 我自己上一轮把 graph 讲太靠前，是误导。
