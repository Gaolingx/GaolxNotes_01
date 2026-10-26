# GPU算子小尺寸瓶颈排序

## 🧑‍💻 User

GPU计算中在前向传播时有大量小算子导致的瓶颈在哪

---
## 😀 默认助手 (7)

## 一句话结论

这个瓶颈的本质是 **"每个算子的固定开销 × 算子数量"**,而不是 GPU 算力不够。绝大多数情况下,你处在 **CPU 发射受限(launch-bound)** 或 **访存受限(memory-bound)** 的区间,GPU 的计算单元实际上有大量时间在空转。

---

## 三层瓶颈模型

| 层次 | 瓶颈来源 | 典型量级 | 在 profiler 里的表现 |
|---|---|---|---|
| **主机/CPU 侧** | `cudaLaunchKernel` + 框架 dispatch + Python/autograd | CUDA ~2–6 μs/算子;PyTorch eager ~10–30 μs/算子 | timeline 上 kernel 之间有**空隙**,GPU 空闲 |
| **GPU 调度侧** | grid 太小填不满 SM、波次量化、尾效应 | 单 kernel < 5 μs | SM 占用率低,但 kernel 时间"看起来很干净" |
| **GPU 执行侧** | 算术强度极低 + 中间张量反复物化 | 逐元素算子 $I\approx 0.125$ FLOP/Byte | HBM/L2 带宽打满,SM 计算单元空闲 |
| **依赖/同步** | 隐式同步、D2H 拷贝、stream 串行、allocator 事件 | 一次全设备 sync 可达 ms 级 | 整条流水线 stall |

下面逐层拆开。

---

## 一、CPU 端发射开销(最常见的主因)

一个 kernel 从 Python 到 GPU 上跑起来,要经过:Python 解释器 → autograd → dispatcher → shape/dtype check → CUDA API → 驱动 → GPU 前端。这一整套**与张量大小无关**。

设第 $i$ 个算子的 CPU 发射耗时 $\tau_c$,GPU 实际执行耗时 $\tau_g$。异步发射可以流水,所以:

$$
T_{\text{total}} \approx \sum_{i=1}^{n} \max(\tau_c,\ \tau_g)
$$

有意义的指标是比值 $r = \tau_c / \tau_g$:

- $r \ll 1$:GPU 打满,发射开销可忽略(大 GEMM、大 attention 就是这种)。
- $r > 1$:**发射受限**,GPU 饿死,$T \approx n\tau_c$,利用率约 $\tau_g/\tau_c$。

以 $\tau_c \approx 10\,\mu s$ 估算,**任何执行时间低于约 10 μs 的算子都在拖后腿**。1000 个小算子 ⇒ 光发射就 10 ms,等价于每秒只能跑 100 个 iteration。

> 关键点:发射是**串行的、受限于单核 CPU 性能**的,增加 GPU 数量完全不解决这个问题。

---

## 二、小 kernel 在 GPU 上本身效率就低

即使不考虑 CPU 发射,小算子本身在 GPU 上也不好跑:

1. **并行度不足**:grid 里的 CTA 数小于 SM 数时,部分 SM 直接闲置;层归一化、softmax 这类按行归约算子尤其明显。
2. **波次量化(wave quantization)**:CTA 数不是 SM 数的整数倍时,最后一"波"只有少量 SM 在工作,尾效应按比例吃掉性能。
3. **延迟隐藏不够**:每个 SM 上活跃 warp 太少,无法用 warp 切换掩盖 HBM 延迟(~400–800 cycle)和指令延迟。
4. **固定启动成本**:block 调度、warp 启动、指令 cache 冷启动、常量/参数加载——这些在 2 μs 的 kernel 里占比极高。

---

## 三、访存带宽与中间张量物化(小算子场景的第二个主因)

用 Roofline 的**脊点(ridge point)**判断:

$$
I^{*}=\frac{P_{\text{peak}}}{BW}
$$

以 A100-40GB 为例,$P_{\text{peak}}\approx 19.5$ TFLOP/s(FP32),$BW\approx 1.55$ TB/s:

$$
I^{*}\approx 12.6\ \text{FLOP/Byte}
$$

而一个逐元素算子(读 4B、写 4B、算 1 FLOP):

$$
I=\frac{1}{8}=0.125\ \text{FLOP/Byte}
$$

**比脊点低约两个数量级**,完全落在访存受限斜坡上。

再看融合带来的流量收益。一条 $n$ 个逐元素算子组成的链,每个算子读输入写输出,总量:

$$
\text{Traffic}_{\text{unfused}} \approx 2nNb,\qquad
\text{Traffic}_{\text{fused}} \approx 2Nb
$$

理想加速比就是 $\approx n$。这就是为什么 TorchInductor / TensorRT / XLA 的逐元素融合能轻松拿到数倍加速——**它们省的不是 FLOPs,是显存往返**。

<details>
<summary><b>为什么小张量反而"看起来"没这么惨(展开)</b></summary>

中间张量如果足够小,能驻留在 L2 里(A100 40MB、H100 50MB),那么中间读写的代价是 L2 带宽而不是 HBM 带宽,能缓解一部分访存墙。但一旦中间张量尺寸超过 L2/算子数,就会持续穿透到 HBM,退化成上面的分析。这也是为什么 batch 很小时"小算子问题"往往比大 batch 时更突出——不是算得少了,而是单位数据的摊销变差了。

</details>

---

## 四、同步、依赖与框架层开销

- **隐式全设备同步**:`.item()`、`print(tensor)`、`.cpu()`、`non_blocking=False` 的 D2H、每步 `loss.item()` 做日志——任意一个都会让流水线彻底打断。
- **allocator**:`cudaMalloc` 是同步且昂贵的,caching allocator 的 block 复用/`cudaFree` 事件在某些形状抖动场景下会引入同步。
- **stream 串行**:默认 stream 上的所有小算子严格顺序执行,无法重叠,即使它们之间没有数据依赖。
- **Python / 动态形状**:形状检查、kernel 选择(heuristics 查表)、graph capture 失败等,都会加到 $\tau_c$ 上。

---

## 五、怎么诊断

<details>
<summary><b>诊断清单(展开)</b></summary>

**Nsight Systems(先看这个)**
- kernel 之间是否存在大段空隙 → 发射受限
- 每个 iteration 的 kernel 数量级(几百还是几万)
- 平均单 kernel 时长:如果普遍 < 10 μs,高度可疑
- GPU 忙碌时间 / wall-clock 时间之比

**Nsight Compute(再看这个)**
- Memory Throughput % vs SM Throughput %:前者高后者低 = 访存受限
- Achieved Occupancy、grid size vs SM 数

**torch.profiler**
- `cudaLaunchKernel` 等 CUDA API 的 CPU 时间总和 vs `kernel` 的 GPU 时间总和
- 如果前者 ≥ 后者,基本坐实 launch-bound

**注意**:`nvidia-smi` 的 "GPU-Util" 是"有 kernel 在跑的时间占比",不是算力/带宽占用。它显示 100% 完全可能意味着 SM 在空转等显存。

</details>

---

## 六、解决方向(按性价比排序)

| 手段 | 解决的瓶颈 | 说明 |
|---|---|---|
| **算子融合** | 访存 + 发射 | `torch.compile`(Inductor)、XLA、TensorRT、nvFuser、Triton 手写。逐元素融合、归约融合是收益最大的 |
| **CUDA Graphs** | 发射 | 把整段 launch 序列 capture 成图回放,每算子摊销成本降到 ~1 μs 以下。`torch.cuda.CUDAGraph`、`torch.compile(mode="reduce-overhead")`、cudagraph trees |
| **增大每条 kernel 的工作量** | 三者都治 | 增大 batch/张量;micro-batch 累积;persistent kernel / megakernel |
| **消除同步点** | 依赖 | 去掉 `.item()`、用 pinned memory + `non_blocking=True`、日志异步化 |
| **合并小 GEMM** | 调度 + 发射 | 用 `bmm`/grouped GEMM 替代循环里的小 `matmul` |
| **多流并发** | 依赖 | 仅对**无依赖**的算子有效,融合通常更优 |
| **更宽的并行策略** | 调度 | 归约算子做 split-K / 两阶段归约,保证 CTA 数 ≥ SM 数 |

**反模式警告**:融合不是免费的。融合会抬高寄存器压力、可能造成 register spilling、降低 occupancy;动态形状会阻碍 CUDA Graph capture;把 reduction 和 elementwise 硬融在一起有时反而更慢。**先量化再动手。**

---

## 七、什么时候这根本不是问题

判断标准就是 $r=\tau_c/\tau_g$:

- 算子本身够大(大 GEMM、FlashAttention、大卷积),$\tau_g \gg \tau_c$,发射开销占比 $<\!1\%$,可以无视。
- 处于 compute-bound 区间($I > I^{*}$),访存不是约束。
- batch 足够大,每个 kernel 的执行时间被摊销到几十微秒以上。

所以"大量小算子"的瓶颈,归根结底是一句话:**要减少的是 kernel 的数量和显存往返次数,而不是让每个 kernel 跑得更快。**
