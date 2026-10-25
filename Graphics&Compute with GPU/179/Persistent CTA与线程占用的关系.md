# Persistent CTA与线程占用的关系

## 🧑‍💻 User

为什么有时候写kernel会鼓励用persistent cta，那vector操作不是很费劲，本身我可以多个CTA处理一个vector，现在同时驻留的CTA数目少了，就感觉会影响性能
本身很多vector的量化之类的，占用的寄存器就少，我可以把整个SM的占用拉满
现在一个SM就128个线程，想想就挺抽象的

---
## 🧑‍💻 User

为什么有时候会鼓励用persistent cta，那vector操作不是很费劲，本身我可以多个CTA处理一个vector，现在同时驻留的CTA数目少了，就感觉会影响性能
本身很多vector的量化之类的，占用的寄存器就少，我可以把整个SM的占用拉满
现在一个SM就128个线程，想想就挺抽象的

---
## 😀 默认助手 (7)

# 先纠正一个前提：persistent CTA ≠ 1 CTA/SM

这是最大的误区。Persistent kernel 的"persistent"指的是**不靠反复 launch 新 block 来喂满 GPU，而是让已经驻留的 block 用循环把剩余工作吃完**。它改变的只是"工作怎么分",**完全没改变你能塞进 SM 的线程数上限**。

常驻线程数的约束是:

$$
\text{resident threads/SM} = \min\Big(\underbrace{\frac{65536}{R}}_{\text{寄存器}}, \frac{\text{SMEM}_{\max}}{S_{\text{block}}}, 2048\Big)
$$

其中 $R$ 是每线程寄存器数,$S_{\text{block}}$ 是每 block 的 smem 用量。这个上限对 persistent 和普通 launch **一模一样**。

所以正常的 persistent 配方是:

$$
\text{gridDim} = \text{numSMs} \times \frac{\text{maxActiveBlocksPerSM}}{1}
$$

`maxActiveBlocksPerSM` 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` 算出来。你的量化 kernel 如果用 30 个寄存器、256 线程/block,那 `maxActiveBlocksPerSM = 8`,就 launch `132 × 8 = 1056` 个 block——**2048 线程/SM,占用率拉满,和你现在的做法完全一样**。

"一个 SM 128 线程"是有人用 `gridDim = numSMs`、`blockDim = 128` 写出来的**错误配置**,不是 persistent 的固有属性。如果谁这么写还没出问题,那纯属工作量太小。

---

# Persistent 被鼓励的真正原因

在推荐它的场景里(LLM 的 decode attention、MoE、fused epilogue、量化/反量化流水线、CUTLASS 的 persistent GEMM),核心收益是这几条:

| 收益 | 机制 | 对向量/量化 kernel 有用吗 |
|---|---|---|
| 消掉 wave quantization / tail | 最后一次不满的 wave 的浪费被摊到循环里 | ✅ 小 kernel 明显 |
| 消掉重复 launch 开销 | 省掉每次 ~3–10 µs 的 launch / drain / refill | ✅ decode 这种 1 µs 级 kernel 极其关键 |
| 跨迭代软件流水 | 循环里可以 prefetch 下一块,和当前块计算重叠 | ✅✅ 对 memory-bound 是刚需 |
| 跨迭代数据复用 | scale / zero-point / 权重留在寄存器或 smem | ✅✅ 量化 kernel 大赢点 |
| 跨 CTA 同步 | grid.sync() / 原子计数,不用重启 kernel 就能做两阶段 | ✅ 融合归约、fused quant |
| 动态负载均衡 | 原子 work counter 抢任务,处理 ragged 长度 | ✅✅ 变长 batch |
| 控制 L2 驻留 | `cudaAccessPolicyWindow` 配合长期驻留的 CTA | 视情况 |

关键的一条:**普通 launch 里,一个 block 干完自己那块就死了,你没法把"当前块的 store"和"下一块的 load"重叠**。persistent 循环天然给你一个跨 tile 的流水线结构,这是 memory-bound kernel 提速的主要手段。

<details>
<summary><b>展开:tail effect / wave quantization 到底浪费多少</b></summary>

设总共 $W$ 个 wave(一个 wave = 所有 SM 同时满载一批 block):

- 普通 launch 的近似损失 $\approx \dfrac{1}{W}\cdot\dfrac{\text{空闲 SM 数}}{\text{SM 总数}}$
- 如果 kernel 只有 1.5 个 wave,损失可以到 **33%**
- 如果有 1000 个 wave,损失只有 **0.05%**,此时 persistent 的这条收益基本为零

所以"要不要 persistent"高度依赖 kernel 的**持续时长和 wave 数**。一个大向量、几千个 wave 的 elementwise kernel,普通 launch 就够了。
</details>

<details>
<summary><b>展开:为什么 SM90/SM100 上常看到"1 CTA/SM + 512~1024 线程"</b></summary>

那是 **GEMM / attention 这类 compute-bound + 张量核 + TMA** 的配方,不是向量 kernel 的配方:

- 需要大块 smem 做 TMA 双缓冲 → 一个 SM 的 smem 只够放一个 CTA
- 需要 warp specialization(producer/consumer warps)→ 线程要够多(512–1024)才能分出专职 warp
- 张量核利用率靠指令级流水,不靠 occupancy

那套东西的目标是让 tensor core 不停转,**和 occupancy 完全不是同一个优化目标**。你把它的 CTA 配置抄到量化 kernel 上,当然会崩。
</details>

---

# 你的向量/量化场景:担心只对了一半

你的直觉"128 线程/SM 肯定不够"是对的,但结论"所以 persistent 有害"是错的。真正的物理量是**在途字节数 (bytes in flight)**:

$$
\text{BW}_{\text{peak}} \approx \frac{\text{bytes in flight}}{\text{memory latency}}
\quad\Longrightarrow\quad
\text{bytes in flight} \ge \text{BW} \times L
$$

以 H100 SXM 为例:$\text{BW} \approx 3.35\ \text{TB/s}$,$L \approx 600\ \text{ns}$:

$$
\text{bytes in flight} \ge 3.35\times10^{12} \times 600\times10^{-9} \approx 2.0\ \text{MB}
$$

除以 132 个 SM,$\approx 15\ \text{KB/SM}$。

| 配置 | 在途字节 | 够不够 |
|---|---|---|
| 128 线程 × 1 个 `float4` | $128 \times 16 = 2\ \text{KB}$ | ❌ 差 ~7.5× |
| 2048 线程 × 1 个 `float4` | $2048 \times 16 = 32\ \text{KB}$ | ✅ 富余 |
| 128 线程 × 8 个独立 `float4` | $128 \times 8 \times 16 = 16\ \text{KB}$ | ✅ 刚好 |

**两个结论**:

1. 如果你真的只有 128 线程/SM,那确实要每线程至少 8 个**独立**的 in-flight load 才够——但这说明配置错了,**直接多开 CTA 就行**,没有任何理由限制成 128。
2. 反过来,一旦占用率到 2048 线程/SM(你现在的做法),**继续加线程对 bandwidth 毫无帮助**——memory-bound kernel 在 50% occupancy 下往往就能跑到 95% 峰值带宽。这时"占用率拉满"本身不再是收益,persistent 省下的 launch/tail **才是纯赚**。

另外量化 kernel 有个 persistent 特别占优的点:**per-block scale / zero-point 的复用**。如果让 CTA 用 block-contiguous 映射循环处理连续的多组 block,scale 可以加载一次进寄存器,并且能做"算当前块时预取下一块的 scale",而 strided 映射下每换一组就要重新取。

<details>
<summary><b>展开:标准 persistent 向量 kernel 该长什么样</b></summary>

```cpp
// 量化/反量化:memory-bound,靠 ILP + 软件流水吃满带宽
__global__ void __launch_bounds__(256, 8)   // 8 blocks/SM -> 2048 threads/SM
quant_persistent(const float4* __restrict__ x,
                 int4*        __restrict__ y,
                 const float* __restrict__ scale,
                 int n_vec)
{
    const int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 展开:一次发出 UNROLL 个独立 load,制造 ILP
    #pragma unroll 1
    for (; i + (UNROLL-1)*stride < n_vec; i += UNROLL*stride) {
        float4 v[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = __ldg(&x[i + u*stride]);  // 并行发出
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            float s = __ldg(&scale[(i + u*stride) >> 8]);  // 复用/邻接 cache line
            y[i + u*stride] = pack_int4(v[u] * s);
        }
    }
    // tail:处理剩余
    for (; i < n_vec; i += stride) { /* ... */ }
}
```

要点:
- `__launch_bounds__(256, 8)` 把寄存器压到不超 65536/(256×8)=32,守住 2048 线程/SM
- `UNROLL` 制造 ILP,这对低占用率是**唯一**的补救手段
- 循环变量的地址计算会吃寄存器——这是 persistent 写法的真实代价,必须用 `__launch_bounds__` 和 `#pragma unroll 1` 控制
</details>

---

# 什么时候**别**用 persistent

| 场景 | 建议 |
|---|---|
| 大数组 elementwise(几千个 wave) | 普通 launch,launch 开销 ≪ 计算时间,收益约等于 0 |
| 想要最大灵活度 / 和别的 kernel 并发 | 普通 launch;persistent 会长期占住 SM,饿死其他 kernel |
| MIG / 多租户 | 慎用,persistent 会霸占整个分区 |
| 需要 kernel 级抢占 / 调试 | persistent 更难调试,且用 grid sync 时死锁风险高 |
| 寄存器本来就很紧 | persistent 的循环开销 + 流水 buffer 可能反而 spill,占用率掉一半 |

**优先级顺序**(从便宜到激进):

1. 普通 launch + grid-stride loop(向量 kernel 的默认答案,就是 persistent 的弱化版)
2. **CUDA Graphs** —— 干掉连续 launch 的开销,不动 kernel
3. **Programmatic Dependent Launch** —— 让下一个 kernel 的 prologue 和当前 kernel 重叠
4. **Persistent kernel** —— 需要跨迭代流水、跨 CTA 同步、动态负载均衡时才上

---

# 总结

- **Persistent 不是"少开 CTA",是"少 launch、block 内循环"**。grid 大小仍然按占用率上限开,你的 2048 线程/SM 一点不少。
- 它对**向量/量化 kernel 恰恰有利**:memory-bound 不需要高占用率的上限,但极需要跨迭代软件流水、scale 复用、以及省掉频繁 launch 和 tail。
- "128 线程/SM"是配置错误,不是性质。要么提到 2048 线程,要么给每线程 8+ 个独立 load(ILP),二者都能补上在途字节。
- 只有在**单个 kernel 已经跑很久(几千 wave)**、且不需要跨迭代流水时,persistent 才是纯粹的负担。

你现在这套"寄存器少、占用率拉满、多 CTA 处理一个 vector"的做法,配上一个 **unroll 过的 grid-stride / persistent 循环**,通常是在它之上再白捡 launch 和 tail 的收益,而不是互相冲突。
