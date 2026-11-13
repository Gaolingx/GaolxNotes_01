# Blackwell GPU上TMA与cp.async选择指南

## 🧑‍💻 User

blackwell gpu上在已经有tma的情况下要不要还去用cp.async？

---
## 😀 默认助手 (7)

# 简短回答

**默认不用 cp.async，TMA 应当是 Blackwell 上的主路径。** 但 `cp.async` 在 sm_100/sm_120 上并没有被移除，PTX 里照常可用，而且确实有一小组场景它仍然更合适，甚至更彻底地说，有一组场景 TMA 根本没法替代。

判断标准其实很简单：如果这个拷贝可以描述成一个矩形块（box），并且有足够的数据量来抵消 mbarrier 流水线的开销，就用 TMA；如果它不规则、长度很小、按线程逐个判断条件，或者布局 TMA 无法描述，那就退回 `cp.async`。

---

# 为什么 Blackwell 上 TMA 应当是默认选择

<details>
<summary><b>展开：TMA 相对 cp.async 在 sm_100 上的具体优势</b></summary>

| 维度 | `cp.async` (LDGSTS) | TMA (`cp.async.bulk.tensor`) |
|---|---|---|
| 指令开销 | 每线程 16B，一个 $128\times64$ fp16 的 tile（16 KB）要 ~1024 条 per-thread 指令，或 32 条 warp 级指令 | **1 条**指令搬整个 tile |
| 发射带宽 | 占用 LSU issue slot，与 `tcgen05.mma`、TMEM 操作争抢 issue | 只占 1 次发射，之后靠 TMA 单元自己跑 |
| 寄存器 | 每个 stage 要保存一堆地址寄存器 | 地址由 descriptor 中的逻辑决定，线程几乎不用寄存器 |
| 跨 CTA 广播 | **不支持**（只有 `cp.async.bulk` 有 `.multicast::cluster`） | 支持 cluster 内 multicast |
| 2-SM MMA（cta_group::2） | 需要经 `st.async` 或 DSMEM 绕一圈 | 直接 multicast 进两个 CTA 的 SMEM |
| 非 128B 对齐的行宽 | warp 内只能合并成部分 sector，DRAM 效率掉 | 整个 box 由硬件重排，请求尺寸最优 |
| 存储方向 | **只有 global→shared，没有 store 版本** | 有 `cp.async.bulk.global.shared::cta` 的异步存回 |

在 Blackwell 上，因为 `tcgen05` 把“发射带宽”变成了比以往更稀缺的资源（MMA 和 TMEM 指令都在抢 issue），上表中“指令开销”这一行会被放大。所以从 Ampere 迁移过来、cp.async 用得很顺的 kernel，到 sm_100 上通常还是应该重写成 TMA。

</details>

---

# `cp.async` 仍然胜出的场景

**1. TMA 描述不了的布局（硬性阻塞，最常见）**

`cuTensorMapEncodeTiled` 有硬性约束：global 地址 16B 对齐，每个 stride 是 16B 的整数倍，box 每个维度 ≤ 256 元素。只要出现以下情况，TMA 就用不了：

- 行 stride 不是 16B 的倍数（比如 fp8 且行长是奇数，或带 padding 的 ragged tensor）
- 需要把 interlaced / 自定义 XOR swizzle 喂给非标准 MMA 布局
- 张量维度是运行时才知道的非 2 的幂形状，且不能接受反复用 `tensormap.replace` 重建 descriptor

这时 `cp.async` + 手算 swizzle 地址就是唯一的异步路径。

**2. 逐线程谓词 / 小到无法分摊开销的拷贝**

- 每个线程拿到的数据量是“不确定有几块”，例如 MoE 的 token 路由、embedding gather、稀疏索引。
- 单次负载 < ~256B 的极细粒度 stage：每次都要做 `mbarrier.arrive.expect_tx` + 等待，mbarrier 的到达计数与相位管理成本会超过收益，而 `cp.async.commit_group` / `wait_group N` 的开销几乎为零。
- 注意 `cp.async` 的 `src-size` 操作数可以做到“越界补零”，这对 ragged 尾部很有用，不用额外 fill。

**3. 需要单代码路径跨 sm_80 / sm_90 / sm_100**

同一份 kernel 想同时在 A100/H100/B200 上跑，`cp.async` 是唯一交集。工程上通常的做法是：`cp.async` 走 fallback 分支，sm_90+ 走 TMA 分支，用 `__CUDA_ARCH__` 分开。

**4. 你想同时用很少的 SMEM 做很多 stage 的小 tile 流水**

`cp.async` 用 commit-group 语义，不需要在 SMEM 里为每个 stage 放一个 mbarrier（每个 8B，但会占 barrier 槽位和初始化成本）。stage 数很大（比如 16+）且 tile 很小时，TMA 的 barrier 管理反而更啰嗦。

---

# 几种“既不是 TMA 也不是 cp.async”的更优选择

- **一维连续拷贝**：`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`（Hopper 起的 1D bulk copy）。它不需要 tensormap，比 tensor TMA 简单，又比 cp.async 省指令。很多人会漏掉这条。
- **数据直接进寄存器**：两个都不用。`cp.async` 只能进 SMEM，要进寄存器还得再 `ld.shared` 一次。此时应该用带 `.L2::128B/256B` 预取的 `ld.global.nc`。
- **写回 global**：`cp.async` 没有 store 版本。要么 `cp.async.bulk` 异步存、要么普通 `st.global` 向量化存。
- **Blackwell 受限 gather**：sm_100a 起 PTX 增加了 `tile::gather4` 一类的受限 gather 模式（2D，一次抓 4 行、列窗口相同），专门给 attention / 这类"行不连续、列连续"的访问用。它比 cp.async 逐线程抓快，但适用范围很窄，别指望它替代通用 gather。

---

# 决策表

| 你的情形 | 选择 |
|---|---|
| 常规 GEMM/attention 主 tile，多 stage 流水，warp-specialized | **TMA tensor** |
| 一维连续 chunk（无 stride） | **TMA 1D bulk** |
| cluster 内广播、2-SM `tcgen05` | **TMA multicast**（cp.async 做不到） |
| Epilogue 存回 global | **TMA bulk store** 或 `st.global` |
| 形状/stride 不满足 tensormap 约束 | **cp.async** |
| 每线程不定量 / 索引 gather / 极小 stage | **cp.async**（或 `ld.global.nc`） |
| 一份代码要跑 sm_80 到 sm_100 | **cp.async** 作 fallback |
| 想省 issue slot、压寄存器 | **TMA** |

---

# 混用时的两个坑

<details>
<summary><b>展开：同时用 cp.async 和 TMA 的同步陷阱</b></summary>

1. **完成机制不同，别混在同一个 SMEM buffer 上。** `cp.async` 靠 `commit_group` / `wait_group N` 的**组序号**推进；TMA 靠 `mbarrier` 的 `expect_tx` + phase flip。两者语义不通约。如果确实要让同一个 stage 里既有 cp.async 又有 TMA 写入，必须 `cp.async.wait_group 0` → `__syncthreads()` → `mbarrier.try_wait` 两段串起来，非常容易写出 race。实践上按 SMEM 区域分开用最安全。

2. **async proxy 的可见性。** TMA 走 async proxy，如果你先用普通 `st.shared` 写好 SMEM 再让 TMA 从这块 SMEM 存回 global，中间必须 `fence.proxy.async.shared::cta`；反向（TMA 载入后由普通指令读）也需要正确的 acquire 语义。`cp.async` 也有类似约束但模式不同，混用时最容易在这里踩坑。

</details>

---

# 结论

Blackwell 上 **TMA 是默认，`cp.async` 是例外通道**。如果只是"因为原来代码用的是 cp.async 所以继续用"，这在 sm_100 上通常是性能损失，主要损失在 issue 带宽和寄存器上。真正值得保留 cp.async 的，基本只有两类：**布局满足不了 tensormap 约束**，以及**细粒度/逐线程谓词的不规则访问**。其余场景用 `cp.async` 换来的只是兼容性，不是性能。

另外提醒一点：`cp.async` 只解决"搬进来"，搬出去和跨 CTA 共享它都无能为力，所以即使保留 cp.async，主流水线之外通常还是得有 TMA。
