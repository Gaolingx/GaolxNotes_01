# 高精度MD引擎优化：JAX骨架+Triton/cuTile热点

## 🧑‍💻 User

针对高精度分子动力学引擎优化方面 使用JAX还是Triton还是cu Tile比较好一些

---
## 😀 默认助手 (7)

## 结论先行

问题本质上是在比较**不同抽象层级**，而不是在比较三个可以互相替换的框架。在“高精度 MD”这个约束下，抽象层级的选择反而应该从“误差累积在哪里”倒推。

| 场景 | 选择 | 理由 |
|---|---|---|
| 整机骨架、积分器、多 GPU 分解、MLIP、可微 MD | **JAX** | `shard_map`/GSPMD 无可替代，且只有它有 autodiff |
| 热点 kernel（非键合力、PME spread/gather、邻居表、SHAKE） | **Triton**（今天）/ **cuTile**（押注 NVIDIA 长期） | 能把“一步一核”塞进单 kernel，省掉全程 global memory 往返 |
| 追求最后 5%~10%，或要 AOT 交付、跑在 sm90+ 新架构 | **cuTile** | TileIR + `export_kernel`，比 Triton 更贴硬件，且能 FFI 回 JAX |
| 完全不设上限、要长期生产级 | **CUDA C++**（GROMACS/OpenMM 路线） | 仍然最快；cuTile 是官方给的“逃离 C++”路径 |

**一句话**：用 JAX 做骨架，用 Triton/cuTile 写热点并通过 `jax.ffi` 或 `ct.jax.cutile_call()` 挂回去。三者是**栈关系而非竞争关系**。

---

## 一、先厘清：“高精度”把这个问题的性质改了

这一步比选框架重要得多，因为它决定了框架能带来多少收益上限。

### 1. FP64 已是稀缺资源

| 代次 | FP64 : FP32 向量吞吐 |
|---|---|
| A100 | 1 : 2 |
| H100 / B200 | ≈ 1 : 2 |
| B300 / GB300 | ≈ 1 : 60（公开规格，FP64 被大幅削减） |

MD 非键合力是纯 FMA、纯 gather，用不上 tensor core —— 也就是说 **MD 恰好落在被砍得最狠的那个单元上**。同时，$r^{-1}$、$r^{-7}$、$r^{-13}$ 这类计算对尾数敏感，朴素的 FP32 会让能量漂移在 $10^6$ 步量级暴露。

$$\Delta E \sim \mathcal{O}(\varepsilon_{\text{mach}} \cdot \sqrt{N_{\text{steps}}}) \cdot E_{\text{kin}}$$

因此真正的杠杆是**混合精度分解**，而不是“全 FP64”：

- 位置/速度积分、力的归约累加 → FP64（误差会累积的地方）
- 对内距离、掩码、cutoff 判定 → FP32 甚至 TF32（误差不累积的地方）
- 长程实空间求和 → 补偿求和（2Sum/2Prod），成本约 $2\times$ FMA 而非 $4\times$

**量级感受**：框架选得好，通常带来 10%~20%；精度策略选得好，带来 2×~4×。别把 90% 的精力花在 20% 的收益上。

### 2. 瓶颈不在 FLOPs，在 gather 和访存

每个对的计算量约 40~50 FLOP，但如果不做邻居重排（GROMACS 的 cluster-pair / tile-sorted 邻居表），访存是随机的、合并度极差，此时算力利用率可以低到 20% 以下。

**推论**：能让邻居索引**重排成 tile 格式**、能控制**寄存器级分块**的方案，比“FP64 峰值高”的方案更值钱。这一条直接把 JAX 从“热点内核”候选里排除了。

### 3. 最大的结构性收益：单步单核

一个 MD 步 = 建表 → 非键合 → 键合 → PME → 积分。如果每个阶段都是独立 kernel，位置/力要在 HBM 上往返 5 次。真正的最优解是把整步融进**一个 persistent kernel**，位置和力全程留在寄存器和共享内存里。

- JAX/XLA：**做不到**（能做单 kernel 的只有 Pallas，但 Pallas 也无法跨阶段串成持久核）
- Triton：能做单 kernel，但跨 kernel 的持久化要靠 manual persistent loop
- cuTile：tile 模型天然就是写持久核的形状

这是 Triton/cuTile 相对 JAX 的**结构性**优势，不是百分点级的。

---

## 二、三者逐一分析

<details>
<summary><b>JAX —— 最强骨架，最弱热点</b></summary>

**强项**

- `jax.jit` + `fori_loop` + `scan`：整步图级融合、CUDA Graph 友好，Python 开销可压到零
- `shard_map` / GSPMD：多 GPU 域分解 + halo 交换几乎是“免费”的，这是 MD 规模化的刚需
- `grad` / `vmap`：MLIP 训练、可微 MD、力 = $-\nabla U$ 的一体化工作流，独此一家
- `jax.ffi` + Pallas（Triton/Mosaic 后端）：随时把热点降级成手写 kernel
- 可移植到 CPU/TPU/AMD，数值路径统一

**致命短板（针对 MD）**

- 邻居表只能表达成稠密 `[N, max_neighbors]`，密度起伏大时 padding 大量浪费
- `at[idx].add()` 的 scatter 在 XLA 上**非确定**且慢，而高精度 MD 需要可复现的归约顺序
- 无法控制寄存器分块、共享内存布局、`cp.async`/TMA 流水
- 无法写出整步持久 kernel

**怎么用才对**：骨架 + `fori_loop` 融合 + FFI 挂热点。**不要**试图用 pure JAX 追 GROMACS 的性能。

</details>

<details>
<summary><b>Triton —— 今天的最优性价比</b></summary>

**强项**

- 真正的 tile/block 级编程：显式 `tl.load/store`、共享内存、软件流水、`tl.dot`、`tl.atomic_add`
- 不规则 gather 就是 `tl.load(ptr + idx)`，配合 tile-sorted 邻居表可以做到很好的合并度
- FP64 完整支持（`tl.float64`）；**没有 FP64 MMA 也无所谓** —— MD 本来就要 FMA
- 已在 PyTorch/JAX 生态里验证过的 FFI 路径，落地风险最低
- 社区与内核库最厚，遇到坑有人踩过

**短板**

- 没有跨 CTA 同步、没有 warp shuffle 的完整暴露、没有动态并行
- `tl.dot` 之外的分块要靠手工，FP64 寄存器压力是 FP32 的两倍，容易 spill
- 编译器启发式对 FP64 的流水做得一般，需要大量 autotune
- 无多 GPU 能力（仍得自己上 NCCL/NVSHMEM）

**判断**：如果你的目标是“6 个月内做出一个能打的高精度引擎”，选 Triton。成熟度差距在工程时间里会直接兑现。

</details>

<details>
<summary><b>cuTile —— 2026 年已经不该再叫“玩具”了</b></summary>

我查了官方文档确认现状（cuTile Python 1.7.0 / CTK 13.4，2026-10）：它已经不是早期实验品。

**能力清单（已具备的）**

- `float64` 是一等 dtype，算术提升表里完整存在 — 高精度可行
- `ct.gather()` / `ct.scatter()`（带 `mask`）+ `ct.load_advanced_indexing()` — 正好对上 MD 的不规则访问
- `ct.atomic_add()`（含 bf16 on sm90+）— 力累加可用，但**顺序不确定**，确定性归约要自己做 tile 内累加 + 定序归约
- `Array.tiled_view(traversal_steps=...)` — 直接对应 cluster-pair / halo 的重叠 tile
- `ct.assume_divisible_by()`、`static_shape_dims` / `static_stride_dims` — 把 stride 变成编译期常量，这对 MD 的 SoA 布局是实打实的收益
- `ct.tune.exhaustive_search()` + `flush_l2` — 有自己的 autotune
- `export_kernel()` AOT 导出 TileIR，交付期不需要 Python
- `ct.jax.cutile_call()` — **原生 JAX FFI**，混合栈的第一类公民
- `grid_dependency_control_*`（PDL）— 多阶段 kernel 重叠启动，对建表→力→积分流水很关键
- 架构覆盖：sm80 / sm90 / sm100 / sm107，跨代可移植

**短板**

- NVIDIA only，无第二供应商出路
- tile 模型把共享内存交给编译器管，**显式控制反而弱于 Triton** —— 想手抠 smem 布局/double buffer 精细度时你会想回去写 CUDA
- 生态薄：没有现成 kernel 库、没有 Pallas 那种通用中间层、调试经验稀缺
- Tile 维度必须编译期常量且为 2 的幂 —— 对变长体系（每个原子邻居数不同）需要重新设计数据组织
- 没有 autodiff

**判断**：如果团队本来就只跑 NVIDIA、且愿意吃 3~6 个月的爬坡，cuTile 是**未来 3 年最正确的投资**；但它现在还不是“拿来就能超车 Triton”的选择。

</details>

---

## 三、关键维度对比

| 维度 | JAX | Triton | cuTile |
|---|---|---|---|
| FP64 算术 | ✅ 完整 | ✅ 完整 | ✅ 完整 |
| 单步 mega-kernel | ❌（Pallas 单核可用） | ✅ | ✅ 最自然 |
| 不规则 gather / 邻居表 | ⚠️ 只能稠密 pad | ✅ | ✅ gather/scatter + mask |
| 寄存器/smem 精细控制 | ❌ | ⚠️ 中 | ⚠️ 编译器托管 |
| 确定性归约 | ❌（XLA atomics） | ✅ | ✅ |
| 多 GPU | ✅✅ 最强 | ❌ | ❌ |
| 自动微分 | ✅ | ❌ | ❌ |
| Autotune | XLA 自动 | 手工/`triton.autotune` | ✅ 内建 exhaustive |
| AOT 交付 | ✅ XLA | ⚠️ | ✅ TileIR export |
| 成熟度 | 高 | 最高 | 中（1.7，跨 4 代架构） |
| 多厂商 | ✅ AMD/TPU/CPU | ⚠️ AMD | ❌ NVIDIA only |
| 性能上限 | 中 | 高 | 很高 |

---

## 四、推荐架构：三层混合栈

```
┌─ Python 编排层 ─────────────────────────────┐
│  体系构建 / I/O / 检查点 / 报告 / 参数扫描    │
├─ JAX 图层 ─────────────────────────────────┤
│  fori_loop 整步融合 · shard_map 域分解        │
│  CUDA Graph 提交 · 可微 MD / MLIP             │
├─ 热点 kernel 层（按需二选一）─────────────────┤
│  Triton：非键合 · PME spread/gather · 邻居表  │
│  cuTile：同上，或整步 persistent kernel       │
├─ CUDA 生态 ────────────────────────────────┤
│  cuFFT (PME) · NCCL/NVSHMEM · cuSPARSE       │
└────────────────────────────────────────────┘
```

<details>
<summary><b>Triton 热点 kernel 骨架</b></summary>

```python
import triton, triton.language as tl

@triton.jit
def nonbonded_f64(
    pos_ptr,      # [N,3] float64
    f_ptr,        # [N,3] float64
    nb_ptr,       # [N, MAXNB] int32, tile-sorted
    nb_cnt_ptr,   # [N] int32
    N, CUT2,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_i = pid * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_i = idx_i < N

    # 位置一次性载入寄存器，整段 tile 复用 —— 这是算术强度的来源
    xi = tl.load(pos_ptr + idx_i * 3 + 0, mask=mask_i, other=0.0)
    yi = tl.load(pos_ptr + idx_i * 3 + 1, mask=mask_i, other=0.0)
    zi = tl.load(pos_ptr + idx_i * 3 + 2, mask=mask_i, other=0.0)

    fx = tl.zeros([BLOCK_I], tl.float64)
    fy = tl.zeros([BLOCK_I], tl.float64)
    fz = tl.zeros([BLOCK_I], tl.float64)

    for k in range(0, MAXNB, BLOCK_J):          # 编译期常量 -> 完全展开
        jj = tl.arange(0, BLOCK_J)[None, :] + k
        nb = tl.load(nb_ptr + idx_i[:, None] * MAXNB + jj,
                     mask=mask_i[:, None] & (jj < MAXNB), other=-1)
        valid = nb >= 0

        # 注意：这里是 gather，合并度取决于邻居表是否已 tile-sorted
        xj = tl.load(pos_ptr + nb * 3 + 0, mask=valid, other=0.0)
        yj = tl.load(pos_ptr + nb * 3 + 1, mask=valid, other=0.0)
        zj = tl.load(pos_ptr + nb * 3 + 2, mask=valid, other=0.0)

        dx = xi[:, None] - xj
        dy = yi[:, None] - yj
        dz = zi[:, None] - zj
        r2 = dx * dx + dy * dy + dz * dz
        # ... 在 FP64 里算 inv_r，或走"对内距离用 FP32 + 补偿"的混合路径
        # ... 力的 tile 内累加 -> tl.sum(axis=1)，顺序确定，不用 atomic

    tl.store(f_ptr + idx_i * 3 + 0, fx, mask=mask_i)
    # ...
```

关键点：`tl.sum(..., axis=1)` 的归约顺序是确定的 —— 这就是你相对 JAX 拿到**可复现高精度**的地方。

</details>

<details>
<summary><b>cuTile 写法与 JAX 接线</b></summary>

```python
import cuda.tile as ct

@ct.kernel
def nonbonded_kernel(pos, force, nb, nb_cnt, CUT2,
                     BLOCK_I: ct.Constant[int]):
    bi = ct.bid(0)
    # 用 tiled_view 把 [BLOCK_I, MAXNB] 的邻居块直接切出来
    # ct.gather / ct.scatter 处理不规则索引，mask 参数处理变长邻居
    ...
```

```python
# 挂到 JAX 图里 —— cutile_call 是官方 FFI，不需要自己写 XLA custom call
from cuda.tile import jax as ctjax

def step(params, state):
    state = ctjax.cutile_call(
        nonbonded_kernel, dims=(grid,), inputs=[pos, force, nb, cnt, cut2]
    )
    return integrate(state)
```

好处：`export_kernel()` 之后可以完全脱离 Python 部署，且 TileIR 能跨 sm90/sm100 复用。

</details>

---

## 五、按你的实际情况选

**如果目标是发论文 / 做方法论原型**
→ JAX 全栈。autodiff + `vmap` 让你能同时跑 32 个温度点，这个生产力优势碾压性能差异。热点暂时接受 XLA 的结果，需要时再降级。

**如果目标是一个真正能跑的引擎（10 万~100 万原子，多卡）**
→ JAX 骨架 + Triton 热点。这是当前风险最低、天花板够高的组合。

**如果目标是在 2~3 nm 制程的新卡上榨干性能，且押注 NVIDIA**
→ JAX 骨架 + cuTile 热点。`ct.jax.cutile_call()` 已经把这条件路修好了，且 cuTile 的 PDL、TileIR AOT、跨代可移植性会在 Blackwell/Rubin 上越来越值钱。

**如果体系里有大量键合约束 + PME + 多时间步**
→ 别自己造。评估 OpenMM 的 GPU platform（可扩展 custom force / plugin）或 GROMACS 的 C++ 接口，把精力放在你的科学差异点上。

---

## 六、落地前必须做的三件事

1. **建 FP64 微基准**，三个方案各写一个非键合 kernel，在**你的体系、你的邻居密度**下测。别信任何通用 benchmark —— MD 性能对邻居分布极其敏感，公开数字没有参考价值。

2. **用四倍精度（mpmath / double-double）做参考解**，测 $10^6$ 步的能量漂移与力的逐对相对误差。这决定你能不能把 FP64 降成混合精度 —— 而这是唯一能带来 2× 以上的改动。

3. **先确定归约顺序**。原子加（atomic add）会毁掉可复现性；高精度场景应做 tile 内寄存器累加 + 定序归约。这一条**必须在写第一行 kernel 前就定**，后改要重写数据布局。
