# Vulkan HPC 框架落地路线：借鉴 Kompute 实现，对标 PETSc 后端，首攻 SpMV+halo 交换

## 🧑‍💻 User

一个基于vulkan compute shader（kcompute）的hpc框架

---
## 😀 默认助手 (7)

我理解这个问题有两种可能：**(a)** 你想设计或评估一个叫 `kcompute` 的「Vulkan compute shader 之上的 HPC 框架」；**(b)** 你在说已有的 **Kompute**（我查了一下：`KomputeProject/kompute`，C++/Apache-2.0，~2.6k stars，Linux Foundation 支持，定位是“general purpose GPU compute framework built on Vulkan”）。我先把 **Kompute 当作直接参照物**来讲，再给出如果要从零做 kcompute 的架构与关键问题。如果你手上已经有仓库想让我 review，把链接或代码贴出来，我按那套代码讲。

---

## 1. 先给结论：Vulkan 做 HPC 框架，收益边界在哪

| 维度 | Vulkan compute (kcompute) | CUDA | HIP | SYCL | OpenCL |
|---|---|---|---|---|---|
| 厂商覆盖 | NVIDIA / AMD / Intel / Qualcomm / Apple(MoltenVK) | NV only | AMD(+部分 NV 翻译) | 多后端 | 广但生态老化 |
| 生态库（BLAS/FFT/稀疏） | **几乎为零，得自己写** | cuBLAS/cuSPARSE/cuFFT/CUB | roc* 系列 | oneMKL | clBLAS 等（半死） |
| 设备端动态并行 | 无（最大短板） | 有 CDP | 有 | 部分 | 无 |
| 显存超配 / UVM | 无，需自己做 residency 管理 | 有 | 有 | 视后端 | 无 |
| 性能上限 | 带宽型内核接近原生；重 FP64 / Tensor 型差很多 | 基准 | ≈CUDA | 视后端 | 中 |
| 容器 / 集群部署 | 需要 ICD + 驱动，headless 可行 | 成熟 | 成熟 | 中 | 中 |

**一句话判断**：如果目标集群是清一色 NVIDIA，Vulkan 框架只是给自己找麻烦；它的真实价值场景是 **异构集群（AMD MI + NV + Intel + 国产/嵌入式加速器）**、**长周期软件想摆脱厂商锁定**、**同一份代码要跑在数据中心和移动/边缘**。Kompute 的定位也基本是这样（mobile-enabled、cross-vendor）。

要接受的一个现实：**瓶颈从来不是 runtime，而是库**。CUDA 的护城河是 CUB/Thrust/cuSPARSE，不是 `<<<>>>` 语法。所以 kcompute 的工作量 70% 会花在 scan / sort / SpMV / stencil / FFT / 求解器上，而不是在 `vkCmdDispatch` 上。

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────┐
│ L6 应用/数值库: BLAS-L1/2/3, SpMV(CSR/ELL/Blocked-ELL),   │
│                FFT, 迭代求解器, checkpoint/restart        │
├──────────────────────────────────────────────────────────┤
│ L5 分布式: MPI rank↔device 绑定, 域分解, halo 交换,        │
│           dma-buf / VK_KHR_external_memory_rdma 零拷贝    │
├──────────────────────────────────────────────────────────┤
│ L4 图调度: kernel DAG, 拓扑排序, 多队列重叠,              │
│           persistent-CTA executor + device-side work queue│
├──────────────────────────────────────────────────────────┤
│ L3 内核层: SPIR-V 缓存(内容哈希), 反射自动绑定,           │
│           specialization constant 当模板参数, 自动调优    │
├──────────────────────────────────────────────────────────┤
│ L2 内存/同步: 子分配器(VMA 风格), 区间级 hazard 追踪,      │
│              自动插 sync2 barrier, timeline semaphore      │
├──────────────────────────────────────────────────────────┤
│ L1 设备/HAL: 物理设备选择, queue family(compute-only),     │
│              feature 探测 → 能力矩阵, 命令缓冲池            │
└──────────────────────────────────────────────────────────┘
```

高层 API 建议长这样（让框架自己管 hazard 和 barrier，这是它相对裸 Vulkan 的核心价值）：

```cpp
kc::Context ctx = kc::Context::create(/*prefer=*/kc::Vendor::Any, /*fp64=*/false);
kc::Graph  g(ctx);

auto x = g.buffer<float>(n);          // device-local
auto y = g.buffer<float>(n);
auto k = ctx.kernel("axpy.spv")       // 编译期嵌入 / 运行时 libshaderc
            .specialize({{"WG", 256}});

g.add(k).in(x).inout(y).scalar(alpha).grid_divisible(n, 256)
 .on(kc::QueueSel::ComputeDedicated);

g.submit();                            // 自动分析读写区间、插 barrier、流水重叠
y.copy_to_host(host_y);                // 内部用 timeline semaphore + staging
```

---

## 3. 五个真正难的点（也是能不能叫“HPC 框架”的分水岭）

<details>
<summary><b>① 没有设备端动态并行 → 用它做稀疏/图/粒子类负载</b></summary>

CUDA 的 CDP 让你在内核里 launch 内核。Vulkan 没有。三条现实路径：

- **Persistent kernel + 设备端软件队列**：固定 `gridDim` 数量的 CTA 常驻，从全局 atomic 计数器取 tile 做 work-stealing。对不规则负载（SpMV merge-path、粒子、图遍历）几乎是唯一高效解。风险：需要保证所有 workgroup 同时驻留，否则死锁——要按 `maxComputeWorkGroupCount` × `maxComputeWorkGroupInvocations` × 每 SM 驻留数精确算 launch 配置。
- **`VK_EXT_device_generated_commands`**：设备端生成 dispatch 命令，接近 work-graph 能力，但驱动支持参差。
- **图的粒度重派发**：host 侧按 DAG 节点反复 submit，配 timeline semaphore + `vkQueueSubmit` 批量，牺牲一点延迟换可移植性。框架应该三种都封装，按 `features` 探测选择。
</details>

<details>
<summary><b>② FP64 与数值可复现性（HPC 的两个“政治正确”）</b></summary>

- `shaderFloat64` 存在 ≠ 快。消费级卡 FP64 常在 FP32 的 $1/32 \sim 1/64$，Vulkan 照样能跑，性能照样废。框架要么在能力矩阵里标出 $R_{fp64} = P_{fp64}/P_{fp32}$ 并在超过阈值时拒绝/降级，要么提供 **double-float (df64, 两个 fp32 模拟双精度)** 路径 + mixed-precision iterative refinement。
- **可复现性**：`atomicAdd` 顺序不定 → 每次跑结果 bit 不一致，HPC 验收/回归测试会直接卡死。框架必须提供 `reduce_deterministic()`（固定二叉树规约，禁止全局原子累加）开关，并默认开启。
- 精度控制：必须用 `VK_KHR_shader_float_controls`（signed zero、denorm 保留、rounding = RTNE）和 `float_controls2`（禁 fast-math 的 reassoc/contract），否则跨厂商结果漂移。这是 Vulkan 相对 CUDA 少有人提、但对 HPC 极其关键的一块。

Roofline 判断内核值不值得搬过来：

$$P \le \min\!\left(P_{\text{peak}},\; I \cdot B_{\text{mem}}\right),\qquad I = \frac{\text{FLOP}}{\text{Byte}}$$
</details>

<details>
<summary><b>③ 显存不够 → 自己做 UVM / residency</b></summary>

Vulkan 没有自动迁移。方案：`VK_EXT_memory_budget` 拿预算 + sparse binding（`VK_KHR_sparse_resources` 系）按 tile 映射/解映射，配合 LRU + 预取实现 page-level 换入换出；或干脆要求“问题规模适配显存 + 域分解”。HPC 里通常是后者（MPI 分块），但框架该把前者作为大数组溢出路径。
</details>

<details>
<summary><b>④ 多卡 / 跨节点：Linux dma-buf 是唯一优雅解法</b></summary>

- 同机多卡：`VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` 导出 dma-buf，另一块卡导入 → 零拷贝共享（NVLink/PCIe P2P 是否真生效看驱动，NVIDIA 的 Vulkan 在这条路上支持有限，AMD 较好）。
- 跨节点：`VK_KHR_external_memory_rdma` 让显存直接注册给 IB verbs，就是 Vulkan 版的 GPUDirect RDMA；不支持时退回 staging 显存→host pinned→IB。
- 同步：`VK_KHR_external_semaphore_fd` 导出的 semaphore 可以在进程间/MPI rank 间传递，天然对上 MPI 的 device-aware 模型。
- 域分解的通信量下界（立方域切成 $P$ 块）：

$$\frac{\text{halo}}{\text{compute}} \propto \frac{6L^2}{L^3} = \frac{6}{L},\qquad L=\left(\frac{N}{P}\right)^{1/3}$$

所以 kcompute 的通信层必须支持 **halo 与计算的重叠提交**（compute queue 与 transfer queue 分离，double-buffer halo）。
</details>

<details>
<summary><b>⑤ 生产可用性：device lost、校验、性能可观测</b></summary>

- `VK_ERROR_DEVICE_LOST` 在 HPC 长跑作业里是致命的，Vulkan 没有优雅恢复。唯一实用解是 **checkpoint/restart**：应用级 dump 到 host 或 `cuda-checkpoint` 式的进程级（Linux + CRIU，需驱动配合）。框架应把 checkpoint 做成一等 API。
- 计时/计数：`VK_EXT_calibrated_timestamps` + timestamp query（纳秒级、可跨队列对齐）+ `VK_KHR_performance_query` 抓硬件计数器（AMD/Intel 支持较好，NV 差）。没有这些，调优无从谈起。
- 务必内置 validation layers 开关 + `VK_LAYER_KHRONOS_profiles` 用于“模拟只能跑 Vulkan 1.1 的卡”。
</details>

---

## 4. 最小可用代码骨架

**Kernel（GLSL → SPIR-V，编译期嵌入）**

```glsl
#version 460
#extension GL_EXT_scalar_block_layout : require
layout(local_size_x_id = 0) in;            // specialization constant 当模板参数

layout(scalar, binding = 0) buffer X { float x[]; };
layout(scalar, binding = 1) buffer Y { float y[]; };
layout(scalar, push_constant) uniform Pc { float a; uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    y[i] = pc.a * x[i] + y[i];
}
```

**Runtime 切片（`vulkan-hpp`，只保留关键骨架）**

```cpp
#include <vulkan/vulkan.hpp>
using namespace vk;

struct Slice {
  PhysicalDevice phys; Device dev; uint32_t qfam; Queue q;
  CommandPool pool; bool has_sync2, has_tl_sem;
};

// 1) 选队列：优先 compute-only，且不与 graphics 共享
uint32_t pick_compute_family(const std::vector<QueueFamilyProperties>& fams) {
  for (uint32_t i = 0; i < fams.size(); ++i)
    if (fams[i].queueFlags & QueueFlagBits::eCompute &&
       !(fams[i].queueFlags & QueueFlagBits::eGraphics)) return i;
  for (uint32_t i = 0; i < fams.size(); ++i)
    if (fams[i].queueFlags & QueueFlagBits::eCompute) return i;
  throw std::runtime_error("no compute queue");
}

// 2) 派发：一次 dispatch 的完整生命周期
void axpy(Slice& s, Buffer x, Buffer y, float a, uint32_t n, Pipeline pipe, DescriptorSet set) {
  auto cb = s.dev.allocateCommandBuffers(
      CommandBufferAllocateInfo{s.pool, CommandBufferLevel::ePrimary, 1})[0];
  cb.begin({CommandBufferUsageFlagBits::eOneTimeSubmit});

  // 自动 hazard：host 写过的 buffer → 需要 transfer/fragment barrier（此处略，由 L2 追踪器生成）
  cb.bindPipeline(PipelineBindPoint::eCompute, pipe);
  cb.bindDescriptorSets(PipelineBindPoint::eCompute, pipe.getLayout(),
                        /*firstSet=*/0, {set}, {});
  cb.pushConstants(pipe.getLayout(), ShaderStageFlagBits::eCompute, 0, sizeof(float) + sizeof(uint32_t), &a);
  cb.dispatch((n + 255) / 256, 1, 1);          // 真实框架按 maxComputeWorkGroupCount 自动切分

  cb.pipelineBarrier2(DependencyInfo{}.setMemoryBarriers(
      {MemoryBarrier2{}.setSrcStageMask(PipelineStageFlagBits2::eComputeShader)
                       .setSrcAccessMask(AccessFlagBits2::eShaderWrite)
                       .setDstStageMask(PipelineStageFlagBits2::eHost)
                       .setDstAccessMask(AccessFlagBits2::eHostRead)}));
  cb.end();

  // timeline semaphore：免去一堆 binary semaphore 的状态机
  Semaphore ts = s.dev.createSemaphore({SemaphoreCreateInfo{},
                                        {SemaphoreType::eTimeline, 0}});
  s.q.submit({SubmitInfo{}.setCommandBuffers({cb})}, /*fence=*/nullptr);
  s.q.waitSemaphores({{}, ts, ++s.tick, UINT64_MAX});
  ts.wait(s.tick);
}
```

真正需要工程量的是那个 **hazard 追踪器**：给每个 buffer 记 `(读区间, 写区间, lastStage, lastAccess)`，合并相邻区间，只在真冲突时插 barrier（sync2 的 `DependencyInfo`），并做 barrier 合并/下沉。Kompute 在这方面的抽象是 `Sequence` + 几种 memory barrier 档位；想比它更强，就得做区间级（甚至 tile 级）追踪。

---

## 5. 值得直接借鉴 / 不必重造的部分

- **内存分配**：直接用 **VulkanMemoryAllocator (VMA)**，别自己写。
- **SPIR-V → 绑定信息**：`SPIRV-Reflect` / `SPIRV-Cross` 做反射，实现“只给 buffer，框架自动 bind”。
- **FFT**：接 **VkFFT**（就是纯 Vulkan compute 的，已经是该领域的成熟件）。
- **着色器编译**：`glslang` / `DXC` / **Slang**（Slang 对泛型 + specialization 不友好的问题的解决比 GLSL 好得多，很适合做“kcompute 内核 DSL”）。
- **抽象层次参照**：Kompute 的 `Manager → Sequence → Algorithm → Tensor` 四层是很好的起点，但要补的是：无 `Sequence` 内部 barrier 自动分析、无 deterministic 规约、无多卡、无 checkpoint——这几点正好是 HPC 与“通用 GPU compute”的分界线。

---

## 6. 路线图（我建议的优先级，如果是从零做）

1. **阶段 0（2–3 周）**：单卡、单队列、手动 barrier 的 `spv + buffer + dispatch`，跑通 3 个微分方程/线性代数基准内核，先把校准计时做好。
2. **阶段 1**：VMA 子分配 + SPIR-V 反射自动绑定 + specialization constant 模板化 + pipeline cache 落盘。此时对带宽型内核应能摸到该卡 80%+ 峰值的量级。
3. **阶段 2**：hazard 追踪 + 多队列重叠 + persistent-CTA executor（解决不规则负载）。这一步决定框架上限。
4. **阶段 3**：库层——scan/reduce/sort/SpMV/stencil，全部带 deterministic 模式 + 自动调优表。
5. **阶段 4**：MPI 绑定（dma-buf / external semaphore / RDMA），checkpoint/restart，performance_query 集成。

第 3 步之后的每一项都至少是“人月”级投入；这也是为什么我认为**这张牌在纯 NV 集群上打不出去，只有在真异构场景下才值**。

---

## 需要你确认的三件事

1. 你说的 `kcompute` 是**已有项目**（给我 repo / 文档，我做 code review 或架构评估）还是**你要新做的东西**（那按上面路线图，我可以直接帮你落第一阶段代码骨架）？
2. 目标硬件到底是**异构混合集群**，还是某一家（比如 AMD MI300 / 国产卡）——这决定要不要做 vendor-specific kernel 变体。
3. 应用侧是**自己写内核**（那 DSL/调优是重点），还是**要 port 现成 Fortran/C++ 代码**（那 Fortran `ISO_C_BINDING` 接口和库层完备度是重点，工作量分布完全不同）？

告诉我后两点，我可以把上面的 L1–L3 直接写成可编译的工程骨架（CMake + vulkan-hpp + VMA + glslang 离线编译 + 一个 AXPY/SpMV 基准）。

---
## 🧑‍💻 User

https://petsc.org/release/
https://github.com/KomputeProject/kompute
这个有用吗

---
## 😀 默认助手 (7)

两个都有用，但**它们的作用完全不同，而且加起来仍然不等于一个 HPC 框架**。我把查到的实际情况列出来，方便你判断。

---

## 1. Kompute：能“借鉴参考”，不适合“作为底座依赖”

我查看了它的提交历史（截至 2026-08-15 的最新提交）：

| 事实 | 数据 |
|---|---|
| 规模 | ~2.6k stars、202 forks、Apache-2.0、C++、LF 支持 |
| 活跃度 | 提交仍在提交，但**全部是维护类改动**：NumPy 2.x pin、fmt/spdlog、MSVC 编译修复、pybind 3.0、gtest 升级、文档、Android 构建 |
| 待办 | 75 open issues、4 open PR |
| 功能方向 | 近两年**没有任何朝 HPC 方向的功能演进** |

**结论：它现在是一个处于“维护模式”的通用 GPU compute 库，不是 HPC 框架。**

- ✅ **值得借鉴的部分**：`Manager → Sequence → Algorithm → Tensor` 四层抽象、内存类型枚举（host / device / device-and-host / device-and-local）、SPIR-V 加载与 SpecConst/PushConst 的封装、CMake 里 glslang 离线编译 SPIR-V 的流水线。这几块你自己写也要花 1–2 个月，读它的实现能省一大截。
- ❌ **不适合依赖的部分**：
  - **性能关键路径不建议依赖**。它的 `Sequence` 使用粗粒度的 memory barrier 档位（本地/远程、读/写/读写），没有区间级 hazard 追踪，因此 barrier 会多插、会串行化；HPC 内核之间 barrier 成本非常关键。
  - 缺的东西恰好是 HPC 的必需品：**确定性规约**（它的 GPU 规约受 atomic 顺序影响，结果 bit 不一致 → 无法做回归测试）、**多设备/多进程**、**FP64 路径与精度控制（`VK_KHR_shader_float_controls`）**、**checkpoint/restart**、**稀疏库**。
  - 抽象层的对象模型耦合在 Vulkan 上，强行在它上面叠一层 HPC 调度（persistent-CTA executor、多队列重叠）会与它的 `Sequence` 串行模型冲突。

**我的建议**：**Reading copy，不要 dependency**。借鉴其 L1/L2 实现，内核层和图调度层自己写。

---

## 2. PETSc：这部分才是真正有价值的

PETSc（现在 3.26.0）本身不是“GPU 框架”，但它是**唯一能把你的框架放到真实 HPC 标准下验收的载体**，价值有三层：

<details>
<summary><b>① 它已经把“可移植 GPU 后端”这个位置空出来了（而且有现成的模板可以仿照）</b></summary>

我查了它的设备抽象：

```c
typedef enum {
  PETSC_DEVICE_HOST, PETSC_DEVICE_CUDA, PETSC_DEVICE_HIP,
  PETSC_DEVICE_SYCL, PETSC_DEVICE_MAX
} PetscDeviceType;
```

**加一个 `PETSC_DEVICE_VULKAN` 在架构上是“被支持的模式”**——照 SYCL 后端那一路的提交改就行（configure 探测、`PetscDeviceType` 分支、`Vec`/`Mat` 的 `ops` 结构体、`PetscSF` 的 device-aware 路径）。这意味着你不用自己造 API、不用自己写 Fortran 绑定、不用自己写 petsc4py——**这些全都免费继承**。这是路线 A 最大的诱惑。

⚠️ 但注意一个**警示信号**：PETSc 主页仍写着 GPU 经 “CUDA, HIP, Kokkos, or OpenCL”，而 `PetscDeviceType` 里**根本没有 OpenCL**。也就是说，历史上那条“可移植后端”路线最终没留在设备抽象里。**Kokkos/SYCL 也没能在 PETSc 里取代 cuSPARSE/hipSPARSE 成为性能主路径。** 这是对 kcompute 最直接的前车之鉴：可移植层在 HPC 数值栈里通常用来做正确性校验和边缘硬件，性能收益仍靠厂商原生。**你打算靠什么打破这个规律，得先想清楚。**
</details>

<details>
<summary><b>② 它精确地定义了你要实现的 API 面（附工程量评估）</b></summary>

| PETSc 对象 | 需要实现的 Vulkan 侧能力 | 难度 |
|---|---|---|
| `PetscDevice` / `PetscDeviceContext` | 设备/队列抽象 + **stream 语义**（→ `VkQueue` + timeline semaphore） | 中 |
| `PetscMemory` | 异步 H2D/D2H、pinned staging、可映射的 device-and-host 内存 | 中 |
| `Vec` (seq/MPI) | AXPY / Dot / Norm / PointwiseMult / Maxpy / Scatter/Gather | 低 |
| `Mat` (AIJ) | **`MatSetValues` 的语义**：host 侧随时可写 → 需要 dirty flag + 延迟上传 | 中（易出错） |
| `MatMult` / `MatMultTranspose` | CSR / ELL / 分块 SpMV，merge-based 负载均衡 | 中高 |
| **`PetscSF`** | halo exchange，device-aware，跨进程/跨设备（dma-buf、external semaphore） | **高** |
| `PCJACOBI` / block Jacobi | 向量级操作 + 子域划分 | 低 |
| `PCILU` / `PCICC` | 顺序依赖的前代 → Vulkan 上基本无解，只能 level-scheduling 或留在 host | **极高，建议不做** |
| `PCGAMG` | 需要 SpGEMM、转置、聚合 | 极高，别碰 |

**最小可交付子集**（能跑到 `-ksp_type cg -pc_type jacobi` 的三维 Poisson）：`PetscDevice` + `PetscMemory` + `Vec` + `MatAIJ` + `MatMult` + `PetscSF`。这条线做完，你就有了一个**可被引用的成果**。
</details>

<details>
<summary><b>③ 它是最严苛、也最有说服力的 benchmark</b></summary>

PETSc 在 GPU 上的真实瓶颈是 **SpMV + halo exchange + 预条件子**，不是算子本身。这三样正好是 Vulkan 最弱的地方（无 cuSPARSE、无 P2P DMA、无 device-side launch）。

所以只要在**同一台异构节点**上跑这组对比，结论就自动成立：

```
PETSc 3D Poisson (ex / SNES ex), 单节点 8 卡混合:
  A) -vec_type cuda -mat_type aijcusparse        (NVIDIA)
  B) -vec_type hip  -mat_type aijhipsparse       (AMD)
  C) -vec_type vk   -mat_type aijvulkan          (kcompute)   ← 你的东西
对比 KSP 迭代墙钟时间、SpMV 有效带宽占比、以及跨厂商的一致性
```

如果 C 在 AMD 上能落在 B 的 90% 以内、并且在 NV 上不需要改一行代码就跑通，这个故事在论文上是站得住的；**如果 C 只有 B 的 60%，那这个框架的现实价值就基本只剩“教学/可移植性演示”**。先把这条线想清楚，再决定要不要投入。注意这里的判据是**有效带宽占峰值百分比**（$I = F/W$ 决定该内核是带宽受限还是算力受限），不是 FLOPS。
</details>

---

## 3. 组合起来的三条路线（我的排序）

| 路线 | 做法 | 真实工作量 | 我的评价 |
|---|---|---|---|
| **A. kcompute 作为 PETSc 的一个 device backend** | 独立库 `libkcompute` + PETSc 里加 `PETSC_DEVICE_VULKAN`，只覆盖 `Vec`/`MatAIJ`/`PCJACOBI`/`PetscSF` | 6–12 人月（`PetscSF` 占一半） | ⭐⭐⭐⭐ **学术价值最高**，成果可按发表级别评估，且免费获得 Fortran/C/Python 接口和整个应用生态（OpenFOAM、FEniCS、MFEM…） |
| **B. kcompute 只做内核供应商** | 不进 PETSc，只提供 SpMV/stencil/scan 的 Vulkan 内核 + 一个极薄 C API，在应用里替换热点 | 2–4 人月 | ⭐⭐⭐ **性价比最高**。不碰`PetscSF`、不碰`MatSetValues` 语义，能快速拿到性能数据，也留了退路 |
| **C. 自己造完整框架（上一轮那套 L1–L6）** | Kompute 借鉴 + 自己写库 + 自己写 MPI 层 | 3 年+ | ⭐⭐ 除非你有团队和明确产品场景，否则就会陷入“框架写完了但没人用”的境地 |

**推荐：先做 B 拿到数据，如果数据支持，再升级成 A。** 具体第一刀：**三维 Poisson 的 CSR SpMV + halo exchange**，因为：
- 这两个是 PETSc GPU 性能的真正瓶颈，赢了才有意义；
- 它们分别对应 Vulkan 的“能做好”和“最难做好”，一次就能把风险测试完；
- 有现成的 PETSc 基准和正确答案可对照（`src/ksp/ksp/tutorials`、`src/mat/tutorials/ex*`）。

---

## 4. 所以“有用吗”的直接回答

- **Kompute 有用吗？** 有用，但只能当**读物和启动脚手架**：它的层级划分和 CMake/SPIR-V 流水线值得借鉴；其抽象不适合承载 HPC 调度，并且已进入维护模式，将它作为依赖等于把自己的上限与它的抽象上限绑定。
- **PETSc 有用吗？** 非常有用，但它是**评分标准，而非基础组件**。它告诉你 HPC 真实需要什么（`PetscSF`、`MatSetValues` 语义、确定性），也给出了现成的后端扩展模板（照抄 SYCL）；它的 OpenCL/Kokkos/SYCL 历史同时是一个警告：可移植后端在 HPC 栈里很难成为性能主路径。
- **两个合起来够不够？** 不够。缺的仍是那三块实打实的工作：**区间级 hazard 追踪 / persistent-CTA 调度 / 多卡零拷贝通信（dma-buf + `VK_KHR_external_memory_rdma`）**。Kompute 一块都没有，PETSc 只负责评判。

---

需要你定下来的是：**目标是“能发表的成果”还是“能用的工具”？** 前者走 A（我可以把 PETSc SYCL 后端那条线的改动点整理成移植清单，包括 configure 探测、`PetscDeviceType` 分支点、`MatAIJ` ops 结构体的填充表）；后者走 B（我可以直接给你三维 Poisson SpMV + halo 的 Vulkan 实现和基准脚本）。告诉我选哪条，我就开始写具体代码。
