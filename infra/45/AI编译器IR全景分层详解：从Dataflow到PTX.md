# AI编译器IR全景分层详解：从Dataflow到PTX

## 🧑‍💻 User

ai编译器开发（tvm triton）中很多涉及IR的东西
比如ptx LLVM IR
还有dataflow的IR吧，偏底层，类似gluon 比ptx高级一点
能不能全部整理出来啊

---
## 😀 默认助手 (7)

我先核实 Gluon 和 Triton 内部 IR 的最新细节，避免只凭记忆答题。

下面的整理按“**设计轴 × 抽象层级**”来铺开，而不是简单罗列名字；因为 AI 编译器里 IR 混乱的根源在于，大家把三条正交的轴混在一起讨论了。

---

# 一、先把坐标系建起来

任何“AI 编译器 IR”都在回答以下 8 个问题。**IR 之间的差异，90% 来自这 8 个开关的取值不同**，而不是名字不同。

| # | 设计轴 | 取值光谱（高 → 低） |
|---|---|---|
| 1 | **抽象单元** | 图节点 → tile/block → n-D vector → 标量/SIMT lane → 单条指令 |
| 2 | **布局是否显式** | 隐式(编译器选) → 布局作为类型属性 → 线性布局(linear layout，可直接算位) |
| 3 | **内存层次是否显式** | 扁平指针 → address space 标注 → 类型化 Buffer(shared/local/tmem/mbarrier) → TMA descriptor |
| 4 | **并行映射是否显式** | 只有 loop + schedule → grid/CTA/warp/lane 显式 → 只到“并行循环” |
| 5 | **同步/异步是否显式** | 隐式 barrier → async token/mbarrier/phase 显式 → 完全无同步概念 |
| 6 | **语义骨架** | 纯数据流(DAG/SSA value graph) / 结构化控制流(region) / CFG |
| 7 | **类型系统** | 静态 shape → 符号 shape(struct info) → layout 也是类型的一部分 |
| 8 | **可扩展性** | 硬编码 op → dialect/opset 插件化(MLIR) |

**Gluon 之所以“比 PTX 高一点”，本质是它在轴 2/3/4/5 上接近 PTX（全显式），但在轴 1 上仍停留在 tile 级、轴 6 上仍是 SSA 数据流 + 结构化控制流。** 这就是它所在的层。

---

# 二、全景分层表（L0 → L6）

## L0｜图 / 张量级 IR（Dataflow Graph）

| IR | 生态 | 骨架 | 关键特征 |
|---|---|---|---|
| **Relay** | TVM（已冻结） | 纯 DAG，`Expr` + `Call` | 有 `Op` 注册、pattern rewrite、`AlterOpLayout` |
| **Relax** | TVM（现行） | Binding block：`DataflowBlock` + `VarBinding` | **StructInfo** 类型系统支持动态 shape；可嵌入 TIR 作 opaque call |
| **torch.fx / ExportedProgram** | PyTorch | `Graph` + `Node`(target 是 ATen op) | `torch.export` 出 ATen dialect，再做 decompositions |
| **ONNX GraphProto** | 跨框架 | Node/Edge + initializer | 事实上的交换格式，语义偏“规范”不偏“优化” |
| **StableHLO** | JAX / XLA / TF | MLIR，functional | 可移植 opset（MHLO 的稳定子集），有版本兼容承诺 |
| **HLO / MHLO** | XLA 内部 | `HloModule` → MLIR | 带 `async-start/async-done` 的异步数据流，是 XLA 的“真身” |
| **XLA 前端 |`jaxpr`**| JAX | 嵌套 `Jaxpr`(letrec/JaxprEqn) | AD 变换(linearize/transpose)主要在这一层做 |
| **TF GraphDef / TF dialect** | TensorFlow | DAG | 正在被 StableHLO 取代 |
| **Glow Graph / Glow LowLevel** | PyTorch 早期 | DAG + 反向自动生成 | 经典“图IR → 指令级IR”两级设计 |
| **Poplar / PopART Graph** | Graphcore | 计算图 + `Vertex` | vertex 是真正的硬件并行单位 |
| **CANN GE Graph** | 昇腾 | 图 + 算子引擎 | Ascend 的图编译入口 |
| **nn.Module → `torch.compile` Dynamo** | PyTorch | 字节码 → FX Graph | 注意 Dynamo 输出仍是 dataflow graph |

> 这一层的共同点：**几乎都是纯数据流（value graph），没有循环、没有地址、没有时间**。也是“dataflow IR”最常见的含义。

## L1｜循环嵌套 / 调度 IR（Loop-nest / Polyhedral / Schedule）

| IR | 生态 | 说明 |
|---|---|---|
| **TensorIR (TIR)** | TVM | `PrimFunc` + `Block` + `iter_var`；`Schedule` 有 split/fuse/reorder/bind/cache_read/pipeline 等原语；本质是带 affine 约束的 loop nest |
| **TVMScript** | TVM | TIR 的 Python 嵌入文本形式（**它才是你读写的对象**） |
| **HalideIR** | TVM 前身 / Halide | TVM 继承的调度模型来源 |
| **MetaSchedule / DLight** | TVM | 调度*空间*描述（不是 IR，但定义了 IR 可变换的边界） |
| **nvFuser `Fusion`** | NVIDIA | `TensorView` + `IterDomain` + Split/Merge/Reorder/ScheduleOp，与 TVM TIR 高度同构；下面还有 `kir::Kernel`(Kernel IR) |
| **Inductor `LoopLevelIR`** | PyTorch | 纯 Python 对象图：`Buffer`/`FixedLayout`/`FlexibleLayout`/`Pointwise`/`Reduction`/`Scan`/`TemplateBuffer` —— 一个“没写成文本的 TIR” |
| **linalg dialect** | MLIR | `linalg.generic` + indexing maps + iterator types，**最标准的“结构化 loop nest”IR** |
| **affine dialect** | MLIR | 多面体（polyhedral）风格，`affine.for`/`affine.if`/`affine.apply` |
| **Tensor Comprehensions** | 历史 | 多面体 + 自动调度的典型失败案例 |
| **TileLang** | 微软/字节 | `T.Kernel`/`T.alloc_shared`/`T.copy`/`T.Pipelined`，**直接 lowering 到 TVM TIR → CUDA/PTX**，是 Gluon 在 tile 层的直接参照物 |
| **Taichi `CHI` IR** | Taichi | offloaded task + **显式 `global tmp` / `shared` / `local` 内存域**，太鼓最早把“布局显式”做成 IR 层概念 |
| **Halide** | Halide | `Func`/`Stage`/`Var`，schedule 与 algorithm 分离的鼻祖 |

> 这一层的核心概念：**algorithm 与 schedule 分离**；`linalg`/`tir.Block` 里都有 “iteration space + indexing map” 这个对偶结构。

## L2｜Tile 级 IR（隐式布局，编译器管一切）

| IR | 生态 | 骨架 | 布局 | 说明 |
|---|---|---|---|---|
| **`tt` (TritonIR)** | Triton | MLIR，SSA + `scf` | **隐式** | op 直接吃 tensor（tile），如 `tt.load`/`tt.dot`/`tt.reduce`；无 shared memory 概念 |
| **`stablehlo`/`linalg` on tensor** | 通用 | 同上 | 隐式 | 与 TTGIR 之间是“tile 化 + 布局分配”这道坎 |

Triton 的 tile 语义细节：`tt.load` 返回的是 “block 内 N 个元素分布到 CTA 全线程”的逻辑张量；`tt.dot` 有 `allow_tf32`、`max_num_imprecise_acc`；`tt.reduce` 带 `region`。

## L3｜布局显式化 / 硬件映射 IR ←—— **Gluon 就住这一层**

这是你说的“偏底层、比 PTX 高级一点”的那条带子。它下面还有 PTX，上面是 tile 级，中间是**布局 + 内存 + 同步显式化**。

| IR | 生态 | 布局 | 内存 | 同步/异步 | 备注 |
|---|---|---|---|---|---|
| **`ttg` (TritonGPU)** | Triton | **显式**：`#ttg.blocked`、`#ttg.nvidia_mma`、`#ttg.dot_op`、`#ttg.memdesc`、`#ttg.shared` | 显式 shared/local | `ttg.async_wait`、`ttg.barrier`、流水 `num_stages` | `tt` 经过 Layout Propagation + ConvertLayout 得到 |
| **`ttng`** | Triton (NVIDIA) | 沿用 ttg | 显式 + **TMA** | **显式 mbarrier**：`ttng.init_barrier`/`wait_barrier`/`arrive_barrier`、`ttng.async_tma_copy_global_to_local` | Hopper/Blackwell 特性 |
| **`amdg`** | Triton (AMD) | `#amdg.*` | 显式 LDS | `amdg.*` | CDNA/RDNA |
| **`nvws`** | Triton | — | — | **Warp Specialization** 显式化 | 新的 ws=warp-specialized 路径 |
| **`gluon`** | Triton (`triton.experimental.gluon`) | **完全显式、用户手写** | `gl.allocate_shared_memory`、`gl.allocate_mbarrier` | `gl.async_copy_global_to_shared`、`mbarrier_wait/arrive` | **不是文本 IR，是 Python DSL 编译进 MLIR `gluon` dialect** |
| **`nvg` (NVGPU)** | MLIR 官方 | — | — | `nvgpu.mbarrier.*`、`nvgpu.tma.*`、`nvgpu.warpgroup.*` | 比 ttng 更“中立”的 NVIDIA 方言 |
| **`gpu`** | MLIR 官方 | implicit/explicit | `gpu.alloc`（workgroup/private） | `gpu.barrier` | `gpu.thread_id`、`gpu.warp_execute_on_lane_0` |
| **`async`** | MLIR 官方 | — | — | **第一类 async token** | `async.execute`/`async.await`，是“异步数据流”的通用抽象 |
| **`xegpu` / `vec`** | Intel | 显式 | 显式 | — | Xe GPU |
| **`arm_sme` / `arm_sve`** | ARM | 显式（ZA tile / SVE 谓词） | 显式 | — | CPU 侧的同类设计 |
| **`amx`** | Intel CPU | 显式 tile | 显式 | — | AMX tile 寄存器 |
| **CUTLASS / CuTe**（C++ 模板，非 IR） | NVIDIA | **纯编译期线性布局 `Layout = Shape × Stride`** | 显式 | 显式 | **Gluon 的线性布局直接对标 CuTe 的 `Layout` / `Swizzle` / `ComposedLayout`** |
| **CuTe DSL (Python)** | NVIDIA | 同上 | 同上 | 同上 | CuTe 的 Python 化，和 Gluon 是竞争关系 |
| **ThunderKittens** | Stanford | 显式 tile 类型 | 显式 | 显式 | C++ 模板 DSL |
| **Mojo `layout` / `tile`** | Modular | 显式 | 显式 | 显式 | 走 MLIR |
| **Mojo/Mosaic** | Modular | — | — | — | `mosaic` 是 GPU kernel 层 |

### Gluon 到底显式了什么（原文口径 + 我的归纳）

Gluon 是 “Triton 的 lower-level GPU programming model”，**exposes layouts, shared memory, warp specialization, and target-specific features directly**。它和 Triton 共享 frontend / JIT，但把编译器替你做的四件事交还给你：

1. **布局**：`gl.BlockedLayout` / `gl.SliceLayout` / `gl.NVMMASharedLayout` / `gl.DotOperandLayout` / **`gl.DistributedLinearLayout`**（线性布局，可直接按位算映射）
2. **共享内存**：`gl.allocate_shared_memory(dtype, shape, layout)` —— 显式分配、显式给 layout
3. **异步与屏障**：`gl.allocate_mbarrier`、`gl.async_copy_global_to_shared`、`mbarrier_wait/arrive`、TMA (`TensorDescriptor`)
4. **warp specialization**：显式把不同 warp 写进不同分支（producer/consumer），而不是 `num_stages` 让编译器展开

所以它的“高度”是：**tile 级数据模型 + 寄存器/共享内存级显式性**。再往下就是 PTX 的 `mma.sync` / `cp.async.bulk.tensor` / `mbarrier.try_wait` 了。

```python
# Gluon：布局是 constexpr，内存/屏障是显式对象
@gluon.jit
def kernel(a_ptr, b_ptr, c_ptr, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    smem_l:  gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])

    a_smem = gl.allocate_shared_memory(gl.float16, [BM, BK], smem_l)
    bar    = gl.allocate_mbarrier()
    gl.async_copy_global_to_shared(a_smem, a_desc, bar)   # TMA
    gl.mbarrier_wait(bar, phase=0)

    acc = gl.convert_layout(gl.dot(a_smem, b_smem, acc), mma_layout)
```

**层级对照（关键一张表）：**

| | 用户指定布局 | 用户管 shared | 用户管 barrier | 用户管 warp 分工 | 数据模型 | 生成物 |
|---|---|---|---|---|---|---|
| Triton (`tt`) | ✗ | ✗ | ✗ | ✗（`num_stages`） | tile | PTX |
| TTGIR (`ttg`) | 编译器填 | 编译器填 | 编译器填 | ✗ | tile | PTX |
| **Gluon** | ✓ | ✓ | ✓ | ✓ | tile | PTX |
| CuTe / CUTLASS | ✓ | ✓ | ✓ | ✓ | tile + 线性布局 | PTX（或 C++） |
| CUDA C++ | 手动下标 | ✓ | ✓ | ✓ | 标量/向量 | PTX |
| PTX | 手动下标 | ✓ | ✓ | ✓ | 标量 + `mma`/`cp.async` 指令 | SASS |

## L4｜SIMT 标量/向量 IR

| IR | 角色 | 要点 |
|---|---|---|
| **LLVM IR** | 万能后端 | `llvm.nvvm.*` / `llvm.amdgcn.*` intrinsics；`addrspace(3)`=shared，`(1)`=global |
| **NVVM IR** | 官方子集 | LLVM IR + NVVM annotations（`.nvvm.annotations` 标 kernel、`maxntidx` 等）+ 受限 intrinsic；libNVVM 消费它 |
| **MLIR `llvm` dialect** | 交接点 | `llvm.mlir` 是 MLIR ↔ LLVM 的桥；`llvm.func`/`llvm.getelementptr` |
| **MLIR `nvvm` / `rocdl`** | 厂商 intrinsic 方言 | `nvvm.shfl.sync`、`nvvm.mma.sync`、`rocdl.mfma` |
| **SPIR-V** | **“可移植的 PTX”** | Khronos；结构化控制流 + 显式内存模型(SSBO/Workgroup/Private)；OpenCL/Vulkan/WebGPU/oneAPI 后端；`mlir.spirv` |
| **AIR (Apple IR)** | Apple GPU | 基于 LLVM 的中间表示 |
| **`emitc`** | MLIR | 降到 C/C++ 源码（“IR 直接生成代码”，TVM `CodeGenC` 同理） |

```llvm
; NVVM/LLVM IR 片段
%tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%v   = call <4 x float> @llvm.nvvm.ldg.global.v4.f32.p0(ptr %p)
%r   = call {<2 x half>,...} @llvm.nvvm.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32(...)
```

## L5｜虚拟 ISA

| ISA | 厂商 | 要点 |
|---|---|---|
| **PTX** | NVIDIA | 虚拟 ISA，**不是机器码**；`ld.global.nc.v4` / `mma.sync` / `cp.async.bulk.tensor` / `mbarrier.try_wait` / `bar.sync`；由 `ptxas` 编到 SASS |
| **AMD GCN/RDNA ISA** | AMD | 经 LLVM AMDGPU backend 出 `amdgcn`，再由 `ld.lld` 出 `hsaco` |
| **SPIR-V** | Khronos | 既是 IR 又是“虚拟 ISA”，定位与 PTX 对偶 |
| **BANG IR** | 寒武纪 | NPU 侧类似物 |

```ptx
ld.global.nc.v4.b32     {%r0,%r1,%r2,%r3}, [%rd1];
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
                        [%s0], [%rd2, {%r4,%r5}], [%mbar];
mbarrier.try_wait.parity.shared::cta.b64 %p, [%mbar], %phase;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
                        {%f0,%f1,%f2,%f3}, {%r6,%r7,%r8,%r9}, {%r10,%r11}, {%f0,%f1,%f2,%f3};
```

## L6｜物理 ISA / 二进制

| 产物 | 说明 |
|---|---|
| **SASS** | NVIDIA 真机器码，`cuobjdump -sass` / `nvdisasm`。不可手写，只能“看着调” |
| **cubin / fatbin** | 打包产物，含多 arch |
| **HSACO** | AMD |
| **`.spv`** | SPIR-V 二进制 |
| 各家 NPU 的 `binary blob` | 通常不可读 |

---

# 三、数据流 IR 专题（你特别问的）

“dataflow” 在 AI 编译器里有 **四种完全不同的含义**，混用是最大的困惑来源：

<details>
<summary><b>① 张量数据流图（Value DAG）—— 最高层</b></summary>

节点是算子，边是张量，无控制流、无地址、无时间。代表：`Relay`、`XLA HLO`、`StableHLO`、`torch.fx.Graph`、`ONNX`、`TF GraphDef`、`Glow Graph`、`Poplar Graph`、`jaxpr`。

**关键性质**：可以做全局代数重写（`a*b + a*c → a*(b+c)`）、常量折叠、CSE、算子融合；但**无法表达 tile 化、流水、寄存器压力**。

</details>

<details>
<summary><b>② 局部数据流块（Dataflow Region）—— 中层的“纯函数区域”</b></summary>

TVM Relax 的 `R.dataflow()` 是最典型的：在一个 binding block 内，所有中间 `DataflowVar` 保证**无副作用、可并行重排、可融合**；出了 block 才是 `Var`（可跨 block 复用）。

```python
@R.function
def main(x: R.Tensor((1024, 1024), "float32")):
    with R.dataflow():
        lv = R.matmul(x, w1)
        gv = R.nn.gelu(lv)
        R.output(gv)
    return gv
```

MLIR 里对应的是 `linalg` on tensor + `bufferization` 的 one-shot 分析：tensor 语义就是数据流，`memref` 语义才有别名。

</details>

<details>
<summary><b>③ 硬件级数据流（Hardware Dataflow IR）—— 真正的“偏底层 dataflow”</b></summary>

| IR | 出处 | 特点 |
|---|---|---|
| **Calyx** | Cornell，与 TVM/VTA 集成 | **真正的硬件数据流 IR**：`cells` + `groups` + `@static`/`@go` 端口；支持 `while`/`par`；可出 Verilog |
| **DaCe SDFG** | ETH Zurich | Stateful DataFlow multigraph：node=计算，edge=memlet(带索引映射)；有 State（控制流）+ Interstate edge |
| **Glow LowLevel IR** | PyTorch 早期 | DAG + 指令（`ConvolutionInst` 等），显式 `WeightVar`/`Placeholder` |
| **Cerebras CSL / Groq TSP** | 专用 | 数据流 + 片上网络显式 |
| **`ttng` mbarrier 图** | Triton | 异步 copy → barrier → wait，本质是**token 化的片上数据流** |
| **MLIR `async` + `gpu`** | MLIR | `async.token` / `async.value` 显式建模异步依赖，可跨 device |

Calyx 片段（体现“端口 + 组”的数据流语义）：

```calyx
cell mbarrier
group load {
  tma[...] -> smem;      // 生产
  mbarrier.arrive;
}
group consume { ... }
```

</details>

<details>
<summary><b>④ Sea-of-Nodes / 值图（参照系，不是 DL 专用）</b></summary>

HotSpot C2 与 Graal 的 sea-of-nodes：节点是纯值，边是 use-def，**控制流也是节点**（`Start`/`If`/`Region`/`Loop`）。它证明了一件事：**“数据流 + 显式控制节点”可以统一成一张图**。Gluon/CuTe 在思想上就是 GPU 版的 sea-of-nodes（layout 是边上的属性，mbarrier 是控制边）。

</details>

**一句话区分**：① 是**图**，② 是**图里的一个纯函数区域**，③ 是**带硬件端口的图**，④ 是**图包含控制流本身**。

---

# 四、四条主流流水线（把上面的层串起来）

## TVM

```
PyTorch/ONNX/Relay
   └─► Relax (L0, StructInfo, DataflowBlock)
        ├─► BYOC: DNNL / TensorRT / CUTLASS / ACL / cuDNN / BNNS / CoreML
        └─► Legalize → FuseOps → DecomposeOps → ...
             └─► TensorIR / PrimFunc (L1)  ──► MetaSchedule / DLight 调度
                  └─► TIR passes (InjectSoftwarePipeline, StorageRewrite,
                      LowerCrossThreadReduction, HoistIfThenElse, ThreadSync, ...)
                       └─► CodeGen{CUDA, ROCm, LLVM/CPU, SPIRV, Metal,
                                  Vulkan, OpenCL, WebGPU, Hexagon} (L4)
                            └─► PTX / CUDA C / .spv / LLVM IR → 后端
```

- TVM 的“布局”概念 = `IndexMap` + `Buffer` 的 `strides` + `T.Buffer` 的 `layout` 参数（`T.Fragment` for MMA 布局，最近版本才有）
- TVM 的“异步” = `T.Pipelined(num_stages=k)` → `InjectSoftwarePipeline` 被动生成
- **对比 Gluon：TVM 把 L2/L3 的显式性全藏在 `Schedule` 的 primitive 后面**

## Triton

```
@triton.jit (Python AST)
   └─► tt (TritonIR)          ← tile 级, 隐式布局
        └─► ttg (TritonGPUIR)  ← 布局显式化 (LayoutPropagation/ConvertLayout)
             ├─► ttng (NVIDIA: TMA/mbarrier)  /  amdg (AMD)  /  nvws (warp spec)
             └─► (Gluon 前端直接进 gluon dialect，跳过自动布局推断)
                  └─► LLVM dialect → LLVM IR (NVVM) → PTX → cubin → SASS
```

- 中间还有 `TritonGPU` 的 `RemoveLayoutConversions`、`AccelerateMatmul`、`Pipeline`(软件流水)、`TMA lowering`、`WarpSpecialization`
- Gluon 的价值：**它绕过 `AccelerateMatmul` / `Pipeline` / 自动 `ConvertLayout` 这些“猜”的 pass**，让你直接写目标形态

## PyTorch (`torch.compile`)

```
bytecode → Dynamo → FX Graph (L0, dataflow)
   └─► AOTAutograd → 前向+反向 joint graph
        └─► Inductor: Scheduler (fuse/ reduction/ template)
             └─► LoopLevelIR (L1, Python 对象) ──► Triton / C++ source
                  └─► (Triton) TritonIR → TTGIR → ... → PTX
```

- `LoopLevelIR` 的 `FixedLayout` / `FlexibleLayout` / `ComputedBuffer` / `TemplateBuffer` 是“布局显式化”的另一种表达（用 Python 类而不是 layout 类型）
- `TemplateBuffer`（Triton/CUTLASS 模板）= 把 L3 层外包给外部 kernel generator

## JAX / XLA / IREE

```
jaxpr (L0) ──► StableHLO (L0, MLIR)
   ├─► XLA:GPU: HLO → GPU HLO → Triton emitter → tt → ttg → ...
   └─► IREE: StableHLO/TOsa
        → flow (dispatch 形成, 数据流分区)
        → stream (跨 device 异步调度, async token)
        → hal (hardware abstraction layer: buffer/executable)
        → VM bytecode  /  LLVM / SPIRV / CUDA
```

IREE 的 `flow → stream → hal` 三层是**“数据流分派 + 异步调度 + 硬件抽象”**的教科书式分层，值得单独读。

---

# 五、其他厂商一览（补全“全部”）

| 厂商/项目 | 图 IR | 中层 IR | 底层 |
|---|---|---|---|
| **NVIDIA** | — | nvFuser `Fusion` + `kir`、CuTe/CuTeDSL、TTGIR/Gluon | NVVM → PTX → SASS |
| **AMD** | — | `amdg` dialect、MLIR `rocdl` | LLVM AMDGPU → amdgcn → hsaco |
| **Intel** | — | MLIR `xegpu`/`vec`/`gpu`、oneDNN graph | SPIR-V / Level Zero |
| **Intel CPU** | — | MLIR `amx`/`x86vector` | LLVM X86 |
| **ARM** | — | MLIR `arm_sme`/`arm_sve` | LLVM AArch64 |
| **Apple** | — | MLIR `gpu` | AIR → Metal |
| **昇腾** | GE graph / MindIR | TIK / AscendC / `Ascend` dialect | 达芬奇 ISA |
| **寒武纪** | MagicMind graph | BANG IR (`BLOCK`/`CLUSTER`/`CORE`) | MLU ISA |
| **Graphcore** | PopART graph | Poplar `Vertex`/`ComputeSet` | IPU |
| **Tenstorrent** | — | TT-Metalium (`CircularBuffer`, `Kernel`) | RISC-V |
| **Cerebras** | — | CSL (dataflow) | WSE |
| **Taichi** | — | `CHI` IR（显式 `shared`/`local` 域） | LLVM / SPIR-V |
| **Modular** | — | Mojo/Mosaic MLIR | PTX→SASS |
| **华为/字节 生态** | — | TileLang → TVM TIR | CUDA/PTX |

---

# 六、动手：怎么把这些 IR 打出来看

<details>
<summary><b>Triton（能一次看到 5 层）</b></summary>

```python
k = my_kernel[grid](...)
print(k.asm.keys())        # ['ttir', 'ttgir', 'llir', 'ptx', 'cubin']
print(k.asm['ttir'])       # L2
print(k.asm['ttgir'])      # L3（布局长什么样，全在这）
print(k.asm['llir'])       # L4 (NVVM)
print(k.asm['ptx'])        # L5
```

```bash
# 落盘全部阶段 + 每个 pass 前后 diff
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump python t.py
# MLIR pass pipeline 级别的 dump
MLIR_ENABLE_DUMP=1 python t.py
# 只看指定 pass
MLIR_ENABLE_DUMP=1 TRITON_DISABLE_LINE_INFO=1 python t.py 2>&1 | grep -A50 "remove-layout-conversions"
# SASS
cuobjdump -sass ./dump/*.cubin | less
```

</details>

<details>
<summary><b>TVM</b></summary>

```python
print(mod.script())                       # Relax / TIR 源码
s = tvm.tir.Schedule(tir_mod)
print(s.mod.script())                     # 调度后的 TIR
lib = tvm.build(s.mod, target="cuda")
print(lib.get_source("ptx"))              # PTX
print(lib.get_source("ll"))               # LLVM IR（target=c 时）
print(lib.get_source("cu"))               # CUDA C

# 逐 pass 观察
with tvm.transform.PassContext(
        opt_level=3,
        config={"tir.print_ir_before": ["LowerCrossThreadReduction"],
                "tir.print_ir_after":  ["StorageRewrite"]}):
    tvm.lower(mod, ...)
```

</details>

<details>
<summary><b>Inductor / nvFuser / TileLang</b></summary>

```bash
TORCH_LOGS=output_code python m.py      # 看 Triton / C++ 源码
TORCH_COMPILE_DEBUG=1 python m.py        # 落盘 fx_graph / loop_level_ir
```

```python
# TileLang
print(tilelang.lower(func).kernel_source)   # CUDA / PTX
# nvFuser
fd = FusionDefinition(); ...
print(fd)                                  # Fusion IR
```

</details>

---

# 七、选型 / 定位速查

| 你的需求 | 该去的层 | 具体东西 |
|---|---|---|
| 做算子融合、代数化简、常量折叠 | L0 | Relax、StableHLO、HLO、FX |
| 做自动调度 / autotuning | L1 | TensorIR + MetaSchedule、nvFuser、Inductor LoopLevelIR |
| 写可移植的高性能 kernel（少管硬件） | L2 | Triton `tt`、TileLang、linalg |
| 压榨最后一档性能、要控 layout/流水/warp 分工 | **L3** | **Gluon**、CuTe DSL、TTGIR、ttng、`nvgpu` |
| 做后端 / 新硬件支持 | L4–L5 | MLIR `llvm`/`nvvm`/`spirv`、LLVM、SPIR-V、PTX |
| 做 NPU / 加速器编译器 | L1 + ③类 dataflow | Calyx、DaCe SDFG、`linalg`+`affine` |
| 做分布式 / 跨设备调度 | IREE `flow/stream`、Relax + `async` | async token 数据流 |

---

## 一句话总结

> **L0 图**管“算什么”，**L1 循环嵌套**管“按什么顺序算”，**L2 tile**管“按什么粒度算”，**L3（Gluon 这一层）**管“算的时候数据躺在哪个寄存器/共享内存/屏障后面”，**L4 LLVM IR** 管“变成标量指令”，**L5 PTX** 管“变成虚拟机器码”，**L6 SASS** 才是硬件真话。
>
> `dataflow` 这条线横切所有层：L0 是 value DAG，L1/L2 是 region 内的纯函数块，L3 是 mbarrier/token 化的片上异步数据流，硬件加速器上则是 Calyx / DaCe 那种带端口的真 dataflow 图。

需要我把某一条流水线（比如 Triton `tt → ttg → ttng → PTX` 每个 pass 到底改了什么，或者 TVM TIR 到 PTX 的完整 pass 清单）展开成逐 pass 的明细吗？
