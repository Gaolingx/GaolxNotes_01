# x86 SIMD 归约缺失原因

### 🧑‍💻 User

为什么 x86 架构没有从 SIMD 向量到标量的水平归约指令？

---

### 🤖 Assistant

简短结论先说：**x86 早期没有专门的“水平归约（horizontal reduction）到标量”的单指令，是历史兼容性 + 微架构代价 + 指令编码压力共同导致的设计取舍**，而不是做不到。

下面说核心原因。

---

## 1. SIMD 在 x86 里的“原始定位”是 **数据并行**，不是“在寄存器内做树形归约”

x86 的 SSE/AVX 设计初衷是：

* 同一条指令 → 处理 **多组独立标量**
* 每个 lane 本质上是“独立标量 ALU”

也就是说逻辑模型是：

```
lane0 op lane0
lane1 op lane1
lane2 op lane2
...
```

而“水平归约”需要的是：

```
((((lane0 + lane1) + lane2) + lane3) + ...)
```

这是**跨 lane 的数据依赖链**，天然违背 SIMD 的设计哲学。

很多 RISC 架构（如部分 DSP、后期 ARM SVE）一开始就是“向量机”思路，所以支持这种跨 lane 归约；而 x86 是从**标量 ISA → SIMD 补丁式扩展**演化来的。

---

## 2. x86 并不是“不能做”，而是历史上**用 shuffle + add 组合就够了**

典型 float4 求和流程是：

```asm
vaddps   ymm0, ymm0, ymm1
vshufps  ymm1, ymm0, ymm0, imm
vaddps   ymm0, ymm0, ymm1
...
```

在 Intel/AMD 的设计者看来：

* 这种模式 **已经足够快**
* 可以由编译器自动合成
* 没有必要专门为“reduce-to-scalar”增加新 opcode

增加一条新指令不是“免费”的：

* 要占用 opcode 空间
* 要改 decode
* 要加特殊执行路径
* 还会影响调度器 & rename 逻辑

而当年使用场景并不多（科学计算才大量需要）。

---

## 3. 微架构层面：**水平归约对硬件非常不友好**

从硬件角度看，一个真正的：

```
horizontal_add ymm → scalar
```

意味着：

* 必须访问 **整个 vector 寄存器的跨 lane 结构**
* 引入 **树形加法器**
* 延迟不可隐藏（完全串行相关）

这会带来问题：

| 问题   | 原因               |
| ---- | ---------------- |
| 高延迟  | 每一层依赖上一层         |
| 不好流水 | 减少 ILP（指令级并行）    |
| 面积大  | 要加额外归约网络         |
| 使用率低 | 大多数 SIMD 代码是吞吐导向 |

对于 Intel/AMD 的目标市场（通用 CPU）来说，不划算。

---

## 4. 注意：其实 x86 **后来还是“妥协性地”加了**

虽然没有“直接归约成标量”的经典指令，但后来还是增加了“半吊子支持”：

### ✅ SSE3：`HADDPS / HADDPS`

例如：

```asm
haddps xmm0, xmm0
```

但问题是：

* 只做**相邻元素**相加
* 仍然需要 **多条指令 + shuffle**
* 延迟并不好

### ✅ AVX512 才真正有接近归约语义的指令

例如：

```asm
VREDUCEPS
```

和 masked reduction 相关指令。

但这已经是 20 年后的事，而且与 HPC 场景明显相关。

---

## 5. x86 的真实设计取向：**宁愿你用 FMA + 向量并行，也不鼓励频繁 collapse 到 scalar**

x86 的优化路线一直是：

* 鼓励 **宽 SIMD 持续并行**
* 尽量 **晚 reduce（late reduction）**
* 累加留在向量中
* 最后只 reduce 一次

这适合：

* 多媒体
* 图形
* AI 前处理
* 科学计算

而不是频繁 scalar 化。

---

## 6. 为什么 GPU 反而有“横向归约”？

因为 GPU 天然是：

* SIMT 模型
* 线程间通信本来就设计好了（warp shuffle / reduction）
* reduction 是 GPU 常见模式（sum/max/min/scan）

而 x86 的 SIMD lane 是“私有 ALU”，不是“线程”。

---

## 最核心的一句话总结

x86 没有早期提供“向量→标量水平归约单指令”，不是技术上做不到，而是因为：

> SIMD 在 x86 的定位是“批量标量并行”，而不是“寄存器内向量机”，
> 而真正的水平归约在硬件上复杂、收益小、编码成本高，所以长期被视为“用 shuffle 组合就好”的低优先级功能。

---

如果你感兴趣，我可以给你画一个 **“如果 x86 真有单指令 reduce，它在硬件里长什么样”** 的微架构示意。


---

### 🧑‍💻 User

在 avx512 架构中，_mm512_reduce_add_ps这是一个序列指令，这意味着它会被翻译成多个微操作？

---

### 🤖 Assistant

是的，**`_mm512_reduce_add_ps` 不是一条 AVX-512 的真实硬件指令**，它会被编译成**多条指令（多个 uop）组成的指令序列**，本质上是一个 *software sequence / pseudo-instruction*。

下面给你讲清楚它在 AVX-512 里的真实情况。

---

## 1. `_mm512_reduce_add_ps` 本质是什么？

它是 **编译器内建函数（builtin）**，不是硬件指令：

* 来自 headers，如 `immintrin.h`
* 表现形式像“一个指令”
* 实际是 **宏 / 内联函数 + 编译器展开**

IQ：没有任何名为 `VREDUCEADDPS` 之类的指令。

也就是说：

```c
float s = _mm512_reduce_add_ps(v);
```

**永远不会**编译成单条 AVX-512 指令。

---

## 2. 它会被展开成什么样的指令序列？

典型 GCC / Clang 展开成类似这种：

### 对 16×float（zmm）

```
; v = zmm0

vextractf64x4 ymm1, zmm0, 1        ; upper 256
vaddps          ymm0, ymm0, ymm1   ; add high + low

vextractf128    xmm1, ymm0, 1
vaddps          xmm0, xmm0, xmm1

vshufps         xmm1, xmm0, xmm0, 0b01001110
vaddps          xmm0, xmm0, xmm1

vshufps         xmm1, xmm0, xmm0, 0b10110001
vaddps          xmm0, xmm0, xmm1

; result in xmm0[0]
```

这是一个**标准的树形 reduction**：

512 → 256 → 128 → 64 → 32 → scalar

每一级都是：

* `extract` + `add`
* 或 `shuffle` + `add`

---

## 3. 这是几个 uop？

这要看你关心什么粒度：

### 指令条数（典型）

大概：

| 用途          | 指令数 |
| ----------- | --- |
| extract 高半  | 1   |
| add 512→256 | 1   |
| extract 128 | 1   |
| add 256→128 | 1   |
| 两次 shuffle  | 2   |
| 两次 add      | 2   |

**约 8～10 条指令**

### uop 数量

按 Intel Skylake-X / Ice Lake-SP / Sapphire Rapids 级别估算：

| 指令          | 典型 uop  |
| ----------- | ------- |
| `vextract*` | 1–2 uop |
| `vaddps`    | 1 uop   |
| `vshufps`   | 1 uop   |

**总计大约是：8–12 uop**

而且关键是：**这是一个强依赖链（serial dependency chain）**

所以：

* 吞吐量几乎没意义
* 延迟才是主瓶颈

延迟大概在 **10–20 cycle** 范围（取决于具体微架构）。

---

## 4. 它是不是“序列指令”？

从微架构角度：**是的，它是“序列化特征非常强的指令串”**：

特点：

✅ 每一步依赖上一步
✅ 无法在这个 reduction 内部并行
✅ OoO 几乎帮不上忙

所以你可以把 `_mm512_reduce_add_ps` 当作：

> 一个由编译器自动生成的“微程序式”的 reduction 序列，而不是硬件原生指令。

---

## 5. 为什么 Intel 没提供真正的单指令 `reduce`？

原因其实跟你前一个问题强相关：

1. 归约是 **延迟型操作**，对 CPU 不友好
2. 额外硬件代价高
3. server / HPC 才需要大量 reduction
4. 绝大部分代码是吞吐导向而不是 collapse

AVX-512 选择的是：

> 给你宽寄存器，但最后一步归约交给软件 pattern。

---

## 6. 什么时候会更糟？

在以下情况 reduction 代价会特别明显：

* 很小的 kernel（归约占比极高）
* 在 tight loop 里频繁 scalar 化
* 无法 unroll / multi-accumulator

---

## 简洁结论

直接回答你问题：

✅ `_mm512_reduce_add_ps` **不是单指令**
✅ 会展开为 **多条指令 → 多个 uop**
✅ 属于 **强序列依赖链的 reduction 指令序列**
✅ latency 远比 throughput 更关键

一句总结：

> 这不是一条“慢指令”，而是一段“慢链条”。

---

如果你愿意的话，我可以给你对比一下：

* GCC vs Clang 的具体 asm 展开差异
* 或者按不同微架构估算 latency/uop 数量


---

### 🧑‍💻 User

如果我们以 NT 的形式（这是 pytorch 线性定义中的默认形式）执行 gemm：{M, K} * {N, K}，最终将对每 m 行和 n 列进行点积运算，最终我们将得到一个水平归约，写入 C 将是标量。有没有更好的办法（不需要水平归约）？

---

### 🤖 Assistant

结论先给你：**在 `C = A(M×K) * Bᵀ(N×K)`（PyTorch 默认的 `NT` layout）下，数学上不可能避免“点积＝水平归约”这件事；但在实现层面可以把“显式的水平归约指令”完全藏起来，让你几乎感觉不到它的存在**。

下面分层讲。

---

## 1. 数学上：你不可能“消灭归约”，只能“改变归约发生的位置”

你这个式子：

[
C_{m,n} = \sum_{k=0}^{K-1} A_{m,k} \cdot B_{n,k}
]

这个求和本质上就是：

* 乘法 → **元素级并行**
* 求和 → **跨 K 维的归约**

所以：

✅ **无法从数学上消灭 reduction**
✅ 只能把它变成更“硬件友好”的形式

---

## 2. 真正的优化目标：避免“横向归约指令”，而不是避免“归约行为”

你关心的其实是这个问题：

> 如何不让 CPU 执行 `vshufps + vaddps` 这种“lane 内树形归约”？

答案是：**让 reduction 沿着“时间维度”发生，而不是“lane 维度”发生。**

也就是说：

### ❌ 低效方式（你不想要的）

```
vector of K lanes
   ↓  （一次性）
horizontal reduce
   ↓
scalar
```

### ✅ 高效方式（BLAS/GEMM 实际做的）

```
for k_tiles:
    acc_vector += A_vec * B_vec
; 很多次 FMA 累积
; 完全不做水平归约
; 最后只需要 lane = 1 的结果
```

这里的关键是：

> 不对一个“宽寄存器”的 lane 做归约，而是让每个 lane 自己沿时间轴积累。

---

## 3. GEMM 内核是怎么“绕开 horizontal reduce”的？

高性能 GEMM 从来不会这样干：

```text
load A row as vector along K
load B col as vector along K
mul
horizontal add   ← 这是你担心的
```

它们会这样干：

### 向量维度 = K 维度

把 **K 维映射到 SIMD lane**：

```text
zmm0 = [a0 a1 a2 ... a15]
zmm1 = [b0 b1 b2 ... b15]

acc = 0
repeat:
    acc += zmm0 * zmm1   ; vfmadd231ps
    下一块 K
```

**注意：**

* 每个 lane 只负责一个固定 k 偏移
* 完全没有 lane 间通信
* FMA 在“时间维”完成累加，而不是在“空间（lane）”归约

直到最后：

```
acc = [c_partial0 c_partial1 ... c_partial15]
```

这时的每个 lane 对应的是 **不同的 (m, n) 位置**，而不是同一个 dot product。

也就是说：

> 一个 SIMD 寄存器同时计算多个 C(m,n)，而不是一个 C(m,n)。

---

## 4. 这就是为什么 GEMM 要用“block + register tiling”

核心思想：

| 维度          | 作用                 |
| ----------- | ------------------ |
| SIMD lane 维 | 映射到 **不同 n 或不同 m** |
| K 维         | 映射为 **时间上的循环累加**   |

结果：

✅ 完全避免 horizontal reduction
✅ 只用 FMA
✅ 没有 shuffle

---

## 5. 那为什么你会“看到” horizontal reduction 出现？

有两种情况会强制出现：

### ① 你把 K 映射成了 lane

例如你自己写了：

```c
a = _mm512_loadu_ps(&A[m, k])
b = _mm512_loadu_ps(&B[n, k])
dot = _mm512_reduce_add_ps(a * b)
```

这一定会触发横向归约。

### ② 你在做的是 GEMV / very skinny GEMM

当：

* N = 1 或 very small
* M = 1 或 very small

lane 没地方放“多个 C 元素”，就会退化成 **dot product kernel**，这时 horizontal reduction 几乎不可避免（但仍可以用 tricks 缓解）。

---

## 6. 实际工业级方案（MKL / oneDNN / cuBLAS / CUTLASS）是怎么做的？

它们都会：

1. block K
2. block M and N
3. 一个 SIMD 向量同时负责多个 `C(m,n)`
4. accumulation 全部用 FMA
5. reduction 只发生在：

   * 非常小的尾部
   * 或 epilogue 阶段（标量化）

**整个主循环完全没有 horizontal reduce。**

---

## 7. 直观图

### ❌ 糟糕映射（会强制 horizontal reduce）

```
SIMD lane → K
accumulate once → horizontal sum
```

### ✅ 正确映射（工业级方式）

```
SIMD lane → N (或 M)
K → 时间维度累加
长期保留 vector accumulator
```

---

## 8. 答案总结（直接回答你的问题）

> 有没有办法不做水平归约？

**回答：**

✅ 数学上：不可能消灭“求和”的本质
✅ 实现上：完全可以避免“横向归约指令”
✅ 方法：不要让 SIMD lane 对应 K 维，而是让 lane 对应多个 `C(m,n)`，让 K 作为时间维用 FMA 累积
✅ 这正是所有高性能 GEMM 内核的标准做法

一句话总结你这个疑问：

> 不是要“消灭 reduction”，而是要把 reduction 从“lane 内横向”变成“时间维纵向”。

---

如果你愿意，我可以帮你画一个 **AVX512 微内核的寄存器布局图**，把这个 lane 映射讲直观一点。


---

### 🧑‍💻 User

我们有两种方案，对于NT形式的gemm：
1. 把B {N, K} repack成 {N/16, K, 16}，这样，对于每一个 $k$，原本跨度很大的 16 个 $N$ 元素就被放在了一起，可以用一条 vmovups 指令直接加载到 zmm 寄存器中。
2. 提前将矩阵 $B$ 转置为 $B^T$，存储为 $K \times N$ 格式。变成 NN：{M, K} * {K, N}，我们正在进行 FMA，将每个 A[row_index] 广播到一个向量，并从 B 列加载 4 或 6 个向量，然后使用 C 作为累加器，我们不需要水平归约，写入 C 将是一个向量化的存储。
这两种方案各有什么优点，有什么区别？

---

### 🤖 Assistant

我直接给你**工程视角下的结论 + 关键差异点**，不绕弯子。

你这两种方案本质是：

| 方案   | 本质                                             |
| ---- | ---------------------------------------------- |
| 方案 1 | **Pack B 成“按 K 走，但 N 连续”的微内存格式**（GEMM 专用 pack） |
| 方案 2 | **全局转置，把 NT 物理变成 NN**（通用转置方案）                  |

两者都能避免 horizontal reduction，但定位完全不同。

---

## 方案 1：`B` pack 成 `{N/16, K, 16}`（典型 GEMM packing）

这一方案是 **工业级 GEMM 的标准做法**。

### ✅ 优点

**1. Cache / TLB 局部性极佳**

你访问 B 的模式从：

```
跳跃式大步长（原始 NT）
↓
连续内存访问
```

pack 后：

```
for k:
    zmm = load B[k][16 连续 N]
```

完全是：

* 顺序流式
* cache line 利用率 100%
* TLB miss 极低

---

**2. 完美匹配微内核设计**

pack 后的布局：

```
B_pack[n_block][k][lane]
```

刚好对应：

```
for k:
    load zmm(B)
    broadcast A scalar
    FMA
```

整个主循环：

✅ 只有 `vmovups` + `vbroadcastss` + `vfmadd231ps`
✅ 没有 shuffle
✅ 没有 reduction

这是 AVX512 GEMM 的黄金路径。

---

**3. 更适合真正的“块 GEMM”**

由于你只 pack 用到的 tile：

* L2/L3 友好
* 可以做 per-tile packing
* 非常利于多线程并行

---

### ❌ 缺点

**1. 有 pack 开销**

需要额外 kernel：

```
原始 B  →  B_pack
```

代价：

* 内存带宽
* 额外 cache 压力
* small K/M/N 时不划算

---

**2. 依赖特定 SIMD 宽度**

你用了 `{…,16}` → 强耦合 AVX512

换到：

* AVX2 = 8
* ARM NEON = 4 / 8

都要重新设计 pack 格式。

---

**3. 数据不是“数学直观格式”**

后续若还要用 `B`：

* attention
* softmax
* reuse

这种 packed layout 不通用。

---

---

## 方案 2：提前把 `B` 真正转置成 `{K, N}`（物理 NN 存储）

这是更“数学上自然”的方案，但更少用于纯 GEMM 内核。

### ✅ 优点

**1. 更通用**

转置完成后：

```
B_T: {K, N}
```

优点：

* 各种算子都能用
* 后续 kernel 逻辑干净
* 不再需要 stride trick

---

**2. 微内核逻辑更简单、更“纯 FMA”**

转置后：

```
for k:
    A_broadcast
    load B_T[k][vector N]
    FMA
```

和方案 1 几乎一样。

---

**3. 更适合“预转置长期缓存”**

如果：

* B 是 weight matrix（如 Linear/FC）
* 会被多次复用

那一次转置、长期使用非常合算。

---

### ❌ 缺点

**1. 转置代价更高**

转置 ≠ packing：

* 要读 K×N
* 写 K×N
* 访存模式不如 pack 连续

尤其当原始 B 是 row-major 的 `{N, K}`：

转置访问是**跨大步长读取**，非常痛苦。

---

**2. 转置后的内存布局仍然不如 pack 优**

BNN：

```
B_T[k][n]
```

虽然 N 连续了，但：

* 多级缓存局部性
* 并行分块粒度
* L2 命中模式

不如专门 pack layout 可控。

---

**3. 线程并行时的写冲突更复杂**

多个线程同时转置：

* 有写地址交错
* cache line 竞争
* 伪共享风险

packing 通常是 tile-local，不容易发生这种冲突。

---

## 核心差异总结表

| 维度           | 方案1：Pack `{N/16,K,16}` | 方案2：真实转置 `{K,N}` |
| ------------ | ---------------------- | ---------------- |
| 目标场景         | GEMM 专用                | 通用算子             |
| 内存连续性        | ✅ 极佳（按 K 流）            | ✅ 连续，但局部性略弱      |
| 预处理代价        | 中（流式可控）                | 高（访存跨步）          |
| Cache 友好性    | ✅ 非常强                  | ⚠ 较强但不极致         |
| 是否依赖 SIMD 宽度 | ✅ 强相关                  | ⚠ 较弱             |
| 并行扩展性        | ✅ 极佳                   | ⚠ 易竞争            |
| 适合权重复用       | ✅ 中等                   | ✅ 很适合            |
| 适合一次性 GEMM   | ✅                      | ❌ 通常不划算          |

---

## 什么时候各自更优？

### ✅ 方案 1 更优的情况

* 追求**极致性能**
* 大矩阵 GEMM
* 只为 GEMM 服务
* 单次或少量复用

### ✅ 方案 2 更优的情况

* B 是**常驻权重矩阵**
* 会被很多 kernel 反复用
* 你希望逻辑统一成标准 NN 格式
* 后续还有融合算子

---

## 一句话核心结论

> 方案 1 是“为 GEMM 微内核量身定制”的数据格式，追求极致局部性；
> 方案 2 是“对数学结构友好”的全局转置，更通用但在 raw GEMM 性能上略逊一筹。

---

如果你愿意，我可以帮你画一个：

* AVX512 微内核寄存器映射图
* pack 布局的 cacheline 访问路径
* 以及两种方案在 L1/L2/L3 下的命中差异图


---

### 🧑‍💻 User

那么，相较于gpu，对于cpu，为什么在编写高性能matmul kernel时候水平规约反而是要尽量避免的？这反映了cpu与gpu编程模型的哪些差异？

---

### 🤖 Assistant

很好的问题，这个点正好是 **CPU vs GPU 编程模型差异的核心本质之一**。我直接给你一个“硬件友好视角”的解释，而不是泛泛的回答。

---

## 一句话总论

**CPU 上“水平归约”是寄存器内部的强串行操作，而 GPU 上的“归约”更多是线程之间的并行协作；两者的硬件模型完全不同，所以优化目标完全相反。**

---

## 1. CPU：SIMD 是“一个指令控制多个 ALU”，而不是“多个线程”

在 CPU 上：

```
一条 SIMD 指令 → 同时驱动多条 lane ALU
```

关键特性：

* SIMD lane 之间 **没有真实通信网络**
* 只是一个“宽操作数 + 宽执行器”
* 水平归约 = 你要在 **一个寄存器内部制造数据依赖链**

### 会发生什么？

以 AVX512 float 归约举例（16 lane）：

```
level 1: 16 → 8  加法
level 2: 8  → 4
level 3: 4  → 2
level 4: 2  → 1
```

这意味着：

* 每一层都依赖上一层结果
* 指令必须串行执行
* OoO（out of order）无法消除依赖
* pipeline bubble 非常明显

结果：**延迟主导，而 CPU 极怕长依赖链**

---

## 2. GPU：归约是“线程协同”，而不是“寄存器内部树形运算”

GPU 是 SIMT：

```
一个 warp = 32/64 个真实线程
```

每个线程有：

* 自己的寄存器文件
* 自己的标量执行器

GPU 的“归约”是这样：

```c
warp_shuffle_xor(val, offset)
val += shuffled_val
```

硬件层面特点：

* 有 **专用 warp shuffle 交换网络**
* 交换是并行完成的
* 延迟几乎固定
* 更多是带宽受限，而非依赖链受限

所以 GPU 的归约：

✅ 更像 **并行通信问题**
❌ 不是单执行器串行问题

---

## 3. CPU 里 horizontal reduction 的“真实代价模型”

在 CPU 微架构里，水平归约涉及：

### ① 跨 lane 交换（shuffle / permute）

这些走的是：

* shuffle unit
* crossbar
* 延迟通常 3–5 cycle+

### ② ALU 操作必须串行

假设每层延迟：

| 层    | 类型            | 延迟 |
| ---- | ------------- | -- |
| 16→8 | shuffle + add | ~4 |
| 8→4  | shuffle + add | ~4 |
| 4→2  | shuffle + add | ~4 |
| 2→1  | shuffle + add | ~4 |

最终延迟：

> 15–20 cycles 甚至更高

而 **你一个 FMA 是 4–5 cycles**，但是吞吐高。

结论：

> reduction 延迟远大于一个向量 FMA 的收益。

---

## 4. GPU 的 latency hiding 能力远强于 CPU

CPU：

* 一个 core：只有 1～2 条关键执行链
* 并行度低
* 几乎没法隐藏长依赖

GPU：

* 同一个 SM：上千线程在飞
* 一个 warp stall，其他 warp 立刻切换
* reduction 延迟被并发 warp 完全掩盖

所以：

| 架构  | 对归约延迟的态度     |
| --- | ------------ |
| CPU | 必须**避免**     |
| GPU | 可以**容忍甚至鼓励** |

---

## 5. 编程模型差异本质

| 维度   | CPU 模型         | GPU 模型                  |
| ---- | -------------- | ----------------------- |
| 执行粒度 | 指令级并行（ILP）     | 线程级并行（TLP）              |
| 向量   | “一条指令、多 lane”  | “多线程同步执行”               |
| 通信   | 不鼓励 lane 互相通信  | 原生支持线程通信                |
| 延迟隐藏 | OoO + cache 预取 | 大量线程切换                  |
| 归约范式 | 避免 horizontal  | 擅长 warp/block reduction |

CPU 在设计上更强调：

> 低延迟、通用性、复杂控制流。

GPU 在设计上更强调：

> 高吞吐、海量并行、规则数据流。

---

## 6. 为什么 GEMM 在 CPU 上一定要“推迟 reduction”？

CPU 的理想模型：

```
多 accumulator（寄存器）
 + 长 K 循环
 + FMA
 + 完全不 shuffle
 + 最后才 collapse
```

也就是：

* 让 reduction 沿时间维发生
* 不在空间维（lane）发生

GPU 则可以：

* 在空间维高效做 warp-level associative reduction

---

## 7. 简短结论

你这个问题可以用一句硬件语言总结：

> CPU 的 SIMD lane 是“宽标量执行器的切片”，而 GPU 的 lane 是“真实线程”。
> 在 CPU 上做水平归约会制造长依赖链并浪费 OoO 潜力；
> 在 GPU 上归约是通过专用通信网络完成的并行操作，代价更可控。

也可以用一句工程结论：

**CPU 优化的是“减少依赖链”，GPU 优化的是“增加并行度”。**

---

如果你愿意，我可以继续给你解释一个更偏微架构层的视角：
👉 为什么 CPU shuffle 单元天然是瓶颈，而 GPU 的 warp shuffle 是 first-class 设计目标。
