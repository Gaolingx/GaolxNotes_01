# 全新架构GPU Microbenchmarking完整方法论

## 🧑‍💻 User

以nvidia h200为例，假设给你一块全新架构的gpu（比如nv下一代的Rubin、Feynman），你要怎么做microbenchmarking，测量指令latency，throughput，存储层级，大小，带宽等数据

---
## 😀 默认助手 (7)

# 全新架构 GPU 的 Microbenchmarking 方法论

以 H200（GH100 / SM90，规格已知）为“标尺卡”，迁移到 Rubin / Feynman（规格未知）上。

核心方法论只有一句话：**先在已知架构上把测量方法标定到与 spec 吻合，再把这套 harness 搬到新架构，只对“测不出来的东西”（新存储层级、新指令）写新的 probe。**

因为新架构最大的风险是：数据看似合理，实际测的是错的东西（编译器优化掉了、测的是 L2 不是 HBM、时钟在浮动）。标定步骤会帮你发现这些问题。

---

## 0. 总流程（建议 1~2 周节奏）

| 阶段 | 目标 | 产出 |
|---|---|---|
| D1–D2 | 环境固化 + 锁频 + 计时原语校准 | 一份 `clock64()` 开销、实际频率、jacobi sanity |
| D2–D3 | 在 H200 上复现 spec（表1） | 方法可信度 $\pm5\%$ 内 |
| D3–D4 | 逆向 ISA：SASS dump + PTX doc + 新指令清单 | 指令白名单 |
| D4–D8 | Latency / Throughput 扫描 | 每指令延迟、每 pipe 吞吐 |
| D8–D12 | 存储层级扫描（容量/延迟/带宽/line/bank/TLB） | 层级表 |
| D12–D14 | 新层级（TMEM / DSMEM / NV-HBI / HBM4）+ 链路带宽 | 完整 spec 补全 |

---

## 1. 环境固化：先让测量可复现

这一节最容易被跳过，但 80% 的“测不准”都来自这里。

```bash
# 独占 + 持久模式 + 锁频（数据中心卡通常需要 root）
nvidia-smi -pm 1
nvidia-smi -c EXCLUSIVE_PROCESS          # 或 MPS 全部关掉
nvidia-smi -lgc 1980,1980                # 锁 SM 时钟，H200 参考 boost=1.98GHz
nvidia-smi -lmc 2619,2619                # 锁 HBM 时钟（可选）
nvidia-smi -e 0                          # 关 ECC（会改可用 BW，测完恢复！）
```

必须持续监控，任何被 throttle 的点都要标出来：

```bash
nvidia-smi dmon -s pucvmet -d 1 | tee dmon.log
# 或 dcgmi dmon -e 100,101,150,155
```

还要固定：
- `CUDA_MODULE_LOADING=EAGER`（避免首次 launch 的 JIT 干扰）
- 关掉 `cudaLimitPersistingL2CacheSize` 的残留配置（它会改变有效 L2）
- 记录 driver / CUDA / ptxas 版本，**这些会改变 SASS 调度**

---

## 2. 第一步永远是：摸清 ISA

新架构上你可能连“有哪些指令”都不知道，所以不要一上来就写 benchmark。

```bash
# 1) 编译器支持到哪一代
nvcc --list-gpu-arch
nvcc -arch=native -cubin -o k.cubin k.cu && cuobjdump -sass k.cubin
nvdisasm -c k.cubin              # 更细的控制码/调度信息

# 2) 看编译器有没有为你生成新指令
cuobjdump -sass gemm.cubin | awk '{print $2}' | sort | uniq -c | sort -rn
```

关键情报来源，按可靠性排序：
1. **PTX ISA 文档**（新版本会列新 PTX 指令，如 Hopper 的 `wgmma`、Blackwell 的 `tcgen05`）
2. **`cuobjdump -sass` 反汇编**：新指令的 opcode 与操作数编码
3. **CUDA C++ Programming Guide 的 Compute Capability 表**（每 SM 的 lane 数、smem 上限、寄存器数）
4. **`cudaDeviceProp` / `cuDeviceGetAttribute`**：先用属性表作为“待验证假设”，不要当真
5. **Nsight Compute 的 Source Counters + Warp Stall Sampling**：直接看每条 SASS 的 stall 原因，等于免费拿到单指令的瓶颈信息

> 若 PTX 还不支持你想测的指令（新架构早期很常见），你有两条路：写 inline PTX 让 `ptxas` 生成；或者从已有的库 kernel（cuBLAS/cuDNN 的 cubin）里 dump 出该指令的用法再模仿。**不能手写 SASS**，所以必须以 PTX 为最低层。

---

## 3. 计时基础设施：三种原语

| 原语 | 分辨率 | 开销 | 适用 |
|---|---|---|---|
| `clock64()` | 1 SM cycle | ~20–40 cyc | 指令级、依赖链（首选） |
| `%globaltimer`（ns） | 1 ns 单位，~32 ns tick | ~20 cyc | 跨 SM / 跨 kernel 对齐 |
| `cudaEventElapsedTime` | ~0.5 µs | launch 开销 | kernel 级、带宽 |

**必须先校准**：

```cuda
// 测 clock64 自身开销
long long t0 = clock64(); long long t1 = clock64();
ovh = t1 - t0;                    // 循环 N 次取 min

// 测实际 SM 时钟：globaltimer 与 clock64 联立
double ghz = (double)(cyc_delta) / (double)(ns_delta);   // cyc/ns = GHz
```

两个原语互相校验，可以顺带得出“名义频率 vs 实际频率”，也是后面所有理论峰值计算的分母。

---

## 4. 指令 Latency：依赖链法

**定义**：issue 到结果可被下一条依赖指令使用之间的周期数。

**方法**：构造纯依赖链，用 `clock64()` 夹住 $N$ 次迭代，除以 $N$。

```cuda
// 只开 1 个 warp/SM，得到"裸延迟"；开满 warp 得到的是吞吐
__global__ void lat_chain(float *out, int iters) {
    float a = out[threadIdx.x];            // 从内存来，防止常量折叠
    float b = 1.0000001f, c = 0.5f;
    long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
    }
    long long t1 = clock64();
    out[threadIdx.x] = a;
    if (threadIdx.x == 0) out[1] = (float)((t1 - t0 - ovh) / (double)iters);
}
```

要点：
- `#pragma unroll 1` + `asm volatile` 双重保险，防止编译器把循环变成乘方或提升到循环外
- 测完后，**回看 SASS 确认**内部就是 N 条 FFMA，没有多余指令
- 循环计数器（IADD/ISETP/BRA）是独立指令，与依赖链并行，但会占用 issue slot —— 只开 1 warp 时无妨；若发现被 issue 限制，就手工展开 8×16 次
- 延迟短的指令（FADD=4 cyc）要展开到循环体内 32~64 条，误差才降到 1% 以下

**要测的清单**：

| 类别 | 指令 | H200 参考值（cycle，**以实测为准**） |
|---|---|---|
| ALU | FFMA / FADD / FMUL / IADD3 | ~4 |
| 逻辑 | LOP3 / SHF | ~4 |
| 转换 | F2F / I2F | ~6 |
| SFU | MUFU (rcp/sqrt/exp) | ~10–20 |
| 分支 | BRA (taken) | ~15–20 |
| Shared | LDS / STS | ~20–30 |
| L1 | LDG (L1 hit) | ~30–40 |
| L2 | LDG (L2 hit) | ~200–260 |
| HBM | LDG (miss) | ~500–700 |
| Tensor | mma / wgmma / tcgen05 | 依赖 accumulator 依赖链 |
| 新架构 | *？* | 你的主要产出 |

**内存延迟用 pointer chasing**，不要用依赖链 FMA 那套：

```cuda
__global__ void chase(const uint32_t *perm, uint32_t *out, int steps) {
    uint32_t idx = perm[threadIdx.x + blockIdx.x * blockDim.x]; // 随机起点
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < steps; ++i)
        idx = __ldg(&perm[idx]);          // 地址依赖，天然不可优化
    uint64_t t1 = clock64();
    out[threadIdx.x] = idx;
    if (threadIdx.x == 0) out[1] = (uint32_t)((t1 - t0) / steps);
}
```
`perm` 是**随机置换**（打乱顺序避免 prefetcher / 顺序预取导致的假命中）。`__ldg` → `ld.global.nc` 走只读路径，更纯净。

---

## 5. 指令 Throughput：饱和法

**定义**：某 pipe 在稳态下每 cycle 能接受的 warp-instruction 数。

两种测法，互相校验：

**(a) FLOPs 法**：N 条**互相独立**的累加链（≥4 条，最好 8~16 条）填满 operand collector 与流水线。只有一条链时测到的是延迟，不是吞吐。

```cuda
float a0..a7;
#pragma unroll
for (int i = 0; i < U; ++i) {
    a0 = fmaf(a0,b,c); a1 = fmaf(a1,b,c); ... a7 = fmaf(a7,b,c);
}
```
$$\text{FP32 TFLOPS} = \frac{\text{threads} \times U \times 8 \times 2}{t_{\text{ns}}} \times 10^{-3}$$

**(b) 指令计数法**：跑 $M$ 条同一指令，用 `clock64()` 测 $\Delta$ cycles，得
$$\text{issue rate} = \frac{M \cdot W}{\Delta \text{cyc}} \quad [\text{warp-inst/cyc/SM}]$$
与理论对比（SM90：4 scheduler × 1 warp-inst/cyc = 4 IPC/SM 上限）。

**必须按 pipe 分类测**，因为各 pipe 宽度不同（SM90 每 SM）：
- FP32：128 lane → 4 warp-inst/cyc
- INT32：64 lane → 2
- SFU (MUFU)：16 lane → 0.5
- LD/ST：16–32 lane
- Tensor core：单独测，见下

**Tensor core 吞吐**要小心：不能只跑一个大 GEMM（会被 tiling / smem BW 限制）。要在寄存器/smem 上做**纯 MMA 背靠背**，并分别测：
- 独立 accumulator（吞吐）vs 同一 accumulator（依赖延迟）
- 不同 dtype（FP8/FP16/BF16/FP4…）、不同 K
- Blackwell 的 `tcgen05` 需要 TMEM，得先摸清 TMEM 的装载/回收路径，否则 `mma` 之间会插入等待，量到的一半是 TMEM 带宽

---

## 6. 存储层级：一招多用（工作集扫描）

**核心方法：固定随机置换 + 扫描工作集大小 $S$，画「延迟 vs $S$」曲线。** 一个实验同时给出**每一级的容量（拐点）与延迟（平台高度）**。

```python
# 伪代码，host 侧驱动
sizes = 2**np.arange(10, 31, 0.25)          # 1KB .. 1GB
for S in sizes:
    perm = build_random_permutation(S)      # 元素为 4B 或 8B
    lat[S] = run_chase_kernel(perm, steps=1e6)
```

判读：
- **平台高度 = 该级延迟**
- **平台宽度 = 该级容量**
- 拐点之后延迟跳变 → 上一级装不下了

同理把「延迟」换成「带宽」，得到 **带宽 vs 工作集** 曲线，两条曲线交叉验证容量（比单纯看延迟更可靠）。

### 6.1 每级怎么测

| 目标 | 做法 |
|---|---|
| **L1 / 纹理容量** | 置换放在 $S \in [8\text{KB}, 4\text{MB}]$，看 ~30cyc 平台能撑多宽 |
| **L2 容量** | $S \in [1\text{MB}, 512\text{MB}]$，~200cyc 平台宽度；与 `prop.l2CacheSize` 对照（Hopper 上 L2 分两半，注意 MIG 会切分） |
| **Shared memory 上限** | 二分 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 直到 launch 失败；**别信属性值**，实测 opt-in 上限（H200 = 227KB/block） |
| **寄存器文件** | `-maxrregcount=255` + 测最大常驻 warp 数 × 255；或 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 反推 |
| **Cache line / sector** | 固定小工作集，扫 **stride** 1B→4KB，看延迟/带宽在 32B、128B 处的台阶 → sector=32B、line=128B |
| **Associativity** | 固定 stride $= k\cdot 2^n$ 制造 set 冲突，延迟尖峰 → ways 与 set 数 |
| **Bank 宽度 / conflict** | LDS 扫 stride 0..32 words，吞吐凹陷位置 → bank 数、bank 宽度（通常 4B）、双端口与否 |
| **TLB** | 用 stride ≥ 2MB 的大跨度访问 vs 64KB 跨度，延迟差 → TLB 覆盖范围与页大小；再测大页（`cudaMallocAsync` / `mmap hugepage`） |
| **SM 数** | 用 `%smid` 收集 + 一个大 grid 反推，或 `multiProcessorCount` 属性（属性可信度高，但顺手验一次） |

### 6.2 一个实用技巧：二分定位

不知道有几级时，先用**粗扫**（每 octave 一个点）找所有拐点，再在每个拐点附近**细扫**（每 1/8 octave）定位精确容量。不要一开始就暴力扫，1GB 的随机置换打表本身就很贵。

---

## 7. 带宽：分层测，注意“有效”二字

### 7.1 HBM（最重要）

```cuda
// read-only：向量化 + 足够 MLP
__global__ void bw_read(const float4 * __restrict__ a, float4 *out, size_t n) {
    float4 acc = {0,0,0,0};
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {           // grid-stride，保证 coalesced
        float4 v = a[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    // 防止 DCE
    if (acc.x == 12345.678f) out[0] = acc;
}
```

必须做对的几件事：
- **grid 要够大**：至少 $4\times$ SM 数的 block，否则填不满内存级并行
- **每线程 in-flight load 数**要扫（ILP 2/4/8），新架构可能不同
- **分别测 read / write / copy / triad**，不要只测 copy（copy 是 50/50 混合，会掩盖单向瓶颈）
- **write-only 要用 streaming store**（`__stcs` / `st.global.cs`），否则 write-allocate 把读带宽也算进去
- **N 要远大于 L2**（≥ 2 GB），并做 warmup
- 用 `cudaEvent` 计整个 kernel，别用 `clock64`

$$\text{BW} = \frac{\text{bytes moved}}{t}\ ,\qquad \text{效率} = \frac{\text{BW}_{\text{实测}}}{\text{HBM 理论}}$$

### 7.2 其他层级与链路

| 目标 | 工具 / 方法 |
|---|---|
| L2 BW | 工作集刚好 ≤ L2，多 block 并发读，$>200$ GB/s per SM 量级 |
| Shared BW | LDS/STS 无 bank conflict 的密集循环；=$128$ B/clk/SM 量级 |
| 寄存器 BW | FFMA 全寄存器操作数的 issue rate 反推 operand collector 极限 |
| NVLink/NVSwitch | **`nvbandwidth`**（NVIDIA 官方开源，首选）；注意区分单向/双向、P2P 是否启用 |
| PCIe | `nvbandwidth -t device_to_host_memcpy`、`cuda-samples/bandwidthTest` |
| 跨 die（B200 类） | 见下一节 |

---

## 8. 新架构专属：那些“以前不存在”的层级

这是 Rubin/Feynman 上真正要花时间的部分。H200 上没有的东西，你的方法论必须临时搭建。

<details>
<summary><b>已出现的新层级 & 对应 probe（点击展开）</b></summary>

**TMEM（Blackwell `tcgen05` 引入的 per-SM tensor memory）**
- 容量：二分 tcgen05.alloc 直到失败
- 延迟/带宽：`tcgen05.ld` / `tcgen05.st` 做依赖链与密集流，扫描列宽
- 关键：`mma` 与 TMEM 之间的 pipeline 耦合，常常是 tensor 吞吐的真正瓶颈

**DSMEM / Thread Block Cluster（Hopper 引入）**
- 用 `cluster.map_shared_rank` 访问远端 SM 的 smem
- 测：本地 LDS vs 远端 DSMEM 的延迟差、带宽上限
- 扫描 cluster 大小 2/4/8/16，看是否随拓扑下降

**双 die（B200/Rubin 类）的 NUMA 效应**
- 单设备两个 die，CUDA 不暴露放置控制
- **技巧**：用大 buffer 做 pointer chase，对延迟做**直方图**。双峰 = 本地 die / 远端 die（NV-HBI）
- 两峰之差 = die 间延迟；用双 die 分别饱和访问可测 NV-HBI 带宽
- 检查 `prop.multiGpuBoardGroupID`、`cudaDevAttrPageableMemoryAccess` 等新属性

**HBM4 / 更大位宽**
- 方法完全同上，但要重扫 ILP 与 block 数（更宽的接口往往需要更多 MLP 才能饱和）
- 注意 HBM3e→HBM4 的 **pseudo-channel / 32B sector** 变化，用 stride sweep 重新确认最小访问粒度

**新 tensor core dtype（FP4/FP6…）**
- 理论峰值 $\text{TFLOPS} = N_{SM}\times \text{lanes}\times 2 \times f$
- 实测/理论 < 0.7 时，先怀疑数据搬运（smem→TMEM）而不是 core

</details>

---

## 9. 陷阱清单（按踩坑频率排序）

<details>
<summary><b>点击展开</b></summary>

1. **编译器把你优化了**：所有 latency/throughput kernel 测完必须 `cuobjdump -sass` 回看指令条数是否等于预期。这是第一优先级检查。
2. **测到的是上一层缓存**：工作集没超过 L2，你以为在测 HBM。永远画“延迟 vs 尺寸”曲线确认处在平台右侧。
3. **时钟漂移**：短 kernel 会 boost，长 kernel 会 throttle。短测必配 `nvidia-smi dmon` 或 `%globaltimer` 联测。
4. **只用 1 个 warp 测吞吐**：那是延迟。用 ≥4 warp/scheduler + ≥4 独立链。
5. **只用 1 条累加链测吞吐**：同上，量到延迟。
6. **随机置换不随机**：置换生成有误（存在短环）会让延迟偏低。验证方法：统计跳数分布、确认无重复访问。
7. **`volatile` 的副作用**：`ld.volatile` 可能绕过某些缓存层，导致测的不是目标层级。优先用地址依赖（pointer chasing）而不是 `volatile`。
8. **write-allocate**：纯写带宽虚高/虚低，用 `.cs` store。
9. **MIG 存在**：L2、SM、带宽全被切分，属性值不变但实际变小。先确认 `nvidia-smi -q | grep MIG`。
10. **ECC 开启**：HBM 有效带宽下降几个百分点，跨机器对比时是噪声源。
11. **首次 launch 的 JIT / 模块加载**：所有 kernel 先 warmup ≥10 次。
12. **取平均而不是取最小**：延迟类测 **min of N**（干扰最少的那次），吞吐类取 **median**，并报告分布宽度。
13. **单点测量**：每个配置至少 5 次独立重复，换个进程/换天再验一次，确保不是偶然。
14. **理论峰值算错**：`$N_{SM}$` 用的是 enabled 而非 physical 数；lane 数、clock 用实测值。

</details>

---

## 10. H200 标定表（用来验方法，不是结论）

先在 H200 上把下表复现到 $\pm5\%$，你的 harness 才可信。

| 指标 | H200 SXM 参考 |
|---|---|
| SM 数 / boost | 132 / ~1.98 GHz |
| FP32 lane/SM | 128 → $\approx 67$ TFLOPS |
| FP32 issue | 4 warp-inst/cyc/SM |
| FP16/BF16 Tensor dense | $\approx 989$ TFLOPS |
| FP8 Tensor dense | $\approx 1979$ TFLOPS |
| FP64 / FP64 Tensor | $\approx 34$ / $67$ TFLOPS |
| HBM | 141 GB HBM3e, $\approx 4.8$ TB/s |
| L2 | $\approx 50$ MB |
| Shared / 寄存器 | 228 KB(227 可用) / 256 KB per SM |
| NVLink | $\approx 900$ GB/s 双向 |
| 每 SM 带宽 | $4.8\text{TB/s}/132 \approx 36$ GB/s |
| Roofline 拐点（FP16） | $\approx 206$ FLOP/byte |

$\text{峰值 FLOPs} = N_{SM}\times L_{lane}\times 2\times f_{实测}$，用这一条公式对照上面每一行，对不上就是有环节测错了。

---

## 11. 最终交付物

一张“新架构规格补全表”，每一格都带两种独立方法的交叉验证：

| | 容量 | 延迟 | 带宽 | 关键参数 |
|---|---|---|---|---|
| 寄存器 | | | | 宽度、bank |
| Shared | | | | bank 数/宽度、conflict 图 |
| TMEM | | | | 分配粒度 |
| L1/纹理 | | | | sector、line、ways |
| L2 | | | | 分区、persisting 能力 |
| HBM | | | | 通道数、最小 burst |
| DSMEM | | | | cluster 拓扑 |
| Die 间 | | | | 远端/本地比 |
| NVLink/PCIe | — | | | 单向/双向 |

配套：一套参数化 CUDA harness（block size / ILP / unroll / occupancy 全部可扫）+ 自动出图脚本，这样换架构时只需重跑而不需要重写。

<details>
<summary><b>附录：最小可用 harness 骨架</b></summary>

```cuda
// 通用：依赖链延迟
template<class F> __global__ void bench_lat(float*out, int iters, F f);

// 通用：饱和吞吐（8 条独立链）
__global__ void bench_tput(float*out, int iters) {
    float a[8];
    #pragma unroll
    for (int i=0;i<8;i++) a[i] = out[threadIdx.x+i];
    long long t0 = clock64();
    #pragma unroll 1
    for (int i=0;i<iters;i++) {
        #pragma unroll
        for (int j=0;j<8;j++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;":"+f"(a[j]):"f"(1.0000001f),"f"(0.5f));
    }
    long long t1 = clock64();
    #pragma unroll
    for (int i=0;i<8;i++) out[threadIdx.x+i]=a[i];
    if (threadIdx.x==0) out[64]=(float)((t1-t0)/(double)iters);
}

// 通用：pointer chase（延迟 + 容量扫描）
__global__ void bench_chase(const uint32_t*perm, unsigned*res, int steps){
    uint32_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long t0=clock64();
    #pragma unroll 1
    for(int i=0;i<steps;i++) idx = __ldg(&perm[idx]);
    long long t1=clock64();
    if(threadIdx.x==0) res[blockIdx.x]=(unsigned)((t1-t0)/steps);
}
```

驱动侧：固定 `blockDim=32`（单 warp，测裸延迟）与 `blockDim=1024 × 满 occupancy`（测吞吐）两套配置各跑一遍，前者给延迟、后者给吞吐，两者永远一起报。

</details>

---

**一句话总结**：在 H200 上把「随机置换扫尺寸」和「依赖链 + 独立链」这两把尺子校准到能复现 spec，然后在 Rubin/Feynman 上第一件事是 `cuobjdump -sass` 摸清 ISA、用属性表和扫描曲线互相验证找出**新增的存储层级**，剩下的都是重跑参数化 harness 而不是重写方法。
