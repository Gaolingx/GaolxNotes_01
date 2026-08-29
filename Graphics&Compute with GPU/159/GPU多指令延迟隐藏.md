# GPU多指令延迟隐藏

### 🧑‍💻 User

gpu发射 多指令 延迟隐藏是怎么实现的

---

### 🤖 Assistant

GPU 的“多指令发射 + 延迟隐藏”是其能够实现高吞吐量计算的核心机制。下面从硬件设计、调度策略和执行流程几个层面来拆解它的实现。

---

## 1. 延迟的来源与为什么要隐藏它

在 GPU 中，指令的执行延迟主要来自两方面：

- **算术延迟**：一个算术指令（如浮点乘法、加法）从发射到结果写回需要多个时钟周期（例如 $10 \sim 20$ cycles）。
- **内存延迟**：访问全局内存（DRAM）可能需要数百个时钟周期。

如果 GPU 像 CPU 一样顺序执行一个线程（或一个 warp），那么在等待长延迟操作完成的时间里，执行单元就会空闲，导致有效吞吐量急剧下降。GPU 的核心策略是 **用大量并行线程的切换来填满这些空闲槽，从而“隐藏”延迟**，而不是去减少延迟本身。

---

## 2. 基础：Warp 与 SIMT 模型

- GPU 的最小调度单位是 **warp**（NVIDIA 通常为 32 个线程，AMD 为 wavefront 64 个线程）。
- 一个 warp 中的所有线程以 **SIMT（单指令多线程）** 方式执行同一条指令，但各自拥有独立的寄存器状态。
- GPU 上同时驻留着大量 warp（例如一个 SM 中可以同时存在 64 个 warp），它们共享执行单元、寄存器和共享内存。

这些 warp 提供了切换的基础：当正在执行的 warp 遇到长延迟操作时，硬件零开销地切换到另一个就绪的 warp，不让执行单元停下来。

---

## 3. 多指令发射：每个周期从多个 Warp 中挑选指令

现代 GPU（如 NVIDIA Volta 及之后的架构）每个 SM 内部有多个 warp scheduler。每个 scheduler 在每个时钟周期可以发射多条指令（dual‑issue 甚至更多）。

典型的硬件拓扑：
- 一个 SM 内有 **4 个 warp scheduler**。
- 每个 scheduler 管理一个 warp 池（例如 16 个 warp）。
- 每个周期，scheduler 可以从它管理的就绪 warp 中选出 **1～2 条指令** 发送到执行单元（例如一条整数/浮点指令 + 一条内存加载指令，或者两条独立的算术指令）。如果两个指令来自不同的 warp，那更是常态。

**关键点**：多指令发射并不是从一个 warp 里提取指令级并行（ILP），而是从多个 warp 里提取 **线程级并行（TLP）** 转换为指令级并行。每个 warp 内部是顺序标量流水线，但不同 warp 之间可以并行发射，从而填满多个执行单元。

---

## 4. 延迟隐藏的具体工作原理

以一个长延迟的全局内存访问为例：

1. **Warp A** 执行一条 `LD`（加载）指令，访问全局内存。这个请求会被送到内存子系统，需要等 300～600 cycles 才能返回数据。
2. 发射完这条 `LD` 后，硬件将 **Warp A** 标记为“未就绪”（等待数据），并将其从可调度集合中移除。
3. 在接下来的 **每一个时钟周期**，warp scheduler 扫描它管理的就绪 warp 列表，选择另一个 warp（例如 **Warp B**）发射指令。如果存在多个就绪 warp，它可以连续发射不同 warp 的指令，不让流水线空闲。
4. 在这些周期里，执行单元在运行 Warp B、Warp C、Warp D 的计算指令、整数操作、或甚至其他 warp 的内存请求等。
5. 经过数百个周期后，Warp A 的内存数据返回，Warp A 被重新标记为“就绪”。当 scheduler 再次选中它时，就可以执行后续的依赖指令（例如用加载的值做加法）。

**结果**：从单个 warp 的视角看，它被“卡住”了几百个周期，但从 SM 的视角看，这几百个周期里执行单元几乎一直在做有用的工作。延迟被“隐藏”在其他 warp 的执行背后。

---

## 5. 硬件支撑：寄存器文件、计分板和零开销切换

要实现以上模型，需要几个关键硬件特性：

- **巨大的寄存器文件**：每个 SM 有数十 MB 的寄存器空间，用于同时保存所有驻留 warp 的上下文。例如每个 warp 需要 32×N 个寄存器，64 个 warp 可以同时存储全部状态，切换时无需保存/恢复到内存。
- **计分板（Scoreboard）**：跟踪每个 warp 中各寄存器的就绪状态（是否被长延迟操作占用）。只有所有源寄存器就绪的指令才能被发射。
- **零开销 warp 切换**：切换 warp 只是更新 scheduler 选中的 warp ID，不需要清空流水线或保存上下文。因此可以 **每个时钟周期选择一个不同的 warp**。

这些硬件使得“每个周期都从一个不同 warp 发射指令”成为可能，极大容忍了指令间的延迟。

---

## 6. 多指令发射与延迟隐藏的协同

单次发射一个指令，执行单元可能仍喂不饱。多指令发射进一步放大了隐藏效果。例如：

- 每个 scheduler 每周期可以同时发射 **一条计算指令** 和 **一条内存访问指令**，两者来自不同的 warp，或者同一个 warp 的独立指令。
- 当大量 warp 都处于计算密集状态时，多发射可以在同一周期内填充浮点、整数、加载/存储等多个执行单元。
- 如果某个 warp 的指令之间存在数据依赖，多发射可以同时从另外两个无依赖的 warp 里各取一条指令发射，从而完全避免流水线气泡。

现代架构甚至引入了 **指令级并行（ILP）感知的调度**：编译器优化或硬件可以识别某个 warp 内连续多条独立指令，将它们单周期多发（例如 NVIDIA 的 dual‑issue，在一个 warp 内同时发射两条独立指令），但这仍是辅助手段，主要依赖 TLP。

---

## 7. 实例：NVIDIA Volta/Turing/Ampere 中的实现

以 Volta SM 为例（后续架构类似）：

- 每个 SM 有 4 个 warp scheduler，每个 scheduler 拥有一个 warp 池。
- 每个 scheduler 每周期可以向 **4 个执行路径** 发出指令：分支、数学运算、加载/存储、张量核心。
- 它可以 **双发**（dual‑issue）来自同一个 warp 的两条独立指令（例如一条浮点 FMA 和一条整数加法），条件是二者无依赖且目标执行单元不同。
- 如果双发条件不满足，它仍然可以从另一个 warp 发出指令，实现单周期内多 warp 并行。

配合共计 64 个 warp 的常驻容量，即使每个 warp 平均每 32 个周期才能执行一条指令，SM 每个周期仍然能从可用的 warp 中找出足够的指令来保持执行单元繁忙。

---

## 8. 总结：核心要点

| 机制 | 作用 |
|------|------|
| **海量 warp 驻留** | 提供大量可切换的指令流，当一个 warp 等待时，立即切换至其他。 |
| **零开销 warp 切换** | 每个周期都允许不同 warp 发射指令，无上下文保存开销。 |
| **计分板 + 寄存器文件** | 自动跟踪每个 warp 的指令依赖，硬件识别就绪 warp。 |
| **多指令发射** | 每周期从多个就绪 warp（或同一 warp 的独立指令）发射多条指令，填充多个执行单元。 |
| **长延迟容忍** | 通过上述组合，内存延迟和算术流水线延迟被其他 warp 的执行完全覆盖。 |

最终效果：GPU 不需要像 CPU 那样依赖复杂的乱序执行和大缓存来降低延迟，而是利用 **海量线程级并行 + 多指令发射** 让延迟“消失”在切换之中，实现接近峰值吞吐量的性能。

---

### 🧑‍💻 User

具体到gpu cs based渲染管线各种pass是怎么实现的？需要举例

---

### 🤖 Assistant

基于 Compute Shader（计算着色器）的渲染管线，是将传统的图形管线（VS→Rasterizer→PS）中的各个 Pass 全部或部分替换为 `Dispatch` 调用，由 Compute Shader 完成光栅化、着色和混合。这种模型赋予了开发者对硬件并行度的极致控制，常用于 **GPU‑Driven Pipeline**、延迟渲染光照、后处理链以及软件光栅化方案。

下面从实现机制、通用 Pass 模板、几个典型 Pass 的完整例子，以及同步这几个维度展开详解。

---

## 1. 为什么要用 Compute Shader 实现 Pass

传统图形管线的局限：
- **固定功能光栅器粗粒度过大**：无法精确控制访问模式，难以实现自定义的 Tile‑Based 分类和剔除。
- **线程组协作困难**：Pixel Shader 中线程彼此独立，无法进行 Group Shared Memory（共享内存）的显式协作（仅能通过纹理缓存）。
- **资源使用不透明**：Draw Call 的状态切换有开销，大量间接绘制需要复杂的参数管理。

Compute Shader 的优势：
- **任意并发模式**：可以按 2D Tile、1D Batch、三角形列表等自由划分线程组。
- **共享内存**：`groupshared` 允许同一 Tile 内的线程高效协同（如 Tile‑based 光照）
- **零开销切换**：多个 `Dispatch` 之间无需绑定渲染目标、深度模板状态，仅通过 UAV/SRV 传递数据。
- **间接调度**：DispatchIndirect 可以根据 GPU 自己计算出的工作分组数量接启动后续 Pass。

---

## 2. CS‑Based Pass 的通用实现骨架

每一个 CS Pass 本质上是：

1. **声明输入**：将上一个 Pass 的输出纹理/缓冲区绑定为 `Texture2D`、`StructuredBuffer` 等 SRV（只读）。
2. **声明输出**：将结果目标绑定为 `RWTexture2D`、`RWStructuredBuffer` 等 UAV（可读写）。
3. **定义线程组大小**：例如 `[numthreads(8,8,1)]`。
4. **Dispatch**：`context.Dispatch(ceil(width/8), ceil(height/8), 1)`。
5. **Barrier 转换**：Pass 之间通过 `ResourceBarrier` 将 UAV 转为 SRV，或将 SRV 转为 UAV。

模板伪代码：
```hlsl
// Input
Texture2D<float4> InputTexture : register(t0);
// Output
RWTexture2D<float4> OutputUAV : register(u0);

[numthreads(8, 8, 1)]
void CS_Pass(uint3 id : SV_DispatchThreadID) {
    float4 color = InputTexture[id.xy];
    // ... 处理 ...
    OutputUAV[id.xy] = result;
}
```
```cpp
// C++ 调度
context.CSSetShaderResources(0, 1, &inputSRV);
context.CSSetUnorderedAccessViews(0, 1, &outputUAV, nullptr);
context.Dispatch(width/8, height/8, 1);
// 下一个 Pass 前
context.ResourceBarrier(1, &barrierOutputToSRV);
```

---

## 3. 具体 Pass 的实现举例

### 3.1 深度金字塔（Hi‑Z）构建 Pass

**用途**：在 GPU‑Driven 管线中做遮挡剔除，需要逐帧生成深度缓冲区的 mipmap，但不同于普通纹理 mip，Hi‑Z 使用 **最远深度**（最大值），保证保守剔除。

**实现**：
- 输入：上一帧的深度纹理（或在深度预 pass 后生成的深度图）。
- 输出：mip level 0 的原深度作为 SRV，level 1~N 的 UAV。
- 每个 Dispatch 生成一个 mip 级别，使用 2×2 核取最大值。
```hlsl
Texture2D<float> DepthSRV : register(t0);
RWTexture2D<float> HiZMip1UAV : register(u0);

[numthreads(8, 8, 1)]
void CS_BuildHiZMip1(uint3 id : SV_DispatchThreadID) {
    uint2 coord = id.xy * 2;
    float d0 = DepthSRV[coord + uint2(0,0)];
    float d1 = DepthSRV[coord + uint2(1,0)];
    float d2 = DepthSRV[coord + uint2(0,1)];
    float d3 = DepthSRV[coord + uint2(1,1)];
    float maxDepth = max(max(d0, d1), max(d2, d3));
    HiZMip1UAV[id.xy] = maxDepth;
}
```
**为什么用 CS**：传统 GenerateMips 是图形管线的固定功能，但只能做平均（或设备相关），无法自定义为最大值。CS 提供了精确控制。

---

### 3.2 Tile‑based 延迟光照 Pass（最经典的 CS 用例）

**用途**：对 G‑Buffer（法线、反照率、粗糙度、深度等）进行着色，需要处理大量动态光源。传统 Pixel Shader 需对每个像素循环所有光源，非常昂贵。CS 方案**分 Tile 剔除光源**。

**流程**：
1. **光源剔除 Pass（可选）**：为每个 Tile 计算影响光源列表，存入 `StructuredBuffer`。
2. **光照计算 Pass**：每个 Tile 加载 G‑Buffer 到共享内存，读取光源列表，计算着色。

**光照 Pass 核心代码**：
```hlsl
Texture2D<float4> GBufferA : register(t0); // normal + roughness
Texture2D<float4> GBufferB : register(t1); // albedo + metallic
Texture2D<float> Depth : register(t2);
StructuredBuffer<Light> Lights : register(t3);
RWTexture2D<float4> Output : register(u0);

groupshared uint uLightIdxList[1024];
groupshared uint uLightCount;
groupshared float4 sGBuffer[16][16]; // 根据 Tile 大小调整

[numthreads(16, 16, 1)]
void CS_DeferredLighting(uint3 groupID : SV_GroupID, 
                         uint3 groupThreadID : SV_GroupThreadID, 
                         uint groupIndex : SV_GroupIndex)
{
    uint2 pixel = groupID.xy * 16 + groupThreadID.xy;
  
    // 1. 加载 G-Buffer 到共享内存
    sGBuffer[groupThreadID.x][groupThreadID.y] = GBufferA[pixel];
    GroupMemoryBarrierWithGroupSync();
  
    // 2. 只有 thread 0 执行光源剔除（或者前面的 CS 已做好）
    if (groupIndex == 0) {
        uLightCount = 0;
        // 计算 Tile 的深度范围
        float minDepth = 1.0; float maxDepth = 0.0;
        // ... 遍历 sGBuffer 中的深度 ...
        // 遍历全局光源列表，进行 Tile/深度剔除
        for (int i = 0; i < totalLightCount; ++i) {
            if (/* 光源与 tile 相交且深度覆盖 */) {
                uLightIdxList[uLightCount++] = i;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
  
    // 3. 每个线程为自己的像素计算光照
    float3 color = float3(0,0,0);
    for (uint i = 0; i < uLightCount; ++i) {
        Light light = Lights[uLightIdxList[i]];
        // 使用 sGBuffer 中的法线等计算 BRDF
        color += EvaluateLight(light, sGBuffer[groupThreadID.x][groupThreadID.y]);
    }
    Output[pixel] = float4(color, 1);
}
```
**关键优势**：利用 `groupshared` 内存使 Tile 内 256 个线程协同，只做一次光源剔除，大幅减少重复判断。传统的 PS 无法做到跨像素共享剔除结果。

---

### 3.3 纯 CS 软件光栅化实现深度预 Pass

**用途**：完全绕过硬件光栅器，在 CS 中手动光栅化三角形，写入深度缓冲区（通常用于 GPU‑Driven 的下逐步剔除或极小三角形的高效光栅化）。

**实现**：
- 输入：三角形顶点缓冲区（经过 MVP 变换的裁剪空间坐标）、索引缓冲区。
- 输出：`RWTexture2D<float>` 深度 UAV。
- 每个线程组处理一个三角形，内部用多个线程并行覆盖像素块。

```hlsl
struct Triangle {
    float4 v0, v1, v2;
};
StructuredBuffer<Triangle> Triangles : register(t0);
RWTexture2D<float> DepthUAV : register(u0);

// 每个线程组处理一个三角形，组内线程处理像素块
[numthreads(8, 8, 1)]
void CS_SoftwareRaster(uint3 groupID : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint triIdx = groupID.x; // 假设一维按三角形 dispatch
    Triangle tri = Triangles[triIdx];
  
    // 将三角形变换到屏幕空间
    float2 v0 = tri.v0.xy / tri.v0.w * 0.5 + 0.5;
    float2 v1 = tri.v1.xy / tri.v1.w * 0.5 + 0.5;
    float2 v2 = tri.v2.xy / tri.v2.w * 0.5 + 0.5;
    // 逆天深度用于插值 (1/w)
    float invW0 = 1.0 / tri.v0.w, invW1 = 1.0 / tri.v1.w, invW2 = 1.0 / tri.v2.w;
  
    // 计算三角形 edge 函数
    float area = edgeFunction(v0, v1, v2);
    if (area <= 0) return; // 背面剔除
  
    // 每个线程负责分块后的一个像素（例如 Tile 方式）
    uint2 pixel = groupID.xy * 8 + gtid.xy;  // 假设每个线程组处理一个 8x8 块
    // 判断像素中心是否在三角形内
    float2 p = float2(pixel) + 0.5;
    float w0 = edgeFunction(v1, v2, p) / area;
    float w1 = edgeFunction(v2, v0, p) / area;
    float w2 = edgeFunction(v0, v1, p) / area;
    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        // 重心插值深度
        float depth = (tri.v0.z * invW0 * w0 + tri.v1.z * invW1 * w1 + tri.v2.z * invW2 * w2)
                      / (invW0 * w0 + invW1 * w1 + invW2 * w2);
        // 原子比较写入
        InterlockedMin(DepthUAV[pixel], asuint(depth));
    }
}
```
**同步与特性**：使用 `InterlockedMin` 或 `InterlockedMax` 保证深度测试的原子性，无需硬件深度单元。这种方案常用于 **Nanite 式虚拟几何**、粒子光栅化或早期 Z 剔除的替代。

---

### 3.4 后处理 Pass 示例：Bloom 降采样

**用途**：对一个 HDR 纹理进行多次 $2\times2$ 降采样，构建 Bloom 金字塔。使用 Compute Shader 可以一次 Dispatch 完成一级，并将下级的输出直接作为下一级的 SRV。

```hlsl
Texture2D<float4> Input : register(t0);
RWTexture2D<float4> Output : register(u0);

[numthreads(8, 8, 1)]
void CS_Downsample(uint3 id : SV_DispatchThreadID) {
    float4 sum = 0;
    // 简单的 2x2 box filter
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            sum += Input[id.xy * 2 + uint2(i, j)];
    Output[id.xy] = sum * 0.25;
}
```
连续多个 `Dispatch`，每次 Dispatch 前交换 SRV/UAV，并在之间插入 `ResourceBarrier`。此模式可轻松扩展到复杂的高斯模糊（水平/垂直分离 pass）。

---

## 4. Pass 间的依赖与同步屏障

CS‑Based 管线的资源流转必须严格遵守屏障规则，否则会出现数据竞争。

典型管线流程：
```
深度预Pass（CS软件光栅） → UAV 写深度
    ↓ UAV Barrier → SRV 深度
Hi-Z构建Pass（CS） → UAV 写 mips
    ↓ UAV Barrier → SRV
几何剔除Pass（CS） → 生成间接绘制参数 / 可见几何
    ↓ (无纹理依赖)
G-Buffer填充Pass（CS软件光栅） → UAV 写 G-BufferA, G-BufferB, 深度
    ↓ Barrier
Tile光照Pass（CS） → SRV 读 G-Buffer，UAV 写颜色
    ↓ Barrier
Bloom / ToneMapping Pass（CS） → 逐级 Dispatch
```

每个箭头都是一个 `ResourceBarrier`，例如从 UAV 到 SRV 的转换：
```cpp
D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
barrier.UAV.pResource = depthTex;
commandList->ResourceBarrier(1, &barrier);
```
在 CS 内部，若需要保证同一 Pass 内的写入对所有后续线程可见，则使用 `GroupMemoryBarrierWithGroupSync()` 处理共享内存；跨线程组无法同步，必须切分 Pass。

---

## 5. 总结：如何设计一个 CS‑Based 渲染 Pass

| 步骤 | 说明 |
|------|------|
| **1. 确定数据布局** | 输入是 Buffer 还是 Texture；输出是 UAV。 |
| **2. 划分线程组** | 按像素、Tile、三角形或实例划分。 |
| **3. 编写 Shader** | 利用 `SV_DispatchThreadID`、`SV_GroupID` 等定位数据；若需要协作，使用 `groupshared` 内存。 |
| **4. Dispatch 调用** | `Dispatch(x, y, z)` 或 `DispatchIndirect`。 |
| **5. Barrier 衔接** | 在调度下一个 Pass 前插入适当的资源屏障。 |
| **6. 调优** | 关注共享内存访问冲突、原子操作开销、寄存器压力、Occupancy。 |

最终，CS‑Based 管线将硬件光栅器看作一个可选的加速器，而不再是唯一绘图手段。这使得渲染器可以灵活实现 **自定义光栅化**、**细粒度剔除** 和 **高度并行的后处理**，充分发挥现代 GPU 的计算能力。