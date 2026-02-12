# UAV支持原子操作

### 🧑‍💻 User

为了在cs中读写texture，我们需要通过uav读写显存，那在uav中是否支持原子操作呢？

---

### 🤖 Assistant

简短的回答是：**是的，UAV（Unordered Access View）完全支持原子操作，但是对 Texture 的格式（Format）有严格的限制。**

在 Compute Shader (CS) 中对 `RWTexture` 进行原子操作是图形编程中的常见需求（例如计算直方图、写入非顺序的像素列表等）。

以下是详细的技术细节、限制以及代码示例。

### 1. 核心限制：纹理格式 (Texture Format)

并不是所有的纹理格式都支持原子操作。硬件和图形 API（DirectX 11/12, Vulkan）通常只保证以下格式的原生原子操作支持：

*   **必须是 32 位整数格式**：
    *   `DXGI_FORMAT_R32_UINT`
    *   `DXGI_FORMAT_R32_SINT`

这意味着你不能直接对 `R8G8B8A8_UNORM` 或 `R16_FLOAT` 这样的纹理调用原子函数。

<details>
<summary><strong>点击展开：关于 64 位和浮点原子操作的支持情况</strong></summary>

虽然 $32$ 位整数是标准，但在现代硬件和 API 中有扩展支持：

1.  **64 位原子操作 ($64$-bit Atomics):**
    *   许多现代 GPU 支持 `R64_UINT` 的原子操作（通常通过 `InterlockedCompareExchange64` 等）。
    *   需要 Shader Model 6.6+ 或特定扩展。

2.  **浮点原子操作 (Float Atomics):**
    *   **原生支持：** 较新的显卡（NVIDIA Maxwell+, AMD GCN+）和 API（DX12, Vulkan）支持 `R32_FLOAT` 的原子加法 (`InterlockedAdd`)。
    *   **旧方法 (CAS 循环)：** 如果硬件不支持原生浮点原子加法，可以使用 `InterlockedCompareExchange` 实现一个自旋锁（CAS Loop）来模拟浮点原子操作，但性能较差。

</details>

### 2. 支持的原子函数 (HLSL)

在 HLSL 中，你可以对 `RWTexture1D`, `RWTexture2D`, `RWTexture3D` 使用 `Interlocked` 系列函数。

常用的原子操作包括：
*   `InterlockedAdd`: 加法
*   `InterlockedMin` / `InterlockedMax`: 最小值/最大值
*   `InterlockedAnd` / `InterlockedOr` / `InterlockedXor`: 位运算
*   `InterlockedExchange`: 交换值（写入并返回旧值）
*   `InterlockedCompareStore`: 比较并写入
*   `InterlockedCompareExchange`: 比较并交换 (CAS)

### 3. 代码示例

假设我们需要在一个 `RWTexture2D` 上统计某些事件发生的次数（例如热力图生成）。

**C++ / Host 端设置：**
创建 Texture 时，Format 必须设置为 `DXGI_FORMAT_R32_UINT`。

**HLSL Shader 代码：**

```hlsl
// 定义一个支持原子操作的 UAV 纹理，类型必须是 uint 或 int
RWTexture2D<uint> g_Heatmap : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    // 假设我们通过某种逻辑计算出了要写入的坐标
    uint2 writePos = uint2(dispatchThreadId.x, dispatchThreadId.y);
  
    // 边界检查
    uint width, height;
    g_Heatmap.GetDimensions(width, height);
    if (writePos.x >= width || writePos.y >= height) return;

    // 原子加法：将对应位置的值加 1
    // 注意：InterlockedAdd 不返回值
    InterlockedAdd(g_Heatmap[writePos], 1);

    // 如果你需要获取加法之前的值（Original Value），可以使用这个重载：
    // uint originalValue;
    // InterlockedAdd(g_Heatmap[writePos], 1, originalValue);
}
```

### 4. 常见问题与解决方案

#### 问题 A：我想对 `R8G8B8A8` 纹理做原子操作怎么办？
你不能直接做。常用的 Workaround 是：
1.  **Format Aliasing (Typeless):** 创建纹理资源为 `DXGI_FORMAT_R32_TYPELESS`。
2.  **UAV 转换:** 创建 UAV 时，指定格式为 `DXGI_FORMAT_R32_UINT`。
3.  **Shader 中:** 将颜色 Pack 成一个 $32$ 位的 `uint`，然后使用 `InterlockedExchange` 或 `InterlockedCompareExchange` 写入。这通常用于 Order Independent Transparency (OIT) 链表构建。

#### 问题 B：原子操作太慢了怎么办？
显存（VRAM）上的原子操作存在**高延迟**和**竞争（Contention）**问题。如果多个线程同时写入同一个像素，性能会急剧下降。
*   **优化方案：** 尽可能使用 **GroupShared Memory (LDS)** 进行原子操作。
    1.  在 `groupshared` 数组中进行原子累加（速度非常快）。
    2.  在 Shader 结束前，仅由一个线程将 `groupshared` 的结果原子写入到全局显存（UAV）中。

**LDS 优化示例：**

```hlsl
groupshared uint lds_histogram[256];

[numthreads(256, 1, 1)]
void CS_Histogram(uint3 groupThreadId : SV_GroupThreadID, uint3 dispatchId : SV_DispatchThreadID)
{
    // 1. 初始化 LDS
    lds_histogram[groupThreadId.x] = 0;
    GroupMemoryBarrierWithGroupSync();

    // 2. 读取纹理并进行 LDS 原子操作 (极快)
    uint colorValue = ReadTexture(dispatchId.xy);
    InterlockedAdd(lds_histogram[colorValue], 1);

    GroupMemoryBarrierWithGroupSync();

    // 3. 将结果合并到全局 UAV (减少了全局竞争)
    if (lds_histogram[groupThreadId.x] > 0)
    {
        InterlockedAdd(g_GlobalHistogram[groupThreadId.x], lds_histogram[groupThreadId.x]);
    }
}
```

### 总结
*   **支持吗？** 支持。
*   **关键条件：** 纹理格式通常必须是 `R32_UINT` 或 `R32_SINT`。
*   **性能建议：** 全局显存原子操作昂贵，尽量先在 `groupshared` 内存中聚合数据。