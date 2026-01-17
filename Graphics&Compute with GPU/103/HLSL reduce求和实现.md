# HLSL reduce求和实现

### 🧑‍💻 User

hlsl compute shader如何实现reduce求和操作？

---

### 🤖 Assistant

在HLSL Compute Shader中实现Reduce（规约）求和操作，核心思想是利用**并行树形规约（Parallel Tree Reduction）**。

由于GPU是大规模并行的，我们不能像CPU那样写一个简单的`for`循环去累加，那样效率极低。我们需要利用**组共享内存（groupshared memory）**在每个线程组内部进行部分求和，然后将结果级联。

以下是实现Reduce求和的详细步骤和代码。

### 核心原理

1.  **加载（Load）：** 每个线程从全局显存（Global Memory）读取数据到组共享内存（Shared Memory）。
2.  **规约（Reduce）：** 在共享内存中进行折半归约。
    *   第一轮：前50%的线程将后50%的数据加到自己身上。
    *   第二轮：前25%的线程将后25%的数据加到自己身上。
    *   以此类推，直到剩下第0号线程持有整个组的总和。
3.  **输出（Store）：** 线程组的第0号线程将结果写入输出缓冲区。

### HLSL 代码实现

假设我们有一个一维的`StructuredBuffer<float>`作为输入，我们需要计算总和。

这里定义一个线程组大小（BLOCK_SIZE）为 512。

```hlsl
// 定义线程组大小，必须是2的幂次方，通常为 256 或 512
#define BLOCK_SIZE 512

// 输入缓冲区
StructuredBuffer<float> g_Input : register(t0);

// 输出缓冲区 (存储每个线程组的部分和)
RWStructuredBuffer<float> g_Output : register(u0);

// 组共享内存：这是同一个线程组内所有线程可见的高速缓存
groupshared float sharedMem[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void ReduceSumCS(
    uint3 DTid : SV_DispatchThreadID, // 全局线程ID
    uint GI : SV_GroupIndex,          // 组内线程索引 (0 ~ BLOCK_SIZE-1)
    uint3 Gid : SV_GroupID            // 线程组ID
)
{
    // ---------------------------------------------------------
    // 1. 加载数据到共享内存
    // ---------------------------------------------------------
    // 注意：需要检查边界，防止读取越界
    uint inputIndex = DTid.x;
  
    // 如果你的输入数据长度可能小于 Dispatch * BLOCK_SIZE，这里需要加判断
    // 假设 BufferSize 通过 ConstantBuffer 传入，或者简单起见直接读取
    // float val = (inputIndex < g_DataSize) ? g_Input[inputIndex] : 0.0f;
  
    // 这里简化为直接读取
    sharedMem[GI] = g_Input[inputIndex];

    // 必须同步，确保组内所有线程都完成了读取操作
    GroupMemoryBarrierWithGroupSync();

    // ---------------------------------------------------------
    // 2. 在共享内存中进行树形规约
    // ---------------------------------------------------------
    // 这是一个标准的折半归约循环
    // s 代表步长 (stride)，每次循环步长减半
  
    // unroll 属性提示编译器展开循环以提高性能
    [unroll]
    for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
        if (GI < s)
        {
            sharedMem[GI] += sharedMem[GI + s];
        }
        // 每一轮加法完成后，必须同步，防止数据竞争
        GroupMemoryBarrierWithGroupSync();
    }

    // ---------------------------------------------------------
    // 3. 将本组的结果写回全局内存
    // ---------------------------------------------------------
    // 此时 sharedMem[0] 保存了当前线程组内 512 个数字的总和
    if (GI == 0)
    {
        g_Output[Gid.x] = sharedMem[0];
    }
}
```

### 关键细节解析

#### 1. 为什么使用 `groupshared`？
全局内存（显存）访问速度很慢。如果每次加法都读写显存，性能会非常差。`groupshared` 内存位于GPU芯片上的L1缓存级别，速度极快，适合线程间交换数据。

#### 2. `GroupMemoryBarrierWithGroupSync()` 的作用
这是一个同步屏障。它强制同一个线程组内的所有线程都执行到这一行代码后，才能继续往下执行。
*   在加载阶段：防止有的线程还没把数据写入 `sharedMem`，别的线程就开始读取并求和了。
*   在规约循环中：防止上一轮计算还没结束，下一轮就开始了。

#### 3. 处理大规模数据（Multi-pass Approach）
上述 Shader 只能将 **一个线程组**（例如512个元素）归约为 1 个数值。
如果你的输入数组有 $N$ 个元素（例如 $N = 1,000,000$），你不能在一个 Pass 中完成。

**标准流程是多重 Dispatch（Ping-Pong）：**

1.  **Pass 1:** 输入 $N$ 个元素。Dispatch $N/512$ 个线程组。
    *   输出结果是一个大小为 $N/512$ 的缓冲区（每个元素是原数组512个数字的和）。
2.  **Pass 2:** 将 Pass 1 的输出作为输入。Dispatch $(N/512)/512$ 个线程组。
    *   输出更小的缓冲区。
3.  **重复:** 直到剩下的元素数量足够小（比如为1，或者CPU直接读取最后几十个数字求和）。

### 进阶优化技巧

#### 1. 优化内存加载（每个线程处理两个元素）
为了提高利用率，通常让一个线程加载 2 个数值并在存入共享内存前先加一次。这样只需要一半的线程数即可处理同样的数据量，或者同样的线程数处理双倍数据。

```hlsl
// 每一个线程读取两个数： i 和 i + BLOCK_SIZE
unsigned int idx = DTid.x;
unsigned int idx2 = idx + BLOCK_SIZE; // 这种偏移需要调整Dispatch逻辑

// 读取两个数相加存入 shared
float val1 = g_Input[idx];
float val2 = g_Input[idx2]; // 需做边界检查
sharedMem[GI] = val1 + val2;
GroupMemoryBarrierWithGroupSync();
// ... 后续逻辑不变
```

#### 2. 消除最后的同步（Loop Unrolling）
当 `stride` 小于 32（Warp/Wavefront 大小）时，在 NVIDIA 和 AMD 显卡上，同一个 Warp 内的线程是隐式同步的（SIMD同步）。虽然 HLSL 标准要求 `GroupMemoryBarrierWithGroupSync`，但为了极致性能，许多开发者会手动展开最后几层循环，并移除 Barrier（注意：这依赖于硬件特性，属于高级优化）。

#### 3. InterlockedAdd (仅限整数)
如果你的求和对象是 `int` 或 `uint`，你可以偷懒：
在 Shader 最后，不写入 `g_Output[Gid.x]`，而是直接对一个全局单一地址进行原子加法：
```hlsl
if (GI == 0) {
    InterlockedAdd(g_FinalResult[0], (uint)sharedMem[0]); // 浮点数不支持原生的 InterlockedAdd
}
```
**注意：** HLSL **不**支持浮点数的 `InterlockedAdd`。如果必须对浮点数做原子加，需要使用 `InterlockedCompareExchange` 编写自旋锁，这通常比上面的多 Pass 方法慢且复杂，不推荐。

### 总结

实现 HLSL Reduce 求和的最佳实践是：
1.  使用 `groupshared` 内存缓存数据。
2.  使用 `for` 循环配合 `GroupMemoryBarrierWithGroupSync` 进行 $O(\log N)$ 复杂度的折半归约。
3.  在 Host 端（C++/C#）编写循环逻辑，多次 Dispatch Shader，直到将数据缩减为 1 个值。