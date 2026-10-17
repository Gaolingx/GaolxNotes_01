# 寻找 NET GC吞吐量测试代码

## 🧑‍💻 User

我记得Microsoft .net文档有测试gc吞吐的c#代码，具体是哪个

---
## 😀 默认助手 (7)

I'll look this up in the .NET docs.

The main GC performance page doesn't have that sample. Let me search the docs directly.

The docs search page is JS-rendered, so let me query the search API directly.

Let me search the .NET docs source repo for the throughput-measuring sample.

Let me search the web for the exact sample.

Let me search from the angle a Chinese developer would encounter it.

Sourcegraph works. Let me use it to locate the sample by file.

我把 Microsoft Learn 的 .NET 文档（`dotnet/docs` 仓库全文 + Learn 搜索）都翻了一遍，先给结论：**Learn 上没有任何一页是用 "GC throughput" 这个说法、并且配一段专门"测 GC 吞吐"的 C# 程序的**（"GC 吞吐量" 在 learn 上也是零命中）。你记忆里的东西，几乎肯定是下面几篇中的一篇——它们才是 .NET 文档里"用 C# 测 GC 性能"的代码来源。

<details>
<summary><b>① 最可能：垃圾回收和性能（Garbage collection and performance）</b></summary>

`https://learn.microsoft.com/zh-cn/dotnet/standard/garbage-collection/performance`

- 这是官方唯一系统讲"GC 性能"的页：`% Time in GC`（GC 时间占比）、`Allocated Bytes/second`、GC ETW 事件、`!eeheap`/`!dumpheap` 等。
- 里面**确实有一段 C# 代码**：
  ```csharp
  GC.Collect();
  GC.WaitForPendingFinalizers();
  GC.Collect();
  ```
- 该页"判断 GC 是否拖慢应用"的整套算法（% Time in GC ≈ 100% − GC 吞吐量），正是很多人记忆里"测 GC 吞吐"的出处。

</details>

<details>
<summary><b>② 唯一"可编程测量 GC"的 C# 代码：Windows 上的大型对象堆（LOH）</b></summary>

`https://learn.microsoft.com/zh-cn/dotnet/standard/garbage-collection/large-object-heap` → 章节 **"Collect performance data for the LOH"**

```csharp
PerformanceCounter performanceCounter = new()
{
    CategoryName = ".NET CLR Memory",
    CounterName = "Large Object Heap size",
    InstanceName = ""
};
Console.WriteLine(performanceCounter.NextValue());
```

- 这是 GC 相关文档里**唯一**用 C# 直接读 **.NET CLR Memory** 性能计数器的示例；把 `CounterName` 换成 `"% Time in GC"` 或 `"Allocated Bytes/sec"`，就是标准的"测 GC 吞吐"代码。
- 如果你记得的代码里有 `PerformanceCounter` / `NextValue()`，就是这一页。

</details>

<details>
<summary><b>③ 唯一的完整 C# GC 监控程序：垃圾回收通知</b></summary>

`https://learn.microsoft.com/zh-cn/dotnet/standard/garbage-collection/notifications`

- 一整段可运行程序：`GC.RegisterForFullGCNotification` + `WaitForFullGCProc` 循环，并用 `GC.CollectionCount(2)` 记录回收次数（`Gen 2 collection count: …`）。
- 如果你记得的是"完整的 `Main` + 循环 + 打印各代 GC 次数"，就是这一页（源码在 `dotnet/docs` 的 `samples/snippets/csharp/VS_Snippets_CLR/GCNotification/cs/Program.cs`）。

</details>

<details>
<summary><b>④ 有压测 GC 的 C# 代码 + 吞吐量数据：ASP.NET Core 内存管理</b></summary>

`https://learn.microsoft.com/zh-cn/aspnet/core/performance/memory`

- 页面用一组 C# 端点不停制造 GC（`bigstring`、`staticstring`、`loh/{size}`、`array`、`pooledarray`…），配图表分析 GC 对吞吐的影响，原文有"**需要更多的 CPU，吞吐量会下降约 50%**"。
- 还指向示例程序 **MemoryLeak**（`github.com/sebastienros/memoryleak`），它会"实时采集内存和 GC 数据"。
- 如果你记得的代码是 `[HttpGet("...")]` 这类端点 + GC 图表，就是这一篇。

</details>

<details>
<summary><b>⑤ 若其实是博客/源码仓库（非 Learn 页）：GCMemoryInfo 测"每请求 GC 停顿"</b></summary>

- `https://devblogs.microsoft.com/dotnet/the-updated-getgcmemoryinfo-api-in-net-5-0-and-how-it-can-help-you/`（Maoni Stephens，.NET GC 架构师）。
- 核心是 `GC.GetGCMemoryInfo().PauseTimePercentage`（"从进程启动至今的 % pause time in GC"，就是吞吐视角的指标），文中那段按请求统计 GC 影响的示例代码用 `Stopwatch` + `GC.CollectionCount(0)` 计算 `GC impact / % requests affected by GC`。文档侧对应 `dotnet/core/diagnostics/built-in-metrics-runtime` 的 `dotnet.gc.pause.time`。

</details>

**帮我确认一下就能锁定是哪一篇**：你记得的那段代码里有没有出现——

| 线索 | 对应文档 |
|---|---|
| `PerformanceCounter` / `.NET CLR Memory` | ② LOH 页 |
| `WaitForFullGCComplete` / `GC.CollectionCount(2)` | ③ 垃圾回收通知 |
| `[HttpGet(...)]` / `ArrayPool` | ④ ASP.NET Core 内存管理 |
| `GetGCMemoryInfo()` / `PauseTimePercentage` / `Stopwatch` 包住一次请求 | ⑤ Maoni 的博客 |
| 只有 `GC.Collect(); GC.WaitForPendingFinalizers();` | ① 垃圾回收和性能 |

如果你要的是"能直接跑、算出 GC 吞吐率"的现成程序，Learn 上并没有（官方只给方法学 + 计数器/ETW/`GCMemoryInfo`），我也可以按 `% Time in GC = GC 暂停总时长 / 进程总时长` 给你写一份完整的 `dotnet run` 示例。
