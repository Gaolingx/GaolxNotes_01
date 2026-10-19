# GC压力大原因分析

## 🧑‍💻 User

这段c#代码为什么gc压力非常大？
```cs
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.Runtime;
using System.Runtime.CompilerServices;
using Microsoft.Diagnostics.NETCore.Client;
using Microsoft.Diagnostics.Tracing;
using Microsoft.Diagnostics.Tracing.Analysis;
using Microsoft.Diagnostics.Tracing.Parsers;
class Program
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static void Main()
    {
        var pauses = new List<double>();

        var client = new DiagnosticsClient(Environment.ProcessId);
        EventPipeSession eventPipeSession = client.StartEventPipeSession([new("Microsoft-Windows-DotNETRuntime",
            EventLevel.Informational, (long)ClrTraceEventParser.Keywords.GC)], false);
        var source = new EventPipeEventSource(eventPipeSession.EventStream);

        source.NeedLoadedDotNetRuntimes();
        source.AddCallbackOnProcessStart(proc =>
        {
            proc.AddCallbackOnDotNetRuntimeLoad(runtime =>
            {
                runtime.GCEnd += (p, gc) =>
                {
                    if (p.ProcessID == Environment.ProcessId)
                    {
                        pauses.Add(gc.PauseDurationMSec);
                    }
                };
            });
        });

        GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);
        GC.WaitForPendingFinalizers();
        GC.WaitForFullGCComplete();
        Thread.Sleep(5000);

        new Thread(() => source.Process()).Start();

        pauses.Clear();

        Test(22);

        source.StopProcessing();
        Console.WriteLine($"Max GC Pause: {pauses.Max()}ms");
        Console.WriteLine($"Average GC Pause: {pauses.Average()}ms");
        pauses.Sort();
        Console.WriteLine($"P99.9 GC Pause: {pauses.Take((int)(pauses.Count * 0.999)).Max()}ms");
        Console.WriteLine($"P99 GC Pause: {pauses.Take((int)(pauses.Count * 0.99)).Max()}ms");
        Console.WriteLine($"P95 GC Pause: {pauses.Take((int)(pauses.Count * 0.95)).Max()}ms");
        Console.WriteLine($"P90 GC Pause: {pauses.Take((int)(pauses.Count * 0.9)).Max()}ms");
        Console.WriteLine($"P80 GC Pause: {pauses.Take((int)(pauses.Count * 0.8)).Max()}ms");

        using (var process = Process.GetCurrentProcess())
        {
            Console.WriteLine($"Peak WorkingSet: {process.PeakWorkingSet64} bytes");
        }
    }

    static void Test(int size)
    {
        var bt = new BinaryTrees.Benchmarks();
        var sw = Stopwatch.StartNew();
        bt.ClassBinaryTree(size);
        Console.WriteLine($"Elapsed: {sw.Elapsed.TotalMilliseconds}ms");
    }

}

public class BinaryTrees
{
    class ClassTreeNode
    {
        class Next { public required ClassTreeNode left, right; }
        readonly Next? next;
        ClassTreeNode(ClassTreeNode left, ClassTreeNode right) =>
            next = new Next { left = left, right = right };
        public ClassTreeNode() { }
        internal static ClassTreeNode Create(int d)
        {
            return d == 1 ? new ClassTreeNode(new ClassTreeNode(), new ClassTreeNode())
                          : new ClassTreeNode(Create(d - 1), Create(d - 1));
        }

        internal int Check()
        {
            int c = 1;
            var current = next;
            while (current != null)
            {
                c += current.right.Check() + 1;
                current = current.left.next;
            }
            return c;
        }
    }

    public class Benchmarks
    {
        const int MinDepth = 4;
        public int ClassBinaryTree(int maxDepth)
        {
            var longLivedTree = ClassTreeNode.Create(maxDepth);
            var nResults = (maxDepth - MinDepth) / 2 + 1;
            for (int i = 0; i < nResults; i++)
            {
                var depth = i * 2 + MinDepth;
                var n = 1 << maxDepth - depth + MinDepth;

                var check = 0;
                for (int j = 0; j < n; j++)
                {
                    check += ClassTreeNode.Create(depth).Check();
                }
            }

            return longLivedTree.Check();
        }
    }
}
```

---
## 😀 默认助手 (7)

I'll investigate this code carefully, including checking whether it matches a known issue.

I've got the code analyzed structurally. Let me quantify the allocation volume precisely to ground my answer.

## 结论

**这不是 EventPipe/事件回调的锅**（那部分开销相对很小）——真正的原因是 `Test(22)` 这个 binary-trees 基准本身：它会在几秒内分配 **约 20 亿个对象、约 53 GB 的短命垃圾**，再叠加两个放大因素：

1. `ClassTreeNode` + `Next` **每个内部节点分配两个对象**；
2. 一棵 **335 MB 的"长寿树"被晋升到 Gen2**，导致又贵又长的 Gen2/晋升回收。

GC 压力巨大是这个 workload 的必然结果，不是 bug。

---

## 主因一：基准是纯分配型 workload

`ClassBinaryTree(22)`（`MinDepth=4`, `maxDepth=22`）的分配量是压倒性的。深度 $d$ 的树有 $2^{d+1}-1$ 个 `ClassTreeNode` 和 $2^{d}-1$ 个 `Next`；第 $i$ 层创建 $n = 2^{\,22-d+4}$ 棵树。我按代码里的公式精确算了一下：

| maxDepth | 层数 | `ClassTreeNode` 个数 | `Next` 个数 | 对象总数 | 分配字节（8 字节引用估计） |
|---|---|---|---|---|---|
| 20（常见参数） | 9 | 3.01 亿 | 1.50 亿 | ~4.5 亿 | ~12 GB |
| **22（本代码）** | 10 | **13.4 亿** | **6.65 亿** | **~20 亿** | **~53 GB** |

关键点：**每一层的工作量几乎完全相同**（`n × 每棵树的节点数 ≈ 2²⁷`），所以总工作量 ≈ `层数 × 2²⁷`。把 maxDepth 从 20 提到 22，层数变多、每层规模翻 4 倍，总共约 **4 倍**。

这些对象几乎全是"立即死亡"的：`Create(depth)` 造出来后马上被 `Check()` 读完就丢。这种极低的存活率 + 极高分配速率 = 每秒几百到几千次 Gen0，GC 压力当然爆表。**这就是 binary-trees 基准存在的意义**。

> `1 << maxDepth - depth + MinDepth` 看着可疑，但 C# 里加法优先级高于移位，它等价于 `1 << (maxDepth - depth + MinDepth)`，是基准的正确写法，不是 bug。

---

## 主因二：`Next` 让每个内部节点变成两个对象

```cs
class ClassTreeNode
{
    class Next { public required ClassTreeNode left, right; }  // 额外一个对象
    readonly Next? next;
    ClassTreeNode(ClassTreeNode left, ClassTreeNode right) =>
        next = new Next { left = left, right = right };        // 每次内部节点都 new
    public ClassTreeNode() { }                                  // 叶子：只有 1 个对象
}
```

标准 benchmark 里一个节点就是**一个对象**（`left`/`right` 直接做字段）。这里为了把 `next` 做成 `readonly`、并让叶子用 `null` 表示，硬生生拆成两层：

- 内部节点：`ClassTreeNode`(24B) + `Next`(32B) = 2 个对象、56B
- 叶子：`ClassTreeNode`(24B) = 1 个对象

代价：

- **对象总数 +50%**（20 亿 vs 1.34 亿… 准确说 20 亿 vs 13.4 亿）；
- **分配字节 ≈ +25%**；
- `Check()` 里多一次指针跳转（`current.left.next`），缓存局部性更差、遍历更慢 → 在同样的观测窗口里观察到更多次 GC。

这个 `Next` 完全是可去掉的。

---

## 主因三：一棵 335 MB 的长寿树被晋升到 Gen2

```cs
var longLivedTree = ClassTreeNode.Create(maxDepth);   // 在循环前创建
...
return longLivedTree.Check();                          // 循环后才使用
```

因为它"循环后还要用"，JIT 必须让它在整个循环期间保持存活，于是：

- `Create(22)` 自身就是 **8,388,607 个 `ClassTreeNode` + 4,194,303 个 `Next` ≈ 12.6M 对象 / ≈ 335 MB**；
- 它在循环中被反复的 Gen0 GC 命中 → 一路晋升 Gen1 → Gen2；
- 之后**每一次 Gen1/Gen2 回收都要标记/搬移这 12.6M 个互相引用的对象**——这正是你看到的 `Max GC Pause` / P99.9 偏大的来源；
- 同时它把工作集推高（对应 `PeakWorkingSet64`）。

这部分是"故意设计"来考验 GC 的（标准基准就这么写），但**它比循环里所有短命垃圾加起来都更贵**，因为它会反复被扫描。

---

<details>
<summary>次要与非因素（点击展开）</summary>

### 次要因素

- **同进程采样的轻微反馈**：`EventPipeEventSource` + `TraceEvent.Analysis` 在解析线程上会为每个事件分配对象。但这些分配相对 53 GB 只是零头，GC 事件也只有几千个量级，所以**不是主因**，但确实会让数字略微偏高。把采样器放在**独立进程**会更干净。
- **`[MethodImpl(AggressiveOptimization)]`、`Stopwatch`、LINQ `Take`、字符串插值**：都可忽略。

### 不是因素

- `GC.Collect(GC.MaxGeneration, Aggressive, true, true)`：这只是**测量前一次性**强制一次全阻塞 GC，不影响之后的持续压力。
- `Keywords.GC`：只开 GC 事件，**不含**分配采样/`GCHeapCollect` 之类的每分配开销。
- `source.Process()` 的 `GCEnd` 回调：回调本身只做 `pauses.Add`，很轻。

### 顺带一个正确性 bug（与 GC 无关）

```cs
new Thread(() => source.Process()).Start();
pauses.Clear();        // 主线程
// 解析线程同时在执行 pauses.Add(...)   ← 无同步 → 数据竞争
```

`List<double>` 不是线程安全的，`Clear()` 与 `Add()` 并发可能丢数据甚至抛异常。生产代码里应加锁或用 `ConcurrentQueue<double>`。

</details>

---

## 怎么把 GC 压力降下来

```cs
// 1) 干掉 Next：一个节点一个对象
sealed class ClassTreeNode
{
    readonly ClassTreeNode? left, right;
    ClassTreeNode(ClassTreeNode? l, ClassTreeNode? r) { left = l; right = r; }
    internal static ClassTreeNode Create(int d) =>
        d == 1
            ? new ClassTreeNode(null, null)   // 需要区分叶子时可用"哨兵"或单独类型
            : new ClassTreeNode(Create(d - 1), Create(d - 1));
    internal int Check() { /* 直接读 left/right，少一次跳转 */ }
}
```

2. **降低 `maxDepth`**：22 → 20 直接砍掉约 3/4 的工作量（想复现"高 GC 压力"再调大）。
3. **别让那棵树一直长寿**：要么循环结束后立刻不再引用，要么干脆别保留（基准里保留它是为了压测 Gen2，业务代码通常不需要）。
4. **配置 GC**：这类 CPU/分配密集场景用 **Server GC + 后台 GC**（`<ServerGarbageCollection>true</ServerGarbageCollection>` / `DOTNET_gcServer=1`），Gen0 可并行回收，暂停显著下降；必要时用 `GCSettings.LatencyMode`。
5. **测量与负载分离**：把 EventPipe 采样放到**单独进程**采集目标进程，避免采样器自身的分配混进被测进程，也顺手修掉上面的并发问题；`List<double>` 换成线程安全容器。
6. 如果只是想压测"暂停时间"，用 `GC.TryStartNoGCRegion`/`GCSettings` 之类的工具，或直接读 `GC.GetTotalAllocatedBytes()` 而不是自己拆事件流。

一句话：**压力来自"分配了约 20 亿个对象"，其中 `Next` 让对象数多了 50%、那棵 335 MB 的长寿树让回收变贵**——监控代码只是测量者，不是肇事者。
