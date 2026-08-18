# Unity6000替换UniTask建议

### 🧑‍💻 User

unity6000既然mono支持原生async Task<T>了，那怎么替换已有的unitask写法？

---

### 🤖 Assistant

**可以替换，但通常不建议“无脑全量替换”。** Unity 6（6000）确实显著改善了 `async`/`await` 和原生 `Task`/`Task<T>` 的支持（Mono 下更友好），`Awaitable` 也更成熟，但 UniTask 在性能、PlayerLoop 控制、Delay 语义、Unity 扩展和取消上仍有明显优势。

### 核心结论与建议优先级

1. **优先评估是否真的需要替换**
   - UniTask 零/低分配、可精确挂到 Update/FixedUpdate/LateUpdate 等、`Delay` 支持 `DeltaTime`/`Unscaled`/`Realtime`、大量 Unity API 扩展（AsyncOperation、Addressables、uGUI 等）、`.Forget()` 和取消更顺手。
   - 原生 `Task` 分配更多，时间语义不同，Unity 集成弱一些。
   - **更推荐的官方轻量替代是 `UnityEngine.Awaitable` / `Awaitable<T>`**（设计理念接近 UniTask，性能好，已内置）。

2. **什么时候适合换成原生 Task**
   - 想减少第三方依赖、代码偏通用 .NET、主要跑在编辑器/Mono、对分配不敏感、不需要复杂 Unity 时间/扩展。
   - 或新写跨平台逻辑。

3. **混合使用完全可行**
   UniTask 提供 `AsTask()` / `AsUniTask()`，可渐进迁移。

### 基础替换对照

```csharp
// ========== Before (UniTask) ==========
using Cysharp.Threading.Tasks;

async UniTask<int> LoadAsync(CancellationToken ct = default)
{
    await UniTask.Delay(1000, cancellationToken: ct);      // 或 DelayFrame、Yield 等
    await UniTask.Yield();                                  // 或 NextFrame()
    await someAsyncOperation;                               // 靠扩展
    return 42;
}

// 调用
LoadAsync().Forget();
await UniTask.WhenAll(t1, t2);

// ========== After (原生 Task) ==========
using System.Threading.Tasks;
using System.Threading;

async Task<int> LoadAsync(CancellationToken ct = default)
{
    await Task.Delay(1000, ct);                             // 注意：见下方时间语义差异
    await Task.Yield();                                     // 回到当前 SynchronizationContext（主线程下一帧左右）
    // AsyncOperation 见下文
    return 42;
}

// 调用（Fire-and-forget 要小心异常）
_ = LoadAsync();                                            // 简单写法，异常可能 unobserved
// 或封装一个安全 Forget
await Task.WhenAll(t1, t2);
```

**类型映射简表**

| UniTask              | 原生 / 推荐替代                  | 备注 |
|----------------------|----------------------------------|------|
| `UniTask`            | `Task` 或 `Awaitable`            | 结构体 vs 类，分配差异大 |
| `UniTask<T>`         | `Task<T>` 或 `Awaitable<T>`      | 同上 |
| `UniTaskVoid`        | `async void` 或返回 `Task` 后 `_ = ` | `async void` 仅限事件等顶层 |
| `UniTask.Delay`      | `Task.Delay` 或 `Awaitable.WaitForSecondsAsync` | **时间语义不同** |
| `UniTask.Yield` / `NextFrame` | `Task.Yield()` 或 `Awaitable.NextFrameAsync()` | 推荐后者更明确 |
| `UniTask.WhenAll/Any`| `Task.WhenAll/Any`               | 基本等价 |
| `.Forget()`          | `_ = task;` + 异常处理           | 见下方 |
| `.WithCancellation`  | 传 `CancellationToken` 或 `WaitAsync(ct)` | 标准 .NET 方式 |
| AsyncOperation 等扩展| `Awaitable.FromAsyncOperation` 或 TCS | 见下方示例 |

### 关键差异与正确写法

**1. Delay 时间语义（最容易踩坑）**
- `Task.Delay`：系统时钟 / 墙钟时间，**不受 `Time.timeScale` 影响**。
- UniTask 默认更贴合游戏逻辑，且可选 `DelayType`。

需要 scaled 时间时自己写或用 Awaitable：

```csharp
// 简易 scaled delay（Task 版）
async Task DelayScaledAsync(float seconds, CancellationToken ct = default)
{
    float start = Time.time;
    while (Time.time - start < seconds)
    {
        ct.ThrowIfCancellationRequested();
        await Task.Yield();
    }
}

// 更推荐官方
await Awaitable.WaitForSecondsAsync(seconds);   // 查看当前文档确认 timeScale 行为
await Awaitable.NextFrameAsync();
```

**2. Fire-and-forget 与异常**

```csharp
// 推荐封装
public static async void Forget(this Task task)
{
    try { await task; }
    catch (Exception e) when (e is not OperationCanceledException)
    {
        Debug.LogException(e);
    }
}

// 使用
LoadAsync().Forget();
// 或
_ = LoadAsync().ContinueWith(t =>
{
    if (t.IsFaulted) Debug.LogException(t.Exception.InnerException ?? t.Exception);
}, TaskScheduler.FromCurrentSynchronizationContext());
```

**3. Unity 异步操作（AsyncOperation / UnityWebRequest / Addressables 等）**

```csharp
// 方式 A：官方 Awaitable（推荐）
var op = someRequest.SendWebRequest();
await Awaitable.FromAsyncOperation(op);
// 或很多新 API 直接返回 Awaitable

// 方式 B：纯 Task + TaskCompletionSource
static Task AsTask(this AsyncOperation op)
{
    if (op.isDone) return Task.CompletedTask;
    var tcs = new TaskCompletionSource<bool>();
    op.completed += _ => tcs.TrySetResult(true);
    return tcs.Task;
}

// 使用
await request.SendWebRequest().AsTask();
```

Addressables、Resource 等同理，用 handle 的 completed 事件 + TCS，或查是否已有官方 async 封装。

**4. 取消**

```csharp
async Task FooAsync(CancellationToken ct = default)
{
    // 传给支持的方法
    await Task.Delay(1000, ct);
    // 或
    await someTask.WaitAsync(ct);
    ct.ThrowIfCancellationRequested();
}
```

`CancellationTokenSource` 用法与 UniTask 时代基本相同（记得 Dispose）。

**5. 切换线程（后台 ↔ 主线程）**

```csharp
// 纯 Task 较繁琐，通常配合 Task.Run + 回到主线程
await Task.Run(() => HeavyWork());          // 后台
await Task.Yield();                         // 或自定义 Post 回主线程

// 更干净：用 Awaitable
await Awaitable.BackgroundThreadAsync();
// ... 后台工作
await Awaitable.MainThreadAsync();
```

### 迁移步骤建议

1. 备份 / 分支。
2. 全局搜索 `UniTask`、`Cysharp.Threading.Tasks`、`.Forget`、`DelayFrame`、`NextFrame`、`ToUniTask` 等。
3. 先把返回类型和 `await` 点改成 `Task`/`Task.Yield`/`Task.Delay`，编译通过。
4. 重点检查：Delay 是否需要 timeScale、所有 fire-and-forget、AsyncOperation 扩展点、频繁 await 的热点（看 Profiler 分配）。
5. 删除 UniTask 包（或暂时保留做互转）。
6. 跑一遍功能 + 性能测试（尤其是加载、UI、循环逻辑）。
7. 考虑把高频路径改成 `Awaitable` 而不是裸 `Task`。

### 更推荐的路径：迁到 Awaitable（接近 UniTask）

很多项目最终会发现：

```csharp
async Awaitable<int> LoadAsync()
{
    await Awaitable.WaitForSecondsAsync(1f);
    await Awaitable.NextFrameAsync();
    await Awaitable.FromAsyncOperation(op);
    return 42;
}
```

语法和心智模型都更接近原来的 UniTask，同时去掉第三方依赖，性能也更好。

### 什么时候继续留着 UniTask？

- 已有大量 `DelayFrame`、PlayerLoop Timing、自定义扩展、DOTween/Addressables 插件用法。
- 对 GC 和帧时间极度敏感。
- 团队已经很熟，迁移成本 > 收益。

UniTask 在 Unity 6 上依然工作得很好，完全可以继续用。

---

**需要我针对你现有的具体写法（Delay 系列、某个 AsyncOperation、事件回调里的 UniTaskVoid、WhenAll 组合、取消链路等）给出直接改写后的代码吗？** 贴几段典型代码，我按「纯 Task」和「Awaitable」两个版本给你对照改。