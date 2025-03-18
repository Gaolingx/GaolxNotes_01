# C#多线程与异步编程（五）

## 1. 简单任务

### 1.1 如何创建异步任务

- Task.Run()
  - 在一个新的线程上执行代码
  - 传入异步匿名方法会被包装成 Task
    - Task.Run(async () => await Task.Delay(100))
    - 保证异步方法在别的线程上运行
- Task.Factory.StartNew()
  - 提供更多功能，比如 TaskCreationOptions.LongRunning
  - Task.Run 相当于简化版
- new Task + Task.Start() 看起来类似 new Thread + Thread.Start()，不常用

---

在C#中，创建异步任务的主要方式包括 `Task.Run()`、`Task.Factory.StartNew()` 和 `new Task()`，但它们的使用场景和行为存在差异。以下从用户提出的几点展开论述：

---

### 1. **`Task.Run()`**
`Task.Run()` 是将代码提交到线程池执行的**推荐方式**，适用于大多数异步任务场景。

#### 新线程上的执行
- 通过 `Task.Run()` 提交的代码会在线程池的线程上运行（不一定是“新线程”，但确保不阻塞主线程）。
- 线程池会根据负载动态分配线程，避免频繁创建/销毁线程的开销。
  
#### 异步匿名方法的处理
当传入**异步匿名方法**时，`Task.Run()` 会自动将异步操作包装为 `Task`，并确保异步方法在线程池线程启动：
```csharp
// 示例：Task.Run 包裹异步方法
Task.Run(async () => {
    await Task.Delay(100); 
    Console.WriteLine("完成延迟");
});
```
- **关键行为**：异步方法的初始部分（`await` 之前）在线程池线程运行，`await` 后的代码可能因同步上下文影响执行位置（如在UI线程需返回主线程）。
- **自动解包**：`Task.Run` 自动处理嵌套 `Task<Task>`，直接返回表示整个异步操作的任务，无需手动 `Unwrap`。

#### Example

```csharp
[Test]
public async Task TestCreateTask()
{
    Helper.PrintThreadId("Before");
    var result = await Task.Run(() => HeavyJob01());
    Console.WriteLine($"result:{result}");
    Helper.PrintThreadId("After");
}

private int HeavyJob01()
{
    Helper.PrintThreadId();
    Thread.Sleep(5000);
    return 10;
}
```

![](imgs/01.PNG)

---

### 2. **`Task.Factory.StartNew()`**
`Task.Factory.StartNew()` 提供了更细粒度的控制，但需谨慎使用以避免常见陷阱。

#### 更多功能与选项
- **`TaskCreationOptions`**：支持 `LongRunning`、`DenyChildAttach` 等选项。
  - `LongRunning` 提示任务可能长时间运行，线程池可能为其分配独立线程（避免线程池阻塞）。
  ```csharp
  Task.Factory.StartNew(async () => {
      await Task.Delay(100);
  }, TaskCreationOptions.LongRunning);
  ```

#### 与 `Task.Run()` 的差异
- **默认行为**：`Task.Run` 是 `Task.Factory.StartNew` 的简化版，默认附加 `TaskScheduler.Default` 和 `DenyChildAttach`。
- **异步委托处理**：`StartNew` 返回 `Task<Task>`，需手动调用 `.Unwrap()` 获取实际任务：
  ```csharp
  var task = Task.Factory.StartNew(async () => await Task.Delay(100)).Unwrap();
  ```

---

### 3. **`new Task() + Task.Start()`**
此方式类似 `new Thread().Start()`，但**不推荐使用**。

#### 缺点与风险
- **手动管理**：需显式调用 `Start()` 启动任务，易遗漏导致任务未执行。
- **缺乏灵活性**：无法直接处理异步委托（构造函数仅接受 `Action`/`Func`，不接受 `async` 方法）。
- **反模式**：违背任务基于线程池的设计初衷，可能导致性能问题。

```csharp
// 不推荐示例
var task = new Task(() => Console.WriteLine("运行"));
task.Start();
```

---

### 总结
| 方法                     | 适用场景                          | 异步支持 | 线程行为               | 推荐度 |
|--------------------------|-----------------------------------|----------|------------------------|--------|
| `Task.Run()`             | 简单异步任务                      | ✔️自动解包 | 线程池动态分配         | ★★★★★ |
| `Task.Factory.StartNew()` | 需要精细控制的场景（如 `LongRunning`） | 需 `Unwrap` | 可配置线程池或独立线程 | ★★★☆☆ |
| `new Task() + Start()`    | 遗留代码或特殊需求                | ❌        | 手动管理               | ★☆☆☆☆ |

**最佳实践**：
- 优先使用 `Task.Run()` 简化异步操作。
- 仅在需要 `TaskCreationOptions` 时使用 `Task.Factory.StartNew()`，并注意处理嵌套任务。
- 避免使用 `new Task()`，除非有明确的低级控制需求。

### 1.2 如何同时开启多个异步任务

在C#中，使用`Task.WhenAll()`和`Task.WhenAny()`可以高效地管理多个异步任务。以下是具体实现方法及示例：

### 使用 `Task.WhenAll()`
`Task.WhenAll()`会等待所有任务完成后再继续执行。

#### 示例1：无返回值任务
```csharp
public async Task RunAllTasksAsync()
{
    Task task1 = DoWorkAsync("Task 1");
    Task task2 = DoWorkAsync("Task 2");
    Task task3 = DoWorkAsync("Task 3");

    await Task.WhenAll(task1, task2, task3);
    Console.WriteLine("所有任务完成！");
}

async Task DoWorkAsync(string taskName)
{
    await Task.Delay(1000); // 模拟耗时操作
    Console.WriteLine($"{taskName} 完成");
}
```

#### 示例2：带返回值的任务
```csharp
public async Task ProcessDataAsync()
{
    Task<int> task1 = CalculateResultAsync(10);
    Task<int> task2 = CalculateResultAsync(20);

    int[] results = await Task.WhenAll(task1, task2);
    Console.WriteLine($"结果总和：{results.Sum()}");
}

async Task<int> CalculateResultAsync(int input)
{
    await Task.Delay(500);
    return input * 2;
}
```

### 使用 `Task.WhenAny()`
`Task.WhenAny()`在任意一个任务完成时立即继续。

#### 示例1：处理首个完成的任务
```csharp
public async Task HandleFirstCompletedTaskAsync()
{
    Task<string> task1 = FetchDataAsync("Source1");
    Task<string> task2 = FetchDataAsync("Source2");

    Task<string> firstTask = await Task.WhenAny(task1, task2);
    string result = await firstTask;
    Console.WriteLine($"最先返回的数据：{result}");
}

async Task<string> FetchDataAsync(string source)
{
    await Task.Delay(new Random().Next(500, 2000)); // 模拟不同响应时间
    return $"{source} 的数据";
}
```

#### 示例2：处理任务并取消其他任务
```csharp
public async Task<string> GetDataWithCancellationAsync()
{
    var cts = new CancellationTokenSource();
    var tasks = new List<Task<string>>
    {
        DownloadWithCancelAsync("http://example.com", cts.Token),
        DownloadWithCancelAsync("http://example.org", cts.Token)
    };

    Task<string> completedTask = await Task.WhenAny(tasks);
    cts.Cancel(); // 取消未完成的任务
    return await completedTask;
}

async Task<string> DownloadWithCancelAsync(string url, CancellationToken token)
{
    using (var client = new HttpClient())
    {
        await Task.Delay(1000, token); // 模拟可取消操作
        return await client.GetStringAsync(url);
    }
}
```

#### 示例3
```csharp
[Test]
public async Task TestRunTasksAsync()
{
    var inputs = Enumerable.Range(1, 10).ToArray();
    var sem = new SemaphoreSlim(2, 2);
    var tasks = inputs.Select(HeavyJob).ToList();

    await Task.WhenAll(tasks);

    var outputs = tasks.Select(x => x.Result).ToArray();

    foreach (var output in outputs)
    {
        Console.WriteLine($"result:{output}");
    }

    async Task<int> HeavyJob(int input)
    {
        await sem.WaitAsync();
        await Task.Delay(1000);
        sem.Release();
        return input * input;
    }
}
```

![](imgs/02.PNG)

通过上述方法，可以灵活管理多个异步任务的执行与结果处理。

### 1.3 异步任务取消

- CancellationTokenSource + CancellationToken
- OperationCanceledException & TaskCanceledException
- 推荐异步方法都带上 CancellationToken 这一传参

---

在C#中取消异步任务是一个重要的设计模式，可以有效避免资源浪费和提升程序健壮性。以下是结合关键点的详细说明：

---

### 1. **CancellationTokenSource + CancellationToken**
这是C#中实现任务取消的核心机制：
- **CancellationTokenSource**：负责生成取消信号（`Cancel()`方法）并管理生命周期。
- **CancellationToken**：传递给异步方法的令牌，用于监听取消请求。

```csharp
var cts = new CancellationTokenSource();
CancellationToken token = cts.Token;

// 异步任务内部监听取消
async Task DoWorkAsync(CancellationToken ct)
{
    while (!ct.IsCancellationRequested)
    {
        // 模拟工作
        await Task.Delay(1000);
    }
    ct.ThrowIfCancellationRequested(); // 主动抛出取消异常
}

// 触发取消（例如按钮点击、超时等）
cts.CancelAfter(3000); // 3秒后自动取消
// 或手动取消：cts.Cancel();
```

---

### 2. **OperationCanceledException & TaskCanceledException**
这两个异常用于处理取消逻辑：
- **OperationCanceledException**：通用取消异常，直接由`ThrowIfCancellationRequested()`抛出。
- **TaskCanceledException**：继承自前者，通常由`Task.Delay`或未启动的任务取消时抛出。

```csharp
try
{
    await DoWorkAsync(token);
}
catch (OperationCanceledException ex)
{
    // 统一处理取消逻辑（例如资源清理）
    Console.WriteLine("Task was cancelled.");
}
// 不需要单独捕获TaskCanceledException，因其继承自OperationCanceledException
```

---

### 3. **推荐异步方法都带上 CancellationToken 参数**
这是编写可取消异步任务的最佳实践：
- **提高可组合性**：允许调用者控制任务取消。
- **资源友好**：避免无用任务继续占用资源。
- **框架集成**：ASP.NET Core等框架会自动传递请求级别的`CancellationToken`。

**示例：**
```csharp
// 正确做法：支持取消
public async Task ProcessDataAsync(CancellationToken ct = default)
{
    await SomeAsyncOperation(ct);
    ct.ThrowIfCancellationRequested();
}

// 错误做法：无法取消
public async Task ProcessDataAsync()
{
    await SomeAsyncOperation(); // 无取消支持
}
```

**设计建议：**
- 为异步方法的参数添加`CancellationToken`（默认值设为`default`）。
- 在耗时操作前通过`ct.ThrowIfCancellationRequested()`主动检查取消。
- 将`CancellationToken`传递给底层异步方法（如`HttpClient.SendAsync`）。

---

### 总结
通过`CancellationTokenSource`触发取消、用`CancellationToken`传递信号、统一处理`OperationCanceledException`，并强制要求异步方法支持取消，可以构建高响应、资源高效的异步程序。这一机制在Web服务（请求中止）、UI应用（用户取消操作）等场景中尤为重要。

### 1.4 异步任务超时

### 1.5 异步任务汇报进度

### 1.6 同步方法中调用异步方法

## 2. 异步编程常见误区

- 异步一定是多线程？（不一定）
  - 异步编程不必需要多线程来实现
    - 时间片轮转调度
  - 比如可以在单个线程上使用异步 I/O 或事件驱动的编程模型（EAP）
  - 单线程异步：自己定好计时器，到时间之前先去做别的事情
  - 多线程异步：将任务交给不同的线程，并由自己来进行指挥调度

- 异步方法一定要写成 async Task？（不一定）
  - async 关键字只是用来配合 await 使用，从而将方法包装为状态机
  - 本质上仍然是 Task，只不过提供了语法糖，并且函数体中可以直接 return Task 的泛型类型
  - 接口中无法声明 async Task

- await 一定会切换同步上下文？（不一定）
  - 在使用 await 关键字调用并等待一个异步任务时，异步方法不一定会立刻来到新的线程上
  - 如果 await 了一个已经完成的任务（包括Task.Delay(0)），会直接获得结果

- 异步可以全面取代多线程？（不行）
  - 异步编程与多线程有一定关系，但两者并不是可以完全互相替代

- Task.Result 一定会阻塞当前线程？
  - 如果任务已经完成，那么 Task.Result 可以直接得到结果

- 开启的异步任务一定不会阻塞当前线程？
  - await 关键字不一定会立刻释放当前线程，所以如果调用的异步方法中存在阻塞（如Thread.Sleep(0)），那么依旧会阻塞当前上下文对应的线程

## 3. 异步编程中的同步机制

- 传统方式（不适用于异步编程）
  - Monitor（lock）
  - Mutex
  - Semaphore
  - EventWaitHandle

- 轻量型
  - **SemaphoreSlim**
  - ManualResetEventSlim

- 并发集合
  - ConcurrentBag、ConcurrentStack、ConcurrentQueue、ConcurrentDictionary
  - BlockingCollection
  - **Channel**

- 第三方库
  - AsyncManualResetEvent(Microsoft.VisualStudio.Threading)
  - AsyncLock
  - Nito.AsyncEx

### Example

使用 `SemaphoreSlim` 实现异步编程的信号量等待可以通过其 `WaitAsync()` 方法来实现。以下是详细的实现步骤和示例：

### 1. 基本用法
`SemaphoreSlim` 用于控制对资源的并发访问，特别适合异步编程场景。它能限制同时执行特定代码段的异步任务数量。

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

public class AsyncSemaphoreExample
{
    private static readonly SemaphoreSlim _semaphore = new SemaphoreSlim(3); // 允许同时3个任务进入

    public static async Task RunConcurrentTasksAsync()
    {
        var tasks = new Task[10];
        for (int i = 0; i < 10; i++)
        {
            tasks[i] = ProcessAsync(i);
        }
        await Task.WhenAll(tasks);
    }

    private static async Task ProcessAsync(int id)
    {
        Console.WriteLine($"Task {id} 等待进入信号量区域...");
        
        await _semaphore.WaitAsync(); // 异步等待信号量
        try
        {
            Console.WriteLine($"Task {id} 进入信号量区域，当前剩余容量: {_semaphore.CurrentCount}");
            await Task.Delay(1000); // 模拟异步操作
        }
        finally
        {
            _semaphore.Release(); // 释放信号量
            Console.WriteLine($"Task {id} 退出，释放信号量");
        }
    }
}
```

### 2. 关键点解释

- **初始化**：`new SemaphoreSlim(3)` 表示初始允许同时 3 个任务进入临界区
- **异步等待**：`await _semaphore.WaitAsync()` 不会阻塞线程，适合异步编程
- **资源释放**：必须确保在 `finally` 块中调用 `Release()`，避免资源泄漏
- **容量监控**：通过 `CurrentCount` 属性可查看当前可用信号量数

## 3. 补充

### 1. 问题：c#对于一个返回值为Task的异步方法（标记为async）不能使用out或ref参数，这是为什么？

在C#中，返回值为`Task`的异步方法（标记为`async`）**不能使用`out`或`ref`参数**，主要原因如下：

---

### 1. **异步方法的执行流程特性**
异步方法会在遇到`await`时挂起当前操作，将控制权返回给调用者，直到异步操作完成后再恢复执行。此时：
   - **`ref`/`out`参数可能指向无效内存**：当异步方法恢复时，原始调用栈可能已释放，而`ref`/`out`参数可能引用的是原始栈上的变量，导致不安全访问。
   - **跨线程风险**：异步操作可能在另一个线程中恢复，而`ref`/`out`参数引用的变量可能无法跨线程安全访问。

---

### 2. **状态机的实现限制**
编译器会将异步方法转换为一个状态机类，用于管理挂起和恢复。在此过程中：
   - **参数需被捕获到状态机中**：普通值类型参数会被复制到状态机的字段中，但`ref`/`out`参数是变量的引用（指针），无法安全地存储到状态机中（可能指向短期存在的栈内存）。
   - **内存安全性问题**：若状态机在恢复时访问已失效的`ref`/`out`参数，会导致未定义行为或内存损坏。

---

### 3. **语言设计约束**
C#编译器直接禁止在异步方法中使用`ref`或`out`参数，以避免潜在风险：
   - **编译错误**：尝试在`async`方法中使用`ref`/`out`参数会触发错误`CS1988: Async methods cannot have ref or out parameters`。
   - **替代方案**：可通过返回`Tuple`或自定义类型封装多个返回值，或通过`ValueTask<TResult>`传递结果。

---

### 代码示例
```csharp
// 错误：异步方法不能有out参数
public async Task<int> GetValueAsync(out int result)
{
    result = 42;
    await Task.Delay(100);
    return result;
}

// 正确替代方案：通过返回值传递多个结果
public async Task<(int Value, int Result)> GetValueAsync()
{
    int result = 42;
    await Task.Delay(100);
    return (Value: result, Result: result);
}
```

---

### 总结
异步方法禁用`ref`/`out`参数的核心原因是**内存安全性和状态机实现的限制**。通过返回值或容器类型替代`ref`/`out`，可以安全地传递多个结果。
