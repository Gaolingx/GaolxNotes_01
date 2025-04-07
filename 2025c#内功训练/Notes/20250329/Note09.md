# C#多线程与异步编程（九）

## 7. 在同步中调用异步方法

虽然说C#基于Task的这套异步编程用起来确实很方便，但是C#实际开发中，很多情况下无法将一个方法改写成异步，而不得不同步调用异步方法，例如：构造函数调用异步方法、接口中必须实现的同步方法里需要异步操作，以及老项目中使用阻断方式控制异步传染性等，因此，在同步中调用异步方法是我们不得不面对的一个问题。

### 1. 以阻塞的方式调用异步方法

在异步方法无返回值的情况下，我们可以直接使用`Task.Wait`，用阻塞的方式等待任务结束

```csharp
[Test]
public void RunFooAsync()
{
    Console.WriteLine("Run FooAsync...");
    // 用阻塞的方式等待任务结束
    FooAsync().Wait();
    Console.WriteLine("Done.");
}

private async Task FooAsync()
{
    await Task.Delay(1000);
}
```

运行结果如下：

![](imgs/01.PNG)

---

对于有返回值的异步方法，我们可以使用`Task.Result`，用阻塞的方式等待任务返回值

```csharp
[Test]
public void RunGetMessageAsync()
{
    Console.WriteLine("Start...");
    // 用阻塞的方式等待任务结束（需要考虑死锁问题）
    var message = GetMessageAsync().Result;
    Console.WriteLine($"message:{message}");

    Console.WriteLine("Done.");
}

private async Task<string> GetMessageAsync()
{
    await Task.Delay(1000);
    return "Hello World!";
}
```

运行结果如下：

![](imgs/02.PNG)

---

以上两种方式虽然都能实现在同步方法中以阻塞的方式调用异步方法，但是也存在一些问题。

在异步方法无返回值的情况下，直接使用 `Task.Wait()`，有返回值情况下使用`Task.Result`以阻塞方式等待任务结束，虽然在某些特定场景下（如简单的控制台程序）可能不会立即出现问题，但**这种做法存在严重隐患，应避免使用**。以下是关键原因和替代方案：

1. **死锁风险**  
   在存在同步上下文的环境（如 UI 线程、ASP.NET 请求上下文）中，`Wait()` 会阻塞当前线程，而异步任务完成后的回调可能依赖该线程继续执行，导致两者互相等待，形成死锁。

2. **异常处理差异**  
   `Wait()` 会将异常封装为 `AggregateException`，而 `await` 直接抛出原始异常，后者更符合直觉且便于调试。

3. **线程资源浪费**  
   阻塞线程会导致线程无法释放去处理其他任务，在高并发场景下可能引发性能问题。

为了解决上述问题，我们应该优先使用 `GetAwaiter().GetResult()` 而非 `.Result`/`.Wait`，在 C# 中，`GetAwaiter().GetResult()` 和 `Task.Result` 都可以用来在同步方法中阻塞等待异步操作的结果，但它们在行为和使用场景上有显著差异。以下是两者的核心区别：

---

### **1. 异常处理方式**
| **方法**                     | **异常抛出方式**                                                                 | **示例**                                 |
|------------------------------|--------------------------------------------------------------------------------|-----------------------------------------|
| **`Task.Result`**            | 抛出 `AggregateException`，将异步操作中的所有异常包装成一个集合类异常         | `catch (AggregateException ex)`        |
| **`GetAwaiter().GetResult()`** | 直接抛出原始异常（如 `InvalidOperationException` 或自定义异常）                | `catch (HttpRequestException ex)`      |

**示例对比**：
```csharp
async Task ThrowAsync()
{
    await Task.Delay(100);
    throw new InvalidOperationException("Oops!");
}

void SyncMethod()
{
    try
    {
        // 使用 .Result
        var result = ThrowAsync().Result; // ❌ 抛出 AggregateException
    }
    catch (AggregateException ex)
    {
        // 需要解包才能获取原始异常
        var originalEx = ex.InnerException;
    }

    try
    {
        // 使用 GetAwaiter().GetResult()
        var result = ThrowAsync().GetAwaiter().GetResult(); // ✅ 直接抛出 InvalidOperationException
    }
    catch (InvalidOperationException ex)
    {
        // 直接捕获目标异常
    }
}
```

---

### **2. 死锁风险**
两者在同步上下文中（如 UI 线程或传统 ASP.NET 请求上下文）都可能引发死锁，但结合 `ConfigureAwait(false)` 的行为不同：

| **方法**                     | **上下文行为**                                                                 |
|------------------------------|------------------------------------------------------------------------------|
| **`Task.Result`**            | 即使使用 `ConfigureAwait(false)`，仍可能因未正确切断上下文导致死锁          |
| **`GetAwaiter().GetResult()`** | 更明确地与 `ConfigureAwait(false)` 配合使用，能更可靠地避免上下文死锁       |

**优化写法**：
```csharp
// 正确避免死锁的写法
var result = asyncTask
    .ConfigureAwait(false)  // 切断同步上下文
    .GetAwaiter()           // 优先使用此方式
    .GetResult();
```

---

### **3. 代码可读性与意图**
| **方法**                     | **代码意图**                                                                 | **适用场景**                         |
|------------------------------|----------------------------------------------------------------------------|-------------------------------------|
| **`Task.Result`**            | 隐式表示“获取任务结果”，可能被误解为任务已自然完成                         | 适用于已知任务已完成的情况（如 `Task.FromResult`） |
| **`GetAwaiter().GetResult()`** | 显式表示“同步阻塞等待任务完成”，代码意图更清晰                             | 强制等待未完成的任务                 |

---

### **4. 性能与底层机制**
| **方法**                     | **底层行为**                                                                 |
|------------------------------|----------------------------------------------------------------------------|
| **`Task.Result`**            | 直接访问 `Task` 的 `Result` 属性，若任务未完成会阻塞并等待                 |
| **`GetAwaiter().GetResult()`** | 通过 `TaskAwaiter` 机制获取结果，与 `await` 的底层行为更一致               |

---

### **总结：如何选择？**
| **场景**                                                                 | **推荐方法**                              |
|--------------------------------------------------------------------------|------------------------------------------|
| 需要直接捕获原始异常（如业务逻辑中的错误处理）                           | `GetAwaiter().GetResult()` + `ConfigureAwait(false)` |
| 已知任务已完成（如缓存的结果或已完成的任务）                             | `Task.Result`                            |
| 在同步上下文中强制等待异步操作（如旧代码改造）                           | `GetAwaiter().GetResult()`               |
| 需要兼容旧代码或第三方库（某些遗留代码可能依赖 `AggregateException`）    | `Task.Result`                            |

---

演示代码：

```csharp
[Test]
public void RunGetMessageAsync2()
{
    Console.WriteLine("Start...");

    try
    {
        FooAsync2().GetAwaiter().GetResult();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
    }

    Console.WriteLine("Done.");
}

private async Task FooAsync2()
{
    await Task.Delay(1000);
    throw new Exception("FooAsync2 Error!");
}
```

运行结果如下：

![](imgs/03.PNG)

---

可以看到我们可以直接捕获到`FooAsync2()`中的异常，而无需对`AggregateException`解包获取原始异常，因此，推荐在任何需要同步调用异步方法的情况下，都使用`GetAwaiter().GetResult()`以阻塞的方式调用异步方法。

### 2. 一发即忘（Fire-and-forget）

```csharp
/// <summary>
/// 不安全的Fire-and-forget（一）
/// </summary>
/// <returns></returns>
[Test]
public async Task RunGetMessageAsync3()
{
    Console.WriteLine("Start...");

    try
    {
        _ = FooAsync2(); //无法捕获异常
        await Task.Delay(2000);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
    }

    Console.WriteLine("Done.");
}

/// <summary>
/// 不安全的Fire-and-forget（二）
/// </summary>
/// <returns></returns>
[Test]
public async Task RunGetMessageAsync4()
{
    Console.WriteLine("Start...");

    try
    {
        VoidFooAsync2(); //无法捕获异常
        await Task.Delay(2000);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
    }

    Console.WriteLine("Done.");
}

private async Task FooAsync2()
{
    await Task.Delay(1000);
    throw new Exception("FooAsync2 Error!");
}

private async void VoidFooAsync2()
{
    await Task.Delay(1000);
    throw new Exception("FooAsync2 Error!");
}
```

运行结果如下，可以看出`RunGetMessageAsync3`方法无法捕获到FooAsync2抛出的异常，也无法判断任务执行的状态，而`RunGetMessageAsync4`因为出现未经处理的异常（异常没有被捕获或处理），直接导致进程崩溃。

![](imgs/04.PNG)

![](imgs/07.PNG)

---

在C#中，Fire-and-forget模式的两个示例存在以下问题：

---

### **1. 不安全的Fire-and-forget（一）`_ = FooAsync2()`**
#### 问题分析：
- **任务被丢弃**：通过`_ =`忽略返回的`Task`，导致异常无法被观察。
- **未处理的Task异常**：当`FooAsync2`抛出异常时，异常会存储在返回的`Task`中。但由于没有`await`或显式检查该`Task`，异常成为“未观察的异常”（Unobserved Exception）。此类异常不会触发调用方的`catch`块，而是在垃圾回收时可能触发`TaskScheduler.UnobservedTaskException`事件（默认不崩溃，但不可靠）。

#### 示例代码的缺陷：
```csharp
_ = FooAsync2(); // 异常被存储在丢弃的Task中，无法被catch捕获
```

---

### **2. 不安全的Fire-and-forget（二）`VoidFooAsync2()`**
#### 问题分析：
- **`async void`的致命缺陷**：`async void`方法没有返回`Task`，因此异常无法通过任务传播。当`VoidFooAsync2`抛出异常时，异常会直接传播到同步上下文（如`SynchronizationContext`），导致：
  - 在UI应用程序中，可能引发全局未处理异常，导致进程崩溃。
  - 在非UI环境（如控制台或测试框架）中，异常可能被静默吞没或触发`AppDomain.UnhandledException`，但**永远不会进入调用方的`catch`块**。

#### 示例代码的缺陷：
```csharp
private async void VoidFooAsync2() // async void 无法被安全捕获异常！
{
    await Task.Delay(1000);
    throw new Exception("Error!"); // 异常直接传播到同步上下文
}
```

---

### **为什么无法捕获异常？**
1. **Fire-and-forget（一）的异常路径**：
   ```text
   FooAsync2 → Task（异常存储于Task） → 未观察的Task → 未被catch捕获
   ```
2. **Fire-and-forget（二）的异常路径**：
   ```text
   VoidFooAsync2 → 异常直接抛出到同步上下文 → 未被catch捕获
   ```

---

### **解决方案**
#### 1. 对返回`Task`的方法：
- **显式处理异常**（推荐）：
  ```csharp
  _ = FooAsync2().ContinueWith(task =>
  {
      if (task.Exception != null)
      {
          Console.WriteLine($"Error: {task.Exception.Message}");
      }
  }, TaskScheduler.Default);
  ```
- **全局处理未观察到的异常**（需谨慎）：
  ```csharp
  TaskScheduler.UnobservedTaskException += (sender, e) =>
  {
      Console.WriteLine($"Unobserved error: {e.Exception.Message}");
      e.SetObserved(); // 标记为已处理，避免进程崩溃
  };
  ```

#### 2. 避免使用`async void`：
- **始终优先使用`async Task`**：
  ```csharp
  private async Task VoidFooAsync2() // 改为返回Task
  {
      await Task.Delay(1000);
      throw new Exception("Error!");
  }
  ```
- **强制调用方处理Task**：
  ```csharp
  await VoidFooAsync2(); // 需要await，不再是Fire-and-forget
  ```
  若必须Fire-and-forget：
  ```csharp
  _ = VoidFooAsync2().ConfigureAwait(false); // 至少保留Task
  ```

---

### **总结**
- **永远不要忽略`Task`**：使用`_ =`丢弃`Task`会导致异常丢失。
- **永远避免`async void`**：除非在事件处理程序（如UI按钮点击）中，且内部已用`try-catch`包裹所有代码。
- **Fire-and-forget需谨慎**：确保异常有明确的处理路径（如日志记录）。

---

### 3. 安全的用Fire-and-forget调用异步方法（一）SafeFireAndForget方案

```csharp
#region Async Call in Ctor 1
internal class SyncAndAsync2
{
    [Test]
    public async Task RunLoadingDataAsync()
    {
        Console.WriteLine("Start...");

        var dataModel = new MyDataModel();
        Console.WriteLine("Loading data...");
        await Task.Delay(2000);
        var data = dataModel.Data;
        Console.WriteLine($"Data is loaded: {dataModel.IsDataLoaded}");

        Console.WriteLine("Done.");
    }
}
internal class MyDataModel
{
    public List<int>? Data { get; private set; }

    public bool IsDataLoaded { get; private set; } = false;

    public MyDataModel()
    {
        //LoadDataAsync(); //直接使用Fire-and-forget方式调用无法处理异常，也无法观察任务状态
        //SafeFireAndForget(LoadDataAsync(), () => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
        SafeFireAndForget(LoadDataAsync2(), () => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
        //LoadDataAsync2().Forget(() => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
    }

    private static async void SafeFireAndForget(Task task, Action? onCompleted = null, Action<Exception>? onError = null)
    {
        try
        {
            await task;
            onCompleted?.Invoke();
        }
        catch (Exception ex)
        {
            onError?.Invoke(ex);
        }
    }

    private async Task LoadDataAsync()
    {
        await Task.Delay(1000);
        Data = Enumerable.Range(1, 10).ToList();
    }

    private async Task LoadDataAsync2()
    {
        await Task.Delay(1000);
        throw new Exception("Failed to load data.");
    }
}

static class TaskExtensions
{
    public static async void Forget(this Task task, Action? onCompleted = null, Action<Exception>? onError = null)
    {
        try
        {
            await task;
            onCompleted?.Invoke();
        }
        catch (Exception ex)
        {
            onError?.Invoke(ex);
        }
    }
}
#endregion
```

运行结果如下：

![](imgs/05.PNG)

---

这段代码通过以下机制安全地实现了一发即忘（Fire-and-forget）并处理异常：

1. **异步异常捕获机制**：
   - `SafeFireAndForget`方法被标记为`async void`，使其成为可独立运行的异步操作
   - 内部使用`try-catch`块包裹`await task`语句，确保捕获所有同步和异步异常
   - 通过`onError`回调参数传递异常信息（示例中输出到控制台）

2. **状态回调机制**：
   - 提供`onCompleted`回调用于处理成功场景（示例中设置IsDataLoaded = false）
   - 通过可选参数设计允许选择性处理完成/失败事件

3. **任务生命周期管理**：
   - 使用`await`确保任务完成前保持对象引用
   - 避免直接使用`.Result`或`.Wait()`可能造成的死锁

4. **扩展方法支持**：
   - `TaskExtensions.Forget`提供链式调用语法
   - 保持统一异常处理逻辑

5. **构造函数安全调用**：
   - 通过包装异步方法避免直接在构造函数中使用`async void`
   - 防止未处理异常导致程序崩溃

典型使用场景：
```csharp
public MyDataModel()
{
    // 安全启动异步初始化
    this.LoadDataAsync()
        .Forget(
            onCompleted: () => IsDataLoaded = true,
            onError: ex => Logger.LogError(ex)
        );
}
```

这种模式适用于不需要等待结果但需要确保异常处理的场景（如后台初始化、事件通知等），但需注意异步操作的生命周期管理。

### 4. 安全的用Fire-and-forget调用异步方法（二） ContinueWith方案

```csharp
#region Async Call in Ctor 2
internal class SyncAndAsync3
{
    [Test]
    public async Task RunLoadingDataAsync2()
    {
        Console.WriteLine("Start...");

        var dataModel = new MyDataModel2();
        Console.WriteLine("Loading data...");
        await Task.Delay(2000);
        var data = dataModel.Data;
        Console.WriteLine($"Data is loaded: {dataModel.IsDataLoaded}");

        Console.WriteLine("Done.");
    }
}
internal class MyDataModel2
{
    public List<int>? Data { get; private set; }

    public bool IsDataLoaded { get; private set; } = false;

    public MyDataModel2()
    {
        //LoadDataAsync(); //直接使用Fire-and-forget方式调用无法处理异常，也无法观察任务状态
        //LoadDataAsync().ContinueWith(t => { OnDataLoaded(t); }, TaskContinuationOptions.None);
        LoadDataAsync2().ContinueWith(t => { OnDataLoaded(t); }, TaskContinuationOptions.None);
    }

    private bool OnDataLoaded(Task task)
    {
        if (task.IsFaulted)
        {
            Console.WriteLine($"Error: {task.Exception.InnerException?.Message}");
            return false;
        }
        return true;
    }

    private async Task LoadDataAsync()
    {
        await Task.Delay(1000);
        Data = Enumerable.Range(1, 10).ToList();
    }

    private async Task LoadDataAsync2()
    {
        await Task.Delay(1000);
        throw new Exception("Failed to load data.");
    }
}
#endregion
```

---

运行结果如下：

![](imgs/06.PNG)

---

### 问题1：以下C#代码是如何安全的实现一发即忘（Fire-and-Forget）？

这段代码通过以下方式安全实现了一发即忘（Fire-and-Forget）：
1. **异步任务的触发**：在 `MyDataModel2` 的构造函数中调用 `LoadDataAsync2().ContinueWith(...)`，启动异步任务后不直接等待它，而是通过 `ContinueWith` 附加一个延续任务。
2. **异常处理**：在延续任务的回调方法 `OnDataLoaded` 中，通过检查 `task.IsFaulted` 捕获异步任务中的异常，并打印错误信息。这避免了未处理的异常被忽略。
3. **状态的显式观察**：通过 `ContinueWith` 强制观察任务状态（包括成功、失败或取消），确保不会因未观察的异常导致程序崩溃（例如触发 `TaskScheduler.UnobservedTaskException`）。

### 问题2：这里使用 `ContinueWith` 可能引发的问题？

使用 `ContinueWith` 可能引发以下问题：
1. **同步上下文丢失**  
   `ContinueWith` 默认不捕获同步上下文（例如UI线程的上下文）。如果回调中需要操作UI或其他线程敏感资源，可能导致代码在错误的上下文中执行。  
   **解决方案**：使用 `TaskScheduler.FromCurrentSynchronizationContext()` 或显式指定调度器。

2. **未处理的延续任务异常**  
   如果 `OnDataLoaded` 方法内部抛出异常（例如访问 `task.Exception` 时未正确处理），会导致新的未观察异常，可能引发程序崩溃。  
   **示例**：若 `task.Exception` 为 `null`，直接访问 `InnerException` 会抛出 `NullReferenceException`。虽然当前代码检查了 `IsFaulted`，但未处理 `task.Exception` 本身为 `null` 的情况。

3. **不明确的延续条件**  
   当前代码使用 `TaskContinuationOptions.None`，无论任务是成功、失败还是取消，都会触发回调。但 `OnDataLoaded` 仅处理了 `IsFaulted` 的情况，忽略了任务取消（`IsCanceled`）或其他状态。  
   **改进建议**：使用 `TaskContinuationOptions.OnlyOnFaulted` 限定仅在任务失败时触发回调。

4. **潜在的资源泄漏**  
   如果任务长时间运行或未正确释放资源，`ContinueWith` 可能导致回调链未及时释放，引发资源泄漏。

### 代码改进建议

1. **显式处理所有任务状态**  
   ```csharp
   LoadDataAsync2().ContinueWith(t => {
       if (t.IsFaulted) 
           Console.WriteLine($"Error: {t.Exception?.InnerException?.Message}");
       else if (t.IsCanceled) 
           Console.WriteLine("Task was canceled.");
   }, TaskContinuationOptions.ExecuteSynchronously);
   ```

2. **使用 `async/await` 替代 `ContinueWith`（如果可行）**  
   ```csharp
   // 在构造函数中无法直接使用 async，但可通过工厂方法实现
   public static async Task<MyDataModel2> CreateAsync() {
       var model = new MyDataModel2();
       await model.LoadDataAsync2().ConfigureAwait(false);
       return model;
   }
   ```

3. **强制捕获同步上下文（如需要）**  
   ```csharp
   LoadDataAsync2().ContinueWith(t => {
       OnDataLoaded(t);
   }, TaskScheduler.FromCurrentSynchronizationContext());
   ```

4. **安全访问异常信息**  
   ```csharp
   private bool OnDataLoaded(Task task) {
       if (task.Exception is AggregateException ex) 
           Console.WriteLine($"Error: {ex.InnerException?.Message}");
       // 其他处理...
   }
   ```

### 总结
- **安全的一发即忘**：通过 `ContinueWith` 观察任务状态并处理异常，避免未处理异常。
- **`ContinueWith` 的陷阱**：需注意同步上下文、异常传播、任务状态过滤和资源管理。在可能的情况下，优先使用 `async/await` 简化逻辑。

---

### **与首个方案的对比**
| 特性               | SafeFireAndForget方案          | ContinueWith方案              |
|--------------------|-------------------------------|------------------------------|
| **异常可见性**      | 通过try-catch直接捕获          | 依赖手动检查`task.Exception`  |
| **上下文保留**      | 默认保留同步上下文（await特性）| 需显式指定`TaskScheduler`     |
| **代码可读性**      | 更符合async/await模式          | 回调嵌套降低可维护性          |
| **状态管理**        | 通过回调参数明确区分成功/失败  | 需手动在回调中处理状态        |
| **资源释放**        | 天然支持using作用域            | 需手动管理CancellationToken   |

---

### 5. 安全的用Fire-and-forget调用异步方法（三） Fire-and-forget Later方案

我们可以先用Task的字段包装一个异步任务，在构造方法里面给Task赋值，然后直到该异步任务被完成（IsCompleted为true）后，才能后续的操作。即我们可以在每个需要访问异步结果的方法前都await这个Task确保任务完成。

```csharp
#region Fire-and-forget Later
internal class SyncAndAsync4
{
    [Test]
    public async Task RunLoadingDataAsync3()
    {
        Console.WriteLine("Start...");

        var dataModel = new MyDataModel3();
        Console.WriteLine("Loading data...");
        await Task.Delay(2000);
        await dataModel.DisplayDataAsync();

        Console.WriteLine("Done.");
    }
}
internal class MyDataModel3
{
    public List<int>? Data { get; private set; }

    public bool IsDataLoaded { get; private set; } = false;

    private readonly Task loadDataTask;

    public MyDataModel3()
    {
        // 把异步任务存储成类中的一个私有字段，任务状态（是否失败、完成等）可被观察
        loadDataTask = LoadDataAsync();
    }

    public async Task DisplayDataAsync()
    {
        await loadDataTask;
        if (Data != null)
        {
            foreach (var data in Data)
            {
                Console.WriteLine(data);
            }
        }
    }

    private async Task LoadDataAsync()
    {
        await Task.Delay(1000);
        Data = Enumerable.Range(1, 10).ToList();
    }
}
#endregion
```

在C#中，构造函数无法直接使用`async/await`，但可以通过以下方式在构造函数中启动异步任务，并在后续方法中等待其完成：

### 实现原理：
1. **启动任务并存储引用**：在构造函数中调用异步方法，将返回的`Task`保存为类的私有字段。
2. **后续方法等待任务**：在需要访问异步结果的方法中，通过`await`该`Task`确保任务完成。

### 关键点：
- **构造函数中的任务启动**：`LoadDataAsync()`在构造函数中被调用，任务立即开始执行。
- **非阻塞构造**：构造函数不等待任务完成，避免阻塞。
- **确保数据就绪**：任何需要数据的方法（如`DisplayDataAsync`）必须先`await loadDataTask`，确保数据加载完成。

### 使用示例：
```csharp
var dataModel = new MyDataModel3();  // 开始加载数据
// ...其他操作...
await dataModel.DisplayDataAsync();  // 等待数据加载完成后显示
```

这种方式通过分离任务启动与等待，实现了构造时异步初始化，是C#中处理异步初始化的常见模式。

### 6. 安全的用Fire-and-forget调用异步方法（四） Async Factory方案

在异步工厂模式中，我们可以将这个类包装成只暴露工厂方法的类（Create、Build等），并将这个类本身的构造方法隐藏起来，不能显式的构造这个类，实现异步构造的方法。