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

private async Task FooAsync2()
{
    await Task.Delay(1000);
    throw new Exception("FooAsync2 Error!");
}
```

运行结果如下，可以看出异常未被正确捕获，我们无法捕获到FooAsync2抛出的异常，也无法判断任务执行的状态

![](imgs/04.PNG)

---

在C#中，调用异步方法时若未使用`await`关键字，异常将无法被常规的`try-catch`捕获，因为此时任务（`Task`）会被“火并忘记”（fire-and-forget）。未等待的任务异常会被静默封装在返回的`Task`对象中，而非传播到当前上下文。以下是关键原因及解决方案：

### 问题原因
1. **未观察任务状态**：  
   `_ = FooAsync2()`丢弃了返回的`Task`，导致该任务的异常未被任何代码观察（如`await`、`.Wait()`或访问`.Exception`属性）。

2. **异步异常传播机制**：  
   C#中，异步方法的异常仅在通过`await`等待任务时才会抛出到当前上下文。若未等待，异常会被保留在任务中，直到被显式处理或成为未观察异常（可能触发`TaskScheduler.UnobservedTaskException`）。

### 解决方案
**使用`await`等待任务**：  
在`try`块内`await`异步调用，使异常能正确传播到`catch`块：

```csharp
[Test]
public async Task RunGetMessageAsync3()
{
    Console.WriteLine("Start...");

    try
    {
        await FooAsync2(); // 使用await等待任务，异常会被捕获
        await Task.Delay(2000);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
    }

    Console.WriteLine("Done.");
}
```

### 其他场景：需后台执行任务但仍需处理异常
若需让任务在后台运行，但需单独处理其异常，可显式捕获任务异常：

```csharp
_ = FooAsync2().ContinueWith(task => 
{
    if (task.Exception != null)
    {
        Console.WriteLine($"后台任务异常: {task.Exception.Message}");
    }
}, TaskScheduler.Default);
```

### 关键结论
- **始终`await`异步调用**：除非明确需要“火并忘记”且接受潜在未处理异常。
- **避免静默丢弃任务**：丢弃任务（`_ =`）易导致异常丢失，建议仅在明确处理异常时使用。

### 3. 构造函数中调用异步方法