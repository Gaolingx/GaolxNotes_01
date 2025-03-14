# C#多线程与异步编程（六）

## 4. 异步取消

当我们执行一个异步任务的时候，可能任务执行中途会以各种理由，取消这个任务，因此，我们需要借助 CancellationTokenSource + CancellationToken 来实现异步任务的取消，许多异步方法（例如SemaphoreSlim.WaitAsync）也允许传入CancellationToken来取消。

### 4.1 CancellationTokenSource + CancellationToken取消异步

案例一（手动取消）：

```csharp
[Test]
public async Task TestCancelTask()
{
    var cts = new CancellationTokenSource();
    var token = cts.Token;
    var sw = Stopwatch.StartNew();

    try
    {
        var cancelTask = Task.Run(async () =>
        {
            await Task.Delay(2000);
            cts.Cancel();
        });
        await Task.WhenAll(Task.Delay(5000, token), cancelTask);
    }
    catch (TaskCanceledException ex)
    {
        Console.WriteLine(ex.ToString());
    }
    finally
    {
        cts.Dispose();
    }
    Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
}
```

![](imgs/01.PNG)

---

### **1. 异步取消的异常处理问题**
- **`Task.WhenAll` 的异常包装机制**：当多个任务中有一个抛出取消异步的异常时，会抛出`TaskCanceledException`，需要处理。

---

### **2. `CancellationTokenSource` 的释放问题**
#### 原代码问题：
- **手动调用 `Dispose` 的风险**：虽然代码在 `finally` 中调用了 `cts.Dispose()`，但若作用域内出现未处理的异常或提前返回，可能跳过释放（尽管此处 `finally` 已经较安全）。
- **更优雅的释放方式**：使用 `using` 语句可以自动管理资源生命周期，确保 `Dispose` 被调用。

#### 改进方案：
- **用 `using` 包裹 `CancellationTokenSource`**：将 `cts` 包裹在 `using` 作用域中，自动释放资源。

```csharp
[Test]
public async Task TestCancelTask2()
{
    using (var cts = new CancellationTokenSource())
    {
        var token = cts.Token;
        var sw = Stopwatch.StartNew();

        try
        {
            var cancelTask = Task.Run(async () =>
            {
                await Task.Delay(2000);
                cts.Cancel();
            });
            await Task.WhenAll(Task.Delay(5000, token), cancelTask);
        }
        catch (TaskCanceledException ex)
        {
            Console.WriteLine(ex.ToString());
        }
        finally
        {
            cts.Dispose();
        }
        Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
    }
}
```

---

此改进方案确保异常被正确处理且资源管理更安全。

---

案例二（超时自动取消）：

```csharp
[Test]
public async Task TestCancelTask3()
{
    using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3.0)))
    {
        //或者用 cts.CancelAfter(3000);
        var token = cts.Token;
        var sw = Stopwatch.StartNew();

        try
        {
            await Task.Delay(5000, token);
        }
        catch (TaskCanceledException ex)
        {
            Console.WriteLine(ex.ToString());
        }
        finally
        {
            cts.Dispose();
        }
        Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
    }
}
```

---

![](imgs/02.PNG)

---

### **TimeSpan 介绍**
#### **作用**：
- `TimeSpan`是C#中表示时间间隔的结构体，用于精确描述时间段（如天、小时、分钟、秒、毫秒）。

#### **常用创建方法**：
- `TimeSpan.FromDays(double)`: 按天数创建。
- `TimeSpan.FromHours(double)`: 按小时数创建。
- `TimeSpan.FromMinutes(double)`: 按分钟数创建。
- `TimeSpan.FromSeconds(double)`: 按秒数创建。
- `TimeSpan.FromMilliseconds(double)`: 按毫秒数创建。

#### **示例**：
```csharp
TimeSpan ts1 = TimeSpan.FromSeconds(3);    // 3秒
TimeSpan ts2 = TimeSpan.FromMinutes(1.5);  // 1分30秒
TimeSpan ts3 = TimeSpan.FromMilliseconds(500); // 500毫秒
```

#### **其他功能**：
- **时间计算**：支持加减运算（`+`, `-`）和比较（`>`, `<`）。
- **属性提取**：可通过`Days`, `Hours`, `Minutes`, `Seconds`, `Milliseconds`等属性获取时间间隔的各部分值。

---

解读：

这段C#代码通过`CancellationTokenSource`和`TimeSpan`实现了异步任务的超时自动取消机制。以下是实现原理的详细分析及`TimeSpan`的介绍：

---

### **超时自动取消的实现原理**
#### 代码关键点：
- **`CancellationTokenSource`**：用于生成取消令牌（`CancellationToken`），并在指定时间后触发取消操作。
  - 构造函数`new CancellationTokenSource(TimeSpan.FromSeconds(3.0))`会在3秒后自动触发取消。
  - 内部通过一个计时器实现超时，超时后调用`Cancel()`方法，将`CancellationToken`的状态标记为“已取消”。
- **`Task.Delay`**：接受一个`CancellationToken`参数，当令牌被取消时，`Task.Delay`会立即抛出`TaskCanceledException`。
  - 本例中`Task.Delay(5000, token)`原本会等待5秒，但3秒后因令牌取消而提前终止。

#### 执行流程：
1. 创建`CancellationTokenSource`并设置3秒超时。
2. 启动`Stopwatch`记录耗时。
3. 执行`Task.Delay(5000, token)`，传入取消令牌。
4. **3秒后**，`CancellationTokenSource`触发取消，`Task.Delay`抛出`TaskCanceledException`。
5. 异常被`catch`块捕获，输出异常信息。
6. 最终输出耗时约为3000ms（而非5000ms），证明任务因超时被取消。

---

### 说明

1. CTS 实现了 IDisposable 接口，所以需要释放。
2. CTS 还可以传入一个 TimeSpan，表示超时后自动取消，或调用 CancelAfter 方法。

### 补充

#### 1. 异步方法中 CancellationToken 重载写法

为了方便的取消异步任务，我们通常会给每个方法都写一个传入CancellationToken的参数，但很多时候，我们并不是总需要传入 CancellationToken 取消，对此我们可以写一个不带CT的方法重载。

```csharp
class Demo
{
    private async Task FooAsync(CancellationToken ct)
    {
        await Task.Delay(5000, ct);
        // ...
    }

    private async Task FooAsync()
    {
        await FooAsync(CancellationToken.None);
    }
    //或者：
    //private async Task FooAsync() => await FooAsync(CancellationToken.None);
    //private Task FooAsync() => FooAsync(CancellationToken.None);

    private async Task FooAsync2(int delay, CancellationToken ct = default)
    {
        await Task.Delay(delay, ct);
    }
}
```
