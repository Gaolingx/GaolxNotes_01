# C#多线程与异步编程（七）

## 5. 异步超时

### 1. thread中的超时机制

```csharp
[Test]
public void TestTimeoutThreadInterrupt()
{
    var thread = new Thread(FooSync);
    thread.Start();
    if (!thread.Join(TimeSpan.FromMilliseconds(2000)))
    {
        thread.Interrupt();
    }

    Console.WriteLine("Task Done.");
}

private void FooSync()
{
    try
    {
        Console.WriteLine("Foo start...");
        Thread.Sleep(5000);
        Console.WriteLine("Foo end...");
    }
    catch (ThreadInterruptedException)
    {
        Console.WriteLine("Foo Interrupted");
    }
}
```

在C#中，这段代码通过结合`Thread.Join`的超时检测和`Thread.Interrupt`方法，实现了对线程执行的超时控制。具体机制如下：

---

### 1. **超时检测机制**
- **`thread.Join(TimeSpan.FromMilliseconds(2000))`**：  
  主线程调用`Join`方法并设置2秒的超时时间。此方法会阻塞主线程，直到以下两种情况之一发生：
  - **子线程在2秒内完成**：`Join`返回`true`，主线程继续执行后续代码（不触发中断）。
  - **子线程未在2秒内完成**：`Join`返回`false`，主线程调用`thread.Interrupt()`中断子线程。

---

### 2. **中断子线程**
- **`thread.Interrupt()`**：  
  当主线程调用此方法时，若子线程正处于**可中断的阻塞状态**（如`Thread.Sleep`、`Monitor.Wait`等），则会抛出`ThreadInterruptedException`异常。在此代码中，子线程正在执行`Thread.Sleep(5000)`，因此会被立即中断。

---

### 3. **子线程的异常处理**
- **`catch (ThreadInterruptedException ex)`**：  
  子线程在睡眠过程中被中断后，会捕获此异常并提前结束阻塞状态，从而终止耗时操作（原本需要5秒的`Sleep`被缩短）。

---

### 执行流程示例
1. 主线程启动子线程，子线程进入`FooSync`并开始睡眠5秒。
2. 主线程等待子线程最多2秒：
   - 若2秒内子线程未完成（实际不可能，因为子线程需5秒），主线程调用`Interrupt`。
3. 子线程在睡眠中被中断，抛出异常，执行`catch`块后结束。
4. 最终控制台输出：
   ```
   Foo start...
   Task Done.
   Foo Interrupted
   ```

---

### 2. async Task中的超时机制
