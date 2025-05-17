# C#多线程与异步编程（十三）

## 11. 在异步任务中调用及取消一个长时间运行的同步方法

**要求**：我们希望能在异步方法中以不阻塞的方式调用一个同步方法，可以随时取消这个同步方法，且不修改原有的同步方法（例如添加CancellationToken）。

**实现思路**：

1. 用 `Thread` 来运行 `LongRunningJob`
2. 如果希望取消，我们可以使用 `Thread.Interrupt` 强制打断
3. 在异步环境下等待：用信号量等方式暴露一个可等待的异步任务

**代码实现**：

```csharp
using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class RunSyncOnAsync
    {
        [Test]
        public async Task RunCancelableThreadTask()
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(2000));
            var task = new CancelableThreadTask(() => { LongRunningJob(); });
            try
            {
                await task.RunAsync(cts.Token);
            }
            catch (TaskCanceledException)
            {
                Console.WriteLine("Task was Canceled.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Task failed: {ex.Message}");
            }

            await Task.Delay(6000);
            Console.WriteLine("Finish.");
        }

        private void LongRunningJob()
        {
            Thread.Sleep(5000);
            Console.WriteLine("Long Running job completed");
        }
    }

    internal class CancelableThreadTask
    {
        private Thread? _thread;
        private TaskCompletionSource? _tcs;
        private readonly Action _action;
        private readonly Action<Exception>? _onError;
        private readonly Action? _onCompleted;

        //原子操作，保证线程安全
        private int _isRunning = 0;

        public CancelableThreadTask(Action action, Action<Exception>? onError = null, Action? onCompleted = null)
        {
            ArgumentNullException.ThrowIfNull(action);
            _action = action;
            _onError = onError;
            _onCompleted = onCompleted;
        }

        public Task RunAsync(CancellationToken token)
        {
            if (Interlocked.CompareExchange(ref _isRunning, 1, 0) == 1)
                throw new InvalidOperationException("Task is already running!");

            _tcs = new TaskCompletionSource();

            _thread = new Thread(() =>
            {
                try
                {
                    _action();
                    _tcs.SetResult();
                    _onCompleted?.Invoke();
                }
                catch (Exception ex)
                {
                    if (ex is ThreadInterruptedException)
                    {
                        _tcs.TrySetCanceled();
                    }
                    else
                    {
                        _tcs.TrySetException(ex);
                        _onError?.Invoke(ex);
                    }
                }
                finally
                {
                    Interlocked.Exchange(ref _isRunning, 0);
                }
            });

            token.Register(() =>
            {
                if (Interlocked.CompareExchange(ref _isRunning, 0, 1) == 1)
                    _thread.Interrupt();
            });

            _thread.Start();

            return _tcs.Task;
        }
    }
}

```

**标准输出**：

```text
Task was Canceled.
Finish.

```

---

### Interlocked 方法讲解

### **1. Interlocked.Exchange**
#### **作用**
以原子方式将变量设置为新值，并返回其原始值。

#### **方法签名**
```csharp
public static int Exchange(ref int location1, int value);
public static object Exchange(ref object location1, object value);
// 其他重载：long、float等（某些类型可能需要转换，如double需通过BitConverter处理）
```

#### **参数**
- `location1`：要修改的变量的引用。
- `value`：要设置的新值。

#### **返回值**
返回`location1`的原始值。

#### **应用场景**
- **原子更新标志位**：多个线程竞争修改共享状态。
- **无锁赋值**：确保变量在读取和写入之间不被其他线程干扰。

#### **示例**
```csharp
int sharedValue = 0;

// 线程安全地将sharedValue设置为5，并获取旧值
int oldValue = Interlocked.Exchange(ref sharedValue, 5);
```

---

### **2. Interlocked.CompareExchange**
#### **作用**
原子操作：比较变量的值与预期值`comparand`，若相等，则将变量设置为新值。无论是否交换，都返回变量的原始值。

#### **方法签名**
```csharp
public static int CompareExchange(ref int location1, int value, int comparand);
public static object CompareExchange(ref object location1, object value, object comparand);
// 其他重载支持long、float等类型
```

#### **参数**
- `location1`：要修改的变量的引用。
- `value`：要设置的新值。
- `comparand`：预期与`location1`当前值比较的值。

#### **返回值**
返回`location1`的原始值。

#### **应用场景**
- **无锁算法**：实现自旋锁、无锁队列或栈。
- **条件更新**：仅在变量符合预期时更新，避免数据竞争。

#### **示例**
```csharp
int sharedValue = 0;

// 仅在sharedValue为0时将其设置为1
int original = Interlocked.CompareExchange(ref sharedValue, 1, 0);
if (original == 0) 
{
    Console.WriteLine("值成功更新为1");
}
```
