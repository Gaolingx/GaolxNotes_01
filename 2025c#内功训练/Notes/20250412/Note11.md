# C#多线程与异步编程（十一）

## 9. 使用 ValueTask

自从 C# 5.0 引入了 `async` 和 `await` 语法以后，异步编程变得非常简单，而 `Task` 类型也在开发中扮演着相当重要的角色，存在感极高。但是在 .NET Core 2.0 这个版本，微软引入了一个新的类型 `ValueTask`，`ValueTask` 和 `ValueTask<TResult>` 是为了优化异步操作的性能而引入的，主要解决 Task 类型在高性能场景中的内存分配和开销问题。

> 参考文档
> ![参考 1](https://blog.coldwind.top/posts/why-we-need-valuetask/)

---

### 1. 传统 Task 类型的问题

首先我们要知道，`Task` 包含泛型版本和非泛型版本，分别对应有无返回值的异步任务。而 `ValueTask` 在诞生之初，只有一个泛型版本。换句话说，设计者认为，`ValueTask` 应当只适用于有返回值的异步任务。所以这里我们来看一个典型的例子：

```csharp
using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class TestValueTask
    {
        [Test]
        public async Task RunTask()
        {
            var service = new MyServices();
            var result = await service.GetValueAsync(42);
            Console.WriteLine(result);
        }
    }

    internal class MyServices
    {
        private readonly Dictionary<int, string> _cacheDict;

        public MyServices()
        {
            _cacheDict = new()
            {
                [42] = "hello"
            };
        }

        public async Task<string> GetValueAsync(int id)
        {
            if (_cacheDict.TryGetValue(id, out var result))
            {
                return result;
            }

            await Task.Delay(500);
            return id.ToString();
        }
    }
}

```

在上面的 `GetMessageAsync` 方法中，我们首先尝试从缓存中获取消息，如果没有找到，就再尝试从数据库中获取。但这里有一个问题，如果缓存中有数据，那么虽然我们好像会直接返回一个值。但是，由于 `GetMessageAsync` 方法是一个异步方法，所以实际上会返回一个 `Task<string>` 类型的对象。这就意味着，即便我们本可以只返回一个值，我们依旧会多创建一个 `Task` 对象，这就导致了无端的内存开销。

> Info
> 这种在异步任务中直接返回一个值的情况，我们称之为“同步完成”，或者“返回同步结果”。线程进入这个异步任务后，并没有碰到 await 关键字，而是直接返回。也就是说，这个异步任务自始至终都是在同一个线程上执行的。

### 2. ValueTask 简介

所以，`ValueTask` 的主要作用就是解决这个问题。它在 .NET Core 2.0 被正式引入，并在 .NET Core 2.1 得到了增强（新增了 `IValueTaskSource<T>` 接口，从而使它可以拥有诸如 IsCompleted 等属性），并且还添加了非泛型的 `ValueTask` 类型（这个我们稍后再说）。

`ValueTask` 我们先不要去思考它是否为值类型，而是可以这么理解：它适用于可能返回一个 `Value`，也可能返回一个 `Task` 的情形。也就是说，它非常适合上面的“缓存命中”的典型场景。我们可以把上面的代码修改为：

```csharp
internal class MyServices
{
    private readonly Dictionary<int, string> _cacheDict;

    public MyServices()
    {
        _cacheDict = new()
        {
            [42] = "hello"
        };
    }

    public async Task<string> GetValueAsync(int id)
    {
        if (_cacheDict.TryGetValue(id, out var result))
        {
            return result;
        }

        await Task.Delay(500);
        return id.ToString();
    }

    public async ValueTask<string> GetValueAsync2(int id)
    {
        if (_cacheDict.TryGetValue(id, out var result))
        {
            return result; // 提前返回结果的情况，不会创建Task，0GC
        }

        await Task.Delay(500);
        return id.ToString();
    }
}
```

此时，如果缓存中有数据，那么我们可以直接返回一个 `ValueTask<T>` 对象，而不需要再创建一个 `Task<T>` 对象。这样就避免了无端的堆内存开销；否则，我们才会创建 `Task<T>` 对象。或者说，在这种情况下，`ValueTask` 的性能会退化为 `Task`（甚至可能还稍微低一丁点，因为涉及到更多的字段，以及值拷贝等）。

> Info
> 至于非泛型版本的 `ValueTask`，它的使用情形就更少了。它只有在即使异步完成也可以无需分配内存的情况下才会派上用场。`ValueTask` 的“发明者”Stephen Toub 在他的文章中提到，除非你借助 profiling 工具确认 `Task` 的这一丁点开销会成为瓶颈，否则不需要考虑使用 `ValueTask`。

这时候我们再来思考它的性能究竟如何：

顾名思义，`ValueTask` 是一个值类型，可以在栈上分配，而不需要在堆上分配。不仅如此，它因为实现了一些接口，从而使它可以像 `Task` 一样被用于异步编程。所以，照理说，`ValueTask` 的性能要比 Task 更好很多（就如同 `ValueTuple` 之于 `Tuple`、`Span` 之于 `Array` 一样）。

### 3. ValueTask 注意事项

但是 `ValueTask` 也存在诸多限制，例如：不可多次等待（await）、不支持在任务未完成时阻塞的功能（像使用 Task 那样在 ValueTask 上调用 Wait、Result、GetAwaiter().GetResult() 等方法）、没有引入线程安全等机制等，因此并非所有的异步任务都适用于 `ValueTask`。

#### 1. ValueTask 不能被多次等待（await）

`ValueTask` 底层会使用一个对象存储异步操作的状态，而它在被 `await` 后（可以认为此时异步操作已经结束），这个对象可能已经被回收，甚至有可能已经被用在别处（或者说，`ValueTask` 可能会从已完成状态变成未完成状态）。而 `Task` 是绝对不可能发生这种情况的，所以可以被多次等待。

#### 2. 不要阻塞 ValueTask

`ValueTask` 所对应的 `IValueTaskSource` 并不需要支持在任务未完成时阻塞的功能，并且通常也不会这样做。这意味着，你无法像使用 `Task` 那样在 `ValueTask` 上调用 `Wait`、`Result`、`GetAwaiter().GetResult()` 等方法。

但换句话说，如果你可以确定一个 ValueTask 已经完成（通过判断 IsCompleted 等属性的值），那么你可以通过 Result 属性来安全地获取 ValueTask 的结果。

```csharp
[Test]
public void RunTask2()
{
    var service = new MyServices();
    var task = service.GetValueAsync2(42); //GetValueAsync2 是一个 ValueTask
    if (task.IsCompleted)
    {
        var result = task.Result;
        Console.WriteLine(result);
    }
}
```

> Info
> 微软专门添加了一个与这个有关的警告：![CA2012](https://learn.microsoft.com/zh-cn/dotnet/fundamentals/code-analysis/quality-rules/ca2012)

#### 3. 不要在多个线程上同时等待一个 ValueTask

`ValueTask` 在设计之初就只是用来解决 `Task` 在个别情况下的开销问题，而不是打算全面取代 `Task`。因此，`Task` 的很多优秀且便捷的特性它都不用有。其中一个就是线程安全的等待。

也就是说，`ValueTask` 底层的对象被设计为只希望被一个消费者（或线程）等待，因此并没有引入线程安全等机制。尝试同时等待它可能很容易引入竞态条件和微妙的程序错误。而 `Task` 支持任意数量的并发等待。

---

#### 完整示例

```csharp
using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class TestValueTask
    {
        [Test]
        public async Task RunTask()
        {
            var service = new MyServices();
            var result = await service.GetValueAsync(42);
            Console.WriteLine(result);
        }

        [Test]
        public async Task RunTask2()
        {
            var service = new MyServices();
            var result = await service.GetValueAsync2(42);
            Console.WriteLine(result);
        }

        [Test]
        public void RunTask3()
        {
            var service = new MyServices();
            var task = service.GetValueAsync2(42);
            if (task.IsCompleted)
            {
                var result = task.Result;
                Console.WriteLine(result);
            }
        }
    }

    internal class MyServices
    {
        private readonly Dictionary<int, string> _cacheDict;

        public MyServices()
        {
            _cacheDict = new()
            {
                [42] = "hello"
            };
        }

        public async Task<string> GetValueAsync(int id)
        {
            if (_cacheDict.TryGetValue(id, out var result))
            {
                return result;
            }

            await Task.Delay(500);
            return id.ToString();
        }

        public async ValueTask<string> GetValueAsync2(int id)
        {
            if (_cacheDict.TryGetValue(id, out var result))
            {
                return result;
            }

            await Task.Delay(500);
            return id.ToString();
        }
    }
}

```

### 4. Task 与 ValueTask 对比

以下是 **C# 中 `Task` 与 `ValueTask` 的核心对比**，帮助你在不同场景中合理选择：

---

### **1. 类型与内存分配**
| **特性**          | `Task` / `Task<T>`                          | `ValueTask` / `ValueTask<T>`              |
|-------------------|---------------------------------------------|-------------------------------------------|
| **类型**          | 引用类型（堆分配）                          | 结构体（值类型，通常栈分配）              |
| **内存开销**      | 每次异步操作都会分配堆内存                  | 同步完成时无堆分配，异步时可能分配        |
| **GC 压力**       | 高（频繁分配导致垃圾回收）                  | 低（减少堆对象）                          |

---

### **2. 性能特点**
| **场景**          | `Task`                                      | `ValueTask`                               |
|-------------------|---------------------------------------------|-------------------------------------------|
| **同步完成**      | 必须分配 `Task` 对象（即使立即返回结果）    | 直接内联结果，无堆分配                    |
| **异步完成**      | 与 `ValueTask` 性能相近                     | 可能包装 `Task` 或 `IValueTaskSource`     |
| **高频调用**      | 堆分配累积导致 GC 压力                      | 显著减少内存分配                          |

---

### **3. 适用场景**
| **场景**          | `Task`                                      | `ValueTask`                               |
|-------------------|---------------------------------------------|-------------------------------------------|
| **通用异步操作**  | 默认选择，适用于大多数异步场景              | 需谨慎评估，仅在优化高频同步完成时使用    |
| **可能同步完成**  | 不高效（例如缓存命中时仍需分配 `Task`）     | **理想选择**（直接返回结果，无分配）      |
| **长期持有结果**  | 安全（可缓存、多次访问）                    | **不安全**（结构体可能失效）              |
| **高频调用**      | 不推荐（如解析器、网络库等热路径代码）      | **推荐**（减少 GC 压力）                  |
| **库开发**        | 简单安全，但可能对调用方不高效              | **推荐**（如 `System.IO.Pipelines`）      |

---

### **4. 使用限制**
| **限制**          | `Task`                                      | `ValueTask`                               |
|-------------------|---------------------------------------------|-------------------------------------------|
| **多次等待**      | 支持多次 `await` 或调用 `Result`            | **仅支持一次等待**，后续操作需转为 `Task` |
| **长期持有**      | 安全                                        | 需转为 `Task` 或确保异步操作已完成        |
| **复杂性**        | 简单易用                                    | 需注意状态管理（如避免装箱、多次访问）    |

---

### **代码示例对比**
#### 使用 `Task`（通用场景）：
```csharp
public async Task<int> ReadDataAsync()
{
    if (_cacheValid)
        return _cachedData; // 同步返回，但隐式生成 Task<int> 对象

    return await ReadFromNetworkAsync(); // 异步路径
}
```

#### 使用 `ValueTask`（优化高频同步场景）：
```csharp
public ValueTask<int> ReadDataOptimizedAsync()
{
    if (_cacheValid)
        return ValueTask.FromResult(_cachedData); // 同步返回，无堆分配

    return new ValueTask<int>(ReadFromNetworkAsync()); // 异步路径
}
```

---

### **5. 如何选择？**
- **优先 `Task` 的情况**：
  - 操作通常需要异步完成（如网络请求）。
  - 需要长期缓存任务结果。
  - 代码简单性优先于极致性能。

- **优先 `ValueTask` 的情况**：
  - 操作可能频繁同步完成（如缓存命中、快速路径）。
  - 高频调用的热路径代码（如解析器、高吞吐服务）。
  - 编写基础库时，减少调用方的内存开销。

---

### **总结**
- **`Task`**：通用、安全、易用，适合大多数异步场景。
- **`ValueTask`**：性能优化工具，专为减少同步完成时的内存分配设计，适用于高频或同步完成率高的场景。

**权衡建议**：除非性能分析表明 `Task` 的内存分配成为瓶颈，否则优先使用 `Task`。在库开发或性能敏感代码中，合理使用 `ValueTask` 可显著提升效率。
