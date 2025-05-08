# C#多线程与异步编程（十二）

## 10. C#如何在异步任务中汇报并展示进度

在执行异步任务时，有时候我们会希望有办法汇报进度。比如在一个 WPF 程序中，我们在界面上放了一个进度条，从而展示当前任务的进度。那么该如何汇报异步任务的进度呢？

其实 .NET 标准库就为我们提供了实现这一功能的接口和类：`IProgress<T>` 与 `Progress<T>`，其中 `T` 是一个泛型类型，表示要汇报的内容。如果我们希望汇报一个百分比进度，那么使用 `double` 类型即可；类似地，如果我们希望汇报一些更加复杂的内容，还可以使用 `string` 甚至一些自定义类与结构体。

### 1. IProgress<T>、Progress<T>

在C#异步编程中，`IProgress<T>` 和 `Progress<T>` 是用于**跨线程/上下文报告进度**的机制，常用于在异步操作中向调用方（如UI线程）反馈进度信息（例如文件下载进度、任务完成百分比等）。

---

### **1. `IProgress<T>` 接口**
- **作用**：定义了一个通用的进度报告接口，通过 `Report(T value)` 方法传递进度数据。
- **特点**：
  - 抽象接口，不依赖具体实现。
  - 支持泛型类型 `T`，可以是自定义的进度模型（例如 `int` 表示百分比，或一个包含详细信息的对象）。
- **典型用途**：作为异步方法的参数，允许调用方传入自定义的进度报告逻辑。

```csharp
public interface IProgress<in T>
{
    void Report(T value);
}
```

---

### **2. `Progress<T>` 类**
- **作用**：`IProgress<T>` 的默认实现类，内部封装了事件触发机制，简化进度报告。
- **特点**：
  - 自动将进度事件**同步到创建它的上下文**（例如UI线程），无需手动处理线程切换。
  - 通过 `ProgressChanged` 事件通知进度更新。
  - `Progress<T>` 在构造时会捕获当前的 `SynchronizationContext`（如UI线程的上下文）。

```csharp
public class Progress<T> : IProgress<T>
{
    public event EventHandler<T>? ProgressChanged;
    public void Report(T value) { /* 触发 ProgressChanged 事件 */ }
}
```

### 2. 在WPF中通过Progress<T>展示异步任务进度

以一个简答的wpf项目为例，我们创建了一个run的按钮（Button）和一个进度条（ProgressBar），我们的要求是当用户点击run 按钮之后，开启一个耗时的异步任务，并在进度条展示任务进度，同时窗口需要能正常响应。