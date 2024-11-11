# C#高级编程之——文件流-Stream流

## 定义

命名空间:
System.IO
程序集:
System.Runtime.dll
Source:
Stream.cs

**简介：**提供字节序列的泛型视图。 这是一个抽象类。

- 继承：Object->MarshalByRefObject->Stream
- 派生：
  - Microsoft.JScript.COMCharStream
  - System.Data.OracleClient.OracleBFile
  - System.Data.OracleClient.OracleLob
  - System.Data.SqlTypes.SqlFileStream
  - System.IO.BufferedStream
- 实现：IDisposable  IAsyncDisposable

## 注解

Stream 是所有流的抽象基类。 流是字节序列的抽象，例如文件、输入/输出设备、进程间通信管道或 TCP/IP 套接字。 Stream 类及其派生类提供了这些不同类型的输入和输出的泛型视图，并将程序员与操作系统和基础设备的特定详细信息隔离开来。

流涉及三个基本操作：

- 可以从流中读取。 读取是将数据从流传输到数据结构，例如字节数组。
- 可以写入流。 写入是将数据从数据结构传输到流中。
- 流可以支持寻求。 查找是指查询和修改流中的当前位置。 查找功能取决于流的支持存储类型。 例如，网络流没有统一的当前位置概念，因此通常不支持寻求。

继承自 Stream 的一些较常用流是 FileStream，MemoryStream。

流可能仅支持其中一些功能，具体取决于基础数据源或存储库。 可以使用 Stream 类的 CanRead、CanWrite和 CanSeek 属性来查询流的功能。

Read 和 Write 方法以各种格式读取和写入数据。 对于支持查找的流，请使用 Seek 和 SetLength 方法和 Position 和 Length 属性来查询和修改流的当前位置和长度。

此类型实现 IDisposable 接口。 使用完该类型后，应直接或间接释放它。 若要直接释放类型，请在 try/catch 块中调用其 Dispose 方法。 若要间接处理它，请使用语言构造（如 using（在 C# 中）或 Using（在 Visual Basic 中）。 有关详细信息，请参阅 IDisposable 接口主题中的“使用实现 IDisposable 的对象”部分。

释放 Stream 对象会刷新任何缓冲数据，并实质上调用 Flush 方法。 Dispose 还会释放操作系统资源，例如文件句柄、网络连接或用于任何内部缓冲的内存。 BufferedStream 类提供将缓冲流环绕另一个流的功能，以提高读取和写入性能。

从 .NET Framework 4.5 开始，Stream 类包括用于简化异步操作的异步方法。 异步方法在其名称中包含 Async，例如 ReadAsync、WriteAsync、CopyToAsync和 FlushAsync。 通过这些方法，无需阻止主线程即可执行资源密集型 I/O 操作。 在 Windows 8.x 应用商店应用或桌面应用中，这种性能注意事项尤其重要，其中耗时的流操作可能会阻止 UI 线程，并使你的应用看起来好像不起作用一样。 异步方法与 Visual Basic 和 C# 中的 async 和 await 关键字结合使用。
