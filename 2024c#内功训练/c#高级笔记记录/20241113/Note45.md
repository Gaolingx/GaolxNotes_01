# C#高级编程之——MemoryStream

## MemoryStream 类

### 定义

命名空间:
System.IO
程序集:
System.Runtime.dll
Source:
MemoryStream.cs

功能:创建一个支持存储为内存的流。

继承:Object->MarshalByRefObject->Stream->MemoryStream

### 注解

流的当前位置是下一次读取或写入操作的发生位置。 可以通过 Seek 方法检索或设置当前位置。 创建新 MemoryStream 实例时，当前位置设置为零。

> 备注
  此类型实现 IDisposable 接口，但实际上没有任何要释放的资源。 这意味着不需要直接调用 Dispose() 或使用 using（在 C# 中）或 Using（在 Visual Basic 中）等语言构造来释放它。

使用无符号字节数组创建的内存流提供不可调整大小的数据流。 使用字节数组时，既不能追加流，也不能收缩流，尽管根据传递给构造函数的参数，可以修改现有内容。 空内存流可以调整大小，并且可以写入和读取。

如果将 MemoryStream 对象添加到 ResX 文件或 .resources 文件，请在运行时调用 GetStream 方法以检索它。

如果将 MemoryStream 对象序列化为资源文件，它实际上将序列化为 UnmanagedMemoryStream。 此行为提供了更好的性能，并且能够直接获取指向数据的指针，而无需通过 Stream 方法。

### 特点以及应用场景

### 一、特点

1. **内存存储**：MemoryStream将数据存储在内存中的缓冲区中，而不是在磁盘或网络中。这使得读写操作更快速，并且可以避免磁盘I/O的开销。
2. **可变大小**：MemoryStream的大小可以根据需要动态增长或缩小，可以通过调整容量来处理不同大小的数据。
3. **支持读写操作**：MemoryStream提供了丰富的读写操作方法，如Read、Write、Seek等，可以方便地读取和写入数据。
4. **支持字节数组**：可以使用字节数组来初始化MemoryStream，也可以从MemoryStream中获取字节数组。
5. **无需释放非托管资源**：MemoryStream中没有任何非托管资源，因此在使用完毕后，不需要显式调用Dispose方法来释放资源（尽管调用Dispose是一个良好的编程习惯，可以确保释放任何可能持有的托管资源）。

### 二、应用场景

1. **处理大量数据**：在处理图像、音频、视频等大量数据时，使用MemoryStream可以避免频繁的磁盘I/O操作，提高数据处理的效率。
2. **临时存储数据**：在网络传输过程中，可以使用MemoryStream作为数据缓冲区，临时存储接收或发送的数据。
3. **实现自定义数据流逻辑**：开发者可以使用MemoryStream来实现自定义的数据流逻辑，如加密、压缩、解密、解压缩等操作。
4. **作为其他流的中间对象**：MemoryStream常作为其他流数据交换时的中间对象操作，方便在不同流之间进行数据转换和传输。

**总结：**允许开发者在内存中处理数据流，而无需涉及磁盘I/O操作。

### 三、常用属性、方法

**属性：**

- Length
  获取流的长度（以字节为单位）。
- Position
  获取或设置流中的当前位置。

**方法：**

- CopyTo(Stream)
  从当前流中读取字节并将其写入另一个流。 这两个流位置都是按复制的字节数进行高级的。(继承自 Stream)
- WriteTo(Stream)
  将此内存流的整个内容写入另一个流。
- Read(Byte[], Int32, Int32)
  从当前流中读取字节块并将数据写入缓冲区。
- Write(Byte[], Int32, Int32)
  使用从缓冲区读取的数据将字节块写入当前流。

### 四、使用

```csharp
// MemoryStream:内存流
// 特点：它不依赖于其他的资源（如具体的文件）
[Test]
public void TestMemoryStream01()
{
    using Stream fStream = new FileStream("D://App/logs/log.txt", FileMode.Open, FileAccess.Read);

    //将文件流中的内容转移到内存中
    //1. 一次性全部转移到内存流中，小文件建议使用（如果文件过大，可能占用过多资源）
    using MemoryStream mStream = new MemoryStream();
    fStream.CopyTo(mStream);

    Console.WriteLine(mStream.Length);
    Console.WriteLine("================");

    //2.使用使用指定的缓冲区大小读取流
    using Stream fStream2 = new FileStream("D://App/logs/log.txt", FileMode.Open, FileAccess.Read);
    using MemoryStream mStream2 = new MemoryStream();
    int bufferSize = 100; //每次只读100bytes
    byte[] buffer = new byte[bufferSize]; //缓冲区

    while (fStream2.Position < fStream2.Length) // Position:文件流当前的位置
    {
        int count =       //返回值(count):每次实际读取的长度 
            fStream2.Read(
            buffer,       //接收文件流中读取的内容
            0,            //每次从第0个位置开始写入到buffer中
            bufferSize    //每次读取的长度
            );

        //写入内存流（把缓冲区中的内容写入到内存流中）
        mStream2.Write(buffer, 0, count); //必须指定count，否则可能出现意料之外的错误
    }

    Console.WriteLine(mStream2.Length);
}
```

运行结果如下：
