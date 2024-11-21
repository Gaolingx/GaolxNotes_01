# C#高级编程之——I/O文件流（三）其他方法

## 七、使用——判断文件存在

1. **File.Exists(String)**

**注解：**确定指定的文件是否存在。

```csharp
// 9. 判断文件是否存在
[Test]
public void TestExists()
{
    string path = "D://App/logs";
    if (File.Exists($"{path}/log233.txt"))
    {
        Console.WriteLine($"{path} 存在同名文件:log233.txt");
    }
    else
    {
        Console.WriteLine($"{path} 不存在同名文件:log233.txt");
    }
}
```

运行结果如下：

## 八、使用——文件打开

1. **File.Open(String, FileMode)**

**注解：**通过不共享的读/写访问权限打开指定路径上的 FileStream。

FileMode 值，用于指定在文件不存在时是否创建该文件，并确定是保留还是覆盖现有文件的内容。

```csharp
// 10. 打开文件
[Test]
public void TestOpen01()
{
    using var stream = File.Open("D://App/logs2/log.txt",FileMode.Open);
    using StreamWriter streamWriter = new StreamWriter(stream);
    streamWriter.WriteLine($"TestOpen01,Date:{DateTime.Now}");
}
```

运行结果如下：

2. **File.Open(String, FileMode, FileAccess)**

**注解：**通过指定的模式和不共享的访问权限打开指定路径上的 FileStream。

```csharp
// 11. 打开文件
[Test]
public void TestOpen02()
{
    // 如果要对File 执行写入操作，我们需要设置文件的写入权限（FileAccess.Write）
    using var stream = File.Open("D://App/logs2/log.txt", FileMode.Open, FileAccess.Read);
    using StreamReader streamReader = new StreamReader(stream);
    string? text = streamReader.ReadLine();
    Console.WriteLine(text);
}
```

运行结果如下：

3. **File.OpenRead(String)**

**注解：**打开现有文件以进行读取。

```csharp
// 12. 打开并读取
[Test]
public void TestOpenRead01()
{
    // OpenRead 只有读取权限
    using var stream = File.OpenRead("D://App/logs2/log.txt");
    using StreamReader streamReader = new StreamReader(stream);
    if (!streamReader.EndOfStream) //判断是否是文件最末尾
    {
        string? text = streamReader.ReadLine();
        Console.WriteLine(text);
    }
}
```

运行结果如下：

4. **File.OpenWrite(String)**

**注解：**打开一个现有文件或创建一个新文件以进行写入。

```csharp
// 13. 打开并写入
[Test]
public void TestOpenWrite01()
{
    // OpenWrite 只有写入权限
    using var stream = File.OpenWrite("D://App/logs2/log.txt");
    using StreamWriter writer = new StreamWriter(stream);
    writer.WriteLine($"TestOpenWrite01,Date:{DateTime.Now}");
}
```

运行结果如下：

5. **File.ReadAllText(String, Encoding)**

**注解：**打开一个文件，使用指定的编码读取文件中的所有文本，然后关闭此文件。

```csharp
// 14. 读取所有内容
[Test]
public void TestReadAllText01()
{
    var stringArr = File.ReadAllText("D://App/logs2/log.txt", System.Text.Encoding.UTF8);
    Console.WriteLine(stringArr);
    Console.WriteLine(stringArr.Length);
}
```

运行结果如下：
