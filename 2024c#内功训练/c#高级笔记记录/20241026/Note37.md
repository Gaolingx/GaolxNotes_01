# C#高级编程之——I/O文件流（二）——创建，移动，复制，删除

## 三、使用——文件复制

1. **Copy(String, String)**

**注解：**将现有文件复制到新文件。不允许覆盖同名的文件。

```csharp
// 5. 文件复制
[Test]
public void TestCopy01()
{
    // 参数：源文件(string),目标文件(string)
    File.Copy("D://App/logs/log.txt", "D://App/logs2/log.txt");
}
```

注：1. 不允许覆盖同名文件。2. 目录路径必须存在

运行代码，观察D://App/logs2 路径下是否有文件生成，说明我们成功复制了文件，如果重复运行，则会抛出异常。

2. **Copy(String, String, Boolean)**

**注解：**将现有文件复制到新文件。允许覆盖同名的文件（如果第三个参数为true）。

### 四、使用——文件创建

1. **Create(String)**

**注解：**在指定路径中创建或覆盖文件。

```csharp
// 6. 文件创建
[Test]
public void TestCreate01()
{
    //创建一个文件流，需要手动释放资源
    using var fileStream = File.Create("D://App/logs/log3.txt");
    //向文件流写入内容
    using StreamWriter  streamWriter = new StreamWriter(fileStream);
    streamWriter.WriteLine("这是在TestCreate01 写入的内容");
}
```

运行结果如下：

2. **Create(String, Int32)**

**注解：**在指定路径中创建或覆盖文件，指定缓冲区大小。

### 五、使用——文件删除

1. **Delete(String)**

**注解：**删除指定文件。

```csharp
// 7. 文件删除
[Test]
public void TestDelete01()
{
    File.Delete("D://App/logs/log3.txt");
}
```

检查D://App/logs 路径下的log3.txt是否被删除

### 六、使用——文件移动

1. **Move(String, String, Boolean)**

**注解：**将指定文件移动到新位置，提供指定新文件名和覆盖目标文件(如果它已存在)的选项。

```csharp
// 8. 文件移动
[Test]
public void TestMove01()
{
    // true:如果存在同名文件则覆盖，默认为false
    File.Move("D://App/logs2/log.txt", "D://App/logs/log233.txt", true);
}
```

检查D://App/logs2 路径下的log.txt是否被移动到 D://App/logs/log233.txt
