# C#高级编程之——I/O文件流（四）FileInfo类

## 一、定义

- 命名空间:
System.IO
- 程序集:
System.Runtime.dll

功能：提供用于创建、复制、删除、移动和打开文件的属性和实例方法，并有助于创建 FileStream 对象。 无法继承此类。

## 二、注解

将 FileInfo 类用于典型操作，例如复制、移动、重命名、创建、打开、删除和追加到文件。

如果要对同一文件执行多个操作，则可以更高效地使用 FileInfo 实例方法，而不是 File 类的相应静态方法，因为安全检查并不总是必要的。

创建或打开文件时，许多 FileInfo 方法返回其他 I/O 类型。 可以使用这些其他类型的进一步操作文件。 有关详细信息，请参阅特定的 FileInfo 成员，例如 Open、OpenRead、OpenText、CreateText或 Create。

默认情况下，向所有用户授予对新文件的完整读/写访问权限。

下表描述了用于自定义各种 FileInfo 方法行为的枚举。

<div class="has-inner-focus"><table aria-label="表 1" class="table table-sm margin-top-none">
<thead>
<tr>
<th>列举</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.fileaccess?view=net-8.0" class="no-loc" data-linktype="relative-path">FileAccess</a></td>
<td>指定对文件的读取和写入访问权限。</td>
</tr>
<tr>
<td><a href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.fileshare?view=net-8.0" class="no-loc" data-linktype="relative-path">FileShare</a></td>
<td>指定已使用的文件允许的访问级别。</td>
</tr>
<tr>
<td><a href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filemode?view=net-8.0" class="no-loc" data-linktype="relative-path">FileMode</a></td>
<td>指定是否保留或覆盖现有文件的内容，以及创建现有文件的请求是否会导致异常。</td>
</tr>
</tbody>
</table></div>

## 三、使用——获取文件信息

**常见属性：**

- CreationTime:获取或设置当前文件或目录的创建时间。(继承自 FileSystemInfo)
- CreationTimeUtc:获取或设置当前文件或目录的协调世界时（UTC）的创建时间。(继承自 FileSystemInfo)
- Directory:获取父目录的实例。
- DirectoryName:获取表示目录的完整路径的字符串。
- Exists:获取一个值，该值指示文件是否存在。
- Extension:获取文件名的扩展名部分，包括前导点 .（即使它是整个文件名）或空字符串（如果没有扩展名）。(继承自 FileSystemInfo)
- FullName:获取目录或文件的完整路径。(继承自 FileSystemInfo)
- IsReadOnly:获取或设置一个值，该值确定当前文件是否为只读。
- LastAccessTime:获取或设置上次访问当前文件或目录的时间。(继承自 FileSystemInfo)
- LastAccessTimeUtc:获取或设置上次访问当前文件或目录的时间（UTC）。(继承自 FileSystemInfo)
- LastWriteTime:获取或设置上次写入当前文件或目录的时间。(继承自 FileSystemInfo)
- LastWriteTimeUtc:获取或设置上次写入当前文件或目录的时间（UTC）。(继承自 FileSystemInfo)
- Length:获取当前文件的大小（以字节为单位）。
- Name:获取文件的名称。

```csharp
// 1. 获取文件基本信息
[Test]
public void TestInfo01()
{
    FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
    Console.WriteLine($"FullName:{fileInfo.FullName},Extension:{fileInfo.Extension}," +
        $"ReadOnly:{fileInfo.IsReadOnly},Length:{fileInfo.Length},CreateTime:{fileInfo.CreationTime}," +
        $"LastWriteTime:{fileInfo.LastWriteTime}");
}
```

运行结果如下：

## 四、使用——操作文本

1. **FileInfo.AppendText()**

**注解：**创建一个 StreamWriter，该 FileInfo实例所表示的文件追加文本。

```csharp
// 2. 操作文件——追加文本
[Test]
public void TestInfo02()
{
    FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
    using StreamWriter writer = fileInfo.AppendText();
    writer.WriteLine($"AppendText:TestInfo02,Date{DateTime.Now}");
}
```

2. **FileInfo.OpenRead()**

**注解：**创建只读 FileStream。

```csharp
// 3. 操作文件——打开并读取
[Test]
public void TestInfo03()
{
    // 如果需要对某个文件进行多次操作，建议使用 FileInfo 对象
    // 如果需要获取文件信息，也需要使用 FileInfo
    // 如果只是对文件单次操作，则可以使用 File 静态类
    FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
    var stream = fileInfo.OpenRead(); //只读
    using StreamReader reader = new StreamReader(stream);
    var str = reader.ReadLine();
    Console.WriteLine(str);
}
```
