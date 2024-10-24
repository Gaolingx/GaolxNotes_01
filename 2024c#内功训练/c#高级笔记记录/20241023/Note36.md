# C#高级编程之——I/O文件流

## 什么是文件流

I/O 文件流（Input/Output File Streams）是指用于在程序与文件系统之间传输数据的机制。它允许程序读取（输入）和写入（输出）文件中的数据。例如日志的写入，图片、音频等二进制的处理。

## 一、File 类

### 定义

- 命名空间:System.IO
- 程序集:System.Runtime.dll
- 功能：提供用于创建、复制、删除、移动和打开单个文件的静态方法，并有助于创建 FileStream 对象。

### 注解

使用 File 类执行一次复制、移动、重命名、创建、打开、删除和追加到单个文件等典型操作。 还可以使用 File 类来获取和设置文件属性或 DateTime 与创建、访问和写入文件相关的信息。 如果要对多个文件执行操作，请参阅 Directory.GetFiles 或 DirectoryInfo.GetFiles。

创建或打开文件时，许多 File 方法返回其他 I/O 类型。 可以使用这些其他类型的进一步操作文件。 有关详细信息，请参阅特定的 File 成员，例如 OpenText、CreateText或 Create。

由于所有 File 方法都是静态的，因此，如果只想执行一个操作，则使用 File 方法比相应的 FileInfo 实例方法更有效。 所有 File 方法都需要你正在操作的文件的路径。

File 类的静态方法对所有方法执行安全检查。 如果要多次重复使用对象，请考虑改用相应的 FileInfo 实例方法，因为安全检查并不总是必要的。

默认情况下，向所有用户授予对新文件的完整读/写访问权限。

下表描述了用于自定义各种 File 方法行为的枚举。

<table aria-label="表 1" class="table table-sm margin-top-none">
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
</table>

注：在接受路径作为输入字符串的成员中，该路径的格式必须正确或引发异常。 例如，如果路径完全限定，但以空格开头，则不会在类的方法中剪裁路径。 因此，路径格式不正确，并引发异常。 同样，路径或路径的组合不能完全限定两次。 例如，在大多数情况下，“c：\temp c：\windows”也会引发异常。 使用接受路径字符串的方法时，请确保路径格式正确。

### 常用方法

1. **创建文件**
   - `File.Create(string path)`: 创建一个新文件，如果文件已存在，则覆盖它。返回一个 `FileStream` 对象，可以用来写入文件。
   - `File.Create(string path, int bufferSize)`: 类似于上面的方法，但允许你指定缓冲区大小。
   - `File.CreateText(string path)`: 创建一个新文件，如果文件已存在，则覆盖它。返回一个 `StreamWriter` 对象，可以用来写入文本。

2. **复制文件**
   - `File.Copy(string sourceFileName, string destFileName)`: 将现有文件复制到新文件。如果目标文件已存在，则覆盖它。
   - `File.Copy(string sourceFileName, string destFileName, bool overwrite)`: 类似于上面的方法，但允许你指定是否覆盖目标文件。

3. **删除文件**
   - `File.Delete(string path)`: 删除指定文件。如果文件不存在，会引发异常。

4. **检查文件是否存在**
   - `File.Exists(string path)`: 返回一个布尔值，指示指定路径的文件是否存在。

5. **获取文件信息**
   - `File.GetAttributes(string path)`: 获取文件的属性（如只读、隐藏等）。
   - `File.GetCreationTime(string path)`: 获取文件的创建时间。
   - `File.GetLastAccessTime(string path)`: 获取上次访问文件的时间。
   - `File.GetLastWriteTime(string path)`: 获取上次写入文件的时间。
   - `File.GetLength(string path)`: 获取文件的大小（以字节为单位）。

6. **移动文件**
   - `File.Move(string sourceFileName, string destFileName)`: 将现有文件移动到新位置，并提供新的文件名。

7. **打开文件**
   - `File.Open(string path, FileMode mode)`: 打开一个现有文件或创建一个新文件。返回一个 `FileStream` 对象。
   - `File.OpenRead(string path)`: 打开一个文件用于读取，并返回一个 `FileStream` 对象。
   - `File.OpenWrite(string path)`: 打开一个文件用于写入，如果文件不存在则创建它，并返回一个 `FileStream` 对象。

8. **读取文件**
   - `File.ReadAllBytes(string path)`: 读取文件的全部字节并将其作为一个字节数组返回。
   - `File.ReadAllLines(string path)`: 读取文件中的所有行并将其作为字符串数组返回。
   - `File.ReadAllText(string path)`: 读取文件的全部文本并将其作为一个字符串返回。

9. **写入文件**
   - `File.WriteAllBytes(string path, byte[] bytes)`: 将字节数组写入指定文件。如果文件已存在，则覆盖它。
   - `File.WriteAllLines(string path, string[] contents)`: 将字符串数组写入文件，每个字符串作为一行。如果文件已存在，则覆盖它。
   - `File.WriteAllText(string path, string contents)`: 将字符串写入文件。如果文件已存在，则覆盖它。

10. **附加内容到文件**
    - `File.AppendAllLines(string path, string[] contents)`: 将一个字符串数组作为单独的行附加到文件末尾。如果文件不存在，则创建它。
    - `File.AppendAllText(string path, string contents)`: 将一个字符串附加到文件末尾。如果文件不存在，则创建它。

### 使用

1. **文件追加**

运行以下代码，观察D://App/logs路径下是否生成log文件生成，并多次运行，再次查看log是否追加了文本。

```csharp
// 1. 添加文件路径
[Test]
public void TestAppendLines01()
{
    File.AppendAllLines("D://App/logs/log.txt", new List<string> {$"Log第一行,Time:{DateTime.Now}",
    $"Log第二行,Time:{DateTime.Now}"});

    //注:1. 路径不存在会抛出异常，但是文件不存在会自动新建
}
```

运行结果如下：
