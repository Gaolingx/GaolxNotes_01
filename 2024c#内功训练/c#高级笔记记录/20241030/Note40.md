# C#高级编程之——I/O文件流（五）目录操作

## 1. Directory

### 一、定义

- 命名空间:
System.IO
- 程序集:
System.Runtime.dll

### 二、注解

公开用于通过目录和子目录进行创建、移动和枚举的**静态方法**。 此类不能被继承。

Directory将 类用于复制、移动、重命名、创建和删除目录等典型操作。

- 若要创建目录，请使用方法之 CreateDirectory 一。
- 若要删除目录，请使用方法之 Delete 一。
- 若要获取或设置应用的当前目录，请使用 GetCurrentDirectory 或 SetCurrentDirectory 方法。
- 若要操作DateTime与目录的创建、访问和写入相关的信息，请使用 和 SetCreationTime等SetLastAccessTime方法。

类的 Directory 静态方法对所有方法执行安全检查。 如果要多次重用对象，请考虑改用 对应的实例方法DirectoryInfo，因为安全检查并非始终是必需的。

如果只执行一个与目录相关的操作，则使用静态 Directory 方法而不是相应的 DirectoryInfo 实例方法可能更有效。 大多数 Directory 方法都需要要操作的目录的路径。

## 2. DirectoryInfo

### 一、定义

- 命名空间:
System.IO
- 程序集:
System.Runtime.dll

公开用于创建、移动和枚举目录和子目录的实例方法。 此类不能被继承。

### 二、注解

DirectoryInfo将 类用于典型的操作，例如复制、移动、重命名、创建和删除目录。

如果要多次重用对象，请考虑使用 的DirectoryInfo**实例方法**，而不是类的Directory相应静态方法，因为安全检查并不总是必要的。

### 三、属性

<div class="has-inner-focus"><table class="table table-sm margin-top-none" aria-label="表 3">
        	<tbody><tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.attributes?view=net-8.0#system-io-filesysteminfo-attributes" data-linktype="relative-path">Attributes</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置当前文件或目录的特性。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.creationtime?view=net-8.0#system-io-filesysteminfo-creationtime" data-linktype="relative-path">Creation<wbr>Time</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置当前文件或目录的创建时间。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.creationtimeutc?view=net-8.0#system-io-filesysteminfo-creationtimeutc" data-linktype="relative-path">Creation<wbr>Time<wbr>Utc</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置当前文件或目录的创建时间，其格式为协调世界时 (UTC)。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr>
	<td>
	<span class="break-text">
		<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo.exists?view=net-8.0#system-io-directoryinfo-exists" data-linktype="relative-path">Exists</a>
	</span>
</td>
	<td class="has-text-wrap">
		<p>获取指示目录是否存在的值。</p>
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.extension?view=net-8.0#system-io-filesysteminfo-extension" data-linktype="relative-path">Extension</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取文件名的扩展名部分，包括前导点 <code data-dev-comment-type="c">.</code> （即使它是整个文件名）或空字符串（如果不存在扩展名）。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 ">
	<td>
	<span class="break-text">
		<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo.fullname?view=net-8.0#system-io-directoryinfo-fullname" data-linktype="relative-path">Full<wbr>Name</a>
	</span>
</td>
	<td class="has-text-wrap">
		<p>获取目录的完整路径。</p>
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.fullname?view=net-8.0#system-io-filesysteminfo-fullname" data-linktype="relative-path">Full<wbr>Name</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取目录或文件的完整目录。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.lastaccesstime?view=net-8.0#system-io-filesysteminfo-lastaccesstime" data-linktype="relative-path">Last<wbr>Access<wbr>Time</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置上次访问当前文件或目录的时间。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.lastaccesstimeutc?view=net-8.0#system-io-filesysteminfo-lastaccesstimeutc" data-linktype="relative-path">Last<wbr>Access<wbr>Time<wbr>Utc</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置上次访问当前文件或目录的时间，其格式为协调世界时 (UTC)。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.lastwritetime?view=net-8.0#system-io-filesysteminfo-lastwritetime" data-linktype="relative-path">Last<wbr>Write<wbr>Time</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置上次写入当前文件或目录的时间。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.lastwritetimeutc?view=net-8.0#system-io-filesysteminfo-lastwritetimeutc" data-linktype="relative-path">Last<wbr>Write<wbr>Time<wbr>Utc</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置上次写入当前文件或目录的时间，其格式为协调世界时 (UTC)。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.linktarget?view=net-8.0#system-io-filesysteminfo-linktarget" data-linktype="relative-path">Link<wbr>Target</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取位于 中的 <a class="no-loc" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.fullname?view=net-8.0#system-io-filesysteminfo-fullname" data-linktype="relative-path">FullName</a>链接的目标路径，如果 <code data-dev-comment-type="langword">null</code> 此 <a class="no-loc" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a> 实例不表示链接，则为 。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
        	<tr>
	<td>
	<span class="break-text">
		<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo.name?view=net-8.0#system-io-directoryinfo-name" data-linktype="relative-path">Name</a>
	</span>
</td>
	<td class="has-text-wrap">
		<p>获取此 <a class="no-loc" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo?view=net-8.0" data-linktype="relative-path">DirectoryInfo</a> 实例的名称。</p>
	</td>
        	</tr>
        	<tr>
	<td>
	<span class="break-text">
		<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo.parent?view=net-8.0#system-io-directoryinfo-parent" data-linktype="relative-path">Parent</a>
	</span>
</td>
	<td class="has-text-wrap">
		<p>获取指定的子目录的父目录。</p>
	</td>
        	</tr>
        	<tr>
	<td>
	<span class="break-text">
		<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.directoryinfo.root?view=net-8.0#system-io-directoryinfo-root" data-linktype="relative-path">Root</a>
	</span>
</td>
	<td class="has-text-wrap">
		<p>获取目录的根部分。</p>
	</td>
        	</tr>
        	<tr data-moniker=" dotnet-uwp-10.0 netcore-1.0 netcore-1.1 netstandard-1.3 netstandard-1.4 netstandard-1.6 net-5.0 net-6.0 net-7.0 net-8.0 net-9.0 netcore-2.0 netcore-2.1 netcore-2.2 netcore-3.0 netcore-3.1 netframework-1.1 netframework-2.0 netframework-3.0 netframework-3.5 netframework-4.0 netframework-4.5 netframework-4.5.1 netframework-4.5.2 netframework-4.6 netframework-4.6.1 netframework-4.6.2 netframework-4.7 netframework-4.7.1 netframework-4.7.2 netframework-4.8 netframework-4.8.1 netstandard-2.0 netstandard-2.1 ">
	<td>
		<span class="break-text">
			<a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo.unixfilemode?view=net-8.0#system-io-filesysteminfo-unixfilemode" data-linktype="relative-path">Unix<wbr>File<wbr>Mode</a>
		</span>
	</td>
	<td class="has-text-wrap">
		<p>获取或设置当前文件或目录的 Unix 文件模式。</p>
		(继承自 <a class="xref" href="https://learn.microsoft.com/zh-cn/dotnet/api/system.io.filesysteminfo?view=net-8.0" data-linktype="relative-path">FileSystemInfo</a>)
	</td>
        	</tr>
	</tbody></table></div>

### 四、使用

#### 3.1 创建目录

#### 1. Directory.CreateDirectory(String)

**注解**：在指定路径中创建所有目录和子目录，除非它们已经存在。

```csharp
// 1. 创建目录
[Test]
public void TestCreateDirectory01()
{
    DirectoryInfo info = Directory.CreateDirectory("D://App/logs3/logs");
}
```

运行结果如下：

#### 3.2 删除目录

#### 1. Directory.Delete(String)

**注解**：从指定路径删除**空目录**。

```csharp
// 2. 删除目录
[Test]
public void TestDeleteDirectory01()
{
    Directory.Delete("D://App/logs3/logs");
}
```

运行效果如下：

#### 2. Directory.Delete(String, Boolean)

**注解**：删除指定的目录，并删除该目录中的所有子目录和文件（如果表示）。

#### 3. DirectoryInfo.Delete()

**注解**：如果此 DirectoryInfo 为空则将其删除。

#### 4. DirectoryInfo.Delete(Boolean)

**注解**：删除 DirectoryInfo 的此实例，指定是否删除子目录和文件。

```csharp
[Test]
public void TestDeleteDirectory02()
{
    DirectoryInfo info = new DirectoryInfo("D://App/logs3");
    // 只能删除空目录
    //info.Delete();
    // 删除整个目录，包括子目录及文件
    info.Delete(true);
}
```

运行效果如下：

#### 3.3 检索目录

**方法及重载**：

- EnumerateDirectories():返回当前目录中的目录信息的可枚举集合。
- EnumerateDirectories(String):返回与指定的搜索模式匹配的目录信息的可枚举集合。
- EnumerateDirectories(String, EnumerationOptions):返回与指定的搜索模式和枚举选项匹配的目录信息的可枚举集合。

```csharp
// 3. 获取所有目录
[Test]
public void TestGetAllDirectory01()
{
    DirectoryInfo info = new DirectoryInfo("D://App/logs3");
    IEnumerable<DirectoryInfo> dirs = info.EnumerateDirectories(); //只能获取目录

    foreach (DirectoryInfo dir in dirs)
    {
        Console.WriteLine(dir.FullName);
    }
    Console.WriteLine("---------------------");

    // 需要正则匹配
    // 获取所有目录
    IEnumerable<DirectoryInfo> dirs2 = info.EnumerateDirectories("*", SearchOption.AllDirectories);
    foreach (DirectoryInfo dir in dirs2)
    {
        Console.WriteLine(dir.FullName);
    }
}
```

运行效果如下：

#### 3.4 获取目录的文件列表

**方法及重载**：

- GetFiles():返回当前目录的文件列表。（不能获取子目录的文件）
- GetFiles(String):返回当前目录中与给定的搜索模式匹配的文件列表。
- GetFiles(String, EnumerationOptions):返回当前目录中与指定的搜索模式和枚举选项匹配的文件列表。
- GetFiles(String, SearchOption):返回与给定的搜索模式匹配并且使用某个值确定是否在子目录中进行搜索的当前目录的文件列表。

```csharp
// 4. 获取目录的文件列表
[Test]
public void TestGetFiles01()
{
    DirectoryInfo info = new DirectoryInfo("D://App/logs3");

    //获取当前目录下的所有文件
    var files = info.GetFiles();
    foreach (var file in files)
    {
        Console.WriteLine($"FullName:{file.FullName},Length:{file.Length}");
    }

    //获取当前目录及子目录所有文件
    //*.*:所有文件名以及所有的文件后缀名
    var files2 = info.GetFiles("*.*",SearchOption.AllDirectories);
    foreach (var file in files2)
    {
        Console.WriteLine($"FullName:{file.FullName},Length:{file.Length}");
    }
}
```

运行效果如下：
