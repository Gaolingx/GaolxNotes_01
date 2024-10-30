# C#高级编程之——I/O文件流（三）DirectoryInfo 目录操作

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

### 三、使用

#### 3.1 创建目录

**1. Directory.CreateDirectory(String)**

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

**1. Directory.Delete(String)**

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

**2. Directory.Delete(String, Boolean)**

**注解**：删除指定的目录，并删除该目录中的所有子目录和文件（如果表示）。

**3. DirectoryInfo.Delete()**

**注解**：如果此 DirectoryInfo 为空则将其删除。

**4. DirectoryInfo.Delete(Boolean)**

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
