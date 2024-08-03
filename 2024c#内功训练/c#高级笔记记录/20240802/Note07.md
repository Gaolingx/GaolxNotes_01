# C#高级编程之——反射（七）

## 三、反射入门——加载/获取程序集

在前两个章节中我们学习了如何通过反射操作属性和字段，今天我们来学习如何加载和获取程序集。

### 详细知识点

**关于Assembly类：**

- 介绍：在C#中，Assembly 类是.NET Framework 和 .NET Core 中非常重要的一个类，它位于 System.Reflection 命名空间中。Assembly 类提供了加载、检查和使用程序集中的类型、方法和字段等成员的能力。
- 主要功能：
  1. 加载和卸载程序集：Assembly 类提供了加载和卸载程序集的方法，允许动态地加载和使用程序集中的代码。
  2. 检索程序集信息：可以获取程序集的名称、版本、文化信息、公钥令牌等元数据信息。
  3. 访问程序集中的类型：通过 Assembly 类，可以获取程序集中定义的所有类型（Type 对象），进而可以创建这些类型的实例、调用方法、访问字段等。
  4. 资源访问：可以访问嵌入在程序集中的资源，如图片、文本文件等。
- 常用方法和属性：
  1. GetExecutingAssembly：获取当前执行的程序集。
  2. Load：根据指定的程序集名称或路径加载程序集。
  3. LoadFile：加载指定的文件作为程序集。
  4. ManifestModuleName：获取包含程序集清单的文件的名称。
  5. FullName：获取程序集的完整名称，包括版本号、文化信息和公钥令牌。
  6. GetTypes：获取当前程序集中定义的所有公共类型的 Type 对象数组。
  7. GetManifestResourceStream：获取嵌入在程序集中的指定名称的资源流。

示例：

```csharp
//加载程序集名称程序必须在当前Bin目录下
var assembly1 = Assembly.Load("ClassLibrary1");

//路径加载程序集名称
var assembly2 = Assembly.LoadFile(@"d:\path\ClassLibrary1.dll");

//加载程序集后创建对象,第二个参数null 是构造函数的参数
object obj = assembly1.CreateInstance("命名空间.类名", false);

//获取程序集完整路径
string location = Assembly.GetExecutingAssembly().Location;

//获取程序集名称
string file = Assembly.GetExecutingAssembly().GetName().Name;

//获取程序集版本号
string version = Assembly.GetExecutingAssembly().GetName().Version.ToString();
```

操作：

以本工程（ConsoleApp4）为例演示，大家可按照自己的解决方案修改

1. 加载程序集

在Assembly.Load(xxx)处断点，运行，观察程序集包含哪些信息。注：如果未找到程序集则会抛出 System.IO.FileNotFoundException:“Could not load file or assembly 'ConsoleApp41, Culture=neutral, PublicKeyToken=null'. 系统找不到指定的文件。” 的异常。诸如此类问题要么是路径不正确要么是缺少程序集引用等，大家可以自行检查。

```csharp
public static void TestGetAssembly01()
{
    var assembly = Assembly.Load("ConsoleApp4");
}
```

运行结果如下

2. 通过路径加载程序集

在Assembly.LoadFile(xxx)处断点，运行，观察程序集包含哪些信息。

```csharp
public static void TestGetAssembly02()
{
    var assembly = Assembly.LoadFile(@"F:\GitHub\GaolxNotes_01\2024c#内功训练\c#高级笔记记录\20240619\ConsoleApp4\ConsoleApp4\bin\Debug\net8.0\ConsoleApp4.dll");
}
```

运行结果如下

3. 外部加载程序集后创建对象


