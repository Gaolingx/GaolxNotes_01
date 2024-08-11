# C#高级编程之——反射（九）

## 三、反射入门——操作方法

在上一个章节中我们学习了如何通过反射操作构造函数并创建对象，今天我们来学习如何通过反射操作方法。

### 详细知识点

**关于Type.GetMethods：**

`Type.GetMethods` 是 .NET 框架中 `System.Type` 类的一个方法，用于获取当前 `Type` 对象所表示的类型中定义的所有公共方法。这个方法有几个重载版本，允许你以不同的方式检索方法信息。下面是对 `Type.GetMethods` 方法及其重载的简要介绍：

1. Type.GetMethods()

这是最基本的重载版本，不需要任何参数。它返回当前类型中定义的所有公共实例方法，但不包括继承的方法。返回的 `MethodInfo` 数组按照声明的顺序（即源代码中的顺序）进行排序。

```csharp
MethodInfo[] methods = typeof(MyClass).GetMethods();
```

2. Type.GetMethods(BindingFlags bindingAttr)

这个重载版本接受一个 `BindingFlags` 枚举值作为参数，允许你指定搜索方法的条件。`BindingFlags` 枚举允许你指定搜索是应该包括公共方法、非公共方法、实例方法、静态方法等。通过组合这些标志，你可以非常灵活地控制搜索的范围。

```csharp
MethodInfo[] methods = typeof(MyClass).GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
```

3. Type.GetMethods(Type returnType)

这个重载版本允许你指定返回类型，并返回所有具有指定返回类型的方法。但是，需要注意的是，在 .NET Framework 和 .NET Core 的官方文档中，并没有直接接受 `Type` 作为参数的 `GetMethods` 重载。这通常是通过结合使用 `BindingFlags` 和 LINQ 来实现的，例如：

```csharp
MethodInfo[] methods = typeof(MyClass).GetMethods(BindingFlags.Public | BindingFlags.Instance)
    .Where(m => m.ReturnType == typeof(int)).ToArray();
```

4. Type.GetMethods(Type returnType, BindingFlags bindingAttr)

虽然直接接受 `Type` 和 `BindingFlags` 作为参数的 `GetMethods` 重载在官方文档中不存在，但你可以通过组合使用 `BindingFlags` 和 LINQ 来达到类似的效果，如上面的示例所示。


操作：

一、获取类的所有方法

在main函数中执行以下方法，观察控制台输出

```csharp
//获取所有方法
public static void TestGetAllMethod01()
{
    Type type01 = typeof(StudentInfo);
    var methodInfos = type01.GetMethods();

    int i = 1;
    foreach (var m in methodInfos)
    {
        Console.WriteLine($"{i++}. {nameof(StudentInfo)} 方法中名称：{m?.Name},返回值类型：{m?.ReturnType}");
    }
}

public static void TestGetAllMethod02()
{
    Type type01 = typeof(StudentInfo);
    var methodInfos = type01.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

    int i = 1;
    foreach (var m in methodInfos)
    {
        Console.WriteLine($"{i++}. {nameof(StudentInfo)} 方法中名称：{m?.Name},返回值类型：{m?.ReturnType}");
    }
}

```

控制台分别输出如下，可以看到GetMethods默认输出了StudentInfo类中所有公有方法，当我们指定 BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic 后，则输出StudentInfo类中公有和私有方法。

拓展：
在C#中，一个类可以包含多种方法，这些方法有的来自于基类（如`System.Object`），有的是自定义的。几乎所有的C#类（除非显式继承自另一个非`Object`的类）都会继承这些方法。下面是对这些方法的简要说明：

1. GetType()
   - `GetType` 方法用于获取当前实例的 `Type`。`Type` 对象包含有关类型的元数据，如类型成员（字段、方法、属性等）和类型声明。这允许你在运行时检查对象的类型信息。

2. MemberwiseClone()
   - `MemberwiseClone` 创建一个浅表副本（shallow copy）的当前对象。这意味着它复制对象的字段，但不复制字段引用的对象本身。对于值类型字段，这基本上等同于赋值；但对于引用类型字段，副本和原始对象将共享对同一个对象的引用。这个方法主要用于对象的浅复制。

3. Finalize()
   - `Finalize`（或称为析构函数，虽然它在C#中不是以析构函数的语法`~ClassName()`声明的）是垃圾回收过程中的一个方法，它在对象被垃圾回收器回收之前被调用。这允许对象在销毁前执行清理操作，如释放非托管资源（如文件句柄、数据库连接等）。然而，由于性能原因，频繁使用 `Finalize` 并不推荐；C#提供了更安全的机制（如`IDisposable`接口）来管理非托管资源。可以简单理解为：对象终结者，作用就是释放资源。GC（垃圾回收器）底层调用的就是Finalize方法（笔者注）

4. ToString()
   - `ToString` 方法返回一个表示当前对象的字符串。默认情况下，它返回类的完全限定名（包括命名空间）。

5. Equals(object obj)
   - `Equals` 方法用于确定指定的对象是否等于当前对象。默认情况下，它比较两个对象的引用（即它们是否是内存中的同一个对象）。但是，许多类（特别是那些表示值的类，如`int`、`double`的包装类）都会重写这个方法，以基于对象的值而不是引用来比较对象。

6. GetHashCode()
   - `GetHashCode` 方法返回一个整数，该整数是根据对象的内部状态生成的哈希码。

二、获取并调用指定方法
已知我们的StudentInfo类有如下方法及重载：

```csharp
public void Run()
{
    Console.WriteLine($"我的名字是{Name}，是米游社的一名创作者");
}

//有参无返回值方法
public void Run2(int age)
{
    Console.WriteLine($"我的名字是{Name}，我今年{age}岁了");
}

//有参有返回值私有方法
public string Run3(string name)
{
    return $"我的名字是{name}，我是一个私有方法";
}
```

1. 获取并调用无参数方法

在main函数中执行以下方法，观察控制台输出

```csharp
//获取并调用单个方法
//无参数，公有方法
public static void TestGetMethod01()
{
    Type type01 = typeof(StudentInfo);

    var methodInfo = type01.GetMethod("Run");
    //调用方法
    // 常规操作：对象.方法名();

    var stu = Activator.CreateInstance(type01); //创建对象
    methodInfo?.Invoke(stu, null); //默认返回值为object类型
}
```

控制台输出如下，说明我们成功调用了StudentInfo类的 Run 方法。

2. 获取并调用带参数方法

在main函数中执行以下方法，观察控制台输出

```csharp
//带参数，公有方法
public static void TestGetMethod02()
{
    Type type01 = typeof(StudentInfo);

    var methodInfo = type01.GetMethod("Run2");
    //调用方法
    // 常规操作：对象.方法名();

    var stu = Activator.CreateInstance(type01); //此处的参数指的是Run2方法的参数age
    methodInfo?.Invoke(stu, new object?[] { 19 }); //默认返回值为object类型
}
```

控制台输出如下，说明我们成功调用了StudentInfo类的 Run2 方法。

3. 获取并调用带参数的私有方法并接收返回值

为了测试调用私有方法，我们在StudentInfo类中新增如下方法：

```csharp
private string Run4(string name)
{
    return $"我的名字是{name}，我是一个私有方法";
}
```

在main函数中执行以下方法，观察控制台输出

```csharp
//带参数，有返回值，私有方法
public static void TestGetMethod03()
{
    Type type01 = typeof(StudentInfo);

    var methodInfo = type01.GetMethod("Run4", BindingFlags.Instance | BindingFlags.NonPublic); //私有方法
    //调用方法
    // 常规操作：对象.方法名();

    var stu = Activator.CreateInstance(type01); //此处的参数指的是Run2方法的参数age
    var resultObj = methodInfo?.Invoke(stu, new object?[] { "爱莉小跟班gaolx" }); //默认返回值为object类型
    Console.WriteLine(resultObj);
}
```

控制台输出如下，说明我们成功调用了StudentInfo类的私有方法 Run4。
