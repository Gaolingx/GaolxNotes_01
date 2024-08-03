# C#高级编程之——反射（四）

## 三、反射入门——Activator 创建对象及其原理

在上期教程中，我们学习了Type类的基本使用，我们使用反射获取属性、获取字段、类的命名空间、全称、基类等，今天我们开始学习如何通过反射创建实例。

### 详细知识点

**关于Activator类：**

- 介绍：Activator 类是一个位于 System 命名空间下的静态类，它提供了在运行时创建对象实例的方法，而无需直接调用类的构造函数。这对于实现依赖注入、动态类型创建或当你需要在运行时根据某些条件动态创建对象时特别有用。
- 用法：
  1. CreateInstance(Type type): 创建一个指定类型的实例。如果类型具有无参数的构造函数，这将非常有用。
  2. CreateInstance(Type type, params object[] args): 创建一个指定类型的实例，并传递一个参数数组给构造函数。这允许你调用具有参数的构造函数。
  3. CreateInstance<T>(): 这是 CreateInstance(Type type) 的泛型版本，它简化了类型参数的传递，使得代码更加简洁。
  4. CreateInstance<T>(params object[] args): 这是 CreateInstance(Type type, params object[] args) 的泛型版本，允许你以类型安全的方式传递参数给构造函数。

```csharp
//方式一
var tp = typeof(StudentInfo);
var obj = Activator.CreateInstance(tp);
```

使用：

1. 调用无参构造方法创建实例
在Main方法中执行如下代码：

```csharp
public static void TestCreateInstance()
{
    Type type01 = typeof(StudentInfo);
    var instance = Activator.CreateInstance(type01) as StudentInfo;
    instance.Age = 19;
    instance.Name = "爱莉小跟班gaolx";
    Console.WriteLine($"我的年龄是{instance?.Age},名字是{instance?.Name}");
}
```

通过对Console.WriteLine($"...)断点调试我们发现实例不为空，说明对象创建成功。

2. 调用有参构造方法创建实例

现在，我们在StudentInfo类中有如下带有两个参数的构造函数：

```csharp
public StudentInfo(string name, int age)
{
    Name = name;
    Age = age;
}
```

在Main方法中执行如下代码：

```csharp
public static void TestCreateInstance02()
{
    Type type01 = typeof(StudentInfo);
    var instance = Activator.CreateInstance(type01, "爱莉小跟班gaolx", 19) as StudentInfo;
    Console.WriteLine($"我的年龄是{instance?.Age},名字是{instance?.Name}");
}
```

运行代码，如果没有抛出异常，观察控制台输出我们可以判断调用了StudentInfo类中的有参构造函数。结论：再次的验证 CreateInstance 其实是调用了StudentInfo类的构造方法。

总结：CreateInstance 底层其实是调用了 无参构造方法。对象创建的唯一途径只能是构造方法被调用。