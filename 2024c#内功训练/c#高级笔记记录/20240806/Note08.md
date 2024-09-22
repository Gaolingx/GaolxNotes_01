# C#高级编程之——反射（八）操作构造函数

在前两个章节中我们学习了如何加载和获取程序集，今天我们来学习如何通过反射操作构造函数。

## 详细知识点

**1.1 关于Type.GetConstructor类：**
Type.GetConstructors 方法是 System.Type 类的一个成员，用于获取表示当前类型构造函数的 ConstructorInfo 对象数组。这些构造函数信息可以用来在运行时动态地创建对象实例或检查类型的构造函数特性。默认情况下，不带参数的 GetConstructors 方法只返回公共构造函数。

该方法有如下重载：

- Type.GetConstructors()
  这是最基本的不带参数的 GetConstructors 方法。它返回当前类型的所有公共构造函数（即那些访问级别为 public 的构造函数）的 ConstructorInfo 数组。如果类型没有公共构造函数，则返回一个空数组。
- Type.GetConstructors(BindingFlags)
  这个重载版本的 GetConstructors 方法允许你通过 BindingFlags 枚举来指定要搜索的构造函数的访问级别和其他特性。BindingFlags 是一个位掩码，允许你组合多个值来精确控制搜索过程。

**1.2 示例：**

```csharp
var type = typeof(studentInfo);

//获取构造方法studentInfo(string nickName, int age)
var constructorInfo = type.GetConstructor(new []{typeof(string), typeof(int)});
//根据构造方法，创建对象
var student = constructorInfo.Invoke(new Type[] { });
var student2 = constructorInfo.Invoke(new object?[] {"xxx", 18}); //有参
//获取所有的方法，包括私有方法与继承至 object对象的方法
var methodInfos = type.GetMethods(BindingFlags.Instance|BindingFlags.NonPub1ic);
//获取指定方法
var runMethod = type.GetMethod("Run");
//第二个参数表示被调用的方法参数，null 表示此方法为无参方法
runMethod.Invoke(student, null);

```

**1.3 操作：**
已知我们的StudentInfo类中有如下两个构造方法：

```csharp
public StudentInfo()
{

}

public StudentInfo(string name, int age)
{
    Name = name;
    Age = age;
}
```

1. 获取无参构造方法创建对象

```csharp
public static void TestConstructor01()
{
    Type type01 = typeof(StudentInfo);
    //获取无参构造方法
    var constructor = type01.GetConstructor(new Type[] { });
    //创建对象 方法1
    var obj = constructor?.Invoke(null); //无参构造，所以参数为空

    //创建对象 方法2
    var obj2 = Activator.CreateInstance(type01);

}
```

2. 获取有参构造方法创建对象

```csharp
public static void TestConstructor02()
{
    Type type01 = typeof(StudentInfo);
    //获取有参构造方法
    var constructor = type01.GetConstructor(new Type[] {typeof(string),typeof(int)}); //需要指定构造方法参数的类型
    //创建对象 方法1
    StudentInfo obj = constructor?.Invoke(new object?[] { "爱莉小跟班gaolx", 18 }) as StudentInfo; //指定参数的值

    //创建对象 方法2
    StudentInfo obj2 = Activator.CreateInstance(type01, "爱莉大跟班gaolx", 19) as StudentInfo;

    Console.WriteLine($"{nameof(obj)} Name is {obj?.Name}");
    Console.WriteLine($"{nameof(obj2)} Name is {obj2?.Name}");
}
```

控制台输出结果如下：
