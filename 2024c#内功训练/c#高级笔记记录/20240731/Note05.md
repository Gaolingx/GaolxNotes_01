# C#高级编程之——反射（五）操作属性

在上期教程中，我们学习了Activator类的基本使用，我们通过Activator.CreateInstance调用有参/有参构造方法反射创建实例，今天我们开始学习通过反射操作属性，即对属性赋值、获取属性的值等操作。

## 详细知识点

**关于Type.GetProperty：**
Type.GetProperty方法用于获取当前Type对象所表示的类型中指定的属性信息，并返回一个PropertyInfo对象，该对象包含有关该属性的元数据（如名称、类型、访问级别等）。注：该方法默认只能获取到公共（public）的成员属性，如需通过反射访问非公共（如私有、受保护等）的成员属性，你可以使用 Type.GetProperty 方法的一个重载版本，它接受 BindingFlags 枚举作为参数。

**关于PropertyInfo.SetValue：**
PropertyInfo类表示某个属性的信息，并且它提供了一个SetValue方法，用于设置与该属性关联的值。这是通过反射API实现的，允许在运行时动态地修改对象的属性。

示例：

```csharp
//1.第一步获取类型
var tp = typeof(StudentInfo);
//获取某个属性
var propInfo = tp.GetProperty("NickName");

//2.通过反射创建对象(其实调用的是构造函数)
var obj = Activator.CreateInstance(tp);
propInfo.SetValue(obj，"李四");
var val = propInfo.GetValue(obj);//获取属性的值
Console.WriteLine(val);

```

操作：

1. 在之前的StudentInfo类中，找到属性Name，将其改为：public string Name { get; set; } = "爱莉大跟班gaolx"; ，便于后期测试。
2. 在Main方法中执行如下代码，观察控制台输出：

```csharp
//操作属性
public static void TestOperationProp01()
{
    /*
     * 操作属性（常规操作）：
     * StudentInfo obj = new();
     * obj.Age = 19; // 对象.属性名 = 属性值
     * var age = obj.Age; // var 变量名 = 对象.属性名
     */

    var tp = typeof(StudentInfo);
    // 获取属性
    var propInfo = tp.GetProperty("Name");

    // 创建对象
    var instance = Activator.CreateInstance(tp);
    // 为属性赋值
    propInfo.SetValue(instance, "爱莉小跟班gaolx");

    // 获取属性的值
    var name = propInfo.GetValue(instance);
    Console.WriteLine($"我的名字是{name}");
}
```

观察控制台输出可以发现我们成功修改了instance实例中Name属性的值。

结论：通过反射（Reflection）操作属性与直接通过new关键字创建对象后直接访问其属性名相比，反射允许程序在运行时动态地查询和操作对象的类型信息，包括其属性、方法等。这意味着你可以编写更加灵活和可扩展的代码，而不需要在编译时就固定对象的类型和行为。这也是许多框架高通用性的根基。
