# C#高级编程之——反射（三）

## 三、反射入门——反射的使用（案例篇）

在上期教程中，我们讲解了Type类型的概念，简单使用了typeof 运算符以及GetType 方法通过实例对象获取类型，接下来我们继续深入讲解Type类的使用。

### 详细知识点

**关于propertyInfo：**

- CanRead：这是一个布尔值（bool），指示是否可以通过该 PropertyInfo 实例获取属性的值。如果属性的访问器（getter）是公共的（public）或者该属性具有公共的访问器，并且当前反射的上下文（比如访问级别）允许访问它，则 CanRead 返回 true。简单理解：如果这个字段带有get访问器，则CanRead为true，如果带有set访问器，则CanWrite为true，CanWrite同理。
- CanWrite：这也是一个布尔值（bool），指示是否可以通过该 PropertyInfo 实例设置属性的值。如果属性的访问器（setter）是公共的（public）或者该属性具有公共的访问器，并且当前反射的上下文允许访问它，则 CanWrite 返回 true。
- GetMethod：这是一个 MethodInfo 对象，代表用于设置属性值的访问器（getter）方法。即反编译后这个get属性代表什么方法，即get访问器被编译成的方法名称，如果属性是只读的（即没有setter），则此属性为 null。SetMethod同理。
- MemberType：这是一个MemberType的枚举，返回成员的种类—字段，构造器，属性，方法等。
- PropertyType：这是一个 Type 对象，表示属性的数据类型。它指明了属性的值是什么类型的。简单理解：属性的类型，如int值类型的字段，PropertyType为System.Int32。
- Name：用于获取变量、类型或成员的简单(非限定)字符串名称。
- Attributes：特性，这是一个 AttributeCollection 对象，它包含了与该属性关联的所有自定义属性的集合。这个集合允许你查询并操作与属性相关联的自定义属性。
- CustomAttributes：自定义特性，这个属性实际上在 PropertyInfo 类中并不直接存在，但你可能是在引用通过反射API可以获取的自定义属性的能力。通常，我们会使用 GetCustomAttributes 或 GetCustomAttributes<T>() 方法来获取与 PropertyInfo 关联的自定义属性。这些方法返回自定义属性的数组，可以让你查询和操作这些属性。
- DeclaringType：这是一个 Type 对象，表示声明该属性的类型。如果属性是接口的一部分，则 DeclaringType 返回接口本身；如果属性是类或结构的一部分，则 DeclaringType 返回类或结构的类型。简单理解：表示属性属于什么类下的，例如int a和int b属于Class C下，则DeclaringType为C。
- BindingFlags：这是一个枚举类型，用于指定在反射操作（如获取类型成员）时应该使用的绑定标志。如Public、NonPublic、Static、Instance等。

**Type类的具体使用案例：**

1. 获取所有属性

```csharp
public static void TestGetAllProperty()
{
    Type type01 = typeof(StudentInfo);
    PropertyInfo[] propList = type01.GetProperties();
    //或 var propList = typeof(StudentInfo).GetProperties();

    foreach (var propertyInfo in propList)
    {
        Console.WriteLine($"{nameof(type01)}类型中属性的名称:{propertyInfo.Name},类型:{propertyInfo.PropertyType}");
    }
}
```

通过对"Console.WriteLine($"StudentInfo类型中属性的名称:{propertyInfo.Name},类型:{propertyInfo.PropertyType}");"进行断点，观察PropertyInfo中各字段的信息。我们发现GetMethod为get_Age()，而property的name为Age，说明get访问器被编译为get_Age方法。

2. 获取指定类型的属性

```csharp
public static void TestGetPropertyByName(string name)
{
    //获取类型
    Type type01 = typeof(StudentInfo);
    var propInfo = type01.GetProperty(name); //获取type01类型中含有Age属性的名称的方法
    Console.WriteLine($"{nameof(type01)}类型中属性的名称:{propInfo?.Name},类型:{propInfo?.PropertyType}");

}
```

运行结果如下：
