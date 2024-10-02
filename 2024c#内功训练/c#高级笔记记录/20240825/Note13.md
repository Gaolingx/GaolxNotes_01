# C#高级编程之——特性（二）自定义特性

在上一节中我们介绍了C#特性的基本概念、用途以及自定义特性相关知识，今天我们来尝试创建一个自定义特性。

## 五、创建自定义特性

**5.1 目标：**创建一个描述类型信息的类，该特性仅对属性和字段生效。

**5.2 创建特性——操作步骤：**

1. 在当前的c#项目里新建一个名为MyDescriptionAttribute的类并继承自Attribute基类。注：自定义特性需要以Attribute结尾并继承自Attribute基类。
2. 在自定义的Attribute前加上AttributeUsage特性，用于定义特性类的使用方式。由于我们只希望该特性在属性和字段上生效，所以我们应该使用 [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = true)] 。
   - 说明：
   - 元特性：对特性进行描述的特性。
   - AttributeTargets.All：这个参数指定了你的自定义特性可以应用于所有的程序元素上。AttributeTargets 是一个枚举，它包含了许多成员，如 Class、Method、Property、Enum 等，用于指定特性可以应用的范围。
   - AllowMultiple = false：这个参数指定在同一个程序元素上不允许多次应用你的自定义特性。如果设置为 true，则表示同一个元素上可以使用多个该特性实例。
   - Inherited = true：这个参数指定如果你的自定义特性被应用于一个类，那么它的派生类也会继承这个特性（如果特性被应用于派生类所继承的成员上）。
3. 在MyDescriptionAttribute中添加属性Name。
4. 在当前的c#项目里新建一个名为Product的类，写一个ProductName的属性，在它前面加上 [MyDescription(Name ="商品名称")] 。至此我们就完成了自定义特性的创建。

创建完自定义特性后，我们通过反射获取ProductName属性上的MyDescription特性。
**5.3 获取特性——操作步骤：**

1. 回到main方法对应的类，运行以下代码，观察控制台输出：

```csharp
//获取属性上的自定义特性
public static void TestGetCustomAttribute01()
{
    var type01 = typeof(Product);
    //获取指定属性
    var propInfo = type01.GetProperty("ProductName");
    //获取自定义特性
    var attr = propInfo?.GetCustomAttribute(typeof(MyDescriptionAttribute)) as MyDescriptionAttribute;
    Console.WriteLine($"{nameof(Product)}的描述是:{attr?.Name}");
}
```

可以看到控制台输出：Product的描述是:商品名称，说明我们成功获取了Product上的ProductName属性的MyDescriptionAttribute自定义特性的值：Name。

**5.4 笔者补充：**

1. 特性本质还是一个类，即不论是框架自带的特性还是我们自定义的特性，都需要继承自Attribute类。
2. Attribute可以理解为是给编译器看的注释，即注释不能直接影响程序的运行，但是特性Attribute可以。
3. Attribute与一般的类不同的点就在于，我们在代码阶段我们就要确定我们写的Attribute中所含所有信息，列如字段属性的类型、值、方法的参数、返回值等等，因为在编译后Attribute就不能再被动态的修改了，就像泛型那样。
4. 总之特性可以在不破坏类型封装的前提下，添加额外的信息和行为。

由此，我们就成功创建了自定义特性，不难看出，特性是对对象的描述，就像给物品贴标签起到标识的作用，特性常常雨反射结合使用才能实现相应的功能（反射获取自定义特性->执行相应操作）。
