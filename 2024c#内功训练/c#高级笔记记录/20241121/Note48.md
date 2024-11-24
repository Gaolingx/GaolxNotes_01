# C#高级编程之——委托（二）匿名方法与Lambda表达式

## 背景

- 在 2.0 之前的 C# 版本中，声明委托的唯一方法是使用命名方法。C# 2.0 引入了匿名方法，而在 C# 3.0 及更高版本中，Lambda 表达式取代了匿名方法，作为编写内联代码的首选方式。这种方式在某些情况下显得不够灵活，因为它要求程序员必须显式地定义一个与委托签名相匹配的方法。
<br>

- 为了简化委托的声明和使用，C# 2.0引入了匿名方法。匿名方法允许程序员在声明委托时直接内联一个代码块，而无需单独定义一个方法。这种方式减少了代码量，提高了代码的可读性和维护性。
- 通过使用匿名方法，由于您不必创建单独的方法，因此减少了实例化委托所需的编码系统开销。

## 概念——匿名方法

匿名方法（Anonymous methods）提供了一种**传递代码块作为委托参数**的技术。匿名方法是没有名称只有主体的方法。在匿名方法中您不需要指定返回类型，它是从方法主体内的return语句推断的。匿名方法是通过使用delegate关键字创建委托实例来声明的

## 概念——Lambda表达式

lambda 表达式是一个可用于创建委托或表达式树类型的匿名函数。通过使用 lambda 表达式，可以可作为参数或返回编写本地函数，该函数调用的值。Lambda 表达式用于编写 LINQ 查询表达式特别有用。

特点：

- Lambda 表达式是一个匿名函数，用它可以高效简化代码，常用作委托，回调
- Lambda 表达式都使用运算符=>，所以当你见到这个符号，基本上就是一个Lambda表达式
- Lambda 运算符的左边是输入参数()，=>，右边是表达式或语句块
- Lambda 表达式，是可以访问到外部变量的
  
语法：() 表示方法的参数，如果只有一个参数，则 () 可以省略，=> 表示 goes to，{ } 是与委托签名相匹配的方法。

## 使用

案例一：

```csharp
private string SpeakLanguages(int type)
{
    return type switch
    {
        0 => "我在说中文",
        1 => "我在说英文",
        _ => "参数无效",
    };
}

/// <summary>
/// 匿名方法
/// </summary>
[Test]
public void TestDelegate5()
{
    //方法一:匿名方法
    DoSpeak speakDel = delegate { Console.WriteLine("你调用了第一个匿名方法"); };
    //方法二:Lambda表达式（本质还是匿名方法）
    speakDel += () => { Console.WriteLine("你调用了第二个匿名方法"); };
    speakDel?.Invoke();

    DoSpeak3 speak3 = delegate (int val) { return SpeakLanguages(val); };
    string? str = speak3?.Invoke(0);
    Console.WriteLine(str);

    DoSpeak3 speak4 = (int val) => { return SpeakLanguages(val); };
    string? str2 = speak4?.Invoke(1);
    Console.WriteLine(str2);
}
```

运行结果如下：

## 闭包

### C#闭包的概念

在C#中，闭包是指一个能够访问并操作其外部作用域（通常是包含它的方法或匿名函数外部的方法）中变量的方法或匿名函数（如lambda表达式或匿名方法）。闭包允许你将数据和操作数据的行为绑定在一起，形成一个不可分割的单元。

### 委托中的匿名函数捕获外部变量

C#中的委托可以指向匿名方法或lambda表达式。这些匿名函数或lambda表达式可以捕获并使用其定义时所在作用域中的变量，即使这些变量在匿名函数被调用时已经不再处于活动状态。

下面是一个简单的示例，展示了闭包和匿名函数如何捕获外部变量：

```csharp
using System;

class Program
{
    delegate int MathOperation(int x, int y); // 定义一个委托类型

    static void Main()
    {
        int a = 5; // 外部变量
        int b = 10; // 另一个外部变量

        // 使用lambda表达式创建一个委托实例，该lambda表达式捕获了外部变量a和b
        MathOperation operation = (x, y) => x * a + y * b;

        // 调用委托，传入参数3和4
        int result = operation(3, 4);

        // 输出结果，应该是5*3 + 10*4 = 15 + 40 = 55
        Console.WriteLine(result);
    }
}
```

在这个示例中：

1. 我们定义了一个名为`MathOperation`的委托类型，它接受两个整数参数并返回一个整数。
2. 在`Main`方法中，我们定义了两个外部变量`a`和`b`。
3. 我们使用lambda表达式创建了一个`MathOperation`委托的实例。这个lambda表达式捕获了外部变量`a`和`b`，并在其表达式中使用了它们。
4. 当我们调用这个委托并传入参数3和4时，lambda表达式中的`x`和`y`被替换为这些参数值，而`a`和`b`则保持为它们被捕获时的值（5和10）。
5. 最终，表达式`x * a + y * b`计算为`3 * 5 + 4 * 10 = 15 + 40 = 55`，并作为委托调用的结果返回。

这个例子展示了闭包的一个关键特性：匿名函数（如lambda表达式）可以捕获并保留其定义时作用域中的变量，即使这些变量在匿名函数被调用时已经不再处于活动状态（例如，它们可能已经被垃圾回收器回收，但由于闭包的存在，它们实际上仍然被引用着）。在这个例子中，`a`和`b`就是被捕获的外部变量。
