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
