# C#高级编程之——扩展知识（一）

## 四、扩展知识——扩展方法

在上一个章节中我们学习了特性的概念以及如何获取类的自定义特性，由此我们对于反射的基础学习算是告一段落了，今天我们将开始学习C#的拓展方法、

一、 扩展方法

1. 概念：
   - 扩展方法使你能够向现有类型“添加”方法，而无需创建新的派生类型、重新编译或以其他方式修改原始类型。 扩展方法是一种特殊的静态方法，但可以像扩展类型上的实例方法一样进行调用。 对于用 C# 和 Visual Basic 编写的客户端代码，调用扩展方法与调用在类型中实际定义的方法之间没有明显的差异。
   - 最常见的扩展方法是 LINQ 标准查询运算符，它将查询功能添加到现有的 System.Collections.IEnumerable 和 System.Collections.Generic.IEnumerable<T> 类型。 若要使用标准查询运算符，请先使用 using System.Linq 指令将它们置于范围中。 然后，任何实现了IEnumerable<T> 的类型看起来都具有 GroupBy、OrderBy、Average 等实例方法。 在 IEnumerable<T> 类型的实例（如 List<T> 或 Array）后键入“dot”时，可以在 IntelliSense 语句完成中看到这些附加方法。
   - 省流：起到工具类的作用，对已有类型进行拓展。
2. 应用场景：
   - 一些原有方法没有提供但是需要经常用到的。
   - System.Linq命名空间中的许多方法都是通过扩展方法实现的,开发者可以利用这些扩展方法来简化数据查询和处理过程。
   - 当使用第三方库或框架时，如果其提供的类不满足特定需求，可以通过扩展方法来为这些类添加新功能，而无需修改第三方库的源代码。

二、OrderBy 示例

下面的示例演示如何对一个整数数组调用标准查询运算符 OrderBy 方法。 括号里面的表达式是一个 lambda 表达式。 很多标准查询运算符采用 Lambda 表达式作为参数，但这不是扩展方法的必要条件。 有关详细信息，请参阅 Lambda 表达式。

```csharp
class ExtensionMethods2
{
​
    static void Main()
    {
        int[] ints = { 10, 45, 15, 39, 21, 26 };
        var result = ints.OrderBy(g => g);
        foreach (var i in result)
        {
            System.Console.Write(i + " ");
        }
    }
}
//Output: 10 15 21 26 39 45
```

将鼠标放在OrderBy方法上，可以发现它并不是int[]的方法，而是由System.Linq.Enumerable拓展出来的方法，该方法允许开发者根据指定的属性或表达式对集合中的元素进行排序。
