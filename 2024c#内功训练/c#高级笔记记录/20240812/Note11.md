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

三、自定义扩展方法

步骤：

1. 在你的解决方案的csharp工程下新建一个名为Utils的文件夹，创建一个静态类（此处为ExtendUtils.cs）
2. 创建一个静态方法，这个方法即是你的扩展方法
3. 方法中必须将你要扩展的类型通过参数传递进来
4. 你需要扩展的参数类型前面加this（重点）

```csharp
// 扩展方法的工具类,注意一定要写成static 静态类
namespace ConsoleApp4.Utils
{
    public static class ExtendUtils
    {
        /**
         * 作者：Gaolingx
         * 功能：将Object类型 提供一个转换为Int类型的扩展方法
         * 注意：
         * 1. 静态类下所有的方法都只能是static 方法
         * 2. 把你需要扩展的类型前面加this 
         * 操作步骤：
         * 1. 将该拓展类设置为static 类
         * 2. 创建一个static 方法
         * 3. 在需要扩展的类型前面加一个this 关键字
         */
        public static int ParseInt(this string str) //需要扩展的类型 + 参数的值
        {
            if (string.IsNullOrWhiteSpace(str))
            {
                return 0;
            }

            int result = 0;

            if (!int.TryParse(str, out result)) //TryParse返回值为ture则表示str可以被转换成int类型，返回输出参数out result的值
            {
                return 0;
            }

            return result;
        }
    }
}
```

完成上述拓展方法后，回到RunMain类中，引用命名空间，测试我们的自定义拓展方法：

```csharp
public static void TestExtendUtils01()
{
    string str = "12345";
    var num = str.ParseInt();
    Console.WriteLine($"input:{str},output:{num}");

    string str2 = "12345abc";
    var num2 = str2.ParseInt();
    Console.WriteLine($"input:{str2},output:{num2}");
}
```

运行结果如下，可以看到str输出值为12345，说明string类型通过我们自定义的拓展方法成功转化为int类型，str2返回值为0，说明转换失败，由于我们在ParseInt 方法中使用了int.TryParse...对异常进行了处理，因此没有因为转换失败而抛出异常，提高了代码的健壮性。
