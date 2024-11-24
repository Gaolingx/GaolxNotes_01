# C#高级编程之——委托（二）内置委托

## Action<T> 委托

- 定义
  - 命名空间:System
  - 程序集:netstandard.dll
  - 概述:封装一个方法，该方法只有一个参数并且不返回值。

```csharp
public delegate void Action<in T>(T obj);
```

- 类型参数
  - `T`
   此委托封装的方法的参数类型。
   这是逆变类型参数。 即，可以使用指定的类型，也可以使用派生程度较低的任何类型。 有关协变和逆变的详细信息，请参阅泛型中的协变和逆变。

- 参数
  - `obj` T
   此委托封装的方法的参数。

Action委托用于封装**没有返回值**（void）的方法。
1、Action 声明无参数委托
2、Action<T> 声明有一个参数委托
3、Action<T1,T2> 声明有2个参数委托
4、Action<T1,T2,T3> 声明有3个参数委托
5、Action<T1,..> 委托输入参数个数最多16个。

- 特点
  - 它有16个重载方法
  - 它有16个输入参数
  - Action的返回值类型是void

示例：

```csharp
/// <summary>
/// 内置委托——Action
/// </summary>
[Test]
public void TestAction()
{
    // 声明委托（0个参数）
    Action action = () => { Console.WriteLine("这是一个无参无返回值的内置委托"); };
    // 调用委托（方式一）
    action?.Invoke();
    // 调用委托（方式二）
    if (action != null)
    {
        action();
    }

    Console.WriteLine("=============");
    // 声明委托（一个参数）
    Action<int> action1 = (int i) =>
    {
        Console.WriteLine($"这是一个带一个参数无返回值的内置委托,value:{i}");
    };
    // 声明委托（多个参数）
    Action<int, string> action2 = (int i, string j) =>
    {
        Console.WriteLine($"这是一个带一个参数无返回值的内置委托,value:{i},{j}");
    };
    Action<int, int, string> action3 = (int i, int j, string k) =>
    {
        Console.WriteLine($"这是一个带一个参数无返回值的内置委托,i+j={i + j},{k}");
    };
    action1?.Invoke(1);
    action2?.Invoke(20, "爱莉小跟班");
    action3?.Invoke(30, 40, "流萤");

}
```

运行结果如下：

## Func<T,TResult> 委托

- 定义
  - 命名空间:System
  - 程序集:netstandard.dll
  - 概述:封装一个方法，该方法具有一个参数，且返回由 `TResult` 参数指定的类型的值。

```csharp
public delegate TResult Func<in T,out TResult>(T arg);
```

- 类型参数
  - `T`
   此委托封装的方法的参数类型。
   这是逆变类型参数。 即，可以使用指定的类型，也可以使用派生程度较低的任何类型。 有关协变和逆变的详细信息，请参阅泛型中的协变和逆变。
  - `TResult`
   此委托封装的方法的返回值类型。这是协变类型参数。 即，可以使用指定的类型，也可以使用派生程度较高的任何类型。 有关协变和逆变的详细信息，请参阅泛型中的协变和逆变。

- 参数
  - `arg` T
   此委托封装的方法的参数。

- 返回值
  - `TResult`
   此委托封装的方法的返回值。
