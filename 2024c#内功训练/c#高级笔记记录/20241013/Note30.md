# C#高级编程之——泛型集合（二）Stack<T>

## 三、泛型集合——Stack<T> 栈

### 3.1 特点：先进后出，后进先出

### 3.2 使用：

`Stack<T>` 是 C# 中提供的一个后进先出（LIFO, Last In First Out）集合类，属于 `System.Collections.Generic` 命名空间。以下是一些常见的用法和示例：

### 1. 初始化栈

```csharp
using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        // 创建一个空的栈
        Stack<int> stack = new Stack<int>();

        // 创建一个包含初始值的栈
        Stack<string> stringStack = new Stack<string>(new string[] { "a", "b", "c" });
    }
}
```

### 2. 入栈（Push）

使用 `Push` 方法将元素添加到栈顶。

```csharp
stack.Push(1);
stack.Push(2);
stack.Push(3);
```

### 3. 出栈（Pop）

使用 `Pop` 方法移除并返回栈顶元素。如果栈为空，调用 `Pop` 会引发 `InvalidOperationException` 异常。

```csharp
int topElement = stack.Pop(); // 返回 3 并从栈中移除
```

### 4. 查看栈顶元素（Peek）

使用 `Peek` 方法查看栈顶元素但不移除它。如果栈为空，调用 `Peek` 会引发 `InvalidOperationException` 异常。

```csharp
int topElement = stack.Peek(); // 返回栈顶元素但不移除，此时为 2
```

### 5. 检查栈是否为空（Count 和 Contains）

使用 `Count` 属性检查栈中的元素数量，使用 `Contains` 方法检查栈中是否包含某个元素。

```csharp
bool isEmpty = stack.Count == 0; // 检查栈是否为空
bool containsTwo = stack.Contains(2); // 检查栈中是否包含元素 2
```

### 6. 清空栈（Clear）

使用 `Clear` 方法移除栈中的所有元素。

```csharp
stack.Clear(); // 清空栈
```

### 7. 遍历栈

由于 `Stack<T>` 是基于 `IEnumerable<T>` 的，可以使用 `foreach` 循环遍历栈中的元素。请注意，由于栈是后进先出的结构，遍历顺序是从栈顶到栈底。

```csharp
foreach (int item in stack)
{
    Console.WriteLine(item);
}
```

不过，更常见的需求是按从栈底到栈顶的顺序遍历，可以通过 `ToArray` 方法将栈转换为数组，再遍历数组。

```csharp
int[] stackArray = stack.ToArray();
Array.Reverse(stackArray); // 反转数组以恢复顺序
foreach (int item in stackArray)
{
    Console.WriteLine(item);
}
```

### 8. TryPop 方法

`TryPop`方法尝试从栈中移除并返回栈顶元素。如果栈不为空，它会将栈顶元素复制到提供的输出参数中，并从栈中移除该元素，然后返回`true`。如果栈为空，则不会执行任何操作，并返回`false`。

使用`TryPop`方法的优点在于，它允许你在不引发异常的情况下检查栈是否为空，并在不为空时安全地移除栈顶元素。

### 9. TryPeek 方法

`TryPeek`方法尝试查看栈顶元素而不移除它。如果栈不为空，它会将栈顶元素复制到提供的输出参数中，并返回`true`。如果栈为空，则不会执行任何操作，并返回`false`。

使用`TryPeek`方法的优点在于，它允许你在不引发异常的情况下检查栈是否为空，并在不为空时安全地查看栈顶元素。

### 3.3 使用案例

运行以下代码，观察输出结果：

```csharp
[Test]
public void TestStack1()
{
    Stack<int> ints =  new Stack<int>(4); //初始容量4，扩容速度2

    ints.Push(100); //stack添加一个元素，压栈
    ints.Push(101);
    ints.Push(102);

    foreach (var item in ints)
    {
        Console.WriteLine(item);
    }

    var peek = ints.Peek();
    Console.WriteLine(peek);

    var peek2 = ints.Peek();
    Console.WriteLine(peek2); //从stack中获取元素，返回最顶端的元素，但不移除

    var pop = ints.Pop();
    Console.WriteLine(pop); //出栈，移除一个元素并返回

    //var peek3 = ints.TryPeek(out _); //弃元
    int result = 0;
    var peek3 = ints.TryPeek(out result);
    Console.WriteLine(result); //result是被取出的元素的值

    var pop2 = ints.Pop();
    Console.WriteLine(pop2); //出栈，移除元素

    int result2 = 0;
    var pop3 = ints.TryPop(out result2);
    Console.WriteLine(pop3); //pop3返回的是操作是否成功
    Console.WriteLine(result2); //result是被移除的元素的值

    Console.WriteLine(ints.Count);
}
```
