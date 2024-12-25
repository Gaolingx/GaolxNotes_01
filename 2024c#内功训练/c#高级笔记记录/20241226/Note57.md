# C# Linq 查询（六）

## 八、Linq 方法语法 Select

在C#中，LINQ（Language Integrated Query）提供了一种简洁且强大的方式来查询和操作集合数据。LINQ查询可以使用查询语法（类似于SQL）或方法语法（基于扩展方法的链式调用）。`Select` 是 LINQ 方法语法中的一个重要方法，它用于从数据源中提取数据，并应用一个投影函数来转换每个元素。

### 基本用法

`Select` 方法的基本签名如下：

```csharp
public static IEnumerable<TResult> Select<TSource, TResult>(
    this IEnumerable<TSource> source,
    Func<TSource, TResult> selector
)
```

- `source`：数据源，实现了 `IEnumerable<TSource>` 接口的集合。
- `selector`：一个函数，它应用于 `source` 中的每个元素并返回转换后的结果。这个函数接受一个 `TSource` 类型的参数，并返回一个 `TResult` 类型的结果。

### 示例

假设我们有一个包含一些整数的列表，我们想要获取这些整数的平方：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        List<int> numbers = new List<int> { 1, 2, 3, 4, 5 };

        // 使用 Select 方法获取每个数字的平方
        IEnumerable<int> squares = numbers.Select(n => n * n);

        // 输出结果
        foreach (int square in squares)
        {
            Console.WriteLine(square);
        }
    }
}
```

在这个例子中，`numbers.Select(n => n * n)` 表示对 `numbers` 列表中的每个元素 `n` 应用一个投影函数 `n => n * n`，生成一个新的整数序列，其中每个元素都是原序列中对应元素的平方。

### 复杂示例

`Select` 方法也可以用于更复杂的对象。例如，假设我们有一个包含 `Person` 对象的列表，我们想要提取每个 `Person` 的名字和年龄：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
}

class Program
{
    static void Main()
    {
        List<Person> people = new List<Person>
        {
            new Person { Name = "Alice", Age = 30 },
            new Person { Name = "Bob", Age = 25 },
            new Person { Name = "Charlie", Age = 35 }
        };

        // 使用 Select 方法提取每个 Person 的名字和年龄
        IEnumerable<string> personInfo = people.Select(p => $"{p.Name}, Age: {p.Age}");

        // 输出结果
        foreach (string info in personInfo)
        {
            Console.WriteLine(info);
        }
    }
}
```

在这个例子中，`people.Select(p => $"{p.Name}, Age: {p.Age}")` 表示对 `people` 列表中的每个 `Person` 对象 `p` 应用一个投影函数，生成一个新的字符串序列，其中每个字符串都包含 `Person` 对象的名字和年龄。

### 使用

根据前面的例子，假设我们现在要我们要提取学生信息中的名字、性别、年龄 三个数据，要求使用Linq的Select方法

```csharp
/// <summary>
/// 提取学生信息中的名字、性别、年龄 三个数据（使用Select方法）
/// </summary>
[Test]
public void TestLinq17()
{
    var students = GetStudentInfos2();
    var studentData = students.Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

    foreach (var item in studentData)
    {
        Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
    }
}
```

运行结果如下：

### 总结

`Select` 方法是 LINQ 方法语法中的一个非常基础且常用的方法，它允许你对集合中的每个元素进行转换或投影，生成一个新的集合。通过 `Select` 方法，你可以轻松地从原始数据集中提取所需的信息，并将其转换成更适合后续操作的形式。

## 九、Linq 方法语法 OrderBy

当然可以！在C#的LINQ（Language Integrated Query）中，`OrderBy` 方法是方法语法中的一个重要成员，它用于对集合中的元素进行排序。`OrderBy` 方法根据指定的键对元素进行升序排序；如果你需要降序排序，可以使用 `OrderByDescending` 方法。

### 基本用法

`OrderBy` 方法的基本签名如下：

```csharp
public static IOrderedEnumerable<TSource> OrderBy<TSource, TKey>(
    this IEnumerable<TSource> source,
    Func<TSource, TKey> keySelector
)
```

- `source`：数据源，实现了 `IEnumerable<TSource>` 接口的集合。
- `keySelector`：一个函数，它应用于 `source` 中的每个元素并返回用于排序的键。这个函数接受一个 `TSource` 类型的参数，并返回一个 `TKey` 类型的结果。

`OrderBy` 方法返回一个 `IOrderedEnumerable<TSource>` 类型的对象，这个对象表示已排序的集合，并且支持进一步的排序操作（例如，然后按照另一个键进行排序）。

### 示例

假设我们有一个包含一些整数的列表，我们想要按照升序对这些整数进行排序：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        List<int> numbers = new List<int> { 5, 3, 8, 1, 2 };

        // 使用 OrderBy 方法对数字进行升序排序
        IEnumerable<int> sortedNumbers = numbers.OrderBy(n => n);

        // 输出结果
        foreach (int number in sortedNumbers)
        {
            Console.WriteLine(number);
        }
    }
}
```

在这个例子中，`numbers.OrderBy(n => n)` 表示对 `numbers` 列表中的元素按照它们自身的值进行升序排序。

### 复杂示例

`OrderBy` 方法也可以用于更复杂的对象。例如，假设我们有一个包含 `Person` 对象的列表，我们想要按照 `Person` 对象的年龄进行排序：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
}

class Program
{
    static void Main()
    {
        List<Person> people = new List<Person>
        {
            new Person { Name = "Alice", Age = 30 },
            new Person { Name = "Bob", Age = 25 },
            new Person { Name = "Charlie", Age = 35 }
        };

        // 使用 OrderBy 方法按照年龄对 Person 对象进行排序
        IEnumerable<Person> sortedPeople = people.OrderBy(p => p.Age);

        // 输出结果
        foreach (Person person in sortedPeople)
        {
            Console.WriteLine($"{person.Name}, Age: {person.Age}");
        }
    }
}
```

在这个例子中，`people.OrderBy(p => p.Age)` 表示对 `people` 列表中的 `Person` 对象按照它们的年龄进行升序排序。

### 然后排序（ThenBy 和 ThenByDescending）

当你想要按照多个键对集合进行排序时，可以在 `OrderBy` 或 `OrderByDescending` 之后使用 `ThenBy` 或 `ThenByDescending` 方法。例如，如果你想要先按照年龄排序，然后再按照名字排序，你可以这样做：

```csharp
IEnumerable<Person> sortedPeopleByName = people.OrderBy(p => p.Age).ThenBy(p => p.Name);
```

这将返回一个先按年龄升序排序，然后按名字升序排序的 `Person` 对象集合。

### 使用

根据前面的例子，假设我们现在要我们要提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从小到大进行排序

```csharp
/// <summary>
/// 提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从小到大进行排序
/// </summary>
[Test]
public void TestLinq18()
{
    var students = GetStudentInfos2();
    var studentData = students.OrderBy(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

    foreach (var item in studentData)
    {
        Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
    }
}
```

运行结果如下：

提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从大到小进行排序（降序排列）

```csharp
/// <summary>
/// 提取学生信息中的名字、性别、年龄 三个数据，并根据年龄大小从大到小进行排序（降序排列）
/// </summary>
[Test]
public void TestLinq19()
{
    var students = GetStudentInfos2();
    var studentData = students.OrderByDescending(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age }); // Func<TSource, TResult>

    foreach (var item in studentData)
    {
        Console.WriteLine($"Student Name:{item.Name}, Student Sex:{item.Sex}, Student Age:{item.Age}");
    }
}
```

运行结果如下：

### 总结

`OrderBy` 方法是 LINQ 方法语法中的一个非常有用的方法，它允许你对集合中的元素进行排序。通过 `OrderBy` 方法，你可以轻松地对原始数据集中的元素进行排序，以满足你的需求。


## 十、Linq 方法语法 First

当然可以！在C#的LINQ（Language Integrated Query）中，`First` 方法是方法语法中的一个重要成员，它用于从集合中检索第一个满足指定条件的元素。如果没有找到满足条件的元素，`First` 方法会抛出一个 `InvalidOperationException` 异常。如果你希望在找不到元素时返回默认值而不是抛出异常，可以使用 `FirstOrDefault` 方法。

### 基本用法

`First` 方法的基本签名如下：

```csharp
public static TSource First<TSource>(
    this IEnumerable<TSource> source
)
```

以及带有谓词（条件）的重载：

```csharp
public static TSource First<TSource>(
    this IEnumerable<TSource> source,
    Func<TSource, bool> predicate
)
```

- `source`：数据源，实现了 `IEnumerable<TSource>` 接口的集合。
- `predicate`（可选）：一个函数，用于定义要检索的元素的条件。这个函数接受一个 `TSource` 类型的参数，并返回一个布尔值，指示元素是否满足条件。

### 示例

假设我们有一个包含一些整数的列表，我们想要检索列表中的第一个元素：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        List<int> numbers = new List<int> { 1, 2, 3, 4, 5 };

        // 使用 First 方法检索列表中的第一个元素
        int firstNumber = numbers.First();

        // 输出结果
        Console.WriteLine(firstNumber); // 输出: 1
    }
}
```

在这个例子中，`numbers.First()` 返回列表 `numbers` 中的第一个元素，即 `1`。

### 带条件的示例

假设我们有一个包含 `Person` 对象的列表，我们想要检索列表中第一个年龄大于30的 `Person` 对象：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
}

class Program
{
    static void Main()
    {
        List<Person> people = new List<Person>
        {
            new Person { Name = "Alice", Age = 30 },
            new Person { Name = "Bob", Age = 35 },
            new Person { Name = "Charlie", Age = 25 }
        };

        // 使用 First 方法检索年龄大于30的第一个 Person 对象
        Person firstOlderPerson = people.First(p => p.Age > 30);

        // 输出结果
        Console.WriteLine($"{firstOlderPerson.Name}, Age: {firstOlderPerson.Age}"); // 输出: Bob, Age: 35
    }
}
```

在这个例子中，`people.First(p => p.Age > 30)` 返回列表中第一个年龄大于 `30` 的 `Person` 对象，即 `Bob`。

### 使用

根据前面的例子，

1. 查询学生信息中的名字、性别、年龄、所在小组 三个数据，并输出年龄最小的学生信息。
2. 查询学生信息中位于第一组的第一条记录的学生信息。

```csharp
/// <summary>
/// 1. 查询学生信息中的名字、性别、年龄 三个数据，并输出年龄最小的学生信息。
/// 2. 查询学生信息中位于第一组的第一条记录的学生信息。
/// </summary>
[Test]
public void TestLinq20()
{
    var students = GetStudentInfos2();
    var student = students.OrderBy(item => item.Age).Select(item => new { item.Name, item.Sex, item.Age, item.GroupId }).First();
    Console.WriteLine($"Student Name:{student.Name}, Student Sex:{student.Sex}, Student Age:{student.Age}, Student Group:{student.GroupId}");

    var student2 = students.First(item => item.GroupId == 1); // 只会查询满足条件的第一条记录（集合首个元素）
    Console.WriteLine($"Student Name:{student2.Name}, Student Sex:{student2.Sex}, Student Age:{student2.Age}, Student Group:{student2.GroupId}");
}
```

运行结果如下：

### 注意事项

- 如果集合为空，`First` 方法会抛出一个 `InvalidOperationException` 异常。如果你想要避免这种情况，可以使用 `FirstOrDefault` 方法，它在找不到元素时会返回类型的默认值（对于引用类型，默认值是 `null`）。
- 如果集合中没有元素满足指定的条件，`First` 方法同样会抛出一个 `InvalidOperationException` 异常。使用 `FirstOrDefault` 可以避免这个异常。

例如：

```csharp
[Test]
public void TestLinq21()
{
    var students = GetStudentInfos2();

    try
    {
        var student2 = students.First(item => item.GroupId == 10); // 如果没有满足条件的数据，则会抛出InvalidOperationException 异常
    }
    catch(Exception ex)
    {
        Console.WriteLine($"Error:{ex.Message}");
    }

    try
    {
        var student2 = students.FirstOrDefault(item => item.GroupId == 10); // 找不到元素时会返回类型的默认值，且不会抛出异常
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error:{ex.Message}");
    }
}
```

运行结果如下：

### 总结

`First` 方法是 LINQ 方法语法中的一个非常有用的方法，它允许你从集合中检索第一个满足指定条件的元素。通过 `First` 方法，你可以轻松地获取满足你需求的第一个元素，但需要注意处理可能的异常情况。