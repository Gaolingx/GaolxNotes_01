# C# Linq 查询（七）

## 十一、Linq 方法语法 Max、Min、Sum、Average

简单介绍Linq查询中几个聚合操作的方法：Max、Min、Sum、Average、Count（前面已经介绍过），在C#的LINQ（Language Integrated Query）中，`Max`、`Min`、`Sum` 和 `Average` 是方法语法中用于聚合运算的几个重要方法。它们分别用于计算集合中的最大值、最小值、总和和平均值。这些方法通常用于数值类型的集合，但也可以用于实现了自定义比较逻辑的集合。

### Max 方法

`Max` 方法用于返回集合中的最大值。它有一个重载版本，可以接受一个选择器函数，用于从集合中的每个元素中提取要比较的值。

```csharp
int maxValue = numbers.Max(); // 对于整数集合
double maxSalary = employees.Max(e => e.Salary); // 对于对象集合，提取 Salary 属性
```

### Min 方法

`Min` 方法与 `Max` 方法类似，但它返回集合中的最小值。

```csharp
int minValue = numbers.Min(); // 对于整数集合
DateTime minDate = events.Min(e => e.EventDate); // 对于对象集合，提取 EventDate 属性
```

### Sum 方法

`Sum` 方法用于计算集合中所有元素的总和。它同样有一个重载版本，可以接受一个选择器函数。

```csharp
int total = numbers.Sum(); // 对于整数集合
decimal totalSales = orders.Sum(o => o.TotalAmount); // 对于对象集合，提取 TotalAmount 属性
```

### Average 方法

`Average` 方法用于计算集合中所有元素的平均值。和前面的方法一样，它也有一个重载版本，可以接受一个选择器函数。

```csharp
double average = numbers.Average(); // 对于整数集合（将隐式转换为 double）
decimal avgScore = students.Average(s => s.Score); // 对于对象集合，提取 Score 属性
```

### 注意事项

- 当使用这些方法时，集合中的元素类型必须支持所需的聚合运算（例如，整数类型支持 `Max`、`Min` 和 `Sum`，而浮点数和十进制数类型还支持 `Average`）。
- 如果集合为空，这些方法通常会抛出一个 `InvalidOperationException` 异常。为了避免这种情况，你可以在使用这些方法之前检查集合是否为空，或者使用 `DefaultIfEmpty` 方法为集合提供一个默认值。
- 对于 `Average` 方法，如果集合中的元素类型不是数值类型，但实现了自定义的数值运算（例如，通过重载运算符），则仍然可以使用 `Average` 方法进行计算。

### 示例

以下是一个综合示例，展示了如何使用 `Max`、`Min`、`Sum` 和 `Average` 方法：

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        List<int> scores = new List<int> { 85, 92, 78, 90, 88 };

        int maxScore = scores.Max();
        int minScore = scores.Min();
        int totalScore = scores.Sum();
        double averageScore = scores.Average();

        Console.WriteLine($"Max Score: {maxScore}");
        Console.WriteLine($"Min Score: {minScore}");
        Console.WriteLine($"Total Score: {totalScore}");
        Console.WriteLine($"Average Score: {averageScore:F2}"); // 使用格式字符串来限制小数位数
    }
}
```

在这个例子中，我们有一个包含学生分数的整数列表，并使用 `Max`、`Min`、`Sum` 和 `Average` 方法来计算最高分、最低分、总分和平均分。

### 使用

案例：

1. 从学生信息的集合中获取年龄最大的学生的年龄
2. 从学生信息的集合中获取年龄最小的学生的年龄
3. 获取学生信息集合中所有学生语文科目的总分
4. 获取学生信息集合中所有学生物理科目的平均分

```csharp
[Test]
public void TestLinq22()
{
    var students = GetStudentInfos2();

    var maxAgeStudent = students.Max(item => item.Age); // 比较依据：Age
    // 从学生信息的集合中获取年龄最大的学生的年龄
    Console.WriteLine($"Max Student Age:{maxAgeStudent}");

    var minAgeStudent = students.Min(item => item.Age);
    // 从学生信息的集合中获取年龄最小的学生的年龄
    Console.WriteLine($"Min Student Age:{minAgeStudent}");

    var totalChineseScore = students.Sum(item => item.Chinese);
    // 获取学生信息集合中所有学生语文科目的总分
    Console.WriteLine($"Student Chinese Total Score:{totalChineseScore}");

    var avgPhysicsScore = students.Average(item => item.Physics);
    // 获取学生信息集合中所有学生物理科目的平均分
    Console.WriteLine($"Student Physics Average Score:{avgPhysicsScore}");
}
```

运行结果如下：
