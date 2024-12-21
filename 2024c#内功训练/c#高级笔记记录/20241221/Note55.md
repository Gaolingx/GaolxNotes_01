# C# Linq 查询（四）

## 四、多条件查询

在LINQ（Language Integrated Query）查询表达式中，可以有多重 `where` 子句，我们可以通过多个 `where` 关键字实现 `&&` 运算符的效果，即返回同时满足所有的 where 字句中的条件的查询结果，你也可以通过逻辑运算符（如 `&&`，即逻辑与）来组合多个条件。

以下是一个示例，展示了如何在LINQ查询表达式中使用多个条件：

```csharp
var query = from item in collection
            where item.Property1 == someValue1 && item.Property2 == someValue2
            select item;
```

在这个例子中，`where` 子句包含了两个条件：`item.Property1 == someValue1` 和 `item.Property2 == someValue2`，它们通过 `&&` 运算符连接在一起。

或者：

```csharp
var query = from item in collection
            where item.Property1 == someValue1 
            where item.Property2 == someValue2
            select item;
```

如果你更喜欢方法语法，你也可以这样写：

```csharp
var query = collection.Where(item => item.Property1 == someValue1 && item.Property2 == someValue2);
```

这个方法同样使用了 `&&` 运算符来组合多个条件。

### 使用

假设我们现在要查询年龄大于19岁且位于第一小组的学生信息。LINQ查询表达式实现如下：

```csharp
/// <summary>
/// 查询年龄大于19岁且位于第一小组的学生信息
/// </summary>
[Test]
public void TestLinq12()
{
    var stuInfos = from p in GetStudentInfos2()
                   where p.Age > 19 
                   where p.GroupId == 1
                   // 等价于 where p.Age > 19 && p.GroupId == 1
                   select p;

    foreach (var stuInfo in stuInfos)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
            $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}, Group = {stuInfo.GroupId}");
    }
}
```

运行结果如下，这样即查询到了符合上述所有条件的学生信息。

</br>

## 五、自定义函数

在LINQ查询中，你可以使用自定义函数来增强查询的灵活性和可读性。在C#中，这通常涉及将自定义函数应用于查询的结果或在查询的投影部分中使用它。

### 使用

### 1. 在 Select 投影中使用自定义函数

假如我们现在要封装一个方法，用于查询小组的详细信息，显示小组ID，小组名称，小组人数，除了使用join组连接实现，也可以通过自定义函数实现。
已知小组ID，小组名称是可以直接从小组信息的集合中获取的，我们并不能直接获取小组中的学生人数，但是在学生信息的集合中是包含小组id的，作为分类的依据，我们只需获取学生信息中学生所在的小组id，即可查询这个分类中所有的学生信息，则实现如下。

```csharp
/// <summary>
/// 根据小组id，查询所在组的学生数量
/// </summary>
/// <param name="groupId"></param>
/// <returns></returns>
private static int GetStudentsCountByGroupId(int groupId)
{
    int count = GetStudentInfos2().Where(item => item.GroupId == groupId).Count();
    return count;
}

/// <summary>
/// 查询小组的详细信息，显示小组ID，小组名称，小组人数
/// </summary>
[Test]
public void TestLinq13()
{
    var groupItems = from classGroup in GetClassGroups2()
                     select new { classGroup.Id, classGroup.GroupName, Count = GetStudentsCountByGroupId(classGroup.Id) };

    foreach (var item in groupItems)
    {
        Console.WriteLine($"Group Id = {item.Id}, Group Name = {item.GroupName}, Student Count:{item.Count}");
    }
}
```

运行结果如下：

### 2. 在 Where 子句中使用自定义函数

根据 第一节 的条件，再追加一条，假设我们需要小组中的学生个数大于0时才返回所在组的学生数量的查询结果，
你可以在LINQ查询的 `Where` 子句中使用这个函数，则实现如下。

```csharp
/// <summary>
/// 查询小组的详细信息，显示小组ID，小组名称，小组人数，且只返回小组中的学生个数大于0的查询结果
/// </summary>
[Test]
public void TestLinq14()
{
    var groupItems = from classGroup in GetClassGroups2()
                     let count = GetStudentsCountByGroupId(classGroup.Id)
                     where count > 0
                     select new { classGroup.Id, classGroup.GroupName, Count = count };

    foreach (var item in groupItems)
    {
        Console.WriteLine($"Group Id = {item.Id}, Group Name = {item.GroupName}, Student Count:{item.Count}");
    }
}
```

运行结果如下：

### 3. 使用方法语法中的自定义函数

如果你更喜欢使用方法语法（而不是查询语法），你可以直接在 `Select` 或 `Where` 方法中使用自定义函数：

```csharp
var query = employees.Select(employee => new
{
    employee.Name,
    CalculatedSalary = CalculateSalary(employee)
});

foreach (var result in query)
{
    Console.WriteLine($"Name: {result.Name}, Calculated Salary: {result.CalculatedSalary}");
}

var highSalaryEmployees = employees.Where(employee => IsHighSalary(employee, highSalaryThreshold));

foreach (var employee in highSalaryEmployees)
{
    Console.WriteLine($"High Salary Employee: {employee.Name}");
}
```

</br>

### 注意事项

1. **性能**：在LINQ to SQL或EF Core等查询提供者中，如果自定义函数包含复杂的逻辑，可能会阻止查询被翻译成SQL。在这种情况下，自定义函数可能会在客户端执行，导致性能下降。

2. **纯函数**：确保你的自定义函数是纯函数（即没有副作用，相同的输入总是产生相同的输出），这样可以更容易地理解和调试你的LINQ查询。

3. **可重用性**：将逻辑封装在自定义函数中可以提高代码的可重用性和可维护性。

通过这些方法，你可以在LINQ查询中灵活地使用自定义函数，使你的代码更加简洁和可读。
