# C# Linq 查询（二）

## 一、let 关键字

`let` 关键字在LINQ查询中用于创建新的变量，这些变量可以在后续的查询子句中使用。它主要用于存储中间结果，以提高查询的可读性和效率。类似允许我们在查询时定义临时变量。

```csharp
/// <summary>
/// 查询年龄大于等于20岁的学生，显示学生Id，名字，总分（Chinese+Math+English+Physics）
/// </summary>
[Test]
public void TestLinq07()
{
    // 方法一：
    var stuInfoLst = from item in GetStudentInfos()
                     where item.Age >= 20
                     select new { item.Id, item.Name, Total = item.Chinese + item.Math + item.English + item.Physics }; // 需要定义属性接受表达式的指

    foreach (var stuInfo in stuInfoLst)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name},Total Score = {stuInfo.Total}");
    }
    Console.WriteLine("====================");

    // 方法二：
    var stuInfoLst2 = from item in GetStudentInfos()
                      let total = item.Chinese + item.Math + item.English + item.Physics
                      where item.Age >= 20
                      select new { item.Id, item.Name, Total = total }; // 需要定义属性接受表达式的指

    foreach (var stuInfo in stuInfoLst2)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name},Total Score = {stuInfo.Total}");
    }
    Console.WriteLine("====================");
}
```

运行结果如下：

在这个例子中，`let` 子句用于创建 `total` 变量，它存储每个学生总分（Chinese+Math+English+Physics）。然后在 `select` 子句中，我们将 total 赋值给匿名类的 Total 属性上。

## 二、多表联合查询（多from字句）

LINQ查询可以包含多个from字句，从多个数据源（例如两个列表）进行联合查询。

例如，假设我们有两个列表，一个包含学生信息，另一个包含小组信息，学生信息中的小组id与小组信息中的id一一对应。我们希望找到所有选修了特定课程的学生，并分别显示学生所在的小组。
显然，我们需要从学生信息和小组学习两个数据源中查数据。

学生信息：

```csharp
namespace TestLinq
{
    /// <summary>
    /// 学生信息
    /// </summary>
    internal class StuInfo
    {
        public int Id { get; set; }
        public string? Name { get; set; }
        public string? Sex { get; set; }
        public int Age { get; set; }
        public double Chinese { get; set; }
        public double Math { get; set; }
        public double English { get; set; }
        public double Physics { get; set; }
        public double Score { get; set; }
        public string? Grade { get; set; }
        public int GroupId { get; set; }
    }
}

```

小组信息：

```csharp
namespace TestLinq
{
    internal class ClassGroup
    {
        public int Id { get; set; }
        public string GroupName { get; set; }
    }
}

```

学生数据：

```csharp
public static List<StuInfo> GetStudentInfos()
{
    List<StuInfo> stuInfos = new List<StuInfo>()
    {
        new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A",GroupId = 1 },
        new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A",GroupId = 1 },
        new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" ,GroupId = 2},
        new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" ,GroupId = 3}
    };
    return stuInfos;
}
```

小组数据：

```csharp
public static List<ClassGroup> GetClassGroups()
{
    List<ClassGroup> classGroups = new List<ClassGroup>()
    {
        new ClassGroup{Id = 1,GroupName="组1"},
        new ClassGroup{Id = 2,GroupName="组2"},
        new ClassGroup{Id = 3,GroupName="组3"},
        new ClassGroup{Id = 4,GroupName="组4"},
    };
    return classGroups;
}
```

分别遍历小组信息和学生信息，假设两个数据源id相同，进行映射，最后执行查询操作。

```csharp
/// <summary>
/// 查询学生详情，显示学生名字和所在的组名（分类名称）
/// </summary>
[Test]
public void TestLinq08()
{
    var stuInfoLst = from p in GetStudentInfos()
                     from g in GetClassGroups()
                     where p.GroupId == g.Id
                     select new { p.Name, g.GroupName };

    foreach (var item in stuInfoLst)
    {
        Console.WriteLine($"Name = {item.Name}, GroupName = {item.GroupName}");
    }
}
```

运行结果如下：每个学生的名称和小组名称都一并显示了
