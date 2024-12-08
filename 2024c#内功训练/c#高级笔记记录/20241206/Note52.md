# C# Linq 查询

## 一、LINQ 简介

- LINQ（Language Integrated Query，语言集成查询）是一系列直接将查询功能集成到 C# 语言的技术统称。
- 数据查询历来都表示为简单的字符串，没有编译时类型检查或 IntelliSense 支持。 此外，需要针对每种类型的数据源了解不同的查询语言：SQL 数据库、XML 文档、各种 Web 服务等。
- 借助 LINQ，查询成为了最高级的语言构造，就像类、方法和事件一样。它允许你在代码中以一种类似 SQL 的方式查询数据，极大地简化了数据检索、筛选和操作的过程。

## 二、LINQ 的主要类型

LINQ 主要有以下几种类型：

- LINQ to Objects：用于查询任何实现了 IEnumerable<T> 或 IQueryable<T> 接口的对象集合。
- LINQ to XML：用于查询和操作 XML 文档。
- LINQ to SQL：用于查询 SQL Server 数据库。现在微软推荐使用 Entity Framework 作为替代方案。
- LINQ to DataSet：用于查询 DataSet 对象。
- LINQ to Entities：用于查询实体框架中的实体集合。

## 三、Linq 的基本语法

### 查询表达式语法

查询表达式语法是一种类似于 SQL 的声明性查询语法，它使用关键字如 `from`、`where`、`select`、`join`、`group by`、`order by` 等来构建查询。以下是一些基本示例：

1. **选择数据**：

```csharp
var query = from item in collection select item;
```

这行代码表示从 `collection` 集合中选择所有元素。

2. **筛选数据**：

```csharp
var query = from item in collection where condition select item;
```

这里，`condition` 是一个布尔表达式，用于筛选满足条件的元素。

3. **排序数据**：

```csharp
var query = from item in collection orderby item.SomeProperty select item;
```

这行代码表示按 `SomeProperty` 属性对集合进行排序。

4. **分组数据**：

```csharp
var query = from item in collection group item by item.SomeProperty into g select new { Key = g.Key, Items = g };
```

这里，`group item by item.SomeProperty` 表示按 `SomeProperty` 属性对集合进行分组，`into g` 将分组结果放入临时变量 `g` 中，然后选择一个包含键和项的新对象。

5. **连接数据**：

```csharp
var query = from item1 in collection1
            join item2 in collection2 on item1.SomeProperty equals item2.SomeProperty
            select new { item1, item2 };
```

这行代码表示根据 `SomeProperty` 属性连接两个集合 `collection1` 和 `collection2`，并选择一个包含两个元素的新对象。

总结：

- 查询表达式必须以 `from` 子句开头。它指定数据源以及范围变量。
- 查询表达式必须以 `select` 子句或 `group` 子句结尾。
- `where` 关键字后跟布尔表达式。用于对数据源进行条件过滤。
- 对查询结果进行排序，可以使用 `orderby` 关键字，默认为升序 ascending ，降序用 descending。
- 对元素分组使用 `group by`，如果需要对分组后的元素进行操作，应使用 `into` 关键字重新定义变量。
- `select` 字后句可以接自定义类型的数据。
- 查询表达式可能会包含多个from子句。

### 方法语法

方法语法是使用 LINQ 扩展方法来构建查询的另一种方式。这些扩展方法定义在 `System.Linq` 命名空间中，并可以在任何实现了 `IEnumerable<T>` 或 `IQueryable<T>` 接口的集合上使用。以下是一些基本示例：

1. **选择数据**：

```csharp
var query = collection.Select(item => item);
```

这行代码表示从 `collection` 集合中选择所有元素。

2. **筛选数据**：

```csharp
var query = collection.Where(item => condition);
```

这里，`condition` 是一个布尔表达式，用于筛选满足条件的元素。

3. **排序数据**：

```csharp
var query = collection.OrderBy(item => item.SomeProperty);
```

这行代码表示按 `SomeProperty` 属性对集合进行排序。

4. **分组数据**：

```csharp
var query = collection.GroupBy(item => item.SomeProperty).Select(g => new { Key = g.Key, Items = g });
```

这里，`GroupBy` 方法按 `SomeProperty` 属性对集合进行分组，然后选择一个包含键和项的新对象。

5. **连接数据**：

虽然 LINQ 没有直接的 `Join` 扩展方法对应 SQL 中的 JOIN 操作，但你可以使用 `Zip` 方法（仅在两个集合长度相同的情况下）或自定义的扩展方法来实现类似的功能。不过，更常见的是使用查询表达式语法中的 `join` 关键字来进行连接操作。

## 四、优点

## 五、使用

开始之前，先准备一个类用于描述学生信息。

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
    }
}
```

1. 查询所有学生信息

```csharp
using NUnit.Framework;

namespace TestLinq
{
    internal class StudyLinq
    {
        #region 测试数据
        public static List<StuInfo> GetStudentInfos()
        {
            List<StuInfo> stuInfos = new List<StuInfo>()
            {
                new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A" },
                new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A" },
                new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" },
                new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" }
            };
            return stuInfos;
        }
        #endregion

        /// <summary>
        /// 查询所有学生信息
        /// </summary>
        [Test]
        public void TestLinq01()
        {
            var stuInfoLst = GetStudentInfos(); // 数据源

            // 语法：var 查询结果 = from 字段 in 数据源 select 字段
            var stuInfos = from prop in stuInfoLst select prop;

            foreach (var stuInfo in stuInfos)
            {
                Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                    $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
            }
        }
    }
}
```

运行结果如下：

2. 查询等级为A的学生信息(Grade = "A")

```csharp
/// <summary>
/// 查询等级为A的学生信息(Grade = "A")
/// </summary>
[Test]
public void TestLinq02()
{
    var stuInfoLst = GetStudentInfos(); // 数据源

    // 语法：var 查询结果 = from 字段 in 数据源 where 条件 select 字段
    var stuInfos = from prop in stuInfoLst
                   where prop.Grade == "A" // where接 bool表达式
                   select prop;

    foreach (var stuInfo in stuInfos)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
            $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
    }
}
```

运行结果如下：

3. 查询所有学生信息，并按照id降序排列

```csharp
/// <summary>
/// 查询所有学生信息，并按照id降序排列
/// </summary>
[Test]
public void TestLinq03()
{
    var stuInfoLst = GetStudentInfos();

    // 语法：var 查询结果 = from 字段 in 数据源 select 字段
    var stuInfos = from prop in stuInfoLst
                   orderby prop.Id descending //descending 降序 ascending升序（默认）
                   select prop;

    foreach (var stuInfo in stuInfos)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
            $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
    }
}
```

运行结果如下：

4. 查询所有学生信息，按照 Grade 进行分组

```csharp
/// <summary>
/// 查询所有学生信息，按照 Grade 进行分组
/// </summary>
[Test]
public void TestLinq04()
{
    var stuInfoLst = GetStudentInfos();

    // 语法：var 查询结果 = from 字段 in 数据源 group 字段 by 分组条件
    var stuInfos = from prop in stuInfoLst
                   group prop by prop.Grade;

    foreach (var groupItem in stuInfos)
    {
        Console.WriteLine(groupItem.Key); //分组依据

        foreach (var stuInfo in groupItem) //遍历groupItem，输出其中的每一个元素
        {
            Console.WriteLine($"   Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
        }
    }
}
```

运行结果如下：

5. 查询所有学生信息，按照 Age 进行分组，并对每组元素的数量进行升序排列

```csharp
/// <summary>
/// 查询所有学生信息，按照 Age 进行分组，并对每组元素的数量进行升序排列
/// </summary>
[Test]
public void TestLinq05()
{
    var stuInfoLst = GetStudentInfos();

    // 语法：var 查询结果 = from 字段 in 数据源 group 字段 by 分组条件
    var stuInfos = from prop in stuInfoLst
                   group prop by prop.Age
                   into groupdata //group by分组后，如果要对分组后的每组元素进行操作，需要into关键字重新赋值，g指每组的元素
                   orderby groupdata.Count() ascending
                   select groupdata;

    foreach (var groupItem in stuInfos)
    {
        Console.WriteLine(groupItem.Key); //分组依据

        foreach (var stuInfo in groupItem) //遍历groupItem，输出其中的每一个元素
        {
            Console.WriteLine($"   Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
                $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}");
        }
    }
}
```

运行结果如下：

6. 只查询学生的Id，名字和年龄

```csharp
/// <summary>
/// 只查询学生的Id，名字和年龄
/// </summary>
[Test]
public void TestLinq06()
{
    var stuInfoLst = GetStudentInfos();

    var stuLst = from prop in stuInfoLst
                 select new { prop.Id, prop.Name, prop.Age, Remark = "备注" }; //匿名类

    foreach (var stuInfo in stuLst)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Age = {stuInfo.Age}, Remark = {stuInfo.Remark}");
    }
}
```

运行结果如下：

## 总结

通过理解和使用 LINQ 的基本语法，你可以更高效地查询和操作数据，提升代码的可读性和维护性。