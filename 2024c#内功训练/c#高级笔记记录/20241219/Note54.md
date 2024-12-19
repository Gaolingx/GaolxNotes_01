# C# Linq 查询（三）

## 三、join 关键字

LINQ中，`join` 关键字用于在多个数据集合之间执行连接操作，类似于 SQL 中的 `JOIN` 语句。
`join` 关键字的主要用途是将两个或多个数据源基于一个或多个键进行匹配，并返回一个新的数据集合，其中包含匹配项的组合。

以下是一些常见的 `join` 用法示例：

### 1. 内连接（Inner Join）

内连接只返回在两个数据集合中都存在匹配项的元素。

```csharp
var query = from person in people
            join pet in pets on person.PersonID equals pet.OwnerID
            select new { PersonName = person.Name, PetName = pet.Name };
```

在这个示例中，`people` 和 `pets` 是两个数据集合，`person.PersonID` 和 `pet.OwnerID` 是用于匹配的键。结果集合包含所有在 `people` 和 `pets` 中都有匹配项的 `Person` 和 `Pet` 的组合。

### 2. 左外连接（Left Outer Join）

左外连接返回左数据集合中的所有元素，即使右数据集合中没有匹配的元素。如果右数据集合中没有匹配的元素，则结果集中的相应字段为 `null`。

在 LINQ 中，左外连接是通过 `DefaultIfEmpty` 方法来实现的。

```csharp
var query = from person in people
            join pet in pets on person.PersonID equals pet.OwnerID into gj
            from subpet in gj.DefaultIfEmpty()
            select new { PersonName = person.Name, PetName = subpet?.Name ?? "No Pet" };
```

在这个示例中，`into gj` 创建了一个分组连接（grouped join），然后通过 `from subpet in gj.DefaultIfEmpty()` 将每个分组展开为单个元素，其中 `subpet` 可以是 `null`（如果没有匹配的宠物）。

### 3. 交叉连接（Cross Join）

交叉连接返回两个数据集合中所有可能的元素组合。在 LINQ 中，这通常通过 `from` 子句的多重嵌套来实现，而不是使用 `join` 关键字。

```csharp
var query = from person in people
            from pet in pets
            select new { PersonName = person.Name, PetName = pet.Name };
```

注意：这种用法并不使用 `join` 关键字，但它实现了交叉连接的效果。

### 4. 多表连接（Multiple Joins）

你可以将多个 `join` 语句组合在一起，以连接多个数据集合。

```csharp
var query = from person in people
            join pet in pets on person.PersonID equals pet.OwnerID
            join address in addresses on person.AddressID equals address.AddressID
            select new { PersonName = person.Name, PetName = pet.Name, Address = address.Street };
```

在这个示例中，`people`、`pets` 和 `addresses` 是三个数据集合，通过两个 `join` 语句将它们连接在一起。

### 使用

开始之前，先修改下存储学生的名称和小组名称的集合

```csharp
public static List<StuInfo> GetStudentInfos2()
{
    List<StuInfo> stuInfos = new List<StuInfo>()
    {
        new StuInfo { Id = 1001, Name = "流萤", Sex = "女", Age = 20, Chinese = 100, Math = 120, English = 95, Physics = 70, Score = 500, Grade = "A",GroupId = 1 },
        new StuInfo { Id = 1002, Name = "符玄", Sex = "女", Age = 20, Chinese = 105, Math = 130, English = 100, Physics = 80, Score = 500, Grade = "A",GroupId = 1 },
        new StuInfo { Id = 1003, Name = "爱莉希雅", Sex = "女", Age = 18, Chinese = 110, Math = 90, English = 105, Physics = 65, Score = 500, Grade = "B" ,GroupId = 2},
        new StuInfo { Id = 1003, Name = "琪亚娜", Sex = "女", Age = 19, Chinese = 90, Math = 85, English = 100, Physics = 60, Score = 500, Grade = "B" ,GroupId = 3},
        new StuInfo { Id = 1003, Name = "派蒙", Sex = "女", Age = 9, Chinese = 80, Math = 95, English = 110, Physics = 60, Score = 500, Grade = "A" ,GroupId = 6} // 分类ID不存在
    };
    return stuInfos;
}

public static List<ClassGroup> GetClassGroups2()
{
    List<ClassGroup> classGroups = new List<ClassGroup>()
    {
        new ClassGroup{Id = 1,GroupName="组1"},
        new ClassGroup{Id = 2,GroupName="组2"},
        new ClassGroup{Id = 3,GroupName="组3"},
        new ClassGroup{Id = 4,GroupName="组4"}, // 不存在任何数据
        new ClassGroup{Id = 5,GroupName="组5"}, // 不存在任何数据
    };
    return classGroups;
}
```

根据第二节教程中，我们已经有学生信息和小组信息两个集合，即两个数据源，当我们要查询查询学生详情，显示学生名字和所在的组名（分类名称）时，两者之间的关联是学生信息中的小组id与小组信息中的id一一对应。

#### 1. 内连接

语法：..join..in..on..

需要从 `学生信息` 中查询 `学生名字` 字段，从 `小组信息` 中查询 `组名` 字段，LINQ查询表达式实现如下：

```csharp
/// <summary>
/// 查询学生详情，显示学生名字和所在的组名（通过join 内连接关联）
/// </summary>
[Test]
public void TestLinq09()
{
    var stuInfoLst = from p in GetStudentInfos2()
                     join g in GetClassGroups2()
                     on p.GroupId equals g.Id //关联数据源，不能用 ==，而是 equals关键字
                     select new { p.Name, g.GroupName };

    foreach (var item in stuInfoLst)
    {
        Console.WriteLine($"Name = {item.Name}, GroupName = {item.GroupName}");
    }
}
```

可以看到，学生的名称和小组名称都一并显示了，但是新增的那个学生信息由于小组id不存在，因为没有对应的分组所以没有显示。因此，**内连接只返回在两个数据集合中都存在匹配项的元素。**

#### 2. 组连接

语法：..join..in..on..into..

假设我们现在希望从小组信息中查询小组id，小组名称，以及这个小组下所有学生的数量，小组id，小组名称可以直接从小组信息的数据源中获取，但是小组下所有学生的数量需要从学生信息中获取学生所在的小组id才能获取到，LINQ查询表达式实现如下：

```csharp
/// <summary>
/// 从小组信息中查询小组id，小组名称，以及这个小组下所有学生的数量
/// </summary>
[Test]
public void TestLinq10()
{
    var groupItems = from g in GetClassGroups2()
                     join p in GetStudentInfos2()
                     on g.Id equals p.GroupId
                     into ps //将分类对象 g 下所有的 p 保存到 ps 中，即ps存储了分类g下所有的学生信息
                     select new { g.Id, g.GroupName, Count = ps.Count() };

    foreach (var item in groupItems)
    {
        Console.WriteLine($"Id = {item.Id}, GroupName = {item.GroupName}, Student Count:{item.Count}");
    }
}
```

观察运行结果我们发现，通过组连接，可以将联合的数据按小组id进行分组。因此，**左外连接返回左数据集合中的所有元素，即使右数据集合中没有匹配的元素。**

#### 3. 左外连接

特点：可以获取内连接的查询结果，并获取内连接没有匹配到的数据。左外连接=内连接的数据+未匹配的数据

假如我们现在既要查询学生信息，要求显示学生的id，学生名字，小组名字。

我们需要考虑如下两种情形：

1. 学生信息中存在未分组的学生的情况
2. 小组信息中存在有小组没有学生的情况（分类存在，但是分类下面没有任何学生对象）

面对上述情形，左外连接可以查询这些情况。LINQ查询表达式实现如下：

```csharp
/// <summary>
/// 既要查询学生信息，要求显示学生的id，学生名字，小组名字。
/// </summary>
[Test]
public void TestLinq11()
{
    // 情况一：有学生未分组（左表：Students）
    var stuInfos = from students in GetStudentInfos2()
                   join classGroup in GetClassGroups2()
                   on students.GroupId equals classGroup.Id
                   into cs
                   // 从分类组 cs 中获取分类信息，如果（右数据集合）没有匹配的元素则使用 默认值：class null，或者new一个分类对象，指定它的初始值
                   from c2 in cs.DefaultIfEmpty(new ClassGroup() { GroupName = "无" }) // 有分类：显示分类，无分类：显示无
                   select new { students.Id, students.Name, c2.GroupName };

    foreach (var item in stuInfos)
    {
        Console.WriteLine($"Id = {item.Id}, Student Name = {item.Name}, Group Name:{item.GroupName}");
    }
    Console.WriteLine("====================");

    // 情况二：有分组无学生（左表：ClassGroups）

    var groupItems = from classGroup in GetClassGroups2()
                     join students in GetStudentInfos2()
                     on classGroup.Id equals students.GroupId
                     into ps
                     from p2 in ps.DefaultIfEmpty(new StuInfo()) // 当分类没有对应的学生信息时需要做相应的null值处理
                     select new { p2.Id, p2.Name, classGroup.GroupName };

    foreach (var item in groupItems)
    {
        Console.WriteLine($"Id = {item.Id}, Student Name = {item.Name}, Group Name:{item.GroupName}");
    }
}
```

运行结果如下，因此，**左外连接返回左数据集合中的所有元素，如果右数据集合中没有匹配的元素，则结果集中的相应字段为 null。**
