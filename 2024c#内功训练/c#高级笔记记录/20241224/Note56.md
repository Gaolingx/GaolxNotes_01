# C# Linq 查询（五）

## 六、Linq 方法语法 Where

`Where` 方法是 LINQ 中用于筛选数据的最常用方法之一。它允许你根据指定的条件从集合中筛选出符合条件的元素。
省流：条件过滤

### 基本用法

`Where` 方法通常与 Lambda 表达式或委托一起使用，以定义筛选条件。

### Lambda 表达式

Lambda 表达式是 `Where` 方法中最常用的条件表示方式。Lambda 表达式的基本形式是 `(parameter) => expression` 或 `(parameters) => { statement; }`。

### 示例

```csharp
/// <summary>
/// 查询年龄大于18岁的所有学生信息
/// </summary>
[Test]
public void TestLinq15()
{
    var students = GetStudentInfos2();
    var data = students.Where(delegate (StuInfo val) { return GetStudentAge(val); }); // Func<TSource, bool>
    //var data2 = students.Where(item => item.Age > 19);

    foreach (var stuInfo in data)
    {
        Console.WriteLine($"Id = {stuInfo.Id}, Name = {stuInfo.Name}, Sex = {stuInfo.Sex}, Age = {stuInfo.Age}, Chinese = {stuInfo.Chinese}, " +
            $"Math = {stuInfo.Math}, English = {stuInfo.English}, Physics = {stuInfo.Physics}, Score = {stuInfo.Score}, Grade = {stuInfo.Grade}, Group = {stuInfo.GroupId}");
    }
}
```

运行结果如下：

### 总结

`Where` 方法是 LINQ 中非常基础且强大的工具，用于从集合中筛选出符合条件的元素。它支持 Lambda 表达式、方法组以及复合条件，并且可以与其他 LINQ 方法结合使用，以形成复杂的查询。

## 七、Linq 方法语法 Count

`Count` 方法是 LINQ 中常用的一个聚合方法，用于返回集合中元素的数量。

### 基本用法

`Count` 方法通常用于返回集合中满足特定条件的元素数量。如果没有指定条件，它将返回集合中的总元素数量。

以下是 `Count` 方法的一些关键点和使用示例：

#### 1. 不带条件的 Count

```csharp
int[] numbers = { 1, 2, 3, 4, 5 };
int count = numbers.Count();
Console.WriteLine(count); // 输出: 5
```

在这个例子中，`Count` 方法返回数组 `numbers` 中的总元素数量。

#### 2. 带条件的 Count

```csharp
int[] numbers = { 1, 2, 3, 4, 5 };
int evenCount = numbers.Count(n => n % 2 == 0);
Console.WriteLine(evenCount); // 输出: 2
```

在这个例子中，`Count` 方法使用了一个 lambda 表达式 `n => n % 2 == 0` 作为条件，返回数组中所有偶数元素的数量。（用于查询满足特定条件的元素数量）

#### 举例

```csharp
/// <summary>
/// 获取集合中所有学生数量
/// </summary>
[Test]
public void TestLinq16()
{
    var students = GetStudentInfos2();
    var studentCount = students.Count(); // 不带条件Count
    var studentCount2 = students.Count(item => item.Age > 18); // 带条件Count

    Console.WriteLine($"All Students Count:{studentCount}");
    Console.WriteLine($"Age Over 18 Students Count:{studentCount2}");
}
```

### 注意事项

1. **性能**：对于大型集合，`Count` 方法在每次调用时都会遍历集合来计算元素数量。如果你需要多次使用集合的大小，可以考虑将其存储在变量中。

2. **空集合**：如果集合为空，`Count` 方法将返回 0。

3. **线程安全**：对于多线程环境中使用的集合，`Count` 方法的线程安全性取决于集合本身。如果集合不是线程安全的，那么在多线程环境中调用 `Count` 可能会导致不可预测的结果。
