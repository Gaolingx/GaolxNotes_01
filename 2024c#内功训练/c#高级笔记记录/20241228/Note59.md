# C# Linq 查询（八）

## 十一、Linq 方法语法 Concat、Union、Intersect、Except

简单介绍Linq查询中几个集合操作的方法：Concat、Union、Intersect、Except，在C#中，LINQ（Language Integrated Query）提供了一种简洁、强大的方式来查询和操作数据集合。在LINQ查询中，方法语法使用一系列扩展方法，这些方法可以链式调用以构建复杂的查询。

以下是对 `Concat`、`Union`、`Intersect` 和 `Except` 这四种方法的介绍：

### 1. Concat

`Concat` 方法用于将两个序列连接成一个序列。它不会去除重复元素，只是简单地将第二个序列追加到第一个序列的末尾。

**用法示例**：

```csharp
var first = new int[] { 1, 2, 3 };
var second = new int[] { 4, 5, 6 };

var result = first.Concat(second);

foreach (var item in result)
{
    Console.WriteLine(item);
}
// 输出: 1 2 3 4 5 6
```

### 2. Union

`Union` 方法用于返回两个序列的并集。它会去除重复的元素，并返回唯一的元素集合。

**用法示例**：

```csharp
var first = new int[] { 1, 2, 3 };
var second = new int[] { 2, 3, 4 };

var result = first.Union(second);

foreach (var item in result)
{
    Console.WriteLine(item);
}
// 输出: 1 2 3 4
```

### 3. Intersect

`Intersect` 方法用于返回两个序列的交集。它只会返回在两个序列中都存在的元素，并去除重复的元素。

**用法示例**：

```csharp
var first = new int[] { 1, 2, 3 };
var second = new int[] { 2, 3, 4 };

var result = first.Intersect(second);

foreach (var item in result)
{
    Console.WriteLine(item);
}
// 输出: 2 3
```

### 4. Except

`Except` 方法用于返回两个序列的差集。它返回在第一个序列中存在但在第二个序列中不存在的元素，并去除重复的元素。

**用法示例**：

```csharp
var first = new int[] { 1, 2, 3 };
var second = new int[] { 2, 3, 4 };

var result = first.Except(second);

foreach (var item in result)
{
    Console.WriteLine(item);
}
// 输出: 1
```

### 使用

案例：

1. Concat:连接两个序列（有重复元素）

```csharp
/// <summary>
/// Concat:连接两个序列（有重复元素）
/// </summary>
[Test]
public void TestLinq23()
{
    int[] numArr1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int[] numArr2 = { 1, 3, 5, 7, 9, 11 };

    var numArr3 = numArr1.Concat(numArr2);

    Console.WriteLine("====== Concat ======");
    foreach (int num in numArr3)
    {
        Console.WriteLine(num);
    }
}
```

运行结果如下：

2. Union:求两个集合并集（无重复元素）

```csharp
/// <summary>
/// Union:求两个集合并集（无重复元素）
/// </summary>
[Test]
public void TestLinq24()
{
    int[] numArr1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int[] numArr2 = { 1, 3, 5, 7, 9, 11 };

    var numArr3 = numArr1.Union(numArr2);

    Console.WriteLine("====== Union ======");
    foreach (int num in numArr3)
    {
        Console.WriteLine(num);
    }
}
```

运行结果如下：

3. Intersect:求两个集合交集（提取集合中相同元素）

```csharp
/// <summary>
/// Intersect:求两个集合交集（提取集合中相同元素）
/// </summary>
[Test]
public void TestLinq25()
{
    int[] numArr1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int[] numArr2 = { 1, 3, 5, 7, 9, 11 };

    var numArr3 = numArr1.Intersect(numArr2);

    Console.WriteLine("====== Intersect ======");
    foreach (int num in numArr3)
    {
        Console.WriteLine(num);
    }
}
```

运行结果如下：

4. Except:求两个集合差集（从某集合中删除与另一个集合中相同的项）

```csharp
/// <summary>
/// Except:求两个集合差集（从某集合中删除与另一个集合中相同的项）
/// </summary>
[Test]
public void TestLinq26()
{
    int[] numArr1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int[] numArr2 = { 1, 3, 5, 7, 9, 11 };

    var numArr3 = numArr1.Except(numArr2);

    Console.WriteLine("====== Except ======");
    foreach (int num in numArr3)
    {
        Console.WriteLine(num);
    }
}
```

运行结果如下：
