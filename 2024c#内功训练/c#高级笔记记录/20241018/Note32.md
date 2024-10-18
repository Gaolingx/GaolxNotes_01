# C#高级编程之——泛型集合（五）SortedList<TKey,TValue>

## 五、泛型集合——SortedList<TKey,TValue> 类

### 5.1 简介

- 命名空间: System.Collections.Generic
- 程序集: System.Collections.dll
- 声明：表示基于相关的 IComparer 实现**按键**进行排序的键/值对的集合。
- 泛型参数：TKey 集合中的键的类型。TValue 集合中值的类型。这两组成一个键值对
- 主要作用：需要对元素排序的情况下

### 5.2 特点

- 键排序：SortedList 使用键的默认比较器（或提供的自定义比较器）对元素进行排序，因此元素总是按照键的升序排列。
- 键值对存储：它像字典一样存储键值对，但不同的是，它维护了元素的排序顺序。
- 索引访问：除了通过键进行查找外，还可以通过索引（从零开始的整数）访问元素。

### 5.3 简单使用

```csharp
[Test]
public void TestSortedList01()
{
    // Key(string):科目,Value(int):成绩
    SortedList<string, int> sList = new SortedList<string, int>();

    // 1. 键值对赋值（索引下标）
    sList["语文"] = 90;
    sList["数学"] = 120;

    // 2. Add方法赋值
    sList.Add("英语", 110); //注意类型匹配

    // 3. 遍历集合
    // SortedList按照键的顺序排序
    foreach (var item in sList)
    {
        Console.WriteLine($"科目名字:{item.Key},成绩:{item.Value}");
    }
}
```

运行结果如下：

### 5.4 进阶使用

```csharp
[Test]
public void TestSortedList02()
{
    // 1. 创建一个键值对都是string 类型的集合
    SortedList<string, string> openWith =
        new SortedList<string, string>();

    // 2. 初始化一些没有重复键的元素，但对应的值，有些元素是重复的
    openWith["语文"] = "120";
    openWith["数学"] = "120";
    openWith.Add("英语", "110");
    openWith.Add("物理", "75");

    // 3. 如果添加一个已经存在的键值对，则会抛出异常（Key不能重复）
    try
    {
        openWith.Add("物理", "80");
    }
    catch (ArgumentException)
    {
        Console.WriteLine("An element with Key = \"物理\" already exists.");
    }

    // 4. 元素的键可作为集合的索引来访问元素（根据Key获取值）
    Console.WriteLine("For key = \"语文\", value = {0}.",
        openWith["语文"]);

    // 5. 通过键索引，可修改其所关联的值
    openWith["数学"] = "135";
    Console.WriteLine("For key = \"数学\", value = {0}.",
        openWith["数学"]);

    // 6. 如果键不存在，则会新增一个键值对数据
    openWith["化学"] = "75";

    // 7. 如果请求的键不存在，则会抛出异常
    try
    {
        Console.WriteLine("For key = \"地理\", value = {0}.",
            openWith["地理"]);
    }
    catch (KeyNotFoundException)
    {
        Console.WriteLine("Key = \"地理\" is not found.");
    }

    // 8. 当一个程序经常要尝试的键，结果却不是  在列表中，TryGetValue可以是一个更有效的  
    // 获取值的方法。  （返回值类型：bool）
    string value = "";
    if (openWith.TryGetValue("地理", out value))
    {
        Console.WriteLine("For key = \"地理\", value = {0}.", value);
    }
    else
    {
        Console.WriteLine("Key = \"地理\" is not found.");
    }

    // 9. 判断是否包含键
    if (!openWith.ContainsKey("地理"))
    {
        openWith.Add("地理", "90");
        Console.WriteLine("Value added for key = \"地理\": {0}",
            openWith["地理"]);
    }

    // 10. 遍历循环，元素被检索为KeyValuePair对象
    Console.WriteLine();
    foreach (KeyValuePair<string, string> kvp in openWith)
    {
        Console.WriteLine("Key = {0}, Value = {1}",
            kvp.Key, kvp.Value);
    }

    // 11. 获取集合中的Values 列表
    IList<string> ilistValues = openWith.Values;

    // 打印出所有的值列表
    Console.WriteLine();
    foreach (string s in ilistValues)
    {
        Console.WriteLine("Value = {0}", s);
    }

    // 通过索引获取值
    Console.WriteLine("\nIndexed retrieval using the Values " +
        "property: Values[2] = {0}", openWith.Values[2]);

    // 获取所有的Key
    IList<string> ilistKeys = openWith.Keys;

    // 12. 打印出所有的键列表
    Console.WriteLine("=========================");
    foreach (string s in ilistKeys)
    {
        Console.WriteLine("Key = {0}", s);
    }

    // 13. 通过索引获取Key
    Console.WriteLine("\nIndexed retrieval using the Keys " +
        "property: Keys[0] = {0}", openWith.Keys[0]);

    // 14. 移除元素（键不存在，不抛异常）
    Console.WriteLine("\nRemove(\"数学\")");
    openWith.Remove("数学");
    openWith.RemoveAt(0); //移除第一个元素

    if (!openWith.ContainsKey("数学"))
    {
        Console.WriteLine("Key \"数学\" is not found.");
    }

    Console.WriteLine("=========================");
    // 输出剩余元素（会自动排序）
    foreach (KeyValuePair<string, string> kvp in openWith)
    {
        Console.WriteLine("Key = {0}, Value = {1}",
            kvp.Key, kvp.Value);
    }

    // SortedList能排序的本质：实现了ICompare接口
}
```
