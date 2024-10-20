# C#高级编程之——泛型集合（六）字典集合

## 六、泛型集合——HashTable与Dictionary<TKey,TValue>类

### 6.2 Dictionary<TKey,TValue>

**6.2.1 声明：**表示键和值的集合。TKey : 字典中的键的类型。TValue : 字典中的值的类型。

`Dictionary<TKey, TValue>` 是 C# 中一个非常重要的集合类，它表示键值对的集合。每个元素都是一个键值对（Key-Value Pair），通过键（Key）可以快速查找、添加、移除或访问值（Value）。
解决了HashTable类型安全问题。效率高。但是线程非安全，多线程场景下需要lock确保线程安全，或者使用ConcurrentDictionary。

**6.2.2 常用属性和方法：**

- **Count**：获取字典中键值对的数量。
- **Item[TKey]**：获取或设置与指定键相关联的值。如果键不存在，获取操作会抛出 `KeyNotFoundException` 异常；设置操作会添加新的键值对。
- **Keys**：获取包含字典中键的集合。
- **Values**：获取包含字典中值的集合。
- **Add(TKey, TValue)**：向字典中添加一个带有指定键和值的元素。如果键已存在，则会抛出 `ArgumentException` 异常。
- **ContainsKey(TKey)**：确定字典是否包含指定的键。
- **ContainsValue(TValue)**：确定字典是否包含指定的值。
- **Remove(TKey)**：移除具有指定键的元素。如果键不存在，则什么也不做。
- **TryGetValue(TKey, out TValue)**：获取与指定键相关联的值（如果存在）。该方法返回一个布尔值，指示是否找到键，并通过 `out` 参数返回值（如果找到）。
- **Clear()**：从字典中移除所有键值对。

**6.2.3 使用：**

```csharp
[Test]
public void TestDictionary01()
{
    Dictionary<string, int> openWith = new Dictionary<string, int>();

    // 初始化数据，不能存在重复键
    openWith.Add("语文", 90);
    openWith.Add("数学", 120);
    openWith.Add("英语", 130);
    openWith.Add("物理", 60);

    // 1. 添加重复键会抛出异常
    try
    {
        openWith.Add("语文", 90);
    }
    catch (ArgumentException)
    {
        Console.WriteLine("An element with Key = \"语文\" already exists.");
    }

    // 2. 通过索引取值
    Console.WriteLine("For key = \"数学\", value = {0}.",
        openWith["数学"]);

    // 3. 给已存在的键值索引赋值
    openWith["数学"] = 130; // Key不存在：新增 Key存在：更新
    Console.WriteLine("For key = \"数学\", value = {0}.",
        openWith["数学"]);

    // 4. 如果不存在，则会新增
    openWith["地理"] = 80;

    // 5. 如果访问一个不存在的索引值，则会抛出异常
    try
    {
        Console.WriteLine("For key = \"化学\", value = {0}.",
            openWith["化学"]);
    }
    catch (KeyNotFoundException)
    {
        Console.WriteLine("Key = \"化学\" is not found.");
    }

    // 6. tryValue 尝试取值
    if (openWith.TryGetValue("地理", out var value))
    {
        Console.WriteLine("For key = \"地理\", value = {0}.", value);
    }
    else
    {
        Console.WriteLine("Key = \"地理\" is not found.");
    }

    // 7. 判断是否包含键
    if (!openWith.ContainsKey("ht"))
    {
        openWith.Add("化学", 75);
        Console.WriteLine("Value added for key = \"化学\": {0}",
            openWith["化学"]);
    }

    // 8. 遍历循环，元素被检索为 KeyValuePair 对象
    Console.WriteLine();
    foreach (KeyValuePair<string, int> kvp in openWith)
    {
        Console.WriteLine("Key = {0}, Value = {1}",
            kvp.Key, kvp.Value);
    }

    // 9. 获取所有的值集合
    Dictionary<string, int>.ValueCollection valueColl =
        openWith.Values;

    // 10. 遍历值集合
    Console.WriteLine();
    foreach (int s in valueColl)
    {
        Console.WriteLine("Value = {0}", s);
    }

    // 11. 获取所有的键集合
    Dictionary<string, int>.KeyCollection keyColl =
        openWith.Keys;

    // 12. 遍历键集合
    Console.WriteLine();
    foreach (string s in keyColl)
    {
        Console.WriteLine("Key = {0}", s);
    }

    // 13. 移除键值对
    Console.WriteLine("\nRemove(\"英语\")");
    openWith.Remove("英语");

    if (!openWith.ContainsKey("英语"))
    {
        Console.WriteLine("Key \"英语\" is not found.");
    }
}
```

运行结果如下：
