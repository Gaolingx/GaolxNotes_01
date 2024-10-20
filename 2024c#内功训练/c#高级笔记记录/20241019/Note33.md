# C#高级编程之——泛型集合（六）字典集合

## 六、泛型集合——HashTable与Dictionary<TKey,TValue>类

### 6.1 HashTable

**简介：**HashTable 哈希表, 也叫散列表，是一种基于哈希函数实现的数据结构,通过哈希函数将键（Key）映射到表中的某个位置，以便快速访问与键相关联的值（Value）。

**特点：**

- 查找效率高：由于哈希函数的存在，HashTable可以在常数时间内完成查找操作，这使得它在处理大规模数据时具有很高的效率。
- 不保证有序性：HashTable中的元素不按照插入顺序或键的顺序存储，因此它不提供排序功能。
- 它是线程安全的（多个线程同时并发操作一段代码，同一时间只有一个线程执行）：因为它的所有方法都使用了同步机制，也导致了它在高并发场景下的性能瓶颈。

**应用场景：**

- 快速查找：HashTable适用于需要快速查找数据的场景，如缓存系统、数据库索引等。
- 去重：可以利用HashTable来检测数据中的重复项，因为每个键在HashTable中只能映射到一个唯一的位置。

值得强调的是：常见的Hash算法有MD5，SHA1，目前推荐的Hash 算法是：SHA2-256。

**使用：**

```csharp
[Test]
public void TestHashtable01()
{
    Hashtable openWith = new Hashtable();

    // 初始化一批数据，不可出现重复键
    openWith.Add("txt", "notepad.exe");
    openWith.Add("bmp", "paint.exe");
    openWith.Add("dib", "paint.exe");
    openWith.Add("rtf", "wordpad.exe");

    // 如果出现重复键，则会抛出异常
    try
    {
        openWith.Add("txt", "winword.exe");
    }
    catch
    {
        Console.WriteLine("An element with Key = \"txt\" already exists.");
    }

    // 通过索引访问
    Console.WriteLine("For key = \"rtf\", value = {0}.", openWith["rtf"]);

    // 修改索引所关联的值
    openWith["rtf"] = "winword.exe";
    Console.WriteLine("For key = \"rtf\", value = {0}.", openWith["rtf"]);

    // 给一个不存在的键赋值，则会新增
    openWith["doc"] = "winword.exe";

    // 判断是否包含
    if (!openWith.ContainsKey("ht"))
    {
        openWith.Add("ht", "hypertrm.exe");
        Console.WriteLine("Value added for key = \"ht\": {0}", openWith["ht"]);
    }

    // 遍历循环，元素被检索为 DictionaryEntry 对象
    Console.WriteLine();
    foreach (DictionaryEntry de in openWith)
    {
        Console.WriteLine("Key = {0}, Value = {1}", de.Key, de.Value);
    }

    // 获取所有的值集合
    ICollection valueColl = openWith.Values;

    // 遍历值集合
    Console.WriteLine();
    foreach (string s in valueColl)
    {
        Console.WriteLine("Value = {0}", s);
    }

    // 获取所有的键
    ICollection keyColl = openWith.Keys;

    // 遍历键集合
    Console.WriteLine();
    foreach (string s in keyColl)
    {
        Console.WriteLine("Key = {0}", s);
    }

    // 移除键值对
    Console.WriteLine("\nRemove(\"doc\")");
    openWith.Remove("doc");

    if (!openWith.ContainsKey("doc"))
    {
        Console.WriteLine("Key \"doc\" is not found.");
    }
}
```

运行结果如下：

**注意：**

不建议将类用于 `Hashtable` 新开发。 相反，我们建议使用泛型 Dictionary 类。非泛型集合的主要问题：1. 可能潜在的拆装箱影响性能 2. 类型安全无法保证
