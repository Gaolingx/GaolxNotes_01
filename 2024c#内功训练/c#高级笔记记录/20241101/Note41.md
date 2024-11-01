# C#高级编程之——泛型集合（九）Set集合

## 九、泛型集合——HashSet<T> 类

**简介：**众所周知，List<T>可以存储重复元素，主要被设计用来存储集合，做高性能集运算，例如两个集合求交集、并集、差集等。从名称可以看出，它是基于Hash的（是ISet的一个实现），可以简单理解为没有Value的Dictionary。
HashSet<T>不能用索引访问，不能存储重复数据。

`HashSet<T>` 是 C# 中的一个集合类型，它表示元素的集合，其中每个元素都是唯一的。通常用于去重。

**特点：**

1. **唯一性**：
   - `HashSet<T>` 保证集合中的每个元素都是唯一的。如果尝试向集合中添加一个已经存在的元素，该操作将被忽略，并且集合的大小不会改变。

2. **性能**：
   - `HashSet<T>` 提供了高效的查找、添加和删除操作，这些操作通常在平均情况下具有 O(1) 的时间复杂度。
   - 这是因为它基于哈希表实现，使得元素的查找和插入非常迅速。

3. **初始化**：
   - 可以使用集合初始化器或向构造函数传递一个已有的集合来初始化 `HashSet<T>`。

   ```csharp
   HashSet<int> numbers = new HashSet<int> { 1, 2, 3, 4, 5 };
   HashSet<string> fruits = new HashSet<string>(new List<string> { "Apple", "Banana", "Cherry" });
   ```

4. **操作方法**：
   - `Add(T item)`：添加一个新元素到集合中。如果元素已经存在，则不添加并且返回 `false`，否则返回 `true`。
   - `Remove(T item)`：从集合中移除一个元素。如果元素存在并且被成功移除，则返回 `true`，否则返回 `false`。
   - `Contains(T item)`：检查集合中是否包含某个元素，返回 `true` 或 `false`。
   - `Clear()`：清空集合中的所有元素。
   - `UnionWith(IEnumerable<T> collection)`：将指定集合中的所有元素添加到当前集合中，但仅添加不在当前集合中的元素。
   - `IntersectWith(IEnumerable<T> collection)`：保留当前集合和指定集合中都存在的元素，其他元素都被移除。
   - `ExceptWith(IEnumerable<T> collection)`：移除当前集合中与指定集合中存在的所有元素。
   - `SymmetricExceptWith(IEnumerable<T> collection)`：保留当前集合和指定集合中一个集合存在但另一个集合不存在的元素。

5. **遍历**：
   - 可以使用 `foreach` 循环来遍历 `HashSet<T>` 中的元素。由于 `HashSet<T>` 不保证元素的顺序，因此遍历的顺序可能是不确定的。

   ```csharp
   foreach (int number in numbers)
   {
       Console.WriteLine(number);
   }
   ```

6. **线程安全**：
   - `HashSet<T>` 不是线程安全的。如果在多线程环境中使用，需要采取适当的同步措施，例如使用 `lock` 语句。

**使用：**

```csharp
[Test]
public void TestHashSet01()
{
    //ISet<int> ints = new HashSet<int>();
    HashSet<int> ints = new HashSet<int>();

    // 1. 添加元素
    ints.Add(1);
    ints.Add(1);
    ints.Add(2);
    ints.Add(2);
    ints.Add(3);

    //该集合之所以可以被遍历。是因为实现了IEnumerable<T>
    foreach (int i in ints)
    {
        Console.WriteLine(i);
    }

    // 2. List<T>集合去除重复元素
    List<int> list = new List<int>() { 10, 10, 20, 20 };

    foreach (int i in list)
    {
        Console.WriteLine($"List:{i}");
    }

    HashSet<int> ints2 = list.ToHashSet();
    foreach (int i in ints2)
    {
        Console.WriteLine($"List to HashSet:{i}");
    }

}
```

运行结果如下：
