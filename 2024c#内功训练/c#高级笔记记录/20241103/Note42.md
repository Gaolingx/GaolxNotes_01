# C#高级编程之——泛型集合（十）Set集合

## 十、泛型集合——SortedSet<T> 类

## 简介

C#SortedSet类可用于存储、删除或查看元素。它保持升序，不存储重复元素。如果必须存储唯一元素并保持升序，建议使用SortedSet类。它位于System.Collections.Generic命名空间中。它基于红黑树（一种自平衡二叉搜索树）实现，从而确保了元素的插入、删除和查找操作在对数时间内完成。

## 特点

1. **排序**：
   - `SortedSet<T>` 会根据元素的默认比较器（或者提供的自定义比较器）自动对元素进行排序。
   - 元素按升序排列，除非你提供了自定义的比较器来改变排序顺序。

2. **唯一性**：
   - `SortedSet<T>` 中的元素是唯一的，不允许有重复元素。

3. **添加和删除**：
   - 你可以使用 `Add` 方法添加元素。如果元素已经存在，则添加操作不会成功，也不会抛出异常。
   - 可以使用 `Remove` 方法删除元素。

4. **遍历**：
   - `SortedSet<T>` 实现了 `IEnumerable<T>` 接口，因此你可以使用 `foreach` 循环来遍历集合中的元素。
   - 它还支持从集合的开头或结尾进行遍历，并提供 `GetEnumerator` 方法以及反向枚举器。

5. **查找**：
   - `SortedSet<T>` 提供了高效的查找操作，比如 `Contains` 方法用于检查某个元素是否存在于集合中。
   - `TryGetValue` 方法可以用于查找特定元素并获取其值（如果存在）。

6. **范围操作**：
   - 你可以使用 `GetViewBetween` 方法获取集合中指定范围内的元素。
   - `GetEnumerator` 和 `GetReverseEnumerator` 方法支持基于比较器的范围遍历。

7. **自定义比较器**：
   - 你可以在创建 `SortedSet<T>` 实例时指定一个自定义的比较器 (`IComparer<T>`)，以改变默认的排序行为。

## 使用

```csharp
[Test]
public void TestSortedSet01()
{
    SortedSet<int> ints = new SortedSet<int>();

    // 1. 添加元素
    ints.Add(5);
    ints.Add(20);
    ints.Add(8);
    ints.Add(10);
    ints.Add(30);

    // 2. 遍历集合
    foreach (var item in ints)
    {
        Console.WriteLine(item); //检查元素是否被排序
    }
}
```

运行结果如下：

## 补充

### 1. SortedDictionary和SortedList对比

#### 1. 数据结构和性能

- **`SortedDictionary<TKey, TValue>`**：
  - **底层实现**：基于红黑树（一种自平衡二叉搜索树）。
  - **查找、插入和删除操作**：平均时间复杂度为O(log n)。
  - **内存使用**：相对较为紧凑，但比`SortedList`稍高，因为每个节点需要额外的引用和存储用于维护树的平衡。
  - **支持空键和空值**：允许键和值为null（如果键类型允许）。

- **`SortedList<TKey, TValue>`**：
  - **底层实现**：基于数组和双向链表（类似于哈希表，但使用键的排序顺序进行存储）。
  - **查找、插入和删除操作**：平均时间复杂度为O(1)（在插入点附近操作），但由于内部使用数组，当数组需要扩展时会有一定的性能开销。
  - **内存使用**：在较小的集合上表现较好，但随着集合大小的增加，性能可能会下降，因为数组扩展需要复制现有数据。
  - **不支持空键**：键不能为空，但值可以为null（如果值类型允许）。

#### 2. 键的类型限制

- **`SortedDictionary<TKey, TValue>`**：
  - 可以使用任何实现了`IComparable<TKey>`接口的键类型，或者通过提供一个`IComparer<TKey>`接口的实现来定义自定义排序。

- **`SortedList<TKey, TValue>`**：
  - 键类型必须是实现了`IComparable`接口的类型，这意味着键类型必须支持比较操作。

#### 3. 初始容量和扩容

- **`SortedDictionary<TKey, TValue>`**：
  - 没有初始容量的概念，因为它使用红黑树进行动态调整。

- **`SortedList<TKey, TValue>`**：
  - 可以指定一个初始容量，以减少在添加元素时的数组扩容开销。

#### 4. 迭代顺序

- **`SortedDictionary<TKey, TValue>`** 和 **`SortedList<TKey, TValue>`**：
  - 都按照键的排序顺序进行迭代。

### 使用场景建议

- 如果你需要一个高效且内存使用相对较少的排序集合，并且需要支持null键（如果键类型允许），`SortedDictionary`可能是更好的选择。
- 如果你处理的集合相对较小，且需要快速查找、插入和删除操作，同时不需要null键，`SortedList`可能是一个更合适的选择。

### 总结

使用上两者的接口都类似字典，并且SortedList的比如Find,FindIndex,RemoveAll常用方法都没提供。
数据结构上二者差异比较大，SortedList查找数据极快，但添加新元素，删除元素较慢，SortedDictionary查找，添加，删除速度都比较平均。
