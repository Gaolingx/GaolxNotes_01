# C#高级编程之——泛型集合（三）Queue<T>

## 四、泛型集合——Queue<T> 队列

### 4.1 特点：与栈相反，先进先出

### 4.2 使用：

在C#中，`Queue<T>` 是一个泛型集合，用于表示先进先出（FIFO, First In First Out）的数据结构。这意味着最早添加到队列中的元素将是第一个被移除的元素。

### 4.3 特点

1. **先进先出**：
   - 队列遵循先进先出的原则，元素按照添加的顺序被移除。

2. **泛型**：
   - `Queue<T>` 是泛型类，这意味着你可以指定队列中存储的元素类型，例如 `Queue<int>`, `Queue<string>` 等。

3. **动态大小**：
   - 队列的大小是动态调整的，你可以根据需要添加或移除元素，而不需要手动管理其容量。

4. **线程安全**：
   - 默认情况下，`Queue<T>` 不是线程安全的。如果在多线程环境中使用，需要额外的同步机制，或者使用 `ConcurrentQueue<T>`。

### 4.4 基本操作

- **Enqueue**：将元素添加到队列的末尾。
- **Dequeue**：移除并返回队列的第一个元素。如果队列为空，则抛出 `InvalidOperationException`。
- **Peek**：返回队列的第一个元素但不移除它。如果队列为空，则抛出 `InvalidOperationException`。
- **Count**：获取队列中元素的数量。
- **Clear**：移除队列中的所有元素。
- **Contains**：检查队列是否包含某个元素。
- **CopyTo**：将队列中的元素复制到数组。
- **ToArray**：将队列中的元素复制到新的数组中并返回该数组。
- **Enumerate**：可以使用 `foreach` 循环遍历队列中的元素。

### 4.5 使用案例

```csharp
[Test]
public void TestQueue01()
{
    Queue<int> q = new Queue<int>(4); //初始化一个队列，容量为4

    // 1. 添加元素到队列：Enqueue
    q.Enqueue(100);
    q.Enqueue(200);
    q.Enqueue(300);

    // 2. 从队列中获取一个元素，但是不移除
    var peek = q.Peek();
    Console.WriteLine($"peek: {peek}"); //先进先出
    Console.WriteLine($"peek: {peek}");

    // 3. 遍历队列
    foreach (var item in q)
    {
        Console.WriteLine(item);
    }

    // 4. 从队列中取出元素并移除，返回值为移除的元素
    var result = q.Dequeue();
    var result2 = q.Dequeue();
    var result3 = q.Dequeue();

    Console.WriteLine($"Count:{q.Count}");

    int result4 = 0;
    bool flag = q.TryDequeue(out result4); //out 是操作失败则返回初始值，成功则返回移除元素的值，
                                           // TryDequeue 为操作是否成功（bool类型）成功为true

    Console.WriteLine($"Success:{flag},result:{result4}");
}
```

运行结果如下：

### 4.6 注意事项

- **空队列操作**：在空队列上调用 `Dequeue` 或 `Peek` 方法会抛出 `InvalidOperationException` 异常。
- **线程安全**：如果需要在多线程环境中使用，考虑使用 `ConcurrentQueue<T>` 或者手动添加同步机制。