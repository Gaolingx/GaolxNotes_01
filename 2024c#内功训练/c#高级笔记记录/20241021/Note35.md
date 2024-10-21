# C#高级编程之——泛型集合（七）字典集合

## 七、泛型集合——ConcurrentDictionary 类

## 线程安全性

只要集合未修改，A Dictionary 就可以同时支持多个读取器。 即便如此，通过集合进行遍历本质上不是线程安全的过程。 在遍历与写入访问竞争的极少数情况下，必须在整个遍历期间锁定集合。 若要允许多个线程访问集合以进行读写操作，则必须实现自己的同步。（如使用lock关键字加锁）

有关线程安全的替代，请参阅 ConcurrentDictionary 类或 ImmutableDictionary 类。

### 7.1 ConcurrentDictionary<TKey,TValue>

**7.1.1 声明：**表示可由多个线程同时访问的键/值对的线程安全集合。TKey ： 字典中的键的类型。TValue ：字典中的值的类型。

所有公共成员和受保护成员 ConcurrentDictionary 都是线程安全的，并且可以从多个线程并发使用。 但是，通过重写（包括扩展方法） ConcurrentDictionary 之一访问的成员不能保证线程安全，并且可能需要由调用方同步。

System.Collections.Generic.Dictionary与类一样，ConcurrentDictionary实现IDictionary接口。 此外， ConcurrentDictionary 还提供了几种方法用于在字典中添加或更新键/值对，如下表所述。

**7.1.2 常用属性和方法：**

<div class="table-wrapper"><table class="md-table">
<thead>
<tr class="md-end-block"><th><span class="td-span"><span class="md-plain">要执行此操作</span></span></th><th><span class="td-span"><span class="md-plain">方法</span></span></th><th><span class="td-span"><span class="md-plain">使用注意事项</span></span></th></tr>
</thead>
<tbody>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">将新键添加到字典（如果字典中尚不存在）</span></span></td>
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.tryadd?view=net-6.0" rel="noopener nofollow"><span class="md-plain">TryAdd</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">如果字典中当前不存在该键，此方法将添加指定的键/值对。 该方法返回 <span class="md-pair-s"><code>true</code><span class="md-plain"> 或 <span class="md-pair-s"><code>false</code><span class="md-plain"> 取决于是否添加了新对。</span></span></span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">更新字典中现有键的值（如果该键具有特定值）</span></span></td>
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.tryupdate?view=net-6.0" rel="noopener nofollow"><span class="md-plain">TryUpdate</span></a></span></span></td>
<td><span class="td-span"><span class="md-plain">此方法检查密钥是否具有指定的值，如果具有指定值，则使用新值更新密钥。 它类似于 <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.threading.interlocked.compareexchange?view=net-6.0" rel="noopener nofollow"><span class="md-plain">CompareExchange</span></a><span class="md-plain"> 该方法，只不过它用于字典元素。</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">无条件地将键/值对存储在字典中，并覆盖已存在的键的值</span></span></td>
<td><span class="td-span"><span class="md-plain">索引器的 setter： <span class="md-pair-s"><code>dictionary[key] = newValue</code></span></span></span></td>
<td>&nbsp;</td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">将键/值对添加到字典，或者如果键已存在，请根据键的现有值更新键的值</span></span></td>
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.addorupdate?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-addorupdate(-0-system-func((-0-1)" rel="noopener nofollow"><span class="md-plain">AddOrUpdate(TKey, Func, Func)</span></a><span class="md-plain">-system-func((-0-1-1)))) - 或 - <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.addorupdate?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-addorupdate(-0-1-system-func((-0-1-1)" rel="noopener nofollow"><span class="md-plain">AddOrUpdate(TKey, TValue, Func)</span></a><span class="md-plain">))</span></span></span></span></span></td>
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.addorupdate?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-addorupdate(-0-system-func((-0-1)" rel="noopener nofollow"><span class="md-plain">AddOrUpdate(TKey, Func, Func)</span></a><span class="md-plain">-system-func((-0-1-1)))) 接受Key和两个委托。 如果字典中不存在Key，它将使用第一个委托;它接受Key并返回应为Key添加的值。 如果密钥存在，它将使用第二个委托;它接受键及其当前值，并返回应为键设置的新值。 <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.addorupdate?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-addorupdate(-0-1-system-func((-0-1-1)" rel="noopener nofollow"><span class="md-plain">AddOrUpdate(TKey, TValue, Func)</span></a><span class="md-plain">)) 接受密钥、要添加的值和更新委托。 这与上一个重载相同，只不过它不使用委托来添加密钥。</span></span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">获取字典中键的值，将值添加到字典中，如果键不存在，则返回它</span></span></td>
<td><span class="td-span"><span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.getoradd?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-getoradd(-0-1)" rel="noopener nofollow"><span class="md-plain">GetOrAdd(TKey, TValue)</span></a><span class="md-plain"> - 或 - <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.getoradd?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-getoradd(-0-system-func((-0-1)" rel="noopener nofollow"><span class="md-plain">GetOrAdd(TKey, Func)</span></a><span class="md-plain">))</span></span></span></span></span></td>
<td><span class="td-span"><span class="md-plain">这些重载为字典中的键/值对提供延迟初始化，仅当它不存在时才添加值。 <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.getoradd?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-getoradd(-0-1)" rel="noopener nofollow"><span class="md-plain">GetOrAdd(TKey, TValue)</span></a><span class="md-plain"> 如果键不存在，则采用要添加的值。 <span class="md-meta-i-c  md-link"><a href="https://docs.microsoft.com/zh-cn/dotnet/api/system.collections.concurrent.concurrentdictionary-2.getoradd?view=net-6.0#system-collections-concurrent-concurrentdictionary-2-getoradd(-0-system-func((-0-1)" rel="noopener nofollow"><span class="md-plain">GetOrAdd(TKey, Func)</span></a><span class="md-plain">)) 获取一个委托，如果键不存在，将生成该值。</span></span></span></span></span></span></td>
</tr>
</tbody>
</table></div>

**说明：**所有这些操作都是原子操作，对于类上 ConcurrentDictionary 所有其他操作都是线程安全的。 唯一的例外是接受委托的方法，即 AddOrUpdate和 GetOrAdd。 若要对字典进行修改和写入操作， ConcurrentDictionary 请使用细粒度锁定来确保线程安全。 (字典上的读取操作以无锁方式执行。) 但是，这些方法的委托在锁外部调用，以避免在锁下执行未知代码时可能出现的问题。 因此，这些委托执行的代码不受操作的原子性的约束。

**如何选择：**如果不涉及线程安全问题，一般使用Dictionary<TKey,TValue>， 否则使用ConcurrentDictionary<TKey,TValue>。

**7.1.3 使用：**

```csharp
[Test]
public void TestConcurrentDictionary01()
{
    ConcurrentDictionary<string, int> dict = new();

    // 1. 添加
    dict["语文"] = 90;
    dict.TryAdd("数学", 110);

    // 2. 是否包含键
    if (dict.ContainsKey("英语"))
    {
        Console.WriteLine("The Dictionary Contain Key:英语");
    }

    // 3. 移除
    dict.Remove("语文", out int result);
    Console.WriteLine($"Remove Key:语文,result:{result}");

    // 4. 添加(AddOrUpdate)
    bool flag = dict.TryAdd("语文", 120);
    Console.WriteLine($"Key:语文,result:{dict["语文"]}");
    dict.AddOrUpdate("语文", 80, (k, v) =>
    {
        Console.WriteLine($"Key:语文,k={k},v={v}");
        return 60;
    }); //如果键存在则修改，但需要声明修改的方法（委托）
    dict.AddOrUpdate("数学", 120, (k, v) => { return 100; });
    Console.WriteLine($"Key:数学,result:{dict["数学"]}");

    // 5. 添加(GetOrAdd)
    var result2 = dict.GetOrAdd("物理", 130); //键不存在则添加
    Console.WriteLine($"Key:物理,result:{dict["物理"]},isSuccess:{result2}");
}
```

运行结果如下：
