# C#高级编程之——泛型集合（一）List< T >

## 一、为什么选择使用泛型集合

**1.1 非泛型集合存在的一些问题**

让我们观察如下代码：

```csharp
ArrayList arrylist = new ArrayList() { 14, "hello", 29.7, true};
arrylist.Add("world");// object
​
double dsum = 0;
foreach(var item in arrylist)
{
    dsum += Convert.ToDouble(item); // 出现异常
}
```

问题：
1、存取数据需要进行装箱拆箱（拆装箱问题）
2、数据类型转换存在隐患（类型安全问题）

**1.2 非泛型集合的性能**

非泛型集合性能：

```csharp
[Test]
public void Test1()
{
    Stopwatch watch = new Stopwatch();
    watch.Start();
    ArrayList arrayList = new();
    for (int i = 0; i < 2000000; i++)
    {
        arrayList.Add(i); // 装箱
    }
​
    long sum = 0;
    foreach (var item in arrayList)
    {
        sum += Convert.ToInt64(item);
    }
    watch.Stop();
    Console.WriteLine("非泛型集合耗时(ms)："+watch.ElapsedMilliseconds);
}
```

输出结果：非泛型集合耗时(ms)：395，too Slow！

泛型集合性能：

```csharp
[Test]
public void Test2()
{
    Stopwatch watch = new Stopwatch();
    watch.Start();
    var arrayList = new List<int>();
    for (int i = 0; i < 2000000; i++)
    {
        arrayList.Add(i); 
    }

    long sum = 0;
    foreach (var item in arrayList)
    {
        sum += Convert.ToInt64(item);
    }
    watch.Stop();
    Console.WriteLine("泛型集合耗时(ms)："+watch.ElapsedMilliseconds);
}
```

输出结果：泛型集合耗时(ms)：61

可以看出同样是循环2000000次取数据，非泛型集合存取数据要远慢于泛型集合

## 二、泛型集合——List<T>

**2.1 使用场景：**

1. 存储和操作一组相同类型的对象
2. 动态添加和移除元素
3. 访问和遍历元素
4. 查找元素
5. 排序和搜索
6. 作为其他集合类型的数据源

举例：运行如下代码，观察控制台输出

```csharp
public void Test3()
{
    List<int> list = new List<int>() { 2, 3, 7, 5 }; // 集合初始化器
    Console.WriteLine($"集合元素个数:{list.Count},容量:{list.Capacity}");
    list.Add(1);
    Console.WriteLine($"集合元素个数:{list.Count},容量:{list.Capacity}");
}
```

**2.2 声明**

声明泛型集合：
List<T> 集合名=new List<T>()

例如:
//值类型
List<int> list = new List<int>();
//引用类型
List<PersonModel> personList = new List<PersonModel>()

注意：
1、T只是占位符，会被传递的数据类型替换。
2、实例化List时传入相对应的数据类型
3、长度以2倍速度扩容

**2.3 List<T> 常用属性**

<div class="table-wrapper"><table class="md-table">
<thead>
<tr class="md-end-block"><th><span class="td-span"><span class="md-plain">Count</span></span></th><th><span class="td-span"><span class="md-plain">List集合中当前存储的元素个数</span></span></th></tr>
</thead>
<tbody>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">Capacity</span></span></td>
<td><span class="td-span"><span class="md-plain">List集合当前容量 Capacity&gt;=Count</span></span></td>
</tr>
</tbody>
</table></div>

例如：

```csharp
List<int> list = new List<int>() { 2, 3, 7, 5, 9 }; // 集合初始化器
```

输出：Count  :  5    Capacity   : 8

**2.4 List<T> 常用方法**

<div class="table-wrapper"><table class="md-table">
<thead>
<tr class="md-end-block"><th><span class="td-span"><span class="md-plain">Add（）</span></span></th><th><span class="td-span"><span class="md-plain">添加到List集合尾部 Add(元素) 如：strlist.Add(“me”)</span></span></th></tr>
</thead>
<tbody>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">Insert()</span></span></td>
<td><span class="td-span"><span class="md-plain">添加到List集合指定位置 Insert(下标，元素) 如：strlist.Insert(2,”Hi”)</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">Remove()</span></span></td>
<td><span class="td-span"><span class="md-plain">删除List集合中的元素 Remove(元素) 如：strlist.Remove(“c”)</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">RemoveAt()</span></span></td>
<td><span class="td-span"><span class="md-plain">删除List集合中指定下标的元素 RemoveAt(下标)如：strlist.RemoveAt(3)</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">RemoveRange()</span></span></td>
<td><span class="td-span"><span class="md-plain">删除List集合中指定范围的元素 RemoveRange(下标，个数)，<span class="md-br md-tag"> <span class="md-plain">如：strlist.RemoveRange(1，2)</span></span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">Clear()</span></span></td>
<td><span class="td-span"><span class="md-plain">清空List集合中所有元素 Clear()</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">First()</span></span></td>
<td><span class="td-span"><span class="md-plain">返回List集合中第一个元素</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">FirstOrDefault ()</span></span></td>
<td><span class="td-span"><span class="md-plain">返回List集合中第一个元素为空是返回默认值</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">Last()</span></span></td>
<td><span class="td-span"><span class="md-plain">返回List集合中最后一个元素</span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">LastOrDefault ()</span></span></td>
<td><span class="td-span"><span class="md-plain">返回List集合最后一个元素为空时返回默认值</span></span></td>
</tr>
</tbody>
</table></div>

例如：

```csharp
List<int> list = new List<int>() { 2, 3, 7, 5, 9 };

list.Add(10); // 2, 3, 7, 5, 9,10
list.Insert(2,6); //   2, 3,6, 7, 5, 9,10
list.Remove(2); // 3,6, 7, 5, 9,10
list.RemoveAt(0); // 6, 7, 5, 9,10
list.RemoveRange(1,2); // 6,9,10
list.First();// 6
list.FirstOrDefault(); // 6
list.Last();// 10
list.LastOrDefault(); // 10
list.Clear(); // 集合为空
```
