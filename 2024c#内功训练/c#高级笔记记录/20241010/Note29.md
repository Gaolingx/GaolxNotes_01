# C#高级编程之——泛型集合（一）

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

2.1 使用场景：

1. 在Linq 中比较常见
2. 存储数据
