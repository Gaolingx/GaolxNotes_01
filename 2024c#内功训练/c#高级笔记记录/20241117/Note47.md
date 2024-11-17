# C#高级编程之——委托（一）

## 委托的本质

委托的本质是一个（匿名函数）类。委托在C#中是通过delegate关键字定义的，但本质上它会被编译器转换为一个类。例如：

```csharp
public delegate int MyDelegate(int x, int y);
```

编译器会生成一个类似于以下的类（伪代码）：

```csharp
public class MyDelegate : System.MulticastDelegate
{
    public MyDelegate(object target, IntPtr method);
    public virtual int Invoke(int x, int y);
    public virtual IAsyncResult BeginInvoke(int x, int y, AsyncCallback callback, object object);
    public virtual int EndInvoke(IAsyncResult result);
}
```

## 用反射查看委托

```csharp
public delegate string DoSpeak3(int type);

[Test]
public void TestDelegate4()
{
    // 1.使用反射查看委托的本质
    Type type = typeof(DoSpeak3);
    Console.WriteLine($"{nameof(DoSpeak3)} is Class:{type.IsAnsiClass}, is Sealed:{type.IsSealed}");
}
```

运行结果如下：

## 委托实例的创建

委托实例的创建和调用与类的实例非常相似。例如：

```csharp
MyDelegate del = (x, y) => x + y;
int result = del(5, 3);  // 调用委托实例
```

## 结论

从以上代码可以看出，所有的委托类型都继承自System.MulticastDelegate，而System.MulticastDelegate又继承自System.Delegate。System.Delegate是一个基类，提供了委托类型的基本功能。因此，委托类型具有类的所有特性，比如可以包含字段、方法、属性等。

## 属性和方法

`System.MulticastDelegate`作为C#中所有委托的基类（委托本身都继承自`System.MulticastDelegate`，而`System.MulticastDelegate`又继承自`System.Delegate`），提供了一系列重要的方法和属性。

1. **构造函数**：

   * `protected MulticastDelegate(object target, string method)`: 这是一个受保护的构造函数，用于初始化`MulticastDelegate`类的新实例。它不能直接在应用程序代码中使用。`target`参数是在其上定义`method`的对象，`method`参数是为其创建委托的方法的名称。

2. **Invoke方法**：

   * `public virtual object Invoke(params object[] args)`: 这是一个虚拟方法，用于以同步方式调用委托对象所引用的方法。参数`args`是一个对象数组，包含要传递给被调用方法的参数。返回值是被调用方法的返回值，其类型与委托的返回类型相同。

   * 对于特定类型的委托，编译器会生成一个具有特定参数类型和返回类型的`Invoke`方法。例如，对于一个返回整数并接受两个整数参数的委托，编译器会生成一个`Invoke(int x, int y)`方法。

3. **BeginInvoke和EndInvoke方法**：

   * `public virtual IAsyncResult BeginInvoke(params object[] args, AsyncCallback callback, object state)`: 这是一个虚拟方法，用于在另一个线程上异步调用委托所引用的方法。它返回一个`IAsyncResult`接口，该接口可用于监视异步操作的进度。

   * `public virtual object EndInvoke(IAsyncResult result)`: 这是一个虚拟方法，用于结束一个异步调用。它接受一个`IAsyncResult`参数，该参数是`BeginInvoke`方法返回的。`EndInvoke`方法会阻塞调用线程，直到异步调用完成，并返回被调用方法的返回值。

4. **其他重要方法**：

   * `public virtual Delegate[] GetInvocationList()`: 此方法返回一个委托数组，其中包含了调用链中的所有委托对象。每个委托对象在数组中都是孤立的，即它们的`_prev`字段都被设置为`null`。

   * `public override bool Equals(object obj)`: 重写`Object`类的`Equals`方法，用于判断两个委托对象是否相等。对于`MulticastDelegate`，它还会比较委托链表。

   * `public override int GetHashCode()`: 重写`Object`类的`GetHashCode`方法，为委托对象生成一个哈希码。

5. **属性**：

   * `public object Target { get; }`: 获取与委托关联的对象（即委托所引用的方法的所属对象）。如果委托是静态方法的引用，则此属性返回`null`。

   * `public MethodInfo Method { get; }`: 获取与委托关联的方法的`MethodInfo`对象。


## 多播委托
