# C#高级编程之——事件（三）

在C#中，事件（Event）是一种特殊的多播委托（Multicast Delegate），用于在对象之间实现发布-订阅（Publisher-Subscriber）模式的通信。事件允许一个类或对象（发布者）通知其他类或对象（订阅者）某个特定的事情已经发生。

## 事件（Event）

在c#中，**事件(Event)**基本上说是一个用户操作，如按键、点击、鼠标移动等等，或者是一些提示信息，如系统生成的通知。应用程序需要在事件发生时响应事件。这意味着，事件处理方法可以事先绑定一系列的事件相应方法，当这个事件被触发时，这些绑定的方法将会被响应。

事件主要用于在对象之间提供松散的耦合，允许对象在保持彼此独立的同时进行通信。使用事件可以简化代码，提高可维护性，因为你可以在不修改发布者代码的情况下添加或移除订阅者。

### `event` 关键字

`event` 关键字用于声明一个事件。这个关键字定义了一个特殊的委托类型的成员，它只能由声明它的类或结构在其内部被触发（Raise）或附加/移除订阅者。

**语法**：
声明事件的语法如下：

```csharp
<访问修饰符> event 委托名称 事件名称; //注：事件其实是委托的实例
private event SpeakChinese SpeakChineseEvent;
```

**使用**：

注：

1. 事件只能通过 `+=` 绑定方法，`-=` 解绑方法，进行操作方法
2. 解绑方法（取消订阅）的操作不能作用在匿名方法上

```csharp
internal class TestEvent
{
    public event Action Event1;
    public event Action<string> Event2;

    /// <summary>
    /// 1. 委托的使用
    /// </summary>
    [Test]
    public void TestEvent01()
    {
        // event只能通过 += 方式订阅方法
        //订阅方法
        Event1 += () => { Console.WriteLine("Speak Chinese."); };
        Event1 += () => { Console.WriteLine("Speak English."); };

        //触发事件
        Event1?.Invoke();
    }
}
```

运行结果如下：

**反射查看事件本质**：

运行以下代码，断点查看Event1的类型信息

```csharp
/// <summary>
/// 2. 反射查看委托本质
/// </summary>
[Test]
public void TestEvent02()
{
    Event1 += () => { Console.WriteLine("Speak Chinese."); };
    Type type = Event1.GetType();
    Console.WriteLine($"{nameof(Event1)} is Class:{type.IsClass}, is Sealed:{type.IsSealed}");
    Console.WriteLine();
}
```

运行结果如下：

结论：
通过反射我们可以发现：

1. Type.IsClass: true
2. Type.BaseType: System.MulticastDelegate
3. Type.IsSealed: true

因此，事件的本质还是多播委托，并且事件还是一个密封类。

### `EventHandler` 和 `EventHandler<TEventArgs>`

`EventHandler` 是一个预定义的委托类型，用于表示处理不包含事件数据的事件的方法。其定义如下：

```csharp
public delegate void EventHandler(object sender, EventArgs e);
```

- `sender`：触发事件的对象。
- `e`：包含事件数据的对象，对于不包含事件数据的事件，可以使用 `EventArgs.Empty`。

当事件需要传递额外信息时，可以使用泛型版本的 `EventHandler<TEventArgs>`，其中 `TEventArgs` 继承自 `EventArgs`：

**操作步骤**：

1. **自定义事件数据类**：`MyEventArgs` 继承自 `EventArgs`，并包含一个 `Message` 属性。

   ```csharp
   // 自定义事件数据类
   public class MyEventArgs : EventArgs
   {
       public string Message { get; }
   
       public MyEventArgs(string message)
       {
           Message = message;
       }
   }
   ```

2. **发布者类**：`Publisher` 类声明了一个 `MyEvent` 事件，并在 `DoSomething` 方法中触发该事件。

   ```csharp
      // 发布者类
   public class Publisher
   {
       // 声明事件
       public event EventHandler<MyEventArgs>? MyEvent;
   
       // 触发事件的方法
       protected virtual void OnMyEvent(MyEventArgs e)
       {
           MyEvent?.Invoke(this, e);
       }
   
       // 模拟某个操作触发事件
       public void DoSomething()
       {
           Console.WriteLine("Doing something...");
           OnMyEvent(new MyEventArgs("Hello, this is a custom event!"));
       }
   }
   ```

3. **订阅者类**：`Subscriber` 类包含一个处理事件的方法 `HandleMyEvent`。

   ```csharp
   // 订阅者类
   public class Subscriber
   {
       public void HandleMyEvent(object? sender, MyEventArgs e)
       {
           Console.WriteLine("Received event with message: " + e.Message);
       }
   }
   ```

4. **主程序**：在 `Test` 方法中，创建 `Publisher` 和 `Subscriber` 实例，并订阅 `MyEvent` 事件。然后调用 `DoSomething` 方法触发事件，最后取消订阅。

   ```csharp
   [Test]
   public void TestEvent03()
   {
       Publisher publisher = new Publisher();
       Subscriber subscriber = new Subscriber();
   
       // 订阅事件
       publisher.MyEvent += subscriber.HandleMyEvent;
   
       // 触发事件
       publisher.DoSomething();
   
       // 取消订阅事件
       publisher.MyEvent -= subscriber.HandleMyEvent;
   }
   ```

完整代码：

```csharp
//------------事件------------//
// 自定义事件数据类
public class MyEventArgs : EventArgs
{
    public string Message { get; }

    public MyEventArgs(string message)
    {
        Message = message;
    }
}

// 发布者类
public class Publisher
{
    // 声明事件
    public event EventHandler<MyEventArgs>? MyEvent;

    // 触发事件的方法
    protected virtual void OnMyEvent(MyEventArgs e)
    {
        MyEvent?.Invoke(this, e);
    }

    // 模拟某个操作触发事件
    public void DoSomething()
    {
        Console.WriteLine("Doing something...");
        OnMyEvent(new MyEventArgs("Hello, this is a custom event!"));
    }
}

// 订阅者类
public class Subscriber
{
    public void HandleMyEvent(object? sender, MyEventArgs e)
    {
        Console.WriteLine("Received event with message: " + e.Message);
    }
}

[Test]
public void TestEvent03()
{
    Publisher publisher = new Publisher();
    Subscriber subscriber = new Subscriber();

    // 订阅事件
    publisher.MyEvent += subscriber.HandleMyEvent;

    // 触发事件
    publisher.DoSomething();

    // 取消订阅事件
    publisher.MyEvent -= subscriber.HandleMyEvent;
}
```

运行结果如下：

## 事件与委托区别

1. 事件只能在方法的外部进行声明，而委托在方法的外部和内部都可以进行声明;
2. 事件只能在类的内部进行触发，不能在类的外部进行触发。而委托在类的内部和外部都可触发;（可以包装成方法并对外暴露）
3. 委托一般用于回调，而事件一般用于外部接口。在观察者模式中，被观察者可在内部声明一个事件作为外部观察者注册的接口。
4. 事件只能通过+=，-=方式绑定/解绑方式
5. 事件是一个特殊的委托，查看反编译工具之后的代码，发现事件是一个private委托，所有的委托类型都继承自System.MulticastDelegate，而System.MulticastDelegate又继承自System.Delegate。
