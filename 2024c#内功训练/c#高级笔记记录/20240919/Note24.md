# C#高级编程之——泛型（五）泛型协变和逆变

## 七、逆变

在上个教程中，我们通过协变的方式实现了 TestClass02<Person> person2 = new TestClass02<Student>(); 这样的操作，但如果反过来，让TestClass02<Student> person2 = new TestClass02<Person>(); 即泛型子类实例化泛型父类显然是不行的，因此这里需要实现逆变。可以实现例如 Action<Student> stu = new Action<Person>(); 这样的行为。

### 7.1 概念

**逆变（Contravariance）**
逆变与协变相反，它允许将基类类型的对象赋值给派生类型的对象，但这是在泛型接口或委托的参数类型上实现的。换句话说，如果泛型接口或委托的参数类型是逆变的，那么你可以将一个接受基类类型参数的接口或委托赋值给一个接受派生类型参数的接口或委托。

**如果基类泛型隐式转换成子类泛型，使用泛型逆变。**

### 7.2 操作

1. 关于通知的一个接口

```csharp
public interface INotification
{
    public string Message { get; }
}

// 关于通知接口的抽象实现。
public abstract class Notification : INotification
{
    public abstract string Message { get; }
}
```

2. 关于通知抽象类的具体实现

```csharp
public class MainNotification : Notification
{
    public override string Message => "您有一封新的邮件";
}
```

3. 接下来，需要把通知的信息发布出去，需要一个发布通知的接口INotifier，该接口依赖INotification，大致INotifier<INotification>，而最终显示通知，我们希望INotifier<MailNotification>，INotifier<INotification>转换成INotifier<MailNotification>，这是逆变，需要关键字in。

```csharp
public interface INotifier<in T> where T : INotification
{
    void Notify(T notification);
}
```

4. 实现INotifier：

```csharp
public class Notifier<T> : INotifier<T> where T : INotification
{
    public void Notify(T notification)
    {
        Console.WriteLine(notification.Message);
    }
}
```

5. 客户端调用

```csharp
[Test]
public void Test4()
{
    INotifier<INotification> notifier = new Notifier<INotification>();
    INotifier<MainNotification> mailNotifier = notifier; // 逆变
    mailNotifier.Notify(new MainNotification());
}
```

最后控制台输出结果如下：

### 7.2 总结

1. INotifier的方法Notify()的参数类型是INotification,逆变后把INotification类型参数隐式转换成了实现类 MailNotificaiton。
2. 泛型接口中的in关键字必不可少。

### 7.3 章节总结

1. 在C#中，通过在泛型接口或委托的返回类型前添加out关键字来标记协变。通过在泛型接口或委托的参数类型前添加in关键字来标记逆变。
2. 协变(Convariant)和逆变(Contravariance)的出现，使数组、委托、泛型类型的隐式转换变得可能。 子类转换成基类，称之为协变；基类转换成子类，称之为逆变。.NET4.0以来，支持了泛型接口的协变和逆变。
3. 逆变与协变只能放在泛型接口和泛型委托的泛型参数里面，在泛型中out修饰泛型称为协变，协变（covariant） 修饰返回值 ，协变的原理是把子类指向父类的关系，拿到泛型中。
   在泛型中in 修饰泛型称为逆变， 逆变（contravariant ）修饰传入参数，逆变的原理是把父类指向子类的关系，拿到泛型中。

### 7.4 内置的协变逆变泛型

<div class="table-wrapper"><table class="md-table">
<thead>
<tr class="md-end-block"><th><span class="td-span"><span class="md-plain">序号</span></span></th><th><span class="td-span"><span class="md-plain">类型</span></span></th><th><span class="td-span"><span class="md-plain">名称</span></span></th></tr>
</thead>
<tbody>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">1</span></span></td>
<td><span class="td-span"><span class="md-plain">接口</span></span></td>
<td><span class="td-span"><span class="md-plain">IEnumerable<span class="md-tag md-raw-inline">&lt;out T&gt;</span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">2</span></span></td>
<td><span class="td-span"><span class="md-plain">委托</span></span></td>
<td><span class="td-span"><span class="md-plain">Action<span class="md-tag md-raw-inline">&lt;in T&gt;</span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">3</span></span></td>
<td><span class="td-span"><span class="md-plain">委托</span></span></td>
<td><span class="td-span"><span class="md-plain">Func<span class="md-tag md-raw-inline">&lt;out T&gt;</span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">4</span></span></td>
<td><span class="td-span"><span class="md-plain">接口</span></span></td>
<td><span class="td-span"><span class="md-plain">IReadOnlyList<span class="md-tag md-raw-inline">&lt;out T&gt;</span></span></span></td>
</tr>
<tr class="md-end-block">
<td><span class="td-span"><span class="md-plain">5</span></span></td>
<td><span class="td-span"><span class="md-plain">接口</span></span></td>
<td><span class="td-span"><span class="md-plain">IReadOnlyCollection<span class="md-tag md-raw-inline">&lt;out T&gt;</span></span></span></td>
</tr>
</tbody>
</table></div>

### 7.5 快速区分

泛型父类给子类赋值——逆变，泛型子类给父类赋值——协变。
