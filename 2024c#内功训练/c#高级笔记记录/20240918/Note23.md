# C#高级编程之——泛型（四）泛型协变和逆变

## 六、协变

### 6.1 前言

先来看一个例子，假如我们现在有一个Student的子类和Person的基类如下：

```csharp
public class Student : Person
{
    public int Id { get; set; }
    public string Name { get; set; }
}

public class Person
{

}
```

众所周知我们通过父类(Person)实例化子类(Student)，当我们使用 TestClass02<Person> person = new TestClass02<Person>(); 不会报错，但是当我使用 TestClass02<Person> person2 = new TestClass02<Student>(); 就会报错，提示"无法将类型 'type' 隐式转换为 'type'"之类的错误(CS0029)，尽管Person和Student是继承关系，但是 TestClass02<Person> 和 TestClass02<Student> 不存在继承关系，所以它两不是相同的类型，因此不能这样实例化，这里就需要用到协变。

### 6.2 协变（Covariance）

**概念：**协变允许将派生类型的对象赋值给基类类型的对象，但这是在泛型接口或委托的返回类型上实现的。简单来说，如果泛型接口或委托的返回类型是协变的，那么你可以将一个返回派生类型对象的接口或委托赋值给一个返回基类类型对象的接口或委托。

**注意：**只能通过泛型接口实现协变，而不是类。

1. 在开始之前先准备好两个子父类

```csharp
public class Animal
{
    public virtual void Run()
    {
        Console.WriteLine("动物在跑");
    }
}

public class Dog:Animal
{
    public override void Run()
    {
        Console.WriteLine("狗在跑");
    }
}
```

2. 定义好泛型协变接口

```csharp
// 协变泛型接口，必须要在泛型前面加上 out关键字
public interface IFactory<out T> // out 协变关键字 只能应用于interface
{
    T Create();
}
```

3. 子类实现协变接口

```csharp
public class FactoryImpl<T> : IFactory<T> where T : new()
{
    public T Create()
    {
        return new T();
    }
}
```

4. 测试泛型协变

```csharp
[Test]
public void Test3()
{
    // 定义泛型接口是为了实现协变（一定要加 out关键字）
    // 实现类是因为接口本身不能实例化，只能通过具体的类实现
    IFactory<Dog> iFactory = new FactoryImpl<Dog>();
    IFactory<Animal> parentFactory = iFactory; // 协变

    Animal animal = parentFactory.Create();
    animal.Run();// 输出结果：狗在跑
}
```

**总结：**

1. 泛型接口中的out关键字必不可少
2. out 协变 只能应用于interface
