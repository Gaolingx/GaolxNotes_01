# C#高级编程之——泛型（三）泛型约束

## 五、泛型约束

### 5.1 为什么要用泛型约束

```csharp

public class TestClass02<T> //如果在类上定义泛型，则其中的方法均为泛型方法
{
    public T Sum(T a, T b) //返回值为泛型，泛型方法的<T>中需要指定类型参数
    {
        // dynamic: 它是在程序运行时才知道是什么类型，但会绕过编译时的类型检查
        return (dynamic)a + b; //由于编译器不知道a和b的类型，需要转换成dynamic类型，运行时确定
    }

    public void Print()
    {
        Console.WriteLine($"{nameof(T)}的类型是{typeof(T).Name}");
    }
}

[Test]
public void Test2()
{
    MyClass my = new MyClass();
    Student s1 = new Student(1,"张三");
    Student s2 = new Student(2,"李四");
    my.Sum<Student>(s1, s2); // 合适吗？
}

record Student(int Id,string Name);
```

先来看上述案例，尽管能通过编译时的类型检查，但如果我们运行这段代码一定会报错，显然，两个Student对象不可能可以直接相加。此时，如果不对Sum 这个泛型方法加以约束，就很有可能出现上述情况。

所谓泛型约束，实际上就是约束的类型T。使T必须遵循一定的规则。比如T必须继承自某个类或者T必须实现某个接口等。 使用where关键字加上约束

### 5.2 官方定义

泛型是C#中一种支持参数化类型的机制，允许类或方法等结构在声明时不指定具体的数据类型，而是使用类型参数（Type Parameters）作为占位符，这些类型参数在类或方法被实例化时由实际的数据类型所替代

**简述：**泛型约束就是对泛型的管束。

### 5.3 泛型常见写法

```csharp
//格式
public class 泛型类<T> where T:约束类型
{
    
}

// 例如
class A<T> where T:new()
```

这是类型参数约束，其中where表示对类型变量T的约束关系。

**注意：约束不能组合或重复，并且必须先在约束列表中进行指定(CS0449)，但是可以有where T: class,new()，意味着不但T要为引用类型，还要具有无参构造.**

最常用的当属class和new()两种约束。泛型可以是值类型也可以是引用类型，class表示这个T为引用类型，new()表示这个泛型必须有构造函数否则不能使用。

### 5.4 约束的类型

除了class和new()之外，还有其他的约束关系。.NET支持的类型参数约束有以下五种：

where T: class        // T类型参数必须是引用类型，包括任何类、接口、委托或数组类型。
where T: new()        // T类型参数必须具有无参数的公共构造函数。当与其他约束一起使用时，new() 约束必须最后指定。
where T: struct       // T类型参数必须是值类型。可以指定除 Nullable 以外的任何值类型
where T: NameOfBaseClass    // T类型参数必须是指定的基类或派生自指定的基类
where T: NameOfInterface    // 类型参数必须是指定的接口或实现指定的接口。可以指定多个接口约束。约束接口也可以是泛型的。

### 5.5 泛型约束--struct

泛型约束中的struct 指定类型参数必须是值类型。可以指定除 Nullable 以外的任何值类型

```csharp
public class MyClass
{
    // 泛型方法
    public T Sum<T>(T a, T b) where T:struct
    {
        return (dynamic) a + b;
    }
}

[Test]
public void Test2()
{
    MyClass my = new MyClass();
    Student s1 = new Student(1,"张三");
    Student s2 = new Student(2,"李四");
    my.Sum<Student>(s1, s2); // 此时编译器直接给出错误提示，编译失败
}

record Student(int Id,string Name);
```

my.Sum<Student>(s1, s2); // 此时编译器直接给出错误提示，编译失败。

### 5.6 泛型约束--class

泛型约束class ,指定类型参数必须是引用类型，包括任何类、接口、委托或数组类型。

```csharp
public interface IRepository<T> where T:class
{
    // 接口也可以有默认实现
    int Add(T model)
    {
        Console.WriteLine("添加了一条数据");
        return 1;
    }

    int Update(T model);

    int Delete(dynamic id);

    T GetModel(dynamic id);

    IList<T> GetList(string condition);
}
```

如果有组合约束时，class约束必须放在最前面。

```csharp
public interface IRepository<T> where T:class,new() // class放前面，否则编译失败
{
    int Add(T model);

    int Update(T model);

    int Delete(dynamic id);

    T GetModel(dynamic id);

    IList<T> GetList(string condition);
}
```

测试效果：

```csharp
IRepository<int> repository = new IRepository<int>(); // 编译失败

IRepository<object> repository = new IRepository<object>(); // 编译通过
```

### 5.7 泛型约束—new()

泛型约束new(),指定类型参数必须具有无参数的公共构造函数。当与其他约束一起使用时，new() 约束必须最后指定。加上该约束后可以在类中或者方法中实例化T类型的对象。

```csharp
public class BaseDAL<T> where T:class,new() //new()放后面
{
    public List<T> GetList<T>()
    {
        List<T> list = new();
        T t = new(); // 可以实例化了
        list.Add(t);
        
        return list;
    }
}
```

测试效果

```csharp
BaseDAL<Student> dal = new BaseDAL<Student>(); // 编译失败，Student并未提供无参构造
 
record Student(int Id,string Name);
```

### 5.8 泛型约束—基类名

类型约束之基类名称，类型参数必须是指定的基类或派生自指定的基类

```csharp
public class StudentDal<T> where T:BaseModel
{
     
}
 
class BaseModel
{
     
}
```

**说明：**基类约束时，基类不能是密封类，即不能是sealed类。sealed类表示该类不能被继承，在这里用作约束就无任何意义，因为sealed类没有子类.

### 5.9 泛型约束—接口名称

泛型约束之接口名称，类型参数必须是指定的接口或实现指定的接口。可以指定多个接口约束。约束接口也可以是泛型的。

```csharp
interface IAnimal
{
    // ...
}
 
interface IPerson
{
    // ...
}
 
 
class 泛型类<T> where T:IAnimal,IPerson
{
     
}
 
class Student:IAnimal,IPerson
{
     
}
 
// 测试使用
泛型类<Student> myClass = new 泛型类<Student>(); // 测试通过
```

### 5.10 总结

当我们使用is、as进行类型转换时候，部分是可以通过设计避免的，例如你想限制传入参数的类型，可以通过泛型约束来做，这样还可以把运行时错误优化成编译时错误，减少一些潜在的类型转换错误，提高类型安全。
