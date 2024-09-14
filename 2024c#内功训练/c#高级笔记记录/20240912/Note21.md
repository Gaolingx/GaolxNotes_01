# C#高级编程之——泛型（二）泛型方法

## 四、泛型方法

泛型方法，在C#中，泛型方法是一种使用泛型类型参数的方法。这意味着你可以在方法定义时指定一个或多个类型参数，这些类型参数在方法被调用时才会被具体的类型所替代。将指定类型参数（Type Parameter，通常以T 表示），紧随方法名，并包含在<>符号内。

4.1 定义泛型方法

泛型方法的定义与泛型类的定义类似，但它是作用在方法级别上。你需要在方法返回类型之前指定类型参数，这些类型参数被放在尖括号<>中。

**格式：**

```csharp
访问修饰符  方法返回类型   方法名<T>(参数列表)
{
    // 方法体...
}
```

**普通类中的泛型:**

案例一：

```csharp
// 泛型方法——求和
public class TestClass02
{
    public T Sum<T>(T a, T b) //返回值为泛型，泛型方法的<T>中需要指定类型参数
    {
        // dynamic: 它是在程序运行时才知道是什么类型，但会绕过编译时的类型检查
        return (dynamic)a + b; //由于编译器不知道a和b的类型，需要转换成dynamic类型，运行时确定
    }
}
```

运行结果如下：

案例二：

```csharp
// 泛型方法——求和
// 如果一个类下面有多个泛型方法，建议将这个类定义成泛型类
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

public class RunTestClass
{
    [Test]
    public void RunTest()
    {
        TestClass02<int> test = new TestClass02<int>(); //实例化时需声明类型
        test.Sum(1, 2);
        Console.WriteLine($"sum的结果是{test}");

        TestClass02<string> test2 = new TestClass02<string>();
        test2.Print();
    }
}
```

运行结果如下：

结论：

1. 如果一个类下面有多个泛型方法，建议将这个类定义成泛型类
2. 如果在类上定义泛型，则其中的方法均为泛型方法
