# C#高级编程之——泛型（一）泛型类型

前面我们提到过，也用到过泛型，例如var site = JsonConvert.DeserializeObject(jsonText, typeof(Site)) as Site; 和 var site = JsonConvert.DeserializeObject<Site>(jsonText); 两种写法效果是等价的，可以说，泛型在c#中的使用非常广泛。

## 一、泛型的概念与作用

1.1 背景：
在上面那个例子中，尽管我们可以为JsonConvert.DeserializeObject<T>单独声明一个JsonConvert.DeserializeObject<Site>方法，但是假如类型出现了变化，就需要频繁的修改DeserializeObject方法，显然我们不可能针对每一个类型专门定义一个方法，那有什么一劳永逸的办法吗？答案就是：泛型，例如我们可以用<T> where T : class，来泛指所有的class。

再看下面一个例子，就应该能明白泛型是如何提将类型抽象出来的了。

**概念：**C# 泛型（Generics）是.NET Framework 2.0 引入的一个强大特性，它允许程序员在编写类、接口、方法时定义一个或多个类型参数（Type Parameters）。这些类型参数在类、接口或方法被实例化时（即创建对象或调用方法时）被具体的类型所替换。**泛型是可以当作任意一种且由编译期间决定其最终类型的数据类型。通俗来讲，泛型，即泛指某种类型。**

**作用：**

1. **泛型增强了代码的可读性（多个类型可共用一个代码）**
2. **泛型有助于实现代码的重用、保护类型的安全以及提高性能。（避免类型转换出错的风险）**
3. 我们可以创建泛型集合类。
4. 泛型实现了类型和方法的参数化（类型当作参数传递）
5. 我们还可以对泛型类进行约束以访问特定数据类型的方法。
6. 关于泛型数据类型中使用的类型的信息可在运行时通过使用反射获取。

**一句话总结：**泛型——泛指某种类型。（不是具体的类型）

## 二、泛型类

2.1 泛型类声明格式
泛型类，将指定类型参数（Type Parameter，通常以T 表示），紧随类名，并包含在<>符号内。

```csharp
public class 泛型类<T>
{
    /// <summary>
    /// 泛型属性
    /// </summary>
    public T ItemName { get; set; }
 
    public string MyName { get; set; } // 也可定义其他的属性
     
}
```

新建一个class，写入以下代码，向MyGenericClass01传入不同类型，分别分析其中的T代表什么类型

```csharp
namespace TestGeneric
{
    public class MyGenericClass01<T> //<T> 代指泛型，泛指某种类型，类型在编译期间确定，T也可以用其他字符标识
    {
        public T ItemName { get; set; }
        public int Total {  get; set; }
    }

    public class TestClass
    {
        MyGenericClass01<int> num = new MyGenericClass01<int>(); //T 代表int类型

        MyGenericClass01<string> str = new MyGenericClass01<string>(); //T 代表string类型
        private void Test()
        {
            num.ItemName = 10; //这里的ItemName变成了int类型
            str.ItemName = "爱莉小跟班gaolx"; //这里的ItemName变成了string类型
        }
    }
}
```

## 三、泛型类的应用

3.1 案例一：分页查询

由于返回的数据的类型是不确定的，显然我们不可能为每一个数据模型创建一个分页对象，因此需要将类型抽象出来，这样即可让不同的数据类型共享一份代码。（因为除了类型不一样，其他的behavior都是一样的）这便是泛型的意义。

背景：分页数据中，除了返回数据，还要包括当前页码、每页大小、总页数、总记录数等分页相关信息。

假设我们有如下两个分页对象

```csharp
namespace TestGeneric
{
    public class Student
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
}
```

```csharp
public class Product
{
    public int Id { get; set; }
    public int Name { get; set; }
}
```

则在 PageModel<T> 中有如下写法，即可将任意类型的数据装进PageModel了，是不是很方便？

```csharp
namespace TestGeneric
{
    //分页对象
    public class PageModel<T>
    {
        public List<T>? Datas { get; set; }
        public int Total {  get; set; }
    }
}
```
