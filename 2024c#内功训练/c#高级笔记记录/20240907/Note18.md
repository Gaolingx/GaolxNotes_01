# C#高级编程之——单元测试（一）基本介绍与使用

## 一、为什么需要单元测试

在我们之前，测试某些功能是否能够正常运行时，我们都将代码写到Main方法中，当我们测试第二个功能时，我们只能选择将之前的代码清掉，重新编写。此时，如果你还想重新测试你之前的功能时，这时你就显得有些难为情了，因为代码都被你清掉了。当然你完全可以把代码写到一个记事本中进行记录，但是这样总归没有那么方便。当然你也可以重新新建一个项目来测试新的功能，但随着功能越来越多，重新新建项目使得项目越来越多，变得不易维护，此时你若选择使用单元测试功能，就可以完美解决你的困扰。

NUnit 提供了单元测试能力，也是目前用的比较流行的单元测试组件。

## 二、什么是NUnit？

NUnit是一个专为.NET平台设计的开源单元测试框架，它属于xUnit家族的一员，最初从 JUnit 移植，当前的生产版本 3 已完全重写，具有许多新功能和对各种 .NET 平台的支持。
作为适用于所有 .Net 语言的单元测试框架，与JUnit（Java）、CPPUnit（C++）等测试框架有着相似的架构和理念。NUnit通过提供一系列丰富的工具和特性，帮助开发者编写、组织和运行单元测试，以确保.NET应用程序的各个部分按预期工作。

## 三、NUnit 使用

步骤：
打开Nuget包管理器，通过Nuget 安装 如下三个包：

- NUnit
- NUnit3TestAdapter
- Microsoft.NET.Test.Sdk

注：

- NUnit 包应由每个测试程序集引用，但不应由任何其他测试程序集引用。
- NUnit使用自定义特性来标识测试。 所有的NUnit属性都包含在NUnit中的 框架的命名空间。 包含测试的每个源文件必须包含该名称空间的using语句（NUnit.Framework），项目必须引用框架程序集nunit.framework.dll。

## 四、创建你的第一个单元测试

为了标识出你的测试用例，你需要在需要进行单元测试的方法前加上 Test Attribute，标记表示测试的 TestFixture 的方法。

其余相关的Attribute参考如下图：

新建一个类，创建一个测试方法，在其前面加上[Test]，代码中右键点击 运行测试（或按Ctrl+R,T），稍等片刻即可在测试资源管理器看到结果：

## 五、TestFixture

功能：此属性标记包含测试以及（可选）设置或拆解方法的类。

现在，对用作测试夹具的类的大多数限制都已消除。TestFixture类：

- 可以是公共的、受保护的、私有的或内部的。
- 可能是静态类。
- 可以是泛型的，只要提供了任何类型参数，或者可以从实际参数中推断出来。
- 可能不是抽象的 - 尽管该属性可以应用于旨在用作TestFixture基类的抽象类。
- 如果 TestFixtureAttribute 中没有提供任何参数，则该类必须具有默认构造函数。
- 如果提供了参数，则它们必须与其中一个构造函数匹配。

如果违反了这些限制中的任何一个，则该类不可作为测试运行，并且将显示为错误。

建议构造函数没有任何副作用，因为 NUnit 可能会在会话过程中多次构造对象。

从 NUnit 2.5 开始，TestFixture 属性对于非参数化、非通用Fixture是可选的。只要该类包含至少一个标有 Test、TestCase 或 TestCaseSource 属性的方法，它就会被视为TestFixture。

```csharp
using NUnit.Framework;

namespace MyTest;

// [TestFixture] // 2.5 版本以后，可选
public class FirstTest
{

    [Test]
    public void Test1()
    {
        Console.WriteLine("test1,hello");
    }
    
}
```

总结：在2.5版本的以前的nunit需要在测试类前加上 TestFixtureAttribute，2.5版本后可选。

## 六、SetUp 设置

功能：此属性在TestFixture内部使用，以提供在调用每个测试方法之前执行的一组通用函数，用于初始化单元测试的一些数据。（意味着假如在某个方法前加上SetUpAttribute，则在执行测试方法前会先执行这个方法）

SetUp 方法可以是静态方法，也可以是实例方法，您可以在夹具中定义多个方法。通常，多个 SetUp 方法仅在继承层次结构的不同级别定义，如下所述。

如果 SetUp 方法失败或引发异常，则不会执行测试，并报告失败或错误。

实践：运行以下测试，观察测试结果：

```csharp
using NUnit.Framework;

namespace ConsoleApp4
{
    // [TestFixture] // 2.5 版本以后，可选
    public class MyTestUnit
    {
        private string? Name;

        [SetUp]
        public void InitTest()
        {
            Console.WriteLine("初始化单元测试...");
            Name = "爱莉小跟班gaolx";
        }

        private int a = 10;
        [OneTimeSetUp] // 只执行一次
        public void OneTime()
        {
            a++;
            Console.WriteLine("我只执行一次");
        }

        [Test]
        public void Test01()
        {
            Console.WriteLine($"我的名字是{Name}");
            Console.WriteLine($"a的值是：{a}");
        }
    }
}

```

**继承**  

SetUp 属性继承自任何基类。因此，如果基类定义了 SetUp 方法，则会在派生类中的每个测试方法之前调用该方法。

您可以在基类中定义一个 SetUp 方法，在派生类中定义另一个方法。NUnit 将在派生类中调用基类 SetUp 方法之前调用基类 SetUp 方法。

**警告**
如果在派生类中重写了基类 SetUp 方法，则 NUnit 将不会调用基类 SetUp 方法;NUnit 预计不会使用包括隐藏基方法在内的用法。请注意，每种方法可能都有不同的名称;只要两者都存在属性，每个属性都将以正确的顺序调用。[SetUp]

**笔记**  

1. 尽管可以在同一类中定义多个 SetUp 方法，但您很少应该这样做。与在继承层次结构中的单独类中定义的方法不同，不能保证它们的执行顺序。
2. 在 .NET 4.0 或更高版本下运行时，如有必要，可以指定异步方法（c# 中的关键字）。
