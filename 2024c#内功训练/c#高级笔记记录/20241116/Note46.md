# C#高级编程之——委托、事件与观察者模式

## Observer 设计模式

**Observer**：监视者，它监视Subject，当Subject 中的某件事发生的时候，会告知Observer，而Observer 则会采取相应的行动。

**Observer（观察者模式）设计模式**：Observer 设计模式是为了定义对象间的一种一对多的依赖关系，以便于当一个对象的状态改变时，其他依赖于它的对象会被自动告知并更新。

在观察者模式中发生改变的对象称为“观察目标”，而被通知的对象称为“观察者”，一个观察目标可以应对多个观察者，而且这些观察者之间可以没有任何相互联系，可以根据需要增加和删除观察者，使得系统更易于扩展。因此Observer 模式是一种松耦合的设计模式。

**使用场景**：发布-订阅(Publish/Subscribe)模式，模型-视图(Model-View)模式、源-监听(Source-Listener) 模式等。

## 委托

委托是一种定义方法签名的类型，它可以将方法作为参数进行传递。在生活中，委托可以理解为一种“代理”关系，即一个人（或实体）代表另一个人（或实体）执行某项任务。

**生活实例**：

1. **快递送货**：当你工作繁忙无法亲自取快递时，你可以委托快递员帮你将包裹送到指定地点。这里，你（委托人）和快递员（代理人）之间就形成了一种委托关系。
2. **代购**：你委托朋友帮你购买某个商品，朋友按照你的要求去执行任务，这也是一种委托关系。

**概念：**

delegate是C#中的一种**类型**，它实际上是一个能够持有对某个方法的引用的类。与其它的类不同，delegate类能够拥有一个签名（signature），并且它"只能持有与它的签名相匹配的方法的引用"。它所实现的功能与C/C++中的函数指针十分相似。它允许你传递一个类A的方法m给另一个类B的对象，使得类B的对象能够调用这个方法m。但与函数指针相比，delegate有许多函数委托和事件在 .Net Framework中的应用非常广泛指针不具备的优点。首先，函数指针只能指向静态函数，而delegate既可以引用静态函数，又可以引用非静态成员函数。在引用非静态成员函数时，delegate不但保存了对此函数入口指针的引用，而且还保存了调用此函数的类实例的引用。其次，与函数指针相比，delegate是面向对象、类型安全、可靠的受控（managed）对象。也就是说，runtime能够保证delegate指向一个有效的方法，你无须担心delegate会指向无效地址或者越界地址。

在c#中，委托也就是delegate是一个引用类型，他相当于一个装着方法的容器，他可以把方法作为对象进行传递，但前提是委托和对应传递方法的签名得是相同的，签名指的是他们的参数类型和返回值类型

**总结：**

1. 可以简单理解为c++中的函数指针，但是有诸多优点。
2. 委托的签名与方法的签名要保持一致。(参数列表和返回值统称为方法签名)

**应用场景：**

1. 窗体应用程序中的按钮点击事件处理：可以使用委托来定义按钮点击事件的处理方法，并将其与按钮的点击事件关联起来。
2. 发布-订阅模式的实现：通过定义事件和委托来实现发布-订阅模式，其中一个对象可以发布事件，而其他对象可以订阅该事件并在事件发生时执行相应的操作。
3. 回调函数：可以使用委托来定义回调函数，以便在某个操作完成时通知调用方。
4. 多线程编程中的异步操作：可以使用委托和事件来实现异步操作，例如在后台线程执行某个长时间运行的任务，并在任务完成时触发事件通知主线程。

### 语法格式

```csharp
// Delegate
<访问修饰符> delegate 返回值 委托名称(<参数列表>)
```

**例如**

```csharp
//格式
//<访问修饰符> delegate 返回值 委托名称(<参数列表>) 
//声明一个带有0个参数，无返回值的委托
public delegate void DoSpeak();
//声明一个带有1个参数，无返回值的委托
public delegate void DoSpeak2(bool val);
```

### 使用

```csharp
using NUnit.Framework;

namespace StudyDelegate
{
    internal class TestDelegate
    {
        // <访问修饰符> delegate 返回值 委托名称(<参数列表>)
        public delegate void DoSpeak(); //无参无返回值的委托
        public delegate string DoSpeak2(); //无参有返回值的委托
        public delegate string DoSpeak3(int type); //有参有返回值的委托

        // 委托执行的行为
        private void SpeakChinese()
        {
            Console.WriteLine("我在说中文");
        }

        [Test]
        public void TestSpeakDelegate()
        {
            // 创建一个委托实例，实例化时需要传入行为
            DoSpeak speak = new DoSpeak(SpeakChinese); // 委托可以将方法名当作参数，委托本质上是方法的容器
            // 执行委托
            speak();
            Console.WriteLine("=============");
            // 直接调用方法
            SpeakChinese();
            Console.WriteLine("=============");

            //委托的第二种写法
            DoSpeak speak1 = SpeakChinese;
            speak1();
            Console.WriteLine("=============");
            //委托的第二种调用方式
            speak1?.Invoke();
            Console.WriteLine("=============");
        }

        private string SpeakEnglish()
        {
            return "我在说英文";
        }

        [Test]
        public void TestSpeakDelegate2()
        {
            // 注：如果方法签名与委托不一致，则无法调用
            DoSpeak2 speak = new DoSpeak2(SpeakEnglish);
            string str = speak();
            Console.WriteLine(str);

            Console.WriteLine("=============");
            string? str2 = speak?.Invoke(); //Invoke的返回值类型与委托返回值类型相同
            Console.WriteLine(str2);
        }

        private string SpeakLanguages(int type)
        {
            return type switch
            {
                0 => "我在说中文",
                1 => "我在说英文",
                _ => "参数无效",
            };
        }

        [Test]
        public void TestSpeakDelegate3()
        {
            DoSpeak3 speak3 = new DoSpeak3(SpeakLanguages);
            string? str = speak3?.Invoke(0);
            string? str2 = speak3?.Invoke(1);
            string? str3 = speak3?.Invoke(2);

            Console.WriteLine(str);
            Console.WriteLine(str2);
            Console.WriteLine(str3);
        }
    }
}

```

运行结果如下：
