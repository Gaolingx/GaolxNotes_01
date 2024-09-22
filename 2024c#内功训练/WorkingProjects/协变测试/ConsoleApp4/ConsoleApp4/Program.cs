using System;
using System.Collections.Generic;

class Program
{

    //补充元数据
    internal interface MyInterface1
    {
        void Foo()
        {
            Console.WriteLine("This is Foo 1");
        }
    }

    internal interface MyInterface2 : MyInterface1
    {
        new void Foo() //此处关键字new的作用并不是一个对象，而是覆盖被补充的元数据
                       //注意：这样做会使得被覆盖的元数据丢失，该例子为MyInterface1中的void Foo()
        {
            Console.WriteLine("This is Foo 2");
        }
    }

    internal class TestClass1 : MyInterface2
    {
        public void TestFunc() //是不是觉得很奇怪，为啥这个方法不需要实现接口MyInterface2，也能调用MyInterface2中的Foo方法？
                               //由于Foo在接口中已经带有元数据（通过as协变），所以无需在此实现接口（即实现接口==补充元数据）
        {

        }
    }

    internal class TestClass2
    {
        TestClass1 testClass1 = new TestClass1();

        public void TestFunc()
        {
            testClass1.TestFunc();
            var meta = testClass1 as MyInterface2; //在这里，我用类as接口得到了接口，这是为什么呢？
                                                   //这个是一个向上的协变，目的是得到它上一级的元数据载体
                                                   //如果还是不能理解，就想想子类为了得到它基类的元数据，我们肯定是要向上as协变才能拿到它的元数据
            meta.Foo(); //因为我们已经通过协变拿到了MyInterface2的元数据，现在，我们就拥有了MyInterface2接口中的Foo方法
        }
    }

    static void Main()
    {
        TestClass2 testClass2 = new TestClass2();
        testClass2.TestFunc();
    }
}