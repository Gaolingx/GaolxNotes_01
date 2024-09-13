
using NUnit.Framework;

namespace TestGeneric
{
    public class MyGenericClass01<T> //<T> 代指泛型，泛指某种类型，类型在编译期间确定，T也可以用其他字符标识
    {
        public T ItemName { get; set; }
        public int Total { get; set; }
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

    // 泛型方法——求和
    public class TestClass02
    {
        public T Sum<T>(T a, T b) //返回值为泛型，泛型方法的<T>中需要指定类型参数
        {
            // dynamic: 它是在程序运行时才知道是什么类型，但会绕过编译时的类型检查
            return (dynamic)a + b; //由于编译器不知道a和b的类型，需要转换成dynamic类型，运行时确定
        }
    }

    public class RunTestClass
    {
        [Test]
        public void RunTest()
        {
            TestClass02 test = new TestClass02();
            test.Sum<int>(1, 2);
            Console.WriteLine($"sum的结果是{test}");
        }
    }

}
