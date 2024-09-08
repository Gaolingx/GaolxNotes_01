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
