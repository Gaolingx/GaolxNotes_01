
using NUnit.Framework;
using System.Collections;
using System.Diagnostics;

namespace TestGenericCollection
{
    public class TestClass1
    {
        [Test]
        public void Test1()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            ArrayList arrayList = new();
            for (int i = 0; i < 2000000; i++)
            {
                arrayList.Add(i); // 装箱
            }

            long sum = 0;
            foreach (var item in arrayList)
            {
                sum += Convert.ToInt64(item);
            }
            watch.Stop();
            Console.WriteLine("非泛型集合耗时(ms)：" + watch.ElapsedMilliseconds);
        }

        [Test]
        public void Test2()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            var arrayList = new List<int>();
            for (int i = 0; i < 2000000; i++)
            {
                arrayList.Add(i);
            }

            long sum = 0;
            foreach (var item in arrayList)
            {
                sum += Convert.ToInt64(item);
            }
            watch.Stop();
            Console.WriteLine("泛型集合耗时(ms)：" + watch.ElapsedMilliseconds);
        }

        [Test]
        public void Test3()
        {
            List<int> list = new List<int>() { 2, 3, 7, 5 }; // 集合初始化器
            Console.WriteLine($"集合元素个数:{list.Count},容量:{list.Capacity}");
            list.Add(1);
            Console.WriteLine($"集合元素个数:{list.Count},容量:{list.Capacity}");
        }

        [Test]
        public void TestStack1()
        {
            Stack<int> ints =  new Stack<int>(4); //初始容量4，扩容速度2

            ints.Push(100); //stack添加一个元素，压栈
            ints.Push(101);
            ints.Push(102);

            foreach (var item in ints)
            {
                Console.WriteLine(item);
            }

            var peek = ints.Peek();
            Console.WriteLine(peek);

            var peek2 = ints.Peek();
            Console.WriteLine(peek2); //从stack中获取元素，返回最顶端的元素，但不移除

            var pop = ints.Pop();
            Console.WriteLine(pop); //出栈，移除一个元素并返回

            //var peek3 = ints.TryPeek(out _); //弃元
            int result = 0;
            var peek3 = ints.TryPeek(out result);
            Console.WriteLine(result); //result是被取出的元素的值

            var pop2 = ints.Pop();
            Console.WriteLine(pop2); //出栈，移除元素

            int result2 = 0;
            var pop3 = ints.TryPop(out result2);
            Console.WriteLine(pop3); //pop3返回的是操作是否成功
            Console.WriteLine(result2); //result是被移除的元素的值

            Console.WriteLine(ints.Count);
        }
    }
}
