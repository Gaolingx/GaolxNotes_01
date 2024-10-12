
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
    }
}
