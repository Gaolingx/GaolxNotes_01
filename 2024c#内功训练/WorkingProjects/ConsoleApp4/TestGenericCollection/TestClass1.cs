﻿
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
            Stack<int> ints = new Stack<int>(4); //初始容量4，扩容速度2

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

        [Test]
        public void TestQueue01()
        {
            Queue<int> q = new Queue<int>(4); //初始化一个队列，容量为4

            // 1. 添加元素到队列：Enqueue
            q.Enqueue(100);
            q.Enqueue(200);
            q.Enqueue(300);

            // 2. 从队列中获取一个元素，但是不移除
            var peek = q.Peek();
            Console.WriteLine($"peek: {peek}"); //先进先出
            Console.WriteLine($"peek: {peek}");

            // 3. 遍历队列
            foreach (var item in q)
            {
                Console.WriteLine(item);
            }

            // 4. 从队列中取出元素并移除，返回值为移除的元素
            var result = q.Dequeue();
            var result2 = q.Dequeue();
            var result3 = q.Dequeue();

            Console.WriteLine($"Count:{q.Count}");

            int result4 = 0;
            bool flag = q.TryDequeue(out result4); //out 是操作失败则返回初始值，成功则返回移除元素的值，
                                                   // TryDequeue 为操作是否成功（bool类型）成功为true

            Console.WriteLine($"Success:{flag},result:{result4}");
        }

        [Test]
        public void TestSortedList01()
        {
            // Key(string):科目,Value(int):成绩
            SortedList<string, int> sList = new SortedList<string, int>();

            // 1. 键值对赋值（索引下标）
            sList["语文"] = 90;
            sList["数学"] = 120;

            // 2. Add方法赋值
            sList.Add("英语", 110); //注意类型匹配

            // 3. 遍历集合
            // SortedList按照键的顺序排序
            foreach (var item in sList)
            {
                Console.WriteLine($"科目名字:{item.Key},成绩:{item.Value}");
            }
        }

        [Test]
        public void TestSortedList02()
        {
            // 1. 创建一个键值对都是string 类型的集合
            SortedList<string, string> openWith =
                new SortedList<string, string>();

            // 2. 初始化一些没有重复键的元素，但对应的值，有些元素是重复的
            openWith["语文"] = "120";
            openWith["数学"] = "120";
            openWith.Add("英语", "110");
            openWith.Add("物理", "75");

            // 3. 如果添加一个已经存在的键值对，则会抛出异常（Key不能重复）
            try
            {
                openWith.Add("物理", "80");
            }
            catch (ArgumentException)
            {
                Console.WriteLine("An element with Key = \"物理\" already exists.");
            }

            // 4. 元素的键可作为集合的索引来访问元素（根据Key获取值）
            Console.WriteLine("For key = \"语文\", value = {0}.",
                openWith["语文"]);

            // 5. 通过键索引，可修改其所关联的值
            openWith["数学"] = "135";
            Console.WriteLine("For key = \"数学\", value = {0}.",
                openWith["数学"]);

            // 6. 如果键不存在，则会新增一个键值对数据
            openWith["化学"] = "75";

            // 7. 如果请求的键不存在，则会抛出异常
            try
            {
                Console.WriteLine("For key = \"地理\", value = {0}.",
                    openWith["地理"]);
            }
            catch (KeyNotFoundException)
            {
                Console.WriteLine("Key = \"地理\" is not found.");
            }

            // 8. 当一个程序经常要尝试的键，结果却不是  在列表中，TryGetValue可以是一个更有效的  
            // 获取值的方法。  （返回值类型：bool）
            string value = "";
            if (openWith.TryGetValue("地理", out value))
            {
                Console.WriteLine("For key = \"地理\", value = {0}.", value);
            }
            else
            {
                Console.WriteLine("Key = \"地理\" is not found.");
            }

            // 9. 判断是否包含键
            if (!openWith.ContainsKey("地理"))
            {
                openWith.Add("地理", "90");
                Console.WriteLine("Value added for key = \"地理\": {0}",
                    openWith["地理"]);
            }

            // 10. 遍历循环，元素被检索为KeyValuePair对象
            Console.WriteLine();
            foreach (KeyValuePair<string, string> kvp in openWith)
            {
                Console.WriteLine("Key = {0}, Value = {1}",
                    kvp.Key, kvp.Value);
            }

            // 11. 获取集合中的Values 列表
            IList<string> ilistValues = openWith.Values;

            // 打印出所有的值列表
            Console.WriteLine();
            foreach (string s in ilistValues)
            {
                Console.WriteLine("Value = {0}", s);
            }

            // 通过索引获取值
            Console.WriteLine("\nIndexed retrieval using the Values " +
                "property: Values[2] = {0}", openWith.Values[2]);

            // 获取所有的Key
            IList<string> ilistKeys = openWith.Keys;

            // 12. 打印出所有的键列表
            Console.WriteLine("=========================");
            foreach (string s in ilistKeys)
            {
                Console.WriteLine("Key = {0}", s);
            }

            // 13. 通过索引获取Key
            Console.WriteLine("\nIndexed retrieval using the Keys " +
                "property: Keys[0] = {0}", openWith.Keys[0]);

            // 14. 移除元素（键不存在，不抛异常）
            Console.WriteLine("\nRemove(\"数学\")");
            openWith.Remove("数学");
            openWith.RemoveAt(0); //移除第一个元素

            if (!openWith.ContainsKey("数学"))
            {
                Console.WriteLine("Key \"数学\" is not found.");
            }

            Console.WriteLine("=========================");
            // 输出剩余元素（会自动排序）
            foreach (KeyValuePair<string, string> kvp in openWith)
            {
                Console.WriteLine("Key = {0}, Value = {1}",
                    kvp.Key, kvp.Value);
            }

            // SortedList能排序的本质：实现了ICompare接口
        }
    }
}
