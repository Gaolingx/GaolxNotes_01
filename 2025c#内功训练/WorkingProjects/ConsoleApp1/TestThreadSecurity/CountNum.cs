using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class CountNum
    {
        private const int total = 100_000;
        private int count = 0;
        private readonly object lockObj = new object();

        /// <summary>
        /// 1. 线程不安全的情况
        /// </summary>
        [Test]
        public void Test01()
        {
            count = 0;
            var thread1 = new Thread(Increment);
            var thread2 = new Thread(Increment);

            thread1.Start();
            thread2.Start();

            thread1.Join();
            thread2.Join();

            Console.WriteLine($"1. Count: {count}");

        }

        /// <summary>
        /// 2. 互斥锁
        /// </summary>
        [Test]
        public void Test02()
        {
            count = 0;
            var thread1 = new Thread(Increment2);
            var thread2 = new Thread(Increment2);

            thread1.Start();
            thread2.Start();

            thread1.Join();
            thread2.Join();

            Console.WriteLine($"2. Count: {count}");

        }

        /// <summary>
        /// 3. 原子操作
        /// </summary>
        [Test]
        public void Test03()
        {
            count = 0;
            var thread1 = new Thread(Increment3);
            var thread2 = new Thread(Increment3);

            thread1.Start();
            thread2.Start();

            thread1.Join();
            thread2.Join();

            Console.WriteLine($"3. Count: {count}");

        }

        private void Increment()
        {
            for (int i = 0; i < total; i++)
                count++;
        }
        private void Increment2()
        {
            for (int i = 0; i < total; i++)
                lock (lockObj)
                    count++;
        }
        private void Increment3()
        {
            for (int i = 0; i < total; i++)
                Interlocked.Increment(ref count);
        }
    }
}
