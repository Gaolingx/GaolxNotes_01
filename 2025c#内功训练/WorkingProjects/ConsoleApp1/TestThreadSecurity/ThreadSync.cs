using NUnit.Framework;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace TestThreadSecurity
{
    internal class ThreadSync
    {
        [Test]
        public void TestSemaphore()
        {
            var inputs = Enumerable.Range(1, 20).ToArray();
            var semaphore = new Semaphore(3, 3); //initialCount: 初始可用资源数 maximumCount: 最大并发资源数

            int HeavyJob(int input)
            {
                semaphore.WaitOne(); //每次最多有 3 个线程同时执行，其他线程等待
                Thread.Sleep(300);
                semaphore.Release();
                return input;
            }

            var sw = Stopwatch.StartNew();
            var plinqOutputs = inputs.AsParallel().Select(HeavyJob).ToArray();
            sw.Stop();
            Console.WriteLine($"Outputs: {string.Join(",", plinqOutputs)}");
            Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds}ms");
            semaphore.Dispose();
        }

        [Test]
        public void TestConcurrentQueue()
        {
            var queue = new ConcurrentQueue<int>();

            var producer = new Thread(Producer);
            var consumer1 = new Thread(Consumer);
            var consumer2 = new Thread(Consumer);

            producer.Start();
            consumer1.Start();
            consumer2.Start();

            producer.Join();
            Thread.Sleep(100); // Wait for consumers to finish

            consumer1.Interrupt();
            consumer2.Interrupt();
            consumer1.Join();
            consumer2.Join();

            void Producer()
            {
                for (int i = 0; i < 20; i++)
                {
                    Thread.Sleep(20);
                    queue.Enqueue(i);
                }
            }

            void Consumer()
            {
                try
                {
                    while (true)
                    {
                        if (queue.TryDequeue(out var res))
                            Console.WriteLine(res);
                        Thread.Sleep(1);
                    }
                }
                catch (ThreadInterruptedException)
                {
                    Console.WriteLine("Thread interrupted.");
                }
            }
        }
    }
}
