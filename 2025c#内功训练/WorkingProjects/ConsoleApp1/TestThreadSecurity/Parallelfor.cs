using NUnit.Framework;
using System.Diagnostics;

namespace TestThreadSecurity
{
    internal class Parallelfor
    {
        /// <summary>
        /// 1. Sequential
        /// </summary>
        [Test]
        public void TestSequential()
        {
            var inputs = Enumerable.Range(1, 20).ToArray();

            int HeavyJob(int input)
            {
                Thread.Sleep(300);
                return input;
            }

            var sw = Stopwatch.StartNew();

            var forOutputs = new int[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                forOutputs[i] = HeavyJob(inputs[i]);
            }

            Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds}");
            PrintArray(forOutputs);
        }

        /// <summary>
        /// 2. Parallel
        /// </summary>
        [Test]
        public void TestParallel()
        {
            var inputs = Enumerable.Range(1, 20).ToArray();

            int HeavyJob(int input)
            {
                Thread.Sleep(300);
                return input;
            }

            var sw = Stopwatch.StartNew();

            var parallelOutputs = new int[inputs.Length];

            // Parallel.For(开始索引, 结束索引, 每个迭代调用一次的委托);
            Parallel.For(0, inputs.Length, i =>
            {
                parallelOutputs[i] = HeavyJob(inputs[i]);
            });

            Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds}");
            PrintArray(parallelOutputs);
        }

        /// <summary>
        /// 3. PLinq
        /// </summary>
        [Test]
        public void TestPLinq()
        {
            var inputs = Enumerable.Range(1, 20).ToArray();

            int HeavyJob(int input)
            {
                Thread.Sleep(300);
                return input;
            }

            var sw = Stopwatch.StartNew();

            var plinqOutputs = inputs.AsParallel().Select(HeavyJob).ToArray();

            Console.WriteLine($"Elapsed time: {sw.ElapsedMilliseconds}");
            PrintArray(plinqOutputs);
        }

        private void PrintArray<T>(T[] values)
        {
            string output = string.Join(", ", values);
            Console.WriteLine(output);
        }

        /// <summary>
        /// 4. 线程终止
        /// </summary>

        [Test]
        public void TestStopThread()
        {
            var thread = new Thread((object? obj) =>
            {
                DoJobBackground();
            })
            {
                IsBackground = true,
                Priority = ThreadPriority.Normal,
            };

            var thread2 = new Thread((object? obj) =>
            {
                DoJobBackground();
            })
            {
                IsBackground = true,
                Priority = ThreadPriority.Normal,
            };

            thread.Start();
            Console.WriteLine($"Running in main thread, id: {Thread.CurrentThread.ManagedThreadId}");
            thread.Join(); // 会导致主线程阻塞，直到子线程的任务完成
            Console.WriteLine("Job 1 Done!");

            Console.WriteLine("===================");
            thread2.Start();
            Console.WriteLine($"Running in main thread, id: {Thread.CurrentThread.ManagedThreadId}");
            Thread.Sleep(3500);
            thread2.Interrupt(); // 立即结束线程
            thread2.Join();
            Console.WriteLine("Job 2 Done!");
        }

        private void DoJobBackground()
        {
            try
            {
                int threadId = Thread.CurrentThread.ManagedThreadId;
                for (int i = 0; i < 10; i++)
                {
                    Thread.Sleep(1000);
                    Console.WriteLine("Running in sub thread: " + i + ", thread id: " + threadId);
                }
                Console.WriteLine("Sub thread task finished !");
            }
            catch (ThreadInterruptedException)
            {
                Console.WriteLine("Thread has been interrupted !");
            }
        }
    }
}
