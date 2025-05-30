﻿using Nito.AsyncEx;
using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class AsyncLockAndSemaphore
    {

    }

    //错误的例子：不能在lock中使用await关键字
    /*
    internal class Demo
    {
        private readonly object _lock = new object();

        public async Task DoJobAsync()
        {
            lock (_lock)
            {
                await Task.Delay(1000); //无法保证同一个线程占据或释放一个锁
            }
        }
    }
    */

    // Nito.AsyncEx 实现Lock机制
    internal class Demo2
    {
        private readonly AsyncLock _lock = new AsyncLock();

        private readonly CancellationTokenSource _cts = new CancellationTokenSource();

        public async Task DoJobAsync()
        {
            using (await _lock.LockAsync(_cts.Token))
            {
                // Do Something...
            }
        }
    }

    internal class Demo3
    {
        [Test]
        public async Task RunDemo()
        {
            var mutex = new AsyncLock();
            var start = DateTime.Now;
            var tasks = Enumerable.Range(0, 10).Select(x => ComputeAsync(x, mutex)).ToList();
            var results = await Task.WhenAll(tasks);
            Console.WriteLine(string.Join(", ", results));

            var end = DateTime.Now;
            Console.WriteLine($"Elapsed Time: {(end - start).TotalMilliseconds:F4} ms");
        }


        private async Task<int> ComputeAsync(int x, AsyncLock mutex)
        {
            using (await mutex.LockAsync())
            {
                await Task.Delay(200);
                return x * x;
            }
        }
    }

    internal class Demo4
    {
        [Test]
        public async Task RunDemo()
        {
            var semaphore = new SemaphoreSlim(1);
            var start = DateTime.Now;
            var tasks = Enumerable.Range(0, 10).Select(x => ComputeAsync(x, semaphore)).ToList();
            var results = await Task.WhenAll(tasks);
            Console.WriteLine(string.Join(", ", results));

            var end = DateTime.Now;
            Console.WriteLine($"Elapsed Time: {(end - start).TotalMilliseconds:F4} ms");
        }


        private async Task<int> ComputeAsync(int x, SemaphoreSlim semaphore)
        {
            await semaphore.WaitAsync();
            await Task.Delay(200);
            semaphore.Release();
            return x * x;
        }
    }

    internal class Demo5
    {
        [Test]
        public async Task RunDemo()
        {
            var signal = new AsyncAutoResetEvent(false);

            var setter = Task.Run(() =>
            {
                Thread.Sleep(1000);
                signal.Set();
            });

            var waiter = Task.Run(async () =>
            {
                await signal.WaitAsync();
                Console.WriteLine("Signal received !");
            });

            await Task.WhenAll(setter, waiter);
        }
    }

    internal class Demo6
    {
        [Test]
        public async Task RunDemo()
        {
            var tcs = new TaskCompletionSource();

            var setter = Task.Run(() =>
            {
                Thread.Sleep(1000);
                tcs.SetResult();
            });

            var waiter = Task.Run(async () =>
            {
                await tcs.Task;
                Console.WriteLine("Signal received !");
            });

            await Task.WhenAll(setter, waiter);
        }
    }
}
