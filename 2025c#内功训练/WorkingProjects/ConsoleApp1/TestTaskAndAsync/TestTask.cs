using NUnit.Framework;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace TestTaskAndAsync
{
    internal class TestTask
    {
        [Test]
        public async Task TestGetThreadId()
        {
            Helper.PrintThreadId("Before");
            await FooAsync();
            Helper.PrintThreadId("After");
        }

        [Test]
        public async Task TestCreateTask()
        {
            Helper.PrintThreadId("Before");
            var result = await Task.Run(() => HeavyJob01());
            Console.WriteLine($"result:{result}");
            Helper.PrintThreadId("After");
        }

        private int HeavyJob01()
        {
            Helper.PrintThreadId();
            Thread.Sleep(5000);
            return 10;
        }

        [Test]
        public async Task TestRunTasksAsync()
        {
            var inputs = Enumerable.Range(1, 10).ToArray();
            var sem = new SemaphoreSlim(2, 2);
            var tasks = inputs.Select(HeavyJob).ToList();

            await Task.WhenAll(tasks);

            var outputs = tasks.Select(x => x.Result).ToArray();

            foreach (var output in outputs)
            {
                Console.WriteLine($"result:{output}");
            }

            async Task<int> HeavyJob(int input)
            {
                await sem.WaitAsync();
                await Task.Delay(1000);
                sem.Release();
                return input * input;
            }
        }

        [Test]
        public async Task TestCancelTask()
        {
            var cts = new CancellationTokenSource();
            var token = cts.Token;
            var sw = Stopwatch.StartNew();

            try
            {
                var cancelTask = Task.Run(async () =>
                {
                    await Task.Delay(2000);
                    cts.Cancel();
                });
                await Task.WhenAll(Task.Delay(5000, token), cancelTask);
            }
            catch (TaskCanceledException ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                cts.Dispose();
            }
            Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
        }

        [Test]
        public async Task TestCancelTask2()
        {
            using (var cts = new CancellationTokenSource())
            {
                var token = cts.Token;
                var sw = Stopwatch.StartNew();

                try
                {
                    var cancelTask = Task.Run(async () =>
                    {
                        await Task.Delay(2000);
                        cts.Cancel();
                    });
                    await Task.WhenAll(Task.Delay(5000, token), cancelTask);
                }
                catch (TaskCanceledException ex)
                {
                    Console.WriteLine(ex.ToString());
                }
                finally
                {
                    cts.Dispose();
                }
                Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
            }
        }

        [Test]
        public async Task TestCancelTask3()
        {
            using (var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3.0)))
            {
                //或者用 cts.CancelAfter(3000);
                var token = cts.Token;
                var sw = Stopwatch.StartNew();

                try
                {
                    await Task.Delay(5000, token);
                }
                catch (TaskCanceledException ex)
                {
                    Console.WriteLine(ex.ToString());
                }
                finally
                {
                    cts.Dispose();
                }
                Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
            }
        }


        private Task FooAsync3(CancellationToken cancellationToken)
        {
            return Task.Run(() =>
            {
                if (cancellationToken.IsCancellationRequested)
                    cancellationToken.ThrowIfCancellationRequested();
                //...
                while (true)
                {
                    if (cancellationToken.IsCancellationRequested)
                        cancellationToken.ThrowIfCancellationRequested();
                    //...
                    Thread.Sleep(1000);
                    Console.WriteLine("Background Task Running...");
                }
            });
        }

        private Task<string> FooAsync4(CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested)
                return Task.FromCanceled<string>(cancellationToken);
            return Task.FromResult("done");
        }

        [Test]
        public async Task ComplexOperationAsync()
        {
            var cts = new CancellationTokenSource();
            cts.Token.Register(() => Console.WriteLine("Task Cancelled 1"));
            cts.Token.Register(() => Console.WriteLine("Task Cancelled 2")); //后注册的先被调用
            var sw = Stopwatch.StartNew();

            try
            {
                var cancelTask = Task.Run(async () =>
                {
                    Console.WriteLine("Background Task Running...");
                    await Task.Delay(2000);
                    cts.Cancel();
                });
                await Task.WhenAll(Task.Delay(5000, cts.Token), cancelTask);
            }
            catch (TaskCanceledException ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                cts.Dispose();
            }
            Console.WriteLine($"Task completed in {sw.ElapsedMilliseconds}ms");
        }

        [Test]
        public void TestTimeoutThreadInterrupt()
        {
            var thread = new Thread(FooSync);
            thread.Start();
            if (!thread.Join(TimeSpan.FromMilliseconds(2000)))
            {
                thread.Interrupt();
            }

            Console.WriteLine("Task Done.");
        }

        private void FooSync()
        {
            try
            {
                Console.WriteLine("Foo start...");
                Thread.Sleep(5000);
                Console.WriteLine("Foo end...");
            }
            catch (ThreadInterruptedException)
            {
                Console.WriteLine("Foo Interrupted");
            }
        }

        async Task FooAsync()
        {
            Helper.PrintThreadId("Before");
            await Task.Delay(1000);
            Helper.PrintThreadId("After");
        }

        class Helper
        {
            private static int index = 1;
            public static void PrintThreadId(string? message = null, [CallerMemberName] string? name = null)
            {
                var title = $"index: {index}, CallerMemberName: {name}";
                if (!string.IsNullOrEmpty(message))
                    title += $" @ {message}";
                Console.WriteLine($"Thread id: {Environment.CurrentManagedThreadId}, {title}");
                Interlocked.Increment(ref index);
            }
        }


    }
}
