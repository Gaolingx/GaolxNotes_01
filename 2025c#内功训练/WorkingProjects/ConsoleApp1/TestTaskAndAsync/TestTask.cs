using NUnit.Framework;
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
