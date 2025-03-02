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
