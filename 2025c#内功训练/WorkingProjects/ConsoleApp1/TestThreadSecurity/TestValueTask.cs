using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class TestValueTask
    {
        [Test]
        public async Task RunTask()
        {
            var service = new MyServices();
            var result = await service.GetValueAsync(42);
            Console.WriteLine(result);
        }

        [Test]
        public async Task RunTask2()
        {
            var service = new MyServices();
            var result = await service.GetValueAsync2(42);
            Console.WriteLine(result);
        }

        [Test]
        public void RunTask3()
        {
            var service = new MyServices();
            var task = service.GetValueAsync2(42);
            if (task.IsCompleted)
            {
                var result = task.Result;
                Console.WriteLine(result);
            }
        }
    }

    internal class MyServices
    {
        private readonly Dictionary<int, string> _cacheDict;

        public MyServices()
        {
            _cacheDict = new()
            {
                [42] = "hello"
            };
        }

        public async Task<string> GetValueAsync(int id)
        {
            if (_cacheDict.TryGetValue(id, out var result))
            {
                return result;
            }

            await Task.Delay(500);
            return id.ToString();
        }

        public async ValueTask<string> GetValueAsync2(int id)
        {
            if (_cacheDict.TryGetValue(id, out var result))
            {
                return result;
            }

            await Task.Delay(500);
            return id.ToString();
        }
    }
}
