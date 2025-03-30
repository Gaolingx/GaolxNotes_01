using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class SyncAndAsync
    {
        [Test]
        public void RunFooAsync()
        {
            Console.WriteLine("Run FooAsync...");
            // 用阻塞的方式等待任务结束
            FooAsync().Wait();
            Console.WriteLine("Done.");
        }

        [Test]
        public void RunGetMessageAsync()
        {
            Console.WriteLine("Start...");
            // 用阻塞的方式等待任务结束（需要考虑死锁问题）
            var message = GetMessageAsync().Result;
            Console.WriteLine($"message:{message}");

            Console.WriteLine("Done.");
        }

        [Test]
        public void RunGetMessageAsync2()
        {
            Console.WriteLine("Start...");

            try
            {
                FooAsync2().GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
            }

            Console.WriteLine("Done.");
        }

        [Test]
        public async Task RunGetMessageAsync3()
        {
            Console.WriteLine("Start...");

            try
            {
                _ = FooAsync2(); //无法捕获异常
                await Task.Delay(2000);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error Type:{ex.GetType()}, Error Message:{ex.Message}");
            }

            Console.WriteLine("Done.");
        }

        private async Task FooAsync()
        {
            await Task.Delay(1000);
        }

        private async Task<string> GetMessageAsync()
        {
            await Task.Delay(1000);
            return "Hello World!";
        }

        private async Task FooAsync2()
        {
            await Task.Delay(1000);
            throw new Exception("FooAsync2 Error!");
        }
    }
}
