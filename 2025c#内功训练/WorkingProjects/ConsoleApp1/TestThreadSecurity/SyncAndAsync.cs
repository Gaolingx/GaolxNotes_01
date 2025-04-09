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

        /// <summary>
        /// 不安全的Fire-and-forget（一）
        /// </summary>
        /// <returns></returns>
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

        /// <summary>
        /// 不安全的Fire-and-forget（二）
        /// </summary>
        /// <returns></returns>
        [Test]
        public async Task RunGetMessageAsync4()
        {
            Console.WriteLine("Start...");

            try
            {
                VoidFooAsync2(); //无法捕获异常
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

        private async void VoidFooAsync2()
        {
            await Task.Delay(1000);
            throw new Exception("FooAsync2 Error!");
        }
    }

    #region Async Call in Ctor 1
    internal class SyncAndAsync2
    {
        [Test]
        public async Task RunLoadingDataAsync()
        {
            Console.WriteLine("Start...");

            var dataModel = new MyDataModel();
            Console.WriteLine("Loading data...");
            await Task.Delay(2000);
            var data = dataModel.Data;
            Console.WriteLine($"Data is loaded: {dataModel.IsDataLoaded}");

            Console.WriteLine("Done.");
        }
    }
    internal class MyDataModel
    {
        public List<int>? Data { get; private set; }

        public bool IsDataLoaded { get; private set; } = false;

        public MyDataModel()
        {
            //LoadDataAsync(); //直接使用Fire-and-forget方式调用无法处理异常，也无法观察任务状态
            //SafeFireAndForget(LoadDataAsync(), () => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
            SafeFireAndForget(LoadDataAsync2(), () => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
            //LoadDataAsync2().Forget(() => { IsDataLoaded = false; }, ex => { Console.WriteLine($"Error Message: {ex.Message}"); });
        }

        private static async void SafeFireAndForget(Task task, Action? onCompleted = null, Action<Exception>? onError = null)
        {
            try
            {
                await task;
                onCompleted?.Invoke();
            }
            catch (Exception ex)
            {
                onError?.Invoke(ex);
            }
        }

        private async Task LoadDataAsync()
        {
            await Task.Delay(1000);
            Data = Enumerable.Range(1, 10).ToList();
        }

        private async Task LoadDataAsync2()
        {
            await Task.Delay(1000);
            throw new Exception("Failed to load data.");
        }
    }

    static class TaskExtensions
    {
        public static async void Forget(this Task task, Action? onCompleted = null, Action<Exception>? onError = null)
        {
            try
            {
                await task;
                onCompleted?.Invoke();
            }
            catch (Exception ex)
            {
                onError?.Invoke(ex);
            }
        }
    }
    #endregion

    #region Async Call in Ctor 2
    internal class SyncAndAsync3
    {
        [Test]
        public async Task RunLoadingDataAsync2()
        {
            Console.WriteLine("Start...");

            var dataModel = new MyDataModel2();
            Console.WriteLine("Loading data...");
            await Task.Delay(2000);
            var data = dataModel.Data;
            Console.WriteLine($"Data is loaded: {dataModel.IsDataLoaded}");

            Console.WriteLine("Done.");
        }
    }
    internal class MyDataModel2
    {
        public List<int>? Data { get; private set; }

        public bool IsDataLoaded { get; private set; } = false;

        public MyDataModel2()
        {
            //LoadDataAsync(); //直接使用Fire-and-forget方式调用无法处理异常，也无法观察任务状态
            //LoadDataAsync().ContinueWith(t => { OnDataLoaded(t); }, TaskContinuationOptions.None);
            LoadDataAsync2().ContinueWith(t => { OnDataLoaded(t); }, TaskContinuationOptions.None);
        }

        private bool OnDataLoaded(Task task)
        {
            if (task.IsFaulted)
            {
                Console.WriteLine($"Error: {task.Exception.InnerException?.Message}");
                return false;
            }
            return true;
        }

        private async Task LoadDataAsync()
        {
            await Task.Delay(1000);
            Data = Enumerable.Range(1, 10).ToList();
        }

        private async Task LoadDataAsync2()
        {
            await Task.Delay(1000);
            throw new Exception("Failed to load data.");
        }
    }
    #endregion

    #region Fire-and-forget Later
    internal class SyncAndAsync4
    {
        [Test]
        public async Task RunLoadingDataAsync3()
        {
            Console.WriteLine("Start...");

            var dataModel = new MyDataModel3();
            Console.WriteLine("Loading data...");
            await Task.Delay(2000);
            await dataModel.DisplayDataAsync();

            Console.WriteLine("Done.");
        }
    }
    internal class MyDataModel3
    {
        public List<int>? Data { get; private set; }

        public bool IsDataLoaded { get; private set; } = false;

        private readonly Task loadDataTask;

        public MyDataModel3()
        {
            // 把异步任务存储成类中的一个私有字段，任务状态（是否失败、完成等）可被观察
            loadDataTask = LoadDataAsync();
        }

        public async Task DisplayDataAsync()
        {
            await loadDataTask;
            if (Data != null)
            {
                foreach (var data in Data)
                {
                    Console.WriteLine(data);
                }
            }
        }

        private async Task LoadDataAsync()
        {
            await Task.Delay(1000);
            Data = Enumerable.Range(1, 10).ToList();
        }
    }
    #endregion

    #region Async Factory
    internal class SyncAndAsync5
    {
        [Test]
        public async Task RunMyService()
        {
            var service = await MyService.CreateMyService();
            Console.WriteLine("MyService is created.");
        }
    }
    internal class MyService
    {
        private MyService()
        {
            Console.WriteLine("This is MyService.");
        }

        private async Task InitSvcAsync()
        {
            Console.WriteLine("Init MyService...");
            await Task.Delay(1000);
        }

        // 异步工厂方法，在这里调用初始化的构造函数，但无法注册给ioc容器，且对单例模式不友好。
        public static async Task<MyService> CreateMyService()
        {
            var service = new MyService();
            await service.InitSvcAsync();
            return service;
        }
    }

    #endregion
}
