using NUnit.Framework;

namespace TestThreadSecurity
{
    internal class RunSyncOnAsync
    {
        [Test]
        public async Task RunCancelableThreadTask()
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(2000));
            var task = new CancelableThreadTask(() => { LongRunningJob(); });
            try
            {
                await task.RunAsync(cts.Token);
            }
            catch (TaskCanceledException)
            {
                Console.WriteLine("Task was Canceled.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Task failed: {ex.Message}");
            }

            await Task.Delay(6000);
            Console.WriteLine("Finish.");
        }

        private void LongRunningJob()
        {
            Thread.Sleep(5000);
            Console.WriteLine("Long Running job completed");
        }
    }

    internal class CancelableThreadTask
    {
        private Thread? _thread;
        private TaskCompletionSource? _tcs;
        private readonly Action _action;
        private readonly Action<Exception>? _onError;
        private readonly Action? _onCompleted;

        //原子操作，保证线程安全
        private int _isRunning = 0;

        public CancelableThreadTask(Action action, Action<Exception>? onError = null, Action? onCompleted = null)
        {
            ArgumentNullException.ThrowIfNull(action);
            _action = action;
            _onError = onError;
            _onCompleted = onCompleted;
        }

        public Task RunAsync(CancellationToken token)
        {
            if (Interlocked.CompareExchange(ref _isRunning, 1, 0) == 1)
                throw new InvalidOperationException("Task is already running!");

            _tcs = new TaskCompletionSource();

            _thread = new Thread(() =>
            {
                try
                {
                    _action();
                    _tcs.SetResult();
                    _onCompleted?.Invoke();
                }
                catch (Exception ex)
                {
                    if (ex is ThreadInterruptedException)
                    {
                        _tcs.TrySetCanceled();
                    }
                    else
                    {
                        _tcs.TrySetException(ex);
                        _onError?.Invoke(ex);
                    }
                }
                finally
                {
                    Interlocked.Exchange(ref _isRunning, 0);
                }
            });

            token.Register(() =>
            {
                if (Interlocked.CompareExchange(ref _isRunning, 0, 1) == 1)
                    _thread.Interrupt();
            });

            _thread.Start();

            return _tcs.Task;
        }
    }
}
