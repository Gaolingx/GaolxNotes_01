namespace TestTaskAndAsync
{
    static class AsyncExtensions
    {
        public static async Task<TResult> TimeoutAfter<TResult>(this Task<TResult> task, TimeSpan timeout)
        {
            using var cts = new CancellationTokenSource();
            var completedTask = await Task.WhenAny(task, Task.Delay(timeout, cts.Token));
            if (completedTask != task)
            {
                cts.Cancel();
                throw new TimeoutException();
            }

            return await task;
        }
    }
}
