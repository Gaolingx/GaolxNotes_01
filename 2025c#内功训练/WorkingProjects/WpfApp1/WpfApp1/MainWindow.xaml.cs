using System.Windows;

namespace WpfApp1
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                button.IsEnabled = false;
                progressBar2.Visibility = Visibility.Visible;
                //更新progressBar的回调
                var progress = new Progress<double>(value => progressBar.Value = value);
                var progress2 = new MyProgress<double>(value => progressBar2.Value = value, () => progressBar2.Visibility
                     = Visibility.Hidden, 100);
                await DoJobAsync(progress, progress2);
                button.IsEnabled = true;
            }
            catch (Exception)
            {

            }
        }

        private async Task DoJobAsync(IProgress<double> progress, IProgress<double> progress1)
        {
            for (int i = 1; i <= 100; i++)
            {
                await Task.Delay(50);
                //progressBar.Value = i;
                //Dispatcher.Invoke(() => progressBar.Value = i);
                progress.Report(i);
                progress1.Report(i);
            }
        }
    }

    internal class MyProgress<T> : Progress<T> where T : notnull
    {
        private readonly Action? _complete;
        private readonly T _maximum;
        private bool _isCompleted;

        public MyProgress(Action<T> handler, Action? complete, T maximum)
            : base(handler)
        {
            _complete = complete;
            _maximum = maximum;

            ProgressChanged += CheckCompletion;
        }

        protected override void OnReport(T value)
        {
            if (_isCompleted)
                return;
            base.OnReport(value);
        }

        private void CheckCompletion(object? sender, T e)
        {
            if (e.Equals(_maximum) && !_isCompleted)
            {
                _isCompleted = true;
                _complete?.Invoke();
            }
        }
    }
}