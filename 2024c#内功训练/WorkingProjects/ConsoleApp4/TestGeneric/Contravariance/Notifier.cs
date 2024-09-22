using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Contravariance
{
    public class Notifier<T> : INotifier<T> where T : INotification
    {
        public void Notify(T notification)
        {
            Console.WriteLine(notification.Message);
        }
    }
}
