using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Contravariance
{
    public interface INotifier<in T> where T : INotification
    {
        void Notify(T notification);
    }
}
