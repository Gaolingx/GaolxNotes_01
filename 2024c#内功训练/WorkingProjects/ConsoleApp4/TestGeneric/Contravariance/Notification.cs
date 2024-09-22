using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Contravariance
{
    // 关于通知接口的抽象实现。
    public abstract class Notification : INotification
    {
        public abstract string Message { get; }
    }
}
