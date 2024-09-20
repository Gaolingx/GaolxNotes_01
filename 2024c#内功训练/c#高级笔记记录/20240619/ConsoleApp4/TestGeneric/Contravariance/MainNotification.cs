using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Contravariance
{
    public class MainNotification : Notification
    {
        public override string Message => "您有一封新的邮件";
    }
}
