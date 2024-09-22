using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Covariance
{
    // 协变泛型接口，必须要在泛型前面加上 out关键字
    public interface IFactory<out T> // out 协变关键字 只能应用于interface
    {
        T Create();
    }
}
