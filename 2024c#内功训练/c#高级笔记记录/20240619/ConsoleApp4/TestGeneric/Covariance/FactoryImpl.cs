using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric.Covariance
{
    public class FactoryImpl<T> : IFactory<T> where T : new()
    {
        public T Create()
        {
            return new T();
        }
    }
}
