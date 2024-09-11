using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestGeneric
{
    //分页对象
    public class PageModel<T>
    {
        public List<T>? Datas { get; set; }
        public int Total {  get; set; }
    }
}
