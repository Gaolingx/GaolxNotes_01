using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp4
{
    public class Product
    {
        [MyDescription(Name ="商品名称")]
        public string? ProductName { get; set; }
    }
}
