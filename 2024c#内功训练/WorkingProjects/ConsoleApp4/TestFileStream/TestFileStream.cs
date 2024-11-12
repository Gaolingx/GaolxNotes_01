using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestFileStream
{
    public class TestFileStream
    {
        [Test]
        public void TestFStream()
        {
            //1. 创建文件流
           using Stream stream = new FileStream("D://App/logs/log.txt",FileMode.Open, FileAccess.Read);
            //2. 读取文件流
            //只能读取流中的字符串（不适合二进制文件）
            StreamReader reader = new StreamReader(stream);

            while(!reader.EndOfStream)
            {
                string? text = reader.ReadLine();
                Console.WriteLine(text);
            }
        }
    }
}
