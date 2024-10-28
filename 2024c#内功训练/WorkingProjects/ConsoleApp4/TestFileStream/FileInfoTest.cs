using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestFileStream
{
    [TestFixture]
    public class FileInfoTest
    {
        // 1. 获取文件基本信息
        [Test]
        public void TestInfo01()
        {
            FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
            Console.WriteLine($"FullName:{fileInfo.FullName},Extension:{fileInfo.Extension}," +
                $"ReadOnly:{fileInfo.IsReadOnly},Length:{fileInfo.Length},CreateTime:{fileInfo.CreationTime}," +
                $"LastWriteTime:{fileInfo.LastWriteTime}");
        }

        // 2. 操作文件——追加文本
        [Test]
        public void TestInfo02()
        {
            FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
            using StreamWriter writer = fileInfo.AppendText();
            writer.WriteLine($"AppendText:TestInfo02,Date{DateTime.Now}");
        }

        // 3. 操作文件——打开并读取
        [Test]
        public void TestInfo03()
        {
            // 如果需要对某个文件进行多次操作，建议使用 FileInfo 对象
            // 如果需要获取文件信息，也需要使用 FileInfo
            // 如果只是对文件单次操作，则可以使用 File 静态类
            FileInfo fileInfo = new FileInfo("D://App/logs2/log.txt");
            var stream = fileInfo.OpenRead(); //只读
            StreamReader reader = new StreamReader(stream);
            var str = reader.ReadLine();
            Console.WriteLine(str);
        }
    }
}
