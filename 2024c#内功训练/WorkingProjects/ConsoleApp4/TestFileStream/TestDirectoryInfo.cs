using NUnit.Framework;

namespace TestFileStream
{
    [TestFixture]
    public class TestDirectoryInfo
    {
        // 1. 创建目录
        [Test]
        public void TestCreateDirectory01()
        {
            DirectoryInfo info = Directory.CreateDirectory("D://App/logs3/logs");
        }

        // 2. 删除目录
        [Test]
        public void TestDeleteDirectory01()
        {
            Directory.Delete("D://App/logs3/logs/123456");
        }

        [Test]
        public void TestDeleteDirectory02()
        {
            DirectoryInfo info = new DirectoryInfo("D://App/logs3");
            // 只能删除空目录
            //info.Delete();
            // 删除整个目录，包括子目录及文件
            info.Delete(true);
        }

        // 3. 获取所有目录
        [Test]
        public void TestGetAllDirectory01()
        {
            DirectoryInfo info = new DirectoryInfo("D://App/logs3");
            IEnumerable<DirectoryInfo> dirs = info.EnumerateDirectories(); //只能获取目录

            foreach (DirectoryInfo dir in dirs)
            {
                Console.WriteLine(dir.FullName);
            }
            Console.WriteLine("---------------------");

            // 需要正则匹配
            // 获取所有目录
            IEnumerable<DirectoryInfo> dirs2 = info.EnumerateDirectories("*", SearchOption.AllDirectories);
            foreach (DirectoryInfo dir in dirs2)
            {
                Console.WriteLine(dir.FullName);
            }
        }

        // 4. 获取目录的文件列表
        [Test]
        public void TestGetFiles01()
        {
            DirectoryInfo info = new DirectoryInfo("D://App/logs3");

            //获取当前目录下的所有文件
            var files = info.GetFiles();
            foreach (var file in files)
            {
                Console.WriteLine($"FullName:{file.FullName},Length:{file.Length}");
            }

            //获取当前目录及子目录所有文件
            //*.*:所有文件名以及所有的文件后缀名
            var files2 = info.GetFiles("*.*",SearchOption.AllDirectories);
            foreach (var file in files2)
            {
                Console.WriteLine($"FullName:{file.FullName},Length:{file.Length}");
            }
        }
    }
}
