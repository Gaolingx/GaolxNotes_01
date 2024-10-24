

using NUnit.Framework;

namespace TestFileStream
{
    [TestFixture]
    public class TestClass
    {
        // 1. 添加文件路径
        [Test]
        public void TestAppendLines01()
        {
            File.AppendAllLines("D://App/logs/log.txt", new List<string> {$"Log第一行,Time:{DateTime.Now}",
            $"Log第二行,Time:{DateTime.Now}"});

            //注:1. 路径不存在会抛出异常，但是文件不存在会自动新建
        }
    }
}
