﻿

using NUnit.Framework;

namespace TestFileStream
{
    [TestFixture]
    public class TestClass
    {
        // 1. 向一个文件中追加行
        [Test]
        public void TestAppendAllLines01()
        {
            File.AppendAllLines("D://App/logs/log.txt", new List<string> {$"Log第一行,Time:{DateTime.Now}",
            $"Log第二行,Time:{DateTime.Now}"});

            //注:1. 路径不存在会抛出异常，但是文件不存在会自动新建
        }

        // 2. 向一个文件中追加行（指定编码类型）
        public void TestAppendAllLines02()
        {
            File.AppendAllLines("D://App/logs/log.txt", new List<string> {$"Log第一行,Time:{DateTime.Now}",
            $"Log第二行,Time:{DateTime.Now}"}, System.Text.Encoding.UTF8);

            //注：如果写入文件的内容出现乱码，你可以尝试指定编码类型
        }

        // 3. 向一个文件中追加行（不换行）
        [Test]
        public void TestAppendAllText01()
        {
            File.AppendAllText("D://App/logs/log.txt", $"Log追加的内容1:TestAppendAllText01,Time:{DateTime.Now}", System.Text.Encoding.UTF8);
            File.AppendAllText("D://App/logs/log.txt", $"Log追加的内容2:TestAppendAllText02,Time:{DateTime.Now}", System.Text.Encoding.UTF8);

            //转义字符手动实现换行（\r\n:回车+换行）
            File.AppendAllText("D://App/logs/log.txt", $"Log追加的内容1:TestAppendAllText01,Time:{DateTime.Now}\r\n", System.Text.Encoding.UTF8);
            File.AppendAllText("D://App/logs/log.txt", $"Log追加的内容2:TestAppendAllText02,Time:{DateTime.Now}\r\n", System.Text.Encoding.UTF8);
        }

        // 4. 创建一个StreamWriter，追加文本到文件
        [Test]
        public void TestAppendText01()
        {
            //步骤:1. 创建StreamWriter 流写入对象
            //2. 操作StreamWriter写入文件流
            //3. 使用完后记得关闭(使用using)
            using var streamWriter = File.AppendText("D://App/logs/log.txt");
            streamWriter.WriteLine($"这是一段通过streamWriter写入的文本");
            streamWriter.WriteLine($"Log追加的内容3:TestAppendText01,Time:{DateTime.Now}");
            streamWriter.Flush();
        }

        // 5. 文件复制
        [Test]
        public void TestCopy01()
        {
            // 参数：源文件(string),目标文件(string)
            File.Copy("D://App/logs/log.txt", "D://App/logs2/log.txt");
        }

        // 6. 文件创建
        [Test]
        public void TestCreate01()
        {
            //创建一个文件流，需要手动释放资源
            using var fileStream = File.Create("D://App/logs/log3.txt");
            //向文件流写入内容
            using StreamWriter  streamWriter = new StreamWriter(fileStream);
            streamWriter.WriteLine("这是在TestCreate01 写入的内容");
        }

        // 7. 文件删除
        [Test]
        public void TestDelete01()
        {
            File.Delete("D://App/logs/log3.txt");
        }

        // 8. 文件移动
        [Test]
        public void TestMove01()
        {
            // true:如果存在同名文件则覆盖，默认为false
            File.Move("D://App/logs2/log.txt", "D://App/logs/log233.txt", true);
        }

        // 9. 判断文件是否存在
        [Test]
        public void TestExists()
        {
            string path = "D://App/logs";
            if (File.Exists($"{path}/log233.txt"))
            {
                Console.WriteLine($"{path} 存在同名文件:log233.txt");
            }
            else
            {
                Console.WriteLine($"{path} 不存在同名文件:log233.txt");
            }
        }

        // 10. 打开文件
        [Test]
        public void TestOpen01()
        {
            using var stream = File.Open("D://App/logs2/log.txt",FileMode.Open);
            using StreamWriter streamWriter = new StreamWriter(stream);
            streamWriter.WriteLine($"TestOpen01,Date:{DateTime.Now}");
        }

        // 11. 打开文件
        [Test]
        public void TestOpen02()
        {
            // 如果要对File 执行写入操作，我们需要设置文件的写入权限（FileAccess.Write）
            using var stream = File.Open("D://App/logs2/log.txt", FileMode.Open, FileAccess.Read);
            using StreamReader streamReader = new StreamReader(stream);
            string? text = streamReader.ReadLine();
            Console.WriteLine(text);
        }

        // 12. 打开并读取
        [Test]
        public void TestOpenRead01()
        {
            // OpenRead 只有读取权限
            using var stream = File.OpenRead("D://App/logs2/log.txt");
            using StreamReader streamReader = new StreamReader(stream);
            if (!streamReader.EndOfStream) //判断是否是文件最末尾
            {
                string? text = streamReader.ReadLine();
                Console.WriteLine(text);
            }
        }

        // 13. 打开并写入
        [Test]
        public void TestOpenWrite01()
        {
            // OpenWrite 只有写入权限
            using var stream = File.OpenWrite("D://App/logs2/log.txt");
            using StreamWriter writer = new StreamWriter(stream);
            writer.WriteLine($"TestOpenWrite01,Date:{DateTime.Now}");
        }

        // 14. 读取所有内容
        [Test]
        public void TestReadAllText01()
        {
            var stringArr = File.ReadAllText("D://App/logs2/log.txt", System.Text.Encoding.UTF8);
            Console.WriteLine(stringArr);
            Console.WriteLine(stringArr.Length);
        }
    }
}
