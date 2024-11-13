using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.IO;
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

        // MemoryStream:内存流
        // 特点：它不依赖于其他的资源（如具体的文件）
        [Test]
        public void TestMemoryStream01()
        {
            using Stream fStream = new FileStream("D://App/logs/log.txt", FileMode.Open, FileAccess.Read);

            //将文件流中的内容转移到内存中
            //1. 一次性全部转移到内存流中，小文件建议使用（如果文件过大，可能占用过多资源）
            using MemoryStream mStream = new MemoryStream();
            fStream.CopyTo(mStream);

            Console.WriteLine(mStream.Length);
            Console.WriteLine("================");

            //2.使用使用指定的缓冲区大小读取流
            using Stream fStream2 = new FileStream("D://App/logs/log.txt", FileMode.Open, FileAccess.Read);
            using MemoryStream mStream2 = new MemoryStream();
            int bufferSize = 100; //每次只读100bytes
            byte[] buffer = new byte[bufferSize]; //缓冲区

            while (fStream2.Position < fStream2.Length) // Position:文件流当前的位置
            {
                int count =       //返回值(count):每次实际读取的长度 
                    fStream2.Read(
                    buffer,       //接收文件流中读取的内容
                    0,            //每次从第0个位置开始写入到buffer中
                    bufferSize    //每次读取的长度
                    );

                //写入内存流（把缓冲区中的内容写入到内存流中）
                mStream2.Write(buffer, 0, count); //必须指定count，否则可能出现意料之外的错误
            }

            Console.WriteLine(mStream2.Length);
        }
    }
}
