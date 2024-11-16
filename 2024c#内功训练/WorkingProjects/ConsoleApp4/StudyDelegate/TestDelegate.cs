using NUnit.Framework;

namespace StudyDelegate
{
    internal class TestDelegate
    {
        // <访问修饰符> delegate 返回值 委托名称(<参数列表>)
        public delegate void DoSpeak(); //无参无返回值的委托
        public delegate string DoSpeak2(); //无参有返回值的委托
        public delegate string DoSpeak3(int type); //有参有返回值的委托

        // 委托执行的行为
        private void SpeakChinese()
        {
            Console.WriteLine("我在说中文");
        }

        [Test]
        public void TestSpeakDelegate()
        {
            // 创建一个委托实例，实例化时需要传入行为
            DoSpeak speak = new DoSpeak(SpeakChinese); // 委托可以将方法名当作参数，委托本质上是方法的容器
            // 执行委托
            speak();
            Console.WriteLine("=============");
            // 直接调用方法
            SpeakChinese();
            Console.WriteLine("=============");

            //委托的第二种写法
            DoSpeak speak1 = SpeakChinese;
            speak1();
            Console.WriteLine("=============");
            //委托的第二种调用方式
            speak1?.Invoke();
            Console.WriteLine("=============");
        }

        private string SpeakEnglish()
        {
            return "我在说英文";
        }

        [Test]
        public void TestSpeakDelegate2()
        {
            // 注：如果方法签名与委托不一致，则无法调用
            DoSpeak2 speak = new DoSpeak2(SpeakEnglish);
            string str = speak();
            Console.WriteLine(str);

            Console.WriteLine("=============");
            string? str2 = speak?.Invoke(); //Invoke的返回值类型与委托返回值类型相同
            Console.WriteLine(str2);
        }

        private string SpeakLanguages(int type)
        {
            return type switch
            {
                0 => "我在说中文",
                1 => "我在说英文",
                _ => "参数无效",
            };
        }

        [Test]
        public void TestSpeakDelegate3()
        {
            DoSpeak3 speak3 = new DoSpeak3(SpeakLanguages);
            string? str = speak3?.Invoke(0);
            string? str2 = speak3?.Invoke(1);
            string? str3 = speak3?.Invoke(2);

            Console.WriteLine(str);
            Console.WriteLine(str2);
            Console.WriteLine(str3);
        }
    }
}
