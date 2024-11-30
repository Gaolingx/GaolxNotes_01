using NUnit.Framework;
using System.Numerics;

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

        [Test]
        public void TestDelegate4()
        {
            // 1.使用反射查看委托的本质
            Type type = typeof(DoSpeak3);
            Console.WriteLine($"{nameof(DoSpeak3)} is Class:{type.IsClass}, is Sealed:{type.IsSealed}");
        }

        private void SpeakA()
        {
            Console.WriteLine("Speak A");
        }

        private void SpeakB()
        {
            Console.WriteLine("Speak B");
        }

        /// <summary>
        /// 多播委托
        /// </summary>
        [Test]
        public void TestMultiDelegate()
        {
            //注册第一个方法
            DoSpeak speakDel = new DoSpeak(SpeakA);
            //注册第二个方法
            speakDel += SpeakB;

            //触发多个方法，依次调用
            speakDel?.Invoke();
            Console.WriteLine("=============");

            //解绑一个方法
            speakDel -= SpeakA;
            speakDel?.Invoke();
        }

        /// <summary>
        /// 匿名方法
        /// </summary>
        [Test]
        public void TestDelegate5()
        {
            //方法一:匿名方法
            DoSpeak speakDel = delegate { Console.WriteLine("你调用了第一个匿名方法"); };
            //方法二:Lambda表达式（本质还是匿名方法）
            speakDel += () => { Console.WriteLine("你调用了第二个匿名方法"); };
            speakDel?.Invoke();

            DoSpeak3 speak3 = delegate (int val) { return SpeakLanguages(val); };
            string? str = speak3?.Invoke(0);
            Console.WriteLine(str);

            DoSpeak3 speak4 = (int val) => { return SpeakLanguages(val); };
            string? str2 = speak4?.Invoke(1);
            Console.WriteLine(str2);
        }

        /// <summary>
        /// 匿名对象
        /// </summary>
        [Test]
        public void TestObject()
        {
            // 匿名对象：没有名称的对象
            var x = new { Name = "流萤", Age = 20, Like = "机甲" };
            var y = new { Name = "青雀", Age = 18, Like = "摸鱼" };

            // 在当前的作用域下可访问
            Console.WriteLine($"Name;{x.Name},Age:{x.Age},Like:{x.Like}");
            Console.WriteLine($"Name;{y.Name},Age:{y.Age},Like:{y.Like}");
        }

        /// <summary>
        /// 内置委托——Action
        /// </summary>
        [Test]
        public void TestAction()
        {
            // 声明委托（0个参数）
            Action action = () => { Console.WriteLine("这是一个无参无返回值的内置委托"); };
            // 调用委托（方式一）
            action?.Invoke();
            // 调用委托（方式二）
            if (action != null)
            {
                action();
            }

            Console.WriteLine("=============");
            // 声明委托（一个参数）
            Action<int> action1 = (int i) =>
            {
                Console.WriteLine($"这是一个带一个参数无返回值的内置委托,value:{i}");
            };
            // 声明委托（多个参数）
            Action<int, string> action2 = (int i, string j) =>
            {
                Console.WriteLine($"这是一个带一个参数无返回值的内置委托,value:{i},{j}");
            };
            Action<int, int, string> action3 = (int i, int j, string k) =>
            {
                Console.WriteLine($"这是一个带一个参数无返回值的内置委托,i+j={i + j},{k}");
            };
            action1?.Invoke(1);
            action2?.Invoke(20, "爱莉小跟班");
            action3?.Invoke(30, 40, "流萤");

        }

        /// <summary>
        /// 内置委托——Func
        /// </summary>
        [Test]
        public void TestFunc()
        {
            // 无参有返回值
            Func<int> func = () => { return 10; };
            // 方法体内如果只有一行代码，可以简写成如下
            Func<int> func2 = () => 10;

            // 1个参数有返回值
            Func<int, string> func3 = (x1) => { return (x1++).ToString(); };
            // 1个参数有返回值
            Func<int, int, string> func4 = (x1, x2) => { return (x1 + x2).ToString(); };
            // more
            Func<float, float, float, float> CalculateMagnitude = (x, y, z) =>
            {
                return (float)Math.Sqrt(x * x + y * y + z * z);
            };

            // 调用
            string? result = func3?.Invoke(10);
            Console.WriteLine($"func3 result:{result}");
            string? result2 = func4?.Invoke(20, 30);
            Console.WriteLine($"func3 result:{result2}");
            float result3 = CalculateMagnitude.Invoke(3, 4, 5);
            Console.WriteLine($"Vector3(3,4,5) Magnitude:{result3}");
        }
    }
}
