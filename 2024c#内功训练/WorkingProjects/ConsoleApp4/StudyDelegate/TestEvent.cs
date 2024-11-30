using NUnit.Framework;
using static StudyDelegate.TestDelegate;

namespace StudyDelegate
{
    internal class TestEvent
    {
        public event Action Event1;
        public event Action<string> Event2;

        /// <summary>
        /// 1. 委托的使用
        /// </summary>
        [Test]
        public void TestEvent01()
        {
            // event只能通过 += 方式订阅方法
            //订阅方法
            Event1 += () => { Console.WriteLine("Speak Chinese."); };
            Event1 += () => { Console.WriteLine("Speak English."); };

            //触发事件
            Event1?.Invoke();
        }

        /// <summary>
        /// 2. 反射查看委托本质
        /// </summary>
        [Test]
        public void TestEvent02()
        {
            Event1 += () => { Console.WriteLine("Speak Chinese."); };
            Type type = Event1.GetType();
            Console.WriteLine($"{nameof(Event1)} is Class:{type.IsClass}, is Sealed:{type.IsSealed}");
            Console.WriteLine();
        }
    }
}
