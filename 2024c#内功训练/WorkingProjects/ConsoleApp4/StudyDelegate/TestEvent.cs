using NUnit.Framework;
using System;
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

        //------------事件------------//
        // 自定义事件数据类
        public class MyEventArgs : EventArgs
        {
            public string Message { get; }

            public MyEventArgs(string message)
            {
                Message = message;
            }
        }

        // 发布者类
        public class Publisher
        {
            // 声明事件
            public event EventHandler<MyEventArgs>? MyEvent;

            // 触发事件的方法
            protected virtual void OnMyEvent(MyEventArgs e)
            {
                MyEvent?.Invoke(this, e);
            }

            // 模拟某个操作触发事件
            public void DoSomething()
            {
                Console.WriteLine("Doing something...");
                OnMyEvent(new MyEventArgs("Hello, this is a custom event!"));
            }
        }

        // 订阅者类
        public class Subscriber
        {
            public void HandleMyEvent(object? sender, MyEventArgs e)
            {
                Console.WriteLine("Received event with message: " + e.Message);
            }
        }

        [Test]
        public void TestEvent03()
        {
            Publisher publisher = new Publisher();
            Subscriber subscriber = new Subscriber();

            // 订阅事件
            publisher.MyEvent += subscriber.HandleMyEvent;

            // 触发事件
            publisher.DoSomething();

            // 取消订阅事件
            publisher.MyEvent -= subscriber.HandleMyEvent;
        }
    }
}
