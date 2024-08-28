# C#高级编程之——单元总结与回顾

在上一节教程中我们尝试创建了自定义特性并结合前面所学，通过运行时反射的方式获取这个特性并执行相应行为，今天我们先不学习新的东西，借助几个案例复习这一章节所学。

一、结合反射，枚举，特性，扩展方法，完成以下练习

1. 声明枚举如下：

    ```csharp
    using System.ComponentModel;
    /**
     * 订单状态
     */
    public enum OrderStateEnum
    {
        [Description("待支付")]
        WaitPay,
        [Description("待发货")]
        WaitSend,
        [Description("待收货")]
        WaitReceive,
        [Description("待评论")]
        WaitComment,
        [Description("已完成")]
        Finish,
        [Description("取消订单")]
        Cancel
    }
    ```

2. 要求 封装OrderStateEnum的扩展 方法 GetDescription() ，效果如下:

    ```csharp
    string desc = OrderStateEnum.WaitPay.GetDescription();
    Console.WriteLine(desc); // 输出：待支付
    ```

解析：根据要求我们可以得知，需要通过访问OrderStateEnum中的枚举，通过拓展方法获取Description上的字段的描述（如：WaitPay上的待支付）

操作：

1. 新建一个c#项目，其中新建一个名为OrderStateEnum的类，添加如图代码：
2. 创建拓展方法，在c#项目中新建一个名为EnumExtend的类：

   ```csharp
   using System.ComponentModel;
   using System.Reflection;


    namespace TestGetDescription
    {
        public static class EnumExtend
        {
            /// <summary>
            /// 获取枚举的字段的描述
            /// </summary>
            /// <param name="stateEnum"></param>
            /// <returns></returns>
            public static string GetDescription(this OrderStateEnum stateEnum)
            {
                //通过反射获取
                var stateType = typeof(OrderStateEnum);
                //获取当前被操作的枚举字段
                var field = stateType.GetField(stateEnum.ToString(), BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public); //根据字段名称获取字段对象
                var descriptionAttribute = field?.GetCustomAttribute<DescriptionAttribute>();
                //var descriptionAttribute = field?.GetCustomAttribute(typeof(DescriptionAttribute)) as DescriptionAttribute;
                return descriptionAttribute?.Description ?? string.Empty;
            }
        }
    }
   ```

3. 接下来测试拓展方法，在Main函数内运行如下代码，注意引用命名空间，观察控制台输出，可以看到我们成功获取了OrderStateEnum上enum对应的描述。

   ```csharp
    public static void TestGetDescription()
    {
        var desc = OrderStateEnum.WaitPay.GetDescription();
        Console.WriteLine($"当前订单状态:{desc}");
    }
   ```
