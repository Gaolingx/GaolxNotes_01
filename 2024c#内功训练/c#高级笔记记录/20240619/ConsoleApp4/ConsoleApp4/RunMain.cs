using System;
using System.ComponentModel;
using System.Reflection;
using Info.Main;
using ConsoleApp4.Utils;
using ConsoleApp4;
using TestGetDescription;
using TestJsonAndCSharp;

class RunMain
{
    public static void TestGetType()
    {
        //通过 typeof 运算符获取Type类型
        Type type01 = typeof(StudentInfo);

        //通过实例对象获取类型
        StudentInfo studentInfo = new StudentInfo(); //创建对象
        Type type02 = studentInfo.GetType(); //GetType 是Object 这个类的方法。
                                             //由于所有类型都继承自System.Object，所以所有的类型都含有GetType 方法
    }

    //获取属性
    public static void TestGetAllProperty()
    {
        //获取类型
        Type type01 = typeof(StudentInfo);
        //通过类型获取其中的属性
        PropertyInfo[] propList = type01.GetProperties(); //注：GetProperties方法有多个重载，默认获取public属性
        //或 var propList = typeof(StudentInfo).GetProperties();

        //打印PropertyInfo（属性）
        foreach (var propertyInfo in propList) //propertyInfo也是一个实例
        {
            Console.WriteLine($"{nameof(type01)}类型中属性的名称:{propertyInfo.Name},类型:{propertyInfo.PropertyType}");
        }
    }

    public static void TestGetPropertyByName(string name)
    {
        //获取类型
        Type type01 = typeof(StudentInfo);
        var propInfo = type01.GetProperty(name); //获取type01类型中含有Age属性的名称的方法
        Console.WriteLine($"{nameof(type01)}类型中属性的名称:{propInfo?.Name},类型:{propInfo?.PropertyType}");

    }

    public static void TestGetAllField()
    {
        Type type01 = typeof(StudentInfo);
        //获取所有私有属性
        var fieldInfos = type01.GetFields(BindingFlags.Instance | BindingFlags.NonPublic); //fieldInfos为FieldInfo[]，| 是位运算符，表示并且的意思

        foreach (var fieldInfo in fieldInfos)
        {
            Console.WriteLine($"{nameof(type01)}类型中私有字段的名称:{fieldInfo?.Name},类型:{fieldInfo?.FieldType}");
        }
    }

    //获取字段
    public static void TestGetFieldByName(string name)
    {
        Type type01 = typeof(StudentInfo);
        var fieldInfo = type01.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);

        Console.WriteLine($"{nameof(type01)}类型中字段的名称:{fieldInfo?.Name},类型:{fieldInfo?.FieldType}"); //?表示可空
    }

    //获取类的信息
    public static void TestGetFullName()
    {
        Type type01 = typeof(StudentInfo);

        Console.WriteLine($"它的全称是：{type01.FullName}"); //命名空间+类名

        Console.WriteLine($"它的基类是：{type01.BaseType}");

        Console.WriteLine($"它的类名是：{type01.Name}");

        Console.WriteLine($"它的命名空间是：{type01.Namespace}");
    }

    //创建对象
    public static void TestCreateInstance()
    {
        Type type01 = typeof(StudentInfo);
        // CreateInstance底层其实是调用了 无参构造方法。
        // 对象创建的唯一途径只能是构造方法被调用
        var instance = Activator.CreateInstance(type01) as StudentInfo;
        instance.Age = 19;
        instance.Name = "爱莉小跟班gaolx";
        Console.WriteLine($"我的年龄是{instance?.Age},名字是{instance?.Name}");
    }

    public static void TestCreateInstance02()
    {
        Type type01 = typeof(StudentInfo);
        // 再次验证 CreateInstance 其实是调用了构造方法
        var instance = Activator.CreateInstance(type01, "爱莉小跟班gaolx", 19) as StudentInfo;
        Console.WriteLine($"我的年龄是{instance?.Age},名字是{instance?.Name}");
    }

    //操作属性
    public static void TestOperationProp01()
    {
        /*
         * 操作属性（常规操作）：
         * StudentInfo obj = new();
         * obj.Age = 19; // 对象.属性名 = 属性值
         * var age = obj.Age; // var 变量名 = 对象.属性名
         */

        var tp = typeof(StudentInfo);
        // 获取属性
        var propInfo = tp.GetProperty("Name");

        // 创建对象
        var instance = Activator.CreateInstance(tp);
        // 为属性赋值
        propInfo.SetValue(instance, "爱莉小跟班gaolx");

        // 获取属性的值
        var name = propInfo.GetValue(instance);
        Console.WriteLine($"我的名字是{name}");
    }

    //操作字段
    public static void TestOperationField01()
    {
        var tp = typeof(StudentInfo);
        var field = tp.GetField("_studentId", BindingFlags.Instance | BindingFlags.NonPublic); //获取私有字段需要GetField的一个重载，BindingFlags指定搜索的访问级别

        var instance = Activator.CreateInstance(tp);

        field?.SetValue(instance, "19"); // 字段赋值

        var obj = field?.GetValue(instance); //获取字段的值
        Console.WriteLine($"我的年龄是{obj}");
    }

    //加载程序集
    public static void TestGetAssembly01()
    {
        var assembly = Assembly.Load("ConsoleApp4");
    }

    public static void TestGetAssembly02()
    {
        var assembly = Assembly.LoadFile(@"F:\GitHub\GaolxNotes_01\2024c#内功训练\c#高级笔记记录\20240619\ConsoleApp4\ConsoleApp4\bin\Debug\net8.0\ConsoleApp4.dll");
    }

    public static void TestGetAssembly03()
    {
        var assembly = Assembly.LoadFile(@"F:\GitHub\GaolxNotes_01\2024c#内功训练\c#高级笔记记录\20240619\ConsoleApp4\TestClassLibrary1\bin\Debug\netstandard2.1\TestClassLibrary1.dll");
        var instance = assembly.CreateInstance("TestClassLibrary1.Class1", false); //第二个参数表示忽略大小写

        if (instance != null)
        {
            Console.WriteLine($"{nameof(instance)} has been created.");
        }

    }

    //操作构造方法（有参）
    public static void TestConstructor01()
    {
        Type type01 = typeof(StudentInfo);
        //获取无参构造方法
        var constructor = type01.GetConstructor(new Type[] { });
        //创建对象 方法1
        var obj = constructor?.Invoke(null); //无参构造，所以参数为空

        //创建对象 方法2
        var obj2 = Activator.CreateInstance(type01);

    }

    //操作构造方法（无参）
    public static void TestConstructor02()
    {
        Type type01 = typeof(StudentInfo);
        //获取有参构造方法
        var constructor = type01.GetConstructor(new Type[] { typeof(string), typeof(int) }); //需要指定构造方法参数的类型
        //创建对象 方法1
        StudentInfo obj = constructor?.Invoke(new object?[] { "爱莉小跟班gaolx", 18 }) as StudentInfo; //指定参数的值

        //创建对象 方法2
        StudentInfo obj2 = Activator.CreateInstance(type01, "爱莉大跟班gaolx", 19) as StudentInfo;

        Console.WriteLine($"{nameof(obj)} Name is {obj?.Name}");
        Console.WriteLine($"{nameof(obj2)} Name is {obj2?.Name}");
    }

    //获取所有方法
    public static void TestGetAllMethod01()
    {
        Type type01 = typeof(StudentInfo);
        var methodInfos = type01.GetMethods();

        int i = 1;
        foreach (var m in methodInfos)
        {
            Console.WriteLine($"{i++}. {nameof(StudentInfo)} 方法中名称：{m?.Name},返回值类型：{m?.ReturnType}");
        }
    }

    public static void TestGetAllMethod02()
    {
        Type type01 = typeof(StudentInfo);
        var methodInfos = type01.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

        int i = 1;
        foreach (var m in methodInfos)
        {
            Console.WriteLine($"{i++}. {nameof(StudentInfo)} 方法中名称：{m?.Name},返回值类型：{m?.ReturnType}");
        }
    }

    //获取并调用单个方法
    //无参数，公有方法
    public static void TestGetMethod01()
    {
        Type type01 = typeof(StudentInfo);

        var methodInfo = type01.GetMethod("Run");
        //调用方法
        // 常规操作：对象.方法名();

        var stu = Activator.CreateInstance(type01); //创建对象
        methodInfo?.Invoke(stu, null); //默认返回值为object类型
    }

    //带参数，公有方法
    public static void TestGetMethod02()
    {
        Type type01 = typeof(StudentInfo);

        var methodInfo = type01.GetMethod("Run2");
        //调用方法
        // 常规操作：对象.方法名();

        var stu = Activator.CreateInstance(type01); //此处的参数指的是Run2方法的参数age
        methodInfo?.Invoke(stu, new object?[] { 19 }); //默认返回值为object类型
    }

    //带参数，有返回值，私有方法
    public static void TestGetMethod03()
    {
        Type type01 = typeof(StudentInfo);

        var methodInfo = type01.GetMethod("Run4", BindingFlags.Instance | BindingFlags.NonPublic); //私有方法
        //调用方法
        // 常规操作：对象.方法名();

        var stu = Activator.CreateInstance(type01); //此处的参数指的是Run2方法的参数age
        var resultObj = methodInfo?.Invoke(stu, new object?[] { "爱莉小跟班gaolx" }); //默认返回值为object类型
        Console.WriteLine(resultObj);
    }

    //操作特性
    //获取类的特性
    public static void TestGetClassAttribute01()
    {
        var type01 = typeof(StudentInfo);
        var descAttribute = type01.GetCustomAttribute<DescriptionAttribute>(); //DescriptionAttribute是[Description]的全称
        Console.WriteLine($"{nameof(StudentInfo)}类的描述是{descAttribute?.Description}");
    }

    //拓展方法
    //OrderBy
    public static void TestOrderBy()
    {
        int[] ints = { 10, 45, 15, 39, 21, 26 };
        var result = ints.OrderBy(g => g);
        foreach (var i in result)
        {
            System.Console.Write(i + " ");
        }
    }

    //自定义拓展方法
    public static void TestExtendUtils01()
    {
        string str = "12345";
        var num = str.ParseInt();
        Console.WriteLine($"input:{str},output:{num}");

        string str2 = "12345abc";
        var num2 = str2.ParseInt();
        Console.WriteLine($"input:{str2},output:{num2}");
    }

    //获取属性上的自定义特性
    public static void TestGetCustomAttribute01()
    {
        var type01 = typeof(Product);
        //获取指定属性
        var propInfo = type01.GetProperty("ProductName");
        //获取自定义特性
        var attr = propInfo?.GetCustomAttribute(typeof(MyDescriptionAttribute)) as MyDescriptionAttribute;
        Console.WriteLine($"{nameof(Product)}的描述是:{attr?.Name}");
    }

    public static void TestGetDescription()
    {
        var desc = OrderStateEnum.WaitPay.GetDescription();
        Console.WriteLine($"当前订单状态:{desc}");
    }

    class Site
    {
        public int id;
        public string name;
        public string url;
    }

    public static void TestJsonToCS()
    {
        Site site = new Site { id = 1, name = "米游社", url = "www.miyoushe.com" };
        Site[] sites = { new Site { id = 1, name = "米游社", url = "www.miyoushe.com" },
            new Site { id = 1, name = "崩坏三", url = "www.bh3.com" },
            new Site { id = 1, name = "崩坏：星穹铁道", url = "sr.mihoyo.com" } };

        Site[] value = {
            new Site { id = 1, name = "米游社", url = "www.miyoushe.com" },
            new Site { id = 1, name = "崩坏三", url = "www.bh3.com" },
            new Site { id = 1, name = "崩坏：星穹铁道", url = "sr.mihoyo.com" }
        };
        var obj = new List<Site>(value);
    }


    static void Main()
    {
        JsonTest.TestObjectToJson();
    }


}