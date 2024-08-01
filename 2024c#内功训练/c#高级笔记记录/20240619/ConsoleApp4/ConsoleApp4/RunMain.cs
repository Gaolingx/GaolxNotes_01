using System;
using System.Reflection;
using Info.Main;

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
        var fieldInfos = type01.GetFields(BindingFlags.Instance|BindingFlags.NonPublic); //fieldInfos为FieldInfo[]，| 是位运算符，表示并且的意思

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
        var field = tp.GetField("_studentId",BindingFlags.Instance|BindingFlags.NonPublic); //获取私有字段需要GetField的一个重载，BindingFlags指定搜索的访问级别

        var instance = Activator.CreateInstance(tp);

        field?.SetValue(instance, "19"); // 字段赋值

        var obj = field?.GetValue(instance); //获取字段的值
        Console.WriteLine($"我的年龄是{obj}");
    }

    static void Main()
    {
        TestOperationField01();
    }

    
}