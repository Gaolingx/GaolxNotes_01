
using System.Reflection;

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

    public void TestGetAllField()
    {
        Type type01 = typeof(StudentInfo);
        var fieldInfos = type01.GetFields(BindingFlags.Instance|BindingFlags.NonPublic); //fieldInfos为FieldInfo[]，| 是位运算符，表示并且的意思
    }

    static void Main()
    {
        TestGetPropertyByName("Age");
    }

    
}