
using System.Reflection;

public class RunMain()
{
    public void TestGetType()
    {
        //通过 typeof 运算符获取Type类型
        Type type01 = typeof(StudentInfo);

        //通过实例对象获取类型
        StudentInfo studentInfo = new StudentInfo(); //创建对象
        Type type02 = studentInfo.GetType(); //GetType 是Object 这个类的方法。
                                             //由于所有类型都继承自Object，所以所有的类型都含有GetType 方法
    }

    public void TestGetProperty()
    {
        //获取类型
        Type type01 = typeof(StudentInfo);
        //通过类型获取其中的属性
        PropertyInfo[] propertyInfos01 = type01.GetProperties(); //注：GetProperties方法有多个重载，默认获取public属性
        //打印PropertyInfo（属性）
        foreach(var propertyInfo in propertyInfos01) //propertyInfo也是一个实例
        {
            Console.WriteLine($"StudentInfo类型中属性的名称:{propertyInfo.Name},类型:{propertyInfo.PropertyType}");
        }
    }

    //关于propertyInfo:
    //propertyInfo中CanRead和CanWrite代表读写权限，例如：如果这个字段带有get访问器，则CanRead为true，如果带有set访问器，则CanWrite为true，以此类推。
    //GetMethod 指反编译后这个get属性代表什么方法，即get访问器被编译成的方法名称，SetMethod同理。
    //PropertyType 属性的类型，如int值类型的字段，PropertyType为Int32
    //Attributes/CustomAttributes 特性，后者代表自定义的特性
    //DeclaringType 表示属性属于什么类下的，例如int a和int b属于Class C下，则DeclaringType为C
}