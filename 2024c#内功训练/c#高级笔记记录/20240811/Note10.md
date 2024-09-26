# C#高级编程之——反射（十）操作特性

在上一个章节中我们学习了如何通过反射操作方法，今天我们来学习什么是特性以及如何获取自定义特性。

## 详细知识点

1. **C#中的特性：**
   - 定义：特性是用于在运行时传递程序中各种元素（比如类、方法、结构、组件等）的行为信息的声明性标签。
   - 用途：特性可以用于多种目的，如描述程序集、控制编译器的行为、实现自定义的编译时检查、在运行时通过反射查询程序元素的信息等。
   - 声明：特性的声明通过在它所作用的元素上方放置方括号（[]）并将特性名称括起来来实现。例如，[Serializable]特性用于标记一个类，表示其实例可以被序列化。
   - 应用：特性可以应用于整个程序集、模块或较小的程序元素（如类和属性）。它们可以像方法和属性一样接收参数。
   - 举例：
  
   ```csharp
   [Test]
   public void TestAttribute()
   {

   }

   [Description("描述")]
   public class Class1
   {

   }
   ```

2. **Type.GetCustomAttribute<T>():**
   描述：该方法允许你检索指定类型（Type）上定义的第一个 T 类型的自定义属性（Custom Attribute）。如果找到了该类型的自定义属性，则该方法返回该属性的实例；如果没有找到，则返回 null。
   举例：使用 Type.GetCustomAttribute<T>() 获取自定义属性：

   ```csharp
    using System;  
    using System.Reflection;  
  
    class Program  
    {  
        static void Main(string[] args)  
        {  
            Type type = typeof(TestClass);  
    
            // 获取 TestClass 上定义的 MyCustomAttribute 属性的实例  
            MyCustomAttribute attribute = type.GetCustomAttribute<MyCustomAttribute>();  
    
            if (attribute != null)  
            {  
                Console.WriteLine($"找到自定义属性: {attribute?.Description}");  
            }  
            else  
            {  
                Console.WriteLine("未找到自定义属性");  
            }  
        }  
    }
   ```

3. 操作：

**3.1 获取类上的attribute**
为了便于测试，我们在StudentInfo类上加上如下特性：

```csharp
using System.ComponentModel;

namespace Info.Main
{
    [Description("学生类的描述信息")]
    public class StudentInfo
    {
        //...<etc>
    }
}
```

在main函数中执行以下方法，观察控制台输出

```csharp
//操作特性
//获取类的特性
public static void TestGetClassAttribute01()
{
    var type01 = typeof(StudentInfo);
    var descAttribute = type01.GetCustomAttribute<DescriptionAttribute>(); //DescriptionAttribute是[Description]的全称
    Console.WriteLine($"{nameof(StudentInfo)}类的描述是{descAttribute?.Description}");
}
```

output:
