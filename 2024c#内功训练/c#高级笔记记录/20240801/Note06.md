# C#高级编程之——反射（六）

## 三、反射入门——操作字段

在上期教程中，我们学习了Type.GetProperty和PropertyInfo.SetValue两个方法，我们通过反射操作属性（即对属性赋值、获取属性的值），今天我们开始学习通过反射操作字段。

### 详细知识点

**关于Type.GetField：**
Type.GetField 允许你在运行时查询和操作对象的类型信息。具体来说，Type.GetField 方法用于获取当前 Type 对象所表示的类或接口的公共字段信息。如果你需要访问非公共字段，你可能需要使用 Type.GetField(string name, BindingFlags bindingAttr) 方法的重载版本，其中 BindingFlags 参数允许你指定搜索的访问级别（如私有、受保护等）和其他搜索选项。

我们以操作StudentInfo中的_studentId私有字段为例：

在Main方法中执行如下代码，观察控制台输出：

```csharp
public static void TestOperationField01()
{
    var tp = typeof(StudentInfo);
    var field = tp.GetField("_studentId",BindingFlags.Instance|BindingFlags.NonPublic);

    var instance = Activator.CreateInstance(tp);

    field?.SetValue(instance, "19"); // 字段赋值

    var obj = field?.GetValue(instance); //获取字段的值
    Console.WriteLine($"我的年龄是{obj}");
}
```

可以看到控制台输出“我的年龄是19”，说明我们成功修改了instance实例中Name属性的值。可以看出操作字段和操作属性有很多相似之处，核心就在于Type.GetField类，下一章我们将讲解如何通过反射获取程序集信息。
