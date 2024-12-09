# C#高级编程之——反射（二）Type的使用（概念篇）

在上期教程中，我们对反射的概念有了简单的了解，知道什么是反射，反射有什么作用，还对c#一些核心概念做了温习，知道了程序集、模块、元数据、类型的作用、构成以及他们的关系，今天我们来开始对反射的使用进行介绍。

</br>

在我们的实验开始之前，我们先准备一个这样的源文件，这个类包含了无参无返回值的方法、有参无返回值方法、有参有返回值私有方法共三个方法，无参、有参两个构造函数，两个字段和属性，我们将使用反射获取字段、调用其中的类和构造函数，并了解属性的本质是什么。

## 一、Type类型

当我们需要使用反射时，type是最常用的一个类，也是我们必须要知道的一个类。反射可以获取模块里面的类型。

1. Type 类：表示类型声明（类型的引用）：类类型、接口类型、数组类型、值类型、枚举类型、类型参数、泛型类型定义，以及开放或封闭构造的泛型类型。
2. TypeInfo 类：表示类类型、接口类型、数组类型、值类型、枚举类型、类型参数、泛型类型定义，以及开放或封闭构造的泛型类型的类型声明。
3. TypeInfo对象表示类型定义本身，而 Type 对象表示对类型定义的引用。 获取 TypeInfo 对象将强制加载包含该类型的程序集。 相比之下，你可以操作 Type 对象，而无需运行时加载它们引用的程序集。

- **解读：**我们在反射的时候要知道一个类的类型信息，我们在声明类型时可以将它赋值给type类，获取里面的信息（元数据）。比如说我想知道这个值类型里面到底是什么。

- **用法**：有两种方法可以生成Type类的对象:一种是Typeof(类名)，一种是对象调用GetType()函数。

```csharp
//方式一
Type type = typeof(StudentInfo);

//方式二
StudentInfo stu = new(); //等价于StudentInfo stu = new StudentInfo;
Type type = stu.GetType();

//注：type.GetType等于typeof(Class)
```

- **概念：**Type为 system.Reflection 功能的根，也是访问元数据的主要方式。使用Type的成员获取关于类型声明的信息，这允许程序在运行时查询任何类型的元数据，如构造函数、方法、字段、属性和类的事件，以及在其中部署该类的模块和程序集。

- **用途：**

1. **获取类型信息**：你可以通过 `Type.GetType()` 方法获取一个类型的 `Type` 实例，或者通过 `typeof` 关键字直接在编译时获取。一旦你有了 `Type` 实例，就可以查询该类型的各种属性，如它的名称、是否可继承、是否为抽象类等。

2. **创建实例**：使用 `Type` 类的 `CreateInstance()` 方法，可以在运行时创建该类型的一个实例，这对于实现泛型工厂模式或动态实例化对象非常有用。

3. **查询成员**：`Type` 类提供了多种方法来查询类型的成员，包括 `GetMethods()`, `GetProperties()`, `GetFields()`, `GetConstructors()` 等。这些方法允许你获取类型的所有方法、属性、字段和构造函数的信息，进而可以进一步查询这些成员的详细信息（如参数、访问级别等）。

4. **类型比较和检查**：`Type` 类提供了 `IsAssignableFrom()`, `IsInstanceOfType()`, `IsSubclassOf()` 等方法，用于在类型之间进行比较和检查。

5. **获取和设置字段/属性值**：通过反射，你还可以使用 `Type` 类提供的方法来动态地获取和设置对象的字段和属性的值。

6. **自定义属性**：`Type` 类还允许你查询附加到类型上的自定义属性（Attribute），这是.NET中用于向程序元素（如类、方法、属性等）添加声明性信息的一种方式。

- **一句话总结：Type就是类型的声明。**

</br>

## 二、TypeInfo和Type的区别

TypeInfo出现于.net framework 4.5之后，这次调整用于区分两个概念：“reference”和“definition”。

- reference is a shallow representation of something
- definition is  a rich representation of something

例如System.Reflection.Assembly就代表了一个“definition” ，而System.Reflection.AssemblyName就代表了一个“reference”
在未区分这两种概念之前，System.Type是这两种概念的混合，通过一个Type实例，既可以获得该类型的“Name、Assembly、……”，也可以获得该类型的“NestTypes、Fields、Properties、Methods、Constructors、Events……”。这也导致了当我们获得一个Type的时候，就把它全部的信息都加载到了内存中，但是很多情况下，这并不是我们想要看到的。举例如下：

```csharp
public MyClass:BaseClass
{
 
}
 
//获得MyClass的类型对象
Type t=MyClass.GetType();
```

- 在 .net framework 4中，获得类型对象时，同时获得了基类(BaseClass)的类型对象，同时包括基类对象的“reference”和“definition”，因此需要把基类所在的程序集也加载进来。
- 在 .net framework 4.5中，如下代码：
  Type baseType = MyClass.GetType().GetTypeInfo().BaseType;
  在获得类型对象时，只获得了基类的"reference"

## 三、Type类的使用

1. 通过 typeof 运算符获取Type类型/通过实例对象获取类型：

```csharp
public static void TestGetType()
{
    //通过 typeof 运算符获取Type类型
    Type type01 = typeof(StudentInfo);

    //通过实例对象获取类型
    StudentInfo studentInfo = new StudentInfo(); //创建对象
    Type type02 = studentInfo.GetType(); //GetType 是Object 这个类的方法。
                                         //由于所有类型都继承自System.Object，所以所有的类型都含有GetType 方法
}
```
