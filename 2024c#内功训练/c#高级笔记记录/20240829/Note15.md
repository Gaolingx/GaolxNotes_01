# C#高级编程之——Json操作（一）

在上节教程中我们结合反射，枚举，特性，扩展方法，自定义了一个获取枚举对应描述的拓展方法，今天我们来学习json的基本概念以及如何在c#中的操作json。

一、JSON基本概念
JSON: JavaScript Object Notation(JavaScript 对象表示法) 是一种轻量级的数据交换格式，作为存储和交换文本信息的语法，类似 XML。

基本结构：

- 对象（Object）：在JSON中，对象被表示为一系列的无序的“名称/值”对。一个对象以左花括号“{”开始，以右花括号“}”结束。每个“名称”后跟一个冒号“:”；“名称/值”对之间使用逗号“,”分隔。
示例：{"name": "John", "age": 30, "city": "New York"}

- 数组（Array）：在JSON中，数组是值（value）的有序集合。一个数组以左方括号“[”开始，以右方括号“]”结束。值之间使用逗号“,”分隔。
示例：["apple", "banana", "cherry"]

- 值（Value）：JSON中的值可以是：数字（整数或浮点数）、字符串（在双引号中）、逻辑值（true 或 false）、数组、对象、null。

二：JSON的优势

1. 易于阅读：JSON的语法结构清晰，易于人阅读和理解，即使对于非开发人员来说也是如此。
2. 易于编写：由于其简单的结构，JSON也易于编写。它支持多种编程语言，可以轻松地在不同系统之间进行数据交换。
3. 轻量级：JSON比XML更小、更快、更易解析。它没有额外的标记，这使得JSON的数据更加紧凑，有利于网络传输。
4. 语言无关性：JSON 使用 Javascript语法来描述数据对象，但是 JSON 仍然独立于语言和平台。JSON 解析器和 JSON 库支持许多不同的编程语言。 目前非常多的动态（PHP，JSP，.NET）编程语言都支持JSON，因此可以在不同的编程环境中使用JSON进行数据交换。
5. 支持复杂数据类型：JSON能够表示嵌套的对象和数组，这为表示复杂的数据结构提供了灵活性。
6. 广泛的支持：随着Web技术的不断发展，JSON已经成为数据交换的标准格式之一，得到了广泛的支持和应用。

总结：

1. JSON 指的是 JavaScript 对象表示法（JavaScript Object Notation）。
2. JSON 是存储和交换文本信息的语法，类似 XML。
3. JSON 比 XML 更小、更快，更易解析。

三、JSON数据示例：

假设我们有如下json文本：

```json
{
    "sites": [
    {id:1, "name":"米游社" , "url":"www.miyoushe.com" }, 
    {id:2, "name":"崩坏三" , "url":"www.bh3.com" }, 
    {id:3, "name":"崩坏：星穹铁道" , "url":"sr.mihoyo.com" }
    ]
}
```

则可以等价于以下c#代码：

```csharp

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
```

在上述json中，sites是匿名对象，它包含一个数组，数值中共包含三个对象，可以说明json是数据传输的载体
