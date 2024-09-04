# C#高级编程之——Json操作（三）

在上节教程中，我们学习了json的语法规则与数据结构，并简单介绍了c#与json序列化/反序列化的工具，今天我们将在项目中实操json序列化/反序列化，先来介绍最基本的操作。

七、JSON序列化/反序列化操作

以Newtonsoft.Json为例，执行以下操作

1. 首先，新建项目，使用nuget 工具 将 Newtonsoft.Json 引用到当前项目中
2. 新建一个Site类，创建以下代码：

   ```csharp
   public class Site
   {
    public int Id { get; set; }
    public string Name { get; set; }
    public string Url { get; set; }
   }
   ```

1、Json反序列化

1. 准备一个json字符串，粘贴到string中时注意使用转义字符。

2. 引用Newtonsoft.Json的命名空间，创建如下静态类，在Main中运行该方法。

   ```csharp
   //json转对象（反序列化）
   // json反序列化：json 字符串转换为C# 对象
    public static void TestJsonToObject()
    {
        string jsonText = "{\"Id\":1,\"Name\":\"米游社\",\"Url\":\"www.miyoushe.com\"}";
        //var site = JsonConvert.DeserializeObject(jsonText, typeof(Site)) as Site;
        var site = JsonConvert.DeserializeObject<Site>(jsonText);
        Console.WriteLine($"ID = {site?.Id},Name = {site?.Name},Url = {site?.Url}");
    }
   ```

3. 若控制台输出以下信息，说明我们成功将json字符串反序列化为c#对象。

2、Json序列化

与前面步骤相同，创建一个C#对象，调用JsonConvert.SerializeObject方法，代码如下：

```csharp
// C# 对象序列化为 Json字符串
public static void TestObjectToJson()
{
    Site site = new()
    {
        Id = 1,Name = "爱莉小跟班gaolx",Url = "https://www.miyoushe.com/sr/accountCenter/postList?id=277273444"
    }; //构造函数初始化器

    var jsonStr = JsonConvert.SerializeObject(site);
    Console.WriteLine(jsonStr);
}
```

若控制台打印以下日志，说明我们成功将类型为Site的c#对象成功序列化为string字符串。
