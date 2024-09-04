using Newtonsoft.Json;

namespace TestJsonAndCSharp
{
    public class JsonTest
    {
        //json转对象（反序列化）
        // json反序列化：json 字符串转换为C# 对象
        public static void TestJsonToObject()
        {
            string jsonText = "{\"Id\":1,\"Name\":\"米游社\",\"Url\":\"www.miyoushe.com\"}";
            //var site = JsonConvert.DeserializeObject(jsonText, typeof(Site)) as Site;
            var site = JsonConvert.DeserializeObject<Site>(jsonText);
            Console.WriteLine($"ID = {site?.Id},Name = {site?.Name},Url = {site?.Url}");
        }

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

        static void Main()
        {

        }
    }
}
